"""Position-only WRF-GS+ adaptation producing complete wideband complex H.

Retained: verified author deformation trunk, deformable Gaussian geometry,
degree-3 spherical harmonics, and front-to-back alpha blending.
Changed: signed real/imaginary full-band signal heads, coherent array readout,
and complex/PAS/PDP supervision. This is not an unchanged paper reproduction.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from round2.paper_baseline_adapters_v1.wrfgs import (
    _author_objects, _quaternion_matrix, _project, AUTHOR_COMMIT,
)
from round2.paper_baseline_training_v1.wrfgs_model import initialize_map_gaussians
from round2.paper_baseline_training_v1.wrfgs_fast_ops import alpha_render_batch_elementwise

C0 = 299792458.0
LANES, TONES = 8, 192

def angular_directions(width, height, camera):
    yy, xx = torch.meshgrid(torch.arange(height, dtype=torch.float64),
                            torch.arange(width, dtype=torch.float64), indexing="ij")
    az = (xx.reshape(-1)+.5)*(2*math.pi/width)-math.pi
    el = (yy.reshape(-1)+.5)*(math.pi/height)-math.pi/2
    local = torch.stack((el.cos()*az.sin(), el.sin(), el.cos()*az.cos()), -1)
    return local @ torch.as_tensor(np.array(camera), dtype=torch.float64).T

class SiteField(nn.Module):
    def __init__(self, xyz, center, camera, input_center, width, height):
        super().__init__()
        author_class, embedder, self.eval_sh = _author_objects(None)
        self.xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float32))
        self.log_scale = nn.Parameter(torch.full_like(self.xyz, math.log(.85)))
        self.quaternion = nn.Parameter(torch.tensor([[1.,0.,0.,0.]]).repeat(len(xyz),1))
        self.opacity_logit = nn.Parameter(torch.full((len(xyz),), -2.1972246))
        self.sh = nn.Parameter(torch.zeros(len(xyz), 16, LANES*2))
        nn.init.normal_(self.sh[:,0], std=.03)
        self.deformation = author_class(is_blender=True, is_6dof=False)
        # Source MLP layout is retained. Meter-based position PE resolves RF variation.
        self.deformation.embed_pos_fn, position_dim = embedder(9, 3)
        self.deformation.posnet[0] = nn.Linear(position_dim, 256)
        self.deformation.gaussian_signal = nn.Linear(256, LANES*TONES*2)
        nn.init.normal_(self.deformation.gaussian_signal.weight, std=.001)
        nn.init.zeros_(self.deformation.gaussian_signal.bias)
        # The public source takes abs(real*exp(i*phase)), erasing its phase head.
        # Our signed two-component head bypasses that operation explicitly.
        for p in self.deformation.gaussian_phase.parameters():
            p.requires_grad_(False)
        for name in ("gaussian_warp", "gaussian_rotation", "gaussian_scaling"):
            nn.init.zeros_(getattr(self.deformation,name).weight)
            nn.init.zeros_(getattr(self.deformation,name).bias)
        self.register_buffer("center", torch.tensor(np.array(center), dtype=torch.float32))
        self.register_buffer("camera", torch.tensor(np.array(camera), dtype=torch.float32))
        self.register_buffer("input_center", torch.tensor(input_center, dtype=torch.float32))
        self.width, self.height = width, height

    def deformation_output(self, xyz, position):
        net = self.deformation
        tx = net.posnet(net.embed_pos_fn(position))
        gx = net.embed_fn(xyz)
        h = torch.cat((gx, tx), -1)
        for i, layer in enumerate(net.linear):
            h = F.relu(layer(h))
            if i in net.skips:
                h = torch.cat((gx, tx, h), -1)
        return (net.gaussian_warp(h), net.gaussian_rotation(h),
                net.gaussian_scaling(h), net.gaussian_signal(h))

    def angular_field(self, positions):
        b, k = len(positions), len(self.xyz)
        xyz_in = (self.xyz.detach()-self.input_center)[None].expand(b,-1,-1)
        pos_in = (positions-self.input_center)[:,None].expand(-1,k,-1)
        warp, rotation, scale, signal = self.deformation_output(
            xyz_in.reshape(b*k,3), pos_in.reshape(b*k,3))
        means = self.xyz[None]+warp.reshape(b,k,3)
        q = F.normalize(self.quaternion,dim=-1,eps=1e-8)[None]+rotation.reshape(b,k,4)
        mat = _quaternion_matrix(q.reshape(b*k,4))
        sizes = (self.log_scale.exp()[None]+scale.reshape(b,k,3)).abs().clamp_min(1e-3)
        cov = mat @ torch.diag_embed(sizes.reshape(b*k,3).square()) @ mat.transpose(-1,-2)
        pixel,cov2,depth = _project(means.reshape(b*k,3),cov,self.center,self.camera,self.width,self.height)
        dirs = F.normalize(means-self.center,dim=-1,eps=1e-8)
        base = self.eval_sh(3,self.sh[None].expand(b,-1,-1,-1).transpose(-1,-2),dirs)
        # Exact linearity of source sh + d_signal broadcast, without a huge SH tensor.
        basis_sum = self.eval_sh(3,dirs.new_ones((b,k,1,16)),dirs)
        signed = base.reshape(b,k,LANES,1,2) + (
            signal.reshape(b,k,LANES,TONES,2)*basis_sum[:,:,:,None,None])
        image = alpha_render_batch_elementwise(
            pixel.reshape(b,k,2), cov2.reshape(b,k,2,2), depth.reshape(b,k),
            self.opacity_logit.sigmoid(), signed.reshape(b,k,LANES*TONES*2),
            self.width,self.height)
        image = image.reshape(b,self.width*self.height,LANES,TONES,2)
        return torch.complex(image[...,0], image[...,1])

class FullHWRFGS(nn.Module):
    def __init__(self,geometry,fit_positions,count=1024,seed=42,width=64,height=32):
        super().__init__()
        xyz,_ = initialize_map_gaussians(geometry,count=count,seed=seed)
        cameras = np.array(geometry.bs_rotations)[:,:,[1,2,0]]
        fit = np.asarray(fit_positions,np.float64)
        self.fields = nn.ModuleList([
            SiteField(xyz[s],geometry.bs_positions_m[s],cameras[s],fit.mean(0),width,height)
            for s in range(2)])
        self.register_buffer("frequencies",torch.tensor(np.array(geometry.frequencies_hz),dtype=torch.float64))
        ports = np.array(geometry.bs_port_positions_m).reshape(2,2,128,3)
        if not np.array_equal(ports[:,0],ports[:,1]):
            raise ValueError("Expected co-located polarization ports")
        offsets = ports[:,0]-np.array(geometry.bs_positions_m)[:,None]
        self.register_buffer("offsets",torch.tensor(offsets,dtype=torch.float64))
        self.register_buffer("directions",torch.stack([
            angular_directions(width,height,cameras[s]) for s in range(2)]))
        self.register_buffer("_manifold",None,persistent=False)
        self.geometry_identity=geometry.geometry_fingerprint
        self.count,self.width,self.height=count,width,height

    def manifold(self):
        if self._manifold is None:
            products = torch.einsum("smd,spd->smp",self.offsets,self.directions)
            # exp(-ik|R*d-r_m|) = exp(-ikR)*exp(+ik d.r_m), common range phase in field.
            phase = (2*math.pi/C0)*self.frequencies[None,:,None,None]*products[:,None]
            self._manifold = torch.polar(torch.ones_like(phase),phase).to(torch.complex64)
        return self._manifold

    def readout(self,angular,site,frequency_ids=None):
        # angular [B,pixel,pol*UE,F] -> [B,pol*H*V,UE,F].
        a = self.manifold()[site]
        if frequency_ids is not None:
            a=a[frequency_ids]
        b,p,c,f=angular.shape
        z=angular.permute(3,1,0,2).reshape(f,p,b*c)
        h=torch.bmm(a,z).reshape(f,128,b,2,4).permute(2,3,1,4,0)
        return h.reshape(b,256,4,f)/math.sqrt(p)

    def forward(self,positions,sites):
        device=self.frequencies.device
        pos=torch.as_tensor(positions,dtype=torch.float32,device=device)
        sites=torch.as_tensor(sites,dtype=torch.long,device=device)
        if pos.shape!=(len(sites),3) or not bool(torch.isfinite(pos).all()):
            raise ValueError("Invalid position input")
        if bool(((sites<0)|(sites>=2)).any()):
            raise ValueError("Invalid site")
        result=torch.zeros((len(pos),256,4,TONES),dtype=torch.complex64,device=device)
        for s,field in enumerate(self.fields):
            index=torch.nonzero(sites==s,as_tuple=False).flatten()
            if len(index):
                h=self.readout(field.angular_field(pos[index]),s)
                result=result.index_copy(0,index,h)
        return result

    def optimizer(self):
        groups=[]
        for field in self.fields:
            groups += [
                dict(params=[field.xyz],lr=1.6e-4),
                dict(params=[field.log_scale],lr=.005),
                dict(params=[field.quaternion],lr=.001),
                dict(params=[field.opacity_logit],lr=.025),
                dict(params=[field.sh],lr=.0025),
                dict(params=[p for p in field.deformation.parameters() if p.requires_grad],lr=1.6e-4)]
        return torch.optim.Adam(groups,eps=1e-8)

    def constrain(self):
        with torch.no_grad():
            for field in self.fields:
                field.log_scale.clamp_(-6.,4.)
                field.opacity_logit.clamp_(-12.,12.)

    def metadata(self):
        return dict(method="WRF-GS+ full-H task adaptation",source_commit=AUTHOR_COMMIT,
            input="position, site and known geometry; no query CSI",
            output="complex64[B,256,4,192]",port_order="BS polarization,horizontal16,vertical8;UE4",
            frequencies=self.frequencies.detach().cpu().tolist(),gaussians_per_site=self.count,
            angular_grid=[self.height,self.width],geometry_identity=self.geometry_identity,
            original_paper_exact_reproduction=False,original_CUDA_reproduction=False,
            adaptation=["signed real/imag signal instead of abs/clamped display values",
              "192 parallel frequency heads; shared geometric deformation within 5.76MHz band",
              "fixed coherent array response instead of learned nonnegative power readout",
              "meter position PE9; author Gaussian PE10 and 8x256 trunk",
              "training complex CSI plus PAS/PDP loss"],
            phase_delay_amplitude_query_alignment=False,units="FIT scalar RMS normalized H")

def spectral_loss(pred,target):
    b=len(pred)
    pp=torch.fft.fft2(pred.reshape(b,2,16,8,4,TONES),dim=(2,3)).abs().square().reshape_as(pred.real)
    tp=torch.fft.fft2(target.reshape(b,2,16,8,4,TONES),dim=(2,3)).abs().square().reshape_as(target.real)
    dp=torch.fft.ifft(pred,dim=-1).abs().square()
    dt=torch.fft.ifft(target,dim=-1).abs().square()
    def cosine(x,y,axis):
        x=x.to(torch.float64);y=y.to(torch.float64)
        dot=(x*y).sum(axis)
        product=x.square().sum(axis)*y.square().sum(axis)
        # Replace zero before sqrt, avoiding 0*infinite backward derivatives.
        den=torch.where(product>0,product,torch.ones_like(product)).sqrt()
        return (dot/den).float().mean()
    pas=cosine(pp,tp,1);pdp=cosine(dp,dt,3)
    mse=(pred-target).abs().square().mean()
    loss=.4*(1-pas)+.4*(1-pdp)+.2*mse
    return loss,dict(pas=float(pas.detach()),pdp=float(pdp.detach()),normalized_mse=float(mse.detach()))
