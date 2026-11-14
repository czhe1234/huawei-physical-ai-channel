"""GPU-batched WRF-GS+ power adapter; original Gaussian/deform/SH core retained.

The task readout is the frozen adapter's learned unshifted DFT-codebook power.
This is not a complex-H predictor or the original CUDA-rasterizer reproduction.
"""
from __future__ import annotations
import copy
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from project_paths import PATHS
from round2.paper_baseline_adapters_v1.wrfgs import (
    WRFGSPlusPowerAdapter, _quaternion_matrix, _project, POWER_CONTRACT,
)
POWER_GAIN = 128.0  # Unnormalized 16x8 FFT2 Parseval gain.


def map_vertices(path=None):
    path = PATHS.round2 / "Round2_Map.ply" if path is None else Path(path)
    with path.open() as f:
        n = None
        while True:
            line = f.readline()
            if not line:
                raise ValueError("PLY header truncated")
            if line.startswith("element vertex "):
                n = int(line.split()[-1])
            if line.strip() == "end_header":
                break
        if not n:
            raise ValueError("PLY vertex count missing")
        xyz = np.loadtxt(f, max_rows=n, usecols=(0,1,2), dtype=np.float64)
    if xyz.shape != (n,3) or not np.isfinite(xyz).all():
        raise ValueError("Finite public map vertices required")
    return xyz


def initialize_map_gaussians(geometry, count=1024, seed=20261101):
    xyz = map_vertices()
    initial = []
    rng = np.random.default_rng(seed)
    for site, center in enumerate(geometry.bs_positions_m):
        region = xyz[(xyz[:,1] >= 0) == bool(site)]
        region = region[np.linalg.norm(region-center[None],axis=1) > 1.0]
        if not len(region):
            raise ValueError("Public map has no site-region initialization")
        selected = rng.choice(len(region), size=count, replace=len(region)<count)
        initial.append(np.array(region[selected],copy=True))
    z_scale = max(float(xyz[:,2].std()), float(np.ptp(xyz[:,2]))/4, .1)
    return initial, z_scale


def alpha_render_batch(pixel, covariance, depth, opacity, color, width, height):
    """Same front-to-back source rule, batched over position/frequency pairs."""
    yy, xx = torch.meshgrid(torch.arange(height,dtype=pixel.dtype,device=pixel.device),
                            torch.arange(width,dtype=pixel.dtype,device=pixel.device),indexing="ij")
    grid = torch.stack((xx,yy),-1).reshape(-1,2)
    delta = pixel[:,:,None,:]-grid[None,None]
    exponent = -.5*torch.einsum("bkpi,bkij,bkpj->bkp",delta,torch.linalg.inv(covariance),delta)
    alpha = (opacity[None,:,None]*exponent.exp()).clamp(max=.99)
    alpha = torch.where((alpha>=1/255)&(depth[:,:,None]>.2),alpha,0.)
    order = depth.argsort(dim=1)
    alpha = alpha.gather(1,order[:,:,None].expand(-1,-1,alpha.shape[-1]))
    color = color.gather(1,order[:,:,None].expand(-1,-1,color.shape[-1]))
    post = torch.cumprod(1-alpha,dim=1)
    pre = torch.cat((torch.ones_like(post[:,:1]),post[:,:-1]),dim=1)
    weights = alpha*pre*(post>=1e-4)
    return torch.einsum("bkp,bkc->bpc",weights,color)


def site_forward_batch(field, positions, normalized_frequency):
    b, k = len(positions), len(field.xyz)
    xyz_input = ((field.xyz.detach()-field.input_center)/field.input_scale)[None].expand(b,-1,-1)
    position_input = (positions-field.input_center)/field.input_scale
    condition = torch.cat((position_input,normalized_frequency[:,None]),dim=-1)
    condition = condition[:,None,:].expand(-1,k,-1)
    warp, rotation, scale, signal = field.deformation(xyz_input.reshape(b*k,3),condition.reshape(b*k,4))
    means = field.xyz[None]+warp.reshape(b,k,3)
    q = F.normalize(field.quaternion,dim=-1,eps=1e-8)[None]+rotation.reshape(b,k,4)
    matrix = _quaternion_matrix(q.reshape(b*k,4))
    scales = (field.log_scale.exp()[None]+scale.reshape(b,k,3)).abs().clamp_min(1e-3)
    covariance = matrix @ torch.diag_embed(scales.reshape(b*k,3).square()) @ matrix.transpose(-1,-2)
    pixel,cov2,depth = _project(means.reshape(b*k,3),covariance,field.center,field.camera,field.width,field.height)
    pixel,cov2,depth = pixel.reshape(b,k,2),cov2.reshape(b,k,2,2),depth.reshape(b,k)
    direction = F.normalize(means-field.center,dim=-1,eps=1e-8)
    sh = (field.sh[None]+signal.reshape(b,k,1,8)).transpose(-1,-2)
    color = F.softplus(field.eval_sh(3,sh,direction)+.5)
    image = alpha_render_batch(pixel,cov2,depth,field.opacity_logit.sigmoid(),color,field.width,field.height)
    readout = F.softplus(field.readout_logits)
    readout = readout/readout.sum(-1,keepdim=True).clamp_min(1e-12)
    power = torch.einsum("cjp,bpc->bcj",readout,image)
    power = power*F.softplus(field.channel_log_gain)[None,:,None]
    return power.reshape(b,2,4,128).permute(0,1,3,2).reshape(b,256,4)*POWER_GAIN


def site_forward_reference(field, position, normalized_frequency):
    """Independent per-pair source calculation with the restored XYZ-warp head."""
    xyz_input = (field.xyz.detach()-field.input_center)/field.input_scale
    position_input = (position-field.input_center)/field.input_scale
    condition = torch.cat((position_input,normalized_frequency.reshape(1))).expand(len(field.xyz),-1)
    warp,rotation,scale,signal = field.deformation(xyz_input,condition)
    means = field.xyz+warp
    q = F.normalize(field.quaternion,dim=-1,eps=1e-8)+rotation
    matrix = _quaternion_matrix(q)
    scales = (field.log_scale.exp()+scale).abs().clamp_min(1e-3)
    covariance = matrix@torch.diag_embed(scales.square())@matrix.transpose(-1,-2)
    pixel,cov2,depth = _project(means,covariance,field.center,field.camera,field.width,field.height)
    direction = F.normalize(means-field.center,dim=-1,eps=1e-8)
    sh = (field.sh+signal[:,None,:]).transpose(1,2)
    color = F.softplus(field.eval_sh(3,sh,direction)+.5)
    image = alpha_render_batch(pixel[None],cov2[None],depth[None],field.opacity_logit.sigmoid(),color[None],field.width,field.height)[0]
    readout = F.softplus(field.readout_logits)
    readout = readout/readout.sum(-1,keepdim=True).clamp_min(1e-12)
    power = torch.einsum("cjp,pc->cj",readout,image)
    power = power*F.softplus(field.channel_log_gain)[:,None]
    return power.reshape(2,4,128).permute(0,2,1).reshape(256,4)*POWER_GAIN


class BatchedWRFGS(WRFGSPlusPowerAdapter):
    def __init__(self,geometry,fit_positions,*,count=1024,seed=20261101,width=32,height=16,initial=None,z_scale=None):
        if initial is None:
            initial,z_scale=initialize_map_gaussians(geometry,count,seed)
        # Frozen geometry columns [normal,horizontal,up] -> camera [horizontal,up,normal].
        axes = np.asarray(geometry.bs_rotations)[:,:,[1,2,0]]
        super().__init__(np.asarray(fit_positions),geometry.bs_positions_m,axes,initial,
                         geometry.frequencies_hz,width=width,height=height,
                         contract_id=POWER_CONTRACT,geometry_identity=geometry.geometry_fingerprint)
        self.gaussian_count = count
        self.initial_seed = seed
        self.public_z_scale = max(float(z_scale),.1)
        self.pair_batch = 32
        for field in self.fields:
            field.input_scale[2] = self.public_z_scale
            # Restore the author's position-dependent XYZ warp; identity initial state.
            for parameter in field.deformation.gaussian_warp.parameters():
                parameter.requires_grad_(True)
            nn.init.zeros_(field.deformation.gaussian_warp.weight)
            nn.init.zeros_(field.deformation.gaussian_warp.bias)

    def forward_pairs(self,positions,sites,frequency_indices):
        device=self.frequencies_hz.device
        positions=torch.as_tensor(positions,device=device,dtype=torch.float32)
        sites=torch.as_tensor(sites,device=device,dtype=torch.long)
        ids=torch.as_tensor(frequency_indices,device=device,dtype=torch.long)
        if positions.ndim!=2 or positions.shape[-1]!=3 or sites.shape!=(len(positions),) or ids.shape!=(len(positions),):
            raise ValueError("Paired position/site/frequency shapes")
        if bool(((sites<0)|(sites>=len(self.fields))).any()) or bool(((ids<0)|(ids>=len(self.frequencies_hz))).any()):
            raise ValueError("Unknown site or frequency")
        if not bool(torch.isfinite(positions).all()):
            raise ValueError("Nonfinite query position")
        normalized=((self.frequencies_hz[ids]-self.frequency_center)/self.frequency_scale).float()
        output=positions.new_zeros((len(positions),256,4))
        for site,field in enumerate(self.fields):
            index=torch.nonzero(sites==site,as_tuple=False).flatten()
            if len(index):
                values=site_forward_batch(field,positions[index],normalized[index])
                output=output.index_copy(0,index,values)
        return output

    def forward(self,positions,sites,frequencies_hz=None):
        device=self.frequencies_hz.device
        positions=torch.as_tensor(positions,device=device,dtype=torch.float32)
        sites=torch.as_tensor(sites,device=device,dtype=torch.long)
        frequencies=self.frequencies_hz if frequencies_hz is None else torch.as_tensor(frequencies_hz,device=device,dtype=torch.float64)
        matches=frequencies[:,None]==self.frequencies_hz[None]
        if not bool((matches.sum(-1)==1).all()):
            raise ValueError("Explicit frequency grid/order required")
        fids=matches.long().argmax(-1)
        n,f=len(positions),len(fids)
        pair_pos=positions[:,None].expand(-1,f,-1).reshape(-1,3)
        pair_site=sites[:,None].expand(-1,f).reshape(-1)
        pair_freq=fids[None].expand(n,-1).reshape(-1)
        pieces=[]
        for lo in range(0,len(pair_freq),self.pair_batch):
            pieces.append(self.forward_pairs(pair_pos[lo:lo+self.pair_batch],pair_site[lo:lo+self.pair_batch],pair_freq[lo:lo+self.pair_batch]))
        if not pieces:
            return positions.new_empty((n,256,4,f))
        return torch.cat(pieces).reshape(n,f,256,4).permute(0,2,3,1).contiguous()

    def optimizer(self):
        groups=[]
        for field in self.fields:
            groups.extend([
                dict(params=[field.xyz],lr=1.6e-4,name="xyz"),
                dict(params=[field.log_scale],lr=.005,name="scale"),
                dict(params=[field.quaternion],lr=.001,name="rotation"),
                dict(params=[field.opacity_logit],lr=.025,name="opacity"),
                dict(params=[field.sh,field.readout_logits,field.channel_log_gain],lr=.0025,name="features_readout"),
                dict(params=[p for p in field.deformation.parameters() if p.requires_grad],lr=1.6e-4,name="deformation"),
            ])
        return torch.optim.Adam(groups,lr=0.,eps=1e-15)

    def constrain(self):
        with torch.no_grad():
            for field in self.fields:
                field.log_scale.clamp_(-6.,4.)
                field.opacity_logit.clamp_(-12.,12.)

    def metadata(self):
        return dict(**self.eligibility(),model="batched_WrFGSplus_power_task_adapter",
                    gaussian_count_per_site=self.gaussian_count,deformation_depth=8,deformation_width=256,
                    spherical_harmonic_degree=3,xyz_warp_restored=True,xyz_warp_initial="zero_identity",
                    z_normalization="public_map_geometry_std_or_quarter_extent",public_z_scale=self.public_z_scale,
                    xy_normalization="role_FIT_only",power_gain=POWER_GAIN,
                    original_author_initial_gaussians=200000,original_author_iterations=200000,
                    batched_execution_preserves_power_contract=True)


def power_loss(prediction,target):
    """Metric-oriented power supervision; all target data must be role-FIT."""
    p=prediction/POWER_GAIN
    t=target/POWER_GAIN
    pn=p.sum(1)
    tn=t.sum(1)
    valid=tn>1e-20
    numerator=(p*t).sum(1,dtype=torch.float64)
    denominator=(p.square().sum(1,dtype=torch.float64).clamp_min(1e-30)*t.square().sum(1,dtype=torch.float64).clamp_min(1e-30)).sqrt()
    cosine=(numerator/denominator).float()
    angular=(1-cosine[valid]).mean() if bool(valid.any()) else p.sum()*0
    power=F.smooth_l1_loss(torch.log1p(pn),torch.log1p(tn))
    return angular+.05*power,dict(angular=float(angular.detach()),log_power=float(power.detach()))
