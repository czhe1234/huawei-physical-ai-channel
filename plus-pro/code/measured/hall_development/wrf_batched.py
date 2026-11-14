"""Batched equivalent of the pinned-author dense WRF power adaptation.

No replacement by a direct position MLP: every prediction passes deformation,
3D covariance, equirectangular alpha rendering and the physical wideband operator.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from measured.hall_adapters.wrf import WRFField,quaternion_matrix,project_gaussians
def fibonacci(count):
    t=np.arange(count,dtype=np.float64)+.5
    z=1-2*t/count;phi=np.pi*(3-np.sqrt(5))*t;r=np.sqrt(1-z*z)
    return np.c_[r*np.cos(phi),r*np.sin(phi),z]
def dense_batch(pixel,cov2,depth,opacity,color,width,height):
    b,k,_=pixel.shape
    yy,xx=torch.meshgrid(torch.arange(height,device=pixel.device,dtype=pixel.dtype),
                         torch.arange(width,device=pixel.device,dtype=pixel.dtype),indexing="ij")
    grid=torch.stack([xx,yy],-1).reshape(-1,2)
    delta=pixel[:,:,None,:]-grid[None,None]
    inv=torch.linalg.inv(cov2)
    exponent=-.5*torch.einsum("bkpi,bkij,bkpj->bkp",delta,inv,delta)
    alpha=(opacity[None,:,None]*exponent.exp()).clamp(max=.99)
    alpha=torch.where((alpha>=1/255)&(depth[:,:,None]>.2),alpha,torch.zeros_like(alpha))
    order=depth.argsort(dim=1)
    alpha=alpha.gather(1,order[:,:,None].expand_as(alpha))
    color=color.gather(1,order[:,:,None].expand_as(color))
    post=torch.cumprod(1-alpha,dim=1)
    pre=torch.cat([torch.ones_like(post[:,:1]),post[:,:-1]],dim=1)
    # Once post-transmittance crosses the source cutoff it cannot increase.
    weights=alpha*pre*(post>=1e-4)
    image=torch.einsum("bkp,bkc->bpc",weights,color).clamp(0,1)
    return image.reshape(b,height,width,-1)
class BatchedWRF(nn.Module):
    def __init__(self,fit_position,spec,count=64,width=360,height=90):
        super().__init__();self.count=count
        # Geometry-only equal-solid-angle initialization covers the complete camera.
        self.fields=nn.ModuleList([WRFField(spec.center[p]+5*fibonacci(count),fit_position,spec,p,width,height) for p in range(2)])
        for f in self.fields:
            f.log_scale.data.fill_(np.log(.85))
            # Start from a valid undeformed Gaussian field, rather than random scales.
            for name in ["gaussian_rotation","gaussian_scaling"]:
                layer=getattr(f.deformation,name);nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
        self.last_diagnostics={}
    def field_forward(self,f,position):
        b=len(position);k=len(f.xyz)
        xyz=((f.xyz.detach()-f.input_center)/f.input_scale)[None].expand(b,-1,-1).reshape(-1,3)
        tx=((position-f.input_center)/f.input_scale)[:,None].expand(-1,k,-1).reshape(-1,3)
        _,rotation,scale,signal=f.deformation(xyz,tx)
        q=(f.quaternion/f.quaternion.norm(dim=-1,keepdim=True).clamp_min(1e-8))[None].expand(b,-1,-1).reshape(-1,4)+rotation
        r=quaternion_matrix(q)
        scales=(f.log_scale.exp()[None].expand(b,-1,-1).reshape(-1,3)+scale).abs().clamp_min(1e-3)
        cov=r@torch.diag_embed(scales.square())@r.transpose(-1,-2)
        means=f.xyz[None].expand(b,-1,-1).reshape(-1,3)
        pixel,cov2,depth=project_gaussians(means,cov,f.center,f.camera,f.width,f.height)
        direction=nn.functional.normalize(means-f.center,dim=-1)
        sh=(f.sh[None].expand(b,-1,-1,-1).reshape(-1,16,1)+signal[:,None,:]).transpose(1,2)
        color=(f.eval_sh(3,sh,direction)+.5).clamp_min(0)
        image=dense_batch(pixel.reshape(b,k,2),cov2.reshape(b,k,2,2),depth.reshape(b,k),f.opacity_logit.sigmoid(),color.reshape(b,k,1),f.width,f.height)
        with torch.no_grad():
            self.last_diagnostics=dict(image_clipped_one_fraction=float((image>=1).float().mean()),zero_pixel_fraction=float((image==0).float().mean()),
                minimum_scale=float(scales.min()),maximum_scale=float(scales.max()),positive_color_fraction=float((color>0).float().mean()))
        return image[...,0].flatten(1)@f.beam_response
    def forward(self,position):
        return torch.stack([self.field_forward(f,position) for f in self.fields],dim=1)
    def optimizer_groups(self,lr=.0005):
        groups=[]
        for index,f in enumerate(self.fields):
            groups.extend([dict(params=[f.xyz],lr=.0002,name="xyz"+str(index)),
              dict(params=[f.log_scale],lr=.002,name="scale"+str(index)),
              dict(params=[f.quaternion],lr=.001,name="rotation"+str(index)),
              dict(params=[f.opacity_logit],lr=.02,name="opacity"+str(index)),
              dict(params=[f.sh],lr=.005,name="SH"+str(index)),
              dict(params=[p for p in f.deformation.parameters() if p.requires_grad],lr=lr,name="deformation"+str(index))])
        return groups

    @torch.inference_mode()
    def predict(self,query):
        from measured.hall_adapters.common import check_prediction
        self.eval()
        x=torch.tensor(np.array(query.position),dtype=torch.float32,device=self.fields[0].xyz.device)
        return check_prediction(dict(method="WRF_GSplus_batched_author_power_adapter",power=self(x).cpu().numpy(),
            diagnostics=dict(original_CUDA_kernel_verified=False,gaussians_per_panel=self.count,query_phase_alignment=False)),query)
