"""Equivalent elementwise 2x2 Gaussian quadratic form; no tiny-bmm expansion."""
from __future__ import annotations
import torch


def alpha_render_batch_elementwise(pixel,covariance,depth,opacity,color,width,height):
    yy,xx=torch.meshgrid(torch.arange(height,dtype=pixel.dtype,device=pixel.device),
                         torch.arange(width,dtype=pixel.dtype,device=pixel.device),indexing="ij")
    gx=xx.reshape(1,1,-1);gy=yy.reshape(1,1,-1)
    dx=pixel[:,:,0,None]-gx;dy=pixel[:,:,1,None]-gy
    inverse=torch.linalg.inv(covariance)
    exponent=-.5*(inverse[:,:,0,0,None]*dx.square()+
                    (inverse[:,:,0,1,None]+inverse[:,:,1,0,None])*dx*dy+
                    inverse[:,:,1,1,None]*dy.square())
    alpha=(opacity[None,:,None]*exponent.exp()).clamp(max=.99)
    alpha=torch.where((alpha>=1/255)&(depth[:,:,None]>.2),alpha,0.)
    order=depth.argsort(dim=1)
    alpha=alpha.gather(1,order[:,:,None].expand(-1,-1,alpha.shape[-1]))
    color=color.gather(1,order[:,:,None].expand(-1,-1,color.shape[-1]))
    post=torch.cumprod(1-alpha,dim=1)
    pre=torch.cat((torch.ones_like(post[:,:1]),post[:,:-1]),dim=1)
    weights=alpha*pre*(post>=1e-4)
    return torch.einsum("bkp,bkc->bpc",weights,color)


def install():
    from . import wrfgs_model
    wrfgs_model.alpha_render_batch=alpha_render_batch_elementwise
    return dict(alpha_quadratic="explicit_elementwise_2x2",inverse="unchanged_torch.linalg.inv",
                gaussian_count_changed=False,parameterization_changed=False)
