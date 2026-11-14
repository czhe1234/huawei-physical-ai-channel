"""Differentiable frequency/relative-delay/receive-energy reconstruction."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from .observables import probability,beam_channel,unit_channel,total_energy,EPS
def relative_anchor(h):
    x=h.flatten(-3,-2)
    cross=(x[...,1:]*x[...,:-1].conj()).sum((-2,-1))
    # A common linear phase changes this continuous anchor by the same delay.
    return -h.shape[-1]*torch.angle(cross)/(2*torch.pi)

def canonicalize(base):
    # The gauge is derived from the predicted FIT-donor carrier, never query CSI.
    anchor=relative_anchor(base).detach()
    k=torch.arange(base.shape[-1],device=base.device,dtype=base.real.dtype)
    ramp=torch.exp(2j*torch.pi*anchor[...,None]*k/base.shape[-1])
    value=base*ramp[...,None,None,:]
    flat=value.flatten(-3)
    index=flat.abs().argmax(-1,keepdim=True)
    chosen=flat.gather(-1,index)[...,0]
    phase=torch.where(chosen.abs()>0,chosen/chosen.abs().clamp_min(EPS),torch.ones_like(chosen)).detach()
    return value/phase[...,None,None,None],dict(anchor=anchor,phase=phase)
def restore_reference(value,reference):
    k=torch.arange(value.shape[-1],device=value.device,dtype=value.real.dtype)
    ramp=torch.exp(-2j*torch.pi*reference["anchor"][...,None]*k/value.shape[-1])
    return value*ramp[...,None,None,:]*reference["phase"][...,None,None,None]

def angle_project(h,q,smooth=True):
    z=beam_channel(h);n,p,b,f=z.shape;s=q.shape[-1]
    current=probability(z.abs().square().reshape(n,p,b,s,f//s).sum(-1),(-2,-1))
    log_gain=.5*(q.clamp_min(1e-12).log()-current.clamp_min(1e-12).log())
    if smooth and s>1:
        gain=F.interpolate(log_gain.reshape(-1,1,s),size=f,mode="linear",align_corners=False).reshape(n,p,b,f).exp()
    else:gain=log_gain[...,None].expand(-1,-1,-1,-1,f//s).reshape(n,p,b,f).exp()
    value=z*gain
    empty=(current<=0)&(q>0)
    if empty.any():
        replacement=(q.clamp_min(0)/(f//s)).sqrt()[...,None].expand(-1,-1,-1,-1,f//s).reshape(n,p,b,f)
        take=empty[...,None].expand(-1,-1,-1,-1,f//s).reshape(n,p,b,f)
        value=torch.where(take,replacement.to(z.dtype),value)
    return torch.fft.ifft2(value.reshape(n,p,8,4,f),dim=(-3,-2),norm="ortho")
def delay_distribution(h):
    v=torch.fft.ifft(beam_channel(h),dim=-1,norm="ortho")
    return probability(v.abs().square(),(-1,))
def delay_project(h,desired):
    z=beam_channel(h);v=torch.fft.ifft(z,dim=-1,norm="ortho")
    power=v.abs().square();energy=power.sum(-1,keepdim=True)
    unit=v/power.clamp_min(EPS).sqrt()
    unit=torch.where(power>0,unit,torch.ones_like(unit))
    v=unit*(desired*energy).clamp_min(EPS).sqrt()
    z=torch.fft.fft(v,dim=-1,norm="ortho")
    return torch.fft.ifft2(z.reshape_as(h),dim=(-3,-2),norm="ortho")
def element_distribution(h):
    return probability(h.abs().square().sum(-1).flatten(-2),(-1,))
def element_project(h,desired):
    current=element_distribution(h)
    gain=(desired/current.clamp_min(EPS)).clamp_min(0).sqrt().reshape(*h.shape[:2],8,4,1)
    return h*gain
def root_blend(a,b,weight,dims):
    while weight.ndim<a.ndim:weight=weight[...,None]
    return probability(((1-weight)*a.clamp_min(1e-12).sqrt()+weight*b.clamp_min(1e-12).sqrt()).square(),dims)
def reconstruct(base,q,delay,element,energy,cycles=4,angular=True,temporal=True,receive=True):
    valid=total_energy(base)>0;value=unit_channel(base)
    for _ in range(cycles):
        if angular:value=angle_project(value,q)
        if temporal:value=delay_project(value,delay)
        if receive:value=element_project(value,element)
        value=unit_channel(value)
    value=value*(energy.clamp_min(0)*base.shape[-1]).sqrt()[...,None,None,None]
    value=torch.where(valid[...,None,None,None],value,torch.zeros_like(value))
    # Residuals refer to the final channel, never to an unverified endpoint only.
    actual_q=probability(beam_channel(value).abs().square().reshape(*q.shape,value.shape[-1]//q.shape[-1]).sum(-1),(-2,-1))
    return value,dict(angle_residual=(actual_q-q).abs().mean(),delay_residual=(delay_distribution(value)-delay).abs().mean(),
      receive_energy_residual=(element_distribution(value)-element).abs().mean(),
      power_relative_residual=((total_energy(value)-energy).abs()/energy.clamp_min(EPS)).max())
