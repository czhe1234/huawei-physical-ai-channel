"""Differentiable frozen metric equivalents; each channel uses its own reference."""
import torch
EPS=1e-30
COHERENCE_MIN=1e-6
KEYS=("pas_cosine","relative_pdp_cosine","receive_energy_TV","receive_delay_joint_TV")

def norm(x,dims):
    total=x.sum(dims,keepdim=True)
    return torch.where(total>0,x/total.clamp_min(EPS),torch.zeros_like(x))

def describe(h):
    if h.ndim!=5 or tuple(h.shape[1:4])!=(2,8,4):raise ValueError("Expected[N,2,8,4,F]")
    total=h.abs().square().sum((-3,-2,-1),keepdim=True)
    x=h/total.clamp_min(EPS).sqrt()
    flat=x.flatten(-3,-2);a,b=flat[...,1:],flat[...,:-1]
    cross=(a*b.conj()).sum((-2,-1))
    den=(a.abs().square().sum((-2,-1))*b.abs().square().sum((-2,-1))).clamp_min(EPS).sqrt()
    valid=(total[...,0,0,0]>0)&(cross.abs()/den>COHERENCE_MIN)
    # arg is fully differentiated for predictions. No pairwise fitted shift.
    phase=torch.angle(torch.where(valid,cross,torch.ones_like(cross)))
    k=torch.arange(h.shape[-1],device=h.device,dtype=h.real.dtype)
    relative=x*torch.exp(-1j*phase[...,None]*k)[...,None,None,:]
    pdp=torch.fft.ifft(relative,dim=-1,norm="ortho").abs().square().flatten(-3,-2)
    energy=flat.abs().square().mean(-1)
    pas=torch.fft.fft2(x,dim=(-3,-2),norm="ortho").abs().square().flatten(-3,-2)
    return dict(pas=pas,pdp=pdp,energy=energy,share=norm(energy,(-1,)),joint=norm(pdp,(-2,-1)),
        panel_valid=total[...,0,0,0]>0,link_valid=energy>0,reference_valid=valid,
        coherence=cross.abs()/den)

def cosine(a,b,dim):
    den=(a.square().sum(dim)*b.square().sum(dim)).clamp_min(EPS).sqrt()
    return ((a*b).sum(dim)/den).clamp(0,1)

def aggregate(x,valid):
    while valid.ndim<x.ndim:valid=valid[...,None]
    valid=valid.expand_as(x)
    if x.ndim>2:
        axes=tuple(range(2,x.ndim));count=valid.sum(axes)
        x=(x*valid).sum(axes)/count.clamp_min(1);valid=count>0
    count=valid.sum(0)
    panel=(x*valid).sum(0)/count.clamp_min(1)
    available=count>0
    if not bool(available.any()):raise ValueError("No valid reference")
    return panel[available].mean()

def metrics(prediction,truth):
    p=describe(prediction);t=describe(truth.detach())
    pas=cosine(p["pas"],t["pas"],-2)
    pdp=cosine(p["pdp"],t["pdp"],-1)
    pv=p["link_valid"]&p["reference_valid"][...,None]
    pdp=torch.where(pv,pdp,torch.zeros_like(pdp))
    energy=.5*(p["share"]-t["share"]).abs().sum(-1)
    energy=torch.where(p["panel_valid"],energy,torch.ones_like(energy))
    joint=.5*(p["joint"]-t["joint"]).abs().sum((-2,-1))
    joint=torch.where(p["panel_valid"]&p["reference_valid"],joint,torch.ones_like(joint))
    return dict(pas_cosine=aggregate(pas,t["pas"].sum(-2)>0),
        relative_pdp_cosine=aggregate(pdp,t["link_valid"]&t["reference_valid"][...,None]),
        receive_energy_TV=aggregate(energy,t["panel_valid"]),
        receive_delay_joint_TV=aggregate(joint,t["panel_valid"]&t["reference_valid"]))

def objective(channel,truth,base,power,residual,variant):
    from measured.hall_structured_v2.observables import losses
    terms=losses(channel,truth,power)
    old=terms["structural"]+.1*terms["power"]+1e-4*residual
    if variant=="old":return old
    m=metrics(channel,truth)
    value=old+(1-m["relative_pdp_cosine"])
    if variant=="joint":value=value+(1-m["pas_cosine"])+.25*m["receive_delay_joint_TV"]
    elif variant!="pdp":raise ValueError(variant)
    return value
