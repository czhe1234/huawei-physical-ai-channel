"""Competition-matched PAS, independently referenced PDP and receive energy."""
from __future__ import annotations
import numpy as np
REFERENCE_COHERENCE_MIN=1e-6
def _validate(h):
    x=np.asarray(h)
    if x.ndim!=5 or x.shape[1:4]!=(2,8,4) or x.shape[-1]<2:raise ValueError("Expected[N,2,8,4,F]")
    if not np.iscomplexobj(x) or not np.isfinite(x).all():raise ValueError("Finite complex channels required")
    return x.astype(np.complex128,copy=False)
def normalize(x,axes):
    den=x.sum(axis=axes,keepdims=True)
    return np.divide(x,den,out=np.zeros_like(x),where=den>0)
def reference(h,bandwidth_hz=50e6):
    """One anchor for all32 elements of each panel, using this channel alone."""
    if not np.isfinite(bandwidth_hz) or bandwidth_hz<=0:raise ValueError("Positive physical bandwidth required")
    h=_validate(h);f=h.shape[-1];x=h.reshape(*h.shape[:2],32,f)
    a,b=x[...,1:],x[...,:-1]
    c=(a*b.conj()).sum(axis=(-2,-1))
    den=np.sqrt((np.abs(a)**2).sum(axis=(-2,-1))*(np.abs(b)**2).sum(axis=(-2,-1)))
    coherence=np.divide(np.abs(c),den,out=np.zeros_like(den),where=den>0)
    valid=(den>0)&(coherence>REFERENCE_COHERENCE_MIN)
    bins=-f*np.angle(c)/(2*np.pi)
    bins=np.where(valid,bins,0.)
    ramp=np.exp(2j*np.pi*bins[...,None]*np.arange(f)/f)
    return dict(channel=h*ramp[...,None,None,:],anchor_bins=bins,
       anchor_seconds=bins/bandwidth_hz,coherence=coherence,valid=valid)
def describe(h,bandwidth_hz=50e6):
    h=_validate(h);n,p=h.shape[:2];f=h.shape[-1]
    e=(np.abs(h)**2).mean(-1).reshape(n,p,32)
    angular=np.abs(np.fft.fft2(h,axes=(-3,-2),norm="ortho"))**2
    angular=angular.reshape(n,p,32,f)
    ref=reference(h,bandwidth_hz)
    pdp=np.abs(np.fft.fftshift(np.fft.ifft(ref["channel"],axis=-1,norm="ortho"),axes=-1))**2
    pdp=pdp.reshape(n,p,32,f)
    return dict(pas=angular,pdp=pdp,energy=e,energy_fraction=normalize(e,(-1,)),
        receive_delay_joint=normalize(pdp,(-2,-1)),panel_valid=e.sum(-1)>0,
        link_valid=e>0,reference=ref)
def cosine(x,y,axis):
    num=(x*y).sum(axis);den=np.sqrt((x*x).sum(axis)*(y*y).sum(axis))
    return np.clip(np.divide(num,den,out=np.zeros_like(num),where=den>0),0,1)
def _mean(value,mask):return float(value[mask].mean()) if mask.any() else None
def _aggregate(value,mask):
    # First average valid cells within each position/panel, then positions,
    # then give the two panels equal weight. Invalidity is reference-only.
    if value.ndim>2:
        axes=tuple(range(2,value.ndim));count=mask.sum(axis=axes)
        value=np.divide((value*mask).sum(axis=axes),count,out=np.zeros(count.shape),where=count>0)
        mask=count>0
    by_panel=[_mean(value[:,j],mask[:,j]) for j in range(value.shape[1])]
    available=[x for x in by_panel if x is not None]
    return (float(np.mean(available)) if available else None),by_panel
def compare(prediction,truth,bandwidth_hz=50e6):
    """No fitted pairwise phase/shift/scale; truth only defines reference masks."""
    p,t=describe(prediction,bandwidth_hz),describe(truth,bandwidth_hz)
    if p["pas"].shape!=t["pas"].shape:raise ValueError("Prediction/reference shape mismatch")
    pas_valid=t["pas"].sum(-2)>0
    pas=cosine(p["pas"],t["pas"],-2)
    pdp_valid=t["link_valid"]&t["reference"]["valid"][...,None]
    pdp=cosine(p["pdp"],t["pdp"],-1)
    pred_valid=p["link_valid"]&p["reference"]["valid"][...,None]
    pdp=np.where(pred_valid,pdp,0.)
    energy=.5*np.abs(p["energy_fraction"]-t["energy_fraction"]).sum(-1)
    energy=np.where(p["panel_valid"],energy,1.)
    joint=.5*np.abs(p["receive_delay_joint"]-t["receive_delay_joint"]).sum((-2,-1))
    joint=np.where(p["panel_valid"]&p["reference"]["valid"],joint,1.)
    joint_valid=t["panel_valid"]&t["reference"]["valid"]
    ps,panel_ps=_aggregate(pas,pas_valid);ds,panel_ds=_aggregate(pdp,pdp_valid)
    es,panel_es=_aggregate(energy,t["panel_valid"]);js,panel_js=_aggregate(joint,joint_valid)
    return dict(pas_cosine=ps,relative_pdp_cosine=ds,
        receive_energy_TV=es,receive_delay_joint_TV=js,
        by_panel=dict(pas_cosine=panel_ps,relative_pdp_cosine=panel_ds,
          receive_energy_TV=panel_es,receive_delay_joint_TV=panel_js),
        reference_valid_pas_cells=int(pas_valid.sum()),reference_valid_pdp_links=int(pdp_valid.sum()),
        reference_valid_energy_panels=int(t["panel_valid"].sum()),reference_valid_joint_panels=int(joint_valid.sum()),
        invalid_reference_anchors=int((t["panel_valid"]&~t["reference"]["valid"]).sum()),
        failed_prediction_pdp_links=int((pdp_valid&~pred_valid).sum()),
        failed_prediction_energy_panels=int((t["panel_valid"]&~p["panel_valid"]).sum()),
        prediction_oracle_alignment=False,combined_score=None)
