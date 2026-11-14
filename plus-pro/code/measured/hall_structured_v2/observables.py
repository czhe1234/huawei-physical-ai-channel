"""Observable structure and fixed-reference losses, never label alignment."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from .config import LAGS
EPS=1e-30
def probability(x,dims):
    x=x.clamp_min(0);total=x.sum(dims,keepdim=True)
    return torch.where(total>0,x/total.clamp_min(EPS),torch.zeros_like(x))
def total_energy(h):
    return h.abs().square().sum((-3,-2,-1))/h.shape[-1]
def unit_channel(h):
    energy=h.abs().square().sum((-3,-2,-1),keepdim=True)
    return h/energy.clamp_min(EPS).sqrt()
def beam_channel(h):
    return torch.fft.fft2(h,dim=(-3,-2),norm="ortho").flatten(-3,-2)
def angle_frequency(h,subbands):
    z=beam_channel(h);f=z.shape[-1]
    if f%subbands:raise ValueError("Subbands must divide the full physical frequency axis")
    a=z.abs().square().reshape(*z.shape[:-1],subbands,f//subbands).sum(-1)
    return probability(a,(-2,-1))
def covariance(h):
    x=unit_channel(h).flatten(-3,-2)
    r=x@x.conj().transpose(-1,-2)
    return r
def correlation(x,lags=LAGS,dimensional_sum=False):
    values=[];valid=[]
    for lag in lags:
        if lag>=x.shape[-1]:continue
        a,b=x[...,lag:],x[...,:-lag]
        dims=(-2,-1) if dimensional_sum else (-1,)
        cross=(a*b.conj()).sum(dims)
        ea=a.abs().square().sum(dims);eb=b.abs().square().sum(dims)
        den=(ea*eb).clamp_min(EPS).sqrt()
        ok=(ea>0)&(eb>0)
        values.append(torch.where(ok,(cross.abs()/den).clamp(0,1),torch.zeros_like(ea)))
        valid.append(ok)
    return torch.stack(values,-1),torch.stack(valid,-1)
def structure(h,subbands=16,lags=LAGS):
    if h.ndim!=5 or h.shape[1:4]!=(2,8,4):raise ValueError("Expected[N,2,8,4,F]")
    if not torch.isfinite(h).all():raise ValueError("Nonfinite channel")
    x=unit_channel(h);z=beam_channel(x)
    c,cv=correlation(x.flatten(-3,-2),lags,True)
    cb,bv=correlation(z,lags)
    return dict(q=angle_frequency(x,subbands),broad=angle_frequency(x,1)[...,0],
                covariance=covariance(x),correlation=c,correlation_valid=cv,
                beam_correlation=cb,beam_correlation_valid=bv,
                energy=total_energy(h),panel_valid=total_energy(h)>0)
@dataclass
class FitPower:
    reference:torch.Tensor
    scale_db:torch.Tensor
    fit_ids:tuple
    @classmethod
    def from_fit(cls,channel,fit_ids):
        e=total_energy(torch.as_tensor(np.array(channel),dtype=torch.complex64))
        if not (e>0).all():raise ValueError("FIT power normalization needs valid nonzero measured panels")
        ref=e.median(0).values
        t=10*torch.log10(e/ref)
        return cls(ref,t.std(0,unbiased=False).clamp_min(1.),tuple(int(x) for x in fit_ids))
    def to(self,device):return FitPower(self.reference.to(device),self.scale_db.to(device),self.fit_ids)
    def relative_db(self,h):
        return 10*torch.log10(total_energy(h).clamp_min(EPS)/self.reference.to(h.device))
def _mean_valid(value,valid):
    return value[valid].mean() if valid.any() else value.sum()*0
def losses(pred,truth,power,scales=(1,4,16)):
    if pred.shape!=truth.shape:raise ValueError("Prediction/reference shapes differ")
    a,b=structure(pred,max(scales)),structure(truth,max(scales));valid=b["panel_valid"]
    failed=valid&~a["panel_valid"]
    angle=pred.real.sum()*0
    for s in scales:
        qp,qt=angle_frequency(pred,s),angle_frequency(truth,s)
        # sqrt at zero has no defined derivative; a tiny floor preserves stability.
        per=.5*(qp.clamp_min(1e-12).sqrt()-qt.clamp_min(1e-12).sqrt()).square().sum((-2,-1))
        angle=angle+_mean_valid(torch.where(failed,torch.ones_like(per),per),valid)/len(scales)
    cov=.5*(a["covariance"]-b["covariance"]).abs().square().sum((-2,-1))
    spatial=_mean_valid(torch.where(failed,torch.ones_like(cov),cov),valid)
    panel_err=(a["correlation"]-b["correlation"]).square()
    panel_err=torch.where(b["correlation_valid"]&~a["correlation_valid"],torch.ones_like(panel_err),panel_err)
    panel=_mean_valid(panel_err,b["correlation_valid"]&valid[...,None])
    be=(a["beam_correlation"]-b["beam_correlation"]).square()
    be=torch.where(b["beam_correlation_valid"]&~a["beam_correlation_valid"],torch.ones_like(be),be)
    mask=b["beam_correlation_valid"]
    perbeam=(be*mask).sum(-1)/mask.sum(-1).clamp_min(1)
    weighted=(perbeam*b["broad"]).sum(-1)
    freq=.5*(panel+_mean_valid(weighted,valid))
    tp=power.relative_db(pred);tt=power.relative_db(truth)
    scale=power.scale_db.to(pred.device)
    lp=torch.nn.functional.huber_loss(tp/scale,tt/scale,reduction="none")
    power_loss=_mean_valid(lp,valid)
    return dict(angle=angle,spatial=spatial,frequency=freq,power=power_loss,
                structural=(angle+spatial+freq)/3,failed_panels=failed.sum())
@torch.no_grad()
def metrics(pred,truth,power):
    a,b=structure(pred),structure(truth);valid=b["panel_valid"];failed=valid&~a["panel_valid"]
    den=a["broad"].norm(dim=-1)*b["broad"].norm(dim=-1)
    cosine=(a["broad"]*b["broad"]).sum(-1)/den.clamp_min(EPS)
    cr=(a["covariance"]-b["covariance"]).abs().square().sum((-2,-1))
    cr=cr/b["covariance"].abs().square().sum((-2,-1)).clamp_min(EPS)
    panel_fail=(b["correlation_valid"]&~a["correlation_valid"]).sum()
    beam_fail=(b["beam_correlation_valid"]&~a["beam_correlation_valid"]).sum()
    pmae=None if panel_fail else float(_mean_valid((a["correlation"]-b["correlation"]).abs(),b["correlation_valid"]&valid[...,None]))
    be=(a["beam_correlation"]-b["beam_correlation"]).abs()
    m=b["beam_correlation_valid"];per=(be*m).sum(-1)/m.sum(-1).clamp_min(1)
    bmae=None if beam_fail else float(_mean_valid((per*b["broad"]).sum(-1),valid))
    terms=losses(pred,truth,power)
    distances={}
    for count in (1,4,16):
        qp,qt=angle_frequency(pred,count),angle_frequency(truth,count)
        error=.5*(qp.clamp_min(1e-12).sqrt()-qt.clamp_min(1e-12).sqrt()).square().sum((-2,-1))
        distances[str(count)]=float(_mean_valid(torch.where(failed,torch.ones_like(error),error),valid))
    beam=a["broad"].argmax(-1)
    received=b["broad"].gather(-1,beam[...,None])[...,0];best=b["broad"].max(-1).values
    db=10*torch.log10(best/received)
    values=db[valid];infinite=bool(torch.isinf(values).any())
    beam_available=bool(valid.any()) and not bool(failed.any())
    p90=values.sort().values[max(0,int(np.ceil(.9*len(values)))-1)] if len(values) else None
    power_error=(power.relative_db(pred)-power.relative_db(truth)).abs()
    by_panel=[float(_mean_valid(power_error[:,j],valid[:,j])) for j in range(2)]
    lag_mae=[]
    for j in range(a["correlation"].shape[-1]):
        fail=(b["correlation_valid"][...,j]&~a["correlation_valid"][...,j]).any()
        lag_mae.append(None if fail else float(_mean_valid((a["correlation"][...,j]-b["correlation"][...,j]).abs(),b["correlation_valid"][...,j]&valid)))
    return dict(angle_frequency_hellinger=distances,frequency_MAE_by_lag=lag_mae,
      relative_CSI_power_MAE_db_by_panel=by_panel,
      mean_beam_loss_db=None if infinite or not beam_available else float(values.mean()),
      infinite_mean_beam_loss=infinite,p90_beam_loss_db=None if not beam_available or torch.isinf(p90) else float(p90),
       beam_percentile="nearest rank ceil(0.9*N); every valid reference panel",
      beam_loss_over3_fraction=float((values>3).float().mean()) if beam_available else None,
      valid_reference_panels=int(valid.sum()),failed_prediction_panels=int(failed.sum()),
      direction_cosine=float(_mean_valid(cosine,valid)),covariance_NMSE=float(_mean_valid(cr,valid)),
      frequency_correlation_MAE=pmae,beam_frequency_correlation_MAE=bmae,
      failed_panel_correlation_cells=int(panel_fail),failed_beam_correlation_cells=int(beam_fail),
      relative_CSI_power_MAE_db=float(_mean_valid((power.relative_db(pred)-power.relative_db(truth)).abs(),valid)),
      structural_selection=float(terms["structural"]),reference_mask_depends_on_prediction=False)
