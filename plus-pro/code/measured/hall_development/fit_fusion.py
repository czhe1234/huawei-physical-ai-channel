"""Phase-invariant calibration of actual complex carrier-bank coefficients."""
from __future__ import annotations
import time,json,traceback
import numpy as np
import torch
from .common import *
from measured.hall_adapters.metrics import training_power_loss
from measured.hall_adapters.foundation import expert_configs
def run():
    start=time.monotonic();seed();dev=DevelopmentData();bank=dev.fit()
    y=normalized_power(directional_power(bank.channel));yv=normalized_power(directional_power(dev.reference("validation")))
    with np.load(WORK/"expert_gram.npz") as z:g=z["gram"];ep=z["expert_power"];rows=z["row_ids"]
    n=len(y);np.testing.assert_array_equal(rows,np.r_[dev.fit_ids,dev.val_ids])
    g=torch.tensor(g);yt=torch.tensor(y,dtype=torch.float32)
    candidates=[];coef=np.zeros((17,2),np.complex64);coef[0]=1
    initial=torch.tensor(coef)
    def powers(c):
        return torch.einsum("ep,npbef,fp->npb",c,g,c.conj()).real.clamp_min(0)
    def evaluate(c):return powers(c).detach().numpy()
    for lr in [.01,.03]:
        parameter=torch.nn.Parameter(torch.tensor(coef));optimizer=torch.optim.Adam([parameter],lr=lr)
        curve=[];best=-1;best_c=None;best_epoch=0
        for step in range(501):
            pred=powers(parameter)
            if step%10==0:
                array=pred.detach().numpy();mf=metrics(array[:n],y);mv=metrics(array[n:],yv)
                row=dict(step=step,fit=mf,validation=mv);curve.append(row)
                if mv["mean_direction_cosine"]>best+.0002:
                    best=mv["mean_direction_cosine"];best_c=parameter.detach().numpy().copy();best_epoch=step
            if step==500:break
            optimizer.zero_grad();loss=training_power_loss(pred[:n],yt)+.001*(parameter-initial).abs().square().mean()
            loss.backward()
            if not torch.isfinite(parameter.grad).all():raise FloatingPointError("Fusion gradient")
            optimizer.step()
        p=evaluate(torch.tensor(best_c))
        candidate=dict(lr=lr,best_step=best_epoch,fit=metrics(p[:n],y),validation=metrics(p[n:],yv),curve=curve,
                       coefficient_real=best_c.real.tolist(),coefficient_imag=best_c.imag.tolist())
        candidates.append(candidate)
    chosen=max(candidates,key=lambda v:v["validation"]["mean_direction_cosine"])
    c=np.array(chosen["coefficient_real"])+1j*np.array(chosen["coefficient_imag"])
    np.save(WORK/"fusion_coefficient.npy",c.astype(np.complex64))
    source=np.load(WORK/"experts.npy",mmap_mode="r")
    base=np.lib.format.open_memmap(WORK/"foundation.npy",mode="w+",dtype=np.complex64,shape=(len(rows),2,8,4,1024))
    for i in range(len(rows)):base[i]=np.einsum("ephvf,ep->phvf",source[i],c).astype(np.complex64)
    base.flush()
    # Independent direct FFT verifies the quadratic optimization measurement operator.
    full=directional_power(base);quadratic=evaluate(torch.tensor(c.astype(np.complex64)))
    error=float(np.max(np.abs(normalized_power(full)-normalized_power(quadratic))))
    if error>2e-5:raise ValueError("Gram/complex writeback inconsistency")
    expert_metrics=[dict(name=e.name,fit=metrics(ep[:n,k],y),validation=metrics(ep[n:,k],yv)) for k,e in enumerate(expert_configs())]
    save_json(REPORT/"fusion.json",dict(complete=True,candidates=candidates,selected={k:v for k,v in chosen.items() if k!="curve"},
       individual_experts=expert_metrics,quadratic_vs_complex_power_max_absolute_error=error,seconds=time.monotonic()-start,
       objective="Fit-only normalized power cosine on complex coefficients; no phase/time fitting to held-out CSI.",test_used=False))
    dev.save_access("fusion");print("FUSION_DONE",json.dumps({k:chosen[k] for k in ["lr","best_step","fit","validation"]}),flush=True)
if __name__=="__main__":
    try:run()
    except BaseException:
        save_json(REPORT/"fusion_failure.json",dict(traceback=traceback.format_exc()));raise
