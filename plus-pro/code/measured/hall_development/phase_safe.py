"""Phase-safe measured adaptation of the final carrier-bank families.

Interpolate sqrt directional power before mixing different measurement phases.
Keep a measured nearest-donor complex carrier and impose predicted magnitudes.
No query-channel phase/time is estimated or used.
"""
from __future__ import annotations
import time,json,traceback
import numpy as np
import torch
from .common import *
from .method import angular_writeback
from measured.hall_adapters.foundation import affine_weights
from measured.hall_adapters.metrics import training_power_loss
def run():
    start=time.monotonic();seed();dev=DevelopmentData();bank=dev.fit()
    rows=np.r_[dev.fit_ids,dev.val_ids];positions=dev.data.meta["position"][rows]
    y=normalized_power(directional_power(bank.channel));yv=normalized_power(directional_power(dev.reference("validation")))
    save_json(REPORT/"phase_safe_contract.json",dict(reason="Complex expert fusion fits but underperforms power learners on spatial validation. Unknown inter-position TX phases make coherent averaging an avoidable sensitivity; test power-first interpolation as one targeted measured adaptation, not a proof of that causal explanation.",
      same_rows=rows.tolist(),same_outer_split=True,same_inner_block_crossfit=True,
      families=["nearest","sqrt_power_IDW_k4_8_16_24","sqrt_power_affine_k8_16_24_32","sqrt_power_steered_affine_k8_16_24_32"],
      radial_families_collapsed="A donor-wide radial phase does not alter its power; duplicate power experts are removed.",
      fusion_lr=[.03,.1],steps=500,core_config=dict(bounds=[.5,.25,0.],gate=.3,lr=.0003),
      query_phase_alignment=False,carrier_phase_source="Nearest permitted measured training donor only.",test_used=False,north_used=False))
    block=np.floor(bank.position[:,:2]/4).astype(int);power=np.zeros((len(rows),13,2,32),np.float32);nearest=[]
    labels=["nearest"]+["sqrt_idw%d"%k for k in [4,8,16,24]]+["sqrt_affine%d"%k for k in [8,16,24,32]]+["sqrt_steered_affine%d"%k for k in [8,16,24,32]]
    for i,(rid,pos) in enumerate(zip(rows,positions)):
        distance=np.linalg.norm(bank.position[:,:2]-pos[:2],axis=-1)
        allowed=~np.all(block==np.floor(pos[:2]/4),axis=-1) if i<len(y) else np.ones(len(distance),bool)
        order=np.flatnonzero(allowed);order=order[np.lexsort((bank.row_ids[order],distance[order]))]
        chosen=order[:32];nearest.append(int(chosen[0]));raw=np.sqrt(y[chosen])
        power[i,0]=y[chosen[0]]
        for e,k in enumerate([4,8,16,24],start=1):
            take=min(k,len(chosen));w=1/np.maximum(distance[chosen[:take]],1e-6)**2;w/=w.sum()
            power[i,e]=np.einsum("k,kpb->pb",w,raw[:take])**2
        steered=[]
        for panel in range(2):
            center=bank.spec.center[panel];ranges=np.linalg.norm(bank.position[chosen]-center,axis=-1)
            u=(bank.position[chosen]-center)/ranges[:,None];uq=(pos-center)/np.linalg.norm(pos-center)
            phase=2*np.pi*bank.spec.carrier_hz/299792458.*np.einsum("ki,hvi->khv",uq[None]-u,bank.spec.element_offsets[panel])
            h=bank.channel[chosen,panel]*np.exp(1j*phase)[...,None]
            p=(np.abs(np.fft.fft2(h,axes=(1,2),norm="ortho"))**2).mean(-1).reshape(len(h),32)
            steered.append(np.sqrt(p/p.sum(-1,keepdims=True)))
        steer=np.stack(steered,axis=1)
        for j,k in enumerate([8,16,24,32]):
            take=min(k,len(chosen));ids=chosen[:take]
            w=affine_weights(bank.position[ids,:2]-pos[:2],distance[ids],.3)
            power[i,5+j]=np.maximum(np.einsum("k,kpb->pb",w,raw[:take]),0)**2
            power[i,9+j]=np.maximum(np.einsum("k,kpb->pb",w,steer[:take]),0)**2
        power[i]/=np.maximum(power[i].sum(-1,keepdims=True),1e-20)
    np.savez_compressed(WORK/"phase_safe_experts.npz",row_ids=rows,power=power,nearest_source_index=nearest)
    pt=torch.tensor(power);yt=torch.tensor(y,dtype=torch.float32);n=len(y);candidates=[]
    for lr in [.03,.1]:
        logit=torch.nn.Parameter(torch.zeros((13,2)));opt=torch.optim.Adam([logit],lr=lr);curve=[];best=-1;weight=None;best_step=0
        for step in range(501):
            weights=torch.softmax(logit,dim=0)
            pred=torch.einsum("nepb,ep->npb",pt,weights)
            if step%10==0:
                pf=pred.detach().numpy();mf=metrics(pf[:n],y);mv=metrics(pf[n:],yv)
                curve.append(dict(step=step,fit=mf,validation=mv))
                if mv["mean_direction_cosine"]>best+.0002:best=mv["mean_direction_cosine"];weight=weights.detach().numpy().copy();best_step=step
            if step==500:break
            opt.zero_grad();loss=training_power_loss(pred[:n],yt)+.0001*weights.square().sum();loss.backward();opt.step()
        p=np.einsum("nepb,ep->npb",power,weight)
        candidates.append(dict(lr=lr,best_step=best_step,weights=weight.tolist(),fit=metrics(p[:n],y),validation=metrics(p[n:],yv),curve=curve))
    selected=max(candidates,key=lambda r:r["validation"]["mean_direction_cosine"]);weights=np.array(selected["weights"])
    desired=np.einsum("nepb,ep->npb",power,weights)
    carrier=np.array(bank.channel[np.array(nearest)],copy=True)
    h=angular_writeback(torch.tensor(carrier.reshape(-1,1,8,4,1024)),torch.tensor(desired.reshape(-1,32),dtype=torch.float32))
    h=h.numpy().reshape(len(rows),2,8,4,1024);actual=normalized_power(directional_power(h))
    error=float(np.max(np.abs(actual-normalized_power(desired))))
    if error>2e-5:raise ValueError("Phase-safe carrier magnitude writeback mismatch")
    np.save(WORK/"foundation_phase_safe.npy",h)
    save_json(REPORT/"phase_safe_fusion.json",dict(complete=True,candidates=candidates,selected={k:v for k,v in selected.items() if k!="curve"},expert_names=labels,
      reconstruction_power_error=error,seconds=time.monotonic()-start,query_label_inputs=False,absolute_channel_claim=False,
      nearest_source_rows=bank.row_ids[np.array(nearest)].tolist(),individual_experts=[dict(name=labels[i],fit=metrics(power[:n,i],y),validation=metrics(power[n:,i],yv)) for i in range(13)]))
    dev.save_access("phase_safe");print("PHASE_SAFE_DONE",json.dumps({k:selected[k] for k in ["lr","best_step","fit","validation"]}),flush=True)
if __name__=="__main__":
    try:run()
    except BaseException:
        save_json(REPORT/"phase_safe_failure.json",dict(traceback=traceback.format_exc()));raise
