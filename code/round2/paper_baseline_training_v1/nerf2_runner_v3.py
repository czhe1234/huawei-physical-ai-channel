"""NeRF2 city-domain unit conditioning; original real/imag MSE supervision.

Only rawXYZ identity features are scaled. All sin/cos physical position and
frequency bands, source network layers,324 rays and32 samples remain fixed.
Extinction=softplus(raw)/far has units1/metre and initial optical depth1;
this is an explicit parameterization adaptation, not same-weight abs equivalence.
"""
import argparse,json,math,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from round2.paper_baseline_training_v1 import nerf2_runner_v2 as previous
from round2.paper_baseline_training_v1 import common,field_runner as legacy

LINEAR_XYZ_SCALE=426.831421/9.


def paths(outer=None,role=None):
    base=common.output_paths("nerf2");a=base["artifact"]/"v3";r=base["report"]/"v3"
    if outer is not None:a=a/f"outer{outer}";r=r/f"outer{outer}"
    if role is not None:a=a/role;r=r/role
    a.mkdir(parents=True,exist_ok=True);r.mkdir(parents=True,exist_ok=True)
    return dict(artifact=a,report=r)


def identity():
    return {**previous.identity(),"nerf2_runner_v3.py":common.sha256(Path(__file__))}


class PerMetreExtinction(nn.Linear):
    def forward(self,x):
        raw=F.linear(x,self.weight,self.bias)
        return torch.stack([F.softplus(raw[...,0])/810.,raw[...,1]],-1)


def condition_embedding(function):
    def conditioned(x):
        encoded=function(x)
        return torch.cat([encoded[...,:3]/LINEAR_XYZ_SCALE,encoded[...,3:]],-1)
    return conditioned


def build(data):
    model=previous.build(data)
    for core in model.fields.values():
        old=core.field.attenuation_output
        new=PerMetreExtinction(old.in_features,old.out_features,bias=True).to(device=old.weight.device,dtype=old.weight.dtype)
        new.load_state_dict(old.state_dict())
        with torch.no_grad():new.weight[0].zero_();new.bias[0].fill_(math.log(math.expm1(1.)))
        core.field.attenuation_output=new
        core.field.embed_pts_fn=condition_embedding(core.field.embed_pts_fn)
        core.field.embed_tx_fn=condition_embedding(core.field.embed_tx_fn)
    return model


def metadata():
    return {**previous.metadata(),"version":"city_conditioning_v3",
      "objective":"0.5*mean(abs(predH-targetH)^2), exact real/imag concatenated MSE source convention",
      "source_loss":"nerf2_runner.py153-156 + model.py12",
      "rawXYZ_identity_scale_m":LINEAR_XYZ_SCALE,
      "PE_sin_cos":"unchanged original physical metre/radian frequency functions; raw frequency feature unscaled",
      "extinction":"softplus(raw)/810 per metre, arbitrary positive density range, initial optical thickness1",
      "not_exact_same_weight_original_abs":True,"query_phase_correction":False}


def fixed_nonzero_positions(data,cache):
    energy=[]
    for first in range(0,len(data.fit_ids),128):
        ids=data.fit_ids[first:first+128]
        slots=torch.tensor(cache.slot[ids],device="cuda:0")
        energy.extend(cache.value.index_select(0,slots).abs().square().mean((1,2,3)).cpu().tolist())
    energy=np.asarray(energy);result=[]
    for s in (0,1):
        rows=data.fit_ids[(data.sites(data.fit_ids)==s)&(energy>0)]
        p=data.positions(rows);chosen=[int(np.argmin(np.linalg.norm(p-p.mean(0),axis=1)))]
        dist=np.linalg.norm(p-p[chosen[0]],axis=1)
        for _ in range(7):
            i=int(np.argmax(dist));chosen.append(i);dist=np.minimum(dist,np.linalg.norm(p-p[i],axis=1))
        result.extend(rows[chosen].tolist())
    return np.asarray(result,np.int64)


def monitor(model,data,fixture):
    model.eval()
    with torch.no_grad():
        value=previous.subset(model,data,fixture)
        h=data.read_fit(fixture,previous.FREQUENCY_IDS)[:,previous.BS_PORTS]
        target=torch.tensor(h,device="cuda:0",dtype=torch.complex64)
        se=(value-target).abs().square().sum((1,2,3),dtype=torch.float64)
        te=target.abs().square().sum((1,2,3),dtype=torch.float64)
        pe=value.abs().square().sum((1,2,3),dtype=torch.float64)
        corr=(value.conj()*target).real.sum(dtype=torch.float64)/torch.sqrt(pe.sum()*te.sum()).clamp_min(1e-30)
        return dict(complex_nmse=float(se.sum()/te.sum()),prediction_to_truth_energy=float(pe.sum()/te.sum()),
          normalized_complex_correlation_real=float(corr),per_position_nmse=(se/te).cpu().tolist(),
          per_position_energy_ratio=(pe/te).cpu().tolist(),shape=list(value.shape),zero_target_rows=int((te==0).sum()))


def fixture_step(model,data,cache,rng,fixture):
    s=int(rng.integers(2));b=int(rng.integers(2));u=int(rng.integers(2))
    allowed=fixture[data.sites(fixture)==s];ids=rng.choice(allowed,size=2,replace=False)
    mm=previous.BS_SPATIAL;ff=np.sort(rng.choice(previous.FREQUENCY_IDS,size=4,replace=False))
    p=torch.tensor(data.positions(ids),device="cuda:0",dtype=torch.float64)
    site=torch.tensor(data.sites(ids),device="cuda:0",dtype=torch.int64)
    world=data.geometry.ue_world_positions(p,site)[:,u*2:(u+1)*2]
    tx=model.bound_bs_ports[s,b].reshape(128,3)[torch.tensor(mm,device="cuda:0")]
    f=torch.tensor(data.frequencies(ff),device="cuda:0",dtype=torch.float64)
    value=model.fields[f"s{s}_b{b}_u{u}"](tx,world,f).permute(1,0,2,3)*model.log_output_gain.exp()
    target=cache.read(ids,ff)[:,b*128+torch.tensor(mm,device="cuda:0"),u*2:(u+1)*2]
    native=.5*(value-target).abs().square().mean()
    return native,dict(complex_mse=float(2*native.detach()),zero_mse=float(target.abs().square().mean()),scalar_labels=target.numel())


def health(args):
    started=time.monotonic();gpu=common.device_guard(legacy.GPU_UUID)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    common.seed_all(42);data=common.FoldData(0,"screen","nerf2")
    cache=legacy.FitGPUCache(data,on_gpu=True);fixture=fixed_nonzero_positions(data,cache)
    model=build(data);opt=previous.optimizer(model);rng=np.random.default_rng(42);source=identity()
    out=paths(0,"health");records=[dict(step=0,monitor=monitor(model,data,fixture))]
    for step in range(1,args.health_steps+1):
        model.train();opt.zero_grad(set_to_none=True)
        loss,detail=fixture_step(model,data,cache,rng,fixture)
        if not bool(torch.isfinite(loss)):raise FloatingPointError("Nonfinite city-conditioned NeRF2 loss")
        loss.backward();finite=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        if not finite:raise FloatingPointError("Nonfinite city-conditioned NeRF2 gradient")
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step()
        if step%250==0:
            event=dict(step=step,monitor=monitor(model,data,fixture),last_training_batch=detail,gradient_norm=float(grad),finite_gradients=finite,seconds=time.monotonic()-started)
            records.append(event);common.atomic_json(out["report"]/"progress.json",event);print(json.dumps(event),flush=True)
        if time.monotonic()-started>args.max_seconds:break
    final=monitor(model,data,fixture)
    result=dict(status="complete",kind="FIT-only16-position overfit health",version="city_conditioning_v3",updates=step,
      source_identity=source,records=records,final=final,FIT_ids=fixture.tolist(),BS_ports=previous.BS_PORTS.tolist(),
      frequency_ids=previous.FREQUENCY_IDS.tolist(),UE_port_count=4,config=metadata(),seconds=time.monotonic()-started,
      GPU=gpu,validation_labels_read=False,outer_truth_read=False,parameter_updates="FIT-only health, no formal model yet")
    common.atomic_json(out["report"]/"result.json",result)
    common.atomic_torch_save(out["artifact"]/"health.pt",dict(model=model.state_dict(),info=result))
    print(json.dumps(dict(status="complete",updates=step,final=final)),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--mode",choices=["health"],default="health")
    ap.add_argument("--health-steps",type=int,default=10000);ap.add_argument("--max-seconds",type=float,default=280.)
    args=ap.parse_args()
    try:health(args)
    except BaseException:
        out=paths(0,"health");(out["report"]/"failure.txt").write_text(traceback.format_exc());raise

if __name__=="__main__":main()
