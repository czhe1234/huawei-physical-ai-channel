"""Restore nGRF batch-global magnitude NMSE with explicit0.1 complex-NMSE H supervision.

All architecture/Gaussian/scale/optimizer settings stay fixed. Outputs live in
ngrf/v2; earlier complex-MSE adapters remain separate diagnostic artifacts.
The source objective cannot identify complex phase uniquely.
"""
from __future__ import annotations
import argparse,json,math,time,traceback
from pathlib import Path
import numpy as np
import torch
from round2.paper_baseline_training_v1 import common,field_runner as legacy

ORIGINAL_PATHS=common.output_paths
ORIGINAL_IDENTITY=legacy.source_identity
ORIGINAL_BUILD=legacy.build_model


def paths(method,outer=None,role=None):
    if method!="ngrf":raise ValueError("v2 paths only support nGRF")
    base=ORIGINAL_PATHS(method);a=base["artifact"]/"v2";r=base["report"]/"v2"
    if outer is not None:a=a/f"outer{outer}";r=r/f"outer{outer}"
    if role is not None:a=a/role;r=r/role
    a.mkdir(parents=True,exist_ok=True);r.mkdir(parents=True,exist_ok=True)
    return dict(artifact=a,report=r,artifacts=a,reports=r)


def objective_metadata():
    return dict(version="native_objective_v2",training="batch-global magnitude NMSE +0.1 complex NMSE + author logit L1",
      formula="[sum((abs(pred)-abs(target))^2)+0.1*sum(abs(pred-target)^2)]/max(sum(abs(target)^2),1e-10) + .01603005049928609*mean(abs(activation_logits))",
      normalization="H divided by one FIT-only scalar RMS before the loss",
      zero_rows="kept in numerator; never filtered",
      all_zero_batch="denominator floored to1e-10, finite gradient; deliberate guard differs from source return-inf",
      source_nmse="utils/loss.py batch-global sums; train.py446-448 receives magnitudes",
      phase="explicit0.1 complex NMSE fixes target-task phase supervision; pure magnitude source loss alone is non-identifying",test_phase_correction=False,
      evaluation="official PAS/PDP/raw complex NMSE/score from actual emittedH; selection uses officialscore")


def identity():
    return {**ORIGINAL_IDENTITY(),"ngrf_native_objective_v2.py":common.sha256(Path(__file__))}


def build(*args,**kwargs):
    model=ORIGINAL_BUILD(*args,**kwargs);native_metadata=model.metadata
    model.metadata=lambda:{**native_metadata(),"objective":objective_metadata()}
    return model


def magnitude_nmse(value,target):
    numerator=(value.abs()-target.abs()).square().sum()
    denominator=target.abs().square().sum()
    return numerator/denominator.clamp_min(1e-10)


def training_step(method,model,data,cache,rng,points=32,tones=8,fit_pool=None):
    if method!="ngrf":raise ValueError("nGRF v2 objective only")
    ff=np.sort(rng.choice(192,size=tones,replace=False));rows=[]
    fit_ids=data.fit_ids if fit_pool is None else np.asarray(fit_pool,np.int64)
    for s in (0,1):
        allowed=fit_ids[data.sites(fit_ids)==s]
        rows.extend(rng.choice(allowed,size=min(points//2,len(allowed)),replace=False).tolist())
    rows=np.array(rows,np.int64)
    p=torch.tensor(data.positions(rows),device="cuda:0",dtype=torch.float64)
    site=torch.tensor(data.sites(rows),device="cuda:0",dtype=torch.int64)
    f=torch.tensor(data.frequencies(ff),device="cuda:0",dtype=torch.float64)
    target=cache.read(rows,ff);value=model(p,site,f)
    native=magnitude_nmse(value,target)
    reg=torch.stack([field.activation_penalty() for field in model.fields.values()]).mean()
    complex_nmse=(value-target).abs().square().sum()/target.abs().square().sum().clamp_min(1e-10)
    loss=native+.1*complex_nmse+.01603005049928609*reg
    return loss,dict(magnitude_nmse=float(native.detach()),complex_nmse=float(complex_nmse.detach()),mse=float((value-target).abs().square().mean().detach()),
       target_energy=float(target.abs().square().mean()),prediction_energy=float(value.abs().square().mean().detach()),
       activation_logit_L1=float(reg.detach()),point_count=len(rows),frequency_count=len(ff),scalar_labels=target.numel(),
       zero_target_rows=int((target.abs().square().sum((1,2,3))==0).sum()))


def install():
    common.output_paths=paths;legacy.output_paths=paths;legacy.source_identity=identity
    legacy.build_model=build;legacy.full_training_step=training_step


def monitor(model,data,ids):
    model.eval()
    with torch.no_grad():
        value=legacy.predict_normalized(model,data,ids)
        target=torch.tensor(data.read_fit(ids),device="cuda:0",dtype=torch.complex64)
        mag=float(magnitude_nmse(value,target));energy=target.abs().square().sum(dtype=torch.float64)
        return dict(magnitude_nmse=mag,complex_nmse=float((value-target).abs().square().sum(dtype=torch.float64)/energy.clamp_min(1e-10)),
          prediction_to_truth_energy=float(value.abs().square().sum(dtype=torch.float64)/energy.clamp_min(1e-10)),
          normalized_complex_correlation_real=float((value.conj()*target).real.sum(dtype=torch.float64)/torch.sqrt(value.abs().square().sum(dtype=torch.float64)*energy).clamp_min(1e-30)),
          zero_target_rows=int((target.abs().square().sum((1,2,3))==0).sum()),complete_frequency_count=192,
          activation_mean={s:float(f.native.get_attributes_and_activation(f.fixed_tx.float())[1].mean()) for s,f in model.fields.items()})


def phase_gradient_probe(model,data,cache,fixture):
    ids=np.asarray(fixture)[[0,3,8,11]];ff=np.array([0,64,128,191],np.int64)
    model.eval()
    value=model(torch.tensor(data.positions(ids),device="cuda:0",dtype=torch.float64),torch.tensor(data.sites(ids),device="cuda:0",dtype=torch.int64),torch.tensor(data.frequencies(ff),device="cuda:0",dtype=torch.float64))
    target=cache.read(ids,ff);loss=(value-target).abs().square().sum()/target.abs().square().sum().clamp_min(1e-10)
    grad=torch.autograd.grad(loss,tuple(model.parameters()),allow_unused=True)
    return dict(complex_nmse=float(loss),gradient_norm=sum(float(g.double().square().sum()) for g in grad if g is not None)**.5,finite=all(g is None or bool(torch.isfinite(g).all()) for g in grad),parameter_update=False)


def health(args):
    gpu=common.device_guard(legacy.GPU_UUID);common.seed_all(42);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=common.FoldData(0,"screen","ngrf")
    fixture=np.concatenate([x[np.linspace(0,len(x)-1,8,dtype=int)] for x in
       [data.fit_ids[data.sites(data.fit_ids)==s] for s in (0,1)]])
    model=build("ngrf",data);opt=legacy.optimizer_for("ngrf",model)
    cache=legacy.FitGPUCache(data,on_gpu=True);rng=np.random.default_rng(42)
    records=[dict(step=0,monitor=monitor(model,data,fixture))];source=identity();started=time.monotonic()
    out=paths("ngrf",0,"health")
    for step in range(1,args.health_steps+1):
        model.train();opt.zero_grad(set_to_none=True)
        legacy.apply_learning_rate("ngrf",opt,step,15000)
        loss,detail=training_step("ngrf",model,data,cache,rng,points=32,tones=8,fit_pool=fixture)
        if not bool(torch.isfinite(loss)):raise FloatingPointError("Nonfinite native nGRF loss")
        loss.backward()
        finite=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        if not finite:raise FloatingPointError("Nonfinite native nGRF gradient")
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step()
        if step%250==0 or step==args.health_steps:
            event=dict(step=step,monitor=monitor(model,data,fixture),training_batch=detail,finite_gradients=finite,
                       gradient_norm=float(grad),seconds=time.monotonic()-started)
            records.append(event);common.atomic_json(out["report"]/"progress.json",event);print(json.dumps(event),flush=True)
        if time.monotonic()-started>args.health_seconds:break
    final=monitor(model,data,fixture)
    phase_probe=phase_gradient_probe(model,data,cache,fixture)
    healthy=final["magnitude_nmse"]<.25 and .5<final["prediction_to_truth_energy"]<1.5 and final["normalized_complex_correlation_real"]>.1 and phase_probe["gradient_norm"]>0 and finite
    result=dict(status="healthy" if healthy else "not_yet_healthy",version="native_objective_v2",updates=step,
      FIT_ids=fixture.tolist(),records=records,final=final,phase_supervision_gradient=phase_probe,seconds=time.monotonic()-started,GPU=gpu,
      source_identity=source,architecture=model.metadata(),validation_labels_used=False,outer_truth_used=False,
      parameter_updates="FIT-only health; formal screen/refit start fresh",objective=objective_metadata())
    common.atomic_json(out["report"]/"result.json",result)
    common.atomic_torch_save(out["artifact"]/"health.pt",dict(model=model.state_dict(),info=result))
    print(json.dumps(dict(status=result["status"],final=final,updates=step)),flush=True)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--mode",choices=["health","main","health-then-main"],default="health")
    ap.add_argument("--health-steps",type=int,default=3000);ap.add_argument("--health-seconds",type=float,default=240.)
    ap.add_argument("--method",default="ngrf");ap.add_argument("--gpu-label-cache",action="store_true",default=True)
    ap.add_argument("--outers",default="0,1,2,3,4");ap.add_argument("--max-steps",type=int,default=15000)
    ap.add_argument("--min-steps",type=int,default=4000);ap.add_argument("--val-every",type=int,default=500)
    ap.add_argument("--patience",type=int,default=6);ap.add_argument("--min-delta",type=float,default=.0005)
    ap.add_argument("--medium-rays",type=int,default=128);ap.add_argument("--tx-chunk",type=int,default=16)
    args=ap.parse_args();install()
    try:
        if args.mode in ("health","health-then-main"):
            result=health(args)
            if result["status"]!="healthy" or args.mode=="health":return
        legacy.main_training(args)
    except BaseException:
        out=paths("ngrf",None,"failure");(out["report"]/"failure.txt").write_text(traceback.format_exc());raise

if __name__=="__main__":main()
