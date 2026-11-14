"""Saved refit weights on exactly replayed recent training scalar labels; no updates."""
from __future__ import annotations
import argparse,json,time
from collections import deque
from pathlib import Path
import numpy as np
import torch
from project_paths import PATHS
from . import common as C
from .saved_scale_diagnosis import stats,check_sources
ROOT=C.REPORT/"small_output_diagnosis_20261101"
class SynthesisTrace:
    """Observe native raw/rho only. Return the untouched native synthesis output."""
    def __init__(self,native):
        self.native=native;self.speedOfLight=native.speedOfLight;self.records=[];self.label=None
    def synthesize(self,raw,z,fc,groups=None):
        if len(self.records)<16:
            with torch.no_grad():
                # Bound diagnostic reductions to first256 actual rays per call.
                r=raw[:256];zz=z[:256];rho=torch.relu(r[...,-1])
                dz=zz[...,1:]-zz[...,:-1]
                dz=torch.cat((dz,torch.full_like(dz[...,:1],1e-10)),-1)
                alpha=1-torch.exp(-rho*dz)
                vals=rho.flatten();signal=torch.tanh(r[...,:2])
                q=torch.quantile(vals.float(),torch.tensor([0.,.5,1.],device=vals.device))
                # Exclusive first element0 is native source convention, kept unchanged.
                transmission=torch.cumprod(1-alpha+1e-10,dim=1)
                exclusive=torch.cat((torch.zeros_like(transmission[:,:1]),transmission[:,:-1]),1)
                weights=alpha*exclusive
                optical=(rho*dz).sum(-1)
                self.records.append(dict(label=self.label,rays=len(r),rho_min_median_max=q.cpu().tolist(),
                    total_optical_depth_mean=float(optical.mean()),total_optical_depth_max=float(optical.max()),
                    alpha_max=float(alpha.max()),sum_volume_weights_mean=float(weights.sum(-1).mean()),
                    raw_signal_abs_mean=float(signal.square().sum(-1).sqrt().mean()),
                    first_alpha_mean=float(alpha[:,0].mean())))
        return self.native.synthesize(raw,z,fc,groups)
def main(args):
    t=time.monotonic();gpu=C.device_guard(args.gpu)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=ROOT/(args.method+"_recent_training.json")
    if out.exists():raise FileExistsError(out)
    if args.method=="nerf2":
        from .nerf2_runner_v3 import build
    else:
        from .newrf_train_v4 import make
        from .newrf_native_batch_v4 import station_forward,station_target
        from .newrf_runner import subset_forward
        from .newrf_query import subset_query
    version="v3" if args.method=="nerf2" else "v4";records=[]
    for outer in range(5):
        data=C.FoldData(outer,"refit",args.method)
        path=C.ART/args.method/version/f"outer{outer}/refit/final.pt"
        before=C.sha256(path);ck=torch.load(path,map_location="cpu",weights_only=False)
        check_sources(args.method,ck)
        if args.method=="nerf2":model=build(data);cache=None;model.load_state_dict(ck["model"],strict=True)
        else:model,cache=make(data,ck["config"]);model.load_state_dict(ck["state_dict"],strict=True)
        model.eval()
        for p in model.parameters():p.requires_grad_(False)
        rng=np.random.default_rng(42+outer*(1009 if args.method=="nerf2" else 1000))
        batches=deque(maxlen=32);coverage=np.zeros((4000,4),np.int64)
        field_steps=np.zeros(8,np.int64)
        allowed=[data.fit_ids[data.sites(data.fit_ids)==s] for s in (0,1)]
        for step in range(1,ck["step"]+1):
            if args.method=="nerf2":
                s=int(rng.integers(2));b=int(rng.integers(2));u=int(rng.integers(2))
                rows=rng.choice(allowed[s],size=min(2,len(allowed[s])),replace=False)
                mm=np.sort(rng.choice(128,size=8,replace=False));ff=np.sort(rng.choice(192,size=4,replace=False))
                batches.append(dict(step=step,s=s,b=b,u=u,rows=rows,mm=mm,ff=ff))
                coverage[rows,b*2+u]+=1;field_steps[s*4+b*2+u]+=1
            else:
                rows=rng.choice(data.fit_ids,size=ck["config"]["batch_positions"],replace=True)
                field=(step-1)%8;mm=rng.integers(0,128,len(rows));ff=rng.integers(0,192,len(rows))
                batches.append(dict(step=step,field=field,rows=rows,mm=mm,ff=ff))
                field_steps[field]+=1
        if rng.bit_generator.state!=ck["rng_state"]:raise AssertionError("Training RNG replay mismatch")
        predictions=[];targets=[];coarses=[];query=[];identities=[]
        if args.method=="newrf":
            trace=SynthesisTrace(model.synthesizer);model.synthesizer=trace
        with torch.no_grad():
            for batch in batches:
                rows=batch["rows"];mm=batch["mm"];ff=batch["ff"]
                if args.method=="nerf2":
                    s,b,u=(batch[k] for k in ("s","b","u"))
                    pp=torch.tensor(data.positions(rows),device="cuda:0",dtype=torch.float64)
                    sites=torch.tensor(data.sites(rows),device="cuda:0",dtype=torch.long)
                    world=data.geometry.ue_world_positions(pp,sites)[:,u*2:(u+1)*2]
                    tx=model.bound_bs_ports[s,b].reshape(128,3)[torch.tensor(mm,device="cuda:0")]
                    f=torch.tensor(data.frequencies(ff),device="cuda:0",dtype=torch.float64)
                    pred=model.fields[f"s{s}_b{b}_u{u}"](tx,world,f).permute(1,0,2,3)*model.log_output_gain.exp()
                    y=data.read_fit(rows,ff,normalized=True)[:,b*128+mm,u*2:(u+1)*2]
                    p=pred.cpu().numpy()*data.scale;y=y*data.scale
                else:
                    field=batch["field"];trace.label=dict(step=batch["step"],field=field)
                    fine,coarse=station_forward(model,cache,data,rows,field,mm,ff)
                    y=station_target(data,rows,field,mm,ff,"cuda:0").cpu().numpy()
                    p=fine.cpu().numpy();coarses.append(coarse.cpu().numpy().reshape(-1))
                predictions.append(p.reshape(-1));targets.append(y.reshape(-1))
                identities.append({k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in batch.items()})
            p=np.concatenate(predictions);y=np.concatenate(targets)
            rec=dict(outer=outer,step=ck["step"],checkpoint=PATHS.relative(path),checkpoint_sha256=before,
                exact_training_RNG_replay=True,recent_updates=32,per_field_updates=field_steps.tolist(),
                metrics=stats(p[None,:],y[None,:],np.array([outer])),batches=identities)
            if args.method=="nerf2":
                rec["position_pol_pairs_never_sampled_fraction"]=float((coverage[data.fit_ids]==0).mean())
                rec["gain"]=float(model.log_output_gain.exp())
            else:
                rec["coarse_metrics"]=stats(np.concatenate(coarses)[None,:],y[None,:],np.array([outer]))
                rec["synthesis_trace_first16_calls"]=trace.records
                # Same actually supervised FIT row/field/BS/F, switch only direction route.
                last=batches[-1];row=last["rows"][:1];spatial=last["mm"][:1];ff=last["ff"][:1];field=last["field"]
                fine,_=subset_forward(model,cache,data,row,field,spatial,ff)
                b,u=divmod(field,4)
                actual=subset_query(model,data,row,b*128+spatial,np.array([u]),ff)
                gt=data.read_fit(row,ff,normalized=False)[:,b*128+spatial,u:u+1]
                rec["one_supervised_scalar_FIT_route"]=stats(fine.cpu().numpy().reshape(1,-1),gt.reshape(1,-1),row)
                rec["one_supervised_scalar_query_route"]=stats(actual.cpu().numpy().reshape(1,-1),gt.reshape(1,-1),row)
            if C.sha256(path)!=before:raise AssertionError("Checkpoint mutated")
            rec["checkpoint_preserved"]=True;records.append(rec)
            C.atomic_json(ROOT/(args.method+"_recent_training_progress.json"),dict(records=records,status="running"))
            print(json.dumps(dict(method=args.method,outer=outer,step=ck["step"],metrics=rec["metrics"],seconds=time.monotonic()-t)),flush=True)
        del model,cache,ck;torch.cuda.empty_cache()
    C.atomic_json(out,dict(status="complete",records=records,GPU=gpu,no_training=True,optimizer_created=False,
        outer_truth_read=False,source_sha256=C.sha256(Path(__file__)),elapsed_seconds=time.monotonic()-t))
if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--method",choices=["nerf2","newrf"],required=True);a.add_argument("--gpu",required=True)
    main(a.parse_args())
