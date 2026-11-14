"""Train task-adapted NeRF2/nGRF on the fixed spatial roles, preserving V16."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

GPU_UUID="GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0"
if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("",GPU_UUID):
    raise RuntimeError("Mask the assigned full H100 UUID before importing CUDA")

import numpy as np
import torch
from torch import nn
from project_paths import PATHS
from round2.paper_baseline_adapters_v1.field_models import CompetitionNeRF2
from round2.paper_baseline_training_v1.field_fast import (
    enable_shared_medium,make_ngrf,ngrf_optimizer,NGRF_SOURCE_CONFIG,equivalence_check)
from round2.paper_baseline_training_v1.native_ngrf import NativeNGRF,native_optimizer
from round2.paper_baseline_training_v1.common import (
    FoldData,evaluate,atomic_json,atomic_npy,atomic_torch_save,output_paths,
    seed_all,device_guard,save_outer_prediction,sha256,digest_ids)

NERF2_CONFIG=dict(depth=8,width=256,skips=(4,),position_levels=10,direction_levels=10,
    transmitter_levels=10,encode_transmitter=True,n_rays=324,n_samples=32,near_m=0.,far_m=5.,
    world_scale_m=1.,reference_frequency_hz=6690000000.,frequency_scale_hz=5760000.,
    propagation_sign=-1,ray_chunk=4096,frequency_chunk=8)
SOURCE_TRAIN_CONFIG=dict(
    nerf2=dict(source_config="plus-pro/code/third_party/paper_baselines/nerf2/configs/mimo-csi.yml",
        lr=.001,weight_decay=.00005,source_total_iterations=70000),
    ngrf=dict(source_repository="https://github.com/muhd-umer/n-grf",
        source_commit="3765d6f1e46a223914908e38430b1ab9d2a5f4fb",
        source_config="configs/default_mimo_od.yaml",source_total_iterations=15000))


def build_model(method,data,*,medium_rays=128,tx_chunk=16,conditioned=False):
    if method=="nerf2":
        cfg={**NERF2_CONFIG,"world_center_m":tuple(np.mean(data.positions(data.fit_ids),axis=0))}
        model=CompetitionNeRF2(data.geometry,field_config=cfg)
        enable_shared_medium(model,medium_ray_chunk=medium_rays,tx_chunk=tx_chunk)
        if conditioned:
            model.register_parameter("log_output_gain",nn.Parameter(torch.tensor(-math.log(cfg["n_rays"]),dtype=torch.float32)))
    else:
        model=NativeNGRF(data.geometry,data.positions(data.fit_ids),data.sites(data.fit_ids))
        for field in model.fields.values():field.native.device=torch.device("cuda:0")
    return model.to(device="cuda:0")


def optimizer_for(method,model):
    if method=="ngrf":return native_optimizer(model)
    if hasattr(model,"log_output_gain"):
        parameters=[p for n,p in model.named_parameters() if n!="log_output_gain"]
        return torch.optim.Adam([dict(params=parameters,lr=.0001,weight_decay=.00005),
                                dict(params=[model.log_output_gain],lr=.001,weight_decay=0.)])
    return torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.00005)


def training_batch(method,model,data,rng,*,points=2,bs_links=4,tones=4,monitor=None):
    if monitor is None:
        site=int(rng.integers(2));bpol=int(rng.integers(2));upol=int(rng.integers(2))
        available=data.fit_ids[data.sites(data.fit_ids)==site]
        rows=rng.choice(available,size=min(points,len(available)),replace=False)
        ff=np.sort(rng.choice(192,size=tones,replace=False))
        mm=np.sort(rng.choice(128,size=bs_links,replace=False))
    else:
        site,bpol,upol,rows,ff,mm=monitor
    pos=torch.tensor(data.positions(rows),device="cuda:0",dtype=torch.float64)
    sites=torch.tensor(data.sites(rows),device="cuda:0",dtype=torch.int64)
    f=torch.tensor(data.frequencies(ff),device="cuda:0",dtype=torch.float64)
    label=torch.tensor(data.read_fit(rows,ff),device="cuda:0",dtype=torch.complex64)
    if method=="nerf2":
        world=data.geometry.ue_world_positions(pos,sites)[:,upol*2:(upol+1)*2]
        core=model.fields[f"s{site}_b{bpol}_u{upol}"]
        tx=model.bound_bs_ports[site,bpol].reshape(128,3)[torch.tensor(mm,device="cuda:0")]
        value=core(tx,world,f).permute(1,0,2,3)
        if hasattr(model,"log_output_gain"):value=value*model.log_output_gain.exp()
        target=label[:,bpol*128+torch.tensor(mm,device="cuda:0"),upol*2:(upol+1)*2]
        loss=(value-target).abs().square().mean()
        scalar_count=target.numel()
    else:
        # Each draw selects a site uniformly; all256x4heads remain supervised.
        value=model(pos,sites,f)
        target=label
        mse=(value-target).abs().square().mean()
        loss=mse+.01603005049928609*model.fields[str(site)].activation_penalty()
        scalar_count=target.numel()
    return loss,dict(rows=np.asarray(rows).tolist(),frequency_ids=np.asarray(ff).tolist(),
        site=site,BS_pol=bpol,UE_pol=upol,BS_spatial_ids=np.asarray(mm).tolist(),
        scalar_labels=scalar_count,mse=float((value-target).abs().square().mean().detach()),
        loss=float(loss.detach()),target_energy=float(target.abs().square().mean().detach()),
        prediction_energy=float(value.abs().square().mean().detach()))


def monitor_tasks(data,points=2,bs_links=4,tones=4):
    tasks=[]
    for site in (0,1):
        allowed=data.fit_ids[data.sites(data.fit_ids)==site]
        pos=data.positions(allowed)
        order=np.lexsort((pos[:,1],pos[:,0]))
        rows=allowed[order[np.linspace(0,len(order)-1,min(points,len(order)),dtype=int)]]
        # Train-only nonzero representative selection avoids a trivial allzero fixture.
        labels=data.read_fit(rows)
        if not np.any(np.abs(labels)>0):
            for candidate in allowed:
                if np.any(np.abs(data.read_fit(np.array([candidate])))>0):
                    rows=np.array([candidate],np.int64);break
        ff=np.array([0,64,128,191],np.int64)[:tones]
        mm=np.array([0,42,85,127],np.int64)[:bs_links]
        for b in (0,1):
            for u in (0,1):tasks.append((site,b,u,rows,ff,mm))
    return tasks


def evaluate_monitor(method,model,data,tasks):
    model.eval();values=[]
    with torch.no_grad():
        for task in tasks:
            _,d=training_batch(method,model,data,np.random.default_rng(0),monitor=task)
            values.append(d["mse"])
    model.train()
    return float(np.mean(values))


def monitor_zero_baseline(method,model,data,tasks):
    values=[]
    model.eval()
    with torch.no_grad():
        for task in tasks:
            _,d=training_batch(method,model,data,np.random.default_rng(0),monitor=task)
            values.append(d["target_energy"])
    model.train()
    return float(np.mean(values))


def predict_normalized(model,data,ids):
    model.eval()
    p=torch.tensor(data.positions(ids),device="cuda:0",dtype=torch.float64)
    s=torch.tensor(data.sites(ids),device="cuda:0",dtype=torch.int64)
    f=torch.tensor(data.frequencies(),device="cuda:0",dtype=torch.float64)
    with torch.no_grad():
        result=model(p,s,f)
        return result*model.log_output_gain.exp() if hasattr(model,"log_output_gain") else result


def pilot(args):
    method=args.method
    paths=output_paths(method,0,args.tag)
    report=paths["report"]
    data=FoldData(0,"screen",method)
    scale=data.scale
    seed_all(42)
    gpu=device_guard(GPU_UUID)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    t=time.monotonic()
    model=build_model(method,data,medium_rays=args.medium_rays,tx_chunk=args.tx_chunk,conditioned=args.conditioned)
    opt=optimizer_for(method,model)
    resumed_steps=0
    if args.resume:
        checkpoint=torch.load(Path(args.resume),map_location="cpu",weights_only=False)
        if checkpoint["info"]["method"] != method:raise ValueError("Resume method mismatch")
        model.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["optimizer"])
        resumed_steps=int(checkpoint["info"]["training_steps"])
    rng=np.random.default_rng(42)
    tasks=monitor_tasks(data,points=args.points,bs_links=args.bs_links,tones=args.tones)
    before=evaluate_monitor(method,model,data,tasks)
    zero_mse=monitor_zero_baseline(method,model,data,tasks)
    torch.cuda.synchronize()
    start=time.monotonic();records=[];durations=[]
    model.train()
    for step in range(1,args.steps+1):
        tick=time.monotonic()
        opt.zero_grad(set_to_none=True)
        if args.memorize:
            task=tasks[(step-1)%len(tasks)]
            loss,detail=training_batch(method,model,data,rng,monitor=task)
        else:
            loss,detail=training_batch(method,model,data,rng,points=args.points,
                bs_links=args.bs_links,tones=args.tones)
        if not torch.isfinite(loss):raise FloatingPointError("Nonfinite loss")
        loss.backward()
        if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise FloatingPointError("Nonfinite gradient")
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
        opt.step()
        torch.cuda.synchronize()
        duration=time.monotonic()-tick
        durations.append(duration)
        if step==1 or step%10==0:
            entry=dict(step=step,seconds=duration,gradient_norm=float(norm),**detail)
            records.append(entry)
            atomic_json(report/"progress.json",dict(stage="pilot",method=method,step=step,total=args.steps,
                elapsed_seconds=time.monotonic()-start,last=entry,GPU_UUID=GPU_UUID))
            print(json.dumps(entry),flush=True)
        if time.monotonic()-start>args.max_seconds:
            break
    after=evaluate_monitor(method,model,data,tasks)
    model.eval()
    row=int(data.probe_ids[0])
    if args.benchmark_full:
        torch.cuda.synchronize();tick=time.monotonic()
        value=predict_normalized(model,data,np.array([row],np.int64))
        torch.cuda.synchronize();full_seconds=time.monotonic()-tick
        full=dict(seconds_per_position=full_seconds,shape=list(value.shape),finite=bool(torch.isfinite(value).all()),
            estimated_500_prediction_seconds=full_seconds*500)
        if not full["finite"]:raise FloatingPointError("Nonfinite complete192-tone H")
    else:full=None
    info=dict(schema="field-baseline-native-pilot-v1",status="passed",method=method,
       training_steps=resumed_steps+len(durations),this_pilot_steps=len(durations),memorize_fixture=bool(args.memorize),FIT_monitor_before=before,FIT_monitor_after=after,
       FIT_monitor_reduction=before-after,FIT_scale=scale,FIT_zero_prediction_MSE=zero_mse,
       FIT_monitor_error_ratio_vs_zero=after/max(zero_mse,1e-30),
       actual_fixture_frequency_ids=np.asarray(tasks[0][4]).tolist(),
       log_output_gain=float(model.log_output_gain.detach()) if hasattr(model,"log_output_gain") else None,
       seconds_per_step_warm=float(np.mean(durations[min(5,len(durations)-1):])),
       total_seconds=time.monotonic()-t,full192_inference=full,
       peak_GPU_bytes=torch.cuda.max_memory_allocated(),gpu=gpu,
       config=NERF2_CONFIG if method=="nerf2" else model.metadata(),
       optimizer_source=SOURCE_TRAIN_CONFIG[method],points=args.points,BS_links=args.bs_links,tones=args.tones,
       model_parameters=sum(p.numel() for p in model.parameters()),records=records,
       data=data.info(),outer_truth_read=False,formal_result=False)
    atomic_json(report/"result.json",info)
    atomic_torch_save(paths["artifact"]/"pilot.pt",dict(model=model.state_dict(),optimizer=opt.state_dict(),info=info))
    print(json.dumps({k:v for k,v in info.items() if k not in ("records","data")}),flush=True)



class FitGPUCache:
    """FIT labels only. Target tensors remain outside the inference model."""
    def __init__(self,data,batch=64,on_gpu=False):
        self.data=data
        self.ids=np.array(data.fit_ids,np.int64)
        self.slot=np.full(4000,-1,np.int64);self.slot[self.ids]=np.arange(len(self.ids))
        self.value=None
        if on_gpu:
            self.value=torch.empty((len(self.ids),256,4,192),device="cuda:0",dtype=torch.complex64)
            for first in range(0,len(self.ids),batch):
                rows=self.ids[first:first+batch]
                self.value[first:first+len(rows)].copy_(torch.from_numpy(data.read_fit(rows)).to("cuda:0"))
    def read(self,rows,frequency_ids):
        rows=np.asarray(rows,np.int64);slots=self.slot[rows]
        if np.any(slots<0):raise PermissionError("Target cache outside FIT")
        if self.value is None:
            return torch.tensor(self.data.read_fit(rows,frequency_ids),device="cuda:0",dtype=torch.complex64)
        indices=torch.tensor(slots,device="cuda:0",dtype=torch.long)
        ff=torch.tensor(frequency_ids,device="cuda:0",dtype=torch.long)
        return self.value.index_select(0,indices).index_select(-1,ff)


def full_training_step(method,model,data,cache,rng,points=32,tones=8):
    ff=np.sort(rng.choice(192,size=tones,replace=False))
    if method=="ngrf":
        rows=[]
        for site in (0,1):
            candidates=data.fit_ids[data.sites(data.fit_ids)==site]
            rows.extend(rng.choice(candidates,size=min(points//2,len(candidates)),replace=False).tolist())
        rows=np.array(rows,np.int64)
        pos=torch.tensor(data.positions(rows),device="cuda:0",dtype=torch.float64)
        site=torch.tensor(data.sites(rows),device="cuda:0",dtype=torch.int64)
        f=torch.tensor(data.frequencies(ff),device="cuda:0",dtype=torch.float64)
        target=cache.read(rows,ff)
        value=model(pos,site,f)
        mse=(value-target).abs().square().mean()
        reg=torch.stack([field.activation_penalty() for field in model.fields.values()]).mean()
        return mse+.01603005049928609*reg,dict(mse=float(mse.detach()),activation_logit_L1=float(reg.detach()),
            scalar_labels=target.numel(),point_count=len(rows),frequency_count=len(ff))
    # NeRF2 samples scalar-link blocks uniformly; every physical link/frequency
    # remains eligible. Full matrix is produced only by the prediction path.
    site=int(rng.integers(2));b=int(rng.integers(2));u=int(rng.integers(2))
    candidates=data.fit_ids[data.sites(data.fit_ids)==site]
    rows=rng.choice(candidates,size=min(points,len(candidates)),replace=False)
    mm=np.sort(rng.choice(128,size=8,replace=False))
    pos=torch.tensor(data.positions(rows),device="cuda:0",dtype=torch.float64)
    sites=torch.tensor(data.sites(rows),device="cuda:0",dtype=torch.int64)
    frequency=torch.tensor(data.frequencies(ff),device="cuda:0",dtype=torch.float64)
    world=data.geometry.ue_world_positions(pos,sites)[:,u*2:(u+1)*2]
    core=model.fields[f"s{site}_b{b}_u{u}"]
    index=torch.tensor(mm,device="cuda:0")
    value=core(model.bound_bs_ports[site,b].reshape(128,3)[index],world,frequency).permute(1,0,2,3)
    if hasattr(model,"log_output_gain"):value=value*model.log_output_gain.exp()
    target=cache.read(rows,ff)[:,b*128+index,u*2:(u+1)*2]
    mse=(value-target).abs().square().mean()
    return mse,dict(mse=float(mse.detach()),scalar_labels=target.numel(),point_count=len(rows),
                    frequency_count=len(ff),site=site,BS_pol=b,UE_pol=u)


def source_identity():
    folder=Path(__file__).resolve().parent
    names=["common.py","field_runner.py","field_fast.py","native_ngrf.py"]
    return {name:sha256(folder/name) for name in names}


def apply_learning_rate(method,opt,equivalent_step,maximum_steps):
    if method=="ngrf":
        fraction=min(max(equivalent_step/maximum_steps,0.),1.)
        xyz=math.exp(math.log(.0002696363015043561)*(1-fraction)+math.log(5.953443116739323e-06)*fraction)
        for group in opt.param_groups:
            if group["name"].endswith("/xyz"):
                group["lr"]=0. if equivalent_step>=int(maximum_steps*.46840533026053566) else xyz
    else:
        # Source schedule has a 10k cosine horizon; this adapted stable run uses
        # a monotone cosine across the preregistered screen budget.
        for group in opt.param_groups:
            initial=group.setdefault("initial_lr",group["lr"])
            group["lr"]=1e-6+(initial-1e-6)*.5*(1+math.cos(math.pi*min(equivalent_step/maximum_steps,1.)))


def train_role(args,outer,role,*,selected_steps=None,screen_fit_count=None):
    method=args.method
    paths=output_paths(method,outer,role)
    artifact=paths["artifact"];report=paths["report"]
    final_file=artifact/"complete.pt"
    if final_file.exists():
        checkpoint=torch.load(final_file,map_location="cpu",weights_only=False)
        return checkpoint["info"]
    data=FoldData(outer,role,method)
    role_ratio=(len(data.fit_ids)/screen_fit_count) if role=="refit" else 1.
    budget=int(math.ceil(selected_steps*role_ratio)) if selected_steps is not None else args.max_steps
    seed_all(42+outer*1009)
    model=build_model(method,data,medium_rays=args.medium_rays,tx_chunk=args.tx_chunk,
                      conditioned=(method=="nerf2"))
    opt=optimizer_for(method,model)
    cache=FitGPUCache(data,on_gpu=args.gpu_label_cache)
    rng=np.random.default_rng(42+outer*1009)
    source=source_identity()
    started=time.monotonic();history=[];best_score=-math.inf;best_step=None;best_metrics=None
    significant=-math.inf;stale=0;start_step=0
    last_file=artifact/"last.pt"
    if last_file.exists():
        previous=torch.load(last_file,map_location="cpu",weights_only=False)
        if previous["source_identity"]!=source:raise RuntimeError("Resume source changed")
        model.load_state_dict(previous["model"]);opt.load_state_dict(previous["optimizer"])
        rng.bit_generator.state=previous["rng_state"]
        state=previous["selection"];start_step=state["step"]
        history=state["history"];best_score=state["best_score"];best_step=state["best_step"]
        best_metrics=state["best_metrics"];significant=state["significant"];stale=state["stale"]
    torch.cuda.reset_peak_memory_stats()
    train_points=32 if method=="ngrf" else 2
    train_tones=8 if method=="ngrf" else 4
    for step in range(start_step+1,budget+1):
        apply_learning_rate(method,opt,step/role_ratio,args.max_steps)
        model.train();opt.zero_grad(set_to_none=True)
        loss,detail=full_training_step(method,model,data,cache,rng,points=train_points,tones=train_tones)
        if not bool(torch.isfinite(loss)):raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise FloatingPointError("Nonfinite training gradient")
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
        opt.step()
        if step==1 or step%100==0:
            event=dict(step=step,loss=float(loss.detach()),gradient_norm=float(grad),**detail,
                       elapsed_seconds=time.monotonic()-started)
            history.append(event)
            atomic_json(report/"progress.json",dict(method=method,outer=outer,role=role,stage="training",
                step=step,budget=budget,last=event,GPU_UUID=GPU_UUID,source_identity=source))
            print(json.dumps(dict(method=method,outer=outer,role=role,**event)),flush=True)
        do_validation=role=="screen" and (step%args.val_every==0 or step==budget)
        if do_validation:
            model.eval();tick=time.monotonic()
            score=evaluate(data,lambda rows:predict_normalized(model,data,rows),
                           ids=data.probe_ids,batch_size=(16 if method=="ngrf" else 1))
            current=float(score["score"])
            history.append(dict(step=step,validation=score,validation_seconds=time.monotonic()-tick))
            if current>best_score:
                best_score=current;best_step=step;best_metrics=score
                atomic_torch_save(artifact/"best.pt",dict(model=model.state_dict(),step=step,
                    validation=score,source_identity=source,data=data.info(),architecture=model.metadata() if method=="ngrf" else NERF2_CONFIG))
            if current>significant+args.min_delta:
                significant=current;stale=0
            else:stale+=1
            print(json.dumps(dict(method=method,outer=outer,role=role,step=step,validation=score,
                                  best_step=best_step,stale=stale)),flush=True)
        should_save=(step%args.val_every==0 or step==budget)
        if should_save:
            selection=dict(step=step,history=history,best_score=best_score,best_step=best_step,
                           best_metrics=best_metrics,significant=significant,stale=stale)
            atomic_torch_save(last_file,dict(model=model.state_dict(),optimizer=opt.state_dict(),
                rng_state=rng.bit_generator.state,selection=selection,source_identity=source,data=data.info()))
            atomic_json(report/"history.json",history)
        if role=="screen" and step>=args.min_steps and stale>=args.patience:
            break
    if source_identity()!=source:raise RuntimeError("Source drift during training")
    if role=="screen":
        best=torch.load(artifact/"best.pt",map_location="cpu",weights_only=False)
        model.load_state_dict(best["model"])
        full_validation=evaluate(data,lambda rows:predict_normalized(model,data,rows),
                                 ids=data.val_ids,batch_size=(16 if method=="ngrf" else 1))
        info=dict(method=method,outer=outer,role=role,selected_steps=best_step,
                  selected_probe_metrics=best_metrics,full_validation_diagnostic=full_validation,
                  completed_steps=step,stopping="validation_plateau" if step<budget else "budget",
                  fit_count=len(data.fit_ids),probe_ids=data.probe_ids.tolist(),seed=42+outer*1009,
                  seconds=time.monotonic()-started,data=data.info(),source_identity=source,
                  config=model.metadata() if method=="ngrf" else NERF2_CONFIG,trained=True)
    else:
        values=[]
        model.eval()
        with torch.no_grad():
            for first in range(0,len(data.outer_ids),(16 if method=="ngrf" else 1)):
                ids=data.outer_ids[first:first+(16 if method=="ngrf" else 1)]
                values.append((predict_normalized(model,data,ids)*data.scale).cpu().numpy())
        manifest=save_outer_prediction(data,method,np.concatenate(values),metadata=dict(
            source_identity=source,selected_screen_updates=selected_steps,refit_updates=budget,
            role_budget_ratio=role_ratio,seed=42+outer*1009,original_paper_exact_reproduction=False,
            adaptation=model.metadata() if method=="ngrf" else NERF2_CONFIG))
        info=dict(method=method,outer=outer,role=role,completed_steps=step,selected_screen_updates=selected_steps,
             role_budget_ratio=role_ratio,fit_count=len(data.fit_ids),prediction=manifest,
             seconds=time.monotonic()-started,data=data.info(),source_identity=source,trained=True,
             query_truth_read=False,outer_scored=False)
    atomic_torch_save(final_file,dict(model=model.state_dict(),info=info))
    atomic_json(report/"complete.json",info)
    del model,opt,cache
    torch.cuda.empty_cache()
    return info


def main_training(args):
    device_guard(GPU_UUID);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    results=[]
    for outer in [int(x) for x in args.outers.split(",")]:
        screen=train_role(args,outer,"screen")
        refit=train_role(args,outer,"refit",selected_steps=screen["selected_steps"],
                         screen_fit_count=screen["fit_count"])
        results.append(dict(outer=outer,screen=screen,refit=refit))
        atomic_json(output_paths(args.method)["report"]/"method_progress.json",
                    dict(method=args.method,status="running",completed_outer_ids=[r["outer"] for r in results],
                         total_outer_ids=[int(x) for x in args.outers.split(",")]))
    atomic_json(output_paths(args.method)["report"]/"complete.json",
                dict(method=args.method,status="complete",folds=results,outer_scored=False,
                     V16_modified=False,GPU_UUID=GPU_UUID))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--method",choices=["nerf2","ngrf"],required=True)
    parser.add_argument("--mode",choices=["pilot","main"],default="pilot")
    parser.add_argument("--gpu-label-cache",action="store_true")
    parser.add_argument("--outers",default="0,1,2,3,4")
    parser.add_argument("--max-steps",type=int,default=15000)
    parser.add_argument("--min-steps",type=int,default=4000)
    parser.add_argument("--val-every",type=int,default=500)
    parser.add_argument("--patience",type=int,default=6)
    parser.add_argument("--min-delta",type=float,default=.0005)
    parser.add_argument("--steps",type=int,default=64)
    parser.add_argument("--tag",default="pilot_native")
    parser.add_argument("--resume")
    parser.add_argument("--points",type=int,default=2)
    parser.add_argument("--bs-links",type=int,default=4)
    parser.add_argument("--tones",type=int,default=4)
    parser.add_argument("--medium-rays",type=int,default=128)
    parser.add_argument("--tx-chunk",type=int,default=16)
    parser.add_argument("--max-seconds",type=float,default=180.)
    parser.add_argument("--conditioned",action="store_true")
    parser.add_argument("--memorize",action="store_true")
    parser.add_argument("--benchmark-full",action="store_true")
    args=parser.parse_args()
    try:
        if args.mode=="pilot":pilot(args)
        else:main_training(args)
    except BaseException:
        r=output_paths(args.method,0,args.tag if args.mode=="pilot" else "main_failure")["report"]
        (r/"failure.txt").write_text(traceback.format_exc())
        raise

if __name__=="__main__":main()
