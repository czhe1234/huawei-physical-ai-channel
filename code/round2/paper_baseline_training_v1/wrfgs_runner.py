"""Train the declared WRF-GS+ power adapter on canonical spatial roles."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from project_paths import PATHS
from round2.paper_baseline_training_v1.common import (
    FoldData,output_paths,seed_all,device_guard,evaluate,atomic_json,atomic_torch_save,
    save_outer_prediction,sha256,
)
from round2.paper_baseline_training_v1.wrfgs_model import BatchedWRFGS,power_loss

GPU_UUID="GPU-70b6fceb-c7c4-9467-3c81-097c9284c160"


def target_power(data,rows,fids):
    h=torch.as_tensor(data.read_fit(rows,fids,normalized=True),device="cuda:0",dtype=torch.complex64)
    p=torch.fft.fft2(h.reshape(len(rows),2,16,8,4,len(fids)),dim=(2,3)).abs().square()
    return p.reshape(len(rows),256,4,len(fids))


def train_step(model,opt,data,rows,fids,*,update=True):
    pos=data.positions(rows);sites=data.sites(rows);freq=data.frequencies(fids)
    truth=target_power(data,rows,fids)
    model.train();model.zero_grad(set_to_none=True)
    prediction=model(pos,sites,freq)
    loss,detail=power_loss(prediction,truth)
    if not bool(torch.isfinite(loss)):raise FloatingPointError("WRF loss nonfinite")
    loss.backward()
    for parameter in model.parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
            raise FloatingPointError("WRF gradient nonfinite")
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
    if update:
        opt.step();model.constrain()
    return float(loss.detach()),detail,float(norm)


def predictor(model,data):
    def fn(rows):
        model.eval()
        return model(data.positions(rows),data.sites(rows),data.frequencies())
    return fn


def code_identity():
    p=Path(__file__).resolve()
    q=p.with_name("wrfgs_model.py")
    return {PATHS.relative(x):sha256(x) for x in [p,q]}


def pilot(args):
    paths=output_paths("wrfgs",args.outer,args.tag)
    report=paths["report"]
    if (report/"result.json").exists():raise FileExistsError("Pilot result already exists; use new tag")
    device=device_guard(GPU_UUID)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    seed_all(args.seed)
    data=FoldData(args.outer,"screen","wrfgs")
    scale=data.scale
    model=BatchedWRFGS(data.geometry,data.positions(data.fit_ids),count=args.gaussians,seed=args.seed).to("cuda:0")
    opt=model.optimizer()
    rng=np.random.default_rng(args.seed)
    # Public role membership selects a fixed FIT fixture, never its target quality.
    fixture=[]
    for site in (0,1):
        candidates=data.fit_ids[data.inputs.site_id[data.fit_ids]==site]
        fixture.extend(rng.choice(candidates,size=16,replace=False).tolist())
    fixture=np.asarray(fixture,np.int64)
    bench=[]
    for batch in (2,4,8,16):
        rows=fixture[:batch//2].tolist()+fixture[16:16+batch-batch//2].tolist()
        rows=np.asarray(rows,np.int64)
        fids=np.array([0,48,96,144],np.int64)
        model.pair_batch=batch*len(fids)
        for _ in range(2):train_step(model,opt,data,rows,fids,update=False)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        last=None
        for _ in range(8):last=train_step(model,opt,data,rows,fids,update=False)
        torch.cuda.synchronize()
        dt=(time.perf_counter()-start)/8
        bench.append(dict(positions=batch,frequencies=4,pairs=batch*4,seconds_per_step=dt,
                          pairs_per_second=batch*4/dt,peak_GPU_bytes=torch.cuda.max_memory_allocated(),
                          initial_fixture_loss=last[0]))
    # Choose batch only from measured throughput, with no score-based setting search.
    choice=max(bench,key=lambda x:x["pairs_per_second"])
    batch=choice["positions"];model.pair_batch=choice["pairs"]
    before=evaluate(data,predictor(model,data),batch_size=1,power_only=True)
    history=[];start=time.perf_counter()
    for step in range(1,args.pilot_steps+1):
        rows=rng.choice(fixture,size=batch,replace=True)
        fids=np.sort(rng.choice(192,size=4,replace=False)).astype(np.int64)
        loss,detail,grad=train_step(model,opt,data,rows,fids)
        if step==1 or step%50==0 or step==args.pilot_steps:
            item=dict(step=step,loss=loss,gradient_norm=grad,**detail)
            history.append(item)
            print(json.dumps(item),flush=True)
    torch.cuda.synchronize();training_seconds=time.perf_counter()-start
    after=evaluate(data,predictor(model,data),batch_size=1,power_only=True)
    result=dict(status="pilot_complete",outer=args.outer,role="screen_FIT_fixed_fixture_only",
                trained=True,pilot_only=True,formal_outer_prediction=False,quality_scored_on_outer=False,
                device=device,gaussians_per_site=args.gaussians,model=model.metadata(),
                fit_scale=scale,fixture_fit_ids=fixture.tolist(),batch_benchmarks=bench,selected_batch=choice,
                train_steps=args.pilot_steps,training_seconds=training_seconds,history=history,
                validation_probe_before=before,validation_probe_after=after,data=data.info(),
                code_identity=code_identity(),formal_native_200k_step_estimate_seconds=200000*choice["seconds_per_step"])
    atomic_json(report/"result.json",result)
    atomic_torch_save(paths["artifact"]/"pilot_model.pt",dict(model=model.state_dict(),config=vars(args),data=data.info()))
    print(json.dumps({k:result[k] for k in ["status","training_seconds","validation_probe_before","validation_probe_after","formal_native_200k_step_estimate_seconds"]}),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=["pilot"],default="pilot")
    parser.add_argument("--outer",type=int,default=0)
    parser.add_argument("--seed",type=int,default=20261101)
    parser.add_argument("--gaussians",type=int,default=1024)
    parser.add_argument("--pilot-steps",type=int,default=500)
    parser.add_argument("--tag",default="pilot_v1")
    args=parser.parse_args()
    if not args.tag.replace("_","").isalnum():raise ValueError("Safe tag required")
    try:pilot(args)
    except BaseException:
        p=output_paths("wrfgs",args.outer,args.tag)["report"]/"failure.txt"
        with p.open("a") as f:f.write(traceback.format_exc()+"\n")
        raise

if __name__=="__main__":main()
