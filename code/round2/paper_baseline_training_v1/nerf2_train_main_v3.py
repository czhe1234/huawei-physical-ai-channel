"""Frozen NeRF2 city full-H baseline, fixed folds; resumable health/screen/refit."""
from __future__ import annotations
import argparse,copy,json,math,time,traceback
from pathlib import Path
import numpy as np
import torch
from project_paths import PATHS
from . import common as C,nerf2_runner_v3 as V,nerf2_runner_v2 as P
from .field_runner import FitGPUCache,GPU_UUID


COMPAT_PREVIOUS_SELF_SHA="396d0f7fb177c857c0bde154716bb96e0454e18a22e491d9e286be69b4abc527"

def verify_resume(previous,current):
    if previous==current:return
    key=PATHS.relative(Path(__file__))
    if previous.get(key)!=COMPAT_PREVIOUS_SELF_SHA or set(previous)!=set(current):
        raise RuntimeError("Resume source mismatch")
    if any(previous[k]!=current[k] for k in current if k!=key):
        raise RuntimeError("Numerical model/config source changed")
    # Only the local loop-resume bookkeeping changed; old source/checkpoint preserved.

def sources(config):
    return {**V.identity(),PATHS.relative(Path(__file__)):C.sha256(Path(__file__)),
            PATHS.relative(config):C.sha256(config)}

def check_sources(source):
    for key,value in source.items():
        p=PATHS.root/key if "/" in key else Path(__file__).parent/key
        if C.sha256(p)!=value:raise RuntimeError("Source/config changed: "+key)

def model_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
def pack(model,opt,rng,step,source,**extra):
    return dict(model=model_state(model),optimizer=opt.state_dict(),rng_state=copy.deepcopy(rng.bit_generator.state),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                step=step,source_identity=source,**extra)

def restore(ck,model,opt,rng):
    model.load_state_dict(ck["model"]);opt.load_state_dict(ck["optimizer"])
    for state in opt.state.values():
        for k,v in state.items():
            if isinstance(v,torch.Tensor):state[k]=v.to("cuda:0")
    rng.bit_generator.state=ck["rng_state"];torch.set_rng_state(ck["torch_rng"])
    torch.cuda.set_rng_state_all(ck["cuda_rng"])

def health(cfg,config):
    out=V.paths(0,"health_10000_fresh");done=out["report"]/"result.json"
    source=sources(config)
    if done.exists():
        d=json.loads(done.read_text());verify_resume(d["source_identity"],source);return d
    C.seed_all(42);data=C.FoldData(0,"screen","nerf2");cache=FitGPUCache(data,on_gpu=True)
    fixture=V.fixed_nonzero_positions(data,cache);model=V.build(data);opt=P.optimizer(model)
    rng=np.random.default_rng(42);start=0;history=[];last=out["artifact"]/"last.pt"
    if last.exists():
        ck=torch.load(last,map_location="cpu",weights_only=False)
        verify_resume(ck["source_identity"],source)
        restore(ck,model,opt,rng);start=ck["step"];history=ck["history"]
    tick=time.monotonic()
    for step in range(start+1,10001):
        model.train();opt.zero_grad(set_to_none=True)
        loss,detail=V.fixture_step(model,data,cache,rng,fixture)
        if not torch.isfinite(loss):raise FloatingPointError("Health loss")
        loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError("Health gradients")
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step()
        if step%500==0:
            event=dict(step=step,monitor=V.monitor(model,data,fixture),gradient_norm=float(norm),
                       seconds=time.monotonic()-tick,training=detail)
            history.append(event);C.atomic_json(out["report"]/"progress.json",event)
            print(json.dumps(dict(stage="health_fresh10000",**event)),flush=True)
            C.atomic_torch_save(last,pack(model,opt,rng,step,source,history=history))
    check_sources(source);final=V.monitor(model,data,fixture)
    finite=all(math.isfinite(final[k]) for k in ["complex_nmse","prediction_to_truth_energy","normalized_complex_correlation_real"])
    if not finite or final["prediction_to_truth_energy"]<=0:raise FloatingPointError("Invalid health outputs")
    d=dict(status="numerically_valid",underfit=(final["complex_nmse"]>.8),updates=10000,
           final=final,history=history,source_identity=source,fit_ids=fixture.tolist(),
           restart="fresh seed42; old4215 model lacked Adam/RNG and remains preserved",
           V16_changed=False,outer_truth_read=False)
    C.atomic_json(done,d);del model,opt,cache;torch.cuda.empty_cache();return d

def save_prediction(data,values,metadata):
    out=V.paths(data.outer,"prediction")["artifact"]
    a=np.asarray(values,np.complex64)
    if a.shape!=(len(data.outer_ids),256,4,192) or not np.isfinite(a).all():raise ValueError("Full rawH required")
    C.atomic_npy(out/"channels.npy",a);C.atomic_npy(out/"row_ids.npy",data.outer_ids)
    d=dict(schema="canonical-baseline-prediction-v1",method="nerf2",outer=data.outer,
      input_identity=data.inputs.identity,row_ids=data.outer_ids.tolist(),row_ids_sha256=C.digest_ids(data.outer_ids),
      prediction_namespace="canonical_train_row_ids",kind="channel",path=PATHS.relative(out/"channels.npy"),
      sha256=C.sha256(out/"channels.npy"),shape=list(a.shape),dtype=str(a.dtype),fit_data=data.info(),
      metadata=metadata,query_truth_read=False,trained=True)
    C.atomic_json(out/"manifest.json",d);return d

def phase(outer,role,cfg,config,selection=None):
    out=V.paths(outer,role);art,report=out["artifact"],out["report"];done=report/"complete.json"
    source=sources(config)
    if done.exists():
        d=json.loads(done.read_text())
        verify_resume(d["source_identity"],source)
        return d
    data=C.FoldData(outer,role,"nerf2");seed=42+outer*1009;C.seed_all(seed)
    model=V.build(data);opt=P.optimizer(model);cache=FitGPUCache(data,on_gpu=True);rng=np.random.default_rng(seed)
    ratio=(len(data.fit_ids)/selection["fit_count"]) if role=="refit" else 1.
    budget=int(math.ceil(selection["selected_step"]*ratio)) if selection else cfg["max_steps"]
    source=sources(config);start=0;best=math.inf;best_step=None;anchor=math.inf;stale=0;history=[]
    last=art/"last.pt"
    if last.exists():
        ck=torch.load(last,map_location="cpu",weights_only=False)
        verify_resume(ck["source_identity"],source)
        if ck["fit_ids_sha256"]!=C.digest_ids(data.fit_ids):raise RuntimeError("Resume role mismatch")
        restore(ck,model,opt,rng);start=ck["step"];best=ck["best"];best_step=ck["best_step"]
        anchor=ck["anchor"];stale=ck["stale"];history=ck["history"]
    tick=time.monotonic();step=start
    for step in range(start+1,budget+1):
        P.set_lr(opt,step/ratio,cfg["max_steps"]);model.train();opt.zero_grad(set_to_none=True)
        raw,detail=P.step_loss(model,data,cache,rng,step=step);loss=.5*raw
        if not torch.isfinite(loss):raise FloatingPointError("Training loss")
        loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError("Training gradients")
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step()
        if step%500==0:
            event=dict(outer=outer,role=role,step=step,budget=budget,loss=float(loss),gradient_norm=float(norm),
                       elapsed_seconds=time.monotonic()-tick,**detail)
            C.atomic_json(report/"progress.json",event);print(json.dumps(event),flush=True)
        validation=role=="screen" and (step%cfg["val_every"]==0 or step==budget)
        if validation:
            check_sources(source);model.eval()
            value=P.subset_stats(model,data,data.probe_ids,"validation");score=value["nmse"]
            if not math.isfinite(score):raise FloatingPointError("Validation score")
            event=dict(step=step,proxy=value);history.append(event)
            if score<best:
                best=score;best_step=step
                C.atomic_torch_save(art/"best.pt",dict(model=model_state(model),step=step,proxy=value,source_identity=source))
            if score<anchor-cfg["min_delta"]:anchor=score;stale=0
            else:stale+=1
            print(json.dumps(dict(outer=outer,role=role,validation=event,best_step=best_step,stale=stale)),flush=True)
        if step%1000==0 or validation or step==budget:
            C.atomic_torch_save(last,pack(model,opt,rng,step,source,best=best,best_step=best_step,
              anchor=anchor,stale=stale,history=history,fit_ids_sha256=C.digest_ids(data.fit_ids)))
        if validation and step>=cfg["min_steps"] and stale>=cfg["patience"]:break
    check_sources(source)
    if role=="screen":
        if best_step is None:raise RuntimeError("No selected model")
        d=dict(status="screen_complete",outer=outer,role=role,selected_step=best_step,selected_proxy_nmse=best,
               actual_steps=step,fit_count=len(data.fit_ids),source_identity=source,history=history,
               seconds=time.monotonic()-tick,query_truth_read=False,config=cfg)
        C.atomic_json(report/"selection.json",d)
        ck=torch.load(art/"best.pt",map_location="cpu",weights_only=False);model.load_state_dict(ck["model"]);model.eval()
        diag=C.evaluate(data,lambda ids:P.predict_full(model,data,ids),ids=data.probe_ids,batch_size=1)
        C.atomic_json(report/"selected_complete192_diagnostic.json",
                      dict(selected_step=best_step,used_for_selection=False,metrics=diag))
    else:
        C.atomic_torch_save(art/"final.pt",pack(model,opt,rng,step,source,selection=selection,
              fit_ids_sha256=C.digest_ids(data.fit_ids)))
        model.eval();work=art/"prediction_work.npy";state=report/"prediction_work.json"
        if state.exists():
            progress=json.loads(state.read_text())
            if progress["source_identity"]!=source:raise RuntimeError("Prediction source drift")
            array=np.load(work,mmap_mode="r+")
        else:
            if work.exists():raise FileExistsError(work)
            array=np.lib.format.open_memmap(work,mode="w+",dtype=np.complex64,shape=(len(data.outer_ids),256,4,192))
            progress=dict(completed=0,source_identity=source);C.atomic_json(state,progress)
        with torch.no_grad():
            for i in range(progress["completed"],len(data.outer_ids)):
                row=data.outer_ids[i:i+1]
                a=(P.predict_full(model,data,row)*data.scale)[0].cpu().numpy()
                if not np.isfinite(a).all():raise FloatingPointError("Full H prediction")
                array[i]=a;array.flush();progress["completed"]=i+1
                C.atomic_json(state,progress)
                if (i+1)%10==0:print(json.dumps(dict(outer=outer,predicted=i+1,total=len(data.outer_ids))),flush=True)
        manifest=save_prediction(data,array,dict(source_identity=source,adapter_version="city_conditioning_v3",
            selected_screen_updates=selection["selected_step"],refit_updates=budget,role_budget_ratio=ratio,
            config=cfg,adaptation=V.metadata(),original_paper_exact_reproduction=False,
            health_report="nerf2/v3/outer0/health_10000_fresh/result.json"))
        d=dict(status="refit_complete",outer=outer,role=role,actual_steps=step,source_identity=source,
               prediction=manifest,seconds=time.monotonic()-tick,config=cfg)
        del array
    C.atomic_json(done,d);del model,opt,cache;torch.cuda.empty_cache();return d

def main(args):
    cfg=json.loads(args.config.read_text())
    if cfg.get("schema")!="nerf2-city-v3-main" or cfg.get("freeze_approved") is not True:raise ValueError("Frozen config required")
    C.device_guard(GPU_UUID);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.stage in ("health","health-then-all"):health(cfg,args.config)
    if args.stage=="health":return
    source=sources(args.config);C.atomic_json(V.paths()["report"]/"formal_source_freeze.json",dict(source_identity=source,config=cfg))
    result=[]
    for outer in [int(x) for x in args.folds.split(",")]:
        check_sources(source);selected=phase(outer,"screen",cfg,args.config)
        result.append(phase(outer,"refit",cfg,args.config,selected))
        C.atomic_json(V.paths()["report"]/"method_progress.json",dict(completed_folds=[x["outer"] for x in result]))
    C.atomic_json(V.paths()["report"]/"all_folds_complete.json",dict(status="complete",results=result,outer_scored=False))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,required=True)
    p.add_argument("--stage",choices=["health","all","health-then-all"],default="health-then-all")
    p.add_argument("--folds",default="0,1,2,3,4");args=p.parse_args()
    try:main(args)
    except BaseException:
        C.atomic_json(V.paths()["report"]/"main_failure.json",dict(traceback=traceback.format_exc()));raise

