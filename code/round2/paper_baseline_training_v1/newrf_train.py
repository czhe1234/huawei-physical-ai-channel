"""Screen/refit NeWRF using a frozen, parent-approved run configuration."""
from __future__ import annotations
import argparse,copy,fcntl,hashlib,json,math,time,traceback
from pathlib import Path
import numpy as np
import torch
from project_paths import PATHS
from .common import (ART,REPORT,FoldData,seed_all,device_guard,output_paths,
                     atomic_json,atomic_torch_save,evaluate,save_outer_prediction,
                     digest_ids,sha256)
from .newrf_runner import (GPU_UUID,GEOMETRY,AUTHOR,FitGeometryBank,FitDirectionCache,
                           build_model,subset_forward,scalar_target,weighted_loss,log_record)
from .newrf_latefreq import build_late_model,ADAPTER_VERSION
from .newrf_query import subset_query
from .selection import evaluate_complex_probe,probe_contract

def make(data,cfg):
    bank=FitGeometryBank(GEOMETRY,data.inputs,outer=data.outer,
                         role="screen_fit" if data.role=="screen" else "refit")
    cache=FitDirectionCache(bank)
    model=build_late_model(data,cache,cfg["n_samples"],cfg["n_importance"],
                      frequency_chunk=cfg["frequency_chunk"])
    # v2 naive/shared H and all-parameter gradient checks passed before freeze.
    model=model.to("cuda:0")
    cache.to("cuda:0")
    return model,cache

def metadata(model,cache,data,cfg,step):
    return dict(method="NeWRF_geometry_assisted_wideband_multiport",adapter_version=ADAPTER_VERSION,step=step,
                density_frequency_independent=True,frequency_PE_levels=10,
                per_field_updates=[(step+7-i)//8 for i in range(8)],
                trained=step>0,formal_config=cfg,model_config=__import__("dataclasses").asdict(model.config),
                ray_cache=cache.info,fit_ids_sha256=digest_ids(data.fit_ids),
                input_identity=data.inputs.identity,fit_RMS=data.scale,
                normalization="raw native synthesis; both loss operands divided by FIT RMS",
                query_provider="current-role FIT image transforms+fixed sphere; no query path/cache/CSI",
                training_objective=".1 coarse+.9 fine sampled global-normalized complex MSE",
                precision=dict(network="FP32 with TF32 enabled",coarse_CDF="FP64",
                               geometry_phase="FP64",H="complex64",BF16=False),
                original_paper_reproduction=False,
                native_reference=dict(lr=.0005,iterations=100000,batch_scalar_labels=32,
                                      n_samples=128,n_importance=128,position_PE_levels=10),
                modifications=["multiport; TX+position trunk, direction+frequency signal branch","FIT-derived ray prior",
                               "sampling positions/BS elements/frequencies for unbiased MSE gradients",
                               "FIT-geometry far bound","unbiased global-FIT RMS loss scale",
                               "explicit FP64 stratified/CDF sampling",
                               "naive/shared-equivalent trunk and branch projection sharing"],
                data=data.info())

def source_identity(config_path):
    paths=[Path(__file__),Path(__file__).with_name("newrf_runner.py"),
           Path(__file__).with_name("newrf_precision_probe.py"),
           Path(__file__).with_name("newrf_latefreq.py"),
           Path(__file__).with_name("newrf_latefreq_pilot.py"),
           PATHS.root/"code/round2/paper_baseline_adapters_v1/newrf.py",
           PATHS.root/"code/round2/simulation_recovery_v1/qualified_newrf.py",
           PATHS.root/"code/round2/simulation_recovery_v1/ray_provider.py",
           Path(__file__).with_name("common.py"),Path(__file__).with_name("newrf_query.py"),
           Path(__file__).with_name("selection.py"),config_path,
           AUTHOR/"config/default.yaml"]
    return {PATHS.relative(p):sha256(p) for p in paths}

def check_sources(expected):
    for relative,digest in expected.items():
        if sha256(PATHS.root/relative)!=digest:
            raise RuntimeError("Runtime source/config drift: "+relative)

def cpu_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

def checkpoint(data,model,optimizer,rng,step,cfg,sources,**extra):
    return dict(schema="newrf-trained-checkpoint-v1",outer=data.outer,role=data.role,
                step=step,config=cfg,source_identities=sources,
                fit_ids_sha256=digest_ids(data.fit_ids),input_identity=data.inputs.identity,
                scale=data.scale,state_dict=cpu_state(model),optimizer=optimizer.state_dict(),
                rng_state=copy.deepcopy(rng.bit_generator.state),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                **extra)

def normalized_predict(model,data,ids):
    with torch.no_grad():
        return model(data.positions(ids),data.sites(ids))/data.scale

def run_phase(outer,role,cfg,config_path,selected=None,train_only=False):
    paths=output_paths("newrf",outer,role)
    art,report=paths["artifact"],paths["report"]
    complete=report/("weights_complete.json" if role=="refit" and train_only else "complete.json")
    if complete.exists():
        old=json.loads(complete.read_text())
        if old["formal_config"]!=cfg:raise RuntimeError("Existing phase belongs to another configuration")
        return old
    data=FoldData(outer,role,method="newrf")
    seed_all(cfg["seed"]);torch.set_num_threads(cfg.get("CPU_threads",4))
    model,cache=make(data,cfg);model.train()
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg["lr"],weight_decay=cfg.get("weight_decay",0.))
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",
                  factor=cfg.get("scheduler_factor",.9),patience=cfg.get("scheduler_patience",10),
                  min_lr=cfg.get("min_lr",1e-5))
    rng=np.random.default_rng(cfg["seed"]+1000*outer)
    sources=source_identity(config_path)
    max_steps=int(cfg["max_steps"])
    if role=="refit":
        if selected is None:raise ValueError("Refit requires frozen screen selection")
        screen_count=int(selected["FIT_count"])
        max_steps=int(math.ceil(selected["selected_step"]*len(data.fit_ids)/screen_count))
    eval_steps=sorted(set(int(v) for v in cfg["eval_steps"] if 0<v<=max_steps))
    if role=="screen" and max_steps not in eval_steps:eval_steps.append(max_steps)
    start_step=0;best_score=math.inf;best_step=None;stale=0;history=[];losses=[];zero_grad_batches=0
    lr_replay={}
    if role=="refit":
        ratio=len(data.fit_ids)/screen_count
        for event in selected["validation_history"]:
            if event["step"]<=selected["selected_step"]:
                lr_replay[int(math.ceil(event["step"]*ratio))]=event["lr_after_scheduler"]
    last=art/"last.pt"
    if last.exists():
        ck=torch.load(last,map_location="cpu",weights_only=False)
        if ck["config"]!=cfg or ck["fit_ids_sha256"]!=digest_ids(data.fit_ids) or ck["input_identity"]!=data.inputs.identity:
            raise RuntimeError("Checkpoint role/config/input mismatch")
        if ck["source_identities"]!=sources:raise RuntimeError("Checkpoint source identity mismatch")
        model.load_state_dict(ck["state_dict"],strict=True)
        optimizer.load_state_dict(ck["optimizer"])
        for state in optimizer.state.values():
            for k,v in state.items():
                if isinstance(v,torch.Tensor):state[k]=v.to("cuda:0")
        start_step=ck["step"];rng.bit_generator.state=ck["rng_state"]
        torch.set_rng_state(ck["torch_rng"]);torch.cuda.set_rng_state_all(ck["cuda_rng"])
        best_score=ck.get("best_score",math.inf);best_step=ck.get("best_step")
        stale=ck.get("stale",0);history=ck.get("history",[]);losses=ck.get("losses",[])
        if "scheduler" in ck:scheduler.load_state_dict(ck["scheduler"])
    base_wall=time.monotonic();last_checkpoint=base_wall
    atomic_json(report/"run_manifest.json",dict(status="running",outer=outer,role=role,
                formal_config=cfg,source_identities=sources,fit_data=data.info(),
                intended_steps=max_steps,resumed_from_step=start_step,model=metadata(model,cache,data,cfg,start_step)))
    step=start_step
    for step in range(start_step+1,max_steps+1):
        if step-1 in lr_replay:
            for group in optimizer.param_groups:group["lr"]=lr_replay[step-1]
        rows=rng.choice(data.fit_ids,size=cfg["batch_positions"],replace=True)
        field=(step-1)%8
        spatial=np.sort(rng.choice(128,size=cfg["BS_spatial_samples"],replace=False))
        frequencies=np.sort(rng.choice(192,size=cfg["frequency_samples"],replace=False))
        target=scalar_target(data,rows,field,spatial,frequencies,"cuda:0")
        pred,coarse=subset_forward(model,cache,data,rows,field,spatial,frequencies)
        loss,lf,lc=weighted_loss(pred,coarse,target,data.scale)
        if not bool(torch.isfinite(loss)):raise FloatingPointError("Nonfinite loss")
        optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.get("gradient_clip",5.))
        if not bool(torch.isfinite(norm)):raise FloatingPointError("Nonfinite gradient")
        if float(norm)==0:zero_grad_batches+=1
        # Zero-density batches are legitimate after training; do not alter
        # learned density or initialization in response to a single such batch.
        optimizer.step()
        losses.append(float(loss.detach()))
        if step%cfg.get("log_every",250)==0:
            value=dict(stage=role,outer=outer,step=step,max_steps=max_steps,
                       loss_recent=float(np.mean(losses[-250:])),last_loss=float(loss.detach()),
                       lr=optimizer.param_groups[0]["lr"],elapsed_seconds=time.monotonic()-base_wall,
                       zero_gradient_batches=zero_grad_batches,
                       peak_gpu_GiB=torch.cuda.max_memory_allocated()/2**30)
            log_record(report/"progress.jsonl",value);atomic_json(report/"progress.json",value)
        if role=="screen" and step in eval_steps:
            check_sources(sources);model.eval()
            metrics=evaluate_complex_probe(data,lambda ids,bs,ue,ff:
                       subset_query(model,data,ids,bs,ue,ff)/data.scale,
                       batch_size=cfg.get("selection_batch_size",16))
            score=float(metrics["selection_value"])
            if not math.isfinite(score):raise FloatingPointError("Nonfinite validation metric")
            scheduler.step(score)
            history.append(dict(step=step,metrics=metrics,
                                lr_after_scheduler=optimizer.param_groups[0]["lr"]))
            eligible=step>=cfg.get("min_steps",0)
            improved=eligible and score<best_score-cfg.get("minimum_improvement",.001)
            if improved:
                best_score=score;best_step=step;stale=0
                atomic_torch_save(art/"best.pt",checkpoint(data,model,optimizer,rng,step,cfg,sources,
                                  best_score=best_score,best_step=best_step,history=history,
                                  metadata=metadata(model,cache,data,cfg,step)))
            elif eligible:
                stale+=1
            log_record(report/"validation.jsonl",dict(step=step,metrics=metrics,
                        best_score=best_score if math.isfinite(best_score) else None,best_step=best_step,
                        selected=improved,stale=stale,complete192_tones=False,selection_contract=probe_contract(data)))
            model.train()
            atomic_json(report/"validation_history.json",dict(history=history,selection_probe_ids=data.probe_ids.tolist()))
        if step%cfg.get("checkpoint_every",1000)==0 or step==max_steps or (role=="screen" and step in eval_steps) or time.monotonic()-last_checkpoint>600:
            atomic_torch_save(last,checkpoint(data,model,optimizer,rng,step,cfg,sources,
                              best_score=best_score,best_step=best_step,stale=stale,
                              history=history,losses=losses,scheduler=scheduler.state_dict()))
            last_checkpoint=time.monotonic()
        if role=="screen" and step in eval_steps and stale>=cfg.get("early_stop_patience",3) and step>=cfg.get("min_steps",0):
            break
    torch.cuda.synchronize()
    if role=="screen":
        if best_step is None:raise RuntimeError("No trained checkpoint eligible for selection")
        value=dict(status="screen_complete",outer=outer,role=role,selected_step=best_step,
                   selected_subtensor_nmse=best_score,lower_is_better=True,
                   selection_contract=probe_contract(data),FIT_count=len(data.fit_ids),actual_steps=step,
                   formal_config=cfg,validation_history=history,probe_ids=data.probe_ids.tolist(),
                   best_checkpoint=PATHS.relative(art/"best.pt"),source_identities=sources,
                   wall_seconds=time.monotonic()-base_wall,query_truth_read=False)
        # Freeze the proxy choice before any complete-band official diagnostic.
        atomic_json(report/"selection.json",value)
        best=torch.load(art/"best.pt",map_location="cpu",weights_only=False)
        model.load_state_dict(best["state_dict"],strict=True);model.eval();check_sources(sources)
        diagnostic=evaluate(data,lambda ids:normalized_predict(model,data,ids),
                            ids=data.probe_ids,batch_size=1)
        atomic_json(report/"selected_complete192_diagnostic.json",dict(
                    selected_step=best_step,selection_metric="validation_subtensor_nmse",
                    selection_was_frozen_first=True,used_for_further_selection=False,
                    metrics=diagnostic,position_ids=data.probe_ids.tolist(),
                    all192_tones=True,outer_quality_scored=False))
    else:
        model.eval();check_sources(sources)
        atomic_torch_save(art/"final.pt",checkpoint(data,model,optimizer,rng,step,cfg,sources,
                          screen_selection=selected,refit_LR_replay=lr_replay,metadata=metadata(model,cache,data,cfg,step)))
        value=dict(status="refit_weights_complete",outer=outer,role=role,formal_config=cfg,
                   actual_steps=step,checkpoint=PATHS.relative(art/"final.pt"),
                   checkpoint_sha256=sha256(art/"final.pt"),
                   wall_seconds=time.monotonic()-base_wall,source_identities=sources,
                   screen_selection=selected)
        atomic_json(report/"weights_complete.json",value)
        if not train_only:
            del model,optimizer,cache
            torch.cuda.empty_cache()
            value=predict_refit(outer,cfg,config_path)
            return value
    check_sources(sources);atomic_json(complete,value)
    del model,optimizer,cache
    torch.cuda.empty_cache()
    return value


def predict_refit(outer,cfg,config_path):
    paths=output_paths("newrf",outer,"refit");art,report=paths["artifact"],paths["report"]
    complete=report/"complete.json"
    if complete.exists():
        value=json.loads(complete.read_text())
        if value["formal_config"]!=cfg:raise RuntimeError("Completed prediction config differs")
        return value
    with (art/"prediction.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        data=FoldData(outer,"refit",method="newrf")
        seed_all(cfg["seed"]);torch.set_num_threads(cfg.get("CPU_threads",4))
        model,cache=make(data,cfg)
        final=art/"final.pt"
        ck=torch.load(final,map_location="cpu",weights_only=False)
        sources=source_identity(config_path)
        if ck["config"]!=cfg or ck["source_identities"]!=sources or ck["fit_ids_sha256"]!=digest_ids(data.fit_ids):
            raise RuntimeError("Frozen checkpoint role/config/source mismatch")
        if ck["input_identity"]!=data.inputs.identity or ck["scale"]!=data.scale:
            raise RuntimeError("Frozen checkpoint input/scale mismatch")
        model.load_state_dict(ck["state_dict"],strict=True);model.eval()
        work=art/"prediction_work.npy";state=report/"prediction_work.json"
        contract=dict(checkpoint_sha256=sha256(final),source_identities=sources,
                      input_identity=data.inputs.identity,row_ids=data.outer_ids.tolist(),
                      fit_ids_sha256=digest_ids(data.fit_ids))
        if state.exists():
            progress=json.loads(state.read_text())
            if progress["contract"]!=contract:raise RuntimeError("Prediction resume identity mismatch")
            array=np.load(work,mmap_mode="r+")
            if array.shape!=(len(data.outer_ids),256,4,192) or array.dtype!=np.complex64:
                raise RuntimeError("Prediction resume header mismatch")
        else:
            if work.exists():raise FileExistsError("Unregistered prediction work; preserve for repair")
            array=np.lib.format.open_memmap(work,mode="w+",dtype=np.complex64,
                                           shape=(len(data.outer_ids),256,4,192))
            progress=dict(contract=contract,completed={})
            atomic_json(state,progress)
        def row_hash(x):
            return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
        started=time.monotonic()
        with torch.no_grad():
            for i,row in enumerate(data.outer_ids):
                key=str(int(row))
                if key in progress["completed"]:
                    if row_hash(array[i])!=progress["completed"][key]:
                        raise RuntimeError("Committed prediction row changed")
                    continue
                value=model(data.positions([row]),data.sites([row]))[0].detach().cpu().numpy()
                if not np.isfinite(value).all():raise FloatingPointError("Nonfinite query prediction")
                array[i]=value;array.flush()
                progress["completed"][key]=row_hash(array[i])
                progress["elapsed_this_session_seconds"]=time.monotonic()-started
                atomic_json(state,progress)
                log_record(report/"prediction_progress.jsonl",dict(outer=outer,done=len(progress["completed"]),
                           total=len(data.outer_ids),row_id=int(row),elapsed_seconds=time.monotonic()-started))
        check_sources(sources)
        manifest=save_outer_prediction(data,"newrf",array,metadata=dict(
                   checkpoint=PATHS.relative(final),checkpoint_sha256=contract["checkpoint_sha256"],
                   formal_config=cfg,source_identities=sources,model=metadata(model,cache,data,cfg,ck["step"]),
                   screen_selection=ck["screen_selection"],query_geometry_payload_reads=0,
                   outer_quality_scored=False,actual_model_forward=True,resumable_per_row_hash=True))
        result=dict(status="refit_prediction_complete",outer=outer,role="refit",formal_config=cfg,
                    actual_steps=ck["step"],prediction=manifest,prediction_seconds=time.monotonic()-started,
                    source_identities=sources)
        atomic_json(complete,result)
        del model,cache,array
        torch.cuda.empty_cache()
        return result

def run(args):
    cfg=json.loads(args.config.read_text())
    if cfg.get("schema")!="newrf-main-training-config-v1" or cfg.get("freeze_approved") is not True:
        raise PermissionError("Parent-approved frozen method config required")
    if cfg["query_rays"]!=64 or cfg["max_training_rays"]!=64:
        raise ValueError("Current accepted NeWRF direction budget is64")
    if cfg.get("BF16",False):raise ValueError("BF16 precision probe failed; not allowed")
    guard=device_guard(GPU_UUID);torch.backends.cuda.matmul.allow_tf32=True
    atomic_json(output_paths("newrf")["report"]/"GPU_assignment.json",guard)
    with (output_paths("newrf")["artifact"]/"run.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        frozen=source_identity(args.config)
        freeze_path=output_paths("newrf")["report"]/"formal_source_freeze.json"
        if freeze_path.exists():
            if json.loads(freeze_path.read_text())["source_identities"]!=frozen:
                raise RuntimeError("Frozen formal source identities differ")
        else:
            atomic_json(freeze_path,dict(source_identities=frozen,formal_config=cfg))
        folds=[int(v) for v in args.folds.split(",")]
        completed=[]
        for outer in folds:
            check_sources(frozen)
            if args.stage=="predict":
                completed.append(predict_refit(outer,cfg,args.config))
            else:
                selected=run_phase(outer,"screen",cfg,args.config)
                completed.append(run_phase(outer,"refit",cfg,args.config,selected,train_only=args.stage=="train"))
            check_sources(frozen)
        atomic_json(output_paths("newrf")["report"]/("all_weights_complete.json" if args.stage=="train" else "all_folds_complete.json"),
                    dict(status="weights_complete" if args.stage=="train" else "complete",method="newrf",
                         folds=folds,outputs=completed,final_method_scored=False,
                         query_truth_read=False,V16_changed=False))

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--folds",default="0,1,2,3,4")
    parser.add_argument("--stage",choices=["all","train","predict"],default="all")
    args=parser.parse_args()
    try:run(args)
    except BaseException:
        with (output_paths("newrf")["report"]/"formal_failure.txt").open("w") as f:f.write(traceback.format_exc())
        raise
