"""One bounded low-LR NeRF2 continuation; preserve parent Adam/RNG/BS state."""
from __future__ import annotations
import argparse,copy,json,math,random,time,traceback
from pathlib import Path
import numpy as np
import torch
from . import nerf2_repair as R
from round2.paper_baseline_training_v1 import nerf2_native_heads_continuation_v4 as CONT

REPORT=R.REPORT/"low_lr"
ART=R.ART/"low_lr"
CONFIG=R.REPORT/"low_lr_config.json"
PARENT=R.ART/"final.pt"
PARENT_RESULT=R.REPORT/"result.json"

def identity():
    out=R.identity()
    for p in [Path(__file__),CONFIG,PARENT,PARENT_RESULT,R.ART/"raw_prediction.npy",Path(CONT.__file__)]:
        out[R.PATHS.relative(p)]=R.C.sha256(p)
    return out

def emit(d):
    REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/"events.jsonl").open("a")as f:f.write(json.dumps(d,allow_nan=False)+"\n")
    print(json.dumps(d,allow_nan=False),flush=True)

def sampler_equal(a,b):
    return (a["cursor"]==b["cursor"] and a["cycles"]==b["cycles"]
        and np.array_equal(a["order"],b["order"]) and np.array_equal(a["counts"],b["counts"])
        and a["rng_state"][0]==b["rng_state"][0]
        and np.array_equal(a["rng_state"][1],b["rng_state"][1])
        and a["rng_state"][2:]==b["rng_state"][2:])

def setup(cfg,seed):
    parent_result=json.loads(PARENT_RESULT.read_text())
    if parent_result["fit_passed"]:raise RuntimeError("Parent already passed; low-LR continuation unnecessary")
    parent=torch.load(PARENT,map_location="cpu",weights_only=False)
    if parent["source_identity"]!=R.identity():raise RuntimeError("Parent source mismatch")
    gpu,data,target,raw,info,renderers,model,opt,sampler=R.setup(cfg,seed)
    if parent["input_identity"]!=info:raise RuntimeError("Parent data mismatch")
    R.restore(model,opt,sampler,parent)
    before=CONT.state_comparison(parent["optimizer"],opt.state_dict())
    expected=copy.deepcopy(parent["optimizer"])
    for group,e in zip(opt.param_groups,expected["param_groups"]):
        group["lr"]*=cfg["LR_multiplier"];e["lr"]*=cfg["LR_multiplier"]
    after=CONT.state_comparison(expected,opt.state_dict())
    for n,x in parent["model"].items():
        if not torch.equal(x.cpu(),model.state_dict()[n].detach().cpu()):raise RuntimeError("Parent model altered")
    state=np.random.get_state();old=parent["numpy_rng"]
    rng=dict(torch_CPU=torch.equal(torch.get_rng_state(),parent["torch_rng"]),
        torch_CUDA=torch.equal(torch.cuda.get_rng_state(),parent["cuda_rng"]),
        python=random.getstate()==parent["python_rng"],
        numpy=state[0]==old[0] and np.array_equal(state[1],old[1]) and state[2:]==old[2:],
        BS_sampler=sampler_equal(sampler.state_dict(),parent["BS_sampler"]))
    if not all(rng.values()):raise RuntimeError(("Parent RNG restore",rng))
    restore=dict(optimizer_before_LR=before,optimizer_after_only_LR=after,RNG=rng,
        parent_LR=[g["lr"]for g in parent["optimizer"]["param_groups"]],
        current_LR=[g["lr"]for g in opt.param_groups])
    return parent,gpu,data,target,raw,info,renderers,model,opt,sampler,restore

def snapshot(model,opt,sampler,step,history,stable,elapsed,src,cfg,info,parent_step):
    d=R.snapshot(model,opt,sampler,step,history,stable,elapsed,src,cfg,info)
    d.update(fresh_joint_initialization=False,starting_checkpoint=R.PATHS.relative(PARENT),
        parent_updates=parent_step,total_optimizer_updates=parent_step+step,
        qa_updates_included_in_budget=True)
    return d

def qa(cfg,seed,src):
    if (REPORT/"qa.json").exists():
        old=json.loads((REPORT/"qa.json").read_text())
        if old["source_identity"]!=src:raise RuntimeError("Preserved QA source mismatch")
        print(json.dumps(dict(status=old["status"],updates=old["updates"])));return
    started=time.monotonic()
    parent,gpu,data,target,raw,info,rs,model,opt,sampler,restore=setup(cfg,seed)
    pred,st=R.evaluate(model,rs,target,cfg["rows"])
    actual=(pred*data.scale).cpu().numpy()
    expected=np.load(R.ART/"raw_prediction.npy",allow_pickle=False)
    error=float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1e-30))
    if error>1e-6:raise RuntimeError(("Parent same-weight replay",error))
    records=[]
    for step in range(1,cfg["qa_updates"]+1):
        event=R.batch(model,opt,rs,target,sampler.draw())
        if step in [1,10,20]:
            torch.cuda.synchronize();records.append(dict(step=step,**event,seconds=time.monotonic()-started))
    out,final=R.evaluate(model,rs,target,cfg["rows"])
    ck=ART/"qa.pt"
    R.C.atomic_torch_save(ck,snapshot(model,opt,sampler,step,[],0,time.monotonic()-started,
        src,cfg,info,int(parent["step"])))
    if identity()!=src:raise RuntimeError("QA source/parent drift")
    result=dict(status="passed",updates=step,restore=restore,parent_replay_relative_L2=error,
        parent_FIT=st,after20_FIT=final,history=records,config=cfg,source_identity=src,GPU=gpu,
        qa_weights_preserved_for_continuation=True,optimizer_updates_count_toward3000=True,
        no_parent_file_update=True,parent_updates=int(parent["step"]),
        seconds_per_update=(records[-1]["seconds"]-records[0]["seconds"])/(cfg["qa_updates"]-1),
        checkpoint_sha256=R.C.sha256(ck),outer_truth_read=False,validation_truth_read=False)
    R.C.atomic_json(REPORT/"qa.json",result)
    emit(dict(stage="qa_complete",updates=step,parent_updates=int(parent["step"]),
        replay_relative_L2=error,seconds_per_update=result["seconds_per_update"],
        current_LR=restore["current_LR"],FIT_nmse=final["pooled"]["nmse"]))

def run(cfg,seed,src):
    done=REPORT/"result.json"
    if done.exists():
        d=json.loads(done.read_text())
        if d["source_identity"]!=src:raise RuntimeError("Completed source mismatch")
        print(json.dumps(dict(status=d["status"],fit_passed=d["fit_passed"])));return
    qa=json.loads((REPORT/"qa.json").read_text())
    if qa["status"]!="passed"or qa["source_identity"]!=src:raise RuntimeError("Matching QA required")
    started=time.monotonic()
    parent,gpu,data,target,raw,info,rs,model,opt,sampler,restoration=setup(cfg,seed)
    last=ART/"last.pt";load=last if last.exists()else ART/"qa.pt"
    ck=torch.load(load,map_location="cpu",weights_only=False)
    if ck["source_identity"]!=src or ck["config"]!=cfg:raise RuntimeError("Child resume mismatch")
    R.restore(model,opt,sampler,ck)
    step=ck["step"];history=ck["history"];stable=ck["stable_checks"];prior=ck["elapsed_seconds"]
    elapsed=lambda:prior+time.monotonic()-started
    parent_step=int(parent["step"])
    R.C.atomic_json(REPORT/"run_manifest.json",dict(source_identity=src,config=cfg,GPU=gpu,
        parent_checkpoint=R.PATHS.relative(PARENT),parent_sha256=R.C.sha256(PARENT),
        parent_updates=parent_step,restoration=restoration,
        numerical_change="current LR times0.2; parent weights, moments, all RNG, BS schedule retained",
        qa20_included_in_additional_budget=True,started_from_QA_checkpoint=True,
        parameter_reinitialization=False,loss_data_geometry_rays_heads_unchanged=True))
    heartbeat=time.monotonic();stop="maximum_additional_updates"
    while step<cfg["max_updates"]:
        if elapsed()>cfg["max_wall_seconds"]-45:
            stop="additional_wall_budget_reserved_for_final_checks";break
        event=R.batch(model,opt,rs,target,sampler.draw());step+=1
        check=step%cfg["eval_every"]==0
        if check:
            pred,stats=R.evaluate(model,rs,target,cfg["rows"])
            stable=stable+1 if R.gate(stats,cfg)else 0
            row=dict(stage="FIT_check",additional_updates=step,total_optimizer_updates=parent_step+step,
                stats=stats,stable_checks=stable,LR=[g["lr"]for g in opt.param_groups],
                elapsed_seconds=elapsed())
            history.append(row);emit(row);R.C.atomic_json(REPORT/"progress.json",row)
        if check or time.monotonic()-heartbeat>=30:
            if not check:
                row=dict(stage="training",additional_updates=step,total_optimizer_updates=parent_step+step,
                    **event,elapsed_seconds=elapsed(),ceiling=cfg["max_updates"])
                emit(row);R.C.atomic_json(REPORT/"progress.json",row)
            R.C.atomic_torch_save(last,snapshot(model,opt,sampler,step,history,stable,elapsed(),src,cfg,info,parent_step))
            heartbeat=time.monotonic()
        if stable>=cfg["stable_checks"]:
            stop="stable_two_position_fit";break
    pred,stats=R.evaluate(model,rs,target,cfg["rows"])
    final=ART/"final.pt"
    R.C.atomic_torch_save(final,snapshot(model,opt,sampler,step,history,stable,elapsed(),src,cfg,info,parent_step))
    R.C.atomic_npy(ART/"raw_prediction.npy",(pred*data.scale).cpu().numpy(),replace=True)
    R.C.atomic_npy(ART/"row_ids.npy",np.asarray(cfg["rows"],np.int64),replace=True)
    clean,_,_=R.build(data,seed)
    clean.load_state_dict(torch.load(final,map_location="cpu",weights_only=False)["model"])
    reloaded,_=R.evaluate(clean,rs,target,cfg["rows"])
    delta=float((reloaded-pred).abs().max())
    if delta!=0:raise RuntimeError(("Child clean reload",delta))
    if identity()!=src:raise RuntimeError("Final source/parent drift")
    passed=stable>=cfg["stable_checks"]and R.gate(stats,cfg)
    result=dict(status="complete",fit_passed=passed,additional_updates=step,
        parent_updates=parent_step,total_optimizer_updates=parent_step+step,
        stop_reason=stop,seconds=elapsed(),source_identity=src,config=cfg,GPU=gpu,
        full_FIT_stats=stats,history=history,stable_checks=stable,
        parent_FIT=json.loads(PARENT_RESULT.read_text())["full_FIT_stats"],
        final_LR=[g["lr"]for g in opt.param_groups],clean_reload_max_abs=delta,
        model_sha256=R.C.sha256(final),raw_prediction_sha256=R.C.sha256(ART/"raw_prediction.npy"),
        parent_checkpoint_unchanged=True,parent_weights_not_overwritten=True,
        additional_per_position_BS_counts=[(sampler.counts-parent["BS_sampler"]["counts"]).tolist()]*2,
        seed_reinitialization=False,optimizer_moments_reset=False,qa_updates_in_budget=True,
        input_identity=info,data_reads=data.info(),outer_truth_read=False,validation_labels_read=False,
        formal500_score=False,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        conclusion="lowLR continuation restored the two-position single-field FIT gate"if passed else
          "bounded lowLR continuation did not pass the two-position FIT gate")
    R.C.atomic_json(done,result)
    emit(dict(stage="complete",fit_passed=passed,additional_updates=step,total_optimizer_updates=parent_step+step,
        FIT_nmse=stats["pooled"]["nmse"],stop_reason=stop,seconds=elapsed()))

def main():
    p=argparse.ArgumentParser();p.add_argument("--mode",choices=["qa","run"],required=True);args=p.parse_args()
    REPORT.mkdir(parents=True,exist_ok=True);ART.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(CONFIG.read_text());seed=json.loads(R.CONFIG.read_text())["seed"];src=identity()
    try:
        if args.mode=="qa":qa(cfg,seed,src)
        else:run(cfg,seed,src)
    except BaseException:
        with (REPORT/f"failure_{args.mode}.log").open("a")as f:f.write(traceback.format_exc()+"\n")
        raise
if __name__=="__main__":main()
