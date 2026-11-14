"""Final matched two-arm NeRF2 FIT continuation: native vs per-position loss."""
from __future__ import annotations
import argparse,copy,json,time,traceback,random,signal
from pathlib import Path
import numpy as np
import torch
from . import nerf2_repair as R,nerf2_low_lr as L
from round2.paper_baseline_training_v1 import nerf2_native_heads_continuation_v4 as CONT

REPORT=R.REPORT/"position_loss"
ART=R.ART/"position_loss"
CONFIG=REPORT/"config.json"
PARENT=L.ART/"final.pt"
PARENT_RESULT=L.REPORT/"result.json"
ARMS=("control","relative")

def identity():
    src=L.identity()
    for p in [Path(__file__),CONFIG,PARENT,PARENT_RESULT,L.ART/"raw_prediction.npy"]:
        src[R.PATHS.relative(p)]=R.C.sha256(p)
    return src

def emit(arm,d):
    d=dict(arm=arm,**d)
    with (REPORT/"events.jsonl").open("a")as f:f.write(json.dumps(d,allow_nan=False)+"\n")
    print(json.dumps(d,allow_nan=False),flush=True)

def setup(cfg,seed,parent):
    gpu,data,target,raw,info,rs,model,opt,sampler=R.setup(cfg,seed)
    if info!=parent["input_identity"]:raise RuntimeError("Parent data differ")
    R.restore(model,opt,sampler,parent)
    state_check=CONT.state_comparison(parent["optimizer"],opt.state_dict())
    for n,x in parent["model"].items():
        if not torch.equal(x.cpu(),model.state_dict()[n].detach().cpu()):raise RuntimeError("Parent model restore")
    if not torch.equal(torch.get_rng_state(),parent["torch_rng"]):raise RuntimeError("CPU RNG restore")
    if not torch.equal(torch.cuda.get_rng_state(),parent["cuda_rng"]):raise RuntimeError("CUDA RNG restore")
    if random.getstate()!=parent["python_rng"]:raise RuntimeError("Python RNG restore")
    old=parent["numpy_rng"];now=np.random.get_state()
    if not(old[0]==now[0]and np.array_equal(old[1],now[1])and old[2:]==now[2:]):raise RuntimeError("NumPy RNG restore")
    if not L.sampler_equal(sampler.state_dict(),parent["BS_sampler"]):raise RuntimeError("BS state restore")
    powers=np.mean(np.abs(target.detach().cpu().numpy().astype(np.complex128))**2,axis=(1,2,3))
    if not np.array_equal(powers,np.asarray(cfg["position_power_normalized"])):raise RuntimeError("Frozen position power changed")
    return gpu,data,target,raw,info,rs,model,opt,sampler,state_check

def update(model,opt,rs,target,bs,arm,cfg):
    model.train();opt.zero_grad(set_to_none=True);losses=[]
    denominators=[1.,1.]if arm=="control"else cfg["position_power_denominators"]
    for i,r in enumerate(rs):
        value=r(model,bs,[0,1]);truth=R.J.label_subset(target[i],bs,[0,1])
        loss=.5*(value-truth).abs().square().mean()/len(rs)/denominators[i]
        if not bool(torch.isfinite(loss)):raise FloatingPointError("Loss")
        loss.backward();losses.append(float(loss.detach()))
    grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True))
    opt.step()
    return dict(loss=sum(losses),position_losses=losses,gradient_norm=grad)

def snapshot(model,opt,sampler,step,history,stable,elapsed,src,cfg,info,parent_step,schedule,arm):
    d=R.snapshot(model,opt,sampler,step,history,stable,elapsed,src,cfg,info)
    d.update(fresh_joint_initialization=False,starting_checkpoint=R.PATHS.relative(PARENT),
       parent_updates=parent_step,total_optimizer_updates=parent_step+step,arm=arm,
       schedule=np.asarray(schedule,np.int64),qa20_included_in_fixed2000=True)
    return d

def run_arm(arm,cfg,seed,parent,src,prior_completed_seconds):
    rr=REPORT/arm;aa=ART/arm;rr.mkdir(parents=True,exist_ok=True);aa.mkdir(parents=True,exist_ok=True)
    done=rr/"result.json"
    if done.exists():
        d=json.loads(done.read_text())
        if d["source_identity"]!=src or d["config"]!=cfg:raise RuntimeError("Completed arm source differs")
        return d
    started=time.monotonic()
    gpu,data,target,raw,info,rs,model,opt,sampler,restore_check=setup(cfg,seed,parent)
    history=[];schedule=[];stable=step=0;prior=0.
    last=aa/"last.pt";parent_step=int(parent["total_optimizer_updates"])
    if last.exists():
        ck=torch.load(last,map_location="cpu",weights_only=False)
        if ck["source_identity"]!=src or ck["config"]!=cfg or ck["arm"]!=arm:raise RuntimeError("Resume mismatch")
        R.restore(model,opt,sampler,ck);history=ck["history"];stable=ck["stable_checks"];step=ck["step"]
        schedule=ck["schedule"].tolist();prior=ck["elapsed_seconds"]
    elapsed=lambda:prior+time.monotonic()-started
    remaining=cfg["global_wall_seconds"]-prior_completed_seconds-elapsed()
    if remaining<=0:raise TimeoutError("Matched global wall budget exhausted")
    signal.setitimer(signal.ITIMER_REAL,remaining)
    if step==0:
        pred,initial=R.evaluate(model,rs,target,cfg["rows"])
        actual=(pred*data.scale).cpu().numpy();expected=np.load(L.ART/"raw_prediction.npy",allow_pickle=False)
        replay=float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1e-30))
        if replay>1e-6:raise RuntimeError(("Same parent replay",replay))
        R.C.atomic_json(rr/"initial_QA.json",dict(parent_step=parent_step,parent_sha256=R.C.sha256(PARENT),
          replay_relative_L2=replay,optimizer_state_restore=restore_check,
          complete_model_and_all_RNG_restored=True,initial_FIT=initial,source_identity=src,
          LR=[g["lr"]for g in opt.param_groups],arm=arm))
    R.C.atomic_json(rr/"run_manifest.json",dict(source_identity=src,config=cfg,GPU=gpu,arm=arm,
       parent_total_optimizer_updates=parent_step,current_LR=[g["lr"]for g in opt.param_groups],
       objective="0.5 mean_i MSE_i"if arm=="control"else"0.5 mean_i MSE_i/max(P_i,1e-6)",
       network_loss_denominator=cfg["position_power_denominators"]if arm=="relative"else[1.,1.],
       target="equal position FIT quality; no vanished-gradient claim",source_parent_unchanged=True,
       changed_item="loss only",input_identity=info,no_validation_or_outer_labels=True))
    heartbeat=time.monotonic();stop="fixed2000_updates"
    while step<cfg["max_updates"]:
        if prior_completed_seconds+elapsed()>cfg["global_wall_seconds"]-45:
            stop="global900s_wall_budget";break
        bs=sampler.draw();schedule.append(bs)
        event=update(model,opt,rs,target,bs,arm,cfg);step+=1
        check=step%cfg["eval_every"]==0
        if check:
            pred,stats=R.evaluate(model,rs,target,cfg["rows"])
            stable=stable+1 if R.gate(stats,cfg)else 0
            row=dict(stage="FIT_check",additional_updates=step,total_optimizer_updates=parent_step+step,
                stats=stats,stable_checks=stable,gate_passed_now=R.gate(stats,cfg),
                elapsed_seconds=elapsed(),LR=[g["lr"]for g in opt.param_groups])
            history.append(row);emit(arm,row);R.C.atomic_json(rr/"progress.json",row)
        if step==cfg["qa_updates"] or check or time.monotonic()-heartbeat>=30:
            if not check:
                row=dict(stage="training",additional_updates=step,**event,elapsed_seconds=elapsed(),
                    prior_completed_arm_seconds=prior_completed_seconds)
                emit(arm,row);R.C.atomic_json(rr/"progress.json",row)
            R.C.atomic_torch_save(last,snapshot(model,opt,sampler,step,history,stable,elapsed(),src,cfg,info,parent_step,schedule,arm))
            heartbeat=time.monotonic()
        if step==cfg["qa_updates"]:
            R.C.atomic_json(rr/"qa20.json",dict(status="passed",updates=step,counted_in_fixed2000=True,
               discarded_or_repeated=False,finite_gradients=True,last_training_batch=event,
               elapsed_seconds=elapsed(),source_identity=src))
        # Always finish the fixed2000 matched budget even after a transient gate pass.
    pred,stats=R.evaluate(model,rs,target,cfg["rows"])
    final=aa/"final.pt"
    R.C.atomic_torch_save(final,snapshot(model,opt,sampler,step,history,stable,elapsed(),src,cfg,info,parent_step,schedule,arm))
    R.C.atomic_npy(aa/"raw_prediction.npy",(pred*data.scale).cpu().numpy(),replace=True)
    R.C.atomic_npy(aa/"BS_schedule.npy",np.asarray(schedule,np.int64),replace=True)
    clean,_,_=R.build(data,seed);clean.load_state_dict(torch.load(final,map_location="cpu",weights_only=False)["model"])
    restored,_=R.evaluate(clean,rs,target,cfg["rows"])
    error=float((restored-pred).abs().max())
    if error!=0:raise RuntimeError(("Clean reload",error))
    if identity()!=src:raise RuntimeError("Source or parent drift")
    result=dict(status="complete"if step==cfg["max_updates"]else"incomplete_budget",arm=arm,
       additional_updates=step,total_optimizer_updates=parent_step+step,parent_updates=parent_step,
       full_FIT_stats=stats,history=history,stable_checks=stable,
       endpoint_gate_passed=R.gate(stats,cfg),fit_passed=stable>=cfg["stable_checks"]and R.gate(stats,cfg),
       stop_reason=stop,seconds=elapsed(),clean_reload_max_abs=error,
       source_identity=src,config=cfg,GPU=gpu,input_identity=info,data_reads=data.info(),
       model_sha256=R.C.sha256(final),raw_prediction_sha256=R.C.sha256(aa/"raw_prediction.npy"),
       BS_schedule_sha256=R.C.sha256(aa/"BS_schedule.npy"),qa20_counted_inside2000=True,
       parent_model_or_optimizer_modified=False,validation_or_outer_truth_read=False,
       gain_correction=False,architecture_changed=False,position_inputs_changed=False,
       peak_cuda_bytes=torch.cuda.max_memory_allocated(),parent_FIT=json.loads(PARENT_RESULT.read_text())["full_FIT_stats"])
    R.C.atomic_json(done,result)
    emit(arm,dict(stage="complete",status=result["status"],fit_passed=result["fit_passed"],updates=step,
       pooled_nmse=stats["pooled"]["nmse"],seconds=elapsed()))
    signal.setitimer(signal.ITIMER_REAL,0)
    del clean,model,opt
    torch.cuda.empty_cache()
    return result

def main():
    REPORT.mkdir(parents=True,exist_ok=True);ART.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(CONFIG.read_text());seed=cfg["seed"]
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError("Matched global900s wall cap")))
    parent=torch.load(PARENT,map_location="cpu",weights_only=False)
    if parent["source_identity"]!=L.identity():raise RuntimeError("LowLR parent source mismatch")
    src=identity();results=[]
    for arm in ARMS:
        results.append(run_arm(arm,cfg,seed,parent,src,sum(x["seconds"]for x in results)))
    schedules=[np.load(ART/arm/"BS_schedule.npy",allow_pickle=False)for arm in ARMS]
    matched=np.array_equal(*schedules)
    if not matched and all(x["status"]=="complete"for x in results):raise RuntimeError("Matched BS schedule failed")
    common=min(len(x)for x in schedules)
    if not np.array_equal(schedules[0][:common],schedules[1][:common]):raise RuntimeError("Schedule prefix differs")
    summary=dict(status="complete"if all(x["status"]=="complete"for x in results)else"incomplete_budget",
       arms=results,matched_entire_BS_schedule=matched,matched_prefix_steps=common,
       parent_total_optimizer_updates=int(parent["total_optimizer_updates"]),
       total_new_optimizer_updates=sum(x["additional_updates"]for x in results),
       seconds=sum(x["seconds"]for x in results),source_identity=src,config=cfg,
       selection_information="FIT-only; no validation/500; no rescaling correction",
       loss_scale_difference_disclosed=True,only_numerical_training_change="declared loss denominator",
       note="Two equal2000-update continuations from same weights/moments/RNG, not a new paper ranking")
    R.C.atomic_json(REPORT/"result.json",summary)
    print(json.dumps({k:summary[k]for k in["status","matched_entire_BS_schedule","total_new_optimizer_updates","seconds"]}))
if __name__=="__main__":
    try:main()
    except BaseException:
        REPORT.mkdir(parents=True,exist_ok=True)
        with (REPORT/"failure.log").open("a")as f:f.write(traceback.format_exc()+"\n")
        raise
