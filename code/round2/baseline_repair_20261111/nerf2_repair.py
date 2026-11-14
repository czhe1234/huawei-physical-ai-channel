"""Fresh joint two-position NeRF2 repair; FIT-only, one of eight fields."""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, random, time, traceback
from pathlib import Path
import numpy as np
import torch
from project_paths import PATHS
from round2.paper_baseline_training_v1 import common as C
from round2.paper_baseline_training_v1 import nerf2_runner_v2 as P, nerf2_runner_v3 as V
from round2.paper_baseline_training_v1 import nerf2_native_frequency_heads_v4 as N
from round2.paper_baseline_training_v1 import nerf2_native_heads_joint_v5 as J
from round2.paper_baseline_training_v1 import field_fast as FF

ROOT=C.REPORT/"repair_20261111"
REPORT=ROOT/"nerf2"
ART=C.ART/"repair_20261111/nerf2"
CONFIG=ROOT/"config.json"
GPU="GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0"

def identity():
    files=[Path(__file__),CONFIG,Path(C.__file__),Path(P.__file__),Path(V.__file__),
           Path(N.__file__),Path(J.__file__),Path(FF.__file__),
           PATHS.code_root/"round2/paper_baseline_training_v1/field_runner.py",
           PATHS.code_root/"round2/paper_baseline_adapters_v1/field_models.py",
           PATHS.root/"plus-pro/code/measured/full_h_adapters_v1/nerf2.py",
           PATHS.root/"plus-pro/code/third_party/paper_baselines/nerf2/model.py",
           PATHS.root/"plus-pro/code/third_party/paper_baselines/nerf2/renderer.py"]
    return {PATHS.relative(p):C.sha256(p) for p in files}

def emit(value):
    REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/"events.jsonl").open("a") as f:f.write(json.dumps(value,allow_nan=False)+"\n")
    print(json.dumps(value,allow_nan=False),flush=True)

def inputs(cfg):
    data=C.FoldData(0,"screen","nerf2")
    rows=np.asarray(cfg["rows"],np.int64)
    if len(rows)!=2 or len(set(rows.tolist()))!=2:raise ValueError("Exactly two distinct FIT rows")
    if not set(rows).issubset(set(data.fit_ids)):raise PermissionError("Non-FIT row")
    if set(rows)&set(data.val_ids) or set(rows)&set(data.outer_ids):raise PermissionError("Held labels")
    if np.any(data.sites(rows)!=0):raise ValueError("This renderer unit binds site0")
    norm=data.read_fit(rows)[:,:128,:2]
    raw=data.read_fit(rows,normalized=False)[:,:128,:2]
    if not np.array_equal(norm,(raw/data.scale).astype(np.complex64)):raise ValueError("Normalization")
    energy=np.sum(np.abs(raw.astype(np.complex128))**2,axis=(1,2,3))
    if not np.isfinite(norm).all() or np.any(energy<=0):raise ValueError("Two nonzero FIT rows required")
    info=dict(rows=rows.tolist(),sites=data.sites(rows).tolist(),positions_m=data.positions(rows).tolist(),
        distance_m=float(np.linalg.norm(data.positions(rows)[0]-data.positions(rows)[1])),
        input_identity=data.inputs.identity,parent_fit_ids_sha256=C.digest_ids(data.fit_ids),
        parent_FIT_RMS=data.scale,raw_channel_identity=data.raw_identity,
        raw_label_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
        normalized_label_sha256=hashlib.sha256(norm.tobytes()).hexdigest(),
        raw_position_energy=energy.tolist(),relative_position_energy_db=(10*np.log10(energy/energy.max())).tolist(),
        actual_output_shape=[2,128,2,192],complete_H_shape=[2,256,4,192],
        trained_field="s0_b0_u0",field_meaning="site0, BS polarization0, UE polarization0 with vertical ports0/1",
        eight_fields="s{site0/1}_b{BS polarization0/1}_u{UE polarization0/1}; only one field is trained",
        BS_ids=list(range(128)),UE_ids=[0,1],frequency_ids=list(range(192)),
        zero_position_count=0,weak_zero_scope="Only these two nonzero FIT rows; no all-energy-range or zero-leakage recovery claim",
        query_truth_read=False,validation_truth_read=False,oracle_phase_delay_scale_correction=False)
    return data,norm,raw,info

def build(data,seed):
    C.seed_all(seed)
    model=V.build(data)
    N.freeze_NN_frequency(model)
    N.expand_head_rows(model)
    opt=P.optimizer(model)
    sampler=J.BSSampler(np.random.get_state())
    return model,opt,sampler

def gate(stats,cfg):
    return (stats["pooled"]["nmse"]<=cfg["pooled_nmse_max"] and all(
        x["pooled"]["nmse"]<=cfg["per_position_nmse_max"]
        and cfg["energy_min"]<=x["pooled"]["energy_ratio"]<=cfg["energy_max"]
        and all(u["nmse"]<=cfg["per_UE_nmse_max"] for u in x["per_UE"])
        for x in stats["per_position"]))

@torch.no_grad()
def evaluate(model,renderers,target,rows):
    predictions=[];records=[]
    for index,renderer in enumerate(renderers):
        pred,st=J.evaluate(model,renderer,target[index],[0,1])
        predictions.append(pred);records.append(dict(row_id=int(rows[index]),**st))
    se=sum(x["pooled"]["squared_error"] for x in records)
    te=sum(x["pooled"]["truth_energy"] for x in records)
    pe=sum(x["pooled"]["prediction_energy"] for x in records)
    return torch.cat(predictions),dict(pooled=dict(nmse=se/te,energy_ratio=pe/te,
        squared_error=se,truth_energy=te,prediction_energy=pe),per_position=records)

def batch(model,opt,renderers,target,bs):
    """Accumulate equally sized position losses, then take exactly one Adam step."""
    model.train();opt.zero_grad(set_to_none=True)
    losses=[]
    for i,r in enumerate(renderers):
        value=r(model,bs,[0,1])
        truth=J.label_subset(target[i],bs,[0,1])
        loss=.5*(value-truth).abs().square().mean()/len(renderers)
        if not bool(torch.isfinite(loss)):raise FloatingPointError("Joint loss")
        loss.backward();losses.append(float(loss.detach()))
    grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True))
    opt.step()
    return dict(loss=sum(losses),gradient_norm=grad,position_losses=losses)

def joint_gradient_qa(model,renderers,target,data,rows):
    bs=[0,17,30,47,64,81,98,127]
    values=torch.cat([r(model,bs,[0,1]) for r in renderers])
    truth=torch.cat([J.label_subset(t,bs,[0,1]) for t in target])
    combined=.5*(values-truth).abs().square().mean()
    expected=N.grad_map(model,combined)
    summed={n:torch.zeros_like(g) for n,g in expected.items()}
    for r,t in zip(renderers,target):
        loss=.25*(r(model,bs,[0,1])-J.label_subset(t,bs,[0,1])).abs().square().mean()
        g=N.grad_map(model,loss)
        for n in summed:summed[n]+=g[n]
    delta=N.relative_gradient(summed,expected,list(expected))
    if delta>1e-5:raise AssertionError(("joint accumulation",delta))
    checks=[J.reference_QA(model,r,data,target[i],dict(row_id=int(rows[i]),scalar_UE_vertical=0,scalar_BS_spatial=30),[0,1]) for i,r in enumerate(renderers)]
    return dict(accumulation_vs_combined_gradient_relative_L2=delta,slow_physical_reference=checks,
                same_BS_for_both_positions=True,optimizer_updates=0)

def snapshot(model,opt,sampler,step,history,stable,elapsed,src,cfg,info):
    return dict(model={n:p.detach().cpu().clone() for n,p in model.state_dict().items()},
        optimizer=opt.state_dict(),BS_sampler=copy.deepcopy(sampler.state_dict()),step=step,
        history=history,stable_checks=stable,elapsed_seconds=elapsed,source_identity=src,
        config=cfg,input_identity=info,torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state(),numpy_rng=np.random.get_state(),
        python_rng=random.getstate(),fresh_joint_initialization=True,starting_checkpoint=None)

def restore(model,opt,sampler,ck):
    model.load_state_dict(ck["model"]);opt.load_state_dict(ck["optimizer"])
    sampler.load_state_dict(copy.deepcopy(ck["BS_sampler"]))
    torch.set_rng_state(ck["torch_rng"]);torch.cuda.set_rng_state(ck["cuda_rng"])
    np.random.set_state(ck["numpy_rng"]);random.setstate(ck["python_rng"])

def setup(cfg,seed):
    gpu=C.device_guard(GPU)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data,norm,raw,info=inputs(cfg)
    target=torch.tensor(norm,device="cuda:0",dtype=torch.complex64)
    renderers=[J.JointRenderer(data,dict(row_id=row)) for row in cfg["rows"]]
    model,opt,sampler=build(data,seed)
    return gpu,data,target,raw,info,renderers,model,opt,sampler

def smoke(cfg,seed,src):
    if (REPORT/"smoke.json").exists():
        old=json.loads((REPORT/"smoke.json").read_text())
        if old["source_identity"]!=src:raise RuntimeError("Preserved smoke source differs")
        print(json.dumps(old));return
    gpu,data,target,raw,info,renderers,model,opt,sampler=setup(cfg,seed)
    check=joint_gradient_qa(model,renderers,target,data,cfg["rows"])
    with torch.no_grad():initial,initial_stats=evaluate(model,renderers,target,cfg["rows"])
    start=time.monotonic();history=[]
    for step in range(1,21):
        event=batch(model,opt,renderers,target,sampler.draw())
        if step in [1,10,20]:
            torch.cuda.synchronize()
            history.append(dict(step=step,**event,seconds=time.monotonic()-start))
    with torch.no_grad():final,final_stats=evaluate(model,renderers,target,cfg["rows"])
    if not torch.isfinite(final).all():raise FloatingPointError("Smoke output")
    if all(x["gradient_norm"]==0 for x in history):raise FloatingPointError("No smoke gradient")
    ck=ART/"smoke"/"checkpoint.pt"
    C.atomic_torch_save(ck,dict(model=model.state_dict(),source_identity=src))
    clone,_,_=build(data,seed)
    clone.load_state_dict(torch.load(ck,map_location="cpu",weights_only=False)["model"])
    reloaded,_=evaluate(clone,renderers,target,cfg["rows"])
    delta=float((reloaded-final).abs().max())
    if delta!=0:raise AssertionError(("Clean reload",delta))
    result=dict(status="passed",updates=20,source_identity=src,config=cfg,input_identity=info,
        GPU=gpu,checks=check,history=history,initial_stats=initial_stats,final_stats=final_stats,
        seconds_per_update=(history[-1]["seconds"]-history[0]["seconds"])/19,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),clean_reload_max_abs=delta,
        discarded_for_formal_run=True,formal_initialization="fresh seed, empty Adam and new balanced BS schedule",
        query_truth_read=False,validation_truth_read=False)
    C.atomic_json(REPORT/"smoke.json",result)
    emit(dict(stage="smoke_complete",status="passed",seconds_per_update=result["seconds_per_update"],
              peak_cuda_bytes=result["peak_cuda_bytes"],final_FIT_nmse=final_stats["pooled"]["nmse"]))

def run(cfg,seed,src):
    done=REPORT/"result.json"
    if done.exists():
        d=json.loads(done.read_text())
        if d["source_identity"]!=src or d["config"]!=cfg:raise RuntimeError("Completed source differs")
        emit(dict(stage="already_complete",status=d["status"],fit_passed=d["fit_passed"]));return
    qa=json.loads((REPORT/"smoke.json").read_text())
    if qa["status"]!="passed" or qa["source_identity"]!=src:raise RuntimeError("Smoke required for this exact source")
    started=time.monotonic()
    gpu,data,target,raw,info,renderers,model,opt,sampler=setup(cfg,seed)
    history=[];step=stable=0;prior=0.
    last=ART/"last.pt"
    if last.exists():
        ck=torch.load(last,map_location="cpu",weights_only=False)
        if ck["source_identity"]!=src or ck["config"]!=cfg or ck["input_identity"]!=info:
            raise RuntimeError("Resume source/data/config mismatch")
        restore(model,opt,sampler,ck)
        history=ck["history"];step=ck["step"];stable=ck["stable_checks"];prior=ck["elapsed_seconds"]
    elapsed=lambda:prior+time.monotonic()-started
    manifest=dict(source_identity=src,config=cfg,seed=seed,GPU=gpu,input_identity=info,
        python=__import__("platform").python_version(),torch=torch.__version__,numpy=np.__version__,
        command="python -m round2.baseline_repair_20261111.nerf2_repair --mode run",
        optimization="fresh both-position combined native MSE; one Adam state and update per joint batch",
        architecture_changed_from_v8=False,initialized_from_old_weights=False,
        frequency_192_heads=True,physical_ray_sample_far=[324,32,810],
        phase_delay_amplitude_query_alignment=False,formal_test=False)
    C.atomic_json(REPORT/"run_manifest.json",manifest)
    if not (ART/"raw_FIT_field_labels.npy").exists():C.atomic_npy(ART/"raw_FIT_field_labels.npy",raw)
    if not (ART/"normalized_FIT_field_labels.npy").exists():C.atomic_npy(ART/"normalized_FIT_field_labels.npy",target.cpu().numpy())
    if step==0:
        initial,st=evaluate(model,renderers,target,cfg["rows"])
        emit(dict(stage="formal_initial",stats=st,elapsed_seconds=elapsed()))
        C.atomic_torch_save(ART/"initial.pt",snapshot(model,opt,sampler,0,[],0,elapsed(),src,cfg,info))
    stop="maximum_updates";last_heartbeat=time.monotonic();last_event=None
    # Reserve a bounded minute for full FIT evaluation, atomic output and clean reload.
    while step<cfg["max_updates"]:
        if elapsed()>cfg["max_wall_seconds"]-60:
            stop="wall_budget_reserved_for_final_checks";break
        event=batch(model,opt,renderers,target,sampler.draw());step+=1
        now=time.monotonic();check=step%cfg["eval_every"]==0
        if check:
            pred,st=evaluate(model,renderers,target,cfg["rows"])
            stable=stable+1 if gate(st,cfg) else 0
            last_event=dict(stage="FIT_check",step=step,stats=st,stable_checks=stable,
                passed=gate(st,cfg),LR=[g["lr"] for g in opt.param_groups],
                elapsed_seconds=elapsed(),BS_count_min=int(sampler.counts.min()),BS_count_max=int(sampler.counts.max()))
            history.append(last_event);emit(last_event)
            C.atomic_json(REPORT/"progress.json",last_event)
        if check or now-last_heartbeat>=30:
            if not check:
                heartbeat=dict(stage="training",step=step,**event,elapsed_seconds=elapsed(),
                    ceiling=cfg["max_updates"],wall_ceiling=cfg["max_wall_seconds"])
                emit(heartbeat);C.atomic_json(REPORT/"progress.json",heartbeat)
            C.atomic_torch_save(last,snapshot(model,opt,sampler,step,history,stable,elapsed(),src,cfg,info))
            last_heartbeat=time.monotonic()
        if stable>=cfg["stable_checks"]:
            stop="stable_two_position_fit";break
    pred,stats=evaluate(model,renderers,target,cfg["rows"])
    final=ART/"final.pt"
    C.atomic_torch_save(final,snapshot(model,opt,sampler,step,history,stable,elapsed(),src,cfg,info))
    C.atomic_npy(ART/"raw_prediction.npy",(pred*data.scale).cpu().numpy(),replace=True)
    C.atomic_npy(ART/"row_ids.npy",np.asarray(cfg["rows"],np.int64),replace=True)
    C.atomic_json(ART/"axes.json",info)
    clean,_,_=build(data,seed)
    clean.load_state_dict(torch.load(final,map_location="cpu",weights_only=False)["model"])
    reloaded,_=evaluate(clean,renderers,target,cfg["rows"])
    delta=float((reloaded-pred).abs().max())
    if delta!=0:raise AssertionError(("Formal clean reload",delta))
    if identity()!=src:raise RuntimeError("Source changed while running")
    fit=stable>=cfg["stable_checks"] and gate(stats,cfg)
    result=dict(status="complete",fit_passed=fit,stop_reason=stop,updates=step,
        max_updates=cfg["max_updates"],seconds=elapsed(),wall_budget_seconds=cfg["max_wall_seconds"],
        stable_checks=stable,full_FIT_stats=stats,history=history,
        source_identity=src,config=cfg,input_identity=info,GPU=gpu,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),clean_reload_max_abs=delta,
        model_sha256=C.sha256(final),raw_prediction_sha256=C.sha256(ART/"raw_prediction.npy"),
        two_position_BS_counts=[sampler.counts.tolist(),sampler.counts.tolist()],
        data_reads=data.info(),initialized_from_old_weights=False,smoke_weights_reused=False,
        validation_labels_read=False,outer_truth_read=False,formal500_score=False,
        conclusion="two-position single-field FIT recovered" if fit else
            "bounded joint optimization did not pass the preregistered two-position FIT gate",
        zero_weak_limit=info["weak_zero_scope"])
    C.atomic_json(done,result)
    emit(dict(stage="formal_complete",status="complete",fit_passed=fit,updates=step,
        stop_reason=stop,FIT_nmse=stats["pooled"]["nmse"],seconds=elapsed()))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--mode",choices=["smoke","run"],required=True)
    args=ap.parse_args()
    REPORT.mkdir(parents=True,exist_ok=True);ART.mkdir(parents=True,exist_ok=True)
    cfgroot=json.loads(CONFIG.read_text());cfg=cfgroot["nerf2"];seed=cfgroot["seed"];src=identity()
    try:
        if args.mode=="smoke":smoke(cfg,seed,src)
        else:run(cfg,seed,src)
    except BaseException:
        with (REPORT/f"failure_{args.mode}.log").open("a") as f:f.write(traceback.format_exc()+"\n")
        raise
if __name__=="__main__":main()
