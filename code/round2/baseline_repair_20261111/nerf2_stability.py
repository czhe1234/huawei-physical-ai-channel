"""Pre-registered500-step stability completion after the fixed2000 comparison."""
from __future__ import annotations
import copy,json
from pathlib import Path
import torch
from . import nerf2_repair as R,nerf2_low_lr as L
from . import nerf2_position_loss as M,nerf2_position_loss_v2 as V

REPORT=R.REPORT/"stability_confirmation"
ART=R.ART/"stability_confirmation"
CONFIG=REPORT/"config.json"
PARENT=V.ART/"relative/final.pt"
PARENT_RESULT=V.REPORT/"relative/result.json"
MATCHED=V.REPORT/"result.json"
BASE_SETUP=M.setup
PARENT_FP=None

def identity():
    src=json.loads(PARENT_RESULT.read_text())["source_identity"].copy()
    for p in [Path(__file__),CONFIG,PARENT,PARENT_RESULT,MATCHED]:
        src[R.PATHS.relative(p)]=R.C.sha256(p)
    return src

def safe_setup(cfg,seed,parent):
    before=V.optimizer_fingerprint(parent["optimizer"])
    if before!=PARENT_FP:raise RuntimeError("Confirmation canonical parent mutated")
    values=BASE_SETUP(cfg,seed,copy.deepcopy(parent))
    actual=V.optimizer_fingerprint(values[7].state_dict())
    if actual!=PARENT_FP or V.optimizer_fingerprint(parent["optimizer"])!=PARENT_FP:
        raise RuntimeError("Confirmation exact optimizer restore failed")
    shared=[]
    for key,st in values[7].state_dict()["state"].items():
        for name,value in st.items():
            old=parent["optimizer"]["state"][key][name]
            if isinstance(value,torch.Tensor)and isinstance(old,torch.Tensor):
                if value.device.type=="cpu"and old.device.type=="cpu"and value.data_ptr()==old.data_ptr():
                    shared.append([key,name])
    if shared:raise RuntimeError(("Shared parent CPU state",shared))
    R.C.atomic_json(REPORT/"initial_optimizer_proof.json",dict(parent=PARENT_FP,branch=actual,
        shared_CPU_storage=shared,deepcopy_before_load=True,all81_state_tensors_exact=True))
    return values

def main():
    global PARENT_FP
    REPORT.mkdir(parents=True,exist_ok=True);ART.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(CONFIG.read_text());prior=json.loads(PARENT_RESULT.read_text())
    if not prior["endpoint_gate_passed"] or prior["stable_checks"]!=2:
        raise RuntimeError("Registered eligibility requires endpoint gate and inherited stable2")
    if cfg["max_updates"]!=500 or cfg["stable_checks"]!=3:
        raise RuntimeError("Confirmation budget or original gate changed")
    for rel,h in prior["source_identity"].items():
        if R.C.sha256(R.PATHS.root/rel)!=h:raise RuntimeError(("Parent source changed",rel))
    parent=torch.load(PARENT,map_location="cpu",weights_only=False)
    PARENT_FP=V.optimizer_fingerprint(parent["optimizer"])
    if PARENT_FP["bias_times"]!=[12717]or PARENT_FP["tensor_count"]!=81:raise RuntimeError("Parent clock")
    initial_hash=R.C.sha256(PARENT);src=identity()
    M.REPORT=REPORT;M.ART=ART;M.CONFIG=CONFIG;M.PARENT=PARENT;M.PARENT_RESULT=PARENT_RESULT
    # Original run_arm's replay reference is L.ART/raw_prediction.npy; point it to the registered parent.
    L.ART=PARENT.parent
    M.setup=safe_setup;M.identity=identity
    M.signal.signal(M.signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError("Confirmation300s wall cap")))
    stage=M.run_arm("continuation",cfg,cfg["seed"],parent,src,0.)
    after=V.optimizer_fingerprint(parent["optimizer"])
    if after!=PARENT_FP or R.C.sha256(PARENT)!=initial_hash:raise RuntimeError("Confirmation parent changed")
    ck=torch.load(ART/"continuation/final.pt",map_location="cpu",weights_only=False)
    times=[int(x["step"])for x in ck["optimizer"]["state"].values()if"step"in x]
    if set(times)!={12717+stage["additional_updates"]}:raise RuntimeError("Confirmation Adam time")
    new_checks=stage["history"]
    if len(new_checks)!=1 or new_checks[0]["additional_updates"]!=500:
        raise RuntimeError("Only one registered500-step stability check")
    passed_now=bool(stage["endpoint_gate_passed"])
    combined=prior["stable_checks"]+1 if passed_now else 0
    result=dict(status="complete",scope="2FIT positions, site0 BS-pol0 UE-pol0;128BSx2UEx192F, one of8fields",
       endpoint_gate_passed=passed_now,inherited_stable_checks=prior["stable_checks"],
       additional_real_500step_checks=1,total_consecutive_stable_checks=combined,
       original_required_stable_checks=3,fit_passed=combined>=3 and passed_now,
       additional_optimizer_updates=stage["additional_updates"],parent_total_optimizer_updates=12717,
       total_optimizer_updates=12717+stage["additional_updates"],actual_Adam_times=times,
       full_FIT_stats=stage["full_FIT_stats"],raw_stage_local_stable_count=stage["stable_checks"],
       source_identity=src,config=cfg,GPU=stage["GPU"],seconds=stage["seconds"],
       clean_reload_max_abs=stage["clean_reload_max_abs"],model_sha256=stage["model_sha256"],
       raw_prediction_sha256=stage["raw_prediction_sha256"],peak_cuda_bytes=stage["peak_cuda_bytes"],
       immutable_parent_optimizer_before=PARENT_FP,immutable_parent_optimizer_after=after,
       immutable_parent_file_sha256=initial_hash,
       optimizer_state_preserved_and_deepcopied=True,all_RNG_and_BS_continued=True,
       additional_budget_not_part_of_matched2000_result=True,
       fixed2000_comparison_unchanged=True,loss_LR_architecture_position_gain_correction_unchanged=True,
       no_validation_or500_read=True,any_check_failure_forbids_further_extension=True,
       note="Stable checks are original candidate1500 and2000 plus actual continuation2500, not repeated evaluation of one weight.")
    R.C.atomic_json(REPORT/"result.json",result)
    print(json.dumps({k:result[k]for k in ["status","fit_passed","total_consecutive_stable_checks",
        "total_optimizer_updates","additional_optimizer_updates","seconds"]}))
if __name__=="__main__":
    try:main()
    except BaseException:
        REPORT.mkdir(parents=True,exist_ok=True)
        with (REPORT/"failure.log").open("a")as f:f.write(__import__("traceback").format_exc()+"\n")
        raise
