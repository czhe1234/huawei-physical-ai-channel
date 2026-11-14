"""Repair CPU Adam step alias; reuse valid first control, rerun candidate only."""
from __future__ import annotations
import copy,hashlib,json
from pathlib import Path
import torch,numpy as np
from . import nerf2_position_loss as M
from . import nerf2_repair as R,nerf2_low_lr as L

OLD_REPORT=R.REPORT/"position_loss"
OLD_ART=R.ART/"position_loss"
REPORT=R.REPORT/"position_loss_v2"
ART=R.ART/"position_loss_v2"
CONFIG=REPORT/"config.json"
BASE_SETUP=M.setup
BASE_IDENTITY=M.identity
CANONICAL_FINGERPRINT=None

def optimizer_fingerprint(state):
    h=hashlib.sha256()
    h.update(json.dumps(state["param_groups"],sort_keys=True,default=str).encode())
    tensor_count=0
    for key in sorted(state["state"]):
        h.update(str(key).encode())
        for name,value in sorted(state["state"][key].items()):
            h.update(name.encode())
            if isinstance(value,torch.Tensor):
                data=value.detach().cpu().contiguous()
                h.update(str(data.dtype).encode());h.update(str(tuple(data.shape)).encode())
                h.update(data.numpy().tobytes());tensor_count+=1
            else:h.update(repr(value).encode())
    return dict(sha256=h.hexdigest(),tensor_count=tensor_count,
        bias_times=sorted({int(x["step"])for x in state["state"].values()if"step"in x}))

def identity():
    src=BASE_IDENTITY()
    files=[Path(__file__),OLD_REPORT/"config.json",OLD_REPORT/"control/result.json",
        OLD_REPORT/"control/initial_QA.json",OLD_ART/"control/final.pt",
        OLD_ART/"control/raw_prediction.npy",OLD_ART/"control/BS_schedule.npy"]
    for p in files:src[R.PATHS.relative(p)]=R.C.sha256(p)
    return src

def safe_setup(cfg,seed,parent):
    before=optimizer_fingerprint(parent["optimizer"])
    if before!=CANONICAL_FINGERPRINT:raise RuntimeError("Canonical parent already mutated")
    # Critical fix: load_state_dict may retain CPU step tensors by reference.
    # Never hand the shared immutable parent object to an optimizer.
    branch_parent=copy.deepcopy(parent)
    values=BASE_SETUP(cfg,seed,branch_parent)
    opt=values[7]
    actual=optimizer_fingerprint(opt.state_dict())
    after=optimizer_fingerprint(parent["optimizer"])
    if actual!=CANONICAL_FINGERPRINT or after!=CANONICAL_FINGERPRINT:
        raise RuntimeError("Immutable optimizer fingerprint failed")
    shared=[]
    for key,st in opt.state_dict()["state"].items():
        reference=parent["optimizer"]["state"][key]
        for name,value in st.items():
            old=reference[name]
            if isinstance(value,torch.Tensor)and isinstance(old,torch.Tensor):
                if value.device.type=="cpu"and old.device.type=="cpu"and value.data_ptr()==old.data_ptr():
                    shared.append([key,name])
    if shared:raise RuntimeError(("Shared parent CPU optimizer storage",shared))
    R.C.atomic_json(REPORT/"candidate_initial_optimizer_proof.json",
        dict(immutable_parent=before,restored_branch=actual,canonical_after_restore=after,
             shared_CPU_storage=shared,deepcopy_before_load_state_dict=True,
             all_parent_states_verified=True,initial_model_replay_checked_by_original_runner=True))
    return values

def verify_valid_control():
    control=json.loads((OLD_REPORT/"control/result.json").read_text())
    for rel,h in control["source_identity"].items():
        if R.C.sha256(R.PATHS.root/rel)!=h:raise RuntimeError(("Original control source changed",rel))
    if control["status"]!="complete"or control["additional_updates"]!=2000:
        raise RuntimeError("Original first control not complete")
    ck=torch.load(OLD_ART/"control/final.pt",map_location="cpu",weights_only=False)
    times=[int(x["step"])for x in ck["optimizer"]["state"].values()if"step"in x]
    if len(times)!=27 or set(times)!={12717}:raise RuntimeError(("Control bias times",times))
    if R.C.sha256(OLD_ART/"control/final.pt")!=control["model_sha256"]:raise RuntimeError("Control model hash")
    if R.C.sha256(OLD_ART/"control/raw_prediction.npy")!=control["raw_prediction_sha256"]:raise RuntimeError("Control raw hash")
    if R.C.sha256(OLD_ART/"control/BS_schedule.npy")!=control["BS_schedule_sha256"]:raise RuntimeError("Control schedule hash")
    qa=json.loads((OLD_REPORT/"control/initial_QA.json").read_text())
    if qa["parent_step"]!=10717 or qa["replay_relative_L2"]>1e-6 or not qa["optimizer_state_restore"]["exact"]:
        raise RuntimeError("Control initial state evidence")
    return control,dict(valid_first_control_reused=True,
       path=R.PATHS.relative(OLD_REPORT/"control/result.json"),
       parent_bias_time=10717,actual_optimizer_calls=2000,final_internal_bias_times=times,
       initial_model_replay_relative_L2=qa["replay_relative_L2"],
       initialization_basis="First branch directly from fresh disk parent before any branch optimizer updates; initial exact state check and final all27 bias counters agree.",
       original_source_identity=control["source_identity"],original_model_sha256=control["model_sha256"])

def main():
    global CANONICAL_FINGERPRINT
    REPORT.mkdir(parents=True,exist_ok=True);ART.mkdir(parents=True,exist_ok=True)
    M.signal.signal(M.signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError("Matched global900s wall cap")))
    cfg=json.loads(CONFIG.read_text());oldcfg=json.loads((OLD_REPORT/"config.json").read_text())
    for k in ["rows","seed","max_updates","eval_every","stable_checks","position_power_normalized",
              "position_power_denominators","pooled_nmse_max","per_position_nmse_max",
              "per_UE_nmse_max","energy_min","energy_max","qa_updates"]:
        if cfg[k]!=oldcfg[k]:raise RuntimeError(("Numerical config changed",k))
    # Keep original source files immutable; redirect this process's new outputs.
    M.REPORT=REPORT;M.ART=ART;M.CONFIG=CONFIG
    M.setup=safe_setup;M.identity=identity
    control,proof=verify_valid_control()
    R.C.atomic_json(REPORT/"valid_control_reuse.json",proof)
    parent=torch.load(M.PARENT,map_location="cpu",weights_only=False)
    if parent["source_identity"]!=L.identity():raise RuntimeError("Disk parent source changed")
    CANONICAL_FINGERPRINT=optimizer_fingerprint(parent["optimizer"])
    if CANONICAL_FINGERPRINT["bias_times"]!=[10717]or CANONICAL_FINGERPRINT["tensor_count"]!=81:
        raise RuntimeError("Unexpected immutable parent optimizer")
    parent_file_hash=R.C.sha256(M.PARENT);src=identity()
    candidate=M.run_arm("relative",cfg,cfg["seed"],parent,src,control["seconds"])
    after=optimizer_fingerprint(parent["optimizer"])
    if after!=CANONICAL_FINGERPRINT or R.C.sha256(M.PARENT)!=parent_file_hash:
        raise RuntimeError("Parent was changed by corrected candidate")
    candidate_ck=torch.load(ART/"relative/final.pt",map_location="cpu",weights_only=False)
    end_times=[int(x["step"])for x in candidate_ck["optimizer"]["state"].values()if"step"in x]
    expected=10717+candidate["additional_updates"]
    if set(end_times)!={expected}or len(end_times)!=27:raise RuntimeError("Corrected candidate actual bias times")
    a=np.load(OLD_ART/"control/BS_schedule.npy",allow_pickle=False)
    b=np.load(ART/"relative/BS_schedule.npy",allow_pickle=False)
    match=np.array_equal(a,b)
    if candidate["status"]=="complete"and not match:raise RuntimeError("Matched schedule differs")
    audit=json.loads((OLD_REPORT/"optimizer_step_alias_audit.json").read_text())
    invalid=audit["relative_partial"]["header_step"]
    summary=dict(status=candidate["status"],control=control,candidate=candidate,
        control_reused=proof,source_identity=src,config=cfg,entire_BS_schedule_equal=match,
        immutable_parent_optimizer_before=CANONICAL_FINGERPRINT,immutable_parent_optimizer_after=after,
        immutable_parent_file_sha256=parent_file_hash,candidate_final_bias_times=end_times,
        candidate_real_optimizer_calls=candidate["additional_updates"],
        valid_new_optimizer_calls=2000+candidate["additional_updates"],
        old_invalid_candidate_saved_optimizer_calls=invalid,
        invalid_candidate_unsaved_optimizer_calls_unknown=True,
        old_invalid_bias_times_are_not_extra_optimizer_calls=True,
        old_attempt_preserved=True,control_not_retrained=True,
        corrected_code_delta="Fresh disk parent; per-arm deepcopy before optimizer load; all81 state fingerprint and CPU storage isolation assertions. Original renderer, objective, LR and2000 schedule unchanged.",
        valid_pair_active_seconds=control["seconds"]+candidate["seconds"],
        current_candidate_source_distinct_from_original_control=True,
        no_validation_or500_test=True)
    R.C.atomic_json(REPORT/"result.json",summary)
    print(json.dumps(dict(status=summary["status"],matched_BS=match,
        optimizer_initial=CANONICAL_FINGERPRINT["bias_times"],optimizer_final=sorted(set(end_times)),
        candidate_calls=candidate["additional_updates"],fit_passed=candidate["fit_passed"])))
if __name__=="__main__":
    try:main()
    except BaseException:
        REPORT.mkdir(parents=True,exist_ok=True)
        with (REPORT/"failure.log").open("a")as f:f.write(__import__("traceback").format_exc()+"\n")
        raise
