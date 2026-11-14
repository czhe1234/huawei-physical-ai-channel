"""Complete only the remaining frozen native updates after a runtime wall cap."""
from __future__ import annotations
import argparse,copy,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from . import newrf_repair as N
from . import newrf_low_lr as L
from project_paths import PATHS
from round2.paper_baseline_training_v1.common import atomic_json,atomic_npy,atomic_torch_save,sha256,seed_all
R=N.R/"native_finish";A=N.A/"native_finish";PARENT=L.A/"final.pt"
def main(args):
    ck=torch.load(PARENT,map_location="cpu",weights_only=False);N.check_identity(ck["source_identity"])
    base_report=json.loads((N.R/"fit_query/result.json").read_text())
    if (L.R/"result.json").exists():
        parent_report=json.loads((L.R/"result.json").read_text())
        if parent_report["status"]!="budget_exhausted":raise RuntimeError("Runtime recovery unnecessary or invalid")
        child_wall_debit=parent_report["elapsed_seconds"]
    else:
        failure=json.loads((L.R/"failure.json").read_text())
        if "TimeoutError: Query deadline" not in failure["traceback"] or "replay=N.full_query(clean" not in failure["traceback"]:
            raise RuntimeError("Only verified final-clean timeout can use checkpoint without completed result")
        if not (L.A/"query_prediction.npy").is_file():raise RuntimeError("Timeout has no saved actual-query endpoint")
        parent_report=dict(status="final_clean_runtime_timeout",
            combined_optimizer_updates=ck["parent_total_updates"]+ck["new_updates"],elapsed_seconds=None)
        child_wall_debit=600.
    cfg=json.loads(L.CONFIG.read_text())
    if ck["stage"]!="native" or not ck["warm_pass"] or not ck["release_done"]:
        raise RuntimeError("Only already-released native phase may be resumed")
    done=int(ck["stage_updates"]);remaining=int(cfg["native_updates"])-done
    if remaining<0:raise RuntimeError("Native target already exceeded")
    active_before=child_wall_debit+base_report["elapsed_seconds"]
    cap=min(300.,2400.-active_before)
    if cap<60:raise RuntimeError("Insufficient registered total active wall reserve")
    source=N.identity()|{PATHS.relative(Path(__file__)):sha256(Path(__file__)),
        PATHS.relative(L.CONFIG):sha256(L.CONFIG),PATHS.relative(PARENT):sha256(PARENT),
        PATHS.relative(Path(L.__file__)):sha256(Path(L.__file__))}
    if (R/"result.json").exists():
        result=json.loads((R/"result.json").read_text());N.check_identity(result["source_identity"]);print(result["status"]);return
    started=time.monotonic();device=N.gpu_setup(args.gpu_uuid);seed_all(42)
    _,data,rows,raw,truth=N.inputs();bank,fitcache,model=N.build(data)
    qcache=N.QueryDirectionsCache(model,fitcache,data,rows);model.load_state_dict(ck["model"])
    slots=np.full(4000,-1,np.int64);slots[rows]=np.arange(2)
    data._native_raw_target_cache=(slots,torch.from_numpy(raw).to("cuda:0"))
    named=dict(model.fine[0].named_parameters())
    active=[p for n,p in named.items() if not n.startswith("network.alpha_out.")]
    alpha=[p for n,p in named.items() if n.startswith("network.alpha_out.")]
    opt=torch.optim.Adam([dict(params=active),dict(params=alpha)],lr=cfg["learning_rate"])
    opt.load_state_dict(ck["optimizer"])
    for st in opt.state.values():
        for k,v in st.items():
            if isinstance(v,torch.Tensor):st[k]=v.to("cuda:0")
    if any(g["lr"]!=cfg["learning_rate"] for g in opt.param_groups):raise RuntimeError("Frozen learning rate changed")
    sampler=N.Sampler(rows,42);sampler.restore(ck["sampler"])
    torch.set_rng_state(ck["torch_rng"]);torch.cuda.set_rng_state_all(ck["cuda_rng"])
    for p in model.parameters():p.requires_grad_(False)
    for p in active+alpha:p.requires_grad_(True)
    if "physical_density" in model.fine[0].__dict__:raise RuntimeError("Warm override unexpectedly present")
    frozen_coarse=N.state(model.coarse[0]);step=done;extra=0;history=[]
    draws=np.zeros((remaining,32,3),np.int64)
    elapsed_before=0.
    if (A/"latest.pt").exists():
        resume=torch.load(A/"latest.pt",map_location="cpu",weights_only=False)
        if resume["source_identity"]!=source:raise RuntimeError("Runtime resume source changed")
        model.load_state_dict(resume["model"]);opt.load_state_dict(resume["optimizer"])
        for st in opt.state.values():
            for k,v in st.items():
                if isinstance(v,torch.Tensor):st[k]=v.to("cuda:0")
        sampler.restore(resume["sampler"]);torch.set_rng_state(resume["torch_rng"]);torch.cuda.set_rng_state_all(resume["cuda_rng"])
        step=resume["native_updates"];extra=resume["extra_updates"];history=resume["history"]
        draws[:extra]=resume["draws"];elapsed_before=resume["elapsed_seconds"]
    protocol=dict(kind="runtime-only native completion",parent_checkpoint_sha256=sha256(PARENT),
        native_updates_before=done,frozen_native_target=cfg["native_updates"],remaining_native_updates=remaining,
        maximum_extra_wall_seconds=cap,parent_and_child_wall_budget_debit_seconds=active_before,
        child_status=parent_report["status"],child_measured_wall_seconds=parent_report["elapsed_seconds"],
        child_checkpoint_elapsed_seconds=ck["elapsed_seconds"],child_wall_budget_debit_seconds=child_wall_debit,
        combined_wall_cap=2400,learning_rate=cfg["learning_rate"],Adam_and_RNG_and_sampler_preserved=True,
        no_warm_retraining=True,no_extra_optimization_updates=True,formal_test=False,outer_truth_reads=0,validation_truth_reads=0)
    atomic_json(R/"run_manifest.json",dict(source_identity=source,protocol=protocol,device=device))
    def elapsed():return elapsed_before+time.monotonic()-started
    def deadline():return time.monotonic()+max(1,cap-elapsed()-10)
    def full():
        model.eval();return N.fixed_field(model,qcache,data,rows,0,N.BS,N.FREQ,deadline())
    def snapshot():
        return dict(model=N.state(model),optimizer=copy.deepcopy(opt.state_dict()),sampler=sampler.snapshot(),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),native_updates=step,
            extra_updates=extra,draws=draws[:extra].copy(),history=history,elapsed_seconds=elapsed(),
            source_identity=source,protocol=protocol)
    partial_replay=N.full_query(model,data,rows,deadline())
    partial_eq=N.compare(np.load(L.A/"query_prediction.npy"),partial_replay)
    atomic_json(R/"parent_endpoint_replay.json",partial_eq)
    if partial_eq["relative_L2"]>1e-5:raise RuntimeError("Saved native endpoint cannot be replayed by original builder")
    last_print=time.monotonic()
    while step<cfg["native_updates"]:
        if elapsed()>cap-45:break
        rr,bb,ff=sampler.draw();draws[extra,:,0]=rr;draws[extra,:,1]=bb;draws[extra,:,2]=ff
        model.train();opt.zero_grad(set_to_none=True)
        y=N.station_target_cached(data,rr,0,bb,ff,"cuda:0")
        h,c=N.station_forward(model,qcache,data,rr,0,bb,ff)
        loss,_,_=N.native_weighted_loss(h,c,y)
        if not torch.isfinite(loss):raise FloatingPointError("Runtime continuation loss")
        loss.backward();gn=torch.nn.utils.clip_grad_norm_(active+alpha,5.,error_if_nonfinite=True);opt.step()
        step+=1;extra+=1
        if time.monotonic()-last_print>30:
            print(json.dumps(dict(stage="native",native_step=step,extra_updates=extra,loss=float(loss),elapsed_seconds=elapsed())),flush=True);last_print=time.monotonic()
        if extra%100==0:atomic_torch_save(A/"latest.pt",snapshot())
        if step%500==0 or step==cfg["native_updates"]:
            pred,_=full();m=N.metrics(pred,truth)
            e=dict(native_updates=step,extra_updates=extra,metrics=m,elapsed_seconds=elapsed())
            history.append(e);atomic_json(R/"progress.json",e);atomic_torch_save(A/"latest.pt",snapshot())
    model.eval();pred,cpred=full();query=N.full_query(model,data,rows,deadline())
    eq=N.compare(pred,query)
    if eq["relative_L2"]>1e-5:raise RuntimeError("Runtime training/query mismatch")
    if not all(torch.equal(v,frozen_coarse[k]) for k,v in N.state(model.coarse[0]).items()):raise RuntimeError("Coarse weights changed")
    atomic_npy(A/"query_prediction.npy",query);atomic_npy(A/"training_provider_prediction.npy",pred)
    atomic_npy(A/"coarse_prediction.npy",cpred);atomic_npy(A/"new_supervised_row_BS_frequency_ids.npy",draws[:extra])
    atomic_torch_save(A/"final.pt",snapshot())
    clean=N.build_native_batch_model(data,fitcache,128,128,frequency_chunk=8).to("cuda:0")
    clean.load_state_dict(torch.load(A/"final.pt",map_location="cpu",weights_only=False)["model"]);clean.eval()
    replay=N.full_query(clean,data,rows,deadline());eqclean=N.compare(query,replay)
    if eqclean["relative_L2"]>1e-5:raise RuntimeError("Runtime clean native builder mismatch")
    N.check_identity(source);m=N.metrics(query,truth)
    combined_updates=parent_report["combined_optimizer_updates"]+extra
    if combined_updates>12000:raise RuntimeError("Registered total updates exceeded")
    passed=bool(step==cfg["native_updates"] and N.fits(m) and active_before+elapsed()<=2400)
    result=dict(status="two_position_actual_query_fit_pass" if passed else "runtime_recovery_incomplete",
        source_identity=source,protocol=protocol,device=device,actual_query_metrics=m,
        training_vs_actual_query=eq,clean_reload=eqclean,clean_builder_no_warm_patch=True,coarse_unchanged=True,
        completed_native_updates=step,extra_optimizer_updates=extra,combined_optimizer_updates=combined_updates,
        elapsed_seconds=elapsed(),
        combined_active_wall_seconds=(None if parent_report["elapsed_seconds"] is None else active_before+elapsed()),
        combined_active_wall_budget_accounted_seconds=active_before+elapsed(),
        combined_active_wall_known_lower_bound_seconds=base_report["elapsed_seconds"]+ck["elapsed_seconds"]+elapsed(),
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),final_checkpoint_sha256=sha256(A/"final.pt"),
        query_prediction_sha256=sha256(A/"query_prediction.npy"),data=data.info(),
        other_fields_untrained=True,formal_test=False,query_truth_read=False,validation_truth_reads=0)
    atomic_json(R/"result.json",result)
    print(json.dumps({k:result[k] for k in ["status","actual_query_metrics","completed_native_updates","extra_optimizer_updates","combined_active_wall_budget_accounted_seconds"]}),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--gpu-uuid",required=True);args=p.parse_args()
    R.mkdir(parents=True,exist_ok=True);A.mkdir(parents=True,exist_ok=True)
    try:main(args)
    except BaseException:
        atomic_json(R/"failure.json",dict(traceback=traceback.format_exc(),formal_test=False));raise
