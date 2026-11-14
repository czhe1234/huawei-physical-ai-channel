"""Single low-LR recovery from the parent warm endpoint, preserving Adam/RNG."""
from __future__ import annotations
import argparse,copy,json,time,traceback,signal
from pathlib import Path
import numpy as np
import torch
from . import newrf_repair as N
from round2.paper_baseline_training_v1.common import atomic_json,atomic_npy,atomic_torch_save,sha256,seed_all
from project_paths import PATHS
R=N.R/"low_lr";A=N.A/"low_lr";PARENT=N.A/"fit_query/final.pt"
CONFIG=N.R/"low_lr_config.json"
def main(args):
    cfg=json.loads(CONFIG.read_text());basecfg,data,rows,raw,truth=N.inputs()
    parent=torch.load(PARENT,map_location="cpu",weights_only=False)
    N.check_identity(parent["source_identity"])
    if parent["stages"].get("warm",{}).get("passed"):raise RuntimeError("Parent passed; low-LR recovery unnecessary")
    endpoint=parent["stages"]["warm"]["fine"]
    if not (endpoint["nmse"]<=.1 and all(x["nmse"]<=.3 and .6<=x["energy_ratio"]<=1.4 for x in endpoint["per_position"])):
        raise RuntimeError("Parent endpoint is not a near-threshold oscillation; do not run hypothesis blindly")
    if parent["total_updates"]+cfg["max_new_updates"]>basecfg["newrf"]["max_updates"]:
        raise RuntimeError("Combined update budget exceeds registered12000")
    source=N.identity()|{PATHS.relative(Path(__file__)):sha256(Path(__file__)),
        PATHS.relative(CONFIG):sha256(CONFIG),PATHS.relative(PARENT):sha256(PARENT)}
    if (R/"result.json").exists():
        d=json.loads((R/"result.json").read_text());N.check_identity(d["source_identity"]);print(d["status"]);return
    device=N.gpu_setup(args.gpu_uuid);seed_all(42);started=time.monotonic()
    bank,fitcache,model=N.build(data);qcache=N.QueryDirectionsCache(model,fitcache,data,rows)
    model.load_state_dict(parent["model"])
    slots=np.full(4000,-1,np.int64);slots[rows]=np.arange(2)
    data._native_raw_target_cache=(slots,torch.from_numpy(raw).to("cuda:0"))
    named=dict(model.fine[0].named_parameters())
    alpha=[p for n,p in named.items() if n.startswith("network.alpha_out.")]
    active=[p for n,p in named.items() if not n.startswith("network.alpha_out.")]
    opt=torch.optim.Adam([dict(params=active),dict(params=alpha)],lr=cfg["learning_rate"])
    opt.load_state_dict(parent["fine_optimizer"])
    for st in opt.state.values():
        for k,v in st.items():
            if isinstance(v,torch.Tensor):st[k]=v.to("cuda:0")
    sampler=N.Sampler(rows,42);sampler.restore(parent["sampler"])
    torch.set_rng_state(parent["torch_rng"]);torch.cuda.set_rng_state_all(parent["cuda_rng"])
    stage="warm";step=0;stable=0;total=0;history=[];elapsed_before=0.;warm_pass=False;release_done=False
    draws=np.zeros((cfg["max_new_updates"],32,3),np.int64)
    latest=A/"latest.pt"
    if latest.exists():
        ck=torch.load(latest,map_location="cpu",weights_only=False)
        if ck["source_identity"]!=source:raise RuntimeError("Child resume source changed")
        model.load_state_dict(ck["model"]);opt.load_state_dict(ck["optimizer"])
        for st in opt.state.values():
            for k,v in st.items():
                if isinstance(v,torch.Tensor):st[k]=v.to("cuda:0")
        sampler.restore(ck["sampler"]);torch.set_rng_state(ck["torch_rng"]);torch.cuda.set_rng_state_all(ck["cuda_rng"])
        stage=ck["stage"];step=ck["stage_updates"];stable=ck["stable"];total=ck["new_updates"]
        history=ck["history"];elapsed_before=ck["elapsed_seconds"];warm_pass=ck["warm_pass"]
        release_done=ck["release_done"];draws[:total]=ck["draws"]
    for g in opt.param_groups:g["lr"]=cfg["learning_rate"]
    frozen_coarse=N.state(model.coarse[0])
    def elapsed():return elapsed_before+time.monotonic()-started
    def deadline():return time.monotonic()+max(1,cfg["max_wall_seconds"]-elapsed()-10)
    def full():
        model.eval()
        return N.fixed_field(model,qcache,data,rows,0,N.BS,N.FREQ,deadline())
    def configure(which):
        for p in model.parameters():p.requires_grad_(False)
        if "physical_density" in model.fine[0].__dict__:delattr(model.fine[0],"physical_density")
        for p in active:p.requires_grad_(True)
        if which=="warm":
            original=model.fine[0].physical_density
            model.fine[0].physical_density=lambda a:original(a).detach()
        else:
            for p in alpha:p.requires_grad_(True)
    def checkpoint():
        return dict(model=N.state(model),optimizer=copy.deepcopy(opt.state_dict()),sampler=sampler.snapshot(),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),stage=stage,stage_updates=step,
            stable=stable,new_updates=total,parent_total_updates=parent["total_updates"],history=history,
            elapsed_seconds=elapsed(),warm_pass=warm_pass,release_done=release_done,draws=draws[:total].copy(),
            source_identity=source,config=cfg)
    protocol=dict(config=cfg,parent_endpoint_metrics=endpoint,parent_checkpoint_sha256=sha256(PARENT),
        initialization="Parent warm endpoint, never selected best",Adam_moments_preserved=True,
        sampler_and_RNG_preserved=True,phase_stability_counter_reset=0,
        only_optimization_change="warm learning rate5e-4->5e-5; all data/model/native loss/physics unchanged",
        final_release="remove backward-only density detach; all fine parameters, same5e-5lr, fixed1000updates",
        coarse_frozen=True,warm_density_values_not_frozen=True,
        axes=dict(row_ids=rows.tolist(),field=0,BS=list(range(128)),UE_port=0,frequency_count=192),
        formal_test=False,validation_reads=0,outer_truth_reads=0)
    atomic_json(R/"run_manifest.json",dict(source_identity=source,protocol=protocol,device=device,data=data.info()))
    if not latest.exists():
        configure("warm");pred,cpred=full()
        eq=N.compare(np.load(N.A/"fit_query/training_provider_prediction.npy"),pred)
        if eq["relative_L2"]>1e-5:raise RuntimeError("Child endpoint replay mismatch")
        atomic_json(R/"parent_endpoint_replay.json",eq)
        atomic_npy(A/"initial_prediction.npy",pred)
    timed_out=False
    while True:
        configure(stage)
        cap=cfg["warm_updates"] if stage=="warm" else cfg["native_updates"]
        last_print=time.monotonic()
        while step<cap:
            if elapsed()>cfg["max_wall_seconds"]-45:timed_out=True;break
            rr,bb,ff=sampler.draw()
            draws[total,:,0]=rr;draws[total,:,1]=bb;draws[total,:,2]=ff
            model.train();opt.zero_grad(set_to_none=True)
            y=N.station_target_cached(data,rr,0,bb,ff,"cuda:0")
            h,c=N.station_forward(model,qcache,data,rr,0,bb,ff)
            loss,lf,lc=N.native_weighted_loss(h,c,y)
            if not torch.isfinite(loss):raise FloatingPointError("Child native loss")
            loss.backward()
            gn=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.,error_if_nonfinite=True)
            opt.step();step+=1;total+=1
            if time.monotonic()-last_print>30:
                print(json.dumps(dict(stage=stage,step=step,new_updates=total,loss=float(loss),elapsed_seconds=elapsed())),flush=True);last_print=time.monotonic()
            if step%250==0:atomic_torch_save(latest,checkpoint())
            if step%cfg["eval_every"] and step!=cap:continue
            pred,cpred=full();fm=N.metrics(pred,truth);cm=N.metrics(cpred,truth)
            now=N.fits(fm);stable=stable+1 if now else 0
            event=dict(stage=stage,step=step,new_updates=total,fine=fm,coarse=cm,gate_now=now,
                stable_full_checks=stable,elapsed_seconds=elapsed())
            history.append(event);atomic_json(R/"progress.json",event);atomic_torch_save(latest,checkpoint())
            print(json.dumps(dict(stage=stage,step=step,pooled_nmse=fm["nmse"],
                per_position_nmse=[x["nmse"] for x in fm["per_position"]],stable=stable,elapsed_seconds=elapsed())),flush=True)
            if stage=="warm" and stable>=cfg["stable_checks"]:warm_pass=True;break
        if timed_out:break
        if stage=="warm" and warm_pass:
            before,beforec=full();configure("native");after,afterc=full()
            nojump=bool(np.array_equal(before,after) and np.array_equal(beforec,afterc))
            atomic_json(R/"release_check.json",dict(forward_bitwise_no_jump=nojump,
                warm_override_removed="physical_density" not in model.fine[0].__dict__,
                active_Adam_moments_preserved=sum(p in opt.state for p in active),
                alpha_Adam_states_before_release=sum(p in opt.state for p in alpha)))
            if not nojump:raise RuntimeError("Native release changed forward values")
            stage="native";step=0;stable=0;release_done=True;atomic_torch_save(latest,checkpoint());continue
        break
    if "physical_density" in model.fine[0].__dict__:delattr(model.fine[0],"physical_density")
    model.eval();pred,cpred=full();q=N.full_query(model,data,rows,deadline())
    eq=N.compare(pred,q)
    if eq["relative_L2"]>1e-5:raise RuntimeError("Child train/query provider mismatch")
    if not all(torch.equal(v,frozen_coarse[k]) for k,v in N.state(model.coarse[0]).items()):raise RuntimeError("Frozen coarse changed")
    atomic_torch_save(A/"final.pt",checkpoint());atomic_npy(A/"query_prediction.npy",q)
    atomic_npy(A/"training_provider_prediction.npy",pred);atomic_npy(A/"coarse_prediction.npy",cpred)
    atomic_npy(A/"new_supervised_row_BS_frequency_ids.npy",draws[:total])
    clean=N.build_native_batch_model(data,fitcache,128,128,frequency_chunk=8).to("cuda:0")
    clean.load_state_dict(torch.load(A/"final.pt",map_location="cpu",weights_only=False)["model"]);clean.eval()
    replay=N.full_query(clean,data,rows,deadline());eqclean=N.compare(q,replay)
    if eqclean["relative_L2"]>1e-5:raise RuntimeError("Child original-builder clean query mismatch")
    N.check_identity(source);qm=N.metrics(q,truth)
    passed=bool(warm_pass and release_done and stage=="native" and step==cfg["native_updates"] and N.fits(qm))
    result=dict(status="two_position_actual_query_fit_pass" if passed else "budget_exhausted" if timed_out else "fit_not_stably_restored",
        source_identity=source,protocol=protocol,device=device,parent_endpoint_metrics=endpoint,
        actual_query_metrics=qm,training_vs_actual_query=eq,clean_reload=eqclean,
        clean_builder_no_warm_patch=True,coarse_unchanged=True,history=history,
        new_optimizer_updates=total,parent_optimizer_updates=parent["total_updates"],
        combined_optimizer_updates=parent["total_updates"]+total,elapsed_seconds=elapsed(),
        parent_elapsed_seconds=parent["elapsed_seconds"],peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        final_checkpoint_sha256=sha256(A/"final.pt"),query_prediction_sha256=sha256(A/"query_prediction.npy"),
        data=data.info(),other_fields_untrained=True,validation_truth_reads=0,query_truth_read=False,formal_test=False)
    atomic_json(R/"result.json",result)
    print(json.dumps({k:result[k] for k in ["status","actual_query_metrics","new_optimizer_updates","combined_optimizer_updates","elapsed_seconds"]}),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--gpu-uuid",required=True);args=p.parse_args()
    R.mkdir(parents=True,exist_ok=True);A.mkdir(parents=True,exist_ok=True)
    try:main(args)
    except BaseException:
        atomic_json(R/"failure.json",dict(traceback=traceback.format_exc(),formal_test=False,query_truth_read=False));raise
