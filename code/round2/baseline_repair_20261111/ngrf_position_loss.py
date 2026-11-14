"""Matched global versus fixed per-position loss; same p0 model/Adam/RNG."""
from __future__ import annotations
import argparse,json,math,os,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
from project_paths import PATHS
from round2.paper_baseline_training_v1 import common as C
from round2.paper_baseline_training_v1 import native_ngrf as N
from round2.baseline_repair_20261111 import ngrf_repair as R
from round2.baseline_repair_20261111 import ngrf_continue_dropout as Q

OUT=R.REPORT/"position_loss"
ART=R.ART/"position_loss"
CONFIG=OUT/"config.json"
def identity():
    return {**Q.identity(),PATHS.relative(Path(__file__)):C.sha256(Path(__file__)),PATHS.relative(CONFIG):C.sha256(CONFIG)}
def actual_dropout(model):
    return {n:float(x.p) for n,x in model.named_modules() if isinstance(x,torch.nn.modules.dropout._DropoutNd)}
def set_dropout(model,off):
    for x in model.modules():
        if isinstance(x,torch.nn.modules.dropout._DropoutNd) and off:x.p=0.
def health(summary,parent_zero):
    full=summary["pooled"];bad=[];zero=[]
    for row in summary["per_point"]:
        if row["zero_truth"]:
            zero.append(dict(row_id=row["row_id"],power=row["prediction_mean_power_FIT_RMS_squared"],
                maximum=parent_zero[row["row_id"]],
                passed=row["prediction_mean_power_FIT_RMS_squared"]<=parent_zero[row["row_id"]]*(1+1e-8)+1e-12))
        elif row["complex_nmse"]>.5 or not .5<=row["prediction_to_truth_energy"]<=1.5:
            bad.append(dict(row_id=row["row_id"],NMSE=row["complex_nmse"],energy_ratio=row["prediction_to_truth_energy"]))
    pooled=full["complex_nmse"]<=.5 and full["magnitude_nmse"]<=.05
    return dict(pooled_gate=bool(pooled),all_nonzero_points_pass=len(bad)==0,failed_nonzero_points=bad,
        zero_leakage_checks=zero,all_zero_leakage_nonincreasing=all(x["passed"] for x in zero),
        all_point_health=bool(pooled and not bad and all(x["passed"] for x in zero)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--gpu-uuid",required=True);args=ap.parse_args()
    cfg=json.loads(CONFIG.read_text());source=identity()
    OUT.mkdir(parents=True,exist_ok=True);ART.mkdir(parents=True,exist_ok=True)
    if (OUT/"results.json").exists():
        previous=json.loads((OUT/"results.json").read_text())
        if previous["status"]=="complete":
            assert previous["source_identity"]==source;print("Completed continuation reused");return
    free=int(subprocess.check_output(["nvidia-smi","-i",args.gpu_uuid,"--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
    if free<24*1024:raise RuntimeError("Requires24GiB free on allowed assigned UUID")
    gpu=C.device_guard(args.gpu_uuid);device=torch.device("cuda:0")
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    parent=json.loads((Q.OUT/"results.json").read_text())
    endpoint=next(x for x in parent["variants"] if x["branch"]=="dropout_off")
    parent_file=Q.ART/"dropout_off/latest.pt";parent_sha=C.sha256(parent_file)
    initial=torch.load(parent_file,map_location="cpu",weights_only=True)
    assert initial["global_step"]==cfg["parent_global_step"]==6000 and initial["branch"]=="dropout_off"
    assert initial["source_identity"]==Q.identity()
    parent_final=torch.load(PATHS.root/endpoint["checkpoint"],map_location="cpu",weights_only=True)
    assert C.sha256(PATHS.root/endpoint["checkpoint"])==endpoint["checkpoint_sha256"]
    assert set(initial["model"])==set(parent_final["model"])
    assert all(torch.equal(initial["model"][k],parent_final["model"][k]) for k in initial["model"])
    for rel,h in endpoint["source_identity"].items():assert C.sha256(PATHS.root/rel)==h,rel
    rows=R.PANEL;data=C.FoldData(0,"screen","ngrf")
    assert not set(rows).intersection(data.outer_ids)
    target_np=data.read_fit(rows);target=torch.tensor(target_np,device=device)
    sites=data.sites(rows);p=torch.tensor(data.positions(rows),dtype=torch.float64,device=device)
    site=torch.tensor(sites,dtype=torch.long,device=device)
    power_np=np.mean(np.abs(target_np.astype(np.complex128))**2,axis=(1,2,3))
    fixed_power=torch.tensor(np.maximum(power_np,cfg["training_power_floor"]),device=device,dtype=torch.float64)
    full_ids,full_ff=R.schedule(data,8000)
    with np.load(Q.ART/"schedule.npz",allow_pickle=False) as z:
        assert np.array_equal(z["point_indices"],full_ids[3000:6000]) and np.array_equal(z["frequency_ids"],full_ff[3000:6000])
    ids=full_ids[6000:];ff=full_ff[6000:];schedule=R.schedule_id(ids,ff)
    if not (ART/"schedule.npz").exists():
        with (ART/"schedule.npz").open("xb") as f:np.savez(f,point_indices=ids,frequency_ids=ff,panel=rows)
    original=json.loads((R.REPORT/"paired/results.json").read_text())
    original_endpoint=next(x for x in original["variants"] if x["weight"]==1.)
    parent_zero={x["row_id"]:x["prediction_mean_power_FIT_RMS_squared"] for x in original_endpoint["final"]["per_point"] if x["zero_truth"]}
    initial_rng=dict(cpu=R.tensor_hash(initial["torch_rng"]),cuda=R.tensor_hash(initial["cuda_rng"]))
    carried=0.
    for file in ART.glob("loss_*/latest.pt"):
        old=torch.load(file,map_location="cpu",weights_only=True);carried=max(carried,float(old["wall_seconds"]));del old
    started=time.monotonic();variants=[];cap=cfg["max_total_seconds"]
    for branch in ["loss_global","loss_position_relative"]:
        off=True;report=OUT/branch;store=ART/branch
        report.mkdir(parents=True,exist_ok=True);store.mkdir(parents=True,exist_ok=True)
        if (report/"result.json").exists():
            done=json.loads((report/"result.json").read_text());assert done["source_identity"]==source
            variants.append(done);continue
        model=R.model_build(data,device);model.load_state_dict(initial["model"])
        original_dropout=actual_dropout(model);set_dropout(model,off)
        actual=actual_dropout(model);assert len(actual)==18 and (not off or all(v==0 for v in actual.values()))
        opt=N.native_optimizer(model);opt.load_state_dict(initial["optimizer"])
        original_hash=R.model_hash(model)
        assert all(torch.equal(initial["model"][k],parent_final["model"][k]) for k in initial["model"])
        torch.set_rng_state(initial["torch_rng"]);torch.cuda.set_rng_state(initial["cuda_rng"])
        assert dict(cpu=R.tensor_hash(torch.get_rng_state()),cuda=R.tensor_hash(torch.cuda.get_rng_state()))==initial_rng
        current=0;history=[];latest=store/"latest.pt"
        if latest.exists():
            ck=torch.load(latest,map_location="cpu",weights_only=True)
            assert ck["source_identity"]==source and ck["branch"]==branch and ck["schedule"]==schedule
            model.load_state_dict(ck["model"]);opt.load_state_dict(ck["optimizer"])
            current=ck["local_step"];history=ck["history"]
            torch.set_rng_state(ck["torch_rng"]);torch.cuda.set_rng_state(ck["cuda_rng"])
            del ck
        def checkpoint():
            C.atomic_torch_save(latest,dict(model=model.state_dict(),optimizer=opt.state_dict(),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
                local_step=current,global_step=6000+current,branch=branch,history=history,
                source_identity=source,schedule=schedule,wall_seconds=carried+time.monotonic()-started))
        if current==0:
            before=R.predict(model,data,rows,device);summary=R.stats(before,target_np,sites,data.scale)
            old=np.load(PATHS.root/endpoint["raw_prediction"],allow_pickle=False)/data.scale
            diff=float(np.linalg.norm((before-old).reshape(-1))/max(np.linalg.norm(old.reshape(-1)),1e-30))
            assert diff<=1e-5
            history.append(dict(local_step=0,global_step=6000,FIT=summary,parent_prediction_relative_L2=diff))
        C.atomic_json(report/"run_manifest.json",dict(source_identity=source,branch=branch,config=cfg,
            parent_checkpoint=PATHS.relative(parent_file),parent_sha256=parent_sha,
            initial_model_hash=original_hash,initial_rng=initial_rng,schedule=schedule,
            fixed_full192_position_power=power_np.tolist(),training_power_floor=cfg["training_power_floor"],floor_is_physical_noise_estimate=False,
            original_native_metadata=endpoint["original_native_metadata"],original_dropout=original_dropout,actual_dropout=actual,parameter_count=sum(x.numel() for x in model.parameters()),data=data.info(),
            device=gpu,validation_labels_read=False,formal_test=False))
        torch.cuda.synchronize();profile_start=time.monotonic();last_print=profile_start
        for step in range(current+1,cfg["additional_updates"]+1):
            model.train();opt.zero_grad(set_to_none=True);R.learning_rate(opt,6000+step)
            ri=torch.tensor(ids[step-1],device=device);fi=torch.tensor(ff[step-1],device=device)
            frequency=torch.tensor(data.frequencies(ff[step-1]),device=device,dtype=torch.float64)
            value=model(p[ri],site[ri],frequency);label=target[ri][:,:,:,fi]
            reg=torch.stack([field.activation_penalty() for field in model.fields.values()]).mean()
            native_loss,detail=R.loss_components(value,label,reg,1.)
            if branch=="loss_global":loss=native_loss
            else:
                mag=(value.abs()-label.abs()).square().mean((1,2,3))
                comp=(value-label).abs().square().mean((1,2,3))
                loss=((mag+comp)/fixed_power[ri]).mean()+R.COEF*reg
            detail={**detail,"actual_objective":float(loss.detach()),"objective_kind":branch}
            if not bool(torch.isfinite(loss)):raise FloatingPointError("Nonfinite continuation loss")
            loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True))
            opt.step();current=step;elapsed=carried+time.monotonic()-started
            if step==20:
                torch.cuda.synchronize();profile_update_seconds=time.monotonic()-profile_start
            full=step%cfg["eval_every"]==0 or step in [20,cfg["additional_updates"]]
            if full:
                value_np=R.predict(model,data,rows,device)
                event=dict(local_step=step,global_step=6000+step,FIT=R.stats(value_np,target_np,sites,data.scale),
                    training_batch=detail,gradient_norm=grad,wall_seconds=elapsed)
                history.append(event);C.atomic_json(report/"progress.json",event);checkpoint()
                print(json.dumps(dict(branch=branch,step=step,total=cfg["additional_updates"],FIT=event["FIT"]["pooled"],wall_seconds=elapsed)),flush=True)
            if step==20:
                torch.cuda.synchronize()
                profile=dict(branch=branch,actual_updates_counted=20,seconds=profile_update_seconds,
                    projected_seconds_for_both_updates=profile_update_seconds/20*(2*cfg["additional_updates"]))
                C.atomic_json(report/"profile.json",profile);print(json.dumps(profile),flush=True)
            if time.monotonic()-last_print>=30:
                checkpoint();print(json.dumps(dict(branch=branch,step=step,total=cfg["additional_updates"],
                    loss=float(loss),gradient_norm=grad,wall_seconds=elapsed)),flush=True);last_print=time.monotonic()
            if elapsed>cap and step<cfg["additional_updates"]:
                checkpoint();C.atomic_json(OUT/"budget_cap.json",dict(status="budget_cap",branch=branch,step=step,wall_seconds=elapsed))
                print(json.dumps(dict(status="budget_cap",branch=branch,step=step)),flush=True);return
        prediction=R.predict(model,data,rows,device);summary=R.stats(prediction,target_np,sites,data.scale)
        checkpoint()
        C.atomic_torch_save(store/"final.pt",dict(model=model.state_dict(),local_step=current,global_step=6000+current,
            original_native_metadata=endpoint["original_native_metadata"],original_dropout=original_dropout,actual_dropout=actual,source_identity=source,schedule=schedule,parent_checkpoint_sha256=parent_sha))
        clone=R.model_build(data,device);set_dropout(clone,off)
        clone.load_state_dict(torch.load(store/"final.pt",map_location=device,weights_only=True)["model"])
        replay=R.predict(clone,data,rows,device)
        difference=float(np.linalg.norm((replay-prediction).reshape(-1))/max(np.linalg.norm(prediction.reshape(-1)),1e-30))
        assert difference<=1e-6
        raw=(prediction*data.scale).astype(np.complex64);C.atomic_npy(store/"FIT_raw_H.npy",raw)
        done=dict(status="complete",branch=branch,local_updates=current,global_updates=6000+current,
            source_identity=source,config=cfg,FIT=summary,health=health(summary,parent_zero),history=history,
            checkpoint=PATHS.relative(store/"final.pt"),checkpoint_sha256=C.sha256(store/"final.pt"),
            raw_prediction=PATHS.relative(store/"FIT_raw_H.npy"),raw_prediction_sha256=C.sha256(store/"FIT_raw_H.npy"),
            clean_reload_relative_L2=difference,initial_model_hash=original_hash,initial_rng=initial_rng,
            parent_checkpoint_sha256=parent_sha,schedule=schedule,original_native_metadata=endpoint["original_native_metadata"],original_dropout=original_dropout,actual_dropout=actual,
            parameter_count=sum(x.numel() for x in model.parameters()),device=gpu,
            peak_cuda_bytes=torch.cuda.max_memory_allocated(),wall_seconds=carried+time.monotonic()-started,
            validation_labels_read=0,outer_truth_read=False,formal_test=False)
        C.atomic_json(report/"result.json",done);variants.append(done)
        del model,clone,opt;torch.cuda.empty_cache()
    a,b=variants
    assert a["initial_model_hash"]==b["initial_model_hash"] and a["initial_rng"]==b["initial_rng"]
    assert a["schedule"]==b["schedule"] and a["parent_checkpoint_sha256"]==b["parent_checkpoint_sha256"]
    assert a["parameter_count"]==b["parameter_count"]
    if C.sha256(parent_file)!=parent_sha:raise RuntimeError("Parent checkpoint changed")
    result=dict(status="complete",source_identity=source,config=cfg,variants=variants,
        POSITION_vs_GLOBAL_complex_relative_improvement=1-b["FIT"]["pooled"]["complex_nmse"]/a["FIT"]["pooled"]["complex_nmse"],
        GLOBAL_vs_parent_complex_relative_improvement=1-a["FIT"]["pooled"]["complex_nmse"]/endpoint["FIT"]["pooled"]["complex_nmse"],
        POSITION_vs_parent_complex_relative_improvement=1-b["FIT"]["pooled"]["complex_nmse"]/endpoint["FIT"]["pooled"]["complex_nmse"],
        parent_checkpoint_unchanged=True,wall_seconds=carried+time.monotonic()-started,
        validation_labels_read=0,outer_truth_read=False,formal_test=False)
    C.atomic_json(OUT/"results.json",result)
    print(json.dumps(dict(status="complete",wall_seconds=result["wall_seconds"],POSITION_vs_GLOBAL=result["POSITION_vs_GLOBAL_complex_relative_improvement"],
        branch_health={x["branch"]:x["health"] for x in variants})),flush=True)
if __name__=="__main__":
    try:main()
    except BaseException:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/"failure.log").open("a") as f:f.write(traceback.format_exc()+"\n")
        raise
