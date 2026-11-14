"""Independent CPU raw-array/state audit for nGRF Adam-copy correction v2."""
from __future__ import annotations
import ast,csv,importlib.util,json,math,os,time
from pathlib import Path
import numpy as np
import torch
from project_paths import PATHS
from common.official_metrics import point_metric_statistics,aggregate_statistics
from round2.paper_baseline_training_v1 import common as C
from round2.baseline_repair_20261111 import ngrf_repair as R
from round2.baseline_repair_20261111 import optimizer_copy_guard as G
OUT=R.REPORT/"adam_copy_fix_v2"
def load(p):return json.loads(Path(p).read_text())
def checkhash(rel,expected):assert C.sha256(PATHS.root/rel)==expected,rel
def independent(pred,target,rows,sites,scale):
    assert pred.shape==target.shape==(16,256,4,192)
    assert pred.dtype==np.complex64 and np.isfinite(pred).all()
    p=pred.astype(np.complex128);t=target.astype(np.complex128)
    e=np.sum(abs(t)**2,(1,2,3));pe=np.sum(abs(p)**2,(1,2,3))
    se=np.sum(abs(p-t)**2,(1,2,3));me=np.sum((abs(p)-abs(t))**2,(1,2,3))
    dot=np.sum((p.conj()*t).real,(1,2,3))
    def pool(ix):
        en=float(e[ix].sum());pn=float(pe[ix].sum())
        return dict(count=int(np.sum(ix)),complex_nmse=float(se[ix].sum()/en) if en else None,
            magnitude_nmse=float(me[ix].sum()/en) if en else None,
            prediction_to_truth_energy=pn/en if en else None,
            signed_complex_correlation_real=float(dot[ix].sum()/math.sqrt(en*pn)) if en*pn else None,
            prediction_mean_power_FIT_RMS_squared=pn/(max(int(np.sum(ix)),1)*256*4*192*scale**2))
    nz=e>0;strength=np.full(16,-np.inf)
    for site in [0,1]:
        ix=(sites==site)&nz
        strength[ix]=10*np.log10(e[ix]/e[ix].max())
    points=[dict(row_id=int(row),site=int(sites[i]),zero_truth=bool(not nz[i]),
        strength_relative_site_max_dB=float(strength[i]) if nz[i] else None,**pool(np.arange(16)==i))
        for i,row in enumerate(rows)]
    return dict(pooled=pool(np.ones(16,bool)),per_point=points,
        per_site={str(s):pool(sites==s) for s in [0,1]},
        energy_groups={name:pool(ix) for name,ix in {
            "strong_ge_minus20dB":nz&(strength>=-20),
            "medium_minus40_to_minus20dB":nz&(strength< -20)&(strength>=-40),
            "weak_lt_minus40dB":nz&(strength< -40),"exact_zero":~nz}.items()})
def compact(s,ceiling):
    nonzero=[x for x in s["per_point"] if not x["zero_truth"]]
    zero=[dict(row_id=x["row_id"],power_FIT_RMS_squared=x["prediction_mean_power_FIT_RMS_squared"],
        ceiling=ceiling[x["row_id"]],passed=x["prediction_mean_power_FIT_RMS_squared"]<=ceiling[x["row_id"]]*(1+1e-8)+1e-12)
        for x in s["per_point"] if x["zero_truth"]]
    failed=[x["row_id"] for x in nonzero if x["complex_nmse"]>.5 or not .5<=x["prediction_to_truth_energy"]<=1.5]
    pooled=s["pooled"]["complex_nmse"]<=.5 and s["pooled"]["magnitude_nmse"]<=.05
    return dict(pooled=s["pooled"],worst_nonzero_nmse=max(x["complex_nmse"] for x in nonzero),
        nonzero_energy_range=[min(x["prediction_to_truth_energy"] for x in nonzero),max(x["prediction_to_truth_energy"] for x in nonzero)],
        zero=zero,failed_nonzero_ids=failed,nonzero_pass_count=len(nonzero)-len(failed),
        strict_all_point_health=bool(pooled and not failed and all(x["passed"] for x in zero)))
def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES")==""
    started=time.monotonic();torch.set_num_threads(4)
    data=C.FoldData(0,"screen","ngrf")
    raw=np.load(C.RAW,mmap_mode="r",allow_pickle=False)
    rows=R.PANEL;assert set(rows).issubset(set(data.fit_ids));assert not set(rows).intersection(data.outer_ids)
    truth=np.asarray(raw[rows]);sites=data.sites(rows);scale=data.scale
    pair=load(R.REPORT/"paired/results.json")
    original=next(x for x in pair["variants"] if x["weight"]==1.)
    ceiling={x["row_id"]:x["prediction_mean_power_FIT_RMS_squared"] for x in original["final"]["per_point"] if x["zero_truth"]}
    records=[];compact_stages={};evidence=[];allpoints=[]
    stages=[("lambda_pair",pair),("dropout_fixed",load(OUT/"dropout/results.json")),("position_fixed",load(OUT/"position_loss/results.json"))]
    for stage,result in stages:
        for rel,h in result["source_identity"].items():checkhash(rel,h)
        outputs=[]
        for item in result["variants"]:
            name=item.get("branch","lambda_"+str(item.get("weight")))
            checkhash(item["raw_prediction"],item["raw_prediction_sha256"])
            checkhash(item["checkpoint"],item["checkpoint_sha256"])
            pred=np.load(PATHS.root/item["raw_prediction"],allow_pickle=False)
            s=independent(pred,truth,rows,sites,scale)
            ref=item.get("FIT",item.get("final"))
            for metric in ["complex_nmse","magnitude_nmse","prediction_to_truth_energy"]:
                assert abs(s["pooled"][metric]-ref["pooled"][metric])<1e-6,(stage,name,metric)
            for x,y in zip(s["per_point"],ref["per_point"]):
                assert x["row_id"]==y["row_id"]
                if not x["zero_truth"]:
                    for metric in ["complex_nmse","prediction_to_truth_energy"]:
                        assert abs(x[metric]-y[metric])<2e-5,(stage,name,x["row_id"],metric)
            latest=(PATHS.root/item["checkpoint"]).with_name("latest.pt")
            state=torch.load(latest,map_location="cpu",weights_only=True)
            expected=3000. if stage=="lambda_pair" else (6000. if stage=="dropout_fixed" else 8000.)
            assert G.steps(state["optimizer"])==[expected]
            moments_finite=all(bool(torch.isfinite(v).all()) for d in state["optimizer"]["state"].values() for v in d.values() if torch.is_tensor(v))
            assert moments_finite
            audit=item.get("optimizer_audit",{})
            if stage!="lambda_pair" and not item.get("reuse_status"):
                parent_latest=R.ART/"paired/weight_1p0/latest.pt" if stage=="dropout_fixed" else R.ART/"adam_copy_fix_v2/dropout/dropout_off/latest.pt"
                parent=torch.load(parent_latest,map_location="cpu",weights_only=True)
                assert audit["parent_optimizer_fingerprint"]==audit["loaded_optimizer_fingerprint"]==G.fingerprint(parent["optimizer"])
                assert audit["CPU_step_and_all_moment_storages_disjoint"] and audit["immutable_initial_preserved"]
                assert audit["end_optimizer_fingerprint"]==G.fingerprint(state["optimizer"])
                assert G.fingerprint(G.immutable_part(parent))==state["immutable_initial_fingerprint"]
                del parent
            checks=dict(stage=stage,branch=name,actual_Adam_steps=G.steps(state["optimizer"]),moments_finite=moments_finite,
                optimizer_fingerprint=G.fingerprint(state["optimizer"]),optimizer_load_audit=audit,
                raw_prediction=item["raw_prediction"],raw_prediction_sha256=item["raw_prediction_sha256"],
                checkpoint=item["checkpoint"],checkpoint_sha256=item["checkpoint_sha256"],
                latest_checkpoint=PATHS.relative(latest),latest_sha256=C.sha256(latest),
                clean_reload_relative_L2=item["clean_reload_relative_L2"])
            c=compact(s,ceiling);outputs.append(dict(branch=name,**c,**{k:checks[k] for k in ["actual_Adam_steps","raw_prediction","raw_prediction_sha256","checkpoint","checkpoint_sha256"]}))
            records.append(dict(**checks,FIT=s,health=c))
            allpoints.extend(dict(stage=stage,branch=name,**x) for x in s["per_point"])
            del state,pred
        compact_stages[stage]=outputs
    vals={}
    for stage in ["dropout","position_loss"]:
        val=load(OUT/stage/"fixed_endpoint_validation/results.json")
        ids=np.array(val["row_ids"],np.int64)
        assert len(ids)==16 and set(ids).issubset(set(data.val_ids)) and not set(ids).intersection(data.outer_ids)
        target=np.asarray(raw[ids])
        variants=[]
        for x in val["variants"]:
            checkhash(x["raw_prediction"],x["raw_prediction_sha256"]);checkhash(x["checkpoint"],x["checkpoint_sha256"])
            p=np.load(PATHS.root/x["raw_prediction"],allow_pickle=False)
            score=aggregate_statistics(point_metric_statistics(p,target))
            difference={k:abs(score[k]-x["metrics"][k]) for k in ["pas","pdp","nmse","score"]}
            assert max(difference.values())<1e-5,(stage,x["branch"],difference)
            variants.append(dict(branch=x["branch"],metrics_cpu=score,GPU_absolute_difference=difference,
                reused=bool(x.get("reuse_status")),raw_prediction=x["raw_prediction"],raw_prediction_sha256=x["raw_prediction_sha256"]))
        vals[stage]=dict(role="fixed original16 development validation, not outer500",variants=variants)
    olddrop=load(R.REPORT/"dropout_continuation/results.json");oldpos=load(R.REPORT/"position_loss/results.json")
    smoke=load(R.REPORT/"smoke/results.json")
    on=next(x for x in olddrop["variants"] if x["branch"]=="dropout_on")
    newdrop=stages[1][1];newpos=stages[2][1]
    training_cost=dict(final_candidate_path_updates=8000,all_valid_controls_updates=16000,
        discarded_bug_updates=7000,discarded_smoke_updates=20,total_observed_optimizer_step_calls=23020,
        stages=[
            dict(stage="smoke_discarded",updates=20,wall_seconds=smoke["wall_seconds"]),
            dict(stage="lambda_pair_valid",updates=6000,wall_seconds=pair["wall_seconds"]),
            dict(stage="v1_dropout_on_valid_reused",updates=3000,wall_seconds=on["wall_seconds"]),
            dict(stage="v1_dropout_off_invalid_bias_time",updates=3000,wall_seconds=olddrop["wall_seconds"]-on["wall_seconds"]),
            dict(stage="v1_position_both_invalid_parent_and_bias_time",updates=4000,wall_seconds=oldpos["wall_seconds"]),
            dict(stage="v2_dropout_off_corrected",updates=3000,wall_seconds=newdrop["wall_seconds"]),
            dict(stage="v2_position_both_corrected",updates=4000,wall_seconds=newpos["wall_seconds"])],
        note="Training stage wall includes stage checks/checkpointing; excludes separate read-only diagnosis, fixed validation, SSH and independent audit. v1 arm2 wall is difference of cumulative process wall.")
    training_cost["all_training_stage_wall_seconds"]=sum(x["wall_seconds"] for x in training_cost["stages"])
    training_cost["successful_controls_stage_wall_seconds"]=pair["wall_seconds"]+on["wall_seconds"]+newdrop["wall_seconds"]+newpos["wall_seconds"]
    training_cost["invalid_bug_stage_wall_seconds"]=olddrop["wall_seconds"]-on["wall_seconds"]+oldpos["wall_seconds"]
    training_cost["peak_torch_allocated_bytes"]=max(x.get("peak_cuda_bytes",0) for _,r in stages for x in r["variants"])
    training_cost["peak_is_not_whole_device_usage"]=True
    files=["optimizer_copy_guard.py","ngrf_dropout_adam_v2.py","ngrf_position_adam_v2.py","ngrf_dropout_eval_adam_v2.py","ngrf_position_eval_adam_v2.py",Path(__file__).name]
    syntax=[];path_hits=[]
    checker_path=PATHS.code_root/"tools/check_hardcoded_paths.py"
    spec=importlib.util.spec_from_file_location("hw_path_check",checker_path);checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
    for n in files:
        p=Path(__file__).parent/n;ast.parse(p.read_text());syntax.append(PATHS.relative(p))
        for lineno,line in enumerate(p.read_text().splitlines(),1):
            for label,pattern in checker.PATTERNS:
                if pattern.search(line):path_hits.append(dict(file=PATHS.relative(p),line=lineno,label=label))
    assert not path_hits,path_hits
    C.atomic_json(OUT/"scoped_path_check.json",dict(status="passed",files=syntax,new_scope_hits=path_hits,syntax_passed=True,legacy_global_exit_code=1,legacy_global_log=PATHS.relative(R.REPORT/"final_global_path_check.log")))
    audit=dict(status="passed",scope="full raw16 FIT192 and fixed16 validation; no training/inference here",
        source=PATHS.relative(Path(__file__)),source_sha256=C.sha256(Path(__file__)),CUDA_visible_devices="",GPU_used=False,
        all_source_and_artifact_hashes_verified=True,records=records,validation=vals,
        syntax_passed=True,syntax_files=syntax,legacy_global_path_check=PATHS.relative(R.REPORT/"final_global_path_check.log"),
        legacy_global_exit_code=1,wall_seconds=time.monotonic()-started)
    C.atomic_json(OUT/"independent_verification.json",audit)
    with (OUT/"points.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(allpoints[0]));writer.writeheader();writer.writerows(allpoints)
    summary=dict(status="complete",corrected_v2=True,task="finite nGRF training health repair; formal500 not run",
        FIT_panel_count=16,FIT_nonzero_count=15,FIT_zero_count=1,FIT_shape=[16,256,4,192],
        groups_definition="Each site max true energy: strong >=-20dB; medium [-40,-20)dB; weak <-40dB; exact-zero separate",
        FIT=compact_stages,validation=vals,training_cost=training_cost,
        invalid_saved_bias_time_steps=dict(dropout_off=9000,position_global=11000,position_relative=13000),
        correction="CPU Adam step shallow sharing in v1; all state loads deepcopy with exact optimizer fingerprint, disjoint storage and immutable parent guards; valid first ON reused",
        limitation="Old ON initial moments were not separately snapshotted; valid reuse is established by first-arm source order, unchanged disk parent, device copy semantics, initial model/RNG/schedule and finite saved moments, not counters alone",
        artifact_audit=PATHS.relative(OUT/"independent_verification.json"),artifact_audit_sha256=C.sha256(OUT/"independent_verification.json"),
        points=PATHS.relative(OUT/"points.csv"),points_sha256=C.sha256(OUT/"points.csv"),
        all_old_results_preserved=True,V16_retrained=False,outer500_truth_read=False)
    C.atomic_json(OUT/"summary.json",summary)
    print(json.dumps(dict(status="passed",FIT={k:[dict(branch=x["branch"],nmse=x["pooled"]["complex_nmse"],worst=x["worst_nonzero_nmse"],strict=x["strict_all_point_health"]) for x in v] for k,v in compact_stages.items()},cost=training_cost)),flush=True)
if __name__=="__main__":main()
