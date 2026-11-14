"""CPU-only final500 scoring for the three frozen formal repair runs."""
from __future__ import annotations
import csv,io,json,os,time
from pathlib import Path
if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
    raise RuntimeError("Formal final scoring requires CPU-only CUDA mask")
import numpy as np
from project_paths import PATHS
from round2.paper_baseline_training_v1 import final_score as F
from round2.paper_baseline_training_v1 import common as C

R=PATHS.analysis/"results/round2_paper_four_baselines_v1/formal_20261111"
A=PATHS.artifacts/"round2/paper_four_baselines_v1/formal_20261111"
METHODS=("nerf2","newrf","ngrf")
LABEL={"nerf2":"NeRF²（修复版）","newrf":"NeWRF（修复版）","ngrf":"nGRF（修复版）"}

def training_evidence(method,manifest):
    fold=int(manifest["outer"])
    records={}
    for role in ("screen","refit"):
        p=R/method/f"outer{fold}"/role/"result.json"
        d=json.loads(p.read_text())
        if d["status"] not in ("complete",role+"_complete"):raise RuntimeError("Incomplete training stage: "+str(p))
        records[role]=d
    d=records["refit"];fit=manifest["fit_data"];n=len(fit["fit_ids"])
    if method=="nerf2":
        cov=d["actual_coverage"]
        if len(cov)!=8 or manifest["all8fields_trained"] is not True:
            raise RuntimeError("NeRF2 field count")
        for v in cov.values():
            if v["actual_optimizer_updates"]<=0 or v["unique_gradient_positions"]!=v["permitted_FIT_positions"] or v["unique_position_BS_links"]!=v["required_position_BS_links"]:
                raise RuntimeError("NeRF2 actual FIT-link coverage")
        model=manifest["checkpoint"];digest=manifest["checkpoint_sha256"]
    elif method=="newrf":
        cov=manifest["metadata"]["gradient_coverage"]
        if len(cov)!=8 or manifest["metadata"]["all8fields_trained"] is not True:
            raise RuntimeError("NeWRF field count")
        for v in cov.values():
            if v["gradient_seen_unique_rows"]!=n or v["gradient_seen_min_count"]<=0:
                raise RuntimeError("NeWRF actual FIT-position coverage")
            if any(st["updates"]<=0 for st in v["stages"].values()):
                raise RuntimeError("NeWRF empty optimization stage")
        model=manifest["metadata"]["checkpoint"];digest=manifest["metadata"]["checkpoint_sha256"]
    else:
        if d["unique_positions_in_gradient_updates"]!=n or d["min_position_visits"]<=0:
            raise RuntimeError("nGRF actual FIT-position coverage")
        if len(d["frequency_step_counts"])!=192 or min(d["frequency_step_counts"])<=0:
            raise RuntimeError("nGRF incomplete tone supervision")
        model=manifest["metadata"]["refit_checkpoint"];digest=manifest["metadata"]["refit_checkpoint_sha256"]
    if C.sha256(PATHS.root/model)!=digest:
        raise RuntimeError("Prediction checkpoint hash changed")
    return {"screen":C.sha256(R/method/f"outer{fold}"/"screen"/"result.json"),
            "refit":C.sha256(R/method/f"outer{fold}"/"refit"/"result.json"),"model":digest}

def preflight(inputs):
    F.SCORE_DIR=R/"evaluation"
    F.method_root=lambda method:A/method
    found={};training={}
    for method in METHODS:
        groups=F.manifests_for(method,inputs,verify_hashes=True)
        if groups is None:raise RuntimeError("Incomplete five-fold prediction: "+method)
        training[method]={}
        for path,manifest,prediction in groups:
            training[method][str(manifest["outer"])]=training_evidence(method,manifest)
            if not manifest.get("fit_data",{}).get("fit_rows_actually_read"):
                raise RuntimeError("No training label reads: "+str(path))
            declared=manifest.get("metadata",{}).get("formal_training",{})
            # Detailed field and sampler coverage is recorded by each method.
            if declared.get("all_required_fields_trained") is False:
                raise RuntimeError("Untrained field declared")
            values=np.load(prediction,mmap_mode="r",allow_pickle=False)
            for lo in range(0,len(values),8):
                if not np.isfinite(values[lo:lo+8]).all():raise RuntimeError("Nonfinite prediction")
        found[method]=groups
    sealed={method:{PATHS.relative(p):C.sha256(p) for p,_,_ in groups} for method,groups in found.items()}
    target=R/"evaluation/predictions_sealed.json"
    if target.exists():
        previous=json.loads(target.read_text())
        if previous["manifests"]!=sealed or previous["training_evidence"]!=training:raise RuntimeError("Previously sealed evidence changed")
    else:
        C.atomic_json(target,{"schema":"three-method-formal-predictions-sealed-v1","manifests":sealed,"training_evidence":training,
             "generated_utc":F.now(),"input_identity":inputs.identity,
             "formal_methods":list(METHODS),"each_query_count":500,
             "each_method_sealed_before_its_truth_scoring":True,
             "combined_snapshot_created_after_individual_method_seals":True,
             "orchestrator_sha256":C.sha256(Path(__file__))})
    return found

def old_rows():
    p=PATHS.analysis/"results/round2_paper_four_baselines_v1/wrfgs_full_h_v1_20261110/evaluation/comparison.json"
    d=json.loads(p.read_text())
    return p,d

def main():
    started=time.monotonic()
    inputs=C.SimulationInputs()
    found=preflight(inputs)
    results={m:F.score_method(m,inputs) for m in METHODS}
    if any(x is None for x in results.values()):raise RuntimeError("Unexpected missing result after preflight")
    # Preserve the exact old comparison payload to avoid silently rescoring V16.
    old_path,old=old_rows()
    C.atomic_json(R/"evaluation/retained_comparison_source.json",
         {"source":PATHS.relative(old_path),"sha256":C.sha256(old_path),"payload":old})
    if old["query_ids_sha256"]!=C.digest_ids(inputs.query_ids):raise RuntimeError("Retained score query set differs")
    by_method={m:results[m]["metrics"] for m in METHODS}
    C.atomic_json(R/"evaluation/new_method_metrics.json",
         {"methods":by_method,"orchestrator_sha256":C.sha256(Path(__file__)),
          "prediction_seal_sha256":C.sha256(R/"evaluation/predictions_sealed.json"),
          "retained_methods":["V16","WRF-GS+ full H"],"retained_methods_rescored":False,
          "elapsed_seconds":time.monotonic()-started,"formula":"0.4*PAS+0.4*PDP+0.2/(1+global_NMSE)",
          "all_zero_truth_rows_retained":True,"phase_delay_amplitude_alignment":False})
    retained={v["key"]:v for v in old["rows"]}
    rows=[{"key":"v16","method":"V16（历史版本保留）","metrics":retained["v16"]["metrics"],
           "origin":"archived summary, not retrained or rescored"}]
    for method in METHODS:
        rows.append({"key":method,"method":LABEL[method],"metrics":by_method[method],
                     "origin":"formal repaired five-fold full-H predictions"})
    rows.append({"key":"wrfgs_full_h_v1","method":"WRF-GS+（完整H适配，保留）",
                 "metrics":retained["wrfgs_full_h_v1"]["metrics"],
                 "origin":"existing verified full-H result, not retrained"})
    deltas={m:{k:by_method[m][k]-retained[m]["metrics"][k] for k in ("pas","pdp","nmse","score")}
            for m in METHODS}
    comparison={"rows":rows,"old_baseline_deltas":deltas,"query_ids_sha256":C.digest_ids(inputs.query_ids),
                "new_methods_prediction_roots":{m:PATHS.relative(A/m) for m in METHODS},
                "all_new_methods_formal500":True,"V16_retrained":False,"WRFGS_retrained":False,
                "V16_historical_training_coverage_difference_retained":True,
                "same_input_output_and_scoring_not_equal_compute":True}
    C.atomic_json(R/"evaluation/comparison.json",comparison)
    buf=io.StringIO();writer=csv.writer(buf)
    writer.writerow(["方法","PAS ↑","PDP ↑","NMSE（线性）↓","综合分 ↑"])
    lines=["# 华为仿真500点：修复版三基线更新","",
           "| 方法 | PAS ↑ | PDP ↑ | NMSE（线性）↓ | 综合分 ↑ |",
           "|---|---:|---:|---:|---:|"]
    for row in rows:
        values=[row["method"]]+[f'{row["metrics"][k]:.6f}' for k in ("pas","pdp","nmse","score")]
        writer.writerow(values);lines.append("| "+" | ".join(values)+" |")
    (R/"evaluation/comparison.csv").write_text("﻿"+buf.getvalue())
    lines+=["","NeRF²、NeWRF、nGRF来自本轮修复版各自5组正式训练、重新拟合与完整H预测。",
            "V16、WRF-GS+保留已有成绩，没有重训或重新评分。V16历史训练覆盖差异仍保留，不能据此声明严格相同训练覆盖下的排名。",
            "全部500点和31个零信道点保留。没有测试真值相位、时延、幅度校正。",
            "总分=0.4 PAS+0.4 PDP+0.2/(1+global NMSE)。各方法预算不同，来源与训练记录见本轮配置及分组报告。",
            "单次固定seed的结果不提供跨seed显著性结论；原论文的任务适配差异继续披露。"]
    (R/"evaluation/RESULTS.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"status":"new_three_methods_scored","metrics":by_method,
                      "retained_source_schema":list(old) if isinstance(old,dict) else "list",
                      "elapsed_seconds":time.monotonic()-started},ensure_ascii=False),flush=True)

if __name__=="__main__":main()

