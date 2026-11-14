"""Score completed canonical outer predictions. No model execution or oracle fits."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
from datetime import datetime, timezone

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("Final scoring is CPU-only; set empty CUDA_VISIBLE_DEVICES")
import numpy as np
from scipy import fft
from common.official_metrics import (ArrayLayout, PointMetricStatistics, official_cosine,
                                      aggregate_statistics, point_metric_statistics)
from project_paths import PATHS
from round2.paper_baseline_training_v1.common import (
    ART, REPORT, SimulationInputs, atomic_json, digest_ids, sha256)

METHODS = ("nerf2", "newrf", "ngrf", "wrfgs")
LABELS = {"nerf2":"NeRF²（全H适配）", "newrf":"NeWRF（全H适配）",
          "ngrf":"nGRF（全H适配）", "wrfgs":"WRF-GS+（功率适配）"}
SCORE_DIR = REPORT / "final_scores"
ARCHIVE = PATHS.analysis / "results/round2_official_sionna_path_calibrator/fixed500_report.json"
OLD_TABLE = PATHS.analysis / "results/round2_existing_comparison_20261101/results.json"


def method_root(method):
    versions={"newrf":"v4","ngrf":"v2","nerf2":"v3"}
    return ART/method/versions[method] if method in versions else ART/method


def now():
    return datetime.now(timezone.utc).isoformat()


def pas_power_one_worker(h):
    a = np.asarray(h, np.complex64)
    if a.ndim != 4 or a.shape[1:] != (256,4,192):
        raise ValueError("Expected canonical full channel")
    sh = a.reshape(len(a),2,16,8,4,192)
    return np.square(np.abs(fft.fft2(sh, axes=(2,3), workers=1)),
                     dtype=np.float32).reshape(a.shape)


def statistics(pred, truth, power_only=False):
    target = np.asarray(truth, np.complex64)
    pp = np.asarray(pred, np.float32) if power_only else pas_power_one_worker(pred)
    tp = pas_power_one_worker(target)
    pas = official_cosine(pp, tp, 1).mean((1,2), dtype=np.float64)
    if power_only:
        return {"pas":pas, "truth_energy":np.sum(np.abs(target)**2, (1,2,3), dtype=np.float64)}
    pred = np.asarray(pred, np.complex64)
    dp = np.square(np.abs(fft.ifft(pred, axis=-1, workers=1)), dtype=np.float32)
    dt = np.square(np.abs(fft.ifft(target, axis=-1, workers=1)), dtype=np.float32)
    pdp = official_cosine(dp, dt, 3).mean((1,2), dtype=np.float64)
    error = np.sum(np.abs(pred-target)**2, (1,2,3), dtype=np.float64)
    energy = np.sum(np.abs(target)**2, (1,2,3), dtype=np.float64)
    return {"pas":pas, "pdp":pdp, "squared_error":error, "truth_energy":energy}


def fixture():
    rng = np.random.default_rng(827)
    truth = (rng.normal(size=(3,256,4,192))+1j*rng.normal(size=(3,256,4,192))).astype(np.complex64)*1e-5
    truth[2] = 0
    pred = .8*truth + (rng.normal(size=truth.shape)+1j*rng.normal(size=truth.shape)).astype(np.complex64)*1e-6
    a=statistics(pred,truth); b=point_metric_statistics(pred,truth)
    differences={k:float(np.max(np.abs(v-getattr(b,k)))) for k,v in a.items()}
    if any(d > 1e-12 for d in differences.values()):
        raise AssertionError(differences)
    identity=statistics(truth,truth)
    st=PointMetricStatistics(**identity)
    score=aggregate_statistics(st)
    if abs(score["pas"]-2/3)>1e-12 or abs(score["pdp"]-2/3)>1e-12:
        raise AssertionError("Zero target row must remain in cosine denominator count")
    power=statistics(pas_power_one_worker(pred),truth,True)
    if not np.array_equal(power["pas"],a["pas"]):
        raise AssertionError("Power and full-H PAS must be identical")
    result=dict(status="passed",scope="synthetic numerical correctness",
                worker1_vs_original_max_absolute=differences,
                zero_row_retained=True,power_H_PAS_identical=True,oracle_alignment=False,
                metric_code_sha256=sha256(PATHS.code_root/"common/official_metrics.py"),
                scorer_sha256=sha256(Path(__file__)))
    atomic_json(REPORT/"final_scorer_fixture.json",result)
    return result


def manifests_for(method,inputs, verify_hashes=True):
    result=[]
    for outer in range(5):
        p=method_root(method)/f"outer{outer}"/"prediction"/"manifest.json"
        if not p.exists():
            return None
        d=json.loads(p.read_text())
        ids=inputs.roles(outer)["outer"]
        want_kind="power" if method=="wrfgs" else "channel"
        if any((d.get("schema")!="canonical-baseline-prediction-v1",
                d.get("method")!=method,d.get("outer")!=outer,
                d.get("input_identity")!=inputs.identity,
                d.get("kind")!=want_kind,d.get("trained") is not True,
                d.get("query_truth_read") is not False,
                d.get("prediction_namespace")!="canonical_train_row_ids")):
            raise ValueError("Prediction manifest identity or role mismatch: "+str(p))
        if d["row_ids"]!=ids.tolist() or d["row_ids_sha256"]!=digest_ids(ids):
            raise ValueError("Query row order differs from canonical fold")
        fd=d["fit_data"]
        if any((fd["role"]!="refit",fd["outer"]!=outer,
                fd["input_identity"]!=inputs.identity,
                fd["fit_ids"]!=inputs.roles(outer)["refit"].tolist(),
                fd["fit_ids_sha256"]!=digest_ids(inputs.roles(outer)["refit"]),
                fd["validation_read_count"]!=0,fd["validation_rows_actually_read"]!=[],
                fd["query_truth_read"] is not False,
                fd["oracle_phase_delay_scale_correction"] is not False,
                not set(fd["fit_rows_actually_read"]).issubset(set(fd["fit_ids"])))):
            raise ValueError("Refit-label isolation mismatch")
        q=PATHS.root/d["path"]
        expected=(method_root(method)/f"outer{outer}"/"prediction"/
                  ("power.npy" if method=="wrfgs" else "channels.npy"))
        if q.resolve()!=expected.resolve() or d["shape"]!=[len(ids),256,4,192]:
            raise ValueError("Canonical prediction path/shape mismatch")
        if not np.array_equal(np.load(q.parent/"row_ids.npy",allow_pickle=False),ids):
            raise ValueError("Stored query IDs mismatch")
        a=np.load(q,mmap_mode="r",allow_pickle=False)
        dtype=np.dtype("float32" if method=="wrfgs" else "complex64")
        if a.shape!=tuple(d["shape"]) or a.dtype!=dtype or d["dtype"]!=str(dtype):
            raise ValueError("Prediction header mismatch")
        if verify_hashes and sha256(q)!=d["sha256"]:
            raise ValueError("Prediction file changed")
        for rel,h in d.get("metadata",{}).get("source_identity",{}).items():
            sp=(PATHS.root/rel) if "/" in rel else Path(__file__).parent/rel
            if not sp.is_file() or sha256(sp)!=h:
                raise ValueError("Training source no longer matches frozen prediction: "+str(sp))
        result.append((p,d,q))
    allids=np.concatenate([np.asarray(x[1]["row_ids"],np.int64) for x in result])
    if len(allids)!=500 or len(np.unique(allids))!=500 or set(allids)!=set(inputs.query_ids):
        raise ValueError("Incomplete or duplicate final500 set")
    return result


def score_method(method,inputs):
    existing=SCORE_DIR/method/"results.json"
    found=manifests_for(method,inputs,verify_hashes=not existing.exists())
    if found is None:
        return None
    out=SCORE_DIR/method
    manifest_hashes={PATHS.relative(p):sha256(p) for p,_,_ in found}
    if existing.exists():
        d=json.loads(existing.read_text())
        if d["prediction_manifests"]!=manifest_hashes or d["input_identity"]!=inputs.identity:
            raise RuntimeError("Already scored artifact identities changed")
        return d
    truth_path=PATHS.round2/"Round2_Train_Channel.npy"
    truth=np.load(truth_path,mmap_mode="r",allow_pickle=False)
    if truth.shape!=(4000,256,4,192) or truth.dtype!=np.complex64:
        raise ValueError("Raw truth shape/dtype")
    st=truth_path.stat()
    raw_identity=dict(path=PATHS.relative(truth_path),bytes=st.st_size,mtime_ns=st.st_mtime_ns,inode=st.st_ino)
    pieces=[]; row_order=[]
    for p,d,q in found:
        if d["fit_data"]["raw_channel_identity"]!=raw_identity:
            raise ValueError("Raw channel file changed since training")
        pred=np.load(q,mmap_mode="r",allow_pickle=False)
        ids=np.asarray(d["row_ids"],np.int64)
        for start in range(0,len(ids),4):
            pp=np.array(pred[start:start+4],copy=True)
            tt=np.array(truth[ids[start:start+4]],copy=True)
            if not np.isfinite(pp).all() or not np.isfinite(tt).all():
                raise ValueError("Nonfinite output/truth")
            if method=="wrfgs" and np.any(pp<0):
                raise ValueError("Power prediction negative")
            pieces.append(statistics(pp,tt,method=="wrfgs"))
            row_order.extend(ids[start:start+4].tolist())
    rows=np.asarray(row_order,np.int64)
    reorder=np.array([int(np.flatnonzero(rows==x)[0]) for x in inputs.query_ids])
    arrays={k:np.concatenate([v[k] for v in pieces])[reorder] for k in pieces[0]}
    if method=="wrfgs":
        metrics=dict(count=500,pas=float(arrays["pas"].mean()),pdp=None,nmse=None,nmse_score=None,score=None)
    else:
        metrics=aggregate_statistics(PointMetricStatistics(**arrays))
    if any(not math.isfinite(v) for k,v in metrics.items() if v is not None):
        raise ValueError("Invalid final score")
    out.mkdir(parents=True,exist_ok=True)
    with (out/"point_statistics.npz").open("xb") as f:
        np.savez(f,row_ids=inputs.query_ids,fold_id=inputs.fold_id,**arrays)
    d=dict(schema="huawei-canonical500-final-score-v1",method=method,
           generated_utc=now(),input_identity=inputs.identity,row_ids_sha256=digest_ids(inputs.query_ids),
           metrics=metrics,raw_prediction_rescored=True,
           raw_truth_identity=raw_identity,prediction_manifests=manifest_hashes,
           prediction_files={x[1]["path"]:x[1]["sha256"] for x in found},
           point_statistics_sha256=sha256(out/"point_statistics.npz"),
           all_five_outer_folds_complete=True,all_zero_truth_rows_retained=True,
           zero_truth_row_count=int(np.sum(arrays["truth_energy"]==0)),
           query_truth_use="final scoring only after all5 predictions frozen",
           phase_delay_amplitude_alignment=False,scoring_device="CPU",fft_workers=1,
           nmse_aggregation="sum squared error / sum truth energy; linear",
           formula="0.4*PAS+0.4*PDP+0.2/(1+global_NMSE)",
           original_paper_exact_reproduction=False,
           fixed_configuration=True,seed=42,
           scorer_sha256=sha256(Path(__file__)),
           official_metric_sha256=sha256(PATHS.code_root/"common/official_metrics.py"))
    atomic_json(existing,d)
    print(json.dumps(dict(event="completed_method_score",method=method,metrics=metrics),ensure_ascii=False),flush=True)
    return d


def table(inputs):
    archive=json.loads(ARCHIVE.read_text())["selected"]
    original=json.loads(OLD_TABLE.read_text())
    if archive!=original["archived_V16_selected"] or archive["count"]!=500:
        raise ValueError("Accepted V16 summary changed")
    total=.4*archive["pas"]+.4*archive["pdp"]+.2/(1+archive["nmse"])
    if abs(total-archive["score"])>1e-12:
        raise ValueError("V16 archival formula")
    rows=[dict(method="V16（用户已验证版本）",key="v16",metrics=archive,status="accepted_archived_fixed500_summary",raw_prediction_rescored=False)]
    for m in METHODS:
        p=SCORE_DIR/m/"results.json"
        d=json.loads(p.read_text()) if p.exists() else None
        count=sum(len(json.loads(f.read_text())["row_ids"]) for f in
                  method_root(m).glob("outer[0-4]/prediction/manifest.json")) if method_root(m).exists() else 0
        rows.append(dict(method=LABELS[m],key=m,metrics=d["metrics"] if d else None,
                         status="completed" if d else "prediction_incomplete",prediction_count=count,
                         raw_prediction_rescored=d is not None,
                         selected_artifact_root=PATHS.relative(method_root(m))))
    def cell(row,key):
        if row["key"]=="wrfgs" and key!="pas":return "不适用"
        if row["metrics"] is None:return "待补齐"
        return f'{row["metrics"][key]:.6f}'
    header=["方法","PAS ↑","PDP ↑","NMSE（线性）↓","比赛总分 ↑"]
    entries=[[r["method"]]+[cell(r,k) for k in ["pas","pdp","nmse","score"]] for r in rows]
    buf=io.StringIO();w=csv.writer(buf);w.writerow(header);w.writerows(entries)
    p=REPORT/"comparison.csv";tmp=p.with_suffix(".csv.partial");tmp.write_text("\ufeff"+buf.getvalue(),encoding="utf-8");tmp.replace(p)
    lines=["# 华为仿真四基线比较","","| "+" | ".join(header)+" |","|---|---:|---:|---:|---:|"]
    lines += ["| "+" | ".join(r)+" |" for r in entries]
    lines += ["","V16沿用用户已验证版本的历史500点汇总；不重训、不替换、不声称重新从H评分。",
              "四个文献方法均是华为任务适配版，具体差异见各方法报告；不称为原论文原样复现。",
              "仅5折全部预测完成的方法进入成绩表。部分折的验证值和外层预测不填正式成绩。",
              "WRF-GS+功率适配版只比较PAS，不能从功率构造完整H、PDP或NMSE。",
              f"V16原线上成绩 {original['archived_online_total']:.6f} 单独保留，不与本地500点各项指标混合。",
              "总分=0.4 PAS+0.4 PDP+0.2/(1+global NMSE)。所有零信道行保留；不做测试相位、时延或幅度校正。",
              "只有1个种子；V16原逐点预测缺失。此表不能提供与V16逐点配对置信区间，也不能仅凭适配版失败宣称原文献方法无效。",""]
    p=REPORT/"comparison.md";tmp=p.with_suffix(".md.partial");tmp.write_text("\n".join(lines),encoding="utf-8");tmp.replace(p)
    atomic_json(REPORT/"comparison.json",dict(generated_utc=now(),rows=rows,
                all_four_complete=all(x["status"]=="completed" for x in rows[1:]),
                original_V16_preserved=True,input_identity=inputs.identity,
                archived_V16_source_sha256=sha256(ARCHIVE),old_table_sha256=sha256(OLD_TABLE)))
    return all(x["status"]=="completed" for x in rows[1:])


def main():
    p=argparse.ArgumentParser();p.add_argument("--watch",action="store_true")
    p.add_argument("--interval",type=int,default=30);p.add_argument("--fixture",action="store_true")
    args=p.parse_args()
    if args.fixture:print(json.dumps(fixture()));return
    inputs=SimulationInputs()
    if inputs.setup["w"]!=[.4,.4,.2] or digest_ids(inputs.query_ids)!="f115290de0dc5f80cfe304e1f2143696694e66eec10c418e1b086d0b179ed08b":
        raise ValueError("Frozen comparison weights/query500 changed")
    SCORE_DIR.mkdir(parents=True,exist_ok=True)
    while True:
        for m in METHODS:score_method(m,inputs)
        complete=table(inputs)
        if complete or not args.watch:return
        time.sleep(max(5,min(args.interval,60)))


if __name__=="__main__":main()

