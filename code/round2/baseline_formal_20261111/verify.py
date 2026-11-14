"""Verify scored formal predictions and preservation without GPU or retraining."""
from __future__ import annotations
import ast,importlib.util,json,os,time
from pathlib import Path
if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise RuntimeError("CPU-only verification")
import numpy as np
from project_paths import PATHS
from common.official_metrics import PointMetricStatistics,point_metric_statistics,aggregate_statistics
from round2.paper_baseline_training_v1 import common as C
from round2.baseline_formal_20261111 import score as S

R=S.R;A=S.A
def check_stat(entry):
    p=PATHS.root/entry["path"];st=p.stat()
    return (st.st_size,st.st_mtime_ns,st.st_ino)==(entry["size"],entry["mtime_ns"],entry["inode"])

def preserve():
    previous=C.REPORT/"repair_20261111/preservation_before.json"
    new=json.loads((R/"preservation_before.json").read_text())
    old=json.loads(previous.read_text())
    hashes={p:v["sha256"] for p,v in new["hashes"].items()}
    hashes.update(old["hashes"])
    bad=[p for p,h in hashes.items() if not (PATHS.root/p).is_file() or C.sha256(PATHS.root/p)!=h]
    if bad:raise RuntimeError(("Protected hash mismatch",bad))
    stats={v["path"]:v for v in new["model_and_array_stats"]+old["stats"]+old["raw_data_stats"]}
    bad=[p for p,v in stats.items() if not check_stat(v)]
    if bad:raise RuntimeError(("Protected model/data stat changed",bad))
    import hashlib
    with (PATHS.root/"experiment_records/EXPERIMENT_RECORD.md").open("rb") as f:prefix=f.read(new["record_prefix_bytes"])
    assert hashlib.sha256(prefix).hexdigest()==new["record_prefix_sha256"]
    return dict(protected_hashes=len(hashes),protected_stats=len(stats),record_prefix_match=True,
                stat_checks_are_not_full_content_hashes=True)

def check_sources():
    p=PATHS.code_root/"tools/check_hardcoded_paths.py"
    spec=importlib.util.spec_from_file_location("source_path_rules",p)
    rules=importlib.util.module_from_spec(spec);spec.loader.exec_module(rules)
    d=PATHS.code_root/"round2/baseline_formal_20261111"
    files={}
    for f in sorted(d.glob("*.py")):
        source=f.read_text();ast.parse(source,filename=f.name)
        bad=[(i,label) for i,line in enumerate(source.splitlines(),1) for label,pat in rules.PATTERNS if pat.search(line)]
        if bad:raise RuntimeError(("New source path rule",f.name,bad))
        files[PATHS.relative(f)]=C.sha256(f)
    return files

def main():
    started=time.monotonic()
    old=preserve()
    sources=check_sources()
    inputs=C.SimulationInputs()
    manifests=S.preflight(inputs)
    truth=np.load(C.RAW,mmap_mode="r",allow_pickle=False)
    assert truth.shape==(4000,256,4,192) and truth.dtype==np.complex64
    query=np.asarray(inputs.query_ids,np.int64);index={int(x):i for i,x in enumerate(query)}
    audits={}
    for name,groups in manifests.items():
        p=R/"evaluation"/name/"results.json"
        recorded=json.loads(p.read_text())
        cached_path=p.parent/"independent_verification.json"
        if cached_path.exists():
            cached=json.loads(cached_path.read_text())
            if (cached["status"]=="passed" and cached["result_sha256"]==C.sha256(p)
                    and cached["five_prediction_manifest_hashes"]==recorded["prediction_manifests"]
                    and cached.get("official_metric_sha256")==C.sha256(PATHS.code_root/"common/official_metrics.py")):
                audits[name]=cached
                continue
        assembled={k:np.zeros(500,np.float64) for k in ("pas","pdp","squared_error","truth_energy")}
        for _,d,path in groups:
            x=np.load(path,mmap_mode="r",allow_pickle=False);rows=np.asarray(d["row_ids"],np.int64)
            for lo in range(0,len(rows),4):
                ids=rows[lo:lo+4]
                st=point_metric_statistics(np.asarray(x[lo:lo+len(ids)]),np.asarray(truth[ids]))
                jj=[index[int(k)] for k in ids]
                for key in assembled:assembled[key][jj]=getattr(st,key)
        check=aggregate_statistics(PointMetricStatistics(**assembled))
        delta={k:abs(float(check[k])-float(recorded["metrics"][k])) for k in check}
        assert max(delta.values())<1e-10,(name,delta)
        zero=int(np.sum(assembled["truth_energy"]==0))
        assert zero==recorded["zero_truth_row_count"]==31
        npz=p.parent/"point_statistics.npz"
        assert C.sha256(npz)==recorded["point_statistics_sha256"]
        with np.load(npz,allow_pickle=False) as z:
            assert np.array_equal(z["row_ids"],query)
            pdiff={k:float(np.max(np.abs(z[k]-assembled[k]))) for k in assembled}
        assert max(pdiff.values())<1e-10,(name,pdiff)
        audits[name]=dict(status="passed",metrics=check,max_aggregate_delta=max(delta.values()),
            max_point_statistic_deltas=pdiff,zero_truth_count=zero,result_sha256=C.sha256(p),
            five_prediction_manifest_hashes=recorded["prediction_manifests"])
    c=json.loads((R/"evaluation/comparison.json").read_text())
    previous=json.loads(S.old_rows()[0].read_text())
    prev={v["key"]:v["metrics"] for v in previous["rows"]}
    for row in c["rows"]:
        k=row["key"]
        if k in ("v16","wrfgs_full_h_v1"):assert row["metrics"]==prev[k]
        else:assert row["metrics"]==json.loads((R/"evaluation"/k/"results.json").read_text())["metrics"]
    report=dict(status="passed",verified_new_methods=list(audits),raw_CPU_recompute=audits,
        old_file_protection=old,new_source_ast_and_path_checks=sources,
        comparison_sha256=C.sha256(R/"evaluation/comparison.json"),
        retained_V16_and_WRFGS_metrics_exact=True,elapsed_seconds=time.monotonic()-started,
        full_training_budget_is_finite_not_convergence_claim=True)
    C.atomic_json(R/"evaluation/completed_verification.json",report)
    print(json.dumps({"status":"passed","methods":list(audits),"old":old,
                      "elapsed_seconds":report["elapsed_seconds"]},ensure_ascii=False),flush=True)
if __name__=="__main__":main()

