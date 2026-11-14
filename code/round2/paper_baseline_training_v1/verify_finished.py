"""Independently verify saved sufficient statistics and final comparison entries."""
from __future__ import annotations
import argparse,json,math
import numpy as np
from pathlib import Path
from .common import REPORT,sha256,atomic_json,digest_ids
from .final_score import method_root
from project_paths import PATHS

def main(args):
    comparison=json.loads((REPORT/"comparison.json").read_text())
    verified=[]
    for row in comparison["rows"]:
        if row["key"]=="v16":
            m=row["metrics"];total=.4*m["pas"]+.4*m["pdp"]+.2/(1+m["nmse"])
            if abs(total-m["score"])>1e-12:raise AssertionError("ArchivedV16 formula")
            continue
        if row["status"]!="completed":continue
        method=row["key"];root=REPORT/"final_scores"/method
        d=json.loads((root/"results.json").read_text())
        with np.load(root/"point_statistics.npz",allow_pickle=False) as z:
            ids=z["row_ids"];energy=z["truth_energy"];pas=z["pas"]
            if len(ids)!=500 or len(np.unique(ids))!=500:raise AssertionError("Canonical500 IDs")
            if digest_ids(ids)!="f115290de0dc5f80cfe304e1f2143696694e66eec10c418e1b086d0b179ed08b":raise AssertionError("Query ID digest")
            if np.any(pas[energy==0]!=0) or int((energy==0).sum())!=31:raise AssertionError("Zero rows")
            result=dict(count=500,pas=float(pas.mean()))
            if method=="wrfgs":result.update(pdp=None,nmse=None,nmse_score=None,score=None)
            else:
                nmse=float(z["squared_error"].sum()/max(energy.sum(),1e-30));pdp=float(z["pdp"].mean())
                result.update(pdp=pdp,nmse=nmse,nmse_score=1/(1+nmse),
                              score=.4*result["pas"]+.4*pdp+.2/(1+nmse))
        for key,value in result.items():
            for observed in [d["metrics"][key],row["metrics"][key]]:
                if value is None:
                    if observed is not None:raise AssertionError("Unsupported metric")
                elif not math.isfinite(observed) or abs(value-observed)>1e-12:
                    raise AssertionError((method,key,value,observed))
        if sha256(root/"point_statistics.npz")!=d["point_statistics_sha256"]:raise AssertionError("Point-stat hash")
        hashes=0
        if args.full_hash:
            for rel,want in d["prediction_files"].items():
                if sha256(PATHS.root/rel)!=want:raise AssertionError("Prediction mutated: "+rel)
                hashes+=1
            for rel,want in d["prediction_manifests"].items():
                if sha256(PATHS.root/rel)!=want:raise AssertionError("Manifest mutated: "+rel)
        verified.append(dict(method=method,global_aggregation_and_table_match=True,
                             zero_rows_retained=31,raw_prediction_files_rehashed=hashes))
    if args.require_all and len(verified)!=4:raise RuntimeError("All4 methods not complete")
    atomic_json(REPORT/"completed_result_verification.json",dict(status="passed",
                completed_methods=len(verified),all_four_complete=len(verified)==4,
                source_statistics_reaggregated=True,new_raw_H_scoring=False,verified=verified,
                V16_retrained=False,scoring_formula="0.4PAS+0.4PDP+0.2/(1+globalNMSE)",
                verifier_sha256=sha256(Path(__file__))))
    print(json.dumps(dict(status="passed",completed_methods=len(verified),full_hash=args.full_hash)),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--require-all",action="store_true")
    p.add_argument("--full-hash",action="store_true");main(p.parse_args())

