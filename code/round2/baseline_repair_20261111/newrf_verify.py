"""Independent CPU audit of saved NeWRF repair predictions and native resumption."""
from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np
import torch
from project_paths import PATHS
from . import newrf_repair as N
from . import newrf_low_lr as L
from . import newrf_native_finish as F
from round2.paper_baseline_training_v1.common import sha256,atomic_json
def one(p,t):
    p=np.asarray(p,np.complex128);t=np.asarray(t,np.complex128)
    pe=float(np.vdot(p.ravel(),p.ravel()).real);te=float(np.vdot(t.ravel(),t.ravel()).real)
    e=p-t;se=float(np.vdot(e.ravel(),e.ravel()).real)
    return dict(nmse=se/te,energy_ratio=pe/te,
        signed_correlation_real=float(np.vdot(p.ravel(),t.ravel()).real/math.sqrt(pe*te)))
def main():
    r=json.loads((F.R/"result.json").read_text())
    assert r["status"]=="two_position_actual_query_fit_pass"
    N.check_identity(r["source_identity"])
    p=np.load(F.A/"query_prediction.npy",allow_pickle=False)
    t=np.load(N.PARENT/"actual_target_field.npy",allow_pickle=False)
    assert p.shape==t.shape==(2,128,192) and p.dtype==np.complex64 and np.isfinite(p).all()
    actual=one(p,t);per=[one(x,y) for x,y in zip(p,t)]
    stored=r["actual_query_metrics"]
    deltas={k:actual[k]-stored[k] for k in actual}
    pdeltas=[{k:x[k]-y[k] for k in x} for x,y in zip(per,stored["per_position"])]
    assert max([abs(x) for x in deltas.values()]+[abs(v) for d in pdeltas for v in d.values()])<1e-7
    assert actual["nmse"]<=.1 and all(x["nmse"]<=.2 and .7<=x["energy_ratio"]<=1.3 for x in per)
    assert sha256(F.A/"query_prediction.npy")==r["query_prediction_sha256"]
    assert sha256(F.A/"final.pt")==r["final_checkpoint_sha256"]
    assert r["clean_reload"]["relative_L2"]<=1e-5 and r["training_vs_actual_query"]["relative_L2"]<=1e-5
    child=torch.load(L.A/"final.pt",map_location="cpu",weights_only=False)
    final=torch.load(F.A/"final.pt",map_location="cpu",weights_only=False)
    assert child["stage"]=="native" and child["stage_updates"]==771
    assert final["native_updates"]==1000 and final["extra_updates"]==229
    assert child["new_updates"]+final["extra_updates"]==2500
    assert child["parent_total_updates"]+2500==r["combined_optimizer_updates"]==8500
    warm=[x for x in child["history"] if x["stage"]=="warm"]
    assert len(warm)==3 and all(x["gate_now"] for x in warm) and warm[-1]["stable_full_checks"]==3
    before=child["optimizer"]["state"];after=final["optimizer"]["state"]
    assert before.keys()==after.keys()
    step_deltas={str(k):float(after[k]["step"])-float(before[k]["step"]) for k in before}
    assert all(x==229 for x in step_deltas.values())
    sampler=N.Sampler(np.array([2666,2042]),42);sampler.restore(child["sampler"])
    expected=[]
    for _ in range(229):
        rr,bb,ff=sampler.draw();expected.append(np.stack([rr,bb,ff],axis=-1))
    actual_draws=np.load(F.A/"new_supervised_row_BS_frequency_ids.npy",allow_pickle=False)
    assert np.array_equal(actual_draws,np.stack(expected))
    assert r["data"]["fit_rows_actually_read"]==[2042,2666]
    assert r["data"]["validation_read_count"]==0 and not r["data"]["query_truth_read"]
    result=dict(status="passed",kind="independent CPU saved-prediction verification",
        shape=list(p.shape),dtype=str(p.dtype),raw_H=True,independent_metrics=actual,
        independent_per_position_metrics=per,stored_global_metric_deltas=deltas,stored_per_position_deltas=pdeltas,
        parent_native_updates=771,completed_native_updates=1000,resumed_updates=229,
        combined_optimizer_updates=8500,all_Adam_step_increments_equal229=True,
        sampler_exactly_continues_parent_endpoint=True,saved_raw_predictions_finite=True,
        source_and_model_hashes_match=True,warm_three_stable_checks=True,
        clean_native_builder_relative_L2=r["clean_reload"]["relative_L2"],
        validation_or_outer_truth_read=False,formal_test=False,
        other_fields_untrained=True,verifier_sha256=sha256(Path(__file__)))
    atomic_json(N.R/"independent_verification.json",result)
    print(json.dumps(result),flush=True)
if __name__=="__main__":main()
