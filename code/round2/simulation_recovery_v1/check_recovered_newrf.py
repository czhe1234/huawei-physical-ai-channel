"""Real-coordinate direction and multiport forward correctness; no CSI/fit/score."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from project_paths import PATHS
from round2.paper_baseline_adapters_v1.contract import SimulationInputs, save_json
from .ray_provider import FitGeometryBank
from .qualified_newrf import direction_readiness
from .integration import RecoveredNeWRFAdapter


def check(manifest):
    started = time.monotonic()
    inputs = SimulationInputs()
    role_checks = []
    bank0 = None
    for outer in range(5):
        roles = inputs.roles(outer)
        for role in ("screen_fit", "refit"):
            bank = FitGeometryBank(manifest, inputs, outer=outer, role=role)
            q = direction_readiness(bank)
            if not q["full_canonical_FIT_covered"]:
                raise RuntimeError("Incomplete geometry cannot pass final direction qualification")
            rejected = []
            forbidden = np.setdiff1d(np.arange(4000, dtype=np.int64), roles[role])
            for row in np.unique(forbidden):
                before = len(bank.opened_ids)
                try: bank.get(int(row))
                except PermissionError: pass
                else: raise AssertionError("Out-of-role geometry access was accepted")
                if len(bank.opened_ids) != before: raise AssertionError("Forbidden payload was opened")
                rejected.append(int(row))
            role_checks.append(dict(outer=outer, role=role, FIT_count=len(bank.source_fit_ids), rejected_count=len(rejected),
                                    source_fingerprint=bank.source_fingerprint))
            if outer == 0 and role == "screen_fit": bank0 = bank
    torch.set_num_threads(2)
    torch.manual_seed(2031)
    cfg=dict(n_layers=2,hidden_dim=8,skip=(),position_levels=1,direction_levels=1,
             n_samples=4,n_importance=2,far_m=2.,frequency_chunk=48)
    adapter=RecoveredNeWRFAdapter(inputs,manifest,outer=0,role="screen_fit",config=cfg,max_query_rays=2,max_training_rays=2)
    model=adapter.model.cpu().eval()
    bank0=model.bank
    chosen=[int(next(row for row in bank0.source_fit_ids if bank0.manifest["points"][str(int(row))]["path_count"] > 0))]
    f=inputs.geometry.frequencies_hz[[0,96,191]].copy()
    with torch.no_grad():
        fit,coarse_fit=model.forward_fit(np.array(chosen,np.int64),f,return_coarse=True)
        empty=[int(r) for r in bank0.source_fit_ids if bank0.manifest["points"][str(int(r))]["path_count"] == 0]
        empty_rows=np.array(empty[:1],np.int64)
        empty_h=model.forward_fit(empty_rows,f) if len(empty_rows) else None
        if empty_h is not None and (empty_h.shape != (len(empty_rows),256,4,3) or not bool(torch.isfinite(empty_h).all())):
            raise AssertionError("Empty-path FIT fallback failed")
        before=list(bank0.opened_ids)
        ids=inputs.roles(0)["outer"]
        # Positions and array geometry are public query inputs. No query payload is opened.
        query_rows=np.array([next(int(r) for r in ids if inputs.site_id[r]==s) for s in (0,1)],np.int64)
        query,coarse_query=model(inputs.positions[query_rows].copy(),inputs.site_id[query_rows].copy(),f,return_coarse=True)
    if before != bank0.opened_ids: raise AssertionError("Query opened a direction payload")
    for h,batch in ((fit,1),(coarse_fit,1),(query,2),(coarse_query,2)):
        if h.shape != (batch,256,4,3) or h.dtype != torch.complex64 or not bool(torch.isfinite(h).all()):
            raise AssertionError("Physical NeWRF forward shape/type/finite mismatch")
    full_query=inputs.query(0,"outer",rows=query_rows[:1])
    full_result=adapter.prediction(full_query)
    try: adapter.evaluate_output(full_result,full_query,None)
    except PermissionError: pass
    else: raise AssertionError("Scoring gate was bypassed")
    if bank0.opened_ids != before:raise AssertionError("Full-frequency query opened geometry payload")
    report=dict(status="passed",role_checks=role_checks,training_rows_used_only_for_direction=chosen,
                zero_path_FIT_forward_ids=empty_rows.tolist(),zero_path_FIT_count=len(empty),public_query_position_ids=query_rows.tolist(),FIT_forward_shape=list(fit.shape),query_forward_shape=list(query.shape),
                query_direction_cache_reads=0,read_CSI=False,optimizer_steps=0,trained=False,scored=False,
                full_frequency_adapter_output_shape=list(full_result["channel"].shape),scoring_disabled=True,
                synthetic_bypass=False,frequency_indices=[0,96,191],
                qualification=model.metadata(),elapsed_seconds=time.monotonic()-started,
                limit="Small random-network forward verifies I/O only. It establishes no convergence or score.")
    out=PATHS.analysis/"audits/simulation_recovery_v1/newrf_recovered_check.json"
    save_json(out,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("qualification","role_checks")}),flush=True)
    return report

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("manifest",type=Path)
    check(parser.parse_args().manifest)
