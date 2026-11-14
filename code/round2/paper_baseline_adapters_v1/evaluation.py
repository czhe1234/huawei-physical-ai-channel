"""Frozen official simulation operators; no measured relative-clock correction."""
from __future__ import annotations
import numpy as np
from common.official_metrics import (
    ArrayLayout,pas_power,pdp_power,official_cosine,point_metric_statistics,aggregate_statistics)
from .contract import DFT_CONTRACT_ID,validate_output

METRICS=dict(pas=dict(direction="higher",operator="unshifted FFT2 over BS H16,V8; keep both polarizations in256-vector"),
 pdp=dict(direction="higher",operator="unmodified H IFFT over192tones; no delay shift/reference"),
 nmse=dict(direction="lower",operator="global summed complex error energy / global truth energy"),
 score=dict(direction="higher",operator="0.4PAS+0.4PDP+0.2/(1+globalNMSE)"))

def eligibility(method):
    if method in {"nerf2","ngrf","v16"}:
        return dict(output="channel",full_H=True,pas=True,pdp=True,nmse=True,score=True,
                    performance_validated=False)
    if method=="wrfgs":
        return dict(output="power",full_H=False,pas=True,pdp=False,nmse=False,score=False,
                    contract_id=DFT_CONTRACT_ID,performance_validated=False,
                    qualification="explicit codebook-power readout; not native spectrum/CSI reproduction")
    if method=="newrf":
        return dict(output="channel",full_H="conditional",pas="conditional",pdp="conditional",
                    nmse="conditional",score="conditional",performance_validated=False,
                    qualification="requires canonicalFIT direction source and training-only query ray search")
    raise ValueError("Unknown method")

def pas_from_power(predicted_power,truth_channel):
    p=np.asarray(predicted_power)
    t=pas_power(np.asarray(truth_channel),ArrayLayout())
    if p.shape!=t.shape or p.shape[1:]!=(256,4,192) or np.iscomplexobj(p) or not np.isfinite(p).all() or np.any(p<0):
        raise ValueError("Actual nonnegative official-codebook powers required")
    return official_cosine(p,t,1).mean(axis=(1,2),dtype=np.float64)

def evaluate_output(result,query,truth,*,explicit_evaluation_enabled=False):
    # This API is wired but never enabled by the correctness entrypoint.
    if not explicit_evaluation_enabled:
        raise PermissionError("Model scoring is disabled in this stage")
    validate_output(result,query)
    reject_online_test_as_oof(prediction_namespace=result["prediction_namespace"],
                              truth_namespace=query.input_namespace)
    if "channel" in result:
        return aggregate_statistics(point_metric_statistics(result["channel"],truth))
    values=pas_from_power(result["power"],truth)
    return dict(count=len(values),pas=float(values.mean()),pdp=None,nmse=None,score=None,
                unsupported_reason="Power output has no complex frequency phase; no fabricated H")

def reject_online_test_as_oof(*,prediction_namespace,truth_namespace):
    if prediction_namespace!=truth_namespace:
        raise ValueError("Online-Test position order cannot be paired with canonical training-row truth")
    if truth_namespace!="canonical_train_row_ids":
        raise ValueError("This comparison contract is canonical spatial holdout only")
