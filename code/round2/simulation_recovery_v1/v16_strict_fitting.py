"""Strict carrier ridge refit using frozen V7 solver and certified features.

No old OOF cross-fold feature bank is implicitly accepted. The feature producer
must regenerate its fitting rows under the same outer/role label pool first.
"""
from __future__ import annotations

import json
from pathlib import Path
import types

import numpy as np

from v16_recovery import (
    PATHS, FOLDS, strict_roles, FitOnlyLabels, source_namespace, sha256,
    file_identity, protected_output, atomic_json, SOURCE_HASHES,
)


def check_feature_provenance(record: dict,outer: int,role: str):
    allowed=strict_roles(outer)[role]
    if record.get("schema")!="v16-strict-carrier-features-v1":
        raise ValueError("Missing strict carrier feature producer provenance")
    if record.get("outer")!=outer or record.get("role")!=role or record.get("protocol_sha256")!=sha256(FOLDS):
        raise ValueError("Carrier feature protocol identity differs")
    if record.get("allowed_label_ids")!=allowed.tolist():
        raise ValueError("Feature producer label pool differs")
    if not record.get("label_sensitive_dependencies"):
        raise ValueError("Transitive feature lineage is required")
    for component,item in record["label_sensitive_dependencies"].items():
        if "label_ids" not in item or not set(item["label_ids"]).issubset(set(allowed.tolist())):
            raise ValueError(f"Forbidden transitive labels in {component}")
    # A checksum of an old OOF bank is not evidence that its other rows were
    # generated without this outer's labels.
    if record.get("cross_fold_old_oof_reuse",False):
        raise ValueError("Old cross-fold OOF features cannot supply strict training features")
    return allowed


def fit_port_stack(outer: int,role: str,feature_record: dict,provider,output_dir: Path,
                   *,ridge: float,sum_penalty: float):
    """provider(ids)->[C,len(ids),256,4,192], all features role-certified above.

    Reuses the exact frozen port8/site ridge solver. Ridge and penalty are explicit
    registered configuration; this function does not select them on held labels.
    """
    if role not in {"screen_fit","refit"}:raise ValueError("Fit role required")
    allowed=check_feature_provenance(feature_record,outer,role)
    if ridge<0 or sum_penalty<0:raise ValueError("Nonnegative ridge/penalty required")
    output_dir=protected_output(output_dir)
    output_dir.mkdir(parents=True,exist_ok=False)
    labels=FitOnlyLabels(allowed)
    # Official site metadata only.
    import sys
    sys.path.insert(0,str(PATHS.round2_code))
    from round2.data import load_positions
    _,_,_,site,_=load_positions()
    gram=[];cross=[]
    carriers=None
    for q in allowed:
        x=np.asarray(provider(np.array([q],np.int64)),np.complex64)
        if x.ndim!=5 or x.shape[1:]!=(1,256,4,192) or not np.isfinite(x).all():
            raise ValueError("Certified feature provider output invalid")
        if carriers is None:carriers=x.shape[0]
        elif carriers!=x.shape[0]:raise ValueError("Carrier count changed")
        y=labels.read(np.array([q],np.int64))[0]
        x=x[:,0].reshape(carriers,2,16,8,4,192).transpose(1,4,0,2,3,5).reshape(8,carriers,-1)
        y=y.reshape(2,16,8,4,192).transpose(0,3,1,2,4).reshape(8,-1)
        gram.append(np.einsum("gcd,ged->gce",x.conj(),x,optimize=True).astype(np.complex64))
        cross.append(np.einsum("gcd,gd->gc",x.conj(),y,optimize=True).astype(np.complex64))
    solver=source_namespace("analyze_v7_complex_carrier_stacking.py",["fit_weights"],
        dict(),())
    cfg=types.SimpleNamespace(scope="site",ports="port8",ridge=float(ridge),sum_penalty=float(sum_penalty))
    cache=dict(gram=np.asarray(gram),cross=np.asarray(cross))
    weight=solver.fit_weights(cache,np.arange(len(allowed)),site[allowed],cfg)
    path=output_dir/"port_stack_weights.npz"
    if path.exists():raise FileExistsError(path)
    np.savez_compressed(path,**{f"site{k}":v for k,v in weight.items()},fit_ids=allowed,
        ridge=np.asarray(ridge),sum_penalty=np.asarray(sum_penalty))
    record=dict(schema="v16-strict-component-v1",component="carrier_stack",outer=outer,role=role,
        protocol_sha256=sha256(FOLDS),fit_ids=allowed.tolist(),
        label_ids=dict(training=allowed.tolist(),normalization=[],basis=[],calibration=[],policy=[]),
        artifact=file_identity(path),feature_provenance=feature_record,sources=dict(SOURCE_HASHES),
        label_access=labels.access,closed_form_fit=True,neural_optimizer_steps=0,
        performance_evaluated=False)
    atomic_json(output_dir/"component.json",record)
    return record
