"""Cheap, explicit native-complex validation selector; never a PAS/PDP score."""
from __future__ import annotations
import numpy as np
import torch
from .common import digest_ids

BS_IDS=np.concatenate([np.linspace(0,127,8,dtype=np.int64),
                       128+np.linspace(0,127,8,dtype=np.int64)])
UE_IDS=np.arange(4,dtype=np.int64)
FREQUENCY_IDS=np.linspace(0,191,12,dtype=np.int64)


def probe_contract(data):
    return dict(kind="fixed-validation-complex-subtensor-NMSE-v1",
      position_ids=data.probe_ids.tolist(),position_ids_sha256=digest_ids(data.probe_ids),
      BS_ids=BS_IDS.tolist(),UE_ids=UE_IDS.tolist(),frequency_ids=FREQUENCY_IDS.tolist(),
      shape=[len(data.probe_ids),len(BS_IDS),len(UE_IDS),len(FREQUENCY_IDS)],
      source="same public-position16probe as full validation; labels read only for selection",
      not_official_full_band_score=True,not_PAS=True,not_PDP=True)


def evaluate_complex_probe(data,predict_subset_fn,*,batch_size=1):
    """predict_subset_fn(ids, BS_IDS, UE_IDS, FREQUENCY_IDS) -> NORMALIZED H.

    Only a native complex-error proxy is returned. The complete-band official
    table is evaluated separately after frozen refit and complete500 prediction.
    """
    if data.role!="screen":raise PermissionError("Refit may not select on validation")
    rows=np.asarray(data.probe_ids,np.int64)
    if not len(rows) or not set(map(int,rows)).issubset(set(map(int,data.val_ids))):
        raise PermissionError("Validation probe role mismatch")
    error=energy=0.
    with torch.no_grad():
        for start in range(0,len(rows),batch_size):
            ids=rows[start:start+batch_size]
            pred=predict_subset_fn(ids,BS_IDS,UE_IDS,FREQUENCY_IDS)
            pred=pred.detach().cpu().numpy() if isinstance(pred,torch.Tensor) else np.asarray(pred)
            want=(len(ids),len(BS_IDS),len(UE_IDS),len(FREQUENCY_IDS))
            if pred.shape!=want or not np.iscomplexobj(pred) or not np.isfinite(pred).all():
                raise ValueError("Validation subtensor output shape/type/finite")
            target=data.read_validation(ids,freq_ids=FREQUENCY_IDS,normalized=True)
            target=np.take(np.take(target,BS_IDS,axis=1),UE_IDS,axis=2)
            pred=pred.astype(np.complex64)
            error+=float(np.sum(np.abs(pred-target)**2,dtype=np.float64))
            energy+=float(np.sum(np.abs(target)**2,dtype=np.float64))
    value=error/max(energy,1e-30)
    return dict(selection_metric="validation_subtensor_nmse",selection_value=value,
      nmse_proxy=value,lower_is_better=True,count=len(rows),scalar_complex_count=int(len(rows)*16*4*12),
      squared_error=error,truth_energy=energy,contract=probe_contract(data))
