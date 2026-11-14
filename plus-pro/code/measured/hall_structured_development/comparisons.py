"""Recompute comparable baseline scores on the unchanged development rows."""
import json,time,traceback
import numpy as np
import torch
from .common import *
from .baselines import SpectralRBF
from measured.hall_structured_v2.foundation import Foundation
from measured.hall_structured_v2.observables import metrics
from measured.hall_adapters.metrics import directional_power,primary_metrics,aggregate_primary
def run():
    init();seed();start=time.monotonic();b=bundle(False)
    power=b["power"];truth=torch.tensor(b["yv"]);rows=b["val"].row_ids
    old=json.loads((PATHS.analysis/"results/hall_south_development_v1/contract.json").read_text())
    assert old["data_identity"]==b["reader"].data.identity and old["validation_rows"]==rows.tolist()
    results={}
    def store(name,pred,extra=None):
        assert tuple(pred.shape)==tuple(truth.shape)
        result=metrics(pred,truth,power)
        results[name]=dict(metrics=result,scope="full-channel structure",details=extra or {})
        print("BASELINE",name,result["structural_selection"],result["direction_cosine"],flush=True)
        return result
    store("nearest_FIT",b["val"].carrier,dict(carrier="nearest permitted FIT channel; no query truth alignment"))
    foundation=Foundation()
    with torch.no_grad():pred,info=foundation(b["val"],power.reference)
    store("uniform_frequency_experts",pred)
    grid=[];best=float("inf");best_model=None
    for length in [.5,1.,2.,4.]:
      for ridge in [.001,.01,.1]:
        model=SpectralRBF(b["bank"],length,ridge);pred=model.predict(b["val"]);m=metrics(pred,truth,power)
        row=dict(length_m=length,ridge=ridge,metrics=m);grid.append(row)
        if m["structural_selection"]<best:
            best=m["structural_selection"];best_model=model;best_row=row;best_pred=pred
        print("KERNEL",length,ridge,round(m["structural_selection"],6),flush=True)
    results["frequency_RBF"]=dict(metrics=best_row["metrics"],scope="full-channel structure",details=dict(length_m=best_row["length_m"],ridge=best_row["ridge"],candidates=12))
    folder=WORK/"baselines";folder.mkdir(parents=True,exist_ok=True)
    atomic_npz(folder/"rbf_best_channel.npz",fit=best_model.predict(b["fit"]).numpy(),validation=best_pred.numpy(),fit_ids=b["fit"].row_ids,validation_ids=rows)
    save_json(REPORT/"kernel_grid.json",dict(candidates=grid,selection="same final-H structural criterion",best=best_row))
    old_root=PATHS.artifacts/"measured/hall_sw_development_v1/models"
    p=old_root/"ours_phase_safe/best_channel.npz"
    with np.load(p,allow_pickle=False) as z:
        pred=torch.tensor(z["validation"])
    store("previous_angular_ours",pred,dict(prediction_file=str(p.relative_to(PATHS.root)),sha256=sha256(p),selection="previous angular-only validation; diagnostic historical comparison"))
    actual_power=directional_power(b["yv"])
    for name in ["wrf_g64","wrf_g128","mlp_levels3","mlp_levels6"]:
        p=old_root/name/"best_power.npz"
        with np.load(p,allow_pickle=False) as z:pred=np.array(z["validation"])
        m=aggregate_primary(primary_metrics(dict(power=pred),actual_power))
        results[name]=dict(metrics=m,scope="direction and beam only",details=dict(prediction_sha256=sha256(p),original_selection="previous angular-only validation",full_complex_metrics_available=False))
        print("POWER_BASELINE",name,m["mean_direction_cosine"],m["mean_beam_loss_db"],flush=True)
    save_json(REPORT/"baseline_comparison.json",dict(complete=True,results=results,seconds=time.monotonic()-start,
        data_identity=b["reader"].data.identity,fit_rows=b["fit"].row_ids.tolist(),validation_rows=rows.tolist(),test=False,north=False))
if __name__=="__main__":
    try:run()
    except BaseException:
        save_json(REPORT/"baseline_comparison_failure.json",dict(traceback=traceback.format_exc()));raise
