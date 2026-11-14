"""Bounded south-only adapter correctness entry point. No optimizer steps."""
from __future__ import annotations
import json
from pathlib import Path
import time
import unittest
import numpy as np
import torch
from .common import SouthDataset,SOUTH,REPORT,ARTIFACTS,save_json,sha256,check_prediction
from .metrics import directional_power,as_power
from .baselines import RBFPower,PositionPowerMLP,DirectPointing
from .foundation import CarrierBank
from .core import MeasuredV16,batch_from_support,FINAL_SOURCE
from .geometry import NativeRT
from .wrf import WRFGSPlusPower,AUTHOR
from . import test_correctness

def independent_los_check(query):
    objects=json.loads((SOUTH/"model_objects.json").read_text())["objects"]
    result=np.ones((len(query.row_ids),2),bool)
    with np.load(SOUTH/"model_measurement_frame.npz",allow_pickle=False) as z:
        for p,origin in enumerate(query.spec.center):
            for n,position in enumerate(query.position):
                direction=position-origin;distance=np.linalg.norm(direction);direction/=distance
                for obj in objects:
                    if not obj["include_in_future_rf_scene"]:continue
                    tri=z[obj["vertices_key"]][z[obj["faces_key"]]]
                    edge1,edge2=tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]
                    cross=np.cross(direction,edge2);det=np.einsum("ij,ij->i",edge1,cross)
                    usable=np.abs(det)>1e-10
                    inv=np.divide(1.,det,out=np.zeros_like(det),where=usable)
                    delta=origin-tri[:,0];u=np.einsum("ij,ij->i",delta,cross)*inv
                    q=np.cross(delta,edge1);v=q@direction*inv;t=np.einsum("ij,ij->i",edge2,q)*inv
                    blocked=usable&(u>=0)&(v>=0)&(u+v<=1)&(t>.001)&(t<distance-.001)
                    if blocked.any():result[n,p]=False
    return result

def source_identity():
    current={str(p.relative_to(Path(__file__).parents[3])):sha256(p)
             for p in sorted(Path(__file__).parent.glob("*.py"))}
    current["frozen_component_state"]=sha256(FINAL_SOURCE/"component_state/model.py")
    current["frozen_synthesis_reference"]=sha256(FINAL_SOURCE/"synthesis/channel_generator.py")
    current["original_physics_converter"]=sha256(Path(__file__).parents[1]/"dc01_geometry.py")
    return current

def run():
    start=time.monotonic();torch.set_num_threads(1);torch.manual_seed(20261024)
    suite=unittest.defaultTestLoader.loadTestsFromModule(test_correctness)
    unit_start=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(suite)
    save_json(REPORT/"unit_checks_final.json",dict(tests=result.testsRun,errors=len(result.errors),failures=len(result.failures),
              passed=result.wasSuccessful(),seconds=time.monotonic()-unit_start,synthetic_optimizer_steps=0))
    if not result.wasSuccessful():raise RuntimeError("Correctness tests failed")

    data=SouthDataset()
    if sha256(SOUTH/"channels.npy")!=data.identity["channels_sha256"]:raise ValueError("Calibrated south channel identity changed")
    fit=data.partition(0,"fit")
    # Two geometry-only queries, one occluded and one independently unobstructed link.
    query=data.query(fit[:2]);los=independent_los_check(query)
    print("SOUTH_FIT_GEOMETRY_SMOKE",query.row_ids.tolist(),flush=True)
    rt=NativeRT(data.spec);rt_prediction,support=rt.trace(query,samples=4096,max_depth=2)
    retained=support["mask"].sum(-1)
    if not los.any() or not np.all(retained[los]>0):
        raise ValueError("Known unobstructed geometry link produced no physical path")
    np.savez(ARTIFACTS/"two_point_geometry_smoke.npz",row_ids=query.row_ids,**support)
    np.save(ARTIFACTS/"two_point_rt_fit_only.npy",rt_prediction["channel"])

    bank=data.read_fit(0,fit[2:6])
    actual_target=directional_power(bank.channel)
    if actual_target.shape!=(4,2,32) or not np.isfinite(actual_target).all():raise ValueError("Full-frequency fit target")
    carrier=CarrierBank(bank);base=carrier.predict(query)
    core=MeasuredV16().eval()
    core_batch=batch_from_support(base["channel"],support)
    # These query IDs belong to FIT, so this is still a FIT-only input normalizer.
    core.set_fit_statistics(core_batch)
    ours=core.predict(query,base,support)
    before=(np.abs(base["channel"].astype(np.complex128))**2).sum((2,3,4))
    after=(np.abs(ours["channel"].astype(np.complex128))**2).sum((2,3,4))
    energy_difference=float(np.max(np.abs(after-before)/np.maximum(before,1e-30)))
    if energy_difference>3e-6:raise ValueError("Full1024-bin synthesis changed foundation array energy")
    kernel=RBFPower(bank);mlp=PositionPowerMLP(data.meta["position"][fit]).eval()
    with np.load(SOUTH/"model_measurement_frame.npz") as z:
        vertex=np.concatenate([z[o["vertices_key"]] for o in json.loads((SOUTH/"model_objects.json").read_text())["objects"] if o["include_in_future_rf_scene"]])
    chosen=np.linspace(0,len(vertex)-1,16,dtype=int)
    wrf=WRFGSPlusPower(data.meta["position"][fit],data.spec,vertex[chosen],width=360,height=90).eval()
    predictions=[ours,kernel.predict(query),mlp.predict(query),rt_prediction,wrf.predict(query),DirectPointing().predict(query)]
    methods=[]
    for prediction in predictions:
        check_prediction(prediction,query);p=as_power(prediction)
        np.testing.assert_array_equal(prediction["row_ids"],query.row_ids)
        methods.append(dict(method=prediction["method"],output="complex_channel" if "channel" in prediction else "direction_power",
            shape=list(prediction["channel"].shape if "channel" in prediction else prediction["power"].shape),
            common_direction_power_shape=list(p.shape),finite=bool(np.isfinite(p).all()),
            row_ids=prediction["row_ids"].tolist(),zero_power_panels=int((p.sum(-1)==0).sum()),
            real_validation_or_test_metric_computed=False))
    for access in data.channel_reads:
        if access["role"]!="fit":raise ValueError("Unexpected channel read")
    summary=dict(status="correctness_passed_not_formal_experiment",source_identity=source_identity(),
        data_identity=data.identity,unit_tests=result.testsRun,unit_tests_passed=True,
        methods=methods,real_channel_access=data.channel_reads,
        real_channel_rows_loaded=sum(len(x["rows"]) for x in data.channel_reads),
        correctness_query_role="fit",correctness_query_rows=query.row_ids.tolist(),
        known_geometric_LOS=los.tolist(),retained_geometry_paths=retained.tolist(),
        native_array_position_checks=rt.array_check,
        rt_configuration=dict(samples_per_source=4096,max_depth=2,full_spatial_cache=False),
        full_frequency_bins=1024,complex_synthesis_relative_energy_difference_max=energy_difference,
        RBF_closed_form_smoke_fit_rows=4,NN_optimizer_steps=0,synthetic_optimizer_steps=0,
        trained_neural_checkpoints_created=False,real_validation_test_scores_computed=False,
        northern_data_or_channel_loaded=False,northern_predictions_produced=False,
        formal_baseline_health_or_convergence_established=False,
        WRF_scope="Author-based scalar-power CPU reference adaptation; original CUDA kernel and original training reproduction not verified.",
        formal_hyperparameters_or_best_baselines_selected=False,seconds=time.monotonic()-start)
    save_json(REPORT/"correctness_summary.json",summary)
    print(json.dumps(summary),flush=True)
    return summary

if __name__=="__main__":
    try:run()
    except BaseException:
        import traceback
        save_json(REPORT/"failure.json",dict(status="failed",traceback=traceback.format_exc()))
        raise
