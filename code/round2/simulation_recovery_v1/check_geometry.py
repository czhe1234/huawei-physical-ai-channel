"""Independent FIT-only and geometry algebra checks, with no model fit/score."""
from __future__ import annotations
import argparse
import json
import time
import numpy as np
import torch
from round2.paper_baseline_adapters_v1.contract import SimulationInputs,save_json
from .geometry import OUTPUT,REPORT
from .ray_provider import FitGeometryBank,TrainingGeometryRayProvider,FitVirtualSourceRayProvider,reflection_transform

def check(manifest=None):
    started=time.monotonic();torch.set_num_threads(2)
    manifest=manifest or OUTPUT/'smoke_outer0_screen_fit/manifest.json'
    inputs=SimulationInputs();bank=FitGeometryBank(manifest,inputs,outer=0,role='screen_fit')
    if not bank.manifest['complete']:raise ValueError("Incomplete geometry cache")
    if len(bank.source_fit_ids)>4:raise ValueError("This bounded smoke check intentionally reads at most4 FIT points")
    held=int(inputs.roles(0)['outer'][0]);opened=list(bank.opened_ids)
    try:bank.get(held)
    except PermissionError:pass
    else:raise AssertionError("Outer direction payload was exposed")
    assert bank.opened_ids==opened
    training=TrainingGeometryRayProvider(bank)
    bundles=training.for_rows(bank.source_fit_ids,inputs.geometry.bs_port_positions_m)
    maximum_error=0.;counts=[]
    for bundle in bundles:
        row=bundle['point_id'];point=bank.get(row)
        theta,phi=point['theta_r'],point['phi_r']
        unit=np.column_stack((np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)))
        maximum_error=max(maximum_error,float(np.max(np.arccos(np.clip((unit*point['direction_center']).sum(-1),-1,1)))) if len(unit) else 0.)
        direction=bundle['directions']
        assert direction.shape==(len(theta),256,4,3)
        if len(direction):np.testing.assert_allclose(np.linalg.norm(direction,axis=-1),1.,atol=1e-10)
        assert point['frequencies_hz'].dtype==np.float64 and np.array_equal(point['frequencies_hz'],inputs.geometry.frequencies_hz)
        counts.append(dict(point_id=row,site=bundle['site_id'],path_count=len(theta),non_LoS=int((point['depth']>0).sum())))
    # Independently specified perfect mirror: the source's image is known before
    # invoking reflection_transform, and transfers to a new query position.
    bs=np.array([-2.,1.,2.]);ue=np.array([2.,1.,1.]);vertex=np.array([[0.,0.,1.5]])
    transform=reflection_transform(vertex,np.array([1]),bs,ue)
    assert transform is not None
    a,b,_=transform
    np.testing.assert_allclose(a,np.diag([1.,-1.,1.]),atol=1e-12)
    np.testing.assert_allclose(b,0.,atol=1e-12)
    query=np.array([3.,2.,1.]);image=np.array([-2.,-1.,2.])
    np.testing.assert_allclose((a@bs+b-query)/np.linalg.norm(a@bs+b-query),(image-query)/np.linalg.norm(image-query),atol=1e-12)
    assert reflection_transform(vertex,np.array([4]),bs,ue) is None
    prior=FitVirtualSourceRayProvider(bank,max_rays=64)
    # A subsequent legal FIT read must not be mistaken for a query cache read.
    bank.get(int(bank.source_fit_ids[0]))
    query_ids=np.array([inputs.roles(0)['outer'][0],inputs.roles(0)['outer'][-1]],np.int64)
    pos=torch.tensor(inputs.positions[query_ids].copy(),dtype=torch.float64)
    sites=torch.tensor(inputs.site_id[query_ids].copy(),dtype=torch.int64)
    origins=inputs.geometry.ue_world_positions(pos,sites)
    tx=torch.tensor(inputs.geometry.bs_port_positions_m.copy(),dtype=torch.float64)[sites].reshape(len(sites),2,128,3)
    opened_before=list(bank.opened_ids)
    original_get=bank.get
    def forbid(*args,**kwargs):raise AssertionError("Query tried to fetch a geometry point")
    bank.get=forbid
    try:directions=prior.directions(origins,sites,tx)
    finally:bank.get=original_get
    assert directions.shape==(2,2,128,4,64,3) and bool(torch.isfinite(directions).all())
    torch.testing.assert_close(torch.linalg.vector_norm(directions,dim=-1),torch.ones(directions.shape[:-1],dtype=torch.float64),atol=1e-10,rtol=1e-10)
    assert bank.opened_ids==opened_before and set(bank.opened_ids).issubset(set(bank.source_fit_ids))
    assert not np.intersect1d(bank.source_fit_ids,query_ids).size
    result=dict(status='prototype_geometry_provider_checks_passed',geometry_points=counts,
      maximum_angle_last_interaction_error_rad=maximum_error,
      perfect_mirror_affine_identity=True,unsupported_refraction_not_mislabeled=True,
      outer_geometry_payload_access_rejected=True,query_direction_cache_reads=0,
      query_source_fit_ids=bank.source_fit_ids.tolist(),query_position_ids=query_ids.tolist(),
      queried_only_position_metadata=True,full_FIT_training_ready=False,
      training_provider=training.metadata(),query_provider=prior.metadata(),
      no_CSI=True,training=False,score=False,wall_seconds=time.monotonic()-started)
    save_json(REPORT/'smoke_provider_check.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--manifest');args=parser.parse_args();check(args.manifest)
