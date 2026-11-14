"""Canonical-FIT geometric direction recovery, using the unchanged V16 engine.

No CSI, learned model, score or held-out direction is read. A pointwise cache
lets FIT providers avoid even opening held-out geometry payloads. Trace output
is an approximate fixed V16 physical prior, not publisher multipath truth.
"""
from __future__ import annotations
import argparse
import ctypes
from dataclasses import asdict,dataclass
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback
import numpy as np
from project_paths import PATHS
from round2.paper_baseline_adapters_v1.contract import SimulationInputs,sha256,save_json

GPU_UUID="GPU-6b53c3c2-3071-5dfb-f306-2f2f142f8009"
VENDOR=PATHS.root/"final_code/V16_reorganized_submission/code/round2/physical_prior"
OUTPUT=PATHS.artifacts/"round2/simulation_recovery_v1/geometry"
REPORT=PATHS.analysis/"audits/simulation_recovery_v1/geometry"
FRAME="official Round2 map/TrainPos world XYZ metres; global Z up"

@dataclass(frozen=True)
class TraceConfig:
    seed:int=42
    max_depth:int=16
    samples_per_src:int=1000000
    max_components_per_source:int=1000000
    diffuse_reflection:bool=False
    scattering_coefficient:float=0.
    batch_size:int=16
    def __post_init__(self):
        if self.seed!=42 or self.max_depth!=16 or self.diffuse_reflection or self.scattering_coefficient!=0.:
            raise ValueError("This stage freezes V16 base depth16 seed42, no diffuse component")
        if min(self.samples_per_src,self.max_components_per_source,self.batch_size)<1:raise ValueError("Positive trace budgets required")

def atomic_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.npz.partial')
    with tmp.open('wb') as stream:np.savez_compressed(stream,**arrays)
    tmp.replace(path)

def vendor_identity():
    return {PATHS.relative(p):sha256(p) for p in sorted(VENDOR.rglob('*.py'))}

def _load_engine():
    if os.environ.get("CUDA_VISIBLE_DEVICES")!=GPU_UUID:raise RuntimeError("Only the allocated full GPU UUID may be visible")
    import torch
    if torch.cuda.device_count()!=1:raise RuntimeError("Expected exactly one visible CUDA device")
    torch.cuda.set_device(0)
    import mitsuba as mi
    import drjit as dr
    if mi.variant() is None:mi.set_variant("cuda_ad_mono_polarized")
    if mi.variant()!="cuda_ad_mono_polarized":raise RuntimeError("Unexpected Mitsuba execution variant")
    lib=ctypes.CDLL(str(Path(dr.__file__).with_name('libdrjit-core.so')))
    select=getattr(lib,'_Z19jit_cuda_set_devicei');select.argtypes=[ctypes.c_int];select.restype=None;select(0)
    # The vendored engine has absolute package imports; bind that original package
    # only in this isolated trace process, never rewrite any vendored source.
    name='round2.physical_prior'
    if name in sys.modules and Path(sys.modules[name].__file__).resolve()!=(VENDOR/'__init__.py').resolve():
        raise RuntimeError("A different physical-prior runtime was already imported")
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,VENDOR/'__init__.py',submodule_search_locations=[str(VENDOR)])
        mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    engine=importlib.import_module(name+'.geometry_prior_engine')
    data=importlib.import_module(name+'.scene_prior_data')
    return engine,data,mi,dr

def extract_point_geometry(components,local,point_id,site,position,ue_ports,bs,seed,max_depth):
    valid=np.asarray(components.valid.numpy(),bool)
    ids=np.flatnonzero(valid[local,0])
    theta=np.asarray(components.theta_r.numpy(),np.float64)[local,0,ids]
    phi=np.asarray(components.phi_r.numpy(),np.float64)[local,0,ids]
    tau=np.asarray(components.tau.numpy(),np.float64)[local,0,ids]
    interactions=np.take(np.asarray(components.interactions.numpy(),np.uint32)[:,local,0,:],ids,axis=1).T
    vertices=np.take(np.asarray(components.vertices.numpy(),np.float64)[:,local,0,:,:],ids,axis=1).transpose(1,0,2)
    depth=np.count_nonzero(interactions,axis=1).astype(np.int64)
    if interactions.shape!=(len(ids),max_depth) or vertices.shape!=(len(ids),max_depth,3):raise ValueError("Vendored path geometry layout changed")
    last=np.broadcast_to(bs,(len(ids),3)).copy()
    for j,count in enumerate(depth):
        if count:
            active=np.flatnonzero(interactions[j])
            if not np.array_equal(active,np.arange(count)):raise ValueError("Noncontiguous interaction sequence")
            if not np.isfinite(vertices[j,:count]).all():raise ValueError("Nonfinite active interaction")
            last[j]=vertices[j,count-1]
    vertices=np.where(np.isfinite(vertices),vertices,0.)
    vector=last-position[None]
    norm=np.linalg.norm(vector,axis=-1)
    if len(ids) and (not np.isfinite(theta).all() or not np.isfinite(phi).all() or np.any(norm<1e-7)):
        raise ValueError("Invalid nonempty directional geometry")
    direction=vector/norm[:,None] if len(ids) else np.empty((0,3),np.float64)
    angular=np.column_stack((np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)))
    error=np.arccos(np.clip(np.sum(angular*direction,axis=-1),-1,1))
    if len(error) and float(error.max())>np.deg2rad(.1):raise ValueError("Arrival-angle/last-interaction convention mismatch")
    port_vector=last[:,None]-ue_ports[None]
    port_norm=np.linalg.norm(port_vector,axis=-1)
    if len(ids) and np.any(port_norm<1e-7):raise ValueError("An interaction lies on a UE port")
    fields=dict(point_id=np.array(point_id,np.int64),site_id=np.array(site,np.int64),
      ue_position_m=np.asarray(position,np.float64),ue_port_positions_m=np.asarray(ue_ports,np.float64),
      bs_position_m=np.asarray(bs,np.float64),theta_r=theta,phi_r=phi,tau_seconds=tau,
      interaction_type=interactions,vertices_m=vertices,depth=depth,last_interaction_m=last,
      direction_center=direction,direction_ue_ports=port_vector/port_norm[...,None],
      trace_seed=np.array(seed,np.int64),source_code=np.array(0,np.int64),
      frequencies_hz=np.array([],np.float64))
    return fields,dict(path_count=len(ids),maximum_angle_geometry_error_rad=float(error.max()) if len(error) else 0.,
      LoS_count=int((depth==0).sum()),non_LoS_count=int((depth>0).sum()))

def select_rows(inputs,mode,outer,role):
    if role=='all_train':allowed=np.arange(4000,dtype=np.int64)
    elif role in ('screen_fit','refit'):allowed=inputs.roles(outer)[role]
    else:raise ValueError("Only canonical FIT/all-training geometric generation is enabled")
    if mode=='full':return allowed
    picked=[]
    for site in (0,1):
        rows=allowed[inputs.site_id[allowed]==site]
        ordered=rows[np.argsort(inputs.positions[rows,1],kind='stable')]
        picked.extend(ordered[[len(ordered)//4,3*len(ordered)//4]].tolist())
    return np.array(sorted(set(picked)),np.int64)

def recover(mode='smoke',outer=0,role='screen_fit',config=None,observer=None):
    cfg=config or TraceConfig();inputs=SimulationInputs()
    rows=select_rows(inputs,mode,outer,role)
    run_name=f'{mode}_outer{outer}_{role}'
    out=OUTPUT/run_name;report_dir=REPORT/run_name
    out.mkdir(parents=True,exist_ok=True);report_dir.mkdir(parents=True,exist_ok=True)
    contract=dict(schema='round2-fit-geometry-v1',input_identity=inputs.identity,
      geometry_fingerprint=inputs.geometry.geometry_fingerprint,source_files=vendor_identity(),
      official_map_sha256=sha256(PATHS.round2/'Round2_Map.ply'),trace_config=asdict(cfg),
      generated_role=role,outer=outer,point_ids=rows.tolist(),coordinate_frame=FRAME,
      frequency_provenance=inputs.geometry.metadata(),geometry_is_approximate=True,
      direction_convention='UE origin toward last interaction; LoS toward BS; theta inclination,phi azimuth world frame',
      no_CSI=True,no_official_Test=True,query_geometry_allowed_for_direction_provider=False,
      statistics_observer=type(observer).__name__ if observer is not None else None)
    key=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    manifest_path=out/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest['contract_hash']!=key:raise ValueError("Existing geometry run has another contract; preserve it")
    else:
        manifest=dict(contract_hash=key,contract=contract,complete=False,points={},batches=[])
        save_json(manifest_path,manifest)
    started=time.monotonic()
    engine,data,mi,dr=_load_engine()
    import torch
    xml=data.prepare_scene_files(PATHS.round2,out/'scene')
    for site in (0,1):
        selected=rows[inputs.site_id[rows]==site]
        scene=engine.load_scene(str(xml));scene.frequency=inputs.geometry.frequency_center_hz
        scene.tx_array=engine.build_transmit_array(4.,.5);scene.rx_array=engine.build_receive_array()
        data.configure_scene_response(scene,0.)
        bs=inputs.geometry.bs_positions_m[site]
        source=engine.Source(name=f'recovery_bs{site}',position=mi.Point3f(*bs.tolist()),
          look_at=data.tx_target(bs,data.SITE_YAW_DEG[site],data.SITE_TILT_DEG[site]))
        scene.add(source);builder=engine.ScenePriorBuilder()
        for start in range(0,len(selected),cfg.batch_size):
            batch=selected[start:start+cfg.batch_size]
            ready=True
            for row in batch:
                entry=manifest['points'].get(str(int(row)))
                if entry is None or sha256(out/entry['file'])!=entry['sha256']:ready=False;break
            if ready:continue
            for row in batch:
                scene.add(engine.Target(name=f'recovery_fit_{row}',position=mi.Point3f(*inputs.positions[row].tolist()),look_at=source))
            trace_seed=cfg.seed+100000*site+int(batch[0])
            tick=time.monotonic()
            components=builder(scene,max_depth=cfg.max_depth,max_components_per_source=cfg.max_components_per_source,
                samples_per_src=cfg.samples_per_src,diffuse_reflection=False,seed=trace_seed)
            ports=inputs.geometry.ue_world_positions(torch.tensor(np.array(inputs.positions[batch]),dtype=torch.float64),
                torch.tensor(np.array(inputs.site_id[batch]),dtype=torch.int64)).numpy()
            batch_info=[]
            for local,row in enumerate(batch):
                arrays,diagnostic=extract_point_geometry(components,local,int(row),site,inputs.positions[row],ports[local],bs,trace_seed,cfg.max_depth)
                arrays['frequencies_hz']=np.array(inputs.geometry.frequencies_hz,np.float64)
                relative=f'points/point_{row:05d}.npz';point_file=out/relative
                previous=manifest['points'].get(str(int(row)))
                if previous is not None and point_file.exists() and sha256(point_file)==previous['sha256']:
                    batch_info.append(dict(point_id=int(row),reused=True));continue
                if point_file.exists():raise FileExistsError("Refuse to overwrite an unregistered point payload")
                atomic_npz(point_file,**arrays)
                manifest['points'][str(int(row))]=dict(file=relative,sha256=sha256(point_file),site=site,**diagnostic)
                batch_info.append(dict(point_id=int(row),**diagnostic))
                save_json(manifest_path,manifest)
            # Stream hook: downstream PAS/PDP/sparse-state accumulators can consume
            # the live components without writing huge complex-coefficient H5 files.
            if observer is not None:
                observer.update(components=components,point_ids=np.array(batch),site=site,trace_seed=trace_seed,
                  positions_m=inputs.positions[batch],frequencies_hz=np.array(inputs.geometry.frequencies_hz),source_config=asdict(cfg))
            dr.sync_thread();del components
            for row in batch:scene.remove(f'recovery_fit_{row}')
            timing=time.monotonic()-tick
            manifest['batches'].append(dict(site=site,point_ids=batch.tolist(),trace_seed=trace_seed,seconds=timing,points=batch_info))
            save_json(manifest_path,manifest)
            progress=dict(stage='geometry',run=run_name,done=len(manifest['points']),total=len(rows),batch_seconds=timing,
              elapsed_seconds=time.monotonic()-started,allocated_GPU_UUID=GPU_UUID,logical_device=0)
            save_json(report_dir/'progress.json',progress);print(json.dumps(progress),flush=True)
        del scene,builder;dr.sync_thread()
    manifest['complete']=len(manifest['points'])==len(rows);save_json(manifest_path,manifest)
    result=dict(status='geometry_complete' if manifest['complete'] else 'incomplete',run=run_name,
      manifest=PATHS.relative(manifest_path),contract_hash=key,point_count=len(rows),
      total_paths=sum(v['path_count'] for v in manifest['points'].values()),
      maximum_angle_geometry_error_rad=max(v['maximum_angle_geometry_error_rad'] for v in manifest['points'].values()),
      elapsed_seconds=time.monotonic()-started,peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
      GPU_UUID=GPU_UUID,logical_device=0,train=False,score=False,read_CSI=False,
      state='approximate_geometry_source; no NeWRF-training or query-direction qualification yet')
    save_json(report_dir/'result.json',result);print(json.dumps(result),flush=True)
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['smoke','full'],default='smoke')
    parser.add_argument('--outer',type=int,default=0);parser.add_argument('--role',choices=['screen_fit','refit','all_train'],default='screen_fit')
    args=parser.parse_args()
    try:recover(args.mode,args.outer,args.role)
    except BaseException:
        REPORT.mkdir(parents=True,exist_ok=True)
        (REPORT/'last_failure.txt').write_text(traceback.format_exc())
        raise
if __name__=='__main__':main()
