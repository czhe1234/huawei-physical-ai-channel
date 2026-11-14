"""Fixed V10 checkpoint: two FIT locations, field0 BS30, actual training/query routes."""
from __future__ import annotations
import argparse,json,math,signal,time,traceback
from pathlib import Path
import numpy as np
import torch
from .common import ART,REPORT,FoldData,device_guard,sha256,atomic_json,atomic_npy
from .newrf_runner import GEOMETRY,FitGeometryBank,subset_forward
from .newrf_native_batch_v4 import build_native_batch_model
from .newrf_runtime_cache_v4 import MemoizedFitDirections
from .newrf_query import subset_query
from .newrf_balanced_dev_v1 import metrics
from . import newrf_minimal_fullH_v2 as unit
from project_paths import PATHS
A=ART/"fit_repair_paper_v11_query_route_20261101/newrf"
R=REPORT/"fit_repair_paper_v11_query_route_20261101/newrf"
PARENT=ART/"fit_repair_paper_v10_coarse_init_signal_warm_20261101/newrf"
ROWS=np.array([2666,2042],np.int64);BS=np.array([30],np.int64);FREQ=np.arange(192,dtype=np.int64)
def alarm(sig,frame):raise TimeoutError("180s entire fixed-model query-route check")
def load_inputs():
    d=FoldData(0,"screen","newrf")
    if not set(map(int,ROWS)).issubset(set(map(int,d.fit_ids))) or not np.all(d.sites(ROWS)==0):raise ValueError("Only authorized FIT rows/site0")
    raw=d.read_fit(ROWS,normalized=False);target=raw[:,BS,0,:]
    frozen=np.load(PARENT/"actual_target_field.npy")[:,BS,:]
    if not np.array_equal(target,frozen):raise RuntimeError("V10 raw target identity")
    return d,target
def preflight():
    started=time.time();A.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
    if (R/"preflight.json").exists():raise FileExistsError("Preserve prior preflight")
    d,y=load_inputs();ck=torch.load(PARENT/"final.pt",map_location="cpu",weights_only=False);unit.source_check(ck["source_identity"])
    if ck["stage"]!="unfrozen" or ck["row_ids"]!=ROWS.tolist() or ck["fine_density_backward_detached"]:raise RuntimeError("V10 final native release required")
    source=dict(ck["source_identity"])
    for p in [Path(__file__),PARENT/"final.pt",Path(__file__).with_name("newrf_query.py"),PATHS.root/"code/round2/simulation_recovery_v1/ray_provider.py",PATHS.root/"code/round2/simulation_recovery_v1/qualified_newrf.py",Path(GEOMETRY)]:
        source[PATHS.relative(p)]=sha256(p)
    atomic_npy(A/"raw_target_BS30.npy",y);atomic_npy(A/"row_ids.npy",ROWS)
    atomic_json(R/"axes.json",dict(row_ids=ROWS.tolist(),positions_m=d.positions(ROWS).tolist(),site_ids=d.sites(ROWS).tolist(),field=0,BS_ids=BS.tolist(),UE_ids=[0],frequency_ids=FREQ.tolist(),physical_frequencies_hz=d.frequencies(FREQ).tolist(),raw_H=True))
    result=dict(status="CPU_preflight_passed",preflight_start_unix=started,elapsed_seconds=time.time()-started,source_identity=source,
        row_ids=ROWS.tolist(),field=0,BS_ids=BS.tolist(),UE_ids=[0],frequency_count=192,target_shape=list(y.shape),target_dtype=str(y.dtype),
        raw_target_bitwise_same_V10=True,raw_target_sha256=sha256(A/"raw_target_BS30.npy"),FIT_RMS=d.scale,
        checkpoint_sha256=sha256(PARENT/"final.pt"),CUDA_initialized=torch.cuda.is_initialized(),query_or_validation_truth_read=False,
        no_alignment_or_correction=True,optimizer_updates=0)
    if result["CUDA_initialized"]:raise RuntimeError("CPU preflight unexpectedly initializes CUDA")
    atomic_json(R/"preflight.json",result);print(json.dumps(result),flush=True)
def compare_predictions(a,b):
    a=a.astype(np.complex128);b=b.astype(np.complex128);delta=float(np.sum(abs(a-b)**2));base=float(np.sum(abs(a)**2))
    return dict(train_norm_squared=base,delta_squared=delta,query_vs_train_relative_L2=math.sqrt(delta/base) if base>0 else None)
def forward():
    pre=json.loads((R/"preflight.json").read_text());elapsed=time.time()-pre["preflight_start_unix"]
    if elapsed>=180:raise TimeoutError("Whole-check wall cap already used")
    signal.signal(signal.SIGALRM,alarm);signal.alarm(max(1,int(180-elapsed)))
    if (R/"result.json").exists():raise FileExistsError("Preserve prior fixed-model result")
    unit.source_check(pre["source_identity"]);d,target=load_inputs()
    # Read-only NVML guard is written by agent immediately before this invocation.
    idle=json.loads((R/"GPU_idle_preflight.json").read_text())
    if idle["GPU_UUID"]!=unit.GPU or not idle["idle_verified"]:raise RuntimeError("Allowed H1001 not idle")
    guard=device_guard(unit.GPU);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    bank=FitGeometryBank(GEOMETRY,d.inputs,outer=0,role="screen_fit");cache=MemoizedFitDirections(bank).to("cuda:0")
    model=build_native_batch_model(d,cache,128,128,frequency_chunk=8).to("cuda:0")
    ck=torch.load(PARENT/"final.pt",map_location="cpu",weights_only=False);model.load_state_dict(ck["model"]);model.eval()
    if "physical_density" in model.fine[0].__dict__:raise RuntimeError("Original builder must have no patch")
    dirs_train=[];query_descriptors=[];angular=[];geometry_reads={}
    with torch.no_grad():
        pos=torch.tensor(d.positions(ROWS),device="cuda:0",dtype=torch.float64);sites=torch.tensor(d.sites(ROWS),device="cuda:0",dtype=torch.long)
        origins=model.spec.ue_world_positions(pos,sites);tx=model.bs_ports[sites]
        before=list(bank.opened_ids);qd=model.ray_provider.directions(origins,sites,tx)
        if before!=bank.opened_ids:raise RuntimeError("Query directions read geometry payload")
        atomic_npy(A/"query_directions_BS30.npy",qd[:,0,30,0].cpu().numpy())
        for i,row in enumerate(ROWS):
            td=cache.scalar(model,int(row),0,torch.tensor(BS,device="cuda:0"),origins[i],sites[i],tx[i])[0]
            t=td.cpu().numpy();q=qd[i,0,30,0].cpu().numpy();dirs_train.append(t);atomic_npy(A/f"FIT_directions_row{int(row)}_BS30.npy",t)
            nn=np.degrees(np.arccos(np.clip(t@q.T,-1.,1.))).min(1)
            direct=(tx[i,0,30]-origins[i,0]);direct=direct/torch.linalg.vector_norm(direct)
            direct_angle=float(torch.rad2deg(torch.acos(torch.clamp(torch.dot(qd[i,0,30,0,0],direct),-1.,1.))))
            angular.append(dict(row=int(row),training_rays=len(t),query_rays=len(q),training_LoS_count=int(cache.los[int(row),:int(cache.count[int(row)])].sum()),
                nearest_train_to_query_angle_deg_min=float(nn.min()),nearest_train_to_query_angle_deg_median=float(np.median(nn)),nearest_train_to_query_angle_deg_max=float(nn.max()),
                matched_training_rays_under1deg=int((nn<1.).sum()),matched_training_rays_under_point1deg=int((nn<.1).sum()),nearest_angles_deg=nn.tolist(),
                query_first_ray_direct_angle_deg=direct_angle))
        # The FIT helper accesses cached allowed FIT rays; query is the actual inference interface.
        train,train_coarse=subset_forward(model,cache,d,ROWS,0,BS,FREQ)
        opened_before_query=list(bank.opened_ids);query=subset_query(model,d,ROWS,BS,np.array([0],np.int64),FREQ)
        if bank.opened_ids!=opened_before_query:raise RuntimeError("Query forward read geometry payload")
        train=train.cpu().numpy();query_original_shape=list(query.shape);query=query[:,:,0,:].cpu().numpy()
    atomic_npy(A/"FIT_route_prediction.npy",train);atomic_npy(A/"query_route_prediction.npy",query)
    modelmeta=model.ray_provider.metadata()
    for site,values in model.ray_provider.transforms.items():
        for k,v in enumerate(values):
            query_descriptors.append(dict(site=int(site),slot=k,kind="forced_direct" if k==0 else "FIT_virtual_source",support_count=len(v["rows"]),depth=int(v["depth"]),source_FIT_rows=sorted(map(int,v["rows"])),a=np.asarray(v["a"]).tolist(),b=np.asarray(v["b"]).tolist()))
    comparison=compare_predictions(train,query);comparison["per_position"]=[dict(row=int(row),**compare_predictions(train[i],query[i])) for i,row in enumerate(ROWS)]
    result=dict(status="fixed_model_dual_route_completed",source_identity=pre["source_identity"],checkpoint_sha256=sha256(PARENT/"final.pt"),GPU=guard,GPU_idle_preflight=idle,
        whole_check_elapsed_seconds=time.time()-pre["preflight_start_unix"],axes=dict(row_ids=ROWS.tolist(),field=0,BS_ids=BS.tolist(),UE_ids=[0],frequency_count=192),
        FIT_route_metrics=metrics(train,target),query_route_metrics=metrics(query,target),prediction_difference=comparison,angular_comparison=angular,
        raw_prediction_shape=list(train.shape),original_query_shape=query_original_shape,raw_H=True,FIT_RMS_metadata_only=d.scale,
        query_provider_metadata=modelmeta,query_candidates=query_descriptors,query_direct_candidate_forced=True,
        training_ray_order="FIT tau ascending stable, cap64; nonLoS last_interaction minus actual UE, no BS correction; noLoS for these two rows",
        query_ray_order="forced direct first; FIT specular image transforms by support descending/depth ascending/translation; then deterministic sphere slots if any",
        geometry_bank_source_ids_sha256=__import__("hashlib").sha256(bank.source_fit_ids.astype(np.int64).tobytes()).hexdigest(),
        geometry_bank_source_count=len(bank.source_fit_ids),geometry_cache_opened_count=len(bank.opened_ids),geometry_cache_opened_unique_count=len(set(bank.opened_ids)),
        query_geometry_payload_reads=0,query_CSI_used_for_construction=False,query_truth_used_for_phase_delay_or_amplitude_alignment=False,
        held_query_truth_reads=0,only_authorized_FIT_labels_used_for_diagnostic_metrics=True,optimizer_updates=0,new_training=False,formal_score=False,formal_test=False,
        prediction_sha256=dict(FIT=sha256(A/"FIT_route_prediction.npy"),query=sha256(A/"query_route_prediction.npy")))
    unit.source_check(pre["source_identity"]);atomic_json(R/"result.json",result);signal.alarm(0)
    print(json.dumps({k:result[k] for k in ["status","whole_check_elapsed_seconds","FIT_route_metrics","query_route_metrics","prediction_difference","angular_comparison","prediction_sha256"]}),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--mode",choices=["preflight","forward"],required=True);args=p.parse_args()
    try:preflight() if args.mode=="preflight" else forward()
    except BaseException:
        R.mkdir(parents=True,exist_ok=True);atomic_json(R/f"{args.mode}_failure.json",dict(traceback=traceback.format_exc(),new_training=False,formal_score=False));raise
