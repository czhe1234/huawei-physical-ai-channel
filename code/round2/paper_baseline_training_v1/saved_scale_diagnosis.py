"""Read-only saved-model scale diagnosis. No optimizer or model/result writes."""
from __future__ import annotations
import argparse,json,math,time,traceback
from pathlib import Path
import numpy as np
import torch
from project_paths import PATHS
from . import common as C
from .newrf_runner import subset_forward
from .newrf_query import subset_query
BS=np.array([0,127,128,255],np.int64)
UE=np.arange(4,dtype=np.int64)
FF=np.linspace(0,191,12,dtype=np.int64)
ROOT=C.REPORT/"small_output_diagnosis_20261101"
def stats(pred,target,rows):
    p=np.asarray(pred,np.complex128);t=np.asarray(target,np.complex128)
    axes=tuple(range(1,p.ndim))
    pe=np.sum(abs(p)**2,axis=axes);te=np.sum(abs(t)**2,axis=axes)
    se=np.sum(abs(p-t)**2,axis=axes);me=np.sum((abs(p)-abs(t))**2,axis=axes)
    cross=np.sum(p.conj()*t,axis=axes);positive=te>0
    total=float(te.sum());predenergy=float(pe.sum())
    out=dict(count=len(rows),complex_scalars=int(p.size),truth_energy=total,
       prediction_energy=predenergy,complex_nmse=float(se.sum()/total) if total else None,
       magnitude_nmse=float(me.sum()/total) if total else None,
       prediction_to_truth_energy=predenergy/total if total else None,
       correlation_real=float(cross.real.sum()/np.sqrt(total*predenergy)) if total*predenergy else None,
       correlation_imag=float(cross.imag.sum()/np.sqrt(total*predenergy)) if total*predenergy else None,
       zero_truth_rows=int((~positive).sum()),zero_prediction_rows=int((pe==0).sum()),
       truth_rms=float(np.sqrt(total/t.size)),prediction_rms=float(np.sqrt(predenergy/p.size)))
    out["per_position"]=[dict(row=int(r),truth_energy=float(a),prediction_energy=float(b),
       energy_ratio=float(b/a) if a else None,nmse=float(e/a) if a else None,
       correlation_real=float(c.real/np.sqrt(a*b)) if a*b else None)
       for r,a,b,e,c in zip(rows,te,pe,se,cross)]
    return out
def panel(data):
    power=np.load(PATHS.artifacts/"round2/pre_sionna/round2_train_spectral_descriptors.npz")["total_power"]
    result=[]
    for site in (0,1):
        ids=data.fit_ids[data.sites(data.fit_ids)==site]
        pos=ids[power[ids]>0]
        pos=pos[np.lexsort((pos,power[pos]))]
        result.extend(pos[np.linspace(0,len(pos)-1,min(12,len(pos)),dtype=int)].tolist())
        result.extend(ids[power[ids]==0][:2].tolist())
    return np.array(sorted(set(result)),np.int64)
def target(data,rows,kind):
    h=data.read_fit(rows,FF,normalized=False) if kind=="FIT" else data.read_validation(rows,FF,normalized=False)
    return h[:,BS]
def nerf_grid(model,data,rows):
    pos=torch.tensor(data.positions(rows),device="cuda:0",dtype=torch.float64)
    site=torch.tensor(data.sites(rows),device="cuda:0",dtype=torch.long)
    freq=torch.tensor(data.frequencies(FF),device="cuda:0",dtype=torch.float64)
    world=data.geometry.ue_world_positions(pos,site)
    out=torch.empty((len(rows),4,4,len(FF)),device="cuda:0",dtype=torch.complex64)
    for s in (0,1):
        rr=torch.nonzero(site==s).flatten()
        if not len(rr):continue
        for b in (0,1):
            tx=model.bound_bs_ports[s,b].reshape(128,3)[torch.tensor([0,127],device="cuda:0")]
            for u in (0,1):
                v=model.fields[f"s{s}_b{b}_u{u}"](tx,world[rr,u*2:(u+1)*2],freq).permute(1,0,2,3)
                out[rr,b*2:(b+1)*2,u*2:(u+1)*2]=v*model.log_output_gain.exp()
    return out
def fit_newrf(model,cache,data,rows):
    fine=torch.empty((len(rows),4,4,len(FF)),device="cuda:0",dtype=torch.complex64)
    coarse=torch.empty_like(fine)
    for field in range(8):
        p,u=divmod(field,4)
        f,c=subset_forward(model,cache,data,rows,field,np.array([0,127]),FF)
        fine[:,p*2:(p+1)*2,u,:]=f
        coarse[:,p*2:(p+1)*2,u,:]=c
    return fine,coarse
def check_sources(method,ck):
    if method=="nerf2":
        from .nerf2_train_main_v3 import check_sources
        check_sources(ck["source_identity"])
    else:
        from .newrf_train_v4 import check_sources
        check_sources(ck["source_identities"])
def npy(x):return x.detach().cpu().numpy()
def main(args):
    global P,V,make_newrf
    if args.method=="nerf2":
        from . import nerf2_runner_v2 as P,nerf2_runner_v3 as V
    else:
        from .newrf_train_v4 import make as make_newrf
    ROOT.mkdir(parents=True,exist_ok=True)
    result_path=ROOT/(args.method+".json")
    if result_path.exists():raise FileExistsError(result_path)
    gpu=C.device_guard(args.gpu)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.50,0)
    version="v3" if args.method=="nerf2" else "v4"
    records=[];start=time.monotonic()
    for outer in range(5):
        for role in ("screen","refit"):
            data=C.FoldData(outer,role,args.method)
            path=C.ART/args.method/version/f"outer{outer}"/role/("best.pt" if role=="screen" else "final.pt")
            before=C.sha256(path);ck=torch.load(path,map_location="cpu",weights_only=False);check_sources(args.method,ck)
            if "fit_ids_sha256" in ck and ck["fit_ids_sha256"]!=C.digest_ids(data.fit_ids):raise ValueError("Wrong FIT roles")
            if "scale" in ck and ck["scale"]!=data.scale:raise ValueError("Wrong checkpoint RMS")
            if args.method=="nerf2":model=V.build(data);cache=None;state=ck["model"]
            else:model,cache=make_newrf(data,ck["config"]);state=ck["state_dict"]
            model.load_state_dict(state,strict=True);model.eval()
            for parameter in model.parameters():parameter.requires_grad_(False)
            rows=panel(data)
            if role=="screen" and set(map(int,rows))&set(map(int,data.val_ids)):raise ValueError("FIT validation overlap")
            rec=dict(method=args.method,outer=outer,role=role,step=ck["step"],checkpoint=PATHS.relative(path),
                checkpoint_sha256=before,FIT_count=len(data.fit_ids),FIT_RMS=data.scale,FIT_rows=rows.tolist(),
                normalization="NeRF2 predicts normalized H and multiplies RMS once; NeWRF predicts raw H",
                input_identity=data.inputs.identity,fit_ids_sha256=C.digest_ids(data.fit_ids))
            with torch.no_grad():
                y=target(data,rows,"FIT")
                yn=data.read_fit(rows,FF,normalized=True)[:,BS]
                rec["target_normalization_relative_L2"]=float(np.linalg.norm(yn-y/data.scale)/max(np.linalg.norm(yn),1e-30))
                if args.method=="nerf2":
                    pn=nerf_grid(model,data,rows);p=npy(pn*data.scale)
                    rec["output_gain"]=float(model.log_output_gain.exp())
                    rec["gain_initial"]=1/324
                    rec["FIT_train_route"]=stats(p,y,rows)
                    rec["FIT_normalized_route"]=stats(npy(pn),yn,rows)
                    # Reference forward: same native subset renderer, not alternate weights.
                    reference=P.subset(model,data,rows[:1])
                    index=np.array([0,7,8,15])
                    rec["native_subset_relative_L2"]=float(torch.linalg.vector_norm(reference[:,index]-pn[:1])/torch.linalg.vector_norm(pn[:1]).clamp_min(1e-30))
                else:
                    pf,pc=fit_newrf(model,cache,data,rows)
                    pq=subset_query(model,data,rows,BS,UE,FF)
                    p=npy(pf);q=npy(pq)
                    rec["FIT_train_route"]=stats(p,y,rows);rec["FIT_coarse_train_route"]=stats(npy(pc),y,rows)
                    rec["FIT_query_route"]=stats(q,y,rows)
                    rec["FIT_query_vs_train_relative_L2"]=float(np.linalg.norm(q-p)/max(np.linalg.norm(p),1e-30))
                    rec["FIT_normalized_route"]=stats(p/data.scale,yn,rows)
                    rec["far_m"]=cache.far_m;rec["FIT_ray_counts"]=[int(cache.count[r]) for r in rows]
                if role=="screen":
                    vr=data.probe_ids;vy=target(data,vr,"validation")
                    vp=nerf_grid(model,data,vr)*data.scale if args.method=="nerf2" else subset_query(model,data,vr,BS,UE,FF)
                    rec["validation_query_route"]=stats(npy(vp),vy,vr)
                    rec["validation_rows"]=vr.tolist()
                else:
                    # No outer truth read. Compare to immutable saved prediction on two public query rows.
                    ids=data.outer_ids[np.array([0,len(data.outer_ids)-1])]
                    op=nerf_grid(model,data,ids)*data.scale if args.method=="nerf2" else subset_query(model,data,ids,BS,UE,FF)
                    saved=C.ART/args.method/version/f"outer{outer}"/"prediction/channels.npy"
                    om=np.load(saved,mmap_mode="r")
                    selected=np.array([om[0][BS][:,:,FF],om[-1][BS][:,:,FF]])
                    actual=npy(op)
                    rec["saved_output_replay_relative_L2"]=float(np.linalg.norm(actual-selected)/max(np.linalg.norm(selected),1e-30))
                    rec["saved_output_replay_max_abs"]=float(np.max(abs(actual-selected)))
                    rec["saved_output_replay_rows"]=ids.tolist()
                    # Raw<->normalized boundary is algebraic, never fit on query values.
                    rec["output_unit_roundtrip_relative_L2"]=float(np.linalg.norm((actual/data.scale)*data.scale-actual)/max(np.linalg.norm(actual),1e-30))
            after=C.sha256(path)
            if before!=after:raise RuntimeError("Checkpoint changed")
            rec["checkpoint_preserved"]=True;rec["elapsed_seconds"]=time.monotonic()-start
            records.append(rec)
            C.atomic_json(ROOT/(args.method+"_progress.json"),dict(status="running",records=records,GPU=gpu,no_training=True))
            print(json.dumps(dict(method=args.method,outer=outer,role=role,step=ck["step"],
                FIT_nmse=rec["FIT_train_route"]["complex_nmse"],FIT_energy=rec["FIT_train_route"]["prediction_to_truth_energy"],
                query_FIT_nmse=rec.get("FIT_query_route",{}).get("complex_nmse"),
                query_FIT_energy=rec.get("FIT_query_route",{}).get("prediction_to_truth_energy"),
                validation_nmse=rec.get("validation_query_route",{}).get("complex_nmse"),
                validation_energy=rec.get("validation_query_route",{}).get("prediction_to_truth_energy"),
                replay=rec.get("saved_output_replay_relative_L2"),seconds=rec["elapsed_seconds"])),flush=True)
            del model,cache,ck,state;torch.cuda.empty_cache()
    C.atomic_json(result_path,dict(status="complete",method=args.method,records=records,GPU=gpu,
        no_training=True,optimizer_created=False,no_old_scores_changed=True,outer_truth_read=False,
        BS_ids=BS.tolist(),UE_ids=UE.tolist(),frequency_ids=FF.tolist(),source_sha256=C.sha256(Path(__file__)),
        selection="FIT nonzero whole-snapshot-energy quantiles12/site, plus up to2zero/site; fixed existing validation16",
        caveat="Point-panel and submatrix diagnostics; not a new official all-axes score.",
        elapsed_seconds=time.monotonic()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated()))
if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--method",choices=["nerf2","newrf"],required=True);a.add_argument("--gpu",required=True)
    args=a.parse_args()
    try:main(args)
    except BaseException:
        ROOT.mkdir(parents=True,exist_ok=True)
        (ROOT/(args.method+"_failure.txt")).write_text(traceback.format_exc());raise
