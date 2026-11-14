"""v4 training semantics:32 iid FIT stations, one scalar CSI per station, native NMSE."""
from __future__ import annotations
import numpy as np
import torch
from torch.nn import functional as F
from .newrf_latefreq import stratified64,density_weights,hierarchical64
from .newrf_conditioned_v3 import (build_conditioned_model,output_paths_v3,
                                  save_outer_prediction_v3,ConditionedNeWRF)
from .common import ART,REPORT,atomic_json,atomic_npy,digest_ids,sha256
from project_paths import PATHS

ADAPTER_VERSION="NeWRF-wideband-city-conditioned-native-batch-v4"

def output_paths_v4(method,outer=None,role=None):
    if method!="newrf":raise ValueError("NeWRF only")
    a=ART/"newrf"/"v4";r=REPORT/"newrf"/"v4"
    if outer is not None:a=a/f"outer{outer}";r=r/f"outer{outer}"
    if role is not None:a=a/role;r=r/role
    a.mkdir(parents=True,exist_ok=True);r.mkdir(parents=True,exist_ok=True)
    return dict(artifact=a,report=r)

def build_native_batch_model(data,cache,coarse=128,importance=128,*,frequency_chunk=8):
    model=build_conditioned_model(data,cache,coarse,importance,frequency_chunk=frequency_chunk)
    return model

def station_target(data,rows,field,bs_ids,freq_ids,device):
    p,u=divmod(field,4);rows=np.asarray(rows,np.int64);bs=np.asarray(bs_ids,np.int64);ff=np.asarray(freq_ids,np.int64)
    h=data.read_fit(rows,normalized=False)
    result=h[np.arange(len(rows)),p*128+bs,u,ff].copy()
    return torch.tensor(result,device=device,dtype=torch.complex64)

def station_forward_reference(model,cache,data,rows,field,bs_ids,freq_ids):
    """Per-station original scalar renderer; identical labels, geometry and weights."""
    from .newrf_runner import subset_forward
    fine=[];coarse=[]
    for row,bs,ff in zip(rows,bs_ids,freq_ids):
        h,hc=subset_forward(model,cache,data,np.asarray([row]),field,np.asarray([bs]),np.asarray([ff]))
        fine.append(h.reshape(()));coarse.append(hc.reshape(()))
    return torch.stack(fine),torch.stack(coarse)

def station_forward(model,cache,data,rows,field,bs_ids,freq_ids):
    """Batch all actual rays without padding, reducing only author per-station sums."""
    rows=np.asarray(rows,np.int64);bs=np.asarray(bs_ids,np.int64);ff=np.asarray(freq_ids,np.int64)
    if rows.ndim!=1 or bs.shape!=rows.shape or ff.shape!=rows.shape or not len(rows):
        raise ValueError("One actual BS and frequency per station required")
    if not set(map(int,rows)).issubset(cache.allowed):raise PermissionError("Non-FIT station")
    if np.any((bs<0)|(bs>=128)) or np.any((ff<0)|(ff>=192)):raise ValueError("Scalar axis range")
    dev=model.bs_ports.device;p,u=divmod(field,4)
    pos=torch.tensor(data.positions(rows),dtype=torch.float64,device=dev)
    sites=torch.tensor(data.sites(rows),dtype=torch.long,device=dev)
    origins=model.spec.ue_world_positions(pos,sites);tx=model.bs_ports[sites]
    dirs=[];oo=[];tt=[];freq=[];groups=[]
    f=torch.tensor(data.frequencies(ff),dtype=torch.float64,device=dev)
    for i,row in enumerate(rows):
        ss=torch.tensor([int(bs[i])],device=dev,dtype=torch.long)
        d=cache.scalar(model,int(row),field,ss,origins[i],sites[i],tx[i])[0]
        r=len(d);groups.append(r);dirs.append(d)
        oo.append(origins[i,u].expand(r,3));tt.append(tx[i,p,int(bs[i])].expand(r,3))
        freq.append(f[i].expand(r,1))
    directions=torch.cat(dirs);origin=torch.cat(oo);transmit=torch.cat(tt);fr=torch.cat(freq)
    points,z=stratified64(origin,directions,model.config.far_m,model.config.n_samples)
    def prepare(field,points):
        view=directions[:,None].expand_as(points);t=transmit[:,None].expand_as(points)
        base,rho=field.prepare(points,view,t)
        ef=field.encoded_frequency(fr)
        split=field.cfg.hidden_dim+field.view_encoder.d_output
        frequency_bias=F.linear(ef,field.network.branch.weight[:,split:],None)
        signal=field.network.output(field.network.act(base+frequency_bias[:,None,:]))
        return torch.cat((signal,rho),-1),rho
    raw,rho=prepare(model.coarse[field],points)
    hc,_,_,_=model.synthesizer.synthesize(raw,z,fr/1e9,groups)
    weights=density_weights(rho,z)
    finepoints,finez=hierarchical64(origin,directions,z,weights,model.config.n_importance)
    raw,_=prepare(model.fine[field],finepoints)
    h,_,_,_=model.synthesizer.synthesize(raw,finez,fr/1e9,groups)
    return h.to(torch.complex64),hc.to(torch.complex64)

def native_weighted_loss(fine,coarse,target,scale=None):
    """Pinned source complex batchNMSE. FIT RMS cancels and is not a weight."""
    energy=target.abs().square().sum(dtype=torch.float64)
    if not bool(torch.isfinite(energy)) or float(energy)<=0:
        raise ValueError("Source batchNMSE undefined for all-zero target batch; do not discard silently")
    lf=(fine-target).abs().square().sum(dtype=torch.float64)/energy
    lc=(coarse-target).abs().square().sum(dtype=torch.float64)/energy
    return .9*lf+.1*lc,lf,lc

def save_outer_prediction_v4(data,method,values,*,metadata=None):
    if data.role!="refit" or method!="newrf":raise PermissionError("NeWRF refit required")
    a=np.asarray(values,dtype=np.complex64)
    if a.shape!=(len(data.outer_ids),256,4,192) or not np.isfinite(a).all():raise ValueError("Complete finite foldH")
    out=output_paths_v4(method,data.outer,"prediction")["artifact"]
    atomic_npy(out/"channels.npy",a);atomic_npy(out/"row_ids.npy",data.outer_ids)
    meta=dict(metadata or {})
    meta.update(adapter_version=ADAPTER_VERSION,raw_H=True,oracle_phase_delay_scale_alignment=False,
                health_evidence=meta.get("formal_config",{}).get("health_evidence",{}),
                training_objective="source .1coarse+.9fine batch complexNMSE",
                training_batch="32 iid FIT station draws, one actual BS/frequency scalar per draw")
    if "source_identities" in meta:meta["source_identity"]=meta["source_identities"]
    manifest=dict(schema="canonical-baseline-prediction-v1",method=method,adapter_version=ADAPTER_VERSION,
       outer=data.outer,input_identity=data.inputs.identity,row_ids=data.outer_ids.tolist(),
       row_ids_sha256=digest_ids(data.outer_ids),prediction_namespace="canonical_train_row_ids",
       kind="channel",path=PATHS.relative(out/"channels.npy"),sha256=sha256(out/"channels.npy"),
       shape=list(a.shape),dtype=str(a.dtype),fit_data=data.info(),metadata=meta,
       query_truth_read=False,trained=True)
    atomic_json(out/"manifest.json",manifest);return manifest

