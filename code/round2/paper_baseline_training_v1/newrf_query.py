"""Tensor subset prediction; the same native scalar-link renderer is preserved."""
from __future__ import annotations
from dataclasses import replace
import torch
import numpy as np

def subset_query(model,data,ids,bs_ids,ue_ids,freq_ids):
    ids=np.asarray(ids,np.int64);bs=np.asarray(bs_ids,np.int64);ue=np.asarray(ue_ids,np.int64)
    ff=np.asarray(freq_ids,np.int64)
    if (ids.ndim!=1 or not len(ids) or bs.ndim!=1 or ue.ndim!=1 or ff.ndim!=1 or
        np.any((bs<0)|(bs>=256)) or np.any((ue<0)|(ue>=4)) or np.any((ff<0)|(ff>=192))):
        raise ValueError("Scalar query axis IDs invalid")
    model._qualify()
    device=model.bs_ports.device
    positions=torch.tensor(data.positions(ids),device=device,dtype=torch.float64)
    sites=torch.tensor(data.sites(ids),device=device,dtype=torch.long)
    origins=model.spec.ue_world_positions(positions,sites)
    tx=model.bs_ports[sites]
    directions=model.ray_provider.directions(origins,sites,tx)
    freq=torch.tensor(data.frequencies(ff),device=device,dtype=torch.float64)
    out=torch.empty((len(ids),len(bs),len(ue),len(ff)),device=device,dtype=torch.complex64)
    for p in (0,1):
        col=np.flatnonzero(bs//128==p)
        if not len(col):continue
        spatial=torch.tensor(bs[col]%128,device=device,dtype=torch.long)
        columns=torch.tensor(col,device=device,dtype=torch.long)
        for j,u in enumerate(ue):
            h,_=model._scalar_links(p*4+int(u),origins[:,int(u)],
                 directions[:,p,:,int(u)].index_select(1,spatial),
                 tx[:,p].index_select(1,spatial),freq)
            out[:,columns,j,:]=h
    if not bool(torch.isfinite(out).all()):raise FloatingPointError("Nonfinite scalar query")
    return out

def change_frequency_chunk(model,chunk):
    model.config=replace(model.config,frequency_chunk=int(chunk))
    for field in list(model.coarse)+list(model.fine):field.cfg=model.config
    return model
