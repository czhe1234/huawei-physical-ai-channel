"""Multi-source path aggregation. No ray tracing is run on import."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .config import Config
C0=299792458.
@dataclass
class Geometry:
    arrays:dict
    presence:np.ndarray
    energy_cv:np.ndarray
    variance_known:np.ndarray
    source_fraction:np.ndarray
    quality:dict
def excess_coordinate(features,bandwidth):
    return np.maximum((features[...,14]-features[...,15])*100/C0,0)*bandwidth
def legacy_geometry(support,presence=None,seed_count=2,bandwidth=50e6):
    a={k:np.array(support[k],copy=True) for k in ["coordinate","mass","mask","features","sequence","interaction","depth"]}
    seen=np.asarray(presence if presence is not None else a["mask"],np.float32)/seed_count
    # Legacy mass already included sqrt(presence/trials); recover mean energy first.
    a["mass"]=np.divide(a["mass"],np.sqrt(np.maximum(seen,1e-12)),out=np.zeros_like(a["mass"]),where=a["mask"])
    a["coordinate"][...,2]=excess_coordinate(a["features"],bandwidth)
    n,p,k=a["mass"].shape
    return Geometry(a,seen,np.zeros((n,p,k),np.float32),np.zeros((n,p,k),bool),
       np.stack([np.ones_like(seen),np.zeros_like(seen)],-1),
       dict(source_configs=["base"],seed_count=seed_count,variance_available=False,ray_convergence_certified=False,
            origin="previous provisional cache; no new RT",delay_reference="geometric excess path length"))
def aggregate(records,bandwidth=50e6,cap=128):
    if not records:raise ValueError("No geometry records")
    ids=[(r["config"],r["seed"]) for r in records]
    if len(set(ids))!=len(ids):raise ValueError("Duplicate configuration/seed record")
    configs=sorted(set(r["config"] for r in records))
    if any(c not in ["base","scatter"] for c in configs):raise ValueError("Unknown propagation source")
    first=records[0]["support"];n,p,_=first["mass"].shape
    a={k:np.zeros((n,p,cap,*first[k].shape[3:]),dtype=first[k].dtype) for k in ["coordinate","mass","mask","features","sequence","interaction","depth"]}
    presence=np.zeros((n,p,cap),np.float32);cv=np.zeros_like(presence);known=np.zeros_like(presence,bool);source=np.zeros((*presence.shape,2),np.float32)
    for i in range(n):
      for panel in range(p):
        groups={}
        for index,record in enumerate(records):
            sp=record["support"]
            if sp["mass"].shape[:2]!=(n,p):raise ValueError("Mixed geometry query layouts")
            for j in np.flatnonzero(sp["mask"][i,panel]):
                d=int(sp["depth"][i,panel,j]);route=np.rint(sp["sequence"][i,panel,j,:d,:3]*3000).astype(np.int64)
                angular=np.rint(np.mod(sp["coordinate"][i,panel,j,:2],[8,4])*4).astype(np.int64)
                key=(d,route.tobytes(),sp["interaction"][i,panel,j,:d].tobytes(),angular.tobytes())
                value=(index,int(j),float(sp["mass"][i,panel,j]))
                per=groups.setdefault(key,{})
                if index not in per or value[2]>per[index][2]:per[index]=value
        ranked=[]
        for values in groups.values():
            energy=np.zeros(len(records))
            for ri,j,e in values.values():energy[ri]=e
            means=[energy[[z["config"]==c for z in records]].mean() for c in configs]
            mean=float(np.mean(means))
            ranked.append((mean*np.sqrt(np.count_nonzero(energy)/len(records)),mean,energy,values))
        ranked.sort(key=lambda z:-z[0])
        for slot,(_,mean,energy,values) in enumerate(ranked[:cap]):
            ri,j,_=max(values.values(),key=lambda z:z[2]);sp=records[ri]["support"]
            for key in a:a[key][i,panel,slot]=sp[key][i,panel,j]
            a["mass"][i,panel,slot]=mean;a["mask"][i,panel,slot]=mean>0
            a["coordinate"][i,panel,slot,2]=excess_coordinate(sp["features"][i,panel,j],bandwidth)
            presence[i,panel,slot]=np.count_nonzero(energy)/len(records)
            known[i,panel,slot]=len(records)>1
            cv[i,panel,slot]=energy.std()/max(energy.mean(),1e-30)
            for c in configs:
                take=np.array([z["config"]==c for z in records])
                source[i,panel,slot,["base","scatter"].index(c)]=energy[take].mean()
            if source[i,panel,slot].sum()>0:source[i,panel,slot]/=source[i,panel,slot].sum()
    return Geometry(a,presence,cv,known,source,dict(source_configs=configs,seed_count=len(records),
       variance_available=len(records)>1,ray_convergence_certified=all(bool(r.get("ray_converged",False)) for r in records),
       variability_is="sampling/model variability, not independently calibrated measurement confidence"))
def subset(geometry,ids):
    return Geometry({k:v[ids] for k,v in geometry.arrays.items()},geometry.presence[ids],geometry.energy_cv[ids],
      geometry.variance_known[ids],geometry.source_fraction[ids],dict(geometry.quality))
