"""South-only, capacity-audited multi-seed geometry development cache."""
from __future__ import annotations
import json,time,traceback
import numpy as np
from .common import *
from .rt_audit import AuditedRT
from measured.hall_adapters.metrics import directional_power
def merge_support(parts,cap=128):
    # Route and interaction geometry, without delay-origin-sensitive identity.
    n,p,k=parts[0]["mass"].shape
    out={key:np.zeros_like(value) for key,value in parts[0].items()}
    presence=np.zeros((n,p,cap),np.int16)
    for i in range(n):
      for panel in range(p):
        groups={}
        for seed_index,part in enumerate(parts):
          for j in np.flatnonzero(part["mask"][i,panel]):
            depth=int(part["depth"][i,panel,j])
            route=np.rint(part["sequence"][i,panel,j,:depth,:3]*3000).astype(np.int64)
            kinds=part["interaction"][i,panel,j,:depth]
            angles=np.rint(part["coordinate"][i,panel,j,:2]*4).astype(np.int64)
            key=(depth,route.tobytes(),kinds.tobytes(),angles.tobytes())
            groups.setdefault(key,[]).append((seed_index,int(j),float(part["mass"][i,panel,j])))
        ranking=[]
        for key,group in groups.items():
            # Remove within-seed duplicate keys before calculating support presence.
            per_seed={}
            for seed_index,j,energy in group:
                if seed_index not in per_seed or energy>per_seed[seed_index][2]:per_seed[seed_index]=(seed_index,j,energy)
            items=list(per_seed.values());mean=sum(v[2] for v in items)/len(parts)
            anchored=mean*np.sqrt(len(items)/len(parts));ranking.append((anchored,items))
        ranking.sort(key=lambda v:-v[0])
        for j,(energy,items) in enumerate(ranking[:cap]):
            si,source_index,_=max(items,key=lambda v:v[2])
            for key in out:out[key][i,panel,j]=parts[si][key][i,panel,source_index]
            out["mass"][i,panel,j]=energy;out["mask"][i,panel,j]=energy>0
            presence[i,panel,j]=len(items)
            out["features"][i,panel,j,9]=np.log(max(energy,1e-30))/30
    return out,presence
def run():
    start=time.monotonic();dev=DevelopmentData();rows=np.r_[dev.fit_ids,dev.val_ids]
    rt=AuditedRT(dev.data.spec,capacity=1048576);folder=WORK/"geometry";folder.mkdir(exist_ok=True)
    records=[];combined={};coherent=[];incoherent=[];presence=[]
    save_json(REPORT/"geometry_cache_contract.json",dict(rows=rows.tolist(),batch=12,rays=1048576,depth=16,candidate_capacity=1048576,seeds=[SEED,SEED+1],
      label_inputs=False,formal_RT_convergence_established=False,
      scope="Development candidate using two-seed geometry presence; frozen thresholds remain; not a certified converged RT baseline."))
    for first in range(0,len(rows),12):
        path=folder/("batch%03d.npz"%(first//12));jp=path.with_suffix(".json")
        if path.exists():
            with np.load(path) as z:
                np.testing.assert_array_equal(z["row_ids"],rows[first:first+12])
                merged={key[len("support_"):]:z[key] for key in z.files if key.startswith("support_")}
                pre=z["presence"];hp=z["coherent_power"];ip=z["incoherent_power"]
            details=json.loads(jp.read_text())
        else:
            query=dev.data.query(rows[first:first+12]);parts=[];cps=[];ips=[];details=[]
            for random_seed in [SEED,SEED+1]:
                pred,support,non,diag=rt.traced(query,samples=1048576,max_depth=16,seed=random_seed)
                if any(d["saturated"] for d in diag):raise ValueError("Geometry development cache candidate capacity saturated")
                parts.append(support);cps.append(directional_power(pred["channel"]));ips.append(non);details.append(dict(seed=random_seed,diagnostics=diag))
            merged,pre=merge_support(parts);hp=np.mean(cps,axis=0);ip=np.mean(ips,axis=0)
            np.savez_compressed(path,row_ids=query.row_ids,presence=pre,coherent_power=hp,incoherent_power=ip,**{"support_"+k:v for k,v in merged.items()})
            save_json(jp,details)
        for key,value in merged.items():combined.setdefault(key,[]).append(value)
        coherent.append(hp);incoherent.append(ip);presence.append(pre);records.append(dict(first=first,details=details))
        print("GEOMETRY",first,len(rows),time.monotonic()-start,flush=True)
        if time.monotonic()-start>45*60:raise TimeoutError("Geometry cache wall budget")
    combined={key:np.concatenate(value) for key,value in combined.items()}
    np.savez_compressed(WORK/"geometry_support.npz",row_ids=rows,presence=np.concatenate(presence),**combined)
    np.savez_compressed(WORK/"rt_power.npz",row_ids=rows,coherent=np.concatenate(coherent),incoherent=np.concatenate(incoherent))
    save_json(REPORT/"geometry_cache.json",dict(complete=True,records=records,seconds=time.monotonic()-start,
         retained_paths=combined["mask"].sum(-1).tolist(),source_CSI_read=False,
         formal_convergence_proven=False))
if __name__=="__main__":
    try:run()
    except BaseException:
        save_json(REPORT/"geometry_cache_failure.json",dict(traceback=traceback.format_exc()));raise
