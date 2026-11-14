"""Geometry-only ray/depth convergence with paired random seeds."""
from __future__ import annotations
import time,json,traceback,itertools
import numpy as np
from .common import *
from measured.hall_adapters.geometry import NativeRT
from measured.hall_adapters.metrics import directional_power,normalized_power
from measured.hall_adapters.check import independent_los_check
def compare(a,b):
    x,y=normalized_power(a),normalized_power(b)
    active=(a.sum(-1)>0)&(b.sum(-1)>0)
    denominator=np.linalg.norm(x,axis=-1)*np.linalg.norm(y,axis=-1)
    cosine=np.divide((x*y).sum(-1),denominator,out=np.zeros_like(denominator),where=denominator>0)
    l1=np.abs(x-y).sum(-1)
    with np.errstate(divide="ignore",invalid="ignore"):
        choice=np.take_along_axis(y,x.argmax(-1)[...,None],-1)[...,0]
        cross=10*np.log10(np.divide(y.max(-1),choice,out=np.full_like(choice,np.inf),where=choice>0))
    n=int(active.sum())
    report=dict(active_panels=n,both_zero_panels=int(((a.sum(-1)==0)&(b.sum(-1)==0)).sum()),
                newly_nonzero_panels=int(((a.sum(-1)==0)&(b.sum(-1)>0)).sum()),
                lost_panels=int(((a.sum(-1)>0)&(b.sum(-1)==0)).sum()),
                mean_cosine=float(cosine[active].mean()) if n else None,
                p10_cosine=float(np.quantile(cosine[active],.1)) if n else None,
                p90_normalized_power_L1=float(np.quantile(l1[active],.9)) if n else None,
                p90_cross_choice_loss_db=float(np.quantile(cross[active],.9)) if n and np.isfinite(cross[active]).all() else None,
                beam_change_fraction=float((x.argmax(-1)[active]!=y.argmax(-1)[active]).mean()) if n else None)
    report["passes"]=bool(n and report["mean_cosine"]>=.995 and report["p10_cosine"]>=.98 and
      report["p90_normalized_power_L1"]<=.05 and report["p90_cross_choice_loss_db"] is not None and
      report["p90_cross_choice_loss_db"]<=.5 and report["newly_nonzero_panels"]==0 and report["lost_panels"]==0)
    return report
def run():
    start=time.monotonic();data,contract=init();rt=NativeRT(data.spec,WORK/"rt_scene")
    rows=np.array(contract["rt_probe_rows"]);query=data.query(rows);save_json(REPORT/"rt_probe_positions.json",dict(row_ids=rows.tolist(),position=query.position.tolist(),triangle_los=independent_los_check(query).tolist(),channel_labels_read=False))
    folder=WORK/"rt_probes";folder.mkdir(exist_ok=True);outputs={};timing=[]
    # Complete depth sweep at65536, then source-budget sweep at depth8.
    pairs=list(dict.fromkeys([(65536,d) for d in [2,4,6,8]]+[(r,8) for r in [4096,16384,262144]]))
    for rays,depth in pairs:
      for random_seed in [SEED,SEED+1]:
        path=folder/("r%d_d%d_s%d.npz"%(rays,depth,random_seed));t=time.monotonic()
        if path.exists():
            with np.load(path) as z:
                np.testing.assert_array_equal(z["row_ids"],rows);p=z["power"];counts=z["counts"]
        else:
            pred,support=rt.trace(query,samples=rays,max_depth=depth,seed=random_seed)
            p=directional_power(pred["channel"]);counts=support["mask"].sum(-1)
            np.savez_compressed(path,row_ids=rows,power=p,counts=counts,**{"support_"+k:v for k,v in support.items()})
        outputs[(rays,depth,random_seed)]=p
        row=dict(rays=rays,depth=depth,seed=random_seed,seconds=time.monotonic()-t,nonzero_panels=int((p.sum(-1)>0).sum()),retained_counts=counts.tolist())
        timing.append(row);print("RT",json.dumps(row),flush=True);save_json(REPORT/"rt_progress.json",dict(completed=timing))
        if time.monotonic()-start>45*60:raise TimeoutError("RT bounded stage exceeded45 minutes")
    comparisons=[]
    for random_seed in [SEED,SEED+1]:
      for first,second in [(2,4),(4,6),(6,8)]:
        comparisons.append(dict(axis="depth",first=first,second=second,seed=random_seed,**compare(outputs[(65536,first,random_seed)],outputs[(65536,second,random_seed)])))
      for first,second in [(4096,16384),(16384,65536),(65536,262144)]:
        comparisons.append(dict(axis="rays",first=first,second=second,seed=random_seed,**compare(outputs[(first,8,random_seed)],outputs[(second,8,random_seed)])))
    for r,d in pairs:
        comparisons.append(dict(axis="seed",rays=r,depth=d,**compare(outputs[(r,d,SEED)],outputs[(r,d,SEED+1)])))
    decisive=[x for x in comparisons if (x["axis"]=="depth" and x["first"]==6) or (x["axis"]=="rays" and x["first"]==65536) or (x["axis"]=="seed" and x["rays"] in [65536,262144] and x["depth"]==8)]
    result=dict(completed=True,comparisons=comparisons,timing=timing,plateau_reached=all(x["passes"] for x in decisive),
                chosen_for_development=dict(rays=262144,max_depth=8,seed=SEED),
                probe_limitations="12 geometry-selected positions, fixed materials, no diffuse scattering/diffraction; both-zero links excluded from stability evidence.",
                source_CSI_read=False,seconds=time.monotonic()-start)
    save_json(REPORT/"rt_convergence.json",result);print("RT_COMPLETE",json.dumps(result),flush=True)
if __name__=="__main__":
    try:run()
    except BaseException:
        save_json(REPORT/"rt_failure.json",dict(traceback=traceback.format_exc()));raise
