"""Final bounded high-capacity RT extension; thresholds remain unchanged."""
from __future__ import annotations
import time,json,traceback
import numpy as np
from .common import *
from .rt_audit import AuditedRT
from .rt_probe import compare
from measured.hall_adapters.metrics import directional_power
def run():
    start=time.monotonic();data,c=init();q=data.query(c["rt_probe_rows"]);rt=AuditedRT(data.spec,4194304)
    settings=[(1048576,8),(4194304,8),(4194304,12),(4194304,16),(1048576,16)]
    save_json(REPORT/"rt_final_extension_contract.json",dict(reason="Capacity also controls native deduplication hash-table size. Eliminate saturation and reduce hash collisions; verify high-ray and high-depth stability separately.",settings=settings,capacity=4194304,seeds=[SEED,SEED+1],thresholds_unchanged=True,source_CSI_read=False))
    folder=WORK/"rt_high_capacity";folder.mkdir(exist_ok=True);out={};records=[]
    for rays,depth in settings:
      for rs in [SEED,SEED+1]:
        path=folder/("r%d_d%d_s%d.npz"%(rays,depth,rs));jp=path.with_suffix(".json");t=time.monotonic()
        if path.exists():
            with np.load(path) as z:p=z["power"];ip=z["incoherent_power"]
            diag=json.loads(jp.read_text())
        else:
            pred,support,ip,diag=rt.traced(q,samples=rays,max_depth=depth,seed=rs);p=directional_power(pred["channel"])
            np.savez_compressed(path,row_ids=q.row_ids,power=p,incoherent_power=ip);save_json(jp,diag)
        records.append(dict(rays=rays,depth=depth,seed=rs,diagnostics=diag,seconds=time.monotonic()-t))
        out[(rays,depth,rs)]=(p,ip);print("HIGH_RT",rays,depth,rs,time.monotonic()-t,flush=True)
        save_json(REPORT/"rt_high_capacity_progress.json",dict(records=records))
        if time.monotonic()-start>45*60:raise TimeoutError("High capacity RT budget")
    comparisons=[]
    for axis,a,b in [("rays",(1048576,8),(4194304,8)),("rays",(1048576,16),(4194304,16)),("depth",(4194304,8),(4194304,12)),("depth",(4194304,12),(4194304,16))]:
      for rs in [SEED,SEED+1]:
        x,y=out[(*a,rs)],out[(*b,rs)]
        comparisons.append(dict(axis=axis,first=a,second=b,seed=rs,coherent=compare(x[0],y[0]),incoherent=compare(x[1],y[1])))
    for rays,depth in settings:
        x,y=out[(rays,depth,SEED)],out[(rays,depth,SEED+1)]
        comparisons.append(dict(axis="seed",rays=rays,depth=depth,coherent=compare(x[0],y[0]),incoherent=compare(x[1],y[1])))
    save_json(REPORT/"rt_high_capacity_convergence.json",dict(complete=True,comparisons=comparisons,records=records,seconds=time.monotonic()-start,capacity=4194304,source_CSI_read=False))
if __name__=="__main__":
    try:run()
    except BaseException:
        save_json(REPORT/"rt_high_failure.json",dict(traceback=traceback.format_exc()));raise
