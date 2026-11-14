"""One final8M-ray check at depth16, with unchanged plateau thresholds."""
from __future__ import annotations
import time,json,traceback
import numpy as np
from .common import *
from .rt_audit import AuditedRT
from .rt_probe import compare
from measured.hall_adapters.metrics import directional_power
def run():
    start=time.monotonic();data,c=init();q=data.query(c["rt_probe_rows"]);rt=AuditedRT(data.spec,4194304)
    save_json(REPORT/"rt_8m_contract.json",dict(reason="4M-ray path-energy predictions were seed-stable at depth16 but the1M-to4M ray increment still exceeded the frozen L1 criterion. Check one last doubling instead of relaxing thresholds.",rays=8388608,depth=16,capacity=4194304,seeds=[SEED,SEED+1],channel_labels_read=False))
    folder=WORK/"rt_8m";folder.mkdir(exist_ok=True);out={};records=[];comparisons=[]
    for rs in [SEED,SEED+1]:
        path=folder/("s%d.npz"%rs);jp=path.with_suffix(".json");t=time.monotonic()
        if path.exists():
            with np.load(path) as z:p=z["power"];ip=z["incoherent_power"]
            diag=json.loads(jp.read_text())
        else:
            pred,support,ip,diag=rt.traced(q,samples=8388608,max_depth=16,seed=rs);p=directional_power(pred["channel"])
            np.savez_compressed(path,row_ids=q.row_ids,power=p,incoherent_power=ip);save_json(jp,diag)
        records.append(dict(seed=rs,seconds=time.monotonic()-t,diagnostics=diag));out[rs]=(p,ip)
        old=WORK/"rt_high_capacity"/("r4194304_d16_s%d.npz"%rs)
        with np.load(old) as z:
            comparisons.append(dict(axis="rays_4M_to_8M",seed=rs,coherent=compare(z["power"],p),incoherent=compare(z["incoherent_power"],ip)))
        print("RT_8M",rs,time.monotonic()-t,flush=True)
    a,b=out[SEED],out[SEED+1];comparisons.append(dict(axis="seed_8M",coherent=compare(a[0],b[0]),incoherent=compare(a[1],b[1])))
    save_json(REPORT/"rt_8m_convergence.json",dict(complete=True,comparisons=comparisons,records=records,seconds=time.monotonic()-start,thresholds_unchanged=True,no_labels_read=True))
if __name__=="__main__":
    try:run()
    except BaseException:
        save_json(REPORT/"rt_8m_failure.json",dict(traceback=traceback.format_exc()));raise
