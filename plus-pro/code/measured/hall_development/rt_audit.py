"""Audit candidate truncation, ray sampling and coherent/noncoherent stability."""
from __future__ import annotations
import time,json,traceback
import numpy as np
from .common import *
from .rt_probe import compare
from measured.hall_adapters.geometry import NativeRT
from measured.hall_adapters.metrics import directional_power
class AuditedRT(NativeRT):
    def __init__(self,spec,capacity=1048576):
        super().__init__(spec,WORK/"rt_scene");self.capacity=capacity;self.calls=[]
        self.original_solver=self.solver;self.solver=self.solve
        self.original_generator=self.original_solver._candidate_generator
        self.original_solver._candidate_generator=self.generate
    def generate(self,*args,**kwargs):
        import drjit as dr
        buffer=self.original_generator(*args,**kwargs);dr.eval(buffer.paths_counter)
        self.candidate_count=int(buffer.paths_counter.numpy()[0])
        return buffer
    def solve(self,*args,**kwargs):
        kwargs["max_num_paths_per_src"]=self.capacity
        paths=self.original_solver(*args,**kwargs)
        ar,ai=paths.a;a=ar.numpy()+1j*ai.numpy()
        a=a[:,0,0];valid=paths.valid.numpy()[:,0].astype(bool)
        a=np.where(valid[:,None,:],a,0)
        power=np.abs(a.astype(np.complex128))**2
        energy=power.sum(1);total=energy.sum(-1)
        captured=np.sort(energy,axis=-1)[:,-128:].sum(-1)
        fraction=np.divide(captured,total,out=np.zeros_like(total),where=total>0)
        angular=np.fft.fft2(a.reshape(len(a),8,4,-1),axes=(1,2),norm="ortho")
        noncoherent=(np.abs(angular)**2).sum(-1).reshape(len(a),32)
        self.calls.append(dict(candidate_count=self.candidate_count,capacity=self.capacity,
           saturated=bool(self.candidate_count>=self.capacity),raw_valid=valid.sum(-1).tolist(),top128_energy_fraction=fraction.tolist(),
           incoherent_power=noncoherent))
        return paths
    def traced(self,query,**kwargs):
        self.calls=[];pred,support=self.trace(query,**kwargs)
        diag=[{k:v for k,v in c.items() if k!="incoherent_power"} for c in self.calls]
        incoherent=np.stack([c["incoherent_power"] for c in self.calls],axis=1)
        return pred,support,incoherent,diag
def run():
    start=time.monotonic();data,contract=init();query=data.query(contract["rt_probe_rows"])
    folder=WORK/"rt_audited";folder.mkdir(exist_ok=True);rt=AuditedRT(data.spec);out={};records=[]
    # Capacity checks are performed before interpreting ray/depth plateaus.
    settings=[(65536,8,32768),(65536,8,131072),(65536,8,1048576),
              (262144,8,1048576),(1048576,8,1048576),(4194304,8,1048576),
              (1048576,12,1048576),(1048576,16,1048576)]
    save_json(REPORT/"rt_extension_contract.json",dict(reason="Initial grid failed stability; inspect candidate capacity and separate coherent from path-energy stability before any task score.",settings=settings,seeds=[SEED,SEED+1],thresholds_unchanged=True,channel_labels_read=False))
    for rays,depth,capacity in settings:
      rt.capacity=capacity
      for random_seed in [SEED,SEED+1]:
        path=folder/("r%d_d%d_c%d_s%d.npz"%(rays,depth,capacity,random_seed));diagpath=path.with_suffix(".json");t=time.monotonic()
        if path.exists():
            with np.load(path) as z:p=z["power"];non=z["incoherent_power"];counts=z["counts"]
            diag=json.loads(diagpath.read_text())
        else:
            pred,support,non,diag=rt.traced(query,samples=rays,max_depth=depth,seed=random_seed)
            p=directional_power(pred["channel"]);counts=support["mask"].sum(-1)
            np.savez_compressed(path,row_ids=query.row_ids,power=p,incoherent_power=non,counts=counts,**{"support_"+k:v for k,v in support.items()})
            save_json(diagpath,diag)
        out[(rays,depth,capacity,random_seed)]=(p,non)
        row=dict(rays=rays,depth=depth,capacity=capacity,seed=random_seed,seconds=time.monotonic()-t,
                 diagnostics=diag,nonzero_panels=int((p.sum(-1)>0).sum()))
        records.append(row);save_json(REPORT/"rt_audit_progress.json",dict(completed=records));print("AUDIT_RT",json.dumps(row),flush=True)
        if time.monotonic()-start>45*60:raise TimeoutError("RT extension exceeded bounded45 minutes")
    comparisons=[]
    pairs=[("capacity",(65536,8,32768),(65536,8,131072)),("capacity",(65536,8,131072),(65536,8,1048576)),
           ("rays",(65536,8,1048576),(262144,8,1048576)),("rays",(262144,8,1048576),(1048576,8,1048576)),
           ("rays",(1048576,8,1048576),(4194304,8,1048576)),("depth",(1048576,8,1048576),(1048576,12,1048576)),
           ("depth",(1048576,12,1048576),(1048576,16,1048576))]
    for axis,first,second in pairs:
      for random_seed in [SEED,SEED+1]:
        a,b=out[(*first,random_seed)],out[(*second,random_seed)]
        comparisons.append(dict(axis=axis,first=first,second=second,seed=random_seed,coherent=compare(a[0],b[0]),incoherent=compare(a[1],b[1])))
    for setting in settings:
        a,b=out[(*setting,SEED)],out[(*setting,SEED+1)]
        comparisons.append(dict(axis="seed",setting=setting,coherent=compare(a[0],b[0]),incoherent=compare(a[1],b[1])))
    save_json(REPORT/"rt_audited_convergence.json",dict(completed=True,comparisons=comparisons,records=records,seconds=time.monotonic()-start,
              no_channel_scoring=True,development_setting=dict(rays=1048576,depth=16,capacity=1048576),
              support_cap=128,scope="Sampling/truncation convergence within declared fixed physics only; no proof of physical fidelity."))
    print("AUDIT_RT_COMPLETE",time.monotonic()-start,flush=True)
if __name__=="__main__":
    try:run()
    except BaseException:
        save_json(REPORT/"rt_audit_failure.json",dict(traceback=traceback.format_exc()));raise
