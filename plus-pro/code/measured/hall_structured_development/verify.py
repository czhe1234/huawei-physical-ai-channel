"""Checkpoint regeneration and independent structure arithmetic, south only."""
from __future__ import annotations
import json,traceback,importlib.util
import numpy as np
import torch
from .common import *
from .baselines import SpectralMLP
from measured.hall_structured_v2.model import HallStructured
from measured.hall_structured_v2.config import Config
from measured.hall_structured_v2.observables import metrics,losses
def literal(h,t):
    h=np.asarray(h,np.complex128);t=np.asarray(t,np.complex128);f=h.shape[-1]
    zh=np.fft.fft2(h,axes=(-3,-2),norm="ortho").reshape(*h.shape[:2],32,f)
    zt=np.fft.fft2(t,axes=(-3,-2),norm="ortho").reshape(*t.shape[:2],32,f)
    def q(z,s):
        v=np.abs(z.reshape(*z.shape[:3],s,f//s))**2;v=v.sum(-1)
        return v/v.sum((-2,-1),keepdims=True)
    angle=np.mean([.5*np.mean(((np.maximum(q(zh,s),1e-12)**.5-np.maximum(q(zt,s),1e-12)**.5)**2).sum((-2,-1))) for s in [1,4,16]])
    def cov(x):
        x=x.reshape(*x.shape[:2],32,f);r=x@x.conj().swapaxes(-1,-2)
        return r/(np.abs(x)**2).sum((-2,-1))[...,None,None]
    rh,rt=cov(h),cov(t);spatial=.5*np.mean((np.abs(rh-rt)**2).sum((-2,-1)))
    def c(x,panel=False):
        out=[]
        for lag in [1,2,4,8,16,32,64,128,256,512]:
            if lag>=f:continue
            a,b=x[...,lag:],x[...,:-lag];axes=(-2,-1) if panel else (-1,)
            cross=(a*b.conj()).sum(axes);den=((np.abs(a)**2).sum(axes)*(np.abs(b)**2).sum(axes))**.5
            out.append(np.clip(np.abs(cross)/den,0,1))
        return np.stack(out,-1)
    cp,ct=c(h.reshape(*h.shape[:2],32,f),True),c(t.reshape(*t.shape[:2],32,f),True)
    cb,cbt=c(zh),c(zt);broad=q(zt,1)[...,0]
    frequency=.5*(np.mean((cp-ct)**2)+np.mean((((cb-cbt)**2).mean(-1)*broad).sum(-1)))
    return dict(angle=float(angle),spatial=float(spatial),frequency=float(frequency),structural=float((angle+spatial+frequency)/3))
def run():
    seed();dev=device();reader=SouthDevelopment()
    bank=reader.fit();qv=reader.query("validation");geom=reader.geometry(qv.row_ids)
    data=prepare(bank,qv,16,"self",geom);power=FitPower.from_fit(bank.channel,bank.row_ids)
    ours_names=["ours_default","ours_delay","ours_trust","ours_frequency"]
    reports={name:json.loads((REPORT/(name+".json")).read_text()) for name in ours_names}
    selected=min(ours_names,key=lambda n:reports[n]["best_validation"]["structural_selection"])
    mlp_names=["spectral_mlp_w1","spectral_mlp_w4"]
    mlps={n:json.loads((REPORT/(n+".json")).read_text()) for n in mlp_names}
    selected_mlp=min(mlp_names,key=lambda n:mlps[n]["best_validation"]["structural_selection"])
    requested=[selected,"no_geometry_default",selected_mlp]
    records=[];regenerated={}
    for name in requested:
        r=json.loads((REPORT/(name+".json")).read_text());ct=r["contract"]
        assert ct["validation_rows"]==qv.row_ids.tolist() and ct["data_identity"]==reader.data.identity
        if ct["kind"]=="mlp":m=SpectralMLP(bank.position,power,ct["levels"],ct["hidden"])
        else:m=HallStructured(bank.spec,power,Config(**ct["config"]))
        checkpoint=torch.load(WORK/"models"/name/"best.pt",map_location="cpu",weights_only=False)
        m.load_state_dict(checkpoint["state_dict"]);m=m.to(dev);m.eval();outputs=[]
        for start in range(0,len(qv.row_ids),32):
            indices=np.arange(start,min(start+32,len(qv.row_ids)));block=slice_prepared(data,indices)
            if ct["kind"]=="mlp":
                with torch.no_grad():pred=m(block.to(dev))["channel"].cpu().numpy()
            else:
                query=reader.query("validation",block.row_ids)
                pred=m.predict(query,block)["channel"]
            outputs.append(pred)
        pred=np.concatenate(outputs)
        with np.load(WORK/"models"/name/"best_channel.npz",allow_pickle=False) as z:
            assert np.array_equal(z["validation_ids"],qv.row_ids);saved=np.array(z["validation"])
        error=float(np.sqrt(np.mean(np.abs(pred-saved)**2))/max(np.sqrt(np.mean(np.abs(saved)**2)),1e-30))
        if not np.isfinite(pred).all() or error>5e-4:raise AssertionError((name,error))
        records.append(dict(name=name,regeneration_relative_RMS_error=error,query_reference_available_during_prediction=False))
        regenerated[name]=pred;del m;torch.cuda.empty_cache()
    assert all(x["role"]=="fit" for x in reader.access)
    # Held-out development labels are made available only after regeneration.
    reader.allow_validation=True;truth=reader.reference("validation")
    for record in records:
        name=record["name"];pred=regenerated[name]
        current=metrics(torch.tensor(pred),torch.tensor(truth),power)
        r=json.loads((REPORT/(name+".json")).read_text())
        delta={k:abs(current[k]-r["best_validation"][k]) for k in ["structural_selection","covariance_NMSE","direction_cosine","frequency_correlation_MAE","beam_frequency_correlation_MAE","relative_CSI_power_MAE_db"]}
        if max(delta.values())>2e-4:raise AssertionError(delta)
        independent=literal(pred[:4],truth[:4]);pytorch=losses(torch.tensor(pred[:4]),torch.tensor(truth[:4]),power)
        differences={k:abs(independent[k]-float(pytorch[k])) for k in independent}
        if max(differences.values())>3e-6:raise AssertionError(differences)
        record.update(recomputed_metric_differences=delta,independent_numpy_differences=differences)
    before=json.loads((REPORT/"preservation_before.json").read_text());after=snapshot()
    changed=[k for k in before if k not in after or before[k]!=after[k]]
    added=sorted(set(after)-set(before))
    if changed or added:raise AssertionError(dict(changed=changed,added=added))
    prefix=(REPORT/"ledger_before.bin").read_bytes();assert (PATHS.root/"experiment_records/EXPERIMENT_RECORD.md").read_bytes().startswith(prefix)
    static=[]
    path=PATHS.code_root/"tools/check_hardcoded_paths.py"
    spec=importlib.util.spec_from_file_location("new_scope_pathcheck",path)
    checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
    for file in SOURCE.glob("*.py"):
        compile(file.read_text(),str(file),"exec")
        for i,line in enumerate(file.read_text().splitlines(),1):
            if any(p.search(line) for _,p in checker.PATTERNS):static.append(dict(file=file.name,line=i))
    if static:raise AssertionError(static)
    source_evidence=[]
    old=WORK/"runtime_revisions/before_frequency_balance"
    for file in REPORT.glob("*.json"):
        r=json.loads(file.read_text())
        if not (r.get("complete") and "contract" in r and "curve" in r):continue
        missing=[]
        for path,digest in r["contract"]["source_sha256"].items():
            current=PATHS.root/path
            match=current.is_file() and sha256(current)==digest
            archived=old/Path(path).name
            if not match:match=archived.is_file() and sha256(archived)==digest
            if not match:missing.append(path)
        if missing:raise AssertionError((file.name,missing))
        source_evidence.append(dict(name=r["name"],source_identity_recoverable=True))
    save_json(REPORT/"verification.json",dict(passed=True,selected_ours=selected,selected_spectral_mlp=selected_mlp,
        regeneration=records,preservation=dict(files=len(before),changed=changed,added_outside_stage=added,ledger_prefix=True),
        source_identity=source_evidence,source_path_failures=static,source_sha256=source_hashes(),
        access=reader.access,formal_test=False,north=False,
        scope="Engineering reproduction and same-row metric recomputation; not independent scientific replication."))
    print("VERIFIED",selected,selected_mlp,json.dumps(records),flush=True)
if __name__=="__main__":
    try:run()
    except BaseException:
        save_json(REPORT/"verification_failure.json",dict(traceback=traceback.format_exc()));raise
