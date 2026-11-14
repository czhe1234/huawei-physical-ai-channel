"""Frozen-output south development reevaluation; no fitting or checkpoint loads."""
from __future__ import annotations
import csv,json,time,traceback,resource
import numpy as np
import torch
from measured.hall_adapters.common import sha256,save_json
from measured.hall_structured_v2.data import SouthDevelopment
from measured.hall_structured_v2.foundation import prepare,Foundation
from measured.hall_metric_protocol_v1.metrics import compare,describe,_aggregate
from measured.hall_adapters.metrics import primary_metrics,aggregate_primary,directional_power
from .common import *
def auxiliary(h,truth):
    power=directional_power(h);target=directional_power(truth)
    return aggregate_primary(primary_metrics(dict(power=power),target))
def write_npz(path,**values):
    if path.exists():raise FileExistsError("Generated outputs are immutable")
    temp=path.with_name(path.name+".partial")
    with temp.open("wb") as f:np.savez_compressed(f,**values)
    temp.replace(path)
def load_rows(path,rows):
    with np.load(path,allow_pickle=False) as z:
        if "validation" not in z.files:raise ValueError("No declared validation prediction")
        p=np.array(z["validation"])
        explicit="validation_ids" in z.files
        if explicit and not np.array_equal(z["validation_ids"],rows):raise ValueError("Prediction row order changed")
    if p.shape!=(len(rows),2,8,4,1024):raise ValueError("Wrong validation shape")
    if not np.iscomplexobj(p) or not np.isfinite(p).all():raise ValueError("Invalid complex prediction")
    return p,explicit
def per_position(h,truth,rows):
    out=[]
    for i,rid in enumerate(rows):
        score=compare(h[i:i+1],truth[i:i+1])
        for panel in range(2):
            out.append(dict(row_id=int(rid),panel=panel,
                **{key:score["by_panel"][key][panel] for key in KEYS}))
    return out
def verify_position_aggregate(scores,points):
    for key in KEYS:
        by=[]
        for panel in range(2):
            values=[x[key] for x in points if x["panel"]==panel and x[key] is not None]
            by.append(float(np.mean(values)) if values else None)
        for actual,target in zip(by,scores["by_panel"][key]):
            if actual is not None and abs(actual-target)>1e-11:raise AssertionError((key,actual,target))
def literal_one(prediction,truth):
    # Independent DFT matrix and direct per-receive-link arithmetic.
    h=prediction.reshape(2,32,-1).astype(np.complex128);t=truth.reshape(2,32,-1).astype(np.complex128);f=h.shape[-1]
    hh,vv=np.meshgrid(np.arange(8),np.arange(4),indexing="ij")
    hgrid,vgrid=hh.ravel(),vv.ravel()
    w=np.exp(-2j*np.pi*(hgrid[:,None]*hgrid[None,:]/8+vgrid[:,None]*vgrid[None,:]/4))/np.sqrt(32)
    zs=[np.abs(np.einsum("ab,pbf->paf",w,x))**2 for x in [h,t]]
    ps=np.mean(np.sum(zs[0]*zs[1],axis=1)/np.sqrt(np.sum(zs[0]**2,axis=1)*np.sum(zs[1]**2,axis=1)),axis=-1)
    pp=[];joint=[];energy=[]
    for x in [h,t]:
        profiles=[]
        for panel in range(2):
            c=sum(np.vdot(x[panel,m,:-1],x[panel,m,1:]) for m in range(32))
            y=x[panel]*np.exp(-1j*np.angle(c)*np.arange(f))
            profiles.append(np.abs(np.fft.ifft(y,axis=-1))**2)
        v=np.array(profiles);joint.append(v/v.sum((-2,-1),keepdims=True))
        e=np.mean(np.abs(x)**2,axis=-1);energy.append(e/e.sum(-1,keepdims=True))
        pp.append(v)
    ds=np.mean(np.sum(pp[0]*pp[1],-1)/np.sqrt(np.sum(pp[0]**2,-1)*np.sum(pp[1]**2,-1)),axis=-1)
    return dict(pas_cosine=float(ps.mean()),relative_pdp_cosine=float(ds.mean()),
       receive_energy_TV=float((.5*np.abs(energy[0]-energy[1]).sum(-1)).mean()),
       receive_delay_joint_TV=float((.5*np.abs(joint[0]-joint[1]).sum((-2,-1))).mean()))
def run():
    init();start=time.monotonic();torch.set_num_threads(2);torch.manual_seed(20261025)
    if (REPORT/"evaluation.json").exists():raise FileExistsError("Completed evaluation exists; do not overwrite")
    reader=SouthDevelopment();bank=reader.fit();query=reader.query("validation");rows=query.row_ids
    old=json.loads((OLD_V1/"contract.json").read_text())
    assert old["data_identity"]==reader.data.identity and old["validation_rows"]==rows.tolist()
    previous=json.loads((OLD_NEW/"baseline_comparison.json").read_text())
    assert previous["data_identity"]==reader.data.identity and previous["validation_rows"]==rows.tolist()
    data=prepare(bank,query,16,"self",None)
    with torch.no_grad():
        uniform,_=Foundation()(data,torch.tensor(np.median(np.sum(np.abs(bank.channel)**2,axis=(2,3,4))/1024,axis=0),dtype=torch.float32))
    write_npz(WORK/"nearest_channel.npz",validation=data.carrier.numpy(),validation_ids=rows)
    write_npz(WORK/"uniform_experts_channel.npz",validation=uniform.numpy(),validation_ids=rows)
    # These outputs and statistical controls precede reference-label access.
    fit_pi=describe(bank.channel)["energy_fraction"]
    controls={"uniform_receive_allocation":np.full((2,32),1/32),"FIT_mean_receive_allocation":fit_pi.mean(0)}
    reader.allow_validation=True;truth=reader.reference("validation");reference=describe(truth)
    assert truth.shape==(48,2,8,4,1024)
    conditioning=dict(positions=len(rows),panels=96,numerically_valid_anchors=int(reference["reference"]["valid"].sum()),
        coherence_min=float(reference["reference"]["coherence"].min()),reference_zero_panels=int((~reference["panel_valid"]).sum()),
        reference_zero_links=int((~reference["link_valid"]).sum()))
    cases=[]
    def add(name,label,path,report=None,family="full",old_metrics=None):
        cases.append(dict(name=name,label=label,path=path,report=report,family=family,old_metrics=old_metrics))
    root=PATHS.artifacts/"measured/hall_sw_structured_development_v2"
    oldroot=PATHS.artifacts/"measured/hall_sw_development_v1/models"
    add("nearest_FIT","最近训练点",WORK/"nearest_channel.npz",old_metrics=previous["results"]["nearest_FIT"]["metrics"])
    add("uniform_frequency_experts","固定专家平均融合",WORK/"uniform_experts_channel.npz",old_metrics=previous["results"]["uniform_frequency_experts"]["metrics"])
    add("frequency_RBF","频率核插值",root/"baselines/rbf_best_channel.npz",old_metrics=previous["results"]["frequency_RBF"]["metrics"])
    labels={"ours_default":"新版含几何：原默认检查点","ours_delay":"新版含几何：宽修正范围",
        "ours_trust":"新版含几何：较低几何权重","ours_frequency":"新版含几何：强化频率监督",
        "no_geometry_default":"新版：去掉几何","spectral_mlp_w1":"位置MLP：标准损失","spectral_mlp_w4":"位置MLP：强化频率监督"}
    for name,label in labels.items():
        r=json.loads((OLD_NEW/(name+".json")).read_text())
        assert r["complete"] and not r["contract"]["health"]
        assert r["contract"]["data_identity"]==reader.data.identity and r["contract"]["validation_rows"]==rows.tolist()
        add(name,label,root/"models"/name/"best_channel.npz",r,"structured" if name.startswith("ours_") else "full",r["best_validation"])
    for name,label in [("ours_phase_safe","旧版：功率融合底座"),("ours_config0_v2","旧版：复数底座设置1"),("ours_config1_v2","旧版：复数底座设置2")]:
        r=json.loads((OLD_V1/(name+".json")).read_text())
        assert r["complete"] and not r["tiny"] and not r["test_used"] and not r["north_used"]
        add(name,label,oldroot/name/"best_channel.npz",r,"historical",r["best_validation"])
    records=[];all_points=[];independent=[]
    for case in cases:
        path=case["path"];digest=sha256(path);prediction,explicit_ids=load_rows(path,rows)
        score=compare(prediction,truth);aux=auxiliary(prediction,truth)
        expected=case["old_metrics"]
        oldcos=expected.get("direction_cosine",expected.get("mean_direction_cosine"))
        oldbeam=expected["mean_beam_loss_db"]
        parity=dict(direction_cosine_difference=abs(aux["mean_direction_cosine"]-oldcos),beam_loss_difference=abs(aux["mean_beam_loss_db"]-oldbeam))
        if max(parity.values())>5e-6:raise AssertionError((case["name"],parity))
        points=per_position(prediction,truth,rows);verify_position_aggregate(score,points)
        for point in points:all_points.append(dict(method=case["name"],**point))
        if case["name"] in ["ours_default","no_geometry_default","frequency_RBF","ours_phase_safe","spectral_mlp_w1"]:
            differences=[]
            for j in range(3):
                actual=compare(prediction[j:j+1],truth[j:j+1]);literal=literal_one(prediction[j],truth[j])
                diff={k:abs(actual[k]-literal[k]) for k in KEYS}
                if max(diff.values())>1e-8:raise AssertionError((case["name"],diff))
                differences.append(diff)
            independent.append(dict(name=case["name"],rows=rows[:3].tolist(),independent_differences=differences))
        record=dict(name=case["name"],label=case["label"],family=case["family"],status="ok",scores=score,auxiliary=aux,
            prediction_file=str(path.relative_to(PATHS.root)),prediction_sha256=digest,explicit_row_ids=explicit_ids,
            row_provenance="explicit NPZ IDs" if explicit_ids else "old fixed48-row contract plus reproduced original direction/beam scores",
            old_metric_parity=parity,best_epoch=case["report"].get("best_epoch") if case["report"] else None,
            checkpoint_selection="unchanged old metric selection; no new epoch search",
            regenerated=case["name"] in ["nearest_FIT","uniform_frequency_experts"])
        save_json(REPORT/(case["name"]+".json"),record);records.append(record)
        assert sha256(path)==digest
        print("SCORED",case["name"],json.dumps({k:score[k] for k in KEYS}),flush=True)
    # Existing broadband-only predictions remain auxiliary-only.
    broadband=[]
    for name,label in [("wrf_g64","WRF-GS+：64基元"),("wrf_g128","WRF-GS+：128基元"),("mlp_levels3","原宽带功率MLP：设置1"),("mlp_levels6","原宽带功率MLP：设置2")]:
        path=oldroot/name/"best_power.npz"
        with np.load(path,allow_pickle=False) as z:power=np.array(z["validation"])
        if power.shape!=(48,2,32):raise AssertionError("Broadband baseline shape")
        aux=aggregate_primary(primary_metrics(dict(power=power),directional_power(truth)))
        oldm=previous["results"][name]["metrics"]
        if abs(aux["mean_direction_cosine"]-oldm["mean_direction_cosine"])>5e-6:raise AssertionError("Old baseline parity")
        broadband.append(dict(name=name,label=label,status="primary_not_applicable",
            scores={k:None for k in KEYS},auxiliary=aux,prediction_file=str(path.relative_to(PATHS.root)),prediction_sha256=sha256(path),
            reason="Broadband directional beam power is not a frequency-resolved complex channel or physical receive-element power."))
    energy=[]
    for name,p in controls.items():
        distance=.5*np.abs(reference["energy_fraction"]-p[None]).sum(-1)
        aggregate,by_panel=_aggregate(distance,reference["panel_valid"])
        energy.append(dict(name=name,receive_energy_TV=aggregate,by_panel=by_panel,primary_not_applicable=True,
            joint_not_applicable=True,scope="Allocation-only fixed statistical reference; no complex signal invented."))
    # Descriptive paired position counts, without independent-block inference.
    paired=[];lookup={r["name"]:r for r in records}
    def point_values(name,key):
        result={}
        for point in all_points:
            if point["method"]==name and point[key] is not None:result.setdefault(point["row_id"],[]).append(point[key])
        return {k:float(np.mean(v)) for k,v in result.items()}
    for other in ["frequency_RBF","no_geometry_default","ours_phase_safe","spectral_mlp_w1"]:
        bymetric={}
        for key in KEYS:
            a,b=point_values("ours_default",key),point_values(other,key);ids=sorted(set(a)&set(b))
            delta=np.array([a[i]-b[i] for i in ids]);good=delta if key in KEYS[:2] else -delta
            bymetric[key]=dict(positions=len(ids),ours_better=int((good>1e-12).sum()),ties=int((np.abs(good)<=1e-12).sum()),
                ours_worse=int((good< -1e-12).sum()),mean_ours_minus_other=float(delta.mean()))
        paired.append(dict(ours="ours_default",other=other,metrics=bymetric,scope="Descriptive48-position counts in one reused block; no p-value or confirmatory CI."))
    fields=["method","row_id","panel",*KEYS]
    with (REPORT/"per_position_panel.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(all_points)
    output=dict(complete=True,scope="New-metric development exploration of fixed existing outputs.",
        checkpoint_selection_unchanged=True,new_training=False,checkpoint_loads=0,optimizer_updates=0,
        north=False,test=False,rows=rows.tolist(),data_identity=reader.data.identity,
        protocol_sha256=sha256(PROTOCOL/"metric_protocol.json"),
        metric_source_sha256=sha256(PATHS.code_root/"measured/hall_metric_protocol_v1/metrics.py"),
        reference_conditioning=conditioning,full_channel_records=records,broadband_only_records=broadband,
        receive_energy_controls=energy,paired_position_comparisons=paired,independent_arithmetic=independent,
        data_access=reader.access,excluded=["8-point tiny/health outputs: not the48-position validation role",
            "uncorrected historical first-attempt power-only outputs: superseded implementation/evaluation error; preserved, not mixed into valid models"],
        seconds=time.monotonic()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    save_json(REPORT/"evaluation.json",output)
    print("DONE",json.dumps(dict(full_outputs=len(records),power_only=len(broadband),energy_controls=len(energy),
        seconds=output["seconds"],reference=conditioning)),flush=True)
if __name__=="__main__":
    try:run()
    except BaseException:
        REPORT.mkdir(parents=True,exist_ok=True)
        save_json(REPORT/"evaluation_failure.json",dict(traceback=traceback.format_exc()));raise
