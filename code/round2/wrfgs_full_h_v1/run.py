"""Bounded verification, training and frozen scoring for WRF-GS+ full-H."""
from __future__ import annotations
import argparse, contextlib, csv, hashlib, json, math, os, platform, resource, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from project_paths import PATHS
from common.official_metrics import PointMetricStatistics, aggregate_statistics, point_metric_statistics
from round2.paper_baseline_training_v1 import common as C
from round2.wrfgs_full_h_v1.model import FullHWRFGS, spectral_loss, C0, angular_directions

METHOD="wrfgs_full_h_v1"
ART=C.ART/METHOD
REPORT=C.REPORT/"wrfgs_full_h_v1_20261110"
CONFIG=REPORT/"config.json"
QUERY_SHA="f115290de0dc5f80cfe304e1f2143696694e66eec10c418e1b086d0b179ed08b"

def source_identity():
    files=list(Path(__file__).parent.glob("*.py"))+[
        PATHS.code_root/"common/official_metrics.py",Path(C.__file__),PATHS.code_root/"round2/paper_baseline_training_v1/final_score.py",
        PATHS.code_root/"round2/paper_baseline_training_v1/wrfgs_model.py",
        PATHS.code_root/"round2/paper_baseline_training_v1/wrfgs_fast_ops.py",
        PATHS.code_root/"round2/paper_baseline_adapters_v1/wrfgs.py",
        PATHS.code_root/"round2/paper_baseline_adapters_v1/contract.py",CONFIG]
    return {PATHS.relative(p):C.sha256(p) for p in files}

def make(data,cfg,seed,device,count=None,width=None,height=None):
    C.seed_all(seed)
    return FullHWRFGS(data.geometry,data.positions(data.fit_ids),
        count=count or cfg["gaussians"],seed=seed,
        width=width or cfg["width"],height=height or cfg["height"]).to(device)

def prediction(model,data,rows):
    model.eval()
    with torch.no_grad():
        return model(data.positions(rows),data.sites(rows))

def configure_device(uuid):
    torch.set_num_threads(4)
    if uuid:
        report=C.device_guard(uuid)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        return torch.device("cuda:0"),report
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("CPU stage requires empty CUDA_VISIBLE_DEVICES")
    return torch.device("cpu"),dict(device="CPU",cuda_visible_devices="")

def verify(cfg):
    data=C.FoldData(0,"screen",METHOD)
    if C.digest_ids(data.inputs.query_ids)!=QUERY_SHA:
        raise ValueError("Frozen query set changed")
    model=make(data,cfg,42,torch.device("cpu"),count=16,width=16,height=8)
    fids=torch.tensor([0,96,191]);pixel=17
    angular=torch.zeros((1,128,8,3),dtype=torch.complex64)
    lanes=torch.arange(1,9).float()[:,None]*(1+1j*torch.tensor([.2,-.3,.7])[None])
    angular[0,pixel]=lanes
    got=model.readout(angular,1,fids).detach().numpy()
    offsets=model.offsets[1].numpy()
    direction=model.directions[1,pixel].numpy()
    freqs=model.frequencies[fids].numpy()
    phase=2*np.pi/C0*offsets@direction[:,None]*freqs[None,:]
    steering=np.exp(1j*phase)/math.sqrt(128)
    expected=(steering[None,:,None,:]*lanes.numpy().reshape(2,1,4,3)).reshape(1,256,4,3)
    error=float(np.max(np.abs(got-expected)))
    if error>3e-6:raise AssertionError(("coherent port/frequency readout",error))
    rotated=model.readout(angular*1j,1,fids).detach().numpy()
    phase_error=float(np.max(np.abs(rotated-1j*got)))
    if phase_error>3e-6:raise AssertionError("Global complex phase lost")
    rows=np.array([data.fit_ids[data.sites(data.fit_ids)==s][0] for s in (0,1)])
    output=model(data.positions(rows),data.sites(rows))
    if output.shape!=(2,256,4,192) or output.dtype!=torch.complex64 or not torch.isfinite(output).all():
        raise AssertionError("Full-H output contract")
    if output.real.std()==0 or output.imag.std()==0:
        raise AssertionError("A complex component is constant")
    identity_loss,_=spectral_loss(output,output.detach())
    if abs(float(identity_loss))>3e-6:raise AssertionError("Identity complex spectral loss")
    target=(torch.randn_like(output.real)+1j*torch.randn_like(output.real)).to(torch.complex64)
    loss,_=spectral_loss(output,target)
    loss.backward()
    gradients={n:float(p.grad.norm()) for n,p in model.named_parameters() if p.grad is not None}
    if not gradients or not all(math.isfinite(x) for x in gradients.values()):
        raise AssertionError("Gradient failure")
    for s in (0,1):
        key=f"fields.{s}.deformation.gaussian_signal.weight"
        if gradients.get(key,0)<=0:raise AssertionError("Signal head receives no gradient")
    ranks=[]
    for site in (0,1):
        camera=np.array(data.geometry.bs_rotations)[site][:,[1,2,0]]
        dirs=angular_directions(cfg["width"],cfg["height"],camera).numpy()
        offsets=model.offsets[site].numpy()
        a=np.exp(2j*np.pi/C0*float(model.frequencies[96])*(offsets@dirs.T))
        singular=np.linalg.svd(a,compute_uv=False)
        rank=int(np.sum(singular>singular[0]*1e-6))
        ranks.append(dict(site=site,rank=rank,condition=float(singular[0]/singular[-1])))
        if rank!=128:raise AssertionError(("Array readout lacks full spatial rank",ranks))
    zero=torch.zeros((1,256,4,192),dtype=torch.complex64,requires_grad=True)
    zero_loss,_=spectral_loss(zero,torch.zeros_like(zero))
    zero_loss.backward()
    if not torch.isfinite(zero_loss) or not torch.isfinite(zero.grad).all():
        raise AssertionError("Zero-channel loss/gradient")
    mixed=output.detach().clone().requires_grad_(True)
    mixed_target=target.detach().clone();mixed_target[0]=0
    mixed_loss,_=spectral_loss(mixed,mixed_target);mixed_loss.backward()
    if not torch.isfinite(mixed_loss) or not torch.isfinite(mixed.grad).all():
        raise AssertionError("Mixed zero/nonzero channel loss/gradient")
    result=dict(status="passed",zero_loss_finite=True,zero_gradient_finite=True,
        mixed_zero_gradient_finite=True,production_array_ranks=ranks,full_H_shape=list(output.shape),coherent_readout_max_abs=error,
        global_phase_equivariance_max_abs=phase_error,identity_loss=float(identity_loss),
        signed_real_and_imag=True,source_identity=source_identity(),
        query_truth_read=False,fit_labels_read=False,geometry_identity=data.geometry.geometry_fingerprint)
    C.atomic_json(REPORT/"correctness.json",result)
    print(json.dumps(result),flush=True)

def pilot(cfg,device,device_report):
    data=C.FoldData(0,"screen",METHOD)
    model=make(data,cfg,42,device);opt=model.optimizer()
    # Geometric selection only, labels remain within screen FIT.
    rows=[]
    for site in (0,1):
        ids=data.fit_ids[data.sites(data.fit_ids)==site]
        rows.extend(ids[np.linspace(0,len(ids)-1,4,dtype=int)].tolist())
    rows=np.asarray(rows,np.int64)
    target=torch.tensor(data.read_fit(rows),device=device)
    positions=torch.tensor(data.positions(rows),device=device,dtype=torch.float32)
    sites=torch.tensor(data.sites(rows),device=device)
    initial=None;history=[];start=time.monotonic()
    for step in range(1,41):
        model.train();opt.zero_grad(set_to_none=True)
        out=model(positions,sites);loss,detail=spectral_loss(out,target)
        if not bool(torch.isfinite(loss)):raise FloatingPointError("pilot loss")
        loss.backward()
        grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True))
        opt.step();model.constrain()
        if initial is None:initial=float(loss)
        if step%10==0:
            torch.cuda.synchronize()
            history.append(dict(step=step,loss=float(loss),gradient_norm=grad,**detail,
                                elapsed_seconds=time.monotonic()-start))
            print(json.dumps(dict(stage="pilot",**history[-1])),flush=True)
    model.eval()
    with torch.no_grad():before=model(positions,sites)
    path=ART/"pilot"/"checkpoint.pt"
    C.atomic_torch_save(path,dict(model=model.state_dict(),source_identity=source_identity()))
    clone=make(data,cfg,42,device)
    clone.load_state_dict(torch.load(path,map_location=device,weights_only=True)["model"])
    clone.eval()
    with torch.no_grad():after=clone(positions,sites)
    reload_error=float((after-before).abs().max())
    if reload_error!=0:raise AssertionError(("reload",reload_error))
    result=dict(status="numerically_healthy",training_objective_decreased=history[-1]["loss"]<initial,
        history=history,fit_row_ids=rows.tolist(),fit_data=data.info(),source_identity=source_identity(),
        clean_reload_max_abs=reload_error,device=device_report,
        seconds_per_update=(history[-1]["elapsed_seconds"]-history[0]["elapsed_seconds"])/30,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),formal_checkpoint=False)
    C.atomic_json(REPORT/"pilot.json",result)
    print(json.dumps(dict(stage="pilot_complete",**{k:result[k] for k in
        ("status","training_objective_decreased","seconds_per_update","peak_cuda_bytes")})),flush=True)

def checkpoint(model,opt,rng,data,cfg,step,best_step,best_score,history,order,cursor,exposure,stale,significant):
    return dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng.bit_generator.state,
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
        config=cfg,source_identity=source_identity(),data=data.info(),step=step,best_step=best_step,
        best_score=best_score,history=history,order=torch.tensor(order),cursor=cursor,
        exposure=torch.tensor(exposure),stale=stale,significant=significant)

def train_role(outer,role,cfg,device,device_report,fixed_steps=None):
    out=REPORT/f"outer{outer}"/role;art=ART/f"outer{outer}"/role
    done=out/"complete.json"
    if done.exists():
        d=json.loads(done.read_text())
        if d["source_identity"]!=source_identity() or d["config"]!=cfg:raise RuntimeError("Completed source/config drift")
        return d
    data=C.FoldData(outer,role,METHOD)
    if C.digest_ids(data.inputs.query_ids)!=QUERY_SHA:raise ValueError("Query set drift")
    seed=cfg["seed"]+1000*outer
    model=make(data,cfg,seed,device);opt=model.optimizer()
    rng=np.random.default_rng(seed);n=len(data.fit_ids)
    order=rng.permutation(n);cursor=0;exposure=np.zeros(n,np.int64)
    current=0;best_step=0;best_score=-1.;significant=-1.;history=[];stale=0
    latest=art/"latest.pt";best=art/"best.pt"
    if latest.exists():
        ck=torch.load(latest,map_location="cpu",weights_only=True)
        if ck["source_identity"]!=source_identity() or ck["config"]!=cfg:raise RuntimeError("Resume identity drift")
        model.load_state_dict(ck["model"]);opt.load_state_dict(ck["optimizer"])
        rng.bit_generator.state=ck["rng"];torch.set_rng_state(ck["torch_rng"]);torch.cuda.set_rng_state(ck["cuda_rng"])
        current=ck["step"];best_step=ck["best_step"];best_score=ck["best_score"];history=ck["history"]
        order=ck["order"].numpy();cursor=ck["cursor"];exposure=ck["exposure"].numpy()
        stale=ck["stale"];significant=ck["significant"]
        del ck
    labels=torch.empty((n,256,4,192),device=device,dtype=torch.complex64)
    for lo in range(0,n,64):
        labels[lo:lo+64]=torch.tensor(data.read_fit(data.fit_ids[lo:lo+64]),device=device)
    positions=torch.tensor(data.positions(data.fit_ids),device=device,dtype=torch.float32)
    sites=torch.tensor(data.sites(data.fit_ids),device=device)
    ceiling=cfg["max_steps"] if role=="screen" else int(fixed_steps)
    C.atomic_json(out/"contract.json",dict(config=cfg,source_identity=source_identity(),
        data=data.info(),model=model.metadata(),ceiling=ceiling,device=device_report,seed=seed))
    losses=[];grads=[];wall=time.monotonic();last_print=wall
    while current<ceiling:
        take=[];remaining=cfg["batch_positions"]
        while remaining:
            end=min(n,cursor+remaining);take.extend(order[cursor:end].tolist())
            remaining-=end-cursor;cursor=end
            if cursor==n:order=rng.permutation(n);cursor=0
        ix=np.asarray(take,np.int64);np.add.at(exposure,ix,1)
        tid=torch.tensor(ix,device=device)
        model.train();opt.zero_grad(set_to_none=True)
        pred=model(positions[tid],sites[tid]);target=labels[tid]
        loss,detail=spectral_loss(pred,target)
        if not bool(torch.isfinite(loss)):raise FloatingPointError("training loss")
        loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True))
        opt.step();model.constrain();current+=1
        losses.append(float(loss));grads.append(grad)
        if time.monotonic()-last_print>30:
            print(json.dumps(dict(outer=outer,role=role,step=current,ceiling=ceiling,
                loss=float(loss),elapsed_seconds=time.monotonic()-wall)),flush=True)
            last_print=time.monotonic()
        if current%cfg["validation_interval"] and current!=ceiling:continue
        metrics=None
        if role=="screen":
            metrics=C.evaluate(data,lambda ids:prediction(model,data,ids),batch_size=2)
            if metrics["score"]>best_score:
                best_score=metrics["score"];best_step=current
                C.atomic_torch_save(best,dict(model=model.state_dict(),step=current,metrics=metrics,
                    config=cfg,source_identity=source_identity(),data=data.info()))
            if metrics["score"]>significant+cfg["min_delta"]:
                significant=metrics["score"];stale=0
            else:stale+=1
        event=dict(step=current,loss_mean=float(np.mean(losses)),gradient_mean=float(np.mean(grads)),
            validation=metrics,best_step=best_step,best_score=best_score if role=="screen" else None,
            elapsed_seconds=time.monotonic()-wall,exposure_min=int(exposure.min()),exposure_max=int(exposure.max()))
        history.append(event);losses=[];grads=[]
        C.atomic_torch_save(latest,checkpoint(model,opt,rng,data,cfg,current,best_step,best_score,history,
            order,cursor,exposure,stale,significant))
        C.atomic_json(out/"progress.json",dict(status="running",outer=outer,role=role,**event))
        print(json.dumps(dict(outer=outer,role=role,**event)),flush=True)
        if role=="screen" and current>=cfg["min_steps"] and stale>=cfg["patience"]:break
    del labels
    if role=="screen":
        selected=torch.load(best,map_location=device,weights_only=True)
        model.load_state_dict(selected["model"]);selected_steps=int(selected["step"])
        full_val=C.evaluate(data,lambda ids:prediction(model,data,ids),ids=data.val_ids,batch_size=2)
        model_path=best
    else:
        selected_steps=current;full_val=None;model_path=art/"final.pt"
        C.atomic_torch_save(model_path,dict(model=model.state_dict(),step=current,config=cfg,
            source_identity=source_identity(),data=data.info()))
    result=dict(status="complete",outer=outer,role=role,config=cfg,source_identity=source_identity(),
        completed_updates=current,selected_updates=selected_steps,history=history,
        full_validation_diagnostic=full_val,wall_seconds=time.monotonic()-wall,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),device=device_report,
        model_path=PATHS.relative(model_path),model_sha256=C.sha256(model_path),data=data.info(),
        stop_reason="fixed_refit_budget" if role=="refit" else
            ("validation_plateau" if current>=cfg["min_steps"] and stale>=cfg["patience"] else "max_updates"),
        exposure=dict(min=int(exposure.min()),max=int(exposure.max()),total=int(exposure.sum())),
        query_truth_read=False)
    if role=="refit":
        folder=ART/f"outer{outer}"/"prediction";folder.mkdir(parents=True,exist_ok=True)
        parts=[]
        for lo in range(0,len(data.outer_ids),cfg["prediction_batch"]):
            rows=data.outer_ids[lo:lo+cfg["prediction_batch"]]
            values=prediction(model,data,rows).detach().cpu().numpy()*data.scale
            if not np.isfinite(values).all():raise FloatingPointError("Prediction nonfinite")
            parts.append(values)
        values=np.concatenate(parts).astype(np.complex64)
        C.atomic_npy(folder/"channels.npy",values)
        C.atomic_npy(folder/"row_ids.npy",data.outer_ids)
        manifest=dict(schema="canonical-baseline-prediction-v1",method=METHOD,outer=outer,
            input_identity=data.inputs.identity,row_ids=data.outer_ids.tolist(),
            row_ids_sha256=C.digest_ids(data.outer_ids),prediction_namespace="canonical_train_row_ids",
            kind="channel",path=PATHS.relative(folder/"channels.npy"),sha256=C.sha256(folder/"channels.npy"),
            shape=list(values.shape),dtype=str(values.dtype),fit_data=data.info(),
            metadata=dict(source_identity=source_identity(),model_path=result["model_path"],
                model_sha256=result["model_sha256"],model=model.metadata(),config=cfg),
            query_truth_read=False,trained=True)
        C.atomic_json(folder/"manifest.json",manifest)
        result["prediction"]=PATHS.relative(folder/"manifest.json")
    C.atomic_json(done,result)
    print(json.dumps(dict(stage="role_complete",outer=outer,role=role,
        selected_updates=selected_steps,completed_updates=current,wall_seconds=result["wall_seconds"])),flush=True)
    del model,opt
    torch.cuda.empty_cache()
    return result

def score(include_new):
    from round2.paper_baseline_training_v1 import final_score as S
    data=C.SimulationInputs()
    if C.digest_ids(data.query_ids)!=QUERY_SHA:raise ValueError("Query set drift")
    S.REPORT=REPORT/"evaluation";S.SCORE_DIR=S.REPORT/"final_scores"
    methods=list(S.METHODS)+([METHOD] if include_new else [])
    for method in methods:
        result=S.score_method(method,data)
        if result is None:raise RuntimeError("Missing predictions: "+method)
        print(json.dumps(dict(stage="scored",method=method,metrics=result["metrics"])),flush=True)
    old=json.loads((C.REPORT/"comparison.json").read_text())
    rows=[dict(method="V16（历史结果保留）",key="v16",metrics=old["rows"][0]["metrics"],
        origin="archived_summary_not_rescored")]
    labels=dict(S.LABELS);labels[METHOD]="WRF-GS+（完整H适配）"
    diffs={}
    for method in methods:
        d=json.loads((S.SCORE_DIR/method/"results.json").read_text())
        rows.append(dict(method=labels[method],key=method,metrics=d["metrics"],origin="fresh_raw_prediction_scoring"))
        previous=next((r for r in old["rows"] if r["key"]==method),None)
        if previous:
            diffs[method]={k:(None if v is None else d["metrics"][k]-v)
                for k,v in previous["metrics"].items()}
    def cell(row,key):
        v=row["metrics"].get(key)
        return "不适用" if v is None else f"{v:.6f}"
    lines=["# 华为仿真本地500点比较","","| 方法 | PAS ↑ | PDP ↑ | NMSE（线性）↓ | 总分 ↑ |",
        "|---|---:|---:|---:|---:|"]
    for row in rows:lines.append("| "+" | ".join([row["method"]]+[cell(row,k) for k in ("pas","pdp","nmse","score")])+" |")
    lines+=["","本轮从保存的预测重新评分；V16保留历史汇总，不重训、不重新评分。",
        "四类文献方法均为任务适配。WRF-GS+功率版与新增完整H版分别保留。",
        "V16历史训练范围与基线存在已记录的差异，此表不证明严格相同训练划分下的领先。",
        "500点、31个零信道点全部保留；不使用评价真值校正相位、时延或幅度。",
        "总分=0.4 PAS+0.4 PDP+0.2/(1+global NMSE)。"]
    S.REPORT.mkdir(parents=True,exist_ok=True)
    (S.REPORT/"comparison.md").write_text("\n".join(lines)+"\n")
    C.atomic_json(S.REPORT/"comparison.json",dict(rows=rows,old_metric_deltas=diffs,
        query_ids_sha256=QUERY_SHA,source_identity=source_identity(),new_full_H_included=include_new))
    with (S.REPORT/"comparison.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["method","PAS","PDP","NMSE","score","origin"])
        for r in rows:w.writerow([r["method"]]+[r["metrics"].get(k) for k in ("pas","pdp","nmse","score")]+[r["origin"]])

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--stage",choices=["verify","pilot","train","score_old","score_all"],required=True)
    p.add_argument("--folds",default="0");p.add_argument("--device-uuid")
    args=p.parse_args()
    cfg=json.loads(CONFIG.read_text())
    device,device_report=configure_device(args.device_uuid)
    try:
        if args.stage=="verify":verify(cfg)
        elif args.stage=="pilot":pilot(cfg,device,device_report)
        elif args.stage=="train":
            if not args.device_uuid:raise ValueError("Training GPU UUID required")
            if json.loads((REPORT/"correctness.json").read_text())["status"]!="passed":raise RuntimeError("Correctness not passed")
            if json.loads((REPORT/"pilot.json").read_text())["status"]!="numerically_healthy":raise RuntimeError("Pilot failed")
            for outer in map(int,args.folds.split(",")):
                screened=train_role(outer,"screen",cfg,device,device_report)
                ns=len(C.FoldData(outer,"screen",METHOD).fit_ids)
                nr=len(C.FoldData(outer,"refit",METHOD).fit_ids)
                budget=math.ceil(screened["selected_updates"]*nr/ns)
                train_role(outer,"refit",cfg,device,device_report,fixed_steps=budget)
        else:score(args.stage=="score_all")
    except BaseException:
        REPORT.mkdir(parents=True,exist_ok=True)
        with (REPORT/f"failure_{args.stage}_{args.folds.replace(',','_')}.log").open("a") as f:
            f.write(traceback.format_exc()+"\n")
        raise

if __name__=="__main__":
    main()
