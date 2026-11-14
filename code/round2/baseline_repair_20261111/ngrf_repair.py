"""Paired FIT-only nGRF phase-weight repair; old sources remain read-only."""
from __future__ import annotations
import argparse,hashlib,json,math,os,time,traceback
from pathlib import Path
import numpy as np
import torch
from project_paths import PATHS
from round2.paper_baseline_training_v1 import common as C
from round2.paper_baseline_training_v1 import native_ngrf as N

BASE=PATHS.analysis/"results/round2_paper_four_baselines_v1/repair_20261111"
REPORT=BASE/"ngrf"
ART=PATHS.artifacts/"round2/paper_four_baselines_v1/repair_20261111/ngrf"
PANEL=np.array([2,565,1120,1717,2295,2877,3447,3998,0,558,1124,1693,2286,2846,3415,3999],np.int64)
COEF=.01603005049928609

def sha(p):return C.sha256(p)
def identity():
    files=[Path(__file__),Path(N.__file__),Path(C.__file__),BASE/"config.json",
        PATHS.code_root/"round2/paper_baseline_training_v1/field_fast.py",
        PATHS.code_root/"round2/paper_baseline_adapters_v1/field_models.py"]
    return {PATHS.relative(p):sha(p) for p in files}
def tensor_hash(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def model_hash(model):
    h=hashlib.sha256()
    for name,x in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode())
        h.update(x.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def raw_hash(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def jsonwrite(p,d):C.atomic_json(p,d)
def model_build(data,device):
    model=N.NativeNGRF(data.geometry,data.positions(data.fit_ids),data.sites(data.fit_ids)).to(device)
    for field in model.fields.values():field.native.device=device
    return model
def learning_rate(opt,step):
    frac=min(max(step/15000.,0.),1.)
    xyz=math.exp(math.log(.0002696363015043561)*(1-frac)+math.log(5.953443116739323e-06)*frac)
    for g in opt.param_groups:
        if g["name"].endswith("/xyz"):g["lr"]=0. if step>=int(15000*.46840533026053566) else xyz
def loss_components(value,target,reg,weight):
    den=target.abs().square().sum().clamp_min(1e-10)
    mag=(value.abs()-target.abs()).square().sum()/den
    comp=(value-target).abs().square().sum()/den
    loss=mag+weight*comp+COEF*reg
    return loss,dict(magnitude_nmse=float(mag.detach()),complex_nmse=float(comp.detach()),
        activation_logit_L1=float(reg.detach()),prediction_energy=float(value.abs().square().mean().detach()),
        target_energy=float(target.abs().square().mean().detach()))
def schedule(data,steps):
    rng=np.random.default_rng(42);indices=[];tones=[]
    for _ in range(steps):
        tones.append(np.sort(rng.choice(192,8,replace=False)))
        ix=[]
        for site in (0,1):
            valid=np.flatnonzero(data.sites(PANEL)==site)
            ix.extend(rng.choice(valid,size=len(valid),replace=False).tolist())
        indices.append(ix)
    return np.array(indices,np.int64),np.array(tones,np.int64)
def schedule_id(indices,tones):return dict(point_indices_sha256=raw_hash(indices),frequency_ids_sha256=raw_hash(tones))
def stats(value,target,sites,scale,row_ids=None):
    v=np.asarray(value,np.complex128);t=np.asarray(target,np.complex128)
    assert v.shape==t.shape and v.shape[1:]==(256,4,192)
    assert np.isfinite(v).all() and np.isfinite(t).all()
    axis=(1,2,3);te=np.sum(np.abs(t)**2,axis);pe=np.sum(np.abs(v)**2,axis)
    se=np.sum(np.abs(v-t)**2,axis);me=np.sum((np.abs(v)-np.abs(t))**2,axis)
    dot=np.sum((v.conj()*t).real,axis)
    def pooled(ix):
        en=float(te[ix].sum());pred=float(pe[ix].sum())
        return dict(count=int(np.sum(ix)),complex_nmse=float(se[ix].sum()/en) if en else None,
            magnitude_nmse=float(me[ix].sum()/en) if en else None,
            prediction_to_truth_energy=pred/en if en else None,
            signed_complex_correlation_real=float(dot[ix].sum()/math.sqrt(en*pred)) if en*pred else None,
            prediction_mean_power_FIT_RMS_squared=pred/max(int(np.sum(ix))*256*4*192,1),
            prediction_mean_power_raw=pred*scale**2/max(int(np.sum(ix))*256*4*192,1))
    allrows=np.ones(len(t),bool);nonzero=te>0;strength=np.full(len(t),-np.inf)
    for site in (0,1):
        ix=(sites==site)&nonzero
        if ix.any():strength[ix]=10*np.log10(te[ix]/te[ix].max())
    per=[]
    for i,row in enumerate(PANEL if row_ids is None else row_ids):
        per.append(dict(row_id=int(row),site=int(sites[i]),zero_truth=bool(not nonzero[i]),
            strength_relative_site_max_dB=float(strength[i]) if nonzero[i] else None,
            **pooled(np.arange(len(t))==i)))
    return dict(pooled=pooled(allrows),per_site={str(s):pooled(sites==s) for s in (0,1)},
        per_point=per,energy_groups={name:pooled(ix) for name,ix in {
            "strong_ge_minus20dB":nonzero&(strength>=-20),
            "medium_minus40_to_minus20dB":nonzero&(strength< -20)&(strength>=-40),
            "weak_lt_minus40dB":nonzero&(strength< -40),"exact_zero":~nonzero}.items()},
        shape=list(v.shape),all_finite=True,zero_target_count=int((~nonzero).sum()),
        power_units="H divided by parent FIT scalar RMS; raw power also reported",
        phase_alignment=False,not_a_full_training_set_assessment=True)
@torch.no_grad()
def predict(model,data,rows,device,batch=2):
    model.eval();pieces=[]
    freq=torch.tensor(data.frequencies(),dtype=torch.float64,device=device)
    for lo in range(0,len(rows),batch):
        ids=rows[lo:lo+batch]
        pieces.append(model(torch.tensor(data.positions(ids),dtype=torch.float64,device=device),
            torch.tensor(data.sites(ids),dtype=torch.long,device=device),freq).cpu().numpy())
    return np.concatenate(pieces)
def verify():
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise RuntimeError("CPU verify requires empty CUDA mask")
    torch.set_num_threads(4);torch.manual_seed(42)
    target=torch.complex(torch.randn(3,4,5),torch.randn(3,4,5));target[0]=0
    value=torch.complex(torch.randn_like(target.real),torch.randn_like(target.real)).requires_grad_(True)
    reg=torch.tensor(.03)
    loss,_=loss_components(value,target,reg,.1)
    den=target.abs().square().sum().clamp_min(1e-10)
    reference=((value.abs()-target.abs()).square().sum()+.1*(value-target).abs().square().sum())/den+COEF*reg
    torch.testing.assert_close(loss,reference,rtol=2e-7,atol=1e-7)
    loss.backward();assert torch.isfinite(value.grad).all()
    zero=torch.zeros((2,3,4),dtype=torch.complex64,requires_grad=True)
    zloss,_=loss_components(zero,torch.zeros_like(zero),reg,1.);zloss.backward()
    assert torch.isfinite(zloss) and torch.isfinite(zero.grad).all()
    native=N.equivalence_check()
    result=dict(status="passed",source_loss_equivalence=True,mixed_zero_gradients_finite=True,
        all_zero_gradients_finite=True,native_renderer=native,source_identity=identity(),GPU_used=False)
    jsonwrite(REPORT/"cpu_checks.json",result);print(json.dumps(result),flush=True)

def run(args):
    expected=identity();cfg=json.loads((BASE/"config.json").read_text())["ngrf"]
    device_info=C.device_guard(args.gpu_uuid);device=torch.device("cuda:0")
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=C.FoldData(0,"screen","ngrf")
    assert set(PANEL).issubset(set(data.fit_ids)) and not set(PANEL).intersection(data.outer_ids)
    if json.loads((REPORT/"cpu_checks.json").read_text())["status"]!="passed":raise RuntimeError("CPU checks required")
    smoke=args.mode=="smoke";root=REPORT/("smoke" if smoke else "paired");art=ART/("smoke" if smoke else "paired")
    steps=20 if smoke else int(cfg["updates"]);weights=[1.] if smoke else list(map(float,cfg["weights"]))
    root.mkdir(parents=True,exist_ok=True);art.mkdir(parents=True,exist_ok=True)
    cap=120. if smoke else float(cfg["max_wall_seconds"])
    if (root/"results.json").exists():
        done=json.loads((root/"results.json").read_text())
        if done["status"]=="complete":
            assert done["source_identity"]==expected;print(json.dumps({"reused":str(root/"results.json")}));return
    target_np=data.read_fit(PANEL);target=torch.tensor(target_np,device=device)
    sites=data.sites(PANEL);positions=torch.tensor(data.positions(PANEL),dtype=torch.float64,device=device)
    site_tensor=torch.tensor(sites,dtype=torch.long,device=device)
    ids,ff=schedule(data,steps);sid=schedule_id(ids,ff)
    schedule_path=art/"draw_schedule.npz"
    if not schedule_path.exists():
        with schedule_path.open("xb") as f:np.savez(f,point_indices=ids,frequency_ids=ff,panel=PANEL)
    else:
        with np.load(schedule_path,allow_pickle=False) as z:
            assert np.array_equal(z["point_indices"],ids) and np.array_equal(z["frequency_ids"],ff)
    if not (art/"labels_raw.npy").exists():C.atomic_npy(art/"labels_raw.npy",(target_np*data.scale).astype(np.complex64))
    init_path=art/"initial.pt"
    if not init_path.exists():
        C.seed_all(42);m=model_build(data,device)
        init=dict(model={k:v.cpu() for k,v in m.state_dict().items()},torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(),model_hash=model_hash(m),source_identity=expected,
            cpu_rng_hash=tensor_hash(torch.get_rng_state()),cuda_rng_hash=tensor_hash(torch.cuda.get_rng_state()))
        C.atomic_torch_save(init_path,init);del m;torch.cuda.empty_cache()
    init=torch.load(init_path,map_location="cpu",weights_only=True)
    assert init["source_identity"]==expected
    sources=dict(source_identity=expected,schedule=sid,initial_model_hash=init["model_hash"],
        initial_cpu_rng_hash=init["cpu_rng_hash"],initial_cuda_rng_hash=init["cuda_rng_hash"],
        data=data.info(),panel=PANEL.tolist(),config=cfg,formal_test=False,device=device_info)
    jsonwrite(root/"run_manifest.json",sources)
    wall_previous=0.
    for path in art.glob("weight_*/latest.pt"):
        ck=torch.load(path,map_location="cpu",weights_only=True)
        wall_previous=max(wall_previous,float(ck["wall_consumed_seconds"]))
        del ck
    started=time.monotonic();result_rows=[]
    for weight in weights:
        tag="weight_"+str(weight).replace(".","p");out=root/tag;store=art/tag
        out.mkdir(parents=True,exist_ok=True);store.mkdir(parents=True,exist_ok=True)
        result_path=out/"result.json"
        if result_path.exists():
            old=json.loads(result_path.read_text())
            assert old["source_identity"]==expected and old["schedule"]==sid
            result_rows.append(old);continue
        model=model_build(data,device);model.load_state_dict(init["model"])
        assert model_hash(model)==init["model_hash"]
        opt=N.native_optimizer(model);history=[];current=0
        torch.set_rng_state(init["torch_rng"]);torch.cuda.set_rng_state(init["cuda_rng"])
        start_rng=dict(cpu=tensor_hash(torch.get_rng_state()),cuda=tensor_hash(torch.cuda.get_rng_state()))
        assert start_rng==dict(cpu=init["cpu_rng_hash"],cuda=init["cuda_rng_hash"])
        latest=store/"latest.pt"
        if latest.exists():
            ck=torch.load(latest,map_location="cpu",weights_only=True)
            assert ck["source_identity"]==expected and ck["schedule"]==sid and ck["weight"]==weight
            model.load_state_dict(ck["model"]);opt.load_state_dict(ck["optimizer"]);current=ck["step"];history=ck["history"]
            torch.set_rng_state(ck["torch_rng"]);torch.cuda.set_rng_state(ck["cuda_rng"])
            del ck
        last_print=time.monotonic();last_detail={};last_grad=0.
        def checkpoint():
            C.atomic_torch_save(latest,dict(model=model.state_dict(),optimizer=opt.state_dict(),step=current,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),weight=weight,history=history,
                schedule=sid,source_identity=expected,wall_consumed_seconds=wall_previous+time.monotonic()-started))
        if current==0:
            before=predict(model,data,PANEL,device)
            history.append(dict(step=0,FIT=stats(before,target_np,sites,data.scale)))
        for step in range(current+1,steps+1):
            model.train();opt.zero_grad(set_to_none=True);learning_rate(opt,step)
            pi=torch.tensor(ids[step-1],device=device);fi=torch.tensor(ff[step-1],device=device)
            freq=torch.tensor(data.frequencies(ff[step-1]),device=device,dtype=torch.float64)
            value=model(positions[pi],site_tensor[pi],freq)
            truth=target[pi][:,:,:,fi]
            reg=torch.stack([field.activation_penalty() for field in model.fields.values()]).mean()
            loss,detail=loss_components(value,truth,reg,weight)
            if not bool(torch.isfinite(loss)):raise FloatingPointError("Nonfinite loss")
            loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True))
            opt.step();current=step;last_detail=detail;last_grad=grad
            elapsed=wall_previous+time.monotonic()-started
            full=(step%int(cfg["eval_every"])==0 or step==steps)
            if full:
                pred=predict(model,data,PANEL,device)
                event=dict(step=step,FIT=stats(pred,target_np,sites,data.scale),training_batch=detail,
                    gradient_norm=grad,wall_consumed_seconds=elapsed)
                history.append(event);jsonwrite(out/"progress.json",event);checkpoint()
                print(json.dumps(dict(weight=weight,step=step,total=steps,FIT=event["FIT"]["pooled"],wall_seconds=elapsed)),flush=True)
            if time.monotonic()-last_print>=30:
                checkpoint();print(json.dumps(dict(weight=weight,step=step,total=steps,loss=float(loss),
                    gradient_norm=grad,wall_seconds=elapsed)),flush=True);last_print=time.monotonic()
            if elapsed>cap and step<steps:
                checkpoint();jsonwrite(root/"budget_cap.json",dict(status="budget_cap",weight=weight,step=step,total=steps,wall_seconds=elapsed))
                print(json.dumps(dict(status="budget_cap",weight=weight,step=step)),flush=True);return
        pred=predict(model,data,PANEL,device);final=stats(pred,target_np,sites,data.scale)
        C.atomic_torch_save(store/"final.pt",dict(model=model.state_dict(),weight=weight,step=current,
            source_identity=expected,schedule=sid,model_hash=model_hash(model),metadata=model.metadata()))
        clone=model_build(data,device)
        clone.load_state_dict(torch.load(store/"final.pt",map_location=device,weights_only=True)["model"])
        replay=predict(clone,data,PANEL,device)
        relative=float(np.linalg.norm((replay-pred).reshape(-1))/max(np.linalg.norm(pred.reshape(-1)),1e-30))
        assert relative<=1e-6,("clean reload mismatch",relative)
        raw=(pred*data.scale).astype(np.complex64);assert raw.shape==(16,256,4,192) and np.isfinite(raw).all()
        C.atomic_npy(store/"FIT_raw_H.npy",raw);C.atomic_npy(store/"row_ids.npy",PANEL)
        result=dict(status="complete",weight=weight,updates=current,final=final,history=history,
            initial_model_hash=init["model_hash"],initial_rng_hashes=start_rng,schedule=sid,source_identity=expected,
            raw_prediction=PATHS.relative(store/"FIT_raw_H.npy"),raw_prediction_sha256=sha(store/"FIT_raw_H.npy"),
            checkpoint=PATHS.relative(store/"final.pt"),checkpoint_sha256=sha(store/"final.pt"),
            clean_reload_relative_L2=relative,training_phase_supervision=True,test_phase_alignment=False,
            data=data.info(),model_metadata=model.metadata(),device=device_info,
            peak_cuda_bytes=torch.cuda.max_memory_allocated(),wall_consumed_seconds=wall_previous+time.monotonic()-started,
            formal_test=False,all_point_health_claim=False)
        jsonwrite(result_path,result);result_rows.append(result)
        checkpoint();del model,clone,opt;torch.cuda.empty_cache()
    verdict=None
    if not smoke:
        a,b=result_rows
        assert a["initial_model_hash"]==b["initial_model_hash"] and a["initial_rng_hashes"]==b["initial_rng_hashes"] and a["schedule"]==b["schedule"]
        av=a["final"]["pooled"];bv=b["final"]["pooled"]
        improvement=1-bv["complex_nmse"]/av["complex_nmse"]
        passed=improvement>=.1 and bv["complex_nmse"]<=.5 and bv["magnitude_nmse"]<=.05
        verdict=dict(preliminary_gate_passed=bool(passed),complex_relative_improvement=improvement,
            criteria="FIT complex NMSE <=0.5, >=10percent relative improvement, magnitudeNMSE<=0.05",
            all_point_health_claim=False,validation_used_for_selection=False)
    validation=None
    if not smoke:
        from common.official_metrics import point_metric_statistics_torch,aggregate_statistics
        valrows=data.probe_ids
        raw_target=data.read_validation(valrows,normalized=False)
        validation=dict(role="fixed_endpoint_development_validation",row_ids=valrows.tolist(),
            checkpoint_selection=False,weight_selection=False,phase_alignment=False,variants=[])
        for row in result_rows:
            model=model_build(data,device)
            model.load_state_dict(torch.load(PATHS.root/row["checkpoint"],map_location=device,weights_only=True)["model"])
            normalized=predict(model,data,valrows,device)
            raw=(normalized*data.scale).astype(np.complex64)
            path=ART/"paired"/("weight_"+str(row["weight"]).replace(".","p"))/"validation_raw_H.npy"
            C.atomic_npy(path,raw)
            detail=stats(normalized,raw_target/data.scale,data.sites(valrows),data.scale,valrows)
            official=aggregate_statistics(point_metric_statistics_torch(
                torch.tensor(raw,device=device),torch.tensor(raw_target,device=device)))
            validation["variants"].append(dict(weight=row["weight"],FIT_endpoint_checkpoint=row["checkpoint"],
                checkpoint_sha256=row["checkpoint_sha256"],diagnostics=detail,official_metrics=official,
                raw_prediction=PATHS.relative(path),raw_prediction_sha256=sha(path)))
            del model;torch.cuda.empty_cache()
        jsonwrite(root/"validation.json",validation)
    final=dict(status="complete",mode=args.mode,source_identity=expected,config=cfg,panel=PANEL.tolist(),
        variants=result_rows,verdict=verdict,wall_seconds=wall_previous+time.monotonic()-started,
        formal_test=False,validation=validation,validation_labels_read=data.info()["validation_read_count"],outer_truth_read=False)
    jsonwrite(root/"results.json",final)
    print(json.dumps(dict(status="complete",mode=args.mode,verdict=verdict)),flush=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--mode",choices=["verify","smoke","train"],required=True)
    ap.add_argument("--gpu-uuid");args=ap.parse_args()
    REPORT.mkdir(parents=True,exist_ok=True);ART.mkdir(parents=True,exist_ok=True)
    try:
        if args.mode=="verify":verify()
        else:
            if not args.gpu_uuid:raise ValueError("GPU UUID required")
            run(args)
    except BaseException:
        with (REPORT/("failure_"+args.mode+".log")).open("a") as f:f.write(traceback.format_exc()+"\n")
        raise
if __name__=="__main__":main()
