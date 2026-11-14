"""Formal five-fold nGRF: all allowed FIT, repaired relative complex objective."""
from __future__ import annotations
import argparse,copy,hashlib,json,math,os,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
from project_paths import PATHS
from round2.paper_baseline_training_v1 import common as C
from round2.paper_baseline_training_v1 import native_ngrf as N
from round2.paper_baseline_adapters_v1 import contract as K
from round2.baseline_repair_20261111 import optimizer_copy_guard as G
from common import official_metrics as M

OUT=C.REPORT/"formal_20261111/ngrf"
ART=C.ART/"formal_20261111/ngrf"
GPU="GPU-70b6fceb-c7c4-9467-3c81-097c9284c160"
def identity(config):
    files=[Path(__file__),Path(C.__file__),Path(N.__file__),Path(K.__file__),Path(G.__file__),
        Path(M.__file__),PATHS.code_root/"round2/paper_baseline_training_v1/field_fast.py",
        PATHS.code_root/"round2/paper_baseline_adapters_v1/field_models.py",config]
    return {PATHS.relative(p):C.sha256(p) for p in files}
def emit(value):print(json.dumps(value,ensure_ascii=False,allow_nan=False),flush=True)
def actual_dropout(m):return {n:float(x.p) for n,x in m.named_modules() if isinstance(x,torch.nn.modules.dropout._DropoutNd)}
def build(data,device):
    m=N.NativeNGRF(data.geometry,data.positions(data.fit_ids),data.sites(data.fit_ids)).to(device)
    for field in m.fields.values():field.native.device=device
    for x in m.modules():
        if isinstance(x,torch.nn.modules.dropout._DropoutNd):x.p=0.
    assert len(actual_dropout(m))==18 and set(actual_dropout(m).values())=={0.}
    return m
def restore(m,state):
    m.load_state_dict(copy.deepcopy(state["model"]))
    for x in m.modules():
        if isinstance(x,torch.nn.modules.dropout._DropoutNd):x.p=0.
    assert actual_dropout(m)==state["actual_dropout"]
def tensor_hash(x):return G.fingerprint(x)
def model_hash(m):return G.fingerprint(m.state_dict())
def xyz_schedule(opt,step):
    fraction=min(max(step/15000.,0.),1.)
    lr=math.exp(math.log(.0002696363015043561)*(1-fraction)+math.log(5.953443116739323e-06)*fraction)
    for group in opt.param_groups:
        if group["name"].endswith("/xyz"):group["lr"]=0. if step>=7026 else lr
def relative_objective(value,label,fixed_power,reg,cfg):
    mag=(value.abs()-label.abs()).square().mean((1,2,3))
    comp=(value-label).abs().square().mean((1,2,3))
    return ((mag+cfg["complex_weight"]*comp)/fixed_power.clamp_min(cfg["training_power_floor"])).mean()+cfg["activation_L1"]*reg
def global_objective(value,label,reg,cfg):
    den=label.abs().square().sum().clamp_min(cfg["warmup_global_denominator_floor"])
    mag=(value.abs()-label.abs()).square().sum()/den
    comp=(value-label).abs().square().sum()/den
    return mag+cfg["complex_weight"]*comp+cfg["activation_L1"]*reg
def phase_for_step(step,cfg):
    return "global" if step<=cfg["warmup_global_steps"] else "relative"
def objective(value,label,fixed_power,reg,cfg,phase):
    if phase=="global":return global_objective(value,label,reg,cfg)
    if phase=="relative":return relative_objective(value,label,fixed_power,reg,cfg)
    raise ValueError("Unknown objective phase")
def schedule(sites,steps,cfg):
    rng=np.random.default_rng(cfg["seed"]);pools=[np.flatnonzero(sites==s) for s in [0,1]]
    order=[np.empty(0,np.int64),np.empty(0,np.int64)];cursor=[0,0]
    indices=np.empty((steps,cfg["batch_size"]),np.int64);tones=np.empty((steps,cfg["tones_per_step"]),np.int64)
    for step in range(steps):
        row=[]
        for s in [0,1]:
            want=cfg["per_site_batch"];chunk=[]
            while want:
                if cursor[s]==len(order[s]):order[s]=rng.permutation(pools[s]);cursor[s]=0
                take=min(want,len(order[s])-cursor[s]);chunk.extend(order[s][cursor[s]:cursor[s]+take]);cursor[s]+=take;want-=take
            row.extend(chunk)
        indices[step]=row;tones[step]=np.sort(rng.choice(192,cfg["tones_per_step"],replace=False))
    return indices,tones
def array_hash(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def schedule_id(ids,ff):return dict(indices=array_hash(ids),frequencies=array_hash(ff))
def gpu_guard(args,cfg):
    if args.gpu_uuid!=GPU or os.environ.get("CUDA_VISIBLE_DEVICES")!=GPU:raise RuntimeError("Only assigned A800 UUID is authorized")
    free=int(subprocess.check_output(["nvidia-smi","-i",GPU,"--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
    if free<cfg["minimum_free_GPU_GiB"]*1024:raise RuntimeError("Insufficient free memory on assigned A800")
    info=C.device_guard(GPU);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.reset_peak_memory_stats();return torch.device("cuda:0"),info
def cache_fit(data,device,cfg):
    started=time.monotonic();rows=data.fit_ids;n=len(rows);power=np.empty(n,np.float64)
    cache=torch.empty((n,192,256,4),dtype=torch.complex64,device=device);last_progress=started
    for lo in range(0,n,cfg["cache_chunk_size"]):
        batch=rows[lo:lo+cfg["cache_chunk_size"]];h=data.read_fit(batch)
        z=h.astype(np.complex128);power[lo:lo+len(batch)]=np.mean(abs(z)**2,axis=(1,2,3))
        cache[lo:lo+len(batch)].copy_(torch.from_numpy(np.ascontiguousarray(h.transpose(0,3,1,2))).to(device))
        if time.monotonic()-last_progress>=30:
            emit(dict(method="ngrf",stage="FIT_cache",outer=data.outer,role=data.role,rows=lo+len(batch),total=n,wall_seconds=time.monotonic()-started));last_progress=time.monotonic()
    torch.cuda.synchronize()
    assert np.isfinite(power).all() and np.all(power>=0)
    # Every label read is allowed FIT; all zero locations remain in the cache.
    assert np.array_equal(np.flatnonzero(data._fit_counts),np.sort(rows))
    return cache,torch.tensor(power,device=device),dict(count=n,bytes=cache.numel()*cache.element_size(),
        wall_seconds=time.monotonic()-started,zero_positions=int(np.sum(power==0)),
        power_sha256=array_hash(power),power_definition="mean full192 |H/FIT_RMS|^2 in FP64; floor applied only in training loss"),power
def forward(m,data,rows,device,frequency=None):
    f=data.frequencies() if frequency is None else data.frequencies(frequency)
    return m(torch.tensor(data.positions(rows),dtype=torch.float64,device=device),
        torch.tensor(data.sites(rows),dtype=torch.long,device=device),torch.tensor(f,dtype=torch.float64,device=device))
@torch.no_grad()
def fit_health(m,data,cache,power,device,cfg):
    previous=m.training;m.eval();parts=[]
    for lo in range(0,len(data.fit_ids),cfg["inference_batch_size"]):
        rows=data.fit_ids[lo:lo+cfg["inference_batch_size"]]
        v=forward(m,data,rows,device).to(torch.complex128)
        y=cache[lo:lo+len(rows)].permute(0,2,3,1).to(torch.complex128)
        te=y.abs().square().sum((1,2,3));pe=v.abs().square().sum((1,2,3))
        se=(v-y).abs().square().sum((1,2,3));me=(v.abs()-y.abs()).square().sum((1,2,3))
        parts.append(torch.stack([te,pe,se,me],-1).cpu().numpy())
    x=np.concatenate(parts);nz=x[:,0]>0
    rows=dict(row_ids=data.fit_ids,truth_energy=x[:,0],prediction_energy=x[:,1],squared_error=x[:,2],magnitude_error=x[:,3])
    summary=dict(count=len(x),nonzero_count=int(nz.sum()),zero_count=int((~nz).sum()),
        complex_nmse=float(x[:,2].sum()/x[:,0].sum()),magnitude_nmse=float(x[:,3].sum()/x[:,0].sum()),
        worst_nonzero_nmse=float(np.max(x[nz,2]/x[nz,0])),
        nonzero_energy_range=[float(np.min(x[nz,1]/x[nz,0])),float(np.max(x[nz,1]/x[nz,0]))],
        nonzero_NMSE_le_half_count=int(np.sum(x[nz,2]/x[nz,0]<=.5)),
        zero_prediction_mean_power_max=float(np.max(x[~nz,1]/196608)) if (~nz).any() else None)
    m.train(previous);return summary,rows
def save_npz(path,arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    with path.open("xb") as f:np.savez(f,**arrays)
def cpu_check(cfg,source):
    assert os.environ.get("CUDA_VISIBLE_DEVICES")=="";torch.set_num_threads(4)
    torch.manual_seed(721);p=torch.complex(torch.randn(4,3,2,5),torch.randn(4,3,2,5)).requires_grad_(True)
    y=torch.complex(torch.randn(4,3,2,5),torch.randn(4,3,2,5));y[0]=0;y[1]*=1e-3
    power=y.abs().square().double().mean((1,2,3));reg=torch.tensor(.03)
    a=objective(p,y,power,reg,cfg,"relative")
    terms=[]
    for i in range(4):
        den=max(float(power[i]),cfg["training_power_floor"])
        terms.append(((p[i].abs()-y[i].abs()).square().mean()+(p[i]-y[i]).abs().square().mean()).double()/den)
    ref=torch.stack(terms).mean()+cfg["activation_L1"]*reg
    torch.testing.assert_close(a,ref,rtol=2e-7,atol=1e-6);a.backward();assert torch.isfinite(p.grad).all()
    g=objective(p,y,power,reg,cfg,"global")
    error_rows=[float((pi.abs()-yi.abs()).square().sum())+float((pi-yi).abs().square().sum()) for pi,yi in zip(p,y)]
    gref=sum(error_rows)/max(sum(float(yi.abs().square().sum()) for yi in y),cfg["warmup_global_denominator_floor"])+cfg["activation_L1"]*float(reg)
    torch.testing.assert_close(g,torch.tensor(gref,dtype=g.dtype),rtol=3e-7,atol=1e-7)
    assert torch.isfinite(torch.autograd.grad(g,p)[0]).all()
    assert phase_for_step(1,cfg)==phase_for_step(6000,cfg)=="global" and phase_for_step(6001,cfg)=="relative"
    checks=[]
    for outer in range(5):
        for role in ["screen","refit"]:
            d=C.FoldData(outer,role,"ngrf");ix,ff=schedule(d.sites(d.fit_ids),500,cfg)
            assert np.all(np.bincount(ix.ravel(),minlength=len(d.fit_ids))>0)
            assert not set(d.fit_ids).intersection(d.outer_ids)
            if role=="screen":assert not set(d.fit_ids).intersection(d.val_ids)
            assert np.all(np.sum(d.sites(d.fit_ids)[ix]==0,axis=1)==16)
            assert ff.shape==(500,8) and np.all((ff>=0)&(ff<192))
            checks.append(dict(outer=outer,role=role,FIT_count=len(d.fit_ids),outer_count=len(d.outer_ids),all_FIT_covered_within_500_updates=True))
    native=N.equivalence_check()
    report=dict(status="passed",source_identity=source,global_and_relative_loss_loop_equivalence=True,mixed_zero_gradients_finite_in_both_phases=True,phase_boundary_6000_6001_verified=True,
        split_and_sampling=checks,native_renderer=native,GPU_used=False,outer_labels_read=False)
    C.atomic_json(OUT/"cpu_checks.json",report);emit(report)
def smoke(args,cfg,source):
    device,info=gpu_guard(args,cfg);data=C.FoldData(0,"screen","ngrf");started=time.monotonic();C.seed_all(cfg["seed"])
    cache,power,cache_info,power_np=cache_fit(data,device,cfg);m=build(data,device);opt=N.native_optimizer(m)
    initial_hash=model_hash(m);ix,ff=schedule(data.sites(data.fit_ids),20,cfg);pos=torch.tensor(data.positions(data.fit_ids),dtype=torch.float64,device=device)
    site=torch.tensor(data.sites(data.fit_ids),device=device,dtype=torch.long);frequencies=torch.tensor(data.frequencies(),device=device,dtype=torch.float64)
    torch.cuda.synchronize();t=time.monotonic();losses=[];zero_diagnostics=[];diagnostic_seconds=0.
    for j in range(20):
        phase="global" if j<cfg["smoke_stage_global_updates"] else "relative"
        m.train();opt.zero_grad(set_to_none=True);xyz_schedule(opt,j+1)
        ri=torch.tensor(ix[j],device=device);fi=torch.tensor(ff[j],device=device)
        value=m(pos[ri],site[ri],frequencies[fi]);label=cache[ri[:,None],fi[None,:]].permute(0,2,3,1)
        reg=torch.stack([x.activation_penalty() for x in m.fields.values()]).mean()
        loss=objective(value,label,power[ri],reg,cfg,phase);assert bool(torch.isfinite(loss))
        if j in [0,9,10,19]:
            torch.cuda.synchronize();diagnostic_started=time.monotonic()
            zmask=power[ri]==0
            if phase=="global":
                den=label.abs().square().sum().clamp_min(cfg["warmup_global_denominator_floor"])
                terms=((value.abs()-label.abs()).square().sum((1,2,3))+(value-label).abs().square().sum((1,2,3)))*len(ri)/den
            else:
                terms=((value.abs()-label.abs()).square().mean((1,2,3))+(value-label).abs().square().mean((1,2,3)))/power[ri].clamp_min(cfg["training_power_floor"])
            zl=terms[zmask].sum()/len(ri);nl=terms[~zmask].sum()/len(ri)
            params=tuple(m.parameters())
            zg=torch.autograd.grad(zl,params,retain_graph=True,allow_unused=True)
            ng=torch.autograd.grad(nl,params,retain_graph=True,allow_unused=True)
            z2=sum(float(x.detach().double().square().sum()) for x in zg if x is not None)
            n2=sum(float(x.detach().double().square().sum()) for x in ng if x is not None)
            dot=sum(float((x.detach().double()*y.detach().double()).sum()) for x,y in zip(zg,ng) if x is not None and y is not None)
            zero_diagnostics.append(dict(step=j+1,phase=phase,zero_count=int(zmask.sum()),nonzero_count=int((~zmask).sum()),
                zero_data_loss=float(zl),nonzero_data_loss=float(nl),zero_loss_share=float(zl/(zl+nl).clamp_min(1e-30)),
                zero_gradient_norm=math.sqrt(z2),nonzero_gradient_norm=math.sqrt(n2),
                gradient_cosine=dot/math.sqrt(z2*n2) if z2*n2 else None,
                gradients_scope="unclipped per-group data term, same batch mean; native activation L1 excluded",
                pooled_complex_nmse=float((value-label).abs().square().sum()/label.abs().square().sum().clamp_min(1e-30))))
            del zg,ng
            torch.cuda.synchronize();diagnostic_seconds+=time.monotonic()-diagnostic_started
        loss.backward()
        grad=float(torch.nn.utils.clip_grad_norm_(m.parameters(),cfg["grad_clip_norm"],error_if_nonfinite=True));opt.step();losses.append(float(loss))
    torch.cuda.synchronize();seconds=time.monotonic()-t
    assert G.steps(opt.state_dict())==[20.]
    m.eval();rows=data.fit_ids[np.linspace(0,len(data.fit_ids)-1,16,dtype=int)]
    with torch.no_grad():out=forward(m,data,rows,device)
    assert out.shape==(16,256,4,192) and out.dtype==torch.complex64 and bool(torch.isfinite(out).all())
    art=ART/"stage_smoke";art.mkdir(parents=True,exist_ok=True)
    state=dict(model=copy.deepcopy(m.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),actual_dropout=actual_dropout(m),source_identity=source)
    C.atomic_torch_save(art/"discarded.pt",state)
    clone=build(data,device);loaded=torch.load(art/"discarded.pt",map_location="cpu",weights_only=True);restore(clone,loaded)
    clone.eval()
    with torch.no_grad():replay=forward(clone,data,rows,device)
    diff=float(torch.linalg.vector_norm(replay-out)/torch.linalg.vector_norm(out).clamp_min(1e-30));assert diff<=1e-6
    report=dict(status="passed",formal=False,discarded=True,updates=20,updates_by_phase=dict(global_updates=10,relative_updates=10),
        phase_probe_only="10+10 actual updates, not a simulation or completion of the 6000-step warm-up",source_identity=source,config=cfg,
        full_FIT_cache=cache_info,parameter_count=sum(x.numel() for x in m.parameters()),native_metadata=m.metadata(),
        actual_dropout=actual_dropout(m),initial_model_hash=initial_hash,final_model_hash=model_hash(m),
        update_seconds=seconds,diagnostic_seconds=diagnostic_seconds,seconds_per_update=(seconds-diagnostic_seconds)/20,zero_nonzero_diagnostics=zero_diagnostics,loss_first=losses[0],loss_last=losses[-1],gradient_norm_last=grad,
        output_shape=list(out.shape),clean_reload_relative_L2=diff,actual_optimizer_steps=G.steps(opt.state_dict()),
        peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),wall_seconds=time.monotonic()-started,
        device=info,outer_labels_read=False,validation_labels_read=False,projected_one_15k_stage_update_seconds=(seconds-diagnostic_seconds)/20*15000)
    C.atomic_json(OUT/"stage_smoke/result.json",report);emit(report)
def train_role(outer,role,steps,args,cfg,source,device,info):
    data=C.FoldData(outer,role,"ngrf");r=OUT/f"outer{outer}"/role;a=ART/f"outer{outer}"/role
    r.mkdir(parents=True,exist_ok=True);a.mkdir(parents=True,exist_ok=True)
    if (r/"result.json").exists():
        previous=json.loads((r/"result.json").read_text());assert previous["source_identity"]==source
        return previous
    started=time.monotonic();C.seed_all(cfg["seed"]);torch.cuda.reset_peak_memory_stats();cache,power,cache_info,power_np=cache_fit(data,device,cfg)
    m=build(data,device);opt=N.native_optimizer(m);initial_model_hash=model_hash(m)
    pos=torch.tensor(data.positions(data.fit_ids),device=device,dtype=torch.float64);site=torch.tensor(data.sites(data.fit_ids),device=device,dtype=torch.long)
    frequencies=torch.tensor(data.frequencies(),device=device,dtype=torch.float64)
    ix,ff=schedule(data.sites(data.fit_ids),steps,cfg);sid=schedule_id(ix,ff)
    if not (a/"schedule.npz").exists():save_npz(a/"schedule.npz",dict(local_indices=ix,frequency_ids=ff,FIT_row_ids=data.fit_ids))
    if not (a/"position_power.npy").exists():C.atomic_npy(a/"position_power.npy",power_np)
    current=0;best_score=None;best_step=None;stale=0;history=[];visits=np.zeros(len(data.fit_ids),np.int64);carried=0.;restore_audit=None;tone_visits=np.zeros((len(data.fit_ids),192),np.int32)
    latest=a/"latest.pt";best=a/"best.pt"
    if latest.exists() or best.exists():
        options=[torch.load(path,map_location="cpu",weights_only=True) for path in [latest,best] if path.exists()]
        state=max(options,key=lambda x:x["step"]);del options
        assert state["source_identity"]==source and state["schedule"]==sid and state["FIT_ids_sha256"]==C.digest_ids(data.fit_ids)
        restore(m,state);restore_audit=G.load_optimizer(opt,state["optimizer"])
        torch.set_rng_state(state["torch_rng"].clone());torch.cuda.set_rng_state(state["cuda_rng"].clone())
        current=state["step"];best_score=state["best_score"];best_step=state["best_step"];stale=state["stale"]
        history=state["history"];visits=state["visits"].numpy().copy();tone_visits=state["tone_visits"].numpy().copy();carried=state["wall_seconds"]
        assert G.steps(opt.state_dict())==[float(current)];del state
    def save(is_best=False):
        if current:assert G.steps(opt.state_dict())==[float(current)]
        payload=dict(model=m.state_dict(),optimizer=opt.state_dict(),actual_dropout=actual_dropout(m),
            step=current,source_identity=source,FIT_ids_sha256=C.digest_ids(data.fit_ids),schedule=sid,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),best_score=best_score,best_step=best_step,
            stale=stale,history=history,visits=torch.tensor(visits),tone_visits=torch.tensor(tone_visits),wall_seconds=carried+time.monotonic()-started)
        if is_best:C.atomic_torch_save(best,payload)
        C.atomic_torch_save(latest,payload);return payload
    C.atomic_json(r/"manifest.json",dict(source_identity=source,config=cfg,outer=outer,role=role,requested_steps=steps,
        initial_model_hash=initial_model_hash,fresh_optimizer_state_count=0,loaded_optimizer_state_count=len(opt.state),schedule=sid,cache=cache_info,
        actual_dropout=actual_dropout(m),native_metadata=m.metadata(),parameter_count=sum(x.numel() for x in m.parameters()),
        data=data.info(),device=info,resume_optimizer_audit=restore_audit,initialization="fresh seed42; no prior weights"))
    last_progress=time.monotonic();stop_reason="maximum_steps" if role=="screen" else "fixed_refit_budget"
    already_stopped=role=="screen" and current>=cfg["screen_min_steps"] and stale>=cfg["patience"]
    if already_stopped:stop_reason="validation_plateau_after_minimum"
    for step in (range(0) if already_stopped else range(current+1,steps+1)):
        phase=phase_for_step(step,cfg)
        m.train();opt.zero_grad(set_to_none=True);xyz_schedule(opt,step)
        ri=torch.tensor(ix[step-1],device=device);fi=torch.tensor(ff[step-1],device=device)
        value=m(pos[ri],site[ri],frequencies[fi]);label=cache[ri[:,None],fi[None,:]].permute(0,2,3,1)
        reg=torch.stack([x.activation_penalty() for x in m.fields.values()]).mean()
        loss=objective(value,label,power[ri],reg,cfg,phase)
        if not bool(torch.isfinite(loss)):raise FloatingPointError("Nonfinite training objective")
        loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(m.parameters(),cfg["grad_clip_norm"],error_if_nonfinite=True))
        opt.step();current=step;np.add.at(visits,ix[step-1],1)
        np.add.at(tone_visits,(ix[step-1][:,None],ff[step-1][None,:]),1)
        is_eval=role=="screen" and step%cfg["validation_every"]==0
        if is_eval:
            m.eval();metrics=C.evaluate(data,lambda rows:forward(m,data,rows,device),ids=data.val_ids,batch_size=cfg["inference_batch_size"])
            eligible=step>=cfg["eligible_selection_from_step"]
            improved=eligible and (best_score is None or metrics["score"]>best_score+cfg["min_delta"])
            if improved:best_score=float(metrics["score"]);best_step=step;stale=0
            elif eligible:stale+=1
            event=dict(step=step,objective_phase=phase,validation=metrics,eligible=eligible,best_score=best_score,best_step=best_step,stale=stale,
                batch_objective=float(loss),gradient_norm=grad,all_FIT_sampled=bool(np.all(visits>0)),
                minimum_position_visits=int(visits.min()),wall_seconds=carried+time.monotonic()-started)
            history.append(event);save(is_best=improved)
            C.atomic_json(r/"progress.json",event);emit(dict(method="ngrf",outer=outer,role=role,**event))
            if step>=cfg["screen_min_steps"] and stale>=cfg["patience"]:
                stop_reason="validation_plateau_after_minimum";break
        if time.monotonic()-last_progress>=30:
            save();event=dict(method="ngrf",outer=outer,role=role,step=step,total=steps,objective_phase=phase,batch_objective=float(loss),
                gradient_norm=grad,minimum_position_visits=int(visits.min()),wall_seconds=carried+time.monotonic()-started)
            C.atomic_json(r/"progress.json",event);emit(event);last_progress=time.monotonic()
    save();assert np.all(visits>0)
    if role=="screen":
        assert best_step is not None and best_step>=cfg["screen_min_steps"]
        state=torch.load(best,map_location="cpu",weights_only=True)
        assert state["step"]==best_step and state["source_identity"]==source
        restore(m,state);selected_step=best_step;selected_path=best
    else:selected_step=current;selected_path=latest
    health,points=fit_health(m,data,cache,power,device,cfg)
    if not (a/"selected_FIT_diagnostic.npz").exists():save_npz(a/"selected_FIT_diagnostic.npz",points)
    exposure_path=a/"actual_training_exposure.npz"
    if not exposure_path.exists():save_npz(exposure_path,dict(FIT_row_ids=data.fit_ids,position_update_count=visits,position_tone_update_count=tone_visits,zero_power_mask=power_np==0,frequency_step_count=np.bincount(ff[:current].ravel(),minlength=192)))
    result=dict(status="complete",outer=outer,role=role,source_identity=source,selected_steps=selected_step,
        completed_steps=current,warmup_global_updates=min(current,cfg["warmup_global_steps"]),relative_updates=max(0,current-cfg["warmup_global_steps"]),best_validation_score=best_score,stopping=stop_reason,requested_steps=steps,
        checkpoint=PATHS.relative(selected_path),checkpoint_sha256=C.sha256(selected_path),
        latest_checkpoint=PATHS.relative(latest),latest_sha256=C.sha256(latest),initial_model_hash=initial_model_hash,
        schedule=sid,all_FIT_rows_sampled=True,min_position_visits=int(visits.min()),max_position_visits=int(visits.max()),
        actual_exposure_path=PATHS.relative(exposure_path),actual_exposure_sha256=C.sha256(exposure_path),
        unique_positions_in_gradient_updates=int(np.sum(visits>0)),minimum_distinct_tones_per_position=int(np.sum(tone_visits>0,axis=1).min()),
        frequency_step_counts=np.bincount(ff[:current].ravel(),minlength=192).tolist(),zero_position_count=int(np.sum(power_np==0)),
        zero_minimum_position_visits=int(visits[power_np==0].min()) if np.any(power_np==0) else None,
        all_FIT_final_health=health,cache=cache_info,native_metadata=m.metadata(),actual_dropout=actual_dropout(m),
        actual_latest_optimizer_step_values=G.steps(opt.state_dict()),resume_optimizer_audit=restore_audit,
        wall_seconds=carried+time.monotonic()-started,peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_scope="this process torch allocator, not whole GPU",data=data.info(),device=info,query_truth_read=False)
    C.atomic_json(r/"result.json",result);del m,opt,cache,power;torch.cuda.empty_cache();return result
@torch.no_grad()
def predict_outer(outer,screen,refit,cfg,source,device):
    data=C.FoldData(outer,"refit","ngrf");a=ART/f"outer{outer}"/"prediction";a.parent.mkdir(parents=True,exist_ok=True)
    if (a/"manifest.json").exists():
        old=json.loads((a/"manifest.json").read_text());assert old["metadata"]["source_identity"]==source
        assert C.sha256(PATHS.root/old["path"])==old["sha256"];return old
    state=torch.load(PATHS.root/refit["checkpoint"],map_location="cpu",weights_only=True)
    assert C.sha256(PATHS.root/refit["checkpoint"])==refit["checkpoint_sha256"]
    C.seed_all(cfg["seed"]);m=build(data,device);restore(m,state);m.eval();before=model_hash(m)
    if a.exists():raise FileExistsError("Unexpected incomplete final prediction directory; retained")
    stage=a.with_name("prediction.partial-"+str(os.getpid())+"-"+str(time.time_ns()));stage.mkdir()
    channels=stage/"channels.npy";rows_path=stage/"row_ids.npy"
    output=np.lib.format.open_memmap(channels,mode="w+",dtype=np.complex64,shape=(len(data.outer_ids),256,4,192))
    for lo in range(0,len(data.outer_ids),cfg["inference_batch_size"]):
        rows=data.outer_ids[lo:lo+cfg["inference_batch_size"]]
        value=(forward(m,data,rows,device).cpu().numpy()*data.scale).astype(np.complex64)
        assert value.shape==(len(rows),256,4,192) and np.isfinite(value).all()
        output[lo:lo+len(rows)]=value
    output.flush();del output;assert model_hash(m)==before;C.atomic_npy(rows_path,data.outer_ids)
    fit_data=copy.deepcopy(refit["data"]);inference_info=data.info()
    assert all(fit_data[k]==inference_info[k] for k in ["input_identity","fit_ids_sha256","scale","role","outer","outer_ids"])
    assert fit_data["validation_read_count"]==0 and not fit_data["query_truth_read"]
    assert inference_info["validation_read_count"]==inference_info["fit_read_count"]==0
    metadata=dict(source_identity=source,method_config_sha256=args_config_sha(source),actual_dropout=actual_dropout(m),
        screen_selected_steps=screen["selected_steps"],screen_completed_steps=screen["completed_steps"],
        screen_checkpoint=screen["checkpoint"],screen_checkpoint_sha256=screen["checkpoint_sha256"],
        refit_steps=refit["completed_steps"],refit_checkpoint=refit["checkpoint"],refit_checkpoint_sha256=refit["checkpoint_sha256"],
        actual_refit_optimizer_steps=refit["actual_latest_optimizer_step_values"],fresh_refit=True,
        actual_exposure_path=refit["actual_exposure_path"],actual_exposure_sha256=refit["actual_exposure_sha256"],
        inference_fit_label_reads=0,inference_validation_label_reads=0,inference_outer_label_reads=0,
        original_paper_exact_reproduction=False,adaptation=refit["native_metadata"],no_truth_correction=True,
        training_objective=cfg["training_objective_schedule"],warmup_global_updates=refit["warmup_global_updates"],relative_updates=refit["relative_updates"])
    manifest=dict(schema="canonical-baseline-prediction-v1",method="ngrf",outer=outer,role="refit",kind="channel",trained=True,
        input_identity=data.inputs.identity,prediction_namespace="canonical_train_row_ids",query_truth_read=False,
        path=PATHS.relative(a/"channels.npy"),sha256=C.sha256(channels),shape=[len(data.outer_ids),256,4,192],dtype="complex64",
        row_ids=data.outer_ids.tolist(),row_ids_sha256=C.digest_ids(data.outer_ids),
        rows_path=PATHS.relative(a/"row_ids.npy"),rows_file_sha256=C.sha256(rows_path),fit_data=fit_data,query_context=inference_info,metadata=metadata,
        source_identity=source,method_config_sha256=args_config_sha(source),phase_delay_amplitude_truth_correction=False)
    C.atomic_json(stage/"manifest.json",manifest);stage.rename(a);del m;torch.cuda.empty_cache();return manifest
def args_config_sha(source):
    keys=[k for k in source if k.endswith("method_config.json") or k.endswith("method_config_draft.json")]
    assert len(keys)==1;return source[keys[0]]
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--mode",choices=["cpu","smoke","train"],required=True)
    ap.add_argument("--gpu-uuid",default=GPU);ap.add_argument("--config",type=Path,default=OUT/"method_config_draft.json")
    ap.add_argument("--fold",type=int,choices=list(range(5)));ap.add_argument("--contract",type=Path)
    ap.add_argument("--contract-sha256");args=ap.parse_args()
    cfg=json.loads(args.config.read_text());source=identity(args.config);OUT.mkdir(parents=True,exist_ok=True);ART.mkdir(parents=True,exist_ok=True)
    assert cfg["batch_size"]==2*cfg["per_site_batch"]==32 and cfg["complex_weight"]==1 and cfg["dropout"]==0
    assert cfg["warmup_global_steps"]==6000 and cfg["smoke_stage_global_updates"]==cfg["smoke_stage_relative_updates"]==10
    if args.mode=="cpu":cpu_check(cfg,source);return
    cpu=json.loads((OUT/"cpu_checks.json").read_text());assert cpu["status"]=="passed"
    assert cpu["source_identity"][PATHS.relative(Path(__file__))]==source[PATHS.relative(Path(__file__))]
    if args.mode=="smoke":smoke(args,cfg,source);return
    assert args.contract is not None and args.contract_sha256 and C.sha256(args.contract)==args.contract_sha256,"Frozen root contract SHA required"
    smoke_result=json.loads((OUT/"stage_smoke/result.json").read_text());assert smoke_result["status"]=="passed"
    assert smoke_result["source_identity"][PATHS.relative(Path(__file__))]==source[PATHS.relative(Path(__file__))]
    assert args_config_sha(smoke_result["source_identity"])==args_config_sha(source),"Smoke and formal config payload differ"
    source={**source,PATHS.relative(args.contract):args.contract_sha256}
    device,info=gpu_guard(args,cfg);records=[]
    for outer in ([args.fold] if args.fold is not None else range(5)):
        screen=train_role(outer,"screen",cfg["screen_max_steps"],args,cfg,source,device,info)
        nscreen=len(C.FoldData(outer,"screen","ngrf").fit_ids);nrefit=len(C.FoldData(outer,"refit","ngrf").fit_ids)
        updates=math.ceil(screen["selected_steps"]*nrefit/nscreen)
        refit=train_role(outer,"refit",updates,args,cfg,source,device,info)
        prediction=predict_outer(outer,screen,refit,cfg,source,device)
        records.append(dict(outer=outer,screen=screen,refit=refit,prediction=prediction))
        C.atomic_json(OUT/f"outer{outer}"/"complete.json",records[-1])
    if args.fold is None:
        assert sum(x["prediction"]["shape"][0] for x in records)==500
        C.atomic_json(OUT/"complete.json",dict(status="complete",method="ngrf",folds=records,source_identity=source,
            outer_truth_read=False,score_computed=False,all500_predictions_frozen=True))
        emit(dict(status="complete",method="ngrf",fold_count=5,predicted_positions=500,score_computed=False))
if __name__=="__main__":
    try:main()
    except BaseException:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/"failure.log").open("a") as f:f.write(traceback.format_exc()+"\n")
        raise
