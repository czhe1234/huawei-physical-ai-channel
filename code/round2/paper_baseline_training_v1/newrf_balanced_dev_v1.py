"""Bounded NeWRF sampling/budget repair on a frozen FIT panel; no formal test."""
from __future__ import annotations
import argparse,copy,json,math,time,traceback,signal
from pathlib import Path
import numpy as np
import torch
from .common import (ART,REPORT,FoldData,seed_all,device_guard,sha256,digest_ids,
                     atomic_json,atomic_npy,atomic_torch_save)
from .newrf_runner import GEOMETRY,FitGeometryBank,subset_forward
from .newrf_native_batch_v4 import build_native_batch_model,station_forward,native_weighted_loss
from .newrf_runtime_cache_v4 import MemoizedFitDirections,station_target_cached
from .newrf_query import subset_query
from project_paths import PATHS

GPU_UUID="GPU-6b53c3c2-3071-5dfb-f306-2f2f142f8009"
A=ART/"fit_repair_dev_20261101/newrf"
R=REPORT/"fit_repair_dev_20261101/newrf"
MON_BS=np.linspace(0,127,8,dtype=np.int64)
MON_F=np.linspace(0,191,12,dtype=np.int64)

class BudgetExpired(RuntimeError):pass

def hard_timeout(signum,frame):
    raise BudgetExpired("Hard SIGALRM wall budget exhausted")

class BalancedSampler:
    """Each batch contains every panel row twice; BS/F marginal permutations."""
    def __init__(self,rows,seed):
        self.rows=np.asarray(rows,np.int64);self.rng=np.random.default_rng(seed)
        if len(self.rows)!=16 or len(np.unique(self.rows))!=16:raise ValueError("16 distinct FIT rows required")
        self.slot={int(row):i for i,row in enumerate(self.rows)}
        self.bs=np.empty(0,np.int64);self.ff=np.empty(0,np.int64);self.bi=0;self.fi=0
        self.position_counts=np.zeros(16,np.int64)
        self.bs_counts=np.zeros((16,128),np.int64);self.ff_counts=np.zeros((16,192),np.int64)
        self.joint_seen=np.zeros((16,128,192),bool)
    def _take(self,n,which):
        size=128 if which=="bs" else 192
        buf=getattr(self,which);idx=getattr(self,"bi" if which=="bs" else "fi");parts=[];remain=n
        while remain:
            if idx==len(buf):buf=self.rng.permutation(size);idx=0
            count=min(remain,len(buf)-idx);parts.append(buf[idx:idx+count]);idx+=count;remain-=count
        setattr(self,which,buf);setattr(self,"bi" if which=="bs" else "fi",idx)
        return np.concatenate(parts)
    def draw(self):
        rows=np.tile(self.rows,2);rows=rows[self.rng.permutation(32)]
        bs=self._take(32,"bs");ff=self._take(32,"ff")
        slots=np.asarray([self.slot[int(r)] for r in rows],np.int64)
        np.add.at(self.position_counts,slots,1)
        np.add.at(self.bs_counts,(slots,bs),1);np.add.at(self.ff_counts,(slots,ff),1)
        self.joint_seen[slots,bs,ff]=True
        return rows,bs,ff
    def state(self):
        return {k:copy.deepcopy(v) for k,v in self.__dict__.items() if k not in ("rng","slot")} | {"rng_state":copy.deepcopy(self.rng.bit_generator.state)}
    def restore(self,state):
        for k,v in state.items():
            if k!="rng_state":setattr(self,k,copy.deepcopy(v))
        self.slot={int(row):i for i,row in enumerate(self.rows)}
        self.rng.bit_generator.state=state["rng_state"]
    def coverage(self):
        return dict(position_draws=self.position_counts.tolist(),
          BS_marginal_covered_per_position=(self.bs_counts>0).sum(1).tolist(),
          frequency_marginal_covered_per_position=(self.ff_counts>0).sum(1).tolist(),
          joint_BS_frequency_unique_per_position=self.joint_seen.sum((1,2)).tolist(),
          joint_BS_frequency_fraction=float(self.joint_seen.mean()),
          marginal_coverage_is_not_full_Cartesian_coverage=True)

def identity(contract,panel):
    names=["newrf_runner.py","newrf_native_batch_v4.py","newrf_runtime_cache_v4.py",
           "newrf_conditioned_v3.py","newrf_latefreq.py","newrf_query.py","common.py"]
    paths=[Path(__file__),contract,panel,*[Path(__file__).with_name(n) for n in names]]
    return {PATHS.relative(p):sha256(p) for p in paths}

def verify_sources(ids):
    for rel,digest in ids.items():
        if sha256(PATHS.root/rel)!=digest:raise RuntimeError("Frozen source changed: "+rel)

def cpu_weights(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

def metrics(pred,target):
    p=np.asarray(pred,np.complex64);t=np.asarray(target,np.complex64)
    if p.shape!=t.shape or not np.isfinite(p).all() or not np.isfinite(t).all():raise ValueError("Metric tensor")
    def one(x,y):
        pe=float(np.sum(np.abs(x)**2,dtype=np.float64));te=float(np.sum(np.abs(y)**2,dtype=np.float64))
        err=float(np.sum(np.abs(x-y)**2,dtype=np.float64))
        dot=np.sum(x.astype(np.complex128).conj()*y.astype(np.complex128),dtype=np.complex128)
        return dict(pred_energy=pe,truth_energy=te,squared_error=err,
                    nmse=err/te if te>0 else None,energy_ratio=pe/te if te>0 else None,
                    signed_correlation_real=float(dot.real/math.sqrt(pe*te)) if pe>0 and te>0 else None)
    out=one(p,t);out["per_position"]=[one(x,y) for x,y in zip(p,t)]
    if p.shape[1:3]==(256,4):
        out["fields"]=[dict(field=f,**one(p[:,(f//4)*128:(f//4+1)*128,f%4],
                                        t[:,(f//4)*128:(f//4+1)*128,f%4])) for f in range(8)]
    return out

def gate(summary,threshold):
    fields=summary["fields"]
    return bool(summary["nmse"] is not None and summary["nmse"]<=threshold["pooled_nmse_max"]
       and all(x["nmse"] is not None and x["nmse"]<=threshold["all_field_nmse_max"]
               and threshold["all_field_energy_ratio_min"]<=x["energy_ratio"]<=threshold["all_field_energy_ratio_max"] for x in fields))

def fixed_field(model,cache,data,rows,field,bs,ff,deadline):
    values=[];coarse=[]
    with torch.no_grad():
        for row in rows:
            if time.monotonic()>deadline:raise BudgetExpired("FIT diagnostic deadline")
            vv=[];cc=[]
            for first in range(0,len(bs),8):
                h,hc=subset_forward(model,cache,data,np.array([row],np.int64),field,bs[first:first+8],ff)
                vv.append(h[0].cpu().numpy());cc.append(hc[0].cpu().numpy())
            values.append(np.concatenate(vv,axis=0));coarse.append(np.concatenate(cc,axis=0))
    return np.stack(values),np.stack(coarse)

def fit_tensor(model,cache,data,rows,deadline):
    result=np.empty((len(rows),256,4,192),np.complex64)
    model.eval()
    for field in range(8):
        p,u=divmod(field,4)
        value,_=fixed_field(model,cache,data,rows,field,np.arange(128,dtype=np.int64),np.arange(192,dtype=np.int64),deadline)
        result[:,p*128:(p+1)*128,u]=value
    return result

def query_tensor(model,data,rows,deadline):
    values=[];model.eval()
    with torch.no_grad():
        for row in rows:
            if time.monotonic()>deadline:raise BudgetExpired("Query diagnostic deadline")
            value=model(data.positions([row]),data.sites([row]))[0].cpu().numpy()
            if not np.isfinite(value).all():raise FloatingPointError("Query H")
            values.append(value)
    return np.stack(values)

def validation_small(model,data,rows,deadline):
    bs=np.asarray([0,127,128,255],np.int64);ff=MON_F;ue=np.arange(4,dtype=np.int64)
    values=[]
    with torch.no_grad():
        for row in rows:
            if time.monotonic()>deadline:raise BudgetExpired("Validation deadline")
            values.append(subset_query(model,data,[row],bs,ue,ff)[0].cpu().numpy())
    value=np.stack(values)
    truth=data.read_validation(rows,ff,normalized=False)[:,bs]
    return value,metrics(value,truth),dict(BS_ids=bs.tolist(),UE_ids=ue.tolist(),frequency_ids=ff.tolist(),
                                         full_H=False,used_for_selection=False)

def main(args):
    contract=json.loads(args.contract.read_text());panel=json.loads(args.panel.read_text())
    if contract["schema"]!="fullH-fit-repair-dev-contract-v1" or panel["schema"]!="balanced-fullH-FIT-dev-panel-v1":raise ValueError("Protocol schema")
    if sha256(args.contract)!=args.contract_sha or sha256(args.panel)!=args.panel_sha:raise RuntimeError("Parent-frozen protocol hash")
    if contract.get("formal_test") is not False or contract.get("outer_truth_reads") is not False:raise ValueError("Dev-only required")
    rows=np.asarray(panel["train_ids"],np.int64);val=np.asarray(panel["validation_ids"],np.int64)
    zero=np.asarray(panel["zero_probe_ids"],np.int64)
    if len(rows)!=16 or len(np.unique(rows))!=16 or not set(zero).issubset(set(rows)):raise ValueError("Panel")
    if not args.check_only:
        signal.signal(signal.SIGALRM,hard_timeout)
        signal.alarm(120 if args.qa_only else int(contract["budgets"]["newrf"]["max_wall_seconds"]))
    data=FoldData(0,"screen","newrf")
    if panel["input_identity"]!=data.inputs.identity or digest_ids(data.fit_ids)!=panel["parent_fit_ids_sha256"]:raise RuntimeError("Parent role changed")
    if not set(rows).issubset(set(data.fit_ids)) or not set(val).issubset(set(data.val_ids)) or np.intersect1d(rows,val).size:raise PermissionError("Panel roles")
    if args.check_only:
        print(json.dumps(dict(status="protocol_valid",train_rows=rows.tolist(),validation_rows=val.tolist(),train=False)));return
    A.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
    if (R/"result.json").exists():raise FileExistsError("Preserve completed repair run")
    if (R/"run_manifest.json").exists() and not args.resume:raise FileExistsError("Use --resume explicitly")
    wall=time.monotonic();elapsed_before=0.;sources=identity(args.contract,args.panel)
    gpu=device_guard(GPU_UUID);torch.set_num_threads(4);seed_all(contract["seed"])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    bank=FitGeometryBank(GEOMETRY,data.inputs,outer=0,role="screen_fit")
    cache=MemoizedFitDirections(bank).to("cuda:0")
    model=build_native_batch_model(data,cache,128,128,frequency_chunk=8).to("cuda:0")
    truth=data.read_fit(rows,normalized=False)
    if not np.isfinite(truth).all():raise ValueError("FIT labels")
    if set(rows[np.sum(np.abs(truth)**2,axis=(1,2,3))==0])!=set(zero):raise ValueError("Zero panel changed")
    slots=np.full(4000,-1,np.int64);slots[rows]=np.arange(len(rows))
    data._native_raw_target_cache=(slots,torch.from_numpy(truth).to("cuda:0"))
    optimizers=[];schedulers=[];samplers=[];parameters=[]
    known=set()
    for field in range(8):
        params=list(model.coarse[field].parameters())+list(model.fine[field].parameters());parameters.append(params)
        known.update(id(x) for x in params)
        opt=torch.optim.Adam(params,lr=.0005,weight_decay=0.);optimizers.append(opt)
        schedulers.append(torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",factor=.9,
                          patience=1,threshold=.001,threshold_mode="abs",min_lr=1e-5))
        samplers.append(BalancedSampler(rows,42+1009*field))
    if any(x.requires_grad and id(x) not in known for x in model.parameters()):raise RuntimeError("Unassigned shared trainable parameter")
    if args.qa_only:
        qa_deadline=wall+115
        representative=int(rows[np.argmax(cache.count[rows])])
        model.eval();torch.cuda.synchronize();tick=time.monotonic()
        native=fit_tensor(model,cache,data,np.array([representative]),qa_deadline)
        torch.cuda.synchronize();native_s=time.monotonic()-tick
        tick=time.monotonic();query=query_tensor(model,data,np.array([representative]),qa_deadline)
        torch.cuda.synchronize();query_s=time.monotonic()-tick
        times=[]
        for k in range(16):
            if time.monotonic()>qa_deadline:raise BudgetExpired("QA120s")
            field=k%8;ids,bs,ff=samplers[field].draw();tick=time.monotonic()
            model.train();optimizers[field].zero_grad(set_to_none=True)
            y=station_target_cached(data,ids,field,bs,ff,"cuda:0")
            pred,coarse=station_forward(model,cache,data,ids,field,bs,ff)
            loss,_,_=native_weighted_loss(pred,coarse,y);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(parameters[field],5.)
            if not torch.isfinite(loss) or not torch.isfinite(norm):raise FloatingPointError("QA update")
            optimizers[field].step();torch.cuda.synchronize();times.append(time.monotonic()-tick)
        estimate=16*1.25*(native_s+query_s)+35
        qa=dict(status="passed",source_identity=sources,GPU=gpu,
          full_native_one_row_seconds=native_s,full_query_one_row_seconds=query_s,
          representative_FIT_row=representative,max_panel_cached_ray_count=int(cache.count[rows].max()),
          mean_update_seconds=float(np.mean(times[8:])),median_update_seconds=float(np.median(times[8:])),
          fitting_plus_query_reserve_seconds=estimate,
          effective_train_wall_seconds=max(0,min(contract["budgets"]["newrf"]["train_wall_seconds"],
                    contract["budgets"]["newrf"]["max_wall_seconds"]-estimate)),
          elapsed_seconds=time.monotonic()-wall,weights_discarded=True,formal_training_started=False,
          native_shape=list(native.shape),query_shape=list(query.shape),validation_reads=0,outer_reads=0)
        atomic_json(R/"qa_runtime.json",qa);print(json.dumps(qa),flush=True);return
    budget=contract["budgets"]["newrf"];target=budget["per_field_target"];minimum=budget["min_before_plateau_stop"]
    updates=[0]*8;best_field=[math.inf]*8;significant=[math.inf]*8;stale=[0]*8
    drops=[0]*8;last_drop_step=[None]*8;zero_gradient=[0]*8;history=[];best_global=math.inf
    start_round=0
    if args.resume and (A/"last.pt").exists():
        ck=torch.load(A/"last.pt",map_location="cpu",weights_only=False)
        if ck["source_identity"]!=sources:raise RuntimeError("Resume sources differ")
        model.load_state_dict(ck["model"]);updates=ck["updates"];best_field=ck["best_field"]
        significant=ck["significant"];stale=ck["stale"];drops=ck["lr_drops"];last_drop_step=ck["last_drop_step"]
        zero_gradient=ck["zero_gradient"];history=ck["history"];best_global=ck["best_global"]
        elapsed_before=ck["elapsed_seconds"];start_round=ck["round"]
        for field in range(8):
            optimizers[field].load_state_dict(ck["optimizers"][field]);schedulers[field].load_state_dict(ck["schedulers"][field])
            samplers[field].restore(ck["samplers"][field])
            for st in optimizers[field].state.values():
                for k,v in st.items():
                    if isinstance(v,torch.Tensor):st[k]=v.to("cuda:0")
        torch.set_rng_state(ck["torch_rng"]);torch.cuda.set_rng_state_all(ck["cuda_rng"])
    def elapsed():return elapsed_before+time.monotonic()-wall
    deadline=wall+budget["max_wall_seconds"]-elapsed_before
    qa=json.loads((R/"qa_runtime.json").read_text())
    # QA preceded control/metadata review fixes and a NeRF-only shared budget edit.
    # Rendering/model/cache/panel sources must remain byte-identical; throughput is not a fit result.
    qa_exceptions={PATHS.relative(Path(__file__)),PATHS.relative(args.contract)}
    if any(qa["source_identity"].get(k)!=v for k,v in sources.items() if k not in qa_exceptions):
        raise RuntimeError("QA numerical model or panel source changed")
    signal.alarm(max(1,int(math.ceil(budget["max_wall_seconds"]-elapsed()))))
    reserve=16*1.25*(qa["full_native_one_row_seconds"]+qa["full_query_one_row_seconds"])+35
    train_wall=max(0,min(budget["train_wall_seconds"],budget["max_wall_seconds"]-reserve))
    train_deadline=wall+train_wall-elapsed_before
    def checkpoint(round_id):
        return dict(schema="newrf-balanced-dev-checkpoint-v1",model=cpu_weights(model),
          optimizers=[x.state_dict() for x in optimizers],schedulers=[x.state_dict() for x in schedulers],
          samplers=[x.state() for x in samplers],torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
          updates=updates.copy(),best_field=best_field.copy(),significant=significant.copy(),stale=stale.copy(),
          lr_drops=drops.copy(),last_drop_step=last_drop_step.copy(),zero_gradient=zero_gradient.copy(),
          history=copy.deepcopy(history),best_global=best_global,elapsed_seconds=elapsed(),round=round_id,
          source_identity=sources,contract=contract,panel=panel,FIT_RMS=data.scale,raw_H=True)
    atomic_json(R/"run_manifest.json",dict(status="running",GPU=gpu,source_identity=sources,contract=contract,panel=panel,
          model_config=__import__("dataclasses").asdict(model.config),initialization="fresh seed42; no old collapsed weights",
          target_cache="only16FIT rawH",geometry_bank="same full parent screenFIT geometry, unchanged ray prior/far",
          scheduler=dict(factor=.9,patience=1,threshold_absolute=.001),best_rule="numerical minimum independent of .001",
          validation_used_for_selection=False,outer_forward=False,effective_train_wall_seconds=train_wall,finalization_reserve_seconds=reserve,
          qa_evidence_sha256=sha256(R/"qa_runtime.json"),qa_source_identity=qa["source_identity"],
          qa_compatibility="only own timeout/metadata controls and shared contract NeRF budget changed; numerical helpers/panel identical"))
    atomic_json(R/"contract_snapshot.json",contract);atomic_json(R/"panel_snapshot.json",panel)
    atomic_npy(A/"fit_row_ids.npy",rows,replace=args.resume)
    atomic_npy(A/"validation_row_ids.npy",val,replace=args.resume)
    stop="field_budget";round_id=start_round;last_supervised=[None]*8
    monitor_summary=None;accepted_current=None
    # Every round performs one independent update for each of the8fields.
    while min(updates)<target:
        if time.monotonic()>=train_deadline:stop="train_wall_budget";break
        round_id+=1
        for field in range(8):
            if updates[field]>=target:continue
            if time.monotonic()>=train_deadline:stop="train_wall_budget";break
            ids,bs,ff=samplers[field].draw();model.train();opt=optimizers[field]
            opt.zero_grad(set_to_none=True)
            y=station_target_cached(data,ids,field,bs,ff,"cuda:0")
            pred,coarse=station_forward(model,cache,data,ids,field,bs,ff)
            loss,lf,lc=native_weighted_loss(pred,coarse,y)
            if not torch.isfinite(loss):raise FloatingPointError("Native loss")
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(parameters[field],5.)
            if not torch.isfinite(norm):raise FloatingPointError("Gradient")
            if float(norm)==0:zero_gradient[field]+=1
            opt.step();updates[field]+=1
            with torch.no_grad():
                e=float(y.abs().square().sum(dtype=torch.float64));pe=float(pred.abs().square().sum(dtype=torch.float64))
                last_supervised[field]=dict(field=field,updates=updates[field],native_loss=float(loss),fine_nmse=float(lf),
                    coarse_nmse=float(lc),energy_ratio=pe/e,gradient_norm_before_clip=float(norm),
                    allzero_target_batch=False,position_draws=ids.tolist(),BS_ids=bs.tolist(),frequency_ids=ff.tolist())
        if round_id%64==0:
            atomic_json(R/"progress.json",dict(elapsed_seconds=elapsed(),updates=updates,lr=[x.param_groups[0]["lr"] for x in optimizers],
               lr_drops=drops,last_supervised=last_supervised,formal_test=False))
            print(json.dumps(dict(stage="training",elapsed=elapsed(),updates=updates,last_native_losses=[x["native_loss"] if x else None for x in last_supervised])),flush=True)
        do_monitor=(round_id%512==0 or min(updates)>=target or stop=="train_wall_budget")
        if do_monitor:
            verify_sources(sources);model.eval();field_results=[]
            for field in range(8):
                value,cvalue=fixed_field(model,cache,data,rows,field,MON_BS,MON_F,min(deadline,train_deadline+60))
                p,u=divmod(field,4);t=truth[:,p*128+MON_BS,u][:,:,MON_F]
                m=metrics(value,t);cm=metrics(cvalue,t)
                score=m["nmse"];before=optimizers[field].param_groups[0]["lr"];schedulers[field].step(score)
                after=optimizers[field].param_groups[0]["lr"]
                if after<before:drops[field]+=1;last_drop_step[field]=updates[field]
                if score<best_field[field]:
                    best_field[field]=score
                    atomic_torch_save(A/f"best_field{field}.pt",dict(coarse=model_state_field(model.coarse[field]),
                       fine=model_state_field(model.fine[field]),optimizer=optimizers[field].state_dict(),
                       scheduler=schedulers[field].state_dict(),sampler=samplers[field].state(),field=field,
                       updates=updates[field],monitor=m,source_identity=sources,not_full_H_gate=True))
                if score<significant[field]-.001:significant[field]=score;stale[field]=0
                else:stale[field]+=1
                field_results.append(dict(field=field,updates=updates[field],fine=m,coarse=cm,lr=after,
                                          lr_drops=drops[field],post_drop_updates=updates[field]-last_drop_step[field] if last_drop_step[field] is not None else 0))
            total_err=sum(x["fine"]["squared_error"] for x in field_results);total_energy=sum(x["fine"]["truth_energy"] for x in field_results)
            score=total_err/total_energy
            monitor_summary=dict(round=round_id,elapsed_seconds=elapsed(),pooled_nmse=score,fields=field_results,
                    BS_ids=MON_BS.tolist(),frequency_ids=MON_F.tolist(),full_H=False,last_supervised=copy.deepcopy(last_supervised))
            history.append(monitor_summary);atomic_json(R/"monitor_history.json",history)
            if score<best_global:
                best_global=score;atomic_torch_save(A/"best_monitor.pt",checkpoint(round_id))
            atomic_torch_save(A/"last.pt",checkpoint(round_id))
            eligible=(min(updates)>=minimum and all(x>=1 for x in drops) and
                      all(updates[f]-last_drop_step[f]>=512 for f in range(8)) and min(stale)>=2)
            if eligible and time.monotonic()+180<train_deadline:
                current=fit_tensor(model,cache,data,rows,train_deadline);summary=metrics(current,truth)
                atomic_json(R/f"full_gate_round{round_id}.json",dict(metrics=summary,training_route=True))
                if gate(summary,contract["acceptance"]["learning_recovered"]):
                    accepted_current=current;stop="full_fit_gate_postLR_plateau";break
        if stop=="train_wall_budget":break
    # Preserve the actual training endpoint; numerical best selection is a separate artifact.
    atomic_torch_save(A/"last.pt",checkpoint(round_id))
    verify_sources(sources);model.eval()
    selected_kind="accepted_full_fit_endpoint" if accepted_current is not None else "numerical_best_fixedFIT_monitor"
    if accepted_current is None and (A/"best_monitor.pt").exists():
        selected=torch.load(A/"best_monitor.pt",map_location="cpu",weights_only=False)
        model.load_state_dict(selected["model"]);selected_updates=selected["updates"]
    else:
        selected=checkpoint(round_id);selected_updates=updates.copy()
    selected_coverage=[]
    for state in selected["samplers"]:
        sampler=BalancedSampler(rows,0);sampler.restore(state);selected_coverage.append(sampler.coverage())
    selected_lr=[x["param_groups"][0]["lr"] for x in selected["optimizers"]]
    atomic_torch_save(A/"selected.pt",dict(model=cpu_weights(model),selected_updates=selected_updates,
          source_identity=sources,selection=selected_kind,contract=contract,panel=panel,raw_H=True))
    full=accepted_current if accepted_current is not None else fit_tensor(model,cache,data,rows,deadline)
    summary=metrics(full,truth)
    atomic_npy(A/"final_fit_prediction.npy",full,replace=args.resume)
    recovered=gate(summary,contract["acceptance"]["learning_recovered"])
    good=gate(summary,contract["acceptance"]["good_fit"])
    result=dict(status="good_fit" if good else ("learning_recovered" if recovered else "underfit_or_budget_limited"),
       stop_reason=stop,elapsed_seconds=elapsed(),actual_updates_per_field=updates,selected_updates_per_field=selected_updates,
       selected_kind=selected_kind,full_FIT_train_route=summary,full_H_gate=True,
       per_field_coverage=[x.coverage() for x in samplers],per_field_coverage_applies_to="training endpoint, not selected",
       endpoint_per_field_coverage=[x.coverage() for x in samplers],selected_per_field_coverage=selected_coverage,
       endpoint_lr_drops=drops,selected_lr_drops=selected["lr_drops"],zero_gradient_batches=zero_gradient,
       endpoint_final_lr=[x.param_groups[0]["lr"] for x in optimizers],selected_lr=selected_lr,
       source_identity=sources,FIT_RMS=data.scale,
       normalization="rawH throughout; no RMS multiply at output",formal_test=False,outer_forward=False,
       validation_used_for_selection=False,weak_point_accuracy_not_implied_by_pooled_gate=True)
    # Validation is diagnostic only, on an explicitly small submatrix.
    if time.monotonic()+20<deadline:
        try:
            vp,vm,axes=validation_small(model,data,val,deadline-5)
            atomic_npy(A/"final_validation_subset_prediction.npy",vp,replace=args.resume)
            axes.update(row_ids=val.tolist(),prediction_file=PATHS.relative(A/"final_validation_subset_prediction.npy"),
                        values="rawH; no RMS multiply")
            atomic_json(R/"validation_axes.json",axes);atomic_json(A/"validation_axes.json",axes)
            result["validation_subset"]=dict(metrics=vm,axes=axes)
        except BudgetExpired:result["validation_subset_skipped"]="wall budget"
    query_reserve=16*qa["full_query_one_row_seconds"]*1.25+10
    if time.monotonic()+query_reserve<deadline:
        try:
            query=query_tensor(model,data,rows,deadline-5)
            atomic_npy(A/"final_fit_query_prediction.npy",query,replace=args.resume)
            result["FIT_query_route"]=metrics(query,truth)
        except BudgetExpired:result["FIT_query_route_skipped"]="wall budget"
    # Full validation is optional; do not consume the fitting or fullFIT acceptance budget.
    if time.monotonic()+query_reserve<deadline:
        try:
            value=query_tensor(model,data,val,deadline-5)
            target_val=data.read_validation(val,normalized=False)
            atomic_npy(A/"final_validation_prediction.npy",value,replace=args.resume)
            result["validation_full_H"]=metrics(value,target_val)
        except BudgetExpired:result["validation_full_H_skipped"]="wall budget"
    result["elapsed_seconds"]=elapsed();result["data_reads"]=data.info()
    atomic_json(R/"result.json",result);signal.alarm(0)
    print(json.dumps(dict(status=result["status"],stop_reason=stop,updates=updates,
                         full_fit_nmse=summary["nmse"],full_fit_energy_ratio=summary["energy_ratio"],elapsed=elapsed())),flush=True)

def model_state_field(field):
    return {k:v.detach().cpu().clone() for k,v in field.state_dict().items()}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--contract",type=Path,required=True);p.add_argument("--panel",type=Path,required=True)
    p.add_argument("--contract-sha",required=True);p.add_argument("--panel-sha",required=True)
    p.add_argument("--check-only",action="store_true");p.add_argument("--qa-only",action="store_true");p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    try:main(args)
    except BaseException:
        # Failure records stay in this independent development output only.
        R.mkdir(parents=True,exist_ok=True)
        atomic_json(R/"failure.json",dict(traceback=traceback.format_exc(),training_started=not args.check_only,
                    formal_test=False,old_results_changed=False))
        raise

