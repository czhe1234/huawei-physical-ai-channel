"""Signal-only trunk gradients during warmup; numerical native rho/render unchanged."""
from __future__ import annotations
import copy,json,math,signal,time,traceback
from pathlib import Path
import numpy as np
import torch
from .common import ART,REPORT,FoldData,seed_all,device_guard,sha256,atomic_json,atomic_npy,atomic_torch_save
from .newrf_runner import GEOMETRY,FitGeometryBank
from .newrf_native_batch_v4 import build_native_batch_model,station_forward,native_weighted_loss
from .newrf_runtime_cache_v4 import MemoizedFitDirections,station_target_cached
from .newrf_balanced_dev_v1 import metrics,fixed_field
from .newrf_paired_initialization_v8 import density_profile,gate
from . import newrf_minimal_fullH_v2 as unit
from project_paths import PATHS
A=ART/"fit_repair_paper_v9_signal_gradient_warm_20261101/newrf"
R=REPORT/"fit_repair_paper_v9_signal_gradient_warm_20261101/newrf"
BASE=ART/"fit_repair_paper_v8_paired_initialization_20261101/newrf"
class BudgetExpired(RuntimeError):pass
def alarm(sig,frame):raise BudgetExpired("600s signal-gradient-warm native unit")
def main():
    started=time.monotonic();signal.signal(signal.SIGALRM,alarm);signal.alarm(600)
    A.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
    if (R/"run_manifest.json").exists():raise FileExistsError("Preserve prior signal-gradient-warm unit")
    panel_path=REPORT/"fit_repair_paper_v2_20261101/panel_v2.json";panel=json.loads(panel_path.read_text())
    parent_path=BASE/"A_reset_old_fine/initial.pt";parent=torch.load(parent_path,map_location="cpu",weights_only=False);unit.source_check(parent["source_identity"])
    source=unit.source_identity(REPORT/"fit_repair_paper_v2_20261101/contract.json",panel_path)
    files=[Path(__file__),Path(__file__).with_name("newrf_paired_initialization_v8.py"),parent_path]+[BASE/n for n in ["warm_row_ids.npy","warm_BS_ids.npy","warm_frequency_ids.npy","warm_raw_labels.npy"]]
    for p in files:source[PATHS.relative(p)]=sha256(p)
    draw_rows=np.load(BASE/"warm_row_ids.npy");draw_bs=np.load(BASE/"warm_BS_ids.npy");draw_f=np.load(BASE/"warm_frequency_ids.npy");draw_y=np.load(BASE/"warm_raw_labels.npy")
    if draw_rows.shape!=(2000,32):raise RuntimeError("Exactly V8A 2000 warm draws")
    rows=np.array([2666,2042],np.int64);field=0;b=30;fid=96;data=FoldData(0,"screen","newrf")
    if not set(map(int,rows)).issubset(set(map(int,data.fit_ids))) or not np.all(data.sites(rows)==0) or data.inputs.identity!=panel["input_identity"]:raise ValueError("Two FIT site0 locations")
    guard=device_guard(unit.GPU);torch.set_num_threads(4);seed_all(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    bank=FitGeometryBank(GEOMETRY,data.inputs,outer=0,role="screen_fit");cache=MemoizedFitDirections(bank).to("cuda:0")
    raw=data.read_fit(rows,normalized=False);truth=raw[:,:128,0];lookup={int(v):i for i,v in enumerate(rows)}
    slots=np.full(4000,-1,np.int64);slots[rows]=np.arange(2);data._native_raw_target_cache=(slots,torch.from_numpy(raw).to("cuda:0"))
    ii=np.vectorize(lookup.__getitem__)(draw_rows)
    if not np.array_equal(draw_y,truth[ii,draw_bs,draw_f]) or not np.all((draw_rows==rows[0]).sum(1)==16) or not np.all((draw_rows==rows[1]).sum(1)==16):raise RuntimeError("Same actual labels and position balance")
    model=build_native_batch_model(data,cache,128,128,frequency_chunk=8).to("cuda:0");model.load_state_dict(parent["model"])
    for p in model.parameters():p.requires_grad_(False)
    named=dict(model.fine[0].named_parameters());alpha_names=[n for n in named if n.startswith("network.alpha_out.")]
    train_names=[n for n in named if n not in alpha_names]
    if len(alpha_names)!=2 or not all(n.startswith(("network.layers.","network.realimag_filters.","network.branch.","network.output.")) for n in train_names):raise RuntimeError("Parameter graph")
    active=[named[n] for n in train_names];alpha=[named[n] for n in alpha_names]
    for p in active:p.requires_grad_(True)
    opt=torch.optim.Adam([dict(params=active),dict(params=alpha)],lr=.0005,weight_decay=0.)
    if opt.state:raise RuntimeError("Fresh fine Adam")
    torch.set_rng_state(parent["torch_rng"]);torch.cuda.set_rng_state_all(parent["cuda_rng"])
    group=unit.groups(model,0);frozen_coarse=unit.state(model.coarse[0])
    atomic_npy(A/"actual_target_field.npy",truth);atomic_npy(A/"field_row_ids.npy",rows)
    for n in ["warm_row_ids.npy","warm_BS_ids.npy","warm_frequency_ids.npy","warm_raw_labels.npy"]:atomic_npy(A/n,np.load(BASE/n))
    protocol=dict(schema="newrf-signal-gradient-warm-v9",initialization="Exactly V8A initial checkpoint; V6 fine, V7A coarse",parent_sha256=sha256(parent_path),
        row_ids=rows.tolist(),field=0,BS_ids=list(range(128)),UE_port=0,frequency_ids=list(range(192)),fresh_fine_Adam=True,
        warm_max_updates=2000,unfrozen_updates_if_stable_pass=1000,total_max_updates=3000,max_wall_seconds=600,
        gate=dict(pooled_nmse_max=.1,each_position_nmse_max=.2,each_energy_min=.7,each_energy_max=1.3,consecutive_full_checks=3,check_interval=100),
        warm_trainable=train_names,warm_frozen=alpha_names,warm_density="Current softplus(raw_alpha)/far numerical value, detached at fine physical_density only; no cache",
        rho_may_change_indirectly_through_trunk=True,coarse_frozen=True,learning_rates=dict(warm=.0005,unfrozen=.00005),
        loss=".1 coarse + .9 fine raw complex batchNMSE",same_V8A_warm_draws=True,release_draws="Same first1000 V8A warm batches",
        no_density_or_z_buffer=True,diagnostic_only=True,formal_test=False,outer_reads=0,validation_reads=0)
    atomic_json(R/"run_manifest.json",dict(protocol=protocol,source_identity=source,GPU=guard))
    atomic_json(R/"axes.json",dict(row_ids=rows.tolist(),positions_m=data.positions(rows).tolist(),site_ids=data.sites(rows).tolist(),field=0,BS_ids=list(range(128)),UE_port=0,frequency_ids=list(range(192)),raw_H=True))
    # QA: no optimizer update. Only rho's backward path is changed.
    ids=draw_rows[0];bs=draw_bs[0];ff=draw_f[0];target=station_target_cached(data,ids,0,bs,ff,"cuda:0")
    opt.zero_grad(set_to_none=True);hn,cn=station_forward(model,cache,data,ids,0,bs,ff);ln,_,_=native_weighted_loss(hn,cn,target);ln.backward()
    native_grad=unit.group_norm(group,True);hn=hn.detach().clone();cn=cn.detach().clone()
    native_physical_density=model.fine[0].physical_density
    detach_checks=dict(calls=0,last_output_requires_grad=None)
    def detached_density(raw_alpha):
        value=native_physical_density(raw_alpha).detach()
        detach_checks["calls"]+=1;detach_checks["last_output_requires_grad"]=value.requires_grad
        return value
    model.fine[0].physical_density=detached_density
    opt.zero_grad(set_to_none=True);hd,cd=station_forward(model,cache,data,ids,0,bs,ff);ld,_,_=native_weighted_loss(hd,cd,target);ld.backward()
    detached_grad=unit.group_norm(group,True)
    qa=dict(source_sha256=sha256(Path(__file__)),native_vs_control_fine_bitwise=bool(torch.equal(hn,hd)),native_vs_control_coarse_bitwise=bool(torch.equal(cn,cd)),
        native_loss=float(ln),controlled_loss=float(ld),native_branch_gradient_norms=native_grad,controlled_branch_gradient_norms=detached_grad,
        controlled_trunk_gradient_nonzero=detached_grad["fine_position_trunk"]>1e-12,alpha_out_all_frozen=all(not p.requires_grad for p in alpha),
        alpha_out_all_grad_none=all(p.grad is None for p in alpha),controlled_density_output_requires_grad=detach_checks["last_output_requires_grad"],
        optimizer_updates=0,optimizer_state_still_empty=not bool(opt.state),same_actual_batch_labels=True,gradient_only_intervention=True,
        rho_values_are_not_held_constant=True,status="passed")
    if not all([qa["native_vs_control_fine_bitwise"],qa["native_vs_control_coarse_bitwise"],qa["controlled_trunk_gradient_nonzero"],qa["alpha_out_all_frozen"],qa["alpha_out_all_grad_none"],qa["optimizer_state_still_empty"]]) or qa["controlled_density_output_requires_grad"]:raise RuntimeError("Gradient-only QA failed")
    atomic_json(R/"qa.json",qa);print(json.dumps(dict(event="QA_PASSED_WAITING_START",qa=qa)),flush=True)
    opt.zero_grad(set_to_none=True);torch.set_rng_state(parent["torch_rng"]);torch.cuda.set_rng_state_all(parent["cuda_rng"])
    # Agent reports source and QA before writing this one-run start record.
    while not (R/"start_training.json").exists():time.sleep(.2)
    start=json.loads((R/"start_training.json").read_text())
    if start.get("source_sha256")!=sha256(Path(__file__)) or start.get("qa_sha256")!=sha256(R/"qa.json"):raise RuntimeError("Start record identity")
    unit.source_check(source);deadline=started+590
    def full(m=model):return fixed_field(m,cache,data,rows,0,np.arange(128,dtype=np.int64),np.arange(192,dtype=np.int64),deadline)
    def decomp():
        result=[];zz=[]
        for row in rows:
            d,z=unit.decomposition(model,cache,data,int(row),0,b,fid);result.append(d);zz.append(z)
        return result,zz
    pred,cpred=full()
    if not gate(metrics(cpred,truth),.05,.1):raise RuntimeError("Shared fitted coarse gate changed")
    atomic_json(R/"initial_metrics.json",dict(fine=metrics(pred,truth),coarse=metrics(cpred,truth)))
    atomic_json(R/"initial_density_profile.json",density_profile(model,cache,data,rows))
    history=[];traces=[];profiles=[];used=[];stages={};total=0;stable=0
    def save_ck(stage,step):
        return dict(model=unit.state(model),optimizer=copy.deepcopy(opt.state_dict()),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            stage=stage,stage_updates=step,total_updates=total,actual_used_draws=copy.deepcopy(used),stable_full_checks=stable,source_identity=source,protocol=protocol,
            fine_density_backward_detached=(stage=="warm"),row_ids=rows.tolist(),field=0,raw_H=True)
    atomic_torch_save(A/"initial.pt",save_ck("warm",0))
    def run(stage,budget):
        nonlocal total,stable
        passed=False
        for step in range(1,budget+1):
            j=step-1;ids=draw_rows[j];bs=draw_bs[j];ff=draw_f[j];used.append(dict(stage=stage,step=step,source_warm_index=int(j)))
            log=step==1 or step%100==0
            before,zb=decomp() if log else (None,None);old={k:[p.detach().clone() for p in ps] for k,ps in group.items()} if log else None
            model.train();opt.zero_grad(set_to_none=True);target=station_target_cached(data,ids,0,bs,ff,"cuda:0")
            h,c=station_forward(model,cache,data,ids,0,bs,ff);loss,lf,lc=native_weighted_loss(h,c,target)
            if not torch.isfinite(loss):raise FloatingPointError("Native loss")
            loss.backward();gn=unit.group_norm(group,True) if log else None
            params=[p for p in model.fine[0].parameters() if p.requires_grad];norm=torch.nn.utils.clip_grad_norm_(params,5.)
            if not torch.isfinite(norm):raise FloatingPointError("Gradient")
            opt.step();total+=1
            if log:
                after,za=decomp();traces.append(dict(stage=stage,update=step,total_updates=total,native_loss=float(loss),fine_loss=float(lf),coarse_loss=float(lc),
                    gradient_before_clip=float(norm),branch_gradient_norms=gn,parameter_update_norms={key:math.sqrt(sum(float((p.detach()-q).double().square().sum()) for p,q in zip(ps,old[key]))) for key,ps in group.items()},
                    before=before,after=after,fine_z_max_change_per_position_m=[float(np.max(abs(y-x))) for x,y in zip(zb,za)]))
                atomic_json(R/"mechanism_traces.json",traces)
            if step%100==0 or step==budget:
                model.eval();pred,cpred=full();fm=metrics(pred,truth);cm=metrics(cpred,truth);now=gate(fm)
                if stage=="warm":stable=stable+1 if now else 0
                profile=density_profile(model,cache,data,rows);profiles.append(dict(stage=stage,step=step,positions=profile));atomic_json(R/"density_profiles_by_checkpoint.json",profiles)
                event=dict(stage=stage,update=step,total_updates=total,fine=fm,coarse=cm,gate_now=now,stable_full_checks=stable,elapsed=time.monotonic()-started)
                history.append(event);atomic_json(R/"progress.json",event);atomic_torch_save(A/"last.pt",save_ck(stage,step))
                print(json.dumps(dict(stage=stage,step=step,pooled_nmse=fm["nmse"],per_position=[x["nmse"] for x in fm["per_position"]],energy=[x["energy_ratio"] for x in fm["per_position"]],stable=stable,elapsed=time.monotonic()-started)),flush=True)
                if stage=="warm" and stable>=3:passed=True;break
        aa=A/stage;aa.mkdir(exist_ok=True);pred,cpred=full()
        atomic_npy(aa/"full_field_prediction.npy",pred);atomic_npy(aa/"coarse_full_field_prediction.npy",cpred);atomic_torch_save(aa/"final.pt",save_ck(stage,step))
        stages[stage]=dict(updates=step,stable_pass=passed if stage=="warm" else None,fine=metrics(pred,truth),coarse=metrics(cpred,truth))
        atomic_json(R/(stage+"_result.json"),stages[stage]);atomic_json(R/(stage+"_density_profile.json"),density_profile(model,cache,data,rows))
        return passed
    warm_pass=run("warm",2000)
    if warm_pass:
        pre,prec=full();atomic_npy(A/"release_before_full_field_prediction.npy",pre)
        delattr(model.fine[0],"physical_density")
        for p in alpha:p.requires_grad_(True)
        for g in opt.param_groups:g["lr"]=.00005
        post,postc=full();atomic_npy(A/"release_after_full_field_prediction.npy",post)
        release=dict(forward_bitwise_no_jump=bool(np.array_equal(pre,post) and np.array_equal(prec,postc)),physical_density_instance_override_removed="physical_density" not in model.fine[0].__dict__,
            alpha_states_empty=all(p not in opt.state for p in alpha),active_moments_preserved=sum(p in opt.state for p in active),all_fine_trainable=all(p.requires_grad for p in model.fine[0].parameters()))
        atomic_json(R/"release_check.json",release)
        if not all([release["forward_bitwise_no_jump"],release["physical_density_instance_override_removed"],release["alpha_states_empty"],release["all_fine_trainable"]]):raise RuntimeError("Native release QA")
        run("unfrozen",1000)
    if not all(torch.equal(v,frozen_coarse[k]) for k,v in unit.state(model.coarse[0]).items()):raise RuntimeError("Coarse changed")
    stage="unfrozen" if warm_pass else "warm";pred,cpred=full();atomic_npy(A/"full_field_prediction.npy",pred);atomic_npy(A/"coarse_full_field_prediction.npy",cpred)
    atomic_torch_save(A/"final.pt",save_ck(stage,stages[stage]["updates"]))
    clean=build_native_batch_model(data,cache,128,128,frequency_chunk=8).to("cuda:0");disk=torch.load(A/"final.pt",map_location="cpu",weights_only=False);clean.load_state_dict(disk["model"]);clean.eval()
    clean_pred,clean_coarse=full(clean);atomic_npy(A/"clean_reload_full_field_prediction.npy",clean_pred)
    if not np.array_equal(clean_pred,pred) or not np.array_equal(clean_coarse,cpred):raise RuntimeError("Clean original native reload mismatch")
    unit.source_check(source)
    result=dict(status="full_two_position_field_pass" if warm_pass and gate(metrics(pred,truth)) else "warm_not_stably_passed" if not warm_pass else "unfrozen_failed_gate",
        stages=stages,total_updates=total,elapsed_seconds=time.monotonic()-started,final_fine=metrics(pred,truth),final_coarse=metrics(cpred,truth),
        clean_reload_bitwise_identical=True,coarse_weights_unchanged=True,history=history,actual_used_draws=used,source_identity=source,
        prediction_sha256=sha256(A/"full_field_prediction.npy"),checkpoint_sha256=sha256(A/"final.pt"),
        rho_changed_indirectly_during_warm=True,no_density_or_z_buffer=True,raw_H=True,diagnostic_only=True,formal_test=False,outer_reads=0,validation_reads=0)
    atomic_json(R/"result.json",result);signal.alarm(0);print(json.dumps({k:v for k,v in result.items() if k not in ["history","actual_used_draws","source_identity"]}),flush=True)
if __name__=="__main__":
    try:main()
    except BaseException:
        R.mkdir(parents=True,exist_ok=True);atomic_json(R/"failure.json",dict(traceback=traceback.format_exc(),formal_test=False));raise
