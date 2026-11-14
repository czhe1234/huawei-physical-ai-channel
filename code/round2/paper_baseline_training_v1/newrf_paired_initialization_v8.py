"""Paired native fine initialization control: old fine vs transfer coarse. No render changes."""
from __future__ import annotations
import copy,json,math,signal,time,traceback
from pathlib import Path
import numpy as np
import torch
from .common import ART,REPORT,FoldData,seed_all,device_guard,sha256,atomic_json,atomic_npy,atomic_torch_save
from .newrf_runner import GEOMETRY,FitGeometryBank
from .newrf_native_batch_v4 import build_native_batch_model,station_forward,native_weighted_loss
from .newrf_runtime_cache_v4 import MemoizedFitDirections,station_target_cached
from .newrf_latefreq import stratified64,density_weights,hierarchical64
from .newrf_balanced_dev_v1 import metrics,fixed_field
from . import newrf_minimal_fullH_v2 as unit
from project_paths import PATHS
A=ART/"fit_repair_paper_v8_paired_initialization_20261101/newrf"
R=REPORT/"fit_repair_paper_v8_paired_initialization_20261101/newrf"
V7A=ART/"fit_repair_paper_v7a_staged_two_positions_20261101/newrf"
V7R=REPORT/"fit_repair_paper_v7a_staged_two_positions_20261101/newrf"
V6=ART/"fit_repair_paper_v6_staged_joint_20261101/newrf/final.pt"
class BudgetExpired(RuntimeError):pass
def alarm(sig,frame):raise BudgetExpired("600s paired-initialization control")
def gate(summary,pooled=.1,each=.2):
    return bool(summary["nmse"]<=pooled and all(x["nmse"]<=each and .7<=x["energy_ratio"]<=1.3 for x in summary["per_position"]))
@torch.no_grad()
def density_profile(model,cache,data,rows,field=0):
    out=[]
    for row in rows:
        dev=model.bs_ports.device
        pos=torch.tensor(data.positions([row]),device=dev,dtype=torch.float64);site=torch.tensor(data.sites([row]),device=dev,dtype=torch.long)
        origins=model.spec.ue_world_positions(pos,site)[0];tx=model.bs_ports[site[0]];profile=[]
        for first in range(0,128,8):
            bs=np.arange(first,min(first+8,128),dtype=np.int64);dd=[];oo=[];tt=[];counts=[]
            actual=cache.scalar(model,int(row),field,torch.tensor(bs,device=dev),origins,site[0],tx)
            for j,bb in enumerate(bs):
                d=actual[j];n=len(d);counts.append(n);dd.append(d);oo.append(origins[0].expand(n,3));tt.append(tx[0,int(bb)].expand(n,3))
            dd=torch.cat(dd);o=torch.cat(oo);t=torch.cat(tt);pp,z=stratified64(o,dd,model.config.far_m,model.config.n_samples)
            _,cr=model.coarse[field].prepare(pp,dd[:,None].expand_as(pp),t[:,None].expand_as(pp));cw=density_weights(cr,z)
            fp,fz=hierarchical64(o,dd,z,cw,model.config.n_importance);_,fr=model.fine[field].prepare(fp,dd[:,None].expand_as(fp),t[:,None].expand_as(fp));fw=density_weights(fr,fz)
            cr=cr.cpu().numpy();fr=fr.cpu().numpy();cw=cw.sum(-1).cpu().numpy();fw=fw.sum(-1).cpu().numpy();slot=0
            for bb,n in zip(bs,counts):
                q=fr[slot:slot+n];profile.append(dict(BS=int(bb),rays=n,fine_rho_mean=float(q.mean()),fine_rho_min=float(q.min()),fine_rho_max=float(q.max()),
                    fine_sumweight_mean=float(fw[slot:slot+n].mean()),coarse_rho_mean=float(cr[slot:slot+n].mean()),coarse_sumweight_mean=float(cw[slot:slot+n].mean())))
                slot+=n
        out.append(dict(row=int(row),BS_profile=profile))
    return out
def main():
    started=time.monotonic();signal.signal(signal.SIGALRM,alarm);signal.alarm(600)
    A.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
    if (R/"run_manifest.json").exists():raise FileExistsError("Preserve prior paired unit")
    panel_path=REPORT/"fit_repair_paper_v2_20261101/panel_v2.json";panel=json.loads(panel_path.read_text());rows=np.array([2666,2042],np.int64);field=0;b=30;fid=96
    parent_path=V7A/"coarse/final.pt";parent=torch.load(parent_path,map_location="cpu",weights_only=False);oldfine=torch.load(V6,map_location="cpu",weights_only=False)
    unit.source_check(parent["source_identity"]);unit.source_check(oldfine["source_identity"])
    source=unit.source_identity(REPORT/"fit_repair_paper_v2_20261101/contract.json",panel_path)
    source_files=[Path(__file__),parent_path,V6,V7R/"label_stages.json"]+[V7A/n for n in ["supervised_row_ids.npy","supervised_BS_ids.npy","supervised_frequency_ids.npy","supervised_raw_labels.npy"]]
    for p in source_files:source[PATHS.relative(p)]=sha256(p)
    stages=json.loads((V7R/"label_stages.json").read_text());warm_indices=np.array([i for i,x in enumerate(stages) if x["stage"]=="warm"],np.int64)
    if len(warm_indices)!=2000:raise RuntimeError("Exactly 2000 actual V7 warm draws required")
    draw_rows=np.load(V7A/"supervised_row_ids.npy")[warm_indices];draw_bs=np.load(V7A/"supervised_BS_ids.npy")[warm_indices]
    draw_f=np.load(V7A/"supervised_frequency_ids.npy")[warm_indices];draw_y=np.load(V7A/"supervised_raw_labels.npy")[warm_indices]
    data=FoldData(0,"screen","newrf")
    if not set(map(int,rows)).issubset(set(map(int,data.fit_ids))) or not np.all(data.sites(rows)==0) or data.inputs.identity!=panel["input_identity"]:raise ValueError("Two FIT site0 locations")
    guard=device_guard(unit.GPU);torch.set_num_threads(4);seed_all(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    bank=FitGeometryBank(GEOMETRY,data.inputs,outer=0,role="screen_fit");cache=MemoizedFitDirections(bank).to("cuda:0")
    raw=data.read_fit(rows,normalized=False);truth=raw[:,:128,0];slotmap={int(v):i for i,v in enumerate(rows)}
    slots=np.full(4000,-1,np.int64);slots[rows]=np.arange(2);data._native_raw_target_cache=(slots,torch.from_numpy(raw).to("cuda:0"))
    ii=np.vectorize(slotmap.__getitem__)(draw_rows)
    if not np.array_equal(draw_y,truth[ii,draw_bs,draw_f]):raise RuntimeError("Actual draws do not match raw labels")
    if not np.all((draw_rows==rows[0]).sum(1)==16) or not np.all((draw_rows==rows[1]).sum(1)==16):raise RuntimeError("32 draw position balance")
    atomic_npy(A/"actual_target_field.npy",truth);atomic_npy(A/"field_row_ids.npy",rows)
    atomic_npy(A/"warm_row_ids.npy",draw_rows);atomic_npy(A/"warm_BS_ids.npy",draw_bs);atomic_npy(A/"warm_frequency_ids.npy",draw_f);atomic_npy(A/"warm_raw_labels.npy",draw_y)
    protocol=dict(schema="newrf-paired-fine-initialization-v8",arms=["A_reset_old_fine","B_transfer_coarse"],row_ids=rows.tolist(),field=0,BS_ids=list(range(128)),UE_port=0,frequency_ids=list(range(192)),
        warm_budget_per_arm=2000,unfrozen_budget_if_pass_per_arm=1000,max_total_updates=6000,max_wall_seconds=600,
        warm_gate=dict(pooled_nmse_max=.1,each_position_nmse_max=.2,each_energy_min=.7,each_energy_max=1.3),
        coarse="Same V7A fitted two-position coarse weights, frozen in both arms",
        A_initialization="Fine field0 from V6 healthy one-position model",B_initialization="Fine field0 copies entire current coarse field0 state",
        only_arm_difference="Fine initialization; both fine Adam states empty",
        warm_trainable="realimag_filters/branch/output",warm_frozen="layers/alpha_out",learning_rates=dict(warm=.0005,unfrozen=.00005),
        warm_draws="Same 2000 actual V7A warm draws in same order",unfrozen_draws="First1000 same actual warm draws reused identically across passing arms",
        loss=".1 coarse + .9 fine raw complex batchNMSE",native_forward_unchanged=True,no_density_or_z_buffer=True,
        diagnostic_only=True,formal_test=False,outer_reads=0,validation_reads=0)
    atomic_json(R/"run_manifest.json",dict(protocol=protocol,source_identity=source,GPU=guard))
    atomic_json(R/"axes.json",dict(row_ids=rows.tolist(),positions_m=data.positions(rows).tolist(),site_ids=data.sites(rows).tolist(),field=0,BS_ids=list(range(128)),UE_port=0,frequency_ids=list(range(192)),raw_H=True))
    deadline=started+590;total=0;arm_results={}
    for arm in protocol["arms"]:
        aa=A/arm;rr=R/arm;aa.mkdir(exist_ok=True);rr.mkdir(exist_ok=True)
        model=build_native_batch_model(data,cache,128,128,frequency_chunk=8).to("cuda:0");model.load_state_dict(parent["model"])
        if arm=="A_reset_old_fine":
            prefix="fine.0.";model.fine[0].load_state_dict({k[len(prefix):]:v for k,v in oldfine["model"].items() if k.startswith(prefix)})
        else:model.fine[0].load_state_dict(model.coarse[0].state_dict())
        for p in model.parameters():p.requires_grad_(False)
        named=dict(model.fine[0].named_parameters());signal_names=[n for n in named if n.startswith(("network.realimag_filters.","network.branch.","network.output."))]
        density_names=[n for n in named if n.startswith(("network.layers.","network.alpha_out."))]
        if set(signal_names)|set(density_names)!=set(named):raise RuntimeError("Parameter dependency classification")
        sp=[named[n] for n in signal_names];dp=[named[n] for n in density_names]
        for p in sp:p.requires_grad_(True)
        opt=torch.optim.Adam([dict(params=sp),dict(params=dp)],lr=.0005,weight_decay=0.)
        if opt.state:raise RuntimeError("Both arms must start empty Adam")
        torch.set_rng_state(parent["torch_rng"]);torch.cuda.set_rng_state_all(parent["cuda_rng"])
        group=unit.groups(model,0);frozen_coarse=unit.state(model.coarse[0]);history=[];traces=[];used=[];stage_results={}
        def full(m=model):return fixed_field(m,cache,data,rows,0,np.arange(128,dtype=np.int64),np.arange(192,dtype=np.int64),deadline)
        def decomp():
            result=[];zz=[]
            for row in rows:
                d,z=unit.decomposition(model,cache,data,int(row),0,b,fid);result.append(d);zz.append(z)
            return result,zz
        def save_ck(stage,step):
            return dict(model=unit.state(model),optimizer=copy.deepcopy(opt.state_dict()),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                stage=stage,stage_updates=step,actual_used_draws=copy.deepcopy(used),arm=arm,source_identity=source,protocol=protocol,row_ids=rows.tolist(),field=0,raw_H=True)
        pred,cpred=full()
        if not gate(metrics(cpred,truth),.05,.1):raise RuntimeError("Shared coarse no longer meets parent gate")
        atomic_json(rr/"initial_metrics.json",dict(fine=metrics(pred,truth),coarse=metrics(cpred,truth)))
        warm_profile=density_profile(model,cache,data,rows);atomic_json(rr/"initial_density_profile.json",warm_profile)
        atomic_torch_save(aa/"initial.pt",save_ck("initial",0))
        def run(stage,budget):
            nonlocal total
            passed=False
            for step in range(1,budget+1):
                j=step-1;ids=draw_rows[j];bs=draw_bs[j];ff=draw_f[j];used.append(dict(stage=stage,step=step,source_warm_index=int(j)))
                log=step==1 or step%100==0
                before,zbefore=decomp() if log else (None,None)
                old={key:[p.detach().clone() for p in ps] for key,ps in group.items()} if log else None
                model.train();opt.zero_grad(set_to_none=True);target=station_target_cached(data,ids,0,bs,ff,"cuda:0")
                h,c=station_forward(model,cache,data,ids,0,bs,ff);loss,lf,lc=native_weighted_loss(h,c,target)
                if not torch.isfinite(loss):raise FloatingPointError("Native loss")
                loss.backward();gn=unit.group_norm(group,True) if log else None
                active=[p for p in model.fine[0].parameters() if p.requires_grad];norm=torch.nn.utils.clip_grad_norm_(active,5.)
                if not torch.isfinite(norm):raise FloatingPointError("Gradient")
                opt.step();total+=1
                if log:
                    after,zafter=decomp()
                    traces.append(dict(stage=stage,update=step,total_pair_updates=total,native_loss=float(loss),fine_loss=float(lf),coarse_loss=float(lc),gradient_before_clip=float(norm),
                        branch_gradient_norms=gn,parameter_update_norms={key:math.sqrt(sum(float((p.detach()-q).double().square().sum()) for p,q in zip(ps,old[key]))) for key,ps in group.items()},
                        before=before,after=after,fine_z_max_change_per_position_m=[float(np.max(abs(y-x))) for x,y in zip(zbefore,zafter)]))
                    atomic_json(rr/"mechanism_traces.json",traces)
                if step%100==0 or step==budget:
                    model.eval();pred,cpred=full();fm=metrics(pred,truth);cm=metrics(cpred,truth);event=dict(stage=stage,update=step,fine=fm,coarse=cm,elapsed=time.monotonic()-started)
                    history.append(event);atomic_json(rr/"progress.json",event);atomic_json(R/"progress.json",dict(arm=arm,total_updates=total,**event))
                    print(json.dumps(dict(arm=arm,stage=stage,update=step,pooled_nmse=fm["nmse"],per_position=[x["nmse"] for x in fm["per_position"]],energy=[x["energy_ratio"] for x in fm["per_position"]],elapsed=time.monotonic()-started)),flush=True)
                    if stage=="warm" and gate(fm):passed=True;break
            stage_dir=aa/stage;stage_dir.mkdir(exist_ok=True);pred,cpred=full()
            atomic_npy(stage_dir/"full_field_prediction.npy",pred);atomic_npy(stage_dir/"coarse_full_field_prediction.npy",cpred);atomic_torch_save(stage_dir/"final.pt",save_ck(stage,step))
            stage_results[stage]=dict(updates=step,gate_pass=passed if stage=="warm" else gate(metrics(pred,truth)),fine=metrics(pred,truth),coarse=metrics(cpred,truth))
            atomic_json(rr/(stage+"_density_profile.json"),density_profile(model,cache,data,rows));atomic_json(rr/(stage+"_result.json"),stage_results[stage])
            return passed
        warm_pass=run("warm",2000)
        if density_profile(model,cache,data,rows)!=warm_profile:raise RuntimeError("Warm density function changed")
        if warm_pass:
            for p in dp:p.requires_grad_(True)
            for g in opt.param_groups:g["lr"]=.00005
            if any(p in opt.state for p in dp):raise RuntimeError("Unfrozen Adam moments not empty")
            run("unfrozen",1000)
        if not all(torch.equal(v,frozen_coarse[k]) for k,v in unit.state(model.coarse[0]).items()):raise RuntimeError("Shared coarse changed")
        pred,cpred=full();atomic_npy(aa/"full_field_prediction.npy",pred);atomic_npy(aa/"coarse_full_field_prediction.npy",cpred)
        laststage="unfrozen" if warm_pass else "warm";atomic_torch_save(aa/"final.pt",save_ck(laststage,stage_results[laststage]["updates"]))
        clean=build_native_batch_model(data,cache,128,128,frequency_chunk=8).to("cuda:0");disk=torch.load(aa/"final.pt",map_location="cpu",weights_only=False);clean.load_state_dict(disk["model"]);clean.eval()
        clean_pred,clean_coarse=full(clean);atomic_npy(aa/"clean_reload_full_field_prediction.npy",clean_pred)
        if not np.array_equal(clean_pred,pred) or not np.array_equal(clean_coarse,cpred):raise RuntimeError("Clean original native reload differs")
        result=dict(status="full_two_position_field_pass" if warm_pass and gate(metrics(pred,truth)) else "warm_failed_gate" if not warm_pass else "unfrozen_failed_gate",
            arm=arm,stage_results=stage_results,final_fine=metrics(pred,truth),final_coarse=metrics(cpred,truth),history=history,
            clean_reload_bitwise_identical=True,coarse_weights_unchanged=True,initial_Adam_empty=True,native_forward_unchanged=True,
            prediction_sha256=sha256(aa/"full_field_prediction.npy"),checkpoint_sha256=sha256(aa/"final.pt"),source_identity=source,actual_used_draws=used,
            no_density_or_z_buffer=True,raw_H=True,diagnostic_only=True,formal_test=False,outer_reads=0,validation_reads=0)
        atomic_json(rr/"result.json",result);arm_results[arm]={k:result[k] for k in ["status","stage_results","final_fine","final_coarse","clean_reload_bitwise_identical","prediction_sha256","checkpoint_sha256"]}
        unit.source_check(source)
        del clean,model,opt,group,disk
        torch.cuda.empty_cache()
    unit.source_check(source)
    final=dict(status="paired_control_completed",arms=arm_results,total_updates=total,elapsed_seconds=time.monotonic()-started,source_identity=source,protocol=protocol,
        diagnostic_only=True,not_paper_performance_score=True,formal_test=False,outer_reads=0,validation_reads=0)
    atomic_json(R/"result.json",final);signal.alarm(0);print(json.dumps({k:v for k,v in final.items() if k not in ["source_identity","protocol"]}),flush=True)
if __name__=="__main__":
    try:main()
    except BaseException:
        R.mkdir(parents=True,exist_ok=True);atomic_json(R/"failure.json",dict(traceback=traceback.format_exc(),formal_test=False));raise
