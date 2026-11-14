"""Native NeWRF optimization schedule diagnostic. No render, rho or z replacement."""
from __future__ import annotations
import copy,json,math,signal,time,traceback
from pathlib import Path
import numpy as np
import torch
from .common import ART,REPORT,FoldData,seed_all,device_guard,sha256,atomic_json,atomic_npy,atomic_torch_save
from .newrf_runner import GEOMETRY,FitGeometryBank
from .newrf_native_batch_v4 import build_native_batch_model,station_forward,native_weighted_loss
from .newrf_runtime_cache_v4 import MemoizedFitDirections,station_target_cached
from .newrf_balanced_dev_v1 import metrics
from . import newrf_minimal_fullH_v2 as unit
from project_paths import PATHS
A=ART/"fit_repair_paper_v5_staged_native_20261101/newrf"
R=REPORT/"fit_repair_paper_v5_staged_native_20261101/newrf"
BASE_A=ART/"fit_repair_paper_v2_20261101/newrf"
class BudgetExpired(RuntimeError):pass
def alarm(sig,frame):raise BudgetExpired("240s staged-native unit")
def cpu_opt_state(value):
    return copy.deepcopy(value)
def main():
    started=time.monotonic();signal.signal(signal.SIGALRM,alarm);signal.alarm(240)
    A.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
    if (R/"run_manifest.json").exists():raise FileExistsError("Preserve prior staged-native unit")
    scalar_path=BASE_A/"scalar/final.pt";coarse_path=BASE_A/"frequency_only/final.pt"
    panel_path=REPORT/"fit_repair_paper_v2_20261101/panel_v2.json"
    panel=json.loads(panel_path.read_text());row=int(panel["primary_row"]);b=int(panel["primary_BS_spatial_id"]);fid=int(panel["primary_frequency_id"]);field=0
    sources=unit.source_identity(REPORT/"fit_repair_paper_v2_20261101/contract.json",panel_path)
    for p in [Path(__file__),scalar_path,coarse_path]:sources[PATHS.relative(p)]=sha256(p)
    scalar=torch.load(scalar_path,map_location="cpu",weights_only=False);coarse=torch.load(coarse_path,map_location="cpu",weights_only=False)
    unit.source_check(scalar["source_identity"]);unit.source_check(coarse["source_identity"])
    data=FoldData(0,"screen","newrf")
    if row not in set(data.fit_ids) or data.inputs.identity!=panel["input_identity"]:raise ValueError("Frozen FIT identity")
    guard=device_guard(unit.GPU);torch.set_num_threads(4);seed_all(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    bank=FitGeometryBank(GEOMETRY,data.inputs,outer=0,role="screen_fit");cache=MemoizedFitDirections(bank).to("cuda:0")
    model=build_native_batch_model(data,cache,128,128,frequency_chunk=8).to("cuda:0");model.load_state_dict(scalar["model"])
    prefix="coarse.0.";model.coarse[field].load_state_dict({k[len(prefix):]:v for k,v in coarse["model"].items() if k.startswith(prefix)})
    raw=data.read_fit(np.array([row]),normalized=False);truth=raw[:,:128,0];target_axis=truth[:,[b],:]
    slots=np.full(4000,-1,np.int64);slots[row]=0;data._native_raw_target_cache=(slots,torch.from_numpy(raw).to("cuda:0"))
    atomic_npy(A/"actual_target_field.npy",truth);atomic_npy(A/"field_row_ids.npy",np.array([row],np.int64))
    # prepare(): rho = softplus(alpha_out(layers(PE(position,tx)))) / far.
    # realimag_filters/branch/output have no dependency path into rho.
    for p in model.parameters():p.requires_grad_(False)
    named=dict(model.fine[field].named_parameters())
    signal_names=[n for n in named if n.startswith(("network.realimag_filters.","network.branch.","network.output."))]
    density_names=[n for n in named if n.startswith(("network.layers.","network.alpha_out."))]
    if set(signal_names)|set(density_names)!=set(named) or set(signal_names)&set(density_names):raise RuntimeError("Unclassified dependency")
    for name in signal_names:named[name].requires_grad_(True)
    signal_params=[named[n] for n in signal_names];density_params=[named[n] for n in density_names]
    opt=torch.optim.Adam(signal_params,lr=.0005,weight_decay=0.)
    # Restore scalar moments only for already-trained signal parameters.
    old_order=list(model.coarse[field].parameters())+list(model.fine[field].parameters())
    old_ids=[i for g in scalar["optimizer"]["param_groups"] for i in g["params"]]
    if len(old_order)!=len(old_ids):raise RuntimeError("Scalar Adam parameter order")
    old_id_by_object={id(p):i for p,i in zip(old_order,old_ids)}
    for p in signal_params:
        old=scalar["optimizer"]["state"].get(old_id_by_object[id(p)])
        if old is not None:
            opt.state[p]=copy.deepcopy(old)
            for key,value in opt.state[p].items():
                if isinstance(value,torch.Tensor):opt.state[p][key]=value.to("cuda:0")
    torch.set_rng_state(scalar["torch_rng"]);torch.cuda.set_rng_state_all(scalar["cuda_rng"])
    sampler=unit.Axes("frequency_only",b,fid,scalar["numpy_rng"]);group=unit.groups(model,field)
    initial_decomp,initial_z=unit.decomposition(model,cache,data,row,field,b,fid)
    initial_rho=initial_decomp["fine"]["rho"];initial_coarse=unit.state(model.coarse[field])
    protocol=dict(schema="newrf-native-staged-diagnostic-v5",warm_max_updates=1000,unfrozen_max_updates=1000,max_updates=2000,max_wall_seconds=240,
        coarse_parent_sha256=sha256(coarse_path),scalar_fine_parent_sha256=sha256(scalar_path),row=row,field=field,BS=b,
        forward="Unmodified newrf_native_batch_v4.station_forward; original dynamic hierarchical64",
        loss=".1 coarse + .9 fine raw complex batchNMSE",coarse_frozen_all_stages=True,
        warm_frozen=density_names,warm_trainable=signal_names,warm_lr=.0005,unfrozen_lr=.00005,
        optimizer="Signal Adam moments restored from scalar, retained across stages; unfrozen params enter with empty state",
        no_density_or_z_buffer=True,diagnostic_only=True,formal_test=False,outer_reads=0,validation_reads=0)
    atomic_json(R/"run_manifest.json",dict(protocol=protocol,source_identity=sources,GPU=guard,model_config=__import__("dataclasses").asdict(model.config)))
    atomic_json(R/"axis.json",dict(row_ids=[row],field=field,BS_ids=[b],UE_port=0,frequency_ids=list(range(192)),raw_H=True))
    atomic_json(R/"initial_decomposition.json",initial_decomp)
    def axis(m=model):
        vv=[];cc=[]
        with torch.no_grad():
            for first in range(0,192,32):
                fs=np.arange(first,first+32,dtype=np.int64);bs=np.full(32,b,np.int64);ids=np.full(32,row,np.int64)
                h,c=station_forward(m,cache,data,ids,field,bs,fs);vv.append(h.cpu().numpy());cc.append(c.cpu().numpy())
        return np.concatenate(vv)[None,None],np.concatenate(cc)[None,None]
    labels=[];traces=[];history=[];stages={};total=0
    def save_checkpoint(stage,step):
        return dict(model=unit.state(model),optimizer=copy.deepcopy(opt.state_dict()),numpy_rng=copy.deepcopy(sampler.rng.bit_generator.state),
            sampler=sampler.snapshot(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),stage=stage,stage_updates=step,total_updates=total,
            source_identity=sources,protocol=protocol,row=row,field=field,raw_H=True)
    def run_stage(stage,budget):
        nonlocal total
        last=None;passed=False
        for step in range(1,budget+1):
            bs,ff=sampler.draw();ids=np.full(32,row,np.int64);labels.append((stage,step,bs.copy(),ff.copy(),truth[0,bs,ff].copy()))
            log=step==1 or step%25==0
            before,zb=unit.decomposition(model,cache,data,row,field,b,fid) if log else (None,None)
            weights_before={key:[p.detach().clone() for p in ps] for key,ps in group.items()} if log else None
            model.train();opt.zero_grad(set_to_none=True)
            target=station_target_cached(data,ids,field,bs,ff,"cuda:0");h,c=station_forward(model,cache,data,ids,field,bs,ff)
            loss,lf,lc=native_weighted_loss(h,c,target)
            if not torch.isfinite(loss):raise FloatingPointError("Native loss")
            loss.backward();gn=unit.group_norm(group,True) if log else None
            trainable=[p for p in model.fine[field].parameters() if p.requires_grad]
            norm=torch.nn.utils.clip_grad_norm_(trainable,5.)
            if not torch.isfinite(norm):raise FloatingPointError("Gradient")
            opt.step();total+=1
            if log:
                after,za=unit.decomposition(model,cache,data,row,field,b,fid)
                if stage=="warm" and after["fine"]["rho"]!=initial_rho:raise RuntimeError("Warm rho dependency changed")
                if not np.array_equal(za,initial_z):raise RuntimeError("Frozen coarse changed native CDF")
                d=dict(stage=stage,update=step,total_updates=total,native_loss=float(loss),fine_loss=float(lf),coarse_loss=float(lc),gradient_before_clip=float(norm),
                    branch_gradient_norms=gn,parameter_update_norms={key:math.sqrt(sum(float((p.detach()-q).double().square().sum()) for p,q in zip(ps,weights_before[key]))) for key,ps in group.items()},
                    before=before,after=after,fine_z_change_max_m=float(np.max(abs(za-zb))),carrier_phase_change_mean_rad=float(np.mean(abs(za-zb))*2*np.pi*data.frequencies([fid])[0]/299792458.))
                traces.append(d);atomic_json(R/"mechanism_traces.json",traces)
            if step%25==0 or step==budget:
                model.eval();pred,cpred=axis();fm=metrics(pred,target_axis);cm=metrics(cpred,target_axis)
                last=dict(stage=stage,update=step,fine=fm,coarse=cm);history.append(last);atomic_json(R/"progress.json",dict(**last,total_updates=total,elapsed=time.monotonic()-started))
                print(json.dumps(dict(stage=stage,step=step,fine_nmse=fm["nmse"],fine_energy=fm["energy_ratio"],elapsed=time.monotonic()-started)),flush=True)
                if stage=="warm" and fm["nmse"]<=.1 and .7<=fm["energy_ratio"]<=1.3:passed=True;break
        stage_dir=A/stage;stage_dir.mkdir(exist_ok=True)
        pred,cpred=axis();atomic_npy(stage_dir/"wideband_prediction.npy",pred);atomic_npy(stage_dir/"wideband_coarse_prediction.npy",cpred)
        atomic_torch_save(stage_dir/"final.pt",save_checkpoint(stage,step))
        stages[stage]=dict(updates=step,fine=metrics(pred,target_axis),coarse=metrics(cpred,target_axis),gate_pass=passed if stage=="warm" else metrics(pred,target_axis)["nmse"]<=.1)
        return passed
    warm_pass=run_stage("warm",1000)
    if warm_pass:
        for p in density_params:p.requires_grad_(True)
        for g in opt.param_groups:g["lr"]=.00005
        if any(p in opt.state for p in density_params):raise RuntimeError("Newly unfrozen parameter moments must be empty")
        opt.add_param_group(dict(params=density_params,lr=.00005))
        atomic_json(R/"unfreeze_optimizer.json",dict(signal_state_entries=sum(p in opt.state for p in signal_params),new_parameter_states_before_step=sum(p in opt.state for p in density_params),
            added_parameter_names=density_names,learning_rates=[g["lr"] for g in opt.param_groups]))
        run_stage("unfrozen",1000)
    if not all(torch.equal(v,initial_coarse[k]) for k,v in unit.state(model.coarse[field]).items()):raise RuntimeError("Coarse weights changed")
    pred,cpred=axis();atomic_npy(A/"wideband_prediction.npy",pred);atomic_npy(A/"wideband_coarse_prediction.npy",cpred)
    final_checkpoint=A/"final.pt";atomic_torch_save(final_checkpoint,save_checkpoint("unfrozen" if warm_pass else "warm",stages["unfrozen" if warm_pass else "warm"]["updates"]))
    # Fresh constructor; no instance monkeypatch, density buffer, coordinate cache or render override.
    clean=build_native_batch_model(data,cache,128,128,frequency_chunk=8).to("cuda:0")
    disk=torch.load(final_checkpoint,map_location="cpu",weights_only=False);clean.load_state_dict(disk["model"]);clean.eval()
    clean_pred,clean_coarse=axis(clean);atomic_npy(A/"clean_reload_wideband_prediction.npy",clean_pred)
    clean_error=float(np.linalg.norm(clean_pred-pred)/max(np.linalg.norm(pred),1e-30))
    if not np.array_equal(clean_pred,pred) or not np.array_equal(clean_coarse,cpred):raise RuntimeError("Fresh unpatched native inference differs")
    atomic_npy(A/"supervised_BS_ids.npy",np.stack([x[2] for x in labels]));atomic_npy(A/"supervised_frequency_ids.npy",np.stack([x[3] for x in labels]));atomic_npy(A/"supervised_raw_labels.npy",np.stack([x[4] for x in labels]))
    atomic_json(R/"label_stages.json",[dict(stage=x[0],step=x[1],row=row) for x in labels])
    unit.source_check(sources)
    result=dict(status="completed_native_diagnostic",stages=stages,total_updates=total,elapsed_seconds=time.monotonic()-started,final_fine=metrics(pred,target_axis),final_coarse=metrics(cpred,target_axis),
        history=history,clean_reload_bitwise_identical=True,clean_reload_relative_L2=clean_error,coarse_weights_unchanged=True,
        no_density_or_z_buffer=True,native_forward_unchanged=True,source_identity=sources,checkpoint_sha256=sha256(final_checkpoint),prediction_sha256=sha256(A/"wideband_prediction.npy"),
        warm_initial_fine_rho=initial_rho,final_decomposition=unit.decomposition(model,cache,data,row,field,b,fid)[0],raw_H=True,
        diagnostic_only=True,not_full_H_result=True,formal_test=False,outer_reads=0,validation_reads=0)
    atomic_json(R/"result.json",result);signal.alarm(0);print(json.dumps({k:v for k,v in result.items() if k not in ["history","final_decomposition","source_identity"]}),flush=True)
if __name__=="__main__":
    try:main()
    except BaseException:
        R.mkdir(parents=True,exist_ok=True);atomic_json(R/"failure.json",dict(traceback=traceback.format_exc(),formal_test=False));raise
