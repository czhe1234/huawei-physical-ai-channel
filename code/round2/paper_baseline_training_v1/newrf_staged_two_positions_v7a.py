"""Two FIT locations at one site, one field; native staged optimization, no render replacement."""
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
A=ART/"fit_repair_paper_v7a_staged_two_positions_20261101/newrf"
R=REPORT/"fit_repair_paper_v7a_staged_two_positions_20261101/newrf"
PARENT=ART/"fit_repair_paper_v6_staged_joint_20261101/newrf/final.pt"
class TwoPointSampler:
    def __init__(self,rows,rngstate):
        self.rows=np.asarray(rows,np.int64);self.rng=np.random.default_rng(42);self.rng.bit_generator.state=copy.deepcopy(rngstate)
        self.buffers={};self.indices={}
    def take(self,row,axis,size):
        key=(int(row),axis);out=[]
        for _ in range(16):
            if key not in self.buffers or self.indices[key]==size:
                self.buffers[key]=self.rng.permutation(size);self.indices[key]=0
            out.append(int(self.buffers[key][self.indices[key]]));self.indices[key]+=1
        return np.asarray(out,np.int64)
    def draw(self):
        rr=[];bs=[];ff=[]
        for row in self.rows:
            rr.append(np.full(16,row,np.int64));bs.append(self.take(row,"BS",128));ff.append(self.take(row,"F",192))
        order=self.rng.permutation(32)
        return np.concatenate(rr)[order],np.concatenate(bs)[order],np.concatenate(ff)[order]
    def snapshot(self):
        return dict(rows=self.rows.copy(),rng=copy.deepcopy(self.rng.bit_generator.state),buffers=copy.deepcopy(self.buffers),indices=self.indices.copy())
class BudgetExpired(RuntimeError):pass
def alarm(sig,frame):raise BudgetExpired("600s two-position native unit")
def main():
    started=time.monotonic();signal.signal(signal.SIGALRM,alarm);signal.alarm(600)
    A.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
    if (R/"run_manifest.json").exists():raise FileExistsError("Preserve prior two-position unit")
    panel_path=REPORT/"fit_repair_paper_v2_20261101/panel_v2.json";panel=json.loads(panel_path.read_text())
    row=int(panel["primary_row"]);rows=np.array([row,2042],np.int64);b=int(panel["primary_BS_spatial_id"]);fid=int(panel["primary_frequency_id"]);field=0
    sources=unit.source_identity(REPORT/"fit_repair_paper_v2_20261101/contract.json",panel_path)
    for p in [Path(__file__),PARENT,Path(__file__).with_name("newrf_staged_joint_v6.py")]:sources[PATHS.relative(p)]=sha256(p)
    ck=torch.load(PARENT,map_location="cpu",weights_only=False);unit.source_check(ck["source_identity"])
    data=FoldData(0,"screen","newrf")
    if not set(map(int,rows)).issubset(set(map(int,data.fit_ids))) or data.inputs.identity!=panel["input_identity"]:raise ValueError("Frozen FIT identity")
    parent_panel=json.loads((REPORT/"fit_repair_dev_20261101/panel.json").read_text())
    if int(rows[1]) not in set(parent_panel["train_ids"]) or not np.all(data.sites(rows)==0):raise ValueError("Added position must be parent-panel FIT; both sites0")
    guard=device_guard(unit.GPU);torch.set_num_threads(4);seed_all(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    bank=FitGeometryBank(GEOMETRY,data.inputs,outer=0,role="screen_fit");cache=MemoizedFitDirections(bank).to("cuda:0")
    model=build_native_batch_model(data,cache,128,128,frequency_chunk=8).to("cuda:0");model.load_state_dict(ck["model"])
    raw=data.read_fit(rows,normalized=False);truth=raw[:,:128,0];row_to_slot={int(r):i for i,r in enumerate(rows)}
    slots=np.full(4000,-1,np.int64);slots[rows]=np.arange(len(rows));data._native_raw_target_cache=(slots,torch.from_numpy(raw).to("cuda:0"))
    atomic_npy(A/"actual_target_field.npy",truth);atomic_npy(A/"field_row_ids.npy",rows)
    for p in model.parameters():p.requires_grad_(False)
    named=dict(model.fine[field].named_parameters());signal_names=[n for n in named if n.startswith(("network.realimag_filters.","network.branch.","network.output."))]
    density_names=[n for n in named if n.startswith(("network.layers.","network.alpha_out."))]
    if set(signal_names)|set(density_names)!=set(named):raise RuntimeError("Unclassified fine dependency")
    sp=[named[n] for n in signal_names];dp=[named[n] for n in density_names];cp=list(model.coarse[field].parameters())
    fine_opt=torch.optim.Adam([dict(params=sp),dict(params=dp)],lr=.0005,weight_decay=0.);fine_opt.load_state_dict(ck["optimizer"])
    for st in fine_opt.state.values():
        for key,value in st.items():
            if isinstance(value,torch.Tensor):st[key]=value.to("cuda:0")
    # V6 fine and coarse moments are preserved through the expanded FIT curriculum.
    coarse_opt=torch.optim.Adam(cp,lr=.0005,weight_decay=0.);coarse_opt.load_state_dict(ck["coarse_optimizer"])
    for st in coarse_opt.state.values():
        for key,value in st.items():
            if isinstance(value,torch.Tensor):st[key]=value.to("cuda:0")
    for g in coarse_opt.param_groups:g["lr"]=.0005
    torch.set_rng_state(ck["torch_rng"]);torch.cuda.set_rng_state_all(ck["cuda_rng"])
    sampler=TwoPointSampler(rows,ck["numpy_rng"]);group=unit.groups(model,field)
    bs_counts=np.zeros((2,128),np.int64);freq_counts=np.zeros((2,192),np.int64);joint_counts=np.zeros((2,128,192),np.int64)
    labels=[];history=[];traces=[];stages={};total=0;deadline=started+590
    protocol=dict(schema="newrf-native-staged-two-positions-v7a",coarse_max_updates=4000,warm_max_updates=2000,unfrozen_updates=1000,total_max_updates=7000,max_wall_seconds=600,
        parent_sha256=sha256(PARENT),rows=rows.tolist(),field=field,BS_ids=list(range(128)),UE_port=0,frequency_ids=list(range(192)),
        initial_guard_correction="Anchor2666 is v2 parentFIT; only added2042 must belong to original16 panel. No numeric training change.",batch_size=32,positions_per_batch=2,draws_per_position_per_batch=16,sampling="Per-position balanced BS/F permutations; 16 paired labels each position then shuffled; marginal coverage not necessarily full Cartesian",
        coarse_gate=dict(pooled_nmse_max=.05,each_position_nmse_max=.1,each_energy_min=.7,each_energy_max=1.3),warm_gate=dict(pooled_nmse_max=.1,each_position_nmse_max=.2),
        learning_rates=dict(coarse=.0005,warm=.0005,unfrozen=.00005),loss=".1 coarse + .9 fine raw complex batchNMSE",
        optimizer="V6 coarse/fine Adam moments preserved through frozen stages",
        native_forward_unchanged=True,no_density_or_z_buffer=True,diagnostic_only=True,formal_test=False,outer_reads=0,validation_reads=0)
    atomic_json(R/"run_manifest.json",dict(protocol=protocol,source_identity=sources,GPU=guard,model_config=__import__("dataclasses").asdict(model.config)))
    atomic_json(R/"axes.json",dict(row_ids=rows.tolist(),positions_m=data.positions(rows).tolist(),site_ids=data.sites(rows).tolist(),field=field,BS_ids=list(range(128)),UE_port=0,frequency_ids=list(range(192)),raw_H=True))
    def full(m=model):
        return fixed_field(m,cache,data,rows,field,np.arange(128,dtype=np.int64),np.arange(192,dtype=np.int64),deadline)
    @torch.no_grad()
    def density_profile_one(profile_row):
        dev=model.bs_ports.device
        pos=torch.tensor(data.positions([profile_row]),device=dev,dtype=torch.float64);site=torch.tensor(data.sites([profile_row]),device=dev,dtype=torch.long)
        origins=model.spec.ue_world_positions(pos,site)[0];tx=model.bs_ports[site[0]];out=[]
        for first in range(0,128,8):
            bs=np.arange(first,min(first+8,128),dtype=np.int64)
            dirs=[];oo=[];tt=[];counts=[]
            actual=cache.scalar(model,profile_row,field,torch.tensor(bs,device=dev),origins,site[0],tx)
            for j,bb in enumerate(bs):
                d=actual[j];n=len(d);counts.append(n);dirs.append(d);oo.append(origins[0].expand(n,3));tt.append(tx[0,int(bb)].expand(n,3))
            dd=torch.cat(dirs);o=torch.cat(oo);t=torch.cat(tt)
            pp,z=stratified64(o,dd,model.config.far_m,model.config.n_samples)
            _,cr=model.coarse[field].prepare(pp,dd[:,None].expand_as(pp),t[:,None].expand_as(pp));cw=density_weights(cr,z)
            fp,fz=hierarchical64(o,dd,z,cw,model.config.n_importance)
            _,fr=model.fine[field].prepare(fp,dd[:,None].expand_as(fp),t[:,None].expand_as(fp));fw=density_weights(fr,fz)
            cr=cr.cpu().numpy();fr=fr.cpu().numpy();cw=cw.sum(-1).cpu().numpy();fw=fw.sum(-1).cpu().numpy();slot=0
            for bb,n in zip(bs,counts):
                cc=cr[slot:slot+n];rr=fr[slot:slot+n]
                out.append(dict(BS=int(bb),rays=n,coarse_rho_mean=float(cc.mean()),coarse_rho_min=float(cc.min()),coarse_rho_max=float(cc.max()),
                    fine_rho_mean=float(rr.mean()),fine_rho_min=float(rr.min()),fine_rho_max=float(rr.max()),coarse_sumweight_mean=float(cw[slot:slot+n].mean()),fine_sumweight_mean=float(fw[slot:slot+n].mean())))
                slot+=n
        return out
    def density_profile():
        return [dict(row=int(rr),BS_profile=density_profile_one(int(rr))) for rr in rows]
    def matrix_gate(summary,gate):
        if summary["nmse"]>gate["pooled_nmse_max"]:return False
        for item in summary["per_position"]:
            if item["nmse"]>gate["each_position_nmse_max"]:return False
            if "each_energy_min" in gate and not gate["each_energy_min"]<=item["energy_ratio"]<=gate["each_energy_max"]:return False
        return True
    initial_fine=unit.state(model.fine[field]);initial_coarse=unit.state(model.coarse[field])
    pred,coarse_pred=full();atomic_json(R/"initial_metrics.json",dict(fine=metrics(pred,truth),coarse=metrics(coarse_pred,truth)))
    atomic_json(R/"initial_density_profile.json",density_profile())
    def coverage():
        return dict(BS_draw_counts=bs_counts.tolist(),frequency_draw_counts=freq_counts.tolist(),joint_unique_per_position=(joint_counts>0).sum((1,2)).tolist(),
            joint_fraction_per_position=(joint_counts>0).mean((1,2)).tolist(),Cartesian_total_per_position=128*192,paired_draws_per_position=joint_counts.sum((1,2)).tolist())
    def checkpoint(stage,step):
        return dict(model=unit.state(model),coarse_optimizer=copy.deepcopy(coarse_opt.state_dict()),optimizer=copy.deepcopy(fine_opt.state_dict()),
            numpy_rng=copy.deepcopy(sampler.rng.bit_generator.state),sampler=sampler.snapshot(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            stage=stage,stage_updates=step,total_updates=total,coverage=coverage(),source_identity=sources,protocol=protocol,rows=rows.tolist(),field=field,raw_H=True)
    def run(stage,budget,opt,gate_key,gate):
        nonlocal total
        passed=False
        for step in range(1,budget+1):
            ids,bs,ff=sampler.draw();ii=np.array([row_to_slot[int(rr)] for rr in ids],np.int64)
            if not np.array_equal(np.sort(np.unique(ids,return_counts=True)[1]),np.array([16,16])):raise RuntimeError("Position batch balance")
            labels.append((stage,step,bs.copy(),ff.copy(),truth[ii,bs,ff].copy(),ids.copy()))
            np.add.at(bs_counts,(ii,bs),1);np.add.at(freq_counts,(ii,ff),1);np.add.at(joint_counts,(ii,bs,ff),1)
            log=step==1 or step%100==0
            before,zb=unit.decomposition(model,cache,data,row,field,b,fid) if log else (None,None)
            old={key:[p.detach().clone() for p in ps] for key,ps in group.items()} if log else None
            model.train();coarse_opt.zero_grad(set_to_none=True);fine_opt.zero_grad(set_to_none=True)
            target=station_target_cached(data,ids,field,bs,ff,"cuda:0");h,c=station_forward(model,cache,data,ids,field,bs,ff);loss,lf,lc=native_weighted_loss(h,c,target)
            if not torch.isfinite(loss):raise FloatingPointError("Native loss")
            loss.backward();gn=unit.group_norm(group,True) if log else None
            active=[p for p in cp+list(model.fine[field].parameters()) if p.requires_grad];norm=torch.nn.utils.clip_grad_norm_(active,5.)
            if not torch.isfinite(norm):raise FloatingPointError("Gradient")
            opt.step();total+=1
            if log:
                after,za=unit.decomposition(model,cache,data,row,field,b,fid)
                traces.append(dict(stage=stage,update=step,total_updates=total,native_loss=float(loss),fine_loss=float(lf),coarse_loss=float(lc),
                    gradient_before_clip=float(norm),branch_gradient_norms=gn,parameter_update_norms={key:math.sqrt(sum(float((p.detach()-q).double().square().sum()) for p,q in zip(ps,old[key]))) for key,ps in group.items()},
                    before=before,after=after,fine_z_change_max_m=float(np.max(abs(za-zb))),carrier_phase_change_mean_rad=float(np.mean(abs(za-zb))*2*np.pi*data.frequencies([fid])[0]/299792458.)))
                atomic_json(R/"mechanism_traces.json",traces)
            if step%100==0 or step==budget:
                model.eval();pred,coarse_pred=full();fm=metrics(pred,truth);cm=metrics(coarse_pred,truth)
                event=dict(stage=stage,update=step,total_updates=total,fine=fm,coarse=cm,elapsed=time.monotonic()-started);history.append(event)
                atomic_json(R/"progress.json",event)
                print(json.dumps(dict(stage=stage,step=step,fine_nmse=fm["nmse"],fine_energy=fm["energy_ratio"],coarse_nmse=cm["nmse"],coarse_energy=cm["energy_ratio"],elapsed=time.monotonic()-started)),flush=True)
                check=cm if gate_key=="coarse" else fm
                if gate is not None and matrix_gate(check,gate):passed=True;break
        stage_dir=A/stage;stage_dir.mkdir(exist_ok=True)
        pred,coarse_pred=full();atomic_npy(stage_dir/"full_field_prediction.npy",pred);atomic_npy(stage_dir/"coarse_full_field_prediction.npy",coarse_pred)
        atomic_torch_save(stage_dir/"final.pt",checkpoint(stage,step));atomic_json(R/(stage+"_density_profile.json"),density_profile())
        stages[stage]=dict(updates=step,gate_pass=passed if gate is not None else None,fine=metrics(pred,truth),coarse=metrics(coarse_pred,truth))
        atomic_json(R/(stage+"_result.json"),stages[stage])
        return passed
    for p in cp:p.requires_grad_(True)
    coarse_pass=run("coarse",4000,coarse_opt,"coarse",protocol["coarse_gate"])
    if not all(torch.equal(v,initial_fine[k]) for k,v in unit.state(model.fine[field]).items()):raise RuntimeError("Fine changed during coarse-only stage")
    warm_pass=False
    frozen_coarse=unit.state(model.coarse[field])
    if coarse_pass:
        for p in cp:p.requires_grad_(False)
        for p in sp:p.requires_grad_(True)
        for g in fine_opt.param_groups:g["lr"]=.0005
        warm_profile=density_profile()
        warm_pass=run("warm",2000,fine_opt,"fine",protocol["warm_gate"])
        if density_profile()!=warm_profile:raise RuntimeError("Warm density function or native coarse sampling changed")
        if warm_pass:
            for p in dp:p.requires_grad_(True)
            for g in fine_opt.param_groups:g["lr"]=.00005
            run("unfrozen",1000,fine_opt,"fine",None)
        if not all(torch.equal(v,frozen_coarse[k]) for k,v in unit.state(model.coarse[field]).items()):raise RuntimeError("Coarse changed during fine stages")
    stage="unfrozen" if warm_pass else "warm" if coarse_pass else "coarse"
    pred,coarse_pred=full();atomic_npy(A/"full_field_prediction.npy",pred);atomic_npy(A/"coarse_full_field_prediction.npy",coarse_pred)
    atomic_torch_save(A/"final.pt",checkpoint(stage,stages[stage]["updates"]))
    clean=build_native_batch_model(data,cache,128,128,frequency_chunk=8).to("cuda:0");disk=torch.load(A/"final.pt",map_location="cpu",weights_only=False);clean.load_state_dict(disk["model"]);clean.eval()
    clean_pred,clean_coarse=full(clean);atomic_npy(A/"clean_reload_full_field_prediction.npy",clean_pred)
    if not np.array_equal(clean_pred,pred) or not np.array_equal(clean_coarse,coarse_pred):raise RuntimeError("Clean native full-matrix reload mismatch")
    atomic_npy(A/"supervised_BS_ids.npy",np.stack([x[2] for x in labels]));atomic_npy(A/"supervised_frequency_ids.npy",np.stack([x[3] for x in labels]));atomic_npy(A/"supervised_raw_labels.npy",np.stack([x[4] for x in labels]));atomic_npy(A/"joint_draw_counts.npy",joint_counts);atomic_npy(A/"supervised_row_ids.npy",np.stack([x[5] for x in labels]))
    atomic_json(R/"label_stages.json",[dict(stage=x[0],step=x[1]) for x in labels]);unit.source_check(sources)
    final_fine=metrics(pred,truth);final_coarse=metrics(coarse_pred,truth)
    result=dict(status="full_field_pass" if warm_pass and matrix_gate(final_fine,protocol["warm_gate"]) else "stage_failed_gate",stages=stages,
        total_updates=total,elapsed_seconds=time.monotonic()-started,final_fine=final_fine,final_coarse=final_coarse,coverage=coverage(),history=history,
        clean_reload_bitwise_identical=True,no_density_or_z_buffer=True,native_forward_unchanged=True,source_identity=sources,
        prediction_sha256=sha256(A/"full_field_prediction.npy"),checkpoint_sha256=sha256(A/"final.pt"),raw_H=True,diagnostic_only=True,
        two_positions_one_site_one_field=True,not_full_8field_H=True,formal_test=False,outer_reads=0,validation_reads=0)
    atomic_json(R/"result.json",result);signal.alarm(0);print(json.dumps({k:v for k,v in result.items() if k not in ["history","source_identity","coverage"]}),flush=True)
if __name__=="__main__":
    try:main()
    except BaseException:
        R.mkdir(parents=True,exist_ok=True);atomic_json(R/"failure.json",dict(traceback=traceback.format_exc(),formal_test=False));raise
