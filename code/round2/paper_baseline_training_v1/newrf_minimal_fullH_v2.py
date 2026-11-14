"""Four-cell single-position NeWRF diagnostic. Native model/loss/scales unchanged."""
from __future__ import annotations
import argparse,copy,json,math,signal,time,traceback
from pathlib import Path
import numpy as np
import torch
from .common import ART,REPORT,FoldData,seed_all,device_guard,sha256,atomic_json,atomic_npy,atomic_torch_save
from .newrf_runner import GEOMETRY,FitGeometryBank,subset_forward
from .newrf_native_batch_v4 import build_native_batch_model,station_forward,native_weighted_loss
from .newrf_runtime_cache_v4 import MemoizedFitDirections,station_target_cached
from .newrf_latefreq import stratified64,density_weights,hierarchical64
from .newrf_balanced_dev_v1 import fixed_field,metrics
from .newrf_v4_health import check as reference_fast_check
from project_paths import PATHS

GPU="GPU-6b53c3c2-3071-5dfb-f306-2f2f142f8009"
A=ART/"fit_repair_paper_v2_20261101/newrf"
R=REPORT/"fit_repair_paper_v2_20261101/newrf"
class BudgetExpired(RuntimeError):pass
def alarm(signum,frame):raise BudgetExpired("600s single-field unit")
def stats(x):
    a=x.detach().double().reshape(-1)
    return dict(min=float(a.min()),median=float(a.median()),mean=float(a.mean()),max=float(a.max()),
                finite=bool(torch.isfinite(a).all()))
def source_identity(contract,panel):
    names=["newrf_runner.py","newrf_native_batch_v4.py","newrf_runtime_cache_v4.py","newrf_conditioned_v3.py",
      "newrf_latefreq.py","newrf_query.py","newrf_v4_health.py","newrf_balanced_dev_v1.py","common.py"]
    return {PATHS.relative(p):sha256(p) for p in [Path(__file__),contract,panel,*[Path(__file__).with_name(n) for n in names]]}
def source_check(source):
    for rel,digest in source.items():
        if sha256(PATHS.root/rel)!=digest:raise RuntimeError("Frozen source changed: "+rel)
def state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

class Axes:
    def __init__(self,mode,b,f,rngstate=None):
        self.mode=mode;self.b=b;self.f=f;self.rng=np.random.default_rng(42)
        if rngstate is not None:self.rng.bit_generator.state=copy.deepcopy(rngstate)
        self.buffers={};self.index={}
    def take(self,key,size):
        out=[]
        for _ in range(32):
            if key not in self.buffers or self.index[key]==size:
                self.buffers[key]=self.rng.permutation(size);self.index[key]=0
            out.append(int(self.buffers[key][self.index[key]]));self.index[key]+=1
        return np.asarray(out,np.int64)
    def draw(self):
        bs=self.take("bs",128) if self.mode in ("BS_only","joint") else np.full(32,self.b,np.int64)
        ff=self.take("freq",192) if self.mode in ("frequency_only","joint") else np.full(32,self.f,np.int64)
        return bs,ff
    def snapshot(self):
        return dict(mode=self.mode,rng=copy.deepcopy(self.rng.bit_generator.state),buffers=copy.deepcopy(self.buffers),index=self.index.copy())

def groups(model,field):
    result={}
    for side in ("coarse","fine"):
        for name,p in getattr(model,side)[field].named_parameters():
            part=("density" if "alpha_out" in name else "signal_output" if ".output." in name else
                  "signal_branch" if ".branch." in name else "signal_features" if "realimag_filters" in name else "position_trunk")
            result.setdefault(side+"_"+part,[]).append(p)
    return result
def group_norm(items,gradient=False):
    out={}
    for key,ps in items.items():
        xs=[p.grad for p in ps if p.grad is not None] if gradient else ps
        out[key]=math.sqrt(sum(float(x.detach().double().square().sum()) for x in xs)) if xs else 0.
    return out

@torch.no_grad()
def decomposition(model,cache,data,row,field,bsid,fid):
    dev=model.bs_ports.device;p,u=divmod(field,4)
    pos=torch.tensor(data.positions([row]),device=dev,dtype=torch.float64)
    sites=torch.tensor(data.sites([row]),device=dev,dtype=torch.long)
    origin=model.spec.ue_world_positions(pos,sites)[0];tx=model.bs_ports[sites[0]]
    dirs=cache.scalar(model,row,field,torch.tensor([bsid],device=dev),origin,sites[0],tx)[0]
    n=len(dirs);o=origin[u].expand(n,3);t=tx[p,bsid].expand(n,3)
    freq=torch.tensor(data.frequencies([fid]),device=dev,dtype=torch.float64).reshape(1,1).expand(n,1)
    points,z=stratified64(o,dirs,model.config.far_m,model.config.n_samples)
    def prepare(net,points):
        captured={}
        hook=net.network.alpha_out.register_forward_hook(lambda m,i,v:captured.update(logit=v.detach()))
        base,rho=net.prepare(points,dirs[:,None].expand_as(points),t[:,None].expand_as(points));hook.remove()
        ef=net.encoded_frequency(freq);cut=net.cfg.hidden_dim+net.view_encoder.d_output
        bias=torch.nn.functional.linear(ef,net.network.branch.weight[:,cut:],None)
        sig=net.network.output(net.network.act(base+bias[:,None,:]))
        return torch.cat((sig,rho),-1),rho,captured["logit"]
    raw,rho,logit=prepare(model.coarse[field],points)
    weights=density_weights(rho,z)
    fp,fz=hierarchical64(o,dirs,z,weights,model.config.n_importance)
    fraw,frho,flogit=prepare(model.fine[field],fp)
    def render(raw,zz,rho,logit):
        dist=torch.cat((zz[:,1:]-zz[:,:-1],1e-10*torch.ones_like(zz[:,:1])),-1)
        alpha=1-torch.exp(-torch.relu(raw[...,-1])*dist)
        def exclusive(v):return torch.cat((torch.zeros_like(v[:,:1]),torch.cumprod(v,1)[:,:-1]),1)
        transmission=exclusive(1-alpha+1e-10)
        phase=torch.exp(-1j*2*torch.pi*freq/model.synthesizer.speedOfLight*dist)
        coeff=alpha*exclusive((1-alpha)*phase+1e-10)
        signal=torch.complex(torch.tanh(raw[...,0]),torch.tanh(raw[...,1]))
        amp=model.synthesizer.speedOfLight/(zz*freq*4*torch.pi)
        terms=signal*amp*coeff
        h=terms.sum();sumabs=terms.abs().sum();eincoh=terms.abs().square().sum()
        actual,_,_,_=model.synthesizer.synthesize(raw,zz,freq/1e9,[n])
        return dict(rho=stats(rho),rho_logit=stats(logit),alpha=stats(alpha),
            first_alpha=stats(alpha[:,0]),integral_weight_sum=stats((alpha*transmission).sum(1)),
            optical_depth=stats((rho[...,0]*dist).sum(1)),final_transmission=stats(torch.prod(1-alpha+1e-10,1)),
            tanh_signal_magnitude=stats(signal.abs()),sum_absolute_contributions=float(sumabs),
            incoherent_contribution_energy=float(eincoh),coherent_energy=float(h.abs().square()),
            coherent_to_absolute_efficiency=float(h.abs().square()/sumabs.square().clamp_min(1e-300)),
            h_real=float(h.real),h_imag=float(h.imag),reconstruction_abs_difference=float((h-actual[0]).abs()),
            fine_z=stats(zz))
    d=dict(row=row,field=field,BS=bsid,frequency_id=fid,rays=n,far_m=model.config.far_m,
           coarse=render(raw,z,rho,logit),fine=render(fraw,fz,frho,flogit))
    return d,fz.detach().cpu().numpy()

def axis_prediction(model,cache,data,row,field,mode,b,f,deadline):
    bs=np.arange(128,dtype=np.int64) if mode in ("BS_only","joint") else np.array([b],np.int64)
    ff=np.arange(192,dtype=np.int64) if mode in ("frequency_only","joint") else np.array([f],np.int64)
    v,c=fixed_field(model,cache,data,[row],field,bs,ff,deadline)
    return v,c,bs,ff
def passed(m,g):
    return m["nmse"]<=g["nmse_max"] and g["energy_min"]<=m["energy_ratio"]<=g["energy_max"]

def main(args):
    started=time.monotonic();signal.signal(signal.SIGALRM,alarm);signal.alarm(600)
    contract=json.loads(args.contract.read_text());panel=json.loads(args.panel.read_text())
    if contract["schema"]!="paper-baseline-repair-unit-contract-v2" or panel["schema"]!="single_location_full_field_fit_discriminator-v2":raise ValueError("Frozen unit schema")
    if contract["max_updates"]!=3000 or contract["max_wall_seconds_per_method"]!=600:raise ValueError("Unit budget")
    A.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
    if (R/"run_manifest.json").exists():raise FileExistsError("Preserve prior unit")
    source=source_identity(args.contract,args.panel)
    data=FoldData(0,"screen","newrf");row=int(panel["primary_row"]);site=int(panel["primary_site"])
    field=0;b=int(panel["primary_BS_spatial_id"]);f=int(panel["primary_frequency_id"]);p,u=divmod(field,4)
    if row not in set(data.fit_ids) or data.sites([row])[0]!=site or site!=0 or panel["primary_UE_port"]!=u:raise ValueError("Single frozen field/site")
    if data.inputs.identity!=panel["input_identity"]:raise RuntimeError("Input identity")
    guard=device_guard(GPU);torch.set_num_threads(4);seed_all(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    bank=FitGeometryBank(GEOMETRY,data.inputs,outer=0,role="screen_fit")
    cache=MemoizedFitDirections(bank).to("cuda:0")
    model=build_native_batch_model(data,cache,128,128,frequency_chunk=8).to("cuda:0")
    alltruth=data.read_fit(np.array([row]),normalized=False);truth=alltruth[:,p*128:(p+1)*128,u]
    slots=np.full(4000,-1,np.int64);slots[row]=0
    data._native_raw_target_cache=(slots,torch.from_numpy(alltruth).to("cuda:0"))
    y0=truth[0,b,f]
    if abs(y0)==0:raise ValueError("Nonzero scalar required")
    atomic_npy(A/"field_row_ids.npy",np.array([row],np.int64));atomic_npy(A/"actual_target_field.npy",truth)
    atomic_json(R/"field_axes.json",dict(field=field,BS_spatial_ids=list(range(128)),UE_port=u,frequency_ids=list(range(192)),raw_H=True))
    atomic_json(R/"run_manifest.json",dict(status="running",source_identity=source,contract=contract,panel=panel,GPU=guard,
         model_config=__import__("dataclasses").asdict(model.config),field=field,site=site,row=row,
         scalar_target=dict(BS=b,F=f,real=float(y0.real),imag=float(y0.imag)),raw_H=True,formal_test=False,
         initialization="same fresh seed42/native model; no preprocessing/target/physics change"))
    eq=reference_fast_check(model,cache,data,np.full(32,row,np.int64),np.full(32,b,np.int64),np.full(32,f,np.int64))
    atomic_json(R/"actual_row_reference_fast_check.json",eq)
    params=list(model.coarse[field].parameters())+list(model.fine[field].parameters())
    opt=torch.optim.Adam(params,lr=.0005,weight_decay=0.)
    group=groups(model,field);labels=[];traces=[];stages={};global_steps=0
    deadline=started+590
    def checkpoint(mode,sampler,steps):
        return dict(model=state(model),optimizer=copy.deepcopy(opt.state_dict()),numpy_rng=copy.deepcopy(sampler.rng.bit_generator.state),
                    sampler=sampler.snapshot(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                    branch=mode,branch_updates=steps,total_updates=global_steps,source_identity=source,
                    row=row,field=field,raw_H=True,parent_FIT_RMS=data.scale)
    def restore(path,mode):
        ck=torch.load(path,map_location="cpu",weights_only=False)
        if ck["source_identity"]!=source:raise RuntimeError("Fork source")
        model.load_state_dict(ck["model"]);opt.load_state_dict(ck["optimizer"])
        for st in opt.state.values():
            for key,value in st.items():
                if isinstance(value,torch.Tensor):st[key]=value.to("cuda:0")
        torch.set_rng_state(ck["torch_rng"]);torch.cuda.set_rng_state_all(ck["cuda_rng"])
        return Axes(mode,b,f,ck["numpy_rng"]),sha256(path)
    def run(mode,budget,gate,parent=None):
        nonlocal global_steps
        sampler,parent_hash=restore(parent,mode) if parent else (Axes(mode,b,f),None)
        stage_dir=A/mode;stage_dir.mkdir(parents=True,exist_ok=True)
        pred0,coarse0,bs0,ff0=axis_prediction(model,cache,data,row,field,mode,b,f,deadline)
        target0=truth[:,bs0][:,:,ff0]
        initial=metrics(pred0,target0);history=[];passed_gate=False;steps=0
        initial_decomp,_=decomposition(model,cache,data,row,field,b,f)
        atomic_json(R/(mode+"_initial_decomposition.json"),initial_decomp)
        for steps in range(1,budget+1):
            if time.monotonic()>deadline:raise BudgetExpired("Unit before600s")
            bs,ff=sampler.draw();ids=np.full(32,row,np.int64)
            label=truth[0,bs,ff]
            labels.append(dict(branch=mode,update=steps,BS=bs.copy(),F=ff.copy(),raw=label.copy()))
            log=(steps==1 or steps%25==0)
            before_decomp,zbefore=decomposition(model,cache,data,row,field,b,f) if log else (None,None)
            before={name:[v.detach().clone() for v in values] for name,values in group.items()} if log else None
            model.train();opt.zero_grad(set_to_none=True)
            target=station_target_cached(data,ids,field,bs,ff,"cuda:0")
            h,hc=station_forward(model,cache,data,ids,field,bs,ff)
            loss,lf,lc=native_weighted_loss(h,hc,target)
            if not torch.isfinite(loss):raise FloatingPointError("Native complex loss")
            loss.backward()
            grad=group_norm(group,True) if log else None
            norm=torch.nn.utils.clip_grad_norm_(params,5.)
            if not torch.isfinite(norm):raise FloatingPointError("Gradient")
            opt.step();global_steps+=1
            if log:
                delta={name:math.sqrt(sum(float((v.detach()-old).double().square().sum()) for v,old in zip(values,before[name]))) for name,values in group.items()}
                after_decomp,zafter=decomposition(model,cache,data,row,field,b,f)
                event=dict(branch=mode,update=steps,total_updates=global_steps,native_loss=float(loss),fine_loss=float(lf),
                  coarse_loss=float(lc),gradient_before_clip=float(norm),branch_gradient_norms=grad,parameter_update_norms=delta,
                  before=before_decomp,after=after_decomp,fine_z_change_max_m=float(np.max(np.abs(zafter-zbefore))),
                  fine_z_change_mean_m=float(np.mean(np.abs(zafter-zbefore))),
                  carrier_phase_change_mean_rad=float(np.mean(np.abs(zafter-zbefore))*2*np.pi*data.frequencies([f])[0]/299792458.),
                  elapsed_seconds=time.monotonic()-started)
                traces.append(event);atomic_json(R/"mechanism_traces.json",traces)
            every=10 if mode=="scalar" else 25 if mode!="joint" else 50
            if steps%every==0 or steps==budget:
                model.eval();v,c,bb,fs=axis_prediction(model,cache,data,row,field,mode,b,f,deadline)
                target_axis=truth[:,bb][:,:,fs];m=metrics(v,target_axis);cm=metrics(c,target_axis)
                history.append(dict(update=steps,metrics=m,coarse=cm))
                atomic_json(R/"progress.json",dict(branch=mode,updates=steps,total_updates=global_steps,metrics=m,
                            elapsed_seconds=time.monotonic()-started,parent_checkpoint_sha256=parent_hash))
                atomic_torch_save(stage_dir/"last.pt",checkpoint(mode,sampler,steps))
                if mode=="scalar":atomic_json(R/"scalar_status.json",dict(updates=steps,metrics=m,passed=passed(m,gate)))
                print(json.dumps(dict(branch=mode,update=steps,nmse=m["nmse"],energy=m["energy_ratio"],elapsed=time.monotonic()-started)),flush=True)
                if passed(m,gate):passed_gate=True;break
        source_check(source);model.eval()
        v,c,bb,fs=axis_prediction(model,cache,data,row,field,mode,b,f,deadline)
        m=metrics(v,truth[:,bb][:,:,fs]);cm=metrics(c,truth[:,bb][:,:,fs])
        full,fullc=fixed_field(model,cache,data,[row],field,np.arange(128,dtype=np.int64),np.arange(192,dtype=np.int64),deadline)
        atomic_npy(stage_dir/"full_field_prediction.npy",full);atomic_npy(stage_dir/"coarse_full_field_prediction.npy",fullc)
        atomic_npy(stage_dir/"axis_prediction.npy",v)
        atomic_torch_save(stage_dir/"final.pt",checkpoint(mode,sampler,steps))
        final_decomp,_=decomposition(model,cache,data,row,field,b,f)
        d=dict(status="passed" if passed_gate else "failed_gate",branch=mode,updates=steps,
          parent_checkpoint_sha256=parent_hash,initial_axis_metrics=initial,axis_metrics=m,coarse_axis_metrics=cm,
          BS_ids=bb.tolist(),frequency_ids=fs.tolist(),full_field_metrics=metrics(full,truth),history=history,
          final_decomposition=final_decomp,checkpoint_sha256=sha256(stage_dir/"final.pt"),
          full_prediction_sha256=sha256(stage_dir/"full_field_prediction.npy"),source_identity=source)
        stages[mode]=d;atomic_json(R/(mode+"_result.json"),d)
        return passed_gate,stage_dir/"final.pt"
    scalar,scalar_ck=run("scalar",contract["stage_budgets"]["newrf"][0],contract["scalar_gate"])
    if scalar:
        freq,freq_ck=run("frequency_only",contract["stage_budgets"]["newrf"][1],contract["wideband_gate"],scalar_ck)
        bs,bs_ck=run("BS_only",contract["stage_budgets"]["newrf"][2],contract["wideband_gate"],scalar_ck)
        if freq and bs:run("joint",contract["stage_budgets"]["newrf"][3],contract["field_gate"],freq_ck)
    source_check(source)
    atomic_npy(A/"supervised_BS_ids.npy",np.stack([x["BS"] for x in labels]))
    atomic_npy(A/"supervised_frequency_ids.npy",np.stack([x["F"] for x in labels]))
    atomic_npy(A/"supervised_raw_labels.npy",np.stack([x["raw"] for x in labels]))
    atomic_json(R/"supervised_label_rows.json",[dict(branch=x["branch"],branch_update=x["update"],row=row) for x in labels])
    result=dict(status="joint_pass" if stages.get("joint",{}).get("status")=="passed" else "failure_localized",
       stages={k:dict(status=v["status"],updates=v["updates"],axis_metrics=v["axis_metrics"],full_field_metrics=v["full_field_metrics"]) for k,v in stages.items()},
       total_updates=global_steps,elapsed_seconds=time.monotonic()-started,source_identity=source,
       field=field,row=row,raw_H=True,formal_test=False,outer_reads=0,validation_reads=0,
       note="One field/location only; neither full8field H nor generalization result")
    atomic_json(R/"result.json",result);signal.alarm(0);print(json.dumps(result),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--contract",type=Path,required=True);p.add_argument("--panel",type=Path,required=True)
    args=p.parse_args()
    try:main(args)
    except BaseException:
        R.mkdir(parents=True,exist_ok=True);atomic_json(R/"failure.json",dict(traceback=traceback.format_exc(),formal_test=False))
        raise

