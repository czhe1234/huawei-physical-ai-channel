"""Single-variable diagnostic: freeze scalar-parent fine coordinates; not deployment baseline."""
from __future__ import annotations
import copy,json,math,signal,time,traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .common import ART,REPORT,FoldData,seed_all,device_guard,sha256,atomic_json,atomic_npy,atomic_torch_save
from .newrf_runner import GEOMETRY,FitGeometryBank
from .newrf_native_batch_v4 import build_native_batch_model,station_forward,native_weighted_loss
from .newrf_runtime_cache_v4 import MemoizedFitDirections,station_target_cached
from .newrf_latefreq import stratified64,density_weights,hierarchical64
from .newrf_balanced_dev_v1 import metrics
from . import newrf_minimal_fullH_v2 as unit
from project_paths import PATHS

A=ART/"fit_repair_paper_v3_fixed_sampling_20261101/newrf"
R=REPORT/"fit_repair_paper_v3_fixed_sampling_20261101/newrf"
BASE_A=ART/"fit_repair_paper_v2_20261101/newrf"
BASE_R=REPORT/"fit_repair_paper_v2_20261101/newrf"
class BudgetExpired(RuntimeError):pass
def alarm(sig,frame):raise BudgetExpired("180s fixed-sampling diagnostic")
def main():
    started=time.monotonic();signal.signal(signal.SIGALRM,alarm);signal.alarm(180)
    A.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
    if (R/"run_manifest.json").exists():raise FileExistsError("Preserve prior fixed-sampling diagnostic")
    parent=BASE_A/"scalar/final.pt"
    panel_path=REPORT/"fit_repair_paper_v2_20261101/panel_v2.json"
    panel=json.loads(panel_path.read_text());row=int(panel["primary_row"]);b=int(panel["primary_BS_spatial_id"]);fid=int(panel["primary_frequency_id"]);field=0
    sources=unit.source_identity(REPORT/"fit_repair_paper_v2_20261101/contract.json",panel_path)
    sources[PATHS.relative(Path(__file__))]=sha256(Path(__file__))
    sources[PATHS.relative(parent)]=sha256(parent)
    ck=torch.load(parent,map_location="cpu",weights_only=False)
    unit.source_check(ck["source_identity"])
    original_source=Path(__file__).with_name("newrf_latefreq.py").read_text()
    if "new=(bg[...,0]+t*(bg[...,1]-bg[...,0])).detach()" not in original_source:raise RuntimeError("Original fine-CDF detach changed")
    data=FoldData(0,"screen","newrf")
    if row not in set(data.fit_ids) or data.inputs.identity!=panel["input_identity"]:raise ValueError("Frozen FIT identity")
    guard=device_guard(unit.GPU);torch.set_num_threads(4);seed_all(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    bank=FitGeometryBank(GEOMETRY,data.inputs,outer=0,role="screen_fit")
    cache=MemoizedFitDirections(bank).to("cuda:0")
    model=build_native_batch_model(data,cache,128,128,frequency_chunk=8).to("cuda:0");model.load_state_dict(ck["model"])
    raw=data.read_fit(np.array([row]),normalized=False);truth=raw[:,:128,0]
    slots=np.full(4000,-1,np.int64);slots[row]=0;data._native_raw_target_cache=(slots,torch.from_numpy(raw).to("cuda:0"))
    params=list(model.coarse[field].parameters())+list(model.fine[field].parameters())
    optimizer=torch.optim.Adam(params,lr=.0005,weight_decay=0.);optimizer.load_state_dict(ck["optimizer"])
    for state in optimizer.state.values():
        for key,value in state.items():
            if isinstance(value,torch.Tensor):state[key]=value.to("cuda:0")
    torch.set_rng_state(ck["torch_rng"]);torch.cuda.set_rng_state_all(ck["cuda_rng"])
    sampler=unit.Axes("frequency_only",b,fid,ck["numpy_rng"])
    group=unit.groups(model,field)
    with torch.no_grad():
        pos=torch.tensor(data.positions([row]),device="cuda:0",dtype=torch.float64)
        site=torch.tensor(data.sites([row]),device="cuda:0",dtype=torch.long)
        origins=model.spec.ue_world_positions(pos,site);tx=model.bs_ports[site]
        dirs=cache.scalar(model,row,field,torch.tensor([b],device="cuda:0"),origins[0],site[0],tx[0])[0]
        n=len(dirs);origin=origins[0,0].expand(n,3);transmit=tx[0,0,b].expand(n,3)
        points,z=stratified64(origin,dirs,model.config.far_m,model.config.n_samples)
        _,rho=model.coarse[field].prepare(points,dirs[:,None].expand_as(points),transmit[:,None].expand_as(points))
        weights=density_weights(rho,z)
        fixed_points,fixed_z=hierarchical64(origin,dirs,z,weights,model.config.n_importance)
        if fixed_z.requires_grad or fixed_points.requires_grad:raise RuntimeError("Expected detached original coordinates")
        fixed_points=fixed_points.detach().clone();fixed_z=fixed_z.detach().clone()
    atomic_npy(A/"fixed_fine_z.npy",fixed_z.cpu().numpy());atomic_npy(A/"fixed_fine_points.npy",fixed_points.cpu().numpy())
    atomic_npy(A/"actual_target_field.npy",truth);atomic_npy(A/"field_row_ids.npy",np.array([row],np.int64))
    def forward(bs,ff):
        bs=np.asarray(bs,np.int64);ff=np.asarray(ff,np.int64)
        if not np.all(bs==b):raise ValueError("Fixed sampler only authorized BS30")
        size=len(bs);dd=dirs.repeat(size,1);tt=transmit.repeat(size,1)
        fr=torch.tensor(data.frequencies(ff),device="cuda:0",dtype=torch.float64).repeat_interleave(n).reshape(-1,1)
        pp,zz=stratified64(origin.repeat(size,1),dd,model.config.far_m,model.config.n_samples)
        def prepare(net,x):
            base,rr=net.prepare(x,dd[:,None].expand_as(x),tt[:,None].expand_as(x))
            ef=net.encoded_frequency(fr);cut=net.cfg.hidden_dim+net.view_encoder.d_output
            bias=F.linear(ef,net.network.branch.weight[:,cut:],None)
            sig=net.network.output(net.network.act(base+bias[:,None,:]))
            return torch.cat((sig,rr),-1)
        coarse=prepare(model.coarse[field],pp);hc,_,_,_=model.synthesizer.synthesize(coarse,zz,fr/1e9,[n]*size)
        fine=prepare(model.fine[field],fixed_points.repeat(size,1,1))
        h,_,_,_=model.synthesizer.synthesize(fine,fixed_z.repeat(size,1),fr/1e9,[n]*size)
        return h.to(torch.complex64),hc.to(torch.complex64)
    def decompose():
        # Only the isolated process helper's sampler binding changes, never source files.
        native=unit.hierarchical64
        def frozen(o,d,z,w,k):
            if o.shape!=origin.shape or not torch.equal(o,origin) or not torch.equal(d,dirs):raise ValueError("Diagnostic geometry mismatch")
            return fixed_points,fixed_z
        unit.hierarchical64=frozen
        try:return unit.decomposition(model,cache,data,row,field,b,fid)
        finally:unit.hierarchical64=native
    def axis():
        fine=[];coarse=[]
        with torch.no_grad():
            for first in range(0,192,32):
                fs=np.arange(first,min(first+32,192),dtype=np.int64)
                h,c=forward(np.full(len(fs),b,np.int64),fs);fine.append(h.cpu().numpy());coarse.append(c.cpu().numpy())
        return np.concatenate(fine)[None,None],np.concatenate(coarse)[None,None]
    # Same weights, actual row and gradients at the freeze point. No optimizer updates.
    check_bs=np.full(32,b,np.int64);check_ff=np.arange(32,dtype=np.int64);check_ids=np.full(32,row,np.int64)
    target=station_target_cached(data,check_ids,field,check_bs,check_ff,"cuda:0")
    optimizer.zero_grad(set_to_none=True);h,c=station_forward(model,cache,data,check_ids,field,check_bs,check_ff);loss,_,_=native_weighted_loss(h,c,target);loss.backward()
    gh=torch.cat([p.grad.detach().reshape(-1).clone() if p.grad is not None else torch.zeros_like(p).reshape(-1) for p in params]);hn=h.detach().clone();cn=c.detach().clone()
    optimizer.zero_grad(set_to_none=True);hf,cf=forward(check_bs,check_ff);lf,_,_=native_weighted_loss(hf,cf,target);lf.backward()
    gf=torch.cat([p.grad.detach().reshape(-1) if p.grad is not None else torch.zeros_like(p).reshape(-1) for p in params])
    equivalence=dict(fine_relative_L2=float(torch.linalg.vector_norm(hf-hn)/torch.linalg.vector_norm(hn)),coarse_relative_L2=float(torch.linalg.vector_norm(cf-cn)/torch.linalg.vector_norm(cn)),
        all_gradient_relative_L2=float(torch.linalg.vector_norm(gf-gh)/torch.linalg.vector_norm(gh)),native_loss=float(loss),fixed_loss=float(lf),original_fine_z_requires_grad=False)
    atomic_json(R/"initial_equivalence.json",equivalence)
    if max(equivalence["fine_relative_L2"],equivalence["coarse_relative_L2"],equivalence["all_gradient_relative_L2"])>1e-5:raise RuntimeError("Freeze point not equal to native")
    optimizer.zero_grad(set_to_none=True)
    protocol=dict(schema="newrf-fixed-fine-sampling-diagnostic-v3",updates=700,max_updates=1200,max_wall_seconds=180,
        scalar_parent_sha256=sha256(parent),single_variable="Fine points/z held at scalar parent's coarse PDF; coarse optimizer remains active",
        original_fine_z_detached=True,loss=".1 coarse + .9 fine raw complex batchNMSE",lr=.0005,clip=5.,row=row,field=field,BS=b,frequency_ids=list(range(192)),
        not_deployable_baseline=True,formal_test=False,outer_reads=0,validation_reads=0)
    atomic_json(R/"run_manifest.json",dict(protocol=protocol,source_identity=sources,GPU=guard,model_config=__import__("dataclasses").asdict(model.config)))
    atomic_json(R/"axis.json",dict(row_ids=[row],field=field,BS_ids=[b],UE_port=0,frequency_ids=list(range(192)),raw_H=True))
    initial,_=decompose();atomic_json(R/"initial_decomposition.json",initial)
    pred,cpred=axis();target_axis=truth[:,[b],:]
    initial_metrics=dict(fine=metrics(pred,target_axis),coarse=metrics(cpred,target_axis))
    labels=[];traces=[];history=[]
    for step in range(1,701):
        bs,ff=sampler.draw();ids=np.full(32,row,np.int64);labels.append((bs.copy(),ff.copy(),truth[0,bs,ff].copy()))
        log=step==1 or step%25==0
        before,zbefore=decompose() if log else (None,None)
        weights_before={key:[p.detach().clone() for p in ps] for key,ps in group.items()} if log else None
        optimizer.zero_grad(set_to_none=True);model.train()
        target=station_target_cached(data,ids,field,bs,ff,"cuda:0");h,c=forward(bs,ff);loss,lf,lc=native_weighted_loss(h,c,target)
        if not torch.isfinite(loss):raise FloatingPointError("Nonfinite loss")
        loss.backward();gn=unit.group_norm(group,True) if log else None
        clipnorm=torch.nn.utils.clip_grad_norm_(params,5.)
        if not torch.isfinite(clipnorm):raise FloatingPointError("Nonfinite gradient")
        optimizer.step()
        if log:
            after,zafter=decompose()
            trace=dict(update=step,native_loss=float(loss),fine_loss=float(lf),coarse_loss=float(lc),gradient_before_clip=float(clipnorm),branch_gradient_norms=gn,
                parameter_update_norms={key:math.sqrt(sum(float((p.detach()-q).double().square().sum()) for p,q in zip(ps,weights_before[key]))) for key,ps in group.items()},
                before=before,after=after,fine_z_change_max_m=float(np.max(abs(zafter-zbefore))),fine_z_change_mean_m=float(np.mean(abs(zafter-zbefore))),
                carrier_phase_change_mean_rad=float(np.mean(abs(zafter-zbefore))*2*np.pi*data.frequencies([fid])[0]/299792458.))
            traces.append(trace);atomic_json(R/"mechanism_traces.json",traces)
        if step%25==0 or step==700:
            pred,cpred=axis();fm=metrics(pred,target_axis);cm=metrics(cpred,target_axis)
            history.append(dict(update=step,fine=fm,coarse=cm));atomic_json(R/"progress.json",dict(update=step,fine=fm,coarse=cm,elapsed=time.monotonic()-started))
            print(json.dumps(dict(update=step,fine_nmse=fm["nmse"],fine_energy=fm["energy_ratio"],coarse_nmse=cm["nmse"],elapsed=time.monotonic()-started)),flush=True)
    pred,cpred=axis();final,zend=decompose()
    atomic_npy(A/"wideband_prediction.npy",pred);atomic_npy(A/"wideband_coarse_prediction.npy",cpred)
    atomic_npy(A/"supervised_BS_ids.npy",np.stack([x[0] for x in labels]));atomic_npy(A/"supervised_frequency_ids.npy",np.stack([x[1] for x in labels]));atomic_npy(A/"supervised_raw_labels.npy",np.stack([x[2] for x in labels]))
    prevbs=np.load(BASE_A/"supervised_BS_ids.npy")[70:770];prevf=np.load(BASE_A/"supervised_frequency_ids.npy")[70:770];prevt=np.load(BASE_A/"supervised_raw_labels.npy")[70:770]
    draws_match=bool(np.array_equal(prevbs,np.stack([x[0] for x in labels])) and np.array_equal(prevf,np.stack([x[1] for x in labels])) and np.array_equal(prevt,np.stack([x[2] for x in labels])))
    if not draws_match:raise RuntimeError("V2 F-only draw sequence mismatch")
    atomic_torch_save(A/"final.pt",dict(model=unit.state(model),optimizer=copy.deepcopy(optimizer.state_dict()),numpy_rng=copy.deepcopy(sampler.rng.bit_generator.state),
        sampler=sampler.snapshot(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),updates=700,scalar_parent_updates=ck["branch_updates"],source_identity=sources,protocol=protocol))
    unit.source_check(sources)
    result=dict(status="completed_diagnostic",updates=700,elapsed_seconds=time.monotonic()-started,initial_metrics=initial_metrics,
        fine_metrics=metrics(pred,target_axis),coarse_metrics=metrics(cpred,target_axis),history=history,final_decomposition=final,
        fixed_z_max_change_m=float(np.max(abs(zend-fixed_z.cpu().numpy()))),fixed_z_max_change_rad=0.,same_v2_F_only_draws_and_labels=draws_match,
        scalar_parent_sha256=sha256(parent),prediction_sha256=sha256(A/"wideband_prediction.npy"),checkpoint_sha256=sha256(A/"final.pt"),source_identity=sources,
        raw_H=True,diagnostic_only=True,not_deployable_baseline=True,formal_test=False,outer_reads=0,validation_reads=0)
    atomic_json(R/"result.json",result);signal.alarm(0);print(json.dumps({k:v for k,v in result.items() if k not in ["history","final_decomposition","source_identity"]}),flush=True)
if __name__=="__main__":
    try:main()
    except BaseException:
        R.mkdir(parents=True,exist_ok=True);atomic_json(R/"failure.json",dict(traceback=traceback.format_exc(),formal_test=False));raise
