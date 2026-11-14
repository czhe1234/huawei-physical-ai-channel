"""NeWRF repair: matched-shape route audit and query-available staged FIT training."""
from __future__ import annotations
import argparse,copy,hashlib,json,math,os,signal,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
from project_paths import PATHS
from round2.paper_baseline_training_v1.common import (
    FoldData,seed_all,device_guard,sha256,atomic_json,atomic_npy,atomic_torch_save)
from round2.paper_baseline_training_v1.newrf_runner import GEOMETRY,FitGeometryBank
from round2.paper_baseline_training_v1.newrf_runtime_cache_v4 import MemoizedFitDirections,station_target_cached
from round2.paper_baseline_training_v1.newrf_native_batch_v4 import build_native_batch_model,station_forward,native_weighted_loss
from round2.paper_baseline_training_v1.newrf_balanced_dev_v1 import fixed_field,metrics
from round2.paper_baseline_training_v1.newrf_query import subset_query
from round2.paper_baseline_training_v1 import newrf_minimal_fullH_v2 as unit

ROOT=PATHS.analysis/"results/round2_paper_four_baselines_v1/repair_20261111"
R=ROOT/"newrf"
A=PATHS.artifacts/"round2/paper_four_baselines_v1/repair_20261111/newrf"
PARENT=PATHS.artifacts/"round2/paper_four_baselines_v1/fit_repair_paper_v10_coarse_init_signal_warm_20261101/newrf"
CONFIG=ROOT/"config.json"
BS=np.arange(128,dtype=np.int64);FREQ=np.arange(192,dtype=np.int64)
ALLOWED={"GPU-6b53c3c2-3071-5dfb-f306-2f2f142f8009",
         "GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0",
         "GPU-70b6fceb-c7c4-9467-3c81-097c9284c160"}

def identity():
    files=[Path(__file__),CONFIG,Path(GEOMETRY),
           PATHS.code_root/"common/official_metrics.py"]
    old=PATHS.code_root/"round2/paper_baseline_training_v1"
    files += [p for p in old.glob("*.py") if p.name.startswith("newrf") or p.name=="common.py"]
    files += [PARENT/"final.pt",PARENT/"actual_target_field.npy",PARENT/"full_field_prediction.npy"]
    return {PATHS.relative(p):sha256(p) for p in files}
def check_identity(source):
    for rel,h in source.items():
        if sha256(PATHS.root/rel)!=h:raise RuntimeError("Frozen source changed: "+rel)
def state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
def compare(a,b):
    x=np.asarray(a,np.complex128);y=np.asarray(b,np.complex128)
    en=float(np.sum(abs(x)**2));err=float(np.sum(abs(x-y)**2))
    return dict(relative_L2=math.sqrt(err/en) if en>0 else math.sqrt(err),
                max_abs=float(np.max(abs(x-y))),reference_energy=en,delta_energy=err)
def fits(m,coarse=False):
    return bool(m["nmse"] <= (.05 if coarse else .1)
        and all(z["nmse"] <= (.1 if coarse else .2) and .7<=z["energy_ratio"]<=1.3 for z in m["per_position"]))
def gpu_setup(uuid):
    if uuid not in ALLOWED or os.environ.get("CUDA_VISIBLE_DEVICES")!=uuid:
        raise RuntimeError("Set exactly the requested allowed CUDA UUID before Python starts")
    free=subprocess.check_output(["nvidia-smi","--id="+uuid,"--query-gpu=memory.free",
        "--format=csv,noheader,nounits"],text=True).strip()
    if int(free)<32768:raise RuntimeError("Need >=32GiB free before this method starts")
    report=device_guard(uuid)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    return report
def inputs():
    cfg=json.loads(CONFIG.read_text());rows=np.asarray(cfg["newrf"]["route_rows"],np.int64)
    if len(rows)!=2 or cfg["newrf"]["field"]!=0:raise ValueError("Registered two-position field0")
    data=FoldData(0,"screen","newrf")
    if not set(rows).issubset(set(data.fit_ids)) or not np.all(data.sites(rows)==0):
        raise PermissionError("Two authorized site0 FIT positions only")
    raw=data.read_fit(rows,normalized=False);truth=raw[:,:128,0]
    frozen=np.load(PARENT/"actual_target_field.npy",allow_pickle=False)
    if not np.array_equal(truth,frozen):raise RuntimeError("V10 raw target differs")
    return cfg,data,rows,raw,truth
def build(data,device="cuda:0"):
    bank=FitGeometryBank(GEOMETRY,data.inputs,outer=0,role="screen_fit")
    cache=MemoizedFitDirections(bank).to(device)
    model=build_native_batch_model(data,cache,128,128,frequency_chunk=8).to(device)
    return bank,cache,model
def full_query(model,data,rows,deadline):
    """Use the original 8-BS block shape, actual query input provider and renderer."""
    vals=[]
    with torch.no_grad():
        for row in rows:
            pieces=[]
            for first in range(0,128,8):
                if time.monotonic()>deadline:raise TimeoutError("Query deadline")
                x=subset_query(model,data,[int(row)],BS[first:first+8],np.array([0]),FREQ)
                pieces.append(x[0,:,0,:].cpu().numpy())
            vals.append(np.concatenate(pieces))
    return np.stack(vals)
class QueryDirectionsCache:
    """FIT row IDs index geometry-only directions obtainable at any query coordinate."""
    def __init__(self,model,fitcache,data,rows):
        self.bank=fitcache.bank;self.far_m=fitcache.far_m;self.allowed=set(map(int,rows))
        self.values={}
        with torch.no_grad():
            for row in rows:
                pos=torch.tensor(data.positions([row]),device=model.bs_ports.device,dtype=torch.float64)
                site=torch.tensor(data.sites([row]),device=model.bs_ports.device,dtype=torch.long)
                origins=model.spec.ue_world_positions(pos,site);tx=model.bs_ports[site]
                before=list(self.bank.opened_ids)
                value=model.ray_provider.directions(origins,site,tx)[0].detach()
                if self.bank.opened_ids!=before:raise RuntimeError("Query direction call read row-specific geometry")
                if value.shape!=(2,128,4,64,3) or not torch.isfinite(value).all():
                    raise RuntimeError("Expected 64 query-available directions for all ports")
                self.values[int(row)]=value
    def scalar(self,model,row,field,spatial,origin_all,site,tx_all):
        if int(row) not in self.allowed:raise PermissionError("Outside repair FIT panel")
        p,u=divmod(int(field),4)
        return self.values[int(row)][p,spatial,u]
class Sampler:
    def __init__(self,rows,seed):
        self.rows=np.asarray(rows,np.int64);self.rng=np.random.default_rng(seed);self.buffers={};self.indices={}
    def take(self,row,axis,size):
        key=(int(row),axis);out=[]
        for _ in range(16):
            if key not in self.buffers or self.indices[key]==size:
                self.buffers[key]=self.rng.permutation(size);self.indices[key]=0
            out.append(int(self.buffers[key][self.indices[key]]));self.indices[key]+=1
        return np.array(out,np.int64)
    def draw(self):
        rr=[];bb=[];ff=[]
        for row in self.rows:
            rr.append(np.full(16,row,np.int64));bb.append(self.take(row,"BS",128));ff.append(self.take(row,"F",192))
        ix=self.rng.permutation(32)
        return np.concatenate(rr)[ix],np.concatenate(bb)[ix],np.concatenate(ff)[ix]
    def snapshot(self):
        return dict(rng=copy.deepcopy(self.rng.bit_generator.state),buffers=copy.deepcopy(self.buffers),indices=self.indices.copy())
    def restore(self,d):
        self.rng.bit_generator.state=d["rng"];self.buffers=d["buffers"];self.indices=d["indices"]

def preflight():
    cfg,data,rows,raw,truth=inputs()
    ck=torch.load(PARENT/"final.pt",map_location="cpu",weights_only=False)
    unit.source_check(ck["source_identity"])
    if ck["stage"]!="unfrozen" or ck["fine_density_backward_detached"]:
        raise RuntimeError("Need clean released V10 checkpoint")
    d=dict(status="passed",source_identity=identity(),config=cfg["newrf"],row_ids=rows.tolist(),
        target_shape=list(truth.shape),parent_checkpoint_sha256=sha256(PARENT/"final.pt"),
        CUDA_initialized=torch.cuda.is_initialized(),query_truth_read=False,optimizer_updates=0)
    if d["CUDA_initialized"]:raise RuntimeError("Preflight initialized CUDA")
    atomic_json(R/"preflight.json",d);print(json.dumps({k:v for k,v in d.items() if k!="source_identity"}),flush=True)

def route(args):
    out=R/"route";art=A/"route"
    if (out/"result.json").exists():
        d=json.loads((out/"result.json").read_text());check_identity(d["source_identity"]);print("Existing frozen route result reused",flush=True);return
    source=identity();check_identity(json.loads((R/"preflight.json").read_text())["source_identity"])
    cfg,data,rows,raw,truth=inputs();device=gpu_setup(args.gpu_uuid)
    bank,cache,model=build(data)
    ck=torch.load(PARENT/"final.pt",map_location="cpu",weights_only=False)
    model.load_state_dict(ck["model"]);model.eval()
    if "physical_density" in model.fine[0].__dict__:raise RuntimeError("Unexpected warm patch")
    start=time.monotonic();deadline=start+600
    with torch.no_grad():fit,coarse=fixed_field(model,cache,data,rows,0,BS,FREQ,deadline)
    reference=np.load(PARENT/"full_field_prediction.npy",allow_pickle=False)
    agreement=compare(reference,fit)
    agreement["per_position"]=[compare(a,b) for a,b in zip(reference,fit)]
    tol=cfg["newrf"]["route_replay_relative_L2_max"]
    okay=agreement["relative_L2"]<=tol and all(x["relative_L2"]<=tol for x in agreement["per_position"])
    atomic_json(out/"saved_replay.json",dict(**agreement,tolerance=tol,passed=okay,
        batch_shape="original per-row blocks of8BS x192tones",device=device))
    atomic_npy(art/"FIT_prediction.npy",fit)
    if not okay:raise RuntimeError("Same-GPU same-shape V10 replay exceeds frozen1e-5 tolerance")
    q=full_query(model,data,rows,deadline);atomic_npy(art/"query_prediction.npy",q)
    qcache=QueryDirectionsCache(model,cache,data,rows)
    with torch.no_grad():unified,uc=fixed_field(model,qcache,data,rows,0,BS,FREQ,deadline)
    atomic_npy(art/"unified_provider_prediction.npy",unified)
    eq=compare(q,unified)
    if eq["relative_L2"]>tol:raise RuntimeError("Query-direction training provider differs from actual query inference")
    qm=metrics(q,truth);fm=metrics(fit,truth)
    result=dict(status="route_complete",source_identity=source,device=device,config=cfg["newrf"],
        row_ids=rows.tolist(),field=0,axes=dict(BS=BS.tolist(),UE=[0],frequency_ids=FREQ.tolist()),
        saved_replay=agreement,same_shape_saved_replay_passed=True,
        FIT_route_metrics=fm,query_route_metrics=qm,FIT_vs_query=compare(fit,q),
        unified_training_provider_vs_actual_query=eq,needs_repair=not fits(qm),
        formal_test=False,optimizer_updates=0,query_truth_read=False,
        target_info=data.info(),elapsed_seconds=time.monotonic()-start,
        parent_checkpoint_sha256=sha256(PARENT/"final.pt"),
        prediction_sha256={str(p.name):sha256(p) for p in art.glob("*.npy")})
    check_identity(source);atomic_json(out/"result.json",result)
    print(json.dumps({k:result[k] for k in ["status","FIT_route_metrics","query_route_metrics","needs_repair","FIT_vs_query"]}),flush=True)

def train(args,smoke=False):
    tag="smoke" if smoke else "fit_query";out=R/tag;art=A/tag
    if (out/"result.json").exists():
        d=json.loads((out/"result.json").read_text());check_identity(d["source_identity"])
        print("Completed "+tag+" reused",flush=True);return
    cfg,data,rows,raw,truth=inputs();source=identity()
    route_result=json.loads((R/"route/result.json").read_text());check_identity(route_result["source_identity"])
    if not route_result["needs_repair"]:
        atomic_json(out/"result.json",dict(status="not_needed",source_identity=source));return
    if not smoke:
        qa=json.loads((R/"smoke/result.json").read_text())
        if qa["status"]!="smoke_passed" or qa["source_identity"]!=source:raise RuntimeError("Smoke is required")
    device=gpu_setup(args.gpu_uuid);seed_all(cfg["seed"]);started=time.monotonic()
    bank,fitcache,model=build(data);qcache=QueryDirectionsCache(model,fitcache,data,rows)
    slots=np.full(4000,-1,np.int64);slots[rows]=np.arange(2)
    data._native_raw_target_cache=(slots,torch.from_numpy(raw).to("cuda:0"))
    atomic_npy(art/"actual_target_field.npy",truth);atomic_npy(art/"row_ids.npy",rows)
    sampler=Sampler(rows,cfg["seed"])
    coarse=list(model.coarse[0].parameters())
    named=dict(model.fine[0].named_parameters())
    alpha=[p for n,p in named.items() if n.startswith("network.alpha_out.")]
    active=[p for n,p in named.items() if not n.startswith("network.alpha_out.")]
    if len(alpha)!=2:raise RuntimeError("Fine alpha head identity")
    cop=torch.optim.Adam(coarse,lr=5e-4)
    fop=torch.optim.Adam([dict(params=active),dict(params=alpha)],lr=5e-4)
    max_steps=cfg["newrf"]["max_updates"];stage_cap=max_steps//3
    max_wall=cfg["newrf"]["max_wall_seconds"]
    draws=np.zeros((max_steps,32,3),np.int64);joint=np.zeros((2,128,192),np.int64)
    history=[];stages={};current_stage=None;current_step=0;stable=0;total=0;elapsed_before=0.
    current_scheduler=None
    latest=art/"latest.pt"
    if latest.exists():
        ck=torch.load(latest,map_location="cpu",weights_only=False)
        if ck["source_identity"]!=source:raise RuntimeError("Resume source/config changed")
        model.load_state_dict(ck["model"]);cop.load_state_dict(ck["coarse_optimizer"]);fop.load_state_dict(ck["fine_optimizer"])
        for opt in [cop,fop]:
            for st in opt.state.values():
                for k,v in st.items():
                    if isinstance(v,torch.Tensor):st[k]=v.to("cuda:0")
        sampler.restore(ck["sampler"]);torch.set_rng_state(ck["torch_rng"]);torch.cuda.set_rng_state_all(ck["cuda_rng"])
        current_stage=ck["current_stage"];current_step=ck["current_step"];stable=ck["stable"]
        history=ck["history"];stages=ck["stages"];total=ck["total_updates"];elapsed_before=ck["elapsed_seconds"]
        draws[:total]=ck["draws"];joint=ck["joint"]
        saved_scheduler=ck["scheduler"]
    else:saved_scheduler=None
    def elapsed():return elapsed_before+time.monotonic()-started
    def deadline():return time.monotonic()+max(1,max_wall-elapsed()-20)
    def full(m=model,cache=qcache):
        m.eval()
        return fixed_field(m,cache,data,rows,0,BS,FREQ,deadline())
    def configure(stage,enter):
        for p in model.parameters():p.requires_grad_(False)
        if "physical_density" in model.fine[0].__dict__:delattr(model.fine[0],"physical_density")
        if stage=="coarse":
            for p in coarse:p.requires_grad_(True)
            return cop
        if enter and stage=="warm":
            model.fine[0].load_state_dict(model.coarse[0].state_dict());fop.state.clear()
        for p in active:p.requires_grad_(True)
        if stage=="warm":
            original=model.fine[0].physical_density
            model.fine[0].physical_density=lambda raw_alpha:original(raw_alpha).detach()
            lr=5e-4
        else:
            for p in alpha:p.requires_grad_(True)
            lr=5e-5
        if enter:
            for g in fop.param_groups:g["lr"]=lr
        return fop
    def checkpoint():
        return dict(model=state(model),coarse_optimizer=copy.deepcopy(cop.state_dict()),fine_optimizer=copy.deepcopy(fop.state_dict()),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),sampler=sampler.snapshot(),
            current_stage=current_stage,current_step=current_step,stable=stable,total_updates=total,stages=stages,
            history=history,elapsed_seconds=elapsed(),draws=draws[:total].copy(),joint=joint.copy(),
            scheduler=current_scheduler.state_dict() if current_scheduler else None,
            source_identity=source,protocol=protocol)
    protocol=dict(schema="newrf-query-available-staged-FIT-v1",row_ids=rows.tolist(),field=0,
        raw_H=True,frequency_count=192,BS_count=128,query_rays=64,n_samples=128,n_importance=128,
        stage_caps=dict(coarse=stage_cap,warm=stage_cap,native=stage_cap),max_wall_seconds=max_wall,
        eval_every=500,checkpoint_every=250,seed=cfg["seed"],batch_size=32,
        initialization="fresh; each minibatch has16 paired labels per position",
        loss="source(.1 sum|coarse-y|^2+.9 sum|fine-y|^2)/sum|y|^2 throughout",
        query_geometry_only=True,query_CSI=False,coordinate_or_synthesis_change=False,
        coarse_stage="fine weights frozen",warm="fine copies coarse; alpha weights frozen; physical-density backward detached; trunk signal gradients active",
        native="warm override removed; all fine parameters trainable; coarse stays frozen",
        learning_rates=dict(coarse=5e-4,warm=5e-4,native=5e-5),
        scheduler="each phase ReduceLROnPlateau factor.5 patience2 threshold1e-4 on full FIT phaseNMSE",
        coarse_gate="pooled<=.05 each<=.1 energy.7..1.3",
        fine_gate="pooled<=.1 each<=.2 energy.7..1.3; three consecutive full checks",
        other_fields_untrained=True,formal_test=False,outer_truth_reads=0,validation_truth_reads=0)
    atomic_json(out/"run_manifest.json",dict(source_identity=source,protocol=protocol,device=device,data=data.info()))
    if smoke:
        opt=configure("coarse",True);model.train()
        rr,bb,ff=sampler.draw();y=station_target_cached(data,rr,0,bb,ff,"cuda:0")
        h,c=station_forward(model,qcache,data,rr,0,bb,ff)
        loss,lf,lc=native_weighted_loss(h,c,y);loss.backward()
        gn=torch.nn.utils.clip_grad_norm_(coarse,5.,error_if_nonfinite=True);opt.step()
        if not torch.isfinite(loss):raise FloatingPointError("Smoke loss")
        opt.zero_grad(set_to_none=True)
        # The query-only scalar batching must agree with the original per-position renderer.
        from round2.paper_baseline_training_v1.newrf_native_batch_v4 import station_forward_reference
        model.eval()
        with torch.no_grad():
            actual,_=station_forward(model,qcache,data,rr[:4],0,bb[:4],ff[:4])
            reference,_=station_forward_reference(model,qcache,data,rr[:4],0,bb[:4],ff[:4])
        eq=compare(reference.cpu().numpy(),actual.cpu().numpy())
        if eq["relative_L2"]>1e-5:raise RuntimeError("Scalar query-provider batch equivalence")
        check_identity(source)
        atomic_json(out/"result.json",dict(status="smoke_passed",source_identity=source,
            finite_loss=float(loss),gradient_norm=float(gn),scalar_batch_equivalence=eq,
            updates=1,weights_discarded=True,device=device,elapsed_seconds=elapsed(),
            peak_cuda_bytes=torch.cuda.max_memory_allocated(),formal_test=False))
        print("SMOKE_PASSED",flush=True);return
    interrupted=False
    for stage in ("coarse","warm","native"):
        if stage in stages:continue
        resuming=current_stage==stage
        opt=configure(stage,not resuming)
        current_stage=stage
        current_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",factor=.5,
            patience=2,threshold=1e-4,threshold_mode="abs",min_lr=1e-5)
        if resuming and saved_scheduler is not None:current_scheduler.load_state_dict(saved_scheduler)
        if not resuming:current_step=0;stable=0
        best=min([e["coarse"]["nmse"] if stage=="coarse" else e["fine"]["nmse"] for e in history if e["stage"]==stage],default=math.inf)
        last_print=time.monotonic();stage_pass=False
        while current_step<stage_cap:
            if elapsed()>max_wall-90:interrupted=True;break
            rr,bb,ff=sampler.draw();slots_rr=np.array([int(np.flatnonzero(rows==x)[0]) for x in rr])
            draws[total,:,0]=rr;draws[total,:,1]=bb;draws[total,:,2]=ff
            np.add.at(joint,(slots_rr,bb,ff),1)
            model.train();cop.zero_grad(set_to_none=True);fop.zero_grad(set_to_none=True)
            y=station_target_cached(data,rr,0,bb,ff,"cuda:0")
            h,c=station_forward(model,qcache,data,rr,0,bb,ff)
            loss,lf,lc=native_weighted_loss(h,c,y)
            if not torch.isfinite(loss):raise FloatingPointError("Nonfinite source batchNMSE")
            loss.backward()
            active_params=[p for p in model.parameters() if p.requires_grad]
            gn=torch.nn.utils.clip_grad_norm_(active_params,5.,error_if_nonfinite=True)
            opt.step();current_step+=1;total+=1
            if time.monotonic()-last_print>30:
                print(json.dumps(dict(stage=stage,step=current_step,total_updates=total,loss=float(loss),elapsed_seconds=elapsed())),flush=True);last_print=time.monotonic()
            if current_step%250==0:atomic_torch_save(latest,checkpoint())
            if current_step%500 and current_step!=stage_cap:continue
            pred,cpred=full();fm=metrics(pred,truth);cm=metrics(cpred,truth)
            selected=cm if stage=="coarse" else fm
            now=fits(selected,coarse=stage=="coarse");stable=stable+1 if now else 0
            current_scheduler.step(selected["nmse"])
            event=dict(stage=stage,step=current_step,total_updates=total,fine=fm,coarse=cm,
                gate_now=now,stable_checks=stable,learning_rates=[g["lr"] for g in opt.param_groups],elapsed_seconds=elapsed())
            history.append(event);atomic_json(out/"progress.json",event)
            if selected["nmse"]<best:
                best=selected["nmse"];atomic_torch_save(art/(stage+"_best.pt"),checkpoint())
            atomic_torch_save(latest,checkpoint())
            print(json.dumps(dict(stage=stage,step=current_step,fine_nmse=fm["nmse"],coarse_nmse=cm["nmse"],
                per_position_nmse=[v["nmse"] for v in selected["per_position"]],stable=stable,elapsed_seconds=elapsed())),flush=True)
            if (stage=="coarse" and now) or (stage!="coarse" and stable>=3):
                stage_pass=True;break
        if interrupted:
            atomic_torch_save(latest,checkpoint());break
        pred,cpred=full()
        stages[stage]=dict(updates=current_step,passed=stage_pass,fine=metrics(pred,truth),coarse=metrics(cpred,truth))
        atomic_json(out/(stage+"_result.json"),stages[stage]);atomic_torch_save(art/(stage+"_final.pt"),checkpoint())
        atomic_torch_save(latest,checkpoint())
        if not stage_pass:break
        current_step=0;stable=0;saved_scheduler=None
    if "physical_density" in model.fine[0].__dict__:delattr(model.fine[0],"physical_density")
    pred,cpred=full()
    actual_query=full_query(model,data,rows,deadline())
    agreement=compare(pred,actual_query)
    if agreement["relative_L2"]>1e-5:raise RuntimeError("Final trained-provider and actual query mismatch")
    model.eval();atomic_npy(art/"query_prediction.npy",actual_query)
    atomic_npy(art/"training_provider_prediction.npy",pred);atomic_npy(art/"coarse_prediction.npy",cpred)
    atomic_npy(art/"supervised_row_BS_frequency_ids.npy",draws[:total]);atomic_npy(art/"joint_draw_counts.npy",joint)
    atomic_torch_save(art/"final.pt",checkpoint())
    clean=build_native_batch_model(data,fitcache,128,128,frequency_chunk=8).to("cuda:0")
    clean.load_state_dict(torch.load(art/"final.pt",map_location="cpu",weights_only=False)["model"]);clean.eval()
    if "physical_density" in clean.fine[0].__dict__:raise RuntimeError("Clean builder has warm patch")
    reloaded=full_query(clean,data,rows,deadline());replay=compare(actual_query,reloaded)
    if replay["relative_L2"]>1e-5:raise RuntimeError("Clean builder query reload failed")
    check_identity(source)
    qm=metrics(actual_query,truth)
    status="two_position_actual_query_fit_pass" if "native" in stages and stages["native"]["passed"] and fits(qm) else "budget_exhausted" if interrupted else "fit_not_restored"
    result=dict(status=status,source_identity=source,protocol=protocol,device=device,stages=stages,
        actual_query_metrics=qm,training_provider_metrics=metrics(pred,truth),
        training_vs_actual_query=agreement,clean_reload=replay,clean_builder_no_warm_patch=True,
        total_optimizer_updates=total,elapsed_seconds=elapsed(),peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        joint_unique_fraction_per_position=(joint>0).mean((1,2)).tolist(),
        final_checkpoint_sha256=sha256(art/"final.pt"),prediction_sha256=sha256(art/"query_prediction.npy"),
        data=data.info(),other_fields_untrained=True,formal_test=False,
        query_truth_read=False,validation_truth_reads=0)
    atomic_json(out/"result.json",result)
    print(json.dumps({k:result[k] for k in ["status","actual_query_metrics","total_optimizer_updates","elapsed_seconds"]}),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--stage",choices=["preflight","route","smoke","train"],required=True)
    p.add_argument("--gpu-uuid")
    args=p.parse_args();R.mkdir(parents=True,exist_ok=True);A.mkdir(parents=True,exist_ok=True)
    try:
        if args.stage=="preflight":preflight()
        elif args.stage=="route":route(args)
        else:train(args,smoke=args.stage=="smoke")
    except BaseException:
        atomic_json(R/(args.stage+"_failure.json"),dict(traceback=traceback.format_exc(),
            formal_test=False,query_truth_read=False))
        raise
if __name__=="__main__":main()
