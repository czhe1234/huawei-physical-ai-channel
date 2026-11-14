"""Formal NeWRF: role-isolated all-field query-route training and raw-H inference.

The field is BS-polarization*4 + physical UE-port. Each gradient block uses
exactly one position, eight BS ports and all192 tones, as does query inference.
Multiple positions accumulate gradients before one Adam update. No outer CSI
reader is used. All writes belong to the new formal artifact/report roots.
"""
from __future__ import annotations
import argparse,copy,fcntl,hashlib,json,math,os,signal,subprocess,time,traceback
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from project_paths import PATHS
from round2.paper_baseline_training_v1 import common as C
from round2.paper_baseline_training_v1.newrf_runner import GEOMETRY,FitGeometryBank
from round2.paper_baseline_training_v1.newrf_native_batch_v4 import build_native_batch_model
from round2.paper_baseline_training_v1.newrf_runtime_cache_v4 import MemoizedFitDirections
from round2.paper_baseline_training_v1.newrf_query import subset_query
from round2.paper_baseline_training_v1 import selection as SELECT

ROOT=C.REPORT/"formal_20261111";ART=C.ART/"formal_20261111"
R=ROOT/"newrf";A=ART/"newrf"
DEFAULT_GPU=C.ALLOWED[0]
STOP=False
BS=np.arange(128,dtype=np.int64);FF=np.arange(192,dtype=np.int64)
STAGES=("coarse","warm","native")
def stop_signal(*_):
    global STOP
    STOP=True
def emit(value,path=None):
    line=json.dumps(value,allow_nan=False);print(line,flush=True)
    if path is not None:
        with Path(path).open("a")as f:f.write(line+"\n")
def sha(p):return C.sha256(p)
def read_json(p):return json.loads(Path(p).read_text())
def cpu_state(model):return {k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
def content_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def load_config(path):
    root=read_json(path);cfg=root.get("newrf",root)
    required=["schema","seed","batch_positions","bs_block","n_samples","n_importance","frequency_chunk",
      "direction_cache_entries","checkpoint_every","health_every","validation_every","stage_min_updates",
      "stage_max_updates","learning_rates","native_patience","minimum_validation_improvement",
      "training_power_floor_FIT_RMS_squared","max_role_wall_seconds","smoke_updates","smoke_batch_options"]
    for k in required:
        if k not in cfg:raise KeyError("Missing frozen NeWRF config key: "+k)
    if cfg["bs_block"]!=8 or cfg["n_samples"]!=128 or cfg["n_importance"]!=128 or cfg["frequency_chunk"]!=8:
        raise ValueError("Keep verified8BS/128coarse/128importance/8frequency-chunk renderer")
    if cfg["batch_positions"]not in(2,4)or cfg["smoke_batch_options"]!=[2,4]or cfg["smoke_updates"]!=20:
        raise ValueError("Declared2/4-position gradient accumulation and20-update smoke")
    if cfg["training_power_floor_FIT_RMS_squared"]!=1e-6:raise ValueError("Registered normalized training denominator floor")
    if any(cfg["stage_max_updates"][s]<cfg["stage_min_updates"][s]for s in STAGES):raise ValueError("Stage caps below minimum")
    return cfg
def source_identity(path):
    old=PATHS.code_root/"round2/paper_baseline_training_v1"
    files=[Path(__file__),Path(path),Path(C.__file__),Path(SELECT.__file__),Path(GEOMETRY),
      PATHS.code_root/"round2/paper_baseline_adapters_v1/contract.py",
      PATHS.code_root/"round2/paper_baseline_adapters_v1/newrf.py",
      PATHS.code_root/"round2/simulation_recovery_v1/qualified_newrf.py",
      PATHS.code_root/"round2/simulation_recovery_v1/ray_provider.py"]
    names=["newrf_runner.py","newrf_native_batch_v4.py","newrf_runtime_cache_v4.py",
      "newrf_query.py","newrf_conditioned_v3.py","newrf_latefreq.py"]
    files.extend(old/n for n in names)
    files.extend(PATHS.root/"plus-pro/code/third_party/paper_baselines/newrf"/n for n in["models.py","synthesizers.py"])
    return {PATHS.relative(p):sha(p)for p in files if p.is_file()}
def check_sources(src):
    for p,h in src.items():
        if sha(PATHS.root/p)!=h:raise RuntimeError(("Frozen source drift",p))
def read_only_scale(data):
    # Reuse immutable role-FIT RMS without touching old cache lock files.
    p=C.ART/"fit_scales"/f"outer{data.outer}_{data.role}.json";d=read_json(p)
    expected=dict(schema="FIT-channel-scalar-RMS-v1",input_identity=data.inputs.identity,
      fit_ids_sha256=C.digest_ids(data.fit_ids),raw=data.raw_identity,outer=data.outer,role=data.role)
    if d["contract"]!=expected:raise RuntimeError("Role FIT scale provenance mismatch")
    data._scale_info=d
    return data.scale
def prepare_data(outer,role):
    data=C.FoldData(outer,role,"newrf");read_only_scale(data)
    return data
def fit_panel(data):
    return C._probe_ids(data.inputs,data.fit_ids,16)
def build(data,cfg,device):
    bank=FitGeometryBank(GEOMETRY,data.inputs,outer=data.outer,role="screen_fit"if data.role=="screen"else"refit")
    if len(bank.source_fit_ids)!=len(data.fit_ids):raise RuntimeError("Incomplete allowed FIT geometry")
    base=MemoizedFitDirections(bank).to(device)
    model=build_native_batch_model(data,base,cfg["n_samples"],cfg["n_importance"],
      frequency_chunk=cfg["frequency_chunk"]).to(device)
    return bank,base,model
def gpu_setup(uuid):
    if uuid not in C.ALLOWED or os.environ.get("CUDA_VISIBLE_DEVICES")!=uuid:raise RuntimeError("Use full allowed GPU UUID before imports")
    free=int(subprocess.check_output(["nvidia-smi","-i",uuid,"--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
    if free<32768:raise RuntimeError("Need32GiB free before new formal process")
    d=C.device_guard(uuid);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    return d

class QueryDirectionLRU:
    """Lazy GPU cache keyed by(field,FITrow), max196608bytes per entry."""
    def __init__(self,model,data,maximum):
        self.model=model;self.data=data;self.allowed=set(map(int,data.fit_ids))
        self.maximum=int(maximum);self.values=OrderedDict();self.hits=self.misses=0;self.build_seconds=0.
    def get(self,row,field,origin,site,tx):
        row=int(row);key=(int(field),row)
        if row not in self.allowed:raise PermissionError("Training cache outside role FIT")
        if key in self.values:
            self.hits+=1;self.values.move_to_end(key);return self.values[key]
        started=time.monotonic();self.misses+=1;p,u=divmod(int(field),4)
        opened=len(self.model.ray_provider.bank.opened_ids)
        with torch.no_grad():value=self.model.ray_provider.directions(origin,site,tx)[0,p,:,u].detach().contiguous()
        if len(self.model.ray_provider.bank.opened_ids)!=opened:raise RuntimeError("Query directions read pointwise geometry payload")
        if value.shape!=(128,64,3)or not torch.isfinite(value).all():raise RuntimeError("Query direction axis mismatch")
        self.build_seconds+=time.monotonic()-started
        if self.maximum:
            self.values[key]=value
            while len(self.values)>self.maximum:self.values.popitem(last=False)
        return value
    def info(self):
        return dict(entries=len(self.values),limit=self.maximum,hits=self.hits,misses=self.misses,
          tensor_bytes=sum(x.numel()*x.element_size()for x in self.values.values()),build_seconds=self.build_seconds,
          query_geometry_payload_reads=0,CSI_cached=False)

class PermutationSampler:
    """Per-site position cycles; per-position BS-block cycles. All192F each draw."""
    def __init__(self,data,batch,seed):
        self.batch=int(batch);self.rng=np.random.default_rng(seed)
        self.pools={s:data.fit_ids[data.sites(data.fit_ids)==s]for s in(0,1)}
        if any(not len(v)for v in self.pools.values()):raise ValueError("Both sites required")
        self.order={};self.cursor={};self.block_order={};self.block_cursor={}
        self.seen=np.zeros(4000,np.int64);self.block_seen=np.zeros((4000,16),np.int32)
    @property
    def coverage_steps(self):return max(math.ceil(len(v)/(self.batch//2))for v in self.pools.values())
    def draw(self):
        rows=[];blocks=[]
        for site in(0,1):
            for _ in range(self.batch//2):
                if site not in self.order or self.cursor[site]==len(self.order[site]):
                    self.order[site]=self.rng.permutation(self.pools[site]);self.cursor[site]=0
                row=int(self.order[site][self.cursor[site]]);self.cursor[site]+=1
                if row not in self.block_order or self.block_cursor[row]==16:
                    self.block_order[row]=self.rng.permutation(16);self.block_cursor[row]=0
                block=int(self.block_order[row][self.block_cursor[row]]);self.block_cursor[row]+=1
                rows.append(row);blocks.append(block)
        return np.asarray(rows,np.int64),np.asarray(blocks,np.int64)
    def commit(self,rows,blocks):
        np.add.at(self.seen,rows,1);np.add.at(self.block_seen,(rows,blocks),1)
    def snapshot(self):
        return copy.deepcopy(dict(batch=self.batch,rng=self.rng.bit_generator.state,order=self.order,cursor=self.cursor,
          block_order=self.block_order,block_cursor=self.block_cursor,seen=self.seen,block_seen=self.block_seen))
    def restore(self,state):
        if state["batch"]!=self.batch:raise RuntimeError("Resumed batch changed")
        for k in["order","cursor","block_order","block_cursor","seen","block_seen"]:setattr(self,k,copy.deepcopy(state[k]))
        self.rng.bit_generator.state=copy.deepcopy(state["rng"])

def block_labels(data,rows,blocks,field):
    rows=np.asarray(rows,np.int64)
    if not set(map(int,rows)).issubset(set(map(int,data.fit_ids))):raise PermissionError("Block labels outside FIT")
    p,u=divmod(field,4);out=[]
    for row,b in zip(rows,blocks):
        ss=np.arange(int(b)*8,int(b)*8+8)
        out.append(np.array(data._h[int(row),p*128+ss,u,:],copy=True))
    np.add.at(data._fit_counts,rows,1)
    return np.stack(out).astype(np.complex64,copy=False)
def cached_block(model,data,row,field,bs,ff,cache):
    dev=model.bs_ports.device;p,u=divmod(field,4)
    pos=torch.tensor(data.positions([row]),device=dev,dtype=torch.float64)
    site=torch.tensor(data.sites([row]),device=dev,dtype=torch.long)
    origin=model.spec.ue_world_positions(pos,site);tx=model.bs_ports[site]
    directions=cache.get(row,field,origin,site,tx)
    ix=torch.tensor(bs,device=dev,dtype=torch.long)
    dirs=directions.index_select(0,ix)[None]
    freq=torch.tensor(data.frequencies(ff),device=dev,dtype=torch.float64)
    return model._scalar_links(field,origin[:,u],dirs,tx[:,p].index_select(1,ix),freq)
def actual_block(model,data,row,field,bs,ff):
    p,u=divmod(field,4)
    return subset_query(model,data,np.asarray([row]),p*128+np.asarray(bs),np.asarray([u]),np.asarray(ff))[0,:,0,:]
def set_stage(model,field,stage,opt,enter):
    for p in model.parameters():p.requires_grad_(False)
    fine=model.fine[field]
    if "physical_density"in fine.__dict__:delattr(fine,"physical_density")
    if stage=="coarse":
        for p in model.coarse[field].parameters():p.requires_grad_(True)
        return
    if enter and stage=="warm":
        fine.load_state_dict(copy.deepcopy(model.coarse[field].state_dict()));opt.state.clear()
    for n,p in fine.named_parameters():
        p.requires_grad_(stage=="native"or not n.startswith("network.alpha_out."))
    if stage=="warm":
        original=fine.physical_density
        fine.physical_density=lambda x:original(x).detach()
def update(model,data,cache,sampler,field,opt,cfg):
    rows,blocks=sampler.draw();labels=block_labels(data,rows,blocks,field)
    energy=float(np.sum(abs(labels.astype(np.complex128))**2))
    floor=float(labels.size*cfg["training_power_floor_FIT_RMS_squared"]*data.scale**2)
    denominator=max(energy,floor);opt.zero_grad(set_to_none=True);model.train()
    loss_total=0.;fine_total=0.;coarse_total=0.
    for row,block,label in zip(rows,blocks,labels):
        bs=np.arange(int(block)*8,int(block)*8+8,dtype=np.int64)
        h,hc=cached_block(model,data,int(row),field,bs,FF,cache)
        y=torch.tensor(label,device=h.device,dtype=torch.complex64)[None]
        # Native pooled complex batchNMSE; accumulated blocks share one denominator.
        lf=(h-y).abs().square().sum(dtype=torch.float64)/denominator
        lc=(hc-y).abs().square().sum(dtype=torch.float64)/denominator
        loss=.9*lf+.1*lc
        if not torch.isfinite(loss):raise FloatingPointError("Nonfinite native loss")
        loss.backward();loss_total+=float(loss.detach());fine_total+=float(lf.detach());coarse_total+=float(lc.detach())
    active=[p for p in model.parameters()if p.requires_grad]
    grad=float(torch.nn.utils.clip_grad_norm_(active,5.,error_if_nonfinite=True));opt.step();sampler.commit(rows,blocks)
    return dict(loss=loss_total,fine_loss=fine_total,coarse_loss=coarse_total,gradient_norm=grad,
      denominator_clamped=energy<floor,true_batch_energy=energy,denominator=denominator,
      positions=len(rows),scalar_complex_labels=int(labels.size),rows=rows.tolist(),blocks=blocks.tolist())
def predict_field(model,data,rows,field,bs=BS,ff=FF,cache=None,coarse=False):
    values=[];coarse_values=[];model.eval()
    with torch.no_grad():
        for row in rows:
            pieces=[];cpieces=[]
            for first in range(0,len(bs),8):
                bb=np.asarray(bs[first:first+8],np.int64)
                if cache is None:pieces.append(actual_block(model,data,int(row),field,bb,ff).cpu().numpy())
                else:
                    h,hc=cached_block(model,data,int(row),field,bb,ff,cache)
                    pieces.append(h[0].cpu().numpy());cpieces.append(hc[0].cpu().numpy())
            values.append(np.concatenate(pieces))
            if cpieces:coarse_values.append(np.concatenate(cpieces))
    return np.stack(values),np.stack(coarse_values)if coarse_values else None
@torch.no_grad()
def query_full_position(model,data,row):
    # Same original-query block operations, but prepare the row directions once.
    model.eval();model._qualify();dev=model.bs_ports.device
    position=torch.tensor(data.positions([row]),device=dev,dtype=torch.float64)
    site=torch.tensor(data.sites([row]),device=dev,dtype=torch.long)
    origin=model.spec.ue_world_positions(position,site);tx=model.bs_ports[site]
    before=len(model.ray_provider.bank.opened_ids)
    dirs=model.ray_provider.directions(origin,site,tx)
    if len(model.ray_provider.bank.opened_ids)!=before:raise RuntimeError("Outer point geometry payload opened")
    freq=torch.tensor(data.frequencies(FF),device=dev,dtype=torch.float64)
    full=np.empty((256,4,192),np.complex64)
    for field in range(8):
        p,u=divmod(field,4)
        for start in range(0,128,8):
            ix=torch.arange(start,start+8,device=dev)
            h,_=model._scalar_links(field,origin[:,u],dirs[:,p,:,u].index_select(1,ix),
                tx[:,p].index_select(1,ix),freq)
            full[p*128+start:p*128+start+8,u]=h[0].cpu().numpy()
    if not np.isfinite(full).all():raise FloatingPointError("Nonfinite complete H")
    return full
def profile_query(config_path,uuid):
    cfg=load_config(config_path);src=source_identity(config_path);gpu=gpu_setup(uuid);C.seed_all(cfg["seed"])
    data=prepare_data(0,"screen");start=time.monotonic();_,_,model=build(data,cfg,"cuda:0");cold=time.monotonic()-start
    row=int(data.fit_ids[0]);torch.cuda.synchronize();start=time.monotonic()
    value=query_full_position(model,data,row);torch.cuda.synchronize();seconds=time.monotonic()-start
    reference=np.empty_like(value);start=time.monotonic()
    for field in range(8):
        p,u=divmod(field,4);v,_=predict_field(model,data,[row],field)
        reference[p*128:(p+1)*128,u]=v[0]
    original=time.monotonic()-start;relative=float(np.linalg.norm(value-reference)/max(np.linalg.norm(reference),1e-30))
    if relative>1e-5:raise RuntimeError(("Complete original query block mismatch",relative))
    assert not data._fit_counts.sum()and not data._val_counts.sum()
    result=dict(status="passed",source_identity=src,GPU=gpu,shape=list(value.shape),dtype=str(value.dtype),
      cold_geometry_build_seconds=cold,new_query_seconds=seconds,original_query_block_seconds=original,
      complete_query_relative_L2=relative,FIT_row=row,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
      optimizer_updates=0,CSI_labels_read=0,outer_truth_reads=0,quality_scored=False)
    C.atomic_json(R/"FIT_query_profile.json",result);emit(result)
def fit_stats(pred,truth,scale):
    p=np.asarray(pred,np.complex128);t=np.asarray(truth,np.complex128)
    se=np.sum(abs(p-t)**2,(1,2));te=np.sum(abs(t)**2,(1,2));pe=np.sum(abs(p)**2,(1,2))
    return dict(nmse=float(se.sum()/te.sum())if te.sum()>0 else None,
      energy_ratio=float(pe.sum()/te.sum())if te.sum()>0 else None,
      per_position=[dict(nmse=float(s/e)if e>0 else None,energy_ratio=float(v/e)if e>0 else None,
        zero_truth=bool(e==0),prediction_power_FIT_RMS_squared=float(v/(128*192*scale**2)))
        for s,e,v in zip(se,te,pe)])
def gate(stats,stage):
    if stats["nmse"]is None:return False
    return bool(stats["nmse"]<=(.05 if stage=="coarse"else .1)and all(
      x["zero_truth"]or(x["nmse"]<=(.1 if stage=="coarse"else .2)and .7<=x["energy_ratio"]<=1.3)
      for x in stats["per_position"]))
def validation_field(model,data,field):
    if data.role!="screen":raise PermissionError("Refit validation labels forbidden")
    p,u=divmod(field,4);bs=np.linspace(0,127,8,dtype=np.int64);ff=SELECT.FREQUENCY_IDS
    pred,_=predict_field(model,data,data.probe_ids,field,bs,ff)
    truth=data.read_validation(data.probe_ids,freq_ids=ff,normalized=False)
    truth=np.take(truth[:,p*128:(p+1)*128,u],bs,axis=1)
    se=float(np.sum(abs(pred.astype(np.complex128)-truth.astype(np.complex128))**2))
    te=float(np.sum(abs(truth.astype(np.complex128))**2))
    return dict(value=se/max(te,1e-30),squared_error=se,truth_energy=te,rows=data.probe_ids.tolist(),
      BS_ids=(p*128+bs).tolist(),UE_ids=[u],frequency_ids=ff.tolist(),kind="original_fixed_validation_subtensor_NMSE_field_slice")
def optimizer_steps(opt):
    return {str(k):float(v["step"])for k,v in opt.state_dict()["state"].items()if"step"in v}
def load_optimizer(opt,state):
    canonical=copy.deepcopy(state);opt.load_state_dict(copy.deepcopy(canonical))
    for key,st in opt.state_dict()["state"].items():
        for name,value in st.items():
            old=canonical["state"][key][name]
            if torch.is_tensor(value)and torch.is_tensor(old):
                if not torch.equal(value.detach().cpu(),old.cpu()):raise RuntimeError("Optimizer restore content mismatch")
                if value.device.type=="cpu"and value.data_ptr()==old.data_ptr():raise RuntimeError("Shared optimizer CPU storage")

def cpu_check(config_path):
    cfg=load_config(config_path);src=source_identity(config_path);data=prepare_data(0,"screen")
    sampler=PermutationSampler(data,cfg["batch_positions"],cfg["seed"])
    seen=np.zeros(4000,np.int64)
    for _ in range(sampler.coverage_steps):
        rr,bb=sampler.draw();sampler.commit(rr,bb);np.add.at(seen,rr,1)
    assert np.all(seen[data.fit_ids]>0)and not np.any(seen[data.outer_ids])
    rows=np.asarray([data.fit_ids[0],data.fit_ids[-1]],np.int64)
    block=block_labels(data,rows,[0,15],0)
    assert block.shape==(2,8,192)and block.dtype==np.complex64 and np.isfinite(block).all()
    bank=FitGeometryBank(GEOMETRY,data.inputs,0,"screen_fit")
    assert set(bank.source_fit_ids)==set(data.fit_ids)and not bank.opened_ids
    assert not torch.cuda.is_initialized()
    all_roles={}
    for k in range(5):
        for role in("screen","refit"):
            d=prepare_data(k,role);s=PermutationSampler(d,cfg["batch_positions"],cfg["seed"])
            all_roles[f"outer{k}_{role}"]=dict(fit=len(d.fit_ids),validation=len(d.val_ids),outer=len(d.outer_ids),
              coverage_steps=s.coverage_steps,minimum_stage_budget=sum(cfg["stage_min_updates"].values()),FIT_RMS=d.scale)
    report=dict(status="CPU_interface_pass",source_identity=src,config=cfg,label_shape=list(block.shape),all_roles=all_roles,
      permutation_coverage_steps=sampler.coverage_steps,FIT_positions=len(data.fit_ids),
      field_count=8,direction_cache_bound_bytes=cfg["direction_cache_entries"]*128*64*3*8,
      full_role_allfield_direction_cache_bytes=len(data.fit_ids)*8*128*64*3*8,
      geometry_FIT_count=len(bank.source_fit_ids),outer_truth_reads=0,validation_truth_reads=0,
      no_model_forward_or_optimizer=True,CUDA_initialized=False)
    C.atomic_json(R/"CPU_interface_check.json",report);emit(report)

def smoke(config_path,uuid):
    cfg=load_config(config_path);src=source_identity(config_path);gpu=gpu_setup(uuid);C.seed_all(cfg["seed"])
    data=prepare_data(0,"screen");started=time.monotonic();bank,base,model=build(data,cfg,"cuda:0");build_seconds=time.monotonic()-started
    initial=cpu_state(model);cpu_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state_all()
    records=[]
    for batch in cfg["smoke_batch_options"]:
        model.load_state_dict(copy.deepcopy(initial));torch.set_rng_state(cpu_rng.clone());torch.cuda.set_rng_state_all(cuda_rng)
        sampler=PermutationSampler(data,batch,cfg["seed"]);cache=QueryDirectionLRU(model,data,cfg["direction_cache_entries"])
        opt=torch.optim.Adam(model.fine[0].parameters(),lr=cfg["learning_rates"]["native"])
        set_stage(model,0,"native",opt,True)
        row=int(data.fit_ids[0]);bb=BS[:8]
        model.eval()
        with torch.no_grad():
            h,_=cached_block(model,data,row,0,bb,FF,cache)
            actual=actual_block(model,data,row,0,bb,FF)
        err=float((h[0]-actual).norm()/actual.norm().clamp_min(1e-30))
        if err>1e-5:raise RuntimeError(("Cached8BSx192F route mismatch",err))
        torch.cuda.reset_peak_memory_stats();times=[];events=[]
        for step in range(1,cfg["smoke_updates"]+1):
            torch.cuda.synchronize();tick=time.monotonic();event=update(model,data,cache,sampler,0,opt,cfg);torch.cuda.synchronize()
            duration=time.monotonic()-tick;times.append(duration)
            if step in(1,5,10,15,20):
                event={**event,"step":step,"batch_positions":batch,"seconds":duration};events.append(event);emit(event,R/"smoke_events.jsonl")
        rec=dict(batch_positions=batch,actual_optimizer_updates=20,seconds=float(sum(times)),median_update_seconds=float(np.median(times[2:])),
          scalar_labels_per_update=batch*8*192,scalar_labels_per_second=batch*8*192/float(np.median(times[2:])),
          peak_cuda_bytes=torch.cuda.max_memory_allocated(),cache=cache.info(),route_relative_L2=err,
          all_gradients_finite=True,optimizer_actual_steps=optimizer_steps(opt),events=events,weights_discarded=True)
        records.append(rec);C.atomic_json(R/"smoke_progress.json",dict(completed=records,source_identity=src,GPU=gpu,build_seconds=build_seconds))
        del opt,cache;torch.cuda.empty_cache()
    check_sources(src);C.atomic_json(R/"smoke_result.json",dict(status="passed",source_identity=src,config=cfg,GPU=gpu,
      build_seconds=build_seconds,records=records,weights_discarded=True,formal_training_started=False,
      outer_truth_reads=0,validation_truth_reads=0))
    emit(dict(status="smoke_passed",build_seconds=build_seconds,records=records))

def role_paths(outer,role):
    return A/f"outer{outer}"/role,R/f"outer{outer}"/role
def run_role(outer,role,config_path,uuid):
    cfg=load_config(config_path);src=source_identity(config_path);aa,rr=role_paths(outer,role)
    aa.mkdir(parents=True,exist_ok=True);rr.mkdir(parents=True,exist_ok=True)
    if(rr/"result.json").exists():
        old=read_json(rr/"result.json")
        if old["status"]=="complete":
            if old["source_identity"]!=src:raise RuntimeError("Completed formal role differs")
            return old
    gpu=gpu_setup(uuid);C.seed_all(cfg["seed"]);data=prepare_data(outer,role)
    build_start=time.monotonic();bank,base,model=build(data,cfg,"cuda:0")
    geometry_build_seconds=time.monotonic()-build_start;cache=QueryDirectionLRU(model,data,cfg["direction_cache_entries"])
    panel=fit_panel(data);panelraw=data.read_fit(panel,normalized=False)
    source_selection=None
    if role=="refit":
        source_selection=read_json(role_paths(outer,"screen")[1]/"result.json")
        if source_selection["status"]!="complete"or source_selection["source_identity"]!=src:raise RuntimeError("Frozen screen selection required")
    saved=aa/"latest.pt";state=dict(field=0,stage="coarse",phase_step=0,total_updates=0,fields={},phase_records={},
      history=[],stable=0,clamped_batches=0,elapsed_seconds=0.,best_value=None,best_step=None,stale=0)
    resumed=None
    if saved.exists():
        resumed=torch.load(saved,map_location="cpu",weights_only=False)
        if resumed["source_identity"]!=src or resumed["method_config_sha256"]!=content_hash(cfg):raise RuntimeError("Resume source/config drift")
        model.load_state_dict(resumed["model"]);state=resumed["state"]
        data._fit_counts+=resumed["fit_read_counts"];data._val_counts+=resumed["validation_read_counts"]
    clock=time.monotonic();elapsed_before=state["elapsed_seconds"];last_checkpoint=time.monotonic();last_print=time.monotonic()
    def elapsed():return elapsed_before+time.monotonic()-clock
    current=None
    def save():
        nonlocal last_checkpoint
        if current is None:return
        state["elapsed_seconds"]=elapsed()
        C.atomic_torch_save(saved,dict(schema="newrf-formal-resume-v1",model=cpu_state(model),
          coarse_optimizer=copy.deepcopy(current["cop"].state_dict()),fine_optimizer=copy.deepcopy(current["fop"].state_dict()),
          sampler=current["sampler"].snapshot(),scheduler=copy.deepcopy(current["scheduler"].state_dict())if current.get("scheduler")else None,state=copy.deepcopy(state),source_identity=src,method_config_sha256=content_hash(cfg),
          torch_rng=torch.get_rng_state().clone(),cuda_rng=torch.cuda.get_rng_state_all(),fit_data=data.info(),
          fit_read_counts=data._fit_counts.copy(),validation_read_counts=data._val_counts.copy(),outer=outer,role=role))
        last_checkpoint=time.monotonic()
    C.atomic_json(rr/"run_manifest.json",dict(source_identity=src,method_config_sha256=content_hash(cfg),config=cfg,GPU=gpu,
      outer=outer,role=role,fit_panel=panel.tolist(),validation_panel=data.probe_ids.tolist()if role=="screen"else[],
      geometry=bank.metadata(),geometry_build_seconds=geometry_build_seconds,data=data.info(),training_shape="per-position8BSx192F; multiple positions gradient-accumulated",
      denominator="max(batch mean raw truth power,1e-6*role_FIT_RMS_squared)*batch_scalar_count",
      original_numeric_gates="coarse pooled<=.05/nonzero point<=.1/E.7..1.3; warm/native pooled<=.1/point<=.2/E.7..1.3",
      gate_scope="fixed FIT panel diagnostic and finite phase stopping; caps do not prevent later stages",
      source_screen_selection_sha256=sha(role_paths(outer,"screen")[1]/"result.json")if role=="refit"else None))
    try:
        for field in range(8):
            if str(field)in state["fields"]:continue
            named=dict(model.fine[field].named_parameters())
            active=[p for n,p in named.items()if not n.startswith("network.alpha_out.")]
            alpha=[p for n,p in named.items()if n.startswith("network.alpha_out.")]
            cop=torch.optim.Adam(model.coarse[field].parameters(),lr=cfg["learning_rates"]["coarse"])
            fop=torch.optim.Adam([dict(params=active),dict(params=alpha)],lr=cfg["learning_rates"]["warm"])
            sampler=PermutationSampler(data,cfg["batch_positions"],cfg["seed"]+1000*outer+field)
            current=dict(cop=cop,fop=fop,sampler=sampler)
            continuing=resumed is not None and state["field"]==field
            if continuing:
                load_optimizer(cop,resumed["coarse_optimizer"]);load_optimizer(fop,resumed["fine_optimizer"]);sampler.restore(resumed["sampler"])
                torch.set_rng_state(resumed["torch_rng"].clone());torch.cuda.set_rng_state_all(resumed["cuda_rng"])
            else:
                state.update(field=field,stage="coarse",phase_step=0,phase_records={},history=[],stable=0,best_value=None,best_step=None,stale=0,
                  native_release_check=None,phase_nonzero_gradient_updates=0)
            if not continuing:state["initial_field_hash"]=content_hash({k:sha_tensor(v)for k,v in model.fine[field].state_dict().items()})
            field_initial_hash=state["initial_field_hash"]
            for stage in STAGES:
                if stage in state["phase_records"]:continue
                same_phase=continuing and state["stage"]==stage and state["phase_step"]>0
                if not same_phase:state.update(stage=stage,phase_step=0,stable=0,phase_nonzero_gradient_updates=0)
                opt=cop if stage=="coarse"else fop
                if stage=="native"and not same_phase:
                    row=int(panel[0])
                    set_stage(model,field,"warm",fop,False);model.eval()
                    with torch.no_grad():before=cached_block(model,data,row,field,BS[:8],FF,cache)[0]
                    set_stage(model,field,"native",fop,False)
                    with torch.no_grad():after=cached_block(model,data,row,field,BS[:8],FF,cache)[0]
                    if not torch.equal(before,after):raise RuntimeError("Native release changed forward values")
                    state["native_release_check"]=dict(forward_bitwise_no_jump=True,alpha_states_before=sum(p in fop.state for p in alpha),
                      active_Adam_states_preserved=sum(p in fop.state for p in active))
                else:set_stage(model,field,stage,fop,not same_phase)
                if not same_phase:
                    for g in opt.param_groups:g["lr"]=cfg["learning_rates"][stage]
                scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",factor=.5,patience=2,threshold=1e-4,threshold_mode="abs",min_lr=1e-5)if stage=="coarse"else None
                current["scheduler"]=scheduler
                if same_phase and resumed.get("scheduler")and scheduler:scheduler.load_state_dict(resumed["scheduler"])
                minimum=int(cfg["stage_min_updates"][stage]);cap=int(cfg["stage_max_updates"][stage])
                previous=sum(x["updates"]for x in state["phase_records"].values())
                coverage_extension=0
                if stage=="native":
                    required=max(1,sampler.coverage_steps-previous)
                    minimum=max(minimum,required);coverage_extension=max(0,required-cap);cap=max(cap,required)
                frozen_stage=None
                if role=="refit":
                    origin=source_selection["fields"][str(field)]["stages"][stage]
                    n=origin["selected_updates"]if stage=="native"else origin["updates"]
                    ratio=len(data.fit_ids)/len(source_selection["fit_ids"])
                    proportional=math.ceil(n*ratio)
                    cap=max(proportional,max(1,sampler.coverage_steps-previous))if stage=="native"else proportional
                    coverage_extension=cap-proportional;minimum=cap;frozen_stage=origin
                phase_seen=np.zeros(4000,np.int64)
                if same_phase:phase_seen=np.asarray(state.get("phase_seen",phase_seen),np.int64)
                lr_events=state.get("lr_events",[])if same_phase else[]
                while state["phase_step"]<cap:
                    if STOP or elapsed()>cfg["max_role_wall_seconds"]:save();raise RuntimeError("Resumable formal pause: signal or registered role wall cap")
                    if role=="refit"and stage=="coarse":
                        replay=[x for x in frozen_stage.get("lr_events",[])if math.ceil(x["step"]*ratio)<=state["phase_step"]]
                        if replay:
                            for g in opt.param_groups:g["lr"]=replay[-1]["lr"]
                    event=update(model,data,cache,sampler,field,opt,cfg)
                    state["phase_step"]+=1;state["total_updates"]+=1;state["clamped_batches"]+=int(event["denominator_clamped"])
                    state["phase_nonzero_gradient_updates"]+=int(event["gradient_norm"]>0.)
                    np.add.at(phase_seen,np.asarray(event["rows"]),1);state["phase_seen"]=phase_seen
                    step=state["phase_step"];check=step%cfg["health_every"]==0 or step==cap
                    validate=(role=="screen"and stage=="native"and(step%cfg["validation_every"]==0 or step==cap)
                      and step>=minimum and previous+step>=sampler.coverage_steps and np.all(sampler.seen[data.fit_ids]>0))
                    if check:
                        p,u=divmod(field,4);truth=panelraw[:,p*128:(p+1)*128,u]
                        fine,coarse=predict_field(model,data,panel,field,cache=cache)
                        fs=fit_stats(fine,truth,data.scale);cs=fit_stats(coarse,truth,data.scale)
                        selected=cs if stage=="coarse"else fs;passed=gate(selected,stage)
                        state["stable"]=state["stable"]+1 if passed else 0
                        if scheduler and role=="screen":
                            scheduler.step(selected["nmse"]if selected["nmse"]is not None else math.inf)
                        lr_events.append(dict(step=step,lr=float(opt.param_groups[0]["lr"])))
                        state["lr_events"]=lr_events
                        health=dict(stage=stage,field=field,step=step,FIT_fine=fs,FIT_coarse=cs,gate_passed=passed,
                          stable=state["stable"],LR=[g["lr"]for g in opt.param_groups],elapsed_seconds=elapsed())
                        state["history"].append(health);emit(health,rr/"events.jsonl");C.atomic_json(rr/"progress.json",health)
                    if validate:
                        val=validation_field(model,data,field);best=state["best_value"]
                        improved=best is None or val["value"]<best-cfg["minimum_validation_improvement"]
                        if improved:
                            state.update(best_value=val["value"],best_step=step,stale=0)
                            C.atomic_torch_save(aa/f"field{field}_selected.pt",dict(coarse=cpu_state(model.coarse[field]),
                              fine=cpu_state(model.fine[field]),native_updates=step,validation=val,source_identity=src,
                              gradient_seen_counts=sampler.seen.copy(),gradient_seen_unique_rows=int(np.count_nonzero(sampler.seen[data.fit_ids])),
                              gradient_seen_counts_sha256=hashlib.sha256(sampler.seen.tobytes()).hexdigest()))
                        else:state["stale"]+=1
                        emit(dict(stage="validation",field=field,step=step,validation=val,best_step=state["best_step"],stale=state["stale"],elapsed_seconds=elapsed()),rr/"events.jsonl")
                    if step%cfg["checkpoint_every"]==0 or time.monotonic()-last_checkpoint>=60:save()
                    if time.monotonic()-last_print>=30:
                        emit(dict(stage=stage,field=field,step=step,cap=cap,loss=event["loss"],gradient_norm=event["gradient_norm"],
                          seen_unique=int(np.count_nonzero(sampler.seen[data.fit_ids])),elapsed_seconds=elapsed()),rr/"events.jsonl");last_print=time.monotonic()
                    if role=="screen"and step>=minimum:
                        if stage!="native"and check and state["stable"]>=(1 if stage=="coarse"else 3):break
                        if stage=="native"and validate and state["stale"]>=cfg["native_patience"]:break
                chosen=state["best_step"]if role=="screen"and stage=="native"else state["phase_step"]
                state["phase_records"][stage]=dict(updates=state["phase_step"],selected_updates=chosen,
                  gate_stable_checks=state["stable"],cap=cap,min_updates=minimum,lr_events=lr_events,
                  gradient_seen_unique_rows=int(np.count_nonzero(phase_seen[data.fit_ids])),
                  gradient_seen_minimum_count=int(phase_seen[data.fit_ids].min()),minimum_coverage_extension_updates=coverage_extension,
                  nonzero_gradient_updates=state["phase_nonzero_gradient_updates"],actual_optimizer_steps=optimizer_steps(opt))
                if state["phase_nonzero_gradient_updates"]<=0:raise RuntimeError("Phase recorded no actual nonzero gradient update")
                state.update(phase_step=0,stable=0,lr_events=[],phase_seen=np.zeros(4000,np.int64))
                save();continuing=False;resumed=None
            if role=="screen":
                best=torch.load(aa/f"field{field}_selected.pt",map_location="cpu",weights_only=False)
                if best["source_identity"]!=src:raise RuntimeError("Selected field source drift")
                selected_counts=np.asarray(best["gradient_seen_counts"],np.int64)
                if (not np.all(selected_counts[data.fit_ids]>0)or np.any(selected_counts[data.outer_ids])
                    or hashlib.sha256(selected_counts.tobytes()).hexdigest()!=best["gradient_seen_counts_sha256"]):
                    raise RuntimeError("Validation-selected checkpoint has invalid gradient coverage")
                model.coarse[field].load_state_dict(best["coarse"]);model.fine[field].load_state_dict(best["fine"])
            if "physical_density"in model.fine[field].__dict__:delattr(model.fine[field],"physical_density")
            coverage=sampler.seen[data.fit_ids]
            if not np.all(coverage>0):raise RuntimeError("Formal field missed allowed FIT positions")
            counts_path=aa/f"field{field}_gradient_counts.npy";C.atomic_npy(counts_path,sampler.seen,replace=True)
            state["fields"][str(field)]=dict(stages=copy.deepcopy(state["phase_records"]),
              gradient_seen_unique_rows=int(np.count_nonzero(coverage)),gradient_seen_min_count=int(coverage.min()),
              gradient_seen_max_count=int(coverage.max()),gradient_seen_counts_path=PATHS.relative(counts_path),
              gradient_seen_counts_sha256=sha(counts_path),per_position_BS_blocks_min=int(np.count_nonzero(sampler.block_seen[data.fit_ids],axis=1).min()),
              initialized_fine_hash=field_initial_hash,native_release_check=state["native_release_check"],
              selected_validation_value=state["best_value"]if role=="screen"else None,
              selected_gradient_seen_unique_rows=best["gradient_seen_unique_rows"]if role=="screen"else int(np.count_nonzero(coverage)),
              selected_gradient_seen_counts_sha256=best["gradient_seen_counts_sha256"]if role=="screen"else hashlib.sha256(sampler.seen.tobytes()).hexdigest())
            save();emit(dict(stage="field_complete",field=field,field_record=state["fields"][str(field)],elapsed_seconds=elapsed()),rr/"events.jsonl")
            del cop,fop;current=None;torch.cuda.empty_cache()
        # All8fields must have actual full-FIT gradient coverage before complete prediction.
        if set(state["fields"])!=set(map(str,range(8))):raise RuntimeError("Incomplete trained fields")
        for net in model.fine:
            if "physical_density"in net.__dict__:raise RuntimeError("Warm backward patch leaked into final model")
        final=aa/"final.pt";C.atomic_torch_save(final,dict(model=cpu_state(model),source_identity=src,
          method_config_sha256=content_hash(cfg),fields=state["fields"],fit_data=data.info(),outer=outer,role=role,
          seed=cfg["seed"],geometry=bank.metadata(),all8fields_trained=True))
        # Original query function from an unpatched builder, all8fields, twoFIT block probes.
        _,_,clean=build(data,cfg,"cuda:0");clean.load_state_dict(torch.load(final,map_location="cpu",weights_only=False)["model"]);clean.eval()
        reload_checks={}
        for field in range(8):
            row=int(panel[field%len(panel)])
            p0,_=predict_field(model,data,[row],field,BS[:8],FF)
            p1,_=predict_field(clean,data,[row],field,BS[:8],FF)
            rel=float(np.linalg.norm(p0-p1)/max(np.linalg.norm(p0),1e-30))
            if rel>1e-5:raise RuntimeError(("Clean original query reload",field,rel))
            reload_checks[str(field)]=rel
        check_sources(src)
        result=dict(status="complete",method="newrf",outer=outer,role=role,source_identity=src,method_config_sha256=content_hash(cfg),
          fields=state["fields"],fit_ids=data.fit_ids.tolist(),fit_data=data.info(),all8fields_trained=True,
          final_checkpoint=PATHS.relative(final),final_checkpoint_sha256=sha(final),elapsed_seconds=elapsed(),geometry_build_seconds=geometry_build_seconds,
          denominator_clamped_batches=state["clamped_batches"],total_optimizer_updates=state["total_updates"],
          scalar_complex_labels_per_update=cfg["batch_positions"]*8*192,cache=cache.info(),clean_query_reload_relative_L2=reload_checks,
          peak_cuda_bytes=torch.cuda.max_memory_allocated(),GPU=gpu,outer_truth_read=False,
          validation_label_reads=int(data._val_counts.sum()),no_oracle_phase_delay_amplitude_correction=True)
        C.atomic_json(rr/"result.json",result);return result
    except BaseException:
        if current is not None:save()
        C.atomic_json(rr/"failure.json",dict(traceback=traceback.format_exc(),field=state["field"],stage=state["stage"],
          phase_step=state["phase_step"],total_updates=state["total_updates"],elapsed_seconds=elapsed(),source_identity=src))
        raise

def sha_tensor(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def predict_outer(outer,config_path,uuid):
    cfg=load_config(config_path);src=source_identity(config_path);aa,rr=role_paths(outer,"prediction")
    aa.mkdir(parents=True,exist_ok=True);rr.mkdir(parents=True,exist_ok=True)
    if(aa/"manifest.json").exists():
        m=read_json(aa/"manifest.json")
        if m["metadata"]["source_identity"]!=src:raise RuntimeError("Existing prediction source differs")
        return m
    gpu=gpu_setup(uuid);data=prepare_data(outer,"refit")
    training=read_json(role_paths(outer,"refit")[1]/"result.json")
    if training["status"]!="complete"or not training["all8fields_trained"]or training["source_identity"]!=src:raise RuntimeError("All8 trained refit fields required")
    final=PATHS.root/training["final_checkpoint"]
    if sha(final)!=training["final_checkpoint_sha256"]:raise RuntimeError("Refit checkpoint changed")
    ck=torch.load(final,map_location="cpu",weights_only=False)
    if ck["role"]!="refit"or ck["outer"]!=outer or not ck["all8fields_trained"]:raise RuntimeError("Prediction checkpoint role")
    bank,base,model=build(data,cfg,"cuda:0");model.load_state_dict(ck["model"]);model.eval()
    work=aa/"channels_work.npy";channels=aa/"channels.npy";progressfile=rr/"progress.json"
    progress=read_json(progressfile)if progressfile.exists()else dict(completed={},checkpoint_sha256=sha(final),source_identity=src)
    if progress["checkpoint_sha256"]!=sha(final)or progress["source_identity"]!=src:raise RuntimeError("Prediction resume identity")
    shape=(len(data.outer_ids),256,4,192)
    if channels.exists():
        if len(progress["completed"])!=len(data.outer_ids)or work.exists():raise RuntimeError("Incomplete final prediction commit")
        array=np.load(channels,mmap_mode="r",allow_pickle=False)
    else:array=np.load(work,mmap_mode="r+",allow_pickle=False)if work.exists()else np.lib.format.open_memmap(work,mode="w+",dtype=np.complex64,shape=shape)
    if array.shape!=shape or array.dtype!=np.complex64:raise RuntimeError("Prediction work axes")
    started=time.monotonic()
    with torch.no_grad():
        for i,row in enumerate(data.outer_ids):
            key=str(int(row))
            if key in progress["completed"]:
                if hashlib.sha256(array[i].tobytes()).hexdigest()!=progress["completed"][key]:raise RuntimeError("Committed outer prediction changed")
                continue
            array[i]=query_full_position(model,data,int(row))
            if not np.isfinite(array[i]).all():raise FloatingPointError("Nonfinite complete outerH")
            array.flush();progress["completed"][key]=hashlib.sha256(array[i].tobytes()).hexdigest()
            progress["elapsed_this_session_seconds"]=time.monotonic()-started
            C.atomic_json(progressfile,progress);emit(dict(outer=outer,stage="predict",done=len(progress["completed"]),total=len(data.outer_ids),elapsed_seconds=time.monotonic()-started),rr/"events.jsonl")
            if STOP:raise RuntimeError("Prediction paused after committed row")
    array.flush();del array
    if not channels.exists():work.replace(channels)
    ids_path=aa/"row_ids.npy"
    if ids_path.exists():
        if not np.array_equal(np.load(ids_path,allow_pickle=False),data.outer_ids):raise RuntimeError("Committed row IDs differ")
    else:C.atomic_npy(ids_path,data.outer_ids)
    check_sources(src)
    if int(data._val_counts.sum())or int(data._fit_counts.sum()):raise RuntimeError("Inference unexpectedly read channel labels")
    query_context=data.info()
    for key in["input_identity","fit_ids","fit_ids_sha256","scale","role","outer"]:
        if query_context[key]!=training["fit_data"][key]:raise RuntimeError(("Training and inference role mismatch",key))
    manifest=dict(schema="canonical-baseline-prediction-v1",method="newrf",outer=outer,role="refit",kind="channel",trained=True,
      input_identity=data.inputs.identity,prediction_namespace="canonical_train_row_ids",query_truth_read=False,
      path=PATHS.relative(channels),sha256=sha(channels),shape=list(shape),dtype="complex64",row_ids=data.outer_ids.tolist(),
      row_ids_sha256=C.digest_ids(data.outer_ids),row_ids_file_sha256=sha(aa/"row_ids.npy"),fit_data=training["fit_data"],
      metadata=dict(source_identity=src,method_config_sha256=content_hash(cfg),checkpoint=PATHS.relative(final),
        checkpoint_sha256=sha(final),all8fields_trained=True,gradient_coverage=training["fields"],
        screen_selection=read_json(role_paths(outer,"screen")[1]/"result.json")["fields"],
        query_context=query_context,query_geometry_payload_reads=0,oracle_phase_delay_scale_alignment=False,raw_H=True,actual_model_forward=True,outer_quality_scored=False,
        prediction_elapsed_seconds=time.monotonic()-started,GPU=gpu))
    C.atomic_json(aa/"manifest.json",manifest);return manifest
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",type=Path,required=True)
    ap.add_argument("--mode",choices=["cpu-check","smoke","profile-query","screen","refit","predict","run"],required=True)
    ap.add_argument("--outer",type=int,default=0);ap.add_argument("--gpu-uuid",default=DEFAULT_GPU);args=ap.parse_args()
    R.mkdir(parents=True,exist_ok=True);A.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGTERM,stop_signal);signal.signal(signal.SIGINT,stop_signal)
    if args.mode=="cpu-check":cpu_check(args.config);return
    if args.mode=="smoke":smoke(args.config,args.gpu_uuid);return
    if args.mode=="profile-query":profile_query(args.config,args.gpu_uuid);return
    lock=A/f"outer{args.outer}.lock"
    with lock.open("a+")as f:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.mode in("run","screen"):run_role(args.outer,"screen",args.config,args.gpu_uuid)
        if args.mode in("run","refit"):run_role(args.outer,"refit",args.config,args.gpu_uuid)
        if args.mode in("run","predict"):predict_outer(args.outer,args.config,args.gpu_uuid)
if __name__=="__main__":
    try:main()
    except BaseException:
        R.mkdir(parents=True,exist_ok=True)
        with(R/"runner_failure.log").open("a")as f:f.write(traceback.format_exc()+"\n")
        raise
