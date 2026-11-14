"""NeRF2 frequency-minibatch continuation of the frozen full192 experiment.

All192 head parameters remain. Training selects head rows before F.linear
and propagates only the selected physical frequencies. The full192 evaluation
functions, allowed data, optimizer update budget and spatial roles are reused.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,math,os,random,signal,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from project_paths import PATHS
from round2.paper_baseline_training_v1 import common as C
from round2.baseline_formal_20261111 import nerf2_portable as IO

REPORT=C.REPORT/"freqminibatch_20261112/nerf2"
ART=C.ART/"freqminibatch_20261112/nerf2"
OLD_REPORT=C.REPORT/"formal_20261111/nerf2"
OLD_ART=C.ART/"formal_20261111/nerf2"
OLD_CONFIG=OLD_REPORT/"method_config.json"
IMMUTABLE=ART/"migration/outer2_screen_immutable.pt"
M=None;CFG=None;ACTIVE=None;BASE_SNAPSHOT=None;BASE_SCHEDULE=None;STAGE_ORIGIN=None

class ToneState:
    def __init__(self,fields,k,seed):
        self.k=int(k);self.rng=np.random.default_rng(seed);self.pending=None
        self.updates={key:np.zeros(192,np.int64)for key in fields}
        self.elements={key:np.zeros(192,np.int64)for key in fields}
        self.legacy_updates={key:0 for key in fields};self.minibatch_updates={key:0 for key in fields}
    def peek(self,key):
        if self.pending is None:
            self.pending=dict(key=key,ids=np.sort(self.rng.choice(192,self.k,replace=False)).tolist())
        if self.pending["key"]!=key:raise RuntimeError("Pending tone batch belongs to another field")
        return self.pending["ids"]
    def commit(self,key,rows,bs):
        if self.pending is None or self.pending["key"]!=key:raise RuntimeError("No matching tone transaction")
        ids=self.pending["ids"];self.updates[key][ids]+=1
        self.elements[key][ids]+=len(rows)*len(bs)*2
        self.minibatch_updates[key]+=1;self.pending=None
    def state_dict(self):
        return copy.deepcopy(dict(k=self.k,rng=self.rng.bit_generator.state,pending=self.pending,
            updates=self.updates,elements=self.elements,legacy_updates=self.legacy_updates,
            minibatch_updates=self.minibatch_updates))
    def load(self,state):
        if state["k"]!=self.k:raise RuntimeError("Frequency minibatch size changed on resume")
        self.rng.bit_generator.state=copy.deepcopy(state["rng"]);self.pending=copy.deepcopy(state["pending"])
        for name in ["updates","elements","legacy_updates","minibatch_updates"]:
            setattr(self,name,copy.deepcopy(state[name]))
    def inherit_full192(self,schedule):
        for key in schedule.counts:
            n=int(schedule.counts[key]);self.updates[key][:]=n
            self.elements[key][:]=int(schedule.BS_supervision_counts[key].sum(dtype=np.uint64))*2
            self.legacy_updates[key]=n
    def summary(self):
        return dict(frequency_count=192,training_K=self.k,sampling="independent Generator.choice without replacement each update",
            objective="mean_K gives an unbiased stochastic data-loss estimator of full192; optimizer trajectory differs",
            normalization="full192 FIT field power and fixed floor, never estimated from selected tones or query CSI",
            per_field={key:dict(full192_pre_migration_updates=self.legacy_updates[key],
                minibatch_updates=self.minibatch_updates[key],tone_step_counts=self.updates[key].tolist(),
                tone_complex_supervision_elements=self.elements[key].tolist(),
                actual_complex_supervision_elements=int(self.elements[key].sum()),
                tone_step_count_min=int(self.updates[key].min()),tone_step_count_max=int(self.updates[key].max()))
                for key in self.updates})

def configure(config_path,gpu):
    global M,CFG,ACTIVE,BASE_SNAPSHOT,BASE_SCHEDULE
    if gpu not in C.ALLOWED:raise PermissionError("Unapproved full GPU UUID")
    cfg=json.loads(config_path.read_text());cfg["_config_rel"]=PATHS.relative(config_path)
    old=json.loads(OLD_CONFIG.read_text())
    for key,value in old.items():
        if key in cfg and cfg[key]!=value and key not in {"schema","formal_frozen","min_free_GiB"}:
            raise RuntimeError("Original formal parameter changed: "+key)
    if cfg["frequency_batch"] not in range(1,193):raise ValueError("K must be1..192")
    m,device_proof=IO.isolated_import(gpu)
    base_schedule=m.Schedule;base_snapshot=m.snapshot;base_restore=m.restore
    M,CFG,BASE_SCHEDULE,BASE_SNAPSHOT=m,cfg,base_schedule,base_snapshot
    class FrequencySchedule(base_schedule):
        def __init__(self,data,batch,bs,seed):
            global ACTIVE
            super().__init__(data,batch,bs,seed)
            tone_seed=cfg["frequency_seed"]+seed+(0 if data.role=="screen"else 1000003)
            self.tones=ToneState(m.FIELDS,cfg["frequency_batch"],tone_seed)
            self.migration=copy.deepcopy(STAGE_ORIGIN)
            ACTIVE=self
        def commit(self,key,rows,bs):
            super().commit(key,rows,bs);self.tones.commit(key,rows,bs)
        def state_dict(self):
            d=super().state_dict();d["frequency_sampling"]=self.tones.state_dict();return d
        def load_state_dict(self,state):
            super().load_state_dict(state)
            if "frequency_sampling"in state:self.tones.load(state["frequency_sampling"])
            else:self.tones.inherit_full192(self)
        def coverage(self):
            out=super().coverage()
            for key,value in out.items():
                value["nominal_full192_equivalent_elements"]=value["supervised_complex_elements"]
                value["supervised_complex_elements"]=int(self.tones.elements[key].sum())
                value["actual_frequency_count_min"]=int(self.tones.updates[key].min())
                value["actual_frequency_count_max"]=int(self.tones.updates[key].max())
            return out
    def frequency_snapshot(*args,**kwargs):
        d=base_snapshot(*args,**kwargs)
        d["frequency_sampling"]=args[2].tones.summary()
        d["training_version"]="freqminibatch_20261112"
        d["migration"]=copy.deepcopy(args[2].migration)
        return d
    def frequency_restore(ck,model,opt,schedule,source,data):
        value=base_restore(ck,model,opt,schedule,source,data)
        schedule.migration=copy.deepcopy(ck.get("migration",schedule.migration))
        return value
    m.Schedule=FrequencySchedule;m.snapshot=frequency_snapshot;m.train_step=train_step;m.restore=frequency_restore
    m.ART=ART;m.REPORT=REPORT;m.GPU=gpu
    source=m.identity(OLD_CONFIG)
    source[PATHS.relative(Path(__file__))]=C.sha256(Path(__file__))
    source[PATHS.relative(Path(IO.__file__))]=C.sha256(Path(IO.__file__))
    source[PATHS.relative(config_path)]=C.sha256(config_path)
    return m,cfg,source,device_proof

class FrequencyRenderer:
    def __init__(self,data,rows,key,frequency_ids):
        self.base=M.Renderer(data,rows,key);self.ids=np.asarray(frequency_ids,np.int64)
        if self.ids.ndim!=1 or len(self.ids)!=len(np.unique(self.ids))or np.any((self.ids<0)|(self.ids>=192)):
            raise ValueError("Distinct physical tone indices required")
        self.f=self.base.f[torch.tensor(self.ids,device="cuda:0")]
    def __call__(self,model,bs):
        r=self.base;core=model.fields[r.key];field=core.field;cfg=core.config
        nr,ns,nf=cfg.n_rays,cfg.n_samples,len(self.ids)
        if(nr,ns,cfg.far_m,cfg.near_m)!=(324,32,810.,0.):raise RuntimeError("Physical sampling changed")
        device=next(field.parameters()).device
        selected=torch.tensor(np.r_[self.ids,192+self.ids],device=device,dtype=torch.int64)
        centre=torch.tensor(cfg.world_center_m,device=device,dtype=torch.float64)
        directions=core.ray_directions.to(dtype=torch.float64)
        directions=directions/torch.linalg.vector_norm(directions,dim=-1,keepdim=True)
        nports=len(r.rows)*2;ray_count=nports*nr;segment=(cfg.far_m-cfg.near_m)/ns
        radii=cfg.near_m+segment*torch.arange(1,ns+1,device=device,dtype=torch.float64)
        points=(r.world.reshape(nports,3)[:,None,None,:]+directions[None,:,None,:]*radii[None,None,:,None]).reshape(ray_count,ns,3)
        xyz=((points-centre)/cfg.world_scale_m).float()
        points_frequency=torch.cat([xyz,torch.zeros((*xyz.shape[:-1],1),device=device)],-1)
        view=directions.float()[None,:,None,:].expand(nports,nr,ns,3).reshape(ray_count,ns,3)
        embedded=field.embed_pts_fn(points_frequency).contiguous()
        shape=embedded.shape[:-1];pts=embedded.reshape(-1,embedded.shape[-1]);x=pts
        for i,layer in enumerate(field.attenuation_linears):
            x=F.relu(layer(x))
            if i in field.skips:x=torch.cat([pts,x],-1)
        # Slice original192 parameter rows BEFORE either expensive output map.
        raw=F.linear(x,field.attenuation_output.weight.index_select(0,selected),
                        field.attenuation_output.bias.index_select(0,selected))
        attenuation=torch.cat([F.softplus(raw[...,:nf])/810.,raw[...,nf:]],-1)
        feature=field.feature_layer(x)
        encoded_view=field.embed_view_fn(view).reshape(len(feature),-1)
        common=torch.cat([feature,encoded_view],-1)
        tx=model.bound_bs_ports[r.site,r.bp].reshape(128,3)
        encoded=field.embed_tx_fn(((tx-centre)/cfg.world_scale_m).float())
        first=field.signal_linears[0];td=encoded.shape[-1];cd=first.in_features-td
        shared=F.linear(common,first.weight[:,:cd],first.bias)
        txpart=F.linear(encoded,first.weight[:,cd:],None)
        ix=torch.as_tensor(bs,device=device,dtype=torch.int64)
        x=F.relu(shared[None]+txpart.index_select(0,ix)[:,None])
        for layer in field.signal_linears[1:]:x=F.relu(layer(x))
        signal=F.linear(x,field.signal_output.weight.index_select(0,selected),
                         field.signal_output.bias.index_select(0,selected))
        att=attenuation.reshape(ray_count,ns,2,nf).permute(0,3,1,2)
        sig=signal.reshape(len(bs),ray_count,ns,2,nf).permute(0,1,4,2,3)
        transport_shape=(ray_count,nf,ns)
        segments=torch.full((ns,),segment,device=device,dtype=torch.float64)
        trans,carrier=M.FF._shared_transport(att.reshape(-1,2),transport_shape,segments,self.f,cfg.propagation_sign,cfg.near_m)
        rays=M.FF._factorized_integral(sig.reshape(len(bs),-1,2),transport_shape,trans,carrier)
        value=rays.reshape(len(bs),len(r.rows),2,nr,nf).sum(3).permute(1,0,2,3)
        return value*model.log_output_gain.exp()

def train_step(model,opt,data,cache,rows,key,bs,floor,objective="relative",audit_components=False):
    ids=ACTIVE.tones.peek(key);model.train();opt.zero_grad(set_to_none=True)
    pred=FrequencyRenderer(data,rows,key,ids)(model,bs)
    full_truth,power=cache.target(rows,key,bs)
    truth=full_truth.index_select(-1,torch.tensor(ids,device="cuda:0"))
    native=.5*(pred-truth).abs().square().mean((1,2,3))
    vector=native if objective=="native_warmup"else native/power.clamp_min(floor)
    loss=vector.mean();zero=power==0
    if not bool(torch.isfinite(loss)):raise FloatingPointError("Nonfinite frequency minibatch loss")
    loss.backward()
    active=[p for p in model.parameters()if p.grad is not None]
    grad=float(torch.nn.utils.clip_grad_norm_(active,10.,error_if_nonfinite=True));opt.step()
    return dict(loss=float(loss.detach()),gradient_norm=grad,scalar_labels=pred.numel(),
        zero_field_positions=int(zero.sum()),field=key,objective=objective,frequency_ids=ids,
        zero_loss_contribution=float(vector[zero].sum().detach()/len(rows))if bool(zero.any())else 0.,
        nonzero_loss_contribution=float(vector[~zero].sum().detach()/len(rows))if bool((~zero).any())else 0.,
        component_gradient_QA={},component_gradient_QA_seconds=0.)

def stable_content_hash(value):
    """Value hash independent of pickle storage identities and reload addresses."""
    h=hashlib.sha256()
    def walk(x):
        if isinstance(x,torch.Tensor):
            z=x.detach().cpu().contiguous()
            h.update(b"torch");h.update(str((str(z.dtype),tuple(z.shape))).encode());h.update(z.numpy().tobytes())
        elif isinstance(x,np.ndarray):
            z=np.ascontiguousarray(x);h.update(b"numpy");h.update(str((z.dtype.str,z.shape)).encode());h.update(z.tobytes())
        elif isinstance(x,dict):
            h.update(b"dict")
            for k in sorted(x,key=repr):walk(k);walk(x[k])
        elif isinstance(x,(list,tuple)):
            h.update(type(x).__name__.encode());h.update(str(len(x)).encode())
            for v in x:walk(v)
        elif isinstance(x,bytes):
            h.update(b"bytes");h.update(str(len(x)).encode());h.update(x)
        else:
            h.update(type(x).__name__.encode());h.update(repr(x).encode());h.update(b";")
    walk(value);return h.hexdigest()

def state_fingerprints(ck):
    h=hashlib.sha256()
    for key,value in ck["model"].items():h.update(key.encode());h.update(stable_content_hash(value).encode())
    return dict(model_sha256=h.hexdigest(),optimizer=M.optimizer_fingerprint(ck["optimizer"]),
        rng_sha256=stable_content_hash(ck["rng"]),
        original_schedule_sha256=stable_content_hash({k:v for k,v in ck["schedule"].items()if k!="frequency_sampling"}))

def migrate(source,cfg):
    """Metadata-compatible copies only. Old working files and readonly parents stay untouched."""
    a,r=M.locations(2,"screen")
    if(a/"last.pt").exists():raise FileExistsError("New migration checkpoint already exists")
    parent=torch.load(IMMUTABLE,map_location="cpu",weights_only=False)
    if C.sha256(IMMUTABLE)!=cfg["migration_checkpoint_sha256"]:raise RuntimeError("Immutable parent SHA differs")
    data=C.FoldData(2,"screen","nerf2")
    schedule=M.Schedule(data,cfg["position_batch"],cfg["BS_batch"],cfg["seed"]+1009*2+97)
    schedule.load_state_dict(parent["schedule"])
    before=state_fingerprints(parent)
    old_best=OLD_ART/"outer2/screen/best.pt"
    best_parent=ART/"migration/outer2_screen_best_immutable.pt"
    if not best_parent.exists():
        import shutil
        shutil.copy2(old_best,best_parent);best_parent.chmod(0o444)
    parents=[]
    for old,dst,kind in[(IMMUTABLE,a/"last.pt","current"),(best_parent,a/"best.pt","best")]:
        ck=torch.load(old,map_location="cpu",weights_only=False)
        old_source=copy.deepcopy(ck["source_identity"]);fp=state_fingerprints(ck)
        schedule=M.Schedule(data,cfg["position_batch"],cfg["BS_batch"],cfg["seed"]+1009*2+97)
        schedule.load_state_dict(ck["schedule"])
        ck["schedule"]=schedule.state_dict();ck["frequency_sampling"]=schedule.tones.summary()
        ck["source_identity"]=source;ck["config"]=cfg
        ck["migration"]=dict(parent=PATHS.relative(old),parent_sha256=C.sha256(old),
            parent_source_identity=old_source,optimizer_updates_inherited=ck["step"],
            frequency_regime_before_migration="all192",frozen_state_fingerprints=fp)
        if state_fingerprints(ck)!=fp:raise RuntimeError("Model/Adam/RNG/original schedule changed in migration")
        C.atomic_torch_save(dst,ck)
        reloaded=torch.load(dst,map_location="cpu",weights_only=False)
        if state_fingerprints(reloaded)!=fp:raise RuntimeError("Reloaded migration state differs from immutable parent")
        parents.append(dict(kind=kind,path=PATHS.relative(dst),sha256=C.sha256(dst),step=ck["step"],parent=ck["migration"],
                            reloaded_model_Adam_LR_RNG_original_sampler_exact=True))
    result=dict(status="passed",source_identity=source,parents=parents,no_optimizer_updates=True,
        current_step=parent["step"],best_step=parent["meta"]["best_step"],full_state_preserved=True)
    C.atomic_json(r/"migration.json",result);return result

def enrich_stage(fold,role,result):
    path=REPORT/f"outer{fold}/{role}/result.json"
    if"frequency_sampling"in result:return result
    final=torch.load(PATHS.root/result["final_checkpoint"],map_location="cpu",weights_only=False)
    out=copy.deepcopy(result);out["frequency_sampling"]=final["frequency_sampling"]
    out["actualsupervision"]={key:value["actual_complex_supervision_elements"]
        for key,value in final["frequency_sampling"]["per_field"].items()}
    out["migration"]=final.get("migration")
    out["training_frequency_regime"]="full192 inherited history plus selected-K stochastic updates"
    selected=torch.load(PATHS.root/result["checkpoint"],map_location="cpu",weights_only=False)
    out["selected_frequency_sampling"]=selected["frequency_sampling"]
    if selected.get("migration"):out["selected_migration"]=selected["migration"]
    C.atomic_json(path.with_name("base_result_before_frequency_annotation.json"),result)
    C.atomic_json(path,out);return out

def publish_prediction(fold,cfg,source,screen,refit,device,proof):
    report=REPORT/f"outer{fold}/prediction";report.mkdir(parents=True,exist_ok=True)
    base=IO.staged_outer_prediction(M,fold,cfg,source,screen,refit,report)
    out=copy.deepcopy(base)
    out["frequency_sampling"]=refit["frequency_sampling"];out["actualsupervision"]=refit["actualsupervision"]
    out["migration"]=refit.get("migration");out["metadata"]["adapter_version"]="freqminibatch_native192_eight_fields"
    out["runtime_adaptation"]=dict(GPU_UUID=device["GPU_UUID"],device_binding=proof,training_frequency_minibatch=True,
        evaluation_frequency_count=192,unbiased_data_loss_estimator_not_numerically_identical_training=True)
    target=ART/f"outer{fold}/prediction/manifest.json"
    if target.exists():
        if json.loads(target.read_text())!=out:raise RuntimeError("Published frequency manifest is immutable")
        return out
    C.atomic_json(report/"result.json",dict(status="complete",manifest=out))
    C.atomic_json(target,out)
    return out

def cpu_check(cfg,source):
    data=C.FoldData(2,"screen","nerf2")
    old=torch.load(IMMUTABLE,map_location="cpu",weights_only=False)
    import pickle
    state=lambda:hashlib.sha256(pickle.dumps((np.random.get_state(),random.getstate(),torch.get_rng_state().tolist()))).hexdigest()
    before=state()
    schedule=M.Schedule(data,cfg["position_batch"],cfg["BS_batch"],cfg["seed"]+1009*2+97)
    schedule.load_state_dict(old["schedule"]);key=M.FIELDS[old["step"]%8]
    pos_before=copy.deepcopy(schedule.state_dict());ff=schedule.tones.peek(key)
    if state()!=before:raise RuntimeError("Independent tone RNG perturbed global RNG")
    if len(ff)!=cfg["frequency_batch"]or len(set(ff))!=len(ff):raise RuntimeError("Tone shape")
    for k in ["rng","counts","states","row_gradient_counts","BS_supervision_counts"]:
        if pickle.dumps(schedule.state_dict()[k])!=pickle.dumps(pos_before[k]):raise RuntimeError("Tone RNG changed spatial schedule")
    for name,counters in schedule.tones.updates.items():
        if not np.all(counters==old["per_field_updates"][name]):raise RuntimeError("Full192 history frequency counts incorrect")
    # Synthetic floor/zero check only: averaging equal frequency partitions
    # preserves the full loss and unclipped gradient with one fixed fullPi.
    z=torch.linspace(-.2,.3,3*192,dtype=torch.float64).reshape(3,192).requires_grad_()
    target=torch.stack([torch.zeros(192,dtype=torch.float64),torch.full((192,),1e-4,dtype=torch.float64),torch.ones(192,dtype=torch.float64)])
    pi=target.square().mean(-1);den=pi.clamp_min(cfg["power_floor"])
    objective=lambda ids:.5*((z[:,ids]-target[:,ids]).square().mean(-1)/den).mean()
    full_loss=objective(list(range(192)));full_grad=torch.autograd.grad(full_loss,z)[0]
    partition=sum(objective(list(range(i,192,12)))/12 for i in range(12))
    partition_grad=torch.autograd.grad(partition,z)[0]
    if not torch.allclose(full_grad,partition_grad,atol=1e-10,rtol=1e-12):raise AssertionError("CPU zero/floor partition gradient")
    d=dict(status="passed",source_identity=source,original_step=old["step"],global_and_position_BS_RNG_unchanged=True,
       zero_and_subfloor_loss_gradient_partition_equivalent=True,
       tone_ids_example=ff,independent_tone_rng=True,optimizer_updates=0,
       immutable_parent_sha256=C.sha256(IMMUTABLE),state_fingerprints=state_fingerprints(old),
       full192_evaluation_functions_unchanged=True)
    C.atomic_json(REPORT/"cpu_check.json",d);print(json.dumps({k:v for k,v in d.items()if k!="state_fingerprints"}),flush=True)

def gpu_qa(cfg,source,device):
    ck=torch.load(IMMUTABLE,map_location="cpu",weights_only=False)
    data=C.FoldData(2,"screen","nerf2");model=M.build(data,cfg["seed"]+1009*2);model.load_state_dict(ck["model"])
    schedule=M.Schedule(data,cfg["position_batch"],cfg["BS_batch"],cfg["seed"]+1009*2+97)
    schedule.load_state_dict(ck["schedule"]);key=M.FIELDS[ck["step"]%8];allrows,allbs=schedule.peek(key)
    rows,bs=allrows[:2],allbs[:8]
    full=torch.tensor(data.read_fit(rows),device="cuda:0",dtype=torch.complex64)
    _,bp,up=M.parts(key)
    power=full[:,bp*128:bp*128+128,up*2:up*2+2].abs().square().mean((1,2,3),dtype=torch.float64).clamp_min(cfg["power_floor"])
    truth=full[:,torch.tensor(bs,device="cuda:0")+bp*128,up*2:up*2+2]
    params=list(model.fields[key].parameters())+[model.log_output_gain]
    def gradients(ids,original=False):
        value=(M.Renderer(data,rows,key)(model,bs)if original else FrequencyRenderer(data,rows,key,ids)(model,bs))
        if original:value=value.index_select(-1,torch.tensor(ids,device="cuda:0"))
        target=truth.index_select(-1,torch.tensor(ids,device="cuda:0"))
        loss=.5*((value-target).abs().square().mean((1,2,3))/power).mean()
        gs=torch.autograd.grad(loss,params,allow_unused=True)
        return value.detach(),[g.detach()if g is not None else torch.zeros_like(p)for g,p in zip(gs,params)]
    def relative(gs,ref):
        n=sum(float((a-b).double().square().sum())for a,b in zip(gs,ref))
        d=sum(float(b.double().square().sum())for b in ref)
        return math.sqrt(n/max(d,1e-60))
    ids=[0,1,17,31,63,95,96,127,159,175,190,191]
    a,ga=gradients(ids,True);b,gb=gradients(ids)
    forward=float((a-b).norm()/a.norm().clamp_min(1e-30));gradient=relative(gb,ga)
    allids=list(range(192));a,gfull=gradients(allids,True);b,gall=gradients(allids)
    all_forward=float((a-b).norm()/a.norm().clamp_min(1e-30));all_gradient=relative(gall,gfull)
    averaged=[torch.zeros_like(p)for p in params]
    for offset in range(4):
        ids2=list(range(offset,192,4));_,g=gradients(ids2)
        averaged=[x+y*.25 for x,y in zip(averaged,g)]
    partition=relative(averaged,gfull)
    if max(forward,all_forward)>1e-4 or max(gradient,all_gradient,partition)>1e-3:
        raise AssertionError((forward,gradient,all_forward,all_gradient,partition))
    timing={}
    for k in [192,32,16]:
        ids2=np.sort(np.random.default_rng(812+k).choice(192,k,replace=False)).tolist()
        gradients(ids2,original=(k==192))
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        for _ in range(3):gradients(ids2,original=(k==192))
        torch.cuda.synchronize()
        timing["small_"+str(k)]=dict(seconds_per_forward_backward=(time.monotonic()-start)/3,
            peak_cuda_bytes=torch.cuda.max_memory_allocated(),position_batch=len(rows),BS_batch=len(bs),
            note="no optimizer update; same trained parent/small positionBS shape")
    # Actual formal position/BS shape is measured for K16 only. The old full192
    # graph at this shape may exceed current external-load headroom.
    bench_full=torch.tensor(data.read_fit(allrows),device="cuda:0",dtype=torch.complex64)
    bench_power=bench_full[:,bp*128:bp*128+128,up*2:up*2+2].abs().square().mean((1,2,3),dtype=torch.float64).clamp_min(cfg["power_floor"])
    bench_truth=bench_full[:,torch.tensor(allbs,device="cuda:0")+bp*128,up*2:up*2+2]
    ids3=np.sort(np.random.default_rng(828).choice(192,16,replace=False)).tolist()
    def actual_batch():
        value=FrequencyRenderer(data,allrows,key,ids3)(model,allbs)
        target=bench_truth.index_select(-1,torch.tensor(ids3,device="cuda:0"))
        loss=.5*((value-target).abs().square().mean((1,2,3))/bench_power).mean()
        return torch.autograd.grad(loss,params,allow_unused=True)
    actual_batch();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    for _ in range(3):actual_batch()
    torch.cuda.synchronize()
    timing["actual_16"]=dict(seconds_per_forward_backward=(time.monotonic()-start)/3,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),position_batch=len(allrows),BS_batch=len(allbs),
        add_full_FIT_GPU_cache_bytes_for_formal=len(data.fit_ids)*256*4*192*8,
        note="actual4x16 graph, no optimizer update, small label cache only")
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():full_prediction=M.predict_full(model,data,allrows[:1],cfg["inference_BS_chunk"])
    if full_prediction.shape!=(1,256,4,192):raise AssertionError("Evaluation output is not full192")
    timing["full192_evaluation"]=dict(peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        shape=list(full_prediction.shape),inference_BS_chunk=cfg["inference_BS_chunk"],
        add_full_FIT_GPU_cache_bytes_for_training_validation=len(data.fit_ids)*256*4*192*8)
    d=dict(status="passed",source_identity=source,device=device,parent_sha256=C.sha256(IMMUTABLE),parent_step=ck["step"],
       field=key,rows=rows,BS_ids=bs,noncontiguous_ids=ids,selected_forward_relative_L2=forward,
       selected_unclipped_gradient_relative_L2=gradient,all192_forward_relative_L2=all_forward,
       all192_unclipped_gradient_relative_L2=all_gradient,partition_average_vs_full_gradient_relative_L2=partition,
       timing=timing,optimizer_updates=0,no_clipping_or_Adam_in_QA=True,physical_rays=324,samples=32,
       output_always_complete192_for_evaluation=True,shared_spatial_trunk_not_frequency_pruned=True,
       no_outer_truth_read=True,training_labels_used=data.info()["fit_rows_actually_read"])
    C.atomic_json(REPORT/"gpu_qa.json",d);print(json.dumps(d),flush=True)

def execute(folds,cfg,source,device,proof):
    global STAGE_ORIGIN
    if cfg.get("formal_frozen")is not True:raise RuntimeError("Root must freeze new frequency contract before training")
    completed=[]
    for fold in folds:
        if fold in [0,1]:raise ValueError("Completed folds0/1 are readonly references, not rerun targets")
        STAGE_ORIGIN=None
        screen=enrich_stage(fold,"screen",M.stage(fold,"screen",cfg,source))
        STAGE_ORIGIN=dict(kind="fresh_refit_following_frozen_screen_selection",
            fresh_parameter_initialization=True,screen_selected_step=screen["selected_step"],
            screen_checkpoint=screen["checkpoint"],screen_checkpoint_sha256=screen["checkpoint_sha256"],
            screen_current_migration=screen.get("migration"),screen_selected_migration=screen.get("selected_migration"))
        refit=enrich_stage(fold,"refit",M.stage(fold,"refit",cfg,source,screen))
        STAGE_ORIGIN=None
        manifest=publish_prediction(fold,cfg,source,screen,refit,device,proof)
        completed.append(dict(fold=fold,manifest=PATHS.relative(ART/f"outer{fold}/prediction/manifest.json"),sha256=manifest["sha256"]))
        C.atomic_json(REPORT/("completed_"+"_".join(map(str,folds))+".json"),dict(status="running",completed=completed))
    C.atomic_json(REPORT/("completed_"+"_".join(map(str,folds))+".json"),dict(status="complete",completed=completed,outer_truth_read=False))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",type=Path,required=True)
    ap.add_argument("--gpu-uuid",default="GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0",choices=C.ALLOWED)
    ap.add_argument("--folds",default="2,3")
    ap.add_argument("--mode",choices=["cpu-check","gpu-qa","migrate","run"],required=True);args=ap.parse_args()
    REPORT.mkdir(parents=True,exist_ok=True);ART.mkdir(parents=True,exist_ok=True)
    m,cfg,source,proof=configure(args.config,args.gpu_uuid)
    if args.mode in["cpu-check","migrate"]:
        if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise RuntimeError("CPU mode needs empty CUDA mask")
        if args.mode=="cpu-check":return cpu_check(cfg,source)
        if not cfg.get("formal_frozen"):raise RuntimeError("Freeze config before checkpoint metadata migration")
        return migrate(source,cfg)
    if os.environ.get("CUDA_VISIBLE_DEVICES")!=args.gpu_uuid:raise RuntimeError("Full UUID CUDA mask mismatch")
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=C.device_guard(args.gpu_uuid)
    if args.mode=="gpu-qa":return gpu_qa(cfg,source,device)
    free,_=torch.cuda.mem_get_info()
    if free<cfg["min_free_GiB"]*(1<<30):raise RuntimeError("Registered memory margin unavailable")
    def stop(_sig,_frame):m.STOP=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    folds=list(map(int,args.folds.split(",")))
    import fcntl
    lock=REPORT/("queue_"+"_".join(map(str,folds))+".lock")
    with lock.open("a+")as f:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        C.atomic_json(REPORT/("device_"+"_".join(map(str,folds))+".json"),dict(device,source_identity=source))
        return execute(folds,cfg,source,device,proof)

if __name__=="__main__":
    try:main()
    except BaseException:
        REPORT.mkdir(parents=True,exist_ok=True)
        with(REPORT/"failure.log").open("a")as f:f.write(traceback.format_exc()+"\n")
        raise

