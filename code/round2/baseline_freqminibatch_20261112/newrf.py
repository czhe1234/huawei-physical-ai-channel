"""Frequency-minibatch continuation of frozen formal NeWRF; full192 inference."""
from __future__ import annotations
import argparse,copy,json,hashlib,math,time,traceback,signal
from pathlib import Path
import numpy as np
import torch
from project_paths import PATHS
from round2.baseline_formal_20261111 import newrf as B
C=B.C
ROOT=C.REPORT/"freqminibatch_20261112";R=ROOT/"newrf"
A=C.ART/"freqminibatch_20261112/newrf"
PARENT=A/"immutable_pause_checkpoint.pt"
PARENT_SHA="74a8361a7ecfee31480153d566b728884b960c72af6d94b261dabed9f45c57d7"
OLD_R=C.REPORT/"formal_20261111/newrf"
OLD_A=C.ART/"formal_20261111/newrf"
ORIG_SRC=B.source_identity;ORIG_LOAD=B.load_config;ORIG_SAMPLER=B.PermutationSampler
ORIG_UPDATE=B.update;ORIG_JSON=C.atomic_json
CONTEXT=dict(outer=4,role="screen",cfg=None)
def load_config(path):
    cfg=ORIG_LOAD(path)
    if cfg["frequency_minibatch"]not in(8,16)or cfg["full_frequency_count"]!=192:raise ValueError("Frozen8/16-tone sampling only")
    parent=B.read_json(R/"parent_method_config.json")
    keys=["seed","batch_positions","bs_block","n_samples","n_importance","frequency_chunk",
      "stage_min_updates","stage_max_updates","learning_rates","validation_every","health_every",
      "native_patience","minimum_validation_improvement","training_power_floor_FIT_RMS_squared"]
    if any(cfg[k]!=parent[k]for k in keys):raise RuntimeError("Numerical protocol changed beyond tone sampling")
    return cfg
def source_identity(path):
    d=ORIG_SRC(path)
    for p in[Path(__file__),Path(__file__).with_name("newrf_qa.py"),PARENT,R/"pause_manifest.json",R/"parent_method_config.json"]:
        d[PATHS.relative(p)]=C.sha256(p)
    if CONTEXT.get("mode")!="qa"and(R/"QA.json").exists():d[PATHS.relative(R/"QA.json")]=C.sha256(R/"QA.json")
    return d
class ToneSampler(ORIG_SAMPLER):
    def __init__(self,data,batch,seed):
        super().__init__(data,batch,seed)
        self.field=int(seed)-int(CONTEXT["cfg"]["seed"])-1000*int(data.outer)
        self.tone_rng=np.random.default_rng(int(seed)+100003+(7 if data.role=="refit"else 0))
        self.tone_selected_updates=np.zeros(192,np.int64);self.tone_scalar_labels=np.zeros(192,np.int64)
        self.legacy_fullband_updates=0;self.minibatch_updates=0
    def draw_tones(self,k):return np.sort(self.tone_rng.choice(192,k,replace=False)).astype(np.int64)
    def commit_tones(self,ff,n):
        self.tone_selected_updates[ff]+=1;self.tone_scalar_labels[ff]+=int(n)*8;self.minibatch_updates+=1
    def info(self):
        return dict(field=self.field,frequency_count=192,sampled_frequency_count=CONTEXT["cfg"]["frequency_minibatch"],
          legacy_fullband_updates=self.legacy_fullband_updates,minibatch_updates=self.minibatch_updates,
          tone_selected_updates=self.tone_selected_updates.tolist(),tone_scalar_gradient_labels=self.tone_scalar_labels.tolist(),
          tones_seen=int(np.count_nonzero(self.tone_selected_updates)),tone_RNG_independent_of_position_BS=True)
    def snapshot(self):
        d=super().snapshot();d.update(tone_rng=copy.deepcopy(self.tone_rng.bit_generator.state),
          tone_selected_updates=self.tone_selected_updates.copy(),tone_scalar_labels=self.tone_scalar_labels.copy(),
          legacy_fullband_updates=self.legacy_fullband_updates,minibatch_updates=self.minibatch_updates)
        rr=R/f"outer{CONTEXT['outer']}"/CONTEXT["role"];rr.mkdir(parents=True,exist_ok=True)
        ORIG_JSON(rr/f"tone_field{self.field}.json",self.info())
        return d
    def restore(self,d):
        super().restore(d)
        if "tone_rng"in d:
            self.tone_rng.bit_generator.state=copy.deepcopy(d["tone_rng"])
            self.tone_selected_updates=d["tone_selected_updates"].copy();self.tone_scalar_labels=d["tone_scalar_labels"].copy()
            self.legacy_fullband_updates=d["legacy_fullband_updates"];self.minibatch_updates=d["minibatch_updates"]
        else:
            n=int(self.seen.sum());assert n%self.batch==0
            self.legacy_fullband_updates=n//self.batch
            self.tone_selected_updates[:]=self.legacy_fullband_updates;self.tone_scalar_labels[:]=n*8
def denominator(full_labels,data,cfg):
    energy=float(np.sum(abs(np.asarray(full_labels,np.complex128))**2))
    floor=float(full_labels.size*cfg["training_power_floor_FIT_RMS_squared"]*data.scale**2)
    return max(energy,floor),energy,floor
def sample_loss(model,data,cache,rows,blocks,field,ff,full_labels,cfg,full_render_mask=False):
    """No clipping: unbiased selected-frequency objective conditional on row/BS batch."""
    den,energy,floor=denominator(full_labels,data,cfg);factor=192/len(ff);terms=[]
    for i,(row,block)in enumerate(zip(rows,blocks)):
        bs=np.arange(int(block)*8,int(block)*8+8,dtype=np.int64)
        frequencies=B.FF if full_render_mask else ff
        h,hc=B.cached_block(model,data,int(row),field,bs,frequencies,cache)
        if full_render_mask:
            ix=torch.tensor(ff,device=h.device);h=h.index_select(-1,ix);hc=hc.index_select(-1,ix)
        y=torch.tensor(np.take(full_labels[i],ff,axis=-1),device=h.device,dtype=torch.complex64)[None]
        # The label is selected on the actual frequency axis before loss.
        terms.append((.9*(h-y).abs().square().sum(dtype=torch.float64)+
                      .1*(hc-y).abs().square().sum(dtype=torch.float64))*factor/den)
    return sum(terms),dict(D_full=den,true_full_energy=energy,floor_full=floor,scale=192/len(ff))
def update(model,data,cache,sampler,field,opt,cfg):
    rows,blocks=sampler.draw();full=B.block_labels(data,rows,blocks,field)
    ff=sampler.draw_tones(cfg["frequency_minibatch"]);den,energy,floor=denominator(full,data,cfg)
    opt.zero_grad(set_to_none=True);model.train();total=lf_total=lc_total=0.;factor=192/len(ff)
    for i,(row,block)in enumerate(zip(rows,blocks)):
        bs=np.arange(int(block)*8,int(block)*8+8,dtype=np.int64)
        # Cut actual Hz BEFORE _scalar_links. No192-tone render followed by slicing.
        h,hc=B.cached_block(model,data,int(row),field,bs,ff,cache)
        y=torch.tensor(np.take(full[i],ff,axis=-1),device=h.device,dtype=torch.complex64)[None]
        lf=factor*(h-y).abs().square().sum(dtype=torch.float64)/den
        lc=factor*(hc-y).abs().square().sum(dtype=torch.float64)/den
        loss=.9*lf+.1*lc
        if not torch.isfinite(loss):raise FloatingPointError("Nonfinite tone-minibatch loss")
        loss.backward();total+=float(loss.detach());lf_total+=float(lf.detach());lc_total+=float(lc.detach())
    active=[p for p in model.parameters()if p.requires_grad]
    grad=float(torch.nn.utils.clip_grad_norm_(active,5.,error_if_nonfinite=True));opt.step()
    sampler.commit(rows,blocks);sampler.commit_tones(ff,len(rows))
    return dict(loss=total,fine_loss=lf_total,coarse_loss=lc_total,gradient_norm=grad,
      denominator_clamped=energy<floor,true_batch_energy=energy,denominator=den,
      positions=len(rows),scalar_complex_labels=len(rows)*8*len(ff),full_denominator_scalar_count=full.size,
      rows=rows.tolist(),blocks=blocks.tolist(),frequency_ids=ff.tolist(),frequency_weight=factor)
def annotate_canonical(path,value):
    target=A/"outer4/prediction/manifest.json"
    if Path(path).resolve()==target.resolve():
        value=copy.deepcopy(value);value["metadata"]["frequency_minibatch"]=dict(
          count=CONTEXT["cfg"]["frequency_minibatch"],full_inference_tones=192,actual_Hz_cut_before_render=True,
          denominator="original full192F batch energy and full-count FIT floor",
          estimator="(192/K)*selected squared error / D_full",SGD_trajectory_equivalence_claimed=False,
          parent_checkpoint_sha256=PARENT_SHA,parent_used_optimizer_updates=1499,refit_initialization="fresh seed42",refit_weights_inherited_from_pause=False,
          reused_formal_folds=[0,1,2,3])
    return ORIG_JSON(path,value)
def install(cfg):
    CONTEXT["cfg"]=cfg
    B.R=R;B.A=A;B.load_config=load_config;B.source_identity=source_identity
    B.PermutationSampler=ToneSampler;B.update=update;C.atomic_json=annotate_canonical
def uninstall():
    B.R=OLD_R;B.A=OLD_A;B.load_config=ORIG_LOAD;B.source_identity=ORIG_SRC
    B.PermutationSampler=ORIG_SAMPLER;B.update=ORIG_UPDATE;C.atomic_json=ORIG_JSON
def verify_qa(path):
    cfg=load_config(path);q=B.read_json(R/"QA.json")
    if q["status"]!="passed"or cfg["frequency_minibatch"]!=q["recommended_K_by_latency_only"]:raise RuntimeError("FixedK must match latency-only QA")
    for p,h in q["source_identity"].items():
        if C.sha256(PATHS.root/p)!=h:raise RuntimeError(("QA source/config changed",p))
    if q["qa_source_sha256"]!=C.sha256(Path(__file__).with_name("newrf_qa.py")):raise RuntimeError("QA helper changed")
    return q

def migrate(path):
    cfg=load_config(path);verify_qa(path);assert C.sha256(PARENT)==PARENT_SHA
    old=torch.load(PARENT,map_location="cpu",weights_only=False);before=copy.deepcopy(old)
    assert old["outer"]==4 and old["role"]=="screen"
    for p,h in old["source_identity"].items():assert C.sha256(PATHS.root/p)==h,p
    data=B.prepare_data(4,"screen");CONTEXT.update(outer=4,role="screen")
    sampler=ToneSampler(data,cfg["batch_positions"],cfg["seed"]+4000);sampler.restore(old["sampler"])
    old["sampler"]=sampler.snapshot();old["source_identity"]=source_identity(path)
    old["method_config_sha256"]=B.content_hash(cfg)
    old["frequency_migration"]=dict(parent_sha256=PARENT_SHA,used_updates=1499,
      legacy_field=0,legacy_stage="warm",legacy_phase_step=749,position_BS_sampler_unchanged=True)
    for key in["model","coarse_optimizer","fine_optimizer","torch_rng","cuda_rng","scheduler","state","fit_read_counts","validation_read_counts"]:
        assert fingerprint(old[key])==fingerprint(before[key]),key
    for key in before["sampler"]:assert fingerprint(old["sampler"][key])==fingerprint(before["sampler"][key]),key
    out=A/"outer4/screen/latest.pt"
    if out.exists():raise FileExistsError(out)
    C.atomic_torch_save(out,old)
    reloaded=torch.load(out,map_location="cpu",weights_only=False)
    stable_keys=["model","coarse_optimizer","fine_optimizer","torch_rng","cuda_rng","scheduler","state","fit_read_counts","validation_read_counts"]
    stable={key:fingerprint(before[key])for key in stable_keys}
    for key in stable_keys:assert fingerprint(reloaded[key])==stable[key],key
    for key in before["sampler"]:assert fingerprint(reloaded["sampler"][key])==fingerprint(before["sampler"][key]),key
    report=dict(status="migrated_no_optimizer_update",parent_sha256=PARENT_SHA,
      migrated_sha256=C.sha256(out),migrated_checkpoint=PATHS.relative(out),model_optimizer_RNG_scheduler_LR_exact=True,
      position_BS_sampler_exact=True,new_tone_rng_independent=True,used_updates=1499,phase="warm",phase_step=749,
      source_identity=old["source_identity"],saved_and_reloaded_stable_hashes=stable,
      sampler_original_subkeys_exact_after_reload=True,QA_sha256=C.sha256(R/"QA.json"),frozen_config_sha256=C.sha256(path))
    ORIG_JSON(R/"migration.json",report);print(json.dumps(report))
def fingerprint(x):
    h=hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v,np.ndarray):h.update(str(v.dtype).encode());h.update(str(v.shape).encode());h.update(v.tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=repr):h.update(repr(k).encode());visit(v[k])
        elif isinstance(v,(tuple,list)):
            for y in v:visit(y)
        else:h.update(repr(v).encode())
    visit(x);return h.hexdigest()
def enrich_role(outer,role,result):
    for field,rec in result["fields"].items():
        p=R/f"outer{outer}"/role/f"tone_field{field}.json";d=B.read_json(p)
        assert d["tones_seen"]==192
        rec["frequency_exposure"]=d
    result["frequency_minibatch"]=CONTEXT["cfg"]["frequency_minibatch"]
    result["full_inference_tones"]=192;result["migration_parent_sha256"]=PARENT_SHA if role=="screen"else None
    result["initialization"]="inherited screen checkpoint1499"if role=="screen"else"fresh seed42; only frozen screen stage counts transferred"
    result["weights_inherited_from_pause"]=role=="screen"
    result["scalar_complex_labels_per_update"]=CONTEXT["cfg"]["batch_positions"]*8*CONTEXT["cfg"]["frequency_minibatch"]
    result["full_denominator_scalar_labels_per_update"]=CONTEXT["cfg"]["batch_positions"]*8*192
    ORIG_JSON(R/f"outer{outer}"/role/"result.json",result);return result
def qa(path,uuid):
    from . import newrf_qa
    return newrf_qa.run(path,uuid,__import__("sys").modules[__name__])
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",type=Path,required=True)
    ap.add_argument("--mode",choices=["cpu-check","qa","migrate","run"],required=True)
    ap.add_argument("--gpu-uuid",default=B.DEFAULT_GPU);args=ap.parse_args()
    R.mkdir(parents=True,exist_ok=True);A.mkdir(parents=True,exist_ok=True)
    cfg=load_config(args.config);CONTEXT["mode"]=args.mode;install(cfg)
    try:
        if args.mode=="cpu-check":
            old=torch.load(PARENT,map_location="cpu",weights_only=False);data=B.prepare_data(4,"screen")
            s=ToneSampler(data,cfg["batch_positions"],cfg["seed"]+4000);s.restore(old["sampler"])
            target=ORIG_SAMPLER(data,cfg["batch_positions"],cfg["seed"]+4000);target.restore(old["sampler"])
            assert all(np.array_equal(x,y)for x,y in zip(s.draw(),target.draw()))
            assert C.sha256(PARENT)==PARENT_SHA and not torch.cuda.is_initialized()
            ORIG_JSON(R/"CPU_check.json",dict(status="passed",position_BS_continuation_exact=True,CUDA_initialized=False,source_identity=source_identity(args.config)));return
        if args.mode=="qa":qa(args.config,args.gpu_uuid);return
        if args.mode=="migrate":migrate(args.config);return
        if not(R/"migration.json").exists():raise RuntimeError("Audited migration required")
        verify_qa(args.config);migration=B.read_json(R/"migration.json")
        if migration["frozen_config_sha256"]!=C.sha256(args.config)or migration["QA_sha256"]!=C.sha256(R/"QA.json"):raise RuntimeError("Frozen migration/QA config binding differs")
        signal.signal(signal.SIGTERM,B.stop_signal);signal.signal(signal.SIGINT,B.stop_signal)
        with(A/"outer4.lock").open("a+")as lock:
            B.fcntl.flock(lock.fileno(),B.fcntl.LOCK_EX|B.fcntl.LOCK_NB)
            for role in["screen","refit"]:
                CONTEXT.update(outer=4,role=role)
                result=B.run_role(4,role,args.config,args.gpu_uuid);enrich_role(4,role,result)
            B.predict_outer(4,args.config,args.gpu_uuid)
    finally:uninstall()
if __name__=="__main__":
    try:main()
    except BaseException:
        R.mkdir(parents=True,exist_ok=True)
        with(R/"failure.log").open("a")as f:f.write(traceback.format_exc()+"\n")
        raise
