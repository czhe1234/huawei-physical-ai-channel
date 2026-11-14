"""One-field diagnostic ladder; frozen NeRF2 renderer/native complex MSE.

This is not a complete256x4 channel result: one fixedUE0 axis block is
trained. Scalar plus two independent axis forks, then all128BS/192tones at fixedUE0.
No validation, outer labels, official scoring, or V16 access is used.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,math,random,signal,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from project_paths import PATHS
from . import common as C,nerf2_runner_v2 as P,nerf2_runner_v3 as V,field_fast as FF

ART=C.ART/"fit_repair_paper_v2_20261101/nerf2"
REPORT=C.REPORT/"fit_repair_paper_v2_20261101/nerf2"
CONFIG=dict(schema="nerf2-minimal-complete-field-ladder-v2",seed=42,outer=0,
 site=0,BS_pol=0,UE_pol=0,max_updates=3000,max_seconds=600,
 stages=[dict(name="scalar",maximum_updates=500,monitor_every=25,minimum_updates=50,nmse_max=.01,energy_min=.8,energy_max=1.2),
         dict(name="F_only",maximum_updates=700,monitor_every=48,minimum_updates=48,nmse_max=.1,energy_min=.7,energy_max=1.3),
         dict(name="BS_only",maximum_updates=700,monitor_every=32,minimum_updates=32,nmse_max=.1,energy_min=.7,energy_max=1.3),
         dict(name="joint_BS_F",maximum_updates=1100,monitor_every=192,minimum_updates=768,nmse_max=.1,energy_min=.7,energy_max=1.3,per_UE_nmse_max=.2)],
 bs_per_batch=8,frequencies_per_batch=4,
 optimizer="unchanged nerf2_runner_v2.optimizer, constant learning rates",
 objective="0.5*mean(abs(predH-targetH)^2)",
 architecture="unchanged v3 depth8 width256 rays324 samples32 far810; parent FIT RMS",
 branch_start="F-only and BS-only fork scalar model/Adam/RNG; joint starts F-only checkpoint",
 isolated_axes_hold_UE_port_fixed=True,
 validation_used=False,outer_truth_read=False,official_scoring=False)



def paths():
 ART.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)


def source_identity(panel):
 source={**V.identity(),PATHS.relative(Path(__file__)):C.sha256(Path(__file__)),
         PATHS.relative(panel):C.sha256(panel),PATHS.relative(panel.parent/"contract.json"):C.sha256(panel.parent/"contract.json")}
 for p in [PATHS.code_root/"round2/paper_baseline_adapters_v1/field_models.py",
           PATHS.code_root/"round2/paper_baseline_adapters_v1/contract.py",
           PATHS.root/"plus-pro/code/measured/full_h_adapters_v1/nerf2.py",
           PATHS.root/"plus-pro/code/third_party/paper_baselines/nerf2/model.py",
           PATHS.root/"plus-pro/code/third_party/paper_baselines/nerf2/renderer.py"]:
  source[PATHS.relative(p)]=C.sha256(p)
 return source


def check_sources(source):
 for key,value in source.items():
  p=PATHS.root/key if "/"in key else Path(__file__).parent/key
  if C.sha256(p)!=value:raise RuntimeError("Numerical source/panel changed: "+key)


def load_data(panel_path):
 data=C.FoldData(0,"screen","nerf2")
 d=json.loads((C.ART/"fit_scales/outer0_screen.json").read_text())
 want=dict(schema="FIT-channel-scalar-RMS-v1",input_identity=data.inputs.identity,
   fit_ids_sha256=C.digest_ids(data.fit_ids),raw=data.raw_identity,outer=0,role="screen")
 if d["contract"]!=want:raise RuntimeError("Parent FIT RMS provenance mismatch")
 data._scale_info=d
 panel=json.loads(panel_path.read_text())
 if panel["input_identity"]!=data.inputs.identity or panel["parent_fit_ids_sha256"]!=C.digest_ids(data.fit_ids):
  raise RuntimeError("Panel/parent FIT mismatch")
 if panel["parent_FIT_RMS"]!=data.scale:raise RuntimeError("Parent RMS changed")
 ids=np.asarray(panel["train_ids"],np.int64)
 if not set(ids).issubset(set(data.fit_ids)):raise PermissionError("Panel includes non-FIT row")
 chosen=ids[data.sites(ids)==CONFIG["site"]]
 if len(chosen)!=1:raise ValueError("Exactly one site0 FIT unit row required")
 row=int(chosen[0])
 full=data.read_fit(np.array([row]))
 target=full[0,:128,:2]
 if not np.isfinite(target).all() or not np.any(np.abs(target)>0):raise ValueError("Nonzero finite field required")
 scalar=(panel["primary_BS_spatial_id"],panel["primary_UE_port"],panel["primary_frequency_id"])
 selection="publisher-independent fixed FIT-only scalar selected in panel before predictions"
 if abs(target[scalar])<=0:raise ValueError("Panel scalar must be nonzero")
 info=dict(panel=panel,row_id=row,site=CONFIG["site"],field=[0,0,0],
  scalar_BS_spatial=scalar[0],scalar_UE_vertical=scalar[1],scalar_frequency_id=scalar[2],
  scalar_selection=selection,scalar_target_normalized_real=float(target[scalar].real),
  scalar_target_normalized_imag=float(target[scalar].imag),parent_FIT_RMS=data.scale,
  axis_block_shape=[1,128,1,192],native_field_shape=[1,128,2,192],complete_H_shape=[1,256,4,192],
  complete_H_has_other_untrained_polarization_blocks=True,
  input_identity=data.inputs.identity,config=CONFIG,outer_truth_read=False)
 return data,target,info


class Sampler:
 def __init__(self,info):
  self.info=info;self.rng=np.random.default_rng(CONFIG["seed"]);self.orders=None;self.stage=None;self.cycle=None
 def task(self,stage,index):
  scalar=dict(mm=[self.info["scalar_BS_spatial"]],uu=[self.info["scalar_UE_vertical"]],ff=[self.info["scalar_frequency_id"]])
  if stage==0:return scalar
  n={1:48,2:16,3:768}[stage];cycle,local=divmod(index,n)
  if self.stage!=stage or self.cycle!=cycle:
   self.orders=dict(mm=self.rng.permutation(128).reshape(16,8),ff=self.rng.permutation(192).reshape(48,4))
   self.stage=stage;self.cycle=cycle
  if stage==1:return dict(mm=scalar["mm"],uu=scalar["uu"],ff=np.sort(self.orders["ff"][local]).tolist())
  if stage==2:return dict(mm=np.sort(self.orders["mm"][local]).tolist(),uu=scalar["uu"],ff=scalar["ff"])
  fi=local%48
  return dict(mm=np.sort(self.orders["mm"][(local//48+fi)%16]).tolist(),uu=scalar["uu"],
              ff=np.sort(self.orders["ff"][fi]).tolist())
 def state_dict(self):return dict(rng_state=copy.deepcopy(self.rng.bit_generator.state),orders=copy.deepcopy(self.orders),stage=self.stage,cycle=self.cycle)
 def load_state_dict(self,s):
  self.rng.bit_generator.state=s["rng_state"];self.orders=s["orders"];self.stage=s["stage"];self.cycle=s["cycle"]

def forward(model,data,info,task):
 p=torch.tensor(data.positions(np.array([info["row_id"]])),device="cuda:0",dtype=torch.float64)
 site=torch.tensor([CONFIG["site"]],device="cuda:0",dtype=torch.int64)
 world=data.geometry.ue_world_positions(p,site)
 rx=world[:,torch.tensor(task["uu"],device="cuda:0")]
 tx=model.bound_bs_ports[0,0].reshape(128,3)[torch.tensor(task["mm"],device="cuda:0")]
 f=torch.tensor(data.frequencies(task["ff"]),device="cuda:0",dtype=torch.float64)
 return model.fields["s0_b0_u0"](tx,rx,f).permute(1,0,2,3)*model.log_output_gain.exp()


def target_slice(target,task):
 mm=torch.tensor(task["mm"],device="cuda:0");uu=torch.tensor(task["uu"],device="cuda:0");ff=torch.tensor(task["ff"],device="cuda:0")
 return target.index_select(0,mm).index_select(1,uu).index_select(2,ff)[None]


def statistics(value,target):
 te=float(target.abs().square().sum(dtype=torch.float64));pe=float(value.abs().square().sum(dtype=torch.float64))
 se=float((value-target).abs().square().sum(dtype=torch.float64));cross=float((value.conj()*target).real.sum(dtype=torch.float64))
 return dict(nmse=se/te if te else None,energy_ratio=pe/te if te else None,
  signed_correlation=cross/math.sqrt(pe*te)if pe*te else None,squared_error=se,truth_energy=te,
  prediction_energy=pe,count=target.numel(),real_mse=float((value.real-target.real).square().mean()),
  imag_mse=float((value.imag-target.imag).square().mean()))


def gate_task(stage,info):
 scalar=dict(mm=[info["scalar_BS_spatial"]],uu=[info["scalar_UE_vertical"]],ff=[info["scalar_frequency_id"]])
 if stage==0:return scalar
 if stage==1:return dict(mm=scalar["mm"],uu=scalar["uu"],ff=list(range(192)))
 if stage==2:return dict(mm=list(range(128)),uu=scalar["uu"],ff=scalar["ff"])
 return dict(mm=list(range(128)),uu=scalar["uu"],ff=list(range(192)))

@torch.no_grad()
def evaluate(model,data,target,info,stage):
 model.eval();task=gate_task(stage,info);value=forward(model,data,info,task)
 truth=target_slice(target,task);st=statistics(value,truth)
 st["per_UE"]=[dict(UE_vertical=int(u),**statistics(value[:,:,i:i+1],truth[:,:,i:i+1]))for i,u in enumerate(task["uu"])]
 return st

def passed(st,stage,updates):
 cfg=CONFIG["stages"][stage]
 ok=updates>=cfg["minimum_updates"]and st["nmse"]is not None and st["nmse"]<=cfg["nmse_max"]and cfg["energy_min"]<=st["energy_ratio"]<=cfg["energy_max"]
 if "per_UE_nmse_max"in cfg:ok=ok and all(x["nmse"]is not None and x["nmse"]<=cfg["per_UE_nmse_max"]for x in st["per_UE"])
 return bool(ok)

def baseline(model):
 return {n:p.detach().cpu().clone()for n,p in model.named_parameters()if n.startswith("fields.s0_b0_u0.")or n=="log_output_gain"}


def parameter_changes(model,before):
 result={}
 for n,p in model.named_parameters():
  if n not in before:continue
  previous=before[n];now=p.detach().cpu()
  result[n]=dict(delta_norm=float((now-previous).norm()),initial_norm=float(previous.norm()),
   relative_delta=float((now-previous).norm()/previous.norm().clamp_min(1e-30)))
 return result


def gradient_details(model,value,target):
 expected=(value.detach()-target)/target.numel()
 relative=float((value.grad-expected).norm()/expected.norm().clamp_min(1e-30))
 gain_analytic=float(((value.detach()-target).conj()*value.detach()).real.mean())
 gain_actual=float(model.log_output_gain.grad)
 if relative>1e-5 or abs(gain_actual-gain_analytic)>1e-5*max(abs(gain_analytic),1e-8):
  raise RuntimeError("Native complex MSE gradient check failed")
 detail=dict(prediction_gradient_relative_error=relative,gain_gradient_analytic=gain_analytic,
  gain_gradient_actual=gain_actual,gain_gradient_descent_direction="shrink"if gain_actual>0 else"increase",
  branches={})
 for n,p in model.named_parameters():
  if p.grad is None or(not n.startswith("fields.s0_b0_u0.")and n!="log_output_gain"):continue
  if any(x in n for x in ["signal_output","attenuation_output","feature_layer","log_output_gain"]):
   gn=float(p.grad.norm());pn=float(p.norm());wd=0. if n=="log_output_gain"else 5e-5
   item=dict(gradient_norm=gn,parameter_norm=pn,weight_decay_gradient_norm=wd*pn,
             data_gradient_over_weight_decay=gn/(wd*pn)if wd*pn else None)
   if n.endswith("output.weight")or n.endswith("output.bias"):
    item["gradient_norm_by_head"]=[float(p.grad[i].norm())for i in range(len(p.grad))]
   detail["branches"][n]=item
 return detail


@torch.no_grad()
def scalar_branches(model,data,info):
 core=model.fields["s0_b0_u0"];saved=dict(attenuation=[],signal=[])
 handles=[core.field.attenuation_output.register_forward_hook(lambda m,i,o:saved["attenuation"].append(o.detach())),
          core.field.signal_output.register_forward_hook(lambda m,i,o:saved["signal"].append(o.detach()))]
 task=gate_task(0,info);model.eval()
 try:value=forward(model,data,info,task)
 finally:
  for h in handles:h.remove()
 if len(saved["attenuation"])!=1 or len(saved["signal"])!=1:raise RuntimeError("Unexpected scalar renderer chunks")
 a=saved["attenuation"][0];sig=saved["signal"][0]
 shape=(324,1,32);seg=torch.full((32,),810/32,device="cuda:0",dtype=torch.float64)
 frequency=torch.tensor(data.frequencies(task["ff"]),device="cuda:0",dtype=torch.float64)
 trans,carrier=FF._shared_transport(a,shape,seg,frequency,-1,0.)
 raw=sig.reshape(1,*shape,2);amp=F.leaky_relu(raw[...,0]).abs()
 phase=torch.sigmoid(raw[...,1])*2*math.pi-math.pi
 emission_phase=torch.complex(phase.cos(),phase.sin()).to(torch.complex128)
 contributions=(amp*trans[None]).double()*carrier[None]*emission_phase
 ray=contributions.sum(-1);total=ray.sum(1)
 denominator=float(contributions.abs().sum())
 def summarize(x):
  z=x.double().reshape(-1)
  return dict(mean=float(z.mean()),minimum=float(z.min()),maximum=float(z.max()),
              q05=float(torch.quantile(z,.05)),median=float(torch.quantile(z,.5)),q95=float(torch.quantile(z,.95)))
 reconstructed=total.to(torch.complex64)*model.log_output_gain.exp()
 optical_depth=F.leaky_relu(a[:,0]).abs().reshape(shape).sum(-1)*(810/32)
 return dict(scalar_task=task,gain=float(model.log_output_gain.exp()),
  emitted_raw_amplitude=summarize(raw[...,0]),negative_emission_raw_fraction=float((raw[...,0]<0).double().mean()),
  emitted_amplitude=summarize(amp),optical_depth=summarize(optical_depth),transmittance=summarize(trans),
  per_sample_absolute_sum=denominator,coherent_sum_abs=float(total.abs().sum()),
  coherent_over_incoherent_amplitude=float(total.abs().sum())/max(denominator,1e-30),
  reconstruction_relative_error=float((reconstructed-value.reshape_as(reconstructed)).abs().norm()/value.abs().norm().clamp_min(1e-30)),
  normalized_prediction_real=float(value.real.reshape(-1)[0]),normalized_prediction_imag=float(value.imag.reshape(-1)[0]),
  note="one FIT scalar; all324x32 learned samples kept; no parameter update")


def emit(event):
 with(REPORT/"raw_log.jsonl").open("a")as f:f.write(json.dumps(event,allow_nan=False)+"\n")
 print(json.dumps(event,allow_nan=False),flush=True)


def pack(model,opt,sampler,stage,stage_step,global_step,info,source,history,elapsed,before):
 return dict(model={k:v.detach().cpu().clone()for k,v in model.state_dict().items()},
  optimizer=opt.state_dict(),sampler=sampler.state_dict(),stage=stage,stage_step=stage_step,
  global_step=global_step,info=info,source_identity=source,history=history,elapsed_seconds=elapsed,
  stage_parameter_baseline=before,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
  numpy_global_rng=np.random.get_state(),python_rng=random.getstate(),outer_truth_read=False)


def cpu_check(data,target,info):
 sampler=Sampler(info);coverage=np.zeros((128,192),np.uint8)
 for j in range(768):
  task=sampler.task(3,j);coverage[np.ix_(task["mm"],task["ff"])]+=1
 if not np.all(coverage==1):raise RuntimeError("Complete-field coverage mismatch")
 state=sampler.state_dict();other=Sampler(info);other.load_state_dict(state)
 if sampler.task(3,768)!=other.task(3,768):raise RuntimeError("Sampler resume mismatch")
 ff=[0,17,96,191];mm=[0,64,127]
 a=np.take(target[mm,:],ff,axis=-1)
 b=data.read_fit(np.array([info["row_id"]]),ff)[0,mm,:2]
 if not np.array_equal(a,b):raise RuntimeError("Port/tone/RMS slicing mismatch")
 return dict(status="passed",joint_BS_frequency_cells=coverage.size,
   axis_block_complex_values=128*192,updates_per_complete_field_cycle=768,
   minimum_count=int(coverage.min()),maximum_count=int(coverage.max()),sampler_resume_identical=True,
   slicing_max_difference=float(np.max(np.abs(a-b))),outer_truth_read=False,parameter_updates=False)



def restore_numerical_state(ck,model,opt,sampler):
 model.load_state_dict(ck["model"]);opt.load_state_dict(ck["optimizer"])
 for state in opt.state.values():
  for k,v in state.items():
   if isinstance(v,torch.Tensor):state[k]=v.to("cuda:0")
 sampler.load_state_dict(ck["sampler"])
 torch.set_rng_state(ck["torch_rng"]);torch.cuda.set_rng_state_all(ck["cuda_rng"])
 np.random.set_state(ck["numpy_global_rng"]);random.setstate(ck["python_rng"])


def actual_reference_QA(model,data,target,info):
 """Actual FIT row, identical weights, native slow integral versus fast renderer."""
 core=model.fields["s0_b0_u0"]
 p=torch.tensor(data.positions(np.array([info["row_id"]])),device="cuda:0",dtype=torch.float64)
 site=torch.tensor([0],device="cuda:0",dtype=torch.int64)
 rx=data.geometry.ue_world_positions(p,site)[:,[info["scalar_UE_vertical"]]]
 params=tuple(core.parameters())+(model.log_output_gain,)
 results=[]
 tasks=[gate_task(0,info),dict(mm=[info["scalar_BS_spatial"],(info["scalar_BS_spatial"]+1)%128],
        uu=[info["scalar_UE_vertical"]],ff=[0,17,52,86,96,121,156,191])]
 for task in tasks:
  tx=model.bound_bs_ports[0,0].reshape(128,3)[torch.tensor(task["mm"],device="cuda:0")]
  f=torch.tensor(data.frequencies(task["ff"]),device="cuda:0",dtype=torch.float64)
  truth=target_slice(target,task)
  values=[];losses=[];gradients=[]
  for slow in (False,True):
   out=(core.__class__.forward(core,tx,rx,f)if slow else core(tx,rx,f)).permute(1,0,2,3)*model.log_output_gain.exp()
   loss=.5*(out-truth).abs().square().mean()
   gs=torch.autograd.grad(loss,params,allow_unused=True)
   gradients.append(torch.cat([(torch.zeros_like(p)if g is None else g).detach().reshape(-1)for p,g in zip(params,gs)]))
   values.append(out.detach());losses.append(float(loss.detach()))
  forward_relative=float((values[0]-values[1]).norm()/values[1].norm().clamp_min(1e-30))
  gradient_relative=float((gradients[0]-gradients[1]).norm()/gradients[1].norm().clamp_min(1e-30))
  result=dict(task=task,fast_loss=losses[0],reference_loss=losses[1],
              forward_relative_L2=forward_relative,gradient_relative_L2=gradient_relative)
  results.append(result)
  if forward_relative>1e-4 or gradient_relative>1e-3:
   raise RuntimeError("Actual-row fast/reference forward or native-loss gradient mismatch: "+str(result))
 return dict(status="passed",checks=results,actual_FIT_row=info["row_id"],
             same_weights=True,parameter_updates=False,outer_truth_read=False,
             reference="core class forward using unfactored original adapted integral",
             fast="instance shared-medium/factorized transport forward")


def unit(panel_path):
 paths();source=source_identity(panel_path);data,target_np,info=load_data(panel_path)
 if(REPORT/"result.json").exists():raise FileExistsError("Completed unit preserved")
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in()).throw(TimeoutError("Minimal unit hard600s cap")))
 signal.setitimer(signal.ITIMER_REAL,CONFIG["max_seconds"])
 started=time.monotonic();elapsed_prior=0.
 gpu=C.device_guard(P.GPU_UUID);torch.set_num_threads(4)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 C.seed_all(CONFIG["seed"]);model=V.build(data);opt=P.optimizer(model);sampler=Sampler(info)
 target=torch.tensor(target_np,device="cuda:0",dtype=torch.complex64)
 history=[];stage=0;stage_step=0;global_step=0;before=baseline(model)
 last=ART/"last.pt"
 if last.exists():
  ck=torch.load(last,map_location="cpu",weights_only=False)
  if ck["source_identity"]!=source or ck["info"]!=info:raise RuntimeError("Resume identity mismatch")
  restore_numerical_state(ck,model,opt,sampler)
  stage=ck["stage"];stage_step=ck["stage_step"];global_step=ck["global_step"]
  history=ck["history"];elapsed_prior=ck["elapsed_seconds"];before=ck["stage_parameter_baseline"]
  del ck
 elapsed=lambda:elapsed_prior+time.monotonic()-started
 signal.setitimer(signal.ITIMER_REAL,max(.001,CONFIG["max_seconds"]-elapsed()))
 C.atomic_json(REPORT/"frozen_unit.json",dict(info=info,source_identity=source,GPU=gpu))
 if global_step==0:
  C.atomic_json(REPORT/"actual_reference_QA.json",actual_reference_QA(model,data,target,info))
  C.atomic_json(REPORT/"initial_scalar_branches.json",scalar_branches(model,data,info))
 stop_reason=None
 while stage<len(CONFIG["stages"]):
  cfg=CONFIG["stages"][stage]
  if stage_step==0:
   before=baseline(model)
   emit(dict(event="stage_start",stage=stage,name=cfg["name"],stats=evaluate(model,data,target,info,stage),elapsed_seconds=elapsed()))
  stage_pass=False
  while stage_step<cfg["maximum_updates"]and global_step<CONFIG["max_updates"]:
   if elapsed()>=CONFIG["max_seconds"]-30:stop_reason="wall_budget_reserve";break
   task=sampler.task(stage,stage_step)
   model.train();opt.zero_grad(set_to_none=True)
   value=forward(model,data,info,task);truth=target_slice(target,task);value.retain_grad()
   loss=.5*(value-truth).abs().square().mean()
   if not bool(torch.isfinite(loss)):raise FloatingPointError("Nonfinite native loss")
   loss.backward()
   if not all(p.grad is None or bool(torch.isfinite(p.grad).all())for p in model.parameters()):raise FloatingPointError("Nonfinite gradient")
   measured=stage_step==0 or(stage_step+1)%cfg["monitor_every"]==0
   gd=gradient_details(model,value,truth)if measured else None
   grad=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
   train_stats=statistics(value.detach(),truth)
   opt.step();stage_step+=1;global_step+=1
   emit(dict(event="train",stage=stage,name=cfg["name"],stage_step=stage_step,global_step=global_step,
       task=task,native_loss=float(loss.detach()),gradient_norm=float(grad),matched_supervised=train_stats,
       gradient_diagnostic=gd,elapsed_seconds=elapsed()))
   if measured or stage_step==cfg["maximum_updates"]:
    check_sources(source);st=evaluate(model,data,target,info,stage)
    event=dict(event="monitor",stage=stage,name=cfg["name"],stage_step=stage_step,global_step=global_step,
       stats=st,gain=float(model.log_output_gain.exp()),elapsed_seconds=elapsed())
    history.append(event);emit(event);C.atomic_json(REPORT/"progress.json",event)
    C.atomic_torch_save(last,pack(model,opt,sampler,stage,stage_step,global_step,info,source,history,elapsed(),before))
    if passed(st,stage,stage_step):stage_pass=True;break
  st=evaluate(model,data,target,info,stage)
  stage_result=dict(stage=stage,name=cfg["name"],updates=stage_step,global_step=global_step,
    passed=stage_pass or passed(st,stage,stage_step),stats=st,
    scalar_branches=scalar_branches(model,data,info),parameter_changes=parameter_changes(model,before),
    elapsed_seconds=elapsed())
  C.atomic_json(REPORT/f"stage{stage}_{cfg['name']}.json",stage_result)
  stage_path=ART/f"stage{stage}_{cfg['name']}.pt"
  C.atomic_torch_save(stage_path,pack(model,opt,sampler,stage,stage_step,global_step,info,source,history,elapsed(),before))
  if stop_reason:break
  if stage==0:
   if not stage_result["passed"]:stop_reason="scalar_gate_failed";break
   next_stage=1;reference=stage_path
  elif stage==1:
   next_stage=2;reference=ART/"stage0_scalar.pt"
  elif stage==2:
   frequency_result=json.loads((REPORT/"stage1_F_only.json").read_text())
   if not stage_result["passed"]or not frequency_result["passed"]:
    stop_reason="isolated_axis_gate_failed";break
   next_stage=3;reference=ART/"stage1_F_only.pt"
  else:
   if not stage_result["passed"]:stop_reason="joint_gate_failed"
   stage=4;break
  ck=torch.load(reference,map_location="cpu",weights_only=False)
  restore_numerical_state(ck,model,opt,sampler);del ck
  emit(dict(event="fork",next_stage=next_stage,source_checkpoint=PATHS.relative(reference),
            source_sha256=C.sha256(reference),global_update_counter_preserved=global_step))
  stage=next_stage;stage_step=0;before=baseline(model)
  C.atomic_torch_save(last,pack(model,opt,sampler,stage,stage_step,global_step,info,source,history,elapsed(),before))
 model.eval()
 with torch.no_grad():value=forward(model,data,info,gate_task(3,info))
 final_truth=target_slice(target,gate_task(3,info))
 full_stats=statistics(value,final_truth)
 full_stats["per_UE"]=[dict(UE_vertical=0,**statistics(value,final_truth))]
 raw=(value*data.scale).cpu().numpy()
 C.atomic_npy(ART/"final_field_prediction.npy",raw,replace=True)
 C.atomic_npy(ART/"field_row_ids.npy",np.array([info["row_id"]],np.int64),replace=True)
 C.atomic_json(ART/"field_axes.json",dict(BS_ids=list(range(128)),UE_ids=[0],
   frequency_ids=list(range(192)),shape=list(raw.shape),raw_H_units=True,
   scope="site0/BS-pol0/UE-pol0/UEvertical0 axis block only; UEvertical1 and other fields untrained",
   all_stages_use_the_same_UE_vertical_port=True))
 check_sources(source)
 stages={}
 for p in sorted(REPORT.glob("stage[0-3]_*.json")):
  d=json.loads(p.read_text());stages[d["name"]]=dict(passed=d["passed"],updates=d["updates"],stats=d["stats"])
 result=dict(status="complete",stop_reason=stop_reason or"all_stage_gates_passed",
   global_updates=global_step,final_stage=stage,elapsed_seconds=elapsed(),config=CONFIG,info=info,
   stages=stages,complete_field_stats=full_stats,
   all_stage_gates_passed=len(stages)==4 and all(d["passed"]for d in stages.values()),
   source_identity=source,GPU=gpu,outer_truth_read=False,validation_used=False,official_scoring=False,
   scope="one UEvertical0 BS-frequency axis block; not complete2UE field/whole256x4H/generalization evidence",
   artifact_sha256=C.sha256(ART/"final_field_prediction.npy"))
 C.atomic_torch_save(ART/"final.pt",pack(model,opt,sampler,stage,stage_step,global_step,info,source,history,elapsed(),before))
 C.atomic_json(REPORT/"result.json",result);emit(dict(event="complete",**result))
 signal.setitimer(signal.ITIMER_REAL,0)

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--panel",type=Path,required=True)
 parser.add_argument("--mode",choices=["check","unit"],default="check");args=parser.parse_args()
 paths()
 if args.mode=="check":
  data,target,info=load_data(args.panel);result=cpu_check(data,target,info)
  C.atomic_json(REPORT/"cpu_check.json",dict(result=result,info=info,source_identity=source_identity(args.panel)))
  print(json.dumps(result),flush=True)
 else:unit(args.panel)

if __name__=="__main__":
 try:main()
 except BaseException:
  if REPORT.exists():
   C.atomic_json(REPORT/"failure.json",dict(error=__import__("traceback").format_exc()))
  raise
