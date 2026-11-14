"""Native-frequency-head paired unit, position-only adaptation.

Both arms use identical scalar weights, fresh Adam, fixed NN frequency0 and
all192 labels at every step. B restores the official grouped multi-frequency
heads; actual-Hz propagation and existing finite-ray physical conventions stay.
This is not an unchanged query-uplink-CSI reproduction or a full-H paper score.
"""
from __future__ import annotations
import copy,hashlib,json,random,signal,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from project_paths import PATHS
from . import common as C,nerf2_runner_v2 as P,nerf2_runner_v3 as V,field_fast as FF
from . import nerf2_minimal_fullH_v2 as U

ART=C.ART/"fit_repair_paper_v4_native_frequency_heads_20261101/nerf2"
REPORT=C.REPORT/"fit_repair_paper_v4_native_frequency_heads_20261101/nerf2"
PANEL=C.REPORT/"fit_repair_paper_v2_20261101/panel_v2.json"
SCALAR=C.ART/"fit_repair_paper_v2_20261101/nerf2/stage0_scalar.pt"
CONFIG=dict(schema="nerf2-native-frequency-head-pair-v4",seed=42,frequencies=192,
 arms=["A_shared2","B_native192"],maximum_updates_per_arm=500,maximum_seconds_total=600,
 monitor_every=25,checkpoint_every=100,
 early_stop=dict(nmse_max=.01,energy_min=.8,energy_max=1.2,stable_monitors=3),
 final_gate=dict(nmse_max=.1,energy_min=.7,energy_max=1.3),
 supervision="same complete192F complex values every update; differs from oldK4 for BOTH arms",
 starting_weights="same v2 scalar50 checkpoint; B copies old head rows into each frequency",
 optimizer="P.optimizer, fresh empty Adam state for both arms",
 NN_frequency="fixed0 before original4D PE",physical_frequency="unchanged actual FP64 Hz",
 native_loss="0.5*mean(abs(pred-target)**2)",ray_sample_far=[324,32,810],
 head_layout="alpha[F],residual_phase[F],emission_amplitude[F],emission_phase[F]",
 conditioning="position/TX only, no query CSI",
 scope="row2666/BS30/UE0/all192F only; not full128BS/allUE or paper performance")


class MIMOExtinction(nn.Linear):
 def __init__(self,in_features,nfreq):
  super().__init__(in_features,2*nfreq);self.nfreq=nfreq
 def forward(self,x):
  raw=F.linear(x,self.weight,self.bias)
  return torch.cat([F.softplus(raw[...,:self.nfreq])/810.,raw[...,self.nfreq:]],-1)


def freeze_NN_frequency(model):
 original=model.fields["s0_b0_u0"].field.embed_pts_fn
 def fixed(x):return original(torch.cat([x[...,:3],torch.zeros_like(x[...,3:4])],-1))
 model.fields["s0_b0_u0"].field.embed_pts_fn=fixed


def expand_head_rows(model,nfreq=192):
 field=model.fields["s0_b0_u0"].field
 old_a,old_s=field.attenuation_output,field.signal_output
 a=MIMOExtinction(old_a.in_features,nfreq).to(old_a.weight.device)
 s=nn.Linear(old_s.in_features,2*nfreq).to(old_s.weight.device)
 with torch.no_grad():
  for new,old in[(a,old_a),(s,old_s)]:
   new.weight[:nfreq].copy_(old.weight[0:1].expand(nfreq,-1))
   new.weight[nfreq:].copy_(old.weight[1:2].expand(nfreq,-1))
   new.bias[:nfreq].fill_(old.bias[0]);new.bias[nfreq:].fill_(old.bias[1])
 field.attenuation_output=a;field.signal_output=s


class SpatialOnceRenderer:
 """Same source spatial MLP/integral; no repeated NN evaluation over frequency."""
 def __init__(self,data,info):
  p=torch.tensor(data.positions(np.array([info["row_id"]])),device="cuda:0",dtype=torch.float64)
  site=torch.tensor([0],device="cuda:0",dtype=torch.int64)
  self.rx=data.geometry.ue_world_positions(p,site)[:,info["scalar_UE_vertical"]:info["scalar_UE_vertical"]+1]
  self.f=torch.tensor(data.frequencies(),device="cuda:0",dtype=torch.float64)
  self.bs=info["scalar_BS_spatial"]
 def __call__(self,model,multihead):
  core=model.fields["s0_b0_u0"];field=core.field;cfg=core.config
  nr,ns,nf=cfg.n_rays,cfg.n_samples,len(self.f)
  if(nr,ns,cfg.far_m,cfg.near_m)!=(324,32,810.,0.):raise RuntimeError("Frozen renderer changed")
  device=next(field.parameters()).device
  centre=torch.tensor(cfg.world_center_m,device=device,dtype=torch.float64)
  directions=core.ray_directions.to(device=device,dtype=torch.float64)
  directions=directions/torch.linalg.vector_norm(directions,dim=-1,keepdim=True)
  segment=(cfg.far_m-cfg.near_m)/ns
  radii=cfg.near_m+segment*torch.arange(1,ns+1,device=device,dtype=torch.float64)
  points=self.rx.reshape(1,3)[:,None,:]+directions[:,None,:]*radii[None,:,None]
  xyz=((points-centre)/cfg.world_scale_m).float()
  frequency0=torch.zeros((*xyz.shape[:-1],1),device=device,dtype=torch.float32)
  view=directions.float()[:,None,:].expand(nr,ns,3)
  attenuation,common,shape=FF._medium(field,torch.cat([xyz,frequency0],-1),view)
  if shape!=(nr,ns):raise RuntimeError("Spatial-only shape mismatch")
  tx=model.bound_bs_ports[0,0].reshape(128,3)[self.bs:self.bs+1]
  tx_encoded=field.embed_tx_fn(((tx-centre)/cfg.world_scale_m).float())
  first=field.signal_linears[0];tx_dim=tx_encoded.shape[-1];common_dim=first.in_features-tx_dim
  shared=F.linear(common,first.weight[:,:common_dim],first.bias)
  tx_projection=F.linear(tx_encoded,first.weight[:,common_dim:],None)
  x=F.relu(shared[None]+tx_projection[:,None])
  for layer in field.signal_linears[1:]:x=F.relu(layer(x))
  signal=field.signal_output(x)
  if multihead:
   if attenuation.shape[-1]!=2*nf or signal.shape[-1]!=2*nf:raise RuntimeError("Grouped frequency head width")
   att=attenuation.reshape(nr,ns,2,nf).permute(0,3,1,2)
   sig=signal.reshape(1,nr,ns,2,nf).permute(0,1,4,2,3)
  else:
   att=attenuation.reshape(nr,ns,2)[:,None].expand(nr,nf,ns,2)
   sig=signal.reshape(1,nr,ns,2)[:,:,None].expand(1,nr,nf,ns,2)
  shape=(nr,nf,ns)
  segments=torch.full((ns,),segment,device=device,dtype=torch.float64)
  transmittance,carrier=FF._shared_transport(att.reshape(-1,2),shape,segments,self.f,cfg.propagation_sign,cfg.near_m)
  ray=FF._factorized_integral(sig.reshape(1,-1,2),shape,transmittance,carrier)
  return(ray.sum(1)*model.log_output_gain.exp()).reshape(1,1,1,nf)


def params(model):
 return[(n,p)for n,p in model.named_parameters()if n=="log_output_gain"or n.startswith("fields.s0_b0_u0.")]


def grad_map(model,loss):
 named=params(model)
 gs=torch.autograd.grad(loss,tuple(p for n,p in named),allow_unused=True)
 return{n:(torch.zeros_like(p)if g is None else g).detach().double()for(n,p),g in zip(named,gs)}


def relative_gradient(a,b,keys):
 numerator=sum(float((a[k]-b[k]).square().sum())for k in keys)
 denominator=sum(float(b[k].square().sum())for k in keys)
 return(numerator/max(denominator,1e-60))**.5


def old_A_reference(model,data,target,info):
 """192-label native mean loss; exact old forward in its eight-tone chunks."""
 outputs=[];grads={n:torch.zeros_like(p,dtype=torch.float64)for n,p in params(model)}
 total_loss=0.
 for start in range(0,192,8):
  ff=list(range(start,min(start+8,192)))
  task=dict(mm=[info["scalar_BS_spatial"]],uu=[info["scalar_UE_vertical"]],ff=ff)
  value=U.forward(model,data,info,task)
  truth=target.index_select(-1,torch.tensor(ff,device="cuda:0"))
  loss=.5*(value-truth).abs().square().sum()/target.numel()
  gs=grad_map(model,loss)
  for n in grads:grads[n]+=gs[n]
  outputs.append(value.detach());total_loss+=float(loss.detach())
 return torch.cat(outputs,-1),total_loss,grads


def pair_QA(A,B,renderer,data,target,info):
 old,old_loss,old_grad=old_A_reference(A,data,target,info)
 valueA=renderer(A,False);lossA=.5*(valueA-target).abs().square().mean();ga=grad_map(A,lossA)
 valueB=renderer(B,True);lossB=.5*(valueB-target).abs().square().mean();gb=grad_map(B,lossB)
 old_forward=float((valueA.detach()-old).norm()/old.norm().clamp_min(1e-30))
 pair_forward=float((valueB.detach()-valueA.detach()).norm()/valueA.detach().norm().clamp_min(1e-30))
 old_gradient=relative_gradient(ga,old_grad,list(ga))
 head_names=[n for n in ga if"attenuation_output"in n or"signal_output"in n]
 summed={}
 head_checks={}
 for name in head_names:
  summed[name]=torch.stack([gb[name][:192].sum(0),gb[name][192:].sum(0)],0)
  head_checks[name]=relative_gradient(summed,ga,[name])
 shared_names=[n for n in ga if n not in head_names]
 shared_gradient=relative_gradient(gb,ga,shared_names)
 result=dict(status="passed",old_A_vs_spatial_once_forward_relative_L2=old_forward,
  old_A_vs_spatial_once_native_gradient_relative_L2=old_gradient,
  A_vs_B_initial_forward_relative_L2=pair_forward,
  B_head_gradient_frequency_sum_vs_A=head_checks,
  A_vs_B_shared_trunk_gain_gradient_relative_L2=shared_gradient,
  old_A_loss=old_loss,A_loss=float(lossA.detach()),B_loss=float(lossB.detach()),
  frequency_count=192,old_A_gradient="native192-label mean accumulated in source8-tone chunks",
  NN_frequency_fixed0=True,physical_Hz_unchanged=True,parameter_updates=False,
  actual_FIT_row=info["row_id"],outer_truth_read=False)
 if old_forward>1e-4 or pair_forward>1e-4 or old_gradient>1e-3 or shared_gradient>1e-3 or max(head_checks.values())>1e-3:
  result["status"]="failed"
  raise RuntimeError("Paired implementation QA failed: "+str(result))
 return result


def restore_rng(scalar):
 torch.set_rng_state(scalar["torch_rng"]);torch.cuda.set_rng_state_all(scalar["cuda_rng"])
 np.random.set_state(scalar["numpy_global_rng"]);random.setstate(scalar["python_rng"])


def emit(event):
 with(REPORT/"raw_log.jsonl").open("a")as f:f.write(json.dumps(event,allow_nan=False)+"\n")
 print(json.dumps(event,allow_nan=False),flush=True)


def pack(model,opt,name,step,source,info,history):
 return dict(model={k:v.detach().cpu().clone()for k,v in model.state_dict().items()},
  optimizer=opt.state_dict(),arm=name,step=step,source_identity=source,info=info,config=CONFIG,
  history=history,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
  numpy_global_rng=np.random.get_state(),python_rng=random.getstate(),outer_truth_read=False)


def criterion(st,limits):return st["nmse"]<=limits["nmse_max"]and limits["energy_min"]<=st["energy_ratio"]<=limits["energy_max"]


def train_arm(name,model,opt,multihead,renderer,target,source,info,scalar,elapsed):
 restore_rng(scalar)
 out=ART/name;report=REPORT/name;out.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
 history=[];stable=0;stop="maximum_updates";step=0
 if len(opt.state)!=0:raise RuntimeError("Both arms must reset Adam")
 C.atomic_torch_save(out/"initial.pt",pack(model,opt,name,0,source,info,history))
 for step in range(1,CONFIG["maximum_updates_per_arm"]+1):
  if elapsed()>=CONFIG["maximum_seconds_total"]-20:stop="wall_budget";step-=1;break
  model.train();opt.zero_grad(set_to_none=True)
  value=renderer(model,multihead);value.retain_grad()
  loss=.5*(value-target).abs().square().mean()
  if not bool(torch.isfinite(loss)):raise FloatingPointError("Nonfinite native loss")
  loss.backward()
  if not all(p.grad is None or bool(torch.isfinite(p.grad).all())for p in model.parameters()):raise FloatingPointError("Nonfinite gradient")
  measured=step==1 or step%CONFIG["monitor_every"]==0
  gd=U.gradient_details(model,value,target)if measured else None
  norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
  matched=U.statistics(value.detach(),target)
  opt.step()
  emit(dict(event="train",arm=name,step=step,complex_labels=192,native_loss=float(loss.detach()),
            matched_supervised=matched,gradient_norm=float(norm),gradient_diagnostic=gd,elapsed_seconds=elapsed()))
  if measured or step==CONFIG["maximum_updates_per_arm"]:
   U.check_sources(source)
   model.eval()
   with torch.no_grad():st=U.statistics(renderer(model,multihead),target)
   event=dict(event="monitor",arm=name,step=step,stats=st,elapsed_seconds=elapsed())
   history.append(event);emit(event);C.atomic_json(report/"progress.json",event)
   stable=stable+1 if criterion(st,CONFIG["early_stop"])else 0
   if stable>=CONFIG["early_stop"]["stable_monitors"]:stop="stable_strong_fit";break
  if step%CONFIG["checkpoint_every"]==0:C.atomic_torch_save(out/"last.pt",pack(model,opt,name,step,source,info,history))
 model.eval()
 with torch.no_grad():prediction=renderer(model,multihead);st=U.statistics(prediction,target)
 C.atomic_npy(out/"F_prediction.npy",(prediction*info["parent_FIT_RMS"]).cpu().numpy(),replace=True)
 C.atomic_torch_save(out/"final.pt",pack(model,opt,name,step,source,info,history))
 result=dict(arm=name,updates=step,supervised_complex_values=step*192,stop_reason=stop,
  final_stats=st,final_gate_passed=criterion(st,CONFIG["final_gate"]),
  strong_gate_passed=criterion(st,CONFIG["early_stop"]),stable_strong_monitors=stable,
  history=history,source_identity=source,optimizer_initial_states=0,outer_truth_read=False,
  prediction_shape=[1,1,1,192],prediction_units="raw H",RMS=info["parent_FIT_RMS"],
  SHA256=C.sha256(out/"F_prediction.npy"))
 C.atomic_json(report/"result.json",result);return result


def run():
 ART.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
 if(REPORT/"result.json").exists():raise FileExistsError("Completed pair preserved")
 source={**U.source_identity(PANEL),PATHS.relative(Path(__file__)):C.sha256(Path(__file__)),PATHS.relative(SCALAR):C.sha256(SCALAR)}
 U.check_sources(source)
 data,target_np,info=U.load_data(PANEL);scalar=torch.load(SCALAR,map_location="cpu",weights_only=False)
 if scalar["source_identity"]!=U.source_identity(PANEL):raise RuntimeError("Scalar provenance mismatch")
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in()).throw(TimeoutError("Native-head pair600s hard wall")))
 signal.setitimer(signal.ITIMER_REAL,CONFIG["maximum_seconds_total"])
 started=time.monotonic();elapsed=lambda:time.monotonic()-started
 gpu=C.device_guard(P.GPU_UUID);torch.set_num_threads(4)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 C.seed_all(42);A=V.build(data);A.load_state_dict(scalar["model"]);freeze_NN_frequency(A)
 C.seed_all(42);B=V.build(data);B.load_state_dict(scalar["model"]);freeze_NN_frequency(B);expand_head_rows(B)
 optA=P.optimizer(A);optB=P.optimizer(B)
 renderer=SpatialOnceRenderer(data,info)
 label=target_np[info["scalar_BS_spatial"],info["scalar_UE_vertical"]]
 target=torch.tensor(label,device="cuda:0",dtype=torch.complex64).reshape(1,1,1,192)
 initial_states=[len(optA.state),len(optB.state)]
 if initial_states!=[0,0]:raise RuntimeError("Adam reset failed")
 qa=pair_QA(A,B,renderer,data,target,info)
 C.atomic_json(REPORT/"paired_implementation_QA.json",qa)
 frozen=dict(config=CONFIG,source_identity=source,info=info,GPU=gpu,
    target_normalized_sha256=hashlib.sha256(label.tobytes()).hexdigest(),
    target_raw_sha256=hashlib.sha256((label*data.scale).astype(np.complex64).tobytes()).hexdigest(),
    optimizer_initial_states=initial_states,extra_head_parameters_per_active_field=147452,
    conditioning="position-only native-frequency-head adaptation; no query uplink CSI",
    old_K4_results_are_not_this_pair=True)
 C.atomic_json(REPORT/"frozen_pair.json",frozen);emit(dict(event="QA_passed",QA=qa,elapsed_seconds=elapsed()))
 results=[]
 for name,model,opt,multihead in[("A_shared2",A,optA,False),("B_native192",B,optB,True)]:
  results.append(train_arm(name,model,opt,multihead,renderer,target,source,info,scalar,elapsed))
 U.check_sources(source)
 C.atomic_json(ART/"axes.json",dict(row_ids=[info["row_id"]],BS_ids=[info["scalar_BS_spatial"]],
    UE_ids=[info["scalar_UE_vertical"]],frequency_ids=list(range(192)),shape=[1,1,1,192],raw_H_units=True))
 result=dict(status="complete",results=results,elapsed_seconds=elapsed(),config=CONFIG,
  source_identity=source,GPU=gpu,matched_same192_labels_every_step=True,both_Adam_reset=True,
  only_changed_model_mechanism="shared output2 vs original-layout frequency-specific384 heads",
  not_original_query_CSI_reproduction=True,outer_truth_read=False,official_scoring=False,V16_changed=False,
  peak_cuda_bytes=torch.cuda.max_memory_allocated())
 C.atomic_json(REPORT/"result.json",result);emit(dict(event="complete",**result))
 signal.setitimer(signal.ITIMER_REAL,0)


if __name__=="__main__":
 try:run()
 except BaseException:
  REPORT.mkdir(parents=True,exist_ok=True)
  C.atomic_json(REPORT/"failure.json",dict(traceback=__import__("traceback").format_exc()))
  raise
