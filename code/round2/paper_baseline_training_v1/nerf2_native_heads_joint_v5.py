"""Single-row native192-head joint BS/frequency and optional second-UE unit.

Restores B1000 model/Adam/RNG. Each update supervises a balanced group of8BS
and all192 tones. UE1 is introduced only after the completeUE0 matrix gate
passes three times. This is one polarization field, not full8field H.
"""
from __future__ import annotations
import copy,hashlib,json,random,signal,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from project_paths import PATHS
from . import common as C,nerf2_runner_v2 as P,nerf2_runner_v3 as V,field_fast as FF
from . import nerf2_minimal_fullH_v2 as U
from . import nerf2_native_frequency_heads_v4 as N
from . import nerf2_native_heads_continuation_v4 as CONT

ART=C.ART/"fit_repair_paper_v5_native_heads_joint_20261101/nerf2"
REPORT=C.REPORT/"fit_repair_paper_v5_native_heads_joint_20261101/nerf2"
PARENT=N.ART/"B_native192_continuation/final.pt"
CONFIG=dict(schema="native-head-single-row-joint-field-v5",parent_step=1000,
 maximum_new_updates=3000,maximum_additional_seconds=600,BS_per_batch=8,
 frequency_count=192,monitor_every=100,stable_monitors_required=3,
 UE0_gate=dict(pooled_nmse_max=.05,energy_min=.8,energy_max=1.2,
               minimum_BS_fraction_nmse_le_dot1=.95),
 UE01_gate=dict(pooled_nmse_max=.05,each_UE_nmse_max=.1,
                each_UE_energy_min=.7,each_UE_energy_max=1.3),
 invariant="same B1000 Adam/lr/nativeMSE/NNfreq0/native192heads/rays324/samples32/far810/actualHz",
 scope="row2666/site0/BS-pol0/UE-pol0 only; no extra rows/fields/formal test",
 optimizer_reset=False,BS_sampler="no replacement in16 groups per cycle; RNG cloned from exact parent NumPy state")


class JointRenderer:
 def __init__(self,data,info):
  self.info=info
  p=torch.tensor(data.positions(np.array([info["row_id"]])),device="cuda:0",dtype=torch.float64)
  site=torch.tensor([0],device="cuda:0",dtype=torch.int64)
  self.world=data.geometry.ue_world_positions(p,site)
  self.f=torch.tensor(data.frequencies(),device="cuda:0",dtype=torch.float64)
 def __call__(self,model,BS_ids,UE_ids):
  core=model.fields["s0_b0_u0"];field=core.field;cfg=core.config
  nr,ns,nf=cfg.n_rays,cfg.n_samples,len(self.f)
  if(nr,ns,cfg.far_m,cfg.near_m)!=(324,32,810.,0.):raise RuntimeError("Renderer convention changed")
  if not set(UE_ids).issubset({0,1}):raise ValueError("This field covers only UE0/1")
  device=next(field.parameters()).device
  centre=torch.tensor(cfg.world_center_m,device=device,dtype=torch.float64)
  directions=core.ray_directions.to(device=device,dtype=torch.float64)
  directions=directions/torch.linalg.vector_norm(directions,dim=-1,keepdim=True)
  nports=len(UE_ids);ntx=len(BS_ids);ray_count=nports*nr
  rx=self.world[:,torch.tensor(UE_ids,device=device)].reshape(nports,3)
  segment=(cfg.far_m-cfg.near_m)/ns
  radii=cfg.near_m+segment*torch.arange(1,ns+1,device=device,dtype=torch.float64)
  points=(rx[:,None,None,:]+directions[None,:,None,:]*radii[None,None,:,None]).reshape(ray_count,ns,3)
  xyz=((points-centre)/cfg.world_scale_m).float()
  frequency0=torch.zeros((*xyz.shape[:-1],1),device=device,dtype=torch.float32)
  view=directions.float()[None,:,None,:].expand(nports,nr,ns,3).reshape(ray_count,ns,3)
  attenuation,common,shape=FF._medium(field,torch.cat([xyz,frequency0],-1),view)
  tx=model.bound_bs_ports[0,0].reshape(128,3)[torch.tensor(BS_ids,device=device)]
  tx_encoded=field.embed_tx_fn(((tx-centre)/cfg.world_scale_m).float())
  first=field.signal_linears[0];td=tx_encoded.shape[-1];cd=first.in_features-td
  shared=F.linear(common,first.weight[:,:cd],first.bias)
  txpart=F.linear(tx_encoded,first.weight[:,cd:],None)
  x=F.relu(shared[None]+txpart[:,None])
  for layer in field.signal_linears[1:]:x=F.relu(layer(x))
  signal=field.signal_output(x)
  if attenuation.shape[-1]!=384 or signal.shape[-1]!=384:raise RuntimeError("Require native192 head")
  att=attenuation.reshape(ray_count,ns,2,nf).permute(0,3,1,2)
  sig=signal.reshape(ntx,ray_count,ns,2,nf).permute(0,1,4,2,3)
  shape=(ray_count,nf,ns)
  segments=torch.full((ns,),segment,device=device,dtype=torch.float64)
  trans,carrier=FF._shared_transport(att.reshape(-1,2),shape,segments,self.f,cfg.propagation_sign,cfg.near_m)
  rays=FF._factorized_integral(sig.reshape(ntx,-1,2),shape,trans,carrier)
  return(rays.reshape(ntx,nports,nr,nf).sum(2)*model.log_output_gain.exp())[None]


class BSSampler:
 def __init__(self,parent_numpy_state):
  self.rng=np.random.RandomState();self.rng.set_state(parent_numpy_state)
  self.order=None;self.cursor=0;self.cycles=0;self.counts=np.zeros(128,np.int64)
 def draw(self):
  if self.order is None or self.cursor==16:
   self.order=self.rng.permutation(128).reshape(16,8);self.cursor=0;self.cycles+=1
  ids=np.sort(self.order[self.cursor]);self.cursor+=1;self.counts[ids]+=1
  return ids.tolist()
 def state_dict(self):return dict(rng_state=self.rng.get_state(),order=self.order,cursor=self.cursor,cycles=self.cycles,counts=self.counts)
 def load_state_dict(self,s):
  self.rng.set_state(s["rng_state"]);self.order=s["order"];self.cursor=s["cursor"];self.cycles=s["cycles"];self.counts=s["counts"]


def label_subset(target,BS_ids,UE_ids):
 return target.index_select(0,torch.tensor(BS_ids,device="cuda:0")).index_select(1,torch.tensor(UE_ids,device="cuda:0"))[None]


def reference_QA(model,renderer,data,target,info,UE_ids):
 BS_ids=[0,17,30,47,64,81,98,127]
 truth=label_subset(target,BS_ids,UE_ids)
 actual=renderer(model,BS_ids,UE_ids);loss=.5*(actual-truth).abs().square().mean()
 ga=N.grad_map(model,loss)
 ref=N.SpatialOnceRenderer(data,info)
 gs={name:torch.zeros_like(p,dtype=torch.float64)for name,p in N.params(model)}
 outputs=[];reference_loss=0.
 for bs in BS_ids:
  per_UE=[]
  for ue in UE_ids:
   ref.bs=bs;ref.rx=renderer.world[:,ue:ue+1]
   value=ref(model,True)
   label=label_subset(target,[bs],[ue])
   native=.5*(value-label).abs().square().sum()/truth.numel()
   g=N.grad_map(model,native)
   for name in gs:gs[name]+=g[name]
   per_UE.append(value.detach());reference_loss+=float(native.detach())
  outputs.append(torch.cat(per_UE,2))
 expected=torch.cat(outputs,1)
 forward_error=float((actual.detach()-expected).norm()/expected.norm().clamp_min(1e-30))
 gradient_error=N.relative_gradient(ga,gs,list(ga))
 result=dict(BS_ids=BS_ids,UE_ids=UE_ids,frequency_count=192,
  forward_relative_L2=forward_error,native_gradient_relative_L2=gradient_error,
  batch_loss=float(loss.detach()),reference_loss=reference_loss,
  reference="same trained native192 model, sequential BS and actual UE physical positions",
  parameter_updates=False,outer_truth_read=False)
 if forward_error>1e-4 or gradient_error>1e-3:raise RuntimeError("NTx8/UE reference QA failed: "+str(result))
 return result


def matrix_stats(prediction,truth,UE_ids):
 z=prediction.detach().cpu().numpy().astype(np.complex128)
 t=truth.detach().cpu().numpy().astype(np.complex128)
 se=np.abs(z-t)**2;te=np.abs(t)**2;pe=np.abs(z)**2
 def stat(a,b,c):
  energy=float(b.sum());pred=float(c.sum());error=float(a.sum())
  return dict(nmse=error/energy if energy>0 else None,energy_ratio=pred/energy if energy>0 else None,
              squared_error=error,truth_energy=energy,prediction_energy=pred)
 result=dict(pooled=stat(se,te,pe),per_UE=[])
 bs_se=se.sum((0,2,3));bs_te=te.sum((0,2,3))
 bs_nmse=np.divide(bs_se,bs_te,out=np.full_like(bs_se,1e30),where=bs_te>0)
 result.update(BS_nmse=bs_nmse.tolist(),BS_nmse_p95=float(np.quantile(bs_nmse,.95)),
   BS_nmse_max=float(bs_nmse.max()),BS_fraction_nmse_le_dot1=float(np.mean(bs_nmse<=.1)),
   zero_target_BS_count=int((bs_te==0).sum()))
 for i,ue in enumerate(UE_ids):
  per=stat(se[:,:,i:i+1],te[:,:,i:i+1],pe[:,:,i:i+1])
  per_se=se[:,:,i].sum((0,2));per_te=te[:,:,i].sum((0,2))
  ratios=np.divide(per_se,per_te,out=np.full_like(per_se,1e30),where=per_te>0)
  result["per_UE"].append(dict(UE_id=ue,**per,BS_nmse=ratios.tolist(),
      BS_nmse_max=float(ratios.max()),BS_nmse_p95=float(np.quantile(ratios,.95))))
 return result


@torch.no_grad()
def evaluate(model,renderer,target,UE_ids):
 model.eval();parts=[]
 for start in range(0,128,8):parts.append(renderer(model,list(range(start,start+8)),UE_ids))
 prediction=torch.cat(parts,1)
 return prediction,matrix_stats(prediction,label_subset(target,list(range(128)),UE_ids),UE_ids)


def gate(stats,stage):
 if stage=="UE0":
  c=CONFIG["UE0_gate"];p=stats["pooled"]
  return p["nmse"]<=c["pooled_nmse_max"]and c["energy_min"]<=p["energy_ratio"]<=c["energy_max"]and stats["BS_fraction_nmse_le_dot1"]>=c["minimum_BS_fraction_nmse_le_dot1"]
 c=CONFIG["UE01_gate"]
 return stats["pooled"]["nmse"]<=c["pooled_nmse_max"]and all(
  x["nmse"]<=c["each_UE_nmse_max"]and c["each_UE_energy_min"]<=x["energy_ratio"]<=c["each_UE_energy_max"]for x in stats["per_UE"])


def emit(event):
 with(REPORT/"raw_log.jsonl").open("a")as f:f.write(json.dumps(event,allow_nan=False)+"\n")
 print(json.dumps(event,allow_nan=False),flush=True)


def snapshot(model,opt,sampler,added,stage,stable,history,source,info,elapsed,seen,pred):
 return dict(model={k:v.detach().cpu().clone()for k,v in model.state_dict().items()},
  optimizer=opt.state_dict(),BS_sampler=sampler.state_dict(),new_updates=added,
  total_optimizer_step=1000+added,stage=stage,stable_monitors=stable,history=history,
  source_identity=source,info=info,config=CONFIG,additional_elapsed_seconds=elapsed,
  stage_BS_seen=seen,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
  numpy_global_rng=np.random.get_state(),python_rng=random.getstate(),
  monitor_prediction_normalized=pred.detach().cpu().clone(),outer_truth_read=False)


def run():
 ART.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
 if(REPORT/"result.json").exists():raise FileExistsError("Completed joint unit preserved")
 parent=torch.load(PARENT,map_location="cpu",weights_only=False)
 required=["model","optimizer","step","torch_rng","cuda_rng","numpy_global_rng","python_rng","source_identity","prediction_normalized"]
 if any(k not in parent for k in required):raise RuntimeError("Missing state; exact restoration unavailable")
 if parent["step"]!=1000:raise RuntimeError("Require B1000 continuation parent")
 U.check_sources(parent["source_identity"])
 source={**parent["source_identity"],PATHS.relative(Path(__file__)):C.sha256(Path(__file__)),PATHS.relative(PARENT):C.sha256(PARENT)}
 data,target_np,info=U.load_data(N.PANEL)
 if info!=parent["info"]:raise RuntimeError("Training input/RMS changed")
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in()).throw(TimeoutError("Joint unit hard600s additional wall")))
 signal.setitimer(signal.ITIMER_REAL,CONFIG["maximum_additional_seconds"])
 started=time.monotonic();prior=0.
 gpu=C.device_guard(P.GPU_UUID);torch.set_num_threads(4)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 C.seed_all(42);model=V.build(data);N.freeze_NN_frequency(model);N.expand_head_rows(model);opt=P.optimizer(model)
 sampler=BSSampler(parent["numpy_global_rng"]);last=ART/"last.pt";ck=parent
 added=0;stage="UE0";stable=0;history=[];seen=np.zeros(128,np.int64)
 if last.exists():
  ck=torch.load(last,map_location="cpu",weights_only=False)
  if ck["source_identity"]!=source or ck["config"]!=CONFIG:raise RuntimeError("Joint resume source mismatch")
  added,stage,stable=ck["new_updates"],ck["stage"],ck["stable_monitors"]
  history=ck["history"];seen=ck["stage_BS_seen"];prior=ck["additional_elapsed_seconds"]
  sampler.load_state_dict(ck["BS_sampler"])
 model.load_state_dict(ck["model"]);opt.load_state_dict(ck["optimizer"])
 optimizer_check=CONT.state_comparison(ck["optimizer"],opt.state_dict())
 for key,value in ck["model"].items():
  if not torch.equal(value.cpu(),model.state_dict()[key].detach().cpu()):raise RuntimeError("Model restore mismatch")
 CONT.restore_rng(ck)
 renderer=JointRenderer(data,info)
 target=torch.tensor(target_np,device="cuda:0",dtype=torch.complex64)
 if not last.exists():
  with torch.no_grad():anchor=renderer(model,[30],[0])
  delta=float((anchor-parent["prediction_normalized"].to("cuda:0")).norm()/parent["prediction_normalized"].norm().clamp_min(1e-30))
  if delta>1e-6:raise RuntimeError("B1000 anchor restore mismatch")
  qa=[reference_QA(model,renderer,data,target,info,[0]),reference_QA(model,renderer,data,target,info,[0,1])]
  C.atomic_json(REPORT/"NTx8_forward_gradient_QA.json",dict(status="passed",checks=qa,anchor_restore_relative_L2=delta,optimizer_state_comparison=optimizer_check,source_identity=source))
  CONT.restore_rng(parent)
 else:
  qa=[]
 elapsed=lambda:prior+time.monotonic()-started
 signal.setitimer(signal.ITIMER_REAL,max(.001,CONFIG["maximum_additional_seconds"]-elapsed()))
 raw_labels=data.read_fit(np.array([info["row_id"]]),normalized=False)[:,:128,:2]
 if not(ART/"raw_FIT_field_labels.npy").exists():
  C.atomic_npy(ART/"raw_FIT_field_labels.npy",raw_labels)
  C.atomic_npy(ART/"normalized_FIT_field_labels.npy",target_np[None])
  C.atomic_npy(ART/"row_ids.npy",np.array([2666],np.int64))
 C.atomic_json(REPORT/"frozen_joint.json",dict(config=CONFIG,source_identity=source,info=info,GPU=gpu,
   raw_label_sha256=C.sha256(ART/"raw_FIT_field_labels.npy"),
   normalized_label_sha256=C.sha256(ART/"normalized_FIT_field_labels.npy"),
   optimizer_state_comparison=optimizer_check,new_BS_sampler_from_parent_numpy_state=True,
   native192_head_position_only_adaptation=True,other_fields_untrained=True))
 initial_UE=[0]if stage=="UE0"else[0,1]
 if added==0:
  pred,st=evaluate(model,renderer,target,[0]);emit(dict(event="initial_fullUE0",stats=st,elapsed_seconds=elapsed()))
 stop="maximum_new_updates";stage_passed={}
 for filename in["UE0_gate_passed.json","UE01_gate_passed.json"]:
  p=REPORT/filename
  if p.exists():stage_passed[filename.split("_gate")[0]]=json.loads(p.read_text())
 while added<CONFIG["maximum_new_updates"]:
  if elapsed()>=CONFIG["maximum_additional_seconds"]-30:stop="additional_wall_budget";break
  UE_ids=[0]if stage=="UE0"else[0,1]
  BS_ids=sampler.draw();seen[BS_ids]+=1
  model.train();opt.zero_grad(set_to_none=True)
  value=renderer(model,BS_ids,UE_ids);value.retain_grad()
  truth=label_subset(target,BS_ids,UE_ids)
  loss=.5*(value-truth).abs().square().mean()
  if not bool(torch.isfinite(loss)):raise FloatingPointError("Nonfinite native loss")
  loss.backward()
  if not all(p.grad is None or bool(torch.isfinite(p.grad).all())for p in model.parameters()):raise FloatingPointError("Nonfinite gradient")
  measured=(added+1)%CONFIG["monitor_every"]==0
  gradient=U.gradient_details(model,value,truth)if measured else None
  norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
  matched=U.statistics(value.detach(),truth);opt.step();added+=1
  emit(dict(event="train",stage=stage,new_updates=added,total_optimizer_step=1000+added,
    BS_spatial_ids=BS_ids,UE_ids=UE_ids,all_frequency_ids_0_to191=True,
    native_loss=float(loss.detach()),matched_supervised=matched,gradient_norm=float(norm),
    gradient_diagnostic=gradient,elapsed_seconds=elapsed()))
  if measured:
   U.check_sources(source);pred,st=evaluate(model,renderer,target,UE_ids)
   stable=stable+1 if gate(st,stage)else 0
   event=dict(event="monitor",stage=stage,new_updates=added,total_optimizer_step=1000+added,
     stats=st,stable_monitors=stable,stage_BS_seen_min=int(seen.min()),stage_BS_seen_max=int(seen.max()),
     elapsed_seconds=elapsed())
   history.append(event);emit(event);C.atomic_json(REPORT/"progress.json",event)
   C.atomic_torch_save(last,snapshot(model,opt,sampler,added,stage,stable,history,source,info,elapsed(),seen,pred))
   if stable>=CONFIG["stable_monitors_required"]:
    stage_passed[stage]=event
    C.atomic_json(REPORT/(stage+"_gate_passed.json"),event)
    C.atomic_npy(ART/(stage+"_prediction.npy"),(pred*info["parent_FIT_RMS"]).cpu().numpy(),replace=True)
    C.atomic_torch_save(ART/(stage+"_passed.pt"),snapshot(model,opt,sampler,added,stage,stable,history,source,info,elapsed(),seen,pred))
    if stage=="UE0":
     stage="UE01";stable=0;seen=np.zeros(128,np.int64)
     emit(dict(event="introduce_UE1",new_updates=added,optimizer_reset=False,same_native_forward=True))
     C.atomic_torch_save(last,snapshot(model,opt,sampler,added,stage,stable,history,source,info,elapsed(),seen,pred))
    else:stop="complete_field_stable_fit";break
 UE_ids=[0]if stage=="UE0"else[0,1]
 pred,stats=evaluate(model,renderer,target,UE_ids)
 C.atomic_npy(ART/"final_prediction.npy",(pred*info["parent_FIT_RMS"]).cpu().numpy(),replace=True)
 C.atomic_json(ART/"axes.json",dict(row_ids=[2666],BS_ids=list(range(128)),UE_ids=UE_ids,
    frequency_ids=list(range(192)),shape=list(pred.shape),raw_H_units=True,
    complete_polarization_field=stage=="UE01"and"UE01"in stage_passed,
    other_polarization_fields_untrained=True))
 final_checkpoint=ART/"final.pt"
 C.atomic_torch_save(final_checkpoint,snapshot(model,opt,sampler,added,stage,stable,history,source,info,elapsed(),seen,pred))
 loaded=torch.load(final_checkpoint,map_location="cpu",weights_only=False)
 reload_model=V.build(data);N.freeze_NN_frequency(reload_model);N.expand_head_rows(reload_model)
 reload_model.load_state_dict(loaded["model"])
 reloaded,_=evaluate(reload_model,renderer,target,UE_ids)
 reload_error=float((reloaded-pred).norm()/pred.norm().clamp_min(1e-30))
 if reload_error>1e-6:raise RuntimeError("Final full-matrix reload mismatch")
 U.check_sources(source)
 result=dict(status="complete",stop_reason=stop,new_updates=added,total_optimizer_step=1000+added,
  stage=stage,stage_gates_passed=list(stage_passed),complete_field_recovered="UE01"in stage_passed,
  final_stats=stats,additional_elapsed_seconds=elapsed(),history=history,
  final_reload_relative_L2=reload_error,config=CONFIG,source_identity=source,GPU=gpu,
  optimizer_reset=False,other_polarization_fields_untrained=True,other_positions_untrained=True,
  no_formal_test=True,outer_truth_read=False,official_scoring=False,V16_changed=False,
  raw_prediction_sha256=C.sha256(ART/"final_prediction.npy"),
  raw_labels_sha256=C.sha256(ART/"raw_FIT_field_labels.npy"),
  stage_BS_seen=seen.tolist(),BS_sampler_counts=sampler.counts.tolist())
 C.atomic_json(REPORT/"result.json",result);emit(dict(event="complete",**result))
 signal.setitimer(signal.ITIMER_REAL,0)


if __name__=="__main__":
 try:run()
 except BaseException:
  REPORT.mkdir(parents=True,exist_ok=True)
  C.atomic_json(REPORT/"failure.json",dict(traceback=__import__("traceback").format_exc()))
  raise
