"""Exact continuation of native192-head B500; no optimizer reset or new factor.

The old A/B500 pair is preserved. Longer B is a fitting-health trajectory,
not an equal-budget comparison with A. All192F labels stay fixed each step.
"""
from __future__ import annotations
import hashlib,json,random,signal,time
from pathlib import Path
import numpy as np
import torch
from . import common as C,nerf2_runner_v2 as P,nerf2_runner_v3 as V
from . import nerf2_minimal_fullH_v2 as U
from . import nerf2_native_frequency_heads_v4 as N
from project_paths import PATHS

ART=N.ART/"B_native192_continuation"
REPORT=N.REPORT/"B_native192_continuation"
PARENT=N.ART/"B_native192/final.pt"
CONFIG=dict(schema="native192-B500-exact-continuation-v1",parent_step=500,
 maximum_total_step=3000,maximum_additional_seconds=600,monitor_every=100,
 stop_gate=dict(nmse_max=.01,energy_min=.8,energy_max=1.2,stable_monitors=3),
 native_loss="unchanged0.5complexMSE",optimizer="restore full B500 Adam; no reset or lr change",
 labels="same complete192F, row2666 BS30 UE0",scope="fitting health; longerB is not same-budget A/B comparison")


def emit(d):
 with(REPORT/"raw_log.jsonl").open("a")as f:f.write(json.dumps(d,allow_nan=False)+"\n")
 print(json.dumps(d,allow_nan=False),flush=True)


def state_comparison(expected,actual):
 if len(expected["param_groups"])!=len(actual["param_groups"]):raise RuntimeError("Optimizer group count changed")
 count=0
 for e,a in zip(expected["param_groups"],actual["param_groups"]):
  if e!=a:raise RuntimeError("Optimizer group parameters or hyperparameters changed")
 if set(expected["state"])!=set(actual["state"]):raise RuntimeError("Optimizer state keys changed")
 for key,old in expected["state"].items():
  new=actual["state"][key]
  if set(old)!=set(new):raise RuntimeError("Missing Adam moment")
  for name,value in old.items():
   current=new[name]
   if isinstance(value,torch.Tensor):
    if not torch.equal(value.cpu(),current.detach().cpu()):raise RuntimeError("Adam tensor changed: "+name)
    count+=1
   elif value!=current:raise RuntimeError("Adam scalar changed: "+name)
 return dict(exact=True,compared_state_tensors=count,parameter_states=len(actual["state"]))


def restore_rng(ck):
 torch.set_rng_state(ck["torch_rng"]);torch.cuda.set_rng_state_all(ck["cuda_rng"])
 np.random.set_state(ck["numpy_global_rng"]);random.setstate(ck["python_rng"])


def snapshot(model,opt,step,source,info,history,elapsed,stable,prediction):
 return dict(model={k:v.detach().cpu().clone()for k,v in model.state_dict().items()},
  optimizer=opt.state_dict(),step=step,arm="B_native192",source_identity=source,info=info,
  config=CONFIG,parent_pair_config=N.CONFIG,history=history,additional_elapsed_seconds=elapsed,
  stable_monitors=stable,prediction_normalized=prediction.detach().cpu().clone(),
  torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
  numpy_global_rng=np.random.get_state(),python_rng=random.getstate(),outer_truth_read=False)


def run():
 ART.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
 if(REPORT/"result.json").exists():raise FileExistsError("Completed continuation preserved")
 parent=torch.load(PARENT,map_location="cpu",weights_only=False)
 required=["model","optimizer","torch_rng","cuda_rng","numpy_global_rng","python_rng",
           "source_identity","info","step","config","history"]
 missing=[k for k in required if k not in parent]
 if missing:raise RuntimeError("Cannot claim exact continuation; missing "+str(missing))
 if parent["step"]!=500 or parent["arm"]!="B_native192":raise RuntimeError("Wrong parent checkpoint")
 U.check_sources(parent["source_identity"])
 source={**parent["source_identity"],PATHS.relative(Path(__file__)):C.sha256(Path(__file__)),
         PATHS.relative(PARENT):C.sha256(PARENT)}
 data,target_np,info=U.load_data(N.PANEL)
 if info!=parent["info"]:raise RuntimeError("Training data/normalization info changed")
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in()).throw(TimeoutError("B continuation hard600s additional wall")))
 signal.setitimer(signal.ITIMER_REAL,CONFIG["maximum_additional_seconds"])
 started=time.monotonic();prior=0.
 gpu=C.device_guard(P.GPU_UUID);torch.set_num_threads(4)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 C.seed_all(42);model=V.build(data);N.freeze_NN_frequency(model);N.expand_head_rows(model)
 opt=P.optimizer(model)
 last=ART/"last.pt";ck=parent
 if last.exists():
  ck=torch.load(last,map_location="cpu",weights_only=False)
  if ck["source_identity"]!=source or ck["config"]!=CONFIG:raise RuntimeError("Continuation resume identity mismatch")
  prior=ck["additional_elapsed_seconds"]
 model.load_state_dict(ck["model"]);opt.load_state_dict(ck["optimizer"])
 for group in opt.param_groups:
  for parameter in group["params"]:
   state=opt.state.get(parameter,{})
   for key in("exp_avg","exp_avg_sq"):
    if key in state and state[key].device!=parameter.device:raise RuntimeError("Adam moment device mismatch")
 optimizer_QA=state_comparison(ck["optimizer"],opt.state_dict())
 for name,tensor in ck["model"].items():
  if not torch.equal(tensor.cpu(),model.state_dict()[name].detach().cpu()):
   raise RuntimeError("Model restoration changed tensor "+name)
 restore_rng(ck)
 cpu_rng_equal=torch.equal(torch.get_rng_state(),ck["torch_rng"])
 cuda_rng_equal=all(torch.equal(a,b)for a,b in zip(torch.cuda.get_rng_state_all(),ck["cuda_rng"]))
 if not cpu_rng_equal or not cuda_rng_equal:raise RuntimeError("RNG restoration failed")
 renderer=N.SpatialOnceRenderer(data,info)
 target_label=target_np[info["scalar_BS_spatial"],info["scalar_UE_vertical"]]
 target=torch.tensor(target_label,device="cuda:0",dtype=torch.complex64).reshape(1,1,1,192)
 frozen=json.loads((N.REPORT/"frozen_pair.json").read_text())
 if hashlib.sha256(target_label.tobytes()).hexdigest()!=frozen["target_normalized_sha256"]:
  raise RuntimeError("Matched192-label tensor changed")
 model.eval()
 with torch.no_grad():restored=renderer(model,True)
 if"prediction_normalized"in ck:
  expected=ck["prediction_normalized"].to("cuda:0")
 else:
  expected=torch.tensor(np.load(N.ART/"B_native192/F_prediction.npy",allow_pickle=False),
                        device="cuda:0",dtype=torch.complex64)/info["parent_FIT_RMS"]
 restoration_error=float((restored-expected).norm()/expected.norm().clamp_min(1e-30))
 if restoration_error>1e-6:raise RuntimeError("Restored prediction differs from parent")
 step=int(ck["step"]);stable=int(ck.get("stable_monitors",0))
 history=ck.get("history",[])
 if last.exists():history=list(history)
 else:history=[]
 elapsed=lambda:prior+time.monotonic()-started
 signal.setitimer(signal.ITIMER_REAL,max(.001,CONFIG["maximum_additional_seconds"]-elapsed()))
 audit=dict(status="passed",parent_checkpoint=PATHS.relative(PARENT),parent_sha256=source[PATHS.relative(PARENT)],
  restored_step=step,required_state_keys_present=required,optimizer_state_comparison=optimizer_QA,
  model_tensors_exact=True,torch_CPU_RNG_exact=cpu_rng_equal,torch_CUDA_RNG_exact=cuda_rng_equal,
  prediction_relative_L2=restoration_error,restored_stats=U.statistics(restored,target),
  source_identity=source,config=CONFIG,GPU=gpu,optimizer_reset=False,
  original_A500_B500_preserved=True,additional_step_counter_starts_at500=True)
 C.atomic_json(REPORT/"restoration_QA.json",audit);emit(dict(event="restored",**audit))
 C.atomic_torch_save(ART/"initial_restored.pt",snapshot(model,opt,step,source,info,history,elapsed(),stable,restored))
 stop="maximum_total_step"
 start_step=step
 while step<CONFIG["maximum_total_step"]:
  if elapsed()>=CONFIG["maximum_additional_seconds"]-20:stop="additional_wall_budget";break
  model.train();opt.zero_grad(set_to_none=True)
  value=renderer(model,True);value.retain_grad()
  loss=.5*(value-target).abs().square().mean()
  if not bool(torch.isfinite(loss)):raise FloatingPointError("Nonfinite native loss")
  loss.backward()
  if not all(p.grad is None or bool(torch.isfinite(p.grad).all())for p in model.parameters()):raise FloatingPointError("Nonfinite gradient")
  next_step=step+1
  measured=next_step%CONFIG["monitor_every"]==0
  diagnostic=U.gradient_details(model,value,target)if measured else None
  norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
  matched=U.statistics(value.detach(),target)
  opt.step();step=next_step
  emit(dict(event="train",step=step,additional_updates=step-500,native_loss=float(loss.detach()),
    matched_supervised=matched,gradient_norm=float(norm),gradient_diagnostic=diagnostic,elapsed_seconds=elapsed()))
  if measured or step==CONFIG["maximum_total_step"]:
   U.check_sources(source);model.eval()
   with torch.no_grad():prediction=renderer(model,True);stats=U.statistics(prediction,target)
   limits=CONFIG["stop_gate"]
   good=stats["nmse"]<=limits["nmse_max"]and limits["energy_min"]<=stats["energy_ratio"]<=limits["energy_max"]
   stable=stable+1 if good else 0
   event=dict(event="monitor",step=step,additional_updates=step-500,stats=stats,
              stable_monitors=stable,elapsed_seconds=elapsed())
   history.append(event);emit(event);C.atomic_json(REPORT/"progress.json",event)
   C.atomic_torch_save(last,snapshot(model,opt,step,source,info,history,elapsed(),stable,prediction))
   if stable>=limits["stable_monitors"]:stop="stable_fullF_fit";break
 model.eval()
 with torch.no_grad():prediction=renderer(model,True);stats=U.statistics(prediction,target)
 C.atomic_npy(ART/"F_prediction.npy",(prediction*info["parent_FIT_RMS"]).cpu().numpy(),replace=True)
 C.atomic_torch_save(ART/"final.pt",snapshot(model,opt,step,source,info,history,elapsed(),stable,prediction))
 U.check_sources(source)
 result=dict(status="complete",stop_reason=stop,final_step=step,additional_updates=step-500,
   current_invocation_updates=step-start_step,additional_elapsed_seconds=elapsed(),final_stats=stats,
   strong_gate_stable=stable>=CONFIG["stop_gate"]["stable_monitors"],stable_monitors=stable,
   history=history,config=CONFIG,source_identity=source,GPU=gpu,optimizer_reset=False,
   exact_parent_restoration=True,original_A500_B500_preserved=True,
   not_equal_budget_comparison=True,not_full_H_or_generalization=True,
   axes=dict(row_ids=[2666],BS_ids=[30],UE_ids=[0],frequency_ids=list(range(192)),shape=[1,1,1,192]),
   raw_prediction_sha256=C.sha256(ART/"F_prediction.npy"),outer_truth_read=False,official_scoring=False,V16_changed=False)
 C.atomic_json(REPORT/"result.json",result);emit(dict(event="complete",**result))
 signal.setitimer(signal.ITIMER_REAL,0)


if __name__=="__main__":
 try:run()
 except BaseException:
  REPORT.mkdir(parents=True,exist_ok=True)
  C.atomic_json(REPORT/"failure.json",dict(traceback=__import__("traceback").format_exc()))
  raise
