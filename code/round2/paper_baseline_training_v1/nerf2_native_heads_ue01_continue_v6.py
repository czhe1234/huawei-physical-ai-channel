"""Exact UE01 continuation of the v5 single-row complete-field unit.

No new row, field, loss, renderer, learning rate, or sampling rule is introduced.
Model, Adam, all global RNGs and the private BS sampler resume from v5 final.
"""
from __future__ import annotations
import copy,hashlib,json,random,signal,time
from pathlib import Path
import numpy as np
import torch
from project_paths import PATHS
from . import common as C,nerf2_runner_v2 as P,nerf2_runner_v3 as V
from . import nerf2_minimal_fullH_v2 as U
from . import nerf2_native_frequency_heads_v4 as N
from . import nerf2_native_heads_continuation_v4 as CONT
from . import nerf2_native_heads_joint_v5 as J

ART=C.ART/"fit_repair_paper_v6_native_heads_ue01_20261101/nerf2"
REPORT=C.REPORT/"fit_repair_paper_v6_native_heads_ue01_20261101/nerf2"
PARENT=J.ART/"final.pt"
CONFIG=dict(schema="native-head-exact-UE01-continuation-v6",maximum_new_updates=1500,
 maximum_additional_seconds=300,monitor_every=100,stable_monitors_required=3,
 gate=copy.deepcopy(J.CONFIG["UE01_gate"]),UE_ids=[0,1],BS_per_batch=8,frequency_count=192,
 invariant="same v5 model/Adam/lr/all global RNG/private BS sampler/native loss/renderer/rays/far",
 scope="row2666/site0/field0 only; no other row/field/formal test")


def emit(d):
 with(REPORT/"raw_log.jsonl").open("a")as f:f.write(json.dumps(d,allow_nan=False)+"\n")
 print(json.dumps(d,allow_nan=False),flush=True)


def sampler_same(a,b):
 return(a["cursor"]==b["cursor"]and a["cycles"]==b["cycles"]
  and np.array_equal(a["order"],b["order"])and np.array_equal(a["counts"],b["counts"])
  and a["rng_state"][0]==b["rng_state"][0]and np.array_equal(a["rng_state"][1],b["rng_state"][1])
  and a["rng_state"][2:]==b["rng_state"][2:])


def snapshot(model,opt,sampler,added,total,stable,seen,history,source,info,elapsed,pred):
 return dict(model={k:v.detach().cpu().clone()for k,v in model.state_dict().items()},
  optimizer=opt.state_dict(),BS_sampler=copy.deepcopy(sampler.state_dict()),
  v6_new_updates=added,total_optimizer_step=total,stage="UE01",stable_monitors=stable,
  stage_BS_seen=seen.copy(),history=history,source_identity=source,info=info,config=CONFIG,
  additional_elapsed_seconds=elapsed,monitor_prediction_normalized=pred.detach().cpu().clone(),
  torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
  numpy_global_rng=np.random.get_state(),python_rng=random.getstate(),outer_truth_read=False)


def run():
 ART.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
 if(REPORT/"result.json").exists():raise FileExistsError("Completed UE01 continuation preserved")
 parent=torch.load(PARENT,map_location="cpu",weights_only=False)
 required=["model","optimizer","BS_sampler","total_optimizer_step","stage","stable_monitors",
  "torch_rng","cuda_rng","numpy_global_rng","python_rng","monitor_prediction_normalized"]
 missing=[k for k in required if k not in parent]
 if missing:raise RuntimeError("Missing exact continuation state: "+str(missing))
 if parent["stage"]!="UE01"or parent["total_optimizer_step"]!=4000:raise RuntimeError("Wrong v5 final parent")
 U.check_sources(parent["source_identity"])
 source={**parent["source_identity"],PATHS.relative(Path(__file__)):C.sha256(Path(__file__)),
         PATHS.relative(PARENT):C.sha256(PARENT)}
 data,target_np,info=U.load_data(N.PANEL)
 if info!=parent["info"]:raise RuntimeError("Data/FIT RMS changed")
 raw_labels=np.load(J.ART/"raw_FIT_field_labels.npy",allow_pickle=False)
 norm_labels=np.load(J.ART/"normalized_FIT_field_labels.npy",allow_pickle=False)
 if not np.array_equal(norm_labels,target_np[None]):raise RuntimeError("Actual labels changed")
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in()).throw(TimeoutError("v6 hard300s additional wall")))
 signal.setitimer(signal.ITIMER_REAL,CONFIG["maximum_additional_seconds"])
 started=time.monotonic();prior=0.
 gpu=C.device_guard(P.GPU_UUID);torch.set_num_threads(4)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 C.seed_all(42);model=V.build(data);N.freeze_NN_frequency(model);N.expand_head_rows(model);opt=P.optimizer(model)
 sampler=J.BSSampler(parent["numpy_global_rng"])
 ck=parent;added=0;stable=parent["stable_monitors"];seen=parent["stage_BS_seen"].copy();history=[]
 last=ART/"last.pt"
 if last.exists():
  ck=torch.load(last,map_location="cpu",weights_only=False)
  if ck["source_identity"]!=source or ck["config"]!=CONFIG:raise RuntimeError("v6 resume identity mismatch")
  added=ck["v6_new_updates"];stable=ck["stable_monitors"];seen=ck["stage_BS_seen"];history=ck["history"]
  prior=ck["additional_elapsed_seconds"]
 model.load_state_dict(ck["model"]);opt.load_state_dict(ck["optimizer"])
 optimizer_QA=CONT.state_comparison(ck["optimizer"],opt.state_dict())
 for key,value in ck["model"].items():
  if not torch.equal(value.cpu(),model.state_dict()[key].detach().cpu()):raise RuntimeError("Model restore changed")
 sampler.load_state_dict(copy.deepcopy(ck["BS_sampler"]))
 if not sampler_same(sampler.state_dict(),ck["BS_sampler"]):raise RuntimeError("BS sampler restoration failed")
 probe=J.BSSampler(ck["numpy_global_rng"]);probe.load_state_dict(copy.deepcopy(ck["BS_sampler"]))
 if ck["BS_sampler"]["cursor"]<16:
  expected_next=np.sort(ck["BS_sampler"]["order"][ck["BS_sampler"]["cursor"]]).tolist()
 else:
  reference_rng=np.random.RandomState();reference_rng.set_state(ck["BS_sampler"]["rng_state"])
  expected_next=np.sort(reference_rng.permutation(128).reshape(16,8)[0]).tolist()
 if probe.draw()!=expected_next:raise RuntimeError("Next-BS draw continuity failed")
 CONT.restore_rng(ck)
 CPU_RNG_exact=torch.equal(torch.get_rng_state(),ck["torch_rng"])
 CUDA_RNG_exact=all(torch.equal(a,b)for a,b in zip(torch.cuda.get_rng_state_all(),ck["cuda_rng"]))
 renderer=J.JointRenderer(data,info)
 target=torch.tensor(target_np,device="cuda:0",dtype=torch.complex64)
 pred,stats=J.evaluate(model,renderer,target,[0,1])
 expected=ck["monitor_prediction_normalized"].to("cuda:0")
 restore_error=float((pred-expected).norm()/expected.norm().clamp_min(1e-30))
 if restore_error>1e-6 or not CPU_RNG_exact or not CUDA_RNG_exact:raise RuntimeError("RNG/full-matrix continuity failed")
 total=int(ck["total_optimizer_step"])
 elapsed=lambda:prior+time.monotonic()-started
 signal.setitimer(signal.ITIMER_REAL,max(.001,CONFIG["maximum_additional_seconds"]-elapsed()))
 audit=dict(status="passed",parent_total_step=total,model_tensors_exact=True,
  optimizer_state_comparison=optimizer_QA,BS_sampler_exact=True,expected_first_BS_ids=expected_next,
  torch_CPU_RNG_exact=CPU_RNG_exact,torch_CUDA_RNG_exact=CUDA_RNG_exact,
  full_matrix_restore_relative_L2=restore_error,restored_stats=stats,
  source_identity=source,GPU=gpu,parameter_updates=False)
 C.atomic_json(REPORT/"continuity_QA.json",audit)
 C.atomic_json(REPORT/"frozen_continuation.json",dict(config=CONFIG,source_identity=source,info=info,GPU=gpu))
 if not(ART/"raw_FIT_field_labels.npy").exists():
  C.atomic_npy(ART/"raw_FIT_field_labels.npy",raw_labels)
  C.atomic_npy(ART/"normalized_FIT_field_labels.npy",norm_labels)
  C.atomic_npy(ART/"row_ids.npy",np.array([2666],np.int64))
 C.atomic_torch_save(ART/"initial_restored.pt",snapshot(model,opt,sampler,added,total,stable,seen,history,source,info,elapsed(),pred))
 emit(dict(event="continuity_passed",step=total,pooled=stats["pooled"],elapsed_seconds=elapsed()))
 stop="maximum_v6_updates"
 while added<CONFIG["maximum_new_updates"]:
  if elapsed()>=CONFIG["maximum_additional_seconds"]-25:stop="additional_wall_budget";break
  BS_ids=sampler.draw();seen[BS_ids]+=1
  model.train();opt.zero_grad(set_to_none=True)
  value=renderer(model,BS_ids,[0,1]);value.retain_grad()
  truth=J.label_subset(target,BS_ids,[0,1])
  loss=.5*(value-truth).abs().square().mean()
  if not bool(torch.isfinite(loss)):raise FloatingPointError("Nonfinite native loss")
  loss.backward()
  if not all(p.grad is None or bool(torch.isfinite(p.grad).all())for p in model.parameters()):raise FloatingPointError("Nonfinite gradient")
  measured=(added+1)%CONFIG["monitor_every"]==0
  gradient=U.gradient_details(model,value,truth)if measured else None
  norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
  matched=U.statistics(value.detach(),truth);opt.step();added+=1;total+=1
  emit(dict(event="train",v6_updates=added,total_optimizer_step=total,BS_spatial_ids=BS_ids,UE_ids=[0,1],
   all192_frequency_labels=True,native_loss=float(loss.detach()),matched_supervised=matched,
   gradient_norm=float(norm),gradient_diagnostic=gradient,elapsed_seconds=elapsed()))
  if measured:
   U.check_sources(source);pred,stats=J.evaluate(model,renderer,target,[0,1])
   stable=stable+1 if J.gate(stats,"UE01")else 0
   event=dict(event="monitor",v6_updates=added,total_optimizer_step=total,stats=stats,
      stable_monitors=stable,elapsed_seconds=elapsed())
   history.append(event);emit(event);C.atomic_json(REPORT/"progress.json",event)
   C.atomic_torch_save(last,snapshot(model,opt,sampler,added,total,stable,seen,history,source,info,elapsed(),pred))
   if stable>=CONFIG["stable_monitors_required"]:stop="complete_field_stable_fit";break
 pred,stats=J.evaluate(model,renderer,target,[0,1])
 C.atomic_npy(ART/"final_prediction.npy",(pred*info["parent_FIT_RMS"]).cpu().numpy(),replace=True)
 C.atomic_json(ART/"axes.json",dict(row_ids=[2666],BS_ids=list(range(128)),UE_ids=[0,1],
    frequency_ids=list(range(192)),shape=[1,128,2,192],raw_H_units=True,
    capability="single site0/BS-pol0/UE-pol0 field; other positions/fields untrained"))
 final=ART/"final.pt";C.atomic_torch_save(final,snapshot(model,opt,sampler,added,total,stable,seen,history,source,info,elapsed(),pred))
 reloaded=torch.load(final,map_location="cpu",weights_only=False)
 clean=V.build(data);N.freeze_NN_frequency(clean);N.expand_head_rows(clean);clean.load_state_dict(reloaded["model"])
 verify,_=J.evaluate(clean,renderer,target,[0,1])
 reload_error=float((verify-pred).norm()/pred.norm().clamp_min(1e-30))
 if reload_error>1e-6:raise RuntimeError("Clean full-matrix reload failed")
 U.check_sources(source)
 result=dict(status="complete",stop_reason=stop,v6_new_updates=added,total_optimizer_step=total,
  complete_field_recovered=stable>=CONFIG["stable_monitors_required"],stable_monitors=stable,
  final_stats=stats,history=history,additional_elapsed_seconds=elapsed(),final_reload_relative_L2=reload_error,
  config=CONFIG,source_identity=source,GPU=gpu,exact_model_Adam_allRNG_BS_sampler_continuation=True,
  other_positions_untrained=True,other_fields_untrained=True,official_scoring=False,
  outer_truth_read=False,no_formal_test=True,V16_changed=False,
  raw_prediction_sha256=C.sha256(ART/"final_prediction.npy"),
  raw_labels_sha256=C.sha256(ART/"raw_FIT_field_labels.npy"),
  final_BS_sampler_counts=sampler.counts.tolist(),UE01_stage_BS_seen=seen.tolist())
 C.atomic_json(REPORT/"result.json",result);emit(dict(event="complete",**result))
 signal.setitimer(signal.ITIMER_REAL,0)


if __name__=="__main__":
 try:run()
 except BaseException:
  REPORT.mkdir(parents=True,exist_ok=True)
  C.atomic_json(REPORT/"failure.json",dict(traceback=__import__("traceback").format_exc()))
  raise
