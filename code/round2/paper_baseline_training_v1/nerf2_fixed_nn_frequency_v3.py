"""Paired F-only control: freeze NN frequency input at0, retain actual-Hz phase.

Restores the exact v2 scalar checkpoint/model/Adam/RNG. All700 frequency draws
must match v2. No head/radius/extinction/gain/native-loss change is introduced.
"""
from __future__ import annotations
import argparse,copy,json,math,random,signal,time
from pathlib import Path
import numpy as np
import torch
from project_paths import PATHS
from . import common as C,nerf2_runner_v2 as P,nerf2_runner_v3 as V
from . import nerf2_minimal_fullH_v2 as U

ART=C.ART/"fit_repair_paper_v3_fixed_nn_frequency_20261101/nerf2"
REPORT=C.REPORT/"fit_repair_paper_v3_fixed_nn_frequency_20261101/nerf2"
CONFIG=dict(schema="nerf2-fixed-NN-frequency-paired-control-v3",
 changed_variable="fourth NN input coordinate fixed0 before PE; actual physical Hz unchanged",
 F_updates=700,joint_maximum_updates=1100,max_seconds=180,
 F_gate=dict(nmse_max=.1,energy_min=.7,energy_max=1.3),
 stage_selection="F-only final700; joint only after F gate passes; same fixed UE0",
 source_scalar="artifacts/round2/paper_four_baselines_v1/fit_repair_paper_v2_20261101/nerf2/stage0_scalar.pt",
 invariant="same scalar weights/Adam/RNG, exact frequency draws, rays324 samples32 far810, same extinction/gain/native loss")
PANEL=C.REPORT/"fit_repair_paper_v2_20261101/panel_v2.json"


def emit(d):
 with(REPORT/"raw_log.jsonl").open("a")as f:f.write(json.dumps(d,allow_nan=False)+"\n")
 print(json.dumps(d,allow_nan=False),flush=True)


def freeze_NN_frequency(model):
 original=model.fields["s0_b0_u0"].field.embed_pts_fn
 def fixed(x):return original(torch.cat([x[...,:3],torch.zeros_like(x[...,3:4])],-1))
 model.fields["s0_b0_u0"].field.embed_pts_fn=fixed


def checkpoint(model,opt,sampler,stage,step,total,info,source,history,elapsed,before):
 ck=U.pack(model,opt,sampler,stage,step,total,info,source,history,elapsed,before)
 ck["control_configuration"]=CONFIG
 return ck


def get_F_tasks():
 tasks=[]
 for line in(U.REPORT/"raw_log.jsonl").read_text().splitlines():
  d=json.loads(line)
  if d.get("event")=="train"and d.get("stage")==1:tasks.append(d["task"])
 if len(tasks)!=700:raise RuntimeError("Need complete v2 F-only700-draw reference")
 return tasks


def run():
 ART.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
 if(REPORT/"result.json").exists():raise FileExistsError("Completed control preserved")
 source={**U.source_identity(PANEL),PATHS.relative(Path(__file__)):C.sha256(Path(__file__)),
         CONFIG["source_scalar"]:C.sha256(PATHS.root/CONFIG["source_scalar"])}
 U.check_sources(source)
 data,target_np,info=U.load_data(PANEL)
 scalar=torch.load(PATHS.root/CONFIG["source_scalar"],map_location="cpu",weights_only=False)
 if scalar["source_identity"]!=U.source_identity(PANEL):raise RuntimeError("Scalar source identity mismatch")
 reference_tasks=get_F_tasks()
 probe=U.Sampler(info);probe.load_state_dict(scalar["sampler"])
 replay=[probe.task(1,j)for j in range(700)]
 if replay!=reference_tasks:raise RuntimeError("Paired supervision draws differ")
 expected_F_rng=copy.deepcopy(probe.rng.bit_generator.state)
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in()).throw(TimeoutError("Fixed-frequency control hard180s cap")))
 signal.setitimer(signal.ITIMER_REAL,CONFIG["max_seconds"])
 start=time.monotonic();prior=0.
 gpu=C.device_guard(P.GPU_UUID);torch.set_num_threads(4)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 C.seed_all(42);model=V.build(data);opt=P.optimizer(model);sampler=U.Sampler(info)
 U.restore_numerical_state(scalar,model,opt,sampler)
 target=torch.tensor(target_np,device="cuda:0",dtype=torch.complex64)
 with torch.no_grad():
  old_anchor=U.forward(model,data,info,U.gate_task(0,info)).detach()
  old_F=U.evaluate(model,data,target,info,1)
 freeze_NN_frequency(model)
 with torch.no_grad():
  new_anchor=U.forward(model,data,info,U.gate_task(0,info)).detach()
 anchor_difference=float((old_anchor-new_anchor).abs().max())
 anchor_relative=float((old_anchor-new_anchor).norm()/old_anchor.norm().clamp_min(1e-30))
 C.atomic_json(REPORT/"anchor_QA.json",dict(max_abs_difference=anchor_difference,relative_L2=anchor_relative,tolerance_relative=1e-6,parameter_updates=False))
 if anchor_relative>1e-6:raise RuntimeError("NN frequency0 changed scalar anchorF96 beyond FP32 tolerance")
 stage=1;step=0;total=0;history=[];before=U.baseline(model);last=ART/"last.pt"
 if last.exists():
  ck=torch.load(last,map_location="cpu",weights_only=False)
  if ck["source_identity"]!=source or ck["control_configuration"]!=CONFIG:raise RuntimeError("Resume identity mismatch")
  U.restore_numerical_state(ck,model,opt,sampler)
  stage,step,total=ck["stage"],ck["stage_step"],ck["global_step"]
  prior=ck["elapsed_seconds"];history=ck["history"];before=ck["stage_parameter_baseline"]
 elapsed=lambda:prior+time.monotonic()-start
 signal.setitimer(signal.ITIMER_REAL,max(.001,CONFIG["max_seconds"]-elapsed()))
 C.atomic_json(REPORT/"frozen_control.json",dict(config=CONFIG,source_identity=source,info=info,GPU=gpu,
   exact_F_draws_match=True,scalar_anchor_max_difference=anchor_difference,scalar_anchor_relative_L2=anchor_relative,
   old_NN_frequency_full_F_stats=old_F,actual_frequency_first_hz=float(data.frequencies()[0]),
   actual_frequency_last_hz=float(data.frequencies()[-1]),outer_truth_read=False))
 if total==0:
  C.atomic_json(REPORT/"actual_reference_QA.json",U.actual_reference_QA(model,data,target,info))
  C.atomic_json(REPORT/"initial_scalar_branches.json",U.scalar_branches(model,data,info))
  emit(dict(event="initial",F_stats=U.evaluate(model,data,target,info,1),
            scalar_stats=U.evaluate(model,data,target,info,0),elapsed_seconds=elapsed()))
 stop_reason=None;F_result=json.loads((REPORT/"F_only_fixed_NN_frequency.json").read_text())if stage==3 else None;joint_result=None
 while stage in(1,3):
  maximum=CONFIG["F_updates"]if stage==1 else CONFIG["joint_maximum_updates"]
  interval=48 if stage==1 else 192
  if step==0:before=U.baseline(model)
  while step<maximum:
   if elapsed()>CONFIG["max_seconds"]-15:stop_reason="wall_reserve";break
   task=sampler.task(stage,step)
   if stage==1 and task!=reference_tasks[step]:raise RuntimeError("Actual F draw mismatch")
   model.train();opt.zero_grad(set_to_none=True)
   value=U.forward(model,data,info,task);truth=U.target_slice(target,task);value.retain_grad()
   loss=.5*(value-truth).abs().square().mean()
   if not bool(torch.isfinite(loss)):raise FloatingPointError("Nonfinite native loss")
   loss.backward()
   if not all(p.grad is None or bool(torch.isfinite(p.grad).all())for p in model.parameters()):raise FloatingPointError("Nonfinite gradient")
   measured=step==0 or(step+1)%interval==0 or step+1==maximum
   gradient=U.gradient_details(model,value,truth)if measured else None
   norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
   stats=U.statistics(value.detach(),truth)
   opt.step();step+=1;total+=1
   emit(dict(event="train",stage=stage,stage_step=step,total_new_updates=total,task=task,
       native_loss=float(loss.detach()),matched_supervised=stats,gradient_norm=float(norm),
       gradient=gradient,elapsed_seconds=elapsed()))
   if measured:
    U.check_sources(source);st=U.evaluate(model,data,target,info,stage)
    e=dict(event="monitor",stage=stage,stage_step=step,total_new_updates=total,stats=st,elapsed_seconds=elapsed())
    history.append(e);emit(e);C.atomic_json(REPORT/"progress.json",e)
    C.atomic_torch_save(last,checkpoint(model,opt,sampler,stage,step,total,info,source,history,elapsed(),before))
    if stage==3 and U.passed(st,3,step):break
  st=U.evaluate(model,data,target,info,stage)
  limits=CONFIG["F_gate"]
  good=st["nmse"]<=limits["nmse_max"]and limits["energy_min"]<=st["energy_ratio"]<=limits["energy_max"]
  if stage==3:good=good and step>=768
  result=dict(stage=stage,updates=step,stats=st,passed=bool(good),
       scalar_branches=U.scalar_branches(model,data,info),parameter_changes=U.parameter_changes(model,before),
       elapsed_seconds=elapsed())
  name="F_only_fixed_NN_frequency"if stage==1 else"joint_fixed_NN_frequency"
  C.atomic_json(REPORT/(name+".json"),result)
  C.atomic_torch_save(ART/(name+".pt"),checkpoint(model,opt,sampler,stage,step,total,info,source,history,elapsed(),before))
  if stage==1:
   F_result=result
   if step==700 and sampler.rng.bit_generator.state!=expected_F_rng:raise RuntimeError("F paired sampler end RNG mismatch")
   with torch.no_grad():F_raw=(U.forward(model,data,info,U.gate_task(1,info))*data.scale).cpu().numpy()
   C.atomic_npy(ART/"F_only_prediction.npy",F_raw,replace=True)
   if stop_reason or not good:stop_reason=stop_reason or"F_gate_failed";break
   stage=3;step=0;before=U.baseline(model)
   emit(dict(event="F_gate_passed_joint_start",source_checkpoint=name+".pt",total_new_updates=total))
   C.atomic_torch_save(last,checkpoint(model,opt,sampler,stage,step,total,info,source,history,elapsed(),before))
  else:
   joint_result=result;stop_reason="joint_passed"if good else"joint_gate_failed";break
 with torch.no_grad():block=U.forward(model,data,info,U.gate_task(3,info));fullstats=U.statistics(block,U.target_slice(target,U.gate_task(3,info)))
 C.atomic_npy(ART/"final_axis_block_prediction.npy",(block*data.scale).cpu().numpy(),replace=True)
 C.atomic_npy(ART/"row_ids.npy",np.array([info["row_id"]],np.int64),replace=True)
 C.atomic_json(ART/"axes.json",dict(BS_ids=list(range(128)),UE_ids=[0],frequency_ids=list(range(192)),
    shape=[1,128,1,192],raw_H_units=True,only_axis_diagnostic=True))
 U.check_sources(source)
 result=dict(status="complete",stop_reason=stop_reason,new_updates=total,elapsed_seconds=elapsed(),
   F_only=F_result,joint=joint_result,final_axis_block_stats=fullstats,
   config=CONFIG,source_identity=source,exact_v2_F_draws=True,
   initial_scalar_anchor_difference=anchor_difference,physical_Hz_not_frozen=True,
   outer_truth_read=False,official_scoring=False,V16_changed=False,
   final_axis_block_was_jointly_trained=joint_result is not None,
   does_not_certify_full_H_or_paper_baseline=True)
 C.atomic_torch_save(ART/"final.pt",checkpoint(model,opt,sampler,stage,step,total,info,source,history,elapsed(),before))
 C.atomic_json(REPORT/"result.json",result);emit(dict(event="complete",**result))
 signal.setitimer(signal.ITIMER_REAL,0)


if __name__=="__main__":
 try:run()
 except BaseException:
  REPORT.mkdir(parents=True,exist_ok=True)
  C.atomic_json(REPORT/"failure_retry1.json",dict(traceback=__import__("traceback").format_exc()))
  raise
