"""Two-position field-fitting development: same native192-head model.

Starts from the stable v7 model/moments/RNG, explicitly restores adaptation LR,
then uses paired BS groups for two alternating FIT positions. No other field,
validation/outer row, loss, gain, calibration, or physical renderer is changed.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,random,signal,time
from pathlib import Path
import numpy as np
import torch
from project_paths import PATHS
from . import common as C,nerf2_runner_v2 as P,nerf2_runner_v3 as V
from . import nerf2_minimal_fullH_v2 as U
from . import nerf2_native_frequency_heads_v4 as N
from . import nerf2_native_heads_continuation_v4 as CONT
from . import nerf2_native_heads_joint_v5 as J
from . import nerf2_low_lr_ue01_v7 as PREVIOUS

ART=C.ART/"fit_repair_paper_v8_two_position_field_20261101/nerf2"
REPORT=C.REPORT/"fit_repair_paper_v8_two_position_field_20261101/nerf2"
PARENT=PREVIOUS.ART/"final.pt"
ROWS=[2666,2042]
CONFIG=dict(schema="native-head-two-FIT-position-field-v8",rows=ROWS,site=0,field=[0,0,0],
 maximum_new_updates=3000,maximum_additional_seconds=600,monitor_every=100,
 initial_LR=[1e-4,1e-3],convergence_after_update=2000,convergence_LR_multiplier=.2,
 stable_monitors_required=3,
 gate=dict(pooled_nmse_max=.1,each_position_nmse_max=.2,each_position_energy_min=.7,
           each_position_energy_max=1.3,each_position_UE_nmse_max=.25),
 supervision="alternate positions, same8BS group for consecutive pair; full2UE/all192F",
 sampler="new paired BS cycle from exact parent private RNG;32updates cover128BS once/position",
 invariant="same native192heads/NNfreq0/model moments/parentFIT RMS/nativeMSE/actualHz/rays324/samples32/far810",
 scope="two parent screenFIT/site0 positions, one polarization field only; no validation/outer/test")


class PositionPairSampler:
 def __init__(self,parent_BS_state,parent_numpy_state):
  self.base=J.BSSampler(parent_numpy_state)
  self.base.rng.set_state(parent_BS_state["rng_state"])
  self.pending=None;self.next_position=0;self.counts=np.zeros((2,128),np.int64)
 def draw(self):
  if self.pending is None:self.pending=self.base.draw()
  index=self.next_position;ids=list(self.pending);self.counts[index,ids]+=1
  if index==0:self.next_position=1
  else:self.next_position=0;self.pending=None
  return index,ids
 def state_dict(self):
  return dict(base=copy.deepcopy(self.base.state_dict()),pending=self.pending,
              next_position=self.next_position,counts=self.counts.copy())
 def load_state_dict(self,s):
  self.base.load_state_dict(copy.deepcopy(s["base"]));self.pending=s["pending"]
  self.next_position=s["next_position"];self.counts=s["counts"].copy()


def load_inputs():
 data,_,base_info=U.load_data(N.PANEL)
 ids=np.asarray(ROWS,np.int64)
 if not set(ids).issubset(set(data.fit_ids))or np.any(data.sites(ids)!=0):
  raise PermissionError("Two positions must both be site0 parent screenFIT")
 if set(ids)&set(data.val_ids)or set(ids)&set(data.outer_ids):
  raise PermissionError("Validation/outer position in two-point FIT unit")
 normalized=data.read_fit(ids)[:,:128,:2]
 raw=data.read_fit(ids,normalized=False)[:,:128,:2]
 reconstructed=(raw/data.scale).astype(np.complex64)
 if not np.array_equal(normalized,reconstructed):raise RuntimeError("RMS/BS/UE/frequency labels mismatch")
 infos=[{**base_info,"row_id":row}for row in ROWS]
 identity=dict(row_ids=ROWS,positions_m=data.positions(ids).tolist(),site_ids=data.sites(ids).tolist(),
  distance_m=float(np.linalg.norm(data.positions(ids)[0]-data.positions(ids)[1])),
  parent_FIT_RMS=data.scale,input_identity=data.inputs.identity,
  both_parent_screenFIT=True,validation_overlap=[],outer_overlap=[],
  native_field_shape=[2,128,2,192],normalized_labels_sha256=hashlib.sha256(normalized.tobytes()).hexdigest(),
  raw_labels_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
  raw_field_energies=[float(np.sum(np.abs(x.astype(np.complex128))**2))for x in raw],
  model_world_centre="unchanged parent screenFIT mean from v3 builder",
  frequency_ids=list(range(192)),BS_ids=list(range(128)),UE_ids=[0,1])
 return data,normalized,raw,infos,identity


def build(parent,data,source,from_checkpoint=None):
 C.seed_all(42);model=V.build(data);N.freeze_NN_frequency(model);N.expand_head_rows(model);opt=P.optimizer(model)
 ck=parent if from_checkpoint is None else from_checkpoint
 model.load_state_dict(ck["model"]);opt.load_state_dict(ck["optimizer"])
 before=CONT.state_comparison(ck["optimizer"],opt.state_dict())
 expected=copy.deepcopy(ck["optimizer"])
 if from_checkpoint is None:
  for group,ex,lr in zip(opt.param_groups,expected["param_groups"],CONFIG["initial_LR"]):
   group["lr"]=lr;ex["lr"]=lr
 only_lr=CONT.state_comparison(expected,opt.state_dict())
 for key,value in ck["model"].items():
  if not torch.equal(value.cpu(),model.state_dict()[key].detach().cpu()):raise RuntimeError("Model restoration mismatch")
 CONT.restore_rng(ck)
 CPU_exact=torch.equal(torch.get_rng_state(),ck["torch_rng"])
 CUDA_exact=all(torch.equal(a,b)for a,b in zip(torch.cuda.get_rng_state_all(),ck["cuda_rng"]))
 if not CPU_exact or not CUDA_exact:raise RuntimeError("Global RNG restoration failed")
 sampler=PositionPairSampler(parent["BS_sampler"],parent["numpy_global_rng"])
 if from_checkpoint is not None:sampler.load_state_dict(ck["position_pair_sampler"])
 return model,opt,sampler,dict(model_tensors_exact=True,optimizer_before_LR=before,
  optimizer_after_only_LR_change=only_lr,parent_LR=[g["lr"]for g in parent["optimizer"]["param_groups"]],
  current_LR=[g["lr"]for g in opt.param_groups],
  torch_CPU_RNG_exact=torch.equal(torch.get_rng_state(),ck["torch_rng"]),
  torch_CUDA_RNG_exact=all(torch.equal(a,b)for a,b in zip(torch.cuda.get_rng_state_all(),ck["cuda_rng"])))


def cpu_sampler_QA(parent):
 s=PositionPairSampler(parent["BS_sampler"],parent["numpy_global_rng"])
 tasks=[s.draw()for _ in range(32)]
 if not np.all(s.counts==1):raise RuntimeError("Each position must cover all128BS once/32updates")
 for i in range(0,32,2):
  if tasks[i][0]!=0 or tasks[i+1][0]!=1 or tasks[i][1]!=tasks[i+1][1]:
   raise RuntimeError("Position alternation/paired BS mismatch")
 saved=copy.deepcopy(s.state_dict());other=PositionPairSampler(parent["BS_sampler"],parent["numpy_global_rng"])
 other.load_state_dict(saved)
 if s.draw()!=other.draw():raise RuntimeError("Paired sampler resume mismatch")
 return dict(status="passed",updates=32,every_position_BS_cell_count_one=True,
   same_BS_for_consecutive_positions=True,resume_draw_identical=True,parameter_updates=False)


def snapshot(model,opt,sampler,added,total,stable,history,source,identity,infos,elapsed,lowLR,pred):
 return dict(model={k:v.detach().cpu().clone()for k,v in model.state_dict().items()},
  optimizer=opt.state_dict(),position_pair_sampler=copy.deepcopy(sampler.state_dict()),
  v8_new_updates=added,total_optimizer_step=total,stable_monitors=stable,history=history,
  source_identity=source,input_identity=identity,position_infos=infos,config=CONFIG,
  additional_elapsed_seconds=elapsed,convergence_LR_applied=lowLR,
  prediction_normalized=pred.detach().cpu().clone(),
  torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
  numpy_global_rng=np.random.get_state(),python_rng=random.getstate(),outer_truth_read=False)


@torch.no_grad()
def evaluate(model,renderers,targets):
 predictions=[];positions=[]
 for index,renderer in enumerate(renderers):
  pred,stats=J.evaluate(model,renderer,targets[index],[0,1])
  predictions.append(pred);positions.append(dict(row_id=ROWS[index],**stats))
 se=sum(x["pooled"]["squared_error"]for x in positions)
 te=sum(x["pooled"]["truth_energy"]for x in positions)
 pe=sum(x["pooled"]["prediction_energy"]for x in positions)
 return torch.cat(predictions,0),dict(pooled=dict(nmse=se/te,energy_ratio=pe/te,
   squared_error=se,truth_energy=te,prediction_energy=pe),per_position=positions)


def gate(stats):
 limits=CONFIG["gate"]
 return stats["pooled"]["nmse"]<=limits["pooled_nmse_max"]and all(
  x["pooled"]["nmse"]<=limits["each_position_nmse_max"]
  and limits["each_position_energy_min"]<=x["pooled"]["energy_ratio"]<=limits["each_position_energy_max"]
  and all(u["nmse"]<=limits["each_position_UE_nmse_max"]for u in x["per_UE"])
  for x in stats["per_position"])


def emit(d):
 with(REPORT/"raw_log.jsonl").open("a")as f:f.write(json.dumps(d,allow_nan=False)+"\n")
 print(json.dumps(d,allow_nan=False),flush=True)


def prepare(mode):
 ART.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
 if mode=="run"and(REPORT/"result.json").exists():raise FileExistsError("Completed two-position unit preserved")
 parent_result=json.loads((PREVIOUS.REPORT/"result.json").read_text())
 parent=torch.load(PARENT,map_location="cpu",weights_only=False)
 if not parent_result["complete_field_recovered"]or parent["stable_monitors"]<3:
  raise PermissionError("v7 original stable3 gate is required before two positions")
 U.check_sources(parent["source_identity"])
 source={**parent["source_identity"],PATHS.relative(Path(__file__)):C.sha256(Path(__file__)),
         PATHS.relative(PARENT):C.sha256(PARENT)}
 data,normalized,raw,infos,identity=load_inputs()
 gpu=C.device_guard(P.GPU_UUID);torch.set_num_threads(4)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 targets=torch.tensor(normalized,device="cuda:0",dtype=torch.complex64)
 renderers=[J.JointRenderer(data,info)for info in infos]
 if mode=="qa":
  model,opt,sampler,restore=build(parent,data,source)
  checks=[J.reference_QA(model,r,data,targets[i],infos[i],[0,1])for i,r in enumerate(renderers)]
  for info,check in zip(infos,checks):check["actual_FIT_row"]=info["row_id"]
  with torch.no_grad():
   old,_=J.evaluate(model,renderers[0],targets[0],[0,1])
  expected=parent["monitor_prediction_normalized"].to("cuda:0")
  delta=float((old-expected).norm()/expected.norm().clamp_min(1e-30))
  if delta>1e-6:raise RuntimeError("Parent position forward continuity failed")
  CPU_QA=cpu_sampler_QA(parent)
  CONT.restore_rng(parent)
  pred,stats=evaluate(model,renderers,targets)
  for filename,array in[("raw_FIT_field_labels.npy",raw),("normalized_FIT_field_labels.npy",normalized),
                       ("row_ids.npy",np.asarray(ROWS,np.int64))]:
   if not(ART/filename).exists():C.atomic_npy(ART/filename,array)
  initial=ART/"prepared_initial.pt"
  C.atomic_torch_save(initial,snapshot(model,opt,sampler,0,parent["total_optimizer_step"],0,[],
      source,identity,infos,0.,False,pred))
  result=dict(status="passed",source_identity=source,input_identity=identity,restore=restore,
    CPU_sampler_QA=CPU_QA,actual_position_slow_reference_checks=checks,
    parent_position_restore_relative_L2=delta,exact_elementwise_raw_over_FIT_RMS=True,
    prepared_initial_sha256=C.sha256(initial),initial_full_matrix_stats=stats,
    parameter_updates=False,GPU=gpu,outer_truth_read=False)
  U.check_sources(source);C.atomic_json(REPORT/"pretraining_QA.json",result)
  print(json.dumps(result,allow_nan=False),flush=True);return
 qa=json.loads((REPORT/"pretraining_QA.json").read_text())
 if qa["status"]!="passed"or qa["source_identity"]!=source or qa["input_identity"]!=identity:
  raise RuntimeError("Reviewed QA source/data identity changed")
 if C.sha256(ART/"prepared_initial.pt")!=qa["prepared_initial_sha256"]:
  raise RuntimeError("Prepared state changed after QA")
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in()).throw(TimeoutError("Two-position hard600s wall")))
 signal.setitimer(signal.ITIMER_REAL,CONFIG["maximum_additional_seconds"])
 started=time.monotonic();prior=0.
 last=ART/"last.pt";ck=torch.load(last if last.exists()else ART/"prepared_initial.pt",map_location="cpu",weights_only=False)
 if ck["source_identity"]!=source or ck["config"]!=CONFIG:raise RuntimeError("Training state identity mismatch")
 model,opt,sampler,restore=build(parent,data,source,ck)
 added,total,stable=ck["v8_new_updates"],ck["total_optimizer_step"],ck["stable_monitors"]
 history=ck["history"];prior=ck["additional_elapsed_seconds"];lowLR=ck["convergence_LR_applied"]
 elapsed=lambda:prior+time.monotonic()-started
 signal.setitimer(signal.ITIMER_REAL,max(.001,CONFIG["maximum_additional_seconds"]-elapsed()))
 C.atomic_json(REPORT/"frozen_run.json",dict(config=CONFIG,source_identity=source,input_identity=identity,
    restored_optimizer=restore,GPU=gpu,pretraining_QA_sha256=C.sha256(REPORT/"pretraining_QA.json")))
 stop="maximum_v8_updates"
 while added<CONFIG["maximum_new_updates"]:
  if elapsed()>=CONFIG["maximum_additional_seconds"]-30:stop="additional_wall_budget";break
  index,BS_ids=sampler.draw();row=ROWS[index]
  model.train();opt.zero_grad(set_to_none=True)
  value=renderers[index](model,BS_ids,[0,1]);value.retain_grad()
  truth=J.label_subset(targets[index],BS_ids,[0,1])
  loss=.5*(value-truth).abs().square().mean()
  if not bool(torch.isfinite(loss)):raise FloatingPointError("Nonfinite native loss")
  loss.backward()
  if not all(p.grad is None or bool(torch.isfinite(p.grad).all())for p in model.parameters()):raise FloatingPointError("Nonfinite gradient")
  measured=(added+1)%CONFIG["monitor_every"]==0
  gradient=U.gradient_details(model,value,truth)if measured else None
  norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
  matched=U.statistics(value.detach(),truth);opt.step();added+=1;total+=1
  emit(dict(event="train",v8_updates=added,total_optimizer_step=total,position_index=index,row_id=row,
    BS_spatial_ids=BS_ids,UE_ids=[0,1],all192_frequency_labels=True,native_loss=float(loss.detach()),
    matched_supervised=matched,gradient_norm=float(norm),gradient_diagnostic=gradient,elapsed_seconds=elapsed()))
  if measured:
   U.check_sources(source);pred,stats=evaluate(model,renderers,targets)
   stable=stable+1 if gate(stats)else 0
   event=dict(event="monitor",v8_updates=added,total_optimizer_step=total,stats=stats,
     stable_monitors=stable,per_position_BS_count_min=sampler.counts.min(1).tolist(),
     per_position_BS_count_max=sampler.counts.max(1).tolist(),LR=[g["lr"]for g in opt.param_groups],
     elapsed_seconds=elapsed())
   history.append(event);emit(event);C.atomic_json(REPORT/"progress.json",event)
   if stable>=CONFIG["stable_monitors_required"]:stop="two_position_stable_fit";break
  if added==CONFIG["convergence_after_update"]and not lowLR:
   for group in opt.param_groups:group["lr"]*=CONFIG["convergence_LR_multiplier"]
   lowLR=True;emit(dict(event="scheduled_LR_reduction",v8_updates=added,
                       LR=[g["lr"]for g in opt.param_groups],elapsed_seconds=elapsed()))
  if measured:
   C.atomic_torch_save(last,snapshot(model,opt,sampler,added,total,stable,history,source,identity,
      infos,elapsed(),lowLR,pred))
 pred,stats=evaluate(model,renderers,targets)
 C.atomic_npy(ART/"final_prediction.npy",(pred*data.scale).cpu().numpy(),replace=True)
 C.atomic_json(ART/"axes.json",dict(row_ids=ROWS,BS_ids=list(range(128)),UE_ids=[0,1],
   frequency_ids=list(range(192)),shape=[2,128,2,192],raw_H_units=True,site=0,field=[0,0,0],
   capability="two training positions, one polarization field; no generalization/full8field claim"))
 final=ART/"final.pt"
 C.atomic_torch_save(final,snapshot(model,opt,sampler,added,total,stable,history,source,identity,infos,elapsed(),lowLR,pred))
 saved=torch.load(final,map_location="cpu",weights_only=False)
 clean=V.build(data);N.freeze_NN_frequency(clean);N.expand_head_rows(clean);clean.load_state_dict(saved["model"])
 reloaded,_=evaluate(clean,renderers,targets)
 reload_error=float((reloaded-pred).norm()/pred.norm().clamp_min(1e-30))
 if reload_error>1e-6:raise RuntimeError("Two-position clean reload mismatch")
 U.check_sources(source)
 result=dict(status="complete",stop_reason=stop,v8_new_updates=added,total_optimizer_step=total,
   two_position_field_fit=stable>=CONFIG["stable_monitors_required"],stable_monitors=stable,
   final_stats=stats,history=history,additional_elapsed_seconds=elapsed(),
   scheduled_LR_reduction_applied=lowLR,final_LR=[g["lr"]for g in opt.param_groups],
   final_reload_relative_L2=reload_error,config=CONFIG,source_identity=source,input_identity=identity,GPU=gpu,
   per_position_BS_seen=sampler.counts.tolist(),other_fields_untrained=True,
   no_validation_or_outer_read=True,no_formal_test=True,official_scoring=False,V16_changed=False,
   raw_prediction_sha256=C.sha256(ART/"final_prediction.npy"),
   raw_labels_sha256=C.sha256(ART/"raw_FIT_field_labels.npy"))
 C.atomic_json(REPORT/"result.json",result);emit(dict(event="complete",**result))
 signal.setitimer(signal.ITIMER_REAL,0)


if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--mode",choices=["qa","run"],default="qa")
 args=parser.parse_args()
 try:prepare(args.mode)
 except BaseException:
  REPORT.mkdir(parents=True,exist_ok=True)
  C.atomic_json(REPORT/("failure_"+args.mode+".json"),dict(traceback=__import__("traceback").format_exc()))
  raise
