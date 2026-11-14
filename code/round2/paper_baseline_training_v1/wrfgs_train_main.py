"""Screen/refit WRF-GS+ power adaptation; outer truth is read only by root scorer."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from project_paths import PATHS
from round2.paper_baseline_training_v1 import common
from round2.paper_baseline_training_v1.common import (
    FoldData,output_paths,seed_all,device_guard,evaluate,atomic_json,
    atomic_torch_save,save_outer_prediction,sha256,
)
from round2.paper_baseline_training_v1.wrfgs_model import BatchedWRFGS,power_loss
from round2.paper_baseline_training_v1.wrfgs_fast_ops import install

GPU_UUID="GPU-70b6fceb-c7c4-9467-3c81-097c9284c160"


def sources():
    here=Path(__file__).resolve()
    paths=[here,here.with_name("wrfgs_model.py"),here.with_name("wrfgs_fast_ops.py"),Path(common.__file__)]
    return {PATHS.relative(p):sha256(p) for p in paths}


def make_model(data,args,seed):
    model=BatchedWRFGS(data.geometry,data.positions(data.fit_ids),count=args.gaussians,seed=seed).cuda()
    model.pair_batch=args.batch_positions*args.frequency_batch
    return model


def step(model,opt,data,rows,fids):
    h=torch.as_tensor(data.read_fit(rows,fids),device="cuda:0",dtype=torch.complex64)
    target=torch.fft.fft2(h.reshape(len(rows),2,16,8,4,len(fids)),dim=(2,3)).abs().square().reshape(len(rows),256,4,len(fids))
    model.train();opt.zero_grad(set_to_none=True)
    prediction=model(data.positions(rows),data.sites(rows),data.frequencies(fids))
    loss,detail=power_loss(prediction,target)
    if not bool(torch.isfinite(loss)):raise FloatingPointError("Nonfinite loss")
    loss.backward()
    grad=torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True,foreach=True)
    opt.step();model.constrain()
    return float(loss.detach()),detail,float(grad)


def predict(model,data,rows):
    model.eval()
    with torch.no_grad():
        return model(data.positions(rows),data.sites(rows),data.frequencies())


def checkpoint(model,opt,rng,config,data,current,best_step,best_pas,history,source_id):
    return dict(schema="wrfgs-role-training-checkpoint-v1",model=model.state_dict(),optimizer=opt.state_dict(),
      numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
      config=config,source_identity=source_id,data=data.info(),step=current,best_step=best_step,
      best_pas=best_pas,history=history)


def train_role(args,outer,role,fixed_steps=None):
    data=FoldData(outer,role,"wrfgs")
    seed=args.seed+outer*1000
    config=dict(base_seed=args.seed,fold_seed=seed,outer=outer,role=role,
      gaussian_count=args.gaussians,batch_positions=args.batch_positions,frequency_batch=args.frequency_batch,
      min_steps=args.min_steps,max_steps=args.max_steps,validation_interval=args.validation_interval,
      patience=args.patience,min_delta=args.min_delta,fixed_steps=fixed_steps,learning_rates="author-derived_constant_group_rates",
      validation_probe_ids=data.probe_ids.tolist(),fit_ids_sha256=common.digest_ids(data.fit_ids),
      loss_contract={"angular_weight":1.0,"angular":"one-minus-power-cosine along all256BS; nonzero target lane mean",
                     "log_total_weight":.05,"log_total":"SmoothL1 log1p(sum_BS power/128); all lanes including zero targets",
                     "validation":"officialPAS includes all lanes/positions"})
    paths=output_paths("wrfgs",outer,role)
    report,artifact=paths["report"],paths["artifact"]
    source_id=sources()
    if (report/"complete.json").exists():
        saved=json.loads((report/"complete.json").read_text())
        if saved["config"]!=config or saved["source_identity"]!=source_id:
            raise RuntimeError("Existing completed phase has different identity")
        return saved
    seed_all(seed);rng=np.random.default_rng(seed)
    model=make_model(data,args,seed);opt=model.optimizer()
    latest=artifact/"latest.pt";best=artifact/"best.pt"
    current=0;best_step=0;best_pas=-1.;significant_pas=-1.;history=[];stale=0
    if latest.exists():
        ck=torch.load(latest,map_location="cpu",weights_only=True)
        if ck["config"]!=config or ck["source_identity"]!=source_id:
            raise RuntimeError("Resume config/source differs; preserve old checkpoint")
        model.load_state_dict(ck["model"]);opt.load_state_dict(ck["optimizer"])
        rng.bit_generator.state=ck["numpy_rng"];torch.set_rng_state(ck["torch_rng"])
        torch.cuda.set_rng_state(ck["cuda_rng"])
        current=ck["step"];best_step=ck["best_step"];best_pas=ck["best_pas"];history=ck["history"]
        stale=ck.get("stale_checks",0);significant_pas=ck.get("significant_pas",best_pas)
    ceiling=args.max_steps if role=="screen" else int(fixed_steps)
    losses=[];gradients=[];wall=time.monotonic();last=time.monotonic()
    atomic_json(report/"run_contract.json",dict(config=config,source_identity=source_id,model=model.metadata(),
      normalization_scale=data.scale,data=data.info(),outer_truth_access=False))
    while current<ceiling:
        rows=rng.choice(data.fit_ids,size=args.batch_positions,replace=False)
        fids=np.sort(rng.choice(192,size=args.frequency_batch,replace=False)).astype(np.int64)
        loss,detail,grad=step(model,opt,data,rows,fids)
        current+=1;losses.append(loss);gradients.append(grad)
        check=current%args.validation_interval==0 or current==ceiling
        if not check:continue
        torch.cuda.synchronize()
        metric=None
        if role=="screen":
            metric=evaluate(data,lambda ids:predict(model,data,ids),batch_size=1,power_only=True)
            if metric["pas"]>best_pas:
                best_pas=float(metric["pas"]);best_step=current
                atomic_torch_save(best,dict(schema="wrfgs-selected-screen-model-v1",model=model.state_dict(),
                  config=config,step=current,selection_metrics=metric,source_identity=source_id,data=data.info()))
            if metric["pas"]>significant_pas+args.min_delta:
                significant_pas=float(metric["pas"]);stale=0
            else:stale+=1
        item=dict(step=current,loss=float(np.mean(losses)),last_loss=loss,gradient_norm_mean=float(np.mean(gradients)),
          validation=metric,best_step=best_step,best_pas=best_pas if role=="screen" else None,
          elapsed_seconds=time.monotonic()-wall,interval_seconds=time.monotonic()-last)
        history.append(item);losses=[];gradients=[];last=time.monotonic()
        ck=checkpoint(model,opt,rng,config,data,current,best_step,best_pas,history,source_id)
        ck["stale_checks"]=stale;ck["significant_pas"]=significant_pas
        atomic_torch_save(latest,ck)
        atomic_json(report/"progress.json",dict(status="running",outer=outer,role=role,step=current,ceiling=ceiling,
          best_step=best_step,best_pas=best_pas if role=="screen" else None,history=history[-2:]))
        print(json.dumps(dict(method="wrfgs",outer=outer,role=role,**item)),flush=True)
        if role=="screen" and current>=args.min_steps and stale>=args.patience:break
    if role=="screen":
        if not best.exists():raise RuntimeError("No selected trained screen model")
        ck=torch.load(best,map_location="cpu",weights_only=True);model.load_state_dict(ck["model"])
        full_validation=evaluate(data,lambda ids:predict(model,data,ids),ids=data.val_ids,batch_size=1,power_only=True)
        final_model=best;selected_steps=int(ck["step"])
    else:
        final_model=artifact/"final.pt";selected_steps=current;full_validation=None
        atomic_torch_save(final_model,dict(schema="wrfgs-refit-final-model-v1",model=model.state_dict(),step=current,
          config=config,source_identity=source_id,data=data.info()))
    result=dict(status="complete",outer=outer,role=role,config=config,source_identity=source_id,
      completed_updates=current,selected_updates=selected_steps,selected_probe_pas=best_pas if role=="screen" else None,
      full_validation_diagnostic=full_validation,history=history,wall_seconds=time.monotonic()-wall,
      model_path=PATHS.relative(final_model),model_sha256=sha256(final_model),model_metadata=model.metadata(),data=data.info(),
      outer_truth_read=False,full_H=False,
      validation_plateau_reached=(role=="screen" and current>=args.min_steps and stale>=args.patience),
      stop_reason=("fixed_refit_budget" if role=="refit" else
                   "validation_plateau" if current>=args.min_steps and stale>=args.patience else "maximum_updates_no_plateau"))
    if role=="refit":
        raw=[]
        for lo in range(0,len(data.outer_ids),args.predict_positions):
            rows=data.outer_ids[lo:lo+args.predict_positions]
            value=predict(model,data,rows).detach().cpu().numpy()
            raw.append(value*(data.scale**2))
        prediction=save_outer_prediction(data,"wrfgs",np.concatenate(raw),power_only=True,
          metadata=dict(model_path=result["model_path"],model_sha256=result["model_sha256"],source_identity=source_id,
                        training_config=config,model=model.metadata(),selected_screen_steps=args._screen_selected_steps,
                        no_outer_selection=True))
        result["prediction"]=prediction
    atomic_json(report/"complete.json",result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--outer",default="all")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--gaussians",type=int,default=1024)
    parser.add_argument("--batch-positions",type=int,default=16)
    parser.add_argument("--frequency-batch",type=int,default=4)
    parser.add_argument("--min-steps",type=int,required=True)
    parser.add_argument("--max-steps",type=int,required=True)
    parser.add_argument("--validation-interval",type=int,required=True)
    parser.add_argument("--patience",type=int,required=True)
    parser.add_argument("--min-delta",type=float,default=.0005)
    parser.add_argument("--predict-positions",type=int,default=1)
    args=parser.parse_args()
    try:
        device=device_guard(GPU_UUID);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        fast=install()
        folds=list(range(5)) if args.outer=="all" else [int(args.outer)]
        completed=[]
        for outer in folds:
            screen=train_role(args,outer,"screen")
            args._screen_selected_steps=screen["selected_updates"]
            n_screen=len(FoldData(outer,"screen","wrfgs").fit_ids)
            n_refit=len(FoldData(outer,"refit","wrfgs").fit_ids)
            budget=int(math.ceil(screen["selected_updates"]*n_refit/n_screen))
            refit=train_role(args,outer,"refit",fixed_steps=budget)
            completed.append(dict(outer=outer,screen=screen["model_path"],refit=refit["model_path"],
                                  prediction=refit["prediction"]["path"],selected_updates=screen["selected_updates"],refit_updates=budget))
        atomic_json(output_paths("wrfgs")["report"]/"method_complete.json",
                    dict(status="predictions_ready",folds=completed,device=device,fast_ops=fast,
                         outer_scored=False,trained=True,full_H=False,eligible_metrics=["pas"]))
    except BaseException:
        p=output_paths("wrfgs")["report"]/"main_failure.txt"
        with p.open("a") as f:f.write(traceback.format_exc()+"\n")
        raise

if __name__=="__main__":main()
