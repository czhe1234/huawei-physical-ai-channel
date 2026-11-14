"""Small-scale measured fitting with health, trajectories and bounded tuning."""
from __future__ import annotations
import argparse,copy,json,math,time,traceback
from dataclasses import replace
import numpy as np
import torch
from .common import *
from .baselines import SpectralMLP
from measured.hall_structured_v2.model import HallStructured
from measured.hall_structured_v2.observables import losses,metrics
from measured.hall_structured_v2.train import rng_state,restore_rng
def configuration(profile):
    c=Config()
    if profile in ["warm","spectrum","delay","trust"]:
        c=replace(c,spectral_log_power_bound=1.4,receive_log_amplitude_bound=.5)
    if profile=="delay":c=replace(c,relative_delay_bound_s=50e-9)
    if profile=="trust":c=replace(c,geometry_gate_cap=.3)
    return c
def build(b,kind,profile,levels,hidden):
    cfg=configuration(profile)
    if kind=="no_geometry":cfg=replace(cfg,use_geometry=False)
    if kind=="mlp":return SpectralMLP(b["bank"].position,b["power"],levels,hidden),cfg
    m=HallStructured(b["bank"].spec,b["power"],cfg)
    if profile in ["warm","spectrum","delay","trust"]:
        recipe=json.loads((PATHS.analysis/"results/hall_south_development_v1/development_recipe.json").read_text())
        weights=np.asarray(recipe["foundation_weights"]).T
        with torch.no_grad():m.foundation.global_logits.copy_(torch.tensor(np.log(weights),dtype=torch.float32))
    return m,cfg
@torch.no_grad()
def evaluate(model,data,truth,power,batch):
    model.eval();pred=[];state=[]
    for first in range(0,len(data.row_ids),batch):
        block=slice_prepared(data,np.arange(first,min(first+batch,len(data.row_ids))))
        out=model(block);pred.append(out["channel"].detach().cpu())
        if "state" in out:
            s=out["state"];bound=model.state.coordinate_bound
            state.append(dict(gate=s["gate"].mean(0).cpu().tolist(),
                delay_saturation=float((s["coordinate_residual"][...,2].abs()>.9*bound[2]).float().mean()),
                spectral_saturation=float((s["spectral_delta"].abs()>.9*model.config.spectral_log_power_bound).float().mean()),
                receive_saturation=float((s["receive_log_gain"].abs()>.9*model.config.receive_log_amplitude_bound).float().mean())))
    pred=torch.cat(pred);target=torch.tensor(np.array(truth),dtype=torch.complex64)
    ms=metrics(pred,target,power.to("cpu"))
    terms=losses(pred,target,power.to("cpu"))
    ms["loss_terms"]={k:float(v) for k,v in terms.items()}
    return pred,ms,state
def checkpoint(path,model,opt,epoch,steps,curve,best,stale,identity,device,grad,complete=False):
    atomic_torch(path,dict(state_dict=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,steps=steps,
      curve=curve,best=best,stale=stale,identity=identity,rng=rng_state(device),
      gradient_max={k:float(v) for k,v in grad.items()},complete=complete))
def run(args):
    if not __import__("re").fullmatch(r"[a-zA-Z0-9_-]{1,64}",args.name):raise ValueError("Invalid run name")
    init();seed(args.seed);dev=device();start=time.monotonic()
    torch.cuda.reset_peak_memory_stats();b=bundle(args.health)
    fit=b["fit"].to(dev);val=b["val"].to(dev) if b["val"] is not None else None
    model,cfg=build(b,args.kind,args.profile,args.levels,args.hidden);model=model.to(dev);model.set_fit_statistics(fit)
    power=b["power"].to(dev);target=torch.tensor(b["y"],device=dev,dtype=torch.complex64)
    folder=WORK/"models"/args.name;report=REPORT/(args.name+".json")
    if report.exists() and json.loads(report.read_text()).get("complete"):print("REUSE",args.name,flush=True);return
    if folder.exists() and not args.resume:raise FileExistsError("Existing run protected; explicitly resume")
    folder.mkdir(parents=True,exist_ok=True)
    contract=dict(name=args.name,kind=args.kind,profile=args.profile,config=cfg.to_dict(),
      seed=args.seed,lr=args.lr,batch=args.batch,epochs=args.epochs,eval_every=args.eval_every,patience=args.patience,
      health=args.health,levels=args.levels,hidden=args.hidden,training_frequency_weight=args.frequency_weight,source_sha256=source_hashes(),
      data_identity=b["reader"].data.identity,fit_rows=fit.row_ids.tolist(),
      validation_rows=val.row_ids.tolist() if val is not None else [],provisional_geometry=True if args.kind=="ours" else None,
      targets=json.loads((REPORT/"contract.json").read_text())["targets"],selection="FIT health only" if args.health else "final-H structural validation",
      environment=environment(),initialization="previous FIT-trained expert weights" if args.profile!="default" else "uniform expert weights")
    identity=__import__("hashlib").sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr,weight_decay=1e-4)
    initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    curve=[];grad={};steps=0;best=float("inf");stale=0;first=0;best_epoch=0
    if args.resume:
        old=json.loads((folder/"contract.json").read_text())
        if old!=contract:raise ValueError("Resume identity changed")
        state=torch.load(folder/"last.pt",map_location=dev,weights_only=False)
        if state["identity"]!=identity or state["complete"]:raise ValueError("Cannot resume changed/completed run")
        model.load_state_dict(state["state_dict"]);opt.load_state_dict(state["optimizer"]);restore_rng(state["rng"],dev)
        curve=state["curve"];grad={k:torch.tensor(v,device=dev) for k,v in state["gradient_max"].items()}
        steps=state["steps"];best=state["best"];stale=state["stale"];first=state["epoch"]+1
        initial=torch.load(folder/"initial_active.pt",map_location="cpu",weights_only=False)
    else:
        save_json(folder/"contract.json",contract);atomic_torch(folder/"initial_active.pt",initial)
    reason="epoch_budget"
    for epoch in range(first,args.epochs+1):
        train_losses=[]
        if epoch:
            model.train()
            for ids in torch.randperm(len(fit.row_ids)).split(args.batch):
                if time.monotonic()-start>args.max_minutes*60:raise TimeoutError("Per-model bound; resume last complete evaluated epoch")
                block=slice_prepared(fit,ids.numpy());opt.zero_grad(set_to_none=True)
                out=model(block);terms=losses(out["channel"],target[ids].to(dev),power)
                fit_structure=(terms["angle"]+terms["spatial"]+args.frequency_weight*terms["frequency"])/(2+args.frequency_weight)
                objective=fit_structure+cfg.power_loss_weight*terms["power"]+cfg.residual_loss_weight*out["residual"]
                if not torch.isfinite(objective):raise FloatingPointError("Nonfinite objective")
                objective.backward()
                bad=[n for n,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
                if bad:
                    save_json(folder/"nonfinite.json",dict(epoch=epoch,steps=steps,rows=block.row_ids.tolist(),parameters=bad,
                        losses={k:float(v) for k,v in terms.items()}));raise FloatingPointError(str(bad))
                for n,p in model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        value=p.grad.detach().abs().max()
                        grad[n]=torch.maximum(grad.get(n,torch.zeros_like(value)),value)
                norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],2.,error_if_nonfinite=True)
                opt.step();steps+=1;train_losses.append(float(objective.detach()))
        if epoch%args.eval_every and epoch!=args.epochs:continue
        pf,mf,sf=evaluate(model,fit,b["y"],b["power"],args.batch)
        if val is None:pv=None;mv=None;sv=[]
        else:pv,mv,sv=evaluate(model,val,b["yv"],b["power"],args.batch)
        value=(mf if args.health else mv)["structural_selection"]
        row=dict(epoch=epoch,steps=steps,train_objective=float(np.mean(train_losses)) if train_losses else None,
            fit=mf,validation=mv,fit_state=sf,validation_state=sv,seconds=time.monotonic()-start)
        curve.append(row)
        if value<best-2e-4:
            best=value;best_epoch=epoch;stale=0
            atomic_torch(folder/"best.pt",dict(state_dict=model.state_dict(),epoch=epoch,steps=steps,identity=identity,config=cfg.to_dict()))
            pred=dict(fit=pf.numpy(),fit_ids=fit.row_ids)
            if pv is not None:pred.update(validation=pv.numpy(),validation_ids=val.row_ids)
            atomic_npz(folder/"best_channel.npz",**pred)
        else:stale+=args.eval_every
        checkpoint(folder/"last.pt",model,opt,epoch,steps,curve,best,stale,identity,dev,grad)
        save_json(REPORT/(args.name+"_trajectory.json"),dict(name=args.name,complete=False,curve=curve))
        print("EPOCH",args.name,epoch,"steps",steps,"fit",round(mf["structural_selection"],6),
            "val",round(mv["structural_selection"],6) if mv else None,"cos",round(mf["direction_cosine"],5),
            "sec",round(time.monotonic()-start,1),flush=True)
        if not args.health and epoch>=40 and stale>=args.patience:reason="validation_plateau";break
        if args.health:
            reduction=1-mf["structural_selection"]/max(curve[0]["fit"]["structural_selection"],1e-12)
            if epoch>=60 and (reduction>=.5 or mf["direction_cosine"]>=.98):reason="tiny_fit_gate";break
    checkpoint(folder/"last.pt",model,opt,epoch,steps,curve,best,stale,identity,dev,grad,True)
    state=torch.load(folder/"best.pt",map_location=dev,weights_only=False);model.load_state_dict(state["state_dict"]);best_epoch=state["epoch"]
    pf,mf,sf=evaluate(model,fit,b["y"],b["power"],args.batch)
    pv,mv,sv=evaluate(model,val,b["yv"],b["power"],args.batch) if val is not None else (None,None,[])
    changes={n:float((p.detach().cpu()-initial[n]).abs().max()) for n,p in model.named_parameters() if p.requires_grad}
    reduction=1-mf["structural_selection"]/max(curve[0]["fit"]["structural_selection"],1e-12)
    health=dict(finite_gradients=True,gradient_max={k:float(v) for k,v in grad.items()},
      parameter_change=changes,active_tensors=len(changes),updated_tensors=sum(x>0 for x in changes.values()),
      structural_fit_reduction=reduction,tiny_gate=bool(reduction>=.5 or mf["direction_cosine"]>=.98))
    result=dict(name=args.name,complete=True,contract=contract,curve=curve,best_epoch=best_epoch,
      best_fit=mf,best_validation=mv,health=health,fit_state=sf,validation_state=sv,epochs=epoch,steps=steps,
      stop_reason=reason,seconds=time.monotonic()-start,peak_GPU_GiB=torch.cuda.max_memory_allocated()/2**30,
      trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
      test=False,north=False,provisional_geometry=args.kind=="ours")
    save_json(report,result)
    print("DONE",args.name,json.dumps(dict(epochs=epoch,steps=steps,best_epoch=best_epoch,stop=reason,fit=mf,validation=mv,seconds=result["seconds"])),flush=True)
def parser():
    p=argparse.ArgumentParser()
    p.add_argument("--name",required=True);p.add_argument("--kind",choices=["ours","no_geometry","mlp"],default="ours")
    p.add_argument("--profile",choices=["default","warm","spectrum","delay","trust"],default="default")
    p.add_argument("--health",action="store_true");p.add_argument("--resume",action="store_true")
    p.add_argument("--epochs",type=int,default=160);p.add_argument("--eval-every",type=int,default=5)
    p.add_argument("--batch",type=int,default=16);p.add_argument("--lr",type=float,default=.0003)
    p.add_argument("--patience",type=int,default=25);p.add_argument("--seed",type=int,default=SEED)
    p.add_argument("--frequency-weight",type=float,default=1.)
    p.add_argument("--levels",type=int,default=6);p.add_argument("--hidden",type=int,default=512)
    p.add_argument("--max-minutes",type=float,default=15)
    return p
if __name__=="__main__":
    args=parser().parse_args()
    try:run(args)
    except BaseException:
        REPORT.mkdir(parents=True,exist_ok=True)
        save_json(REPORT/(args.name+"_failure.json"),dict(traceback=traceback.format_exc()));raise
