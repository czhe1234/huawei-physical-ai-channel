"""Explicit, resumable future development entry; never invoked by correctness."""
from __future__ import annotations
import argparse,hashlib,json,os,re,time,traceback
from pathlib import Path
import numpy as np
import torch
from measured.hall_adapters.common import save_json,sha256
from .config import Config,WORK,REPORT,SOURCE,ALLOWED_GPUS
from .data import SouthDevelopment,slice_prepared
from .cache import convergence_evidence
from .foundation import prepare
from .model import HallStructured
from .observables import FitPower,losses,metrics
def atomic_torch(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".partial.pt");torch.save(value,tmp);tmp.replace(path)
def atomic_npz(path,**value):
    tmp=path.with_name(path.name+".partial")
    with tmp.open("wb") as f:np.savez_compressed(f,**value)
    tmp.replace(path)
def validate_args(args):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}",args.name):raise ValueError("Use a simple run name, without folders")
    if args.epochs<1 or args.batch<1 or args.patience<1 or args.lr<=0 or args.max_minutes<=0:raise ValueError("Invalid training bound")
    if args.measurement_only and (args.geometry_cache or args.convergence_report):raise ValueError("Measurement-only runs do not use a geometry cache")
    if args.convergence_report and not args.geometry_cache:raise ValueError("Convergence evidence must identify a new geometry cache")
def training_device(name):
    if name=="cpu":return torch.device("cpu")
    visible=os.environ.get("CUDA_VISIBLE_DEVICES","").split(",")
    if len(visible)!=1 or visible[0] not in ALLOWED_GPUS:raise RuntimeError("Set exactly one permitted full GPU UUID before Python starts")
    if torch.cuda.device_count()!=1:raise RuntimeError("Unexpected CUDA mapping")
    return torch.device("cuda:0")
def source_identity():return {p.name:sha256(p) for p in sorted(SOURCE.glob("*.py"))}
def contract_hash(contract):return hashlib.sha256(json.dumps(contract,sort_keys=True,allow_nan=False).encode()).hexdigest()
@torch.no_grad()
def evaluate(model,data,truth,power,batch=4):
    model.eval();outputs=[]
    for first in range(0,len(data.row_ids),batch):
        block=slice_prepared(data,np.arange(first,min(first+batch,len(data.row_ids)))).to(model.power_reference.device)
        outputs.append(model(block)["channel"].detach().cpu())
    pred=torch.cat(outputs)
    return pred,metrics(pred,torch.tensor(np.array(truth),dtype=torch.complex64),power.to("cpu"))
def rng_state(device):
    return dict(torch=torch.get_rng_state(),numpy=np.random.get_state(),
        cuda=torch.cuda.get_rng_state() if device.type=="cuda" else None)
def restore_rng(state,device):
    torch.set_rng_state(state["torch"].cpu());np.random.set_state(state["numpy"])
    if device.type=="cuda" and state["cuda"] is not None:torch.cuda.set_rng_state(state["cuda"].cpu())
def run(args):
    validate_args(args);device=training_device(args.device)
    cfg=Config(use_geometry=not args.measurement_only)
    reader=SouthDevelopment(allow_validation=True,geometry_cache=args.geometry_cache);bank=reader.fit()
    qf=reader.query("fit");qv=reader.query("validation")
    gf=reader.geometry(qf.row_ids) if cfg.use_geometry else None
    gv=reader.geometry(qv.row_ids) if cfg.use_geometry else None
    evidence=None
    if cfg.use_geometry and args.convergence_report:
        evidence=convergence_evidence(args.convergence_report,gf,reader.data.identity,qf.row_ids)
        convergence_evidence(args.convergence_report,gv,reader.data.identity,qv.row_ids)
    provisional=bool(cfg.use_geometry and not (evidence and gf.quality.get("ray_convergence_certified") and gv.quality.get("ray_convergence_certified")))
    if provisional and not args.allow_provisional_geometry:
        raise RuntimeError("Geometry lacks matching convergence evidence. Only an explicitly provisional development run is available; no automatic RT or formal promotion.")
    folder=WORK/"training"/args.name
    if folder.exists() and not args.resume:raise FileExistsError("Use a fresh name, or explicitly resume the same interrupted run")
    if args.resume and not (folder/"last.pt").exists():raise FileNotFoundError("No resumable epoch checkpoint")
    torch.set_num_threads(8);torch.manual_seed(args.seed);np.random.seed(args.seed)
    fit=prepare(bank,qf,cfg.subbands,"block",gf);val=prepare(bank,qv,cfg.subbands,"self",gv)
    power=FitPower.from_fit(bank.channel,bank.row_ids)
    model=HallStructured(bank.spec,power,cfg).to(device);model.set_fit_statistics(fit.to(device))
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr,weight_decay=1e-4)
    y=torch.tensor(np.array(bank.channel),dtype=torch.complex64);yv=reader.reference("validation")
    contract=dict(config=cfg.to_dict(),source_sha256=source_identity(),data_identity=reader.data.identity,
        fit_rows=qf.row_ids.tolist(),validation_rows=qv.row_ids.tolist(),max_epochs=args.epochs,
        patience=args.patience,selection="final-H structural loss; angle/spatial/frequency equally weighted",
        minimum_selection_improvement=2e-4,seed=args.seed,lr=args.lr,batch=args.batch,
        fit_donor_exclusion="same 4m block",validation_donors="declared FIT bank only",
        geometry=gf.quality if gf is not None else None,convergence_evidence=evidence,
        provisional_geometry=provisional,formal_test=False,north=False)
    identity=contract_hash(contract);curve=[];best=float("inf");stale=0;steps=0;first=0
    if args.resume:
        old=json.loads((folder/"contract.json").read_text())
        if old!=contract:raise ValueError("Resume source/data/config identity mismatch")
        state=torch.load(folder/"last.pt",map_location=device,weights_only=False)
        if state["contract_sha256"]!=identity or state["complete"]:raise ValueError("Cannot resume a changed or completed run")
        model.load_state_dict(state["state_dict"]);optimizer.load_state_dict(state["optimizer"])
        curve=state["curve"];best=state["best"];stale=state["stale"];steps=state["steps"];first=state["epoch"]+1
        restore_rng(state["rng"],device)
    else:
        folder.mkdir(parents=True);save_json(folder/"contract.json",contract)
    start=time.monotonic()
    for epoch in range(first,args.epochs+1):
        if epoch:
            model.train()
            for ids in torch.randperm(len(fit.row_ids)).split(args.batch):
                if time.monotonic()-start>args.max_minutes*60:raise TimeoutError("Wall-time bound; resume from the last complete epoch")
                block=slice_prepared(fit,ids.numpy()).to(device);target=y[ids].to(device)
                optimizer.zero_grad(set_to_none=True);output=model(block)
                terms=losses(output["channel"],target,power.to(device))
                objective=terms["structural"]+cfg.power_loss_weight*terms["power"]+cfg.residual_loss_weight*output["residual"]
                objective.backward()
                if not torch.isfinite(objective) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError("Nonfinite structured training")
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],2.)
                optimizer.step();steps+=1
        pf,mf=evaluate(model,fit,bank.channel,power,args.batch);pv,mv=evaluate(model,val,yv,power,args.batch)
        row=dict(epoch=epoch,steps=steps,fit=mf,validation=mv,invocation_seconds=time.monotonic()-start);curve.append(row)
        value=mv["structural_selection"]
        if value<best-2e-4:
            best=value;stale=0
            atomic_torch(folder/"best.pt",dict(state_dict=model.state_dict(),config=cfg.to_dict(),epoch=epoch,
                contract_sha256=identity,power_fit_ids=power.fit_ids,data_identity=reader.data.identity))
            atomic_npz(folder/"best_predictions.npz",fit=pf.numpy(),validation=pv.numpy(),fit_ids=fit.row_ids,validation_ids=val.row_ids)
        else:stale+=1
        complete=bool(epoch==args.epochs or (epoch>=args.patience and stale>=args.patience))
        atomic_torch(folder/"last.pt",dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),
            epoch=epoch,steps=steps,curve=curve,best=best,stale=stale,rng=rng_state(device),
            contract_sha256=identity,complete=complete))
        save_json(folder/"trajectory.json",dict(complete=complete,curve=curve,steps=steps,test=False,north=False,source_access=reader.access))
        print(json.dumps(row,allow_nan=False),flush=True)
        if complete:break
        if time.monotonic()-start>args.max_minutes*60:raise TimeoutError("Wall-time bound; resume from the last complete epoch")
def parser():
    p=argparse.ArgumentParser(description="Explicit development training; not part of implementation correctness")
    p.add_argument("--name",required=True);p.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    p.add_argument("--resume",action="store_true");p.add_argument("--geometry-cache",type=Path)
    p.add_argument("--convergence-report",type=Path)
    p.add_argument("--measurement-only",action="store_true");p.add_argument("--allow-provisional-geometry",action="store_true")
    p.add_argument("--epochs",type=int,default=120);p.add_argument("--patience",type=int,default=25)
    p.add_argument("--lr",type=float,default=3e-4);p.add_argument("--batch",type=int,default=4)
    p.add_argument("--seed",type=int,default=20261024);p.add_argument("--max-minutes",type=float,default=45)
    return p
if __name__=="__main__":
    args=parser().parse_args()
    try:run(args)
    except BaseException:
        REPORT.mkdir(parents=True,exist_ok=True)
        safe_name=args.name if re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}",args.name) else "invalid_name"
        save_json(REPORT/("training_entry_failure_"+safe_name+".json"),dict(traceback=traceback.format_exc()))
        raise
