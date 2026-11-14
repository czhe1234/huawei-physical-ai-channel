"""Bounded learnable-baseline fitting with complete development trajectories."""
from __future__ import annotations
import argparse,time,json,traceback,copy
import numpy as np
import torch
from .common import *
from .wrf_batched import BatchedWRF
from measured.hall_adapters.baselines import RBFPower,PositionPowerMLP,DirectPointing
from measured.hall_adapters.metrics import training_power_loss
def atomic_torch(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(".partial.pt");torch.save(value,tmp);tmp.replace(path)
@torch.no_grad()
def evaluate(model,x,batch=4):
    model.eval();return torch.cat([model(x[i:i+batch]).detach().cpu() for i in range(0,len(x),batch)]).numpy()
def fit_model(name,model,optimizer,x,y,xv,yv,max_epochs=180,batch=4,tiny=False,step_cap=None):
    folder=WORK/"models"/name;folder.mkdir(parents=True,exist_ok=True)
    report=REPORT/(name+".json")
    if report.exists() and json.loads(report.read_text()).get("complete"):
        print("REUSE",name,flush=True);return json.loads(report.read_text())
    start=time.monotonic();curve=[];best=-1.;best_epoch=0;stale=0;steps=0
    initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    grad_seen={n:0. for n in initial};group_grad={};max_gradient=0.;bad=0
    target=y.detach().cpu().numpy();targetv=yv.detach().cpu().numpy()
    def record(epoch,trainloss=None):
        nonlocal best,best_epoch,stale
        pf=evaluate(model,x,batch);pv=evaluate(model,xv,batch)
        mf,mv=metrics(pf,target),metrics(pv,targetv)
        row=dict(epoch=epoch,steps=steps,train_loss=trainloss,fit=mf,validation=mv,seconds=time.monotonic()-start)
        curve.append(row);value=mf["mean_direction_cosine"] if tiny else mv["mean_direction_cosine"]
        if value>best+.0002:
            best=value;best_epoch=epoch;stale=0
            atomic_torch(folder/"best.pt",dict(state_dict=model.state_dict(),epoch=epoch,steps=steps,fit=mf,validation=mv))
            np.savez_compressed(folder/"best_power.npz",fit=pf,validation=pv)
        else:stale+=5
        save_json(REPORT/(name+"_trajectory.json"),dict(name=name,complete=False,curve=curve))
        print("TRAIN",name,json.dumps(row),flush=True)
    record(0)
    for epoch in range(1,max_epochs+1):
        model.train();losses=[]
        for ids in torch.randperm(len(x),device=x.device).split(batch):
            optimizer.zero_grad(set_to_none=True);pred=model(x[ids]);loss=training_power_loss(pred,y[ids])
            if not torch.isfinite(loss):raise FloatingPointError("Nonfinite loss in "+name)
            loss.backward()
            for n,p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    if not torch.isfinite(p.grad).all():bad+=1;raise FloatingPointError("Nonfinite gradient "+n)
                    grad_seen[n]=max(grad_seen[n],float(p.grad.detach().abs().max()))
            norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.))
            max_gradient=max(max_gradient,norm);optimizer.step();steps+=1;losses.append(float(loss.detach()))
            if step_cap and steps>=step_cap:break
        if epoch%5==0 or epoch==max_epochs or (step_cap and steps>=step_cap):record(epoch,float(np.mean(losses)))
        if not tiny and stale>=30 and epoch>=40:break
        if step_cap and steps>=step_cap:break
        if time.monotonic()-start>45*60:raise TimeoutError("Bounded neural stage exceeded45 minutes")
    atomic_torch(folder/"last.pt",dict(state_dict=model.state_dict(),epoch=epoch,steps=steps))
    change={n:float((p.detach().cpu()-initial[n]).abs().max()) for n,p in model.named_parameters() if p.requires_grad}
    checkpoint=torch.load(folder/"best.pt",map_location=x.device,weights_only=False);model.load_state_dict(checkpoint["state_dict"])
    best_pred=evaluate(model,x,batch);best_val=evaluate(model,xv,batch)
    reduction=(1-curve[-1]["fit"]["mean_direction_cosine"])/(1-curve[0]["fit"]["mean_direction_cosine"]+1e-20)
    health=dict(nonfinite_gradients=bad,parameter_tensors=len(change),tensors_updated=sum(v>0 for v in change.values()),
       active_gradient_tensors=sum(v>0 for v in grad_seen.values()),max_gradient_norm=max_gradient,
       tiny_overfit_pass=bool(tiny and (metrics(best_pred,target)["mean_direction_cosine"]>=.98 or reduction<=.2)),
       final_fit_loss_reduction_fraction=1-reduction,
       gradients_by_parameter=grad_seen,maximum_parameter_change=change)
    result=dict(name=name,complete=True,tiny=tiny,epochs=epoch,steps=steps,best_epoch=best_epoch,curve=curve,
       best_fit=metrics(best_pred,target),best_validation=metrics(best_val,targetv),health=health,
       seconds=time.monotonic()-start,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
       model_diagnostics=getattr(model,"last_diagnostics",{}),source="pinned-author deformation and dense alpha renderer adaptation" if "wrf" in name else "shared position MLP",
       development_only=True,test_score_computed=False,north_read=False)
    save_json(report,result);print("DONE",name,json.dumps({k:result[k] for k in ["epochs","steps","best_epoch","best_fit","best_validation","seconds","peak_gpu_gib"]}),flush=True)
    return result
def run(mode):
    seed();dev=DevelopmentData();bank=dev.fit()
    truth=directional_power(bank.channel);truthv=directional_power(dev.reference("validation"))
    x=torch.tensor(np.array(bank.position),dtype=torch.float32,device=device())
    xv=torch.tensor(np.array(dev.query("validation").position),dtype=torch.float32,device=x.device)
    y=torch.tensor(normalized_power(truth),dtype=torch.float32,device=x.device)
    yv=torch.tensor(normalized_power(truthv),dtype=torch.float32,device=x.device)
    save_json(REPORT/"environment_training.json",dict(torch=torch.__version__,numpy=np.__version__,python=platform.python_version(),CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),visible_count=torch.cuda.device_count(),device=torch.cuda.get_device_name(0)))
    results=[]
    if mode=="mlp":
        grid=[]
        for length in [.5,1.,2.,4.]:
          for ridge in [.001,.01,.1]:
            m=RBFPower(bank,length,ridge);pf=m.predict(dev.query("fit"))["power"];pv=m.predict(dev.query("validation"))["power"]
            grid.append(dict(length_m=length,ridge=ridge,fit=metrics(pf,truth),validation=metrics(pv,truthv)))
        grid.sort(key=lambda r:r["validation"]["mean_direction_cosine"],reverse=True)
        save_json(REPORT/"kernel_grid.json",dict(candidates=grid,selection="highest validation mean cosine"))
        direct=DirectPointing()
        save_json(REPORT/"direct_reference.json",dict(fit=metrics(direct.predict(dev.query("fit"))["power"],truth),validation=metrics(direct.predict(dev.query("validation"))["power"],truthv)))
        for levels in [3,6]:
            seed();m=PositionPowerMLP(bank.position,levels=levels,hidden=96).to(x.device)
            optimizer=torch.optim.Adam(m.parameters(),lr=.001)
            results.append(fit_model("mlp_levels%d"%levels,m,optimizer,x,y,xv,yv,batch=16))
    elif mode in ("wrf_tiny","wrf"):
        for count in ([64] if mode=="wrf_tiny" else [64,128]):
            seed();m=BatchedWRF(bank.position,bank.spec,count).to(x.device)
            optimizer=torch.optim.Adam(m.optimizer_groups(lr=.0005),eps=1e-8)
            if mode=="wrf_tiny":
                results.append(fit_model("wrf_tiny_g%d"%count,m,optimizer,x[:8],y[:8],x[:8],y[:8],max_epochs=300,batch=4,tiny=True,step_cap=600))
            else:
                results.append(fit_model("wrf_g%d"%count,m,optimizer,x,y,xv,yv,batch=4))
            del m,optimizer;torch.cuda.empty_cache()
    dev.save_access(mode)
    return results
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--mode",choices=["mlp","wrf_tiny","wrf"],required=True);args=parser.parse_args()
    try:run(args.mode)
    except BaseException:
        save_json(REPORT/(args.mode+"_failure.json"),dict(traceback=traceback.format_exc()));raise
