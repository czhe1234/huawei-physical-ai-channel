"""Train only the south development path-state core; no formal-test entry."""
from __future__ import annotations
import argparse,time,json,traceback
import numpy as np
import torch
from .common import *
from .method import DevelopedV16
from .train_baselines import atomic_torch
from measured.hall_adapters.core import batch_from_support,render_endpoints,synthesize
from measured.hall_adapters.metrics import training_power_loss,torch_directional_power
def slice_batch(data,ids):
    values={k:v[ids] for k,v in data.items()}
    k=max(1,int(values["mask"].sum(-1).max()))
    for key in ["base","sequence_geometry","interactions","depth","coordinate","base_log_mass","mask"]:values[key]=values[key][:,:k]
    return values
@torch.no_grad()
def evaluate(model,data,ids,batch=8,channels=False):
    model.eval();out=[];gates=[];energy=[];power_error=[]
    for group in ids.split(batch):
        d=slice_batch(data,group);p,state=model(d);gates.append(state["gate"].detach().cpu())
        if channels:
            h,_=model(d,full_channel=True)
            actual=torch_directional_power(h).reshape(len(h),32)
            norm=actual/actual.sum(-1,keepdim=True).clamp_min(1e-20)
            power_error.append(float((norm-p).abs().max()))
            before=d["channel"].abs().square().sum((-3,-2,-1));after=h.abs().square().sum((-3,-2,-1))
            energy.append(float(((after-before).abs()/before.clamp_min(1e-20)).max()))
            out.append(h.detach().cpu())
        else:out.append(p.detach().cpu())
    result=torch.cat(out).numpy();gate=torch.cat(gates).numpy()
    return result,dict(gate_mean=float(gate.mean()),gate_min=float(gate.min()),gate_max=float(gate.max()),
       maximum_energy_relative_error=max(energy,default=0.),maximum_power_writeback_error=max(power_error,default=0.))
def train(name,config,data,target,nfit,tiny=False):
    start=time.monotonic();seed();model=DevelopedV16(config["bounds"],config["gate"]).to(target.device)
    fit_ids=torch.arange(16 if tiny else nfit,device=target.device)
    val_ids=fit_ids if tiny else torch.arange(nfit,len(target),device=target.device)
    model.set_fit_statistics(slice_batch(data,fit_ids))
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config["lr"],weight_decay=.0001)
    folder=WORK/"models"/name;folder.mkdir(parents=True,exist_ok=True);report=REPORT/(name+".json")
    if report.exists() and json.loads(report.read_text()).get("complete"):return json.loads(report.read_text())
    initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    gradients={n:0. for n in initial};curve=[];best=-1.;best_epoch=0;stale=0;steps=0;max_norm=0.
    yf=target[fit_ids].detach().cpu().numpy().reshape(-1,2,32);yv=target[val_ids].detach().cpu().numpy().reshape(-1,2,32)
    def record(epoch,loss=None):
        nonlocal best,best_epoch,stale
        pf,gf=evaluate(model,data,fit_ids);pv,gv=evaluate(model,data,val_ids)
        mf,mv=metrics(pf.reshape(-1,2,32),yf),metrics(pv.reshape(-1,2,32),yv)
        row=dict(epoch=epoch,steps=steps,loss=loss,fit=mf,validation=mv,fit_gate=gf,validation_gate=gv,seconds=time.monotonic()-start);curve.append(row)
        value=mf["mean_direction_cosine"] if tiny else mv["mean_direction_cosine"]
        if value>best+.0002:
            best=value;best_epoch=epoch;stale=0
            atomic_torch(folder/"best.pt",dict(state_dict=model.state_dict(),config=config,epoch=epoch,steps=steps))
            np.savez_compressed(folder/"best_power.npz",fit=pf.reshape(-1,2,32),validation=pv.reshape(-1,2,32))
        else:stale+=5
        save_json(REPORT/(name+"_trajectory.json"),dict(name=name,complete=False,curve=curve))
        print("CORE",name,json.dumps(row),flush=True)
    record(0)
    for epoch in range(1,(300 if tiny else 180)+1):
        model.train();losses=[]
        order=fit_ids[torch.randperm(len(fit_ids),device=target.device)]
        for ids in order.split(8):
            batch=slice_batch(data,ids);optimizer.zero_grad(set_to_none=True)
            pred,state=model(batch);loss=training_power_loss(pred,target[ids])+model.regularizer(state,batch["mask"])
            if not torch.isfinite(loss):raise FloatingPointError("Nonfinite core loss")
            loss.backward()
            for n,p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    if not torch.isfinite(p.grad).all():raise FloatingPointError("Nonfinite core gradient:"+n)
                    gradients[n]=max(gradients[n],float(p.grad.detach().abs().max()))
            max_norm=max(max_norm,float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],2.)))
            optimizer.step();steps+=1;losses.append(float(loss.detach()))
        if epoch%5==0:record(epoch,float(np.mean(losses)))
        if not tiny and epoch>=40 and stale>=30:break
        if tiny and steps>=600:break
        if time.monotonic()-start>45*60:raise TimeoutError("Bounded core stage exceeded45 minutes")
    atomic_torch(folder/"last.pt",dict(state_dict=model.state_dict(),config=config,epoch=epoch,steps=steps))
    change={n:float((p.detach().cpu()-initial[n]).abs().max()) for n,p in model.named_parameters() if p.requires_grad}
    checkpoint=torch.load(folder/"best.pt",map_location=target.device,weights_only=False);model.load_state_dict(checkpoint["state_dict"])
    pf,gf=evaluate(model,data,fit_ids);pv,gv=evaluate(model,data,val_ids)
    hf,ef=evaluate(model,data,fit_ids,channels=True);hv,ev=evaluate(model,data,val_ids,channels=True)
    # Full1024-bin complex output must match the exact optimized broadband power.
    if max(ef["maximum_power_writeback_error"],ev["maximum_power_writeback_error"])>3e-5:raise ValueError("Core power/channel mismatch")
    if max(ef["maximum_energy_relative_error"],ev["maximum_energy_relative_error"])>5e-6:raise ValueError("Core synthesis energy drift")
    np.savez_compressed(folder/"best_channel.npz",fit=hf.reshape(-1,2,8,4,1024),validation=hv.reshape(-1,2,8,4,1024))
    mf,mv=metrics(pf.reshape(-1,2,32),yf),metrics(pv.reshape(-1,2,32),yv)
    reduction=1-(1-curve[-1]["fit"]["mean_direction_cosine"])/(1-curve[0]["fit"]["mean_direction_cosine"]+1e-20)
    result=dict(name=name,complete=True,tiny=tiny,config=config,epochs=epoch,steps=steps,best_epoch=best_epoch,
       best_fit=mf,best_validation=mv,curve=curve,fit_gate=gf,validation_gate=gv,writeback_fit=ef,writeback_validation=ev,
       health=dict(nonfinite_gradients=0,gradient_max_by_parameter=gradients,parameter_change=change,
                   updated_tensors=sum(v>0 for v in change.values()),trainable_tensors=len(change),maximum_gradient_norm=max_norm,
                   tiny_pass=bool(tiny and (mf["mean_direction_cosine"]>=.98 or reduction>=.8)),fit_loss_reduction_fraction=reduction),
       seconds=time.monotonic()-start,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,test_used=False,north_used=False)
    save_json(report,result);print("CORE_DONE",name,json.dumps({k:result[k] for k in ["epochs","steps","best_epoch","best_fit","best_validation","seconds"]}),flush=True)
    del model,optimizer;torch.cuda.empty_cache();return result
def run(tiny=False):
    seed();dev=DevelopmentData();bank=dev.fit();y=normalized_power(directional_power(bank.channel))
    yv=normalized_power(directional_power(dev.reference("validation")))
    base=np.load(WORK/"foundation.npy",mmap_mode="r")
    with np.load(WORK/"geometry_support.npz") as z:support={k:z[k] for k in ["coordinate","mass","mask","features","sequence","interaction","depth","identity"]}
    data=batch_from_support(base,support);d=device();data={k:v.to(d) for k,v in data.items()}
    target=torch.tensor(np.concatenate([y,yv]).reshape(-1,32),dtype=torch.float32,device=d)
    cfgs=[dict(bounds=[.5,.25,0.],gate=.3,lr=.0003),dict(bounds=[1.,.5,0.],gate=.6,lr=.001)]
    for i,config in enumerate(cfgs[:1] if tiny else cfgs):
        train("ours_tiny_v2" if tiny else "ours_config%d_v2"%i,config,data,target,2*len(y),tiny=tiny)
    with np.load(WORK/"rt_power.npz") as z:
        save_json(REPORT/"rt_development_scores.json",{kind:dict(fit=metrics(z[kind][:len(y)],y),validation=metrics(z[kind][len(y):],yv),convergence_certified=False) for kind in ["coherent","incoherent"]})
    dev.save_access("ours_tiny" if tiny else "ours")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--tiny",action="store_true");a=p.parse_args()
    try:run(a.tiny)
    except BaseException:
        save_json(REPORT/("ours_tiny_v2_failure.json" if a.tiny else "ours_v2_failure.json"),dict(traceback=traceback.format_exc()));raise
