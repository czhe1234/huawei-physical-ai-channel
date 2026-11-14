"""v3 geometry-only small-FIT panel health; no validation or outer scoring."""
import json,math,time
import numpy as np
import torch
from .common import FoldData,seed_all,device_guard,atomic_json
from .newrf_runner import GPU_UUID,GEOMETRY,FitGeometryBank,FitDirectionCache,subset_forward,scalar_target,weighted_loss
from .newrf_conditioned_v3 import build_conditioned_model,output_paths_v3

def main():
    torch.set_num_threads(4);seed_all(42)
    report=output_paths_v3("newrf",0,"multi_FIT_health")["report"]
    if (report/"results.json").exists():raise FileExistsError("Preserve health")
    data=FoldData(0,"screen",method="newrf")
    bank=FitGeometryBank(GEOMETRY,data.inputs,outer=0,role="screen_fit");cache=FitDirectionCache(bank)
    seed_all(42);model=build_conditioned_model(data,cache)
    guard=device_guard(GPU_UUID);torch.backends.cuda.matmul.allow_tf32=False
    model=model.to("cuda:0");cache.to("cuda:0")
    panel=[]
    for site in (0,1):
        ids=data.fit_ids[data.inputs.site_id[data.fit_ids]==site]
        order=np.lexsort((data.inputs.positions[ids,1],data.inputs.positions[ids,0]))
        panel.extend(ids[order[np.linspace(0,len(ids)-1,8,dtype=int)]].tolist())
    panel=np.asarray(panel,np.int64);ss=np.linspace(0,127,8,dtype=np.int64);ff=np.array([0,63,127,191],np.int64)
    targets=[[scalar_target(data,np.array([row]),field,ss,ff,"cuda:0") for row in panel] for field in range(8)]
    def diagnostic():
        out=[]
        with torch.no_grad():
            for field in range(8):
                err=power=energy=0.;zeros=total=0
                for i,row in enumerate(panel):
                    h,_=subset_forward(model,cache,data,np.array([row]),field,ss,ff);t=targets[field][i]
                    err+=float((h-t).abs().square().sum());power+=float(h.abs().square().sum())
                    energy+=float(t.abs().square().sum());zeros+=int((h.abs()==0).sum());total+=h.numel()
                out.append(dict(field=field,nmse=err/max(energy,1e-30),power_ratio=power/max(energy,1e-30),
                                zero_fraction=zeros/total,target_energy=energy))
        return out
    initial=diagnostic();optimizer=torch.optim.Adam(model.parameters(),lr=5e-4,weight_decay=0)
    rng=np.random.default_rng(42);history=[];zero=[0]*8;losses=[];start=time.monotonic()
    for step in range(10000):
        field=step%8;i=int(rng.integers(len(panel)));row=int(panel[i])
        pred,hc=subset_forward(model,cache,data,np.array([row]),field,ss,ff)
        loss,_,_=weighted_loss(pred,hc,targets[field][i],data.scale)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        gn=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        if float(gn)==0:zero[field]+=1
        if not math.isfinite(float(loss)) or not math.isfinite(float(gn)):raise FloatingPointError("v3 multiFIT nonfinite")
        optimizer.step();losses.append(float(loss))
        if (step+1)%2500==0:
            diag=diagnostic();event=dict(step=step+1,fields=diag,loss_recent=float(np.mean(losses[-2500:])))
            history.append(event);atomic_json(report/"progress.json",event);print(json.dumps(event),flush=True)
        if time.monotonic()-start>260:break
    final=diagnostic()
    passed=all(v["nmse"]<.9 and v["power_ratio"]>.01 and v["zero_fraction"]<.01 for v in final)
    result=dict(status="passed" if passed else "needs_diagnosis",optimizer_steps=step+1,wall_seconds=time.monotonic()-start,
          panel=panel.tolist(),panel_basis="geometry-only16FIT evenly spread across2sites",all_rows_kept=True,
          axes=dict(BS_spatial=ss.tolist(),frequency=ff.tolist()),initial=initial,history=history,final=final,
          zero_gradient_batches_by_field=zero,GPU=guard,native_loss_unchanged=True,ray_sample_budgets_unchanged=True,
          validation_truth_reads=0,outer_truth_reads=0,not_formal_checkpoint=True)
    atomic_json(report/"results.json",result);print(json.dumps(result),flush=True)
if __name__=="__main__":main()

