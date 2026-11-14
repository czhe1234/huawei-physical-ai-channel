"""Bounded fixed-FIT optimization health after input-column initialization conditioning."""
import json,math,time
import numpy as np
import torch
from .common import FoldData,output_paths,atomic_json,device_guard,seed_all
from .newrf_runner import (GPU_UUID,GEOMETRY,FitGeometryBank,FitDirectionCache,
                           subset_forward,scalar_target,weighted_loss)
from .newrf_latefreq import build_late_model
from .newrf_conditioning_check import raw_coordinate_condition

def main():
    torch.set_num_threads(4);seed_all(42)
    report=output_paths("newrf",0,"conditioning_health_pilot")["report"]
    if (report/"results.json").exists():raise FileExistsError("Preserve pilot")
    data=FoldData(0,"screen",method="newrf")
    bank=FitGeometryBank(GEOMETRY,data.inputs,outer=0,role="screen_fit")
    cache=FitDirectionCache(bank)
    model=build_late_model(data,cache,128,128,frequency_chunk=8)
    raw_coordinate_condition(model,9./426.831421)
    with torch.no_grad():
        for field in list(model.coarse)+list(model.fine):field.network.alpha_out.bias.fill_(.1)
    guard=device_guard(GPU_UUID);torch.backends.cuda.matmul.allow_tf32=True
    model=model.to("cuda:0");cache.to("cuda:0")
    optimizer=torch.optim.Adam(model.parameters(),lr=5e-4,weight_decay=0)
    rows=np.array([350],np.int64);ss=np.linspace(0,127,8,dtype=np.int64);ff=np.array([0,63,127,191],np.int64)
    targets=[scalar_target(data,rows,f,ss,ff,"cuda:0") for f in range(8)]
    if any(float(v.abs().square().sum())<=0 for v in targets):raise RuntimeError("Health labels must be nonzero")
    def diagnostic():
        out=[]
        with torch.no_grad():
            for field in range(8):
                h,hc=subset_forward(model,cache,data,rows,field,ss,ff);t=targets[field]
                out.append(dict(field=field,nmse=float((h-t).abs().square().sum()/t.abs().square().sum()),
                                power_ratio=float(h.abs().square().sum()/t.abs().square().sum()),
                                zero_fraction=float((h.abs()==0).float().mean())))
        return out
    initial=diagnostic();curve=[];zero=[0]*8;start=time.monotonic()
    for step in range(2048):
        field=step%8
        h,hc=subset_forward(model,cache,data,rows,field,ss,ff)
        loss,_,_=weighted_loss(h,hc,targets[field],data.scale)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        gn=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        if float(gn)==0:zero[field]+=1
        if not math.isfinite(float(loss)) or not math.isfinite(float(gn)):raise FloatingPointError("Nonfinite health")
        optimizer.step()
        if (step+1)%256==0:
            d=diagnostic();curve.append(dict(step=step+1,fields=d))
            print(json.dumps(curve[-1]),flush=True)
        if time.monotonic()-start>180:break
    result=dict(status="pilot_complete",optimizer_steps=step+1,wall_seconds=time.monotonic()-start,
                initialization="raw-coordinate columns*=9/publicmap426.831421m; alpha_out.bias=.1",
                row=350,BS_spatial_ids=ss.tolist(),frequency_ids=ff.tolist(),effective_labels_per_step=32,
                source_network_or_integral_changed=False,initial=initial,curve=curve,
                final=diagnostic(),zero_gradient_batches_by_field=zero,GPU=guard,
                training_health_only=True,validation_truth_reads=0,outer_truth_reads=0,
                not_a_formal_checkpoint=True)
    atomic_json(report/"results.json",result)
    print(json.dumps(result),flush=True)
if __name__=="__main__":main()

