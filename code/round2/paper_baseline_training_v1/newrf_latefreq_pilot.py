"""Bounded equivalence, FIT-health and throughput pilot of late-frequency v2."""
from __future__ import annotations
import copy,json,math,time,traceback
from dataclasses import asdict
import numpy as np
import torch
from torch import nn
from .common import FoldData,seed_all,device_guard,output_paths,atomic_json
from .newrf_runner import (GPU_UUID,GEOMETRY,FitGeometryBank,FitDirectionCache,
                           subset_forward,scalar_target,weighted_loss,log_record,UniformRaySynthesizer)
from .newrf_latefreq import LateFrequencyNeWRF,LateFrequencyConfig,LateFrequencyField,build_late_model
from .newrf_query import subset_query
from .selection import BS_IDS,UE_IDS,FREQUENCY_IDS

def parameter_gradient(model):
    return torch.cat([p.grad.detach().reshape(-1).float() if p.grad is not None
                      else torch.zeros_like(p).reshape(-1) for p in model.parameters()])

def tiny_model(execution):
    m=LateFrequencyNeWRF.__new__(LateFrequencyNeWRF);nn.Module.__init__(m)
    cfg=LateFrequencyConfig(n_layers=3,hidden_dim=16,skip=(),position_levels=3,direction_levels=2,
              frequency_levels=3,n_samples=7,n_importance=5,far_m=7.,coordinate_scale_m=1.,frequency_chunk=2)
    m.config=cfg;m.coarse=nn.ModuleList([LateFrequencyField(cfg)])
    m.fine=nn.ModuleList([LateFrequencyField(cfg)]);m.synthesizer=UniformRaySynthesizer()
    m.execution=execution
    # Prevent a zero-density initial test from producing meaningless zero equality.
    for field in list(m.coarse)+list(m.fine):field.network.alpha_out.bias.data.fill_(.5)
    return m

def equivalence(model,origins,dirs,tx,frequencies,target,scale=1.,train_callback=None):
    naive=copy.deepcopy(model);naive.execution="naive"
    fast=model
    raw=[];grads=[];losses=[]
    for current in (naive,fast):
        current.zero_grad(set_to_none=True)
        if train_callback is None:
            pred,coarse=current._scalar_links(0,origins,dirs,tx,frequencies)
        else:pred,coarse=train_callback(current)
        loss,_,_=weighted_loss(pred,coarse,target,scale)
        loss.backward()
        raw.append((pred.detach(),coarse.detach()));grads.append(parameter_gradient(current));losses.append(float(loss))
    def difference(a,b):
        return dict(max_abs=float((a-b).abs().max()),
                    relative_L2=float((a-b).norm()/a.norm().clamp_min(1e-30)))
    result=dict(fine=difference(raw[0][0],raw[1][0]),coarse=difference(raw[0][1],raw[1][1]),
                all_parameter_gradient=difference(grads[0],grads[1]),losses=losses,
                reference_gradient_norm=float(grads[0].norm()))
    if result["reference_gradient_norm"]<=1e-12:raise AssertionError("Vacuous zero gradient")
    if max(result["fine"]["relative_L2"],result["coarse"]["relative_L2"])>1e-4 or result["all_parameter_gradient"]["relative_L2"]>2e-3:
        raise AssertionError(result)
    del naive
    return result

def main():
    paths=output_paths("newrf",0,"latefreq_v2_pilot");report=paths["report"]
    if (report/"results.json").exists():raise FileExistsError("Preserve completed v2 pilot")
    torch.set_num_threads(4);seed_all(42)
    small=tiny_model("shared")
    origins=torch.tensor([[.2,.5,1.3]],dtype=torch.float64)
    dirs=torch.tensor([[[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],
                        [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]]],dtype=torch.float64)
    tx=torch.tensor([[[1.,2.,2.],[1.1,2.,2.]]],dtype=torch.float64)
    f=torch.tensor([6.68712e9,6.69e9,6.69285e9],dtype=torch.float64)
    target=torch.full((1,2,3),.001+.002j,dtype=torch.complex64)
    cpu=equivalence(small,origins,dirs,tx,f,target)
    atomic_json(report/"CPU_equivalence.json",cpu)
    data=FoldData(0,"screen",method="newrf")
    bank=FitGeometryBank(GEOMETRY,data.inputs,outer=0,role="screen_fit")
    cache=FitDirectionCache(bank)
    seed_all(42);model=build_late_model(data,cache,128,128,frequency_chunk=8)
    guard=device_guard(GPU_UUID);torch.backends.cuda.matmul.allow_tf32=True
    model=model.to("cuda:0");cache.to("cuda:0")
    row=np.array([int(data.fit_ids[np.argmax(cache.count[data.fit_ids])])],np.int64)
    spatial=np.arange(0,128,16);ff=np.array([0,96,191],np.int64)
    target=scalar_target(data,row,0,spatial,ff,"cuda:0")
    gpu_eq=equivalence(model,None,None,None,None,target,data.scale,
                  train_callback=lambda m:subset_forward(m,cache,data,row,0,spatial,ff))
    atomic_json(report/"GPU_equivalence.json",gpu_eq)
    model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
    seed_all(42)
    optimizer=torch.optim.Adam(model.parameters(),lr=5e-4,weight_decay=0)
    rng=np.random.default_rng(42)
    losses=[];timings=[];health=[]
    # These labels are real FIT-only values. The pilot is not a comparison score.
    for step in range(80):
        rows=np.asarray([rng.choice(data.fit_ids)],np.int64);field=step%8
        ss=np.sort(rng.choice(128,8,replace=False));ff=np.sort(rng.choice(192,4,replace=False))
        target=scalar_target(data,rows,field,ss,ff,"cuda:0")
        torch.cuda.synchronize();tick=time.monotonic()
        pred,hc=subset_forward(model,cache,data,rows,field,ss,ff)
        loss,_,_=weighted_loss(pred,hc,target,data.scale)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        gn=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        if not math.isfinite(float(loss)) or not math.isfinite(float(gn)):
            raise FloatingPointError("Nonfinite v2 FIT health")
        if step<8:health.append(dict(field=field,loss=float(loss),gradient_norm=float(gn),
                                    row=int(rows[0]),rays=int(cache.count[rows[0]]) or 64))
        optimizer.step();torch.cuda.synchronize()
        timings.append(time.monotonic()-tick);losses.append(float(loss))
    if any(v["gradient_norm"]<=0 for v in health):
        raise RuntimeError("Dead field initialization; preserve pilot and report to parent")
    model.eval();bench=[];query=data.probe_ids[:1]
    for chunk in (8,):
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.monotonic()
        with torch.no_grad():h=model(data.positions(query),data.sites(query))
        torch.cuda.synchronize()
        bench.append(dict(frequency_chunk=chunk,seconds=time.monotonic()-tick,
                          shape=list(h.shape),finite=bool(torch.isfinite(h).all()),
                          peak_GPU_GiB=torch.cuda.max_memory_allocated()/2**30))
        del h
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.monotonic()
    with torch.no_grad():subset=subset_query(model,data,data.probe_ids,BS_IDS,UE_IDS,FREQUENCY_IDS)
    torch.cuda.synchronize()
    selector=dict(seconds=time.monotonic()-tick,shape=list(subset.shape),
                  finite=bool(torch.isfinite(subset).all()),
                  peak_GPU_GiB=torch.cuda.max_memory_allocated()/2**30)
    result=dict(status="passed",CPU_equivalence=cpu,GPU_equivalence=gpu_eq,
                full192_H_benchmarks=bench,proxy_forward_benchmark=selector,
                GPU=guard,model_config=asdict(model.config),ray_cache=cache.info,
                effective_scalar_labels_per_step=32,training_steps=80,
                loss_first16_mean=float(np.mean(losses[:16])),loss_last16_mean=float(np.mean(losses[-16:])),
                all_losses=losses,median_step_seconds=float(np.median(timings[8:])),
                q90_step_seconds=float(np.quantile(timings[8:],.9)),
                first8_field_health=health,trained_checkpoint_used_for_formal=False,
                outer_truth_reads=0,validation_truth_reads=0,
                native_original_exact_reproduction=False)
    atomic_json(report/"results.json",result)
    print(json.dumps(result),flush=True)
if __name__=="__main__":
    try:main()
    except BaseException:
        with (output_paths("newrf",0,"latefreq_v2_pilot")["report"]/"failure.txt").open("w") as f:
            f.write(traceback.format_exc())
        raise

