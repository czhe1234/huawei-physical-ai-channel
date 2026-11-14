"""v4 same-weight ragged equivalence and source-semantics FIT health."""
from __future__ import annotations
import argparse,copy,json,math,time,traceback
import numpy as np
import torch
from .common import FoldData,seed_all,device_guard,atomic_json
from .newrf_runner import GPU_UUID,GEOMETRY,FitGeometryBank,FitDirectionCache,scalar_target
from .newrf_native_batch_v4 import (build_native_batch_model,station_forward,station_forward_reference,
                    station_target,native_weighted_loss,output_paths_v4)

def grad_vector(model):
    return torch.cat([p.grad.detach().reshape(-1) if p.grad is not None else torch.zeros_like(p).reshape(-1)
                      for p in model.parameters()])

def check(model,cache,data,rows,bs,ff):
    ref=copy.deepcopy(model);outs=[];grads=[];losses=[]
    target=station_target(data,rows,0,bs,ff,"cuda:0")
    for m,fn in [(ref,station_forward_reference),(model,station_forward)]:
        m.zero_grad(set_to_none=True);h,hc=fn(m,cache,data,rows,0,bs,ff)
        loss,_,_=native_weighted_loss(h,hc,target);loss.backward()
        outs.append((h.detach(),hc.detach()));grads.append(grad_vector(m));losses.append(float(loss))
    def diff(a,b):
        return dict(max_abs=float((a-b).abs().max()),relative_L2=float((a-b).norm()/a.norm().clamp_min(1e-30)))
    result=dict(fine=diff(outs[0][0],outs[1][0]),coarse=diff(outs[0][1],outs[1][1]),
       all_parameter_gradient=diff(grads[0],grads[1]),losses=losses,reference_gradient_norm=float(grads[0].norm()),
       stations=len(rows),actual_ray_counts=[int(cache.count[r]) or 64 for r in rows])
    if result["fine"]["relative_L2"]>1e-3 or result["coarse"]["relative_L2"]>1e-3 or result["all_parameter_gradient"]["relative_L2"]>2e-3:
        raise AssertionError(result)
    del ref;model.zero_grad(set_to_none=True);return result

def main(tag):
    torch.set_num_threads(4);seed_all(42);report=output_paths_v4("newrf",0,tag)["report"]
    if (report/"results.json").exists():raise FileExistsError("Preserve health")
    data=FoldData(0,"screen",method="newrf")
    bank=FitGeometryBank(GEOMETRY,data.inputs,outer=0,role="screen_fit");cache=FitDirectionCache(bank)
    seed_all(42);model=build_native_batch_model(data,cache)
    guard=device_guard(GPU_UUID);torch.backends.cuda.matmul.allow_tf32=False
    model=model.to("cuda:0");cache.to("cuda:0")
    rng=np.random.default_rng(42)
    # Include both real variable-count rays and zero-path fallback in the equality check.
    eqrows=np.asarray([350,3707,1775,1876,3374,2281,2773,615]*4,np.int64)
    if not set(eqrows).issubset(set(data.fit_ids)):eqrows=rng.choice(data.fit_ids,32,replace=True)
    bs=rng.integers(0,128,32);ff=rng.integers(0,192,32)
    eq=check(model,cache,data,eqrows,bs,ff);atomic_json(report/"ragged_equivalence.json",eq)
    panel=[]
    for site in (0,1):
        ids=data.fit_ids[data.inputs.site_id[data.fit_ids]==site]
        order=np.lexsort((data.inputs.positions[ids,1],data.inputs.positions[ids,0]))
        choices=ids[order[np.linspace(0,len(ids)-1,8,dtype=int)]]
        panel.extend(choices.tolist())
    original_panel=panel.copy()
    spatial=np.linspace(0,127,8,dtype=np.int64);freqs=np.array([0,63,127,191],np.int64)
    selected=[];zero_candidates=[]
    for anchor in original_panel:
        site=data.inputs.site_id[anchor]
        candidates=data.fit_ids[data.inputs.site_id[data.fit_ids]==site]
        distance=np.linalg.norm(data.inputs.positions[candidates]-data.inputs.positions[anchor],axis=-1)
        for row in candidates[np.argsort(distance,kind="stable")]:
            row=int(row)
            if row in selected:continue
            h=data.read_fit(np.array([row]),freq_ids=freqs,normalized=False)
            selected_h=np.take(h,np.concatenate((spatial,128+spatial)),axis=1)
            if float(np.abs(selected_h).sum())>0:
                selected.append(row);break
            zero_candidates.append(row)
        else:raise RuntimeError("Cannot find16nonzeroFIT health locations")
    panel=np.asarray(selected,np.int64)
    atomic_json(report/"nonzero_health_panel_selection.json",dict(original_geometry_anchors=original_panel,
        nonzero_panel=panel.tolist(),zero_candidates=zero_candidates,
        rule="Nearest same-site FIT point with nonzero health sub-tensor; FIT labels only, health only",
        formal_FIT_zero_rows_kept=True))
    pool=torch.stack([scalar_target(data,panel,f,spatial,freqs,"cuda:0") for f in range(8)])
    # All panel rows stay. The source loss has one joint energy denominator.
    if bool((pool.abs().square().sum((0,2,3))<=0).any()):raise RuntimeError("Health needs nonzero geometry-selected positions")
    def diagnostic():
        result=[]
        with torch.no_grad():
            for field in range(8):
                error=power=energy=0.;zeros=0;total=0
                for i,row in enumerate(panel):
                    bi=np.repeat(spatial,4);fi=np.tile(freqs,8);rr=np.repeat(row,32)
                    h,_=station_forward(model,cache,data,rr,field,bi,fi);t=pool[field,i].reshape(-1)
                    error+=float((h-t).abs().square().sum());power+=float(h.abs().square().sum())
                    energy+=float(t.abs().square().sum());zeros+=int((h.abs()==0).sum());total+=h.numel()
                result.append(dict(field=field,nmse=error/max(energy,1e-30),
                   power_ratio=power/max(energy,1e-30),zero_fraction=zeros/total))
        return result
    initial=diagnostic();rng=np.random.default_rng(42);optimizer=torch.optim.Adam(model.parameters(),lr=5e-4,weight_decay=0)
    start=time.monotonic();zero=[0]*8;history=[];recent=[]
    for step in range(10000):
        field=step%8;pi=rng.integers(0,len(panel),32);bi=rng.integers(0,8,32);fi=rng.integers(0,4,32)
        rows=panel[pi];bs=spatial[bi];ff=freqs[fi]
        t=pool[field,torch.tensor(pi,device="cuda:0"),torch.tensor(bi,device="cuda:0"),torch.tensor(fi,device="cuda:0")]
        h,hc=station_forward(model,cache,data,rows,field,bs,ff)
        loss,_,_=native_weighted_loss(h,hc,t);optimizer.zero_grad(set_to_none=True);loss.backward()
        gn=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        if float(gn)==0:zero[field]+=1
        if not math.isfinite(float(loss)) or not math.isfinite(float(gn)):raise FloatingPointError("Native batch health nonfinite")
        optimizer.step();recent.append(float(loss))
        if (step+1)%2500==0:
            event=dict(step=step+1,fields=diagnostic(),native_batchNMSE_recent=float(np.mean(recent[-2500:])),
                       elapsed_seconds=time.monotonic()-start)
            history.append(event);atomic_json(report/"progress.json",event);print(json.dumps(event),flush=True)
            if all(v["nmse"]<.05 for v in event["fields"]):break
    final=diagnostic();health_steps=step+1;health_wall=time.monotonic()-start
    passed=all(v["nmse"]<.9 and v["power_ratio"]>.01 and v["zero_fraction"]<.01 for v in final)
    # Actual fullFIT I/O +32 independent stations + full scalar axes throughput.
    times=[];full_zero=0;full_losses=[]
    for step in range(200):
        tick=time.monotonic();field=step%8;rows=rng.choice(data.fit_ids,32,replace=True)
        bs=rng.integers(0,128,32);ff=rng.integers(0,192,32)
        target=station_target(data,rows,field,bs,ff,"cuda:0")
        h,hc=station_forward(model,cache,data,rows,field,bs,ff)
        loss,_,_=native_weighted_loss(h,hc,target)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        gn=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        if float(gn)==0:full_zero+=1
        torch.cuda.synchronize();times.append(time.monotonic()-tick);full_losses.append(float(loss))
    model.eval();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.monotonic()
    with torch.no_grad():full=model(data.positions(data.probe_ids[:1]),data.sites(data.probe_ids[:1]))
    torch.cuda.synchronize()
    full_benchmark=dict(seconds=time.monotonic()-tick,shape=list(full.shape),finite=bool(torch.isfinite(full).all()),
                        peak_gpu_GiB=torch.cuda.max_memory_allocated()/2**30,query_truth_read=False)
    result=dict(status="passed" if passed else "needs_diagnosis",optimizer_steps=health_steps,
         health_optimizer_steps=health_steps,additional_fullFIT_throughput_updates=200,
         full192_inference_benchmark=full_benchmark,
         health_wall_seconds=health_wall,initial=initial,history=history,final=final,
         zero_gradient_batches_by_field=zero,ragged_equivalence=eq,GPU=guard,panel=panel.tolist(),
         batch="32 iid station draws; one independently sampled BS/frequency label per draw",
         health_axes=dict(BS_spatial=spatial.tolist(),frequency=freqs.tolist()),
         source_loss="(.1 sum|coarse-target|²+.9 sum|fine-target|²)/sum|target|²",
         fullFIT_throughput=dict(steps=200,median_seconds=float(np.median(times[10:])),
            mean_seconds=float(np.mean(times[10:])),q90_seconds=float(np.quantile(times[10:],.9)),
            zero_gradient_batches=full_zero,mean_native_loss=float(np.mean(full_losses)),
            weights_discarded=True),
         validation_truth_reads=0,outer_truth_reads=0,not_formal_checkpoint=True)
    atomic_json(report/"results.json",result);print(json.dumps(result),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--tag",default="health_nonzero");args=p.parse_args();tag=args.tag
    try:main(tag)
    except BaseException:
        atomic_json(output_paths_v4("newrf",0,tag)["report"]/"failure.json",dict(traceback=traceback.format_exc()))
        raise

