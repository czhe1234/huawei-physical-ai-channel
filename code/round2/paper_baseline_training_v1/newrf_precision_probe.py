"""Check streaming PE and optional fine-only BF16, without scoring query H."""
from __future__ import annotations
import argparse,copy,json,math,time,traceback
from dataclasses import replace,asdict
import numpy as np
import torch
from torch import nn
from .common import FoldData,output_paths,atomic_json,device_guard,seed_all,sha256
from .newrf_runner import (GPU_UUID,GEOMETRY,FitGeometryBank,FitDirectionCache,
                           build_model,subset_forward,scalar_target,weighted_loss)

def streaming_encode(x,encoder,dtype=torch.float32):
    """Same author sin/cos order; bounded temporaries, no trainable PE."""
    d=encoder.d_input
    out=torch.empty((*x.shape[:-1],encoder.d_output),dtype=dtype,device=x.device)
    out[...,:d]=x
    for i in range(encoder.n_freqs):
        # Frequency is the exact scalar captured by the pinned author's lambda.
        frequency=encoder.embed_fns[1+2*i].__defaults__[0]
        value=x*frequency
        out[...,d*(1+2*i):d*(2+2*i)]=torch.sin(value)
        out[...,d*(2+2*i):d*(3+2*i)]=torch.cos(value)
    return out

class StreamField(nn.Module):
    def __init__(self,field,bf16=False):
        super().__init__();self.field=field;self.bf16=bool(bf16)
    @property
    def network(self):return self.field.network
    @property
    def cfg(self):return self.field.cfg
    @cfg.setter
    def cfg(self,v):self.field.cfg=v
    def forward(self,points,view,tx,frequencies):
        cfg=self.field.cfg
        x=torch.cat((points/cfg.coordinate_scale_m,tx/cfg.coordinate_scale_m,
                     (frequencies-cfg.frequency_center_hz)/cfg.frequency_scale_hz),-1).float()
        dtype=torch.bfloat16 if self.bf16 else torch.float32
        ex=streaming_encode(x,self.field.encoder,dtype)
        ev=streaming_encode(view.float(),self.field.view_encoder,dtype)
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=self.bf16):
            raw=self.field.network(ex,ev)
        return raw.float()

def apply_streaming(model,fine_bf16=False):
    model.coarse=nn.ModuleList([StreamField(x,False) for x in model.coarse])
    model.fine=nn.ModuleList([StreamField(x,fine_bf16) for x in model.fine])
    return model

def replace_cfg(model,**kw):
    model.config=replace(model.config,**kw)
    for field in list(model.coarse)+list(model.fine):field.cfg=model.config

def encoder_check():
    from round2.paper_baseline_adapters_v1.newrf import _source
    e=_source("encoders.py").PositionalEncoder(7,10,True)
    x=torch.randn(71,7,requires_grad=True)
    y=x.detach().clone().requires_grad_(True)
    a=e(x);b=streaming_encode(y,e)
    ga=torch.autograd.grad(a.square().sum(),x)[0]
    gb=torch.autograd.grad(b.square().sum(),y)[0]
    result=dict(maximum_forward_abs=float((a-b).abs().max()),maximum_input_gradient_abs=float((ga-gb).abs().max()))
    if result["maximum_forward_abs"]!=0 or result["maximum_input_gradient_abs"]>1e-3:
        raise AssertionError(result)
    return result

def run(args):
    torch.set_num_threads(4);seed_all(42)
    paths=output_paths("newrf",0,"precision_probe");report=paths["report"]
    if (report/"results.json").exists():raise FileExistsError("Preserve precision probe")
    enc=encoder_check()
    data=FoldData(0,"screen",method="newrf")
    bank=FitGeometryBank(GEOMETRY,data.inputs,outer=0,role="screen_fit")
    cache=FitDirectionCache(bank)
    original=build_model(data,cache,128,128)
    control=copy.deepcopy(original)
    streamed=apply_streaming(copy.deepcopy(original),False)
    finebf=apply_streaming(copy.deepcopy(original),True)
    del original
    gpu=device_guard(GPU_UUID);torch.backends.cuda.matmul.allow_tf32=True
    control=control.to("cuda:0").eval();streamed=streamed.to("cuda:0").eval();finebf=finebf.to("cuda:0").eval()
    cache.to("cuda:0")
    rows=data.probe_ids[:1];freq_ids=np.array([0,32,64,96,128,160,191],np.int64)
    freq=torch.tensor(data.frequencies(freq_ids),device="cuda:0",dtype=torch.float64)
    results={}
    with torch.no_grad():
        torch.cuda.synchronize();tick=time.monotonic()
        a=control(data.positions(rows),data.sites(rows),freq)
        torch.cuda.synchronize();results["reference_subset_seconds"]=time.monotonic()-tick
        b=streamed(data.positions(rows),data.sites(rows),freq)
        c=finebf(data.positions(rows),data.sites(rows),freq)
    den=float(a.abs().square().sum())
    def comparison(v):
        return dict(maximum_abs=float((a-v).abs().max()),
                    relative_L2=math.sqrt(float((a-v).abs().square().sum())/max(den,1e-30)),
                    RMS_absolute_over_FIT_RMS=float((a-v).abs().square().mean().sqrt())/data.scale,
                    full_subset_shape=list(v.shape),finite=bool(torch.isfinite(v).all()))
    results["stream_FP32"]=comparison(b);results["fine_BF16"]=comparison(c)
    train_row=np.array([int(data.fit_ids[np.argmax(cache.count[data.fit_ids])])],np.int64)
    spatial=np.arange(0,128,4);ff=np.array([0,96,191])
    target=scalar_target(data,train_row,0,spatial,ff,"cuda:0")
    losses=[];gradients=[]
    for model in (control,streamed,finebf):
        model.zero_grad(set_to_none=True)
        pred,coarse=subset_forward(model,cache,data,train_row,0,spatial,ff)
        loss,_,_=weighted_loss(pred,coarse,target,data.scale);loss.backward()
        losses.append(float(loss))
        params=list(model.coarse[0].parameters())+list(model.fine[0].parameters())
        gradients.append(torch.cat([p.grad.detach().reshape(-1).float() if p.grad is not None else torch.zeros_like(p).reshape(-1) for p in params]))
    results["gradient_FP32_stream_relative_L2"]=float((gradients[1]-gradients[0]).norm()/gradients[0].norm().clamp_min(1e-30))
    results["gradient_fine_BF16_relative_L2"]=float((gradients[2]-gradients[0]).norm()/gradients[0].norm().clamp_min(1e-30))
    results["FIT_subset_losses"]=losses
    results["reference_gradient_norm"]=float(gradients[0].norm())
    atomic_json(report/"comparison_before_full_benchmark.json",results)
    stream_pass=results["stream_FP32"]["relative_L2"]<1e-6 and results["gradient_FP32_stream_relative_L2"]<1e-5
    bfpass=(results["reference_gradient_norm"]>1e-12 and results["fine_BF16"]["relative_L2"]<.01 and
            results["fine_BF16"]["RMS_absolute_over_FIT_RMS"]<.01 and
            results["gradient_fine_BF16_relative_L2"]<.05)
    results.update(stream_FP32_equivalent=stream_pass,fine_BF16_accepted=bfpass,
                   acceptance="relative channel L2<1%, absolute delta/FIT_RMS<1%, gradient relL2<5%",
                   coarse_precision="float32 TF32 matmul as reference; no BF16",
                   sampling_geometry_phase="float64",H_dtype="complex64",query_truth_scored=False,
                   fit_gradient_used_only_for_precision_check=True,optimizer_steps=0)
    chosen=finebf if bfpass else streamed
    if not stream_pass:raise AssertionError(results)
    full=[]
    for chunk in ([8,16] if args.full else []):
        replace_cfg(chosen,frequency_chunk=chunk)
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.monotonic()
        with torch.no_grad():out=chosen(data.positions(rows),data.sites(rows))
        torch.cuda.synchronize()
        full.append(dict(frequency_chunk=chunk,precision="fine_BF16" if bfpass else "stream_FP32",
                         elapsed_seconds=time.monotonic()-tick,shape=list(out.shape),
                         finite=bool(torch.isfinite(out).all()),
                         peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30))
        del out
        print(json.dumps(full[-1]),flush=True)
    results.update(status="passed",encoder=enc,full_H_benchmarks=full,GPU=gpu,
                   config=asdict(chosen.config),ray_cache=cache.info,
                   data=data.info(),source_sha256=sha256(__import__("pathlib").Path(__file__)))
    atomic_json(report/"results.json",results)
    print(json.dumps({k:v for k,v in results.items() if k not in ("data","ray_cache")}),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--full",action="store_true")
    args=parser.parse_args()
    try:run(args)
    except BaseException:
        with (output_paths("newrf")["report"]/"precision_probe_failure.txt").open("w") as f:f.write(traceback.format_exc())
        raise
