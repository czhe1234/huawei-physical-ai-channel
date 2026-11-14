"""Train the NeWRF full-H adaptation with native complex volume synthesis.

Only scalar-link/frequency labels are sampled for gradients. Query prediction
still returns the complete256x4x192 H. Frozen adapters and pinned sources stay
unchanged; the uniform-ray reduction is algebraically vectorized here.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict,replace
import json
import math
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from project_paths import PATHS
from round2.paper_baseline_adapters_v1.newrf import _source,NeWRFFullHConfig
from round2.simulation_recovery_v1.qualified_newrf import QualifiedNeWRFFullH
from round2.simulation_recovery_v1.ray_provider import FitGeometryBank
from .common import (FoldData,ART,REPORT,atomic_json,atomic_torch_save,
                     device_guard,seed_all,output_paths,evaluate,save_outer_prediction,
                     digest_ids,sha256)

GPU_UUID="GPU-6b53c3c2-3071-5dfb-f306-2f2f142f8009"
GEOMETRY=PATHS.artifacts/"round2/simulation_recovery_v1/geometry/full_outer0_all_train/manifest.json"
AUTHOR=PATHS.root/"plus-pro/code/third_party/paper_baselines/newrf"

class UniformRaySynthesizer:
    """Original volume algebra; replace only Python group sums by one reduction."""
    def __init__(self):
        self.author=_source("synthesizers.py").Synthesizer()
        self.speedOfLight=self.author.speedOfLight

    def synthesize(self,raw,z_vals,fc,ray_batches=None):
        if not ray_batches:
            raise ValueError("Explicit positive ray groups required")
        r=int(ray_batches[0])
        if r<1 or any(int(v)!=r for v in ray_batches) or len(raw)!=len(ray_batches)*r:
            return self.author.synthesize(raw,z_vals,fc,ray_batches)
        dists=z_vals[...,1:]-z_vals[...,:-1]
        dists=torch.cat([dists,1e-10*torch.ones_like(dists[...,:1])],-1)
        alpha=1-torch.exp(-torch.nn.functional.relu(raw[...,-1])*dists)
        def exclusive(x):
            v=torch.cumprod(x,dim=1)
            return torch.cat((torch.zeros_like(v[:,:1]),v[:,:-1]),1)
        weights=alpha*exclusive(1-alpha+1e-10)
        depth=(weights*z_vals).sum(-1)
        phs_shift=torch.exp(-1j*(2*torch.pi*fc*1e9/self.speedOfLight)*dists)
        coeffs=alpha*exclusive((1-alpha)*phs_shift+1e-10)
        amp_decay=self.speedOfLight/(z_vals*fc*1e9*4*torch.pi)
        re_ch=torch.tanh(raw[...,0]);im_ch=torch.tanh(raw[...,1])
        per_ray=((re_ch+1j*im_ch)*amp_decay*coeffs).sum(1)
        cfr=per_ray.reshape(len(ray_batches),r).sum(1)
        return cfr,depth,torch.abs(re_ch+1j*im_ch),weights


class FitDirectionCache:
    """Load only the capability's FIT rows; never load query geometry."""
    def __init__(self,bank):
        self.bank=bank;self.allowed=frozenset(map(int,bank.source_fit_ids))
        self.last=np.zeros((4000,64,3),np.float64)
        self.los=np.zeros((4000,64),bool)
        self.count=np.zeros(4000,np.int64)
        max_distance=0.;zero=0
        for row in bank.source_fit_ids:
            row=int(row);p=bank.get(row)
            chosen=np.argsort(p["tau_seconds"],kind="stable")[:64]
            n=len(chosen);self.count[row]=n
            if n:
                self.last[row,:n]=p["last_interaction_m"][chosen]
                self.los[row,:n]=p["depth"][chosen]==0
                distance=np.linalg.norm(self.last[row,:n]-p["ue_position_m"],axis=-1)
                max_distance=max(max_distance,float(distance.max()))
            else:zero+=1
        self.far_m=float(math.ceil(max_distance*1.05+2.))
        self.info=dict(source_FIT_ids_sha256=digest_ids(bank.source_fit_ids),
                       source_fingerprint=bank.source_fingerprint,
                       FIT_count=len(bank.source_fit_ids),zero_path_FIT_count=zero,
                       longest_selected_FIT_last_segment_m=max_distance,
                       far_m=self.far_m,far_rule="ceil(1.05*max_selected_FIT_last_segment+2m)",
                       max_training_rays=64,query_rays=64,query_geometry_payload_reads=0)
        self.gpu_last=None;self.gpu_los=None

    def to(self,device):
        self.gpu_last=torch.tensor(self.last,dtype=torch.float64,device=device)
        self.gpu_los=torch.tensor(self.los,dtype=torch.bool,device=device)
        return self

    def scalar(self,model,row,field,spatial,origin_all,site,tx_all):
        if int(row) not in self.allowed:
            raise PermissionError("FIT geometry access outside role")
        p,u=divmod(int(field),4);n=int(self.count[int(row)])
        if not n:
            d=model.ray_provider.directions(origin_all[None],site[None],tx_all[None])
            return d[0,p,spatial,u]
        v=self.gpu_last[int(row),:n][None]-origin_all[u][None,None]
        v=v.expand(len(spatial),n,3).clone()
        los=self.gpu_los[int(row),:n]
        if bool(los.any()):
            v[:,los,:]=tx_all[p,spatial,None]-origin_all[u][None,None]
        norm=torch.linalg.vector_norm(v,dim=-1,keepdim=True)
        if bool((norm<1e-8).any()):raise ValueError("Degenerate FIT ray")
        return v/norm


def build_model(data,cache,coarse,importance,*,frequency_chunk=8):
    cfg=NeWRFFullHConfig(n_layers=6,hidden_dim=128,skip=(4,),position_levels=10,
                        direction_levels=4,n_samples=int(coarse),n_importance=int(importance),
                        far_m=cache.far_m,coordinate_scale_m=1.,frequency_chunk=frequency_chunk)
    m=QualifiedNeWRFFullH(cache.bank,cfg,max_query_rays=64,max_training_rays=64)
    m.synthesizer=UniformRaySynthesizer()
    return m


def subset_forward(model,cache,data,rows,field,spatial,freq_ids):
    rows=np.asarray(rows,np.int64)
    if not set(map(int,rows)).issubset(cache.allowed):
        raise PermissionError("Training rows outside FIT")
    dev=model.bs_ports.device;p,u=divmod(int(field),4)
    pos=torch.tensor(data.positions(rows),dtype=torch.float64,device=dev)
    sites=torch.tensor(data.sites(rows),dtype=torch.long,device=dev)
    origins=model.spec.ue_world_positions(pos,sites)
    tx=model.bs_ports[sites]
    ss=torch.as_tensor(spatial,dtype=torch.long,device=dev)
    f=torch.tensor(data.frequencies(freq_ids),dtype=torch.float64,device=dev)
    fine=[];coarse=[]
    for i,row in enumerate(rows):
        dirs=cache.scalar(model,int(row),field,ss,origins[i],sites[i],tx[i])
        h,hc=model._scalar_links(field,origins[i:i+1,u],dirs[None],tx[i:i+1,p][:,ss],f)
        fine.append(h);coarse.append(hc)
    return torch.cat(fine),torch.cat(coarse)


def scalar_target(data,rows,field,spatial,freq_ids,device):
    p,u=divmod(int(field),4)
    h=data.read_fit(rows,freq_ids=freq_ids,normalized=False)
    v=np.take(h[:,p*128:(p+1)*128,u,:],np.asarray(spatial,np.int64),axis=1)
    return torch.as_tensor(v,device=device,dtype=torch.complex64)


def weighted_loss(fine,coarse,target,scale):
    fine_loss=((fine-target)/scale).abs().square().mean()
    coarse_loss=((coarse-target)/scale).abs().square().mean()
    return .9*fine_loss+.1*coarse_loss,fine_loss,coarse_loss


def synthesis_check(device="cpu"):
    torch.manual_seed(374)
    groups=[5]*3;n=sum(groups);samples=9
    base=torch.randn(n,samples,3,device=device,dtype=torch.float32)
    base[...,-1]=base[...,-1].abs()+.2
    z=torch.linspace(1e-10,8,samples,device=device,dtype=torch.float64).expand(n,samples)
    f=torch.tensor([6.69]*n,device=device,dtype=torch.float64)[:,None]
    a=base.detach().clone().requires_grad_(True);b=base.detach().clone().requires_grad_(True)
    reference=_source("synthesizers.py").Synthesizer().synthesize(a,z,f,groups)
    fast=UniformRaySynthesizer().synthesize(b,z,f,groups)
    errors=[float((x-y).abs().max()) for x,y in zip(reference,fast)]
    la=reference[0].abs().square().sum()+reference[-1].sum()*.01
    lb=fast[0].abs().square().sum()+fast[-1].sum()*.01
    ga=torch.autograd.grad(la,a)[0];gb=torch.autograd.grad(lb,b)[0]
    grad=float((ga-gb).abs().max())
    if max(errors)>1e-9 or grad>1e-6:raise AssertionError((errors,grad))
    return dict(output_max_abs=errors,gradient_max_abs=grad,groups=groups,device=device,
                original_first_exclusive_entry_zero_preserved=True)


def log_record(path,value):
    text=json.dumps(value,ensure_ascii=False,allow_nan=False)
    with path.open("a") as f:f.write(text+"\n")
    print(text,flush=True)


def pilot(args):
    torch.set_num_threads(4)
    paths=output_paths("newrf",args.outer,args.pilot_tag)
    report=paths["report"];log=report/"pilot.jsonl"
    if (report/"pilot_results.json").exists():
        raise FileExistsError("Preserve completed pilot")
    data=FoldData(args.outer,"screen",method="newrf")
    bank=FitGeometryBank(GEOMETRY,data.inputs,outer=args.outer,role="screen_fit")
    cache=FitDirectionCache(bank)
    scale=data.scale
    requested=[tuple(map(int,c.split(":"))) for c in args.pilot_configs.split(",")]
    seed_all(42)
    prototype=build_model(data,cache,requested[0][0],requested[0][1])
    models=[]
    for ns,ni in requested:
        m=copy.deepcopy(prototype)
        cfg=replace(m.config,n_samples=ns,n_importance=ni)
        m.config=cfg
        for field in list(m.coarse)+list(m.fine):
            field.cfg=cfg
        models.append(((ns,ni),m))
    del prototype
    gpu=device_guard(GPU_UUID)
    torch.set_num_threads(4)
    # NN matrix products may use TF32; all geometry/phase remains FP64 and
    # synthesized channels remain complex64. No AMP is used.
    torch.backends.cuda.matmul.allow_tf32=True
    cache.to("cuda:0")
    eq=synthesis_check("cuda:0")
    rng=np.random.default_rng(42)
    panel=[]
    for site in (0,1):
        ids=data.fit_ids[data.inputs.site_id[data.fit_ids]==site]
        order=np.lexsort((data.inputs.positions[ids,1],data.inputs.positions[ids,0]))
        panel.extend(ids[order[np.linspace(0,len(ids)-1,4,dtype=int)]].tolist())
    panel=np.array(panel,np.int64)
    configs=[];started=time.monotonic()
    for (ns,ni),model in models:
        if time.monotonic()-started>230:break
        model=model.to("cuda:0").train()
        optimizer=torch.optim.Adam(model.parameters(),lr=5e-4,weight_decay=0)
        torch.cuda.reset_peak_memory_stats()
        initial_grad=[];density_fixed=[];losses=[];elapsed=[]
        for step in range(args.pilot_steps):
            row=np.array([panel[step%len(panel)]],np.int64)
            field=step%8
            spatial=np.sort(rng.choice(128,16,replace=False))
            freq=np.sort(rng.choice(192,4,replace=False))
            target=scalar_target(data,row,field,spatial,freq,"cuda:0")
            torch.cuda.synchronize();tick=time.monotonic()
            fine,coarse=subset_forward(model,cache,data,row,field,spatial,freq)
            loss,lf,lc=weighted_loss(fine,coarse,target,scale)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            grads=[p.grad for p in list(model.coarse[field].parameters())+list(model.fine[field].parameters()) if p.grad is not None]
            gn=float(torch.sqrt(sum(g.detach().float().square().sum() for g in grads)))
            if not math.isfinite(float(loss)) or not math.isfinite(gn):raise FloatingPointError("Nonfinite NeWRF training")
            if step<8:
                initial_grad.append(dict(field=field,gradient_norm=gn,loss=float(loss)))
            if gn==0:
                with torch.no_grad():
                    for net in (model.coarse[field],model.fine[field]):
                        net.network.alpha_out.bias.fill_(.1)
                density_fixed.append(dict(step=step,field=field,bias=.1))
                # Retry the same observed FIT mini-batch once after preflight init repair.
                fine,coarse=subset_forward(model,cache,data,row,field,spatial,freq)
                loss,lf,lc=weighted_loss(fine,coarse,target,scale)
                optimizer.zero_grad(set_to_none=True);loss.backward()
                grads=[p.grad for p in list(model.coarse[field].parameters())+list(model.fine[field].parameters()) if p.grad is not None]
                gn=float(torch.sqrt(sum(g.detach().float().square().sum() for g in grads)))
                if not math.isfinite(gn) or gn==0:raise RuntimeError("Dead density persists after bias init repair")
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step();torch.cuda.synchronize()
            elapsed.append(time.monotonic()-tick);losses.append(float(loss.detach()))
        model.eval()
        query=data.probe_ids[:1]
        torch.cuda.synchronize();tick=time.monotonic()
        with torch.no_grad():
            h=model(data.positions(query),data.sites(query))
        torch.cuda.synchronize();query_seconds=time.monotonic()-tick
        if h.shape!=(1,256,4,192) or h.dtype!=torch.complex64 or not bool(torch.isfinite(h).all()):
            raise RuntimeError("Full192-tone query failed")
        result=dict(config=asdict(model.config),steps=len(losses),FIT_RMS=scale,
                    fit_loss_first8=float(np.mean(losses[:8])),fit_loss_last8=float(np.mean(losses[-8:])),
                    initial_field_gradients=initial_grad,density_initialization_repairs=density_fixed,
                    step_seconds_median=float(np.median(elapsed[8:] or elapsed)),
                    full_H_one_query_seconds=query_seconds,full_H_shape=list(h.shape),
                    full_H500_inference_extrapolated_seconds=query_seconds*500,
                    peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                    scale_convention="native raw NeWRF H; pred and target both divided by FIT RMS in loss",
                    query_truth_scored=False,outer_truth_read=False,neural_tf32=True,complex_amp=False)
        configs.append(result);log_record(log,result)
        del h,model,optimizer
        torch.cuda.empty_cache()
        if time.monotonic()-started>270:break
    result=dict(status="bounded_pilot_complete",GPU=gpu,elapsed_seconds=time.monotonic()-started,
                configs=configs,synthesis_equivalence=eq,ray_cache=cache.info,
                author_defaults=dict(config=PATHS.relative(AUTHOR/"config/default.yaml"),
                                     sha256=sha256(AUTHOR/"config/default.yaml"),lr=5e-4,
                                     coarse_weight=.1,fine_weight=.9,n_samples=128,n_importance=128,
                                     native_n_iters=100000,native_batch_scalar_labels=32),
                adaptations=["multiport+frequency conditioning","geometry-assisted FIT rays",
                             "sample scalar links/frequencies for unbiased squared-error gradients",
                             "world-coordinate PE10 matching native length units",
                             "far bound from permitted FIT geometry, not query channel",
                             "uniform-group tensor reduction algebraically verified"],
                formal_run_started=False,validation_quality_scored=False,
                source_sha256=sha256(Path(__file__)),data=data.info())
    atomic_json(report/"pilot_results.json",result)
    print(json.dumps({"status":result["status"],"configs":[(c["config"]["n_samples"],c["config"]["n_importance"]) for c in configs]}))
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=["cpu-check","pilot"],default="cpu-check")
    p.add_argument("--outer",type=int,default=0)
    p.add_argument("--pilot-steps",type=int,default=40)
    p.add_argument("--pilot-tag",default="pilot_v2")
    p.add_argument("--pilot-configs",default="128:128")
    args=p.parse_args()
    if args.mode=="cpu-check":
        value=synthesis_check("cpu")
        atomic_json(output_paths("newrf")["report"]/"synthesis_equivalence_cpu.json",value)
        print(json.dumps(value))
    else:
        pilot(args)

if __name__=="__main__":
    try:main()
    except BaseException:
        r=output_paths("newrf")["report"]
        with (r/"last_failure.txt").open("w") as f:f.write(traceback.format_exc())
        raise
