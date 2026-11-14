"""Wide-domain NeRF2 training with FIT-only initialization and fixed-H subset selection.

Original324 rays,32 samples,8x256 network and raw physical positional encoding
are retained. All coordinates are metres and extinction has units1/metre.
The original MIMO source consumes query uplinkCSI; this position-only fullH
adaptation uses known TX/RX positions and a shared physical-frequency input.
"""
from __future__ import annotations
import argparse,json,math,os,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from round2.paper_baseline_training_v1 import field_runner as legacy
from round2.paper_baseline_training_v1.field_fast import enable_shared_medium
from round2.paper_baseline_training_v1.common import (FoldData,seed_all,device_guard,sha256,
    atomic_json,atomic_torch_save,output_paths,save_outer_prediction)
from round2.paper_baseline_adapters_v1.field_models import CompetitionNeRF2

GPU_UUID=legacy.GPU_UUID
CONFIG={**legacy.NERF2_CONFIG,"far_m":810.,"near_m":0.}
BS_SPATIAL=np.linspace(0,127,8,dtype=np.int64)
BS_PORTS=np.concatenate([BS_SPATIAL,128+BS_SPATIAL])
FREQUENCY_IDS=np.linspace(0,191,12,dtype=np.int64)


def identity():
    return {**legacy.source_identity(),"nerf2_runner_v2.py":sha256(Path(__file__))}


def build(data):
    cfg={**CONFIG,"world_center_m":tuple(np.mean(data.positions(data.fit_ids),axis=0))}
    model=CompetitionNeRF2(data.geometry,field_config=cfg)
    enable_shared_medium(model,medium_ray_chunk=324,tx_chunk=32,factor_transport=True)
    model.register_parameter("log_output_gain",nn.Parameter(torch.tensor(-math.log(324.),dtype=torch.float32)))
    with torch.no_grad():
        for core in model.fields.values():
            core.field.attenuation_output.weight[0].zero_()
            core.field.attenuation_output.bias[0].fill_(1./CONFIG["far_m"])
    return model.to(device="cuda:0")


def optimizer(model):
    return torch.optim.Adam([
        dict(params=[p for n,p in model.named_parameters() if n!="log_output_gain"],lr=1e-4,initial_lr=1e-4,weight_decay=5e-5),
        dict(params=[model.log_output_gain],lr=1e-3,initial_lr=1e-3,weight_decay=0.)])


def metadata():
    return dict(config=CONFIG,source_commit="130b55ea3856aa2f5a42f9329cfa2e845c78e243",
      adaptation="position-only shared physical-frequency fullH, no query uplinkCSI",
      attenuation_initialization="amplitude output row weights0,bias1/far per metre; initial total optical thickness1",
      attenuation_function="original abs(leakyReLU), complete original range retained",
      emission_gain="positive exp(loggain), initial1/324, FIT-trained, no query correction",
      acceleration="forward/gradient-checked medium/TX projection/transport factoring",
      precision="NN FP32, TF32 disabled; actual Hz and propagation phase FP64; coherent sum complex128 to complex64",
      validation_subset=dict(BS_ports=BS_PORTS.tolist(),UE_ports=list(range(4)),frequencies=FREQUENCY_IDS.tolist()),
      selection="global complex NMSE on fixed16 validation positions and fixed16x4x12 submatrix; no phase alignment")


def subset(model,data,ids):
    p=torch.tensor(data.positions(ids),device="cuda:0",dtype=torch.float64)
    site=torch.tensor(data.sites(ids),device="cuda:0",dtype=torch.int64)
    frequency=torch.tensor(data.frequencies(FREQUENCY_IDS),device="cuda:0",dtype=torch.float64)
    result=torch.zeros((len(ids),16,4,12),device="cuda:0",dtype=torch.complex64)
    world=data.geometry.ue_world_positions(p,site)
    for s in (0,1):
        rows=torch.nonzero(site==s).flatten()
        if not len(rows):continue
        for b in (0,1):
            tx=model.bound_bs_ports[s,b].reshape(128,3)[torch.tensor(BS_SPATIAL,device="cuda:0")]
            for u in (0,1):
                value=model.fields[f"s{s}_b{b}_u{u}"](tx,world[rows,u*2:(u+1)*2],frequency).permute(1,0,2,3)
                result[rows,b*8:(b+1)*8,u*2:(u+1)*2]=value*model.log_output_gain.exp()
    return result


def subset_stats(model,data,ids,kind):
    if kind not in ("FIT","validation"):raise ValueError("subset role")
    model.eval()
    with torch.no_grad():
        value=subset(model,data,ids)
        h=data.read_fit(ids,FREQUENCY_IDS) if kind=="FIT" else data.read_validation(ids,FREQUENCY_IDS)
        target=torch.tensor(h[:,BS_PORTS],device="cuda:0",dtype=torch.complex64)
        error=float((value-target).abs().square().sum(dtype=torch.float64))
        energy=float(target.abs().square().sum(dtype=torch.float64))
        return dict(nmse=error/max(energy,1e-30),mse=error/target.numel(),zero_mse=energy/target.numel(),
                    prediction_energy=float(value.abs().square().mean()),count=len(ids),shape=list(value.shape))


def step_loss(model,data,cache,rng,*,single=None,step=1):
    if single is None:
        s=int(rng.integers(2));b=int(rng.integers(2));u=int(rng.integers(2))
        allowed=data.fit_ids[data.sites(data.fit_ids)==s]
        ids=rng.choice(allowed,size=min(2,len(allowed)),replace=False)
        mm=np.sort(rng.choice(128,size=8,replace=False));ff=np.sort(rng.choice(192,size=4,replace=False))
    else:
        ids=np.asarray([single],np.int64);s=int(data.sites(ids)[0]);b=(step//2)%2;u=step%2
        mm=BS_SPATIAL;ff=np.sort(rng.choice(FREQUENCY_IDS,size=4,replace=False))
    p=torch.tensor(data.positions(ids),device="cuda:0",dtype=torch.float64)
    site=torch.tensor(data.sites(ids),device="cuda:0",dtype=torch.int64)
    world=data.geometry.ue_world_positions(p,site)[:,u*2:(u+1)*2]
    tx=model.bound_bs_ports[s,b].reshape(128,3)[torch.tensor(mm,device="cuda:0")]
    f=torch.tensor(data.frequencies(ff),device="cuda:0",dtype=torch.float64)
    value=model.fields[f"s{s}_b{b}_u{u}"](tx,world,f).permute(1,0,2,3)*model.log_output_gain.exp()
    target=cache.read(ids,ff)[:,b*128+torch.tensor(mm,device="cuda:0"),u*2:(u+1)*2]
    loss=(value-target).abs().square().mean()
    return loss,dict(mse=float(loss.detach()),zero_mse=float(target.abs().square().mean()),prediction_energy=float(value.abs().square().mean().detach()),scalar_labels=target.numel())


def set_lr(opt,effective,max_steps):
    for group in opt.param_groups:
        initial=group["initial_lr"]
        group["lr"]=1e-6+(initial-1e-6)*.5*(1+math.cos(math.pi*min(effective/max_steps,1.)))


def train_one(model,opt,data,cache,rng,*,single=None,step=1):
    model.train();opt.zero_grad(set_to_none=True)
    loss,detail=step_loss(model,data,cache,rng,single=single,step=step)
    if not bool(torch.isfinite(loss)):raise FloatingPointError("Nonfinite NeRF2 loss")
    loss.backward()
    if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):raise FloatingPointError("Nonfinite NeRF2 gradient")
    grad=torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step()
    return dict(**detail,gradient_norm=float(grad))


def health(args):
    gpu=device_guard(GPU_UUID);free,total=torch.cuda.mem_get_info()
    if free<36*(1<<30):raise RuntimeError("Insufficient free GPU memory for concurrent bounded health")
    torch.cuda.set_per_process_memory_fraction((32*(1<<30))/total,0)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    seed_all(42);data=FoldData(0,"screen","nerf2");cache=legacy.FitGPUCache(data,on_gpu=True)
    paths=output_paths("nerf2",0,"health_city810_v2");source=identity()
    # Median-energy positive FIT position: nonzero fixture, never a query label.
    energies=[]
    for first in range(0,len(data.fit_ids),128):
        ids=data.fit_ids[first:first+128]
        slots=torch.tensor(cache.slot[ids],device="cuda:0")
        energies.extend(cache.value.index_select(0,slots).abs().square().mean((1,2,3)).cpu().tolist())
    energies=np.asarray(energies);positive=np.flatnonzero(energies>0)
    order=positive[np.argsort(energies[positive])];single=int(data.fit_ids[order[len(order)//2]])
    fixture=np.concatenate([x[np.linspace(0,len(x)-1,4,dtype=int)] for x in
       [data.fit_ids[data.sites(data.fit_ids)==s] for s in (0,1)]])
    started=time.monotonic();all_records=[]
    for stage,steps in (("single_position",1000),("full_FIT",3000)):
        seed_all(42);model=build(data);opt=optimizer(model);rng=np.random.default_rng(42)
        rows=np.array([single]) if stage=="single_position" else fixture
        records=[dict(step=0,**subset_stats(model,data,rows,"FIT"))]
        for step in range(1,steps+1):
            detail=train_one(model,opt,data,cache,rng,single=single if stage=="single_position" else None,step=step)
            if step%250==0:
                event=dict(stage=stage,step=step,monitor=subset_stats(model,data,rows,"FIT"),last_training_batch=detail,elapsed_seconds=time.monotonic()-started)
                records.append(event);print(json.dumps(event),flush=True)
                atomic_json(paths["report"]/"progress.json",dict(stage=stage,records=records))
            if time.monotonic()-started>args.max_seconds:break
        all_records.append(dict(stage=stage,records=records,steps=step))
        atomic_torch_save(paths["artifact"]/(stage+".pt"),dict(model=model.state_dict(),source_identity=source,records=records,steps=step,config=metadata()))
        del model,opt;torch.cuda.empty_cache()
        if time.monotonic()-started>args.max_seconds:break
    result=dict(status="complete",kind="bounded FIT-only city-domain health",records=all_records,single_FIT_id=single,
      full_FIT_fixture_ids=fixture.tolist(),seconds=time.monotonic()-started,GPU=gpu,initial_free_bytes=free,
      maximum_allocator_bytes=32*(1<<30),peak_cuda_bytes=torch.cuda.max_memory_allocated(),source_identity=source,
      config=metadata(),validation_labels_read=False,outer_truth_read=False,
      scheduling="root-authorized <=5min diagnostic exception sharing GPU3 with unchanged nGRF main")
    atomic_json(paths["report"]/"result.json",result);print(json.dumps(result),flush=True)


def predict_full(model,data,ids):
    model.eval()
    p=torch.tensor(data.positions(ids),device="cuda:0",dtype=torch.float64)
    s=torch.tensor(data.sites(ids),device="cuda:0",dtype=torch.int64)
    f=torch.tensor(data.frequencies(),device="cuda:0",dtype=torch.float64)
    with torch.no_grad():return model(p,s,f)*model.log_output_gain.exp()


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--mode",choices=["health"],default="health");ap.add_argument("--max-seconds",type=float,default=260.)
    args=ap.parse_args()
    try:health(args)
    except BaseException:
        paths=output_paths("nerf2",0,"health_city810_v2");(paths["report"]/"failure.txt").write_text(traceback.format_exc());raise

if __name__=="__main__":main()
