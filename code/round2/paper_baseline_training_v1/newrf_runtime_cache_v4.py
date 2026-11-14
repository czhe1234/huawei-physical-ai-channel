"""Exact FIT-only runtime caches; never store validation/outer targets."""
from __future__ import annotations
import time
import numpy as np
import torch
from .newrf_runner import FitDirectionCache

class MemoizedFitDirections(FitDirectionCache):
    def __init__(self,bank):
        super().__init__(bank);self._fallback={};self.info["FIT_zero_path_memoization"]=True
    def scalar(self,model,row,field,spatial,origin_all,site,tx_all):
        row=int(row)
        if row not in self.allowed:raise PermissionError("FIT-only direction cache")
        if self.count[row]:return super().scalar(model,row,field,spatial,origin_all,site,tx_all)
        p,u=divmod(int(field),4)
        if row not in self._fallback:
            provider=model.ray_provider
            if hasattr(provider,"parameters") and any(x.requires_grad for x in provider.parameters()):
                raise RuntimeError("Cannot cache a trainable ray provider")
            with torch.no_grad():
                self._fallback[row]=provider.directions(origin_all[None],site[None],tx_all[None])[0].detach()
        return self._fallback[row][p,spatial,u]
    @property
    def memoized_bytes(self):
        return sum(t.numel()*t.element_size() for t in self._fallback.values())

def cache_raw_fit_targets(data):
    if hasattr(data,"_native_raw_target_cache"):return
    ids=np.asarray(data.fit_ids,np.int64)
    slots=np.full(4000,-1,np.int64);slots[ids]=np.arange(len(ids))
    value=torch.empty((len(ids),256,4,192),device="cuda:0",dtype=torch.complex64)
    for first in range(0,len(ids),64):
        rows=ids[first:first+64]
        value[first:first+len(rows)].copy_(torch.from_numpy(data.read_fit(rows,normalized=False)).to("cuda:0"))
    data._native_raw_target_cache=(slots,value)

def station_target_cached(data,rows,field,bs_ids,freq_ids,device):
    rows=np.asarray(rows,np.int64);bs=np.asarray(bs_ids,np.int64);ff=np.asarray(freq_ids,np.int64)
    p,u=divmod(int(field),4)
    if rows.ndim!=1 or bs.shape!=rows.shape or ff.shape!=rows.shape:raise ValueError("Paired scalar labels")
    if p not in (0,1) or u not in range(4) or np.any((bs<0)|(bs>=128)) or np.any((ff<0)|(ff>=192)):raise ValueError("Port/frequency")
    if not hasattr(data,"_native_raw_target_cache"):
        from .newrf_native_batch_v4 import station_target
        return station_target(data,rows,field,bs,ff,device)
    slots,value=data._native_raw_target_cache
    index=slots[rows]
    if np.any(index<0):raise PermissionError("Label cache outside this role FIT")
    return value[torch.tensor(index,device=value.device),p*128+torch.tensor(bs,device=value.device),
                 u,torch.tensor(ff,device=value.device)]

def main():
    from .common import FoldData,seed_all,device_guard,atomic_json
    from .newrf_runner import GPU_UUID,GEOMETRY,FitGeometryBank,FitDirectionCache
    from .newrf_native_batch_v4 import build_native_batch_model,station_forward,station_target,native_weighted_loss,output_paths_v4
    torch.set_num_threads(4);seed_all(42);device_guard(GPU_UUID);torch.backends.cuda.matmul.allow_tf32=False
    data=FoldData(0,"screen","newrf");bank=FitGeometryBank(GEOMETRY,data.inputs,outer=0,role="screen_fit")
    original=FitDirectionCache(bank).to("cuda:0");memo=MemoizedFitDirections(bank).to("cuda:0")
    model=build_native_batch_model(data,memo).to("cuda:0")
    cache_raw_fit_targets(data)
    rng=np.random.default_rng(731)
    exact_targets=True;exact_directions=True
    for field in range(8):
        rows=rng.choice(data.fit_ids,32,replace=True);bs=rng.integers(0,128,32);ff=rng.integers(0,192,32)
        left=station_target(data,rows,field,bs,ff,"cuda:0");right=station_target_cached(data,rows,field,bs,ff,"cuda:0")
        exact_targets &= bool(torch.equal(left,right))
    zeros=np.array([r for r in data.fit_ids if memo.count[r]==0],np.int64)[:8]
    with torch.no_grad():
        for row in zeros:
            pos=torch.tensor(data.positions([row]),device="cuda:0",dtype=torch.float64)
            site=torch.tensor(data.sites([row]),device="cuda:0",dtype=torch.long)
            origins=model.spec.ue_world_positions(pos,site)[0];tx=model.bs_ports[site[0]]
            for field in range(8):
                bs=torch.tensor([0,61,127],device="cuda:0")
                a=original.scalar(model,int(row),field,bs,origins,site[0],tx)
                b=memo.scalar(model,int(row),field,bs,origins,site[0],tx)
                c=memo.scalar(model,int(row),field,bs,origins,site[0],tx)
                exact_directions &= bool(torch.equal(a,b) and torch.equal(b,c))
    if not exact_targets or not exact_directions:raise AssertionError("Exact cache mismatch")
    opt=torch.optim.Adam(model.parameters(),lr=.0005);times=[]
    for step in range(200):
        tick=time.monotonic();rows=rng.choice(data.fit_ids,32,replace=True)
        bs=rng.integers(0,128,32);ff=rng.integers(0,192,32);field=step%8
        target=station_target_cached(data,rows,field,bs,ff,"cuda:0")
        h,hc=station_forward(model,memo,data,rows,field,bs,ff)
        loss,_,_=native_weighted_loss(h,hc,target)
        opt.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        if not torch.isfinite(loss) or not torch.isfinite(norm):raise FloatingPointError("Cache throughput batch")
        opt.step();torch.cuda.synchronize();times.append(time.monotonic()-tick)
    d=dict(status="passed",raw_scalar_labels_torch_equal=exact_targets,FIT_zero_path_directions_torch_equal=exact_directions,
           zero_rows_tested=zeros.tolist(),FIT_only=True,held_targets_cached=False,model_or_loss_changed=False,
           benchmark_updates=200,weights_discarded=True,mean_seconds=float(np.mean(times[50:])),
           median_seconds=float(np.median(times[50:])),q90_seconds=float(np.quantile(times[50:],.9)),
           memoized_geometry_bytes=memo.memoized_bytes,label_cache_bytes=data._native_raw_target_cache[1].numel()*8,
           peak_gpu_GiB=torch.cuda.max_memory_allocated()/2**30)
    atomic_json(output_paths_v4("newrf")["report"]/"runtime_cache_check.json",d);print(d,flush=True)

if __name__=="__main__":main()

