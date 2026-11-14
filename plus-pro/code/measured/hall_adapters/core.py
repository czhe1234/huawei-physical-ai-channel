"""Actual frozen V16 encoder core, measured single-TX heads and synthesis."""
from __future__ import annotations
import sys
import numpy as np
import torch
from torch import nn
from project_paths import PATHS
from .metrics import torch_directional_power
from .common import check_prediction

FINAL_SOURCE = PATHS.root / "final_code/V16_reorganized_submission/code/round2/physical_ai_submission"
import importlib.util
import importlib
name = "hall_frozen_v16"
if name not in sys.modules:
    spec = importlib.util.spec_from_file_location(name, FINAL_SOURCE / "__init__.py", submodule_search_locations=[str(FINAL_SOURCE)])
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
original = importlib.import_module(name + ".component_state.model")
ComponentStateModel, ModelConfig = original.ComponentStateModel, original.ModelConfig

def norm(x,dims):
    return x.clamp_min(0)/x.clamp_min(0).sum(dims,keepdim=True).clamp_min(1e-20)

def unit(x):
    magnitude=x.abs()
    return torch.where(magnitude>1e-20,x/magnitude.clamp_min(1e-20),torch.ones_like(x))

def delay_power(h):
    return norm(torch.fft.ifft(h,dim=-1,norm="ortho").abs().square().sum((-3,-2)),(-1,))

def render_endpoints(mass,coordinate,frequency):
    kernels=[]
    for axis,n in enumerate([8,4,frequency]):
        index=torch.arange(n,device=coordinate.device,dtype=coordinate.dtype)
        wave=torch.exp((2j if axis<2 else -2j)*torch.pi*coordinate[...,axis,None]*index/n)
        transform=torch.fft.fft(wave,dim=-1) if axis<2 else torch.fft.ifft(wave,dim=-1)
        kernels.append(norm(transform.abs().square(),(-1,)))
    power=norm(torch.einsum("bk,bkh,bkv->bhv",mass,kernels[0],kernels[1]),(-2,-1))
    delay=norm(torch.einsum("bk,bkd->bd",mass,kernels[2]),(-1,))
    return power,delay

def synthesize(base,endpoint_power,endpoint_delay,blend=.2,cycles=4):
    if blend==0:return base.clone()
    p=norm(torch_directional_power(base).reshape(len(base),8,4),(-2,-1))
    d=delay_power(base).reshape(len(base),-1)
    desired_p=norm(((1-blend)*p.clamp_min(1e-20).sqrt()+blend*endpoint_power.clamp_min(1e-20).sqrt()).square(),(-2,-1))
    desired_d=norm(((1-blend)*d.clamp_min(1e-20).sqrt()+blend*endpoint_delay.clamp_min(1e-20).sqrt()).square(),(-1,))
    total=base.abs().square().sum((-3,-2,-1),keepdim=True)
    value=base
    for _ in range(cycles):
        angle=torch.fft.fft2(value,dim=(-3,-2),norm="ortho")
        current=angle.abs().square().mean(-1,keepdim=True)
        shape=angle/torch.sqrt(current.clamp_min(1e-20))
        shape=torch.where(current>1e-20,shape,torch.ones_like(shape))
        desired=desired_p[:,None,:,:,None]*total/base.shape[-1]
        value=torch.fft.ifft2(shape*desired.clamp_min(1e-30).sqrt(),dim=(-3,-2),norm="ortho")
        delay=torch.fft.ifft(value,dim=-1,norm="ortho")
        current=delay.abs().square().sum((-3,-2),keepdim=True)
        shape=delay/current.clamp_min(1e-20).sqrt()
        shape=torch.where(current>1e-20,shape,torch.ones_like(shape)/np.sqrt(32))
        value=torch.fft.fft(shape*(desired_d[:,None,None,None,:]*total).clamp_min(1e-30).sqrt(),dim=-1,norm="ortho")
        value=value*torch.sqrt(total/value.abs().square().sum((-3,-2,-1),keepdim=True).clamp_min(1e-20))
    return value

class HallPathState(ComponentStateModel):
    """Same sequence/set encoders. No fictitious UE/polarization port head."""
    def __init__(self):
        super().__init__(120,21,ModelConfig())
        self.port_head=nn.Identity()
        for module in [self.object_embedding,self.primitive_embedding,self.hash_embedding_a,self.hash_embedding_b]:
            for parameter in module.parameters():parameter.requires_grad_(False)

    def forward(self,batch):
        mask=batch["mask"]
        safe=mask.clone();empty=~safe.any(-1);safe[empty,0]=True
        raw=batch["base"].masked_fill(~mask[...,None],0)
        seq=batch["sequence_geometry"].masked_fill(~mask[...,None,None],0)
        sequence=self.sequence_state(seq,batch["interactions"].masked_fill(~mask[...,None],0),batch["depth"])
        hidden=self.fusion(torch.cat([self.base_input(raw),sequence],-1))
        hidden=self.set_encoder(hidden,src_key_padding_mask=~safe)
        score=self.config.score_limit*torch.tanh(self.score_head(hidden)[...,0])
        offset=torch.tanh(self.coordinate_head(hidden))*hidden.new_tensor([2.,4.,2.])
        mass=torch.softmax((batch["base_log_mass"]+score).masked_fill(~safe,-1e9),-1)*mask
        return dict(mass=norm(mass,(-1,)),coordinate=batch["coordinate"]+offset,
                    score_residual=score,coordinate_residual=offset)

def batch_from_support(base,support):
    """Geometry plus predicted foundation only; never a query channel label."""
    h=torch.as_tensor(np.array(base),dtype=torch.complex64)
    n,p,_,_,f=h.shape;k=support["mass"].shape[-1]
    coordinate=np.array(support["coordinate"],copy=True)
    anchor=delay_power(h).argmax(-1).cpu().numpy()
    coordinate[...,2]=(coordinate[...,2]+anchor[...,None])%f
    feature=np.array(support["features"],copy=True)
    period=np.array([8,4,f]);phase=2*np.pi*coordinate/period
    feature[...,:3]=np.sin(phase);feature[...,3:6]=np.cos(phase);feature[...,6:9]=coordinate/period
    feature[...,17:19]=0
    power=norm(torch_directional_power(h),(-1,))
    delay=delay_power(h).reshape(n,p,64,f//64).sum(-1)
    context=torch.cat([power.clamp_min(1e-8).log(),delay.clamp_min(1e-8).log()],-1).cpu().numpy()
    feature=np.concatenate([feature,np.broadcast_to(context[:,:,None,:],(n,p,k,96))],-1)
    return dict(base=torch.tensor(feature.reshape(-1,k,120),dtype=torch.float32),
                sequence_geometry=torch.tensor(np.array(support["sequence"]).reshape(-1,k,16,21),dtype=torch.float32),
                interactions=torch.tensor(np.array(support["interaction"]).reshape(-1,k,16),dtype=torch.long),
                depth=torch.tensor(np.array(support["depth"]).reshape(-1,k),dtype=torch.long),
                coordinate=torch.tensor(coordinate.reshape(-1,k,3),dtype=torch.float32),
                base_log_mass=torch.tensor(np.log(np.maximum(support["mass"],1e-30)).reshape(-1,k),dtype=torch.float32),
                mask=torch.tensor(np.array(support["mask"]).reshape(-1,k),dtype=torch.bool),
                channel=h.reshape(-1,1,8,4,f))

class MeasuredV16(nn.Module):
    def __init__(self):
        super().__init__();self.state=HallPathState()
        self.register_buffer("feature_mean",torch.zeros(120))
        self.register_buffer("feature_scale",torch.ones(120))

    def set_fit_statistics(self,fit_batch):
        observed=fit_batch["base"][fit_batch["mask"]]
        if not len(observed):raise ValueError("No fit geometry tokens for normalization")
        self.feature_mean.copy_(observed.mean(0));self.feature_scale.copy_(observed.std(0,unbiased=False).clamp_min(1e-3))

    def forward(self,batch):
        data=dict(batch);data["base"]=(batch["base"]-self.feature_mean)/self.feature_scale
        state=self.state(data)
        power,delay=render_endpoints(state["mass"],state["coordinate"],batch["channel"].shape[-1])
        value=synthesize(batch["channel"],power,delay)
        valid=batch["mask"].any(-1)[:,None,None,None,None]
        return torch.where(valid,value,batch["channel"]),state

    @torch.inference_mode()
    def predict(self,query,base_prediction,support):
        self.eval();device=self.feature_mean.device
        data={k:v.to(device) for k,v in batch_from_support(base_prediction["channel"],support).items()}
        value,_=self(data)
        out=value.cpu().numpy().reshape(len(query.row_ids),2,8,4,query.spec.frequency_count)
        return check_prediction(dict(method="V16_measured_single_TX_core",channel=out,
             source_ids=base_prediction.get("source_ids",np.array([],int))),query)
