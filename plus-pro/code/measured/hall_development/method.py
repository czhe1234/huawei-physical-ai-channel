"""Measured V16 core with bounded angular correction and controlled trust."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from measured.hall_adapters.core import HallPathState,MeasuredV16,norm,render_endpoints,synthesize
from measured.hall_adapters.metrics import torch_directional_power
class BoundedState(HallPathState):
    def __init__(self,coordinate_bound=(.5,.25,0.),gate_cap=.3):
        super().__init__();self.gate_cap=gate_cap
        self.register_buffer("coordinate_bound",torch.tensor(coordinate_bound,dtype=torch.float32))
        self.gate_head=nn.Linear(self.config.hidden,1)
        for head in [self.score_head,self.coordinate_head,self.gate_head]:nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        self.gate_head.bias.data.fill_(np.log(.05/(gate_cap-.05)))
    def forward(self,batch):
        mask=batch["mask"];safe=mask.clone();empty=~safe.any(-1);safe[empty,0]=True
        raw=batch["base"].masked_fill(~mask[...,None],0)
        seq=batch["sequence_geometry"].masked_fill(~mask[...,None,None],0)
        sequence=self.sequence_state(seq,batch["interactions"].masked_fill(~mask[...,None],0),batch["depth"])
        hidden=self.fusion(torch.cat([self.base_input(raw),sequence],-1))
        hidden=self.set_encoder(hidden,src_key_padding_mask=~safe)
        score=self.config.score_limit*torch.tanh(self.score_head(hidden)[...,0])
        offset=torch.tanh(self.coordinate_head(hidden))*self.coordinate_bound
        mass=torch.softmax((batch["base_log_mass"]+score).masked_fill(~safe,-1e9),-1)*mask
        pooled=(hidden*mask[...,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        gate=self.gate_cap*torch.sigmoid(self.gate_head(pooled)[:,0])*(~empty)
        return dict(mass=norm(mass,(-1,)),coordinate=batch["coordinate"]+offset,score_residual=score,coordinate_residual=offset,gate=gate)
def angular_writeback(base,desired):
    # Keeps the carrier's per-frequency spatial phase and writes only broadband
    # angular magnitudes. No current query-channel phase or delay enters this.
    total=base.abs().square().sum((-3,-2,-1),keepdim=True)
    angle=torch.fft.fft2(base,dim=(-3,-2),norm="ortho")
    current=angle.abs().square().mean(-1,keepdim=True)
    shape=angle/current.clamp_min(1e-20).sqrt()
    shape=torch.where(current>1e-20,shape,torch.ones_like(shape))
    desired_energy=desired.reshape(len(base),1,8,4,1)*total/base.shape[-1]
    value=torch.fft.ifft2(shape*desired_energy.clamp_min(1e-30).sqrt(),dim=(-3,-2),norm="ortho")
    return value*torch.sqrt(total/value.abs().square().sum((-3,-2,-1),keepdim=True).clamp_min(1e-20))
class DevelopedV16(MeasuredV16):
    def __init__(self,coordinate_bound=(.5,.25,0.),gate_cap=.3):
        nn.Module.__init__(self);self.state=BoundedState(coordinate_bound,gate_cap)
        self.register_buffer("feature_mean",torch.zeros(120));self.register_buffer("feature_scale",torch.ones(120))
    def forward(self,batch,full_channel=False):
        data=dict(batch);data["base"]=(batch["base"]-self.feature_mean)/self.feature_scale
        state=self.state(data)
        endpoint,_=render_endpoints(state["mass"],state["coordinate"],1)
        base=norm(torch_directional_power(batch["channel"]),(-1,)).reshape(-1,8,4)
        carrier_valid=batch["channel"].abs().square().sum((-3,-2,-1)).reshape(-1)>0
        state["gate"]=state["gate"]*carrier_valid
        a=state["gate"][:,None,None]
        desired=norm(((1-a)*base.clamp_min(1e-20).sqrt()+a*endpoint.clamp_min(1e-20).sqrt()).square(),(-2,-1))
        desired=torch.where((batch["mask"].any(-1)&carrier_valid)[:,None,None],desired,base)
        if full_channel:return angular_writeback(batch["channel"],desired),state
        return desired.flatten(-2),state
    def regularizer(self,state,mask):
        if not mask.any():return state["gate"].sum()*0
        return .0001*state["score_residual"][mask].square().mean()+.0001*state["coordinate_residual"][mask].square().mean()


    @torch.inference_mode()
    def predict(self,query,base_prediction,support):
        from measured.hall_adapters.core import batch_from_support
        from measured.hall_adapters.common import check_prediction
        check_prediction(base_prediction,query)
        self.eval()
        data={k:v.to(self.feature_mean.device) for k,v in batch_from_support(base_prediction["channel"],support).items()}
        value,state=self(data,full_channel=True)
        return check_prediction(dict(method="V16_measured_bounded_angular_core",channel=value.cpu().numpy().reshape(len(query.row_ids),2,8,4,1024),
            source_ids=base_prediction.get("source_ids",np.array([],int)),
            diagnostics=dict(gate=state["gate"].cpu().numpy().reshape(-1,2).tolist(),query_phase_alignment=False)),query)
