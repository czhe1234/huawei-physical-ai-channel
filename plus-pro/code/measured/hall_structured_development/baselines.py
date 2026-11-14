"""Full-channel interpolation baselines and a position-only spectral MLP."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from scipy.spatial.distance import cdist
from measured.hall_structured_v2.observables import probability,unit_channel
from measured.hall_structured_v2.synthesis import angle_project
from measured.hall_structured_v2.foundation import band_energy
def write_band(carrier,band):
    q=probability(band,(-2,-1));energy=band.sum((-2,-1))
    h=angle_project(unit_channel(carrier),q)
    return unit_channel(h)*(energy*carrier.shape[-1]).clamp_min(0).sqrt()[...,None,None,None]
class SpectralMLP(nn.Module):
    def __init__(self,position,power,levels=6,hidden=512,subbands=16):
        super().__init__();self.subbands=subbands
        p=np.asarray(position);self.register_buffer("center",torch.tensor(p.mean(0),dtype=torch.float32))
        self.register_buffer("scale",torch.tensor(np.maximum(p.std(0),.1),dtype=torch.float32))
        self.register_buffer("frequency",2**torch.arange(levels,dtype=torch.float32)*torch.pi)
        self.register_buffer("power_reference",power.reference.clone());self.register_buffer("power_scale_db",power.scale_db.clone())
        self.network=nn.Sequential(nn.Linear(3+6*levels,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,2*(32*subbands+1)))
        nn.init.zeros_(self.network[-1].bias)
    def set_fit_statistics(self,data):pass
    def forward(self,data):
        pos=torch.tensor(data.position,device=data.carrier.device,dtype=torch.float32)
        x=(pos-self.center)/self.scale;phase=x[...,None]*self.frequency
        encoded=torch.cat([x,phase.sin().flatten(-2),phase.cos().flatten(-2)],-1)
        raw=self.network(encoded).reshape(-1,2,32*self.subbands+1)
        q=raw[...,:-1].softmax(-1).reshape(-1,2,32,self.subbands)
        energy=self.power_reference*torch.pow(10.,3*raw[...,-1].tanh()*self.power_scale_db/10)
        channel=write_band(data.carrier,q*energy[...,None,None])
        return dict(channel=channel,residual=channel.real.sum()*0)
class SpectralRBF:
    def __init__(self,bank,length,ridge):
        self.position=bank.position;self.length=length
        target=np.sqrt(np.maximum(band_energy(bank.channel,16),0)).reshape(len(bank.row_ids),-1)
        k=np.exp(-cdist(bank.position[:,:2],bank.position[:,:2],"sqeuclidean")/(2*length**2))
        self.coefficient=np.linalg.solve(k+ridge*np.eye(len(k)),target)
    def predict(self,data):
        k=np.exp(-cdist(data.position[:,:2],self.position[:,:2],"sqeuclidean")/(2*self.length**2))
        band=np.maximum(k@self.coefficient,0).reshape(-1,2,32,16)**2
        with torch.no_grad():return write_band(data.carrier,torch.tensor(band,dtype=torch.float32,device=data.carrier.device))
