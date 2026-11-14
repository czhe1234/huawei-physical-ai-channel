"""Frequency-resolved phase-safe experts and trainable donor/expert calibration."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from measured.hall_adapters.foundation import affine_weights
from .observables import probability,unit_channel,EPS
from .synthesis import angle_project
from .cache import spec_signature
@dataclass
class Prepared:
    row_ids:np.ndarray
    position:np.ndarray
    spec:object
    expert_energy:torch.Tensor
    carrier:torch.Tensor
    donor_distance:torch.Tensor
    donor_span:torch.Tensor
    source_ids:list
    geometry:object=None
    def to(self,device):
        return Prepared(self.row_ids,self.position,self.spec,self.expert_energy.to(device),self.carrier.to(device),
            self.donor_distance.to(device),self.donor_span.to(device),self.source_ids,self.geometry)
def band_energy(h,s):
    z=np.fft.fft2(h,axes=(-3,-2),norm="ortho").reshape(*h.shape[:-3],32,s,h.shape[-1]//s)
    return (np.abs(z)**2).sum(-1)/h.shape[-1]
def prepare(bank,query,subbands=16,exclude="block",geometry=None):
    if exclude not in ["block","self","none"]:raise ValueError("Unknown crossfit policy")
    if spec_signature(bank.spec)!=spec_signature(query.spec):raise ValueError("Bank/query physical array specs differ")
    lookup={int(r):i for i,r in enumerate(bank.row_ids)}
    for rid,pos in zip(query.row_ids,query.position):
        if int(rid) in lookup and not np.allclose(pos,bank.position[lookup[int(rid)]],rtol=0,atol=1e-7):raise ValueError("Known query row has a different position")
    f=bank.spec.frequency_count
    if f%subbands:raise ValueError("Full physical frequency axis must divide into subbands")
    out=np.zeros((len(query.row_ids),13,2,32,subbands),np.float32)
    carriers=[];distances=[];spans=[];donors=[]
    original=band_energy(bank.channel,subbands)
    for i,(rid,pos) in enumerate(zip(query.row_ids,query.position)):
        d=np.linalg.norm(bank.position[:,:2]-pos[:2],axis=-1)
        allowed=bank.row_ids!=rid
        if exclude=="block" and rid in bank.row_ids:
            allowed&=np.any(np.floor(bank.position[:,:2]/4)!=np.floor(pos[:2]/4),axis=-1)
        if exclude=="none":allowed=np.ones(len(d),bool)
        order=np.flatnonzero(allowed);order=order[np.lexsort((bank.row_ids[order],d[order]))][:32]
        if len(order)<3:raise ValueError("At least3 permitted distinct donors are needed")
        carriers.append(bank.channel[order[0]]);distances.append(d[order[0]]);spans.append(d[order[-1]])
        donors.append(bank.row_ids[order].tolist());roots=np.sqrt(np.maximum(original[order],0))
        out[i,0]=original[order[0]]
        for j,k in enumerate([4,8,16,24],1):
            take=min(k,len(order));ids=order[:take];w=1/np.maximum(d[ids],1e-6)**2;w/=w.sum()
            out[i,j]=np.einsum("k,kpbs->pbs",w,roots[:take])**2
        steered=[]
        for p in range(2):
            delta=bank.position[order]-bank.spec.center[p];r=np.linalg.norm(delta,axis=-1)
            u=delta/r[:,None];uq=(pos-bank.spec.center[p])/np.linalg.norm(pos-bank.spec.center[p])
            phase=2*np.pi*bank.spec.carrier_hz/299792458.*np.einsum("ki,hvi->khv",uq[None]-u,bank.spec.element_offsets[p])
            h=bank.channel[order,p]*np.exp(1j*phase)[...,None]
            steered.append(np.sqrt(np.maximum(band_energy(h,subbands),0)))
        steer=np.stack(steered,axis=1)
        for j,k in enumerate([8,16,24,32]):
            take=min(k,len(order));ids=order[:take]
            w=affine_weights(bank.position[ids,:2]-pos[:2],d[ids],.3)
            out[i,5+j]=np.maximum(np.einsum("k,kpbs->pbs",w,roots[:take]),0)**2
            out[i,9+j]=np.maximum(np.einsum("k,kpbs->pbs",w,steer[:take]),0)**2
    if not np.isfinite(out).all():raise ValueError("Nonfinite expert energy")
    return Prepared(np.array(query.row_ids),np.array(query.position),query.spec,torch.tensor(out),torch.tensor(np.array(carriers),dtype=torch.complex64),
       torch.tensor(distances,dtype=torch.float32),torch.tensor(spans,dtype=torch.float32),donors,geometry)
class Foundation(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_logits=nn.Parameter(torch.zeros(2,13))
        self.donor_correction=nn.Sequential(nn.Linear(33,64),nn.SiLU(),nn.Linear(64,13))
        nn.init.zeros_(self.donor_correction[-1].weight);nn.init.zeros_(self.donor_correction[-1].bias)
    def forward(self,data,power_reference):
        energy=data.expert_energy.permute(0,2,1,3,4);total=energy.sum((-2,-1))
        q=probability(energy,(-2,-1));entropy=-(q*q.clamp_min(1e-12).log()).sum((-2,-1))
        position=torch.tensor(data.position,dtype=torch.float32,device=energy.device)
        center=torch.tensor(np.array(data.spec.center),dtype=torch.float32,device=energy.device)
        relative=(position[:,None]-center[None])/30
        near=data.donor_distance[:,None,None].expand(-1,2,1)/10
        span=data.donor_span[:,None,None].expand(-1,2,1)/10
        panel=torch.eye(2,device=energy.device)[None].expand(len(position),-1,-1)
        context=torch.cat([(total/power_reference[None,:,None]).clamp_min(1e-12).log(),entropy,relative,near,span,panel],-1)
        logits=self.global_logits[None]+.5*torch.tanh(self.donor_correction(context))
        weights=logits.softmax(-1)
        band=torch.einsum("npe,npebs->npbs",weights,energy)
        predicted_energy=band.sum((-2,-1));q=probability(band,(-2,-1))
        h=angle_project(unit_channel(data.carrier),q)
        h=unit_channel(h)*(predicted_energy*data.spec.frequency_count).sqrt()[...,None,None,None]
        return h,dict(expert_weights=weights,band_energy=band,foundation_energy=predicted_energy,donor_context=context)
