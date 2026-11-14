"""V16 sequence/set route with measured frequency, delay and array-energy heads."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from measured.hall_adapters.core import ComponentStateModel,ModelConfig
from measured.hall_adapters.common import check_prediction
from .config import Config,LAGS
from .cache import spec_signature
from .foundation import Foundation
from .observables import structure,probability,total_energy,FitPower,beam_channel,EPS
from .synthesis import relative_anchor,delay_distribution,element_distribution,root_blend,reconstruct,canonicalize,restore_reference
PATH_FEATURES=("sin_h","sin_v","sin_relative_delay","cos_h","cos_v","cos_relative_delay",
 "h_fraction","v_fraction","relative_delay_fraction","log_prior_mass","depth","geometry_source","measurement_source",
 "seed_presence","energy_log_cv","variance_known","base_source_fraction","scatter_source_fraction","path_length",
 "direct_range","position_x","position_y","position_z","ray_converged","geometric_direct_path",
 "nearest_donor_distance","donor_span","variance_unknown","valid")
def render_joint(mass,coordinate,receive_log_gain,anchor,frequency):
    n,k,_=coordinate.shape;dtype=coordinate.dtype;device=coordinate.device
    h,v=torch.meshgrid(torch.arange(8,device=device,dtype=dtype),torch.arange(4,device=device,dtype=dtype),indexing="ij")
    phase=2*torch.pi*(coordinate[...,0,None]*h.flatten()/8+coordinate[...,1,None]*v.flatten()/4)
    wave=torch.exp(1j*phase)*receive_log_gain[:,None].exp()
    wave=wave/wave.abs().square().sum(-1,keepdim=True).clamp_min(EPS).sqrt()
    angular=torch.fft.fft2(wave.reshape(n,k,8,4),dim=(-2,-1),norm="ortho").abs().square().flatten(-2)
    angular=probability(angular,(-1,))
    f=torch.arange(frequency,device=device,dtype=dtype)
    tau=coordinate[...,2]+anchor[:,None]
    delay=torch.fft.ifft(torch.exp(-2j*torch.pi*tau[...,None]*f/frequency),dim=-1,norm="ortho").abs().square()
    delay=probability(delay,(-1,))
    joint=(mass[...,None]*angular).transpose(1,2)@delay
    covariance=torch.einsum("bk,bka,bkc->bac",mass,wave,wave.conj())
    element=torch.einsum("bk,bka->ba",mass,wave.abs().square())
    return probability(joint,(-2,-1)),covariance,probability(element,(-1,))
class State(ComponentStateModel):
    def __init__(self,input_dim,config,spec):
        super().__init__(input_dim,21,ModelConfig());self.measured_config=config
        self.port_head=nn.Identity()
        for module in [self.object_embedding,self.primitive_embedding,self.hash_embedding_a,self.hash_embedding_b]:
            for p in module.parameters():p.requires_grad_(False)
        self.lifecycle_head=nn.Linear(96,1)
        self.gate_head=nn.Linear(96,3)
        self.spectral_head=nn.Linear(96,32*config.subbands)
        self.receive_head=nn.Linear(96,32)
        self.power_head=nn.Linear(96,1)
        self.register_buffer("coordinate_bound",torch.tensor([config.horizontal_bound_bins,config.vertical_bound_bins,config.relative_delay_bound_s*spec.bandwidth_hz]))
        for head in [self.score_head,self.coordinate_head,self.lifecycle_head,self.gate_head,self.spectral_head,self.receive_head,self.power_head]:
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        self.lifecycle_head.bias.data.fill_(2.)
        self.gate_head.bias.data.fill_(np.log(.03/(config.geometry_gate_cap-.03)))
    def forward(self,batch):
        mask=batch["mask"];safe=mask.clone();safe[~safe.any(-1),0]=True
        seq=self.sequence_state(batch["sequence"],batch["interaction"],batch["depth"])
        if not self.measured_config.use_sequence:seq=seq*0
        hidden=self.fusion(torch.cat([self.base_input(batch["features"]),seq],-1))
        hidden=self.set_encoder(hidden,src_key_padding_mask=~safe)
        score=self.config.score_limit*torch.tanh(self.score_head(hidden)[...,0])
        life=self.lifecycle_head(hidden)[...,0].sigmoid()
        log_mass=batch["mass"].clamp_min(1e-20).log()+score+life.clamp_min(1e-6).log()+batch["confidence"].clamp_min(1e-6).log()
        mass=log_mass.masked_fill(~safe,-1e9).softmax(-1)*mask
        mass=probability(mass,(-1,))
        offset=self.coordinate_head(hidden).tanh()*self.coordinate_bound
        if not self.measured_config.use_relative_delay:offset=offset*offset.new_tensor([1.,1.,0.])
        pooled=(hidden*mask[...,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        gate=self.measured_config.geometry_gate_cap*self.gate_head(pooled).sigmoid()
        if not self.measured_config.use_relative_delay:gate=gate*gate.new_tensor([1.,0.,1.])
        if not self.measured_config.use_receive_energy:gate=gate*gate.new_tensor([1.,1.,0.])
        return dict(mass=mass,coordinate=batch["coordinate"]+offset,score_residual=score,coordinate_residual=offset,
          lifecycle=life,gate=gate,spectral_delta=self.measured_config.spectral_log_power_bound*self.spectral_head(pooled).tanh().reshape(-1,32,self.measured_config.subbands),
          receive_log_gain=self.measured_config.receive_log_amplitude_bound*self.receive_head(pooled).tanh(),
          power_residual=self.measured_config.power_bound_fit_std*self.power_head(pooled)[:,0].tanh())
class HallStructured(nn.Module):
    def __init__(self,spec,power:FitPower,config=Config()):
        super().__init__();self.spec=spec;self.config=config;self.fit_ids=tuple(power.fit_ids)
        self.foundation=Foundation();self.register_buffer("power_reference",power.reference.clone());self.register_buffer("power_scale_db",power.scale_db.clone())
        self.lags=tuple(x for x in LAGS if x<spec.frequency_count)
        self.context_dim=32*config.subbands+2048+33*len(self.lags)+9
        self.register_buffer("context_mean",torch.zeros(self.context_dim));self.register_buffer("context_scale",torch.ones(self.context_dim))
        self.register_buffer("statistics_ready",torch.tensor(False))
        self.context_encoder=nn.Sequential(nn.Linear(self.context_dim,128),nn.SiLU(),nn.Linear(128,config.context_width))
        self.state=State(len(PATH_FEATURES)+config.context_width,config,spec)
    def power_stats(self):return FitPower(self.power_reference,self.power_scale_db,self.fit_ids)
    def _tokens(self,data,base):
        device=base.device;n,p=base.shape[:2];f=base.shape[-1];parts=[]
        quality=torch.zeros(n,p,5,device=device)
        if self.config.use_geometry and data.geometry is not None:
            geo=data.geometry;a=geo.arrays;k=min(a["mass"].shape[-1],self.config.geometry_candidates)
            mask=torch.as_tensor(a["mask"][...,:k],device=device)
            mass=torch.as_tensor(a["mass"][...,:k],device=device,dtype=torch.float32)
            coord=torch.as_tensor(a["coordinate"][...,:k,:],device=device,dtype=torch.float32)
            presence=torch.as_tensor(geo.presence[...,:k],device=device)
            cv=torch.as_tensor(geo.energy_cv[...,:k],device=device)
            known=torch.as_tensor(geo.variance_known[...,:k],device=device)
            sources=torch.as_tensor(geo.source_fraction[...,:k,:],device=device)
            certified=float(bool(geo.quality.get("ray_convergence_certified",False)))
            w=probability(mass,(-1,))
            quality=torch.stack([(w*presence).sum(-1),(w*known).sum(-1),(w*cv).sum(-1),
                (w*sources[...,1]).sum(-1),torch.full((n,p),certified,device=device)],-1)
            confidence=presence/(1+cv)*torch.where(known,torch.ones_like(cv),torch.full_like(cv,.5))
            parts.append(dict(mask=mask,mass=w,coordinate=coord,presence=presence,cv=cv,known=known,sources=sources,
              confidence=confidence,sequence=torch.as_tensor(a["sequence"][...,:k,:,:],device=device,dtype=torch.float32),
              interaction=torch.as_tensor(a["interaction"][...,:k,:],device=device,dtype=torch.long),
              depth=torch.as_tensor(a["depth"][...,:k],device=device,dtype=torch.long),
              length=torch.as_tensor(a["features"][...,:k,14],device=device),direct=torch.as_tensor(a["features"][...,:k,15],device=device),
              geometric=torch.ones_like(mass),certified=torch.full_like(mass,certified)))
        anchor=relative_anchor(base)
        # Reserve components are extracted in the already canonical carrier frame.
        ad=torch.fft.ifft(beam_channel(base),dim=-1,norm="ortho").abs().square()
        reserve=min(self.config.reserve_candidates,32*f)
        energy,index=ad.flatten(-2).topk(reserve,dim=-1)
        angular=index//f;delay=(index%f).to(torch.float32)
        relative=(delay+f/2)%f-f/2
        coord=torch.stack([(angular//4).float(),(angular%4).float(),relative],-1)
        zero=torch.zeros(n,p,reserve,device=device);mask=energy>0
        parts.append(dict(mask=mask,mass=probability(energy,(-1,)),coordinate=coord,presence=torch.ones_like(zero),
           cv=zero,known=torch.ones_like(mask),sources=torch.zeros(n,p,reserve,2,device=device),
           confidence=torch.exp(-data.donor_distance[:,None,None].expand(n,p,reserve)/10),
           sequence=torch.zeros(n,p,reserve,16,21,device=device),interaction=torch.zeros(n,p,reserve,16,dtype=torch.long,device=device),
           depth=torch.zeros(n,p,reserve,dtype=torch.long,device=device),length=zero,direct=zero,geometric=zero,certified=zero))
        for part in parts:
            # Align each source in a relative-delay gauge before source competition.
            weights=probability(part["mass"],(-1,))
            phasor=(weights*torch.exp(-2j*torch.pi*part["coordinate"][...,2]/f)).sum(-1)
            center=-f*torch.angle(phasor)/(2*torch.pi)
            part["coordinate"]=part["coordinate"].clone()
            part["coordinate"][...,2]=(part["coordinate"][...,2]-center[...,None]+f/2)%f-f/2
        mass=torch.cat([z["mass"] for z in parts],-1)
        # Equal source priors in probability units, never raw RT watts versus CSI units.
        mass=probability(mass,(-1,))
        all_data={key:torch.cat([z[key] for z in parts],dim=2) for key in parts[0] if key!="mass"}
        all_data["mass"]=mass
        coordinate=all_data["coordinate"];k=coordinate.shape[-2];period=coordinate.new_tensor([8.,4.,float(f)])
        phase=2*torch.pi*coordinate/period
        pos=torch.as_tensor(data.position,dtype=torch.float32,device=device)[:,None,None].expand(n,p,k,3)/30
        near=data.donor_distance[:,None,None,None].expand(n,p,k,1)/10;span=data.donor_span[:,None,None,None].expand(n,p,k,1)/10
        depth=all_data["depth"].float()[...,None]/16;geometric=all_data["geometric"][...,None]
        feat=torch.cat([phase.sin(),phase.cos(),coordinate/period,mass.clamp_min(1e-12).log()[...,None]/10,depth,
          geometric,1-geometric,all_data["presence"][...,None],all_data["cv"].clamp_min(0).log1p()[...,None],
          all_data["known"][...,None],all_data["sources"],all_data["length"][...,None],all_data["direct"][...,None],pos,
          all_data["certified"][...,None],((all_data["depth"]==0)&(all_data["geometric"]>0))[...,None],
          near,span,(~all_data["known"])[...,None],all_data["mask"][...,None]],-1)
        if feat.shape[-1]!=len(PATH_FEATURES):raise ValueError("Path feature semantics changed")
        all_data["path_features"]=feat.float();return all_data,quality,anchor
    def _context(self,data,base,quality):
        ob=structure(base,self.config.subbands,self.lags)
        position=torch.as_tensor(data.position,dtype=torch.float32,device=base.device)[:,None].expand(-1,2,-1)/30
        cov=ob["covariance"]
        raw=torch.cat([ob["q"].clamp_min(1e-12).sqrt().flatten(-2),cov.real.flatten(-2),cov.imag.flatten(-2),
            ob["correlation"],(ob["beam_correlation"]*ob["broad"].clamp_min(1e-12).sqrt()[...,None]).flatten(-2),
            (ob["energy"]/self.power_reference).clamp_min(1e-12).log()[...,None],position,quality],-1)
        if raw.shape[-1]!=self.context_dim:raise ValueError("Invariant context size changed")
        return raw,ob
    @torch.no_grad()
    def set_fit_statistics(self,data):
        if not set(int(x) for x in data.row_ids).issubset(self.fit_ids):raise ValueError("Context normalization outside FIT")
        base,_=self.foundation(data,self.power_reference)
        base,_=canonicalize(base)
        _,quality,_=self._tokens(data,base);raw,_=self._context(data,base,quality)
        x=raw.reshape(-1,self.context_dim);self.context_mean.copy_(x.mean(0));self.context_scale.copy_(x.std(0,unbiased=False).clamp_min(1e-3))
        self.statistics_ready.fill_(True)
    def forward(self,data):
        if not bool(self.statistics_ready):raise RuntimeError("FIT-only input statistics have not been set")
        raw_base,foundation=self.foundation(data,self.power_reference)
        base,reference=canonicalize(raw_base)
        tokens,quality,anchor=self._tokens(data,base)
        context,_=self._context(data,base,quality)
        hidden=self.context_encoder((context-self.context_mean)/self.context_scale)
        n,p,k=tokens["mask"].shape
        features=torch.cat([tokens["path_features"],hidden[:,:,None].expand(-1,-1,k,-1)],-1)
        flat=dict(features=features.reshape(n*p,k,-1),mass=tokens["mass"].reshape(n*p,k),
           confidence=tokens["confidence"].reshape(n*p,k),coordinate=tokens["coordinate"].reshape(n*p,k,3),
           mask=tokens["mask"].reshape(n*p,k),sequence=tokens["sequence"].reshape(n*p,k,16,21),
           interaction=tokens["interaction"].reshape(n*p,k,16),depth=tokens["depth"].reshape(n*p,k))
        state=self.state(flat)
        joint,endpoint_cov,endpoint_element=render_joint(state["mass"],state["coordinate"],state["receive_log_gain"],torch.zeros_like(anchor.reshape(-1)),base.shape[-1])
        s=self.config.subbands;joint=joint.reshape(n,p,32,-1);angular=joint.sum(-1)
        qbase=structure(base,s,self.lags)["q"];band=qbase.sum(-2)
        qgeom=angular[...,None]*band[...,None,:]
        gate=state["gate"].reshape(n,p,3)
        q=root_blend(qbase,qgeom,gate[...,0],(-2,-1))
        q=probability(q*state["spectral_delta"].reshape(n,p,32,s).exp(),(-2,-1))
        db=delay_distribution(base);dg=probability(joint,(-1,))
        desired_delay=root_blend(db,dg,gate[...,1],(-1,))
        eb=element_distribution(base);eg=endpoint_element.reshape(n,p,32)
        desired_element=root_blend(eb,eg,gate[...,2],(-1,))
        power_delta=state["power_residual"].reshape(n,p)*self.power_scale_db
        energy=foundation["foundation_energy"]*torch.pow(10.,power_delta/10)
        value,projection=reconstruct(base,q,desired_delay,desired_element,energy,self.config.projection_cycles,
           temporal=self.config.use_relative_delay,receive=self.config.use_receive_energy)
        value=restore_reference(value,reference)
        bound=self.state.coordinate_bound.clamp_min(1e-6)
        residual=(state["score_residual"]/4).square().mean()+(state["coordinate_residual"]/bound).square().mean()+(state["spectral_delta"]/self.config.spectral_log_power_bound).square().mean()+(state["receive_log_gain"]/self.config.receive_log_amplitude_bound).square().mean()
        return dict(channel=value,residual=residual,state=state,foundation=foundation,projection=projection,
           endpoint_covariance=endpoint_cov.reshape(n,p,32,32),geometry_quality=quality)
    @torch.inference_mode()
    def predict(self,query,data):
        if not np.array_equal(query.row_ids,data.row_ids):raise ValueError("Query/prepared row identities differ")
        if not np.allclose(query.position,data.position,rtol=0,atol=1e-7):raise ValueError("Query/prepared positions differ")
        if spec_signature(query.spec)!=spec_signature(self.spec) or spec_signature(data.spec)!=spec_signature(self.spec):raise ValueError("Query/prepared/model physical specs differ")
        self.eval();device=self.power_reference.device
        out=self(data.to(device));source_ids=sorted(set(x for ids in data.source_ids for x in ids))
        return check_prediction(dict(method="hall_structured_v2_single_TX",channel=out["channel"].cpu().numpy().astype(np.complex64),
           row_ids=np.array(query.row_ids),source_ids=np.array(source_ids),
           diagnostics=dict(structural_recipe=True,use_geometry=self.config.use_geometry,query_truth_alignment=False,
              original_multi_UE_head=False,reference_source="predicted_FIT_donor_carrier_only",geometry_quality=data.geometry.quality if data.geometry is not None else None,
              projection={k:float(v) for k,v in out["projection"].items()})),query)
