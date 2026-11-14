"""Pinned NeWRF complex core with explicit Round2 multiport adaptation.

CPU synthetic correctness implementation only. Real-data execution is gated:
the canonical-FIT receiver-side direction and query ray-prior provenance
have not been independently qualified. V16 frequency conventions are disclosed. No data/RT/training packages
are imported. Fixed BS -> moving UE is preserved; no reciprocity is assumed.
"""
from __future__ import annotations
import hashlib
import importlib.util
import math
import sys
from dataclasses import asdict, dataclass
from typing import Protocol
import torch
from torch import nn
from project_paths import PATHS

SOURCE_COMMIT = "78feb38471f766f17764a28e8518a05697944107"
SOURCE_SHA256 = {
 "models.py":"4557df4726a3c71033ceddd3e90432e0113af5a4f048df8cfb35b5ac1e5bb476",
 "encoders.py":"5a13b798f362de09f6b8da309ebb258e973c09288e2001f09c9adce470b36354",
 "synthesizers.py":"8482dbb4be0f5aa9c1f03b03953b5270c3233dae20a64ab38b4db9d83f478ae5",
 "samplers.py":"0b784323ebde0045de09f7575609b36495ba2e51a8109f9df7c8ad578b51f118",
}
SOURCE_ROOT = PATHS.root / "plus-pro/code/third_party/paper_baselines/newrf"

class NeWRFQualificationError(RuntimeError):
    """An unresolved information contract, not a model-performance failure."""

REAL_DATA_MISSING = (
 "Canonical-FIT-only mobile-UE-side training directions: source, row IDs, physical frame and split fingerprint are not certified",
 "FIT-only virtual-source/ray-search inference or independent query geometry prior is not certified; held-out LastXPts/DoA must never be used",
)

def real_data_qualification():
    return dict(qualified=False, missing=list(REAL_DATA_MISSING),
      training=False, scored=False, ray_tracing_executed=False,
      source_commit=SOURCE_COMMIT)

def require_real_data_qualification():
    raise NeWRFQualificationError("NeWRF real-data gate: " + "; ".join(REAL_DATA_MISSING))

def _source(name):
    path=SOURCE_ROOT/name
    if hashlib.sha256(path.read_bytes()).hexdigest()!=SOURCE_SHA256[name]:
        raise RuntimeError("Pinned NeWRF source differs: "+name)
    module_name="_round2_newrf_78feb_"+name.replace(".py","")
    if module_name not in sys.modules:
        spec=importlib.util.spec_from_file_location(module_name,path)
        module=importlib.util.module_from_spec(spec)
        sys.modules[module_name]=module
        try:spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name,None)
            raise
    return sys.modules[module_name]

def _frequency64(value,device):
    if hasattr(value,"dtype") and str(value.dtype) not in ("torch.float64","float64"):
        raise TypeError("Physical frequency must arrive in FP64 Hz")
    f=torch.as_tensor(value,dtype=torch.float64,device=device)
    if f.ndim!=1 or f.numel()==0 or not bool(torch.isfinite(f).all()) or bool((f<=0).any()):
        raise ValueError("Positive finite frequency axis [F] required")
    return f

class RayProvider(Protocol):
    def directions(self,ue_origins,site_ids,bs_tx_positions):
        """Return unit[B,2,128,4,R,3], outward from actual UE-port origins."""
    def metadata(self):
        """Report split/FIT provenance and absence of query CSI/path labels."""

class SyntheticRayProvider:
    """Fixed artificial directions. Not an accepted real-data direction source."""
    def __init__(self,directions):
        d=torch.as_tensor(directions,dtype=torch.float64)
        if d.ndim!=2 or d.shape[1]!=3 or not len(d) or not bool(torch.isfinite(d).all()):
            raise ValueError("Synthetic directions must be finite[R,3]")
        if not torch.allclose(torch.linalg.vector_norm(d,dim=-1),torch.ones(len(d),dtype=d.dtype),atol=1e-8,rtol=1e-8):
            raise ValueError("Synthetic directions must be unit vectors")
        self.value=d.detach().clone()
    def directions(self,ue_origins,site_ids,bs_tx_positions):
        return self.value.to(ue_origins.device)[None,None,None,None].expand(len(ue_origins),2,128,4,-1,-1)
    def metadata(self):
        return dict(kind="synthetic_only",query_CSI=False,query_path_labels=False,
                    canonical_fit_verified=False,real_data_qualified=False)

@dataclass(frozen=True)
class NeWRFFullHConfig:
    n_layers:int=6
    hidden_dim:int=128
    skip:tuple[int,...]=(4,)
    position_levels:int=6
    direction_levels:int=4
    n_samples:int=16
    n_importance:int=8
    far_m:float=80.0
    coordinate_scale_m:float=100.0
    frequency_center_hz:float=6690000000.0
    frequency_scale_hz:float=5760000.0
    frequency_chunk:int=16
    def __post_init__(self):
        object.__setattr__(self,"skip",tuple(self.skip))
        for key in ("n_layers","hidden_dim","position_levels","direction_levels","n_samples","frequency_chunk"):
            x=getattr(self,key)
            if isinstance(x,bool) or not isinstance(x,int) or x<1:raise ValueError(key+" must be a positive integer")
        if self.hidden_dim<4 or self.n_samples<3 or self.n_importance<0 or not isinstance(self.n_importance,int):
            raise ValueError("Require width>=4, samples>=3 and nonnegative importance count")
        if any(x<0 or x>=self.n_layers-1 for x in self.skip):raise ValueError("Invalid NeWRF skip index")
        for key in ("far_m","coordinate_scale_m","frequency_center_hz","frequency_scale_hz"):
            if not math.isfinite(getattr(self,key)) or getattr(self,key)<=0:raise ValueError("Positive finite "+key+" required")

class _ConditionalField(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        encoder=_source("encoders.py").PositionalEncoder
        self.encoder=encoder(7,cfg.position_levels,log_space=True)
        self.view_encoder=encoder(3,cfg.direction_levels,log_space=True)
        self.network=_source("models.py").NeWRF(d_input=self.encoder.d_output,
          n_layers=cfg.n_layers,d_filter=cfg.hidden_dim,skip=cfg.skip,
          d_viewdirs=self.view_encoder.d_output)
        self.cfg=cfg
    def forward(self,points,view,tx,frequencies):
        # Physical sampling/distances stay FP64; only bounded network inputs are FP32.
        x=torch.cat((points/self.cfg.coordinate_scale_m,
                     tx/self.cfg.coordinate_scale_m,
                     ((frequencies-self.cfg.frequency_center_hz)/self.cfg.frequency_scale_hz)),dim=-1).float()
        return self.network(self.encoder(x),self.view_encoder(view.float()))

class NeWRFFullH(nn.Module):
    """forward(ue_position[B,3], site_id[B], frequencies_hz[F])->complex[B,256,4,F].

    `spec.bs_port_positions_m` is [sites,2,16,8,3], and
    `spec.ue_world_positions(position,site_id)` returns [B,4,3] in UE pol/v order.
    Eight independent scalar fields (BS2pol x UE4ports) condition on the actual
    TX-element coordinates and physical frequency; no channel is broadcast over
    ports/polarizations. All BS spatial ports share each conditional field.

    The default real-data route raises qualification gaps. Explicit synthetic_only
    is for artificial positions/directions and never grants dataset eligibility.
    """
    def __init__(self,spec,train_only_ray_provider:RayProvider,config=None,*,synthetic_only=False):
        super().__init__()
        self.config=config if isinstance(config,NeWRFFullHConfig) else NeWRFFullHConfig(**(config or {}))
        self.spec=spec;self.ray_provider=train_only_ray_provider
        self.synthetic_only=bool(synthetic_only)
        if not callable(getattr(spec,"ue_world_positions",None)):
            raise TypeError("A dynamic UE-world-position function is required; fixed offsets are insufficient")
        bs=torch.as_tensor(spec.bs_port_positions_m,dtype=torch.float64)
        if bs.ndim!=5 or tuple(bs.shape[1:])!=(2,16,8,3) or not bool(torch.isfinite(bs).all()):
            raise ValueError("spec.bs_port_positions_m must be finite[sites,2,16,8,3]")
        self.register_buffer("bs_ports",bs.reshape(len(bs),2,128,3).detach().clone())
        f=_frequency64(spec.frequencies_hz,"cpu")
        if f.shape!=(192,):raise ValueError("Round2 requires a declared192-tone frequency grid")
        self.register_buffer("physical_frequencies_hz",f.detach().clone())
        if not callable(getattr(train_only_ray_provider,"directions",None)):
            raise TypeError("An explicit direction provider is required")
        info=train_only_ray_provider.metadata()
        if info.get("query_CSI",True) or info.get("query_path_labels",True):
            raise ValueError("Query CSI/path labels are not legal direction inputs")
        self.coarse=nn.ModuleList([_ConditionalField(self.config) for _ in range(8)])
        self.fine=nn.ModuleList([_ConditionalField(self.config) for _ in range(8)]) if self.config.n_importance else None
        self.synthesizer=_source("synthesizers.py").Synthesizer()

    def metadata(self):
        return dict(method="NeWRF_Round2_source_conditioned_multiport_full_H_adapter",
          source_commit=SOURCE_COMMIT,source_sha256=dict(SOURCE_SHA256),config=asdict(self.config),
          direction="fixed BS TX -> moving UE RX; backtracing starts at actual UE ports",
          input_information=["UE position","site ID","physical BS/UE array coordinates","frequencies","explicit permitted ray prior"],
          output="complex[B,256,4,F]; BS polarization,h,v; UE polarization,v",
          independent_fields="8 independent BS-pol x UE-port scalar fields; shared across128 real BS element coordinates",
          preserved=["unmodified author NeWRF/PositionalEncoder class bodies","author complex volume synthesis including first-exclusive-entry zero","author deterministic stratified and optional hierarchical sampling"],
          adapted=["fixed-source MLP inputs extended to actual TX XYZ and shared frequency","8 independent polarization/UE-port fields","actual TX and dynamic UE element geometry","FP64 Hz->GHz propagation input","canonical FIT provider contract replaces80-percent loader and filename cache"],
          mode="synthetic_correctness_only" if self.synthetic_only else "real_data_gate",
          ray_provider=self.ray_provider.metadata(),qualification=real_data_qualification(),
          limits=["not a native original experiment reproduction","not an accepted training/DoA/frequency or performance qualification","no polarization coupling physics beyond separate learned fields","V16 physical frequency/geometry convention; publisher values not independently certified","no checkpoint,optimizer,data loader,RT or score"])

    def _scalar_links(self,field_index,origins,dirs,tx,f):
        cfg=self.config;sampler=_source("samplers.py")
        b,m,r,_=dirs.shape;links=b*m
        origins=origins[:,None,:].expand(b,m,3).reshape(links,3)
        tx=tx.reshape(links,3);dirs=dirs.reshape(links,r,3)
        points,z=sampler.sample_stratified(origins[:,None,:].expand(links,r,3),dirs,
          near=0.,far=cfg.far_m,n_samples=cfg.n_samples,perturb=False,inverse_depth=False)
        pieces=[];coarse_pieces=[]
        for start in range(0,len(f),cfg.frequency_chunk):
            freq=f[start:start+cfg.frequency_chunk];k=len(freq)
            pt=points[:,None].expand(links,k,r,cfg.n_samples,3).reshape(links*k*r,cfg.n_samples,3)
            zz=z[:,None].expand(links,k,r,cfg.n_samples).reshape(links*k*r,cfg.n_samples)
            view=dirs[:,None].expand(links,k,r,3).reshape(links*k*r,3)
            origin=origins[:,None,None].expand(links,k,r,3).reshape(links*k*r,3)
            txray=tx[:,None,None].expand(links,k,r,3).reshape(links*k*r,3)
            fr=freq[None,:,None].expand(links,k,r).reshape(links*k*r,1)
            raw=self.coarse[field_index](pt,view[:,None].expand_as(pt),
              txray[:,None].expand_as(pt),fr[:,None].expand(links*k*r,cfg.n_samples,1))
            groups=[r]*(links*k)
            cfr,_,_,weights=self.synthesizer.synthesize(raw,zz,fr/1e9,groups)
            coarse_pieces.append(cfr.reshape(b,m,k))
            if self.fine is not None:
                pp,zz,_,_=sampler.sample_hierarchical(origin,view,zz,weights,cfg.n_importance,perturb=False)
                raw=self.fine[field_index](pp,view[:,None].expand_as(pp),txray[:,None].expand_as(pp),
                      fr[:,None].expand(len(pp),pp.shape[-2],1))
                cfr,_,_,_=self.synthesizer.synthesize(raw,zz,fr/1e9,groups)
            pieces.append(cfr.reshape(b,m,k))
        return (torch.cat(pieces,dim=-1).to(torch.complex64),
                torch.cat(coarse_pieces,dim=-1).to(torch.complex64))

    def forward(self,ue_position,site_id,frequencies_hz=None,*,return_coarse=False):
        if not self.synthetic_only:require_real_data_qualification()
        device=self.bs_ports.device
        pos=torch.as_tensor(ue_position,dtype=torch.float64,device=device)
        sites=torch.as_tensor(site_id,device=device)
        if pos.ndim!=2 or pos.shape[1]!=3 or not len(pos) or not bool(torch.isfinite(pos).all()):raise ValueError("Finite nonempty UE[B,3] required")
        if sites.shape!=(len(pos),) or sites.dtype not in (torch.int32,torch.int64) or bool(((sites<0)|(sites>=len(self.bs_ports))).any()):raise ValueError("Valid integer site IDs[B] required")
        f=_frequency64(self.physical_frequencies_hz if frequencies_hz is None else frequencies_hz,device)
        origins=torch.as_tensor(self.spec.ue_world_positions(pos,sites),dtype=torch.float64,device=device)
        if origins.shape!=(len(pos),4,3) or not bool(torch.isfinite(origins).all()):raise ValueError("Dynamic UE positions must be finite[B,4,3]")
        tx=self.bs_ports[sites]
        d=torch.as_tensor(self.ray_provider.directions(origins,sites,tx),dtype=torch.float64,device=device)
        if d.ndim!=6 or d.shape[:4]!=(len(pos),2,128,4) or d.shape[-1]!=3 or d.shape[-2]<1 or not bool(torch.isfinite(d).all()):raise ValueError("Direction prior must be finite[B,2,128,4,R,3]")
        if not torch.allclose(torch.linalg.vector_norm(d,dim=-1),torch.ones(d.shape[:-1],device=device,dtype=torch.float64),atol=1e-6,rtol=1e-6):raise ValueError("Direction prior must contain unit vectors")
        polarizations=[];coarse_polarizations=[]
        for p in range(2):
            ue=[];coarse_ue=[]
            for u in range(4):
                value,coarse_value=self._scalar_links(p*4+u,origins[:,u],d[:,p,:,u],tx[:,p],f)
                ue.append(value);coarse_ue.append(coarse_value)
            polarizations.append(torch.stack(ue,dim=2))
            coarse_polarizations.append(torch.stack(coarse_ue,dim=2))
        h=torch.stack(polarizations,dim=1).reshape(len(pos),256,4,len(f))
        coarse_h=torch.stack(coarse_polarizations,dim=1).reshape_as(h)
        if not bool(torch.isfinite(h).all()) or not bool(torch.isfinite(coarse_h).all()):raise FloatingPointError("Nonfinite NeWRF complex channel")
        # A future fit must supervise both outputs as in the author pipeline.
        return (h,coarse_h) if return_coarse else h


def synthetic_self_check():
    """CPU artificial-input checks; no optimizer, dataset, cache or score access."""
    import numpy as np
    import time
    started=time.monotonic()
    torch.set_num_threads(2);torch.manual_seed(2031)
    class SyntheticSpec:
        frequencies_hz=6.69e9+(np.arange(192,dtype=np.float64)-96)*30000.
        bs_port_positions_m=np.array([[[[[s*2+h*.03,v*.07,2.] for v in range(8)] for h in range(16)] for p in range(2)] for s in range(2)],np.float64)
        def ue_world_positions(self,pos,site):
            off=torch.tensor([[0.,0.,-.01],[0.,0.,.01],[0.,0.,-.01],[0.,0.,.01]],dtype=torch.float64,device=pos.device)
            return pos[:,None]+off[None]
    spec=SyntheticSpec();provider=SyntheticRayProvider([[1.,0.,0.]])
    cfg=dict(n_layers=2,hidden_dim=8,skip=(),position_levels=1,direction_levels=1,n_samples=4,n_importance=2,far_m=2.,frequency_chunk=48)
    model=NeWRFFullH(spec,provider,cfg,synthetic_only=True).cpu().eval()
    x=torch.tensor([[.3,.5,1.]],dtype=torch.float64);site=torch.tensor([0])
    f=torch.tensor(spec.frequencies_hz,dtype=torch.float64)
    with torch.no_grad():
        h=model(x,site)
        one=model(x,site,f[[0,96,191]])
    assert h.shape==(1,256,4,192) and h.dtype==torch.complex64 and bool(torch.isfinite(h).all())
    torch.testing.assert_close(h[:,:,:,[0,96,191]],one,rtol=1e-5,atol=1e-7)
    assert not torch.allclose(h,h[...,96,None].expand_as(h))
    assert not torch.allclose(h[:,0],h[:,128])
    assert not torch.allclose(h[:,:,0],h[:,:,2])
    y,coarse_y=model(x,site,f[[0,96,191]],return_coarse=True)
    gradients=torch.autograd.grad(y.abs().square().mean()+coarse_y.abs().square().mean(),tuple(model.parameters()),allow_unused=True)
    assert all(a is None or bool(torch.isfinite(a).all()) for a in gradients)
    assert any(a is not None and bool((a.abs()>0).any()) for a in gradients)
    # Source algebra independently reproduced for two rays, grouped coherently.
    z=torch.tensor([[1e-10,.5,1.,1.5],[1e-10,.5,1.,1.5]],dtype=torch.float64)
    raw=torch.zeros((2,4,3),dtype=torch.float64)
    raw[...,0]=.3;raw[...,1]=-.2;raw[...,-1]=1.1
    ghz=torch.tensor([[6.69],[6.69]],dtype=torch.float64)
    actual,_,_,_=model.synthesizer.synthesize(raw,z,ghz,[2])
    expected=torch.zeros((),dtype=torch.complex128)
    for r in range(2):
        product=torch.ones((),dtype=torch.complex128)
        for j in range(4):
            distance=(z[r,j+1]-z[r,j]) if j<3 else torch.tensor(1e-10,dtype=torch.float64)
            alpha=1-torch.exp(-raw[r,j,2].relu()*distance)
            exclusive=torch.zeros_like(product) if j==0 else product
            emitted=torch.complex(raw[r,j,0].tanh(),raw[r,j,1].tanh())
            expected=expected+emitted*(299792458./(z[r,j]*ghz[r,0]*1e9*4*torch.pi))*alpha*exclusive
            product=product*((1-alpha)*torch.exp(-1j*2*torch.pi*ghz[r,0]*1e9/299792458.*distance)+1e-10)
    torch.testing.assert_close(actual[0],expected,rtol=1e-11,atol=1e-13)
    model.synthetic_only=False
    try:model(x,site)
    except NeWRFQualificationError:pass
    else:raise AssertionError("Real gate did not close")
    assert not torch.cuda.is_initialized()
    return dict(status="synthetic_core_correct_real_dataset_unqualified",output_shape=list(h.shape),
      finite_forward_backward=True,coarse_and_fine_exposed=True,tone_subset_parity=True,independent_BS_and_UE_polarizations=True,
      author_complex_integral_scalar_reference=True,dataset_gate_closed=True,
      source_sha256=dict(SOURCE_SHA256),device="CPU",cuda_initialized=False,
      optimizer_updates=0,trained=False,scored=False,channel_arrays_read=False,path_arrays_read=False,
      wall_seconds=time.monotonic()-started)

if __name__=="__main__":
    import json
    print(json.dumps(synthetic_self_check(),indent=2))
