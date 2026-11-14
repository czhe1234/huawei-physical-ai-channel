"""NeWRF wideband adaptation v2: frequency-conditioned signal, physical density.

The pinned author's NeWRF MLP class and native complex volume algebra remain.
Position+TX XYZ enter the position trunk. Direction PE4 and independent frequency
PE10 enter only the complex signal branch. This differs from the older 7D-trunk
adapter, which remains unchanged. Shared and naive execution represent one model.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from round2.paper_baseline_adapters_v1.newrf import _source,NeWRFFullHConfig
from round2.simulation_recovery_v1.qualified_newrf import QualifiedNeWRFFullH
from .newrf_runner import UniformRaySynthesizer
from .newrf_precision_probe import streaming_encode

ADAPTER_VERSION="NeWRF-wideband-late-frequency-v2"

@dataclass(frozen=True)
class LateFrequencyConfig(NeWRFFullHConfig):
    frequency_levels:int=10

class LateFrequencyField(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg
        encoder=_source("encoders.py").PositionalEncoder
        self.encoder=encoder(6,cfg.position_levels,log_space=True)
        self.view_encoder=encoder(3,cfg.direction_levels,log_space=True)
        self.frequency_encoder=encoder(1,cfg.frequency_levels,log_space=True)
        self.network=_source("models.py").NeWRF(d_input=self.encoder.d_output,
             n_layers=cfg.n_layers,d_filter=cfg.hidden_dim,skip=cfg.skip,
             d_viewdirs=self.view_encoder.d_output+self.frequency_encoder.d_output)

    def encoded_position(self,points,tx):
        x=torch.cat((points/self.cfg.coordinate_scale_m,
                     tx/self.cfg.coordinate_scale_m),-1).float()
        return streaming_encode(x,self.encoder)

    def encoded_frequency(self,frequencies):
        x=((frequencies-self.cfg.frequency_center_hz)/self.cfg.frequency_scale_hz).float()
        return streaming_encode(x,self.frequency_encoder)

    def forward(self,points,view,tx,frequencies):
        """Naive author forward: repeated position trunk for each frequency."""
        ev=streaming_encode(view.float(),self.view_encoder)
        ef=self.encoded_frequency(frequencies)
        return self.network(self.encoded_position(points,tx),torch.cat((ev,ef),-1))

    def prepare(self,points,view,tx):
        """Compute the frequency-independent trunk, density and branch prefix."""
        network=self.network
        x=self.encoded_position(points,tx);x_input=x
        for i,layer in enumerate(network.layers):
            x=network.act(layer(x))
            if i in network.skip:x=torch.cat((x,x_input),dim=-1)
        density=network.alpha_out(x)
        feature=network.realimag_filters(x)
        ev=streaming_encode(view.float(),self.view_encoder)
        prefix=torch.cat((feature,ev),dim=-1)
        d=prefix.shape[-1]
        base=F.linear(prefix,network.branch.weight[:,:d],network.branch.bias)
        return base,density

    def frequency_output(self,base,density,frequencies,links,rays):
        """Exact linear split before the author's ReLU and complex output."""
        # base [L*R,S,64]; output ordering [L,K,R,S,3].
        k=len(frequencies);s=base.shape[-2]
        ef=self.encoded_frequency(frequencies[:,None])
        d=self.cfg.hidden_dim+self.view_encoder.d_output
        freq_bias=F.linear(ef,self.network.branch.weight[:,d:],None)
        hidden=self.network.act(base.reshape(links,1,rays,s,-1)+freq_bias[None,:,None,None,:])
        signal=self.network.output(hidden)
        rho=density.reshape(links,1,rays,s,1).expand(links,k,rays,s,1)
        return torch.cat((signal,rho),-1).reshape(links*k*rays,s,3)

def stratified64(origins,directions,far,n_samples):
    """Pinned stratified algorithm, explicitly deterministic FP64 sampling."""
    t=torch.linspace(0.,1.,n_samples,device=origins.device,dtype=torch.float64)
    z=far*t
    z[0]=1e-10
    z=z.expand(*origins.shape[:-1],n_samples)
    return origins[...,None,:]+directions[...,None,:]*z[...,None],z

def density_weights(density,z):
    dists=torch.cat((z[...,1:]-z[...,:-1],1e-10*torch.ones_like(z[...,:1])),-1)
    alpha=1-torch.exp(-F.relu(density[...,0])*dists)
    cumulative=torch.cumprod(1-alpha+1e-10,dim=-1)
    exclusive=torch.cat((torch.zeros_like(cumulative[...,:1]),cumulative[...,:-1]),-1)
    return alpha*exclusive

def hierarchical64(origins,directions,z,weights,n_new):
    """Pinned inverse-CDF algorithm with FP64 bins, CDF and deterministic u."""
    bins=.5*(z[...,1:]+z[...,:-1]);w=weights[...,1:-1]
    pdf=(w+1e-5)/torch.sum(w+1e-5,-1,keepdim=True)
    cdf=torch.cat((torch.zeros_like(pdf[...,:1]),torch.cumsum(pdf,dim=-1)),-1)
    u=torch.linspace(0.,1.,n_new,device=cdf.device,dtype=torch.float64)
    u=u.expand(*cdf.shape[:-1],n_new).contiguous()
    inds=torch.searchsorted(cdf,u,right=True)
    below=torch.clamp(inds-1,min=0);above=torch.clamp(inds,max=cdf.shape[-1]-1)
    ig=torch.stack((below,above),-1)
    shape=[*ig.shape[:-1],cdf.shape[-1]]
    cg=torch.gather(cdf.unsqueeze(-2).expand(shape),-1,ig)
    bg=torch.gather(bins.unsqueeze(-2).expand(shape),-1,ig)
    den=cg[...,1]-cg[...,0]
    den=torch.where(den<1e-5,torch.ones_like(den),den)
    t=(u-cg[...,0])/den
    new=(bg[...,0]+t*(bg[...,1]-bg[...,0])).detach()
    zz=torch.sort(torch.cat((z,new),-1),-1).values
    return origins[...,None,:]+directions[...,None,:]*zz[...,None],zz

class LateFrequencyNeWRF(QualifiedNeWRFFullH):
    def __init__(self,bank,config,*,execution="shared"):
        super().__init__(bank,config,max_query_rays=64,max_training_rays=64)
        self.coarse=nn.ModuleList([LateFrequencyField(config) for _ in range(8)])
        self.fine=nn.ModuleList([LateFrequencyField(config) for _ in range(8)]) if config.n_importance else None
        self.synthesizer=UniformRaySynthesizer()
        self.execution=execution

    def _scalar_links(self,field_index,origins,dirs,tx,f):
        cfg=self.config;b,m,r,_=dirs.shape;links=b*m
        origins=origins[:,None,:].expand(b,m,3).reshape(links,3)
        tx=tx.reshape(links,3);dirs=dirs.reshape(links,r,3)
        ray_origins=origins[:,None,:].expand(links,r,3).reshape(links*r,3)
        ray_dirs=dirs.reshape(links*r,3)
        ray_tx=tx[:,None,:].expand(links,r,3).reshape(links*r,3)
        points,z=stratified64(ray_origins,ray_dirs,cfg.far_m,cfg.n_samples)
        view=ray_dirs[:,None,:].expand_as(points);txpoints=ray_tx[:,None,:].expand_as(points)
        coarse=self.coarse[field_index]
        # The same density/CDF is used for every frequency, as required by v2.
        if self.execution=="shared":
            cbase,cdensity=coarse.prepare(points,view,txpoints)
        else:
            first=f[:1,None,None].expand(1,len(points),points.shape[-2]).permute(1,2,0)
            craw=coarse(points,view,txpoints,first)
            cdensity=craw[...,-1:];cbase=None
        weights=density_weights(cdensity,z)
        fbase=fdensity=None
        if self.fine is not None:
            finepoints,finez=hierarchical64(ray_origins,ray_dirs,z,weights,cfg.n_importance)
            fineview=ray_dirs[:,None,:].expand_as(finepoints)
            finetx=ray_tx[:,None,:].expand_as(finepoints)
            fine=self.fine[field_index]
            if self.execution=="shared":
                fbase,fdensity=fine.prepare(finepoints,fineview,finetx)
        pieces=[];coarse_pieces=[]
        for start in range(0,len(f),cfg.frequency_chunk):
            freq=f[start:start+cfg.frequency_chunk];k=len(freq)
            fr=freq[None,:,None].expand(links,k,r).reshape(links*k*r,1)
            def render(field,pp,zz,base=None,rho=None):
                samples=pp.shape[-2]
                if self.execution=="shared":
                    raw=field.frequency_output(base,rho,freq,links,r)
                else:
                    # Deliberately repeat the author trunk per frequency.
                    raw=torch.stack([field(pp,ray_dirs[:,None].expand_as(pp),
                        ray_tx[:,None].expand_as(pp),q.expand(len(pp),samples,1))
                        for q in freq],dim=0)
                    raw=raw.reshape(k,links,r,samples,3).permute(1,0,2,3,4).reshape(links*k*r,samples,3)
                zf=zz.reshape(links,1,r,samples).expand(links,k,r,samples).reshape(links*k*r,samples)
                h,_,_,_=self.synthesizer.synthesize(raw,zf,fr/1e9,[r]*(links*k))
                return h.reshape(b,m,k)
            hc=render(coarse,points,z,cbase,cdensity)
            coarse_pieces.append(hc)
            h=render(fine,finepoints,finez,fbase,fdensity) if self.fine is not None else hc
            pieces.append(h)
        return torch.cat(pieces,-1).to(torch.complex64),torch.cat(coarse_pieces,-1).to(torch.complex64)

    def metadata(self):
        value=super().metadata()
        value.update(adapter_version=ADAPTER_VERSION,density_frequency_independent=True,
                     frequency_PE_levels=self.config.frequency_levels,
                     position_input="sample XYZ+actual TX XYZ",
                     signal_branch_input="position feature+direction PE4+frequency PE10",
                     execution=self.execution,
                     sampling="native stratified+hierarchical algorithms, explicit FP64",
                     old_7D_trunk_adapter_changed=False)
        return value

def build_late_model(data,cache,coarse=128,importance=128,*,frequency_chunk=8,execution="shared"):
    cfg=LateFrequencyConfig(n_layers=6,hidden_dim=128,skip=(4,),position_levels=10,
                  direction_levels=4,frequency_levels=10,n_samples=int(coarse),
                  n_importance=int(importance),far_m=cache.far_m,coordinate_scale_m=1.,
                  frequency_chunk=int(frequency_chunk))
    return LateFrequencyNeWRF(cache.bank,cfg,execution=execution)

