"""Exact-computation acceleration and source-parameterized fresh field models.

No old source, model, data, or score is modified. NeRF2 shares only its
TX-independent medium computation. nGRF restores separately specified author
attribute/decoder widths; all official class bodies and covariance remain.
"""
from __future__ import annotations
import dataclasses
import math
import sys
import types
import torch
from torch import nn
from torch.nn import functional as F
from round2.paper_baseline_adapters_v1.field_models import CompetitionNeRF2, CompetitionNGRF, _load_previous_adapter


def _medium(field, points_frequency, view):
    embedded = field.embed_pts_fn(points_frequency).contiguous()
    view_encoded = field.embed_view_fn(view).contiguous()
    shape = embedded.shape[:-1]
    pts = embedded.reshape(-1, embedded.shape[-1])
    x = pts
    for i, layer in enumerate(field.attenuation_linears):
        x = F.relu(layer(x))
        if i in field.skips:
            x = torch.cat([pts, x], -1)
    attenuation = field.attenuation_output(x)
    feature = field.feature_layer(x)
    return attenuation, torch.cat([feature, view_encoded.reshape(len(feature), -1)], -1), shape



def _shared_transport(attenuation,shape,segments,frequency,sign,near):
    alpha=F.leaky_relu(attenuation[:,0]).abs().reshape(shape)
    residual=(torch.sigmoid(attenuation[:,1])*(2.*math.pi)-math.pi).reshape(shape)
    transmittance=torch.exp(torch.cumsum(-alpha*segments.to(alpha.dtype),dim=-1))
    propagation=sign*(2.*math.pi/299792458.)*(frequency[:,None]*segments[None])
    phase=torch.cumsum(residual.double()+propagation,dim=-1)
    phase=phase+sign*(2.*math.pi/299792458.)*(frequency*near)[None,:,None]
    phase=torch.remainder(phase+math.pi,2.*math.pi)-math.pi
    return transmittance,torch.complex(phase.cos(),phase.sin())


def _factorized_integral(signal,shape,transmittance,carrier):
    raw=signal.reshape(signal.shape[0],*shape,2)
    amplitude=F.leaky_relu(raw[...,0]).abs()
    phase=torch.sigmoid(raw[...,1])*(2.*math.pi)-math.pi
    # Only learned residual radians use FP32 trig. Absolute GHz propagation
    # remains FP64 and is shared; sums stay complex128 before finalcomplex64.
    learned=torch.complex(phase.cos(),phase.sin()).to(torch.complex128)
    amplitude=(amplitude*transmittance[None]).double()
    return (amplitude*carrier[None]*learned).sum(-1).to(torch.complex64)

def shared_medium_forward(self, tx_position, rx_position, frequencies_hz):
    source = sys.modules[self.__class__.__module__]
    device = next(self.field.parameters()).device
    dtype = next(self.field.parameters()).dtype
    if dtype != torch.float32:
        raise TypeError("Fresh NeRF2 training keeps field FP32")
    tx = torch.as_tensor(tx_position, dtype=torch.float64, device=device)
    rx = torch.as_tensor(rx_position, dtype=torch.float64, device=device)
    frequency = source._frequencies64(frequencies_hz, device)
    if tx.ndim != 2 or rx.ndim != 3 or tx.shape[-1] != 3 or rx.shape[-1] != 3:
        raise ValueError("Physical TX/RX shape mismatch")
    n_tx, panels, elements = len(tx), rx.shape[0], rx.shape[1]
    ports = panels * elements
    cfg = self.config
    center = torch.tensor(cfg.world_center_m, dtype=torch.float64, device=device)
    directions = self.ray_directions.to(device=device, dtype=torch.float64)
    directions = directions / torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
    step = (cfg.far_m-cfg.near_m)/cfg.n_samples
    segments = torch.full((cfg.n_samples,), step, dtype=torch.float64, device=device)
    radii = cfg.near_m + step*torch.arange(1,cfg.n_samples+1,dtype=torch.float64,device=device)
    rx_flat = rx.reshape(ports,3)
    tx_encoded = self.field.embed_tx_fn(((tx-center)/cfg.world_scale_m).to(dtype)).contiguous()
    first = self.field.signal_linears[0]
    tx_dim = tx_encoded.shape[-1]
    common_dim = first.in_features-tx_dim
    tx_projection = F.linear(tx_encoded,first.weight[:,common_dim:],None)
    tx_ranges = [(a,min(a+self.fast_tx_chunk,n_tx)) for a in range(0,n_tx,self.fast_tx_chunk)]
    frequency_blocks=[]
    for f0 in range(0,len(frequency),cfg.frequency_chunk):
        freq=frequency[f0:f0+cfg.frequency_chunk]
        f_encoded=((freq-cfg.reference_frequency_hz)/cfg.frequency_scale_hz).to(dtype)
        result=[torch.zeros((b-a,ports,len(freq)),dtype=torch.complex64,device=device) for a,b in tx_ranges]
        for r0 in range(0,ports*cfg.n_rays,self.fast_medium_ray_chunk):
            ray_id=torch.arange(r0,min(r0+self.fast_medium_ray_chunk,ports*cfg.n_rays),device=device)
            port=torch.div(ray_id,cfg.n_rays,rounding_mode="floor")
            direction=directions[ray_id%cfg.n_rays]
            points=rx_flat[port,None,:]+direction[:,None,:]*radii[None,:,None]
            n=len(ray_id)
            xyz=((points-center)/cfg.world_scale_m).to(dtype)[:,None].expand(n,len(freq),cfg.n_samples,3)
            ff=f_encoded[None,:,None,None].expand(n,len(freq),cfg.n_samples,1)
            view=direction.to(dtype)[:,None,None].expand(n,len(freq),cfg.n_samples,3)
            attenuation,common,shape=_medium(self.field,torch.cat([xyz,ff],-1),view)
            if common.shape[-1] != common_dim:
                raise ValueError("Author signal topology changed")
            projection=F.linear(common,first.weight[:,:common_dim],first.bias)
            if self.fast_factor_transport:
                transmittance,carrier=_shared_transport(attenuation,shape,segments,freq,cfg.propagation_sign,cfg.near_m)
            for k,(a,b) in enumerate(tx_ranges):
                x=F.relu(projection[None]+tx_projection[a:b,None])
                for layer in self.field.signal_linears[1:]:
                    x=F.relu(layer(x))
                signal=self.field.signal_output(x)
                if self.fast_factor_transport:
                    rays=_factorized_integral(signal,shape,transmittance,carrier)
                else:
                    raw=torch.cat([attenuation[None].expand(b-a,-1,-1),signal],-1)
                    raw=raw.reshape(b-a,*shape,4)
                    rays=source.coherent_nerf2_integral(raw,segments,freq,cfg.propagation_sign,cfg.near_m)
                result[k]=result[k].index_add(1,port,rays)
        frequency_blocks.append(torch.cat(result,0))
    return torch.cat(frequency_blocks,-1).reshape(n_tx,panels,elements,len(frequency))


def enable_shared_medium(model, tx_chunk=16, medium_ray_chunk=128,factor_transport=False):
    if type(model) is not CompetitionNeRF2:
        raise TypeError("Reviewed NeRF2 competition adapter required")
    for core in model.fields.values():
        core.fast_tx_chunk=int(tx_chunk)
        core.fast_medium_ray_chunk=int(medium_ray_chunk)
        core.fast_factor_transport=bool(factor_transport)
        core.forward=types.MethodType(shared_medium_forward,core)
    return model


NGRF_SOURCE_CONFIG = dict(
    num_gaussians=1200,latent_dim=32,initial_scale=0.1478421594100344,
    attribute_hidden_dim=64,attribute_num_layers=8,attribute_dropout_p=0.14651508945335504,
    decoder_hidden_dim=128,decoder_num_layers=3,decoder_dropout_p=0.22766411069474934,
    pos_encoding_freqs=16,frequency_encoding_levels=4,frequency_chunk_size=8)


def make_ngrf(geometry, config=None):
    config={**NGRF_SOURCE_CONFIG,**(config or {})}
    previous=_load_previous_adapter("ngrf")
    networks,_,_=previous._official_components()
    core_config={k:v for k,v in config.items() if k not in ("attribute_hidden_dim","attribute_dropout_p","decoder_hidden_dim")}
    core_config["hidden_dim"]=config["attribute_hidden_dim"]
    model=CompetitionNGRF(geometry,field_config=core_config)
    for core in model.fields.values():
        core.attribute_network=networks.AttributeNetwork(
            latent_dim=config["latent_dim"],mlp_hidden_dim=config["attribute_hidden_dim"],
            mlp_num_layers=config["attribute_num_layers"],pos_encoding_freqs=config["pos_encoding_freqs"],
            dropout_p=config["attribute_dropout_p"])
        fdim=1+2*config["frequency_encoding_levels"]
        core.contribution_decoders=nn.ModuleList([
            networks.ContributionDecoderNetwork(
                latent_dim=config["latent_dim"]+fdim,output_dim=2*core.num_ports,
                hidden_dim=config["decoder_hidden_dim"],num_layers=config["decoder_num_layers"],
                dropout_p=config["decoder_dropout_p"]) for _ in range(core.num_panels)])
        core.configuration.update(config)
    model.native_training_configuration=config
    return model


def ngrf_optimizer(model):
    if type(model) is not CompetitionNGRF:
        raise TypeError("Reviewed nGRF competition adapter required")
    learning_rates={
        "xyz":0.0002696363015043561,"rotation":0.0006292551064997657,
        "scaling":0.0011355962014815962,"attribute_net":0.00031728462884002734,
        "decoder":0.0005534331508483368}
    groups=[]
    for name,core in model.fields.items():
        for label,parameters in [
            ("xyz",[core.gaussian_means]),("rotation",[core.gaussian_quaternions]),
            ("scaling",[core.gaussian_log_scales]),("attribute_net",list(core.attribute_network.parameters())),
            ("decoder",list(core.contribution_decoders.parameters()))]:
            groups.append(dict(params=parameters,lr=learning_rates[label],initial_lr=learning_rates[label],name=name+"/"+label))
    return torch.optim.AdamW(groups,eps=1.3209624572295576e-09,weight_decay=3.146520107386117e-08)


def equivalence_check(factor_transport=False):
    from round2.paper_baseline_adapters_v1.field_models import synthetic_geometry
    torch.set_num_threads(2)
    torch.manual_seed(1947)
    geometry=synthetic_geometry()
    cfg=dict(depth=3,width=16,skips=(),position_levels=2,direction_levels=2,transmitter_levels=2,
             n_rays=3,n_samples=4,near_m=.1,far_m=2.,ray_chunk=41,frequency_chunk=2)
    reference=CompetitionNeRF2(geometry,field_config=cfg)
    optimized=CompetitionNeRF2(geometry,field_config=cfg)
    optimized.load_state_dict(reference.state_dict())
    enable_shared_medium(optimized,tx_chunk=17,medium_ray_chunk=5,factor_transport=factor_transport)
    pos=torch.tensor([[1.,.5,1.2],[2.,-.3,1.]],dtype=torch.float64)
    site=torch.tensor([0,1])
    frequency=torch.tensor([6690000000.,6690030000.,6692850000.],dtype=torch.float64)
    left=reference(pos,site,frequency)
    right=optimized(pos,site,frequency)
    torch.testing.assert_close(right,left,rtol=8e-5,atol=8e-6)
    loss_l=(left.abs().square().mean()+.1*left.real.mean())
    loss_r=(right.abs().square().mean()+.1*right.real.mean())
    loss_l.backward();loss_r.backward()
    gmax=0.
    for (name,a),(name_b,b) in zip(reference.named_parameters(),optimized.named_parameters()):
        assert name==name_b
        if a.grad is None and b.grad is None:continue
        if a.grad is None or b.grad is None:raise AssertionError("Disconnected parameter "+name)
        torch.testing.assert_close(b.grad,a.grad,rtol=3e-4,atol=3e-5)
        gmax=max(gmax,float((b.grad-a.grad).abs().max()))
    return dict(status="passed",shape=list(right.shape),forward_max_abs=float((left-right).abs().max()),
                gradient_max_abs=gmax,trained=False,real_data=False,factorized_carrier=bool(factor_transport),
                optimization="shared TX-independent medium and first signal-linear projection; unchanged ray/sample quadrature")

if __name__=="__main__":
    import json
    print(json.dumps(equivalence_check()))
