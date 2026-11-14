"""Native fixed-TX/moving-RX nGRF with shared frequency conditioning.

Official nGRF model/network/covariance/spatial-weight bodies are reused.
The decoder gains a frequency feature input and fixed256x4 complex port heads.
Coordinates are translated only; kernel distances remain metres. FIT geometry
sets Gaussian centres/scales. No CSI is an inference input.
"""
from __future__ import annotations
import ast
import hashlib
import math
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace
import numpy as np
import torch
from torch import nn
from scipy.spatial import cKDTree
from round2.paper_baseline_adapters_v1.field_models import _load_previous_adapter
from round2.paper_baseline_training_v1.field_fast import NGRF_SOURCE_CONFIG


MODEL_SHA="475016eda3add7dba39fa83c730079e830bf03c63a8db3137eb49d01a4ae47dd"


def official_components():
    adapter=_load_previous_adapter("ngrf")
    networks,transforms,renderer=adapter._official_components()
    name="_native_ngrf_model_3765d6f"
    if name not in sys.modules:
        p=adapter.SOURCE_ROOT/"models/ngrf_model.py"
        payload=p.read_bytes()
        if hashlib.sha256(payload).hexdigest()!=MODEL_SHA:raise RuntimeError("Pinned nGRF class changed")
        tree=ast.parse(payload)
        tree.body=[n for n in tree.body if not (isinstance(n,ast.ImportFrom) and n.module in
                    ("omegaconf","utils","networks"))]
        module=ModuleType(name);module.__file__=str(p)
        module.DictConfig=object
        module.AttributeNetwork=networks.AttributeNetwork
        module.ContributionDecoderNetwork=networks.ContributionDecoderNetwork
        module.build_rotation=transforms.build_rotation
        module.build_covariance_inverse=transforms.build_covariance_inverse
        module.inverse_sigmoid=lambda x:torch.log(x/(1-x))
        sys.modules[name]=module
        exec(compile(tree,str(p),"exec"),module.__dict__)
    return sys.modules[name].nGRF,networks,transforms,renderer,adapter


def source_config():
    return dict(NGRF_SOURCE_CONFIG)


class NativeSite(nn.Module):
    def __init__(self,bs_position,fit_positions,config=None,sigma_factor=3.):
        super().__init__()
        config={**NGRF_SOURCE_CONFIG,**(config or {})}
        self.config=config
        xyz=np.asarray(fit_positions,np.float64)
        if xyz.ndim!=2 or xyz.shape[1]!=3 or len(xyz)<2:raise ValueError("FIT coordinate set")
        center=xyz.mean(0)
        nearest=cKDTree(xyz).query(xyz,k=2)[0][:,1]
        sigma=float(np.median(nearest[nearest>1e-8]))*float(sigma_factor)
        if not math.isfinite(sigma) or sigma<=0:raise ValueError("FIT geometry has no positive spacing")
        self.register_buffer("coordinate_center",torch.tensor(center,dtype=torch.float64))
        self.register_buffer("fixed_tx",torch.tensor(np.asarray(bs_position)-center,dtype=torch.float64))
        self.sigma_m=sigma
        cls,networks,_,_,_=official_components()
        self.native=cls(num_tx_ant=256,num_rx_ant=4,latent_dim=config["latent_dim"],
            attribute_hidden_dim=config["attribute_hidden_dim"],attribute_num_layers=config["attribute_num_layers"],
            attribute_pos_enc_freqs=config["pos_encoding_freqs"],attribute_dropout_p=config["attribute_dropout_p"],
            decoder_hidden_dim=config["decoder_hidden_dim"],decoder_num_layers=config["decoder_num_layers"],
            decoder_dropout_p=config["decoder_dropout_p"],initial_gaussians=config["num_gaussians"],
            init_opacity_value=.08290565177642037,init_scale_value=sigma,device=torch.device("cpu"))
        self.native.init_gaussians(point_cloud=torch.tensor(xyz-center,dtype=torch.float32),
                                   num_points=min(config["num_gaussians"],len(xyz)))
        self.native.contribution_decoder=networks.ContributionDecoderNetwork(
            latent_dim=config["latent_dim"]+1+2*config["frequency_encoding_levels"],
            output_dim=2*256*4,hidden_dim=config["decoder_hidden_dim"],
            num_layers=config["decoder_num_layers"],dropout_p=config["decoder_dropout_p"])
        # Unit calibration is learned on FIT; it preserves strong channels under
        # the shared RMS normalization despite the original bounded per-Gaussian head.
        self.log_output_gain=nn.Parameter(torch.zeros(()))
        self.fit_geometry_count=len(xyz)
        self.median_nearest_fit_spacing_m=sigma/float(sigma_factor)
        self.sigma_factor=float(sigma_factor)

    def prepare(self,frequencies_hz):
        _,_,_,_,adapter=official_components()
        f=adapter.encode_physical_frequencies(frequencies_hz,6690000000.,5760000.,
                                              self.config["frequency_encoding_levels"])
        latent,activation=self.native.get_attributes_and_activation(self.fixed_tx.to(torch.float32))
        _,inverse=self.native.get_covariance(return_inverse=True,eps=1e-10)
        pieces=[]
        for start in range(0,len(f),self.config["frequency_chunk_size"]):
            feature=f[start:start+self.config["frequency_chunk_size"]]
            g=len(latent);count=len(feature)
            inp=torch.cat([latent[:,None].expand(g,count,-1),feature[None].expand(g,-1,-1)],-1)
            raw=self.native.contribution_decoder(inp).reshape(g,count,2,256,4)
            pieces.append(torch.complex(raw[:,:,0],raw[:,:,1]).permute(0,2,3,1))
        return torch.cat(pieces,-1),activation,inverse

    def weights(self,positions,activation,inverse):
        _,_,_,renderer,_=official_components()
        rx=(positions.to(torch.float64)-self.coordinate_center).to(torch.float32)
        d=rx[:,None]-self.native.get_xyz[None]
        b,g,_=d.shape
        value=renderer.compute_spatial_weight(d.reshape(-1,3),
          inverse[None].expand(b,-1,-1,-1).reshape(-1,3,3),
          activation[None].expand(b,-1,-1).reshape(-1,1))
        return value.reshape(b,g)

    def forward(self,positions,frequencies_hz):
        # Same native sum, immediately reduced per frequency chunk. This avoids
        # materializing G x1024 x192 when memory is shared with another user.
        _,_,_,_,adapter=official_components()
        f=adapter.encode_physical_frequencies(frequencies_hz,6690000000.,5760000.,
                                              self.config["frequency_encoding_levels"])
        latent,activation=self.native.get_attributes_and_activation(self.fixed_tx.to(torch.float32))
        _,inverse=self.native.get_covariance(return_inverse=True,eps=1e-10)
        weights=self.weights(positions,activation,inverse).to(torch.complex64)
        pieces=[]
        for start in range(0,len(f),self.config["frequency_chunk_size"]):
            feature=f[start:start+self.config["frequency_chunk_size"]]
            g=len(latent);count=len(feature)
            inp=torch.cat([latent[:,None].expand(g,count,-1),feature[None].expand(g,-1,-1)],-1)
            raw=self.native.contribution_decoder(inp).reshape(g,count,2,256,4)
            contribution=torch.complex(raw[:,:,0],raw[:,:,1]).permute(0,2,3,1)
            pieces.append(torch.einsum("bg,gmnf->bmnf",weights,contribution))
        return torch.cat(pieces,-1)*self.log_output_gain.exp()

    def activation_penalty(self):
        return self.native.get_base_activation_logits(self.fixed_tx.to(torch.float32)).abs().mean()


class NativeNGRF(nn.Module):
    def __init__(self,geometry,fit_positions,fit_sites,config=None,sigma_factor=3.):
        super().__init__()
        self.geometry=geometry
        self.bound_geometry_fingerprint=geometry.geometry_fingerprint
        self.fields=nn.ModuleDict()
        for s in (0,1):
            pos=np.asarray(fit_positions)[np.asarray(fit_sites)==s]
            self.fields[str(s)]=NativeSite(geometry.bs_positions_m[s],pos,config,sigma_factor)
        self.config={**NGRF_SOURCE_CONFIG,**(config or {})}

    def forward(self,positions,sites,frequencies_hz):
        if positions.shape!=(len(sites),3) or sites.dtype!=torch.int64:
            raise ValueError("Native nGRF query contract")
        if frequencies_hz.dtype!=torch.float64:raise TypeError("Actual Hz must remain FP64")
        result=torch.zeros((len(positions),256,4,len(frequencies_hz)),device=positions.device,dtype=torch.complex64)
        for s in (0,1):
            rows=torch.nonzero(sites==s).flatten()
            if len(rows):
                value=self.fields[str(s)](positions[rows],frequencies_hz)
                result=result.index_copy(0,rows,value)
        return result

    def metadata(self):
        return dict(method="nGRF_native_fixed_BS_TX_moving_UE_RX_full_H",
            source_commit="3765d6f1e46a223914908e38430b1ab9d2a5f4fb",source_model_sha256=MODEL_SHA,
            config=self.config,geometry_fingerprint=self.bound_geometry_fingerprint,
            output="complex64[B,256,4,F], BS p,h,v; UE p,v",
            native_mechanism="fixed BS-conditioned Gaussian attributes/contributions; moving UE Gaussian spatial weights; coherent complex sum",
            frequency_adaptation="shared FP64-Hz Fourier features extend original decoder input",
            port_binding="1024 independent native MIMO complex heads at fixed declared BS/UE port order",
            role_remapping=False,query_CSI=False,oracle_correction=False,
            coordinate_translation_only=True,
            sigma_m_by_site={s:f.sigma_m for s,f in self.fields.items()},
            sigma_factor_by_site={s:f.sigma_factor for s,f in self.fields.items()},
            amplitude_calibration="one positive learnable FIT-trained gain per site; no query correction",
            limits=["fixed-array task adaptation, no arbitrary-array claim",
                    "world geometry/frequency follows declared V16 prior convention",
                    "not an unchanged original-paper experiment"])


def native_optimizer(model):
    rates=dict(xyz=.0002696363015043561,rotation=.0006292551064997657,
               scaling=.0011355962014815962,attribute_net=.00031728462884002734,decoder=.0005534331508483368)
    groups=[]
    for site,field in model.fields.items():
        for group in field.native.get_params(rates):
            group={**group,"initial_lr":group["lr"],"name":site+"/"+group["name"]}
            groups.append(group)
        groups.append(dict(params=[field.log_output_gain],lr=.001,initial_lr=.001,name=site+"/output_gain"))
    return torch.optim.AdamW(groups,eps=1.3209624572295576e-09,weight_decay=3.146520107386117e-08)


def equivalence_check():
    """Compare new frequency-conditioned native rendering with official scalar renderer."""
    torch.set_num_threads(2);torch.manual_seed(481)
    fit=np.array([[0.,0.,1.5],[2.,0.,1.5],[0.,2.,1.5],[2.,2.,1.5]],np.float64)
    cfg=dict(num_gaussians=4,latent_dim=4,attribute_hidden_dim=8,attribute_num_layers=2,
             attribute_dropout_p=0.,decoder_hidden_dim=8,decoder_num_layers=2,decoder_dropout_p=0.,
             pos_encoding_freqs=2,frequency_chunk_size=2)
    field=NativeSite(np.array([0.,0.,20.]),fit,cfg,sigma_factor=1.).eval()
    frequency=torch.tensor([6690030000.],dtype=torch.float64)
    query=torch.tensor([[.3,.4,1.5],[1.1,.7,1.5]],dtype=torch.float64)
    value=field(query,frequency)[...,0]
    _,_,_,renderer,adapter=official_components()
    encoded=adapter.encode_physical_frequencies(frequency,6690000000.,5760000.,4)[0]
    def decoder(latent):
        return field.native.contribution_decoder(torch.cat([latent,encoded.expand(len(latent),-1)],-1))
    native=SimpleNamespace(device=torch.device("cpu"),get_xyz=field.native.get_xyz,
        get_attributes_and_activation=field.native.get_attributes_and_activation,
        get_covariance=field.native.get_covariance,contribution_decoder=decoder)
    reference=renderer.render_channel((query-field.coordinate_center).float(),native,field.fixed_tx.float(),256,4)
    torch.testing.assert_close(value,reference,rtol=1e-5,atol=1e-6)
    grads_a=torch.autograd.grad(value.abs().square().mean()+.1*value.real.mean(),
        tuple(field.native.parameters()),retain_graph=True,allow_unused=True)
    grads_b=torch.autograd.grad(reference.abs().square().mean()+.1*reference.real.mean(),
        tuple(field.native.parameters()),allow_unused=True)
    max_grad=0.
    for a,b in zip(grads_a,grads_b):
        if a is None and b is None:continue
        if a is None or b is None:raise AssertionError("Native parameter gradient disconnected")
        torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
        max_grad=max(max_grad,float((a-b).abs().max()))
    frequencies=torch.tensor([6690000000.,6690030000.,6692850000.],dtype=torch.float64)
    streamed=field(query,frequencies)
    contribution,activation,inverse=field.prepare(frequencies)
    materialized=torch.einsum("bg,gmnf->bmnf",field.weights(query,activation,inverse).to(torch.complex64),contribution)*field.log_output_gain.exp()
    torch.testing.assert_close(streamed,materialized,rtol=1e-5,atol=1e-6)
    return dict(status="passed",streaming_max_abs=float((streamed-materialized).abs().max()),forward_max_abs=float((value-reference).abs().max()),
                gradient_max_abs=max_grad,shape=list(value.shape),actual_data=False,
                reference="pinned official render_channel, full1024 native MIMO heads at one physical frequency")

if __name__=="__main__":
    import json
    print(json.dumps(equivalence_check()))
