"""NeWRF v3: declared city-scale coordinate and positive-opacity parameterization."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .common import ART,REPORT,atomic_json,atomic_npy,digest_ids,sha256
from project_paths import PATHS
from .newrf_latefreq import LateFrequencyConfig,LateFrequencyField,LateFrequencyNeWRF

ADAPTER_VERSION="NeWRF-wideband-city-conditioned-v3"
# Public Round2_Map.ply max absolute coordinate; no label-dependent input scaling.
PUBLIC_MAP_MAXABS_M=426.831421
NATIVE_REFERENCE_LENGTH_M=9.
RAW_COORDINATE_SCALE_M=PUBLIC_MAP_MAXABS_M/NATIVE_REFERENCE_LENGTH_M
INITIAL_RAW_DENSITY_BIAS=math.log(math.expm1(1.))

@dataclass(frozen=True)
class ConditionedConfig(LateFrequencyConfig):
    raw_coordinate_scale_m:float=RAW_COORDINATE_SCALE_M
    density_parameterization:str="softplus(raw_alpha)/far_m"
    initial_optical_depth:float=1.

class ConditionedField(LateFrequencyField):
    def encoded_position(self,points,tx):
        x=super().encoded_position(points,tx)
        # Only the six identity channels are bounded. Physical sin/cos stay exact.
        return torch.cat((x[...,:6]/self.cfg.raw_coordinate_scale_m,x[...,6:]),-1)

    def physical_density(self,raw_alpha):
        # Dimension: inverse metres. Equivalent reference9m opacity times9/far.
        return F.softplus(raw_alpha)/self.cfg.far_m

    def raw_forward(self,points,view,tx,frequencies):
        return super().forward(points,view,tx,frequencies)

    def forward(self,points,view,tx,frequencies):
        raw=self.raw_forward(points,view,tx,frequencies)
        return torch.cat((raw[...,:2],self.physical_density(raw[...,-1:])),dim=-1)

    def prepare(self,points,view,tx):
        base,density=super().prepare(points,view,tx)
        return base,self.physical_density(density)

class ConditionedNeWRF(LateFrequencyNeWRF):
    def __init__(self,bank,config,*,execution="shared"):
        super().__init__(bank,config,execution=execution)
        self.coarse=nn.ModuleList([ConditionedField(config) for _ in range(8)])
        self.fine=nn.ModuleList([ConditionedField(config) for _ in range(8)])
        with torch.no_grad():
            for field in list(self.coarse)+list(self.fine):
                field.network.alpha_out.bias.fill_(INITIAL_RAW_DENSITY_BIAS)

    def metadata(self):
        out=super().metadata()
        out.update(adapter_version=ADAPTER_VERSION,raw_XYZ_identity_scale_m=RAW_COORDINATE_SCALE_M,
          physical_sinusoidal_PE_unchanged=True,density_units="inverse metres",
          density_parameterization="softplus(raw_alpha)/far_m",
          initial_raw_alpha_bias=INITIAL_RAW_DENSITY_BIAS,
          exact_old_ReLU_weight_equivalence=False,
          density_expressivity="all positive rho via inverse softplus; zero is limiting value")
        return out

def build_conditioned_model(data,cache,coarse=128,importance=128,*,frequency_chunk=8,execution="shared"):
    cfg=ConditionedConfig(n_layers=6,hidden_dim=128,skip=(4,),position_levels=10,
         direction_levels=4,frequency_levels=10,n_samples=coarse,n_importance=importance,
         far_m=cache.far_m,coordinate_scale_m=1.,frequency_chunk=frequency_chunk)
    return ConditionedNeWRF(cache.bank,cfg,execution=execution)

def output_paths_v3(method,outer=None,role=None):
    if method!="newrf":raise ValueError("v3 belongs to NeWRF only")
    a=ART/"newrf"/"v3";r=REPORT/"newrf"/"v3"
    if outer is not None:a=a/f"outer{outer}";r=r/f"outer{outer}"
    if role is not None:a=a/role;r=r/role
    a.mkdir(parents=True,exist_ok=True);r.mkdir(parents=True,exist_ok=True)
    return dict(artifact=a,report=r)

def save_outer_prediction_v3(data,method,values,*,metadata=None):
    if data.role!="refit" or method!="newrf":raise PermissionError("NeWRF refit required")
    a=np.asarray(values,dtype=np.complex64)
    if a.shape!=(len(data.outer_ids),256,4,192) or not np.isfinite(a).all():
        raise ValueError("Complete finite foldH required")
    out=output_paths_v3(method,data.outer,"prediction")["artifact"]
    atomic_npy(out/"channels.npy",a);atomic_npy(out/"row_ids.npy",data.outer_ids)
    meta=dict(metadata or {})
    meta.update(adapter_version=ADAPTER_VERSION,raw_H=True,
                oracle_phase_delay_scale_alignment=False,
                health_evidence=meta.get("formal_config",{}).get("health_evidence",{}))
    if "source_identities" in meta:meta["source_identity"]=meta["source_identities"]
    manifest=dict(schema="canonical-baseline-prediction-v1",method=method,adapter_version=ADAPTER_VERSION,
       outer=data.outer,input_identity=data.inputs.identity,row_ids=data.outer_ids.tolist(),
       row_ids_sha256=digest_ids(data.outer_ids),prediction_namespace="canonical_train_row_ids",
       kind="channel",path=PATHS.relative(out/"channels.npy"),sha256=sha256(out/"channels.npy"),
       shape=list(a.shape),dtype=str(a.dtype),fit_data=data.info(),metadata=meta,
       query_truth_read=False,trained=True)
    atomic_json(out/"manifest.json",manifest)
    return manifest

