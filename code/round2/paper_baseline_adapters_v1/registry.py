"""Declared output interfaces. No automatic fit, checkpoint load or ray tracing."""
from __future__ import annotations
import numpy as np
import torch
from .contract import PositionQuery,DFT_CONTRACT_ID,validate_output

class SimulationAdapter:
    def __init__(self,method,model,inputs):
        from .field_models import CompetitionNeRF2,CompetitionNGRF
        from .wrfgs import WRFGSPlusPowerAdapter
        from .newrf import NeWRFFullH
        expected={"nerf2":CompetitionNeRF2,"ngrf":CompetitionNGRF,"wrfgs":WRFGSPlusPowerAdapter,"newrf":NeWRFFullH}
        if method not in expected or type(model) is not expected[method]:
            raise TypeError("Only the reviewed position-forward adapter class is accepted; no stored V16 file wrapper")
        if method=="newrf":
            from .newrf import require_real_data_qualification
            require_real_data_qualification()
        fingerprint=inputs.geometry.geometry_fingerprint
        if method in {"nerf2","ngrf"} and model.bound_geometry_fingerprint!=fingerprint:
            raise ValueError("Model and input geometry identities differ")
        if method=="wrfgs" and model.geometry_identity!=fingerprint:
            raise ValueError("WRF model and input geometry identities differ")
        self.method=method;self.model=model;self.inputs=inputs
    def forward(self,query):
        if type(query) is not PositionQuery or query.input_identity!=self.inputs.identity:
            raise ValueError("Query belongs to a different input contract")
        pos,site=query.tensors()
        f=torch.tensor(np.array(self.inputs.geometry.frequencies_hz),dtype=torch.float64)
        return self.model(pos,site,f)
    def prediction(self,query):
        with torch.no_grad():a=self.forward(query).detach().cpu().numpy()
        key="power" if self.method=="wrfgs" else "channel"
        out=dict(method=self.method,row_ids=np.array(query.row_ids),
          prediction_namespace=query.input_namespace,input_identity=self.inputs.identity,
          geometry_identity=self.inputs.geometry.geometry_fingerprint,
          prediction_origin="position_only_forward",**{key:a})
        if key=="power":
            info=self.model.eligibility()
            if not info.get("official_PAS") or info.get("PAS_requires")!=DFT_CONTRACT_ID:
                raise ValueError("WRF power output lacks a verified target operator binding")
            out["eligibility"]={**info,"contract_id":DFT_CONTRACT_ID}
        return validate_output(out,query)

def make_field_model(method,inputs,config):
    from .field_models import CompetitionNeRF2,CompetitionNGRF
    if method=="nerf2":return CompetitionNeRF2(inputs.geometry,field_config=config)
    if method=="ngrf":return CompetitionNGRF(inputs.geometry,field_config=config)
    raise ValueError("Use method-specific explicit factory for WRF or NeWRF")

def make_wrf_model(inputs,fit_position,initial_xyz_by_site,*,width=32,height=16,outer=0,synthetic_fixture=False):
    from .wrfgs import WRFGSPlusPowerAdapter
    geometry=inputs.geometry
    if not synthetic_fixture:
        permitted=inputs.positions[inputs.roles(outer)["screen_fit"]]
        if not np.array_equal(np.asarray(fit_position),permitted):
            raise ValueError("WRF position normalization must use exactly the declared screenFIT positions")
    # R columns are local normal,horizontal,up; camera order is horizontal,up,normal.
    axes=geometry.bs_rotations[:,:,[1,2,0]]
    return WRFGSPlusPowerAdapter(fit_position,geometry.bs_positions_m,axes,initial_xyz_by_site,
       torch.tensor(np.array(geometry.frequencies_hz),dtype=torch.float64),width=width,height=height,
       contract_id=DFT_CONTRACT_ID,geometry_identity=geometry.geometry_fingerprint)

def make_newrf_model(inputs,ray_provider,config):
    from .newrf import require_real_data_qualification,NeWRFFullH
    require_real_data_qualification()
    return NeWRFFullH(inputs.geometry,ray_provider,config,synthetic_only=False)
