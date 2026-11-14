"""South-only physical I/O contract, FIT-only scaling and explicit model wrapper."""
from __future__ import annotations
from dataclasses import dataclass, fields
import json
import numpy as np
import torch
from torch import nn
from project_paths import PATHS
from measured.hall_adapters.common import SouthDataset, Query, frozen_array, sha256, check_prediction

REPORT = PATHS.analysis / "audits/full_h_adaptation_20261031"
SOURCE = PATHS.code_root / "measured/full_h_adapters_v1"
DEVELOPMENT = PATHS.analysis / "results/hall_south_development_v1/contract.json"

@dataclass(frozen=True)
class PhysicalContract:
    frequencies_hz: np.ndarray
    receiver_position_m: np.ndarray
    raw_port_indices: np.ndarray
    panel_rows: int = 8
    panel_columns: int = 4
    coordinate_frame: str = "publisher measurement XYZ, metres"
    complex_convention: str = "H(f)=sum_l a_l(f)*exp(-j*2*pi*f*tau_l)"
    def __post_init__(self):
        f = np.asarray(self.frequencies_hz)
        r = np.asarray(self.receiver_position_m)
        p = np.asarray(self.raw_port_indices)
        if f.dtype != np.float64 or f.shape != (1024,) or not np.isfinite(f).all():
            raise ValueError("Actual FP64 1024-tone frequency axis required")
        expected = 3438000000. + (np.arange(1024, dtype=np.float64)-512)*50000000./1024
        if not np.array_equal(f, expected):
            raise ValueError("Changed, reversed, rounded or subsampled frequency grid")
        if r.shape != (2,32,3) or not np.isfinite(r).all():
            raise ValueError("Physical receiver coordinates must be [2,32,3]")
        if p.shape != (2,32) or p.dtype.kind not in "iu" or not np.array_equal(np.sort(p.ravel()), np.arange(64)):
            raise ValueError("Every raw port must occur exactly once")
        if (self.panel_rows,self.panel_columns)!=(8,4):
            raise ValueError("This version is south 8x4 only")
        object.__setattr__(self,"frequencies_hz",frozen_array(f,np.float64))
        object.__setattr__(self,"receiver_position_m",frozen_array(r,np.float64))
        object.__setattr__(self,"raw_port_indices",frozen_array(p,np.int64))
    @classmethod
    def from_south(cls, data):
        # assignments is [panel,vertical,horizontal]; H uses [panel,horizontal,vertical,F].
        ports=np.asarray(data.meta["antenna_assignments"]).transpose(0,2,1).reshape(2,32)
        xyz=data.spec.center[:,None,None,:]+data.spec.element_offsets
        return cls(data.meta["frequency_hz"],xyz.reshape(2,32,3),ports)
    def describe(self):
        f=self.frequencies_hz
        return dict(shape=["N",2,8,4,1024],dtype="complex64",
          flatten_rule="port_local=horizontal*4+vertical; raw_port=assignments[panel,vertical,horizontal]",
          raw_port_indices=self.raw_port_indices.tolist(),receiver_position_m=self.receiver_position_m.tolist(),
          coordinate_frame=self.coordinate_frame,frequency_order="fftshifted ascending; all bins retained",
          frequency_count=len(f),frequency_first_hz=float(f[0]),frequency_last_hz=float(f[-1]),
          frequency_step_hz=float(f[1]-f[0]),carrier_index=512,carrier_hz=float(f[512]),
          bandwidth_hz=50000000.,complex_convention=self.complex_convention)
    def tensor_inputs(self, query, device="cpu"):
        if type(query) is not Query or {x.name for x in fields(query)}!={"row_ids","position","spec"}:
            raise TypeError("Position-only Query required; channel/DoA/gauge fields are not accepted")
        if not np.array_equal(query.spec.frequencies,self.frequencies_hz):
            raise ValueError("Query frequency contract differs")
        query_xyz=(query.spec.center[:,None,None,:]+query.spec.element_offsets).reshape(2,32,3)
        if not np.array_equal(query_xyz,self.receiver_position_m):
            raise ValueError("Query array/port coordinates differ")
        if len(np.unique(query.row_ids))!=len(query.row_ids):
            raise ValueError("Duplicate query row IDs")
        return (torch.tensor(np.array(query.position),device=device,dtype=torch.float64),
                torch.tensor(np.array(self.receiver_position_m),device=device,dtype=torch.float64),
                torch.tensor(np.array(self.frequencies_hz),device=device,dtype=torch.float64))
    def unflatten(self, h):
        if h.ndim!=4 or tuple(h.shape[1:])!=(2,32,1024) or not torch.is_complex(h):
            raise ValueError("Expected complex [N,2,32,1024]")
        if not bool(torch.isfinite(h).all()):
            raise ValueError("Nonfinite complex output")
        return h.reshape(len(h),2,8,4,1024)

@dataclass(frozen=True)
class FitScale:
    """One positive real RMS per panel; never one scale per query/port/tone."""
    value: np.ndarray
    fit_row_ids: np.ndarray
    data_identity: dict
    def __post_init__(self):
        x=np.asarray(self.value,dtype=np.float64)
        ids=np.asarray(self.fit_row_ids,dtype=np.int64)
        if x.shape!=(2,) or not np.isfinite(x).all() or np.any(x<=0) or len(np.unique(ids))!=len(ids) or not len(ids):
            raise ValueError("Invalid FIT-only physical scale")
        object.__setattr__(self,"value",frozen_array(x,np.float64))
        object.__setattr__(self,"fit_row_ids",frozen_array(ids,np.int64))
    @classmethod
    def from_fit(cls, bank, identity):
        if bank.channel.shape!=(len(bank.row_ids),2,8,4,1024):
            raise ValueError("Unexpected FIT tensor")
        # Float64 accumulation preserves I/Q geometry; no separate real/imag min-max.
        x=np.asarray(bank.channel,np.complex128)
        scale=np.sqrt(np.mean(x.real*x.real+x.imag*x.imag,axis=(0,2,3,4)))
        return cls(scale,bank.row_ids,dict(identity))
    def normalize(self,h):
        if tuple(h.shape[1:])!=(2,8,4,1024) or not np.iscomplexobj(h):
            raise ValueError("Expected south complex channels")
        return np.asarray(h)/self.value[None,:,None,None,None]
    def restore(self,h):
        if tuple(h.shape[1:])!=(2,8,4,1024) or not np.iscomplexobj(h):
            raise ValueError("Expected south complex channels")
        return np.asarray(h)*self.value[None,:,None,None,None]
    def describe(self):
        return dict(rule="one positive real RMS per physical panel over only declared96 FIT positions/ports/tones",
          scale=self.value.tolist(),fit_row_ids=self.fit_row_ids.tolist(),data_identity=self.data_identity,
          preserves_relative_complex_geometry=True,query_scale_fit=False,separate_IQ_normalization=False)

class SouthInputs:
    """No north/custom directory or held-out-channel reader exists here."""
    def __init__(self):
        self.data=SouthDataset()
        self.development=json.loads(DEVELOPMENT.read_text())
        if self.development["data_identity"]!=self.data.identity:
            raise ValueError("Frozen development source identity changed")
        self.fold=int(self.development["fold"])
        self.fit_ids=frozen_array(self.development["fit_rows"],np.int64)
        self.validation_ids=frozen_array(self.development["validation_rows"],np.int64)
        if len(self.fit_ids)!=96 or len(self.validation_ids)!=48 or self.fold!=0:
            raise ValueError("Development sample selection changed")
        if not set(self.fit_ids).issubset(set(self.data.partition(self.fold,"fit"))) or not set(self.validation_ids).issubset(set(self.data.partition(self.fold,"validation"))):
            raise ValueError("Development selection crosses spatial role")
        if set(self.fit_ids)&set(self.validation_ids):
            raise ValueError("Overlapping data roles")
        self.physical=PhysicalContract.from_south(self.data)
        self.fit_reads=[]
    def query(self,role="fit",rows=None):
        if role not in {"fit","validation"}:
            raise ValueError("Formal TEST/north queries are disabled")
        allowed=self.fit_ids if role=="fit" else self.validation_ids
        ids=allowed if rows is None else np.asarray(rows,dtype=np.int64)
        if len(np.unique(ids))!=len(ids) or not set(ids).issubset(set(allowed)):
            raise ValueError("Query role or row mismatch")
        return self.data.query(ids)
    def read_fit(self,rows=None):
        ids=self.fit_ids if rows is None else np.asarray(rows,dtype=np.int64)
        if not set(ids).issubset(set(self.fit_ids)) or len(np.unique(ids))!=len(ids):
            raise ValueError("Channel reads limited to declared96 FIT positions")
        bank=self.data.read_fit(self.fold,ids)
        self.fit_reads.extend(self.data.channel_reads)
        self.data.channel_reads.clear()
        return bank
    def reference(self,*args,**kwargs):
        raise PermissionError("No held-out truth in adapter correctness stage")

class FullHAdapter(nn.Module):
    """Model predicts complex coefficients. Only FIT-derived real scale restores units."""
    def __init__(self,model,physical,scale):
        super().__init__()
        if not isinstance(physical,PhysicalContract) or not isinstance(scale,FitScale):
            raise TypeError("Explicit physical contract and FIT-only scale required")
        self.model=model
        self.physical=physical
        self.scale_metadata=scale
        self.register_buffer("panel_scale",torch.tensor(np.array(scale.value),dtype=torch.float32))
    def forward(self,query):
        tx,rx,f=self.physical.tensor_inputs(query,self.panel_scale.device)
        native=self.model(tx,rx,f)
        if native.shape[0]!=len(query.row_ids):
            raise ValueError("Model output/query batch mismatch")
        normalized=self.physical.unflatten(native)
        return (normalized*self.panel_scale[None,:,None,None,None]).to(torch.complex64)
    def prediction(self,query):
        # No truth argument, pairwise alignment, donor or amplitude fit.
        with torch.no_grad():
            h=self.forward(query).detach().cpu().numpy()
        return check_prediction(dict(channel=h,method=type(self.model).__name__),query)

def file_identity():
    return dict(development_contract_sha256=sha256(DEVELOPMENT),
      reference_policy="panel-common independently self-referenced relative PDP only; raw H unchanged",
      score_policy="reuse frozen evaluators; no combined score; raw unaligned NMSE supplemental",
      north_enabled=False,test_enabled=False)
