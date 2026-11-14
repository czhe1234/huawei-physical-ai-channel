"""Immutable Huawei Round2 I/O, canonical nested splits, and V16 conventions."""
from __future__ import annotations
from dataclasses import dataclass
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from project_paths import PATHS

SOURCE=PATHS.code_root/"round2/paper_baseline_adapters_v1"
REPORT=PATHS.analysis/"audits/simulation_baseline_adapters_20261031"
FOLDS=PATHS.artifacts/"round2/foundation/round2_spatial_hole_folds.npz"
META=PATHS.artifacts/"round2/pre_sionna/round2_pre_sionna_oof_foundation_meta.npz"
V16=PATHS.root/"final_code/V16_reorganized_submission/code/round2"
DFT_CONTRACT_ID="official_fft2_power_unshifted_P2_H16_V8_unnormalized"
C0=299792458.

def frozen(a,dtype=None):
    x=np.array(a,dtype=dtype,copy=True);x.setflags(write=False);return x
def sha256(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda:f.read(8<<20),b""):h.update(block)
    return h.hexdigest()
def save_json(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+".partial");tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+"\n");tmp.replace(p)
def literal_constants(path,names):
    out={}
    for n in ast.parse(Path(path).read_text()).body:
        if isinstance(n,ast.Assign):
            for target in n.targets:
                if isinstance(target,ast.Name) and target.id in names:out[target.id]=ast.literal_eval(n.value)
    if set(out)!=set(names):raise ValueError("V16 constant source mismatch")
    return out
def rotation(yaw,pitch):
    ca,sa=np.cos(yaw),np.sin(yaw);cb,sb=np.cos(pitch),np.sin(pitch)
    return np.array([[ca*cb,-sa,ca*sb],[sa*cb,ca,sa*sb],[-sb,0.,cb]],np.float64)

@dataclass(frozen=True)
class PositionQuery:
    row_ids: np.ndarray
    ue_position: np.ndarray
    site_id: np.ndarray
    input_identity: str
    input_namespace: str = "canonical_train_row_ids"
    def __post_init__(self):
        if self.input_namespace!="canonical_train_row_ids":
            raise ValueError("Only canonical training-row queries are enabled")
        ids=np.asarray(self.row_ids);pos=np.asarray(self.ue_position);site=np.asarray(self.site_id)
        if ids.ndim!=1 or ids.dtype.kind not in "iu" or len(np.unique(ids))!=len(ids):
            raise ValueError("Unique integer query IDs required")
        if pos.shape!=(len(ids),3) or not np.isfinite(pos).all() or site.shape!=(len(ids),) or not np.isin(site,[0,1]).all():
            raise ValueError("Invalid position/site query")
        object.__setattr__(self,"row_ids",frozen(ids,np.int64));object.__setattr__(self,"ue_position",frozen(pos,np.float64));object.__setattr__(self,"site_id",frozen(site,np.int64))
    def tensors(self):
        return torch.tensor(np.array(self.ue_position),dtype=torch.float64),torch.tensor(np.array(self.site_id),dtype=torch.int64)

class V16Geometry:
    """Traceable V16 physical-prior convention, not publisher-certified array survey."""
    def __init__(self,setup):
        path=V16/"physical_prior/scene_prior_data.py"
        names=["SCENE_FREQUENCY_HZ","SITE_YAW_DEG","SITE_TILT_DEG","TX_VERTICAL_SPACING","TX_HORIZONTAL_SPACING"]
        c=literal_constants(path,names)
        config_path=V16/"physical_ai_submission/config.py"
        tree=ast.parse(config_path.read_text())
        config={}
        for node in tree.body:
            if isinstance(node,ast.ClassDef) and node.name=="PipelineConfig":
                for n in node.body:
                    if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name) and n.target.id in ("physical_carrier_frequency_hz","subcarrier_spacing_hz","subcarriers"):
                        config[n.target.id]=ast.literal_eval(n.value)
        if config!=dict(physical_carrier_frequency_hz=6690000000.,subcarrier_spacing_hz=30000.,subcarriers=192):raise ValueError("V16 frequency configuration changed")
        if c["SCENE_FREQUENCY_HZ"]!=config["physical_carrier_frequency_hz"]:raise ValueError("V16 scene and field carrier differ")
        self.frequencies_hz=frozen(c["SCENE_FREQUENCY_HZ"]+(np.arange(192,dtype=np.float64)-96)*30000.)
        self.bs_positions_m=frozen(setup["X"],np.float64)
        self.bs_polarization_labels=("vertical","horizontal")
        self.ue_polarization_labels=("slant_minus45","slant_plus45")
        self.frequency_center_hz=float(c["SCENE_FREQUENCY_HZ"])
        self.frequency_spacing_hz=30000.
        wavelength=C0/self.frequency_center_hz
        h,v=np.meshgrid(np.arange(16)-7.5,3.5-np.arange(8),indexing="ij")
        local=np.stack([np.zeros_like(h),h*c["TX_HORIZONTAL_SPACING"]*wavelength,v*c["TX_VERTICAL_SPACING"]*wavelength],-1)
        rots=np.stack([rotation(np.deg2rad(y),-np.deg2rad(t)) for y,t in zip(c["SITE_YAW_DEG"],c["SITE_TILT_DEG"])])
        self.bs_rotations=frozen(rots)
        xyz=self.bs_positions_m[:,None,None,:]+np.einsum("sij,hvj->shvi",rots,local)
        self.bs_port_positions_m=frozen(np.repeat(xyz[:,None],2,axis=1))
        ue=np.zeros((2,2,1,2,3),np.float64)
        ue[:,:,:,:,2]=np.array([.25,-.25])*wavelength
        self.ue_port_offsets_m=frozen(ue)
        self.ue_orientation_policy="look_at_site_bs"
        self.provenance=dict(kind="frozen_V16_physical_prior_implementation_convention",
          publisher_independently_verified=False,compatible_with_V16_prior=True,
          frequency_axis="centered bin convention fc+(k-96)*30kHz, k0..191; no extra fftshift",
          frequency_grid_anchor_independently_certified=False,
          source_files={PATHS.relative(p):sha256(p) for p in [path,config_path,V16/"physical_prior/geometry_prior_engine/array_model.py",V16/"physical_prior/geometry_prior_engine/endpoints.py",V16/"physical_prior/geometry_prior_engine/utils/geometry.py"]},
          bs_yaw_deg=list(c["SITE_YAW_DEG"]),bs_tilt_deg=list(c["SITE_TILT_DEG"]),
          bs_spacing_wavelengths=[c["TX_HORIZONTAL_SPACING"],c["TX_VERTICAL_SPACING"]],
          ue_spacing_wavelengths=.5,ue_orientation="look_at_each_site_BS; roll0",
          bs_port_order="polarization,horizontal,vertical",ue_port_order="polarization,horizontal=0,vertical",
          limitations=["V16 prior assumptions are not independently released array/frequency truth",
                      "A global frequency-origin phase is learned, never repaired with query truth"])
        payload=dict(f=self.frequencies_hz.tolist(),bs=self.bs_port_positions_m.tolist(),ue=self.ue_port_offsets_m.tolist(),provenance=self.provenance)
        self.geometry_fingerprint=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    def ue_world_positions(self,ue_position,site_id):
        # All operations differentiable in Torch; geometry is fixed, no query labels.
        if ue_position.ndim!=2 or ue_position.shape[-1]!=3 or site_id.shape!=(len(ue_position),):
            raise ValueError("UE position/site shape")
        if site_id.dtype!=torch.int64 or not bool(torch.isin(site_id,torch.tensor([0,1],device=site_id.device)).all()):
            raise ValueError("Unknown site IDs")
        bs=torch.tensor(np.array(self.bs_positions_m),device=ue_position.device,dtype=ue_position.dtype)[site_id]
        d=bs-ue_position
        if bool((d.square().sum(-1)==0).any()):raise ValueError("UE coincides with BS")
        yaw=torch.atan2(d[:,1],d[:,0]);pitch=torch.atan2(-d[:,2],torch.linalg.vector_norm(d[:,:2],dim=-1))
        ca,sa=torch.cos(yaw),torch.sin(yaw);cb,sb=torch.cos(pitch),torch.sin(pitch);zero=torch.zeros_like(ca)
        rot=torch.stack([ca*cb,-sa,ca*sb,sa*cb,ca,sa*sb,-sb,zero,cb],-1).reshape(-1,3,3)
        local=torch.tensor(np.array(self.ue_port_offsets_m),device=ue_position.device,dtype=ue_position.dtype)[site_id].reshape(-1,4,3)
        return ue_position[:,None]+torch.einsum("bij,bmj->bmi",rot,local)
    def metadata(self):
        return dict(frequency_first_hz=float(self.frequencies_hz[0]),frequency_last_hz=float(self.frequencies_hz[-1]),
          frequency_step_hz=30000.,frequency_count=192,center_index=96,geometry_fingerprint=self.geometry_fingerprint,
          bs_shape=list(self.bs_port_positions_m.shape),ue_shape=list(self.ue_port_offsets_m.shape),provenance=self.provenance)

class SimulationInputs:
    def __init__(self):
        self.setup_path=PATHS.round2/"Round2_Setup.json"
        self.setup=json.loads(self.setup_path.read_text())
        for key,want in dict(P_Train=4000,P_Test=500,M=256,M_P=2,M_H=16,M_V=8,N=4,N_P=2,N_H=1,N_V=2,S=192,Q=2).items():
            if self.setup[key]!=want:raise ValueError("Official shape changed")
        self.positions=frozen(np.load(PATHS.round2/"Round2_Train_Pos.npy",allow_pickle=False),np.float64)
        if self.positions.shape!=(4000,3) or not np.isfinite(self.positions).all():raise ValueError("Train position shape")
        self.site_id=frozen((self.positions[:,1]>=0).astype(np.int64))
        if not np.array_equal(np.bincount(self.site_id),[2000,2000]):raise ValueError("Site mapping changed")
        self.geometry=V16Geometry(self.setup)
        self.folds={}
        with np.load(FOLDS,allow_pickle=False) as z:
            for k in z.files:
                a=z[k]
                if a.ndim!=1 or a.dtype.kind not in "iu" or np.any((a<0)|(a>=4000)) or len(np.unique(a))!=len(a):
                    raise ValueError("Invalid original canonical role IDs")
                self.folds[k]=frozen(a,np.int64)
        with np.load(META,allow_pickle=False) as z:
            q=z["query_ids"];f=z["fold_id"]
            if q.dtype.kind not in "iu" or f.dtype.kind not in "iu" or q.shape!=(500,) or f.shape!=(500,) or np.any((q<0)|(q>=4000)) or not np.isin(f,np.arange(5)).all():
                raise ValueError("Invalid canonical IDs/fold dtype or domain")
            self.query_ids=frozen(q,np.int64);self.fold_id=frozen(f,np.int64)
        if self.query_ids.shape!=(500,) or len(np.unique(self.query_ids))!=500:raise ValueError("Expected500 unique canonical query IDs")
        for outer in range(5):
            if not np.array_equal(self.folds[f"fold{outer}_query"],self.query_ids[self.fold_id==outer]):
                raise ValueError("Canonical metadata/outer query identity mismatch")
        union=np.concatenate([self.folds[f"fold{i}_query"] for i in range(5)])
        if len(np.unique(union))!=500 or not np.array_equal(np.sort(union),np.sort(self.query_ids)):
            raise ValueError("Canonical outer union differs")
        self.identity=hashlib.sha256(json.dumps(dict(setup=sha256(self.setup_path),positions=sha256(PATHS.round2/"Round2_Train_Pos.npy"),folds=sha256(FOLDS),meta=sha256(META),geometry=self.geometry.geometry_fingerprint),sort_keys=True).encode()).hexdigest()
        self.access=[]
    def roles(self,outer):
        if outer not in range(5):raise ValueError("Outer fold")
        val=(outer+1)%5
        fit=self.folds[f"fold{outer}_fit"];vfit=self.folds[f"fold{val}_fit"]
        roles=dict(screen_fit=frozen(np.intersect1d(fit,vfit),np.int64),validation=self.folds[f"fold{val}_query"],
                   refit=fit,outer=self.folds[f"fold{outer}_query"],outer_ignored=self.folds[f"fold{outer}_ignored"])
        if any(np.intersect1d(roles[a],roles[b]).size for a,b in [("screen_fit","validation"),("screen_fit","outer"),("refit","outer")]):
            raise ValueError("Label leakage in canonical roles")
        return roles
    def query(self,outer,role,rows=None):
        if role not in {"screen_fit","validation","outer"}:raise ValueError("No official-Test input in correctness stage")
        allowed=self.roles(outer)[role]
        ids=allowed if rows is None else np.asarray(rows,np.int64)
        if ids.ndim!=1 or len(np.unique(ids))!=len(ids) or not set(ids).issubset(set(allowed)):raise ValueError("Query role mismatch")
        return PositionQuery(ids,self.positions[ids],self.site_id[ids],self.identity)
    def read_fit(self,outer,rows):
        rows=np.asarray(rows,np.int64)
        if not len(rows) or len(np.unique(rows))!=len(rows) or not set(rows).issubset(set(self.roles(outer)["screen_fit"])):
            raise PermissionError("Only canonical screenFIT labels may be read")
        x=np.load(PATHS.round2/"Round2_Train_Channel.npy",mmap_mode="r",allow_pickle=False)
        if x.shape!=(4000,256,4,192) or x.dtype!=np.complex64:raise ValueError("Official channel header")
        value=np.array(x[rows],copy=True)
        if not np.isfinite(value).all():raise ValueError("Nonfinite FIT")
        self.access.append(dict(outer=outer,role="screen_fit",rows=rows.tolist()))
        return value
    def reference(self,*args,**kwargs):raise PermissionError("No held-out-label reader in correctness stage")

def validate_output(result,query):
    if set(result)-{"channel","power","row_ids","method","eligibility","prediction_namespace","input_identity","geometry_identity","prediction_origin"}:raise ValueError("Undeclared output")
    if result.get("prediction_namespace")!=query.input_namespace or result.get("input_identity")!=query.input_identity:
        raise ValueError("Prediction source namespace/input contract mismatch")
    if result.get("prediction_origin")!="position_only_forward":
        raise ValueError("Stored prediction files need a separate verified OOF loader; not accepted here")
    if result.get("method") not in {"nerf2","ngrf","newrf","wrfgs"}:
        raise ValueError("V16 online file is not a canonical OOF forward artifact")
    if result.get("method")=="wrfgs" and "channel" in result:
        raise ValueError("WRF power adapter cannot fabricate H eligibility")
    if result.get("method")!="wrfgs" and "power" in result:
        raise ValueError("Full-H methods must emit their actual complex H")
    if result.get("method")=="newrf":
        from .newrf import require_real_data_qualification
        require_real_data_qualification()
    if not result.get("geometry_identity"):
        raise ValueError("Missing prediction geometry identity")
    if not np.array_equal(np.asarray(result["row_ids"]),query.row_ids):raise ValueError("Output ID order")
    expected=(len(query.row_ids),256,4,192)
    if ("channel" in result)==("power" in result):raise ValueError("Declare exactly one actual output kind")
    if "channel" in result:
        a=np.asarray(result["channel"])
        if a.shape!=expected or a.dtype not in (np.complex64,np.complex128) or not np.isfinite(a).all():raise ValueError("Full H contract")
    else:
        a=np.asarray(result["power"])
        if a.shape!=expected or np.iscomplexobj(a) or not np.isfinite(a).all() or np.any(a<0):raise ValueError("Power contract")
        if result.get("eligibility",{}).get("contract_id")!=DFT_CONTRACT_ID:raise ValueError("Unknown power target operator")
    return result
