"""Shared spatial roles, FIT-only scales, metrics and isolated artifact writers."""
from __future__ import annotations
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time

ALLOWED = (
 "GPU-6b53c3c2-3071-5dfb-f306-2f2f142f8009",
 "GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0",
 "GPU-70b6fceb-c7c4-9467-3c81-097c9284c160")
if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", *ALLOWED):
    raise RuntimeError("Set one allowed full GPU UUID or empty CPU mask before imports")

import numpy as np
import torch
from project_paths import PATHS
from common.official_metrics import (PointMetricStatistics,aggregate_statistics,
                                     point_metric_statistics_torch)
from round2.paper_baseline_adapters_v1.contract import SimulationInputs

ART = PATHS.artifacts / "round2/paper_four_baselines_v1"
REPORT = PATHS.analysis / "results/round2_paper_four_baselines_v1"
RAW = PATHS.round2 / "Round2_Train_Channel.npy"
ART.mkdir(parents=True,exist_ok=True)
REPORT.mkdir(parents=True,exist_ok=True)


def digest_ids(ids):
    return hashlib.sha256(np.asarray(ids,np.int64).tobytes()).hexdigest()


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8<<20),b""):h.update(block)
    return h.hexdigest()


def _new_path(path):
    p=Path(path).resolve()
    if not any(p.is_relative_to(r.resolve()) for r in (ART,REPORT)):
        raise PermissionError("Writes are limited to the isolated training artifact/report roots")
    p.parent.mkdir(parents=True,exist_ok=True)
    return p


def atomic_json(path,value):
    p=_new_path(path);temp=p.with_suffix(p.suffix+".partial")
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    temp.replace(p)


def atomic_npy(path,array,*,replace=False):
    p=_new_path(path)
    if p.exists() and not replace:raise FileExistsError(p)
    temp=p.with_suffix(p.suffix+".partial")
    with temp.open("wb") as f:np.save(f,np.asarray(array),allow_pickle=False)
    temp.replace(p)


def atomic_torch_save(path,value):
    p=_new_path(path);temp=p.with_suffix(p.suffix+".partial")
    torch.save(value,temp);temp.replace(p)


def output_paths(method,outer=None,role=None):
    if method not in ("nerf2","ngrf","newrf","wrfgs"):raise ValueError("Method")
    a=ART/method;r=REPORT/method
    if outer is not None:a=a/f"outer{outer}";r=r/f"outer{outer}"
    if role is not None:a=a/role;r=r/role
    a.mkdir(parents=True,exist_ok=True);r.mkdir(parents=True,exist_ok=True)
    return dict(artifact=a,report=r,artifacts=a,reports=r)


def seed_all(seed=42):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if os.environ["CUDA_VISIBLE_DEVICES"]:torch.cuda.manual_seed_all(seed)


def device_guard(expected_uuid):
    if expected_uuid not in ALLOWED or os.environ["CUDA_VISIBLE_DEVICES"]!=expected_uuid:
        raise RuntimeError("Assigned GPU UUID mismatch")
    if torch.cuda.device_count()!=1:raise RuntimeError("Exactly one visible CUDA device required")
    torch.cuda.set_device(0)
    torch.empty(1,device="cuda:0")
    # CUDA context identity works with MPS: NVML may account only the server PID.
    import ctypes
    import uuid
    driver=ctypes.CDLL("libcuda.so.1")
    device=ctypes.c_int()
    get_device=driver.cuCtxGetDevice
    get_device.argtypes=[ctypes.POINTER(ctypes.c_int)];get_device.restype=ctypes.c_int
    if get_device(ctypes.byref(device))!=0:raise RuntimeError("No actual current CUDA context")
    buffer=(ctypes.c_ubyte*16)()
    get_uuid=getattr(driver,"cuDeviceGetUuid_v2",driver.cuDeviceGetUuid)
    get_uuid.argtypes=[ctypes.c_void_p,ctypes.c_int];get_uuid.restype=ctypes.c_int
    if get_uuid(ctypes.byref(buffer),device.value)!=0:raise RuntimeError("CUDA UUID lookup failed")
    actual="GPU-"+str(uuid.UUID(bytes=bytes(buffer)))
    if actual!=expected_uuid:raise RuntimeError("Actual CUDA context UUID mismatch: "+actual)
    pci=ctypes.create_string_buffer(32)
    get_pci=driver.cuDeviceGetPCIBusId
    get_pci.argtypes=[ctypes.c_char_p,ctypes.c_int,ctypes.c_int];get_pci.restype=ctypes.c_int
    if get_pci(pci,32,device.value)!=0:raise RuntimeError("CUDA PCI lookup failed")
    result=subprocess.run(["nvidia-smi","--query-compute-apps=pid,gpu_uuid,process_name",
                           "--format=csv,noheader,nounits"],capture_output=True,text=True,check=True).stdout
    rows=[line.split(",") for line in result.splitlines()]
    owned=[r[1].strip() for r in rows if r[0].strip()==str(os.getpid())]
    if owned and any(x!=expected_uuid for x in owned):raise RuntimeError("NVML client GPU mismatch")
    mps=[dict(pid=int(r[0].strip()),process_name=r[2].strip()) for r in rows
         if len(r)>=3 and r[1].strip()==expected_uuid and "mps" in r[2].lower()]
    return dict(GPU_UUID=actual,CUDA_context_UUID=actual,PCI_bus=pci.value.decode(),
                logical_device=0,PID=os.getpid(),MPS_servers=mps,
                NVML_direct_client_visible=bool(owned),mapping_verification="current CUDA context UUID",
                name=torch.cuda.get_device_name(0),total_bytes=torch.cuda.get_device_properties(0).total_memory)


def _probe_ids(inputs,ids,count=16):
    picked=[]
    for site in (0,1):
        candidates=ids[inputs.site_id[ids]==site]
        order=np.lexsort((inputs.positions[candidates,1],inputs.positions[candidates,0]))
        candidates=candidates[order]
        if len(candidates):picked.extend(candidates[np.linspace(0,len(candidates)-1,min(count//2,len(candidates)),dtype=int)].tolist())
    if len(picked)<min(count,len(ids)):
        picked.extend([int(x) for x in ids if x not in picked][:min(count,len(ids))-len(picked)])
    return np.array(sorted(set(picked)),np.int64)


class FoldData:
    """Only FIT labels for training and disjoint validation labels for selection.

    Outer truth has no public reader in this class. Predictions use public row
    coordinates; final scoring is performed separately after complete-method output.
    """
    def __init__(self,outer,role="screen",method=None):
        if role not in ("screen","refit"):raise ValueError("role=screen/refit")
        self.inputs=SimulationInputs();self.geometry=self.inputs.geometry
        self.outer=int(outer);self.role=role;self.method=method
        roles=self.inputs.roles(self.outer);self.roles=roles
        self.fit_ids=np.array(roles["screen_fit" if role=="screen" else "refit"],np.int64)
        self.train_ids=self.fit_ids
        self.val_ids=np.array(roles["validation"],np.int64)
        self.outer_ids=np.array(roles["outer"],np.int64)
        self.probe_ids=_probe_ids(self.inputs,self.val_ids)
        self._fit=set(map(int,self.fit_ids));self._val=set(map(int,self.val_ids))
        self._fit_counts=np.zeros(4000,np.int64);self._val_counts=np.zeros(4000,np.int64)
        self._h=np.load(RAW,mmap_mode="r",allow_pickle=False)
        if self._h.shape!=(4000,256,4,192) or self._h.dtype!=np.complex64:
            raise ValueError("Official raw-H header mismatch")
        st=RAW.stat()
        self.raw_identity=dict(path=PATHS.relative(RAW),bytes=st.st_size,mtime_ns=st.st_mtime_ns,inode=st.st_ino)
        self._scale_info=None

    @property
    def scale(self):
        if self._scale_info is None:
            cache=ART/"fit_scales"/f"outer{self.outer}_{self.role}.json"
            cache.parent.mkdir(parents=True,exist_ok=True)
            with (cache.with_suffix(".lock")).open("a+") as lock:
                fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
                contract=dict(schema="FIT-channel-scalar-RMS-v1",input_identity=self.inputs.identity,
                              fit_ids_sha256=digest_ids(self.fit_ids),raw=self.raw_identity,outer=self.outer,role=self.role)
                if cache.exists():
                    d=json.loads(cache.read_text())
                    if d["contract"]!=contract:raise RuntimeError("FIT RMS cache provenance mismatch")
                else:
                    descriptor=PATHS.artifacts/"round2/pre_sionna/round2_train_spectral_descriptors.npz"
                    if descriptor.exists():
                        # Existing point energies, verified against the identical raw-H
                        # definition. Only the permitted FIT slice enters the scale.
                        with np.load(descriptor,allow_pickle=False) as z:
                            power=np.asarray(z["total_power"],np.float64)
                        if power.shape!=(4000,):raise ValueError("Power descriptor header")
                        selected=power[self.fit_ids]
                        if not np.isfinite(selected).all() or np.any(selected<0):raise ValueError("FIT power descriptors")
                        energy=float(selected.sum(dtype=np.float64))
                    else:
                        energy=0.
                        for start in range(0,len(self.fit_ids),32):
                            rows=self.fit_ids[start:start+32]
                            h=self._read(rows,"fit",None,False)
                            energy+=float(np.sum(np.abs(h)**2,dtype=np.float64))
                    denom=len(self.fit_ids)*256*4*192
                    value=math.sqrt(energy/denom)
                    if not math.isfinite(value) or value<=0:raise ValueError("FIT labels have no finite positive power")
                    d=dict(contract=contract,scale=value,fit_total_energy=energy,complex_scalar_count=denom)
                    atomic_json(cache,d)
                self._scale_info=d
        return float(self._scale_info["scale"])

    @property
    def rms(self):return self.scale

    def positions(self,ids):return np.array(self.inputs.positions[np.asarray(ids,np.int64)],copy=True)
    def sites(self,ids):return np.array(self.inputs.site_id[np.asarray(ids,np.int64)],copy=True)
    def frequencies(self,freq_ids=None):
        f=self.geometry.frequencies_hz
        return np.array(f if freq_ids is None else f[np.asarray(freq_ids,np.int64)],copy=True)

    def _read(self,ids,kind,freq_ids,normalized):
        rows=np.asarray(ids)
        if rows.ndim!=1 or rows.dtype.kind not in "iu" or not len(rows):raise ValueError("Integer row vector required")
        allowed=self._fit if kind=="fit" else self._val
        if kind=="validation" and self.role!="screen":raise PermissionError("A refit model cannot select on validation")
        if not set(map(int,rows)).issubset(allowed):raise PermissionError("Label read outside permitted "+kind)
        h=np.array(self._h[rows],copy=True)
        if freq_ids is not None:
            ff=np.asarray(freq_ids)
            if ff.ndim!=1 or ff.dtype.kind not in "iu" or not len(ff) or np.any((ff<0)|(ff>=192)):raise ValueError("Frequency IDs")
            h=np.take(h,ff,axis=-1)
        if not np.isfinite(h).all():raise ValueError("Nonfinite label")
        np.add.at(self._fit_counts if kind=="fit" else self._val_counts,rows,1)
        return (h/self.scale).astype(np.complex64) if normalized else h

    def read_fit(self,ids,freq_ids=None,normalized=True):return self._read(ids,"fit",freq_ids,normalized)
    def read_validation(self,ids=None,freq_ids=None,normalized=True):
        return self._read(self.probe_ids if ids is None else ids,"validation",freq_ids,normalized)

    def info(self):
        return dict(input_identity=self.inputs.identity,raw_channel_identity=self.raw_identity,
          outer=self.outer,role=self.role,fit_ids=self.fit_ids.tolist(),fit_ids_sha256=digest_ids(self.fit_ids),
          validation_ids=self.val_ids.tolist(),selection_probe_ids=self.probe_ids.tolist(),
          outer_ids=self.outer_ids.tolist(),scale=self.scale,
          fit_rows_actually_read=np.flatnonzero(self._fit_counts).tolist(),
          validation_rows_actually_read=np.flatnonzero(self._val_counts).tolist(),
          fit_read_count=int(self._fit_counts.sum()),validation_read_count=int(self._val_counts.sum()),
          query_truth_read=False,oracle_phase_delay_scale_correction=False)


def evaluate(data,predict_fn,ids=None,batch_size=1,power_only=False):
    """Evaluate complete192-tone normalized predictions on disjoint validation.

    predict_fn(row_ids) -> Tensor/ndarray[N,256,4,192]. Physical renderers may
    return native_raw_H / FIT_RMS at this boundary; their internal units stay raw.
    """
    rows=data.probe_ids if ids is None else np.asarray(ids,np.int64)
    if data.role!="screen" or not set(map(int,rows)).issubset(set(data.val_ids)):
        raise PermissionError("Validation selection requires screen-only model and validation rows")
    if not len(rows):raise ValueError("Nonempty validation")
    parts=[];pas_parts=[]
    with torch.no_grad():
        for start in range(0,len(rows),batch_size):
            batch=rows[start:start+batch_size]
            value=predict_fn(batch)
            value=value if isinstance(value,torch.Tensor) else torch.as_tensor(value)
            if tuple(value.shape)!=(len(batch),256,4,192) or not bool(torch.isfinite(value).all()):
                raise ValueError("Validation output shape/finite")
            truth=torch.as_tensor(data.read_validation(batch,normalized=False),device=value.device,dtype=torch.complex64)
            if power_only:
                if value.is_complex() or bool((value<0).any()):raise ValueError("Actual nonnegative codebook power required")
                p=value.float()*(data.scale**2)
                shaped=truth.reshape(len(batch),2,16,8,4,192)
                t=torch.fft.fft2(shaped,dim=(2,3)).abs().square().reshape_as(p)
                numerator=(p*t).sum(1,dtype=torch.float64)
                denominator=torch.sqrt(p.square().sum(1,dtype=torch.float64)*t.square().sum(1,dtype=torch.float64))
                denominator=torch.where(denominator==0,torch.full_like(denominator,1e-12),denominator)
                pas_parts.append((numerator/denominator).float().mean((1,2),dtype=torch.float64).cpu().numpy())
            else:
                if not value.is_complex():raise ValueError("Complete complexH required")
                parts.append(point_metric_statistics_torch(value.to(torch.complex64)*data.scale,truth))
    if power_only:
        ps=np.concatenate(pas_parts)
        return dict(count=len(ps),pas=float(ps.mean()),pdp=None,nmse=None,score=None)
    st=PointMetricStatistics(*[np.concatenate([getattr(p,name) for p in parts]) for name in
                              ("pas","pdp","squared_error","truth_energy")])
    return aggregate_statistics(st)


def save_outer_prediction(data,method,values,*,metadata=None,power_only=False):
    """Save RAW model predictions; no outer truth read or scoring here."""
    if data.role!="refit":raise PermissionError("Final outer prediction must use a refit model")
    a=np.asarray(values)
    if a.shape!=(len(data.outer_ids),256,4,192) or not np.isfinite(a).all():
        raise ValueError("Complete fold output required")
    if power_only:
        if np.iscomplexobj(a) or np.any(a<0):raise ValueError("Nonnegative codebook power required")
        a=a.astype(np.float32);name="power.npy"
    else:
        if not np.iscomplexobj(a):raise ValueError("ComplexH output required")
        a=a.astype(np.complex64);name="channels.npy"
    out=output_paths(method,data.outer,"prediction")["artifact"]
    atomic_npy(out/name,a)
    atomic_npy(out/"row_ids.npy",data.outer_ids)
    manifest=dict(schema="canonical-baseline-prediction-v1",method=method,outer=data.outer,
      input_identity=data.inputs.identity,row_ids=data.outer_ids.tolist(),row_ids_sha256=digest_ids(data.outer_ids),
      prediction_namespace="canonical_train_row_ids",kind="power" if power_only else "channel",
      path=PATHS.relative(out/name),sha256=sha256(out/name),shape=list(a.shape),dtype=str(a.dtype),
      fit_data=data.info(),metadata=metadata or {},query_truth_read=False,trained=True)
    atomic_json(out/"manifest.json",manifest)
    return manifest
