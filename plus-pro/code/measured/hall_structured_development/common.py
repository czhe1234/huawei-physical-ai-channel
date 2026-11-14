"""Immutable development scope and resource/source bookkeeping."""
from __future__ import annotations
import json,os,platform,re,time
from pathlib import Path
import numpy as np
import torch
from project_paths import PATHS
from measured.hall_adapters.common import save_json,sha256
from measured.hall_structured_v2.data import SouthDevelopment,slice_prepared
from measured.hall_structured_v2.foundation import prepare
from measured.hall_structured_v2.observables import FitPower
from measured.hall_structured_v2.config import Config,ALLOWED_GPUS
REPORT=PATHS.analysis/"results/hall_south_structured_development_v2"
WORK=PATHS.artifacts/"measured/hall_sw_structured_development_v2"
SOURCE=PATHS.code_root/"measured/hall_structured_development"
SEED=20261024
def snapshot():
    records={}
    for root in [PATHS.code_root,PATHS.artifacts/"measured",PATHS.analysis,PATHS.root/"final_code",PATHS.datasets]:
      for p in root.rglob("*"):
        if not p.is_file() or p.is_symlink() or "__pycache__" in p.parts or any(x in p.parents for x in [REPORT,WORK,SOURCE]):continue
        st=p.stat();records[str(p.relative_to(PATHS.root))]=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns,sha256=sha256(p) if st.st_size<1048576 else None)
    return records
def init():
    REPORT.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
    if (REPORT/"contract.json").exists():return
    reader=SouthDevelopment()
    contract=dict(primary_question="Can restored frequency/relative-delay/receive-energy structure learn from south FIT and improve the same held-out development metrics over comparable baselines?",
      scope="Small-scale south FIT/validation development only; existing geometry is provisional.",
      data_identity=reader.data.identity,fold=reader.fold,fit_rows=reader.fit_ids.tolist(),validation_rows=reader.validation_ids.tolist(),
      tiny_rows=reader.fit_ids[:8].tolist(),seed=SEED,
      targets="Final-H 1/4/16-band angle-frequency Hellinger, half covariance Frobenius loss and panel/per-beam frequency-correlation MSE equally weighted; FIT-relative power auxiliary0.1",
      selection="Lowest equally weighted final-H structural validation loss; each metric and beam loss also reported",
      health=dict(finite_losses_gradients=True,key_modules_update=True,
        tiny_fitting_gate="At least50% structural loss reduction OR direction cosine>=0.98, with initial/final per-domain errors and non-improving domains reported.",
        convergence="Measured trajectories; plateau means no selection improvement0.0002 for25epochs after40epochs; budget exhaustion is not convergence."),
      comparison=dict(full_channel=["nearest FIT","frequency-power RBF with donor carrier","position MLP with donor carrier","same model without geometry","previous angular version"],
        power_only=["existing healthy WRF-GS+64/128 and power MLP; reuse and independently recompute compatible direction/beam metrics"],
        calibration="unchanged publisher per-file tables",alignment="no query-truth phase/delay/amplitude adjustment"),
      tuning=dict(max_full_ours_candidates=4,max_new_MLP_candidates=2,max_tiny_candidates=3,
        levers=["learning rate","FIT-learned expert initialization","spectral correction range","relative-delay correction range","receive-energy correction range","conservative geometry gate"],
        rule="Use tiny-fit component losses/saturation and development validation evidence to choose bounded changes; every attempted setting retained."),
      resources=dict(max_stage_gpu_minutes=75,max_model_minutes=15,allowed_GPU_UUIDs=list(ALLOWED_GPUS),precision="FP32/complex64, TF32 disabled"),
      non_goals=["Formal south TEST","North channels or tuning","New RT","Full five-fold/multi-seed confirmation","Absolute-H NMSE","Proof of independent geometry causality"])
    save_json(REPORT/"preservation_before.json",snapshot())
    ledger=PATHS.root/"experiment_records/EXPERIMENT_RECORD.md";prefix=ledger.read_bytes();(REPORT/"ledger_before.bin").write_bytes(prefix)
    n=max(map(int,re.findall(r"^## (\d+)\.",prefix.decode(),re.M)))+1
    contract["ledger_item"]=n;save_json(REPORT/"contract.json",contract)
    with ledger.open("a") as f:f.write(f"\n\n## {n}. 2026-10-24 South structured bounded training and baseline comparison\n\n### Contract\n- User authorizes small-scale fitting/convergence, targeted tuning around competition structural advantages and same-split/metric baseline comparison. North and formal south TEST stay unused.\n- Reuse fixed96 FIT/48 validation; existing geometry provisional, no new RT. Final-H structural selection, explicit comparable-output boundaries and bounded tuning.\n- Contract: analysis/results/hall_south_structured_development_v2/contract.json. Actual results follow completed runs.\n")
def device():
    visible=os.environ.get("CUDA_VISIBLE_DEVICES","")
    if visible not in ALLOWED_GPUS:raise RuntimeError("Require exactly one permitted full GPU UUID")
    if torch.cuda.device_count()!=1:raise RuntimeError("Unexpected visible devices")
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    return torch.device("cuda:0")
def seed(value=SEED):
    np.random.seed(value);torch.manual_seed(value);torch.set_num_threads(4)
def source_hashes():
    roots=[SOURCE,PATHS.code_root/"measured/hall_structured_v2"]
    return {str(p.relative_to(PATHS.root)):sha256(p) for root in roots for p in sorted(root.glob("*.py"))}
def bundle(health=False):
    reader=SouthDevelopment(allow_validation=not health);bank=reader.fit()
    qf=reader.query("fit",reader.fit_ids[:8] if health else None);gf=reader.geometry(qf.row_ids)
    fit=prepare(bank,qf,16,"block",gf);y=np.array(bank.channel[:8] if health else bank.channel)
    val=None;yv=None
    if not health:
        qv=reader.query("validation");gv=reader.geometry(qv.row_ids)
        val=prepare(bank,qv,16,"self",gv);yv=reader.reference("validation")
    save_json(REPORT/("health_access.json" if health else "development_access.json"),dict(access=reader.access,test=False,north=False))
    return dict(reader=reader,bank=bank,fit=fit,val=val,y=y,yv=yv,power=FitPower.from_fit(bank.channel,bank.row_ids))
def atomic_torch(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(".partial.pt");torch.save(value,tmp);tmp.replace(path)
def atomic_npz(path,**value):
    tmp=path.with_name(path.name+".partial")
    with tmp.open("wb") as f:np.savez_compressed(f,**value)
    tmp.replace(path)
def environment():
    return dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
      visible_UUID=os.environ.get("CUDA_VISIBLE_DEVICES"),device=torch.cuda.get_device_name(0),
      properties=str(torch.cuda.get_device_properties(0)),threads=torch.get_num_threads())
