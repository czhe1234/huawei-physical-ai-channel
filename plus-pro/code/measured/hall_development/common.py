"""Versioned development contract and explicit held-out reference access."""
from __future__ import annotations
import json, os, time, platform
from pathlib import Path
import numpy as np
import torch
from project_paths import PATHS
from measured.hall_adapters.common import SouthDataset, SOUTH, save_json, sha256, frozen_array
from measured.hall_adapters.metrics import directional_power, normalized_power, primary_metrics, aggregate_primary
REPORT=PATHS.analysis/"results/hall_south_development_v1"
WORK=PATHS.artifacts/"measured/hall_sw_development_v1"
FOLD=0
SEED=20261024
GPU="GPU-6b53c3c2-3071-5dfb-f306-2f2f142f8009"
def fps(ids,position,count):
    ids=np.sort(np.asarray(ids,dtype=np.int64));p=np.asarray(position)[ids,:2]
    chosen=[0];distance=np.sum((p-p[0])**2,axis=-1)
    while len(chosen)<min(count,len(ids)):
        distance[chosen]=-1;index=int(np.argmax(distance));chosen.append(index)
        distance=np.minimum(distance,np.sum((p-p[index])**2,axis=-1))
    return ids[chosen]
def init():
    REPORT.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
    data=SouthDataset();fit=fps(data.partition(FOLD,"fit"),data.meta["position"],96)
    val=fps(data.partition(FOLD,"validation"),data.meta["position"],48)
    probes=np.r_[fps(fit,data.meta["position"],8),fps(val,data.meta["position"],4)]
    contract=dict(scope="Bounded south development; fit and validation only; no south test or north channel access.",
      hypothesis="The measured final-method core and learnable baselines can optimize the common wideband power target; increasing RT budgets should stabilize geometry-derived predictions.",
      data_identity=data.identity,fold=FOLD,fit_rows=fit.tolist(),validation_rows=val.tolist(),rt_probe_rows=probes.tolist(),
      tiny_fit_rows=fit[:8].tolist(),selection="Deterministic farthest-position sampling inside the unchanged spatial roles, with no channel-dependent sample selection.",
      shared_donors=96,fit_reference_panels=192,validation_reference_panels=96,seed=SEED,
      core_crossfit="For every fit query, all donors in its own native 4m XY block are excluded. Validation uses only the selected96 fit donors.",
      health=dict(tiny_target_mean_cosine=.98,alternative_loss_reduction_fraction=.8,finite_gradients=True,active_parameter_updates=True,real_train_improvement=True,
                  validation_plateau="No mean-cosine improvement of0.0002 for30 epochs; maximum180 epochs. Tiny memorization is a diagnostic, never evidence of generalization."),
      rt=dict(probes=12,seeds=[SEED,SEED+1],rays=[4096,16384,65536,262144],depths=[2,4,6,8],
              thresholds=dict(mean_cosine=.995,p10_cosine=.98,p90_normalized_power_L1=.05,p90_cross_choice_loss_db=.5),
              zero_links="Both-zero links are reported separately and do not count as stability evidence.",
              scope="Numerical convergence within fixed specular/refraction material model; not material accuracy or omitted-scattering validation."),
      tuning=dict(kernel_lengths_m=[.5,1.,2.,4.],kernel_ridges=[.001,.01,.1],
                  mlp=[dict(levels=3,hidden=96,lr=.001),dict(levels=6,hidden=96,lr=.001)],
                  wrf=dict(gaussians_per_panel=[64,128],author_deformation=True,grid=[360,90],max_epochs=180,
                           health_stage="8 real fit positions, <=600 updates; then common96 fit/48 validation development"),
                  ours=dict(fusion="phase-invariant power loss on complex17-expert fusion; fit-block crossfitted features",
                            coordinate_bounds=[[.5,.25,0.],[1.,.5,0.]],gate_caps=[.3,.6],lr=[.0003,.001],
                            loss="Common normalized broadband-power cosine plus small residual regularization; no raw-H regression.",
                            delay_policy="No absolute delay supervision; angle-only complex writeback is primary, legacy joint projection is checked as an explicit control.")),
      non_goals=["Formal test scores","Northern data or model selection","Full five-fold three-seed benchmark","Proof of geometry causality","Absolute complex-H reconstruction"],
      compute=dict(cpu_threads=8,training_device_uuid=GPU,hard_stage_wall_minutes=45,small_scale_only=True))
    path=REPORT/"contract.json"
    if path.exists() and json.loads(path.read_text())!=contract:raise ValueError("Frozen development contract changed")
    if not path.exists():save_json(path,contract)
    return data,contract
class DevelopmentData:
    def __init__(self):
        self.data,self.contract=init();self.access=[]
        self.fit_ids=np.array(self.contract["fit_rows"]);self.val_ids=np.array(self.contract["validation_rows"])
    def fit(self):
        bank=self.data.read_fit(FOLD,self.fit_ids);self.access.extend(self.data.channel_reads);self.data.channel_reads.clear();return bank
    def reference(self,role):
        if role not in ("fit","validation"):raise ValueError("Formal TEST reference is forbidden")
        rows=self.fit_ids if role=="fit" else self.val_ids
        source=np.load(SOUTH/"channels.npy",mmap_mode="r",allow_pickle=False)
        h=np.array(source[rows],copy=True);self.access.append(dict(fold=FOLD,role=role,rows=rows.tolist(),purpose="training target" if role=="fit" else "development evaluator only"))
        return h
    def query(self,role):
        if role not in ("fit","validation"):raise ValueError("Formal TEST query is forbidden")
        return self.data.query(self.fit_ids if role=="fit" else self.val_ids)
    def save_access(self,name):save_json(REPORT/(name+"_access.json"),dict(access=self.access,test_read=False,north_read=False))
def device():
    if os.environ.get("CUDA_VISIBLE_DEVICES")!=GPU:raise RuntimeError("Training must be isolated to the permitted full UUID")
    if torch.cuda.device_count()!=1:raise RuntimeError("Unexpected CUDA visibility")
    info=torch.cuda.get_device_properties(0)
    if "H100" not in info.name:raise RuntimeError("Unexpected permitted GPU mapping")
    return torch.device("cuda:0")
def seed(value=SEED):
    torch.manual_seed(value);np.random.seed(value);torch.set_num_threads(8)
def metrics(power,truth):
    return aggregate_primary(primary_metrics(dict(power=np.asarray(power)),np.asarray(truth)))
def score(power,truth):
    return metrics(power,truth)["mean_direction_cosine"]
def snapshot():
    roots=[PATHS.code_root,PATHS.artifacts/"measured",PATHS.analysis,PATHS.root/"final_code",PATHS.root/"experiment_records"]
    records={}
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file() or path.is_symlink() or WORK in path.parents or REPORT in path.parents or Path(__file__).parent in path.parents:continue
            rel=str(path.relative_to(PATHS.root))
            if "__pycache__" in path.parts:continue
            st=path.stat();records[rel]=dict(size=st.st_size,mtime_ns=st.st_mtime_ns,sha256=sha256(path) if st.st_size<1048576 else None)
    return records
if __name__=="__main__":
    d,c=init()
    if (REPORT/"preservation_before.json").exists():raise RuntimeError("Development setup already exists; reuse the frozen contract without resetting preservation or ledger.")
    save_json(REPORT/"preservation_before.json",snapshot())
    ledger=PATHS.root/"experiment_records/EXPERIMENT_RECORD.md"
    (REPORT/"ledger_before.bin").write_bytes(ledger.read_bytes())
    with ledger.open("a") as f:
        f.write("\n\n## 840. 2026-10-24 South-blocker bounded fit/validation development\n\n### Contract\n- User authorizes small-scale south training, targeted final-method tuning, WRF-GS+ implementation/training health checks, and RT rays/depth convergence. No formal south-test scoring or north channel access.\n- Common fold0 position-selected96 fit and48 validation points; no split alteration. Original source, preparation and results preserved. Source and outputs in measured/hall_development and hall_south_development_v1.\n- Frozen contract: analysis/results/hall_south_development_v1/contract.json. Tiny-fit optimization is health only; validation selects development settings. Actual completion and failures will follow saved runs.\n")
    print(json.dumps(dict(fit=len(c["fit_rows"]),validation=len(c["validation_rows"]),rt_probes=c["rt_probe_rows"]),indent=2))
