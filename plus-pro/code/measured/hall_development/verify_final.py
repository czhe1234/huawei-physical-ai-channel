"""Final saved-checkpoint inference, literal DFT and isolation verification."""
from __future__ import annotations
import json,time,traceback,runpy
import numpy as np
import torch
from .common import *
from .method import DevelopedV16
from .wrf_batched import BatchedWRF
from measured.hall_adapters.metrics import codebook
def run():
    start=time.monotonic();seed();dev=DevelopmentData();bank=dev.fit();reference=dev.reference("validation");truth=directional_power(reference);d=device()
    with np.load(WORK/"geometry_support.npz") as z:support={k:z[k][len(bank.row_ids):] for k in ["coordinate","mass","mask","features","sequence","interaction","depth","identity"]}
    records=[]
    for name,base_name in [("ours_config0_v2","foundation.npy"),("ours_config1_v2","foundation.npy"),("ours_phase_safe","foundation_phase_safe.npy")]:
        checkpoint=torch.load(WORK/"models"/name/"best.pt",map_location=d,weights_only=False);cfg=checkpoint["config"]
        model=DevelopedV16(cfg["bounds"],cfg["gate"]).to(d);model.load_state_dict(checkpoint["state_dict"]);model.eval()
        base=np.load(WORK/base_name,mmap_mode="r")[len(bank.row_ids):];values=[];returned=[]
        for first in range(0,len(dev.val_ids),4):
            rows=dev.val_ids[first:first+4];q=dev.data.query(rows)
            bp=dict(channel=np.array(base[first:first+4]),method="fit_donor_foundation",source_ids=bank.row_ids,row_ids=rows)
            sp={k:v[first:first+4] for k,v in support.items()};out=model.predict(q,bp,sp)
            values.append(out["channel"]);returned.extend(out["row_ids"].tolist())
        np.testing.assert_array_equal(returned,dev.val_ids)
        h=np.concatenate(values)
        # Literal W^H h, independent of FFT helper and torch optimized objective.
        literal=np.einsum("ab,npaf->npbf",codebook().conj(),h.reshape(len(h),2,32,1024))
        p=(np.abs(literal)**2).mean(-1);pn=normalized_power(p)
        with np.load(WORK/"models"/name/"best_power.npz") as z:
            pe=float(np.max(np.abs(pn-normalized_power(z["validation"]))))
        with np.load(WORK/"models"/name/"best_channel.npz") as z:
            he=float(np.max(np.abs(h-z["validation"]))/max(np.max(np.abs(z["validation"])),1e-20))
        if pe>3e-5 or he>1e-4:raise ValueError("Saved core inference mismatch")
        records.append(dict(name=name,checkpoint_inference_relative_channel_error=he,literal_DFT_power_max_error=pe,
             validation_recomputed=metrics(p,truth),position_only_predictor=True))
        del model;torch.cuda.empty_cache()
    for count in [64,128]:
        name="wrf_g%d"%count;m=BatchedWRF(bank.position,bank.spec,count).to(d)
        m.load_state_dict(torch.load(WORK/"models"/name/"best.pt",map_location=d,weights_only=False)["state_dict"])
        values=[]
        for first in range(0,len(dev.val_ids),4):values.append(m.predict(dev.data.query(dev.val_ids[first:first+4]))["power"])
        p=np.concatenate(values)
        with np.load(WORK/"models"/name/"best_power.npz") as z:pe=float(np.max(np.abs(p-z["validation"])))
        if pe>1e-5:raise ValueError("Saved WRF inference mismatch")
        records.append(dict(name=name,prediction_error=pe,validation_recomputed=metrics(p,truth),position_only_predictor=True))
        del m;torch.cuda.empty_cache()
    for role in ["test","north"]:
        try:dev.reference(role)
        except ValueError:pass
        else:raise ValueError("Forbidden role accepted")
    before=json.loads((REPORT/"preservation_before.json").read_text());after=snapshot();changed=[];missing=[]
    ledger_name="experiment_records/EXPERIMENT_RECORD.md"
    for name,identity in before.items():
        if name==ledger_name:continue
        if name not in after:missing.append(name)
        elif identity!=after[name]:changed.append(name)
    ledger=(PATHS.root/ledger_name).read_bytes();prefix=(REPORT/"ledger_before.bin").read_bytes()
    if changed or missing or not ledger.startswith(prefix):raise ValueError("Preservation failure:"+str((changed,missing)))
    save_json(REPORT/"preservation_after.json",dict(checked_files=len(before)-1,missing=missing,changed=changed,ledger_prefix_unchanged=True,
        raw_writes_performed=False,old_sources_data_results_preserved=True,whole_large_raw_files_rehashed=False))
    path_module=runpy.run_path(str(PATHS.code_root/"tools/check_hardcoded_paths.py"));bad=[]
    for path in Path(__file__).parent.glob("*.py"):
        compile(path.read_text(),str(path),"exec")
        for i,line in enumerate(path.read_text().splitlines(),1):
          for label,pattern in path_module["PATTERNS"]:
            if pattern.search(line):bad.append(dict(file=str(path.relative_to(PATHS.root)),line=i,issue=label))
    if bad:raise ValueError("New maintained-source forbidden paths:"+str(bad))
    unit=(REPORT/"unit_checks_final.log").read_text()
    if "Ran 6 tests" not in unit or "\nOK\n" not in unit:raise ValueError("Unit test completion unverified")
    access_files=sorted(REPORT.glob("*_access.json"));access=[]
    for path in access_files:
        payload=json.loads(path.read_text())
        for row in payload["access"]:
            if row["role"] not in ["fit","validation"]:raise ValueError("Forbidden channel role")
            expected=set(dev.fit_ids if row["role"]=="fit" else dev.val_ids)
            if not set(row["rows"]).issubset(expected):raise ValueError("Unexpected access IDs")
            access.append(dict(file=path.name,**row))
    save_json(REPORT/"acceptance.json",dict(passed=True,unit_checks=6,original_correctness_checks_reused=30,
        independent_trained_WRF_reference_parity=json.loads((REPORT/"trained_wrf_parity.json").read_text())["passed"],
        checkpoint_inference=records,new_maintained_source_path_check_passed=True,new_sources_compiled=True,
        south_test_reference_rejected=True,north_reference_rejected=True,saved_access_logs_checked=len(access_files),
        model_label_access_roles=["fit","validation"],north_training_or_scoring=False,formal_test_score_computed=False,
        seconds=time.monotonic()-start,verification_peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
        source_sha256={str(p.relative_to(PATHS.root)):sha256(p) for p in Path(__file__).parent.glob("*.py")}))
    dev.save_access("final_verification");print("ACCEPTED",len(records),len(before),time.monotonic()-start,flush=True)
if __name__=="__main__":
    try:run()
    except BaseException:
        save_json(REPORT/"final_verification_failure.json",dict(traceback=traceback.format_exc()));raise
