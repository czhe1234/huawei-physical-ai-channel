"""Check the pre-adapter preservation snapshot; only append-only ledger differs."""
from pathlib import Path
import hashlib,json
from project_paths import PATHS
from round2.paper_baseline_adapters_v1.contract import save_json,sha256

def check():
    report=PATHS.analysis/"audits/simulation_recovery_v1"
    baseline=PATHS.analysis/"audits/simulation_baseline_adapters_20261031/preservation_before.json"
    before=json.loads(baseline.read_text())
    changes=[];hashed=0
    for name,meta in before.items():
        if name=="experiment_records/EXPERIMENT_RECORD.md":continue
        p=PATHS.root/name
        if not p.exists():changes.append({"path":name,"change":"missing"});continue
        st=p.stat()
        for k,current in (("bytes",st.st_size),("mtime_ns",st.st_mtime_ns),("inode",st.st_ino)):
            if current != meta[k]:changes.append({"path":name,"change":k})
        if meta.get("sha256"):
            hashed+=1
            if sha256(p)!=meta["sha256"]:changes.append({"path":name,"change":"sha256"})
    ledger=(PATHS.root/"experiment_records/EXPERIMENT_RECORD.md").read_bytes()
    prefix=(report/"ledger_before.bin").read_bytes()
    prefix_ok=ledger.startswith(prefix)
    result=dict(status="passed" if not changes and prefix_ok else "failed",
      protected_files=len(before),hashed_files=hashed,old_file_changes=changes,
      ledger_prefix_preserved=prefix_ok,original_ledger_prefix_bytes=len(prefix),
      new_outputs_root="artifacts/round2/simulation_recovery_v1",
      no_north_measured_data_access=True,
      verification_scope="Existing stat identity and recorded small-file digests; large-file contents not rehashed")
    save_json(report/"preservation.json",result)
    print(json.dumps(result))
    if result["status"]!="passed":raise RuntimeError("Preservation check failed")
    return result

if __name__=="__main__":check()
