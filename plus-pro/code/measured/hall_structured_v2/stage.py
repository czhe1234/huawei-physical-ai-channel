"""Create a fresh implementation stage while preserving all earlier results."""
import json,re
from pathlib import Path
from project_paths import PATHS
from measured.hall_adapters.common import save_json,sha256
from .config import REPORT,WORK,SOURCE,Config
def snapshot():
    records={}
    for root in [PATHS.code_root,PATHS.artifacts/"measured",PATHS.analysis,PATHS.root/"final_code",PATHS.datasets]:
      for p in root.rglob("*"):
        if not p.is_file() or p.is_symlink() or "__pycache__" in p.parts or any(x in p.parents for x in [REPORT,WORK,SOURCE]):continue
        st=p.stat();records[str(p.relative_to(PATHS.root))]=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns,sha256=sha256(p) if st.st_size<1048576 else None)
    return records
def init_stage():
    REPORT.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
    if (REPORT/"preservation_before.json").exists():raise RuntimeError("Stage already initialized; preservation cannot be reset")
    save_json(REPORT/"preservation_before.json",snapshot())
    ledger=PATHS.root/"experiment_records/EXPERIMENT_RECORD.md";prefix=ledger.read_bytes()
    (REPORT/"ledger_before.bin").write_bytes(prefix)
    number=max(int(n) for n in re.findall(r"^## (\d+)\.",prefix.decode(),flags=re.M))+1
    contract=dict(scope="Implementation plus synthetic correctness and bounded FIT-only integration. No optimizer updates, real scoring, new RT, TEST or north channels.",
       primary_question="Does the single-TX implementation retain frequency-resolved measurement information, path energy/lifecycle, relative delay, real-array energy organization and joint complex synthesis with final-H structural losses?",
       config=Config().to_dict(),success=["invariant observable mathematics","joint projection and relative-delay gradients","all restored heads connected","source/donor isolation","training entry prepared but unexecuted","old data and outputs unchanged"],
       limitations=["Original multi-UE/polarization head cannot be reproduced from single TX","Absolute phase/time and inter-array coherence lack reference","Existing geometry cache is provisional and lacks scatter/variance evidence"],
       modules=["frequency-resolved measurement expert fusion and donor correction","path reliability and lifecycle","sequence/set encoding with invariant context","relative angular-delay endpoints","real receive-array energy correction","multi-domain trust gates and spectral calibration","joint full-frequency synthesis","final-H structural supervision and validation selection"])
    save_json(REPORT/"contract.json",contract)
    with ledger.open("a") as f:f.write(f"\n\n## {number}. 2026-10-24 South structured adaptation implementation\n\n### Contract\n- User asks to carry competition advantages into the measured algorithm. Implement a new version and correctness checks; no new training/scoring/RT or TEST/north usage. Preserve all prior versions.\n- Restore clock-safe frequency, relative-delay, receive-array energy and joint-generation capabilities plus final-H structural targets. Physical single-TX and missing absolute references remain explicit.\n- Contract: analysis/results/hall_south_structured_v2/contract.json. Completion follows actual checks.\n")
    print("INITIALIZED",number,len(contract["modules"]),flush=True)
if __name__=="__main__":init_stage()
