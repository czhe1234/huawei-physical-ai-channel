"""Create a fresh metric-feasibility stage while preserving all previous work."""
import json,re
from project_paths import PATHS
from measured.hall_adapters.common import save_json,sha256
SOURCE=PATHS.code_root/"measured/hall_metric_protocol_v1"
REPORT=PATHS.analysis/"audits/hall_south_metric_protocol_20261025"
def snapshot():
    records={}
    for root in [PATHS.code_root,PATHS.analysis,PATHS.artifacts/"measured",PATHS.datasets,PATHS.root/"final_code"]:
      for p in root.rglob("*"):
        if not p.is_file() or p.is_symlink() or "__pycache__" in p.parts or any(x in p.parents for x in [SOURCE,REPORT]):continue
        st=p.stat();records[str(p.relative_to(PATHS.root))]=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns,sha256=sha256(p) if st.st_size<1048576 else None)
    return records
def init():
    REPORT.mkdir(parents=True,exist_ok=True)
    if (REPORT/"contract.json").exists():return
    save_json(REPORT/"preservation_before.json",snapshot())
    ledger=PATHS.root/"experiment_records/EXPERIMENT_RECORD.md";old=ledger.read_bytes()
    (REPORT/"ledger_before.bin").write_bytes(old)
    number=max(map(int,re.findall(r"^## (\d+)\.",old.decode(),re.M)))+1
    save_json(REPORT/"contract.json",dict(scope="Candidate metric implementation and numerical feasibility only.",
      user_request="Advance the discussion; retain preceding no-training/no-rescoring boundary.",
      question="Are competition-matched PAS, independent common-panel relative PDP and receive-energy/coupling metrics well-defined and numerically stable on declared south FIT?",
      acceptance=["Official operator agreement after physical-layout adaptation",
        "Common fractional delay/phase invariance without pairwise alignment",
        "Sensitivity to relative inter-channel delay and energy errors",
        "Zero/unstable-reference policy tested","FIT-only conditioning checks",
        "Old sources/data/results and ledger prefix preserved"],
      fit_only=True,trained_models_loaded=False,validation_channels=False,test=False,north=False,new_training=False,new_algorithm_scores=False,
      ledger_item=number))
    with ledger.open("a") as f:f.write(f"\n\n## {number}. 2026-10-25 South metric protocol feasibility\n\n### Contract\n- Advance metric preparation following the discussion. Implement and check candidate definitions; no new training, model rescoring/ranking, validation/TEST or north channel access.\n- Check competition PAS/PDP correspondence, one self-derived common reference per32-element panel and receive energy/delay coupling. FIT-only data conditioning is not performance validation.\n- Source: code/measured/hall_metric_protocol_v1; report: analysis/audits/hall_south_metric_protocol_20261025.\n")
if __name__=="__main__":init()
