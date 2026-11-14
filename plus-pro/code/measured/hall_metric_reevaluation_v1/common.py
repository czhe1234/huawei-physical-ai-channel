"""Existing-prediction rescoring with frozen metrics and unchanged south rows."""
from __future__ import annotations
import json,re
from pathlib import Path
from project_paths import PATHS
from measured.hall_adapters.common import save_json,sha256
SOURCE=PATHS.code_root/"measured/hall_metric_reevaluation_v1"
REPORT=PATHS.analysis/"results/hall_south_metric_reevaluation_20261025"
WORK=PATHS.artifacts/"measured/hall_sw_metric_reevaluation_v1"
PROTOCOL=PATHS.analysis/"audits/hall_south_metric_protocol_20261025"
OLD_NEW=PATHS.analysis/"results/hall_south_structured_development_v2"
OLD_V1=PATHS.analysis/"results/hall_south_development_v1"
KEYS=("pas_cosine","relative_pdp_cosine","receive_energy_TV","receive_delay_joint_TV")
def snapshot():
    out={}
    for root in [PATHS.code_root,PATHS.analysis,PATHS.artifacts/"measured",PATHS.datasets,PATHS.root/"final_code"]:
      for p in root.rglob("*"):
        if not p.is_file() or p.is_symlink() or "__pycache__" in p.parts or any(x in p.parents for x in [SOURCE,REPORT,WORK]):continue
        st=p.stat();out[str(p.relative_to(PATHS.root))]=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns,sha256=sha256(p) if st.st_size<1048576 else None)
    return out
def init():
    from measured.hall_structured_v2.data import SouthDevelopment
    REPORT.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
    if (REPORT/"contract.json").exists():return
    old_protocol=json.loads((PROTOCOL/"source_index.json").read_text())
    for path,record in old_protocol["new_source"].items():
        assert sha256(PATHS.root/path)==record["sha256"],path
    reader=SouthDevelopment()
    save_json(REPORT/"preservation_before.json",snapshot())
    ledger=PATHS.root/"experiment_records/EXPERIMENT_RECORD.md";prefix=ledger.read_bytes();(REPORT/"ledger_before.bin").write_bytes(prefix)
    number=max(map(int,re.findall(r"^## (\d+)\.",prefix.decode(),re.M)))+1
    contract=dict(question="How do fixed existing south checkpoints and baseline outputs compare on frequency-resolved PAS, self-referenced relative PDP and receive energy/coupling?",
      scope="Reevaluate existing development predictions only; no retraining/reselection/RT/formal TEST/north.",
      data_identity=reader.data.identity,fit_rows=reader.fit_ids.tolist(),validation_rows=reader.validation_ids.tolist(),
      metric_protocol_sha256=sha256(PROTOCOL/"metric_protocol.json"),
      frozen_metric_source_sha256=sha256(PATHS.code_root/"measured/hall_metric_protocol_v1/metrics.py"),
      metrics=list(KEYS),combined_score=None,checkpoint_selection="Retain old selected epochs and settings; list all valid variants without promotion.",
      deterministic_regeneration=["nearest FIT channel","untrained uniform13-expert fusion with original formula"],
      energy_only_controls=["uniform32-element share","mean of96 FIT-normalized receive shares; no optimizer"],
      unavailable="Broadband beam powers do not define per-frequency PAS, relative PDP or physical receive energy; report NA.",
      invalidity="Frozen reference-only policy; failed predictions retained with declared worst scores.",
      statistics="48 positions in one validation spatial block, one seed, reused development data. No confirmatory significance or independent-sample CI.",
      ledger_item=number,new_training=False,test=False,north=False)
    save_json(REPORT/"contract.json",contract)
    with ledger.open("a") as f:f.write(f"\n\n## {number}. 2026-10-25 South existing predictions under new metrics\n\n### Contract\n- User authorizes rescoring existing south models/baselines on the prepared metric protocol. No retraining, new checkpoint selection, formal TEST or north data. Keep old scores and predictions.\n- Fixed96 FIT/48 validation, independent self-reference per panel, separate PAS/PDP/receive energy/joint scores, no new total score. Deterministic missing baselines may regenerate; uniform/FIT-average energy references are statistical controls only.\n- Contract: analysis/results/hall_south_metric_reevaluation_20261025/contract.json.\n")
if __name__=="__main__":init()
