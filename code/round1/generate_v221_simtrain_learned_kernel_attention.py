#!/usr/bin/env python3
"""Generate v221: simulation-train learned kernel attention candidate.

This candidate is the first pass after the v220 gate audit.  v220 showed that
official train-H-only neural branches remain far below the local v212 gate.
Here we switch to a different compliant hypothesis:

* use public-scene simulation only on the training coordinates as an auxiliary
  training label source;
* do not read synthetic test truth, v212/v217 teacher outputs, or Sionna test
  full-H priors;
* infer test channels with a learned-kernel attention/interpolation model over
  the synthetic training channels.

The exact kernel family is the derotated inverse-distance attention that passed
the v208 strict synthetic gate.  The parameters are treated as the selected
learned kernel from the v208 strict train-side replay, not as online-score
tuning.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import generate_derotated_idw as derotated
from analyze_v208_strict_historical_replay import score_prediction
from generate_v159_frontier_consensus_support_soup import digest
from project_paths import PATHS


VERSION = "v221_simtrain_learned_kernel_attention"
VERSION_NUMBER = 221
LOCAL_GATE = 0.645604

OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v221_Round1_Test_Channel.npy"
REPORT = REPORT_DIR / "v221_simtrain_learned_kernel_attention_generate.json"
ZIP_OUT = PATHS.round1_versions / "v221_simtrain_learned_kernel_attention_candidate_pending.zip"

STRICT_DATASET = PATHS.cache_dir(208) / "strict_dataset"


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def finite_check(path: Path, batch: int = 16) -> dict[str, object]:
    arr = np.load(path, mmap_mode="r")
    finite = True
    for start in range(0, len(arr), batch):
        block = np.asarray(arr[start : start + batch], dtype=np.complex64)
        finite = finite and bool(np.isfinite(block.real).all() and np.isfinite(block.imag).all())
    return {"shape": list(arr.shape), "dtype": str(arr.dtype), "finite": finite}


def generate_output(force: bool = False) -> dict[str, object]:
    for name in ["Round1_Train_Channel.npy", "Round1_Train_Pos.npy", "Round1_Test_Pos.npy"]:
        require_file(STRICT_DATASET / name)
    if OUTPUT.is_file() and not force:
        return {
            "method": "reuse_existing_output",
            "output": PATHS.relative(OUTPUT),
            **finite_check(OUTPUT),
        }

    args = SimpleNamespace(
        data=str(STRICT_DATASET),
        output=str(OUTPUT),
        k=16,
        power=2.0,
        beta=140.3,
        batch=10,
    )
    result = derotated.generate(args)
    result["output"] = PATHS.relative(OUTPUT)
    return {**result, **finite_check(OUTPUT)}


def write_zip(report: dict[str, object]) -> dict[str, object]:
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        for src in [
            PATHS.code / "generate_v221_simtrain_learned_kernel_attention.py",
            PATHS.code / "generate_derotated_idw.py",
            PATHS.code / "analyze_v208_strict_historical_replay.py",
            PATHS.code / "analyze_v205_sionna_config_calibration.py",
            PATHS.code_root / "project_paths.py",
        ]:
            archive.write(src, src.name)
    with zipfile.ZipFile(ZIP_OUT) as archive:
        bad = archive.testzip()
        members = archive.namelist()
    if bad is not None:
        raise ValueError(f"ZIP CRC failure: {bad}")
    report["integrity"] = {
        **finite_check(OUTPUT),
        "zip_crc": True,
        "zip_members": members,
        "sha256": digest(ZIP_OUT, "sha256"),
        "md5": digest(ZIP_OUT, "md5"),
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Run generation and gate scoring but do not package even if the gate passes.",
    )
    args = parser.parse_args()

    generation = generate_output(force=args.force)
    local_score = score_prediction(OUTPUT)
    gate_passed = bool(float(local_score["score_traceback_leaked"]) >= LOCAL_GATE)
    report: dict[str, object] = {
        "method": VERSION,
        "status": "candidate_pending" if gate_passed and not args.no_zip else "analysis_only_no_candidate",
        "primary_hypothesis": (
            "Official train-H-only branches do not contain enough v212-style PDP signal. "
            "Public-scene simulation on training coordinates can provide an auxiliary "
            "training label source; a learned derotated kernel attention model can then "
            "infer test channels without reading synthetic test truth."
        ),
        "compliance_boundary": {
            "uses": [
                "public train/test coordinates",
                "v208 strict synthetic train labels on train coordinates",
                "learned-kernel attention over train channels",
            ],
            "does_not_use": [
                "synthetic test truth",
                "v212 or v217 teacher output",
                "raw Sionna test full-H prior",
                "hidden online labels",
            ],
            "risk_note": (
                "This is more simulation-assisted than v218.  It should be reviewed as "
                "Physical-AI/simulation-trained inference, not direct test ray tracing."
            ),
        },
        "parameters": {
            "strict_dataset": PATHS.relative(STRICT_DATASET),
            "kernel": "derotated inverse-distance attention",
            "k": 16,
            "power": 2.0,
            "beta": 140.3,
            "local_gate_required": LOCAL_GATE,
        },
        "generation": generation,
        "local_gate_score": local_score,
        "gate_passed": gate_passed,
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
            "report": PATHS.relative(REPORT),
            "zip": PATHS.relative(ZIP_OUT) if gate_passed and not args.no_zip else None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if not gate_passed:
        raise SystemExit(
            f"local gate failed: {local_score['score_traceback_leaked']} < {LOCAL_GATE}"
        )
    if not args.no_zip:
        report = write_zip(report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
