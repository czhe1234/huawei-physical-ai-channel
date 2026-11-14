#!/usr/bin/env python3
"""v269: pure official-trained analytic scatterer renderer diagnostic."""

from __future__ import annotations

import json

import numpy as np

import analyze_v208_strict_historical_replay as v208
import analyze_v266_learned_scatterer_joint_projection as v266
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 269
VERSION = "v269_pure_learned_scatterer_renderer"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"
OUTPUT = OUT_DIR / "v269_Round1_Test_Channel.npy"
ACTION = OUT_DIR / "v269_joint_projection_actions.npz"

V266_TARGETS = PATHS.cache_dir(266) / "v266_joint_targets.npz"
BASE = PATHS.cache_dir(239) / "v239_path_on_v237_Round1_Test_Channel.npy"
LOCAL_GATE = 0.645604


def main() -> None:
    for path in (V266_TARGETS, BASE):
        if not path.is_file():
            raise FileNotFoundError(path)
    targets = np.load(V266_TARGETS)
    target_sh = v266.normalize(targets["predicted_sh"].astype(np.float32))
    target_d = v266.normalize(targets["predicted_d"].astype(np.float32))
    actions = v266.alternating_project(BASE, OUTPUT, target_sh, target_d)
    np.savez_compressed(ACTION, **actions)
    local_score = v208.score_prediction(OUTPUT)
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "A pure analytic wall-scatterer target with official-H-learned path weights is "
            "closer to the Sionna-like local gate than direct official neighbor spectra."
        ),
        "compliance_boundary": {
            "uses": [
                "official-H-trained v264 patch Transformer",
                "official point-cloud analytic wall reflection delay/angle renderer",
                "train-only patch angle calibration",
                "strict v239 carrier",
            ],
            "does_not_use": [
                "Sionna or synthetic labels for training or generation",
                "teacher output",
                "local replay truth for parameter selection",
                "hidden labels",
                "raw RT H",
            ],
            "note": "Fixed pure-renderer diagnostic; no local-tuned blend strength.",
        },
        "parameters": {"alpha_sh": 1.0, "alpha_d": 1.0, "projection_steps": 4},
        "projection_diagnostics": {key: summarize(value) for key, value in actions.items()},
        "local_gate": {
            "required": LOCAL_GATE,
            "score": local_score,
            "passed": bool(float(local_score["score_traceback_leaked"]) > LOCAL_GATE),
            "gap": float(local_score["score_traceback_leaked"]) - LOCAL_GATE,
        },
        "outputs": {
            "prediction": PATHS.relative(OUTPUT),
            "actions": PATHS.relative(ACTION),
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
