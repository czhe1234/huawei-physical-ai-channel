#!/usr/bin/env python3
"""v258: transfer only the v257 neural transport delta onto strict v239.

v257 improved official holdouts, but its local deployment replaced the entire
v239 spectrum with a domain-mismatched official KNN target. This experiment
keeps the v239 S-H/PDP backbone and applies only the already selected neural
transport residual (transported target minus raw official KNN target).
"""

from __future__ import annotations

import json

import numpy as np

import analyze_v208_strict_historical_replay as v208
import analyze_v247_delaunay_spectral_field as v247
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 258
VERSION = "v258_transport_delta_on_v239"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v258_Round1_Test_Channel.npy"
TARGETS = OUT_DIR / "v258_transport_delta_targets.npz"
ACTIONS = OUT_DIR / "v258_transport_delta_actions.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"
BASE = PATHS.cache_dir(239) / "v239_path_on_v237_Round1_Test_Channel.npy"
V257_TARGETS = PATHS.cache_dir(257) / "v257_neural_transport_targets.npz"
V257_REPORT = PATHS.version_report_dir(257) / "v257_neural_pairwise_spectral_transport.json"

LOCAL_GATE = 0.645604


def normalize(x: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(x, dtype=np.float32), 0.0)
    total = out.reshape(len(out), -1).sum(axis=1)
    valid = total > 1e-20
    out[valid] /= total[valid].reshape(
        (int(np.sum(valid)),) + (1,) * (out.ndim - 1)
    )
    out[~valid] = 0.0
    return out.astype(np.float32)


def main() -> None:
    for path in (BASE, V257_TARGETS, V257_REPORT):
        if not path.is_file():
            raise FileNotFoundError(path)
    v257_report = json.loads(V257_REPORT.read_text(encoding="utf-8"))
    if not v257_report["selection"]["official_selected"]:
        raise RuntimeError("v257 official selection did not pass")
    base = np.load(BASE, mmap_mode="r")
    current_sh, current_d = v247.extract_current(np.asarray(base, dtype=np.complex64))
    transported = np.load(V257_TARGETS)
    delta_sh = transported["sh"].astype(np.float32) - transported["raw_sh"].astype(np.float32)
    delta_d = transported["d"].astype(np.float32) - transported["raw_d"].astype(np.float32)
    target_sh = normalize(current_sh + delta_sh)
    target_d = normalize(current_d + delta_d)
    np.savez_compressed(
        TARGETS,
        sh=target_sh,
        d=target_d,
        current_sh=current_sh,
        current_d=current_d,
        delta_sh=delta_sh,
        delta_d=delta_d,
        source_method=np.asarray(v257_report["selection"]["best_name"]),
    )
    actions = v247.project(target_sh, target_d, OUTPUT)
    np.savez_compressed(ACTIONS, **actions)
    score = v208.score_prediction(OUTPUT)
    report = {
        "method": VERSION,
        "status": (
            "gate_passed_needs_packaging"
            if float(score["score_traceback_leaked"]) > LOCAL_GATE
            else "analysis_only_no_candidate"
        ),
        "primary_hypothesis": (
            "The v257 neural shift signal is valid, but must be transferred as a small "
            "mass-moving residual on the strict v239 spectrum rather than as an absolute "
            "official KNN target replacement."
        ),
        "compliance_boundary": {
            "uses": [
                "v257 PyTorch shift network output trained only on official train pairs",
                "official train/test coordinates and point-cloud features inherited from v257",
                "strict v239 official-data carrier",
            ],
            "does_not_use": [
                "synthetic or Sionna channel labels",
                "v212/v217 teacher output",
                "local replay truth for target construction or parameter selection",
                "hidden labels",
                "raw RT H",
            ],
            "local_truth_usage": "one final diagnostic score after the official-selected residual was fixed",
        },
        "parameters": {
            "source_v257_config": v257_report["selection"]["best_name"],
            "residual_scale": 1.0,
            "transfer": "normalized max(v239 marginal + v257 transported-minus-raw residual, 0)",
            "additional_local_tuning": False,
        },
        "delta_diagnostics": {
            "sh_l1": summarize(np.abs(delta_sh).sum(axis=(1, 2))),
            "d_l1": summarize(np.abs(delta_d).sum(axis=1)),
            "target_sh_l1_from_v239": summarize(np.abs(target_sh - current_sh).sum(axis=(1, 2))),
            "target_d_l1_from_v239": summarize(np.abs(target_d - current_d).sum(axis=1)),
        },
        "projection_diagnostics": {key: summarize(value) for key, value in actions.items()},
        "local_gate": {
            "required": LOCAL_GATE,
            "score": score,
            "passed": bool(float(score["score_traceback_leaked"]) > LOCAL_GATE),
            "gap": float(score["score_traceback_leaked"]) - LOCAL_GATE,
        },
        "outputs": {
            "prediction": PATHS.relative(OUTPUT),
            "targets": PATHS.relative(TARGETS),
            "actions": PATHS.relative(ACTIONS),
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
