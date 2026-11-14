#!/usr/bin/env python3
"""Isolate corrected-source value while keeping the original v531 quality gate."""

from __future__ import annotations

import json

import numpy as np

import analyze_v494_sionna_configuration_stacking as v494
import analyze_v531_expanded_sbi_configuration_library as v531
import analyze_v541_corrected_full_library_sbi as v541
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 542
VERSION = "v542_old_quality_corrected_union_stack"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
OLD_OOF = PATHS.cache_dir(531) / "v531_expanded_sbi_configuration_library_oof.npz"

MIN_MEAN_GAIN = 0.001
MIN_FOLD_WINS = 4
MAX_REGIME_LOSS = 0.0005


def main() -> None:
    for path in (OLD_OOF, v494.OFFICIAL, v494.POSITIONS):
        if not path.is_file():
            raise FileNotFoundError(path)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    old = v531.load_expanded_sources()
    corrected = v541.load_corrected_sources()
    if not np.allclose(old[0], corrected[0]):
        raise ValueError("old and corrected truth arrays differ")
    union = (
        old[0],
        list(old[1]) + list(corrected[1]),
        old[2],
        [f"old_{name}" for name in old[3]] + list(corrected[3]),
    )

    v541.CACHE_DIR = CACHE_DIR
    v541.REPORT_DIR = REPORT_DIR
    variant_report, variant_oof = v541.run_v531_variant(
        VERSION,
        lambda include_nonmetal=False: union,
        1024,
    )
    with np.load(OLD_OOF) as old_cache, np.load(variant_oof) as new_cache:
        old_score = np.asarray(
            old_cache[f"{v531.HYBRID_METHOD}_score"], dtype=np.float64
        )
        new_score = np.asarray(new_cache["expanded_stack_score"], dtype=np.float64)
        fold_id = np.asarray(old_cache["fold_id"], dtype=np.int16)
    geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    regimes = regime_of(geometry["train_blocked"]).astype("U8")
    gain = new_score - old_score
    fold_gains = [float(np.mean(gain[fold_id == fold])) for fold in range(5)]
    regime_gains = {
        regime: float(np.mean(gain[regimes == regime])) for regime in v494.REGIMES
    }
    passed = bool(
        np.mean(gain) >= MIN_MEAN_GAIN
        and sum(value > 0.0 for value in fold_gains) >= MIN_FOLD_WINS
        and min(regime_gains.values()) >= -MAX_REGIME_LOSS
    )
    report = {
        "method": VERSION,
        "status": "analysis_passed_deployment_gate" if passed else "analysis_negative",
        "primary_hypothesis": (
            "The v541 union-source gain was partly suppressed by replacing the original v531 "
            "source-confidence field with max(old, corrected). Keeping the online-validated old "
            "quality gate isolates the new 4-wavelength source information, while deterministic "
            "stacking avoids posterior smoothing across incompatible array modes."
        ),
        "fixed_decision": {
            "quality": "original v531 quality",
            "source_library": "old nine plus corrected nine",
            "method": "expanded deterministic convex stack",
            "sbi_draws_only_for_shared_runner": 1024,
            "test_sionna_used": False,
            "online_feedback_used_for_parameters": False,
        },
        "score": {
            "old_v531_mean": float(np.mean(old_score)),
            "new_mean": float(np.mean(new_score)),
            "mean_gain": float(np.mean(gain)),
            "fold_gains": fold_gains,
            "fold_wins": int(sum(value > 0.0 for value in fold_gains)),
            "regime_gains": regime_gains,
        },
        "gate": {
            "pass": passed,
            "mean_gain_min": MIN_MEAN_GAIN,
            "fold_wins_min": MIN_FOLD_WINS,
            "each_regime_gain_min": -MAX_REGIME_LOSS,
        },
        "shared_variant_decision": variant_report["decision_gate"],
        "outputs": {
            "oof": PATHS.relative(variant_oof),
            "report": PATHS.relative(REPORT),
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
