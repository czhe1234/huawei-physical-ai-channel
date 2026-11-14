#!/usr/bin/env python3
"""Derive the fixed seed42 medium-only deployment gate from the v537 OOF run."""

from __future__ import annotations

import json

import numpy as np

import analyze_v537_corrected_array_sbi_extension as v537
import analyze_v538_corrected_array_medium_only as v538
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 540
VERSION = "v540_seed42_corrected_array_medium_only"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
SOURCE_NAME = v537.SOURCE_NAME
SEEDS = (42,)
SOURCE_PATHS = (v537.NEW_TRAIN,)
SOURCE_DONE = (
    PATHS.cache_dir(537)
    / "physical_sources/corrected_array_d10_sc015_seed42_train_done.npy",
)


load_corrected_sources = v537.append_corrected_source


def main() -> None:
    for path in (v537.REPORT, v537.OOF_CACHE, v537.OLD_REPORT, v537.OLD_OOF):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_report = json.loads(v537.REPORT.read_text(encoding="utf-8"))
    old_report = json.loads(v537.OLD_REPORT.read_text(encoding="utf-8"))
    with np.load(v537.OOF_CACHE) as new, np.load(v537.OLD_OOF) as old:
        method = str(source_report["decision_gate"]["best_method"])
        old_method = str(old_report["decision_gate"]["best_method"])
        gain = np.asarray(new[f"{method}_score"], dtype=np.float64) - np.asarray(
            old[f"{old_method}_score"], dtype=np.float64
        )
        fold_id = np.asarray(new["fold_id"], dtype=np.int16)
    regimes = regime_of(
        np.load(PATHS.shared_features / "height_geometry_cache.npz")["train_blocked"]
    ).astype("U8")
    medium = regimes == "medium"
    deployed_gain = np.where(medium, gain, 0.0)
    medium_fold_gains = [
        float(np.mean(gain[medium & (fold_id == fold)])) for fold in range(5)
    ]
    all_fold_gains = [
        float(np.mean(deployed_gain[fold_id == fold])) for fold in range(5)
    ]
    medium_gain = float(np.mean(gain[medium]))
    all_gain = float(np.mean(deployed_gain))
    medium_fold_wins = int(sum(value > 0 for value in medium_fold_gains))
    passed = bool(
        all_gain >= v538.MIN_ALL_GAIN
        and medium_gain >= v538.MIN_MEDIUM_GAIN
        and medium_fold_wins >= v538.MIN_MEDIUM_FOLD_WINS
    )
    gate = {
        "pass": passed,
        "new_method": method,
        "old_v531_method": old_method,
        "deployment_regime": "medium",
        "seed": 42,
        "all_gain": all_gain,
        "medium_gain": medium_gain,
        "all_fold_gains": all_fold_gains,
        "medium_fold_gains": medium_fold_gains,
        "medium_fold_wins": medium_fold_wins,
        "nonmedium_gain_by_construction": 0.0,
        "requirements": {
            "all_gain_min": v538.MIN_ALL_GAIN,
            "medium_gain_min": v538.MIN_MEDIUM_GAIN,
            "medium_fold_wins_min": v538.MIN_MEDIUM_FOLD_WINS,
        },
    }
    report = source_report
    report["method"] = VERSION
    report["status"] = "analysis_complete_gate_pass" if passed else "analysis_negative"
    report["primary_hypothesis"] = (
        "The preselected default PathSolver seed42 corrected-array source improves medium "
        "joint-S-H, while exact retention of v533 outside medium avoids the light regression."
    )
    report["medium_only_incremental_gate_vs_v533_oof"] = gate
    report["deployment_rule"] = {
        "medium": method,
        "clear_light_heavy": "retain_v533",
    }
    report["derived_from"] = {
        "analysis": PATHS.relative(v537.REPORT),
        "oof": PATHS.relative(v537.OOF_CACHE),
        "no_new_fit": True,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"medium_only_incremental_gate_vs_v533_oof": gate}, ensure_ascii=False))


if __name__ == "__main__":
    main()
