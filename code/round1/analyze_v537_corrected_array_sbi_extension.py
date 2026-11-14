#!/usr/bin/env python3
"""Extend the v531 physical posterior with the corrected-array source."""

from __future__ import annotations

import json

import numpy as np

import analyze_v494_sionna_configuration_stacking as v494
import analyze_v531_expanded_sbi_configuration_library as v531
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 537
VERSION = "v537_corrected_array_sbi_extension"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
OOF_CACHE = CACHE_DIR / f"{VERSION}_oof.npz"
SOURCE_DIR = CACHE_DIR / "physical_sources"
NEW_TRAIN = SOURCE_DIR / "corrected_array_d10_sc015_seed42_train_joint_sh.npy"
OLD_REPORT = PATHS.version_report_dir(531) / "v531_expanded_sbi_configuration_library.json"
OLD_OOF = PATHS.cache_dir(531) / "v531_expanded_sbi_configuration_library_oof.npz"
SOURCE_NAME = "corrected_array_d10_sc015_seed42"

MIN_MEAN_GAIN = 0.0005
MIN_MEDIUM_GAIN = 0.0010
MAX_CLEAR_HEAVY_LOSS = 0.0003
MIN_FOLD_WINS = 4


def append_corrected_source(
    include_nonmetal: bool = False,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, list[str]]:
    truth, sources, quality, names = ORIGINAL_LOADER(include_nonmetal=include_nonmetal)
    if not NEW_TRAIN.is_file():
        raise FileNotFoundError(NEW_TRAIN)
    value = np.load(NEW_TRAIN, mmap_mode="r")
    if value.shape[0] != len(truth):
        raise ValueError(f"corrected source shape mismatch: {value.shape}")
    corrected = v531.clean_source(value.reshape(len(truth), -1), sources[0])
    sources.append(corrected)
    names.append(SOURCE_NAME)
    return truth, sources, quality, names


def add_incremental_gate() -> dict[str, object]:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    old_report = json.loads(OLD_REPORT.read_text(encoding="utf-8"))
    with np.load(OOF_CACHE) as new, np.load(OLD_OOF) as old:
        method = str(report["decision_gate"]["best_method"])
        old_method = str(old_report["decision_gate"]["best_method"])
        new_score = np.asarray(new[f"{method}_score"], dtype=np.float64)
        old_score = np.asarray(old[f"{old_method}_score"], dtype=np.float64)
        fold_id = np.asarray(new["fold_id"], dtype=np.int16)
        if not np.array_equal(fold_id, old["fold_id"]):
            raise ValueError("v531 and v537 fold IDs differ")
    geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    regimes = regime_of(geometry["train_blocked"]).astype("U8")
    gain = new_score - old_score
    fold_gains = [float(np.mean(gain[fold_id == fold])) for fold in range(5)]
    regime_gains = {
        regime: float(np.mean(gain[regimes == regime])) for regime in v494.REGIMES
    }
    mean_gain = float(np.mean(gain))
    fold_wins = int(sum(value > 0.0 for value in fold_gains))
    passed = bool(
        mean_gain >= MIN_MEAN_GAIN
        and regime_gains["medium"] >= MIN_MEDIUM_GAIN
        and regime_gains["clear"] >= -MAX_CLEAR_HEAVY_LOSS
        and regime_gains["heavy"] >= -MAX_CLEAR_HEAVY_LOSS
        and fold_wins >= MIN_FOLD_WINS
    )
    gate = {
        "pass": passed,
        "new_method": method,
        "old_v531_method": old_method,
        "mean_gain": mean_gain,
        "fold_gains": fold_gains,
        "fold_wins": fold_wins,
        "regime_gains": regime_gains,
        "requirements": {
            "mean_gain_min": MIN_MEAN_GAIN,
            "medium_gain_min": MIN_MEDIUM_GAIN,
            "clear_gain_min": -MAX_CLEAR_HEAVY_LOSS,
            "heavy_gain_min": -MAX_CLEAR_HEAVY_LOSS,
            "fold_wins_min": MIN_FOLD_WINS,
        },
    }
    report["incremental_gate_vs_v533_oof"] = gate
    report["primary_hypothesis"] = (
        "The v534/v535-corrected array source adds medium-regime joint-S-H information "
        "missing from the nine-source v531 posterior without degrading clear or heavy regimes."
    )
    report["status"] = "analysis_complete_incremental_gate_pass" if passed else "analysis_negative"
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return gate


ORIGINAL_LOADER = v531.load_expanded_sources


def main() -> None:
    for path in (NEW_TRAIN, OLD_REPORT, OLD_OOF):
        if not path.is_file():
            raise FileNotFoundError(path)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    v531.VERSION_NUMBER = VERSION_NUMBER
    v531.VERSION = VERSION
    v531.REPORT_DIR = REPORT_DIR
    v531.REPORT = REPORT
    v531.CACHE_DIR = CACHE_DIR
    v531.OOF_CACHE = OOF_CACHE
    v531.load_expanded_sources = append_corrected_source
    v531.main()
    gate = add_incremental_gate()
    print(json.dumps({"incremental_gate_vs_v533_oof": gate}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
