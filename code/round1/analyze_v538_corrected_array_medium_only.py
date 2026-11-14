#!/usr/bin/env python3
"""Validate a three-seed corrected-array source only in the medium regime."""

from __future__ import annotations

import json

import numpy as np

import analyze_v494_sionna_configuration_stacking as v494
import analyze_v531_expanded_sbi_configuration_library as v531
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 538
VERSION = "v538_corrected_array_medium_only"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
OOF_CACHE = CACHE_DIR / f"{VERSION}_oof.npz"
SOURCE_DIR = PATHS.cache_dir(537) / "physical_sources"
SEEDS = (42, 206, 420)
SOURCE_PATHS = tuple(
    SOURCE_DIR / f"corrected_array_d10_sc015_seed{seed}_train_joint_sh.npy"
    for seed in SEEDS
)
SOURCE_DONE = tuple(
    SOURCE_DIR / f"corrected_array_d10_sc015_seed{seed}_train_done.npy"
    for seed in SEEDS
)
SOURCE_NAME = "corrected_array_d10_sc015_seed_marginal_3"
OLD_REPORT = PATHS.version_report_dir(531) / "v531_expanded_sbi_configuration_library.json"
OLD_OOF = PATHS.cache_dir(531) / "v531_expanded_sbi_configuration_library_oof.npz"

MIN_ALL_GAIN = 0.00015
MIN_MEDIUM_GAIN = 0.001
MIN_MEDIUM_FOLD_WINS = 4
REQUIRED_SEED_COUNT = 3


ORIGINAL_LOADER = v531.load_expanded_sources


def load_corrected_sources(
    include_nonmetal: bool = False,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, list[str]]:
    truth, sources, quality, names = ORIGINAL_LOADER(include_nonmetal=include_nonmetal)
    complete = []
    for path, done_path in zip(SOURCE_PATHS, SOURCE_DONE, strict=True):
        if not path.is_file() or not done_path.is_file():
            continue
        done = np.load(done_path)
        value = np.load(path, mmap_mode="r")
        if value.shape[0] != len(truth) or not np.all(done):
            raise ValueError(f"incomplete corrected-array source: {path}")
        complete.append(v531.clean_source(value.reshape(len(truth), -1), sources[0]))
    if len(complete) != REQUIRED_SEED_COUNT:
        raise RuntimeError(
            f"v538 requires {REQUIRED_SEED_COUNT} complete seeds, found {len(complete)}"
        )
    sources.append(v494.normalize_rows(np.mean(complete, axis=0)))
    names.append(SOURCE_NAME)
    return truth, sources, quality, names


def add_medium_only_gate() -> dict[str, object]:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    old_report = json.loads(OLD_REPORT.read_text(encoding="utf-8"))
    with np.load(OOF_CACHE) as new, np.load(OLD_OOF) as old:
        method = str(report["decision_gate"]["best_method"])
        old_method = str(old_report["decision_gate"]["best_method"])
        new_score = np.asarray(new[f"{method}_score"], dtype=np.float64)
        old_score = np.asarray(old[f"{old_method}_score"], dtype=np.float64)
        fold_id = np.asarray(new["fold_id"], dtype=np.int16)
        if not np.array_equal(fold_id, old["fold_id"]):
            raise ValueError("v531 and v538 fold IDs differ")

    geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    regimes = regime_of(geometry["train_blocked"]).astype("U8")
    medium = regimes == "medium"
    raw_gain = new_score - old_score
    deployed_gain = np.where(medium, raw_gain, 0.0)
    medium_fold_gains = [
        float(np.mean(raw_gain[medium & (fold_id == fold)])) for fold in range(5)
    ]
    all_fold_gains = [
        float(np.mean(deployed_gain[fold_id == fold])) for fold in range(5)
    ]
    medium_gain = float(np.mean(raw_gain[medium]))
    all_gain = float(np.mean(deployed_gain))
    medium_fold_wins = int(sum(value > 0.0 for value in medium_fold_gains))
    passed = bool(
        all_gain >= MIN_ALL_GAIN
        and medium_gain >= MIN_MEDIUM_GAIN
        and medium_fold_wins >= MIN_MEDIUM_FOLD_WINS
    )
    gate = {
        "pass": passed,
        "new_method": method,
        "old_v531_method": old_method,
        "deployment_regime": "medium",
        "seed_count": REQUIRED_SEED_COUNT,
        "all_gain": all_gain,
        "medium_gain": medium_gain,
        "all_fold_gains": all_fold_gains,
        "medium_fold_gains": medium_fold_gains,
        "medium_fold_wins": medium_fold_wins,
        "nonmedium_gain_by_construction": 0.0,
        "requirements": {
            "all_gain_min": MIN_ALL_GAIN,
            "medium_gain_min": MIN_MEDIUM_GAIN,
            "medium_fold_wins_min": MIN_MEDIUM_FOLD_WINS,
            "required_seed_count": REQUIRED_SEED_COUNT,
        },
    }
    report["medium_only_incremental_gate_vs_v533_oof"] = gate
    report["primary_hypothesis"] = (
        "A three-seed corrected-array physical source adds stable medium-regime "
        "joint-S-H information, while v533 remains exactly unchanged outside medium."
    )
    report["deployment_rule"] = {
        "medium": method,
        "clear_light_heavy": "retain_v533",
    }
    report["status"] = "analysis_complete_gate_pass" if passed else "analysis_negative"
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return gate


def main() -> None:
    for path in (*SOURCE_PATHS, *SOURCE_DONE, OLD_REPORT, OLD_OOF):
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
    v531.load_expanded_sources = load_corrected_sources
    v531.main()
    gate = add_medium_only_gate()
    print(json.dumps({"medium_only_incremental_gate_vs_v533_oof": gate}, ensure_ascii=False))


if __name__ == "__main__":
    main()
