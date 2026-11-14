#!/usr/bin/env python3
"""Validate a three-seed corrected-array support union in the medium regime."""

from __future__ import annotations

import json

import numpy as np

import analyze_v494_sionna_configuration_stacking as v494
import analyze_v531_expanded_sbi_configuration_library as v531
import analyze_v538_corrected_array_medium_only as v538
from project_paths import PATHS


VERSION_NUMBER = 539
VERSION = "v539_corrected_array_support_union"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
OOF_CACHE = CACHE_DIR / f"{VERSION}_oof.npz"
SOURCE_NAME = "corrected_array_d10_sc015_seed_support_union_3"
SEEDS = v538.SEEDS
SOURCE_PATHS = v538.SOURCE_PATHS
SOURCE_DONE = v538.SOURCE_DONE


ORIGINAL_LOADER = v538.ORIGINAL_LOADER


def load_support_union_sources(
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
    if len(complete) != v538.REQUIRED_SEED_COUNT:
        raise RuntimeError(
            f"v539 requires {v538.REQUIRED_SEED_COUNT} complete seeds, found {len(complete)}"
        )
    sources.append(v494.normalize_rows(np.max(np.stack(complete), axis=0)))
    names.append(SOURCE_NAME)
    return truth, sources, quality, names


load_corrected_sources = load_support_union_sources


def main() -> None:
    for path in (*SOURCE_PATHS, *SOURCE_DONE, v538.OLD_REPORT, v538.OLD_OOF):
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
    v531.load_expanded_sources = load_support_union_sources
    v531.main()

    v538.REPORT = REPORT
    v538.OOF_CACHE = OOF_CACHE
    gate = v538.add_medium_only_gate()
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    report["primary_hypothesis"] = (
        "A max-normalized union of three corrected-array PathSolver seeds preserves "
        "stochastic path births better than probability averaging and improves medium "
        "joint-S-H while retaining v533 outside medium."
    )
    report["source_combination"] = "elementwise_max_then_row_normalize"
    report["status"] = "analysis_complete_gate_pass" if gate["pass"] else "analysis_negative"
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"medium_only_incremental_gate_vs_v533_oof": gate}, ensure_ascii=False))


if __name__ == "__main__":
    main()
