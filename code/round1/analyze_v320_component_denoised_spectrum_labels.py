#!/usr/bin/env python3
"""Test continuous-path reconstructed spectra as denoised spatial labels."""

from __future__ import annotations

import json

import numpy as np

import analyze_v254_route_topology_spectral_field as v254
import analyze_v297_continuous_delay_component_transport as v297
import analyze_v300_path_gain_fingerprint_oracle as v300
import analyze_v308_joint_matched_patch_renderer as v308
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 320
VERSION = "v320_component_denoised_spectrum_labels"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
TARGET_CACHE = OUT_DIR / "v320_component_denoised_targets.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"
OUTER_SEEDS = (320, 1320, 2320, 3320, 4320)
OUTER_COUNT = 320
MIN_DELTA = 0.005


def exact_atoms(center: np.ndarray, size: int) -> np.ndarray:
    bins = np.arange(size, dtype=np.float64)[None, None, :]
    delta = bins - center[:, :, None].astype(np.float64)
    delta = np.remainder(delta + size / 2.0, size) - size / 2.0
    atom = np.square(np.sinc(delta) / np.sinc(delta / size))
    atom /= np.maximum(atom.sum(axis=2, keepdims=True), 1e-30)
    return atom.astype(np.float32)


def build_targets() -> tuple[np.ndarray, np.ndarray]:
    if TARGET_CACHE.is_file():
        data = np.load(TARGET_CACHE)
        return data["sh"], data["d"]
    component = np.load(v297.COMPONENT_CACHE)
    fingerprint = np.load(v300.FINGERPRINT_CACHE)
    delay = component["delay"].astype(np.float32)
    valid = component["valid"].astype(bool)
    mass = fingerprint["gain_power"].astype(np.float32) * valid
    mass /= np.maximum(mass.sum(axis=1, keepdims=True), 1e-30)
    d_atom = exact_atoms(delay, v297.N_DELAY)
    s_atom = exact_atoms(np.remainder(-delay, v297.N_DELAY), v297.N_DELAY)
    d = np.einsum("bp,bpd->bd", mass, d_atom, optimize=True).astype(np.float32)
    sh = np.einsum(
        "bp,bps,bph->bsh",
        mass,
        s_atom,
        fingerprint["horizontal"].astype(np.float32),
        optimize=True,
    ).astype(np.float32)
    d = v297.normalize_rows(d)
    sh = v297.normalize_rows(sh)
    np.savez_compressed(TARGET_CACHE, sh=sh, d=d)
    return sh, d


def main() -> None:
    denoised_sh, denoised_d = build_targets()
    raw = np.load(PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz")
    raw_sh = raw["sh"].astype(np.float32)
    raw_d = raw["d"].astype(np.float32)
    valid_ids = np.flatnonzero(raw["valid"].astype(bool)).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    fold_rows = []
    raw_sh_parts = []
    denoised_sh_parts = []
    truth_sh_parts = []
    raw_d_parts = []
    denoised_d_parts = []
    truth_d_parts = []
    for fold, seed in enumerate(OUTER_SEEDS):
        query_ids = np.intersect1d(
            v254.choose_seeded_matched_indices(
                train_pos[:, :2], test_pos[:, :2], OUTER_COUNT, seed
            ),
            valid_ids,
        )
        bank_ids = np.setdiff1d(valid_ids, query_ids)
        raw_anchor_sh = v308.generic_knn_anchor(query_ids, bank_ids, train_pos, raw_sh)
        denoised_anchor_sh = v308.generic_knn_anchor(
            query_ids, bank_ids, train_pos, denoised_sh
        )
        raw_anchor_d = v308.generic_knn_anchor(query_ids, bank_ids, train_pos, raw_d)
        denoised_anchor_d = v308.generic_knn_anchor(
            query_ids, bank_ids, train_pos, denoised_d
        )
        raw_sh_score = float(np.mean(v297.row_cos(raw_anchor_sh, raw_sh[query_ids])))
        denoised_sh_score = float(
            np.mean(v297.row_cos(denoised_anchor_sh, raw_sh[query_ids]))
        )
        raw_d_score = float(np.mean(v297.row_cos(raw_anchor_d, raw_d[query_ids])))
        denoised_d_score = float(
            np.mean(v297.row_cos(denoised_anchor_d, raw_d[query_ids]))
        )
        delta = 0.52 * (denoised_sh_score - raw_sh_score) + 0.48 * (
            denoised_d_score - raw_d_score
        )
        fold_rows.append(
            {
                "fold": fold,
                "seed": seed,
                "query_count": int(len(query_ids)),
                "raw_anchor_sh": raw_sh_score,
                "denoised_anchor_sh": denoised_sh_score,
                "raw_anchor_d": raw_d_score,
                "denoised_anchor_d": denoised_d_score,
                "weighted_delta": delta,
            }
        )
        raw_sh_parts.append(raw_anchor_sh)
        denoised_sh_parts.append(denoised_anchor_sh)
        truth_sh_parts.append(raw_sh[query_ids])
        raw_d_parts.append(raw_anchor_d)
        denoised_d_parts.append(denoised_anchor_d)
        truth_d_parts.append(raw_d[query_ids])
        print(f"v320 fold={fold} delta={delta:+.6f}", flush=True)
    raw_sh_all = np.concatenate(raw_sh_parts)
    denoised_sh_all = np.concatenate(denoised_sh_parts)
    truth_sh_all = np.concatenate(truth_sh_parts)
    raw_d_all = np.concatenate(raw_d_parts)
    denoised_d_all = np.concatenate(denoised_d_parts)
    truth_d_all = np.concatenate(truth_d_parts)
    pooled = {
        "raw_sh": float(np.mean(v297.row_cos(raw_sh_all, truth_sh_all))),
        "denoised_sh": float(np.mean(v297.row_cos(denoised_sh_all, truth_sh_all))),
        "raw_d": float(np.mean(v297.row_cos(raw_d_all, truth_d_all))),
        "denoised_d": float(np.mean(v297.row_cos(denoised_d_all, truth_d_all))),
    }
    pooled["raw_weighted"] = 0.52 * pooled["raw_sh"] + 0.48 * pooled["raw_d"]
    pooled["denoised_weighted"] = 0.52 * pooled["denoised_sh"] + 0.48 * pooled["denoised_d"]
    pooled["delta"] = pooled["denoised_weighted"] - pooled["raw_weighted"]
    fold_wins = int(sum(row["weighted_delta"] > 0.0 for row in fold_rows))
    passed = pooled["delta"] >= MIN_DELTA and fold_wins >= 4
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "Continuous shared-path reconstruction removes complex Gaussian observation noise "
            "and yields better spatial S-H/PDP supervision than raw official H spectra."
        ),
        "reconstruction_vs_raw": {
            "sh_cosine": summarize(v297.row_cos(denoised_sh[valid_ids], raw_sh[valid_ids])),
            "d_cosine": summarize(v297.row_cos(denoised_d[valid_ids], raw_d[valid_ids])),
        },
        "fold_rows": fold_rows,
        "pooled": pooled | {"fold_wins": fold_wins},
        "decision_rule": {
            "weighted_delta": f">={MIN_DELTA}",
            "fold_wins": ">=4/5",
            "passed": bool(passed),
        },
        "outputs": {
            "targets": PATHS.relative(TARGET_CACHE),
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
