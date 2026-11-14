#!/usr/bin/env python3
"""Audit path-depth and interaction-filtered Sionna delay support."""

from __future__ import annotations

import json

import numpy as np

import analyze_v432_sionna_link_energy_calibration as v432
import analyze_v434_sionna_joint_sh_calibration as v434
import analyze_v494_sionna_configuration_stacking as v494
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 498
VERSION = "v498_sionna_path_lifecycle_profiles"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
PROFILE_CACHE = CACHE_DIR / "v498_path_delay_profiles.npy"
PROFILE_VALID = CACHE_DIR / "v498_path_delay_profile_valid.npy"
OOF_CACHE = CACHE_DIR / "v498_sionna_path_lifecycle_oof.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"
PATH_ROOT = PATHS.cache_dir(475) / "shards"
PROFILE_NAMES = (
    "robust_v434",
    "los_only",
    "depth_le1",
    "depth_le2",
    "depth_le3",
    "depth_le5",
    "reflection_only",
    "reflection_depth5",
    "refraction_present",
    "full_depth16",
)
REGIMES = ("clear", "light", "medium", "heavy")
N_DELAY = 192
SUBCARRIER_SPACING = 30_000.0


def normalize_rows(value: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    return value / np.maximum(value.sum(axis=1, keepdims=True), TINY)


def delay_fingerprint(tau: np.ndarray) -> np.ndarray:
    # leaked_forward uses FFT over subcarriers; the path peak therefore uses +k/N.
    k = np.arange(N_DELAY, dtype=np.float64) / N_DELAY
    delta = SUBCARRIER_SPACING * tau[:, None].astype(np.float64) + k[None, :]
    numerator = np.sinc(N_DELAY * delta)
    denominator = np.sinc(delta)
    value = (numerator / np.maximum(np.abs(denominator), 1e-30)) ** 2
    value /= np.maximum(value.sum(axis=1, keepdims=True), 1e-30)
    return value.astype(np.float32)


def profile_masks(data: np.lib.npyio.NpzFile, row: int) -> list[np.ndarray]:
    valid = data["valid"][row].astype(bool)
    count = data["interaction_count"][row].astype(np.int16)
    interactions = data["interactions"][row].astype(np.int16)
    has_refraction = np.any(interactions == 4, axis=1)
    reflection_only = ~has_refraction
    return [
        valid & (count == 0),
        valid & (count <= 1),
        valid & (count <= 2),
        valid & (count <= 3),
        valid & (count <= 5),
        valid & reflection_only,
        valid & reflection_only & (count <= 5),
        valid & has_refraction,
        valid,
    ]


def build_profiles() -> tuple[np.ndarray, np.ndarray]:
    if PROFILE_CACHE.is_file() and PROFILE_VALID.is_file():
        return (
            np.load(PROFILE_CACHE, mmap_mode="r"),
            np.load(PROFILE_VALID, mmap_mode="r"),
        )
    paths = sorted(PATH_ROOT.glob("shard_*/path_targets.npz"))
    if len(paths) != 25:
        raise ValueError(f"Expected 25 v475 shards, found {len(paths)}")
    output = np.lib.format.open_memmap(
        PROFILE_CACHE,
        mode="w+",
        dtype=np.float32,
        shape=(2500, len(PROFILE_NAMES) - 1, N_DELAY),
    )
    profile_valid = np.zeros((2500, len(PROFILE_NAMES) - 1), dtype=bool)
    offset = 0
    for shard_path in paths:
        data = np.load(shard_path)
        for row in range(len(data["valid"])):
            valid = data["valid"][row].astype(bool)
            tau = data["tau"][row, valid].astype(np.float32)
            power = data["normalized_power"][row, valid].astype(np.float32)
            if len(tau):
                kernels = delay_fingerprint(tau)
                valid_ids = np.flatnonzero(valid)
                index = {int(value): local for local, value in enumerate(valid_ids)}
            else:
                kernels = np.empty((0, N_DELAY), dtype=np.float32)
                index = {}
            for profile_index, mask in enumerate(profile_masks(data, row)):
                ids = np.flatnonzero(mask)
                if len(ids):
                    local = np.asarray([index[int(value)] for value in ids], dtype=np.int64)
                    weights = power[local]
                    value = np.sum(kernels[local] * weights[:, None], axis=0)
                    value /= np.maximum(value.sum(), TINY)
                    output[offset + row, profile_index] = value.astype(np.float32)
                    profile_valid[offset + row, profile_index] = True
                else:
                    output[offset + row, profile_index] = 0.0
        offset += len(data["valid"])
        output.flush()
        print(f"v498 path profiles {offset}/2500", flush=True)
    np.save(PROFILE_VALID, profile_valid)
    return np.load(PROFILE_CACHE, mmap_mode="r"), np.load(PROFILE_VALID, mmap_mode="r")


def path_conditioned_sources(
    robust: np.ndarray,
    profiles: np.ndarray,
    valid: np.ndarray,
) -> list[np.ndarray]:
    robust = normalize_rows(robust)
    grid = robust.reshape(len(robust), N_DELAY, 16)
    current_delay = normalize_rows(grid.sum(axis=2))
    sources = [robust]
    for profile_index in range(profiles.shape[1]):
        desired_delay = np.asarray(profiles[: len(robust), profile_index], dtype=np.float32)
        local_valid = np.asarray(valid[: len(robust), profile_index], dtype=bool)
        desired_delay = normalize_rows(desired_delay)
        desired_delay[~local_valid] = current_delay[~local_valid]
        candidate = grid * (
            desired_delay / np.maximum(current_delay, 1e-10)
        )[:, :, None]
        sources.append(normalize_rows(candidate.reshape(len(robust), -1)))
    return sources


def summarize(value: np.ndarray) -> dict[str, float]:
    value = np.asarray(value, dtype=np.float64)
    return {
        "mean": float(np.mean(value)),
        "p10": float(np.quantile(value, 0.10)),
        "median": float(np.median(value)),
        "p90": float(np.quantile(value, 0.90)),
    }


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    truth, base_sources, quality = v494.load_sources()
    robust = base_sources[0]
    profiles, profile_valid = build_profiles()
    sources = path_conditioned_sources(robust, profiles, profile_valid)
    positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    blocked = geometry["train_blocked"].astype(np.float32)
    regimes = regime_of(blocked).astype("U8")
    fold_id = np.load(v434.OOF_CACHE)["fold_id"].astype(np.int16)
    all_ids = np.arange(len(truth), dtype=np.int64)
    prediction = np.zeros_like(truth)
    old_prediction = np.zeros_like(truth)
    source_score = np.zeros((len(sources), len(truth)), dtype=np.float32)
    selected_source = np.zeros(len(truth), dtype=np.int16)
    fold_rows: list[dict[str, object]] = []

    for outer in range(5):
        query_ids = np.flatnonzero(fold_id == outer)
        fit_ids = np.flatnonzero(fold_id != outer)
        inner_baseline = np.zeros_like(truth)
        inner_candidates = np.zeros(
            (len(sources), len(truth), truth.shape[1]), dtype=np.float32
        )
        for inner in range(5):
            if inner == outer:
                continue
            inner_query = np.flatnonzero(fold_id == inner)
            inner_fit = np.flatnonzero((fold_id != outer) & (fold_id != inner))
            inner_baseline[inner_query] = v432.neighbor_prediction(
                inner_fit,
                inner_query,
                positions,
                blocked,
                regimes,
                truth,
                k=48,
            )
            inner_candidates[:, inner_query] = v494.calibrated_sources(
                truth, sources, inner_fit, inner_query, regimes
            )
        query_baseline = v432.neighbor_prediction(
            fit_ids, query_ids, positions, blocked, regimes, truth, k=48
        )
        query_candidates = v494.calibrated_sources(
            truth, sources, fit_ids, query_ids, regimes
        )
        source_alpha: list[dict[str, float]] = []
        inner_score = np.full((len(sources), len(REGIMES)), -np.inf, dtype=np.float64)
        for source_index in range(len(sources)):
            alpha = v494.select_alpha(
                truth,
                inner_baseline,
                inner_candidates[source_index],
                quality,
                fit_ids,
                regimes,
            )
            source_alpha.append(alpha)
            candidate = v494.blend_target(
                inner_baseline[fit_ids],
                inner_candidates[source_index, fit_ids],
                quality,
                fit_ids,
                regimes,
                alpha,
            )
            score = v432.row_cosine(truth[fit_ids], candidate)
            for regime_index, regime in enumerate(REGIMES):
                local = regimes[fit_ids] == regime
                if np.any(local):
                    inner_score[source_index, regime_index] = float(np.mean(score[local]))
        chosen = {
            regime: int(np.argmax(inner_score[:, regime_index]))
            for regime_index, regime in enumerate(REGIMES)
        }
        query_prediction = np.zeros((len(query_ids), truth.shape[1]), dtype=np.float32)
        for source_index in range(len(sources)):
            source_candidate = v494.blend_target(
                query_baseline,
                query_candidates[source_index],
                quality,
                np.arange(len(query_ids)),
                regimes[query_ids],
                source_alpha[source_index],
            )
            source_score[source_index, query_ids] = v432.row_cosine(
                truth[query_ids], source_candidate
            )
            use = np.asarray(
                [chosen[str(regime)] == source_index for regime in regimes[query_ids]],
                dtype=bool,
            )
            query_prediction[use] = source_candidate[use]
            selected_source[query_ids[use]] = source_index
        prediction[query_ids] = query_prediction
        old_prediction[query_ids] = v494.blend_target(
            query_baseline,
            query_candidates[0],
            quality,
            np.arange(len(query_ids)),
            regimes[query_ids],
            source_alpha[0],
        )
        selected_score = v432.row_cosine(truth[query_ids], prediction[query_ids])
        old_score = v432.row_cosine(truth[query_ids], old_prediction[query_ids])
        fold_rows.append(
            {
                "fold": outer,
                "count": int(len(query_ids)),
                "selected": float(np.mean(selected_score)),
                "old": float(np.mean(old_score)),
                "gain_vs_old": float(np.mean(selected_score - old_score)),
                "chosen_by_regime": {
                    regime: PROFILE_NAMES[index] for regime, index in chosen.items()
                },
            }
        )
        print(
            f"v498 outer {outer + 1}/5 gain={fold_rows[-1]['gain_vs_old']:+.6f}",
            flush=True,
        )

    old_score = v432.row_cosine(truth, old_prediction)
    selected_score = v432.row_cosine(truth, prediction)
    oracle_score = np.max(source_score, axis=0)
    gain = selected_score - old_score
    pass_gate = bool(
        float(np.mean(gain)) >= 0.005
        and sum(float(row["gain_vs_old"]) > 0.0 for row in fold_rows) >= 4
    )
    report = {
        "method": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "Exact per-path delay fingerprints filtered by interaction depth and type may "
            "identify stable Sionna support that aggregate spectra mix with unreliable paths."
        ),
        "path_reconstruction": {
            "v475_pdp_cosine_first_shard": 0.9931486320818216,
            "joint_delay_mapping": "FFT sign +1; equivalent to reversed conventional PDP",
        },
        "profiles": list(PROFILE_NAMES),
        "scores": {
            "old": summarize(old_score),
            "selected": summarize(selected_score),
            "oracle": summarize(oracle_score),
            "gain_vs_old": summarize(gain),
            "mean_gain_vs_old": float(np.mean(gain)),
            "oracle_gain_vs_old": float(np.mean(oracle_score - old_score)),
            "fold_wins": int(sum(float(row["gain_vs_old"]) > 0.0 for row in fold_rows)),
        },
        "folds": fold_rows,
        "profile_valid_fraction": {
            PROFILE_NAMES[index + 1]: float(np.mean(profile_valid[:2000, index]))
            for index in range(profile_valid.shape[1])
        },
        "selected_counts": {
            PROFILE_NAMES[index]: int(np.sum(selected_source == index))
            for index in range(len(PROFILE_NAMES))
        },
        "deployment_gate": {
            "required_gain": 0.005,
            "required_fold_wins": 4,
            "pass": pass_gate,
        },
        "outputs": {
            "profiles": PATHS.relative(PROFILE_CACHE),
            "valid": PATHS.relative(PROFILE_VALID),
            "oof": PATHS.relative(OOF_CACHE),
            "report": PATHS.relative(REPORT),
        },
    }
    np.savez_compressed(
        OOF_CACHE,
        fold_id=fold_id,
        profile_names=np.asarray(PROFILE_NAMES),
        prediction=prediction,
        old_prediction=old_prediction,
        source_score=source_score,
        selected_source=selected_source,
        old_score=old_score,
        selected_score=selected_score,
        oracle_score=oracle_score,
        gain_vs_old=gain,
    )
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
