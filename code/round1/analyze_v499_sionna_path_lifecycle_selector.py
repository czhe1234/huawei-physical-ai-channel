#!/usr/bin/env python3
"""Select a path-lifecycle delay profile from explicit Sionna path statistics."""

from __future__ import annotations

import json
import math

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor

import analyze_v432_sionna_link_energy_calibration as v432
import analyze_v494_sionna_configuration_stacking as v494
import analyze_v495_sionna_configuration_selector as v495
import analyze_v498_sionna_path_lifecycle_profiles as v498
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 499
VERSION = "v499_sionna_path_lifecycle_selector"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
PATH_SUMMARY = CACHE_DIR / "v499_path_summary.npy"
OOF_CACHE = CACHE_DIR / "v499_sionna_path_lifecycle_selector_oof.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"
V498_CACHE = PATHS.cache_dir(498) / "v498_sionna_path_lifecycle_oof.npz"
PCA_COMPONENTS = 24
SEED = 49901


def normalize_rows(value: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    return value / np.maximum(value.sum(axis=1, keepdims=True), 1e-30)


def build_path_summary() -> np.ndarray:
    if PATH_SUMMARY.is_file():
        return np.load(PATH_SUMMARY, mmap_mode="r")
    rows: list[np.ndarray] = []
    for path in sorted((PATHS.cache_dir(475) / "shards").glob("shard_*/path_targets.npz")):
        data = np.load(path)
        block = np.zeros((len(data["valid"]), 32), dtype=np.float32)
        for row in range(len(block)):
            valid = data["valid"][row].astype(bool)
            count = data["interaction_count"][row, valid].astype(np.float32)
            power = data["normalized_power"][row, valid].astype(np.float32)
            tau = data["tau"][row, valid].astype(np.float32)
            interactions = data["interactions"][row, valid].astype(np.int16)
            if not len(power):
                continue
            power /= np.maximum(power.sum(), 1e-30)
            has_refraction = np.any(interactions == 4, axis=1)
            reflection_only = ~has_refraction
            sorted_power = np.sort(power)[::-1]
            features = [
                len(power) / 128.0,
                float(np.mean(count)) / 16.0,
                float(np.std(count)) / 16.0,
                float(np.max(count)) / 16.0,
                float(-np.sum(power * np.log(np.maximum(power, 1e-10)))) / math.log(128.0),
                float(sorted_power[0]),
                float(sorted_power[:4].sum()),
                float(sorted_power[:16].sum()),
                float(np.mean(tau)) * 1e7,
                float(np.std(tau)) * 1e7,
                float(np.sum(power * tau)) * 1e7,
            ]
            for threshold in (0, 1, 2, 3, 5, 8):
                mask = count <= threshold
                features.extend(
                    [float(np.mean(mask)), float(np.sum(power[mask]))]
                )
            features.extend(
                [
                    float(np.mean(reflection_only)),
                    float(np.sum(power[reflection_only])),
                    float(np.mean(has_refraction)),
                    float(np.sum(power[has_refraction])),
                ]
            )
            for kind in (1, 4):
                kind_count = np.sum(interactions == kind, axis=1)
                features.extend(
                    [
                        float(np.mean(kind_count)) / 16.0,
                        float(np.sum(power * kind_count)) / 16.0,
                    ]
                )
            block[row, : len(features)] = np.asarray(features, dtype=np.float32)
        rows.append(block)
    output = np.concatenate(rows, axis=0)
    if len(output) != 2500:
        raise ValueError(f"Expected 2500 path summaries, found {len(output)}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PATH_SUMMARY, output)
    return np.load(PATH_SUMMARY, mmap_mode="r")


def profile_features(
    profiles: np.ndarray,
    valid: np.ndarray,
    robust_delay: np.ndarray,
) -> np.ndarray:
    blocks: list[np.ndarray] = []
    for index in range(profiles.shape[1]):
        value = normalize_rows(np.asarray(profiles[:, index], dtype=np.float32))
        entropy = -np.sum(value * np.log(np.maximum(value, 1e-10)), axis=1)
        top = np.sort(value, axis=1)[:, ::-1]
        phase = np.exp(2j * np.pi * np.arange(192) / 192.0)
        moment = value @ phase
        blocks.append(
            np.stack(
                [
                    valid[:, index].astype(np.float32),
                    v432.row_cosine(value, robust_delay),
                    np.sum(np.abs(value - robust_delay), axis=1),
                    entropy / math.log(192.0),
                    top[:, 0],
                    top[:, :4].sum(axis=1),
                    top[:, :16].sum(axis=1),
                    moment.real,
                    moment.imag,
                    np.abs(moment),
                ],
                axis=1,
            ).astype(np.float32)
        )
    for left in range(profiles.shape[1]):
        for right in range(left + 1, profiles.shape[1]):
            a = normalize_rows(profiles[:, left])
            b = normalize_rows(profiles[:, right])
            blocks.append(
                np.stack(
                    [v432.row_cosine(a, b), np.sum(np.abs(a - b), axis=1)], axis=1
                ).astype(np.float32)
            )
    return np.concatenate(blocks, axis=1).astype(np.float32)


def make_model(seed: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=800,
        min_samples_leaf=12,
        max_features=0.75,
        bootstrap=False,
        n_jobs=-1,
        random_state=seed,
    )


def summarize(value: np.ndarray) -> dict[str, float]:
    value = np.asarray(value, dtype=np.float64)
    return {
        "mean": float(np.mean(value)),
        "p10": float(np.quantile(value, 0.10)),
        "median": float(np.median(value)),
        "p90": float(np.quantile(value, 0.90)),
    }


def main() -> None:
    for path in (
        V498_CACHE,
        PATHS.cache_dir(498) / "v498_path_delay_profiles.npy",
        PATHS.cache_dir(498) / "v498_path_delay_profile_valid.npy",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cache = np.load(V498_CACHE)
    source_score = cache["source_score"].astype(np.float32)
    old_score = cache["old_score"].astype(np.float32)
    oracle_score = cache["oracle_score"].astype(np.float32)
    fold_id = cache["fold_id"].astype(np.int16)
    names = cache["profile_names"].astype("U32")
    profiles = np.load(
        PATHS.cache_dir(498) / "v498_path_delay_profiles.npy", mmap_mode="r"
    )[:2000]
    valid = np.load(
        PATHS.cache_dir(498) / "v498_path_delay_profile_valid.npy", mmap_mode="r"
    )[:2000]
    _, sources, _ = v494.load_sources()
    robust_grid = sources[0].reshape(2000, 192, 16)
    robust_delay = normalize_rows(robust_grid.sum(axis=2))
    summary = np.asarray(build_path_summary()[:2000], dtype=np.float32)
    profile_summary = profile_features(profiles, valid, robust_delay)
    positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    blocked = geometry["train_blocked"].astype(np.float32)
    regimes = regime_of(blocked).astype("U8")
    position = v495.position_features(positions, blocked, regimes)
    raw_pca = np.asarray(profiles, dtype=np.float32).reshape(2000, -1)
    gains = (source_score - source_score[0:1]).T.astype(np.float32)

    selected_score = np.zeros(2000, dtype=np.float32)
    selected_source = np.zeros(2000, dtype=np.int16)
    shuffled_score = np.zeros(2000, dtype=np.float32)
    fold_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(SEED)
    for fold in range(5):
        fit_ids = np.flatnonzero(fold_id != fold)
        query_ids = np.flatnonzero(fold_id == fold)
        pca = PCA(
            n_components=PCA_COMPONENTS,
            whiten=True,
            svd_solver="randomized",
            random_state=SEED + fold,
        )
        pca.fit(raw_pca[fit_ids])
        features = np.concatenate(
            [summary, profile_summary, position, pca.transform(raw_pca)], axis=1
        ).astype(np.float32)
        model = make_model(SEED + fold)
        model.fit(features[fit_ids], gains[fit_ids])
        predicted = model.predict(features[query_ids]).astype(np.float32)
        predicted[:, 0] = 0.0
        choice = np.argmax(predicted, axis=1).astype(np.int16)
        margin = predicted[np.arange(len(query_ids)), choice]
        choice[margin < 0.001] = 0
        selected_source[query_ids] = choice
        selected_score[query_ids] = source_score[choice, query_ids]

        shuffled = gains[fit_ids][rng.permutation(len(fit_ids))]
        control = make_model(SEED + 100 + fold)
        control.fit(features[fit_ids], shuffled)
        control_pred = control.predict(features[query_ids]).astype(np.float32)
        control_pred[:, 0] = 0.0
        control_choice = np.argmax(control_pred, axis=1).astype(np.int16)
        control_margin = control_pred[np.arange(len(query_ids)), control_choice]
        control_choice[control_margin < 0.001] = 0
        shuffled_score[query_ids] = source_score[control_choice, query_ids]

        gain = selected_score[query_ids] - old_score[query_ids]
        control_gain = shuffled_score[query_ids] - old_score[query_ids]
        fold_rows.append(
            {
                "fold": fold,
                "count": int(len(query_ids)),
                "selected": float(np.mean(selected_score[query_ids])),
                "old": float(np.mean(old_score[query_ids])),
                "oracle": float(np.mean(oracle_score[query_ids])),
                "gain_vs_old": float(np.mean(gain)),
                "shuffled_gain_vs_old": float(np.mean(control_gain)),
                "selected_counts": {
                    str(names[index]): int(np.sum(choice == index))
                    for index in range(len(names))
                },
            }
        )
        print(
            f"v499 fold {fold + 1}/5 gain={fold_rows[-1]['gain_vs_old']:+.6f} "
            f"control={fold_rows[-1]['shuffled_gain_vs_old']:+.6f}",
            flush=True,
        )

    gain = selected_score - old_score
    control_gain = shuffled_score - old_score
    pass_gate = bool(
        float(np.mean(gain)) >= 0.005
        and sum(float(row["gain_vs_old"]) > 0.0 for row in fold_rows) >= 4
        and float(np.mean(gain - control_gain)) >= 0.003
    )
    report = {
        "method": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "Explicit path count, interaction depth/type, energy fractions and depth-filtered "
            "delay profiles can predict which lifecycle profile is reliable per point."
        ),
        "model": {
            "type": "ExtraTrees multi-output profile-gain selector",
            "estimators": 800,
            "min_samples_leaf": 12,
            "max_features": 0.75,
            "pca_components": PCA_COMPONENTS,
            "fallback_margin": 0.001,
        },
        "profile_names": names.tolist(),
        "scores": {
            "old": summarize(old_score),
            "selected": summarize(selected_score),
            "oracle": summarize(oracle_score),
            "gain_vs_old": summarize(gain),
            "shuffled_gain_vs_old": summarize(control_gain),
            "mean_gain_vs_old": float(np.mean(gain)),
            "mean_shuffled_gain_vs_old": float(np.mean(control_gain)),
            "gain_over_shuffled": float(np.mean(gain - control_gain)),
            "fold_wins": int(sum(float(row["gain_vs_old"]) > 0.0 for row in fold_rows)),
        },
        "selected_counts": {
            str(names[index]): int(np.sum(selected_source == index))
            for index in range(len(names))
        },
        "folds": fold_rows,
        "deployment_gate": {
            "required_gain": 0.005,
            "required_fold_wins": 4,
            "required_gain_over_shuffled": 0.003,
            "pass": pass_gate,
        },
        "outputs": {
            "path_summary": PATHS.relative(PATH_SUMMARY),
            "oof": PATHS.relative(OOF_CACHE),
            "report": PATHS.relative(REPORT),
        },
    }
    np.savez_compressed(
        OOF_CACHE,
        fold_id=fold_id,
        profile_names=names,
        selected_source=selected_source,
        selected_score=selected_score,
        old_score=old_score,
        oracle_score=oracle_score,
        shuffled_score=shuffled_score,
        gain_vs_old=gain,
    )
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
