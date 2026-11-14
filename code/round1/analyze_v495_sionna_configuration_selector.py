#!/usr/bin/env python3
"""Predict which calibrated Sionna configuration is most reliable per point."""

from __future__ import annotations

import json
import math

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor

import analyze_v432_sionna_link_energy_calibration as v432
import analyze_v494_sionna_configuration_stacking as v494
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 495
VERSION = "v495_sionna_configuration_selector"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
OOF_CACHE = CACHE_DIR / f"{VERSION}_oof.npz"
V494_CACHE = PATHS.cache_dir(494) / "v494_sionna_configuration_stacking_oof.npz"
REGIMES = ("clear", "light", "medium", "heavy")
PCA_COMPONENTS = 32
SEED = 49501


def spectral_features(
    sources: list[np.ndarray],
    baseline: np.ndarray,
) -> np.ndarray:
    blocks: list[np.ndarray] = []
    for source in sources:
        grid = source.reshape(len(source), 192, 16)
        delay = grid.sum(axis=2)
        angle = grid.sum(axis=1)
        entropy = -np.sum(source * np.log(np.maximum(source, 1e-10)), axis=1)
        delay_entropy = -np.sum(delay * np.log(np.maximum(delay, 1e-10)), axis=1)
        angle_entropy = -np.sum(angle * np.log(np.maximum(angle, 1e-10)), axis=1)
        top = np.sort(source, axis=1)[:, ::-1]
        delay_phase = np.exp(2j * np.pi * np.arange(192) / 192.0)
        angle_phase = np.exp(2j * np.pi * np.arange(16) / 16.0)
        delay_moment = delay @ delay_phase
        angle_moment = angle @ angle_phase
        cosine = v432.row_cosine(source, baseline)
        l1 = np.sum(np.abs(source - baseline), axis=1)
        blocks.append(
            np.stack(
                [
                    cosine,
                    l1,
                    entropy / math.log(source.shape[1]),
                    delay_entropy / math.log(192.0),
                    angle_entropy / math.log(16.0),
                    top[:, 0],
                    top[:, :4].sum(axis=1),
                    top[:, :16].sum(axis=1),
                    top[:, :64].sum(axis=1),
                    delay_moment.real,
                    delay_moment.imag,
                    np.abs(delay_moment),
                    angle_moment.real,
                    angle_moment.imag,
                    np.abs(angle_moment),
                ],
                axis=1,
            ).astype(np.float32)
        )
    for left in range(len(sources)):
        for right in range(left + 1, len(sources)):
            blocks.append(
                np.stack(
                    [
                        v432.row_cosine(sources[left], sources[right]),
                        np.sum(np.abs(sources[left] - sources[right]), axis=1),
                    ],
                    axis=1,
                ).astype(np.float32)
            )
    return np.concatenate(blocks, axis=1).astype(np.float32)


def position_features(
    positions: np.ndarray,
    blocked: np.ndarray,
    regimes: np.ndarray,
) -> np.ndarray:
    xy = positions[:, :2].astype(np.float32)
    normalized = (xy - xy.mean(axis=0, keepdims=True)) / np.maximum(
        xy.std(axis=0, keepdims=True), 1.0
    )
    blocks = [normalized, positions[:, 2:3].astype(np.float32) / 20.0]
    for frequency in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        angle = np.float32(math.pi * frequency) * normalized
        blocks.extend([np.sin(angle), np.cos(angle)])
    block_z = (blocked - blocked.mean()) / max(float(blocked.std()), 1.0)
    blocks.append(block_z[:, None].astype(np.float32))
    blocks.append(
        np.stack(
            [(regimes == name).astype(np.float32) for name in REGIMES], axis=1
        )
    )
    return np.concatenate(blocks, axis=1).astype(np.float32)


def pca_input(sources: list[np.ndarray]) -> np.ndarray:
    anchor = np.maximum(sources[0], 1e-10)
    return np.concatenate(
        [
            np.clip(
                np.log(np.maximum(source, 1e-10)) - np.log(anchor), -3.0, 3.0
            )
            for source in sources[1:]
        ],
        axis=1,
    ).astype(np.float32)


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
    for path in (V494_CACHE, PATHS.round1 / "Round1_Train_Pos.npy"):
        if not path.is_file():
            raise FileNotFoundError(path)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _, sources, _ = v494.load_sources()
    cache = np.load(V494_CACHE)
    baseline = cache["baseline"].astype(np.float32)
    source_score = cache["source_score"].astype(np.float32)
    old_score = cache["old_score"].astype(np.float32)
    oracle_score = cache["oracle_score"].astype(np.float32)
    fold_id = cache["fold_id"].astype(np.int16)
    names = cache["source_names"].astype("U32")
    positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    blocked = geometry["train_blocked"].astype(np.float32)
    regimes = regime_of(blocked).astype("U8")
    summary = spectral_features(sources, baseline)
    position = position_features(positions, blocked, regimes)
    raw_pca = pca_input(sources)
    gains = (source_score - source_score[0:1]).T.astype(np.float32)

    selected_score = np.zeros(len(positions), dtype=np.float32)
    selected_source = np.zeros(len(positions), dtype=np.int16)
    shuffled_score = np.zeros(len(positions), dtype=np.float32)
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
            [summary, position, pca.transform(raw_pca).astype(np.float32)], axis=1
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
            f"v495 fold {fold + 1}/5 gain={fold_rows[-1]['gain_vs_old']:+.6f} "
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
            "Sionna configuration disagreement, geometry and obstruction features may predict "
            "which of six separately calibrated simulation targets is most reliable per point."
        ),
        "evidence_boundary": {
            "selection": "five mutually exclusive spatial folds",
            "features": "official positions/geometry and Sionna spectra only",
            "test_sionna_used": False,
            "online_feedback_used_for_parameters": False,
        },
        "model": {
            "type": "ExtraTrees multi-output score-gain selector",
            "estimators": 800,
            "min_samples_leaf": 12,
            "max_features": 0.75,
            "pca_components": PCA_COMPONENTS,
            "fallback_margin": 0.001,
        },
        "source_names": names.tolist(),
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
            "oof": PATHS.relative(OOF_CACHE),
            "report": PATHS.relative(REPORT),
        },
    }
    np.savez_compressed(
        OOF_CACHE,
        fold_id=fold_id,
        source_names=names,
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
