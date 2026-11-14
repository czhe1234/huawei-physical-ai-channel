#!/usr/bin/env python3
"""Test whether Sionna-to-official complex path corrections transfer across space."""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

import analyze_v501_full_official_sionna_inverse_calibration as v501
import analyze_v503_fixed_material_sionna_version_inverse as v503
import analyze_v507_complex_path_frequency_oracle as v507
import analyze_v510_joint_delay_h_complex_oracle as v510
from project_paths import PATHS


VERSION_NUMBER = 516
VERSION = "v516_shared_complex_path_correction"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
COEFFICIENT_CACHE = CACHE_DIR / "v516_path_coefficients_points160.npz"
POINT_METRICS = CACHE_DIR / "v516_point_metrics.npz"
TRAIN_POS = PATHS.round1 / "Round1_Train_Pos.npy"
TRAIN_H = PATHS.round1 / "Round1_Train_Channel.npy"
V503_REPORT = PATHS.version_report_dir(503) / "v503_refine.json"
V515_CFR = PATHS.cache_dir(515) / "v515_base_cfr_points160.npy"

PATH_LIMIT = 32
REGIME_NAMES = ("clear", "light", "medium", "heavy")
MODE_NAMES = (
    "direct_sionna",
    "source_atom_fit",
    "global_complex_scalar",
    "predicted_magnitude_only",
    "predicted_complex_scalar",
    "shuffled_label_control",
    "oracle_complex_scalar",
    "oracle_full_coefficients",
)
TINY = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trees", type=int, default=600)
    parser.add_argument("--min-samples-leaf", type=int, default=8)
    parser.add_argument("--max-features", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=516_001)
    parser.add_argument("--force-coefficients", action="store_true")
    return parser.parse_args()


def indices() -> np.ndarray:
    return np.asarray(
        json.loads(V503_REPORT.read_text(encoding="utf-8"))["indices"],
        dtype=np.int64,
    )


def select_source_atoms(
    matrix: np.ndarray,
    source_target: np.ndarray,
    depth: np.ndarray,
) -> np.ndarray:
    if matrix.shape[1] <= PATH_LIMIT:
        return np.arange(matrix.shape[1], dtype=np.int64)
    coefficients = v507.ridge_coefficients(matrix, source_target)
    energy = np.sum(np.abs(coefficients) ** 2, axis=1, dtype=np.float64)
    score = np.log(np.maximum(energy, TINY)) - 1e-6 * depth
    return np.argsort(score)[-PATH_LIMIT:].astype(np.int64)


def build_coefficient_cache(point_ids: np.ndarray, force: bool) -> dict[str, np.ndarray]:
    if COEFFICIENT_CACHE.is_file() and not force:
        with np.load(COEFFICIENT_CACHE) as data:
            return {name: np.asarray(data[name]) for name in data.files}

    union = v507.load_union(point_ids)
    source = np.load(V515_CFR, mmap_mode="r")
    truth = np.load(TRAIN_H, mmap_mode="r")
    tau_out = np.zeros((len(point_ids), PATH_LIMIT), dtype=np.float64)
    h_out = np.zeros_like(tau_out)
    depth_out = np.zeros((len(point_ids), PATH_LIMIT), dtype=np.float32)
    valid_out = np.zeros((len(point_ids), PATH_LIMIT), dtype=bool)
    source_coefficients = np.zeros(
        (len(point_ids), PATH_LIMIT, 64), dtype=np.complex64
    )
    official_coefficients = np.zeros_like(source_coefficients)
    source_fit_coherence = np.zeros(len(point_ids), dtype=np.float32)
    official_fit_coherence = np.zeros(len(point_ids), dtype=np.float32)

    for row, point_id in enumerate(point_ids):
        valid = union["valid"][row].astype(bool)
        tau, h_center, depth = v510.path_candidates(
            union["tau"][row, valid],
            union["phi_t"][row, valid],
            union["interaction_count"][row, valid],
        )
        if not len(tau):
            continue
        matrix = v510.joint_basis(tau, h_center)
        source_target = v510.target_matrix(np.asarray(source[row], dtype=np.complex64))
        official_target = v510.target_matrix(
            np.asarray(truth[int(point_id)], dtype=np.complex64)
        )
        selected = select_source_atoms(matrix, source_target, depth)
        matrix = matrix[:, selected]
        local_tau = tau[selected]
        local_h = h_center[selected]
        local_depth = depth[selected]
        source_local = v507.ridge_coefficients(matrix, source_target)
        official_local = v507.ridge_coefficients(matrix, official_target)
        count = len(selected)
        tau_out[row, :count] = local_tau
        h_out[row, :count] = local_h
        depth_out[row, :count] = local_depth
        valid_out[row, :count] = True
        source_coefficients[row, :count] = source_local.astype(np.complex64)
        official_coefficients[row, :count] = official_local.astype(np.complex64)
        source_fit = matrix @ source_local
        official_fit = matrix @ official_local
        source_fit_coherence[row] = v507.complex_metrics(
            v510.channel_from_matrix(source_fit), np.asarray(source[row])
        )["complex_coherence"]
        official_fit_coherence[row] = v507.complex_metrics(
            v510.channel_from_matrix(official_fit), np.asarray(truth[int(point_id)])
        )["complex_coherence"]
        if (row + 1) % 10 == 0 or row + 1 == len(point_ids):
            print(f"v516 coefficient fit {row + 1}/{len(point_ids)}", flush=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        COEFFICIENT_CACHE,
        indices=point_ids,
        tau=tau_out,
        h_center=h_out,
        depth=depth_out,
        valid=valid_out,
        source_coefficients=source_coefficients,
        official_coefficients=official_coefficients,
        source_fit_coherence=source_fit_coherence,
        official_fit_coherence=official_fit_coherence,
    )
    return {
        "indices": point_ids,
        "tau": tau_out,
        "h_center": h_out,
        "depth": depth_out,
        "valid": valid_out,
        "source_coefficients": source_coefficients,
        "official_coefficients": official_coefficients,
        "source_fit_coherence": source_fit_coherence,
        "official_fit_coherence": official_fit_coherence,
    }


def path_features(
    data: dict[str, np.ndarray],
    positions: np.ndarray,
    regimes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid = data["valid"].astype(bool)
    tau = data["tau"].astype(np.float64)
    h_center = data["h_center"].astype(np.float64)
    depth = data["depth"].astype(np.float64)
    source = data["source_coefficients"].astype(np.complex128)
    official = data["official_coefficients"].astype(np.complex128)

    source_energy = np.sum(np.abs(source) ** 2, axis=2, dtype=np.float64)
    official_energy = np.sum(np.abs(official) ** 2, axis=2, dtype=np.float64)
    source_total = np.maximum(np.sum(source_energy, axis=1, keepdims=True), TINY)
    official_total = np.maximum(np.sum(official_energy, axis=1, keepdims=True), TINY)
    source_fraction = source_energy / source_total
    official_fraction = official_energy / official_total

    numerator = np.sum(np.conj(source) * official, axis=2, dtype=np.complex128)
    denominator = np.maximum(source_energy, TINY)
    complex_scalar = numerator / denominator
    magnitude = np.clip(np.abs(complex_scalar), 1e-3, 1e3)
    target = np.stack(
        [
            np.log(magnitude),
            np.cos(np.angle(complex_scalar)),
            np.sin(np.angle(complex_scalar)),
        ],
        axis=2,
    )

    coefficient_power = np.abs(source.reshape(len(source), PATH_LIMIT, 2, 8, 4)) ** 2
    coefficient_power /= np.maximum(
        np.sum(coefficient_power, axis=(2, 3, 4), keepdims=True), TINY
    )
    pol = np.sum(coefficient_power, axis=(3, 4))
    vertical = np.sum(coefficient_power, axis=(2, 4))
    ue = np.sum(coefficient_power, axis=(2, 3))
    coherent_sum = np.abs(np.sum(source, axis=2)) / np.maximum(
        np.sqrt(64.0 * source_energy), TINY
    )
    strongest = np.argmax(np.abs(source), axis=2)
    reference = np.take_along_axis(source, strongest[..., None], axis=2)[..., 0]

    relative = positions - v501.BS_POSITION[None, :]
    distance = np.linalg.norm(relative, axis=1)
    bearing = np.arctan2(relative[:, 1], relative[:, 0])
    point = np.concatenate(
        [
            relative / np.asarray([200.0, 200.0, 50.0]),
            (distance / 200.0)[:, None],
            np.sin(bearing)[:, None],
            np.cos(bearing)[:, None],
            np.eye(4, dtype=np.float64)[regimes],
        ],
        axis=1,
    )
    point = np.repeat(point[:, None, :], PATH_LIMIT, axis=1)
    delay_center = np.mod(
        v507.N_SUBCARRIERS * v507.SUBCARRIER_SPACING * tau,
        v507.N_SUBCARRIERS,
    )
    delay_angle = 2.0 * np.pi * delay_center / v507.N_SUBCARRIERS
    h_angle = 2.0 * np.pi * h_center / 16.0
    rank = np.repeat(
        (np.arange(PATH_LIMIT, dtype=np.float64) / PATH_LIMIT)[None, :, None],
        len(source),
        axis=0,
    )
    features = np.concatenate(
        [
            point,
            np.sin(delay_angle)[..., None],
            np.cos(delay_angle)[..., None],
            (delay_center / v507.N_SUBCARRIERS)[..., None],
            np.sin(h_angle)[..., None],
            np.cos(h_angle)[..., None],
            (depth / 16.0)[..., None],
            np.log(np.maximum(source_energy, TINY))[..., None],
            np.log(np.maximum(source_fraction, TINY))[..., None],
            source_fraction[..., None],
            coherent_sum[..., None],
            np.cos(np.angle(reference))[..., None],
            np.sin(np.angle(reference))[..., None],
            rank,
            pol,
            vertical,
            ue,
        ],
        axis=2,
    ).astype(np.float32)
    sample_weight = np.sqrt(np.maximum(official_fraction, 0.0))
    sample_weight *= valid
    return features, target.astype(np.float32), sample_weight.astype(np.float64), complex_scalar


def make_model(args: argparse.Namespace, seed: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=args.trees,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        n_jobs=-1,
        random_state=seed,
    )


def scalar_from_prediction(prediction: np.ndarray) -> np.ndarray:
    magnitude = np.exp(np.clip(prediction[:, 0], -4.0, 4.0))
    direction = prediction[:, 1] + 1j * prediction[:, 2]
    direction /= np.maximum(np.abs(direction), 1e-8)
    return magnitude * direction


def render_point(
    tau: np.ndarray,
    h_center: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    matrix = v510.joint_basis(tau, h_center)
    return v510.channel_from_matrix(matrix @ coefficients)


def summarize(values: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    selected = np.asarray(values, dtype=np.float64)[mask]
    return {
        "count": int(len(selected)),
        "mean": float(np.mean(selected)),
        "median": float(np.median(selected)),
        "p10": float(np.quantile(selected, 0.10)),
        "p90": float(np.quantile(selected, 0.90)),
    }


def main() -> None:
    args = parse_args()
    for path in (TRAIN_POS, TRAIN_H, V503_REPORT, V515_CFR):
        if not path.is_file():
            raise FileNotFoundError(path)
    point_ids = indices()
    data = build_coefficient_cache(point_ids, args.force_coefficients)
    all_positions = np.load(TRAIN_POS).astype(np.float32)
    all_regimes = v501.regime_codes()
    positions = all_positions[point_ids]
    regimes = all_regimes[point_ids]
    fold_codes = v501.spatial_fold_codes(all_positions, all_regimes)[point_ids]
    features, targets, sample_weight, oracle_scalar = path_features(
        data, positions, regimes
    )
    valid = data["valid"].astype(bool)
    source_coefficients = data["source_coefficients"].astype(np.complex128)
    official_coefficients = data["official_coefficients"].astype(np.complex128)

    predicted_complex = np.ones((len(point_ids), PATH_LIMIT), dtype=np.complex128)
    predicted_magnitude = np.ones_like(predicted_complex)
    global_scalar = np.ones_like(predicted_complex)
    shuffled_scalar = np.ones_like(predicted_complex)
    fold_details = []

    for fold in sorted(np.unique(fold_codes).tolist()):
        train_points = fold_codes != fold
        test_points = fold_codes == fold
        train_mask = train_points[:, None] & valid
        test_mask = test_points[:, None] & valid
        x_train = features[train_mask]
        y_train = targets[train_mask]
        w_train = sample_weight[train_mask]
        x_test = features[test_mask]

        model = make_model(args, args.seed + int(fold))
        model.fit(x_train, y_train, sample_weight=w_train)
        prediction = model.predict(x_test)
        predicted_complex[test_mask] = scalar_from_prediction(prediction)
        predicted_magnitude[test_mask] = np.exp(
            np.clip(prediction[:, 0], -4.0, 4.0)
        )

        weight = w_train / max(np.sum(w_train), TINY)
        global_target = np.sum(y_train * weight[:, None], axis=0)
        global_scalar[test_mask] = scalar_from_prediction(global_target[None])[0]

        rng = np.random.default_rng(args.seed + 10_000 + int(fold))
        shuffled = y_train[rng.permutation(len(y_train))]
        control = make_model(args, args.seed + 20_000 + int(fold))
        control.fit(x_train, shuffled, sample_weight=w_train)
        shuffled_scalar[test_mask] = scalar_from_prediction(control.predict(x_test))
        fold_details.append(
            {
                "fold": int(fold),
                "train_points": int(np.sum(train_points)),
                "test_points": int(np.sum(test_points)),
                "train_paths": int(np.sum(train_mask)),
                "test_paths": int(np.sum(test_mask)),
            }
        )
        print(f"v516 fold {fold} fitted", flush=True)

    truth = np.load(TRAIN_H, mmap_mode="r")
    direct_sionna = np.load(V515_CFR, mmap_mode="r")
    metric_names = ("pdp", "leaked_full", "complex_coherence", "nmse", "traceback_like")
    metrics = {
        name: np.zeros((len(MODE_NAMES), len(point_ids)), dtype=np.float32)
        for name in metric_names
    }
    fold_gain = {name: [] for name in MODE_NAMES}

    for row, point_id in enumerate(point_ids):
        local_valid = valid[row]
        tau = data["tau"][row, local_valid]
        h_center = data["h_center"][row, local_valid]
        source_local = source_coefficients[row, local_valid]
        official_local = official_coefficients[row, local_valid]
        coefficient_sets = {
            "source_atom_fit": source_local,
            "global_complex_scalar": source_local * global_scalar[row, local_valid, None],
            "predicted_magnitude_only": source_local
            * predicted_magnitude[row, local_valid, None],
            "predicted_complex_scalar": source_local
            * predicted_complex[row, local_valid, None],
            "shuffled_label_control": source_local
            * shuffled_scalar[row, local_valid, None],
            "oracle_complex_scalar": source_local
            * oracle_scalar[row, local_valid, None],
            "oracle_full_coefficients": official_local,
        }
        target = np.asarray(truth[int(point_id)], dtype=np.complex64)
        predictions = {"direct_sionna": np.asarray(direct_sionna[row])}
        predictions.update(
            {
                name: render_point(tau, h_center, coefficients)
                for name, coefficients in coefficient_sets.items()
            }
        )
        for mode_index, mode in enumerate(MODE_NAMES):
            result = v507.complex_metrics(predictions[mode], target)
            for name in metric_names:
                metrics[name][mode_index, row] = result[name]
        if (row + 1) % 10 == 0 or row + 1 == len(point_ids):
            print(f"v516 render {row + 1}/{len(point_ids)}", flush=True)

    source_mode = MODE_NAMES.index("source_atom_fit")
    for mode_index, mode in enumerate(MODE_NAMES):
        for fold in sorted(np.unique(fold_codes).tolist()):
            mask = fold_codes == fold
            fold_gain[mode].append(
                float(
                    np.mean(metrics["traceback_like"][mode_index, mask])
                    - np.mean(metrics["traceback_like"][source_mode, mask])
                )
            )

    summary = []
    for mode_index, mode in enumerate(MODE_NAMES):
        row = {"mode": mode, "overall": {}, "heavy": {}}
        for name, value in metrics.items():
            row["overall"][name] = summarize(
                value[mode_index], np.ones(len(point_ids), dtype=bool)
            )
            row["heavy"][name] = summarize(value[mode_index], regimes == 3)
        row["spatial_fold_gain_vs_source_atom_fit"] = fold_gain[mode]
        row["positive_folds"] = int(np.sum(np.asarray(fold_gain[mode]) > 0.0))
        summary.append(row)

    source_row = next(row for row in summary if row["mode"] == "source_atom_fit")
    global_row = next(row for row in summary if row["mode"] == "global_complex_scalar")
    predicted = next(row for row in summary if row["mode"] == "predicted_complex_scalar")
    magnitude = next(row for row in summary if row["mode"] == "predicted_magnitude_only")
    shuffled = next(row for row in summary if row["mode"] == "shuffled_label_control")
    predicted_gain = float(np.mean(fold_gain["predicted_complex_scalar"]))
    magnitude_gain = float(np.mean(fold_gain["predicted_magnitude_only"]))
    shuffled_gain = float(np.mean(fold_gain["shuffled_label_control"]))
    global_gain = float(np.mean(fold_gain["global_complex_scalar"]))
    predicted_vs_global_folds = (
        np.asarray(fold_gain["predicted_complex_scalar"])
        - np.asarray(fold_gain["global_complex_scalar"])
    )
    magnitude_vs_global_folds = (
        np.asarray(fold_gain["predicted_magnitude_only"])
        - np.asarray(fold_gain["global_complex_scalar"])
    )
    predicted_spectral_gain = float(
        0.5
        * (
            predicted["overall"]["pdp"]["mean"]
            + predicted["overall"]["leaked_full"]["mean"]
            - source_row["overall"]["pdp"]["mean"]
            - source_row["overall"]["leaked_full"]["mean"]
        )
    )
    magnitude_spectral_gain = float(
        0.5
        * (
            magnitude["overall"]["pdp"]["mean"]
            + magnitude["overall"]["leaked_full"]["mean"]
            - source_row["overall"]["pdp"]["mean"]
            - source_row["overall"]["leaked_full"]["mean"]
        )
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        POINT_METRICS,
        indices=point_ids,
        regimes=regimes,
        fold_codes=fold_codes,
        mode_names=np.asarray(MODE_NAMES),
        predicted_complex=predicted_complex.astype(np.complex64),
        predicted_magnitude=np.abs(predicted_magnitude).astype(np.float32),
        **metrics,
    )
    report = {
        "method": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "A shared function of point geometry, path delay/angle/depth, and Sionna complex "
            "port coefficients can predict the per-path complex scalar needed to approach "
            "official H at spatially unseen points."
        ),
        "new_information": (
            "Sionna and official H are decomposed into the same source-selected 32 complex "
            "delay-H atoms, exposing path-level complex correction labels rather than only "
            "aggregate PAS/PDP residuals."
        ),
        "point_count": int(len(point_ids)),
        "path_limit": PATH_LIMIT,
        "split": "v501 five spatial folds",
        "model": {
            "type": "ExtraTrees multi-output path scalar regressor",
            "trees": args.trees,
            "min_samples_leaf": args.min_samples_leaf,
            "max_features": args.max_features,
            "targets": ["log magnitude", "phase cosine", "phase sine"],
            "selection": "atoms selected only from Sionna CFR energy",
        },
        "coefficient_fit": {
            "source_mean_coherence": float(np.mean(data["source_fit_coherence"])),
            "official_mean_coherence": float(np.mean(data["official_fit_coherence"])),
        },
        "fold_details": fold_details,
        "summary": summary,
        "decision": {
            "global_complex_gain_vs_source": global_gain,
            "global_complex_positive_folds": global_row["positive_folds"],
            "predicted_complex_gain_vs_source": predicted_gain,
            "predicted_complex_positive_folds": predicted["positive_folds"],
            "predicted_complex_gain_vs_global": float(
                np.mean(predicted_vs_global_folds)
            ),
            "predicted_complex_positive_folds_vs_global": int(
                np.sum(predicted_vs_global_folds > 0.0)
            ),
            "predicted_complex_spectral_gain_vs_source": predicted_spectral_gain,
            "predicted_magnitude_gain_vs_source": magnitude_gain,
            "predicted_magnitude_positive_folds": magnitude["positive_folds"],
            "predicted_magnitude_gain_vs_global": float(
                np.mean(magnitude_vs_global_folds)
            ),
            "predicted_magnitude_positive_folds_vs_global": int(
                np.sum(magnitude_vs_global_folds > 0.0)
            ),
            "predicted_magnitude_spectral_gain_vs_source": magnitude_spectral_gain,
            "shuffled_gain_vs_source": shuffled_gain,
            "independent_margin_vs_shuffled": predicted_gain - shuffled_gain,
            "global_amplitude_mismatch_supported": bool(
                global_gain >= 0.02 and global_row["positive_folds"] == 5
            ),
            "shared_complex_correction_supported": bool(
                np.mean(predicted_vs_global_folds) >= 0.005
                and np.sum(predicted_vs_global_folds > 0.0) >= 4
                and predicted_spectral_gain >= 0.005
            ),
            "shared_magnitude_correction_supported": bool(
                np.mean(magnitude_vs_global_folds) >= 0.005
                and np.sum(magnitude_vs_global_folds > 0.0) >= 4
                and magnitude_spectral_gain >= 0.005
            ),
            "gate": (
                "path-specific gain over global scalar >= 0.005, at least 4/5 folds "
                "positive versus global, and spectral PAS/PDP gain >= 0.005"
            ),
        },
        "outputs": {
            "coefficient_cache": PATHS.relative(COEFFICIENT_CACHE),
            "point_metrics": PATHS.relative(POINT_METRICS),
            "report": PATHS.relative(REPORT),
            "submission_zip": None,
        },
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "source_fit_coherence": report["coefficient_fit"]["source_mean_coherence"],
                "official_fit_coherence": report["coefficient_fit"]["official_mean_coherence"],
                "magnitude_gain": magnitude_gain,
                "magnitude_gain_vs_global": report["decision"][
                    "predicted_magnitude_gain_vs_global"
                ],
                "complex_gain": predicted_gain,
                "complex_gain_vs_global": report["decision"][
                    "predicted_complex_gain_vs_global"
                ],
                "complex_positive_folds": predicted["positive_folds"],
                "shuffled_gain": shuffled_gain,
                "supported": report["decision"]["shared_complex_correction_supported"],
                "report": PATHS.relative(REPORT),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
