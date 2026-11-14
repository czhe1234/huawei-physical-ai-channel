#!/usr/bin/env python3
"""Apply v325 phase transport only inside geometry-supported path subspaces."""

from __future__ import annotations

import json

import numpy as np
from scipy.spatial import cKDTree

import analyze_v273_planar_barrier_kernel as v273
import analyze_v297_continuous_delay_component_transport as v297
import analyze_v300_path_gain_fingerprint_oracle as v300
import analyze_v304_inverse_scatterer_identifiability as v304
import analyze_v305_pointcloud_path_identity_match as v305
import analyze_v306_matched_patch_energy_transformer as v306
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v325_path_phase_transport_identifiability as v325
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS
from spectrum_projection import MetricAccumulator


VERSION_NUMBER = 326
VERSION = "v326_complex_path_subspace_transport"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
GAIN_CACHE = OUT_DIR / "v326_matched_path_gain_tensors.npz"
OUTPUT = OUT_DIR / "v326_Round1_Test_Channel.npy"
REPORT = REPORT_DIR / f"{VERSION}.json"

BASE_TEST = PATHS.cache_dir(284) / "v284_Round1_Test_Channel.npy"
K_BASE = 16
BASE_POWER = 2.0
BASE_BETA = 140.3
MAX_PATHS = 8
MAX_PAIR_DISTANCE = 25.0
PATH_DISTANCE_SCALE = 12.0
RIDGE = 1e-3
ALPHA_GRID = np.asarray([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)
MIN_SCORE_DELTA = 0.01
MAX_SPECTRAL_DEGRADATION = 0.001


def recover_matched_gain_tensors(rows: dict[str, np.ndarray]) -> np.ndarray:
    if GAIN_CACHE.is_file():
        data = np.load(GAIN_CACHE)
        if not np.array_equal(data["point"], rows["point"]):
            raise ValueError("v326 gain cache point ordering does not match v325 rows")
        if not np.array_equal(data["component"], rows["component"]):
            raise ValueError("v326 gain cache component ordering does not match v325 rows")
        return data["gain"].astype(np.complex64)

    component = np.load(v297.COMPONENT_CACHE)
    delay = component["delay"].astype(np.float64)
    valid = component["valid"].astype(bool)
    channel = np.load(PATHS.round1 / "Round1_Train_Channel.npy", mmap_mode="r")
    gain_out = np.zeros((len(rows["point"]), 2, 16, 8, 4), dtype=np.complex64)
    frequency = np.arange(v297.N_DELAY, dtype=np.float64)
    by_point: dict[int, list[int]] = {}
    for row, point_id in enumerate(rows["point"]):
        by_point.setdefault(int(point_id), []).append(row)

    for progress, (point_id, output_rows) in enumerate(by_point.items(), start=1):
        active = np.flatnonzero(valid[point_id])
        tau = delay[point_id, active]
        steering = np.exp(
            -2j * np.pi * frequency[:, None] * tau[None, :] / v297.N_DELAY
        ).astype(np.complex64)
        h = np.asarray(channel[point_id], dtype=np.complex64).reshape(-1, v297.N_DELAY)
        gram = steering.conj().T @ steering
        regularized = gram + np.eye(len(active), dtype=np.complex64) * (
            v300.RIDGE * np.trace(gram).real / max(len(active), 1)
        )
        gain = np.linalg.solve(regularized, steering.conj().T @ h.T)
        tensor = gain.reshape(len(active), 2, 16, 8, 4)
        lookup = {int(component_id): index for index, component_id in enumerate(active)}
        for output_row in output_rows:
            component_id = int(rows["component"][output_row])
            gain_out[output_row] = tensor[lookup[component_id]]
        if progress % 200 == 0 or progress == len(by_point):
            print(f"v326 matched tensors {progress}/{len(by_point)}", flush=True)

    np.savez_compressed(
        GAIN_CACHE,
        point=rows["point"],
        component=rows["component"],
        gain=gain_out,
    )
    return gain_out


def estimate_beta(
    rows: dict[str, np.ndarray], fit_points: np.ndarray, train_xy: np.ndarray
) -> float:
    left, right, _distance = v325.nearest_patch_pairs(
        rows, fit_points, fit_points, train_xy
    )
    phase, length, weight = v325.pair_values(rows, left, right)
    return v325.fit_beta(phase, length, weight)


def base_predictions(
    query_pos: np.ndarray,
    bank_ids: np.ndarray,
    train_pos: np.ndarray,
    channel: np.ndarray,
) -> np.ndarray:
    tree = cKDTree(train_pos[bank_ids, :2])
    distance, local = tree.query(query_pos[:, :2], k=min(K_BASE, len(bank_ids)))
    if distance.ndim == 1:
        distance = distance[:, None]
        local = local[:, None]
    neighbor_ids = bank_ids[local]
    weight = 1.0 / np.maximum(distance, 0.05) ** BASE_POWER
    weight /= weight.sum(axis=1, keepdims=True)
    bs = np.asarray([50.0, 0.0, 25.0], dtype=np.float64)
    train_distance = np.linalg.norm(train_pos - bs[None, :], axis=1)
    query_distance = np.linalg.norm(query_pos - bs[None, :], axis=1)
    output = np.empty(
        (len(query_pos), channel.shape[1], channel.shape[2], channel.shape[3]),
        dtype=np.complex64,
    )
    for row in range(len(query_pos)):
        phase = np.exp(
            1j
            * BASE_BETA
            * (train_distance[neighbor_ids[row]] - query_distance[row])
        ).astype(np.complex64)
        coefficient = (weight[row] * phase).astype(np.complex64)
        neighbors = np.asarray(channel[neighbor_ids[row]], dtype=np.complex64)
        output[row] = np.sum(
            coefficient[:, None, None, None] * neighbors,
            axis=0,
            dtype=np.complex64,
        )
    return output


class PatchBank:
    def __init__(
        self,
        rows: dict[str, np.ndarray],
        gain: np.ndarray,
        fit_points: np.ndarray,
        train_xy: np.ndarray,
    ) -> None:
        self.rows = rows
        self.gain = gain
        self.train_xy = train_xy
        bank_rows = np.flatnonzero(np.isin(rows["point"], fit_points))
        self.patch: dict[int, tuple[np.ndarray, cKDTree]] = {}
        for patch_id in np.unique(rows["patch"][bank_rows]):
            selected = bank_rows[rows["patch"][bank_rows] == patch_id]
            self.patch[int(patch_id)] = (
                selected,
                cKDTree(train_xy[rows["point"][selected]]),
            )

    def candidates(
        self,
        query_xy: np.ndarray,
        candidate: dict[str, np.ndarray],
        query_row: int,
    ) -> list[tuple[float, int, int, float]]:
        valid = np.flatnonzero(candidate["score"][query_row] > 0.0)
        selected: list[tuple[float, int, int, float]] = []
        used_patch: set[int] = set()
        for candidate_id in valid:
            patch_id = int(candidate["patch_id"][query_row, candidate_id])
            if patch_id in used_patch or patch_id not in self.patch:
                continue
            bank_rows, tree = self.patch[patch_id]
            distance, local = tree.query(query_xy, k=1)
            if distance > MAX_PAIR_DISTANCE:
                continue
            bank_row = int(bank_rows[int(local)])
            score = float(
                self.rows["mass"][bank_row]
                * np.exp(-float(distance) / PATH_DISTANCE_SCALE)
            )
            selected.append((score, int(candidate_id), bank_row, float(distance)))
            used_patch.add(patch_id)
        selected.sort(reverse=True)
        return selected[:MAX_PATHS]


def project_one(
    base: np.ndarray,
    query_xy: np.ndarray,
    candidate: dict[str, np.ndarray],
    query_row: int,
    bank: PatchBank,
    beta: float,
) -> tuple[np.ndarray, dict[str, float]]:
    selected = bank.candidates(query_xy, candidate, query_row)
    if not selected:
        return base.copy(), {"path_count": 0.0, "distance_mean": 0.0}
    candidate_ids = np.asarray([item[1] for item in selected], dtype=np.int64)
    bank_rows = np.asarray([item[2] for item in selected], dtype=np.int64)
    distance = np.asarray([item[3] for item in selected], dtype=np.float32)
    path_length = candidate["path_length"][query_row, candidate_ids].astype(np.float64)
    calibration = json.loads(v304.REPORT.read_text(encoding="utf-8"))["los_calibration"]
    delay = (
        float(calibration["intercept_bins"])
        + float(calibration["slope_bins_per_meter"]) * path_length
    )
    frequency = np.arange(v297.N_DELAY, dtype=np.float64)
    steering = np.exp(
        -2j * np.pi * frequency[:, None] * delay[None, :] / v297.N_DELAY
    ).astype(np.complex64)
    h = base.reshape(-1, v297.N_DELAY)
    gram = steering.conj().T @ steering
    regularized = gram + np.eye(len(selected), dtype=np.complex64) * (
        RIDGE * np.trace(gram).real / max(len(selected), 1)
    )
    base_gain = np.linalg.solve(regularized, steering.conj().T @ h.T)

    transported = bank.gain[bank_rows].reshape(len(selected), -1).copy()
    delta_length = path_length - bank.rows["length"][bank_rows]
    transported *= np.exp(-1j * beta * delta_length)[:, None].astype(np.complex64)
    reflection = candidate["reflection_point"][query_row, candidate_ids]
    query_h = v305.expected_h_bin(reflection).astype(np.float32)
    delta_h = np.remainder(
        query_h - bank.rows["h_center"][bank_rows] + 8.0, 16.0
    ) - 8.0
    h_coordinate = np.arange(16, dtype=np.float32) - 7.5
    h_phase = np.exp(
        1j * 2.0 * np.pi * delta_h[:, None] * h_coordinate[None, :] / 16.0
    ).astype(np.complex64)
    transported = transported.reshape(len(selected), 2, 16, 8, 4)
    transported *= h_phase[:, None, :, None, None]
    transported = transported.reshape(len(selected), -1)

    base_norm = np.linalg.norm(base_gain, axis=1)
    transported_norm = np.linalg.norm(transported, axis=1)
    transported *= (
        base_norm / np.maximum(transported_norm, 1e-20)
    )[:, None].astype(np.float32)
    correction = (steering @ (transported - base_gain)).T.reshape(base.shape)
    output = (base + correction).astype(np.complex64)
    base_power = float(np.sum(np.abs(base) ** 2, dtype=np.float64))
    output_power = float(np.sum(np.abs(output) ** 2, dtype=np.float64))
    if base_power > 0.0 and output_power > 0.0:
        output *= np.float32(np.sqrt(base_power / output_power))
    return output, {
        "path_count": float(len(selected)),
        "distance_mean": float(np.mean(distance)),
    }


def corrected_predictions(
    base: np.ndarray,
    query_pos: np.ndarray,
    candidate: dict[str, np.ndarray],
    bank: PatchBank,
    beta: float,
) -> tuple[np.ndarray, dict[str, object]]:
    output = np.empty_like(base)
    path_count = np.zeros(len(base), dtype=np.float32)
    distance = np.zeros(len(base), dtype=np.float32)
    for row in range(len(base)):
        output[row], diagnostics = project_one(
            base[row], query_pos[row, :2], candidate, row, bank, beta
        )
        path_count[row] = diagnostics["path_count"]
        distance[row] = diagnostics["distance_mean"]
    return output, {
        "path_count": summarize(path_count),
        "active_fraction": float(np.mean(path_count > 0)),
        "distance_mean_active": summarize(distance[path_count > 0]),
    }


def blend(base: np.ndarray, corrected: np.ndarray, alpha: float) -> np.ndarray:
    output = (base + float(alpha) * (corrected - base)).astype(np.complex64)
    base_power = np.sum(np.abs(base) ** 2, axis=(1, 2, 3), dtype=np.float64)
    output_power = np.sum(np.abs(output) ** 2, axis=(1, 2, 3), dtype=np.float64)
    scale = np.sqrt(base_power / np.maximum(output_power, 1e-30)).astype(np.float32)
    output *= scale[:, None, None, None]
    return output


def score(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    accumulator = MetricAccumulator("sumpol")
    for start in range(0, len(prediction), 4):
        end = min(start + 4, len(prediction))
        accumulator.add(prediction[start:end], np.asarray(truth[start:end]))
    return accumulator.result()


def choose_alpha(
    base: np.ndarray, corrected: np.ndarray, truth: np.ndarray
) -> tuple[float, list[dict[str, float]]]:
    rows = []
    for alpha in ALPHA_GRID:
        metrics = score(blend(base, corrected, float(alpha)), truth)
        rows.append({"alpha": float(alpha), **metrics})
    best = max(rows, key=lambda row: row["score"])
    return float(best["alpha"]), rows


def slice_candidates(
    candidate: dict[str, np.ndarray], ids: np.ndarray
) -> dict[str, np.ndarray]:
    return {key: value[ids] for key, value in candidate.items()}


def main() -> None:
    required = (
        v297.COMPONENT_CACHE,
        v305.MATCH_CACHE,
        v306.TRAIN_CANDIDATES,
        v306.TEST_CANDIDATES,
        BASE_TEST,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    gain_data = v325.recover_center_gains()
    rows = v325.matched_rows(gain_data)
    gain = recover_matched_gain_tensors(rows)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    channel = np.load(PATHS.round1 / "Round1_Train_Channel.npy", mmap_mode="r")
    component = np.load(v297.COMPONENT_CACHE)
    valid_ids = np.flatnonzero(component["valid"].any(axis=1)).astype(np.int64)
    folds = v315.disjoint_folds(valid_ids, train_pos[:, :2])
    train_data = np.load(v306.TRAIN_CANDIDATES)
    train_candidate = {key: train_data[key] for key in train_data.files}

    fold_rows = []
    selected_alphas = []
    for fold, outer_ids in enumerate(folds):
        outer_fit = np.setdiff1d(valid_ids, outer_ids)
        inner_ids = v273.matched_subset(
            outer_fit,
            train_pos[:, :2],
            test_pos[:, :2],
            min(240, max(len(outer_fit) // 5, 1)),
            32600 + fold,
        )
        inner_fit = np.setdiff1d(outer_fit, inner_ids)
        inner_beta = estimate_beta(rows, inner_fit, train_pos[:, :2])
        inner_base = base_predictions(train_pos[inner_ids], inner_fit, train_pos, channel)
        inner_bank = PatchBank(rows, gain, inner_fit, train_pos[:, :2])
        inner_corrected, inner_diagnostics = corrected_predictions(
            inner_base,
            train_pos[inner_ids],
            slice_candidates(train_candidate, inner_ids),
            inner_bank,
            inner_beta,
        )
        alpha, alpha_rows = choose_alpha(
            inner_base, inner_corrected, np.asarray(channel[inner_ids])
        )
        selected_alphas.append(alpha)

        outer_beta = estimate_beta(rows, outer_fit, train_pos[:, :2])
        outer_base = base_predictions(train_pos[outer_ids], outer_fit, train_pos, channel)
        outer_bank = PatchBank(rows, gain, outer_fit, train_pos[:, :2])
        outer_corrected, outer_diagnostics = corrected_predictions(
            outer_base,
            train_pos[outer_ids],
            slice_candidates(train_candidate, outer_ids),
            outer_bank,
            outer_beta,
        )
        truth = np.asarray(channel[outer_ids])
        base_metrics = score(outer_base, truth)
        prediction_metrics = score(blend(outer_base, outer_corrected, alpha), truth)
        fold_rows.append(
            {
                "fold": fold,
                "outer_fit_count": int(len(outer_fit)),
                "outer_query_count": int(len(outer_ids)),
                "inner_fit_count": int(len(inner_fit)),
                "inner_query_count": int(len(inner_ids)),
                "inner_beta": inner_beta,
                "outer_beta": outer_beta,
                "selected_alpha": alpha,
                "inner_alpha_rows": alpha_rows,
                "inner_diagnostics": inner_diagnostics,
                "outer_diagnostics": outer_diagnostics,
                "base": base_metrics,
                "prediction": prediction_metrics,
                "score_delta": prediction_metrics["score"] - base_metrics["score"],
                "pas_delta": prediction_metrics["pas"] - base_metrics["pas"],
                "pdp_delta": prediction_metrics["pdp"] - base_metrics["pdp"],
                "inv_nmse_delta": prediction_metrics["inv_nmse"]
                - base_metrics["inv_nmse"],
            }
        )
        print(
            f"v326 fold={fold} alpha={alpha:.2f} beta={outer_beta:.3f} "
            f"score={fold_rows[-1]['score_delta']:+.6f} "
            f"nmse={fold_rows[-1]['inv_nmse_delta']:+.6f}",
            flush=True,
        )

    score_delta = float(np.mean([row["score_delta"] for row in fold_rows]))
    pas_delta = float(np.mean([row["pas_delta"] for row in fold_rows]))
    pdp_delta = float(np.mean([row["pdp_delta"] for row in fold_rows]))
    inv_nmse_delta = float(np.mean([row["inv_nmse_delta"] for row in fold_rows]))
    fold_wins = int(sum(row["score_delta"] > 0.0 for row in fold_rows))
    passed = (
        score_delta >= MIN_SCORE_DELTA
        and fold_wins >= 4
        and pas_delta >= -MAX_SPECTRAL_DEGRADATION
        and pdp_delta >= -MAX_SPECTRAL_DEGRADATION
    )

    deployment = None
    if passed:
        final_alpha = float(np.median(selected_alphas))
        final_beta = estimate_beta(rows, valid_ids, train_pos[:, :2])
        test_data = np.load(v306.TEST_CANDIDATES)
        test_candidate = {key: test_data[key] for key in test_data.files}
        test_base = np.load(BASE_TEST, mmap_mode="r")
        full_bank = PatchBank(rows, gain, valid_ids, train_pos[:, :2])
        test_corrected, diagnostics = corrected_predictions(
            np.asarray(test_base), test_pos, test_candidate, full_bank, final_beta
        )
        prediction = blend(np.asarray(test_base), test_corrected, final_alpha)
        np.save(OUTPUT, prediction.astype(np.complex64))
        deployment = {
            "alpha": final_alpha,
            "beta": final_beta,
            "base": PATHS.relative(BASE_TEST),
            "prediction": PATHS.relative(OUTPUT),
            "shape": list(prediction.shape),
            "dtype": str(prediction.dtype),
            "finite": bool(np.isfinite(prediction).all()),
            "diagnostics": diagnostics,
        }

    report = {
        "method": VERSION,
        "status": "official_selection_passed" if passed else "analysis_only_no_candidate",
        "primary_hypothesis": (
            "Path-length phase transport improves a full complex official-H predictor when "
            "restricted to base-supported, same-wall one-bounce delay subspaces."
        ),
        "compliance_boundary": {
            "uses": [
                "official train H and train/test coordinates",
                "official-pointcloud wall candidate identity, delay, and departure angle",
                "fit-only nearest matched path complex tensors",
                "strict official-only v284 test carrier after OOF selection",
            ],
            "does_not_use": [
                "test truth, synthetic channel labels, or teacher outputs",
                "RT path coefficients or restored scene materials",
                "online score for parameter selection",
            ],
        },
        "parameters": {
            "base_k": K_BASE,
            "base_power": BASE_POWER,
            "base_beta": BASE_BETA,
            "max_paths": MAX_PATHS,
            "max_pair_distance_m": MAX_PAIR_DISTANCE,
            "path_distance_scale_m": PATH_DISTANCE_SCALE,
            "ridge": RIDGE,
            "alpha_grid": ALPHA_GRID.tolist(),
        },
        "fold_rows": fold_rows,
        "oof": {
            "score_delta": score_delta,
            "pas_delta": pas_delta,
            "pdp_delta": pdp_delta,
            "inv_nmse_delta": inv_nmse_delta,
            "fold_wins": fold_wins,
        },
        "decision_rule": {
            "score_delta": f">={MIN_SCORE_DELTA}",
            "fold_wins": ">=4/5",
            "pas_delta": f">=-{MAX_SPECTRAL_DEGRADATION}",
            "pdp_delta": f">=-{MAX_SPECTRAL_DEGRADATION}",
            "passed": bool(passed),
        },
        "deployment": deployment,
        "outputs": {
            "gain_tensors": PATHS.relative(GAIN_CACHE),
            "prediction": PATHS.relative(OUTPUT) if OUTPUT.is_file() else None,
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"oof": report["oof"], "decision": report["decision_rule"]}, indent=2))


if __name__ == "__main__":
    main()
