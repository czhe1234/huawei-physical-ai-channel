#!/usr/bin/env python3
"""Infer official Sionna system parameters from high-confidence LOS channels."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution, least_squares
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupKFold

from project_paths import PATHS


VERSION = "v534_los_official_sionna_configuration_inverse"
REPORT_DIR = PATHS.version_report_dir(534)
REPORT = REPORT_DIR / f"{VERSION}.json"
CACHE_DIR = PATHS.cache_dir(534)
METRICS_CACHE = CACHE_DIR / "v534_los_point_metrics.npz"

BS_POSITION = np.asarray([50.0, 0.0, 25.0], dtype=np.float64)
LIGHT_SPEED = 299_792_458.0
N_SUBCARRIERS = 192
TILE_SIZE_M = 55.0


@dataclass(frozen=True)
class LinearFit:
    intercept: float
    coefficient: np.ndarray

    def predict(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        if value.ndim == 1 and self.coefficient.size == 1:
            return self.intercept + value * self.coefficient[0]
        return self.intercept + value @ self.coefficient


def weighted_median(value: np.ndarray, weight: np.ndarray) -> float:
    order = np.argsort(value)
    sorted_value = np.asarray(value, dtype=np.float64)[order]
    sorted_weight = np.asarray(weight, dtype=np.float64)[order]
    cutoff = 0.5 * np.sum(sorted_weight)
    return float(sorted_value[np.searchsorted(np.cumsum(sorted_weight), cutoff)])


def weighted_mae(value: np.ndarray, weight: np.ndarray) -> float:
    return float(np.sum(np.asarray(weight) * np.abs(value)) / np.sum(weight))


def robust_linear_fit(
    x: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    *,
    intercept: bool = True,
    f_scale: float = 0.02,
) -> LinearFit:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weight = np.maximum(np.asarray(weight, dtype=np.float64), 1e-8)
    if x.ndim == 1:
        x = x[:, None]
    design = np.column_stack([np.ones(len(x)), x]) if intercept else x
    initial = np.linalg.lstsq(design * np.sqrt(weight[:, None]), y * np.sqrt(weight), rcond=None)[0]

    def residual(parameter: np.ndarray) -> np.ndarray:
        return np.sqrt(weight) * (design @ parameter - y)

    result = least_squares(residual, initial, loss="soft_l1", f_scale=f_scale, max_nfev=400)
    if intercept:
        return LinearFit(float(result.x[0]), np.asarray(result.x[1:], dtype=np.float64))
    return LinearFit(0.0, np.asarray(result.x, dtype=np.float64))


def adjacent_frequency_statistics(channel: np.ndarray, batch_size: int = 24) -> tuple[np.ndarray, np.ndarray]:
    phase = np.empty(len(channel), dtype=np.float64)
    coherence = np.empty(len(channel), dtype=np.float64)
    for start in range(0, len(channel), batch_size):
        stop = min(start + batch_size, len(channel))
        value = np.asarray(channel[start:stop])
        product = np.conj(value[..., :-1]) * value[..., 1:]
        summed = np.sum(product, axis=(1, 2, 3), dtype=np.complex128)
        denominator = np.sum(np.abs(product), axis=(1, 2, 3), dtype=np.float64)
        phase[start:stop] = -np.angle(summed) / (2.0 * np.pi)
        coherence[start:stop] = np.abs(summed) / np.maximum(denominator, 1e-30)
    return phase, coherence


def extract_continuous_los(
    channel: np.ndarray,
    delay_cycles: np.ndarray,
    batch_size: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    coefficient = np.empty((len(channel), 256, 4), dtype=np.complex64)
    total_energy = np.empty(len(channel), dtype=np.float64)
    carrier = np.arange(N_SUBCARRIERS, dtype=np.float64)
    for start in range(0, len(channel), batch_size):
        stop = min(start + batch_size, len(channel))
        value = np.asarray(channel[start:stop])
        steering = np.exp(2j * np.pi * delay_cycles[start:stop, None] * carrier[None, :])
        coefficient[start:stop] = np.einsum(
            "nurf,nf->nur", value, steering, optimize=True
        ) / N_SUBCARRIERS
        total_energy[start:stop] = np.mean(
            np.sum(np.abs(value) ** 2, axis=(1, 2), dtype=np.float64), axis=-1
        )
    los_energy = np.sum(np.abs(coefficient) ** 2, axis=(1, 2), dtype=np.float64)
    return coefficient.reshape(len(channel), 2, 16, 8, 2, 2), los_energy / np.maximum(total_energy, 1e-30)


def adjacent_array_statistics(value: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    left = [slice(None)] * value.ndim
    right = [slice(None)] * value.ndim
    left[axis] = slice(None, -1)
    right[axis] = slice(1, None)
    product = np.conj(value[tuple(left)]) * value[tuple(right)]
    reduce_axes = tuple(range(1, value.ndim))
    summed = np.sum(product, axis=reduce_axes, dtype=np.complex128)
    denominator = np.sum(np.abs(product), axis=reduce_axes, dtype=np.float64)
    return np.angle(summed) / np.pi, np.abs(summed) / np.maximum(denominator, 1e-30)


def fit_array_axis(
    unit_direction: np.ndarray,
    observed_phase: np.ndarray,
    weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fit = robust_linear_fit(
        unit_direction,
        observed_phase,
        weight,
        intercept=False,
        f_scale=0.025,
    )
    prediction = fit.predict(unit_direction)
    return fit.coefficient, prediction


def circular_residual(prediction: np.ndarray, observed: np.ndarray) -> np.ndarray:
    return (np.asarray(prediction) - np.asarray(observed) + 1.0) % 2.0 - 1.0


def fit_wrapped_vertical_axis(
    unit_direction: np.ndarray,
    observed_phase: np.ndarray,
    weight: np.ndarray,
    horizontal_vector: np.ndarray,
    *,
    fixed_spacing_wavelengths: float | None = None,
    seed: int = 534,
) -> tuple[np.ndarray, float, float]:
    h_axis = horizontal_vector / np.linalg.norm(horizontal_vector)
    vertical_seed = np.asarray([0.0, 0.0, 1.0])
    first = vertical_seed - np.dot(vertical_seed, h_axis) * h_axis
    first /= np.linalg.norm(first)
    second = np.cross(h_axis, first)
    weight = np.maximum(np.asarray(weight, dtype=np.float64), 1e-8)

    def objective(parameter: np.ndarray) -> float:
        if fixed_spacing_wavelengths is None:
            spacing, angle = parameter
        else:
            spacing, angle = fixed_spacing_wavelengths, parameter[0]
        vector = 2.0 * spacing * (math.cos(angle) * first + math.sin(angle) * second)
        residual = unit_direction @ vector - observed_phase
        return float(np.sum(weight * (1.0 - np.cos(np.pi * residual))) / np.sum(weight))

    bounds = [(0.1, 8.0), (0.0, 2.0 * np.pi)] if fixed_spacing_wavelengths is None else [(0.0, 2.0 * np.pi)]
    result = differential_evolution(
        objective,
        bounds,
        seed=seed,
        popsize=30,
        maxiter=600,
        tol=1e-10,
        polish=True,
        workers=1,
    )
    if fixed_spacing_wavelengths is None:
        spacing, angle = float(result.x[0]), float(result.x[1])
    else:
        spacing, angle = float(fixed_spacing_wavelengths), float(result.x[0])
    vector = 2.0 * spacing * (math.cos(angle) * first + math.sin(angle) * second)
    residual = circular_residual(unit_direction @ vector, observed_phase)
    rmse = float(math.sqrt(np.sum(weight * residual**2) / np.sum(weight)))
    return vector, float(result.fun), rmse


def orthogonalize_axes(horizontal: np.ndarray, vertical: np.ndarray, unit_direction: np.ndarray) -> dict[str, object]:
    horizontal_spacing = float(np.linalg.norm(horizontal) / 2.0)
    vertical_spacing = float(np.linalg.norm(vertical) / 2.0)
    h_axis = horizontal / np.linalg.norm(horizontal)
    v_residual = vertical - np.dot(vertical, h_axis) * h_axis
    v_axis = v_residual / np.linalg.norm(v_residual)
    cross = np.cross(h_axis, v_axis)
    median_direction = np.median(unit_direction, axis=0)
    front_sign = 1.0 if np.dot(cross, median_direction) >= 0.0 else -1.0
    boresight = front_sign * cross
    azimuth = math.degrees(math.atan2(boresight[1], boresight[0]))
    tilt = math.degrees(math.asin(np.clip(boresight[2], -1.0, 1.0)))
    raw_angle = math.degrees(
        math.acos(
            np.clip(
                np.dot(horizontal, vertical) / np.linalg.norm(horizontal) / np.linalg.norm(vertical),
                -1.0,
                1.0,
            )
        )
    )
    return {
        "horizontal_fitted_vector": horizontal.tolist(),
        "vertical_fitted_vector": vertical.tolist(),
        "horizontal_axis_unit": h_axis.tolist(),
        "vertical_axis_unit": v_axis.tolist(),
        "raw_axis_angle_deg": raw_angle,
        "horizontal_spacing_wavelengths": horizontal_spacing,
        "vertical_spacing_wavelengths": vertical_spacing,
        "front_normal_choice": "cross(H,V)" if front_sign > 0 else "-cross(H,V)",
        "boresight_unit": boresight.tolist(),
        "boresight_azimuth_deg": azimuth,
        "boresight_tilt_deg": tilt,
        "front_back_note": (
            "Spatial phase fixes the array plane but has a front/back ambiguity; the reported normal is the side "
            "containing the high-confidence LOS receivers."
        ),
    }


def rotate_z(vector: np.ndarray, angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    rotation = np.asarray(
        [[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    return rotation @ vector


def make_groups(position: np.ndarray) -> np.ndarray:
    xy = position[:, :2]
    tile = np.floor((xy - np.min(xy, axis=0, keepdims=True)) / TILE_SIZE_M).astype(np.int32)
    width = int(np.max(tile[:, 1])) + 1
    return tile[:, 0] * width + tile[:, 1]


def audit_tensor_layouts(
    coefficient: np.ndarray,
    selected_local: np.ndarray,
    unit_direction: np.ndarray,
    weight: np.ndarray,
) -> dict[str, object]:
    flat = coefficient.reshape(len(coefficient), 256, 4)
    dimensions = {"P": 2, "H": 16, "V": 8}
    orders = (
        ("P", "H", "V"),
        ("P", "V", "H"),
        ("H", "P", "V"),
        ("H", "V", "P"),
        ("V", "P", "H"),
        ("V", "H", "P"),
    )
    result: dict[str, dict[str, float]] = {}
    for order_index, order in enumerate(orders):
        reshaped = flat.reshape(
            (len(flat),) + tuple(dimensions[name] for name in order) + (4,)
        )
        permutation = (
            0,
            1 + order.index("P"),
            1 + order.index("H"),
            1 + order.index("V"),
            4,
        )
        candidate = reshaped.transpose(permutation)[selected_local]
        h_phase, h_coherence = adjacent_array_statistics(candidate, axis=2)
        v_phase, v_coherence = adjacent_array_statistics(candidate, axis=3)
        h_vector, _ = fit_array_axis(unit_direction, h_phase, weight)
        v_vector, _, v_rmse = fit_wrapped_vertical_axis(
            unit_direction,
            v_phase,
            weight,
            h_vector,
            seed=600 + order_index,
        )
        h_error = circular_residual(unit_direction @ h_vector, h_phase)
        h_rmse = float(math.sqrt(np.sum(weight * h_error**2) / np.sum(weight)))
        result["".join(order)] = {
            "horizontal_coherence_median": float(np.median(h_coherence)),
            "vertical_coherence_median": float(np.median(v_coherence)),
            "horizontal_spacing_wavelengths": float(np.linalg.norm(h_vector) / 2.0),
            "vertical_spacing_wavelengths": float(np.linalg.norm(v_vector) / 2.0),
            "horizontal_circular_rmse_pi": h_rmse,
            "vertical_circular_rmse_pi": v_rmse,
            "combined_rmse_pi": float(math.sqrt(0.5 * (h_rmse**2 + v_rmse**2))),
        }
    best = min(result, key=lambda name: result[name]["combined_rmse_pi"])
    return {
        "candidate_layouts": result,
        "best_layout": best,
        "interpretation": (
            "PHV means flat TX index = [polarization,16 horizontal,8 vertical]. The selected layout must fit "
            "both physical axes; a low error on only one transposed axis is insufficient."
        ),
    }


def audit_receiver_layout(
    coefficient: np.ndarray,
    selected_local: np.ndarray,
) -> dict[str, object]:
    current = coefficient[selected_local]
    transposed = current.transpose(0, 1, 2, 3, 5, 4)
    result = {}
    for name, candidate in (("polarization_then_vertical", current), ("vertical_then_polarization", transposed)):
        phase, coherence = adjacent_array_statistics(candidate, axis=5)
        result[name] = {
            "abs_tracking_phase_median_pi": float(np.median(np.abs(phase))),
            "tracking_phase_coherence_median": float(np.median(coherence)),
        }
    return {
        "candidate_layouts": result,
        "best_layout": min(
            result,
            key=lambda name: result[name]["abs_tracking_phase_median_pi"],
        ),
        "interpretation": (
            "A source-tracking two-element spatial axis should have near-zero LOS phase. This distinguishes the "
            "RX vertical element index from the polarization index, although it does not assign semantic V/H "
            "names to the polarization channels."
        ),
    }


def build_port_vectors(
    coefficient: np.ndarray,
    horizontal_phase: np.ndarray,
    vertical_phase: np.ndarray,
    receiver_phase: np.ndarray,
) -> np.ndarray:
    h_index = np.arange(16, dtype=np.float64)[None, None, :, None, None, None]
    v_index = np.arange(8, dtype=np.float64)[None, None, None, :, None, None]
    r_index = np.arange(2, dtype=np.float64)[None, None, None, None, None, :]
    correction = np.exp(
        -1j
        * np.pi
        * (
            horizontal_phase[:, None, None, None, None, None] * h_index
            + vertical_phase[:, None, None, None, None, None] * v_index
            + receiver_phase[:, None, None, None, None, None] * r_index
        )
    )
    reduced = np.mean(coefficient * correction, axis=(2, 3, 5))
    flattened = reduced.reshape(len(reduced), -1).astype(np.complex128)
    norm = np.linalg.norm(flattened, axis=1)
    return flattened / np.maximum(norm[:, None], 1e-30)


def build_local_pairs(
    position: np.ndarray,
    unit_direction: np.ndarray,
    distance: np.ndarray,
    vector: np.ndarray,
    quality: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.asarray(indices, dtype=np.int32)
    if len(indices) < 8:
        return np.empty(0), np.empty(0, dtype=np.complex128), np.empty(0)
    tree = cKDTree(position[indices, :2])
    neighbor_count = min(16, len(indices))
    query_distance, query_index = tree.query(position[indices, :2], k=neighbor_count)
    delta_distance: list[float] = []
    phase: list[complex] = []
    weight: list[float] = []
    for local_i, source in enumerate(indices):
        for neighbor_distance, local_j in zip(query_distance[local_i, 1:], query_index[local_i, 1:]):
            target = int(indices[int(local_j)])
            if target <= source or neighbor_distance > 35.0:
                continue
            if float(np.dot(unit_direction[source], unit_direction[target])) < 0.985:
                continue
            inner = np.vdot(vector[source], vector[target])
            magnitude = abs(inner)
            if magnitude < 0.20:
                continue
            delta_distance.append(float(distance[target] - distance[source]))
            phase.append(inner / magnitude)
            weight.append(
                float(
                    math.sqrt(max(quality[source], 1e-8) * max(quality[target], 1e-8))
                    * magnitude
                    * math.exp(-float(neighbor_distance) / 35.0)
                )
            )
    return (
        np.asarray(delta_distance, dtype=np.float64),
        np.asarray(phase, dtype=np.complex128),
        np.asarray(weight, dtype=np.float64),
    )


def beta_score(beta: float, delta_distance: np.ndarray, phase: np.ndarray, weight: np.ndarray) -> float:
    corrected = phase * np.exp(1j * beta * delta_distance)
    return float(np.sum(weight * np.real(corrected)) / np.sum(weight))


def search_beta(
    delta_distance: np.ndarray,
    phase: np.ndarray,
    weight: np.ndarray,
) -> tuple[float, float, list[dict[str, float]]]:
    if len(delta_distance) < 10:
        raise RuntimeError("Not enough local LOS pairs for carrier-phase identification")
    coarse = np.concatenate(
        [np.arange(-200.0, -79.999, 0.01), np.arange(80.0, 200.001, 0.01)]
    )
    score = np.empty(len(coarse), dtype=np.float64)
    for start in range(0, len(coarse), 512):
        stop = min(start + 512, len(coarse))
        corrected = phase[None, :] * np.exp(1j * coarse[start:stop, None] * delta_distance[None, :])
        score[start:stop] = np.sum(weight[None, :] * np.real(corrected), axis=1) / np.sum(weight)
    order = np.argsort(score)[::-1]
    separated: list[int] = []
    for candidate in order:
        if all(abs(coarse[candidate] - coarse[chosen]) >= 0.30 for chosen in separated):
            separated.append(int(candidate))
        if len(separated) == 5:
            break
    refined_peaks: list[dict[str, float]] = []
    for candidate in separated:
        center = coarse[candidate]
        refined = np.arange(center - 0.02, center + 0.02001, 0.0002)
        refined_score = np.asarray(
            [beta_score(value, delta_distance, phase, weight) for value in refined], dtype=np.float64
        )
        best = int(np.argmax(refined_score))
        refined_peaks.append({"beta_rad_per_m": float(refined[best]), "score": float(refined_score[best])})
    refined_peaks.sort(key=lambda item: item["score"], reverse=True)
    return refined_peaks[0]["beta_rad_per_m"], refined_peaks[0]["score"], refined_peaks


def evaluate_frequency_controls(
    distance: np.ndarray,
    observed: np.ndarray,
    weight: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for spacing in (15_000.0, 30_000.0, 60_000.0, 120_000.0):
        slope = spacing / LIGHT_SPEED
        intercept = weighted_median(observed[train] - slope * distance[train], weight[train])
        result[f"{int(spacing)}Hz"] = weighted_mae(
            observed[validation] - (intercept + slope * distance[validation]), weight[validation]
        )
    slope = -30_000.0 / LIGHT_SPEED
    intercept = weighted_median(observed[train] - slope * distance[train], weight[train])
    result["reversed_30000Hz"] = weighted_mae(
        observed[validation] - (intercept + slope * distance[validation]), weight[validation]
    )
    return result


def amplitude_audit(
    distance: np.ndarray,
    unit_direction: np.ndarray,
    amplitude: np.ndarray,
    weight: np.ndarray,
    groups: np.ndarray,
) -> dict[str, object]:
    log_amplitude = np.log(np.maximum(amplitude, 1e-30))
    log_distance = np.log(distance)
    directional = np.column_stack(
        [
            log_distance,
            unit_direction,
            unit_direction**2,
            unit_direction[:, 0] * unit_direction[:, 1],
            unit_direction[:, 0] * unit_direction[:, 2],
            unit_direction[:, 1] * unit_direction[:, 2],
        ]
    )
    full_distance = robust_linear_fit(log_distance, log_amplitude, weight, f_scale=0.12)
    full_directional = robust_linear_fit(directional, log_amplitude, weight, f_scale=0.12)
    folds: list[dict[str, float]] = []
    splitter = GroupKFold(n_splits=5)
    for train, validation in splitter.split(distance, groups=groups):
        constant = weighted_median(log_amplitude[train], weight[train])
        inverse_distance_offset = weighted_median(
            log_amplitude[train] + log_distance[train], weight[train]
        )
        distance_fit = robust_linear_fit(
            log_distance[train], log_amplitude[train], weight[train], f_scale=0.12
        )
        direction_fit = robust_linear_fit(
            directional[train], log_amplitude[train], weight[train], f_scale=0.12
        )

        def rmse(prediction: np.ndarray) -> float:
            error = prediction - log_amplitude[validation]
            return float(
                math.sqrt(np.sum(weight[validation] * error**2) / np.sum(weight[validation]))
            )

        folds.append(
            {
                "constant_rmse": rmse(np.full(len(validation), constant)),
                "fixed_1_over_d_rmse": rmse(inverse_distance_offset - log_distance[validation]),
                "free_distance_rmse": rmse(distance_fit.predict(log_distance[validation])),
                "directional_rmse": rmse(direction_fit.predict(directional[validation])),
                "free_distance_exponent": float(distance_fit.coefficient[0]),
            }
        )
    mean_free_exponent = float(np.mean([fold["free_distance_exponent"] for fold in folds]))
    fixed_inverse_better = float(
        np.mean([fold["fixed_1_over_d_rmse"] < fold["constant_rmse"] for fold in folds])
    )
    return {
        "field_amplitude_distance_exponent": float(full_distance.coefficient[0]),
        "direction_adjusted_distance_exponent": float(full_directional.coefficient[0]),
        "five_fold": folds,
        "five_fold_mean": {
            key: float(np.mean([fold[key] for fold in folds])) for key in folds[0]
        },
        "fixed_1_over_d_better_than_constant_fold_fraction": fixed_inverse_better,
        "interpretation": (
            "The held-out field-amplitude exponent is "
            f"{mean_free_exponent:.3f}, and a fixed 1/d law beats a constant in only "
            f"{fixed_inverse_better:.0%} of folds. LOS phase is highly identifiable, but LOS amplitude is "
            "confounded by the BS element pattern, point selection, normalization, and residual multipath; it "
            "does not identify an absolute transmit-power normalization."
        ),
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    positions_all = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    channels_all = np.load(PATHS.round1 / "Round1_Train_Channel.npy", mmap_mode="r")
    power = np.load(PATHS.shared_features / "train_total_power.npy").astype(np.float64)
    blocked = np.load(PATHS.shared_features / "height_geometry_cache.npz")["train_blocked"]
    clear_nonzero = np.flatnonzero((blocked == 0) & (power > 0.0))
    threshold = float(np.quantile(power[clear_nonzero], 0.25))
    initial_indices = clear_nonzero[power[clear_nonzero] >= threshold]
    position = positions_all[initial_indices]
    channel = np.asarray(channels_all[initial_indices], dtype=np.complex64)
    direction = position - BS_POSITION
    distance = np.linalg.norm(direction, axis=1)
    unit_direction = direction / distance[:, None]

    frequency_phase, frequency_coherence = adjacent_frequency_statistics(channel)
    initial_frequency_fit = robust_linear_fit(
        distance,
        frequency_phase,
        np.maximum(frequency_coherence, 0.05),
        f_scale=0.0008,
    )
    initial_spacing = float(initial_frequency_fit.coefficient[0] * LIGHT_SPEED)
    initial_delay_cycles = initial_frequency_fit.predict(distance)
    coefficient, direct_fraction = extract_continuous_los(channel, initial_delay_cycles)
    horizontal_phase, horizontal_coherence = adjacent_array_statistics(coefficient, axis=2)
    vertical_phase, vertical_coherence = adjacent_array_statistics(coefficient, axis=3)
    receiver_phase, receiver_coherence = adjacent_array_statistics(coefficient, axis=5)

    quality = (
        np.maximum(direct_fraction, 1e-6)
        * np.sqrt(np.maximum(frequency_coherence, 1e-6))
        * np.sqrt(np.maximum(horizontal_coherence * vertical_coherence, 1e-6))
    )
    keep_count = min(256, max(128, int(round(0.75 * len(initial_indices)))))
    selected_local = np.sort(np.argsort(quality)[-keep_count:])
    selected_indices = initial_indices[selected_local]
    selected_weight = quality[selected_local]
    selected_weight = selected_weight / np.median(selected_weight)

    final_frequency_fit = robust_linear_fit(
        distance[selected_local],
        frequency_phase[selected_local],
        np.maximum(frequency_coherence[selected_local], 0.05),
        f_scale=0.0006,
    )
    inferred_spacing = float(final_frequency_fit.coefficient[0] * LIGHT_SPEED)
    delay_cycles = final_frequency_fit.predict(distance)
    coefficient, direct_fraction = extract_continuous_los(channel, delay_cycles)
    horizontal_phase, horizontal_coherence = adjacent_array_statistics(coefficient, axis=2)
    vertical_phase, vertical_coherence = adjacent_array_statistics(coefficient, axis=3)
    receiver_phase, receiver_coherence = adjacent_array_statistics(coefficient, axis=5)

    h_vector, h_prediction = fit_array_axis(
        unit_direction[selected_local], horizontal_phase[selected_local], selected_weight
    )
    v_vector_free, vertical_circular_loss, vertical_circular_rmse = fit_wrapped_vertical_axis(
        unit_direction[selected_local],
        vertical_phase[selected_local],
        selected_weight,
        h_vector,
    )
    v_vector, vertical_fixed4_loss, vertical_fixed4_rmse = fit_wrapped_vertical_axis(
        unit_direction[selected_local],
        vertical_phase[selected_local],
        selected_weight,
        h_vector,
        fixed_spacing_wavelengths=4.0,
    )
    v_prediction = unit_direction[selected_local] @ v_vector
    array_geometry = orthogonalize_axes(h_vector, v_vector, unit_direction[selected_local])
    tensor_layout_audit = audit_tensor_layouts(
        coefficient,
        selected_local,
        unit_direction[selected_local],
        selected_weight,
    )
    receiver_layout_audit = audit_receiver_layout(coefficient, selected_local)

    selected_position = position[selected_local]
    selected_distance = distance[selected_local]
    selected_unit = unit_direction[selected_local]
    selected_groups = make_groups(selected_position)
    selected_coefficient = coefficient[selected_local]
    selected_h_phase = horizontal_phase[selected_local]
    selected_v_phase = vertical_phase[selected_local]
    selected_r_phase = receiver_phase[selected_local]
    selected_frequency_phase = frequency_phase[selected_local]
    selected_frequency_coherence = frequency_coherence[selected_local]

    port_vector = build_port_vectors(
        selected_coefficient, selected_h_phase, selected_v_phase, selected_r_phase
    )
    all_local = np.arange(len(selected_local), dtype=np.int32)
    pair_distance, pair_phase, pair_weight = build_local_pairs(
        selected_position,
        selected_unit,
        selected_distance,
        port_vector,
        selected_weight,
        all_local,
    )
    beta, beta_fit_score, beta_peaks = search_beta(pair_distance, pair_phase, pair_weight)

    rng = np.random.default_rng(534)
    shuffled_pair_distance = rng.permutation(pair_distance)
    _, shuffled_score, _ = search_beta(shuffled_pair_distance, pair_phase, pair_weight)

    fold_results: list[dict[str, object]] = []
    splitter = GroupKFold(n_splits=5)
    for fold, (train, validation) in enumerate(
        splitter.split(selected_position, groups=selected_groups), start=1
    ):
        frequency_fit = robust_linear_fit(
            selected_distance[train],
            selected_frequency_phase[train],
            np.maximum(selected_frequency_coherence[train], 0.05),
            f_scale=0.0006,
        )
        frequency_error = selected_frequency_phase[validation] - frequency_fit.predict(
            selected_distance[validation]
        )
        frequency_controls = evaluate_frequency_controls(
            selected_distance,
            selected_frequency_phase,
            np.maximum(selected_frequency_coherence, 0.05),
            train,
            validation,
        )

        fold_h, _ = fit_array_axis(selected_unit[train], selected_h_phase[train], selected_weight[train])
        fold_v_free, fold_v_loss, _ = fit_wrapped_vertical_axis(
            selected_unit[train],
            selected_v_phase[train],
            selected_weight[train],
            fold_h,
            seed=534 + fold,
        )
        fold_v, fold_v_fixed4_loss, _ = fit_wrapped_vertical_axis(
            selected_unit[train],
            selected_v_phase[train],
            selected_weight[train],
            fold_h,
            fixed_spacing_wavelengths=4.0,
            seed=544 + fold,
        )
        fold_geometry = orthogonalize_axes(fold_h, fold_v, selected_unit[train])
        true_h_error = selected_h_phase[validation] - selected_unit[validation] @ fold_h
        true_v_error = circular_residual(
            selected_unit[validation] @ fold_v, selected_v_phase[validation]
        )
        swapped_h_error = circular_residual(
            selected_unit[validation] @ fold_v, selected_h_phase[validation]
        )
        swapped_v_error = circular_residual(
            selected_unit[validation] @ fold_h, selected_v_phase[validation]
        )
        rotated_h = rotate_z(fold_h, 15.0)
        rotated_v = rotate_z(fold_v, 15.0)
        rotated_h_error = circular_residual(
            selected_unit[validation] @ rotated_h, selected_h_phase[validation]
        )
        rotated_v_error = circular_residual(
            selected_unit[validation] @ rotated_v, selected_v_phase[validation]
        )
        vertical_spacing_controls = {}
        for spacing_control in (0.5, 1.0, 2.0, 4.0, 6.0):
            control_v, _, _ = fit_wrapped_vertical_axis(
                selected_unit[train],
                selected_v_phase[train],
                selected_weight[train],
                fold_h,
                fixed_spacing_wavelengths=spacing_control,
                seed=540 + fold,
            )
            control_error = circular_residual(
                selected_unit[validation] @ control_v, selected_v_phase[validation]
            )
            vertical_spacing_controls[f"{spacing_control:.1f}lambda"] = float(
                math.sqrt(
                    np.sum(selected_weight[validation] * control_error**2)
                    / np.sum(selected_weight[validation])
                )
            )

        track_offset = weighted_median(selected_r_phase[train], selected_weight[train])
        track_error = selected_r_phase[validation] - track_offset
        fixed_positive_offset = weighted_median(
            selected_r_phase[train] - selected_unit[train, 2], selected_weight[train]
        )
        fixed_negative_offset = weighted_median(
            selected_r_phase[train] + selected_unit[train, 2], selected_weight[train]
        )
        arbitrary_fixed = robust_linear_fit(
            selected_unit[train],
            selected_r_phase[train],
            selected_weight[train],
            f_scale=0.02,
        )

        train_pair_distance, train_pair_phase, train_pair_weight = build_local_pairs(
            selected_position,
            selected_unit,
            selected_distance,
            port_vector,
            selected_weight,
            train,
        )
        validation_pair_distance, validation_pair_phase, validation_pair_weight = build_local_pairs(
            selected_position,
            selected_unit,
            selected_distance,
            port_vector,
            selected_weight,
            validation,
        )
        fold_beta, fold_train_score, _ = search_beta(
            train_pair_distance, train_pair_phase, train_pair_weight
        )
        fold_beta_validation_score = beta_score(
            fold_beta, validation_pair_distance, validation_pair_phase, validation_pair_weight
        )

        def combined_rmse(first: np.ndarray, second: np.ndarray) -> float:
            return float(math.sqrt(np.mean(np.concatenate([first, second]) ** 2)))

        fold_results.append(
            {
                "fold": fold,
                "train_count": int(len(train)),
                "validation_count": int(len(validation)),
                "subcarrier_spacing_hz": float(frequency_fit.coefficient[0] * LIGHT_SPEED),
                "frequency_validation_mae_cycles": weighted_mae(
                    frequency_error, np.maximum(selected_frequency_coherence[validation], 0.05)
                ),
                "frequency_controls_validation_mae_cycles": frequency_controls,
                "array_validation_rmse": combined_rmse(true_h_error, true_v_error),
                "array_swap_control_rmse": combined_rmse(swapped_h_error, swapped_v_error),
                "array_rotate_15deg_control_rmse": combined_rmse(rotated_h_error, rotated_v_error),
                "array_h_sign_control_rmse": combined_rmse(
                    circular_residual(
                        -selected_unit[validation] @ fold_h, selected_h_phase[validation]
                    ),
                    true_v_error,
                ),
                "vertical_spacing_wavelengths_free": float(np.linalg.norm(fold_v_free) / 2.0),
                "vertical_train_circular_loss": float(fold_v_loss),
                "vertical_fixed4_train_circular_loss": float(fold_v_fixed4_loss),
                "vertical_spacing_controls_validation_rmse": vertical_spacing_controls,
                "boresight_azimuth_deg_fixed4": float(fold_geometry["boresight_azimuth_deg"]),
                "boresight_tilt_deg_fixed4": float(fold_geometry["boresight_tilt_deg"]),
                "ue_track_validation_mae": weighted_mae(track_error, selected_weight[validation]),
                "ue_fixed_vertical_positive_mae": weighted_mae(
                    selected_r_phase[validation]
                    - (fixed_positive_offset + selected_unit[validation, 2]),
                    selected_weight[validation],
                ),
                "ue_fixed_vertical_negative_mae": weighted_mae(
                    selected_r_phase[validation]
                    - (fixed_negative_offset - selected_unit[validation, 2]),
                    selected_weight[validation],
                ),
                "ue_arbitrary_fixed_axis_mae": weighted_mae(
                    selected_r_phase[validation] - arbitrary_fixed.predict(selected_unit[validation]),
                    selected_weight[validation],
                ),
                "ue_arbitrary_fixed_axis_norm": float(np.linalg.norm(arbitrary_fixed.coefficient)),
                "beta_rad_per_m": float(fold_beta),
                "carrier_frequency_hz": float(abs(fold_beta) * LIGHT_SPEED / (2.0 * np.pi)),
                "beta_train_pair_score": float(fold_train_score),
                "beta_validation_pair_score": float(fold_beta_validation_score),
                "beta_train_pair_count": int(len(train_pair_distance)),
                "beta_validation_pair_count": int(len(validation_pair_distance)),
            }
        )

    compensated_port = port_vector * np.exp(1j * beta * selected_distance[:, None])
    compensated_port /= np.maximum(np.linalg.norm(compensated_port, axis=1, keepdims=True), 1e-30)
    weighted_matrix = compensated_port * np.sqrt(selected_weight[:, None])
    singular = np.linalg.svd(weighted_matrix, compute_uv=False)
    template = np.sum(selected_weight[:, None] * compensated_port, axis=0)
    template /= np.linalg.norm(template)
    strongest = int(np.argmax(np.abs(template)))
    relative_template = template * np.exp(-1j * np.angle(template[strongest]))
    energy_matrix = np.median(
        np.abs(
            np.mean(
                selected_coefficient
                * np.exp(
                    -1j
                    * np.pi
                    * (
                        selected_h_phase[:, None, None, None, None, None]
                        * np.arange(16)[None, None, :, None, None, None]
                        + selected_v_phase[:, None, None, None, None, None]
                        * np.arange(8)[None, None, None, :, None, None]
                        + selected_r_phase[:, None, None, None, None, None]
                        * np.arange(2)[None, None, None, None, None, :]
                    )
                ),
                axis=(2, 3, 5),
            )
        ),
        axis=0,
    )
    energy_matrix /= np.sum(energy_matrix)

    amplitude = np.sqrt(np.sum(np.abs(selected_coefficient) ** 2, axis=(1, 2, 3, 4, 5)))
    amplitude_result = amplitude_audit(
        selected_distance,
        selected_unit,
        amplitude,
        selected_weight,
        selected_groups,
    )

    fold_numeric_keys = (
        "subcarrier_spacing_hz",
        "frequency_validation_mae_cycles",
        "array_validation_rmse",
        "array_swap_control_rmse",
        "array_rotate_15deg_control_rmse",
        "array_h_sign_control_rmse",
        "vertical_spacing_wavelengths_free",
        "vertical_train_circular_loss",
        "vertical_fixed4_train_circular_loss",
        "boresight_azimuth_deg_fixed4",
        "boresight_tilt_deg_fixed4",
        "ue_track_validation_mae",
        "ue_fixed_vertical_positive_mae",
        "ue_fixed_vertical_negative_mae",
        "ue_arbitrary_fixed_axis_mae",
        "ue_arbitrary_fixed_axis_norm",
        "beta_rad_per_m",
        "carrier_frequency_hz",
        "beta_train_pair_score",
        "beta_validation_pair_score",
    )
    fold_mean = {
        key: float(np.mean([float(fold[key]) for fold in fold_results])) for key in fold_numeric_keys
    }
    fold_std = {
        key: float(np.std([float(fold[key]) for fold in fold_results])) for key in fold_numeric_keys
    }

    np.savez_compressed(
        METRICS_CACHE,
        official_indices=initial_indices,
        selected_official_indices=selected_indices,
        selected_local_indices=selected_local,
        distance_m=distance,
        frequency_phase_cycles=frequency_phase,
        frequency_coherence=frequency_coherence,
        direct_fraction=direct_fraction,
        horizontal_phase_pi=horizontal_phase,
        vertical_phase_pi=vertical_phase,
        receiver_phase_pi=receiver_phase,
        horizontal_coherence=horizontal_coherence,
        vertical_coherence=vertical_coherence,
        receiver_coherence=receiver_coherence,
        quality=quality,
        group_id=selected_groups,
        pair_delta_distance_m=pair_distance,
        pair_phase=pair_phase,
        pair_weight=pair_weight,
    )

    discrete_spacing_error = {}
    for spacing in (15_000.0, 30_000.0, 60_000.0, 120_000.0):
        slope = spacing / LIGHT_SPEED
        offset = weighted_median(
            selected_frequency_phase - slope * selected_distance,
            np.maximum(selected_frequency_coherence, 0.05),
        )
        discrete_spacing_error[str(int(spacing))] = {
            "delay_offset_bins": float(N_SUBCARRIERS * offset),
            "weighted_mae_bins": float(
                N_SUBCARRIERS
                * weighted_mae(
                    selected_frequency_phase - (offset + slope * selected_distance),
                    np.maximum(selected_frequency_coherence, 0.05),
                )
            ),
        }

    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "purpose": (
            "Use official train-H LOS components as a calibration ruler for the literal or equivalence-class "
            "Sionna system configuration; do not modify v533 and do not generate a submission."
        ),
        "inputs": {
            "positions": "datasets/round1/Round1_Train_Pos.npy",
            "channel": "datasets/round1/Round1_Train_Channel.npy",
            "visibility_proxy": "artifacts/shared_features/height_geometry_cache.npz:train_blocked",
            "bs_position_m": BS_POSITION.tolist(),
            "channel_layout": "[point, TX=(polarization,16 horizontal,8 vertical), RX=(polarization,2 vertical), 192 subcarriers]",
        },
        "selection": {
            "clear_nonzero_count": int(len(clear_nonzero)),
            "initial_high_power_clear_count": int(len(initial_indices)),
            "power_quantile_threshold": threshold,
            "selected_high_confidence_los_count": int(len(selected_indices)),
            "selection_rule": (
                "Top 75% (capped at 256) by continuous geometric-delay energy, frequency coherence, and H/V "
                "plane-wave coherence after requiring blocked=0, nonzero power, and clear-power >= q25."
            ),
            "selected_direct_fraction_median": float(np.median(direct_fraction[selected_local])),
            "selected_frequency_coherence_median": float(np.median(frequency_coherence[selected_local])),
            "selected_horizontal_coherence_median": float(np.median(horizontal_coherence[selected_local])),
            "selected_vertical_coherence_median": float(np.median(vertical_coherence[selected_local])),
        },
        "frequency_axis_and_delay": {
            "initial_continuous_spacing_hz": initial_spacing,
            "final_continuous_spacing_hz": inferred_spacing,
            "final_delay_offset_cycles_per_subcarrier": float(final_frequency_fit.intercept),
            "final_delay_offset_bins": float(N_SUBCARRIERS * final_frequency_fit.intercept),
            "discrete_spacing_controls": discrete_spacing_error,
            "sign_convention": (
                "Adjacent CFR phase follows exp(-j*2*pi*Delta_f*tau); reversing subcarrier order gives the "
                "wrong distance slope. Centered versus uncentered absolute frequency labels remain a constant "
                "path-phase gauge and cannot be distinguished from H alone."
            ),
        },
        "bs_array": array_geometry
        | {
            "vertical_free_fit_spacing_wavelengths": float(np.linalg.norm(v_vector_free) / 2.0),
            "vertical_free_fit_vector": v_vector_free.tolist(),
            "horizontal_fit_rmse_pi": float(
                math.sqrt(np.mean((selected_h_phase - h_prediction) ** 2))
            ),
            "vertical_free_fit_circular_rmse_pi": vertical_circular_rmse,
            "vertical_free_fit_circular_loss": vertical_circular_loss,
            "vertical_fixed4_circular_rmse_pi": vertical_fixed4_rmse,
            "vertical_fixed4_circular_loss": vertical_fixed4_loss,
            "spacing_formula": "norm(fitted phase vector)/2 wavelengths",
            "vertical_phase_note": (
                "The 8-element axis wraps modulo 2*pi because its spacing is close to 4 wavelengths. A plain "
                "linear phase fit incorrectly aliases this to roughly 1.1 wavelengths."
            ),
        },
        "tensor_layout_audit": tensor_layout_audit,
        "receiver_layout_audit": receiver_layout_audit,
        "ue_array": {
            "observed_abs_phase_median_pi": float(np.median(np.abs(selected_r_phase))),
            "observed_phase_coherence_median": float(np.median(receiver_coherence[selected_local])),
            "interpretation": (
                "Near-zero spatial phase with spatial-fold superiority over fixed-world-axis controls identifies "
                "source-tracking/boresight reception. LOS does not identify roll around the boresight."
            ),
        },
        "carrier_phase": {
            "local_pair_count": int(len(pair_distance)),
            "beta_rad_per_m": float(beta),
            "effective_carrier_frequency_hz": float(abs(beta) * LIGHT_SPEED / (2.0 * np.pi)),
            "phase_sign": "negative propagation phase" if beta > 0 else "positive propagation phase",
            "fit_score": float(beta_fit_score),
            "distance_shuffle_best_score": float(shuffled_score),
            "top_separated_peaks": beta_peaks,
            "identifiability_note": (
                "The frequency is effective under the chosen baseband-index origin. Shifting that origin changes "
                "the reported carrier by at most the occupied 5.76 MHz band, so LOS identifies the carrier to a "
                "narrow equivalence interval rather than exposing the generator's literal frequency variable."
            ),
        },
        "port_response": {
            "normalized_energy_txpol_by_rxpol": energy_matrix.tolist(),
            "first_singular_energy_fraction": float(singular[0] ** 2 / np.sum(singular**2)),
            "relative_complex_template_real": np.real(relative_template).tolist(),
            "relative_complex_template_imag": np.imag(relative_template).tolist(),
            "template_order": "flattened [TX polarization, RX polarization] after H/V/RX-row de-steering",
            "semantic_limit": (
                "The average relative port response and pairing are observable, but the first singular component "
                "does not dominate enough to justify a point-invariant fixed complex gain. Naming a port V or H "
                "is also a convention without the antenna-pattern definition."
            ),
        },
        "los_amplitude": amplitude_result,
        "spatial_five_fold": {
            "tile_size_m": TILE_SIZE_M,
            "folds": fold_results,
            "mean": fold_mean,
            "std": fold_std,
        },
        "configuration_identifiability": {
            "uniquely_or_strongly_identified_from_los": [
                "TX tensor axes are 16 horizontal x 8 vertical under the existing dual-polarization reshape",
                "TX horizontal/vertical port increment signs and array plane",
                "approximately 0.5-wavelength TX horizontal spacing and 4-wavelength TX vertical spacing",
                "BS boresight azimuth and downtilt up to the array front/back convention",
                "receiver tracks the BS/source rather than using a fixed world orientation",
                "30 kHz subcarrier spacing and frequency-order sign",
                "continuous delay zero-point under the selected frequency convention",
                "effective carrier propagation coefficient and a narrow carrier-frequency equivalence interval",
            ],
            "only_equivalence_class_from_los": [
                "centered versus uncentered baseband frequency labels",
                "literal carrier variable within one occupied-band-width phase-reference gauge",
                "array front/back side without using the high-gain receiver distribution",
                "receiver roll around its source-tracking boresight",
                "semantic V/H names and any common complex channel scalar",
                "average polarization-port pairing, but not a point-invariant full complex port gain",
                "absolute transmit power versus channel normalization versus antenna gain",
            ],
            "not_identifiable_from_los": [
                "building material reflection coefficients",
                "max_depth",
                "specular reflection order and path pruning",
                "refraction",
                "diffraction and edge diffraction",
                "diffuse scattering coefficient",
                "random ray seed and finite path budget",
                "NLOS path birth/death rules",
            ],
            "next_stage": (
                "Freeze the LOS-identified signal-chain parameters, subtract or mask the fitted LOS component, "
                "then compare NLOS residual delay-angle support across material/depth/diffraction/scattering "
                "configurations. Those propagation parameters cannot be inferred honestly from LOS alone."
            ),
        },
        "outputs": {
            "report": PATHS.relative(REPORT),
            "point_metrics": PATHS.relative(METRICS_CACHE),
            "submission_zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
