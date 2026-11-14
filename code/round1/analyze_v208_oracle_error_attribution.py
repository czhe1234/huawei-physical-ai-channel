#!/usr/bin/env python3
"""Locate synthetic v200 errors and measure their causal score headroom with oracles."""

from __future__ import annotations

import csv
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

import analyze_v205_sionna_config_calibration as calibration
from powerfield_common import regime_of
from project_paths import PATHS


VERSION = "v208_oracle_error_attribution"
REPORT_DIR = PATHS.version_report_dir(208)
REPORT = REPORT_DIR / "v208_oracle_error_attribution.json"
POINT_CSV = REPORT_DIR / "v208_oracle_point_errors.csv"
GAIN_FIGURE = REPORT_DIR / "v208_oracle_score_gains.png"
REGIME_FIGURE = REPORT_DIR / "v208_oracle_regime_gains.png"
MAP_FIGURE = REPORT_DIR / "v208_oracle_error_maps.png"
PREDICTION = PATHS.cache_dir(208) / "v208_replay_scratch_output.npy"
TRUTH = PATHS.cache_dir(207) / "v207_sionna_test_reference_H.npy"
POWER_HEAD_REPORT = REPORT_DIR / "v208_v200_strict_power_head.json"
GEOMETRY = PATHS.shared_features / "height_geometry_cache.npz"
BATCH = 8
TINY = 1e-30
REGIMES = ("clear", "light", "medium", "heavy")


def summarize(values: np.ndarray) -> dict[str, float | int]:
    value = np.asarray(values, dtype=np.float64)
    value = value[np.isfinite(value)]
    if len(value) == 0:
        return {"count": 0}
    return {
        "count": int(len(value)),
        "mean": float(np.mean(value)),
        "std": float(np.std(value)),
        "min": float(np.min(value)),
        "p10": float(np.quantile(value, 0.10)),
        "p50": float(np.quantile(value, 0.50)),
        "p90": float(np.quantile(value, 0.90)),
        "max": float(np.max(value)),
    }


def point_power(value: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(value) ** 2, axis=tuple(range(1, value.ndim)), dtype=np.float64)


def rescale_h(value: np.ndarray, desired_power: np.ndarray) -> np.ndarray:
    current = point_power(value)
    scale = np.sqrt(desired_power / np.maximum(current, TINY)).astype(np.float32)
    scale[desired_power <= TINY] = 0.0
    return (value * scale.reshape((-1,) + (1,) * (value.ndim - 1))).astype(np.complex64)


def phase_like(value: np.ndarray) -> np.ndarray:
    return np.exp(1j * np.angle(value)).astype(np.complex64)


def grouped_shape_oracle(
    source: np.ndarray,
    target: np.ndarray,
    distribution_axes: tuple[int, ...],
) -> np.ndarray:
    """Use target normalized power inside each group and preserve source group power."""
    source_power = np.abs(source).astype(np.float32) ** 2
    target_power = np.abs(target).astype(np.float32) ** 2
    source_group = np.sum(source_power, axis=distribution_axes, keepdims=True, dtype=np.float64)
    target_group = np.sum(target_power, axis=distribution_axes, keepdims=True, dtype=np.float64)
    desired = source_group * target_power / np.maximum(target_group, TINY)
    output = np.sqrt(np.maximum(desired, 0.0)).astype(np.float32) * phase_like(source)
    return np.where(target_group > TINY, output, source).astype(np.complex64)


def to_angle(value: np.ndarray) -> np.ndarray:
    value6 = value.reshape(len(value), 2, 16, 8, 4, 192)
    return np.fft.fft2(value6, axes=(2, 3)).astype(np.complex64)


def from_angle(value: np.ndarray) -> np.ndarray:
    return np.fft.ifft2(value, axes=(2, 3)).reshape(len(value), 256, 4, 192).astype(np.complex64)


def to_delay(value: np.ndarray) -> np.ndarray:
    return np.fft.ifft(value, axis=-1).astype(np.complex64)


def from_delay(value: np.ndarray) -> np.ndarray:
    return np.fft.fft(value, axis=-1).astype(np.complex64)


def to_leaked(value: np.ndarray) -> np.ndarray:
    value6 = value.reshape(len(value), 2, 16, 8, 4, 192)
    arranged = value6.transpose(0, 1, 5, 2, 4, 3)
    return np.fft.fft2(arranged, axes=(2, 3)).astype(np.complex64)


def from_leaked(value: np.ndarray) -> np.ndarray:
    arranged = np.fft.ifft2(value, axes=(2, 3))
    value6 = arranged.transpose(0, 1, 3, 5, 4, 2)
    return value6.reshape(len(value), 256, 4, 192).astype(np.complex64)


def to_ad(value: np.ndarray) -> np.ndarray:
    return np.fft.ifft(to_angle(value), axis=-1).astype(np.complex64)


def from_ad(value: np.ndarray) -> np.ndarray:
    return from_angle(np.fft.fft(value, axis=-1).astype(np.complex64))


def exact_pas_shape(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    output = from_angle(grouped_shape_oracle(to_angle(source), to_angle(target), (1, 2, 3)))
    return rescale_h(output, point_power(source))


def exact_pdp_shape(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    output = from_delay(grouped_shape_oracle(to_delay(source), to_delay(target), (3,)))
    return rescale_h(output, point_power(source))


def exact_leaked_shape(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    output = from_leaked(grouped_shape_oracle(to_leaked(source), to_leaked(target), (2, 3)))
    return rescale_h(output, point_power(source))


def iterative_metric_oracle(
    source: np.ndarray,
    target: np.ndarray,
    modes: tuple[str, ...],
    rounds: int = 4,
) -> np.ndarray:
    desired_power = point_power(source)
    output = source.copy()
    for _ in range(rounds):
        for mode in modes:
            if mode == "pas":
                output = exact_pas_shape(output, target)
            elif mode == "pdp":
                output = exact_pdp_shape(output, target)
            elif mode == "leaked":
                output = exact_leaked_shape(output, target)
            else:
                raise ValueError(mode)
    return rescale_h(output, desired_power)


def collapsed_ad_shape_oracle(source_ad: np.ndarray, target_ad: np.ndarray) -> np.ndarray:
    source_power = np.abs(source_ad).astype(np.float32) ** 2
    target_power = np.abs(target_ad).astype(np.float32) ** 2
    source_collapsed = np.sum(source_power, axis=(1, 4), keepdims=True, dtype=np.float64)
    target_collapsed = np.sum(target_power, axis=(1, 4), keepdims=True, dtype=np.float64)
    source_total = np.sum(source_collapsed, axis=(2, 3, 5), keepdims=True, dtype=np.float64)
    target_total = np.sum(target_collapsed, axis=(2, 3, 5), keepdims=True, dtype=np.float64)
    desired_collapsed = source_total * target_collapsed / np.maximum(target_total, TINY)
    conditional = source_power / np.maximum(source_collapsed, TINY)
    output_power = desired_collapsed * conditional
    output = np.sqrt(np.maximum(output_power, 0.0)).astype(np.float32) * phase_like(source_ad)
    valid = (source_collapsed > TINY) & (target_total > TINY)
    return np.where(valid, output, source_ad).astype(np.complex64)


def polue_allocation_oracle(source_ad: np.ndarray, target_ad: np.ndarray) -> np.ndarray:
    source_power = np.abs(source_ad).astype(np.float32) ** 2
    target_power = np.abs(target_ad).astype(np.float32) ** 2
    source_collapsed = np.sum(source_power, axis=(1, 4), keepdims=True, dtype=np.float64)
    target_collapsed = np.sum(target_power, axis=(1, 4), keepdims=True, dtype=np.float64)
    target_conditional = target_power / np.maximum(target_collapsed, TINY)
    output_power = source_collapsed * target_conditional
    output = np.sqrt(np.maximum(output_power, 0.0)).astype(np.float32) * phase_like(source_ad)
    return np.where(target_collapsed > TINY, output, source_ad).astype(np.complex64)


def full_ad_magnitude_shape_oracle(source_ad: np.ndarray, target_ad: np.ndarray) -> np.ndarray:
    source_total = point_power(source_ad)
    target_power = np.abs(target_ad).astype(np.float32) ** 2
    target_total = point_power(target_ad)
    desired = target_power * (
        source_total / np.maximum(target_total, TINY)
    ).reshape(-1, 1, 1, 1, 1, 1)
    output = np.sqrt(np.maximum(desired, 0.0)).astype(np.float32) * phase_like(source_ad)
    valid = (target_total > TINY).reshape(-1, 1, 1, 1, 1, 1)
    return np.where(valid, output, source_ad).astype(np.complex64)


def unit_phase_oracle(
    source: np.ndarray,
    target: np.ndarray,
    reduce_axes: tuple[int, ...],
) -> np.ndarray:
    inner = np.sum(np.conj(source) * target, axis=reduce_axes, keepdims=True, dtype=np.complex128)
    magnitude = np.abs(inner)
    factor = np.ones_like(inner, dtype=np.complex128)
    np.divide(inner, magnitude, out=factor, where=magnitude > TINY)
    return (source * factor).astype(np.complex64)


def make_variants(source: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    source_power = point_power(source)
    target_power = point_power(target)
    source_ad = to_ad(source)
    target_ad = to_ad(target)

    common_phase = unit_phase_oracle(source, target, (1, 2, 3))
    point_inner = np.sum(
        np.conj(source) * target, axis=(1, 2, 3), keepdims=True, dtype=np.complex128
    )
    point_den = np.sum(np.abs(source) ** 2, axis=(1, 2, 3), keepdims=True, dtype=np.float64)
    point_scalar = (source * point_inner / np.maximum(point_den, TINY)).astype(np.complex64)

    collapsed_ad = from_ad(collapsed_ad_shape_oracle(source_ad, target_ad))
    polue_ad = from_ad(polue_allocation_oracle(source_ad, target_ad))
    full_mag_shape_ad = full_ad_magnitude_shape_oracle(source_ad, target_ad)
    full_mag_shape = from_ad(full_mag_shape_ad)
    full_mag_absolute = from_ad(np.abs(target_ad).astype(np.float32) * phase_like(source_ad))
    block_phase = from_ad(unit_phase_oracle(source_ad, target_ad, (2, 3, 5)))
    component_phase = from_ad(unit_phase_oracle(source_ad, target_ad, (1, 4)))
    full_phase_ad = (np.abs(source_ad).astype(np.float32) * phase_like(target_ad)).astype(np.complex64)
    truth_nonzero = (target_power > TINY).reshape(-1, 1, 1, 1, 1, 1)
    full_phase_ad = np.where(truth_nonzero, full_phase_ad, source_ad).astype(np.complex64)
    full_phase = from_ad(full_phase_ad)

    variants = {
        "base": source,
        "oracle_total_power": rescale_h(source, target_power),
        "oracle_common_phase": common_phase,
        "oracle_total_power_common_phase": rescale_h(common_phase, target_power),
        "oracle_point_complex_scalar": point_scalar,
        "oracle_pas_shape": exact_pas_shape(source, target),
        "oracle_pdp_shape": exact_pdp_shape(source, target),
        "oracle_leaked_shape": exact_leaked_shape(source, target),
        "oracle_pas_pdp_shape": iterative_metric_oracle(source, target, ("pas", "pdp")),
        "oracle_leaked_pdp_shape": iterative_metric_oracle(source, target, ("leaked", "pdp")),
        "oracle_collapsed_joint_ad_shape": rescale_h(collapsed_ad, source_power),
        "oracle_polue_allocation": rescale_h(polue_ad, source_power),
        "oracle_full_ad_magnitude_shape": rescale_h(full_mag_shape, source_power),
        "oracle_full_ad_magnitude": rescale_h(full_mag_absolute, target_power),
        "oracle_polue_block_phase": rescale_h(block_phase, source_power),
        "oracle_hvd_component_phase": rescale_h(component_phase, source_power),
        "oracle_full_ad_phase": rescale_h(full_phase, source_power),
        "oracle_full_ad_phase_total": rescale_h(full_phase, target_power),
        "truth": target,
    }
    return {name: np.asarray(value, dtype=np.complex64) for name, value in variants.items()}


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = a.reshape(len(a), -1).astype(np.float64)
    bb = b.reshape(len(b), -1).astype(np.float64)
    numerator = np.sum(aa * bb, axis=1, dtype=np.float64)
    denominator = np.sqrt(
        np.sum(aa * aa, axis=1, dtype=np.float64)
        * np.sum(bb * bb, axis=1, dtype=np.float64)
    )
    return (numerator / np.maximum(denominator, TINY)).astype(np.float32)


def baseline_extra_metrics(source: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    source_ad = to_ad(source)
    target_ad = to_ad(target)
    source_power = np.abs(source_ad).astype(np.float32) ** 2
    target_power = np.abs(target_ad).astype(np.float32) ** 2
    full_magnitude = cosine_rows(source_power, target_power)

    source_pu = source_power.transpose(0, 2, 3, 5, 1, 4).reshape(len(source), 16, 8, 192, 8)
    target_pu = target_power.transpose(0, 2, 3, 5, 1, 4).reshape(len(source), 16, 8, 192, 8)
    numerator = np.sum(source_pu * target_pu, axis=-1, dtype=np.float64)
    denominator = np.sqrt(
        np.sum(source_pu * source_pu, axis=-1, dtype=np.float64)
        * np.sum(target_pu * target_pu, axis=-1, dtype=np.float64)
    )
    local_cosine = numerator / np.maximum(denominator, TINY)
    weights = np.sum(target_pu, axis=-1, dtype=np.float64)
    polue = np.sum(local_cosine * weights, axis=(1, 2, 3), dtype=np.float64) / np.maximum(
        np.sum(weights, axis=(1, 2, 3), dtype=np.float64), TINY
    )

    overlap = np.abs(source_ad).astype(np.float64) * np.abs(target_ad).astype(np.float64)
    phase_vector = np.sum(np.conj(source_ad) * target_ad, axis=(1, 2, 3, 4, 5), dtype=np.complex128)
    phase_resultant = np.abs(phase_vector) / np.maximum(
        np.sum(overlap, axis=(1, 2, 3, 4, 5), dtype=np.float64), TINY
    )
    return {
        "full_ad_magnitude_cosine": full_magnitude,
        "polue_conditional_cosine": polue.astype(np.float32),
        "ad_phase_resultant": phase_resultant.astype(np.float32),
    }


def append_variant_metrics(
    accum: dict[str, defaultdict[str, list[np.ndarray]]],
    name: str,
    prediction: np.ndarray,
    target: np.ndarray,
    target_views: dict[str, np.ndarray],
) -> None:
    scores = calibration.spectral_scores(prediction, target_views)
    pred_power = point_power(prediction)
    truth_power = target_views["energy"]
    true_zero = truth_power <= 1e-25
    pred_zero = pred_power <= 1e-25
    for key in ("pas", "pdp", "leaked_sh", "ad_support"):
        value = scores[key].copy()
        value[true_zero & pred_zero] = 1.0
        value[true_zero & ~pred_zero] = 0.0
        accum[name][key].append(value)
    accum[name]["complex_coherence"].append(scores["complex_coherence"])
    accum[name]["pred_log_power"].append(np.log(np.maximum(pred_power, TINY)))
    accum[name]["truth_log_power"].append(np.log(np.maximum(truth_power, TINY)))
    accum[name]["e2"].append(
        np.sum(np.abs(prediction - target) ** 2, axis=(1, 2, 3), dtype=np.float64)
    )
    accum[name]["t2"].append(truth_power.astype(np.float64))


def finalize_variant(
    raw: defaultdict[str, list[np.ndarray]],
    regimes: np.ndarray,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    values = {key: np.concatenate(parts) for key, parts in raw.items()}
    global_nmse = float(np.sum(values["e2"]) / max(float(np.sum(values["t2"])), TINY))
    nmse_score = 1.0 / (1.0 + global_nmse)
    pas = float(np.mean(values["pas"]))
    pdp = float(np.mean(values["pdp"]))
    leaked = float(np.mean(values["leaked_sh"]))
    truth_nonzero = values["t2"] > 1e-25
    log_error = values["pred_log_power"] - values["truth_log_power"]
    power_rank = float(spearmanr(values["pred_log_power"], values["truth_log_power"]).statistic)
    report: dict[str, object] = {
        "score_traceback": 0.4 * leaked + 0.4 * pdp + 0.2 * nmse_score,
        "score_conventional": 0.4 * pas + 0.4 * pdp + 0.2 * nmse_score,
        "pas": pas,
        "pdp": pdp,
        "leaked_sh": leaked,
        "ad_support": float(np.mean(values["ad_support"])),
        "complex_coherence": float(np.mean(values["complex_coherence"])),
        "global_nmse": global_nmse,
        "nmse_score": nmse_score,
        "power_spearman": power_rank,
        "log_power_error_nonzero": summarize(log_error[truth_nonzero]),
        "abs_log_power_error_nonzero": summarize(np.abs(log_error[truth_nonzero])),
        "point_nmse_nonzero": summarize(values["e2"][truth_nonzero] / np.maximum(values["t2"][truth_nonzero], TINY)),
        "truth_zero_count": int(np.sum(~truth_nonzero)),
        "predicted_zero_count": int(np.sum(np.exp(values["pred_log_power"]) <= 1e-25)),
        "by_regime": {},
    }
    for regime in REGIMES:
        mask = regimes == regime
        regime_nmse = float(np.sum(values["e2"][mask]) / max(float(np.sum(values["t2"][mask])), TINY))
        regime_nmse_score = 1.0 / (1.0 + regime_nmse)
        regime_pas = float(np.mean(values["pas"][mask]))
        regime_pdp = float(np.mean(values["pdp"][mask]))
        regime_leaked = float(np.mean(values["leaked_sh"][mask]))
        report["by_regime"][regime] = {
            "count": int(np.sum(mask)),
            "score_traceback": 0.4 * regime_leaked + 0.4 * regime_pdp + 0.2 * regime_nmse_score,
            "score_conventional": 0.4 * regime_pas + 0.4 * regime_pdp + 0.2 * regime_nmse_score,
            "pas": regime_pas,
            "pdp": regime_pdp,
            "leaked_sh": regime_leaked,
            "ad_support": float(np.mean(values["ad_support"][mask])),
            "complex_coherence": float(np.mean(values["complex_coherence"][mask])),
            "global_nmse": regime_nmse,
            "nmse_score": regime_nmse_score,
        }
    return report, values


def plot_score_gains(results: dict[str, dict[str, object]]) -> None:
    names = [name for name in results if name not in ("base", "truth")]
    names.sort(key=lambda name: float(results[name]["gain_traceback_vs_base"]))
    gains = [float(results[name]["gain_traceback_vs_base"]) for name in names]
    colors = ["#c44e52" if gain < 0 else "#3b7c6f" for gain in gains]
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(range(len(names)), gains, color=colors)
    ax.set_yticks(range(len(names)), [name.removeprefix("oracle_") for name in names])
    ax.axvline(0.0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Traceback composite gain versus synthetic v200 base")
    ax.set_title("Synthetic v200 oracle headroom")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(GAIN_FIGURE, dpi=180)
    plt.close(fig)


def plot_regime_gains(results: dict[str, dict[str, object]]) -> None:
    names = [
        "oracle_total_power",
        "oracle_leaked_pdp_shape",
        "oracle_collapsed_joint_ad_shape",
        "oracle_polue_allocation",
        "oracle_full_ad_magnitude_shape",
        "oracle_common_phase",
        "oracle_hvd_component_phase",
        "oracle_full_ad_phase",
    ]
    matrix = np.asarray(
        [
            [
                float(results[name]["by_regime"][regime]["score_traceback"])
                - float(results["base"]["by_regime"][regime]["score_traceback"])
                for regime in REGIMES
            ]
            for name in names
        ],
        dtype=np.float64,
    )
    bound = max(float(np.max(np.abs(matrix))), 1e-6)
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-bound, vmax=bound)
    ax.set_xticks(range(len(REGIMES)), REGIMES)
    ax.set_yticks(range(len(names)), [name.removeprefix("oracle_") for name in names])
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, f"{matrix[row, col]:+.3f}", ha="center", va="center", fontsize=8)
    ax.set_title("Oracle traceback gain by propagation regime")
    fig.colorbar(image, ax=ax, label="gain versus base")
    fig.tight_layout()
    fig.savefig(REGIME_FIGURE, dpi=180)
    plt.close(fig)


def plot_error_maps(point: dict[str, np.ndarray], positions: np.ndarray) -> None:
    fields = (
        ("abs_log_power_error", "|log power error|"),
        ("pas_deficit", "PAS deficit"),
        ("pdp_deficit", "PDP deficit"),
        ("ad_deficit", "joint AD deficit"),
        ("phase_deficit", "AD phase deficit"),
        ("log10_point_nmse", "log10 point NMSE"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    for ax, (key, title) in zip(axes.flat, fields):
        values = point[key]
        finite = np.isfinite(values)
        color = np.clip(values, np.quantile(values[finite], 0.02), np.quantile(values[finite], 0.98))
        scatter = ax.scatter(positions[:, 0], positions[:, 1], c=color, s=22, cmap="viridis", linewidths=0)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes[-1]:
        ax.set_xlabel("x (m)")
    for ax in axes[:, 0]:
        ax.set_ylabel("y (m)")
    fig.suptitle("Where the synthetic v200 replay is inaccurate", fontsize=14)
    fig.tight_layout()
    fig.savefig(MAP_FIGURE, dpi=180)
    plt.close(fig)


def write_point_csv(
    point: dict[str, np.ndarray],
    positions: np.ndarray,
    blocked: np.ndarray,
    regimes: np.ndarray,
) -> None:
    columns = [
        "index",
        "x",
        "y",
        "blocked",
        "regime",
        *point.keys(),
    ]
    with POINT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for index in range(len(positions)):
            row: dict[str, object] = {
                "index": index,
                "x": float(positions[index, 0]),
                "y": float(positions[index, 1]),
                "blocked": float(blocked[index]),
                "regime": str(regimes[index]),
            }
            row.update({key: float(value[index]) for key, value in point.items()})
            writer.writerow(row)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for path in (PREDICTION, TRUTH, POWER_HEAD_REPORT, GEOMETRY):
        if not path.is_file():
            raise FileNotFoundError(path)
    prediction = np.load(PREDICTION, mmap_mode="r")
    truth = np.load(TRUTH, mmap_mode="r")
    expected_shape = (500, 256, 4, 192)
    if prediction.shape != expected_shape or truth.shape != expected_shape:
        raise ValueError((prediction.shape, truth.shape))
    if prediction.dtype != np.complex64 or truth.dtype != np.complex64:
        raise ValueError((prediction.dtype, truth.dtype))

    geometry = np.load(GEOMETRY)
    blocked = geometry["test_blocked"].astype(np.float32)
    regimes = regime_of(blocked).astype("U8")
    positions = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    accum: dict[str, defaultdict[str, list[np.ndarray]]] = {}
    extra_parts: defaultdict[str, list[np.ndarray]] = defaultdict(list)

    for start in range(0, len(prediction), BATCH):
        end = min(start + BATCH, len(prediction))
        source = np.asarray(prediction[start:end], dtype=np.complex64)
        target = np.asarray(truth[start:end], dtype=np.complex64)
        target_views = calibration.spectral_views(target)
        variants = make_variants(source, target)
        for name, value in variants.items():
            if name not in accum:
                accum[name] = defaultdict(list)
            append_variant_metrics(accum, name, value, target, target_views)
        base_scores = calibration.spectral_scores(source, target_views)
        base_power = point_power(source)
        truth_power = target_views["energy"]
        extras = baseline_extra_metrics(source, target)
        extra_parts["pred_power"].append(base_power)
        extra_parts["truth_power"].append(truth_power)
        extra_parts["log_power_error"].append(
            np.log(np.maximum(base_power, TINY)) - np.log(np.maximum(truth_power, TINY))
        )
        for key in ("pas", "pdp", "leaked_sh", "ad_support", "complex_coherence"):
            extra_parts[key].append(base_scores[key])
        for key, value in extras.items():
            extra_parts[key].append(value)
        e2 = np.sum(np.abs(source - target) ** 2, axis=(1, 2, 3), dtype=np.float64)
        extra_parts["point_nmse"].append(e2 / np.maximum(truth_power, TINY))
        print(f"oracle {end}/{len(prediction)}", flush=True)

    results: dict[str, dict[str, object]] = {}
    raw_values: dict[str, dict[str, np.ndarray]] = {}
    for name, raw in accum.items():
        results[name], raw_values[name] = finalize_variant(raw, regimes)
    base_traceback = float(results["base"]["score_traceback"])
    base_conventional = float(results["base"]["score_conventional"])
    for result in results.values():
        result["gain_traceback_vs_base"] = float(result["score_traceback"]) - base_traceback
        result["gain_conventional_vs_base"] = float(result["score_conventional"]) - base_conventional

    expected = json.loads(POWER_HEAD_REPORT.read_text(encoding="utf-8"))["synthetic_v200_power_head"]
    if abs(base_traceback - float(expected["score_traceback_leaked"])) > 2e-6:
        raise RuntimeError(
            "v208 scratch no longer contains the strict v200 output: "
            f"{base_traceback} != {expected['score_traceback_leaked']}"
        )

    extra = {key: np.concatenate(parts) for key, parts in extra_parts.items()}
    point = {
        "truth_power": extra["truth_power"],
        "pred_power": extra["pred_power"],
        "log_power_error": extra["log_power_error"],
        "abs_log_power_error": np.abs(extra["log_power_error"]),
        "pas": extra["pas"],
        "pas_deficit": 1.0 - extra["pas"],
        "pdp": extra["pdp"],
        "pdp_deficit": 1.0 - extra["pdp"],
        "leaked_sh": extra["leaked_sh"],
        "ad_support": extra["ad_support"],
        "ad_deficit": 1.0 - extra["ad_support"],
        "full_ad_magnitude_cosine": extra["full_ad_magnitude_cosine"],
        "polue_conditional_cosine": extra["polue_conditional_cosine"],
        "ad_phase_resultant": extra["ad_phase_resultant"],
        "phase_deficit": 1.0 - extra["ad_phase_resultant"],
        "complex_coherence": extra["complex_coherence"],
        "point_nmse": extra["point_nmse"],
        "log10_point_nmse": np.log10(np.maximum(extra["point_nmse"], TINY)),
    }
    write_point_csv(point, positions, blocked, regimes)
    plot_score_gains(results)
    plot_regime_gains(results)
    plot_error_maps(point, positions)

    ranking = sorted(
        (name for name in results if name not in ("base", "truth")),
        key=lambda name: float(results[name]["gain_traceback_vs_base"]),
        reverse=True,
    )
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "hypothesis": (
            "Direct prediction-vs-clean-truth errors locate inaccurate factors; single-factor oracle "
            "replacements measure which factors causally limit the inferred traceback score."
        ),
        "scope_boundary": (
            "These conclusions are exact for the current 6.0GHz synthetic dataset and strict v200 "
            "power-head replay, but transfer online only after a simulator configuration reproduces "
            "historical online ordering."
        ),
        "inputs": {
            "prediction": PATHS.relative(PREDICTION),
            "truth": PATHS.relative(TRUTH),
            "prediction_definition": "synthetic v9 -> v189-like -> exact-family v196 -> retrained v200 power head",
            "noise": "relative_25db with fixed floor and official 16 exact-zero train rows",
            "count": int(len(prediction)),
            "regime_counts": {regime: int(np.sum(regimes == regime)) for regime in REGIMES},
        },
        "baseline_diagnostics": {
            "score": results["base"],
            "full_ad_magnitude_cosine": summarize(extra["full_ad_magnitude_cosine"]),
            "polue_conditional_cosine": summarize(extra["polue_conditional_cosine"]),
            "ad_phase_resultant": summarize(extra["ad_phase_resultant"]),
            "abs_log_power_error": summarize(np.abs(extra["log_power_error"])[extra["truth_power"] > 1e-25]),
            "point_nmse_nonzero": summarize(extra["point_nmse"][extra["truth_power"] > 1e-25]),
        },
        "oracle_ranking_traceback": ranking,
        "oracle_results": results,
        "outputs": {
            "report": PATHS.relative(REPORT),
            "point_csv": PATHS.relative(POINT_CSV),
            "gain_figure": PATHS.relative(GAIN_FIGURE),
            "regime_figure": PATHS.relative(REGIME_FIGURE),
            "error_map": PATHS.relative(MAP_FIGURE),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "base_traceback": base_traceback,
        "base_conventional": base_conventional,
        "oracle_ranking": [
            [name, results[name]["gain_traceback_vs_base"]] for name in ranking
        ],
        "outputs": report["outputs"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
