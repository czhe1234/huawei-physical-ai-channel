#!/usr/bin/env python3
"""Decompose Round 2 V3 OOF NMSE into power, phase, ramp, and port factors."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import fft as scipy_fft
from scipy.stats import spearmanr

from data import load_positions
from evaluate_v3_anchored_projection import (
    METRICS,
    TARGET_AD,
    TARGET_PAS,
    TARGET_PDP,
    target_batch,
)
from official_metrics import aggregate_statistics, point_metric_statistics
from pre_sionna_spectral import project_channel
from project_paths import PATHS
from run_rbf_spectral_field import ACTIONS, OOF_BASE, OOF_META


ARTIFACT_DIR = PATHS.artifacts / "round2" / "v3_nmse_decomposition"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v3_nmse_decomposition"
FIGURE_DIR = PATHS.analysis / "visualizations" / "复赛V3_NMSE误差分解"
REPORT = ANALYSIS_DIR / "v3_nmse_decomposition.json"
POINT_CSV = ANALYSIS_DIR / "v3_nmse_point_diagnostics.csv"

TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"
GEOMETRY = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_geometry_visibility_features.npz"
)

TINY = 1e-30
MP = 2
MH = 16
MV = 8
N_UE = 4
S_SC = 192

FIGURE_LABELS = {
    "current": "当前V3",
    "oracle_zero_state": "零覆盖状态",
    "oracle_total_power": "逐点总功率",
    "oracle_common_phase": "逐点公共相位",
    "oracle_total_power_common_phase": "总功率+公共相位",
    "oracle_point_complex_scalar": "逐点复数系数",
    "oracle_delay_ramp_point_scalar": "时延相位斜率+逐点系数",
    "oracle_horizontal_ramp_point_scalar": "水平阵列斜率+逐点系数",
    "oracle_vertical_ramp_point_scalar": "垂直阵列斜率+逐点系数",
    "oracle_joint_ramps_point_scalar": "三轴斜率+逐点系数",
    "oracle_pol_complex_scalar": "分极化复系数",
    "oracle_ue_complex_scalar": "分UE复系数",
    "oracle_polue_complex_scalar": "极化×UE复系数",
    "oracle_joint_ramps_polue_scalar": "三轴斜率+极化×UE",
    "oracle_link_complex_scalar": "逐天线链路复系数",
    "oracle_subcarrier_complex_scalar": "逐子载波复系数",
    "oracle_cfr_true_magnitude": "CFR真实幅度",
    "oracle_cfr_true_phase": "CFR真实相位",
    "oracle_delay_true_magnitude": "时延域真实幅度",
    "oracle_delay_true_phase": "时延域真实相位",
    "oracle_angle_true_magnitude": "角域真实幅度",
    "oracle_angle_true_phase": "角域真实相位",
    "oracle_ad_true_magnitude": "角时延域真实幅度",
    "oracle_ad_true_phase": "角时延域真实相位",
}


def configure_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]
    plt.rcParams["axes.unicode_minus"] = False


def summarize(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"count": 0}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
    }


def point_energy(value: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(value) ** 2, axis=tuple(range(1, value.ndim)), dtype=np.float64)


def point_error(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.sum(
        np.abs(prediction - truth) ** 2,
        axis=tuple(range(1, prediction.ndim)),
        dtype=np.float64,
    )


def rescale_power(value: np.ndarray, desired_power: np.ndarray) -> np.ndarray:
    current = point_energy(value)
    scale = np.sqrt(
        np.divide(
            desired_power,
            np.maximum(current, TINY),
            out=np.zeros_like(desired_power, dtype=np.float64),
            where=desired_power > 0.0,
        )
    ).astype(np.float32)
    return (value * scale.reshape((-1,) + (1,) * (value.ndim - 1))).astype(np.complex64)


def unit_phase(value: np.ndarray) -> np.ndarray:
    magnitude = np.abs(value)
    return np.divide(
        value,
        np.maximum(magnitude, TINY),
        out=np.ones_like(value, dtype=np.complex64),
        where=magnitude > 0.0,
    ).astype(np.complex64)


def complex_scalar_oracle(
    source: np.ndarray,
    truth: np.ndarray,
    reduce_axes: tuple[int, ...],
) -> np.ndarray:
    inner = np.sum(
        np.conj(source) * truth,
        axis=reduce_axes,
        keepdims=True,
        dtype=np.complex128,
    )
    denominator = np.sum(
        np.abs(source) ** 2,
        axis=reduce_axes,
        keepdims=True,
        dtype=np.float64,
    )
    factor = np.divide(
        inner,
        np.maximum(denominator, TINY),
        out=np.zeros_like(inner),
        where=denominator > 0.0,
    )
    return (source * factor).astype(np.complex64)


def common_phase_oracle(source: np.ndarray, truth: np.ndarray) -> np.ndarray:
    inner = np.sum(
        np.conj(source) * truth,
        axis=tuple(range(1, source.ndim)),
        keepdims=True,
        dtype=np.complex128,
    )
    factor = unit_phase(inner)
    return (source * factor).astype(np.complex64)


def apply_best_axis_ramp(
    source: np.ndarray,
    truth: np.ndarray,
    axis: int,
    nfft: int,
) -> tuple[np.ndarray, np.ndarray]:
    reduce_axes = tuple(index for index in range(1, source.ndim) if index != axis)
    cross = np.sum(
        np.conj(source) * truth,
        axis=reduce_axes,
        dtype=np.complex128,
    )
    spectrum = scipy_fft.fft(cross, n=nfft, axis=1, workers=-1)
    best = np.argmax(np.abs(spectrum), axis=1)
    slope = 2.0 * np.pi * best.astype(np.float64) / float(nfft)
    slope = (slope + np.pi) % (2.0 * np.pi) - np.pi
    index = np.arange(source.shape[axis], dtype=np.float64)
    ramp = np.exp(1j * slope[:, None] * index[None, :]).astype(np.complex64)
    shape = [len(source)] + [1] * (source.ndim - 1)
    shape[axis] = source.shape[axis]
    return (source * ramp.reshape(shape)).astype(np.complex64), slope


def ramp_scalar_oracle(
    source: np.ndarray,
    truth: np.ndarray,
    axes: tuple[tuple[int, int], ...],
    iterations: int,
    scalar_axes: tuple[int, ...],
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    output = source.copy()
    slopes = {axis: np.zeros(len(source), dtype=np.float64) for axis, _ in axes}
    for _ in range(iterations):
        for axis, nfft in axes:
            output, delta = apply_best_axis_ramp(output, truth, axis, nfft)
            slopes[axis] = (slopes[axis] + delta + np.pi) % (2.0 * np.pi) - np.pi
    return complex_scalar_oracle(output, truth, scalar_axes), slopes


def transformed_magnitude_phase_errors(
    source: np.ndarray,
    truth: np.ndarray,
    factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    true_magnitude = np.abs(truth).astype(np.float32) * unit_phase(source)
    true_phase = np.abs(source).astype(np.float32) * unit_phase(truth)
    return (
        point_error(true_magnitude, truth) / factor,
        point_error(true_phase, truth) / factor,
    )


def grouped_nmse(
    error: np.ndarray,
    energy: np.ndarray,
    mask: np.ndarray,
) -> float:
    denominator = float(np.sum(energy[mask], dtype=np.float64))
    return float(np.sum(error[mask], dtype=np.float64) / max(denominator, TINY))


def grouped_rows(
    errors: dict[str, np.ndarray],
    energy: np.ndarray,
    groups: dict[str, np.ndarray],
    names: tuple[str, ...],
) -> dict[str, dict[str, dict[str, float | int]]]:
    output: dict[str, dict[str, dict[str, float | int]]] = {}
    for group_name, labels in groups.items():
        output[group_name] = {}
        for label in np.unique(labels):
            mask = labels == label
            output[group_name][str(label)] = {
                "count": int(np.sum(mask)),
                "truth_energy_fraction": float(
                    np.sum(energy[mask], dtype=np.float64)
                    / max(float(np.sum(energy, dtype=np.float64)), TINY)
                ),
                "current_error_fraction": float(
                    np.sum(errors["current"][mask], dtype=np.float64)
                    / max(float(np.sum(errors["current"], dtype=np.float64)), TINY)
                ),
                **{
                    name: grouped_nmse(errors[name], energy, mask)
                    for name in names
                },
            }
    return output


def plot_oracle_headroom(results: dict[str, dict[str, object]]) -> None:
    names = (
        "oracle_total_power",
        "oracle_common_phase",
        "oracle_point_complex_scalar",
        "oracle_delay_ramp_point_scalar",
        "oracle_horizontal_ramp_point_scalar",
        "oracle_vertical_ramp_point_scalar",
        "oracle_joint_ramps_point_scalar",
        "oracle_polue_complex_scalar",
        "oracle_joint_ramps_polue_scalar",
        "oracle_link_complex_scalar",
        "oracle_ad_true_magnitude",
        "oracle_ad_true_phase",
    )
    gains = np.asarray(
        [float(results["current"]["nmse"]) - float(results[name]["nmse"]) for name in names]
    )
    order = np.argsort(gains)
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ["#ba4a46" if gains[index] < 0 else "#277a68" for index in order]
    ax.barh(
        np.arange(len(names)),
        gains[order],
        color=colors,
    )
    ax.set_yticks(
        np.arange(len(names)),
        [FIGURE_LABELS[names[index]] for index in order],
    )
    ax.axvline(0.0, color="#333333", linewidth=1)
    ax.set_xlabel("相对当前V3可降低的NMSE（越大越重要）")
    ax.set_title("V3复信道误差来源的单项Oracle上限")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_NMSE误差来源上限.png", dpi=190)
    plt.close(fig)


def plot_group_nmse(
    grouped: dict[str, dict[str, dict[str, float | int]]],
) -> None:
    methods = (
        "current",
        "oracle_point_complex_scalar",
        "oracle_delay_ramp_point_scalar",
        "oracle_polue_complex_scalar",
        "oracle_link_complex_scalar",
    )
    method_labels = [FIGURE_LABELS[name] for name in methods]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, group_name, title in zip(
        axes,
        ("site", "fold"),
        ("不同基站", "不同空间折"),
    ):
        labels = list(grouped[group_name])
        x = np.arange(len(labels), dtype=np.float64)
        width = 0.15
        for index, (method, method_label) in enumerate(zip(methods, method_labels)):
            values = [float(grouped[group_name][label][method]) for label in labels]
            ax.bar(x + (index - 2) * width, values, width, label=method_label)
        ax.set_xticks(x, labels)
        ax.set_ylabel("NMSE（越低越好）")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    axes[1].legend(fontsize=9, ncol=2)
    fig.suptitle("V3 NMSE在基站和空间洞上的误差分解")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "02_基站与空间折NMSE分解.png", dpi=190)
    plt.close(fig)


def plot_power_coherence(
    truth_energy: np.ndarray,
    coherence: np.ndarray,
    point_nmse: np.ndarray,
    site: np.ndarray,
) -> None:
    nonzero = truth_energy > TINY
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for site_value, color in ((0, "#2878b5"), (1, "#d05a45")):
        mask = nonzero & (site == site_value)
        axes[0].scatter(
            np.log10(truth_energy[mask]),
            coherence[mask],
            s=22,
            alpha=0.7,
            color=color,
            label=f"基站{site_value}",
        )
        axes[1].scatter(
            coherence[mask],
            np.log10(np.maximum(point_nmse[mask], 1e-8)),
            s=22,
            alpha=0.7,
            color=color,
            label=f"基站{site_value}",
        )
    axes[0].set_xlabel("真实总功率 log10")
    axes[0].set_ylabel("复数相干度")
    axes[0].set_title("功率与复数相干度")
    axes[1].set_xlabel("复数相干度")
    axes[1].set_ylabel("逐点NMSE log10")
    axes[1].set_title("相干度决定复信道误差")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "03_功率相干度与NMSE关系.png", dpi=190)
    plt.close(fig)


def plot_spatial_errors(
    positions: np.ndarray,
    site: np.ndarray,
    coherence: np.ndarray,
    error_share: np.ndarray,
    point_scalar_gain: np.ndarray,
) -> None:
    fields = (
        (coherence, "复数相干度", "viridis", None),
        (np.log10(np.maximum(error_share, 1e-10)), "总误差贡献 log10", "magma", None),
        (
            point_scalar_gain,
            "逐点复系数可降低的全局NMSE贡献",
            "RdYlGn",
            0.0,
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    for ax, (values, title, cmap, center) in zip(axes, fields):
        finite = np.isfinite(values)
        lower = float(np.quantile(values[finite], 0.02))
        upper = float(np.quantile(values[finite], 0.98))
        if center is not None:
            bound = max(abs(lower), abs(upper), 1e-6)
            lower, upper = -bound, bound
        image = ax.scatter(
            positions[:, 0],
            positions[:, 1],
            c=np.clip(values, lower, upper),
            s=30,
            cmap=cmap,
            vmin=lower,
            vmax=upper,
            linewidths=0,
        )
        ax.set_title(title)
        ax.set_xlabel("X坐标（米）")
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("Y坐标（米）")
    fig.suptitle("V3空间洞中的NMSE误差位置")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "04_NMSE空间误差分布.png", dpi=190)
    plt.close(fig)


def write_point_csv(
    query_ids: np.ndarray,
    fold_id: np.ndarray,
    site: np.ndarray,
    positions: np.ndarray,
    selected_action: np.ndarray,
    selected_scale: np.ndarray,
    truth_energy: np.ndarray,
    predicted_energy: np.ndarray,
    coherence: np.ndarray,
    point_nmse: np.ndarray,
    error_share: np.ndarray,
    pas: np.ndarray,
    pdp: np.ndarray,
    center_hit_count: np.ndarray,
) -> None:
    fields = (
        "row",
        "train_id",
        "fold",
        "site",
        "x",
        "y",
        "selected_action",
        "selected_scale",
        "truth_energy",
        "predicted_power_ratio",
        "complex_coherence",
        "point_nmse",
        "global_error_share",
        "pas",
        "pdp",
        "center_hit_count",
    )
    with POINT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in range(len(query_ids)):
            writer.writerow(
                {
                    "row": row,
                    "train_id": int(query_ids[row]),
                    "fold": int(fold_id[row]),
                    "site": int(site[row]),
                    "x": float(positions[row, 0]),
                    "y": float(positions[row, 1]),
                    "selected_action": str(ACTIONS[int(selected_action[row])].name),
                    "selected_scale": float(selected_scale[row]),
                    "truth_energy": float(truth_energy[row]),
                    "predicted_power_ratio": float(
                        predicted_energy[row] / max(truth_energy[row], TINY)
                    ),
                    "complex_coherence": float(coherence[row]),
                    "point_nmse": float(point_nmse[row]),
                    "global_error_share": float(error_share[row]),
                    "pas": float(pas[row]),
                    "pdp": float(pdp[row]),
                    "center_hit_count": float(center_hit_count[row]),
                }
            )


def analyze(device: torch.device, batch_size: int) -> dict[str, object]:
    metadata = np.load(OOF_META)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    metrics = np.load(METRICS)
    if not np.array_equal(metrics["query_ids"], query_ids):
        raise RuntimeError("V3 metric/query order mismatch")
    selected_action = metrics["selected_action"].astype(np.int16)
    selected_scale = metrics["selected_scale"].astype(np.float32)
    site = metrics["site"].astype(np.int8)
    selected_pas = np.take_along_axis(
        metrics["pas"], selected_action[None, :], axis=0
    )[0]
    selected_pdp = np.take_along_axis(
        metrics["pdp"], selected_action[None, :], axis=0
    )[0]

    _, train_positions, _, train_site, _ = load_positions()
    if not np.array_equal(train_site[query_ids], site):
        raise RuntimeError("site labels do not match")
    positions = train_positions[query_ids, :2].astype(np.float64)
    geometry_cache = np.load(GEOMETRY)
    geometry_names = [str(name) for name in geometry_cache["names"]]
    center_index = geometry_names.index("center_hit_count")
    center_hit_count = geometry_cache["train"][query_ids, center_index].astype(np.float64)

    ad_cache = np.load(TARGET_AD, mmap_mode="r")
    pas_cache = np.load(TARGET_PAS, mmap_mode="r")
    pdp_cache = np.load(TARGET_PDP, mmap_mode="r")
    base = np.load(OOF_BASE, mmap_mode="r")
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")

    errors_raw: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    truth_energy_parts: list[np.ndarray] = []
    predicted_energy_parts: list[np.ndarray] = []
    inner_parts: list[np.ndarray] = []
    point_pas_parts: list[np.ndarray] = []
    point_pdp_parts: list[np.ndarray] = []
    slope_parts: defaultdict[str, list[np.ndarray]] = defaultdict(list)

    def append_error(name: str, prediction: np.ndarray, truth: np.ndarray) -> None:
        errors_raw[name].append(point_error(prediction, truth))

    for start in range(0, len(query_ids), batch_size):
        end = min(start + batch_size, len(query_ids))
        ad, pas, pdp = target_batch(
            ad_cache, pas_cache, pdp_cache, start, end, device
        )
        base_tensor = torch.as_tensor(
            np.array(base[start:end], dtype=np.complex64, copy=True), device=device
        )
        prediction = np.empty_like(np.asarray(base[start:end]))
        local_actions = selected_action[start:end]
        for action_index in np.unique(local_actions):
            rows = np.flatnonzero(local_actions == action_index)
            action = ACTIONS[int(action_index)]
            with torch.inference_mode():
                projected = project_channel(
                    base_tensor[rows],
                    ad[rows],
                    pas[rows],
                    pdp[rows],
                    action.alpha_ad,
                    action.alpha_pas,
                    action.alpha_pdp,
                )
            prediction[rows] = projected.cpu().numpy()
        prediction *= selected_scale[start:end, None, None, None]
        prediction = prediction.astype(np.complex64, copy=False)
        truth = np.array(truth_all[query_ids[start:end]], dtype=np.complex64, copy=True)

        official = point_metric_statistics(prediction, truth)
        point_pas_parts.append(official.pas)
        point_pdp_parts.append(official.pdp)
        truth_power = point_energy(truth)
        predicted_power = point_energy(prediction)
        inner = np.sum(
            np.conj(prediction) * truth,
            axis=(1, 2, 3),
            dtype=np.complex128,
        )
        truth_energy_parts.append(truth_power)
        predicted_energy_parts.append(predicted_power)
        inner_parts.append(inner)

        append_error("current", prediction, truth)
        zero_state = prediction.copy()
        zero_state[truth_power <= TINY] = 0.0
        append_error("oracle_zero_state", zero_state, truth)
        append_error("oracle_total_power", rescale_power(prediction, truth_power), truth)
        common_phase = common_phase_oracle(prediction, truth)
        append_error("oracle_common_phase", common_phase, truth)
        append_error(
            "oracle_total_power_common_phase",
            rescale_power(common_phase, truth_power),
            truth,
        )
        append_error(
            "oracle_point_complex_scalar",
            complex_scalar_oracle(prediction, truth, (1, 2, 3)),
            truth,
        )

        prediction6 = prediction.reshape(end - start, MP, MH, MV, N_UE, S_SC)
        truth6 = truth.reshape(end - start, MP, MH, MV, N_UE, S_SC)
        append_error(
            "oracle_pol_complex_scalar",
            complex_scalar_oracle(prediction6, truth6, (2, 3, 4, 5)),
            truth6,
        )
        append_error(
            "oracle_ue_complex_scalar",
            complex_scalar_oracle(prediction6, truth6, (1, 2, 3, 5)),
            truth6,
        )
        append_error(
            "oracle_polue_complex_scalar",
            complex_scalar_oracle(prediction6, truth6, (2, 3, 5)),
            truth6,
        )
        append_error(
            "oracle_link_complex_scalar",
            complex_scalar_oracle(prediction6, truth6, (5,)),
            truth6,
        )
        append_error(
            "oracle_subcarrier_complex_scalar",
            complex_scalar_oracle(prediction6, truth6, (1, 2, 3, 4)),
            truth6,
        )

        delay_ramp, delay_slopes = ramp_scalar_oracle(
            prediction6,
            truth6,
            ((5, 3072),),
            1,
            (1, 2, 3, 4, 5),
        )
        append_error("oracle_delay_ramp_point_scalar", delay_ramp, truth6)
        slope_parts["delay"].append(delay_slopes[5])
        horizontal_ramp, horizontal_slopes = ramp_scalar_oracle(
            prediction6,
            truth6,
            ((2, 512),),
            1,
            (1, 2, 3, 4, 5),
        )
        append_error("oracle_horizontal_ramp_point_scalar", horizontal_ramp, truth6)
        slope_parts["horizontal"].append(horizontal_slopes[2])
        vertical_ramp, vertical_slopes = ramp_scalar_oracle(
            prediction6,
            truth6,
            ((3, 256),),
            1,
            (1, 2, 3, 4, 5),
        )
        append_error("oracle_vertical_ramp_point_scalar", vertical_ramp, truth6)
        slope_parts["vertical"].append(vertical_slopes[3])
        joint_ramp, joint_slopes = ramp_scalar_oracle(
            prediction6,
            truth6,
            ((5, 3072), (2, 512), (3, 256)),
            2,
            (1, 2, 3, 4, 5),
        )
        append_error("oracle_joint_ramps_point_scalar", joint_ramp, truth6)
        joint_polue = complex_scalar_oracle(joint_ramp, truth6, (2, 3, 5))
        append_error("oracle_joint_ramps_polue_scalar", joint_polue, truth6)
        slope_parts["joint_delay"].append(joint_slopes[5])
        slope_parts["joint_horizontal"].append(joint_slopes[2])
        slope_parts["joint_vertical"].append(joint_slopes[3])

        cfr_mag, cfr_phase = transformed_magnitude_phase_errors(
            prediction6, truth6, 1.0
        )
        errors_raw["oracle_cfr_true_magnitude"].append(cfr_mag)
        errors_raw["oracle_cfr_true_phase"].append(cfr_phase)
        prediction_delay = scipy_fft.ifft(prediction6, axis=5, workers=-1)
        truth_delay = scipy_fft.ifft(truth6, axis=5, workers=-1)
        delay_mag, delay_phase = transformed_magnitude_phase_errors(
            prediction_delay, truth_delay, 1.0 / S_SC
        )
        errors_raw["oracle_delay_true_magnitude"].append(delay_mag)
        errors_raw["oracle_delay_true_phase"].append(delay_phase)
        prediction_angle = scipy_fft.fft2(prediction6, axes=(2, 3), workers=-1)
        truth_angle = scipy_fft.fft2(truth6, axes=(2, 3), workers=-1)
        angle_mag, angle_phase = transformed_magnitude_phase_errors(
            prediction_angle, truth_angle, float(MH * MV)
        )
        errors_raw["oracle_angle_true_magnitude"].append(angle_mag)
        errors_raw["oracle_angle_true_phase"].append(angle_phase)
        prediction_ad = scipy_fft.ifft(prediction_angle, axis=5, workers=-1)
        truth_ad = scipy_fft.ifft(truth_angle, axis=5, workers=-1)
        ad_mag, ad_phase = transformed_magnitude_phase_errors(
            prediction_ad, truth_ad, float(MH * MV) / S_SC
        )
        errors_raw["oracle_ad_true_magnitude"].append(ad_mag)
        errors_raw["oracle_ad_true_phase"].append(ad_phase)
        print(f"V3 NMSE decomposition {end}/{len(query_ids)}", flush=True)

    errors = {name: np.concatenate(parts) for name, parts in errors_raw.items()}
    truth_energy = np.concatenate(truth_energy_parts)
    predicted_energy = np.concatenate(predicted_energy_parts)
    inner = np.concatenate(inner_parts)
    point_pas = np.concatenate(point_pas_parts)
    point_pdp = np.concatenate(point_pdp_parts)
    slopes = {name: np.concatenate(parts) for name, parts in slope_parts.items()}

    coherence = np.divide(
        np.abs(inner),
        np.sqrt(np.maximum(predicted_energy * truth_energy, TINY)),
        out=np.zeros_like(truth_energy),
        where=(predicted_energy > TINY) & (truth_energy > TINY),
    )
    point_nmse = errors["current"] / np.maximum(truth_energy, TINY)
    error_share = errors["current"] / max(float(np.sum(errors["current"])), TINY)
    total_energy = float(np.sum(truth_energy, dtype=np.float64))
    current_pas = float(np.mean(point_pas, dtype=np.float64))
    current_pdp = float(np.mean(point_pdp, dtype=np.float64))
    spectral_constant = 0.4 * current_pas + 0.4 * current_pdp
    current_nmse = float(np.sum(errors["current"], dtype=np.float64) / max(total_energy, TINY))
    current_nmse_score = 1.0 / (1.0 + current_nmse)

    def energy_decomposition(mask: np.ndarray) -> dict[str, float | int]:
        local_truth = float(np.sum(truth_energy[mask], dtype=np.float64))
        local_prediction = float(np.sum(predicted_energy[mask], dtype=np.float64))
        local_inner = np.sum(inner[mask], dtype=np.complex128)
        valid_scale = mask & (selected_scale > 0.0)
        pre_shrink_prediction = float(
            np.sum(
                predicted_energy[valid_scale]
                / np.square(selected_scale[valid_scale].astype(np.float64)),
                dtype=np.float64,
            )
        )
        denominator = np.sqrt(max(local_truth * local_prediction, TINY))
        energy_ratio = local_prediction / max(local_truth, TINY)
        real_cross_ratio = float(np.real(local_inner)) / max(local_truth, TINY)
        return {
            "count": int(np.sum(mask)),
            "truth_energy_fraction": local_truth / max(total_energy, TINY),
            "final_predicted_to_truth_energy": energy_ratio,
            "pre_shrink_predicted_to_truth_energy": (
                pre_shrink_prediction / max(local_truth, TINY)
            ),
            "real_cross_to_truth_energy": real_cross_ratio,
            "aggregate_real_cosine": float(np.real(local_inner)) / denominator,
            "aggregate_complex_cosine": float(np.abs(local_inner)) / denominator,
            "nmse_from_identity": 1.0 + energy_ratio - 2.0 * real_cross_ratio,
        }

    results: dict[str, dict[str, object]] = {}
    for name, values in errors.items():
        nmse = float(np.sum(values, dtype=np.float64) / max(total_energy, TINY))
        nmse_score = 1.0 / (1.0 + nmse)
        results[name] = {
            "label_zh": FIGURE_LABELS.get(name, name),
            "nmse": nmse,
            "nmse_reduction_vs_current": current_nmse - nmse,
            "nmse_score": nmse_score,
            "nmse_score_gain_vs_current": nmse_score - current_nmse_score,
            "equivalent_total_if_pas_pdp_fixed": spectral_constant + 0.2 * nmse_score,
            "equivalent_total_gain_if_pas_pdp_fixed": 0.2 * (nmse_score - current_nmse_score),
            "point_nmse": summarize(values / np.maximum(truth_energy, TINY)),
        }

    nonzero = truth_energy > TINY
    power_group = np.full(len(query_ids), "zero", dtype="U12")
    if np.any(nonzero):
        thresholds = np.quantile(np.log10(truth_energy[nonzero]), (0.25, 0.50, 0.75))
        power_group[nonzero] = np.asarray(
            ["q1_low", "q2", "q3", "q4_high"], dtype="U12"
        )[np.digitize(np.log10(truth_energy[nonzero]), thresholds)]
    else:
        thresholds = np.zeros(3, dtype=np.float64)
    visibility = np.where(center_hit_count > 0.0, "center_blocked", "center_clear")
    key_group_methods = (
        "current",
        "oracle_total_power",
        "oracle_point_complex_scalar",
        "oracle_delay_ramp_point_scalar",
        "oracle_joint_ramps_point_scalar",
        "oracle_polue_complex_scalar",
        "oracle_joint_ramps_polue_scalar",
        "oracle_link_complex_scalar",
        "oracle_ad_true_magnitude",
        "oracle_ad_true_phase",
    )
    groups = grouped_rows(
        errors,
        truth_energy,
        {
            "site": site.astype("U4"),
            "fold": fold_id.astype("U4"),
            "power_quartile": power_group,
            "center_visibility": visibility,
        },
        key_group_methods,
    )

    def correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        mask = np.isfinite(x) & np.isfinite(y)
        if np.sum(mask) < 3:
            return {"spearman": float("nan"), "pvalue": float("nan")}
        result = spearmanr(x[mask], y[mask])
        return {"spearman": float(result.statistic), "pvalue": float(result.pvalue)}

    correlations = {
        "point_nmse_vs_log_truth_power": correlation(
            point_nmse[nonzero], np.log10(truth_energy[nonzero])
        ),
        "point_nmse_vs_complex_coherence": correlation(point_nmse, coherence),
        "error_share_vs_log_truth_power": correlation(
            error_share[nonzero], np.log10(truth_energy[nonzero])
        ),
        "point_nmse_vs_pas_deficit": correlation(point_nmse, 1.0 - point_pas),
        "point_nmse_vs_pdp_deficit": correlation(point_nmse, 1.0 - point_pdp),
        "point_nmse_vs_center_hit_count": correlation(point_nmse, center_hit_count),
    }

    point_scalar_reduction = (
        errors["current"] - errors["oracle_point_complex_scalar"]
    ) / max(total_energy, TINY)
    top_error_order = np.argsort(error_share)[::-1]
    diagnostics = {
        "current_pas": current_pas,
        "current_pdp": current_pdp,
        "current_nmse": current_nmse,
        "current_nmse_score": current_nmse_score,
        "current_total_score": spectral_constant + 0.2 * current_nmse_score,
        "perfect_nmse_total_score_if_pas_pdp_fixed": spectral_constant + 0.2,
        "maximum_total_gain_from_nmse_if_pas_pdp_fixed": 0.2 * (
            1.0 - current_nmse_score
        ),
        "truth_zero_count": int(np.sum(~nonzero)),
        "selected_scale": {
            "all": summarize(selected_scale),
            "site_0": summarize(selected_scale[site == 0]),
            "site_1": summarize(selected_scale[site == 1]),
        },
        "energy_decomposition": {
            "all": energy_decomposition(np.ones(len(query_ids), dtype=bool)),
            "site_0": energy_decomposition(site == 0),
            "site_1": energy_decomposition(site == 1),
        },
        "predicted_power_ratio": summarize(
            predicted_energy[nonzero] / np.maximum(truth_energy[nonzero], TINY)
        ),
        "complex_coherence": summarize(coherence[nonzero]),
        "point_nmse_nonzero": summarize(point_nmse[nonzero]),
        "top_10_percent_error_share": float(
            np.sum(error_share[top_error_order[: max(1, len(query_ids) // 10)]])
        ),
        "top_20_points_error_share": float(np.sum(error_share[top_error_order[:20]])),
        "top_error_train_ids": query_ids[top_error_order[:20]].astype(int).tolist(),
        "power_quartile_log10_thresholds": thresholds.tolist(),
        "correlations": correlations,
        "ramp_slopes_rad_per_index": {
            name: summarize(value) for name, value in slopes.items()
        },
    }

    expected = aggregate_statistics(
        point_metric_statistics(
            np.zeros((1, 256, 4, 192), dtype=np.complex64),
            np.zeros((1, 256, 4, 192), dtype=np.complex64),
        )
    )
    del expected
    reference_nmse = 0.9526311803959002
    reference_score = 0.6345084646755442
    if abs(current_nmse - reference_nmse) > 2e-5:
        raise RuntimeError(f"reconstructed NMSE {current_nmse} != {reference_nmse}")
    if abs(diagnostics["current_total_score"] - reference_score) > 2e-5:
        raise RuntimeError(
            f"reconstructed score {diagnostics['current_total_score']} != {reference_score}"
        )
    if not np.allclose(point_pas, selected_pas, atol=2e-5, rtol=2e-5):
        raise RuntimeError("reconstructed PAS does not match cached V3 metrics")
    if not np.allclose(point_pdp, selected_pdp, atol=2e-5, rtol=2e-5):
        raise RuntimeError("reconstructed PDP does not match cached V3 metrics")

    write_point_csv(
        query_ids,
        fold_id,
        site,
        positions,
        selected_action,
        selected_scale,
        truth_energy,
        predicted_energy,
        coherence,
        point_nmse,
        error_share,
        point_pas,
        point_pdp,
        center_hit_count,
    )
    plot_oracle_headroom(results)
    plot_group_nmse(groups)
    plot_power_coherence(truth_energy, coherence, point_nmse, site)
    plot_spatial_errors(
        positions,
        site,
        coherence,
        error_share,
        point_scalar_reduction,
    )

    ranking = sorted(
        results,
        key=lambda name: float(results[name]["nmse"]),
    )
    return {
        "experiment": "Round2-E5-V3-NMSE-error-decomposition",
        "status": "analysis_only",
        "sionna_used": False,
        "official_inputs_only": True,
        "scope": (
            "Strict V3 five-fold OOF predictions. Oracle truth is used only to attribute "
            "headroom; no test output or submission is generated."
        ),
        "count": int(len(query_ids)),
        "diagnostics": diagnostics,
        "oracle_results": results,
        "oracle_ranking_low_nmse_first": ranking,
        "grouped_nmse": groups,
        "files": {
            "point_csv": PATHS.relative(POINT_CSV),
            "figure_dir": PATHS.relative(FIGURE_DIR),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    configure_chinese_font()
    payload = analyze(torch.device(args.device), args.batch_size)
    payload["seconds"] = float(time.time() - started)
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "diagnostics": payload["diagnostics"],
                "oracle_ranking": payload["oracle_ranking_low_nmse_first"],
                "report": PATHS.relative(REPORT),
                "seconds": payload["seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
