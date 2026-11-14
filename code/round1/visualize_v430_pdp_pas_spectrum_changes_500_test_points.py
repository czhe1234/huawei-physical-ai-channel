#!/usr/bin/env python3
"""Visualize PDP and validated leaked-PAS spectrum changes for all test points."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import generate_v423_joint_sionna_pas_pdp_diagnostic as v423
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 430
VERSION = "v430_pdp_pas_spectrum_changes_500_test_points"
OUTPUT_DIR = PATHS.analysis / "visualizations" / "500个测试点的PDP与PAS谱变化"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
CACHE = REPORT_DIR / "v430_500个测试点PDP与PAS谱缓存.npz"

REFERENCE = PATHS.cache_dir(207) / "v207_sionna_test_reference_H.npy"
CHANNELS = {
    "v209": PATHS.cache_dir(209) / "v209_Round1_Test_Channel.npy",
    "v212": PATHS.cache_dir(212) / "v212_Round1_Test_Channel.npy",
    "v424": PATHS.cache_dir(424) / "v424_Round1_Test_Channel.npy",
    "v426": PATHS.cache_dir(426) / "v426_Round1_Test_Channel.npy",
}
ALL_CHANNELS = {**CHANNELS, "reference": REFERENCE}
TEST_POS = PATHS.round1 / "Round1_Test_Pos.npy"
GEOMETRY = PATHS.shared_features / "height_geometry_cache.npz"
V429_METRICS = PATHS.version_report_dir(429) / "v429_500个测试点逐点指标.npz"

PDP_EVOLUTION = OUTPUT_DIR / "01_500点PDP谱随版本变化.png"
PDP_TRANSPORT = OUTPUT_DIR / "02_500点PDP谱三阶段能量搬运.png"
PAS_EVOLUTION = OUTPUT_DIR / "03_500点PAS谱随版本变化.png"
PAS_TRANSPORT = OUTPUT_DIR / "04_500点PAS谱三阶段能量搬运.png"
PDP_CURVES = OUTPUT_DIR / "05_代表性测试点PDP曲线.png"
PAS_CURVES = OUTPUT_DIR / "06_代表性测试点PAS曲线.png"
INTERACTIVE = OUTPUT_DIR / "07_500点PDP和PAS交互谱图.html"
CSV_DETAILS = OUTPUT_DIR / "500点谱图排序与指标.csv"
JSON_SUMMARY = OUTPUT_DIR / "500点谱变化摘要.json"

VERSION_KEYS = ("v209", "v212", "v424", "v426", "reference")
DISPLAY_NAMES = {
    "v209": "v209：修正前底座",
    "v212": "v212：PDP补全后",
    "v424": "v424：PAS修正后",
    "v426": "v426：逐链路PDP后",
    "reference": "Sionna同坐标诊断参考",
}
CURVE_COLORS = {
    "v209": "#6b7280",
    "v212": "#2563eb",
    "v424": "#db2777",
    "v426": "#059669",
    "reference": "#111827",
}
REGIMES = ("clear", "light", "medium", "heavy")
REGIME_CN = {
    "clear": "无遮挡",
    "light": "轻遮挡",
    "medium": "中遮挡",
    "heavy": "重遮挡",
}
STAGES = (
    ("阶段一：v209→v212\n补全归一化时延/PDP", "v209", "v212"),
    ("阶段二：v212→v424\n修正已验证PAS口径", "v212", "v424"),
    ("阶段三：v424→v426\n补全1024条链路PDP", "v424", "v426"),
)
ROW_DB_FLOOR = -40.0
BATCH = 4
TINY = 1e-30


def configure_chinese_font() -> None:
    selected = "DejaVu Sans"
    for family in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Droid Sans Fallback"):
        try:
            font_path = font_manager.findfont(
                font_manager.FontProperties(family=family), fallback_to_default=False
            )
            font_manager.fontManager.addfont(font_path)
            selected = font_manager.FontProperties(fname=font_path).get_name()
            break
        except ValueError:
            continue
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [selected, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 130,
            "savefig.dpi": 220,
        }
    )


def validate_inputs() -> None:
    required = (*ALL_CHANNELS.values(), TEST_POS, GEOMETRY, V429_METRICS)
    missing = [PATHS.relative(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"缺少v430输入文件：{missing}")


def normalize_rows(value: np.ndarray) -> np.ndarray:
    output = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    total = output.sum(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    output /= np.maximum(total, TINY)
    return output.astype(np.float32)


def linkwise_mean_pdp(channel: np.ndarray) -> np.ndarray:
    """Average normalized PDP shapes after normalizing every 256x4 link."""
    cir = np.fft.ifft(channel, axis=-1)
    magnitude = np.abs(cir).astype(np.float32)
    power = magnitude * magnitude
    link_total = power.sum(axis=-1, dtype=np.float64).astype(np.float32)
    valid = link_total > 1e-20
    shape = power / np.maximum(link_total[..., None], TINY)
    summed = np.sum(shape * valid[..., None], axis=(1, 2), dtype=np.float64)
    count = np.sum(valid, axis=(1, 2), dtype=np.int32)
    mean_shape = summed / np.maximum(count[:, None], 1)
    return normalize_rows(mean_shape)


def validated_pas(channel: np.ndarray) -> np.ndarray:
    """Collapse the online-effective leaked support to its 16 H-angle bins."""
    support, _ = v423.leaked_support(channel)
    pas = np.sum(support, axis=(1, 2, 4, 5), dtype=np.float64)
    return normalize_rows(pas)


def compute_spectra() -> dict[str, np.ndarray]:
    count = int(np.load(TEST_POS, mmap_mode="r").shape[0])
    output: dict[str, np.ndarray] = {}
    for name, path in ALL_CHANNELS.items():
        channel = np.load(path, mmap_mode="r")
        if channel.shape != (count, 256, 4, 192):
            raise ValueError(f"{PATHS.relative(path)} shape异常：{channel.shape}")
        pdp = np.zeros((count, 192), dtype=np.float32)
        pas = np.zeros((count, 16), dtype=np.float32)
        for start in range(0, count, BATCH):
            end = min(start + BATCH, count)
            block = np.asarray(channel[start:end], dtype=np.complex64)
            pdp[start:end] = linkwise_mean_pdp(block)
            pas[start:end] = validated_pas(block)
            if end == count or end % 100 == 0:
                print(f"v430 计算{name}谱：{end}/{count}", flush=True)
        output[f"{name}_pdp"] = pdp
        output[f"{name}_pas"] = pas
    return output


def load_or_compute_spectra() -> dict[str, np.ndarray]:
    expected = {f"{name}_{metric}" for name in VERSION_KEYS for metric in ("pdp", "pas")}
    if CACHE.is_file():
        cached = np.load(CACHE)
        if expected.issubset(cached.files):
            print("v430 复用已生成的PDP/PAS谱缓存", flush=True)
            return {key: cached[key] for key in expected}
    spectra = compute_spectra()
    np.savez_compressed(CACHE, **spectra)
    return spectra


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=1, dtype=np.float64)
    left_norm = np.sum(left * left, axis=1, dtype=np.float64)
    right_norm = np.sum(right * right, axis=1, dtype=np.float64)
    output = numerator / np.maximum(np.sqrt(left_norm * right_norm), TINY)
    both_zero = (left_norm <= TINY) & (right_norm <= TINY)
    output[both_zero] = 1.0
    return output.astype(np.float32)


def relative_db(value: np.ndarray) -> np.ndarray:
    peak = np.max(value, axis=1, keepdims=True)
    ratio = value / np.maximum(peak, TINY)
    return np.maximum(10.0 * np.log10(np.maximum(ratio, 10.0 ** (ROW_DB_FLOOR / 10.0))), ROW_DB_FLOOR)


def amplitude_delta(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    return (np.sqrt(np.maximum(after, 0.0)) - np.sqrt(np.maximum(before, 0.0))).astype(np.float32)


def build_sort_order(
    test_pos: np.ndarray, regimes: np.ndarray
) -> tuple[np.ndarray, list[dict[str, object]]]:
    rank = np.array([REGIMES.index(str(value)) for value in regimes], dtype=np.int16)
    order = np.lexsort((test_pos[:, 0], test_pos[:, 1], rank)).astype(np.int32)
    sorted_regimes = regimes[order]
    groups: list[dict[str, object]] = []
    for regime in REGIMES:
        rows = np.flatnonzero(sorted_regimes == regime)
        if len(rows) == 0:
            continue
        groups.append(
            {
                "regime": regime,
                "start": int(rows[0]),
                "end": int(rows[-1] + 1),
                "mid": float((rows[0] + rows[-1]) / 2.0),
                "count": int(len(rows)),
            }
        )
    return order, groups


def add_group_guides(axis: plt.Axes, groups: list[dict[str, object]], show_labels: bool) -> None:
    for group in groups[1:]:
        axis.axhline(float(group["start"]) - 0.5, color="white", linewidth=1.2, alpha=0.95)
        axis.axhline(float(group["start"]) - 0.5, color="#111827", linewidth=0.35, alpha=0.9)
    if show_labels:
        axis.set_yticks([float(group["mid"]) for group in groups])
        axis.set_yticklabels(
            [f"{REGIME_CN[str(group['regime'])]}\n{group['count']}点" for group in groups]
        )
        axis.set_ylabel("500个测试点（按传播难度、y、x排序）")
    else:
        axis.tick_params(axis="y", left=False, labelleft=False)


def save_figure(figure: plt.Figure, png_path: Path) -> list[Path]:
    pdf_path = png_path.with_suffix(".pdf")
    figure.savefig(png_path, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return [png_path, pdf_path]


def plot_evolution(
    spectra: dict[str, np.ndarray],
    order: np.ndarray,
    groups: list[dict[str, object]],
    metric: str,
    output: Path,
) -> list[Path]:
    is_pdp = metric == "pdp"
    bins = 192 if is_pdp else 16
    title = "500个测试点的PDP谱如何随版本变化" if is_pdp else "500个测试点的PAS谱如何随版本变化"
    x_label = "时延bin（0-191）" if is_pdp else "水平角H-bin（0-15）"
    fig, axes = plt.subplots(1, len(VERSION_KEYS), figsize=(27, 12), sharey=True)
    image = None
    for column, (axis, name) in enumerate(zip(axes, VERSION_KEYS)):
        image = axis.imshow(
            relative_db(spectra[f"{name}_{metric}"])[order],
            aspect="auto",
            interpolation="nearest",
            origin="upper",
            cmap="viridis",
            vmin=ROW_DB_FLOOR,
            vmax=0.0,
        )
        axis.set_title(DISPLAY_NAMES[name], pad=10, fontweight="bold")
        axis.set_xlabel(x_label)
        axis.set_xlim(-0.5, bins - 0.5)
        if is_pdp:
            axis.set_xticks([0, 48, 96, 144, 191])
        else:
            axis.set_xticks([0, 4, 8, 12, 15])
        add_group_guides(axis, groups, show_labels=column == 0)
    fig.suptitle(title, fontsize=20, fontweight="bold", y=0.965)
    fig.subplots_adjust(left=0.075, right=0.91, bottom=0.075, top=0.90, wspace=0.06)
    if image is not None:
        color_axis = fig.add_axes([0.93, 0.16, 0.012, 0.66])
        colorbar = fig.colorbar(image, cax=color_axis)
        colorbar.set_label("每个测试点相对自身谱峰值（dB）")
    note = (
        "PDP：每条256×4链路先对192个时延bin归一化，再对有效链路求平均。"
        if is_pdp
        else "PAS：使用v424线上验证有效的traceback泄露口径，聚合为16个H角度bin；不是传统256阵元PAS。"
    )
    fig.text(0.5, 0.025, note, ha="center", fontsize=11, color="#334155")
    fig.text(
        0.5,
        0.008,
        "最右列是同坐标Sionna诊断参考，只用于解释谱被修向哪里，不是官方隐藏测试真值。",
        ha="center",
        fontsize=10,
        color="#b42318",
    )
    return save_figure(fig, output)


def plot_transport(
    spectra: dict[str, np.ndarray],
    order: np.ndarray,
    groups: list[dict[str, object]],
    metric: str,
    output: Path,
) -> list[Path]:
    is_pdp = metric == "pdp"
    bins = 192 if is_pdp else 16
    deltas = [
        amplitude_delta(spectra[f"{before}_{metric}"], spectra[f"{after}_{metric}"])
        for _, before, after in STAGES
    ]
    magnitude = np.concatenate([np.abs(value).reshape(-1) for value in deltas])
    limit = max(float(np.quantile(magnitude, 0.995)), 1e-6)
    fig, axes = plt.subplots(1, 3, figsize=(20, 13), sharey=True)
    image = None
    for column, (axis, stage, delta) in enumerate(zip(axes, STAGES, deltas)):
        mean_motion = float(np.mean(np.sum(np.abs(delta), axis=1)))
        image = axis.imshow(
            delta[order],
            aspect="auto",
            interpolation="nearest",
            origin="upper",
            cmap="RdBu",
            vmin=-limit,
            vmax=limit,
        )
        axis.set_title(f"{stage[0]}\n平均搬运量={mean_motion:.4f}", pad=10, fontweight="bold")
        axis.set_xlabel("时延bin（0-191）" if is_pdp else "水平角H-bin（0-15）")
        axis.set_xlim(-0.5, bins - 0.5)
        if is_pdp:
            axis.set_xticks([0, 48, 96, 144, 191])
        else:
            axis.set_xticks([0, 4, 8, 12, 15])
        add_group_guides(axis, groups, show_labels=column == 0)
    metric_name = "PDP" if is_pdp else "PAS"
    fig.suptitle(f"500个测试点的{metric_name}谱：三个阶段把能量搬到了哪里", fontsize=20, fontweight="bold", y=0.975)
    fig.subplots_adjust(left=0.085, right=0.89, bottom=0.07, top=0.82, wspace=0.08)
    if image is not None:
        color_axis = fig.add_axes([0.92, 0.17, 0.015, 0.58])
        colorbar = fig.colorbar(image, cax=color_axis)
        colorbar.set_label("sqrt(归一化功率)变化：蓝色增强，红色减弱")
    fig.text(
        0.5,
        0.025,
        "差分使用sqrt功率，既保留正负方向，也让弱多径的变化可见；三个面板共用同一颜色尺度。",
        ha="center",
        fontsize=11,
        color="#334155",
    )
    if not is_pdp:
        fig.text(
            0.5,
            0.008,
            "PAS仍指v424已验证的16-bin traceback泄露口径，不是传统PAS。",
            ha="center",
            fontsize=10,
            color="#b42318",
        )
    return save_figure(fig, output)


def select_representative_points(spectra: dict[str, np.ndarray]) -> list[tuple[str, int]]:
    metrics = np.load(V429_METRICS)
    requests = (
        ("阶段一PDP改善最大", "阶段1_PDP增量", None),
        ("阶段一PDP改善次大", "阶段1_PDP增量", None),
        ("阶段二PAS改善最大", "阶段2_leakedPAS增量", None),
        ("阶段三逐链路PDP改善最大", "阶段3_逐链路PDP增量", None),
        ("v426剩余PDP误差最大", "v426剩余PDP误差", "pdp"),
        ("v426剩余PAS误差最大", "v426剩余leakedPAS误差", "pas"),
    )
    selected: list[tuple[str, int]] = []
    used: set[int] = set()
    for label, key, required_reference in requests:
        candidates = np.argsort(metrics[key])[::-1]
        point = next(
            int(index)
            for index in candidates
            if int(index) not in used
            and (
                required_reference is None
                or float(np.sum(spectra[f"reference_{required_reference}"][int(index)])) > 0.5
            )
        )
        selected.append((label, point))
        used.add(point)
    return selected


def plot_representative_curves(
    spectra: dict[str, np.ndarray],
    selected: list[tuple[str, int]],
    test_pos: np.ndarray,
    regimes: np.ndarray,
    metric: str,
    output: Path,
) -> list[Path]:
    is_pdp = metric == "pdp"
    bins = 192 if is_pdp else 16
    reference = spectra[f"reference_{metric}"]
    fig, axes = plt.subplots(3, 2, figsize=(19, 15), sharex=False, sharey=True)
    for subplot, (axis, (reason, point)) in enumerate(zip(axes.ravel(), selected)):
        for name in VERSION_KEYS:
            values = relative_db(spectra[f"{name}_{metric}"][[point]])[0]
            line_style = "--" if name == "reference" else "-"
            marker = "o" if (not is_pdp and name != "reference") else None
            axis.plot(
                np.arange(bins),
                values,
                color=CURVE_COLORS[name],
                linewidth=2.1 if name == "reference" else 1.6,
                linestyle=line_style,
                marker=marker,
                markersize=3.2,
                label=DISPLAY_NAMES[name],
                alpha=0.95,
            )
        scores = [
            float(cosine_rows(spectra[f"{name}_{metric}"], reference)[point])
            for name in VERSION_KEYS[:-1]
        ]
        axis.text(
            0.985,
            0.03,
            "对参考谱余弦：\n"
            + " → ".join(f"{name} {score:.3f}" for name, score in zip(VERSION_KEYS[:-1], scores)),
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.86},
        )
        axis.set_title(
            f"T{point:03d}｜{reason}\n{REGIME_CN[str(regimes[point])]}，x={test_pos[point, 0]:.1f}米，y={test_pos[point, 1]:.1f}米",
            fontsize=11,
            fontweight="bold",
        )
        axis.set_xlim(0, bins - 1)
        axis.set_ylim(ROW_DB_FLOOR - 1.0, 2.0)
        axis.grid(True, color="#cbd5e1", linewidth=0.5, alpha=0.55)
        axis.set_ylabel("相对自身谱峰值（dB）")
        axis.set_xlabel("时延bin（0-191）" if is_pdp else "水平角H-bin（0-15）")
        if is_pdp:
            axis.set_xticks([0, 32, 64, 96, 128, 160, 191])
        else:
            axis.set_xticks(np.arange(0, 16, 2))
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.012))
    title = "代表性测试点的PDP曲线变化" if is_pdp else "代表性测试点的PAS曲线变化"
    fig.suptitle(title, fontsize=20, fontweight="bold", y=0.985)
    note = (
        "同一子图内每条曲线都按自身峰值归一化，关注的是谱形变化，不是总功率。"
        if is_pdp
        else "PAS采用v424已验证的16-bin traceback泄露口径；同一子图内每条曲线按自身峰值归一化。"
    )
    fig.text(0.5, 0.043, note, ha="center", fontsize=10.5, color="#334155")
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.08, top=0.93, hspace=0.34, wspace=0.14)
    return save_figure(fig, output)


def spectrum_states(spectra: dict[str, np.ndarray]) -> list[dict[str, object]]:
    states: list[dict[str, object]] = []
    for name in VERSION_KEYS:
        states.append(
            {
                "name": DISPLAY_NAMES[name],
                "kind": "absolute",
                "pdp": relative_db(spectra[f"{name}_pdp"]),
                "pas": relative_db(spectra[f"{name}_pas"]),
            }
        )
    for title, before, after in STAGES:
        states.append(
            {
                "name": title.replace("\n", "："),
                "kind": "delta",
                "pdp": amplitude_delta(spectra[f"{before}_pdp"], spectra[f"{after}_pdp"]),
                "pas": amplitude_delta(spectra[f"{before}_pas"], spectra[f"{after}_pas"]),
            }
        )
    return states


def write_interactive(
    spectra: dict[str, np.ndarray],
    order: np.ndarray,
    groups: list[dict[str, object]],
    test_pos: np.ndarray,
    regimes: np.ndarray,
) -> Path:
    states = spectrum_states(spectra)
    pdp_delta_limit = max(
        float(np.quantile(np.concatenate([np.abs(state["pdp"]).reshape(-1) for state in states if state["kind"] == "delta"]), 0.995)),
        1e-6,
    )
    pas_delta_limit = max(
        float(np.quantile(np.concatenate([np.abs(state["pas"]).reshape(-1) for state in states if state["kind"] == "delta"]), 0.995)),
        1e-6,
    )
    point_labels = [
        f"T{point:03d}｜{REGIME_CN[str(regimes[point])]}｜x={test_pos[point, 0]:.1f}米，y={test_pos[point, 1]:.1f}米"
        for point in order
    ]
    fig = make_subplots(
        rows=2,
        cols=1,
        vertical_spacing=0.09,
        subplot_titles=("PDP谱（192个时延bin）", "PAS谱（16个H角度bin）"),
    )
    for state_index, state in enumerate(states):
        absolute = state["kind"] == "absolute"
        visible = state_index == 0
        pdp_limit = 0.0 if absolute else pdp_delta_limit
        pas_limit = 0.0 if absolute else pas_delta_limit
        fig.add_trace(
            go.Heatmap(
                z=np.asarray(state["pdp"])[order],
                x=np.arange(192),
                y=point_labels,
                visible=visible,
                colorscale="Viridis" if absolute else "RdBu",
                zmin=ROW_DB_FLOOR if absolute else -pdp_limit,
                zmax=0.0 if absolute else pdp_limit,
                colorbar={
                    "title": "相对峰值dB" if absolute else "sqrt功率变化",
                    "len": 0.38,
                    "y": 0.79,
                    "x": 1.01,
                },
                hovertemplate=(
                    "测试点：%{y}<br>时延bin：%{x}<br>相对峰值：%{z:.2f} dB<extra></extra>"
                    if absolute
                    else "测试点：%{y}<br>时延bin：%{x}<br>sqrt功率变化：%{z:+.5f}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Heatmap(
                z=np.asarray(state["pas"])[order],
                x=np.arange(16),
                y=point_labels,
                visible=visible,
                colorscale="Viridis" if absolute else "RdBu",
                zmin=ROW_DB_FLOOR if absolute else -pas_limit,
                zmax=0.0 if absolute else pas_limit,
                colorbar={
                    "title": "相对峰值dB" if absolute else "sqrt功率变化",
                    "len": 0.38,
                    "y": 0.21,
                    "x": 1.01,
                },
                hovertemplate=(
                    "测试点：%{y}<br>H角度bin：%{x}<br>相对峰值：%{z:.2f} dB<extra></extra>"
                    if absolute
                    else "测试点：%{y}<br>H角度bin：%{x}<br>sqrt功率变化：%{z:+.5f}<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )
    buttons = []
    for state_index, state in enumerate(states):
        visibility = [False] * (2 * len(states))
        visibility[2 * state_index] = True
        visibility[2 * state_index + 1] = True
        buttons.append(
            {
                "label": str(state["name"]),
                "method": "update",
                "args": [
                    {"visible": visibility},
                    {"title": f"500个测试点PDP/PAS交互谱图｜{state['name']}"},
                ],
            }
        )
    tick_values = [point_labels[int(round(float(group["mid"])))] for group in groups]
    tick_text = [f"{REGIME_CN[str(group['regime'])]}（{group['count']}点）" for group in groups]
    fig.update_yaxes(
        autorange="reversed",
        tickmode="array",
        tickvals=tick_values,
        ticktext=tick_text,
        title_text="测试点分组",
        row=1,
        col=1,
    )
    fig.update_yaxes(
        autorange="reversed",
        tickmode="array",
        tickvals=tick_values,
        ticktext=tick_text,
        title_text="测试点分组",
        row=2,
        col=1,
    )
    fig.update_xaxes(title_text="时延bin", row=1, col=1)
    fig.update_xaxes(title_text="水平角H-bin", row=2, col=1)
    fig.update_layout(
        title="500个测试点PDP/PAS交互谱图｜v209：修正前底座",
        width=1500,
        height=1150,
        margin={"l": 170, "r": 135, "t": 145, "b": 95},
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 0.0,
                "xanchor": "left",
                "y": 1.105,
                "yanchor": "top",
                "showactive": True,
            }
        ],
        annotations=list(fig.layout.annotations)
        + [
            {
                "text": "下拉菜单可切换五套绝对谱和三个阶段差分；悬停可查看T编号、坐标、传播难度和bin。PAS为v424已验证的16-bin泄露口径。",
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": -0.075,
                "showarrow": False,
                "font": {"size": 13, "color": "#334155"},
            },
            {
                "text": "Sionna同坐标谱仅是诊断参考，不是官方隐藏测试真值。",
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": -0.105,
                "showarrow": False,
                "font": {"size": 12, "color": "#b42318"},
            },
        ],
    )
    fig.write_html(INTERACTIVE, include_plotlyjs=True, full_html=True)
    return INTERACTIVE


def normalized_entropy(value: np.ndarray) -> np.ndarray:
    width = value.shape[1]
    return (
        -np.sum(value * np.log(np.maximum(value, TINY)), axis=1, dtype=np.float64)
        / np.log(width)
    ).astype(np.float32)


def write_details_and_summary(
    spectra: dict[str, np.ndarray],
    order: np.ndarray,
    selected: list[tuple[str, int]],
    test_pos: np.ndarray,
    regimes: np.ndarray,
    blocked: np.ndarray,
) -> tuple[Path, Path, dict[str, object]]:
    metrics = np.load(V429_METRICS)
    representative = {point: label for label, point in selected}
    frame = pd.DataFrame(
        {
            "谱图排序行": np.arange(1, len(order) + 1),
            "测试点编号": [f"T{point:03d}" for point in order],
            "数组索引": order,
            "x坐标_米": test_pos[order, 0],
            "y坐标_米": test_pos[order, 1],
            "z坐标_米": test_pos[order, 2],
            "传播难度": [REGIME_CN[str(regimes[point])] for point in order],
            "遮挡采样数": blocked[order],
            "代表点用途": [representative.get(int(point), "") for point in order],
            "阶段1_PDP接近度增量": metrics["阶段1_PDP增量"][order],
            "阶段2_PAS接近度增量": metrics["阶段2_leakedPAS增量"][order],
            "阶段3_逐链路PDP接近度增量": metrics["阶段3_逐链路PDP增量"][order],
            "v426剩余PDP误差": metrics["v426剩余PDP误差"][order],
            "v426剩余PAS误差": metrics["v426剩余leakedPAS误差"][order],
        }
    )
    ref_pdp = spectra["reference_pdp"]
    ref_pas = spectra["reference_pas"]
    version_cosines: dict[str, dict[str, float]] = {}
    for name in VERSION_KEYS:
        pdp = spectra[f"{name}_pdp"]
        pas = spectra[f"{name}_pas"]
        pdp_cosine = cosine_rows(pdp, ref_pdp)
        pas_cosine = cosine_rows(pas, ref_pas)
        frame[f"{name}_PDP主峰bin"] = np.argmax(pdp, axis=1)[order]
        frame[f"{name}_PDP平均时延bin"] = np.sum(pdp * np.arange(192)[None, :], axis=1)[order]
        frame[f"{name}_PDP归一化熵"] = normalized_entropy(pdp)[order]
        frame[f"{name}_PDP对参考谱余弦"] = pdp_cosine[order]
        frame[f"{name}_PAS主峰H_bin"] = np.argmax(pas, axis=1)[order]
        frame[f"{name}_PAS归一化熵"] = normalized_entropy(pas)[order]
        frame[f"{name}_PAS对参考谱余弦"] = pas_cosine[order]
        version_cosines[name] = {
            "PDP对参考谱平均余弦": float(np.mean(pdp_cosine)),
            "PAS对参考谱平均余弦": float(np.mean(pas_cosine)),
        }
    stage_motion: dict[str, dict[str, float]] = {}
    for stage_title, before, after in STAGES:
        pdp_delta = amplitude_delta(spectra[f"{before}_pdp"], spectra[f"{after}_pdp"])
        pas_delta = amplitude_delta(spectra[f"{before}_pas"], spectra[f"{after}_pas"])
        stage_key = stage_title.split("\n", maxsplit=1)[0]
        frame[f"{stage_key}_PDP_sqrt功率搬运量"] = np.sum(np.abs(pdp_delta), axis=1)[order]
        frame[f"{stage_key}_PAS_sqrt功率搬运量"] = np.sum(np.abs(pas_delta), axis=1)[order]
        stage_motion[stage_key] = {
            "PDP平均sqrt功率搬运量": float(np.mean(np.sum(np.abs(pdp_delta), axis=1))),
            "PAS平均sqrt功率搬运量": float(np.mean(np.sum(np.abs(pas_delta), axis=1))),
        }
    frame.to_csv(CSV_DETAILS, index=False, encoding="utf-8-sig")

    summary = {
        "分析版本": "v430",
        "测试点数量": int(len(test_pos)),
        "谱定义": {
            "PDP": "对每个测试点做192点IFFT；每条256×4链路先按自身总能量归一化，再平均所有有效链路，得到192-bin谱。",
            "PAS": "按官方traceback暴露的reshape/fft2口径得到B×2×192×16×4×8功率，再对polar、S、UE、V求和，得到16个H角度bin。该口径经v424线上验证，不是传统PAS。",
            "热图颜色": "绝对谱按每个测试点自身峰值转为dB并截断到-40dB；阶段差分使用sqrt归一化功率之差。",
        },
        "排序规则": "先按无遮挡、轻遮挡、中遮挡、重遮挡分组，再按y坐标、x坐标升序。",
        "各版本对同坐标诊断参考的平均谱余弦": version_cosines,
        "三个阶段的平均谱搬运量": stage_motion,
        "代表性测试点": [
            {
                "用途": label,
                "测试点编号": f"T{point:03d}",
                "数组索引": int(point),
                "传播难度": REGIME_CN[str(regimes[point])],
                "坐标_米": [float(value) for value in test_pos[point]],
            }
            for label, point in selected
        ],
        "证据边界警告": "v207是官方测试坐标上的Sionna诊断参考，不是官方隐藏测试真值；这些图解释v209到v426注入了什么，不能当作真实逐点线上误差图。",
    }
    JSON_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return CSV_DETAILS, JSON_SUMMARY, summary


def main() -> None:
    configure_chinese_font()
    validate_inputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    test_pos = np.load(TEST_POS).astype(np.float64)
    geometry = np.load(GEOMETRY)
    blocked = geometry["test_blocked"].astype(np.float32)
    regimes = regime_of(blocked).astype("U8")
    order, groups = build_sort_order(test_pos, regimes)
    spectra = load_or_compute_spectra()
    selected = select_representative_points(spectra)

    outputs: list[Path] = [CACHE]
    outputs.extend(plot_evolution(spectra, order, groups, "pdp", PDP_EVOLUTION))
    outputs.extend(plot_transport(spectra, order, groups, "pdp", PDP_TRANSPORT))
    outputs.extend(plot_evolution(spectra, order, groups, "pas", PAS_EVOLUTION))
    outputs.extend(plot_transport(spectra, order, groups, "pas", PAS_TRANSPORT))
    outputs.extend(plot_representative_curves(spectra, selected, test_pos, regimes, "pdp", PDP_CURVES))
    outputs.extend(plot_representative_curves(spectra, selected, test_pos, regimes, "pas", PAS_CURVES))
    outputs.append(write_interactive(spectra, order, groups, test_pos, regimes))
    csv_path, summary_path, summary = write_details_and_summary(
        spectra, order, selected, test_pos, regimes, blocked
    )
    outputs.extend([csv_path, summary_path])

    report = {
        "方法": VERSION,
        "状态": "分析完成_不生成候选",
        "主要问题": "把全部500个测试点的PDP和线上已验证PAS口径在v209、v212、v424、v426及Sionna诊断参考之间的谱变化画清楚。",
        "输入": {name: PATHS.relative(path) for name, path in ALL_CHANNELS.items()},
        "摘要": summary,
        "输出": [PATHS.relative(path) for path in outputs],
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
