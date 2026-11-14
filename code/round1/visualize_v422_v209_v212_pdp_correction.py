#!/usr/bin/env python3
"""Create a Chinese visual explanation of the v209-to-v212 PDP correction chain."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

from project_paths import PATHS


VERSION_NUMBER = 422
VERSION = "v422_v209_v212_pdp_visual_explanation"
FIGURE_DIR = PATHS.analysis / "visualizations" / "v209到v212的PDP修正可视化"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
POINT_METRICS = PATHS.version_report_dir(347) / "v347_point_metrics.npz"

REFERENCE = PATHS.cache_dir(207) / "v207_sionna_test_reference_H.npy"
CHANNELS = {
    "v209": PATHS.cache_dir(209) / "v209_Round1_Test_Channel.npy",
    "v210": PATHS.cache_dir(210) / "v210_Round1_Test_Channel.npy",
    "v211": PATHS.cache_dir(211) / "v211_Round1_Test_Channel.npy",
    "v212": PATHS.cache_dir(212) / "v212_Round1_Test_Channel.npy",
}
ONLINE = {
    "v209": 0.511934,
    "v210": 0.534149,
    "v211": 0.551384,
    "v212": 0.559160,
}
TRANSITIONS = (("v209", "v210"), ("v210", "v211"), ("v211", "v212"))
REGIMES = ("clear", "light", "medium", "heavy")
REGIME_CN = {"clear": "清晰", "light": "轻度", "medium": "中度", "heavy": "重度"}


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "Noto Sans CJK JP",
            "axes.unicode_minus": False,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "figure.dpi": 130,
            "savefig.dpi": 220,
        }
    )


def normalized_pdp(channel: np.ndarray) -> np.ndarray:
    cir = np.fft.ifft(np.asarray(channel, dtype=np.complex64), axis=-1)
    power = np.sum(np.abs(cir) ** 2, axis=(1, 2), dtype=np.float64)
    return np.divide(
        power,
        np.maximum(np.sum(power, axis=1, keepdims=True), 1e-30),
        out=np.zeros_like(power),
        where=np.sum(power, axis=1, keepdims=True) > 0.0,
    ).astype(np.float32)


def link_delay_power(channel: np.ndarray) -> np.ndarray:
    cir = np.fft.ifft(np.asarray(channel, dtype=np.complex64), axis=-1)
    return (np.abs(cir).astype(np.float32) ** 2).astype(np.float32)


def normalize_last(value: np.ndarray) -> np.ndarray:
    total = np.sum(value, axis=-1, keepdims=True, dtype=np.float64)
    return np.divide(
        value,
        np.maximum(total, 1e-30),
        out=np.zeros_like(value, dtype=np.float32),
        where=total > 0.0,
    ).astype(np.float32)


def cosine_last(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=-1, dtype=np.float64)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    return np.divide(
        numerator,
        np.maximum(denominator, 1e-30),
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    ).astype(np.float32)


def relative_db(value: np.ndarray) -> np.ndarray:
    peak = max(float(np.max(value)), 1e-30)
    return np.maximum(10.0 * np.log10(np.maximum(value / peak, 1e-8)), -50.0)


def select_representative_points(metrics: np.lib.npyio.NpzFile) -> dict[str, int]:
    heavy = metrics["regimes"].astype(str) == "heavy"
    selected: dict[str, int] = {}
    used: set[int] = set()
    for before, after in TRANSITIONS:
        key = f"{before}_to_{after}_delta_pdp"
        score = metrics[key].astype(np.float64).copy()
        score[~heavy] = -np.inf
        for index in np.argsort(score)[::-1]:
            point_id = int(index)
            if point_id not in used and np.isfinite(score[point_id]):
                selected[f"{before}→{after}"] = point_id
                used.add(point_id)
                break
    return selected


def plot_pipeline() -> Path:
    path = FIGURE_DIR / "01_从复信道骨架到PDP修正的实现流程.png"
    fig, ax = plt.subplots(figsize=(16, 8.2))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    boxes = [
        (0.4, 5.1, 2.4, 2.0, "v209 复信道骨架\n\n保留已有相位、功率\n极化和阵列纹理", "#d9e8f5"),
        (3.4, 5.1, 2.4, 2.0, "IFFT 到时延域\n\n频率轴 → 到达时延\n得到复数 CIR", "#e8e8e8"),
        (6.4, 5.1, 2.5, 2.0, "提取归一化谱形\n\nD：总 PDP\nU-D：端口 PDP\nH-D：水平阵列 PDP", "#f8edcf"),
        (9.5, 5.1, 2.7, 2.0, "构造目标谱形\n\n官方训练插值锚点\n+ Sionna 时延形状\n+ 置信度控制", "#dfead7"),
        (12.9, 5.1, 2.7, 2.0, "IPF 幅度投影\n\n反复调整各格能量\n复相位不直接回归\n总功率重新归一", "#f4d8d5"),
    ]
    for x, y, width, height, text, color in boxes:
        ax.add_patch(Rectangle((x, y), width, height, facecolor=color, edgecolor="#343434", linewidth=1.4))
        ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", linespacing=1.45)
    for left, right in zip(boxes[:-1], boxes[1:]):
        ax.add_patch(
            FancyArrowPatch(
                (left[0] + left[2] + 0.08, left[1] + left[3] / 2),
                (right[0] - 0.08, right[1] + right[3] / 2),
                arrowstyle="-|>", mutation_scale=18, linewidth=1.6, color="#444444",
            )
        )

    stage_text = [
        (1.0, 1.0, 3.8, 2.2, "v210：首次引入高预算 Sionna PDP\n把缺失或错位的时延峰拉向物理参考\n线上 +0.022215", "#f5b7a8"),
        (6.1, 1.0, 3.8, 2.2, "v211：补完 v210 尚未修够的差距\n只继续修可信的中度/重度点\n线上 +0.017235", "#f7d794"),
        (11.2, 1.0, 3.8, 2.2, "v212：用 2.6/6.0/6.7 GHz 一致性筛选\n多频都支持的路径才继续增强\n线上 +0.007776", "#afd7bf"),
    ]
    for x, y, width, height, text, color in stage_text:
        ax.add_patch(Rectangle((x, y), width, height, facecolor=color, edgecolor="#444444", linewidth=1.2))
        ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", linespacing=1.5)
    ax.add_patch(FancyArrowPatch((4.85, 2.1), (6.0, 2.1), arrowstyle="-|>", mutation_scale=18, color="#555555"))
    ax.add_patch(FancyArrowPatch((9.95, 2.1), (11.1, 2.1), arrowstyle="-|>", mutation_scale=18, color="#555555"))
    ax.text(
        8,
        8.35,
        "核心动作：只借用 Sionna 的‘能量落在哪些时延位置’，不复制其原始相位和总功率",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
    )
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_score_and_metric(metrics: np.lib.npyio.NpzFile) -> Path:
    path = FIGURE_DIR / "02_线上分数与PDP接近程度同步上升.png"
    names = list(ONLINE)
    online = np.asarray([ONLINE[name] for name in names])
    pdp = np.asarray([float(np.mean(metrics[f"{name}_pdp"])) for name in names])
    pas = np.asarray([float(np.mean(metrics[f"{name}_leaked_sh"])) for name in names])
    nmse = np.asarray([float(np.mean(metrics[f"{name}_nmse_score"])) for name in names])

    fig, axes = plt.subplots(2, 1, figsize=(12.8, 10), constrained_layout=True)
    axes[0].plot(names, online, marker="o", linewidth=2.8, color="#b83b2d")
    axes[0].fill_between(names, online, online.min() - 0.003, color="#b83b2d", alpha=0.08)
    for name, value in zip(names, online):
        axes[0].annotate(f"{value:.6f}", (name, value), xytext=(0, 9), textcoords="offset points", ha="center")
    axes[0].set_title("真实线上结果：三次 PDP 修正连续涨分")
    axes[0].set_ylabel("线上分数")
    axes[0].grid(alpha=0.25)

    axes[1].plot(names, pdp, marker="o", linewidth=2.6, label="PDP 接近度", color="#087f5b")
    axes[1].plot(names, pas, marker="s", linewidth=2.2, label="PAS 接近度", color="#7048a8")
    axes[1].plot(names, nmse, marker="^", linewidth=2.2, label="复信道 NMSE 分量", color="#d97706")
    axes[1].set_title("诊断参考：主要变化发生在 PDP，而不是复信道 NMSE")
    axes[1].set_ylabel("与 v207 Sionna reference 的局部指标")
    axes[1].set_xlabel("版本")
    axes[1].grid(alpha=0.25)
    axes[1].legend(ncol=3, loc="upper left")
    axes[1].text(
        0.01,
        0.03,
        "注意：v207 只是解释性参考，不是官方隐藏测试真值。",
        transform=axes[1].transAxes,
        color="#8b1e1e",
    )
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_regime_gains(metrics: np.lib.npyio.NpzFile) -> Path:
    path = FIGURE_DIR / "03_分传播难度观察PDP改善集中在哪里.png"
    matrix = np.zeros((len(TRANSITIONS), len(REGIMES)), dtype=np.float64)
    regimes = metrics["regimes"].astype(str)
    for row, (before, after) in enumerate(TRANSITIONS):
        delta = metrics[f"{before}_to_{after}_delta_pdp"].astype(np.float64)
        for col, regime in enumerate(REGIMES):
            matrix[row, col] = float(np.mean(delta[regimes == regime]))

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    limit = max(float(np.max(np.abs(matrix))), 0.01)
    image = ax.imshow(matrix, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(len(REGIMES)), [REGIME_CN[value] for value in REGIMES])
    ax.set_yticks(np.arange(len(TRANSITIONS)), [f"{a}→{b}" for a, b in TRANSITIONS])
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, f"{matrix[row, col]:+.4f}", ha="center", va="center", fontweight="bold")
    ax.set_title("PDP 改善几乎全部集中在中度和重度传播点")
    ax.set_xlabel("传播难度分区")
    ax.set_ylabel("版本跃迁")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    colorbar.set_label("PDP 接近度增量")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def load_point_pdps(
    selected: dict[str, int],
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    point_ids = list(selected.values())
    power: dict[str, np.ndarray] = {}
    for name, path in {"Sionna参考": REFERENCE, **CHANNELS}.items():
        channel = np.load(path, mmap_mode="r")
        power[name] = link_delay_power(np.asarray(channel[point_ids], dtype=np.complex64))

    output = {name: np.empty((len(point_ids), 192), dtype=np.float32) for name in power}
    details: list[dict[str, object]] = []
    reference = power["Sionna参考"]
    for row, transition in enumerate(selected):
        before, after = transition.split("→")
        before_cosine = cosine_last(power[before][row], reference[row])
        after_cosine = cosine_last(power[after][row], reference[row])
        gain = after_cosine - before_cosine
        valid = np.sum(reference[row], axis=-1) > 1e-20
        gain[~valid] = -np.inf
        flat_id = int(np.argmax(gain.reshape(-1)))
        antenna, port = np.unravel_index(flat_id, gain.shape)
        link_cosines: dict[str, float] = {}
        for name, values in power.items():
            curve = values[row, antenna, port]
            output[name][row] = normalize_last(curve[None, :])[0]
            if name != "Sionna参考":
                link_cosines[name] = float(
                    cosine_last(curve[None, :], reference[row, antenna, port][None, :])[0]
                )
        details.append(
            {
                "transition": transition,
                "antenna": int(antenna),
                "ue_port": int(port),
                "link_gain": float(gain[antenna, port]),
                "link_cosines": link_cosines,
            }
        )
    return output, details


def plot_point_evolution(
    metrics: np.lib.npyio.NpzFile,
    selected: dict[str, int],
    pdps: dict[str, np.ndarray],
    link_details: list[dict[str, object]],
) -> Path:
    path = FIGURE_DIR / "04_三个真实重度点的PDP峰如何逐步修正.png"
    colors = {"v209": "#737373", "v210": "#c53b32", "v211": "#e28e1b", "v212": "#16845b"}
    fig, axes = plt.subplots(len(selected), 1, figsize=(15, 12), sharex=True, constrained_layout=True)
    positions = metrics["positions"]
    for row, ((transition, point_id), ax) in enumerate(zip(selected.items(), np.atleast_1d(axes))):
        reference = pdps["Sionna参考"][row]
        detail = link_details[row]
        ax.plot(relative_db(reference), color="#111111", linewidth=2.7, linestyle="--", label="Sionna 时延形状参考")
        for name in CHANNELS:
            alpha = 0.72 if name != "v212" else 1.0
            width = 1.4 if name != "v212" else 2.2
            ax.plot(relative_db(pdps[name][row]), color=colors[name], linewidth=width, alpha=alpha, label=name)
        cosines = detail["link_cosines"]
        coordinate = positions[point_id]
        ax.set_title(
            f"点 {point_id}，坐标 ({coordinate[0]:.1f}, {coordinate[1]:.1f})，BS 天线 {detail['antenna']} / UE 端口 {detail['ue_port']}\n"
            f"该链路 PDP 接近度：v209 {cosines['v209']:.3f} → v210 {cosines['v210']:.3f} → "
            f"v211 {cosines['v211']:.3f} → v212 {cosines['v212']:.3f}"
        )
        ax.set_ylabel("相对功率 / dB")
        ax.set_ylim(-50, 3)
        ax.grid(alpha=0.22)
    axes[-1].set_xlabel("时延 bin（0～191）")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.01))
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def transition_diagnostics() -> dict[str, dict[str, np.ndarray]]:
    output: dict[str, dict[str, np.ndarray]] = {}
    for before, after in TRANSITIONS:
        h0 = np.load(CHANNELS[before], mmap_mode="r")
        h1 = np.load(CHANNELS[after], mmap_mode="r")
        power_ratio: list[np.ndarray] = []
        phase_agreement: list[np.ndarray] = []
        pdp_l1: list[np.ndarray] = []
        for start in range(0, len(h0), 5):
            end = min(start + 5, len(h0))
            block0 = np.asarray(h0[start:end], dtype=np.complex64)
            block1 = np.asarray(h1[start:end], dtype=np.complex64)
            cir0 = np.fft.ifft(block0, axis=-1)
            cir1 = np.fft.ifft(block1, axis=-1)
            p0 = np.sum(np.abs(cir0) ** 2, axis=(1, 2, 3), dtype=np.float64)
            p1 = np.sum(np.abs(cir1) ** 2, axis=(1, 2, 3), dtype=np.float64)
            power_ratio.append((p1 / np.maximum(p0, 1e-30)).astype(np.float32))
            weight = np.abs(cir0) * np.abs(cir1)
            phase = np.real(cir1 * np.conj(cir0)) / np.maximum(weight, 1e-30)
            phase_agreement.append(
                (np.sum(weight * phase, axis=(1, 2, 3)) / np.maximum(np.sum(weight, axis=(1, 2, 3)), 1e-30)).astype(np.float32)
            )
            d0 = normalized_pdp(block0)
            d1 = normalized_pdp(block1)
            pdp_l1.append(np.sum(np.abs(d1 - d0), axis=1).astype(np.float32))
        output[f"{before}→{after}"] = {
            "power_ratio": np.concatenate(power_ratio),
            "phase_agreement": np.concatenate(phase_agreement),
            "pdp_l1": np.concatenate(pdp_l1),
        }
    return output


def plot_preservation(diagnostics: dict[str, dict[str, np.ndarray]]) -> Path:
    path = FIGURE_DIR / "05_真正被改动的是PDP而不是总功率和相位.png"
    names = list(diagnostics)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)

    axes[0].bar(names, [float(np.mean(diagnostics[name]["pdp_l1"])) for name in names], color=["#c53b32", "#e28e1b", "#16845b"])
    axes[0].set_title("归一化 PDP 的实际改动")
    axes[0].set_ylabel("平均 L1 距离")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].grid(axis="y", alpha=0.22)

    power_mean = np.asarray([float(np.mean(diagnostics[name]["power_ratio"])) for name in names])
    power_median = np.asarray([float(np.median(diagnostics[name]["power_ratio"])) for name in names])
    power_p05 = np.asarray([float(np.quantile(diagnostics[name]["power_ratio"], 0.05)) for name in names])
    power_p95 = np.asarray([float(np.quantile(diagnostics[name]["power_ratio"], 0.95)) for name in names])
    x = np.arange(len(names))
    axes[1].errorbar(
        x,
        power_mean,
        yerr=np.vstack([power_mean - power_p05, power_p95 - power_mean]),
        fmt="o",
        color="#b83b2d",
        capsize=6,
        linewidth=2,
        label="均值及 5%～95%",
    )
    axes[1].scatter(x, power_median, marker="x", s=75, color="#222222", label="中位数")
    axes[1].axhline(1.0, color="#222222", linestyle="--", linewidth=1.3)
    axes[1].set_title("逐点总功率比值")
    axes[1].set_ylabel("后版本功率 / 前版本功率")
    axes[1].set_xticks(x, names)
    axes[1].set_ylim(min(0.95, float(np.min(power_p05)) - 0.003), max(1.01, float(np.max(power_p95)) + 0.003))
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].legend(loc="lower right")

    phase_mean = [float(np.mean(diagnostics[name]["phase_agreement"])) for name in names]
    axes[2].bar(names, phase_mean, color=["#4477aa", "#66a1c9", "#7aa974"])
    axes[2].set_ylim(min(0.995, min(phase_mean) - 0.001), 1.0002)
    axes[2].set_title("时延域复相位加权一致性")
    axes[2].set_ylabel("1 表示相位方向完全保留")
    axes[2].tick_params(axis="x", rotation=15)
    axes[2].grid(axis="y", alpha=0.22)
    for index, value in enumerate(phase_mean):
        axes[2].text(index, value, f"{value:.6f}", ha="center", va="bottom")

    fig.suptitle("修正器移动的是能量分布；v211、v212 基本严格保持总功率和相位", fontsize=16, fontweight="bold")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    required = [POINT_METRICS, REFERENCE, *CHANNELS.values()]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    configure_plotting()

    metrics = np.load(POINT_METRICS)
    selected = select_representative_points(metrics)
    point_ids = list(selected.values())
    point_pdps, link_details = load_point_pdps(selected)
    diagnostics = transition_diagnostics()

    figures = [
        plot_pipeline(),
        plot_score_and_metric(metrics),
        plot_regime_gains(metrics),
        plot_point_evolution(metrics, selected, point_pdps, link_details),
        plot_preservation(diagnostics),
    ]

    metric_summary = {
        name: {
            "online": ONLINE[name],
            "mean_pdp_to_v207": float(np.mean(metrics[f"{name}_pdp"])),
            "mean_leaked_pas_to_v207": float(np.mean(metrics[f"{name}_leaked_sh"])),
            "mean_nmse_score_to_v207": float(np.mean(metrics[f"{name}_nmse_score"])),
        }
        for name in CHANNELS
    }
    transition_summary = {
        name: {
            "mean_pdp_l1_change": float(np.mean(values["pdp_l1"])),
            "power_ratio_mean": float(np.mean(values["power_ratio"])),
            "power_ratio_median": float(np.median(values["power_ratio"])),
            "phase_agreement_mean": float(np.mean(values["phase_agreement"])),
        }
        for name, values in diagnostics.items()
    }
    report = {
        "method": VERSION,
        "status": "analysis_only_visual_explanation",
        "purpose": "Explain with actual outputs how v209's complex channel was changed through v210-v212 by normalized PDP projection.",
        "evidence_boundary": {
            "online_scores": "actual leaderboard evidence",
            "v207_reference_metrics": "diagnostic explanation only; not official hidden-test truth",
            "representative_points": "selected among heavy points by stage-specific PDP gain to the v207 diagnostic reference",
        },
        "selected_points": {
            name: {
                "index": int(point_id),
                "position": metrics["positions"][point_id].astype(float).tolist(),
                "regime": str(metrics["regimes"][point_id]),
                "selected_link": link_details[row],
            }
            for row, (name, point_id) in enumerate(selected.items())
        },
        "version_metrics": metric_summary,
        "transition_diagnostics": transition_summary,
        "figures": [PATHS.relative(path) for path in figures],
        "report": PATHS.relative(REPORT),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
