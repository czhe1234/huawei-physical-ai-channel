#!/usr/bin/env python3
"""Plot canonical Round 2 local and online scores for V1-V11."""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from project_paths import PATHS


INPUT = (
    PATHS.analysis
    / "results"
    / "round2_canonical_500_version_replay"
    / "canonical_500_version_scores.json"
)
OUTPUT_DIR = PATHS.analysis / "visualizations" / "复赛固定500点评价"
OUTPUT_PNG = OUTPUT_DIR / "各版本固定500点本地分数与线上分数曲线.png"
OUTPUT_PDF = OUTPUT_DIR / "各版本固定500点本地分数与线上分数曲线.pdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v5-online-approx",
        type=float,
        default=0.66,
        help="Approximate V5 online score supplied by the user.",
    )
    return parser.parse_args()


def configure_chinese_font() -> str:
    candidates = (
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "WenQuanYi Micro Hei",
        "SimHei",
        "Microsoft YaHei",
    )
    for name in candidates:
        try:
            font_manager.findfont(name, fallback_to_default=False)
        except ValueError:
            continue
        plt.rcParams["font.family"] = name
        plt.rcParams["axes.unicode_minus"] = False
        return name
    raise RuntimeError("未找到可用的中文字体")


def annotate_values(
    axis: plt.Axes,
    x: np.ndarray,
    values: np.ndarray,
    color: str,
    offset: float | np.ndarray,
    skip: set[int] | None = None,
) -> None:
    excluded = skip or set()
    for index, value in enumerate(values):
        if index in excluded or not np.isfinite(value):
            continue
        local_offset = float(offset[index]) if isinstance(offset, np.ndarray) else offset
        axis.annotate(
            f"{value:.3f}",
            (x[index], value),
            xytext=(0, local_offset),
            textcoords="offset points",
            ha="center",
            va="bottom" if local_offset >= 0 else "top",
            fontsize=9,
            color=color,
            fontweight="medium",
        )


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.v5_online_approx <= 1.0:
        raise ValueError("V5线上约值必须位于[0, 1]")
    configure_chinese_font()
    payload = json.loads(INPUT.read_text(encoding="utf-8"))
    rows = payload["versions"]
    versions = [str(row["version"]) for row in rows]
    if versions != [f"V{index}" for index in range(1, 12)]:
        raise ValueError(f"版本顺序异常：{versions}")

    x = np.arange(len(rows), dtype=np.float64)
    local = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
    online = np.asarray(
        [
            np.nan if row["online_score"] is None else float(row["online_score"])
            for row in rows
        ],
        dtype=np.float64,
    )
    v5_index = versions.index("V5")
    online_with_approx = online.copy()
    online_with_approx[v5_index] = args.v5_online_approx

    local_color = "#2468A2"
    online_color = "#C43E38"
    figure, axis = plt.subplots(figsize=(13.5, 7.6), constrained_layout=False)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("#FAFBFC")

    axis.plot(
        x,
        local,
        color=local_color,
        linewidth=2.6,
        marker="o",
        markersize=7.5,
        markeredgecolor="white",
        markeredgewidth=1.2,
        label="固定500点本地总分",
        zorder=4,
    )
    axis.plot(
        x[:v5_index],
        online[:v5_index],
        color=online_color,
        linewidth=2.6,
        marker="s",
        markersize=7,
        markeredgecolor="white",
        markeredgewidth=1.2,
        label="线上分数",
        zorder=4,
    )
    axis.plot(
        x[v5_index + 1 :],
        online[v5_index + 1 :],
        color=online_color,
        linewidth=2.6,
        marker="s",
        markersize=7,
        markeredgecolor="white",
        markeredgewidth=1.2,
        zorder=4,
    )
    axis.plot(
        x[v5_index - 1 : v5_index + 2],
        online_with_approx[v5_index - 1 : v5_index + 2],
        color=online_color,
        linewidth=1.8,
        linestyle="--",
        alpha=0.72,
        zorder=3,
    )
    axis.scatter(
        [x[v5_index]],
        [args.v5_online_approx],
        s=80,
        marker="s",
        facecolors="white",
        edgecolors=online_color,
        linewidths=2.0,
        label="V5线上约值",
        zorder=5,
    )

    local_offsets = np.where(local >= online_with_approx, 11.0, -18.0)
    online_offsets = np.where(online_with_approx >= local, 11.0, -20.0)
    annotate_values(axis, x, local, local_color, local_offsets)
    annotate_values(axis, x, online, online_color, online_offsets)
    axis.annotate(
        f"约{args.v5_online_approx:.3f}",
        (x[v5_index], args.v5_online_approx),
        xytext=(0, float(online_offsets[v5_index])),
        textcoords="offset points",
        ha="center",
        va="bottom" if online_offsets[v5_index] >= 0 else "top",
        fontsize=9,
        color=online_color,
        fontweight="medium",
    )

    lower = min(float(local.min()), float(np.nanmin(online_with_approx))) - 0.012
    upper = max(float(local.max()), float(np.nanmax(online_with_approx))) + 0.014
    axis.set_ylim(lower, upper)
    axis.set_xlim(-0.45, len(rows) - 0.55)
    axis.set_xticks(x, versions)
    axis.set_ylabel("官方总分", fontsize=12)
    axis.set_xlabel("复赛版本（按版本编号排列）", fontsize=12, labelpad=10)
    axis.grid(axis="y", color="#D7DCE2", linewidth=0.9, alpha=0.8)
    axis.grid(axis="x", visible=False)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#AEB6BF")
    axis.tick_params(axis="both", labelsize=10.5, colors="#27313B")

    figure.suptitle(
        "复赛各版本本地分数与线上分数",
        fontsize=19,
        fontweight="bold",
        y=0.965,
    )
    axis.set_title(
        "本地统一使用固定500个空间洞OOF点（包含31个零信道点）",
        fontsize=11.5,
        color="#53606C",
        pad=17,
    )
    axis.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#D7DCE2",
        fontsize=10.5,
        ncol=3,
    )
    figure.text(
        0.5,
        0.025,
        "说明：V5线上分数为用户提供的约值；V5与V6均从V4分叉，横轴连线只表示版本编号顺序。",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#5E6872",
    )
    figure.tight_layout(rect=(0.03, 0.065, 0.99, 0.91))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PNG, dpi=240, bbox_inches="tight", facecolor="white")
    figure.savefig(OUTPUT_PDF, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(PATHS.relative(OUTPUT_PNG))
    print(PATHS.relative(OUTPUT_PDF))


if __name__ == "__main__":
    main()
