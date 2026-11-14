#!/usr/bin/env python3
"""Build static visual diagnostics for Round1 submissions."""

from __future__ import annotations
from project_paths import PATHS

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_map_features import BS_POS
from evaluate_local_ad_paths import ad_transform


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def maybe_extract_npy(zip_path: Path, out_path: Path) -> Path | None:
    if not zip_path.exists():
        return None
    if out_path.exists():
        return out_path
    ensure_dir(out_path.parent)
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("Round1_Test_Channel.npy") as src, out_path.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
    return out_path


def point_power(arr) -> np.ndarray:
    out = np.empty(arr.shape[0], dtype=np.float64)
    for i in range(arr.shape[0]):
        out[i] = float(np.sum(np.abs(np.asarray(arr[i])) ** 2, dtype=np.float64))
    return out


def point_diff_power(a, b) -> np.ndarray:
    out = np.empty(a.shape[0], dtype=np.float64)
    for i in range(a.shape[0]):
        da = np.asarray(a[i]) - np.asarray(b[i])
        out[i] = float(np.sum(np.abs(da) ** 2, dtype=np.float64))
    return out


def scatter_map(ax, train_pos, test_pos, values, title, cmap="viridis", s=28, vmin=None, vmax=None):
    ax.scatter(train_pos[:, 0], train_pos[:, 1], s=4, c="#d0d0d0", alpha=0.25, linewidths=0)
    sc = ax.scatter(test_pos[:, 0], test_pos[:, 1], s=s, c=values, cmap=cmap, vmin=vmin, vmax=vmax, edgecolors="k", linewidths=0.15)
    ax.scatter([BS_POS[0]], [BS_POS[1]], marker="*", s=180, c="red", edgecolors="black", linewidths=0.5)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)


def save_spatial_overview(out_dir: Path, train_pos, test_pos, train_blocked, test_blocked, powers: dict[str, np.ndarray], ratios: dict[str, np.ndarray]) -> dict:
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    log_v68 = np.log10(np.maximum(powers["v68"], 1e-30))
    scatter_map(axes[0, 0], train_pos, test_pos, log_v68, "v68 log10 total power")
    scatter_map(axes[0, 1], train_pos, test_pos, np.log10(np.maximum(ratios["v68_vs_v36"], 1e-12)), "log10 ||v68-v36||^2 / ||v36||^2", cmap="magma")
    scatter_map(axes[0, 2], train_pos, test_pos, test_blocked, "test occlusion count", cmap="plasma")
    delta = (powers["v68"] - powers["v36"]) / np.maximum(powers["v36"], 1e-30)
    lim = np.percentile(np.abs(delta), 98)
    scatter_map(axes[1, 0], train_pos, test_pos, delta, "(power v68 - v36) / v36", cmap="coolwarm", vmin=-lim, vmax=lim)
    if "v55" in powers:
        scatter_map(axes[1, 1], train_pos, test_pos, np.log10(np.maximum(ratios["v55_vs_v36"], 1e-12)), "log10 ||v55-v36||^2 / ||v36||^2", cmap="magma")
        scatter_map(axes[1, 2], train_pos, test_pos, np.log10(np.maximum(ratios["v68_vs_v55"], 1e-12)), "log10 ||v68-v55||^2 / ||v55||^2", cmap="magma")
    else:
        axes[1, 1].axis("off")
        axes[1, 2].axis("off")
    path = out_dir / "spatial_overview.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    bins = np.arange(5)
    labels = ["clear", "light", "medium", "heavy"]
    regimes = np.digitize(test_blocked, [1, 26, 101])
    vals = []
    for r in range(4):
        m = regimes == r
        vals.append(float(np.mean(np.log10(np.maximum(ratios["v68_vs_v36"][m], 1e-12)))) if np.any(m) else np.nan)
    ax.bar(np.arange(4), vals, color=["#4daf4a", "#377eb8", "#ff7f00", "#e41a1c"])
    ax.set_xticks(np.arange(4), labels)
    ax.set_ylabel("mean log10 residual ratio")
    ax.set_title("v68-v36 residual strength by occlusion regime")
    path2 = out_dir / "residual_by_regime.png"
    fig.savefig(path2, dpi=160)
    plt.close(fig)
    return {"spatial_overview": path.name, "residual_by_regime": path2.name}


def ad_power_maps(h):
    ad = ad_transform(np.asarray(h, dtype=np.complex64)[None])[0]
    p = np.sum(np.abs(ad) ** 2, axis=(0, 3), dtype=np.float64)  # MH, MV, S
    h_delay = np.sum(p, axis=1)  # MH, S
    pdp = np.sum(p, axis=(0, 1))
    hv = np.sum(p, axis=2)
    return p, h_delay, pdp, hv


def plot_ad_point(out_dir: Path, idx: int, arrays: dict[str, object]) -> str:
    names = [n for n in ["v36", "v68", "v55"] if n in arrays]
    maps = {n: ad_power_maps(np.asarray(arrays[n][idx])) for n in names}
    residual = np.asarray(arrays["v68"][idx]) - np.asarray(arrays["v36"][idx])
    _, h_delay_res, pdp_res, hv_res = ad_power_maps(residual)

    ncols = 3
    fig, axes = plt.subplots(3, ncols, figsize=(16, 11), constrained_layout=True)

    ax = axes[0, 0]
    for n in names:
        ax.plot(10 * np.log10(np.maximum(maps[n][2], 1e-30)), label=n)
    ax.plot(10 * np.log10(np.maximum(pdp_res, 1e-30)), label="v68-v36", linestyle="--")
    ax.set_title(f"Point {idx} PDP")
    ax.set_xlabel("delay bin")
    ax.set_ylabel("dB")
    ax.legend()

    for col, n in enumerate(names[:2]):
        ax = axes[0, col + 1]
        im = ax.imshow(10 * np.log10(np.maximum(maps[n][1], 1e-30)), aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(f"{n} h-delay power")
        ax.set_xlabel("delay")
        ax.set_ylabel("h bin")
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    if len(names) < 2:
        axes[0, 2].axis("off")

    im = axes[1, 0].imshow(10 * np.log10(np.maximum(h_delay_res, 1e-30)), aspect="auto", origin="lower", cmap="magma")
    axes[1, 0].set_title("v68-v36 residual h-delay power")
    axes[1, 0].set_xlabel("delay")
    axes[1, 0].set_ylabel("h bin")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.035, pad=0.02)

    im = axes[1, 1].imshow(10 * np.log10(np.maximum(maps["v68"][3], 1e-30)), aspect="auto", origin="lower", cmap="viridis")
    axes[1, 1].set_title("v68 h-v integrated power")
    axes[1, 1].set_xlabel("v bin")
    axes[1, 1].set_ylabel("h bin")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.035, pad=0.02)

    im = axes[1, 2].imshow(10 * np.log10(np.maximum(hv_res, 1e-30)), aspect="auto", origin="lower", cmap="magma")
    axes[1, 2].set_title("v68-v36 residual h-v power")
    axes[1, 2].set_xlabel("v bin")
    axes[1, 2].set_ylabel("h bin")
    fig.colorbar(im, ax=axes[1, 2], fraction=0.035, pad=0.02)

    flat = maps["v68"][0].reshape(-1)
    top = np.argpartition(flat, -12)[-12:]
    top = top[np.argsort(flat[top])[::-1]]
    rows = []
    for rank, fid in enumerate(top, 1):
        h = fid // (8 * 192)
        rem = fid % (8 * 192)
        v = rem // 192
        s = rem % 192
        rows.append([rank, int(h), int(v), int(s), float(flat[fid])])
    axes[2, 0].axis("off")
    table = axes[2, 0].table(
        cellText=[[r[0], r[1], r[2], r[3], f"{r[4]:.3e}"] for r in rows],
        colLabels=["rank", "h", "v", "delay", "power"],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.2)
    axes[2, 0].set_title("v68 top AD components")

    if "v55" in maps:
        diff = np.asarray(arrays["v68"][idx]) - np.asarray(arrays["v55"][idx])
        _, h_delay_d, pdp_d, _ = ad_power_maps(diff)
        axes[2, 1].plot(10 * np.log10(np.maximum(pdp_d, 1e-30)))
        axes[2, 1].set_title("PDP of v68-v55")
        axes[2, 1].set_xlabel("delay")
        axes[2, 1].set_ylabel("dB")
        im = axes[2, 2].imshow(10 * np.log10(np.maximum(h_delay_d, 1e-30)), aspect="auto", origin="lower", cmap="magma")
        axes[2, 2].set_title("v68-v55 h-delay power")
        axes[2, 2].set_xlabel("delay")
        axes[2, 2].set_ylabel("h bin")
        fig.colorbar(im, ax=axes[2, 2], fraction=0.035, pad=0.02)
    else:
        axes[2, 1].axis("off")
        axes[2, 2].axis("off")

    path = out_dir / f"point_{idx:03d}_ad.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def plot_paths_for_point(out_dir: Path, idx: int, train_pos, test_pos, paths) -> str:
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    q = test_pos[idx]
    ax.scatter(train_pos[:, 0], train_pos[:, 1], s=4, c="#d0d0d0", alpha=0.12, linewidths=0)
    ax.scatter(test_pos[:, 0], test_pos[:, 1], s=8, c="#aaaaaa", alpha=0.25, linewidths=0)
    ax.scatter([BS_POS[0]], [BS_POS[1]], marker="*", s=180, c="red", edgecolors="black", linewidths=0.5, label="BS")
    ax.scatter([q[0]], [q[1]], marker="o", s=80, c="blue", edgecolors="black", linewidths=0.5, label=f"test {idx}")
    ax.plot([BS_POS[0], q[0]], [BS_POS[1], q[1]], color="black", lw=1.2, alpha=0.6, label="LOS")

    valid = paths["test_valid"][idx]
    score = np.asarray(paths["test_score"][idx])
    ids = np.flatnonzero(valid)
    ids = ids[np.argsort(score[ids])[::-1]]
    colors = ["#e41a1c", "#ff7f00", "#984ea3", "#4daf4a", "#377eb8"]
    for j, p in enumerate(ids[:5]):
        if int(paths["test_path_type"][idx, p]) == 0:
            continue
        rp = paths["test_reflect_point"][idx, p]
        col = colors[j % len(colors)]
        ax.plot([BS_POS[0], rp[0], q[0]], [BS_POS[1], rp[1], q[1]], color=col, lw=1.4, alpha=0.8, label=f"path {p} score {score[p]:.2f}")
        ax.scatter([rp[0]], [rp[1]], s=35, c=col, edgecolors="black", linewidths=0.3)
        plane = int(paths["test_plane_id"][idx, p])
        if plane >= 0:
            tangent = paths["plane_tangent"][plane]
            normal = paths["plane_normal"][plane]
            t0 = max(float(paths["plane_t_min"][plane]), float(np.dot(rp[:2], tangent)) - 25)
            t1 = min(float(paths["plane_t_max"][plane]), float(np.dot(rp[:2], tangent)) + 25)
            c = float(paths["plane_c"][plane])
            p0 = tangent * t0 + normal * c
            p1 = tangent * t1 + normal * c
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=col, lw=3, alpha=0.35)

    ax.set_xlim(q[0] - 90, q[0] + 90)
    ax.set_ylim(q[1] - 90, q[1] + 90)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Geometry path candidates for test point {idx}")
    ax.legend(fontsize=7, loc="best")
    path = out_dir / f"point_{idx:03d}_paths.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def choose_points(test_blocked, power_v68, ratio_v68_v36) -> list[int]:
    candidates = []
    for arr in [
        np.argsort(power_v68)[::-1][:4],
        np.argsort(ratio_v68_v36)[::-1][:5],
        np.flatnonzero(test_blocked > 100)[np.argsort(ratio_v68_v36[test_blocked > 100])[::-1][:5]] if np.any(test_blocked > 100) else [],
        np.flatnonzero(test_blocked == 0)[np.argsort(power_v68[test_blocked == 0])[::-1][:3]] if np.any(test_blocked == 0) else [],
    ]:
        for x in arr:
            i = int(x)
            if i not in candidates:
                candidates.append(i)
    return candidates[:10]


def rel(path: str) -> str:
    return f"figures/{path}"


def write_analysis_md(out_dir: Path, images: dict, point_rows: list[dict], summary: dict) -> None:
    lines = [
        "# Round1 v68 可视化分析",
        "",
        "目的：把 `v36` 基线、`v68` no-s1 路径相对 residual、`v55` 当前 overall best 放在一起比较，分析我们到底改动了哪些位置、哪些传播路径、哪些 angle-delay component。",
        "",
        "## 文件结构",
        "",
        "- `figures/`：所有可视化 PNG 图。",
        "- `data/summary.json`：本次可视化统计数据。",
        "- `analysis.md`：当前分析文档。",
        "",
        "## 版本含义",
        "",
        "| 版本 | 含义 |",
        "| --- | --- |",
        "| `v36` | 几何/局部仿射强基线 |",
        "| `v68` | 当前 no-s1 最好版本，线上 `0.470780` |",
        "| `v55` | 当前 overall best，线上 `0.475165`，含外部 `s1-v9` residual |",
        "",
        "## 核心统计",
        "",
        "```json",
        json.dumps(summary, indent=2, ensure_ascii=False),
        "```",
        "",
        "## 全局空间图",
        "",
        f"![spatial overview]({rel(images['spatial_overview'])})",
        "",
        "这张图包含 6 个子图：",
        "",
        "| 子图 | 含义 | 分析重点 |",
        "| --- | --- | --- |",
        "| `v68 log10 total power` | v68 在每个测试点预测出的总功率 | 强功率区域是否合理集中在 LOS/近距离区域 |",
        "| `log10 ||v68-v36||^2 / ||v36||^2` | v68 相比 v36 的逐点改动强度 | 亮色区域就是 v68 residual 主要作用的位置 |",
        "| `test occlusion count` | BS 到测试点连线附近的点云遮挡计数 | 遮挡越大，越像复杂 NLOS/heavy 区域 |",
        "| `(power v68 - v36) / v36` | v68 是否整体抬高或压低某些点的功率 | 红色是增功率，蓝色是降功率 |",
        "| `log10 ||v55-v36||^2 / ||v36||^2` | v55 相对 v36 的改动 | v55 是当前最高分，可作为有效 residual 的参考 |",
        "| `log10 ||v68-v55||^2 / ||v55||^2` | v68 和 v55 的差异 | 这里是追 v55 或超越 v55 的重点区域 |",
        "",
        "初步结论：",
        "",
        "- v68 的改动不是均匀铺开，而是更集中在遮挡/heavy 或边界区域。",
        "- v55 相比 v36 的改动更激进，特别是左下和右上边缘区域。",
        "- v68 与 v55 的差异仍有明显空间结构，说明 v68 不是完全错方向，而是 residual 覆盖、强度或相位还不够。",
        "",
        "## 按遮挡区域统计 residual",
        "",
        f"![residual by regime]({rel(images['residual_by_regime'])})",
        "",
        "这张图按 `clear/light/medium/heavy` 四类区域统计 `v68-v36` 的平均 residual 强度。",
        "",
        "初步结论：",
        "",
        "- clear 区域 residual 最弱，heavy 区域 residual 最强。",
        "- 这说明 v68 主要在复杂多径区域补偿，不是在所有点上平均乱改。",
        "- 下一步不能大幅动 clear 高功率点，否则容易破坏当前可靠的 LOS/直达径预测。",
        "",
        "## 代表测试点",
        "",
        "| point | blocked | log10 power v68 | v68/v36 residual ratio | AD 图 | 路径图 |",
        "| ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in point_rows:
        lines.append(
            f"| {row['idx']} | {row['blocked']} | {row['log_power']:.3f} | {row['ratio']:.3e} | "
            f"[AD]({rel(row['ad'])}) | [paths]({rel(row['paths'])}) |"
        )

    lines.extend(
        [
            "",
            "### 读图方法",
            "",
            "- `paths` 图：蓝点是测试点，红星是 BS，黑线是 LOS，彩色折线是候选反射路径，粗半透明线段是对应墙面 patch。",
            "- `AD` 图：PDP 看 delay 能量；h-delay 看水平角/时延结构；h-v 看角度结构；`v68-v36 residual` 看新方法到底补了什么。",
            "- 如果 residual 集中在少数清晰条纹或亮点，说明它像真实路径 component；如果大面积散开，更像噪声或过拟合。",
            "",
        ]
    )

    for row in point_rows:
        idx = row["idx"]
        lines.extend(
            [
                f"## Point {idx}",
                "",
                f"- blocked = `{row['blocked']}`",
                f"- log10 power v68 = `{row['log_power']:.3f}`",
                f"- v68/v36 residual ratio = `{row['ratio']:.3e}`",
                "",
                f"![point {idx} paths]({rel(row['paths'])})",
                "",
                f"![point {idx} AD]({rel(row['ad'])})",
                "",
            ]
        )

    lines.extend(
        [
            "## 当前图像分析结论",
            "",
            "1. `v68` 的提升主要来自 heavy/遮挡区域的路径级 residual 补偿，不是来自 clear 区域。",
            "2. clear 高功率点，例如 Point 120，`v36/v68/v55` 的 PDP 很接近，说明直达径区域已经比较稳，后续不宜激进改。",
            "3. Point 38/55 这类点更有价值：residual 在 h-delay/h-v 上呈现清晰条纹或小块，说明 component 级补偿确实有物理形状。",
            "4. `v68-v55` 差异仍集中在类似的早 delay + 特定 h-bin 路径带，说明 v68 没有完全走错，但强度、相位或路径覆盖还没到 v55 水平。",
            "5. 简单放宽 same-plane fallback 会降低相干性；更合理的下一步是路径族聚类/学习，而不是无脑扩大 residual 范围。",
            "",
            "## 下一步建议",
            "",
            "下一步建议围绕 `路径族聚类 + component residual 校准` 做，而不是继续做全局 heavy gain：",
            "",
            "- 用路径候选的 `delay/AoD/AoA/plane normal/plane offset/excess length/incidence angle` 做路径族聚类。",
            "- 对每个路径族统计训练集 residual 的相干性、平均复标量、可迁移范围。",
            "- 只允许高相干路径族跨 plane 或跨邻域借用 residual。",
            "- 对 clear 高功率点设置保护，避免破坏已经稳定的 LOS/直达径预测。",
            "",
        ]
    )
    (out_dir / "analysis.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--out-dir", default=str(PATHS.analysis / "visualizations/v68_analysis"))
    parser.add_argument("--v36", default=str(PATHS.artifacts / "external/tmp_blend_inspect/v36.npy"))
    parser.add_argument("--v68", default=str(PATHS.temporary / "Round1_Test_Channel.npy"))
    parser.add_argument("--v55-zip", default=str(PATHS.round1_versions / "v55_s1_residual_component_gate_top4_s075_w010_pres_c100_l100_m100_h400_p0_l10_m3_h2_d-1_h-1_v1_online_0.475165.zip"))
    parser.add_argument("--paths", default=str(PATHS.artifacts / "shared_features/path_candidates.npz"))
    parser.add_argument("--geometry-cache", default=str(PATHS.artifacts / "shared_features/height_geometry_cache.npz"))
    parser.add_argument("--no-v55", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    figures_dir = out_dir / "figures"
    data_dir = out_dir / "data"
    ensure_dir(out_dir)
    ensure_dir(figures_dir)
    ensure_dir(data_dir)
    cache_dir = Path("/tmp/round1_viz_cache")
    ensure_dir(cache_dir)

    data = Path(args.data)
    train_pos = np.load(data / "Round1_Train_Pos.npy")
    test_pos = np.load(data / "Round1_Test_Pos.npy")
    geom = np.load(args.geometry_cache)
    train_blocked = geom["train_blocked"]
    test_blocked = geom["test_blocked"]
    paths = np.load(args.paths)

    arrays = {
        "v36": np.load(args.v36, mmap_mode="r"),
        "v68": np.load(args.v68, mmap_mode="r"),
    }
    v55_path = None if args.no_v55 else maybe_extract_npy(Path(args.v55_zip), cache_dir / "v55.npy")
    if v55_path is not None:
        arrays["v55"] = np.load(v55_path, mmap_mode="r")

    powers = {name: point_power(arr) for name, arr in arrays.items()}
    ratios = {
        "v68_vs_v36": point_diff_power(arrays["v68"], arrays["v36"]) / np.maximum(powers["v36"], 1e-30),
    }
    if "v55" in arrays:
        ratios["v55_vs_v36"] = point_diff_power(arrays["v55"], arrays["v36"]) / np.maximum(powers["v36"], 1e-30)
        ratios["v68_vs_v55"] = point_diff_power(arrays["v68"], arrays["v55"]) / np.maximum(powers["v55"], 1e-30)

    images = save_spatial_overview(figures_dir, train_pos, test_pos, train_blocked, test_blocked, powers, ratios)
    selected = choose_points(test_blocked, powers["v68"], ratios["v68_vs_v36"])
    point_rows = []
    for idx in selected:
        ad_img = plot_ad_point(figures_dir, idx, arrays)
        path_img = plot_paths_for_point(figures_dir, idx, train_pos, test_pos, paths)
        point_rows.append(
            {
                "idx": int(idx),
                "blocked": int(test_blocked[idx]),
                "log_power": float(np.log10(max(powers["v68"][idx], 1e-30))),
                "ratio": float(ratios["v68_vs_v36"][idx]),
                "ad": ad_img,
                "paths": path_img,
            }
        )

    summary = {
        "versions": list(arrays.keys()),
        "point_count": int(len(test_pos)),
        "selected_points": selected,
        "power": {k: {"min": float(np.min(v)), "median": float(np.median(v)), "max": float(np.max(v))} for k, v in powers.items()},
        "ratios": {k: {"min": float(np.min(v)), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "max": float(np.max(v))} for k, v in ratios.items()},
        "regime_counts": {
            "clear": int(np.sum(test_blocked == 0)),
            "light": int(np.sum((test_blocked > 0) & (test_blocked <= 25))),
            "medium": int(np.sum((test_blocked > 25) & (test_blocked <= 100))),
            "heavy": int(np.sum(test_blocked > 100)),
        },
    }
    (data_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_analysis_md(out_dir, images, point_rows, summary)
    if v55_path is not None and Path(v55_path).parent == cache_dir:
        Path(v55_path).unlink(missing_ok=True)
        try:
            cache_dir.rmdir()
        except OSError:
            pass
    print(json.dumps({"out_dir": str(out_dir), "analysis": str(out_dir / "analysis.md"), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
