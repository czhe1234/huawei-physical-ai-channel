#!/usr/bin/env python3
"""Visualize the official Round1 inputs without using any predicted submission."""

from __future__ import annotations
from project_paths import PATHS

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_map_features import BS_POS, read_ply_memmap
from evaluate_local_ad_paths import ad_transform


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def percentile_dict(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    return {
        "min": float(np.min(x)),
        "p05": float(np.percentile(x, 5)),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
    }


def load_feature_by_name(path: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    names = [str(x) for x in z["feature_names"]]
    idx = names.index(name)
    return z["train_features"][:, idx].astype(np.float64), z["test_features"][:, idx].astype(np.float64)


def sample_map(vertices, max_points: int, seed: int) -> dict[str, np.ndarray]:
    n = len(vertices)
    rng = np.random.default_rng(seed)
    count = min(max_points, n)
    idx = rng.choice(n, size=count, replace=False)
    return {
        "x": np.asarray(vertices["x"][idx], dtype=np.float64),
        "y": np.asarray(vertices["y"][idx], dtype=np.float64),
        "z": np.asarray(vertices["z"][idx], dtype=np.float64),
        "nx": np.asarray(vertices["nx"][idx], dtype=np.float64),
        "ny": np.asarray(vertices["ny"][idx], dtype=np.float64),
        "nz": np.asarray(vertices["nz"][idx], dtype=np.float64),
    }


def plot_positions_and_map(fig_dir: Path, train_pos, test_pos, map_sample) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    ax = axes[0]
    sc = ax.scatter(map_sample["x"], map_sample["y"], s=1, c=map_sample["z"], cmap="terrain", alpha=0.18, linewidths=0)
    ax.scatter(train_pos[:, 0], train_pos[:, 1], s=8, c="#1f77b4", alpha=0.65, label="train RX")
    ax.scatter(test_pos[:, 0], test_pos[:, 1], s=16, c="#ff7f0e", alpha=0.85, marker="x", label="test RX")
    ax.scatter([BS_POS[0]], [BS_POS[1]], marker="*", s=220, c="red", edgecolors="black", linewidths=0.5, label="BS")
    ax.set_title("Map point cloud sample + official RX positions")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=8)
    fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, label="point-cloud z")

    ax = axes[1]
    wallness = 1.0 - np.abs(map_sample["nz"])
    sc = ax.scatter(map_sample["x"], map_sample["y"], s=1, c=wallness, cmap="magma", alpha=0.2, linewidths=0)
    ax.scatter(train_pos[:, 0], train_pos[:, 1], s=7, c="#1f77b4", alpha=0.55)
    ax.scatter(test_pos[:, 0], test_pos[:, 1], s=14, c="#ff7f0e", alpha=0.85, marker="x")
    ax.scatter([BS_POS[0]], [BS_POS[1]], marker="*", s=220, c="red", edgecolors="black", linewidths=0.5)
    ax.set_title("Vertical-surface proxy: 1 - |normal_z|")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, label="wall proxy")

    out = fig_dir / "positions_and_map.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out.name


def plot_position_distributions(fig_dir: Path, train_pos, test_pos) -> tuple[str, dict]:
    tree = cKDTree(train_pos[:, :2])
    test_nn, _ = tree.query(test_pos[:, :2], k=1)
    train_nn, _ = tree.query(train_pos[:, :2], k=2)
    train_loo = train_nn[:, 1]
    train_dist_bs = np.linalg.norm(train_pos - BS_POS[None, :], axis=1)
    test_dist_bs = np.linalg.norm(test_pos - BS_POS[None, :], axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes[0, 0].scatter(train_pos[:, 0], train_pos[:, 1], s=9, alpha=0.45, label="train")
    axes[0, 0].scatter(test_pos[:, 0], test_pos[:, 1], s=18, marker="x", alpha=0.85, label="test")
    axes[0, 0].scatter([BS_POS[0]], [BS_POS[1]], marker="*", s=180, c="red", edgecolors="black")
    axes[0, 0].set_title("Train/test RX locations")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    axes[0, 0].set_aspect("equal", adjustable="box")
    axes[0, 0].legend()

    axes[0, 1].hist(train_dist_bs, bins=40, alpha=0.6, density=True, label="train")
    axes[0, 1].hist(test_dist_bs, bins=30, alpha=0.6, density=True, label="test")
    axes[0, 1].set_title("Distance to BS")
    axes[0, 1].set_xlabel("meters")
    axes[0, 1].legend()

    axes[1, 0].hist(train_loo, bins=40, alpha=0.6, density=True, label="train leave-one-out NN")
    axes[1, 0].hist(test_nn, bins=30, alpha=0.6, density=True, label="test -> train NN")
    axes[1, 0].set_title("Nearest training point distance")
    axes[1, 0].set_xlabel("meters")
    axes[1, 0].legend()

    axes[1, 1].hist(train_pos[:, 0], bins=40, alpha=0.55, density=True, label="train x")
    axes[1, 1].hist(test_pos[:, 0], bins=30, alpha=0.55, density=True, label="test x")
    ax2 = axes[1, 1].twinx()
    ax2.hist(train_pos[:, 1], bins=40, alpha=0.25, density=True, label="train y", color="#2ca02c")
    ax2.hist(test_pos[:, 1], bins=30, alpha=0.25, density=True, label="test y", color="#d62728")
    axes[1, 1].set_title("x/y coordinate distributions")
    axes[1, 1].set_xlabel("coordinate value")
    h1, l1 = axes[1, 1].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axes[1, 1].legend(h1 + h2, l1 + l2, fontsize=8)

    out = fig_dir / "position_distributions.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    stats = {
        "train_distance_to_bs": percentile_dict(train_dist_bs),
        "test_distance_to_bs": percentile_dict(test_dist_bs),
        "train_loo_nn_distance": percentile_dict(train_loo),
        "test_to_train_nn_distance": percentile_dict(test_nn),
    }
    return out.name, stats


def scatter_metric(ax, pos, values, title, cmap="viridis", s=14, test_pos=None):
    sc = ax.scatter(pos[:, 0], pos[:, 1], s=s, c=values, cmap=cmap, edgecolors="none")
    if test_pos is not None:
        ax.scatter(test_pos[:, 0], test_pos[:, 1], s=12, marker="x", c="black", alpha=0.35, linewidths=0.6)
    ax.scatter([BS_POS[0]], [BS_POS[1]], marker="*", s=160, c="red", edgecolors="black", linewidths=0.5)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)


def plot_geometry_features(fig_dir: Path, train_pos, test_pos, train_blocked, test_blocked, map_features_path: Path) -> tuple[str, dict]:
    train_los_min, test_los_min = load_feature_by_name(map_features_path, "los_min_dist")
    train_wall_dist, test_wall_dist = load_feature_by_name(map_features_path, "rx_wall_dist2")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
    scatter_metric(axes[0, 0], train_pos, train_blocked, "Train occlusion count", cmap="plasma", test_pos=test_pos)
    scatter_metric(axes[0, 1], test_pos, test_blocked, "Test occlusion count", cmap="plasma")
    scatter_metric(axes[1, 0], train_pos, train_los_min, "Train LOS min distance to wall points", cmap="viridis", test_pos=test_pos)
    scatter_metric(axes[1, 1], test_pos, test_los_min, "Test LOS min distance to wall points", cmap="viridis")

    out = fig_dir / "geometry_features.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    stats = {
        "train_blocked": percentile_dict(train_blocked),
        "test_blocked": percentile_dict(test_blocked),
        "train_los_min_dist": percentile_dict(train_los_min),
        "test_los_min_dist": percentile_dict(test_los_min),
        "train_rx_wall_dist2": percentile_dict(train_wall_dist),
        "test_rx_wall_dist2": percentile_dict(test_wall_dist),
    }
    return out.name, stats


def compute_channel_power_stats(channel_path: Path, batch: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = np.load(channel_path, mmap_mode="r")
    n = h.shape[0]
    power = np.empty(n, dtype=np.float64)
    top_delay = np.empty(n, dtype=np.int32)
    delay_centroid = np.empty(n, dtype=np.float64)
    mean_pdp = np.zeros(h.shape[-1], dtype=np.float64)
    delay_index = np.arange(h.shape[-1], dtype=np.float64)
    for start in range(0, n, batch):
        end = min(start + batch, n)
        hb = np.asarray(h[start:end])
        p = np.abs(hb) ** 2
        power[start:end] = np.sum(p, axis=(1, 2, 3), dtype=np.float64)
        cir = np.fft.ifft(hb, axis=-1)
        pdp = np.sum(np.abs(cir) ** 2, axis=(1, 2), dtype=np.float64)
        mean_pdp += np.sum(pdp, axis=0)
        top_delay[start:end] = np.argmax(pdp, axis=1)
        denom = np.sum(pdp, axis=1)
        delay_centroid[start:end] = np.sum(pdp * delay_index[None, :], axis=1) / np.maximum(denom, 1e-30)
    mean_pdp /= n
    return power, top_delay, delay_centroid, mean_pdp


def plot_train_power(fig_dir: Path, train_pos, train_power, top_delay, delay_centroid, train_blocked) -> tuple[str, dict]:
    dist_bs = np.linalg.norm(train_pos - BS_POS[None, :], axis=1)
    logp = np.log10(np.maximum(train_power, 1e-30))
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
    scatter_metric(axes[0, 0], train_pos, logp, "Train truth log10 channel power", cmap="viridis")
    scatter_metric(axes[0, 1], train_pos, top_delay, "Dominant delay bin per train point", cmap="turbo")
    sc = axes[1, 0].scatter(dist_bs, logp, s=10, c=train_blocked, cmap="plasma", alpha=0.7, linewidths=0)
    axes[1, 0].set_title("Power vs distance to BS; color = occlusion")
    axes[1, 0].set_xlabel("distance to BS")
    axes[1, 0].set_ylabel("log10 total power")
    plt.colorbar(sc, ax=axes[1, 0], fraction=0.035, pad=0.02)
    sc = axes[1, 1].scatter(delay_centroid, logp, s=10, c=dist_bs, cmap="viridis", alpha=0.7, linewidths=0)
    axes[1, 1].set_title("Power vs delay centroid; color = distance")
    axes[1, 1].set_xlabel("delay centroid")
    axes[1, 1].set_ylabel("log10 total power")
    plt.colorbar(sc, ax=axes[1, 1], fraction=0.035, pad=0.02)
    out = fig_dir / "train_power_geometry.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    stats = {
        "train_channel_power": percentile_dict(train_power),
        "train_log10_channel_power": percentile_dict(logp),
        "dominant_delay_bin": percentile_dict(top_delay),
        "delay_centroid": percentile_dict(delay_centroid),
        "corr_logpower_distance": float(np.corrcoef(logp, dist_bs)[0, 1]),
        "corr_logpower_blocked": float(np.corrcoef(logp, train_blocked)[0, 1]),
        "corr_delay_centroid_distance": float(np.corrcoef(delay_centroid, dist_bs)[0, 1]),
    }
    return out.name, stats


def compute_ad_summary(channel_path: Path, batch: int, max_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    h = np.load(channel_path, mmap_mode="r")
    n = h.shape[0]
    rng = np.random.default_rng(seed)
    ids = np.sort(rng.choice(n, size=min(max_samples, n), replace=False))
    h_delay = np.zeros((16, 192), dtype=np.float64)
    hv = np.zeros((16, 8), dtype=np.float64)
    component_power = np.zeros((16, 8, 192), dtype=np.float64)
    for start in range(0, len(ids), batch):
        batch_ids = ids[start : start + batch]
        ad = ad_transform(np.asarray(h[batch_ids]))
        p = np.sum(np.abs(ad) ** 2, axis=(1, 4), dtype=np.float64)  # B,H,V,S
        component_power += np.sum(p, axis=0)
        h_delay += np.sum(p, axis=(0, 2))
        hv += np.sum(p, axis=(0, 3))
    component_power /= len(ids)
    h_delay /= len(ids)
    hv /= len(ids)
    flat = component_power.reshape(-1)
    top = np.argpartition(flat, -20)[-20:]
    top = top[np.argsort(flat[top])[::-1]]
    rows = []
    for rank, fid in enumerate(top, 1):
        hidx = int(fid // (8 * 192))
        rem = int(fid % (8 * 192))
        vidx = int(rem // 192)
        sidx = int(rem % 192)
        rows.append({"rank": rank, "h": hidx, "v": vidx, "delay": sidx, "power": float(flat[fid])})
    return h_delay, hv, component_power, rows


def plot_pdp_ad_summary(fig_dir: Path, mean_pdp, h_delay, hv, top_components) -> tuple[str, dict]:
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    axes[0, 0].plot(10 * np.log10(np.maximum(mean_pdp, 1e-30)))
    axes[0, 0].set_title("Mean train PDP")
    axes[0, 0].set_xlabel("delay bin")
    axes[0, 0].set_ylabel("dB")
    im = axes[0, 1].imshow(10 * np.log10(np.maximum(h_delay, 1e-30)), aspect="auto", origin="lower", cmap="viridis")
    axes[0, 1].set_title("Mean angle-delay power, summed over v")
    axes[0, 1].set_xlabel("delay bin")
    axes[0, 1].set_ylabel("h bin")
    plt.colorbar(im, ax=axes[0, 1], fraction=0.035, pad=0.02)
    im = axes[1, 0].imshow(10 * np.log10(np.maximum(hv, 1e-30)), aspect="auto", origin="lower", cmap="viridis")
    axes[1, 0].set_title("Mean h-v angular power")
    axes[1, 0].set_xlabel("v bin")
    axes[1, 0].set_ylabel("h bin")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.035, pad=0.02)
    axes[1, 1].axis("off")
    table_rows = [[r["rank"], r["h"], r["v"], r["delay"], f'{r["power"]:.3e}'] for r in top_components[:12]]
    table = axes[1, 1].table(
        cellText=table_rows,
        colLabels=["rank", "h", "v", "delay", "power"],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.2)
    axes[1, 1].set_title("Top mean AD components")
    out = fig_dir / "train_pdp_ad_summary.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    stats = {
        "mean_pdp_top_bins": [
            {"delay": int(i), "power": float(mean_pdp[i])}
            for i in np.argsort(mean_pdp)[::-1][:12]
        ],
        "top_mean_ad_components": top_components,
    }
    return out.name, stats


def plot_representative_train_points(fig_dir: Path, channel_path: Path, train_pos, train_power, train_blocked, top_delay) -> tuple[str, list[int]]:
    h = np.load(channel_path, mmap_mode="r")
    candidates = []
    for arr in [
        np.argsort(train_power)[::-1][:2],
        np.argsort(train_power)[:2],
        np.flatnonzero(train_blocked > np.percentile(train_blocked, 95))[:2],
        np.argsort(np.abs(top_delay - np.median(top_delay)))[::-1][:2],
    ]:
        for idx in arr:
            i = int(idx)
            if i not in candidates:
                candidates.append(i)
    ids = candidates[:6]
    fig, axes = plt.subplots(len(ids), 3, figsize=(15, 3.2 * len(ids)), constrained_layout=True)
    if len(ids) == 1:
        axes = axes[None, :]
    for row, idx in enumerate(ids):
        hb = np.asarray(h[idx : idx + 1])
        cir = np.fft.ifft(hb, axis=-1)
        pdp = np.sum(np.abs(cir[0]) ** 2, axis=(0, 1), dtype=np.float64)
        ad = ad_transform(hb)
        p = np.sum(np.abs(ad[0]) ** 2, axis=(0, 3), dtype=np.float64)
        h_delay = np.sum(p, axis=1)
        hv = np.sum(p, axis=2)
        axes[row, 0].plot(10 * np.log10(np.maximum(pdp, 1e-30)))
        axes[row, 0].set_title(f"train {idx} PDP, blocked={int(train_blocked[idx])}")
        axes[row, 0].set_xlabel("delay")
        axes[row, 0].set_ylabel("dB")
        im = axes[row, 1].imshow(10 * np.log10(np.maximum(h_delay, 1e-30)), aspect="auto", origin="lower", cmap="viridis")
        axes[row, 1].set_title(f"train {idx} h-delay")
        axes[row, 1].set_xlabel("delay")
        axes[row, 1].set_ylabel("h bin")
        plt.colorbar(im, ax=axes[row, 1], fraction=0.035, pad=0.02)
        im = axes[row, 2].imshow(10 * np.log10(np.maximum(hv, 1e-30)), aspect="auto", origin="lower", cmap="viridis")
        axes[row, 2].set_title(f"train {idx} h-v")
        axes[row, 2].set_xlabel("v bin")
        axes[row, 2].set_ylabel("h bin")
        plt.colorbar(im, ax=axes[row, 2], fraction=0.035, pad=0.02)
    out = fig_dir / "representative_train_channels.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out.name, ids


def write_analysis(out_dir: Path, figures: dict, stats: dict) -> None:
    lines = [
        "# Round1 官方数据可视化分析",
        "",
        "本目录只分析官方输入数据，不使用任何提交预测结果。",
        "",
        "## 文件结构",
        "",
        "- `figures/`：官方数据可视化 PNG。",
        "- `data/summary.json`：统计数据。",
        "- `analysis.md`：当前文档。",
        "",
        "## 官方数据",
        "",
        "| 文件 | 含义 |",
        "| --- | --- |",
        "| `Round1_Train_Pos.npy` | 2000 个训练接收点坐标 |",
        "| `Round1_Test_Pos.npy` | 500 个官方测试接收点坐标，无信道真值 |",
        "| `Round1_Train_Channel.npy` | 训练真值信道，shape `(2000,256,4,192)` |",
        "| `Round1_Map.ply` | 场景点云，含点坐标和法向量 |",
        "| `Round1_Setup.json` | 阵列规模、子载波数、BS 位置和评分权重 |",
        "",
        "## 核心统计",
        "",
        "```json",
        json.dumps(stats, indent=2, ensure_ascii=False),
        "```",
        "",
        "## 1. 地图点云与官方接收点",
        "",
        f"![positions and map](figures/{figures['positions_and_map']})",
        "",
        "左图用点云高度 `z` 着色，叠加训练/测试接收点和 BS；右图用 `1-|normal_z|` 表示垂直墙面 proxy。",
        "",
        "观察重点：",
        "",
        "- 训练点和测试点覆盖区域高度重合，测试点不是明显跑到训练区域外。",
        "- BS 在左侧 `(50,0,25)`，接收点都在 `z=1.5` 平面。",
        "- 墙面 proxy 能显示建筑/障碍的主要边界，后续路径候选应尽量贴合这些结构。",
        "",
        "## 2. 训练/测试点分布",
        "",
        f"![position distributions](figures/{figures['position_distributions']})",
        "",
        "这张图比较训练点与测试点的空间分布、到 BS 的距离分布、测试点到最近训练点距离、以及 x/y 坐标分布。",
        "",
        "观察重点：",
        "",
        "- 如果 `test -> train NN` 和 `train leave-one-out NN` 很接近，说明测试点局部密度并不比训练点困难很多。",
        "- 如果距离 BS 分布相似，说明线上不是简单的远距离外推问题。",
        "- 这支持我们之前的判断：问题主要不是测试点选得更远，而是 H 的路径相位/component 很难预测。",
        "",
        "## 3. 官方几何/遮挡特征",
        "",
        f"![geometry features](figures/{figures['geometry_features']})",
        "",
        "这张图只基于官方点云和位置，展示训练/测试点的遮挡计数与 LOS 最近墙面距离。",
        "",
        "观察重点：",
        "",
        "- heavy/遮挡点主要集中在部分区域，和之前 v68 residual 强区域有对应关系。",
        "- 测试集 heavy 点数量不少，说明多径/NLOS 是关键。",
        "- 但遮挡计数只是粗 proxy，不能直接代表真实路径；之前 v46 严重下降也说明不能把“建筑内部功率 0”等规则硬塞进去。",
        "",
        "## 4. 训练真值功率与几何关系",
        "",
        f"![train power geometry](figures/{figures['train_power_geometry']})",
        "",
        "这张图只用训练真值 `Round1_Train_Channel.npy`：总功率、主 delay bin、功率与 BS 距离/遮挡/delay centroid 的关系。",
        "",
        "观察重点：",
        "",
        "- 训练真值功率有明显空间结构，不能只用距离解释。",
        "- 主 delay bin 和 delay centroid 与距离有相关性，但不等于完整相位可预测。",
        "- 强功率区域是模型最应该保护的区域；低功率 heavy 区域可以更激进，但线上权重可能有限。",
        "",
        "## 5. 训练真值 PDP / angle-delay 结构",
        "",
        f"![train PDP AD summary](figures/{figures['train_pdp_ad_summary']})",
        "",
        "这张图展示训练真值的平均 PDP、平均 h-delay 功率、平均 h-v 角度功率，以及全局最强 AD component。",
        "",
        "观察重点：",
        "",
        "- 能量高度集中在早 delay，这和我们长期观察到的 delay 稀疏性一致。",
        "- h-delay 上存在稳定的水平角带，说明 H 内部 component 结构非常重要。",
        "- 这支持后续从“整信道平均”转向“component/path-family 校准”。",
        "",
        "## 6. 代表训练点真值信道",
        "",
        f"![representative train channels](figures/{figures['representative_train_channels']})",
        "",
        "这些点来自官方训练真值，包含高功率、低功率、强遮挡和 delay 结构特殊的点。",
        "",
        "观察重点：",
        "",
        "- 单点信道通常由少数 delay/angle 结构主导。",
        "- 不同区域的主 h-bin、v-bin、delay 会迁移，因此绝对 component ID 不一定稳定。",
        "- 这解释了为什么 v68 的“路径相对 component tracking”比绝对 AD component 更有物理意义。",
        "",
        "## 当前结论",
        "",
        "1. 官方训练/测试点空间覆盖相近，线上困难不是简单外推距离问题。",
        "2. 官方点云提供了有价值的遮挡和墙面结构，但粗规则不够，必须服务于 component/path-family 建模。",
        "3. 训练真值信道高度稀疏，主要能量集中在早 delay 和少数 h-v 角度区域。",
        "4. H 内部维度结构非常关键，不能只把整个 H 当作一个整体插值。",
        "5. 下一步应重点做：官方几何路径候选 + 训练真值 component 统计 + 测试点路径族归因。",
        "",
    ]
    (out_dir / "analysis.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--out-dir", default=str(PATHS.analysis / "visualizations/official_data"))
    parser.add_argument("--map-features", default=str(PATHS.artifacts / "shared_features/map_features.npz"))
    parser.add_argument("--geometry-cache", default=str(PATHS.artifacts / "shared_features/height_geometry_cache.npz"))
    parser.add_argument("--map-sample", type=int, default=180000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--ad-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260707)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    data_dir = out_dir / "data"
    ensure_dir(fig_dir)
    ensure_dir(data_dir)

    data = Path(args.data)
    train_pos = np.load(data / "Round1_Train_Pos.npy")
    test_pos = np.load(data / "Round1_Test_Pos.npy")
    setup = json.loads((data / "Round1_Setup.json").read_text())
    geom = np.load(args.geometry_cache)
    train_blocked = geom["train_blocked"].astype(np.float64)
    test_blocked = geom["test_blocked"].astype(np.float64)
    vertices = read_ply_memmap(data / "Round1_Map.ply")
    map_sample = sample_map(vertices, args.map_sample, args.seed)

    figures = {}
    stats = {
        "setup": setup,
        "train_pos_shape": list(train_pos.shape),
        "test_pos_shape": list(test_pos.shape),
        "train_pos_min": train_pos.min(axis=0).astype(float).tolist(),
        "train_pos_max": train_pos.max(axis=0).astype(float).tolist(),
        "test_pos_min": test_pos.min(axis=0).astype(float).tolist(),
        "test_pos_max": test_pos.max(axis=0).astype(float).tolist(),
        "map_vertex_count": int(len(vertices)),
        "map_sample_count": int(len(map_sample["x"])),
    }

    figures["positions_and_map"] = plot_positions_and_map(fig_dir, train_pos, test_pos, map_sample)
    figures["position_distributions"], pos_stats = plot_position_distributions(fig_dir, train_pos, test_pos)
    stats.update(pos_stats)
    figures["geometry_features"], geom_stats = plot_geometry_features(
        fig_dir, train_pos, test_pos, train_blocked, test_blocked, Path(args.map_features)
    )
    stats.update(geom_stats)

    channel_path = data / "Round1_Train_Channel.npy"
    train_power, top_delay, delay_centroid, mean_pdp = compute_channel_power_stats(channel_path, args.batch)
    figures["train_power_geometry"], power_stats = plot_train_power(
        fig_dir, train_pos, train_power, top_delay, delay_centroid, train_blocked
    )
    stats.update(power_stats)
    h_delay, hv, _, top_components = compute_ad_summary(channel_path, args.batch, args.ad_samples, args.seed)
    figures["train_pdp_ad_summary"], ad_stats = plot_pdp_ad_summary(fig_dir, mean_pdp, h_delay, hv, top_components)
    stats.update(ad_stats)
    figures["representative_train_channels"], rep_ids = plot_representative_train_points(
        fig_dir, channel_path, train_pos, train_power, train_blocked, top_delay
    )
    stats["representative_train_points"] = [int(x) for x in rep_ids]

    (data_dir / "summary.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    write_analysis(out_dir, figures, stats)
    print(json.dumps({"out_dir": str(out_dir), "analysis": str(out_dir / "analysis.md"), "figures": figures}, indent=2))


if __name__ == "__main__":
    main()
