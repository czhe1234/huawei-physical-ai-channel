#!/usr/bin/env python3
"""Visualize training-set path/component families.

This diagnostic uses only official training truth plus geometric path candidates.
It groups dominant angle-delay components by:

    geometric path candidate + relative component offset

The goal is to inspect whether stable path families exist before building a new
submission model.
"""

from __future__ import annotations
from project_paths import PATHS

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_map_features import BS_POS
from evaluate_local_ad_paths import MH, MV, ad_transform


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def signed_circular_delta(values, centers, period):
    return ((values - centers + period / 2.0) % period) - period / 2.0


def percentile_dict(x) -> dict:
    x = np.asarray(x, dtype=np.float64)
    return {
        "min": float(np.min(x)),
        "p05": float(np.percentile(x, 5)),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
    }


def plane_line_xy(paths, plane_id: int, center_xy: np.ndarray, radius: float = 55.0):
    if plane_id < 0:
        return None
    tangent = paths["plane_tangent"][plane_id].astype(np.float64)
    normal = paths["plane_normal"][plane_id].astype(np.float64)
    c = float(paths["plane_c"][plane_id])
    center_t = float(np.dot(center_xy, tangent))
    t0 = max(float(paths["plane_t_min"][plane_id]), center_t - radius)
    t1 = min(float(paths["plane_t_max"][plane_id]), center_t + radius)
    p0 = tangent * t0 + normal * c
    p1 = tangent * t1 + normal * c
    return p0, p1


def make_records(assign, paths, train_pos, train_blocked, top_per_point, power_floor):
    n = assign["component_power"].shape[0]
    thresholds = assign["thresholds"].astype(np.float32)
    rows = []
    comp_power = assign["component_power"][:, :top_per_point].astype(np.float64)
    total_power = assign["total_power"].astype(np.float64)
    valid_point = total_power > power_floor
    best = assign["best_path"][:, :top_per_point].astype(np.int64)
    row_idx = np.arange(n)[:, None]

    assigned = (
        (assign["delay_err"][:, :top_per_point] <= thresholds[0])
        & (assign["h_err"][:, :top_per_point] <= thresholds[1])
        & (assign["v_err"][:, :top_per_point] <= thresholds[2])
        & np.isfinite(assign["assignment_score"][:, :top_per_point])
        & valid_point[:, None]
    )
    path_type = assign["path_type"][row_idx, best].astype(np.int16)
    plane_id = paths["train_plane_id"][row_idx, best].astype(np.int32)
    path_delay = assign["path_delay_bin"][row_idx, best].astype(np.float64)
    path_h = assign["path_h_bin"][row_idx, best].astype(np.float64)
    path_v = assign["path_v_bin"][row_idx, best].astype(np.float64)
    path_length = paths["train_path_length"][row_idx, best].astype(np.float64)
    path_excess = paths["train_excess_length"][row_idx, best].astype(np.float64)
    incidence = paths["train_incidence_cos"][row_idx, best].astype(np.float64)

    hbin = assign["hbin"][:, :top_per_point].astype(np.float64)
    vbin = assign["vbin"][:, :top_per_point].astype(np.float64)
    sbin = assign["sbin"][:, :top_per_point].astype(np.float64)
    dh = signed_circular_delta(hbin, path_h, MH)
    dv = signed_circular_delta(vbin, path_v, MV)
    ds = sbin - path_delay

    for i in range(n):
        for r in range(top_per_point):
            if not assigned[i, r]:
                continue
            ptype = int(path_type[i, r])
            pid = int(plane_id[i, r]) if ptype == 1 else -1
            dsi = int(np.rint(ds[i, r]))
            dhi = int(np.rint(dh[i, r]))
            dvi = int(np.rint(dv[i, r]))
            key = (ptype, pid, dsi, dhi, dvi)
            rows.append(
                {
                    "point": i,
                    "rank": r,
                    "key": key,
                    "path_type": ptype,
                    "plane_id": pid,
                    "ds": float(ds[i, r]),
                    "dh": float(dh[i, r]),
                    "dv": float(dv[i, r]),
                    "ds_bin": dsi,
                    "dh_bin": dhi,
                    "dv_bin": dvi,
                    "component_id": int(assign["component_ids"][i, r]),
                    "hbin": int(assign["hbin"][i, r]),
                    "vbin": int(assign["vbin"][i, r]),
                    "sbin": int(assign["sbin"][i, r]),
                    "component_power": float(comp_power[i, r]),
                    "point_power": float(total_power[i]),
                    "power_frac": float(comp_power[i, r] / max(total_power[i], 1e-30)),
                    "score": float(assign["assignment_score"][i, r]),
                    "path_length": float(path_length[i, r]),
                    "path_excess": float(path_excess[i, r]),
                    "incidence_cos": float(incidence[i, r]) if np.isfinite(incidence[i, r]) else None,
                    "x": float(train_pos[i, 0]),
                    "y": float(train_pos[i, 1]),
                    "blocked": float(train_blocked[i]),
                }
            )
    return rows


def compute_record_phases(rows, channel_path: Path, beta: float, batch: int):
    by_point = defaultdict(list)
    for rid, rec in enumerate(rows):
        by_point[rec["point"]].append(rid)
    channel = np.load(channel_path, mmap_mode="r")
    points = np.asarray(sorted(by_point), dtype=np.int64)
    phase_raw = np.zeros(len(rows), dtype=np.complex64)
    phase_aligned = np.zeros(len(rows), dtype=np.complex64)
    amp = np.zeros(len(rows), dtype=np.float32)
    for start in range(0, len(points), batch):
        pts = points[start : start + batch]
        ad = ad_transform(np.asarray(channel[pts]))
        local = {int(p): j for j, p in enumerate(pts)}
        for p in pts:
            j = local[int(p)]
            for rid in by_point[int(p)]:
                rec = rows[rid]
                coeff = ad[j, :, rec["hbin"], rec["vbin"], :, rec["sbin"]].reshape(-1)
                mag = np.abs(coeff)
                if np.max(mag) <= 1e-30:
                    continue
                z = coeff[int(np.argmax(mag))]
                amp[rid] = float(np.abs(z))
                phase_raw[rid] = z / np.abs(z)
                za = z * np.exp(1j * beta * rec["path_length"])
                phase_aligned[rid] = za / max(abs(za), 1e-30)
    return phase_raw, phase_aligned, amp


def aggregate_clusters(rows, phase_raw, phase_aligned, min_points):
    groups = defaultdict(list)
    for i, rec in enumerate(rows):
        groups[rec["key"]].append(i)
    clusters = []
    for key, ids in groups.items():
        unique_points = sorted({rows[i]["point"] for i in ids})
        if len(unique_points) < min_points:
            continue
        power = np.asarray([rows[i]["component_power"] for i in ids], dtype=np.float64)
        frac = np.asarray([rows[i]["power_frac"] for i in ids], dtype=np.float64)
        score = np.asarray([rows[i]["score"] for i in ids], dtype=np.float64)
        blocked = np.asarray([rows[i]["blocked"] for i in ids], dtype=np.float64)
        raw = phase_raw[ids]
        aligned = phase_aligned[ids]
        good_raw = np.abs(raw) > 0
        good_aligned = np.abs(aligned) > 0
        ptype, pid, dsb, dhb, dvb = key
        label = "LOS" if ptype == 0 else f"P{pid}"
        label = f"{label} ds{dsb:+d} dh{dhb:+d} dv{dvb:+d}"
        clusters.append(
            {
                "key": key,
                "label": label,
                "path_type": int(ptype),
                "plane_id": int(pid),
                "ds_bin": int(dsb),
                "dh_bin": int(dhb),
                "dv_bin": int(dvb),
                "record_count": int(len(ids)),
                "point_count": int(len(unique_points)),
                "total_component_power": float(np.sum(power)),
                "median_power_frac": float(np.median(frac)),
                "mean_power_frac": float(np.mean(frac)),
                "median_score": float(np.median(score)),
                "median_blocked": float(np.median(blocked)),
                "raw_phase_coherence": float(abs(np.mean(raw[good_raw]))) if np.any(good_raw) else 0.0,
                "path_aligned_phase_coherence": float(abs(np.mean(aligned[good_aligned]))) if np.any(good_aligned) else 0.0,
                "record_ids": ids,
                "point_ids": unique_points,
            }
        )
    clusters.sort(key=lambda x: x["total_component_power"], reverse=True)
    return clusters


def plot_cluster_overview(fig_dir: Path, clusters):
    top = clusters[:24]
    labels = [c["label"] for c in top]
    power = np.asarray([c["total_component_power"] for c in top])
    counts = np.asarray([c["point_count"] for c in top])
    coh = np.asarray([c["path_aligned_phase_coherence"] for c in top])
    refl = np.asarray([c["path_type"] == 1 for c in top])

    fig, axes = plt.subplots(1, 2, figsize=(17, 7), constrained_layout=True)
    colors = np.where(refl, "#ff7f0e", "#1f77b4")
    y = np.arange(len(top))[::-1]
    axes[0].barh(y, power[::-1], color=colors[::-1], alpha=0.85)
    axes[0].set_yticks(y, labels[::-1], fontsize=7)
    axes[0].set_xlabel("sum component power")
    axes[0].set_title("Top path-relative component clusters")

    sc = axes[1].scatter(counts, coh, s=60 + 800 * power / max(np.max(power), 1e-30), c=np.log10(np.maximum(power, 1e-30)), cmap="viridis", alpha=0.8, edgecolors="black", linewidths=0.3)
    axes[1].set_xlabel("unique training points in cluster")
    axes[1].set_ylabel("path-length-aligned phase coherence")
    axes[1].set_title("Cluster support vs phase coherence")
    for i, c in enumerate(top[:10]):
        axes[1].annotate(str(i + 1), (counts[i], coh[i]), fontsize=8)
    plt.colorbar(sc, ax=axes[1], fraction=0.035, pad=0.02, label="log10 cluster power")
    out = fig_dir / "cluster_overview.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out.name


def plot_dominant_cluster_map(fig_dir: Path, rows, clusters, train_pos, test_pos):
    top = clusters[:10]
    key_to_rank = {c["key"]: i for i, c in enumerate(top)}
    per_point = defaultdict(lambda: defaultdict(float))
    for rec in rows:
        if rec["key"] in key_to_rank:
            per_point[rec["point"]][key_to_rank[rec["key"]]] += rec["component_power"]
    dom = np.full(len(train_pos), -1, dtype=np.int32)
    dom_power = np.zeros(len(train_pos), dtype=np.float64)
    for p, vals in per_point.items():
        if not vals:
            continue
        k = max(vals, key=vals.get)
        dom[p] = k
        dom_power[p] = vals[k]
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    ax.scatter(train_pos[:, 0], train_pos[:, 1], s=6, c="#d0d0d0", alpha=0.35, linewidths=0, label="other train")
    for k, c in enumerate(top):
        m = dom == k
        if np.any(m):
            ax.scatter(train_pos[m, 0], train_pos[m, 1], s=10 + 60 * dom_power[m] / max(np.max(dom_power), 1e-30), color=cmap(k), alpha=0.72, linewidths=0, label=f"{k+1}: {c['label']}")
    ax.scatter(test_pos[:, 0], test_pos[:, 1], s=12, marker="x", c="black", alpha=0.3, linewidths=0.6, label="test")
    ax.scatter([BS_POS[0]], [BS_POS[1]], marker="*", s=200, c="red", edgecolors="black", linewidths=0.5, label="BS")
    ax.set_title("Dominant top path-relative cluster per training point")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
    out = fig_dir / "dominant_cluster_map.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out.name


def plot_top_cluster_maps(fig_dir: Path, rows, clusters, paths, train_pos, test_pos, top_n=8):
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    axes = axes.reshape(-1)
    for ax, c in zip(axes, clusters[:top_n]):
        ids = c["record_ids"]
        point_power = defaultdict(float)
        for rid in ids:
            rec = rows[rid]
            point_power[rec["point"]] += rec["component_power"]
        pts = np.asarray(list(point_power.keys()), dtype=np.int64)
        vals = np.asarray([point_power[int(p)] for p in pts], dtype=np.float64)
        ax.scatter(train_pos[:, 0], train_pos[:, 1], s=4, c="#d0d0d0", alpha=0.22, linewidths=0)
        if pts.size:
            sc = ax.scatter(train_pos[pts, 0], train_pos[pts, 1], s=18, c=np.log10(np.maximum(vals, 1e-30)), cmap="viridis", alpha=0.85, linewidths=0)
            plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
        if c["path_type"] == 1:
            line = plane_line_xy(paths, c["plane_id"], train_pos[pts].mean(axis=0)[:2] if pts.size else np.array([150.0, -50.0]))
            if line is not None:
                p0, p1 = line
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="red", lw=2.0, alpha=0.75)
            test_has = np.any((paths["test_plane_id"] == c["plane_id"]) & (paths["test_path_type"] == 1), axis=1)
        else:
            test_has = np.any(paths["test_path_type"] == 0, axis=1)
        ax.scatter(test_pos[test_has, 0], test_pos[test_has, 1], s=12, marker="x", c="black", alpha=0.35, linewidths=0.6)
        ax.scatter([BS_POS[0]], [BS_POS[1]], marker="*", s=120, c="red", edgecolors="black", linewidths=0.4)
        ax.set_title(f"{c['label']}\npts={c['point_count']} coh={c['path_aligned_phase_coherence']:.2f}", fontsize=9)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    for ax in axes[len(clusters[:top_n]) :]:
        ax.axis("off")
    out = fig_dir / "top_cluster_maps.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out.name


def plot_offset_distributions(fig_dir: Path, rows, clusters):
    top_keys = {c["key"] for c in clusters[:12]}
    ds = np.asarray([rec["ds"] for rec in rows if rec["key"] in top_keys], dtype=np.float64)
    dh = np.asarray([rec["dh"] for rec in rows if rec["key"] in top_keys], dtype=np.float64)
    dv = np.asarray([rec["dv"] for rec in rows if rec["key"] in top_keys], dtype=np.float64)
    powv = np.asarray([rec["component_power"] for rec in rows if rec["key"] in top_keys], dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    axes[0].hist(ds, bins=np.arange(-3.5, 3.6, 0.5), weights=powv, alpha=0.8)
    axes[0].set_title("Power-weighted delay offset")
    axes[0].set_xlabel("component delay - path delay")
    axes[1].hist(dh, bins=np.arange(-4.5, 4.6, 0.5), weights=powv, alpha=0.8)
    axes[1].set_title("Power-weighted h offset")
    axes[1].set_xlabel("component h - path h")
    axes[2].hist(dv, bins=np.arange(-3.5, 3.6, 0.5), weights=powv, alpha=0.8)
    axes[2].set_title("Power-weighted v offset")
    axes[2].set_xlabel("component v - path v")
    out = fig_dir / "offset_distributions.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out.name


def write_analysis(out_dir: Path, figures: dict, summary: dict, clusters_public):
    lines = [
        "# 训练集路径族 / Component 聚类可视化",
        "",
        "本分析只使用官方训练真值 `Round1_Train_Channel.npy`、官方位置/点云派生的几何路径候选，以及 component-path assignment 缓存。",
        "",
        "## 路径族定义",
        "",
        "这里的路径族不是黑盒聚类，而是一个物理可解释分组：",
        "",
        "```text",
        "路径族 = 几何路径候选(LOS 或某个反射平面) + 相对 component 偏移(Δdelay, Δh, Δv)",
        "```",
        "",
        "例如 `P37 ds+0 dh-1 dv+1` 表示：component 被归因到第 37 个反射平面，对应 delay/h/v bin 相对几何预测分别偏移 `0/-1/+1`。",
        "",
        "## 文件结构",
        "",
        "- `figures/`：可视化 PNG。",
        "- `data/summary.json`：聚类统计。",
        "- `analysis.md`：当前文档。",
        "",
        "## 核心统计",
        "",
        "```json",
        json.dumps(summary, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Top 路径族总览",
        "",
        f"![cluster overview](figures/{figures['cluster_overview']})",
        "",
        "左图是总 component power 最大的路径族；右图看每个路径族的训练点覆盖数和 path-length 相位校准后的相干性。",
        "",
        "阅读重点：",
        "",
        "- 高 power 但低 coherence：这类路径族能量大，但相位不稳定，直接跨点平均风险高。",
        "- 中等 power 且高 coherence：这类路径族更适合做可迁移 residual/scalar 校准。",
        "- 覆盖点数太少的 cluster 不适合泛化到测试集。",
        "",
        "## 主导路径族空间分布",
        "",
        f"![dominant cluster map](figures/{figures['dominant_cluster_map']})",
        "",
        "这张图给每个训练点标出它在 top 路径族里最强的那个。测试点以黑色叉号叠加。",
        "",
        "阅读重点：",
        "",
        "- 如果路径族在空间上连续成片，说明它更像真实传播状态。",
        "- 如果路径族碎片化严重，说明仅用该 path family 做测试点外推会不稳定。",
        "",
        "## Top 路径族局部地图",
        "",
        f"![top cluster maps](figures/{figures['top_cluster_maps']})",
        "",
        "每个子图展示一个 top 路径族在训练点上的空间支持。红线是该反射平面，黑色叉号表示测试点中也有该平面候选路径的位置。",
        "",
        "阅读重点：",
        "",
        "- 训练点支持和测试点候选如果空间重合，说明该路径族有机会用于测试预测。",
        "- 如果测试点候选很多但训练支持少，直接外推风险高。",
        "",
        "## 相对 offset 分布",
        "",
        f"![offset distributions](figures/{figures['offset_distributions']})",
        "",
        "这张图展示 top 路径族的 component 相对几何路径预测的偏移分布。",
        "",
        "阅读重点：",
        "",
        "- 如果 offset 集中，说明几何路径预测和 AD component 有稳定关系。",
        "- 如果 offset 分散，说明我们现在的路径几何/阵列映射还不够准。",
        "",
        "## Top 路径族表",
        "",
        "| rank | label | points | records | power | median frac | score | blocked | raw coh | aligned coh |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for i, c in enumerate(clusters_public[:30], 1):
        lines.append(
            f"| {i} | `{c['label']}` | {c['point_count']} | {c['record_count']} | "
            f"{c['total_component_power']:.3e} | {c['median_power_frac']:.3e} | "
            f"{c['median_score']:.3f} | {c['median_blocked']:.1f} | "
            f"{c['raw_phase_coherence']:.3f} | {c['path_aligned_phase_coherence']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 当前启发",
            "",
            "1. 如果 top 路径族能形成空间连续区域，那么我们可以把测试点先归到路径族，再做 component residual/scalar 校准。",
            "2. 如果 path-length 对齐后 coherence 仍低，说明单条几何路径长度还不足以解释该 component 的相位，可能需要路径族 residual phase 或更细的多反射候选。",
            "3. 之前简单 same-plane fallback 失败的原因可能是：平面相同还不够，必须连 `relative component offset` 和相位相干性一起约束。",
            "4. 下一步更合理的模型不是扩大所有 heavy residual，而是只对高相干、空间连续、测试点可覆盖的路径族增强。",
            "",
        ]
    )
    (out_dir / "analysis.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--assignments", default=str(PATHS.artifacts / "shared_features/component_path_assignments_calibrated_v1.npz"))
    parser.add_argument("--paths", default=str(PATHS.artifacts / "shared_features/path_candidates.npz"))
    parser.add_argument("--geometry-cache", default=str(PATHS.artifacts / "shared_features/height_geometry_cache.npz"))
    parser.add_argument("--out-dir", default=str(PATHS.analysis / "visualizations/path_families"))
    parser.add_argument("--top-per-point", type=int, default=8)
    parser.add_argument("--power-floor", type=float, default=1e-12)
    parser.add_argument("--min-points", type=int, default=8)
    parser.add_argument("--phase-beta", type=float, default=140.3)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    data_dir = out_dir / "data"
    ensure_dir(fig_dir)
    ensure_dir(data_dir)

    data = Path(args.data)
    train_pos = np.load(data / "Round1_Train_Pos.npy")
    test_pos = np.load(data / "Round1_Test_Pos.npy")
    geom = np.load(args.geometry_cache)
    train_blocked = geom["train_blocked"].astype(np.float64)
    assign = np.load(args.assignments)
    paths = np.load(args.paths)

    rows = make_records(assign, paths, train_pos, train_blocked, args.top_per_point, args.power_floor)
    phase_raw, phase_aligned, amp = compute_record_phases(rows, data / "Round1_Train_Channel.npy", args.phase_beta, args.batch)
    clusters = aggregate_clusters(rows, phase_raw, phase_aligned, args.min_points)

    figures = {
        "cluster_overview": plot_cluster_overview(fig_dir, clusters),
        "dominant_cluster_map": plot_dominant_cluster_map(fig_dir, rows, clusters, train_pos, test_pos),
        "top_cluster_maps": plot_top_cluster_maps(fig_dir, rows, clusters, paths, train_pos, test_pos),
        "offset_distributions": plot_offset_distributions(fig_dir, rows, clusters),
    }

    clusters_public = []
    for c in clusters:
        item = {k: v for k, v in c.items() if k not in ("record_ids", "point_ids", "key")}
        clusters_public.append(item)
    summary = {
        "assignment_cache": args.assignments,
        "path_candidates": args.paths,
        "top_per_point": int(args.top_per_point),
        "power_floor": float(args.power_floor),
        "min_points": int(args.min_points),
        "phase_beta": float(args.phase_beta),
        "record_count": int(len(rows)),
        "cluster_count": int(len(clusters)),
        "raw_record_phase_amplitude": percentile_dict(amp[amp > 0]) if np.any(amp > 0) else {},
        "top_clusters": clusters_public[:50],
    }
    (data_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_analysis(out_dir, figures, summary, clusters_public)
    print(json.dumps({"out_dir": str(out_dir), "analysis": str(out_dir / "analysis.md"), "figures": figures, "cluster_count": len(clusters)}, indent=2))


if __name__ == "__main__":
    main()
