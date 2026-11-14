#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


BS_POS = np.array([50.0, 0.0, 25.0], dtype=np.float64)


def qtiles(x, qs=(0, 10, 25, 50, 75, 90, 100)):
    x = np.asarray(x, dtype=np.float64)
    return {f"p{q}": float(np.percentile(x, q)) for q in qs}


def robust_standardize(train, other):
    med = np.median(train, axis=0)
    q25 = np.percentile(train, 25, axis=0)
    q75 = np.percentile(train, 75, axis=0)
    scale = np.maximum(q75 - q25, 1e-6)
    return (train - med[None, :]) / scale[None, :], (other - med[None, :]) / scale[None, :]


def greedy_unique_from_neighbors(neigh):
    used = set()
    out = []
    for row in neigh:
        chosen = None
        for idx in row:
            idx = int(idx)
            if idx not in used:
                chosen = idx
                break
        if chosen is None:
            continue
        used.add(chosen)
        out.append(chosen)
    return np.asarray(out, dtype=np.int64)


def make_feature_matrix(pos, map_features, blocked, knn_dist, selected_feature_indices):
    vec = pos - BS_POS[None, :]
    dist = np.linalg.norm(vec, axis=1)
    unit = vec / np.maximum(dist[:, None], 1e-12)
    az = np.arctan2(vec[:, 1], vec[:, 0])
    base = [
        pos[:, 0] / 300.0,
        pos[:, 1] / 300.0,
        dist / 300.0,
        unit[:, 0],
        unit[:, 1],
        unit[:, 2],
        np.sin(az),
        np.cos(az),
        knn_dist[:, 0] / 20.0,
        knn_dist[:, 3] / 20.0,
        knn_dist[:, 7] / 20.0,
        knn_dist[:, 15] / 20.0,
        knn_dist[:, 23] / 20.0,
        np.log1p(blocked) / np.log(256.0),
    ]
    blocks = [np.stack(base, axis=1).astype(np.float32)]
    if map_features is not None and selected_feature_indices:
        blocks.append(map_features[:, selected_feature_indices].astype(np.float32))
    return np.concatenate(blocks, axis=1).astype(np.float32)


def los_label(blocked):
    if blocked <= 0:
        return "clear"
    if blocked <= 25:
        return "light"
    if blocked <= 100:
        return "medium"
    return "heavy"


def density_label(nn1, q25, q75, q90):
    if nn1 <= q25:
        return "dense"
    if nn1 <= q75:
        return "normal"
    if nn1 <= q90:
        return "sparse"
    return "very_sparse"


def novelty_label(d, q50, q75, q90):
    if d <= q50:
        return "familiar"
    if d <= q75:
        return "normal"
    if d <= q90:
        return "novel"
    return "very_novel"


def boundary_label(neigh_blocked):
    labels = np.asarray([b <= 0 for b in neigh_blocked], dtype=np.float32)
    clear_frac = float(labels.mean())
    std = float(np.std(np.log1p(neigh_blocked)))
    if 0.2 <= clear_frac <= 0.8:
        return "los_boundary", clear_frac, std
    if std > 1.0:
        return "blockage_gradient", clear_frac, std
    return "stable", clear_frac, std


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def md_table(headers, rows):
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--map-features", default=str(PATHS.artifacts / "shared_features/map_features.npz"))
    parser.add_argument("--height-cache", default=str(PATHS.artifacts / "shared_features/height_geometry_cache.npz"))
    parser.add_argument("--out-dir", default=str(PATHS.artifacts / "test_regime_analysis"))
    parser.add_argument("--feature-indices", default="0,1,2,3,5,9,13,17,21,25,28,32,33,34,35,39,40,41")
    parser.add_argument("--clusters", type=int, default=12)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_pos = np.load(os.path.join(args.data, "Round1_Train_Pos.npy")).astype(np.float64)
    test_pos = np.load(os.path.join(args.data, "Round1_Test_Pos.npy")).astype(np.float64)
    z = np.load(args.map_features)
    train_map = z["train_features"].astype(np.float32)
    test_map = z["test_features"].astype(np.float32)
    names = [str(x) for x in z["feature_names"]]
    h = np.load(args.height_cache)
    train_blocked = h["train_blocked"].astype(np.float64)
    test_blocked = h["test_blocked"].astype(np.float64)
    feature_indices = [int(x) for x in args.feature_indices.split(",") if x.strip()]

    tree_xy = cKDTree(train_pos[:, :2])
    test_knn_dist, test_knn_idx = tree_xy.query(test_pos[:, :2], k=32, workers=-1)
    train_knn_dist, train_knn_idx_raw = tree_xy.query(train_pos[:, :2], k=33, workers=-1)
    train_knn_dist = train_knn_dist[:, 1:33]
    train_knn_idx = train_knn_idx_raw[:, 1:33]

    train_feat = make_feature_matrix(train_pos, train_map, train_blocked, train_knn_dist, feature_indices)
    test_feat = make_feature_matrix(test_pos, test_map, test_blocked, test_knn_dist, feature_indices)
    train_feat_z, test_feat_z = robust_standardize(train_feat, test_feat)

    feat_tree = cKDTree(train_feat_z)
    test_feat_dist, test_feat_idx = feat_tree.query(test_feat_z, k=32, workers=-1)
    train_feat_dist_raw, train_feat_idx_raw = feat_tree.query(train_feat_z, k=33, workers=-1)
    train_feat_dist = train_feat_dist_raw[:, 1:33]
    train_feat_idx = train_feat_idx_raw[:, 1:33]

    # KMeans is only used for diagnosis; fall back to scipy labels if sklearn is unavailable.
    try:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=args.clusters, random_state=20260705, n_init=20)
        train_cluster = km.fit_predict(train_feat_z)
        test_cluster = km.predict(test_feat_z)
    except Exception:
        train_cluster = np.zeros(len(train_pos), dtype=np.int64)
        test_cluster = np.zeros(len(test_pos), dtype=np.int64)

    nn_q = qtiles(train_knn_dist[:, 0])
    feat_q = qtiles(train_feat_dist[:, 0])
    rows = []
    for i in range(len(test_pos)):
        b_label = los_label(test_blocked[i])
        d_label = density_label(test_knn_dist[i, 0], nn_q["p25"], nn_q["p75"], nn_q["p90"])
        n_label = novelty_label(test_feat_dist[i, 0], feat_q["p50"], feat_q["p75"], feat_q["p90"])
        boundary, clear_frac, block_std = boundary_label(train_blocked[test_knn_idx[i, :16]])
        mf = {name: float(test_map[i, j]) for j, name in enumerate(names)}
        rows.append({
            "test_id": i,
            "x": float(test_pos[i, 0]),
            "y": float(test_pos[i, 1]),
            "z": float(test_pos[i, 2]),
            "bs_dist": float(np.linalg.norm(test_pos[i] - BS_POS)),
            "cluster": int(test_cluster[i]),
            "height_blocked": int(test_blocked[i]),
            "los_regime": b_label,
            "density_regime": d_label,
            "feature_novelty": n_label,
            "boundary_regime": boundary,
            "local_clear_frac": clear_frac,
            "local_block_std": block_std,
            "nn1_dist": float(test_knn_dist[i, 0]),
            "nn4_dist": float(test_knn_dist[i, 3]),
            "nn8_dist": float(test_knn_dist[i, 7]),
            "nn16_dist": float(test_knn_dist[i, 15]),
            "nn24_dist": float(test_knn_dist[i, 23]),
            "feature_nn1_dist": float(test_feat_dist[i, 0]),
            "feature_nn4_dist": float(test_feat_dist[i, 3]),
            "feature_nn8_dist": float(test_feat_dist[i, 7]),
            "nearest_train": int(test_knn_idx[i, 0]),
            "feature_nearest_train": int(test_feat_idx[i, 0]),
            "los_min_dist": mf.get("los_min_dist", np.nan),
            "los_frac_r0p5": mf.get("los_frac_r0p5", np.nan),
            "los_frac_r1p0": mf.get("los_frac_r1p0", np.nan),
            "los_frac_r2p0": mf.get("los_frac_r2p0", np.nan),
            "rx_wall_dist2": mf.get("rx_wall_dist2", np.nan),
        })

    fieldnames = list(rows[0].keys())
    write_csv(out_dir / "test_regime_table.csv", rows, fieldnames)

    feature_matched = greedy_unique_from_neighbors(test_feat_idx)
    spatial_matched = greedy_unique_from_neighbors(test_knn_idx)

    cluster_pick = []
    rng = np.random.default_rng(20260705)
    for c in sorted(set(test_cluster.tolist())):
        need = int(np.sum(test_cluster == c))
        pool = np.where(train_cluster == c)[0]
        if len(pool) == 0:
            continue
        if len(pool) >= need:
            chosen = rng.choice(pool, size=need, replace=False)
        else:
            chosen = pool
        cluster_pick.extend(chosen.tolist())
    cluster_matched = np.asarray(cluster_pick[:500], dtype=np.int64)

    stress_mask = np.array([
        (r["feature_novelty"] in {"novel", "very_novel"})
        or (r["density_regime"] in {"sparse", "very_sparse"})
        or (r["boundary_regime"] != "stable")
        for r in rows
    ])
    stress_train = greedy_unique_from_neighbors(test_feat_idx[stress_mask])
    clear_train = greedy_unique_from_neighbors(test_feat_idx[np.array([r["los_regime"] == "clear" for r in rows])])
    heavy_train = greedy_unique_from_neighbors(test_feat_idx[np.array([r["los_regime"] == "heavy" for r in rows])])

    np.savez_compressed(
        out_dir / "test_matched_val_sets.npz",
        feature_matched=feature_matched,
        spatial_matched=spatial_matched,
        cluster_matched=cluster_matched,
        stress_feature_boundary=stress_train,
        clear_los=clear_train,
        heavy_blocked=heavy_train,
    )

    summary = {
        "spatial_train_loo_nn1": qtiles(train_knn_dist[:, 0]),
        "spatial_test_to_train_nn1": qtiles(test_knn_dist[:, 0]),
        "feature_train_loo_nn1": qtiles(train_feat_dist[:, 0]),
        "feature_test_to_train_nn1": qtiles(test_feat_dist[:, 0]),
        "test_los_regime_counts": dict(Counter(r["los_regime"] for r in rows)),
        "test_density_counts": dict(Counter(r["density_regime"] for r in rows)),
        "test_feature_novelty_counts": dict(Counter(r["feature_novelty"] for r in rows)),
        "test_boundary_counts": dict(Counter(r["boundary_regime"] for r in rows)),
        "val_set_sizes": {
            "feature_matched": int(len(feature_matched)),
            "spatial_matched": int(len(spatial_matched)),
            "cluster_matched": int(len(cluster_matched)),
            "stress_feature_boundary": int(len(stress_train)),
            "clear_los": int(len(clear_train)),
            "heavy_blocked": int(len(heavy_train)),
        },
    }

    cluster_rows = []
    for c in range(args.clusters):
        tr = np.where(train_cluster == c)[0]
        te = np.where(test_cluster == c)[0]
        if len(tr) == 0 and len(te) == 0:
            continue
        cluster_rows.append({
            "cluster": c,
            "train_count": int(len(tr)),
            "test_count": int(len(te)),
            "test_train_ratio": float(len(te) / max(len(tr), 1)),
            "train_clear_frac": float(np.mean(train_blocked[tr] <= 0)) if len(tr) else 0.0,
            "test_clear_frac": float(np.mean(test_blocked[te] <= 0)) if len(te) else 0.0,
            "train_blocked_mean": float(np.mean(train_blocked[tr])) if len(tr) else 0.0,
            "test_blocked_mean": float(np.mean(test_blocked[te])) if len(te) else 0.0,
            "test_xy_mean": (
                [float(np.mean(test_pos[te, 0])), float(np.mean(test_pos[te, 1]))] if len(te) else [None, None]
            ),
        })
    summary["clusters"] = cluster_rows
    with open(out_dir / "test_regime_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    outlier_rows = sorted(rows, key=lambda r: (r["feature_nn1_dist"], r["nn1_dist"]), reverse=True)[:25]
    cluster_table = sorted(cluster_rows, key=lambda r: r["test_count"], reverse=True)
    md = []
    md.append("# Test Regime Analysis")
    md.append("")
    md.append("## Key Distributions")
    md.append("")
    md.append(md_table(
        ["metric", "train_loo_p50", "train_loo_p90", "test_p50", "test_p90", "test_max"],
        [
            [
                "spatial nn1",
                f"{summary['spatial_train_loo_nn1']['p50']:.3f}",
                f"{summary['spatial_train_loo_nn1']['p90']:.3f}",
                f"{summary['spatial_test_to_train_nn1']['p50']:.3f}",
                f"{summary['spatial_test_to_train_nn1']['p90']:.3f}",
                f"{summary['spatial_test_to_train_nn1']['p100']:.3f}",
            ],
            [
                "feature nn1",
                f"{summary['feature_train_loo_nn1']['p50']:.3f}",
                f"{summary['feature_train_loo_nn1']['p90']:.3f}",
                f"{summary['feature_test_to_train_nn1']['p50']:.3f}",
                f"{summary['feature_test_to_train_nn1']['p90']:.3f}",
                f"{summary['feature_test_to_train_nn1']['p100']:.3f}",
            ],
        ],
    ))
    md.append("")
    md.append("## Test Regime Counts")
    md.append("")
    for key in ["test_los_regime_counts", "test_density_counts", "test_feature_novelty_counts", "test_boundary_counts", "val_set_sizes"]:
        md.append(f"### {key}")
        md.append("")
        md.append(md_table(["name", "count"], sorted(summary[key].items())))
        md.append("")
    md.append("## Largest Test Clusters")
    md.append("")
    md.append(md_table(
        ["cluster", "train", "test", "ratio", "test_clear", "test_blocked_mean", "test_xy_mean"],
        [
            [
                r["cluster"],
                r["train_count"],
                r["test_count"],
                f"{r['test_train_ratio']:.3f}",
                f"{r['test_clear_frac']:.3f}",
                f"{r['test_blocked_mean']:.1f}",
                r["test_xy_mean"],
            ]
            for r in cluster_table[:12]
        ],
    ))
    md.append("")
    md.append("## Top Feature-Outlier Test Points")
    md.append("")
    md.append(md_table(
        ["id", "x", "y", "cluster", "los", "density", "novelty", "boundary", "nn1", "feat_nn1", "blocked"],
        [
            [
                r["test_id"],
                f"{r['x']:.1f}",
                f"{r['y']:.1f}",
                r["cluster"],
                r["los_regime"],
                r["density_regime"],
                r["feature_novelty"],
                r["boundary_regime"],
                f"{r['nn1_dist']:.2f}",
                f"{r['feature_nn1_dist']:.2f}",
                r["height_blocked"],
            ]
            for r in outlier_rows
        ],
    ))
    md.append("")
    md.append("## Suggested Use")
    md.append("")
    md.append("- Use `feature_matched` as a geometry-aware validation set.")
    md.append("- Use `stress_feature_boundary` to test whether a method breaks on sparse/novel/boundary regimes.")
    md.append("- Inspect methods by `los_regime`, `density_regime`, and `feature_novelty` instead of only aggregate C.")
    md.append("")
    (out_dir / "test_regime_report.md").write_text("\n".join(md))

    print(json.dumps({
        "out_dir": str(out_dir),
        "summary": str(out_dir / "test_regime_summary.json"),
        "table": str(out_dir / "test_regime_table.csv"),
        "val_sets": str(out_dir / "test_matched_val_sets.npz"),
        "report": str(out_dir / "test_regime_report.md"),
        "counts": {
            "los": summary["test_los_regime_counts"],
            "density": summary["test_density_counts"],
            "novelty": summary["test_feature_novelty_counts"],
            "boundary": summary["test_boundary_counts"],
        },
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
