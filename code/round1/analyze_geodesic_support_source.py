#!/usr/bin/env python3
"""Diagnose geodesic/corridor features as a new support information source.

This script does not create a submission.  It asks whether the official
point-cloud occupancy grid contains propagation-mode information not already
captured by the current template/family stack:

1. Build a 2D obstacle grid from vertical point-cloud cells.
2. Run a shortest-path distance transform from the BS projection.
3. Extract path features for train/test points:
   geodesic distance, excess over direct distance, turn count, first/last
   path directions, and clearance bottleneck along the path.
4. On train points, compare v125-equivalent support to truth support and test
   whether geodesic features help retrieve bad points with similar residual
   signatures.
"""

from __future__ import annotations

import json
import math
import heapq
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy import sparse
from scipy.sparse import csgraph

from project_paths import PATHS
from powerfield_common import ASSETS, TINY, regime_of


OUT_NPZ = ASSETS + "/geodesic_support_features.npz"
OUT_JSON = ASSETS + "/geodesic_support_source_analysis.json"
OUT_MD = ASSETS + "/geodesic_support_source_analysis.md"

REGIMES = ("clear", "light", "medium", "heavy")


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = a.reshape(a.shape[0], -1).astype(np.float64)
    bb = b.reshape(b.shape[0], -1).astype(np.float64)
    num = np.sum(aa * bb, axis=1)
    den = np.sqrt(np.sum(aa * aa, axis=1) * np.sum(bb * bb, axis=1)) + 1e-30
    return (num / den).astype(np.float32)


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train, dtype=np.float32)
    test = np.asarray(test, dtype=np.float32)
    med = np.nanmedian(train, axis=0, keepdims=True)
    q25 = np.nanpercentile(train, 25, axis=0, keepdims=True)
    q75 = np.nanpercentile(train, 75, axis=0, keepdims=True)
    scale = np.maximum(q75 - q25, 1e-6)
    return (
        np.nan_to_num((train - med) / scale, nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32),
        np.nan_to_num((test - med) / scale, nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32),
    )


def weights_from_dist(dist: np.ndarray, power: float = 2.0) -> np.ndarray:
    w = 1.0 / np.maximum(dist, 1e-4) ** float(power)
    w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-30)
    return w.astype(np.float32)


def build_bad_mask(
    train_base: np.ndarray,
    truth: np.ndarray,
    regimes: np.ndarray,
    bad_quantile: float,
    min_bad_cos: float,
) -> tuple[np.ndarray, np.ndarray]:
    cos = row_cos(train_base, truth).astype(np.float32)
    badness = (1.0 - cos).astype(np.float32)
    threshold = max(float(np.quantile(badness, bad_quantile)), 1.0 - float(min_bad_cos))
    mask = badness >= threshold
    mask |= ((regimes == "medium") | (regimes == "heavy")) & (badness >= 0.55 * threshold)
    if int(mask.sum()) < 256:
        ids = np.argsort(badness)[-256:]
        mask[ids] = True
    return mask.astype(bool), cos


def summarize(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(np.nanmean(x)),
        "std": float(np.nanstd(x)),
        "p10": float(np.nanpercentile(x, 10)),
        "p50": float(np.nanpercentile(x, 50)),
        "p90": float(np.nanpercentile(x, 90)),
        "min": float(np.nanmin(x)),
        "max": float(np.nanmax(x)),
    }


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def corr(x: np.ndarray, y: np.ndarray, spearman: bool = False) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return 0.0
    x = x[mask]
    y = y[mask]
    if spearman:
        x = rankdata(x)
        y = rankdata(y)
    x = x - x.mean()
    y = y - y.mean()
    den = math.sqrt(float(np.dot(x, x) * np.dot(y, y))) + 1e-12
    return float(np.dot(x, y) / den)


def nearest_free_indices(free: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # For every cell, return distance and nearest free-cell row/col.
    dist, inds = ndimage.distance_transform_edt(~free, return_indices=True)
    return dist.astype(np.float32), inds[0].astype(np.int32), inds[1].astype(np.int32)


def dijkstra_grid(
    free: np.ndarray,
    source_rc: tuple[int, int],
    cost_grid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = free.shape
    if cost_grid is not None:
        free = np.ones_like(free, dtype=bool)
        sid = np.arange(h * w, dtype=np.int32).reshape(h, w)
        n = h * w
        source = int(sid[source_rc])
        rows = []
        cols = []
        data = []
        offsets = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        ]
        grid_ids = sid
        for dr, dc, step in offsets:
            r0a = max(0, -dr)
            r0b = min(h, h - dr)
            c0a = max(0, -dc)
            c0b = min(w, w - dc)
            src = grid_ids[r0a:r0b, c0a:c0b].ravel()
            dst = grid_ids[r0a + dr : r0b + dr, c0a + dc : c0b + dc].ravel()
            c_src = cost_grid[r0a:r0b, c0a:c0b].ravel()
            c_dst = cost_grid[r0a + dr : r0b + dr, c0a + dc : c0b + dc].ravel()
            rows.append(src)
            cols.append(dst)
            data.append((step * 0.5 * (c_src + c_dst)).astype(np.float32))
        row = np.concatenate(rows)
        col = np.concatenate(cols)
        dat = np.concatenate(data)
        graph = sparse.csr_matrix((dat, (row, col)), shape=(n, n), dtype=np.float32)
        dist, pred = csgraph.dijkstra(
            graph,
            directed=True,
            indices=source,
            return_predecessors=True,
            unweighted=False,
        )
        return sid, dist.astype(np.float32), pred.astype(np.int32)
    sid = np.full((h, w), -1, dtype=np.int32)
    rr, cc = np.nonzero(free)
    sid[rr, cc] = np.arange(len(rr), dtype=np.int32)
    n = len(rr)
    source = int(sid[source_rc])
    if source < 0:
        raise RuntimeError("source cell is not free")

    dist = np.full(n, np.inf, dtype=np.float32)
    pred = np.full(n, -1, dtype=np.int32)
    dist[source] = 0.0
    heap: list[tuple[float, int]] = [(0.0, source)]
    offsets = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    ]
    while heap:
        d, u = heapq.heappop(heap)
        if d > float(dist[u]) + 1e-6:
            continue
        r = int(rr[u])
        c = int(cc[u])
        for dr, dc, cost in offsets:
            nr = r + dr
            nc = c + dc
            if nr < 0 or nr >= h or nc < 0 or nc >= w:
                continue
            v = int(sid[nr, nc])
            if v < 0:
                continue
            if cost_grid is None:
                step_cost = cost
            else:
                step_cost = cost * 0.5 * (float(cost_grid[r, c]) + float(cost_grid[nr, nc]))
            nd = d + step_cost
            if nd < float(dist[v]):
                dist[v] = nd
                pred[v] = u
                heapq.heappush(heap, (nd, v))
    return sid, dist, pred


def trace_path(node: int, pred: np.ndarray, rr: np.ndarray, cc: np.ndarray, max_steps: int = 4096) -> np.ndarray:
    if node < 0:
        return np.empty((0, 2), dtype=np.int32)
    out = []
    cur = int(node)
    for _ in range(max_steps):
        out.append((int(rr[cur]), int(cc[cur])))
        nxt = int(pred[cur])
        if nxt < 0:
            break
        cur = nxt
    out.reverse()
    return np.asarray(out, dtype=np.int32)


def build_features_for_points(
    pos: np.ndarray,
    cache: np.lib.npyio.NpzFile,
    free: np.ndarray,
    sid: np.ndarray,
    dist: np.ndarray,
    pred: np.ndarray,
    near_r: np.ndarray,
    near_c: np.ndarray,
    clearance: np.ndarray,
    name: str,
) -> tuple[np.ndarray, list[str]]:
    xmin = float(cache["x_min"])
    ymin = float(cache["y_min"])
    res = float(cache["resolution"])
    h, w = free.shape
    rr_nodes, cc_nodes = np.nonzero(free)

    bs_xy = np.array([50.0, 0.0], dtype=np.float32)
    names = [
        "geo_dist",
        "direct_dist",
        "geo_excess",
        "geo_ratio",
        "snap_dist",
        "turn_count",
        "turn_rate",
        "first_dir_sin",
        "first_dir_cos",
        "last_dir_sin",
        "last_dir_cos",
        "bottleneck_clearance",
        "mean_clearance",
        "path_len_cells",
        "unreachable",
    ]
    feat = np.zeros((len(pos), len(names)), dtype=np.float32)
    for i, p in enumerate(pos):
        col = int(round((float(p[0]) - xmin) / res))
        row = int(round((float(p[1]) - ymin) / res))
        row = int(np.clip(row, 0, h - 1))
        col = int(np.clip(col, 0, w - 1))
        fr = int(near_r[row, col])
        fc = int(near_c[row, col])
        snap = float(math.hypot(fr - row, fc - col) * res)
        node = int(sid[fr, fc])
        gd = float(dist[node]) * res if node >= 0 else math.inf
        direct = float(np.linalg.norm(p[:2].astype(np.float32) - bs_xy))
        unreachable = 0.0 if np.isfinite(gd) else 1.0
        path = trace_path(node, pred, rr_nodes, cc_nodes) if np.isfinite(gd) else np.empty((0, 2), dtype=np.int32)
        if len(path) >= 2:
            step = np.diff(path.astype(np.float32), axis=0)
            # Convert row/col to x/y direction: row is y, col is x.
            dirs = np.stack([step[:, 1], step[:, 0]], axis=1)
            ang = np.arctan2(dirs[:, 1], dirs[:, 0])
            d_ang = np.abs(np.angle(np.exp(1j * np.diff(ang)))) if len(ang) > 1 else np.array([], dtype=np.float32)
            turns = float(np.sum(d_ang > np.deg2rad(35.0)))
            first = float(ang[0])
            last = float(ang[-1])
            clear_vals = clearance[path[:, 0], path[:, 1]].astype(np.float32) * res
            bottleneck = float(np.min(clear_vals))
            mean_clear = float(np.mean(clear_vals))
        else:
            turns = 0.0
            first = 0.0
            last = 0.0
            bottleneck = 0.0
            mean_clear = 0.0
        feat[i] = np.array(
            [
                gd if np.isfinite(gd) else 1e6,
                direct,
                (gd - direct) if np.isfinite(gd) else 1e6,
                (gd / max(direct, 1e-6)) if np.isfinite(gd) else 1e6,
                snap,
                turns,
                turns / max(float(len(path)), 1.0),
                math.sin(first),
                math.cos(first),
                math.sin(last),
                math.cos(last),
                bottleneck,
                mean_clear,
                float(len(path)),
                unreachable,
            ],
            dtype=np.float32,
        )
        if (i + 1) % 500 == 0:
            print(f"{name} geodesic features {i+1}/{len(pos)}", flush=True)
    return feat, names


def residual_signature(truth: np.ndarray, base: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    lr = np.log(np.maximum(truth, floor)) - np.log(np.maximum(base, floor))
    pos = np.maximum(lr, 0.0).astype(np.float32)
    neg = np.maximum(-lr, 0.0).astype(np.float32)
    parts = [
        pos.sum(axis=3).reshape(pos.shape[0], -1),
        pos.sum(axis=(1, 2)),
        pos.sum(axis=(2, 3)),
        pos.sum(axis=(1, 3)),
        neg.sum(axis=3).reshape(neg.shape[0], -1),
        neg.sum(axis=(1, 2)),
    ]
    sig = np.concatenate(parts, axis=1).astype(np.float32)
    sig /= np.maximum(np.linalg.norm(sig, axis=1, keepdims=True), TINY)
    return sig


def knn_residual_similarity(feat: np.ndarray, sig: np.ndarray, ids: np.ndarray, k: int = 16) -> dict[str, float]:
    x = feat[ids].astype(np.float32)
    s = sig[ids].astype(np.float32)
    n = len(ids)
    sims = []
    for start in range(0, n, 256):
        end = min(start + 256, n)
        d2 = ((x[start:end, None, :] - x[None, :, :]) ** 2).sum(axis=2)
        rows = np.arange(end - start)
        d2[rows, np.arange(start, end)] = np.inf
        nn = np.argpartition(d2, kth=min(k, n - 1), axis=1)[:, :k]
        dist = np.take_along_axis(d2, nn, axis=1)
        w = weights_from_dist(np.sqrt(np.maximum(dist, 0.0)), power=2.0)
        sim = (w * (s[start:end] @ s[nn].transpose(0, 2, 1)).diagonal(axis1=1, axis2=2)).sum(axis=1)
        sims.append(sim.astype(np.float32))
    out = np.concatenate(sims)
    return summarize(out)


def main() -> None:
    cache = np.load(ASSETS + "/official_pointcloud_scene_cache.npz")
    pos_train = np.load(str(PATHS.round1 / "Round1_Train_Pos.npy")).astype(np.float32)
    pos_test = np.load(str(PATHS.round1 / "Round1_Test_Pos.npy")).astype(np.float32)
    blocked_train = np.load(ASSETS + "/height_geometry_cache.npz")["train_blocked"].astype(np.float32)
    blocked_test = np.load(ASSETS + "/height_geometry_cache.npz")["test_blocked"].astype(np.float32)
    regimes = regime_of(blocked_train)

    occ = (cache["grid_vertical_count"] > 0) & (cache["grid_z_max"] > 2.0)
    raw_free = ~occ
    free = np.ones_like(raw_free, dtype=bool)
    xmin = float(cache["x_min"])
    ymin = float(cache["y_min"])
    res = float(cache["resolution"])
    bs_col = int(round((50.0 - xmin) / res))
    bs_row = int(round((0.0 - ymin) / res))
    free[bs_row, bs_col] = True

    print("soft propagation cost field", flush=True)
    vertical = cache["grid_vertical_count"].astype(np.float32)
    zmax = np.nan_to_num(cache["grid_z_max"].astype(np.float32), nan=0.0)
    wall_cost = np.clip(vertical / 8.0, 0.0, 5.0)
    height_cost = np.clip((zmax - 2.0) / 16.0, 0.0, 3.0)
    cost_grid = (1.0 + 2.5 * occ.astype(np.float32) + 0.45 * wall_cost + 0.35 * height_cost).astype(np.float32)
    cost_grid[bs_row, bs_col] = 1.0

    yy, xx = np.indices(free.shape)
    snap_dist = np.zeros_like(cost_grid, dtype=np.float32)
    near_r = yy.astype(np.int32)
    near_c = xx.astype(np.int32)
    print("clearance", flush=True)
    clearance = ndimage.distance_transform_edt(raw_free).astype(np.float32)
    print("dijkstra grid", flush=True)
    sid, dist, pred = dijkstra_grid(free, (bs_row, bs_col), cost_grid=cost_grid)

    print("point features", flush=True)
    train_geo, geo_names = build_features_for_points(
        pos_train, cache, free, sid, dist, pred, near_r, near_c, clearance, "train"
    )
    test_geo, _ = build_features_for_points(
        pos_test, cache, free, sid, dist, pred, near_r, near_c, clearance, "test"
    )
    np.savez_compressed(OUT_NPZ, train_features=train_geo, test_features=test_geo, feature_names=np.asarray(geo_names))

    print("support diagnostics", flush=True)
    truth = np.load(ASSETS + "/truth_coll_power.npy", mmap_mode="r").astype(np.float32)
    base = np.load(ASSETS + "/cache_v128/train_v125eq_collapsed_support.npy", mmap_mode="r").astype(np.float32)
    truth = truth / np.maximum(truth.sum(axis=(1, 2, 3), keepdims=True), TINY)
    base = base / np.maximum(base.sum(axis=(1, 2, 3), keepdims=True), TINY)
    base_cos = row_cos(base, truth).astype(np.float32)
    badness = (1.0 - base_cos).astype(np.float32)
    bad_mask, _ = build_bad_mask(base, truth, regimes, 0.68, 0.62)
    bad_ids = np.flatnonzero(bad_mask)
    sig = residual_signature(truth, base)

    scene_train = cache["train_features"].astype(np.float32)
    tmpl = np.load(ASSETS + "/train_v125eq_crossfit_template_probs.npy").astype(np.float32)
    tmpl_entropy = -(tmpl * np.log(np.maximum(tmpl, 1e-12))).sum(axis=1) / math.log(tmpl.shape[1])
    tmpl_top = tmpl.max(axis=1)

    # Feature-correlation scan.
    corr_rows = []
    extra = np.column_stack([tmpl_entropy, tmpl_top, blocked_train]).astype(np.float32)
    extra_names = ["template_entropy", "template_top_prob", "blocked_count"]
    all_diag = np.concatenate([train_geo, extra], axis=1)
    all_names = geo_names + extra_names
    for j, name in enumerate(all_names):
        corr_rows.append(
            {
                "feature": name,
                "pearson_badness": corr(all_diag[:, j], badness, False),
                "spearman_badness": corr(all_diag[:, j], badness, True),
                "bad_mean": float(np.mean(all_diag[bad_mask, j])),
                "good_mean": float(np.mean(all_diag[~bad_mask, j])),
            }
        )
    corr_rows.sort(key=lambda r: abs(r["spearman_badness"]), reverse=True)

    # Does geodesic help retrieve same residual shape among bad points?
    scene_std, _ = standardize(scene_train, scene_train)
    tmpl_std, _ = standardize(tmpl, tmpl)
    geo_std, _ = standardize(train_geo, train_geo)
    base_feat = np.concatenate([0.9 * scene_std, 0.45 * tmpl_std], axis=1)
    geo_feat = np.concatenate([base_feat, 0.85 * geo_std], axis=1)
    geo_only = geo_std
    knn_report = {
        "bad_count": int(len(bad_ids)),
        "scene_template": knn_residual_similarity(base_feat, sig, bad_ids, 16),
        "scene_template_plus_geodesic": knn_residual_similarity(geo_feat, sig, bad_ids, 16),
        "geodesic_only": knn_residual_similarity(geo_only, sig, bad_ids, 16),
    }

    # Regime summaries.
    regime_report = {}
    for r in REGIMES:
        m = regimes == r
        regime_report[r] = {
            "count": int(m.sum()),
            "bad_fraction": float(bad_mask[m].mean()) if m.any() else 0.0,
            "support_cos": summarize(base_cos[m]) if m.any() else {},
            "geo_ratio": summarize(train_geo[m, geo_names.index("geo_ratio")]) if m.any() else {},
            "turn_count": summarize(train_geo[m, geo_names.index("turn_count")]) if m.any() else {},
            "bottleneck_clearance": summarize(train_geo[m, geo_names.index("bottleneck_clearance")]) if m.any() else {},
        }

    report = {
        "method": "geodesic_support_source_analysis",
        "occupancy": {
            "grid_shape": list(map(int, free.shape)),
            "occupied_fraction": float(occ.mean()),
            "free_nodes": int(raw_free.sum()),
            "resolution": float(res),
            "cost_mean": float(cost_grid.mean()),
            "cost_p90": float(np.percentile(cost_grid, 90)),
        },
        "train": {
            "support_cos": summarize(base_cos),
            "badness": summarize(badness),
            "bad_count": int(len(bad_ids)),
            "bad_fraction": float(bad_mask.mean()),
        },
        "top_feature_correlations": corr_rows[:16],
        "knn_residual_similarity": knn_report,
        "by_regime": regime_report,
        "outputs": {
            "features": OUT_NPZ,
            "json": OUT_JSON,
            "md": OUT_MD,
        },
    }
    Path(OUT_JSON).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Geodesic Support Source Analysis",
        "",
        "Purpose: test whether point-cloud corridor/geodesic topology provides new support/template information.",
        "",
        f"- occupied fraction: `{report['occupancy']['occupied_fraction']:.4f}`",
        f"- bad train fraction: `{report['train']['bad_fraction']:.4f}`",
        "",
        "## Top Feature Correlations",
        "",
        "| feature | spearman_badness | bad_mean | good_mean |",
        "| --- | ---: | ---: | ---: |",
    ]
    for r in corr_rows[:12]:
        lines.append(
            f"| `{r['feature']}` | `{r['spearman_badness']:.4f}` | `{r['bad_mean']:.4f}` | `{r['good_mean']:.4f}` |"
        )
    lines += [
        "",
        "## Bad-Point Residual KNN Similarity",
        "",
        "| feature set | mean | p50 | p10 | p90 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for k, v in knn_report.items():
        if not isinstance(v, dict):
            continue
        lines.append(f"| `{k}` | `{v['mean']:.4f}` | `{v['p50']:.4f}` | `{v['p10']:.4f}` | `{v['p90']:.4f}` |")
    lines += ["", "## Regime Summary", "", "| regime | count | bad_frac | support_cos_mean | geo_ratio_p50 | turns_p50 | clearance_p10 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r, v in regime_report.items():
        lines.append(
            f"| `{r}` | `{v['count']}` | `{v['bad_fraction']:.4f}` | `{v['support_cos']['mean']:.4f}` | "
            f"`{v['geo_ratio']['p50']:.4f}` | `{v['turn_count']['p50']:.1f}` | `{v['bottleneck_clearance']['p10']:.2f}` |"
        )
    Path(OUT_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report["knn_residual_similarity"], indent=2), flush=True)
    print(f"wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
