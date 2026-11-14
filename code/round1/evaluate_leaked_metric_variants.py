#!/usr/bin/env python3
"""Evaluate local metric variants using the leaked platform PAS reshape.

The platform traceback exposed this intermediate shape in ``cal_pas``:

    (B, 2, 192, 16, 4, 8)

followed by:

    np.fft.fft2(..., axes=(2, 3))

This script computes PAS variants around that exact reshape/FFT so we can
compare local ordering against known online scores.  It is a metric diagnostic;
it does not generate a submission.
"""

from __future__ import annotations
from project_paths import PATHS

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from evaluate_derotated_loo_grid import choose_val_indices
from evaluate_local_ad_paths import BS_POS, idw_weights
from evaluate_reflection_steering import phase_from_units, reflection_lengths_and_units
from evaluate_s1style_geom_blend import geometry_cache
from generate_geometric_steering_blend import steering_phase, unit_vectors
from spectrum_projection import cosine_last, exact_pdp_scores, exact_pas_scores


MP = 2
MH = 16
MV = 8
N_ANT = 4
S_SC = 192


ONLINE = {
    "derot_k16_p2": 0.450782,
    "derot_k4_p3": 0.437548,
    "blend_k24_a05": 0.454453,
    "blend_k8_a05": 0.451313,
    "map_reflect_k24_r025": 0.450868,
}


def make_split(pos, test_pos, mode, val_count, seed, k):
    if mode == "random":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(pos))
        val_idx = np.sort(perm[:val_count])
        train_idx = np.sort(perm[val_count:])
        tree = cKDTree(pos[train_idx, :2])
        distances, local = tree.query(pos[val_idx, :2], k=k, workers=-1)
        if k == 1:
            distances = distances[:, None]
            local = local[:, None]
        return val_idx, distances, train_idx[local]
    val_idx = choose_val_indices(pos[:, :2], test_pos[:, :2], val_count, mode, seed)
    tree = cKDTree(pos[:, :2])
    distances_all, local_all = tree.query(pos[val_idx, :2], k=k + 1, workers=-1)
    neighbor_idx = np.empty((len(val_idx), k), dtype=np.int64)
    distances = np.empty((len(val_idx), k), dtype=np.float64)
    for i, vi in enumerate(val_idx):
        keep = local_all[i] != vi
        neighbor_idx[i] = local_all[i, keep][:k]
        distances[i] = distances_all[i, keep][:k]
    return val_idx, distances, neighbor_idx


def full_predict(neigh_h, weights, radial, phase=None):
    coeff = weights.astype(np.float32) * radial.astype(np.complex64)
    if phase is None:
        return np.sum(coeff[:, None, None, None] * neigh_h, axis=0, keepdims=True).astype(np.complex64)
    return np.sum(coeff[:, None, None, None] * phase[:, :, None, None] * neigh_h, axis=0, keepdims=True).astype(np.complex64)


def leaked_pas_power(h: np.ndarray) -> np.ndarray:
    """Return power after the leaked platform reshape and fft2.

    Input shape: B,M,N,S = B,256,4,192.
    Intermediate shape matches traceback: B,2,192,16,4,8.
    """

    arr = h.reshape(h.shape[0], MP, MH, MV, N_ANT, S_SC)
    arr = arr.transpose(0, 1, 5, 2, 4, 3)  # B,2,192,16,4,8
    spec = np.fft.fft2(arr, axes=(2, 3))
    return np.abs(spec) ** 2


def cosine_vectors(pred_p: np.ndarray, true_p: np.ndarray, vector_axes: tuple[int, ...]) -> np.ndarray:
    """Cosine over vector_axes, average remaining axes per point."""

    axes = tuple(int(a) for a in vector_axes)
    keep_axes = [a for a in range(pred_p.ndim) if a not in axes]
    # Keep batch first, move vector axes to the end.
    out_order = keep_axes + list(axes)
    p = np.transpose(pred_p, out_order)
    t = np.transpose(true_p, out_order)
    batch = pred_p.shape[0]
    vector_len = int(np.prod([pred_p.shape[a] for a in axes]))
    p = p.reshape(batch, -1, vector_len)
    t = t.reshape(batch, -1, vector_len)
    scores = cosine_last(p.reshape(-1, vector_len), t.reshape(-1, vector_len))
    return scores.reshape(batch, -1).mean(axis=1)


def leaked_pas_scores_from_power(pred_p: np.ndarray, truth_p: np.ndarray, mode: str) -> np.ndarray:
    if mode == "leaked_all":
        return cosine_vectors(pred_p, truth_p, (1, 2, 3, 4, 5))
    if mode == "leaked_sh_per_pol_rx_v":
        return cosine_vectors(pred_p, truth_p, (2, 3))
    if mode == "leaked_shv_per_pol_rx":
        return cosine_vectors(pred_p, truth_p, (2, 3, 5))
    if mode == "leaked_sum_pol_shv_per_rx":
        return cosine_vectors(pred_p.sum(axis=1), truth_p.sum(axis=1), (1, 2, 4))
    if mode == "leaked_sum_rx_shv_per_pol":
        return cosine_vectors(pred_p.sum(axis=4), truth_p.sum(axis=4), (2, 3, 4))
    if mode == "leaked_sum_polrx_shv":
        return cosine_vectors(pred_p.sum(axis=(1, 4)), truth_p.sum(axis=(1, 4)), (1, 2, 3))
    raise ValueError(mode)


def leaked_pas_scores(pred: np.ndarray, truth: np.ndarray, mode: str) -> np.ndarray:
    pred_p = leaked_pas_power(pred)
    truth_p = leaked_pas_power(truth)
    return leaked_pas_scores_from_power(pred_p, truth_p, mode)


class MultiMetric:
    def __init__(self, pas_modes):
        self.pas_modes = list(pas_modes)
        self.num = 0.0
        self.den = 0.0
        self.pdp = []
        self.pas = {m: [] for m in self.pas_modes}

    def add(self, pred, truth):
        self.num += float(np.sum(np.abs(pred - truth) ** 2, dtype=np.float64))
        self.den += float(np.sum(np.abs(truth) ** 2, dtype=np.float64))
        self.pdp.append(exact_pdp_scores(pred, truth))
        leaked_pred_p = None
        leaked_truth_p = None
        for mode in self.pas_modes:
            if mode.startswith("old_"):
                old_mode = mode.replace("old_", "")
                self.pas[mode].append(exact_pas_scores(pred, truth, old_mode))
            else:
                if leaked_pred_p is None:
                    leaked_pred_p = leaked_pas_power(pred)
                    leaked_truth_p = leaked_pas_power(truth)
                self.pas[mode].append(leaked_pas_scores_from_power(leaked_pred_p, leaked_truth_p, mode))

    def result(self):
        nmse = self.num / max(self.den, 1e-30)
        pdp = float(np.mean(np.concatenate(self.pdp)))
        inv_nmse = float(1.0 / (1.0 + nmse))
        out = {"nmse": float(nmse), "pdp": pdp, "inv_nmse": inv_nmse}
        for mode in self.pas_modes:
            pas = float(np.mean(np.concatenate(self.pas[mode])))
            out[f"pas_{mode}"] = pas
            out[f"score_{mode}"] = 0.4 * pas + 0.4 * pdp + 0.2 * inv_nmse
        return out


def rank_corr(rows, score_key):
    known = [r for r in rows if r["online"] is not None]
    if len(known) < 3:
        return {"pearson": None, "spearman": None}
    x = np.asarray([r[score_key] for r in known], dtype=np.float64)
    y = np.asarray([r["online"] for r in known], dtype=np.float64)
    pearson = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else None
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    spearman = float(np.corrcoef(rx, ry)[0, 1]) if np.std(rx) > 0 and np.std(ry) > 0 else None
    return {"pearson": pearson, "spearman": spearman}


def evaluate(args):
    data = Path(args.data)
    pos = np.load(data / "Round1_Train_Pos.npy")
    test_pos = np.load(data / "Round1_Test_Pos.npy")
    h = np.load(data / "Round1_Train_Channel.npy", mmap_mode="r")
    train_blocked, _test_blocked = geometry_cache(args.data, args.cache, False)
    val_idx, distances, neighbor_idx = make_split(pos, test_pos, args.mode, args.val_count, args.seed, args.max_k)
    dist_all = np.linalg.norm(pos - BS_POS[None, :], axis=1)
    u_all = unit_vectors(pos)

    map_data = np.load(args.map_features, allow_pickle=True)
    features = map_data["train_features"][val_idx]
    names = [str(x) for x in map_data["feature_names"]]
    clear = features[:, names.index("los_min_dist")] >= 0.5
    plane_points = features[:, [names.index("rx_wall_x2"), names.index("rx_wall_y2"), names.index("rx_wall_z2")]]
    plane_normals = features[:, [names.index("rx_wall_nx2"), names.index("rx_wall_ny2"), names.index("rx_wall_nz2")]]

    pas_modes = [
        "old_sumpol",
        "old_fullpol",
        "leaked_all",
        "leaked_sh_per_pol_rx_v",
        "leaked_shv_per_pol_rx",
        "leaked_sum_pol_shv_per_rx",
        "leaked_sum_rx_shv_per_pol",
        "leaked_sum_polrx_shv",
    ]
    candidates = [
        "derot_k16_p2",
        "derot_k4_p3",
        "blend_k24_a05",
        "blend_k8_a05",
        "map_reflect_k24_r025",
    ]
    metrics = {name: MultiMetric(pas_modes) for name in candidates}

    for ii, vi in enumerate(val_idx):
        truth = np.asarray(h[vi:vi + 1])
        pred_cache = {}
        for k, power in [(24, 2.0), (16, 2.0), (8, 2.0), (4, 3.0)]:
            idx = neighbor_idx[ii, :k]
            neigh = np.asarray(h[idx])
            weights = idw_weights(distances[ii, :k], power)
            radial = np.exp(1j * args.beta * (dist_all[idx] - dist_all[vi])).astype(np.complex64)
            pred_cache[(k, power, "derot")] = full_predict(neigh, weights, radial)
            phase = steering_phase(u_all, idx, vi, args.steering_scale)
            steer = full_predict(neigh, weights, radial, phase)
            pred_cache[(k, power, "steer")] = steer
            pred_cache[(k, power, "blend05")] = (0.5 * pred_cache[(k, power, "derot")] + 0.5 * steer).astype(np.complex64)

        direct0 = pred_cache[(24, 2.0, "derot")]
        direct1 = pred_cache[(24, 2.0, "steer")]
        direct025 = (0.75 * direct0 + 0.25 * direct1).astype(np.complex64)
        idx = neighbor_idx[ii, :24]
        neigh = np.asarray(h[idx])
        weights = idw_weights(distances[ii, :24], 2.0)
        all_points = np.concatenate([pos[idx], pos[vi:vi + 1]], axis=0)
        lengths, units, _ = reflection_lengths_and_units(all_points, plane_points[ii], plane_normals[ii])
        radial_ref = np.exp(1j * args.beta * (lengths[:-1] - lengths[-1])).astype(np.complex64)
        ref_phase = phase_from_units(units[:-1], units[-1], args.steering_scale)
        reflect = full_predict(neigh, weights, radial_ref, ref_phase)
        map_reflect = direct1 if clear[ii] else (0.75 * direct025 + 0.25 * reflect).astype(np.complex64)

        preds = {
            "derot_k16_p2": pred_cache[(16, 2.0, "derot")],
            "derot_k4_p3": pred_cache[(4, 3.0, "derot")],
            "blend_k24_a05": pred_cache[(24, 2.0, "blend05")],
            "blend_k8_a05": pred_cache[(8, 2.0, "blend05")],
            "map_reflect_k24_r025": map_reflect,
        }
        for name, pred in preds.items():
            metrics[name].add(pred, truth)
        if (ii + 1) % args.progress == 0 or ii + 1 == len(val_idx):
            print(f"evaluated {ii + 1}/{len(val_idx)}", flush=True)

    rows = []
    for name in candidates:
        row = {"name": name, "online": ONLINE.get(name), **metrics[name].result()}
        rows.append(row)

    correlations = {}
    for mode in pas_modes:
        correlations[mode] = rank_corr(rows, f"score_{mode}")

    payload = {
        "method": "evaluate_leaked_metric_variants",
        "mode": args.mode,
        "val_count": int(args.val_count),
        "seed": int(args.seed),
        "pas_modes": pas_modes,
        "rows": rows,
        "correlations": correlations,
        "leaked_pas_note": "reshape B,256,4,192 -> B,2,16,8,4,192 -> transpose B,2,192,16,4,8 -> fft2 axes=(2,3)",
    }
    return payload


def write_report(payload: dict, out_md: str):
    modes = payload["pas_modes"]
    lines = [
        "# Leaked Platform Metric Variants",
        "",
        "目的：利用平台报错暴露的 PAS reshape/FFT 口径，比较不同本地指标与已知线上结果的相关性。",
        "",
        "泄露线索：",
        "",
        "```text",
        "true_reshaped shape = (B, 2, 192, 16, 4, 8)",
        "np.fft.fft2(true_reshaped, axes=(2, 3))",
        "```",
        "",
        "说明：本脚本关注评分数学口径；是否整块计算或逐点计算只影响内存/速度，不改变这些均值型指标。",
        "",
        "## Correlation",
        "",
        "| PAS mode | Pearson(local score, online) | Spearman(rank) |",
        "| --- | ---: | ---: |",
    ]
    for mode in modes:
        c = payload["correlations"][mode]
        pearson = "-" if c["pearson"] is None else f"{c['pearson']:.4f}"
        spearman = "-" if c["spearman"] is None else f"{c['spearman']:.4f}"
        lines.append(f"| `{mode}` | {pearson} | {spearman} |")

    lines.extend(["", "## Rows", ""])
    for mode in modes:
        lines.append(f"### {mode}")
        lines.append("")
        lines.append("| candidate | online | local score | PAS | PDP | NMSE |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        key = f"score_{mode}"
        for row in sorted(payload["rows"], key=lambda r: r[key], reverse=True):
            lines.append(
                f"| `{row['name']}` | {row['online']:.6f} | {row[key]:.6f} | "
                f"{row[f'pas_{mode}']:.6f} | {row['pdp']:.6f} | {row['nmse']:.6f} |"
            )
        lines.append("")
    Path(out_md).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--cache", default=str(PATHS.artifacts / "shared_features/height_geometry_cache.npz"))
    parser.add_argument("--map-features", default=str(PATHS.artifacts / "shared_features/map_features.npz"))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/leaked_metric_variants_matched500.json"))
    parser.add_argument("--out-md", default=str(PATHS.analysis / "results/leaked_metric_variants_matched500.md"))
    parser.add_argument("--mode", choices=["random", "matched-test"], default="matched-test")
    parser.add_argument("--val-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--max-k", type=int, default=24)
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--steering-scale", type=float, default=4.71238898038)
    parser.add_argument("--progress", type=int, default=20)
    args = parser.parse_args()

    payload = evaluate(args)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(payload, args.out_md)
    print(json.dumps({"out": args.out, "out_md": args.out_md, "correlations": payload["correlations"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
