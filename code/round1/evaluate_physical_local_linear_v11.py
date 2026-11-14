#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from evaluate_derotated_loo_grid import choose_val_indices
from evaluate_local_ad_paths import BS_POS, parse_floats, parse_ints, parse_words
from evaluate_physical_local_linear import affine_coefficients, predict_affine
from generate_geometric_steering_blend import unit_vectors
from generate_physical_local_linear_v11 import (
    apply_policy,
    diagnose,
    predict_derotated_idw,
    predict_steered_idw,
    summarize_diags,
)
from spectrum_projection import MetricAccumulator


class PolicyArgs:
    pass


def make_policy_args(base_args, candidate):
    policy = PolicyArgs()
    for key in [
        "power",
        "beta",
        "gamma",
        "alpha",
        "ridge",
        "steering_scale",
        "severe_ratio_small_thresh",
        "severe_neg_frac_thresh",
        "blend_affine",
    ]:
        setattr(policy, key, getattr(base_args, key))
    for key, value in candidate.items():
        setattr(policy, key, value)
    return policy


def candidate_key(candidate):
    return (
        f"fk{candidate['fallback_k']}_ik{candidate['idw_k']}"
        f"_rs{candidate['ratio_small_thresh']:g}_ri{candidate['ratio_idw_thresh']:g}"
        f"_nf{candidate['neg_frac_thresh']:g}_cs{candidate['cos_thresh']:g}"
        f"_gc{candidate['gain_cap']:g}_{candidate['mode']}"
    )


def build_candidates(args):
    return [
        {
            "k": int(args.k),
            "fallback_k": int(fallback_k),
            "idw_k": int(idw_k),
            "ratio_small_thresh": float(ratio_small),
            "ratio_idw_thresh": float(ratio_idw),
            "neg_frac_thresh": float(neg_frac),
            "cos_thresh": float(cos_thresh),
            "gain_cap": float(gain_cap),
            "mode": mode,
        }
        for fallback_k in args.fallback_k_values
        for idw_k in args.idw_k_values
        for ratio_small in args.ratio_small_thresh
        for ratio_idw in args.ratio_idw_thresh
        for neg_frac in args.neg_frac_thresh
        for cos_thresh in args.cos_thresh
        for gain_cap in args.gain_caps
        for mode in args.modes
    ]


def validate_shapes(pos, test_pos, h):
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"unexpected train position shape {pos.shape}")
    if test_pos.ndim != 2 or test_pos.shape[1] != 3:
        raise ValueError(f"unexpected test position shape {test_pos.shape}")
    if h.ndim != 4 or h.shape[1:] != (256, 4, 192):
        raise ValueError(f"unexpected train channel shape {h.shape}")


def prepare_rows(pos, test_pos, h, args):
    val_idx = choose_val_indices(pos[:, :2], test_pos[:, :2], args.val_count, args.val_mode, args.seed)
    max_k = max(args.k, max(args.fallback_k_values), max(args.idw_k_values))
    tree = cKDTree(pos[:, :2])
    distances_all, local_all = tree.query(pos[val_idx, :2], k=max_k + 1, workers=-1)
    neighbor_idx = np.empty((len(val_idx), max_k), dtype=np.int64)
    distances = np.empty((len(val_idx), max_k), dtype=np.float64)
    for i, vi in enumerate(val_idx):
        keep = local_all[i] != vi
        neighbor_idx[i] = local_all[i, keep][:max_k]
        distances[i] = distances_all[i, keep][:max_k]

    dist_all = np.linalg.norm(pos - BS_POS[None, :], axis=1)
    u_all = unit_vectors(pos)
    rows = []
    for row, vi in enumerate(val_idx):
        idx = neighbor_idx[row]
        neigh = np.asarray(h[idx])
        rel_xy = pos[idx[:args.k], :2] - pos[vi, :2]
        coeff = affine_coefficients(rel_xy, distances[row, :args.k], args.power, args.ridge)
        pred_aff = predict_affine(
            neigh[:args.k],
            coeff,
            dist_all[idx[:args.k]] - dist_all[vi],
            u_all,
            idx[:args.k],
            vi,
            args.beta,
            args.gamma,
            args.alpha,
            args.steering_scale,
        )
        small_cache = {}
        for fallback_k in args.fallback_k_values:
            small_cache[int(fallback_k)] = predict_steered_idw(
                neigh,
                distances[row],
                dist_all[idx] - dist_all[vi],
                u_all,
                idx,
                vi,
                int(fallback_k),
                args.power,
                args.beta,
                args.gamma,
                args.alpha,
                args.steering_scale,
            )
        idw_cache = {}
        for idw_k in args.idw_k_values:
            idw_cache[int(idw_k)] = predict_derotated_idw(
                neigh,
                distances[row],
                dist_all[idx] - dist_all[vi],
                int(idw_k),
                args.power,
                args.beta,
                args.gamma,
            )
        rows.append({
            "vi": int(vi),
            "truth": np.asarray(h[vi]),
            "pred_aff": pred_aff,
            "coeff": coeff,
            "small": small_cache,
            "idw": idw_cache,
        })
        if (row + 1) % 10 == 0 or row + 1 == len(val_idx):
            print(f"prepared {row + 1}/{len(val_idx)}", flush=True)
    return val_idx, rows


def eval_nmse(rows, candidates, args):
    results = []
    for ci, candidate in enumerate(candidates):
        policy_args = make_policy_args(args, candidate)
        num = 0.0
        den = 0.0
        diags = []
        for row in rows:
            pred_small = row["small"][candidate["fallback_k"]]
            pred_idw = row["idw"][candidate["idw_k"]]
            diag = diagnose(row["pred_aff"], pred_small, pred_idw, row["coeff"])
            pred, branch, gain = apply_policy(row["pred_aff"], pred_small, pred_idw, diag, policy_args)
            diag["branch"] = branch
            diag["gain"] = float(gain)
            diff = pred - row["truth"]
            num += float(np.sum(np.abs(diff) ** 2, dtype=np.float64))
            den += float(np.sum(np.abs(row["truth"]) ** 2, dtype=np.float64))
            diags.append(diag)
        result = copy.deepcopy(candidate)
        result.update({
            "name": candidate_key(candidate),
            "global_nmse_direct": float(num / max(den, 1e-30)),
            "diag_summary": summarize_diags(diags),
        })
        results.append(result)
        if (ci + 1) % 20 == 0 or ci + 1 == len(candidates):
            print(f"nmse evaluated {ci + 1}/{len(candidates)}", flush=True)
    results.sort(key=lambda row: row["global_nmse_direct"])
    return results


def add_metric_results(rows, results, args):
    selected = results[:args.metric_top]
    baseline_acc = MetricAccumulator(args.pas_mode)
    for row in rows:
        baseline_acc.add(row["pred_aff"][None], row["truth"][None])
    metric_rows = [{"name": "v9_affine_baseline", **baseline_acc.result()}]

    for ri, result in enumerate(selected):
        candidate = {key: result[key] for key in [
            "k",
            "fallback_k",
            "idw_k",
            "ratio_small_thresh",
            "ratio_idw_thresh",
            "neg_frac_thresh",
            "cos_thresh",
            "gain_cap",
            "mode",
        ]}
        policy_args = make_policy_args(args, candidate)
        acc = MetricAccumulator(args.pas_mode)
        for row in rows:
            pred_small = row["small"][candidate["fallback_k"]]
            pred_idw = row["idw"][candidate["idw_k"]]
            diag = diagnose(row["pred_aff"], pred_small, pred_idw, row["coeff"])
            pred, _, _ = apply_policy(row["pred_aff"], pred_small, pred_idw, diag, policy_args)
            acc.add(pred[None], row["truth"][None])
        metric = {"name": result["name"], **acc.result()}
        metric_rows.append(metric)
        result.update({f"metric_{k}": v for k, v in metric.items() if k != "name"})
        print(f"metric evaluated {ri + 1}/{len(selected)}", flush=True)
    return metric_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--out", default=str(PATHS.version_report_dir(11) / "physical_local_linear_v11_eval.json"))
    parser.add_argument("--val-count", type=int, default=100)
    parser.add_argument("--val-mode", choices=["matched-test", "random"], default="matched-test")
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--gamma", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=0.3)
    parser.add_argument("--steering-scale", type=float, default=4.71238898038)
    parser.add_argument("--fallback-k-values", type=parse_ints, default="4")
    parser.add_argument("--idw-k-values", type=parse_ints, default="16")
    parser.add_argument("--ratio-small-thresh", type=parse_floats, default="0.40,0.50,0.60")
    parser.add_argument("--ratio-idw-thresh", type=parse_floats, default="0.50,0.60,0.70")
    parser.add_argument("--neg-frac-thresh", type=parse_floats, default="0.10,0.15,0.20")
    parser.add_argument("--cos-thresh", type=parse_floats, default="0.985,0.99")
    parser.add_argument("--gain-caps", type=parse_floats, default="1.15,1.25,1.35")
    parser.add_argument("--modes", type=parse_words, default="gain_only,gain_then_small")
    parser.add_argument("--severe-ratio-small-thresh", type=float, default=0.30)
    parser.add_argument("--severe-neg-frac-thresh", type=float, default=0.30)
    parser.add_argument("--blend-affine", type=float, default=0.25)
    parser.add_argument("--metric-top", type=int, default=8)
    parser.add_argument("--pas-mode", choices=["sumpol", "fullpol"], default="sumpol")
    args = parser.parse_args()

    pos = np.load(os.path.join(args.data, "Round1_Train_Pos.npy"))
    test_pos = np.load(os.path.join(args.data, "Round1_Test_Pos.npy"))
    h = np.load(os.path.join(args.data, "Round1_Train_Channel.npy"), mmap_mode="r")
    validate_shapes(pos, test_pos, h)

    candidates = build_candidates(args)
    val_idx, rows = prepare_rows(pos, test_pos, h, args)
    results = eval_nmse(rows, candidates, args)
    metric_rows = add_metric_results(rows, results, args)

    payload = {
        "seed": int(args.seed),
        "val_count": int(len(val_idx)),
        "val_mode": args.val_mode,
        "base": {
            "k": int(args.k),
            "power": float(args.power),
            "beta": float(args.beta),
            "gamma": float(args.gamma),
            "alpha": float(args.alpha),
            "ridge": float(args.ridge),
            "steering_scale": float(args.steering_scale),
        },
        "metric_rows": metric_rows,
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    for row in results[:20]:
        printable = {k: v for k, v in row.items() if k != "diag_summary"}
        print(json.dumps(printable, indent=2), flush=True)
    print("metric rows:", flush=True)
    for row in metric_rows:
        print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
