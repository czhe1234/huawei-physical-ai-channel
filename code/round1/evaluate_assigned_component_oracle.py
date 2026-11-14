#!/usr/bin/env python3
from project_paths import PATHS
"""Oracle diagnostics for geometry-assigned angle-delay components.

This evaluates whether the newly built geometric path/component assignment can
actually improve a strong physical-local-linear baseline if the component
scalar were known. It is intentionally an oracle diagnostic, not a submission
generator.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from build_component_path_assignments import circular_abs_delta
from evaluate_derotated_loo_grid import choose_val_indices
from evaluate_local_ad_paths import (
    BS_POS,
    MH,
    MV,
    S_SC,
    ad_transform,
    flat_components,
    inverse_ad,
    parse_ints,
)
from evaluate_physical_local_linear import affine_coefficients, predict_affine
from generate_geometric_steering_blend import unit_vectors
from spectrum_projection import MetricAccumulator


def parse_words(text):
    return [x for x in str(text).split(",") if x.strip()]


def parse_floats(text):
    return [float(x) for x in str(text).split(",") if x.strip()]


def top_components_from_ad(ad, count):
    power = np.sum(np.abs(ad) ** 2, axis=(1, 4), dtype=np.float64)  # B,H,V,S
    flat = power.reshape(power.shape[0], -1)
    count = min(int(count), flat.shape[1])
    ids = np.argpartition(flat, -count, axis=1)[:, -count:]
    rows = np.arange(flat.shape[0])[:, None]
    order = np.argsort(flat[rows, ids], axis=1)[:, ::-1]
    return ids[rows, order].astype(np.int32), flat[rows, ids[rows, order]].astype(np.float32)


def assign_dynamic(component_ids, path_delay, path_h, path_v, valid, weights):
    hbin, vbin, sbin = flat_components(component_ids.reshape(-1))
    hbin = hbin.reshape(component_ids.shape).astype(np.float32)
    vbin = vbin.reshape(component_ids.shape).astype(np.float32)
    sbin = sbin.reshape(component_ids.shape).astype(np.float32)
    de = np.abs(sbin[:, :, None] - path_delay[:, None, :])
    he = circular_abs_delta(hbin[:, :, None], path_h[:, None, :], MH)
    ve = circular_abs_delta(vbin[:, :, None], path_v[:, None, :], MV)
    valid3 = valid[:, None, :] & np.isfinite(de) & np.isfinite(he) & np.isfinite(ve)
    score = (de / weights[0]) ** 2 + (he / weights[1]) ** 2 + (ve / weights[2]) ** 2
    score = np.where(valid3, score, np.inf)
    best = np.argmin(score, axis=2)
    row = np.arange(component_ids.shape[0])[:, None]
    col = np.arange(component_ids.shape[1])[None, :]
    return {
        "component_ids": component_ids,
        "hbin": hbin.astype(np.int16),
        "vbin": vbin.astype(np.int16),
        "sbin": sbin.astype(np.int16),
        "best_path": best.astype(np.int16),
        "score": score[row, col, best].astype(np.float32),
        "delay_err": de[row, col, best].astype(np.float32),
        "h_err": he[row, col, best].astype(np.float32),
        "v_err": ve[row, col, best].astype(np.float32),
    }


def assignment_mask(assign, thresholds):
    return (
        (assign["delay_err"] <= thresholds[0])
        & (assign["h_err"] <= thresholds[1])
        & (assign["v_err"] <= thresholds[2])
        & np.isfinite(assign["score"])
    )


def component_coeff(ad_one, component_id):
    h, v, s = flat_components(np.asarray([component_id], dtype=np.int64))
    return ad_one[:, h[0], v[0], :, s[0]]


def set_component_coeff(ad_one, component_id, coeff):
    h, v, s = flat_components(np.asarray([component_id], dtype=np.int64))
    ad_one[:, h[0], v[0], :, s[0]] = coeff


def scalar_fit(src, dst, eps=1e-20):
    den = np.sum(np.abs(src) ** 2, dtype=np.float64)
    if den <= eps:
        return np.complex64(1.0 + 0.0j)
    return np.complex64(np.sum(dst * np.conj(src), dtype=np.complex128) / den)


def clip_scalar(c, max_gain, max_phase):
    mag = float(np.abs(c))
    phase = float(np.angle(c))
    mag = float(np.clip(mag, 1.0 / max_gain, max_gain))
    phase = float(np.clip(phase, -max_phase, max_phase))
    return np.complex64(mag * np.exp(1j * phase))


def apply_oracle(base_ad, truth_ad, component_ids, mask, method, max_components, max_gain, max_phase):
    out = np.array(base_ad, copy=True)
    used = 0
    for comp, ok in zip(component_ids, mask):
        if not ok:
            continue
        src = component_coeff(out, int(comp))
        dst = component_coeff(truth_ad, int(comp))
        if method == "replace":
            new_coeff = dst
        elif method == "scalar":
            new_coeff = scalar_fit(src, dst) * src
        elif method == "scalar_clip":
            new_coeff = clip_scalar(scalar_fit(src, dst), max_gain, max_phase) * src
        elif method == "phase":
            c = scalar_fit(src, dst)
            if np.abs(c) > 1e-20:
                new_coeff = (c / np.abs(c)).astype(np.complex64) * src
            else:
                new_coeff = src
        elif method == "magnitude":
            phase = src / np.maximum(np.abs(src), 1e-20)
            new_coeff = np.abs(dst).astype(np.float32) * phase
        else:
            raise ValueError(f"unknown oracle method {method}")
        set_component_coeff(out, int(comp), new_coeff.astype(np.complex64))
        used += 1
        if used >= max_components:
            break
    return out, used


def predict_v9_baseline(pos, test_pos, channels, val_idx, args):
    max_k = int(args.k)
    tree = cKDTree(pos[:, :2])
    d_all, local_all = tree.query(pos[val_idx, :2], k=max_k + 1, workers=-1)
    neighbor_idx = np.empty((len(val_idx), max_k), dtype=np.int64)
    distances = np.empty((len(val_idx), max_k), dtype=np.float64)
    for i, vi in enumerate(val_idx):
        keep = local_all[i] != vi
        neighbor_idx[i] = local_all[i, keep][:max_k]
        distances[i] = d_all[i, keep][:max_k]

    dist_all = np.linalg.norm(pos - BS_POS[None, :], axis=1)
    u_all = unit_vectors(pos)
    preds = []
    coeff_rows = []
    for row, vi in enumerate(val_idx):
        idx = neighbor_idx[row]
        neigh = np.asarray(channels[idx])
        rel_xy = pos[idx, :2] - pos[vi, :2]
        coeff = affine_coefficients(rel_xy, distances[row], args.power, args.ridge)
        pred = predict_affine(
            neigh,
            coeff,
            dist_all[idx] - dist_all[vi],
            u_all,
            idx,
            vi,
            args.beta,
            args.gamma,
            args.alpha,
            args.steering_scale,
        )
        preds.append(pred.astype(np.complex64))
        coeff_rows.append(coeff)
        if (row + 1) % 20 == 0 or row + 1 == len(val_idx):
            print(f"baseline predicted {row + 1}/{len(val_idx)}", flush=True)
    return np.asarray(preds), np.asarray(coeff_rows)


def nmse_num_den(pred, truth):
    return (
        float(np.sum(np.abs(pred - truth) ** 2, dtype=np.float64)),
        float(np.sum(np.abs(truth) ** 2, dtype=np.float64)),
    )


def make_result(name, num, den, acc, used_counts, mask_counts):
    metric = acc.result()
    return {
        "name": name,
        "global_nmse_direct": float(num / max(den, 1e-30)),
        "used_components_mean": float(np.mean(used_counts)) if used_counts else 0.0,
        "used_components_p90": float(np.percentile(used_counts, 90)) if used_counts else 0.0,
        "eligible_components_mean": float(np.mean(mask_counts)) if mask_counts else 0.0,
        **metric,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--assignments", default=str(PATHS.artifacts / "shared_features/component_path_assignments.npz"))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/assigned_component_oracle.json"))
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
    parser.add_argument("--baseline-component-counts", type=parse_ints, default="16,32,64")
    parser.add_argument("--methods", type=parse_words, default="replace,scalar,scalar_clip,phase")
    parser.add_argument("--max-components", type=int, default=64)
    parser.add_argument("--thresholds", type=parse_floats, default="2.0,3.0,2.0")
    parser.add_argument("--strict-thresholds", type=parse_floats, default="1.25,2.0,1.25")
    parser.add_argument("--max-gain", type=float, default=2.0)
    parser.add_argument("--max-phase", type=float, default=1.57079632679)
    parser.add_argument("--pas-mode", choices=["sumpol", "fullpol"], default="sumpol")
    args = parser.parse_args()

    data = Path(args.data)
    pos = np.load(data / "Round1_Train_Pos.npy")
    test_pos = np.load(data / "Round1_Test_Pos.npy")
    channels = np.load(data / "Round1_Train_Channel.npy", mmap_mode="r")
    val_idx = choose_val_indices(pos[:, :2], test_pos[:, :2], args.val_count, args.val_mode, args.seed)

    truth = np.asarray(channels[val_idx])
    baseline, coeff = predict_v9_baseline(pos, test_pos, channels, val_idx, args)
    truth_ad = ad_transform(truth)
    base_ad = ad_transform(baseline)

    cache = np.load(args.assignments)
    path_delay = cache["path_delay_bin"][val_idx].astype(np.float32)
    path_h = cache["path_h_bin"][val_idx].astype(np.float32)
    path_v = cache["path_v_bin"][val_idx].astype(np.float32)
    path_valid = cache["path_valid"][val_idx]
    weights = cache["weights"].astype(np.float32)
    default_thresholds = np.asarray(args.thresholds, dtype=np.float32)
    strict_thresholds = np.asarray(args.strict_thresholds, dtype=np.float32)

    truth_assign = {
        "component_ids": cache["component_ids"][val_idx],
        "delay_err": cache["delay_err"][val_idx],
        "h_err": cache["h_err"][val_idx],
        "v_err": cache["v_err"][val_idx],
        "score": cache["assignment_score"][val_idx],
    }
    baseline_component_sets = {}
    for count in args.baseline_component_counts:
        ids, _ = top_components_from_ad(base_ad, count)
        baseline_component_sets[int(count)] = assign_dynamic(ids, path_delay, path_h, path_v, path_valid, weights)

    candidates = [("baseline", "none", 0, "none", default_thresholds)]
    for method in args.methods:
        candidates.append(("truth_support", method, args.max_components, "default", default_thresholds))
        candidates.append(("truth_support", method, args.max_components, "strict", strict_thresholds))
        for count in args.baseline_component_counts:
            candidates.append((f"baseline_top{count}", method, args.max_components, "default", default_thresholds))
            candidates.append((f"baseline_top{count}", method, args.max_components, "strict", strict_thresholds))

    stats = {}
    accs = {}
    used = {}
    eligible = {}
    for source, method, max_comp, thresh_name, _ in candidates:
        name = f"{source}_{method}_{thresh_name}"
        stats[name] = [0.0, 0.0]
        accs[name] = MetricAccumulator(args.pas_mode)
        used[name] = []
        eligible[name] = []

    for i in range(len(val_idx)):
        for source, method, max_comp, thresh_name, thresholds in candidates:
            name = f"{source}_{method}_{thresh_name}"
            if source == "baseline":
                pred = baseline[i:i + 1]
                n_used = 0
                n_eligible = 0
            else:
                if source == "truth_support":
                    assign = truth_assign
                else:
                    count = int(source.replace("baseline_top", ""))
                    assign = baseline_component_sets[count]
                mask = assignment_mask({k: v[i:i + 1] for k, v in assign.items() if k != "component_ids"}, thresholds)[0]
                comp_ids = assign["component_ids"][i]
                n_eligible = int(np.sum(mask))
                patched_ad, n_used = apply_oracle(
                    base_ad[i],
                    truth_ad[i],
                    comp_ids,
                    mask,
                    method,
                    int(max_comp),
                    args.max_gain,
                    args.max_phase,
                )
                pred = inverse_ad(patched_ad[None])
            num, den = nmse_num_den(pred, truth[i:i + 1])
            stats[name][0] += num
            stats[name][1] += den
            accs[name].add(pred, truth[i:i + 1])
            used[name].append(n_used)
            eligible[name].append(n_eligible)
        if (i + 1) % 10 == 0 or i + 1 == len(val_idx):
            print(f"oracle evaluated {i + 1}/{len(val_idx)}", flush=True)

    results = [
        make_result(name, stats[name][0], stats[name][1], accs[name], used[name], eligible[name])
        for name in stats
    ]
    results.sort(key=lambda row: row["global_nmse_direct"])
    payload = {
        "method": "assigned_component_oracle",
        "val_count": int(len(val_idx)),
        "val_mode": args.val_mode,
        "seed": int(args.seed),
        "baseline": {
            "k": int(args.k),
            "power": float(args.power),
            "beta": float(args.beta),
            "gamma": float(args.gamma),
            "alpha": float(args.alpha),
            "ridge": float(args.ridge),
        },
        "thresholds": {
            "default": [float(x) for x in default_thresholds],
            "strict": [float(x) for x in strict_thresholds],
        },
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    for row in results[:30]:
        print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
