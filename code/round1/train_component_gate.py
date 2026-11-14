#!/usr/bin/env python3
from project_paths import PATHS
"""Train a component-level gate for geometry-assigned scalar corrections."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from evaluate_derotated_loo_grid import choose_val_indices
from evaluate_assigned_component_oracle import component_coeff
from evaluate_local_ad_paths import inverse_ad
from spectrum_projection import MetricAccumulator
from train_assigned_scalar_expert import (
    build_dataset,
    clip_complex_scalar,
    patch_ad,
    parse_floats,
)


LOCAL_START = 9 + 7 + 4 + 4 + 4
PATH_START = LOCAL_START + 12


def parse_thresholds(text):
    return [float(x) for x in str(text).split(",") if x.strip()]


def proposal_scalar(features, proposal, gain_cap):
    if proposal == "local_idw":
        raw = features[:, LOCAL_START].astype(np.float32) + 1j * features[:, LOCAL_START + 1].astype(np.float32)
    elif proposal == "local_near":
        raw = features[:, LOCAL_START + 4].astype(np.float32) + 1j * features[:, LOCAL_START + 5].astype(np.float32)
    else:
        raise ValueError(proposal)
    return clip_complex_scalar(raw.astype(np.complex64), gain_cap)


def group_ids_from_features(features, args):
    if args.group_mode == "global":
        return np.zeros(features.shape[0], dtype=np.int32)
    path_type = np.rint(features[:, PATH_START]).astype(np.int32)
    plane_id = np.rint(features[:, PATH_START + 1] * 200.0).astype(np.int32)
    delay_band = np.floor(np.clip(features[:, 2], 0.0, 1.0) * args.delay_groups).astype(np.int32)
    delay_band = np.clip(delay_band, 0, args.delay_groups - 1)
    if args.group_mode == "path_type":
        return path_type.astype(np.int32)
    if args.group_mode == "delay_band":
        return delay_band.astype(np.int32)
    if args.group_mode == "path_delay":
        return (path_type * args.delay_groups + delay_band).astype(np.int32)
    if args.group_mode == "plane_mod":
        # LOS and invalid plane ids go to group 0. Reflection planes are bucketed
        # to avoid tiny per-plane models.
        bucket = np.where(plane_id >= 0, 1 + (plane_id % args.plane_groups), 0)
        return bucket.astype(np.int32)
    if args.group_mode == "path_plane_mod":
        bucket = np.where(plane_id >= 0, 1 + (plane_id % args.plane_groups), 0)
        return (path_type * (args.plane_groups + 1) + bucket).astype(np.int32)
    raise ValueError(args.group_mode)


def make_patch_scalar(raw_scalar, rho):
    return (1.0 + rho * (raw_scalar - 1.0)).astype(np.complex64)


def component_errors(base_ad_one, truth_ad_one, comp_ids, patch_scalar):
    base_err = np.empty(len(comp_ids), dtype=np.float32)
    patch_err = np.empty(len(comp_ids), dtype=np.float32)
    for i, comp in enumerate(comp_ids):
        base = component_coeff(base_ad_one, int(comp))
        truth = component_coeff(truth_ad_one, int(comp))
        patched = patch_scalar[i] * base
        base_err[i] = float(np.sum(np.abs(base - truth) ** 2, dtype=np.float64))
        patch_err[i] = float(np.sum(np.abs(patched - truth) ** 2, dtype=np.float64))
    return base_err, patch_err


def build_gate_examples(dataset, proposal, rho, gain_cap, args, label):
    xs = []
    ys = []
    weights = []
    groups = []
    cursor = 0
    positives = 0
    total = 0
    for i in range(len(dataset["truth"])):
        mask = dataset["mask"][i]
        n = int(np.sum(mask))
        if n == 0:
            continue
        feat = dataset["features"][cursor:cursor + n]
        cursor += n
        comp_ids = dataset["component_ids"][i][mask]
        raw = proposal_scalar(feat, proposal, gain_cap)
        patch = make_patch_scalar(raw, rho)
        base_err, patch_err = component_errors(dataset["base_ad"][i], dataset["truth_ad"][i], comp_ids, patch)
        benefit = base_err - patch_err
        norm = benefit / np.maximum(base_err, 1e-12)
        xs.append(feat)
        ys.append(np.clip(norm, -2.0, 2.0).astype(np.float32))
        w = np.maximum(base_err, 1e-12)
        weights.append(w.astype(np.float32))
        groups.append(group_ids_from_features(feat, args))
        positives += int(np.sum(benefit > 0.0))
        total += int(n)
        if (i + 1) % 50 == 0 or i + 1 == len(dataset["truth"]):
            print(f"{label} gate examples {i + 1}/{len(dataset['truth'])}", flush=True)
    if xs:
        x = np.concatenate(xs, axis=0).astype(np.float32)
        y = np.concatenate(ys, axis=0).astype(np.float32)
        w = np.concatenate(weights, axis=0).astype(np.float32)
        g = np.concatenate(groups, axis=0).astype(np.int32)
    else:
        x = np.empty((0, 1), dtype=np.float32)
        y = np.empty((0,), dtype=np.float32)
        w = np.empty((0,), dtype=np.float32)
        g = np.empty((0,), dtype=np.int32)
    w = w / max(float(np.mean(w)), 1e-12)
    w = np.clip(w, 0.05, 20.0).astype(np.float32)
    print(
        f"{label} examples={len(y)} positive_frac={positives / max(total, 1):.4f}",
        flush=True,
    )
    return x, y, w, g


def normalize_train(x):
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=10.0, neginf=-10.0)
    mean = x.mean(axis=0).astype(np.float32)
    std = (x.std(axis=0) + 1e-6).astype(np.float32)
    return ((x - mean) / std).astype(np.float32), mean, std


def normalize_apply(x, mean, std):
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=10.0, neginf=-10.0)
    return ((x - mean[None, :]) / std[None, :]).astype(np.float32)


def fit_hgb(x_norm, y, w, args, seed):
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        max_leaf_nodes=args.max_leaf_nodes,
        l2_regularization=args.l2,
        early_stopping=True,
        random_state=seed,
    )
    model.fit(x_norm, y, sample_weight=w)
    return model


def train_gate(x, y, w, groups, args):
    x_norm, mean, std = normalize_train(x)
    global_model = fit_hgb(x_norm, y, w, args, args.seed)
    if args.group_mode == "global":
        print(f"gate train_score={global_model.score(x_norm, y):.6f}", flush=True)
        return {"global": global_model, "groups": {}, "mode": "global"}, mean, std

    models = {}
    for gi, group in enumerate(np.unique(groups)):
        mask = groups == group
        if int(np.sum(mask)) < args.min_group_samples:
            continue
        model = fit_hgb(x_norm[mask], y[mask], w[mask], args, args.seed + int(group) + 17)
        models[int(group)] = model
    pred = predict_grouped({"global": global_model, "groups": models, "mode": args.group_mode}, x_norm, groups)
    score = 1.0 - float(np.sum((pred - y) ** 2) / max(np.sum((y - np.mean(y)) ** 2), 1e-30))
    print(
        f"gate grouped mode={args.group_mode} groups={len(models)} train_score={score:.6f} "
        f"global_score={global_model.score(x_norm, y):.6f}",
        flush=True,
    )
    return {"global": global_model, "groups": models, "mode": args.group_mode}, mean, std


def predict_grouped(model_pack, x_norm, groups):
    out = model_pack["global"].predict(x_norm).astype(np.float32)
    for group, model in model_pack["groups"].items():
        mask = groups == int(group)
        if np.any(mask):
            out[mask] = model.predict(x_norm[mask]).astype(np.float32)
    return out


def add_metric(name, pred, truth, stats, accs):
    if name not in stats:
        stats[name] = [0.0, 0.0]
        accs[name] = MetricAccumulator("sumpol")
    stats[name][0] += float(np.sum(np.abs(pred - truth) ** 2, dtype=np.float64))
    stats[name][1] += float(np.sum(np.abs(truth) ** 2, dtype=np.float64))
    accs[name].add(pred, truth)


def evaluate_gate(dataset, model, mean, std, args):
    stats = {}
    accs = {}
    cursor = 0
    gate_counts = {thr: [] for thr in args.thresholds_eval}
    oracle_counts = []
    for i in range(len(dataset["truth"])):
        truth = dataset["truth"][i:i + 1]
        base = dataset["baseline"][i:i + 1]
        add_metric("baseline", base, truth, stats, accs)
        mask = dataset["mask"][i]
        n = int(np.sum(mask))
        if n == 0:
            for thr in args.thresholds_eval:
                add_metric(f"learned_gate_thr{thr:g}", base, truth, stats, accs)
            add_metric("all_patch", base, truth, stats, accs)
            add_metric("oracle_gate", base, truth, stats, accs)
            continue

        feat = dataset["features"][cursor:cursor + n]
        cursor += n
        comp_ids = dataset["component_ids"][i][mask]
        raw = proposal_scalar(feat, args.proposal, args.proposal_gain_cap)
        patch_scalar = make_patch_scalar(raw, args.rho)
        base_err, patch_err = component_errors(dataset["base_ad"][i], dataset["truth_ad"][i], comp_ids, patch_scalar)
        benefit = base_err - patch_err

        all_ad = patch_ad(dataset["base_ad"][i], comp_ids, raw, args.rho)
        add_metric("all_patch", inverse_ad(all_ad[None]), truth, stats, accs)

        oracle_mask = benefit > 0.0
        oracle_counts.append(int(np.sum(oracle_mask)))
        if np.any(oracle_mask):
            oracle_ad = patch_ad(dataset["base_ad"][i], comp_ids[oracle_mask], raw[oracle_mask], args.rho)
            oracle_pred = inverse_ad(oracle_ad[None])
        else:
            oracle_pred = base
        add_metric("oracle_gate", oracle_pred, truth, stats, accs)

        feat_norm = normalize_apply(feat, mean, std)
        groups = group_ids_from_features(feat, args)
        if isinstance(model, dict):
            pred_benefit = predict_grouped(model, feat_norm, groups).astype(np.float32)
        else:
            pred_benefit = model.predict(feat_norm).astype(np.float32)
        for thr in args.thresholds_eval:
            gate = pred_benefit > thr
            gate_counts[thr].append(int(np.sum(gate)))
            if np.any(gate):
                patched_ad = patch_ad(dataset["base_ad"][i], comp_ids[gate], raw[gate], args.rho)
                pred = inverse_ad(patched_ad[None])
            else:
                pred = base
            add_metric(f"learned_gate_thr{thr:g}", pred, truth, stats, accs)

        if (i + 1) % 10 == 0 or i + 1 == len(dataset["truth"]):
            print(f"eval {i + 1}/{len(dataset['truth'])}", flush=True)

    results = []
    for name, (num, den) in stats.items():
        row = {
            "method": name,
            "global_nmse_direct": float(num / max(den, 1e-30)),
            **accs[name].result(),
        }
        if name == "oracle_gate":
            row["patched_components_mean"] = float(np.mean(oracle_counts)) if oracle_counts else 0.0
        elif name.startswith("learned_gate_thr"):
            thr = float(name.split("thr", 1)[1])
            row["patched_components_mean"] = float(np.mean(gate_counts[thr])) if gate_counts[thr] else 0.0
        results.append(row)
    results.sort(key=lambda r: r["global_nmse_direct"])
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--paths", default=str(PATHS.artifacts / "shared_features/path_candidates.npz"))
    parser.add_argument("--assignments", default=str(PATHS.artifacts / "shared_features/component_path_assignments.npz"))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/component_gate.json"))
    parser.add_argument("--val-count", type=int, default=200)
    parser.add_argument("--val-mode", choices=["matched-test", "random"], default="matched-test")
    parser.add_argument("--train-query-count", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--component-count", type=int, default=64)
    parser.add_argument("--thresholds", type=parse_floats, default="2.0,3.0,2.0")
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--gamma", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=0.3)
    parser.add_argument("--steering-scale", type=float, default=4.71238898038)
    parser.add_argument("--clip-gain", type=float, default=3.0)
    parser.add_argument("--clip-phase", type=float, default=3.14159265359)
    parser.add_argument("--target-mode", default="realimag")
    parser.add_argument("--weight-mode", default="base")
    parser.add_argument("--proposal", choices=["local_idw", "local_near"], default="local_near")
    parser.add_argument("--proposal-gain-cap", type=float, default=1.5)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--thresholds-eval", type=parse_thresholds, default="-0.05,-0.02,0,0.02,0.05,0.1")
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2", type=float, default=0.05)
    parser.add_argument(
        "--group-mode",
        choices=["global", "path_type", "delay_band", "path_delay", "plane_mod", "path_plane_mod"],
        default="global",
    )
    parser.add_argument("--delay-groups", type=int, default=6)
    parser.add_argument("--plane-groups", type=int, default=16)
    parser.add_argument("--min-group-samples", type=int, default=800)
    args = parser.parse_args()

    pos = np.load(os.path.join(args.data, "Round1_Train_Pos.npy"))
    test_pos = np.load(os.path.join(args.data, "Round1_Test_Pos.npy"))
    channels = np.load(os.path.join(args.data, "Round1_Train_Channel.npy"), mmap_mode="r")
    path_cache = np.load(args.paths)
    assign_cache = np.load(args.assignments)
    val_idx = choose_val_indices(pos[:, :2], test_pos[:, :2], args.val_count, args.val_mode, args.seed)
    train_pool = np.setdiff1d(np.arange(len(pos), dtype=np.int64), val_idx, assume_unique=False)
    rng = np.random.default_rng(args.seed)
    train_queries = np.array(train_pool, copy=True)
    if 0 < args.train_query_count < len(train_queries):
        train_queries = rng.choice(train_queries, size=args.train_query_count, replace=False)

    print(f"building train dataset queries={len(train_queries)} val={len(val_idx)}", flush=True)
    train_data = build_dataset(pos, channels, train_queries, train_pool, path_cache, assign_cache, args, "train")
    x, y, w, groups = build_gate_examples(train_data, args.proposal, args.rho, args.proposal_gain_cap, args, "train")
    model, mean, std = train_gate(x, y, w, groups, args)

    print("building validation dataset", flush=True)
    val_data = build_dataset(pos, channels, val_idx, train_pool, path_cache, assign_cache, args, "val")
    results = evaluate_gate(val_data, model, mean, std, args)
    payload = {
        "args": vars(args),
        "train_examples": int(x.shape[0]),
        "feature_dim": int(x.shape[1]),
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
