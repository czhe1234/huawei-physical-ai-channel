#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from evaluate_local_ad_paths import BS_POS, idw_weights
from evaluate_metric_variants import MultiMetric, full_predict, load_map_geometry
from evaluate_reflection_steering import phase_from_units, reflection_lengths_and_units
from generate_geometric_steering_blend import steering_phase, unit_vectors


def spatial_density_features(train_pos, test_pos, ks):
    tree = cKDTree(train_pos[:, :2])
    train_k = max(ks) + 1
    train_d, _ = tree.query(train_pos[:, :2], k=train_k, workers=-1)
    test_d, _ = tree.query(test_pos[:, :2], k=max(ks), workers=-1)
    train_d = train_d[:, 1:]
    out_train = []
    out_test = []
    for k in ks:
        out_train.append(train_d[:, k - 1])
        out_test.append(test_d[:, k - 1])
    return np.stack(out_train, axis=1), np.stack(out_test, axis=1)


def make_query_features(data_dir, features_path, map_feature_names):
    train_pos = np.load(os.path.join(data_dir, "Round1_Train_Pos.npy"))
    test_pos = np.load(os.path.join(data_dir, "Round1_Test_Pos.npy"))
    map_data = np.load(features_path, allow_pickle=True)
    train_map = map_data["train_features"].astype(np.float64)
    test_map = map_data["test_features"].astype(np.float64)
    names = [str(x) for x in map_data["feature_names"]]

    train_xy = train_pos[:, :2].astype(np.float64)
    test_xy = test_pos[:, :2].astype(np.float64)
    train_vec = train_pos - BS_POS[None, :]
    test_vec = test_pos - BS_POS[None, :]
    train_range = np.linalg.norm(train_vec, axis=1)
    test_range = np.linalg.norm(test_vec, axis=1)
    train_unit = train_vec / np.maximum(train_range[:, None], 1e-30)
    test_unit = test_vec / np.maximum(test_range[:, None], 1e-30)
    train_density, test_density = spatial_density_features(train_pos, test_pos, [1, 4, 8, 16, 24])

    selected_map = []
    for name in map_feature_names:
        if name not in names:
            raise ValueError(f"unknown map feature {name}")
        selected_map.append(names.index(name))
    selected_map = np.asarray(selected_map, dtype=np.int64)

    train_feat = np.concatenate([
        train_xy,
        train_range[:, None],
        train_unit,
        train_density,
        train_map[:, selected_map],
    ], axis=1)
    test_feat = np.concatenate([
        test_xy,
        test_range[:, None],
        test_unit,
        test_density,
        test_map[:, selected_map],
    ], axis=1)
    feature_names = (
        ["x", "y", "range", "ux", "uy", "uz"]
        + [f"nn{k}" for k in [1, 4, 8, 16, 24]]
        + [f"map:{name}" for name in map_feature_names]
    )
    return train_feat, test_feat, feature_names


def robust_standardize(train_feat, test_feat):
    both = np.concatenate([train_feat, test_feat], axis=0)
    center = np.median(both, axis=0)
    q25, q75 = np.percentile(both, [25, 75], axis=0)
    scale = q75 - q25
    fallback = np.std(both, axis=0)
    scale = np.where(scale > 1e-12, scale, fallback)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (train_feat - center) / scale, (test_feat - center) / scale, center, scale


def choose_feature_matched_indices(train_feat, test_feat, count, seed, candidate_k):
    train_z, test_z, _, _ = robust_standardize(train_feat, test_feat)
    tree = cKDTree(train_z)
    first_dist, _ = tree.query(test_z, k=1, workers=-1)
    test_order = np.argsort(-first_dist)
    query_k = min(candidate_k, len(train_z))
    cand_dist, cand_idx = tree.query(test_z[test_order], k=query_k, workers=-1)
    if query_k == 1:
        cand_dist = cand_dist[:, None]
        cand_idx = cand_idx[:, None]

    rng = np.random.default_rng(seed)
    used = np.zeros(len(train_z), dtype=bool)
    chosen = []
    for row_dist, row_idx in zip(cand_dist, cand_idx):
        best = None
        best_score = np.inf
        for d, idx in zip(row_dist, row_idx):
            idx = int(idx)
            if used[idx]:
                continue
            jitter = 1e-9 * rng.random()
            score = float(d) + jitter
            if score < best_score:
                best = idx
                best_score = score
        if best is None:
            remaining = np.flatnonzero(~used)
            if remaining.size == 0:
                break
            d = np.linalg.norm(train_z[remaining] - test_z[test_order[len(chosen)]], axis=1)
            best = int(remaining[np.argmin(d)])
        used[best] = True
        chosen.append(best)
        if len(chosen) >= count:
            break
    if len(chosen) < count:
        rest = np.flatnonzero(~used)
        chosen.extend(rest[: count - len(chosen)].astype(int).tolist())
    return np.sort(np.asarray(chosen, dtype=np.int64))


def make_neighbors(pos, val_idx, max_k):
    tree = cKDTree(pos[:, :2])
    distances_all, local_all = tree.query(pos[val_idx, :2], k=max_k + 1, workers=-1)
    neighbor_idx = np.empty((len(val_idx), max_k), dtype=np.int64)
    distances = np.empty((len(val_idx), max_k), dtype=np.float64)
    for i, vi in enumerate(val_idx):
        keep = local_all[i] != vi
        neighbor_idx[i] = local_all[i, keep][:max_k]
        distances[i] = distances_all[i, keep][:max_k]
    return distances, neighbor_idx


def feature_summary(train_feat, test_feat, val_idx, feature_names):
    rows = []
    for j, name in enumerate(feature_names):
        test_mean = float(np.mean(test_feat[:, j]))
        val_mean = float(np.mean(train_feat[val_idx, j]))
        train_mean = float(np.mean(train_feat[:, j]))
        scale = float(np.std(np.concatenate([train_feat[:, j], test_feat[:, j]])))
        if scale <= 1e-12:
            scale = 1.0
        rows.append({
            "feature": name,
            "train_mean": train_mean,
            "test_mean": test_mean,
            "val_mean": val_mean,
            "val_test_abs_zdiff": abs(val_mean - test_mean) / scale,
            "train_test_abs_zdiff": abs(train_mean - test_mean) / scale,
        })
    rows.sort(key=lambda r: r["val_test_abs_zdiff"], reverse=True)
    return rows


def evaluate_candidates(args, val_idx, train_feat, test_feat, feature_names):
    data = Path(args.data)
    pos = np.load(data / "Round1_Train_Pos.npy")
    test_pos = np.load(data / "Round1_Test_Pos.npy")
    h = np.load(data / "Round1_Train_Channel.npy", mmap_mode="r")
    distances, neighbor_idx = make_neighbors(pos, val_idx, args.max_k)
    clear, plane_points, plane_normals = load_map_geometry(args.features, val_idx)
    dist_all = np.linalg.norm(pos - BS_POS[None, :], axis=1)
    u_all = unit_vectors(pos)
    variants = ["hvp_sumpol", "phv_sumpol"]
    candidates = [
        "derot_k16_p2",
        "derot_k4_p3",
        "blend_k24_a05",
        "blend_k24_a075",
        "blend_k24_a1",
        "blend_k8_a05",
        "map_alpha_k24",
        "map_reflect_k24_r025",
    ]
    metrics = {name: MultiMetric(variants) for name in candidates}

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
        map_alpha = direct1 if clear[ii] else direct025

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
            "blend_k24_a075": (0.25 * pred_cache[(24, 2.0, "derot")] + 0.75 * pred_cache[(24, 2.0, "steer")]).astype(np.complex64),
            "blend_k24_a1": pred_cache[(24, 2.0, "steer")],
            "blend_k8_a05": pred_cache[(8, 2.0, "blend05")],
            "map_alpha_k24": map_alpha,
            "map_reflect_k24_r025": map_reflect,
        }
        for name, pred in preds.items():
            metrics[name].add(pred, truth)
        if (ii + 1) % args.progress_every == 0 or ii + 1 == len(val_idx):
            print(f"evaluated {ii + 1}/{len(val_idx)}", flush=True)

    online = {
        "derot_k16_p2": 0.450782,
        "derot_k4_p3": 0.437548,
        "blend_k24_a05": 0.454453,
        "blend_k8_a05": 0.451313,
        "map_reflect_k24_r025": 0.450868,
    }
    results = []
    for name in candidates:
        row = {"candidate": name, "online": online.get(name)}
        row.update(metrics[name].result())
        results.append(row)
    return {
        "val_indices": val_idx.astype(int).tolist(),
        "test_count": int(len(test_pos)),
        "feature_names": feature_names,
        "feature_summary": feature_summary(train_feat, test_feat, val_idx, feature_names),
        "variants": variants,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--features", default=str(PATHS.artifacts / "shared_features/map_features.npz"))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/feature_matched_candidates_500.json"))
    parser.add_argument("--val-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--candidate-k", type=int, default=128)
    parser.add_argument("--max-k", type=int, default=24)
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--steering-scale", type=float, default=4.71238898038)
    parser.add_argument("--progress-every", type=int, default=20)
    args = parser.parse_args()

    map_feature_names = [
        "los_min_dist",
        "los_p05_dist",
        "los_p10_dist",
        "los_mean_inv_dist",
        "los_frac_r0p5",
        "los_frac_r0p75",
        "los_frac_r1p0",
        "los_frac_r1p5",
        "los_frac_r2p0",
        "los_frac_r3p0",
        "rx_wall_dist3",
        "rx_wall_dist2",
        "rx_wall_nx2",
        "rx_wall_ny2",
        "rx_wall_nz2",
    ]
    train_feat, test_feat, feature_names = make_query_features(args.data, args.features, map_feature_names)
    val_idx = choose_feature_matched_indices(train_feat, test_feat, args.val_count, args.seed, args.candidate_k)
    payload = evaluate_candidates(args, val_idx, train_feat, test_feat, feature_names)
    payload.update({
        "seed": int(args.seed),
        "val_count": int(args.val_count),
        "candidate_k": int(args.candidate_k),
        "selection": "feature_matched_test",
    })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    for variant in payload["variants"]:
        print(f"\nvariant={variant}", flush=True)
        for row in sorted(payload["results"], key=lambda r: r[f"score_{variant}"], reverse=True):
            print(
                f"{row['candidate']:24s} online={row['online']} score={row[f'score_{variant}']:.6f} "
                f"pas={row[f'pas_{variant}']:.6f} pdp={row['pdp']:.6f} nmse={row['nmse']:.6f}",
                flush=True,
            )
    print("\nworst feature mean gaps:", flush=True)
    for row in payload["feature_summary"][:10]:
        print(
            f"{row['feature']:24s} val-test-z={row['val_test_abs_zdiff']:.3f} "
            f"train-test-z={row['train_test_abs_zdiff']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
