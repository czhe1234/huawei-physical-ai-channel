#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from evaluate_derotated_loo_grid import choose_val_indices
from evaluate_local_ad_paths import BS_POS, idw_weights
from evaluate_reflection_steering import phase_from_units, reflection_lengths_and_units
from generate_geometric_steering_blend import steering_phase, unit_vectors
from spectrum_projection import cosine_last, exact_pdp_scores


MP, MH, MV = 2, 16, 8


def build_split(n, val_count, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return np.sort(perm[val_count:]), np.sort(perm[:val_count])


def make_split(pos, test_pos, mode, val_count, seed, k):
    if mode == "random":
        train_idx, val_idx = build_split(len(pos), val_count, seed)
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


def pas_scores(pred, truth, variant):
    if variant.startswith("phv"):
        pred_a = np.fft.fft2(pred.reshape(pred.shape[0], MP, MH, MV, pred.shape[2], pred.shape[3]), axes=(2, 3))
        truth_a = np.fft.fft2(truth.reshape(truth.shape[0], MP, MH, MV, truth.shape[2], truth.shape[3]), axes=(2, 3))
        if variant == "phv_sumpol":
            pred_p = np.sum(np.abs(pred_a) ** 2, axis=1)  # B,H,V,N,S
            truth_p = np.sum(np.abs(truth_a) ** 2, axis=1)
            pred_p = np.moveaxis(pred_p, (1, 2), (-2, -1)).reshape(pred.shape[0], pred.shape[2], pred.shape[3], MH * MV)
            truth_p = np.moveaxis(truth_p, (1, 2), (-2, -1)).reshape(pred.shape[0], pred.shape[2], pred.shape[3], MH * MV)
            scores = cosine_last(pred_p.reshape(-1, MH * MV), truth_p.reshape(-1, MH * MV))
            return scores.reshape(pred.shape[0], -1).mean(axis=1)
        if variant == "phv_fullpol":
            pred_p = np.abs(pred_a) ** 2
            truth_p = np.abs(truth_a) ** 2
            pred_p = pred_p.transpose(0, 4, 5, 1, 2, 3).reshape(pred.shape[0], pred.shape[2], pred.shape[3], MP * MH * MV)
            truth_p = truth_p.transpose(0, 4, 5, 1, 2, 3).reshape(pred.shape[0], pred.shape[2], pred.shape[3], MP * MH * MV)
            scores = cosine_last(pred_p.reshape(-1, MP * MH * MV), truth_p.reshape(-1, MP * MH * MV))
            return scores.reshape(pred.shape[0], -1).mean(axis=1)
        if variant == "phv_avgpol":
            pred_p = np.abs(pred_a) ** 2
            truth_p = np.abs(truth_a) ** 2
            pred_p = pred_p.transpose(0, 1, 4, 5, 2, 3).reshape(pred.shape[0], MP, pred.shape[2], pred.shape[3], MH * MV)
            truth_p = truth_p.transpose(0, 1, 4, 5, 2, 3).reshape(pred.shape[0], MP, pred.shape[2], pred.shape[3], MH * MV)
            scores = cosine_last(pred_p.reshape(-1, MH * MV), truth_p.reshape(-1, MH * MV))
            return scores.reshape(pred.shape[0], -1).mean(axis=1)
    if variant.startswith("hvp"):
        pred_a = np.fft.fft2(pred.reshape(pred.shape[0], MH, MV, MP, pred.shape[2], pred.shape[3]), axes=(1, 2))
        truth_a = np.fft.fft2(truth.reshape(truth.shape[0], MH, MV, MP, truth.shape[2], truth.shape[3]), axes=(1, 2))
        if variant == "hvp_sumpol":
            pred_p = np.sum(np.abs(pred_a) ** 2, axis=3)  # B,H,V,N,S
            truth_p = np.sum(np.abs(truth_a) ** 2, axis=3)
            pred_p = np.moveaxis(pred_p, (1, 2), (-2, -1)).reshape(pred.shape[0], pred.shape[2], pred.shape[3], MH * MV)
            truth_p = np.moveaxis(truth_p, (1, 2), (-2, -1)).reshape(pred.shape[0], pred.shape[2], pred.shape[3], MH * MV)
            scores = cosine_last(pred_p.reshape(-1, MH * MV), truth_p.reshape(-1, MH * MV))
            return scores.reshape(pred.shape[0], -1).mean(axis=1)
        if variant == "hvp_fullpol":
            pred_p = np.abs(pred_a) ** 2
            truth_p = np.abs(truth_a) ** 2
            pred_p = pred_p.transpose(0, 4, 5, 1, 2, 3).reshape(pred.shape[0], pred.shape[2], pred.shape[3], MH * MV * MP)
            truth_p = truth_p.transpose(0, 4, 5, 1, 2, 3).reshape(pred.shape[0], pred.shape[2], pred.shape[3], MH * MV * MP)
            scores = cosine_last(pred_p.reshape(-1, MH * MV * MP), truth_p.reshape(-1, MH * MV * MP))
            return scores.reshape(pred.shape[0], -1).mean(axis=1)
        if variant == "hvp_avgpol":
            pred_p = np.abs(pred_a) ** 2
            truth_p = np.abs(truth_a) ** 2
            pred_p = pred_p.transpose(0, 3, 4, 5, 1, 2).reshape(pred.shape[0], MP, pred.shape[2], pred.shape[3], MH * MV)
            truth_p = truth_p.transpose(0, 3, 4, 5, 1, 2).reshape(pred.shape[0], MP, pred.shape[2], pred.shape[3], MH * MV)
            scores = cosine_last(pred_p.reshape(-1, MH * MV), truth_p.reshape(-1, MH * MV))
            return scores.reshape(pred.shape[0], -1).mean(axis=1)
    raise ValueError(variant)


class MultiMetric:
    def __init__(self, variants):
        self.variants = variants
        self.num = 0.0
        self.den = 0.0
        self.pdp = []
        self.pas = {v: [] for v in variants}

    def add(self, pred, truth):
        self.num += float(np.sum(np.abs(pred - truth) ** 2, dtype=np.float64))
        self.den += float(np.sum(np.abs(truth) ** 2, dtype=np.float64))
        self.pdp.append(exact_pdp_scores(pred, truth))
        for v in self.variants:
            self.pas[v].append(pas_scores(pred, truth, v))

    def result(self):
        nmse = self.num / max(self.den, 1e-30)
        pdp = float(np.mean(np.concatenate(self.pdp)))
        out = {"nmse": float(nmse), "pdp": pdp, "inv_nmse": float(1.0 / (1.0 + nmse))}
        for v in self.variants:
            pas = float(np.mean(np.concatenate(self.pas[v])))
            out[f"pas_{v}"] = pas
            out[f"score_{v}"] = 0.4 * pas + 0.4 * pdp + 0.2 / (1.0 + nmse)
        return out


def load_map_geometry(features_path, val_idx):
    data = np.load(features_path, allow_pickle=True)
    features = data["train_features"][val_idx]
    names = [str(x) for x in data["feature_names"]]
    clear = features[:, names.index("los_min_dist")] >= 0.5
    plane_points = features[:, [names.index("rx_wall_x2"), names.index("rx_wall_y2"), names.index("rx_wall_z2")]]
    plane_normals = features[:, [names.index("rx_wall_nx2"), names.index("rx_wall_ny2"), names.index("rx_wall_nz2")]]
    return clear, plane_points, plane_normals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--features", default=str(PATHS.artifacts / "shared_features/map_features.npz"))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/metric_variants.json"))
    parser.add_argument("--mode", choices=["random", "matched-test"], default="matched-test")
    parser.add_argument("--val-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--max-k", type=int, default=24)
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--steering-scale", type=float, default=4.71238898038)
    args = parser.parse_args()

    data = Path(args.data)
    pos = np.load(data / "Round1_Train_Pos.npy")
    test_pos = np.load(data / "Round1_Test_Pos.npy")
    h = np.load(data / "Round1_Train_Channel.npy", mmap_mode="r")
    val_idx, distances, neighbor_idx = make_split(pos, test_pos, args.mode, args.val_count, args.seed, args.max_k)
    clear, plane_points, plane_normals = load_map_geometry(args.features, val_idx)
    dist_all = np.linalg.norm(pos - BS_POS[None, :], axis=1)
    u_all = unit_vectors(pos)

    variants = ["hvp_sumpol", "hvp_fullpol", "hvp_avgpol", "phv_sumpol", "phv_fullpol", "phv_avgpol"]
    candidates = [
        "derot_k16_p2",
        "derot_k4_p3",
        "blend_k24_a05",
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
            "blend_k8_a05": pred_cache[(8, 2.0, "blend05")],
            "map_alpha_k24": map_alpha,
            "map_reflect_k24_r025": map_reflect,
        }
        for name, pred in preds.items():
            metrics[name].add(pred, truth)
        if (ii + 1) % 20 == 0 or ii + 1 == len(val_idx):
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
    payload = {
        "mode": args.mode,
        "seed": int(args.seed),
        "val_count": int(args.val_count),
        "variants": variants,
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    for v in variants:
        print(f"\nvariant={v}", flush=True)
        for row in sorted(results, key=lambda r: r[f"score_{v}"], reverse=True):
            print(
                f"{row['candidate']:24s} online={row['online']} score={row[f'score_{v}']:.6f} "
                f"pas={row[f'pas_{v}']:.6f} pdp={row['pdp']:.6f} nmse={row['nmse']:.6f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
