#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


BS_POS = np.array([50.0, 0.0, 25.0], dtype=np.float64)
MH, MV, MP = 16, 8, 2
NV, NP = 2, 2


def build_split(n, val_count, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return np.sort(perm[val_count:]), np.sort(perm[:val_count])


def parse_ints(text):
    return [int(x) for x in str(text).split(",") if x]


def parse_floats(text):
    return [float(x) for x in str(text).split(",") if x]


def idw_weights(distances, power):
    d = np.asarray(distances, dtype=np.float64)
    if d[0] < 1e-12:
        w = np.zeros_like(d)
        w[0] = 1.0
        return w.astype(np.float32)
    w = 1.0 / np.maximum(d, 1e-6) ** power
    w /= np.sum(w)
    return w.astype(np.float32)


def unit_vectors(pos):
    vec = pos - BS_POS[None, :]
    return vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-30)


def axis_value(u, name):
    sign = -1.0 if name.startswith("-") or name.startswith("n") else 1.0
    key = name[1:] if name.startswith("-") else name
    if name.startswith("n") and name not in ("none",):
        key = name[1:]
    if key == "none":
        return np.zeros(len(u), dtype=np.float64)
    if key == "ux":
        return sign * u[:, 0]
    if key == "uy":
        return sign * u[:, 1]
    if key == "uz":
        return sign * u[:, 2]
    if key == "azsin":
        xy = np.maximum(np.linalg.norm(u[:, :2], axis=1), 1e-30)
        return sign * u[:, 1] / xy
    if key == "azcos":
        xy = np.maximum(np.linalg.norm(u[:, :2], axis=1), 1e-30)
        return sign * u[:, 0] / xy
    raise ValueError(name)


def bs_indices(order):
    if order == "phv":
        arr = np.empty((MP, MH, MV, 2), dtype=np.float64)
        for p in range(MP):
            for h in range(MH):
                for v in range(MV):
                    arr[p, h, v] = (h - (MH - 1) / 2.0, v - (MV - 1) / 2.0)
        return arr.reshape(-1, 2)
    if order == "hvp":
        arr = np.empty((MH, MV, MP, 2), dtype=np.float64)
        for h in range(MH):
            for v in range(MV):
                for p in range(MP):
                    arr[h, v, p] = (h - (MH - 1) / 2.0, v - (MV - 1) / 2.0)
        return arr.reshape(-1, 2)
    if order == "hvw":
        return bs_indices("hvp")
    if order == "vhp":
        arr = np.empty((MV, MH, MP, 2), dtype=np.float64)
        for v in range(MV):
            for h in range(MH):
                for p in range(MP):
                    arr[v, h, p] = (h - (MH - 1) / 2.0, v - (MV - 1) / 2.0)
        return arr.reshape(-1, 2)
    raise ValueError(order)


def ue_indices():
    arr = np.empty((NV, NP), dtype=np.float64)
    for v in range(NV):
        for p in range(NP):
            arr[v, p] = v - (NV - 1) / 2.0
    return arr.reshape(-1)


def make_configs(args):
    configs = [{"name": "baseline", "order": "phv", "h_axis": "none", "v_axis": "none", "bs_scale": 0.0, "ue_scale": 0.0}]
    for order in args.orders:
        for h_axis in args.h_axes:
            for v_axis in args.v_axes:
                for bs_scale in args.bs_scales:
                    if bs_scale == 0.0:
                        continue
                    configs.append({
                        "name": f"bs_{order}_h{h_axis}_v{v_axis}_s{bs_scale:g}",
                        "order": order,
                        "h_axis": h_axis,
                        "v_axis": v_axis,
                        "bs_scale": float(bs_scale),
                        "ue_scale": 0.0,
                    })
    if args.include_ue:
        base = list(configs[1:])
        for cfg in base:
            for ue_scale in args.ue_scales:
                c = dict(cfg)
                c["ue_scale"] = float(ue_scale)
                c["name"] = c["name"] + f"_ue{ue_scale:g}"
                configs.append(c)
    return configs


def predict_config(neigh, weights, radial, u_all, idx, qidx, cfg):
    coeff = weights * radial
    out = coeff[:, None, None, None] * neigh
    if cfg["bs_scale"] != 0.0:
        ant = bs_indices(cfg["order"])
        h_coord = ant[:, 0]
        v_coord = ant[:, 1]
        h_val = axis_value(u_all, cfg["h_axis"])
        v_val = axis_value(u_all, cfg["v_axis"])
        delta_h = h_val[qidx] - h_val[idx]
        delta_v = v_val[qidx] - v_val[idx]
        phase_arg = cfg["bs_scale"] * (
            delta_h[:, None] * h_coord[None, :] + delta_v[:, None] * v_coord[None, :]
        )
        phase = np.exp(1j * phase_arg).astype(np.complex64)
        out = out * phase[:, :, None, None]
    if cfg["ue_scale"] != 0.0:
        ant_n = ue_indices()
        v_val = axis_value(u_all, cfg["v_axis"])
        delta_v = v_val[qidx] - v_val[idx]
        phase_n = np.exp(1j * cfg["ue_scale"] * delta_v[:, None] * ant_n[None, :]).astype(np.complex64)
        out = out * phase_n[:, None, :, None]
    return np.sum(out, axis=0).astype(np.complex64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/geometric_steering.json"))
    parser.add_argument("--val-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--orders", nargs="+", default=["phv", "hvp", "vhp"])
    parser.add_argument("--h-axes", nargs="+", default=["ux", "nux", "uy", "nuy", "azsin", "nazsin"])
    parser.add_argument("--v-axes", nargs="+", default=["none", "uz", "nuz"])
    parser.add_argument("--bs-scales", type=parse_floats, default="1.57079632679,3.14159265359,-1.57079632679,-3.14159265359")
    parser.add_argument("--include-ue", action="store_true")
    parser.add_argument("--ue-scales", type=parse_floats, default="3.14159265359,-3.14159265359")
    args = parser.parse_args()

    pos = np.load(os.path.join(args.data, "Round1_Train_Pos.npy"))
    h = np.load(os.path.join(args.data, "Round1_Train_Channel.npy"), mmap_mode="r")
    train_idx, val_idx = build_split(len(pos), args.val_count, args.seed)
    tree = cKDTree(pos[train_idx, :2])
    distances, local = tree.query(pos[val_idx, :2], k=args.k, workers=-1)
    if args.k == 1:
        distances = distances[:, None]
        local = local[:, None]
    neighbor_idx = train_idx[local]
    dist_all = np.linalg.norm(pos - BS_POS[None, :], axis=1)
    u_all = unit_vectors(pos)
    configs = make_configs(args)
    nums = np.zeros(len(configs), dtype=np.float64)
    dens = np.zeros(len(configs), dtype=np.float64)

    for c, vi in enumerate(val_idx):
        idx = neighbor_idx[c]
        neigh = np.asarray(h[idx])
        truth = np.asarray(h[vi])
        den = float(np.sum(np.abs(truth) ** 2, dtype=np.float64))
        weights = idw_weights(distances[c], args.power)
        radial = np.exp(1j * args.beta * (dist_all[idx] - dist_all[vi])).astype(np.complex64)
        for j, cfg in enumerate(configs):
            pred = predict_config(neigh, weights, radial, u_all, idx, vi, cfg)
            diff = pred - truth
            nums[j] += float(np.sum(np.abs(diff) ** 2, dtype=np.float64))
            dens[j] += den
        if (c + 1) % 10 == 0 or c + 1 == len(val_idx):
            print(f"evaluated {c + 1}/{len(val_idx)}", flush=True)

    results = []
    for cfg, num, den in zip(configs, nums, dens):
        nmse = float(num / max(den, 1e-30))
        row = dict(cfg)
        row.update({
            "k": int(args.k),
            "power": float(args.power),
            "beta": float(args.beta),
            "global_nmse": nmse,
            "inv_nmse": float(1.0 / (1.0 + nmse)),
        })
        results.append(row)
    results.sort(key=lambda r: r["global_nmse"])
    payload = {"seed": int(args.seed), "val_count": int(args.val_count), "results": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    for row in results[:30]:
        print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
