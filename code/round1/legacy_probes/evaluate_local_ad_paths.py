#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from spectrum_projection import MetricAccumulator


BS_POS = np.array([50.0, 0.0, 25.0], dtype=np.float64)
MP, MH, MV = 2, 16, 8
N_ANT, S_SC = 4, 192


def parse_ints(text):
    return [int(x) for x in str(text).split(",") if x]


def parse_floats(text):
    return [float(x) for x in str(text).split(",") if x]


def parse_words(text):
    return [x for x in str(text).split(",") if x]


def build_split(n, val_count, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return np.sort(perm[val_count:]), np.sort(perm[:val_count])


def idw_weights(distances, power):
    d = np.asarray(distances, dtype=np.float64)
    if d[0] < 1e-12:
        w = np.zeros_like(d)
        w[0] = 1.0
        return w.astype(np.float32)
    w = 1.0 / np.maximum(d, 1e-6) ** power
    w /= np.sum(w)
    return w.astype(np.float32)


def ad_transform(h):
    # M flatten order is treated as (P,H,V), which matched the geometric steering experiments.
    cir = np.fft.ifft(h, axis=-1)
    arr = cir.reshape(h.shape[0], MP, MH, MV, N_ANT, S_SC)
    return np.fft.fft2(arr, axes=(2, 3)).astype(np.complex64)


def inverse_ad(ad):
    arr = np.fft.ifft2(ad, axes=(2, 3))
    cir = arr.reshape(ad.shape[0], MP * MH * MV, N_ANT, S_SC)
    return np.fft.fft(cir, axis=-1).astype(np.complex64)


def flat_components(component_ids):
    h = component_ids // (MV * S_SC)
    rem = component_ids % (MV * S_SC)
    v = rem // S_SC
    s = rem % S_SC
    return h.astype(np.int64), v.astype(np.int64), s.astype(np.int64)


def top_components(power, count):
    flat = power.reshape(-1)
    count = min(count, flat.size)
    ids = np.argpartition(flat, -count)[-count:]
    ids = ids[np.argsort(flat[ids])[::-1]]
    return ids.astype(np.int64)


def select_components(neigh_power, weights, mode, per_neighbor, total_count):
    if mode == "weighted":
        score = np.tensordot(weights.astype(np.float32), neigh_power, axes=(0, 0))
        return top_components(score, total_count)
    if mode == "nearest":
        return top_components(neigh_power[0], total_count)
    if mode == "union":
        ids = []
        for j in range(neigh_power.shape[0]):
            ids.extend(top_components(neigh_power[j], per_neighbor).tolist())
        if not ids:
            return np.empty(0, dtype=np.int64)
        ids = np.unique(np.asarray(ids, dtype=np.int64))
        if ids.size > total_count:
            score = np.tensordot(weights.astype(np.float32), neigh_power, axes=(0, 0)).reshape(-1)
            ids = ids[np.argsort(score[ids])[::-1][:total_count]]
        return ids.astype(np.int64)
    raise ValueError(mode)


def predict_sparse_ad(neigh_ad, weights, component_ids, align):
    pred_ad = np.zeros((1, MP, MH, MV, N_ANT, S_SC), dtype=np.complex64)
    if component_ids.size == 0:
        return pred_ad
    hh, vv, ss = flat_components(component_ids)
    coeff = np.empty((neigh_ad.shape[0], component_ids.size, MP, N_ANT), dtype=np.complex64)
    for local, (h_idx, v_idx, s_idx) in enumerate(zip(hh, vv, ss)):
        coeff[:, local] = neigh_ad[:, :, h_idx, v_idx, :, s_idx]
    if align and coeff.shape[0] > 1:
        ref = coeff[0].reshape(coeff.shape[1], -1)
        for j in range(1, coeff.shape[0]):
            cur = coeff[j].reshape(coeff.shape[1], -1)
            corr = np.sum(cur * np.conj(ref), axis=1)
            phase = np.ones_like(corr, dtype=np.complex64)
            good = np.abs(corr) > 1e-30
            phase[good] = (corr[good] / np.abs(corr[good])).astype(np.complex64)
            coeff[j] *= phase[:, None, None]
    mixed = np.tensordot(weights.astype(np.float32), coeff, axes=(0, 0)).astype(np.complex64)
    for local, comp in enumerate(component_ids):
        h, v, s = flat_components(np.asarray([comp]))
        pred_ad[0, :, h[0], v[0], :, s[0]] = mixed[local]
    return pred_ad


def full_derot_idw(neigh_h, weights, neighbor_dist, query_dist, beta):
    phase = np.exp(1j * beta * (neighbor_dist - query_dist)).astype(np.complex64)
    coeff = weights.astype(np.float32) * phase
    return np.sum(coeff[:, None, None, None] * neigh_h, axis=0, keepdims=True).astype(np.complex64)


def eval_one(truth, pred):
    diff = pred - truth
    return (
        float(np.sum(np.abs(diff) ** 2, dtype=np.float64)),
        float(np.sum(np.abs(truth) ** 2, dtype=np.float64)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/local_ad_paths.json"))
    parser.add_argument("--val-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--k-values", type=parse_ints, default="4,8")
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--component-counts", type=parse_ints, default="128,256,512")
    parser.add_argument("--per-neighbor", type=int, default=64)
    parser.add_argument("--modes", type=parse_words, default="weighted,nearest,union")
    parser.add_argument("--align-options", type=parse_words, default="no,yes")
    parser.add_argument("--derot-options", type=parse_words, default="yes,no")
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()

    pos = np.load(os.path.join(args.data, "Round1_Train_Pos.npy"))
    h = np.load(os.path.join(args.data, "Round1_Train_Channel.npy"), mmap_mode="r")
    train_idx, val_idx = build_split(len(pos), args.val_count, args.seed)
    max_k = max(args.k_values)
    tree = cKDTree(pos[train_idx, :2])
    distances, local = tree.query(pos[val_idx, :2], k=max_k, workers=-1)
    if max_k == 1:
        distances = distances[:, None]
        local = local[:, None]
    neighbor_idx = train_idx[local]
    dist_all = np.linalg.norm(pos - BS_POS[None, :], axis=1)

    candidates = []
    for k in args.k_values:
        candidates.append(("full_derot", k, 0, "weighted", False, True))
        for comp_count in args.component_counts:
            for mode in args.modes:
                for align_text in args.align_options:
                    for derot_text in args.derot_options:
                        candidates.append((f"ad_{mode}", k, comp_count, mode, align_text == "yes", derot_text == "yes"))

    stats = {str(c): {"num": 0.0, "den": 0.0} for c in candidates}
    accs = {str(c): MetricAccumulator("sumpol") for c in candidates}

    for ii, vi in enumerate(val_idx):
        truth = np.asarray(h[vi:vi + 1])
        for k in args.k_values:
            idx = neighbor_idx[ii, :k]
            neigh_h = np.asarray(h[idx])
            w = idw_weights(distances[ii, :k], args.power)
            base = full_derot_idw(neigh_h, w, dist_all[idx], dist_all[vi], args.beta)
            key = str(("full_derot", k, 0, "weighted", False, True))
            num, den = eval_one(truth, base)
            stats[key]["num"] += num
            stats[key]["den"] += den
            accs[key].add(base, truth)

            for derot in [text == "yes" for text in args.derot_options]:
                if derot:
                    neigh_work = neigh_h * np.exp(1j * args.beta * dist_all[idx])[:, None, None, None].astype(np.complex64)
                else:
                    neigh_work = neigh_h
                neigh_ad = ad_transform(neigh_work)
                neigh_power = np.sum(np.abs(neigh_ad) ** 2, axis=(1, 4), dtype=np.float64)  # k,H,V,S
                for comp_count in args.component_counts:
                    for mode in args.modes:
                        component_ids = select_components(neigh_power, w, mode, args.per_neighbor, comp_count)
                        for align in [text == "yes" for text in args.align_options]:
                            pred_ad = predict_sparse_ad(neigh_ad, w, component_ids, align)
                            pred = inverse_ad(pred_ad)
                            if derot:
                                pred *= np.exp(-1j * args.beta * dist_all[vi]).astype(np.complex64)
                            key = str((f"ad_{mode}", k, comp_count, mode, align, derot))
                            num, den = eval_one(truth, pred)
                            stats[key]["num"] += num
                            stats[key]["den"] += den
                            accs[key].add(pred, truth)
        if (ii + 1) % 10 == 0 or ii + 1 == len(val_idx):
            print(f"evaluated {ii + 1}/{len(val_idx)}", flush=True)

    results = []
    for candidate in candidates:
        key = str(candidate)
        den = stats[key]["den"]
        nmse = stats[key]["num"] / max(den, 1e-30)
        metric = accs[key].result()
        results.append({
            "candidate": candidate[0],
            "k": int(candidate[1]),
            "component_count": int(candidate[2]),
            "mode": candidate[3],
            "align": bool(candidate[4]),
            "derot": bool(candidate[5]),
            "global_nmse_direct": float(nmse),
            **metric,
        })
    results.sort(key=lambda r: r["global_nmse"])
    payload = {
        "seed": int(args.seed),
        "val_count": int(args.val_count),
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    for row in results[:30]:
        print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
