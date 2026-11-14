#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


MH, MV, MP = 16, 8, 2
NH, NV, NP = 1, 2, 2


def delay_transform(h):
    return np.fft.ifft(h, axis=-1).astype(np.complex64)


def build_split(n, val_count, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return np.sort(perm[val_count:]), np.sort(perm[:val_count])


def idw_weights(d, power):
    d = np.asarray(d, dtype=np.float64)
    if d[0] < 1e-12:
        w = np.zeros_like(d)
        w[0] = 1.0
        return w
    w = 1.0 / np.maximum(d, 1e-6) ** power
    return w / np.sum(w)


def affine_weights(query_xy, neigh_xy, distances, power, ridge, clip_negative):
    base = idw_weights(distances, power)
    dx = np.asarray(neigh_xy, dtype=np.float64) - np.asarray(query_xy, dtype=np.float64)[None, :]
    a = np.vstack([np.ones(len(base), dtype=np.float64), dx.T])
    b = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    mat = (a * base[None, :]) @ a.T
    mat.flat[::4] += ridge
    try:
        lag = np.linalg.solve(mat, b)
        w = base * (a.T @ lag)
    except np.linalg.LinAlgError:
        w = base.copy()
    if clip_negative:
        w = np.maximum(w, 0.0)
        s = np.sum(w)
        if s <= 1e-30:
            return base
        w = w / s
    return w.astype(np.float64)


def select_taps(train_h, train_idx, tap_count, batch):
    energy = np.zeros(train_h.shape[-1], dtype=np.float64)
    for start in range(0, len(train_idx), batch):
        idx = train_idx[start:start + batch]
        cir = delay_transform(np.asarray(train_h[idx]))
        energy += np.sum(np.abs(cir) ** 2, axis=(0, 1, 2), dtype=np.float64)
    taps = np.sort(np.argpartition(energy, -tap_count)[-tap_count:])
    return taps, float(energy[taps].sum() / max(energy.sum(), 1e-30))


def nmse_batch(pred, truth):
    diff = pred - truth
    num = np.sum(np.abs(diff) ** 2, axis=(1, 2, 3), dtype=np.float64)
    den = np.sum(np.abs(truth) ** 2, axis=(1, 2, 3), dtype=np.float64)
    return num / np.maximum(den, 1e-30)


def cosine_rows(a, b):
    a = a.reshape(a.shape[0], -1).astype(np.float64, copy=False)
    b = b.reshape(b.shape[0], -1).astype(np.float64, copy=False)
    num = np.sum(a * b, axis=1)
    den = np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1))
    return num / np.maximum(den, 1e-30)


def pdp_batch(h):
    ht = np.fft.ifft(h, axis=-1)
    return np.sum(np.abs(ht) ** 2, axis=(1, 2))


def pas_batch(h):
    arr = h.reshape(h.shape[0], MH, MV, MP, NH, NV, NP, h.shape[-1])
    ang = np.fft.fftn(arr, axes=(1, 2, 4, 5))
    return np.sum(np.abs(ang) ** 2, axis=(3, 6, 7)).reshape(h.shape[0], -1)


def summarize(nmse, pdp, pas, power):
    score = 0.4 * pas + 0.4 * pdp + 0.2 / (1.0 + nmse)
    return {
        'score_mean': float(np.mean(score)),
        'pas_mean': float(np.mean(pas)),
        'pdp_mean': float(np.mean(pdp)),
        'nmse_mean': float(np.mean(nmse)),
        'nmse_median': float(np.median(nmse)),
        'nmse_q90': float(np.quantile(nmse, 0.9)),
        'nmse_power_weighted': float(np.sum(nmse * power) / np.sum(power)),
    }


def predict_full(train_h, train_idx, neighbor_idx, weights):
    n = neighbor_idx.shape[0]
    pred = np.empty((n, 256, 4, 192), dtype=np.complex64)
    for i in range(n):
        neigh = np.asarray(train_h[train_idx[neighbor_idx[i]]])
        pred[i] = np.tensordot(weights[i].astype(np.float32), neigh, axes=(0, 0)).astype(np.complex64)
    return pred


def predict_delay_taps(train_h, train_idx, neighbor_idx, weights, taps):
    n = neighbor_idx.shape[0]
    pred_cir = np.zeros((n, 256, 4, 192), dtype=np.complex64)
    for i in range(n):
        neigh_cir = delay_transform(np.asarray(train_h[train_idx[neighbor_idx[i]]]))[..., taps]
        pred_cir[i][..., taps] = np.tensordot(weights[i].astype(np.float32), neigh_cir, axes=(0, 0)).astype(np.complex64)
    return np.fft.fft(pred_cir, axis=-1).astype(np.complex64)


def make_weights(pos_train_xy, query_xy, neighbor_idx, distances, method, power, ridge, clip_negative):
    weights = np.empty(neighbor_idx.shape, dtype=np.float64)
    for i in range(neighbor_idx.shape[0]):
        if method == 'idw':
            weights[i] = idw_weights(distances[i], power)
        elif method == 'affine':
            weights[i] = affine_weights(query_xy[i], pos_train_xy[neighbor_idx[i]], distances[i], power, ridge, clip_negative)
        else:
            raise ValueError(method)
    return weights


def evaluate(args):
    pos = np.load(os.path.join(args.data, 'Round1_Train_Pos.npy'))
    train_h = np.load(os.path.join(args.data, 'Round1_Train_Channel.npy'), mmap_mode='r')
    train_idx, val_idx = build_split(len(pos), args.val_count, args.seed)
    train_xy = pos[train_idx, :2]
    val_xy = pos[val_idx, :2]
    tree = cKDTree(train_xy)
    taps_by_count = {}
    results = []
    for mode in args.modes:
        tap_items = [0]
        if mode == 'delay':
            tap_items = args.tap_counts
        for tap_count in tap_items:
            taps = None
            frac = None
            if mode == 'delay':
                if tap_count not in taps_by_count:
                    taps_by_count[tap_count] = select_taps(train_h, train_idx, tap_count, args.batch)
                taps, frac = taps_by_count[tap_count]
            for k in args.k_values:
                distances, neighbor_idx = tree.query(val_xy, k=k, workers=-1)
                if k == 1:
                    distances = distances[:, None]
                    neighbor_idx = neighbor_idx[:, None]
                for method in args.methods:
                    nmse_all, pdp_all, pas_all, power_all = [], [], [], []
                    for start in range(0, len(val_idx), args.eval_batch):
                        end = min(start + args.eval_batch, len(val_idx))
                        w = make_weights(
                            train_xy,
                            val_xy[start:end],
                            neighbor_idx[start:end],
                            distances[start:end],
                            method,
                            args.power,
                            args.ridge,
                            args.clip_negative,
                        )
                        if mode == 'full':
                            pred = predict_full(train_h, train_idx, neighbor_idx[start:end], w)
                        else:
                            pred = predict_delay_taps(train_h, train_idx, neighbor_idx[start:end], w, taps)
                        truth = np.asarray(train_h[val_idx[start:end]])
                        nmse_all.append(nmse_batch(pred, truth))
                        pdp_all.append(cosine_rows(pdp_batch(pred), pdp_batch(truth)))
                        pas_all.append(cosine_rows(pas_batch(pred), pas_batch(truth)))
                        power_all.append(np.sum(np.abs(truth) ** 2, axis=(1, 2, 3), dtype=np.float64))
                    row = {
                        'mode': mode,
                        'method': method,
                        'tap_count': int(tap_count) if mode == 'delay' else None,
                        'taps': taps.tolist() if taps is not None else None,
                        'tap_energy_fraction_train': frac,
                        'k': int(k),
                        'power': float(args.power),
                        'ridge': float(args.ridge),
                        'clip_negative': bool(args.clip_negative),
                        **summarize(np.concatenate(nmse_all), np.concatenate(pdp_all), np.concatenate(pas_all), np.concatenate(power_all)),
                    }
                    results.append(row)
                    print(json.dumps(row, indent=2), flush=True)
    return results


def parse_int_list(text):
    return [int(x) for x in text.split(',') if x]


def parse_str_list(text):
    return [x for x in text.split(',') if x]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=str(PATHS.round1))
    parser.add_argument('--out', default=str(PATHS.analysis / 'results/json/local_linear_results.json'))
    parser.add_argument('--val-count', type=int, default=400)
    parser.add_argument('--seed', type=int, default=20260703)
    parser.add_argument('--modes', type=parse_str_list, default='full,delay')
    parser.add_argument('--methods', type=parse_str_list, default='affine')
    parser.add_argument('--tap-counts', type=parse_int_list, default='8,16,24')
    parser.add_argument('--k-values', type=parse_int_list, default='3,5,8,12,16')
    parser.add_argument('--power', type=float, default=2.0)
    parser.add_argument('--ridge', type=float, default=1e-8)
    parser.add_argument('--clip-negative', action='store_true')
    parser.add_argument('--batch', type=int, default=50)
    parser.add_argument('--eval-batch', type=int, default=10)
    args = parser.parse_args()
    result = evaluate(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()
