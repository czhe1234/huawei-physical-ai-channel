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


def predict_delay_tap_idw(train_h, train_idx, neighbor_idx, distances, taps, power, align_phase=False):
    n = neighbor_idx.shape[0]
    pred_cir = np.zeros((n, 256, 4, 192), dtype=np.complex64)
    for i in range(n):
        neigh_cir = delay_transform(np.asarray(train_h[train_idx[neighbor_idx[i]]]))[..., taps]
        if align_phase and neigh_cir.shape[0] > 1:
            ref = neigh_cir[0].reshape(-1)
            for j in range(1, neigh_cir.shape[0]):
                c = np.vdot(neigh_cir[j].reshape(-1), ref)
                if abs(c) > 1e-30:
                    neigh_cir[j] *= np.complex64(c / abs(c))
        w = idw_weights(distances[i], power).astype(np.float32)
        pred_cir[i][..., taps] = np.tensordot(w, neigh_cir, axes=(0, 0)).astype(np.complex64)
    return np.fft.fft(pred_cir, axis=-1).astype(np.complex64)


def evaluate(args):
    pos = np.load(os.path.join(args.data, 'Round1_Train_Pos.npy'))
    train_h = np.load(os.path.join(args.data, 'Round1_Train_Channel.npy'), mmap_mode='r')
    train_idx, val_idx = build_split(len(pos), args.val_count, args.seed)
    tree = cKDTree(pos[train_idx, :2])
    results = []
    for tap_count in args.tap_counts:
        taps, frac = select_taps(train_h, train_idx, tap_count, args.batch)
        for k in args.k_values:
            distances, neighbor_idx = tree.query(pos[val_idx, :2], k=k, workers=-1)
            if k == 1:
                distances = distances[:, None]
                neighbor_idx = neighbor_idx[:, None]
            nmse_all = []
            pdp_all = []
            pas_all = []
            power_all = []
            for start in range(0, len(val_idx), args.eval_batch):
                end = min(start + args.eval_batch, len(val_idx))
                pred = predict_delay_tap_idw(
                    train_h,
                    train_idx,
                    neighbor_idx[start:end],
                    distances[start:end],
                    taps,
                    args.power,
                    align_phase=args.align_phase,
                )
                truth = np.asarray(train_h[val_idx[start:end]])
                nmse_all.append(nmse_batch(pred, truth))
                pdp_all.append(cosine_rows(pdp_batch(pred), pdp_batch(truth)))
                pas_all.append(cosine_rows(pas_batch(pred), pas_batch(truth)))
                power_all.append(np.sum(np.abs(truth) ** 2, axis=(1, 2, 3), dtype=np.float64))
            metrics = summarize(
                np.concatenate(nmse_all),
                np.concatenate(pdp_all),
                np.concatenate(pas_all),
                np.concatenate(power_all),
            )
            row = {
                'tap_count': int(tap_count),
                'taps': taps.tolist(),
                'tap_energy_fraction_train': frac,
                'k': int(k),
                'power': float(args.power),
                'align_phase': bool(args.align_phase),
                **metrics,
            }
            results.append(row)
            print(json.dumps(row, indent=2), flush=True)
    return results


def generate(args):
    pos = np.load(os.path.join(args.data, 'Round1_Train_Pos.npy'))
    test_pos = np.load(os.path.join(args.data, 'Round1_Test_Pos.npy'))
    train_h = np.load(os.path.join(args.data, 'Round1_Train_Channel.npy'), mmap_mode='r')
    train_idx = np.arange(len(pos))
    taps, frac = select_taps(train_h, train_idx, args.tap_count, args.batch)
    tree = cKDTree(pos[:, :2])
    distances, neighbor_idx = tree.query(test_pos[:, :2], k=args.k, workers=-1)
    if args.k == 1:
        distances = distances[:, None]
        neighbor_idx = neighbor_idx[:, None]
    out = np.lib.format.open_memmap(args.output, mode='w+', dtype=np.complex64, shape=(len(test_pos), 256, 4, 192))
    for start in range(0, len(test_pos), args.eval_batch):
        end = min(start + args.eval_batch, len(test_pos))
        pred = predict_delay_tap_idw(
            train_h,
            train_idx,
            neighbor_idx[start:end],
            distances[start:end],
            taps,
            args.power,
            align_phase=args.align_phase,
        )
        out[start:end] = pred
        print(f'wrote {end}/{len(test_pos)}', flush=True)
    out.flush()
    return {
        'tap_count': args.tap_count,
        'taps': taps.tolist(),
        'tap_energy_fraction_train': frac,
        'k': args.k,
        'power': args.power,
        'align_phase': bool(args.align_phase),
        'output': args.output,
    }


def parse_int_list(text):
    return [int(x) for x in text.split(',') if x]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=str(PATHS.round1))
    parser.add_argument('--out', default=str(PATHS.analysis / 'results/json/delay_tap_idw_results.json'))
    parser.add_argument('--val-count', type=int, default=400)
    parser.add_argument('--seed', type=int, default=20260703)
    parser.add_argument('--tap-counts', type=parse_int_list, default='4,8,12,16,24,32')
    parser.add_argument('--k-values', type=parse_int_list, default='1,2,3,5,8')
    parser.add_argument('--power', type=float, default=2.0)
    parser.add_argument('--align-phase', action='store_true')
    parser.add_argument('--batch', type=int, default=50)
    parser.add_argument('--eval-batch', type=int, default=10)
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--tap-count', type=int, default=8)
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--output', default=str(PATHS.temporary / "Round1_Test_Channel.npy"))
    args = parser.parse_args()

    if args.generate:
        result = generate(args)
    else:
        result = evaluate(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()
