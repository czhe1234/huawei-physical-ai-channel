#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import os

import numpy as np
from scipy.spatial import cKDTree


C = 299792458.0
BS_POS = np.array([50.0, 0.0, 25.0])


def nmse(pred, truth):
    den = np.vdot(truth.ravel(), truth.ravel()).real
    if den == 0:
        return 0.0 if np.vdot(pred.ravel(), pred.ravel()).real == 0 else np.inf
    return float(np.vdot((pred - truth).ravel(), (pred - truth).ravel()).real / den)


def best_complex_gain(pred, truth):
    den = np.vdot(pred.ravel(), pred.ravel())
    if abs(den) == 0:
        return 0.0 + 0.0j
    return np.vdot(pred.ravel(), truth.ravel()) / den


def fft_delay(h):
    return np.fft.ifft(h, axis=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=str(PATHS.round1))
    parser.add_argument('--val-count', type=int, default=200)
    parser.add_argument('--seed', type=int, default=20260703)
    args = parser.parse_args()

    pos3 = np.load(os.path.join(args.data, 'Round1_Train_Pos.npy'))
    pos = pos3[:, :2]
    h = np.load(os.path.join(args.data, 'Round1_Train_Channel.npy'), mmap_mode='r')

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(pos))
    val_idx = np.sort(perm[:args.val_count])
    train_idx = np.sort(perm[args.val_count:])
    tree = cKDTree(pos[train_idx])
    dists, nbr_local = tree.query(pos[val_idx], k=8)
    nbr_idx = train_idx[nbr_local]

    rows = []
    phase_rows = []
    for ii, vi in enumerate(val_idx):
        truth = np.asarray(h[vi])
        neighs = np.asarray(h[nbr_idx[ii]])
        nn = neighs[0]
        avg3 = np.mean(neighs[:3], axis=0).astype(np.complex64)
        w = 1.0 / np.maximum(dists[ii, :3], 1e-6) ** 2
        w = (w / w.sum()).astype(np.float32)
        idw3 = np.tensordot(w, neighs[:3], axes=(0, 0)).astype(np.complex64)
        gain_nn = best_complex_gain(nn, truth)
        gain_idw = best_complex_gain(idw3, truth)
        rows.append([
            dists[ii, 0], nmse(nn, truth), nmse(gain_nn * nn, truth),
            nmse(avg3, truth), nmse(idw3, truth), nmse(gain_idw * idw3, truth)
        ])

        # dominant delay-bin phase slope for BS antenna 0/user antenna 0
        ht = fft_delay(truth[0, 0])
        k = int(np.argmax(np.abs(ht)))
        spec = truth[0, 0]
        phase = np.unwrap(np.angle(spec))
        slope, intercept = np.polyfit(np.arange(spec.size), phase, 1)
        dist_to_bs = float(np.linalg.norm(pos3[vi] - BS_POS))
        phase_rows.append([dist_to_bs, k, slope, intercept, np.abs(ht[k]), np.mean(np.abs(truth)**2)])

        if (ii + 1) % 25 == 0:
            print('done', ii + 1, flush=True)

    rows = np.asarray(rows)
    labels = ['nn', 'gain*nn', 'avg3', 'idw3', 'gain*idw3']
    print('distance quantiles', np.quantile(rows[:,0], [0,.25,.5,.75,.9,1]))
    for j, lab in enumerate(labels, start=1):
        vals = rows[:, j]
        finite = vals[np.isfinite(vals)]
        print(lab, 'mean', finite.mean(), 'median', np.median(finite), 'q', np.quantile(finite, [0,.25,.5,.75,.9,1]))

    phase_rows = np.asarray(phase_rows)
    print('dominant delay bin counts')
    bins, counts = np.unique(phase_rows[:,1].astype(int), return_counts=True)
    order = np.argsort(counts)[::-1]
    for b, c in zip(bins[order][:20], counts[order][:20]):
        print(int(b), int(c))
    print('slope quantiles', np.quantile(phase_rows[:,2], [0,.1,.25,.5,.75,.9,1]))
    print('delay abs quantiles', np.quantile(phase_rows[:,4], [0,.25,.5,.75,.9,1]))
    print('power quantiles', np.quantile(phase_rows[:,5], [0,.25,.5,.75,.9,1]))


if __name__ == '__main__':
    main()
