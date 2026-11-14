#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.preprocessing import StandardScaler


MH, MV, MP = 16, 8, 2
NH, NV, NP = 1, 2, 2


def load_h(data):
    return np.load(os.path.join(data, 'Round1_Train_Channel.npy'), mmap_mode='r')


def transform(h, mode):
    if mode == 'freq':
        x = h
    elif mode == 'delay':
        x = np.fft.ifft(h, axis=-1).astype(np.complex64)
    elif mode == 'angle_delay':
        cir = np.fft.ifft(h, axis=-1)
        arr = cir.reshape(h.shape[0], MH, MV, MP, NH, NV, NP, h.shape[-1])
        x = np.fft.fftn(arr, axes=(1, 2, 4, 5)).astype(np.complex64)
    else:
        raise ValueError(mode)
    return x


def inverse_transform(x, mode):
    if mode == 'freq':
        return x.astype(np.complex64)
    if mode == 'delay':
        return np.fft.fft(x, axis=-1).astype(np.complex64)
    if mode == 'angle_delay':
        arr = np.fft.ifftn(x, axes=(1, 2, 4, 5))
        arr = arr.reshape(x.shape[0], 256, 4, 192)
        return np.fft.fft(arr, axis=-1).astype(np.complex64)
    raise ValueError(mode)


def flatten_complex(x):
    xr = np.stack([x.real, x.imag], axis=-1).reshape(x.shape[0], -1)
    return xr.astype(np.float32)


def unflatten_complex(z, shape):
    z = z.reshape((z.shape[0],) + shape[1:] + (2,))
    return (z[..., 0] + 1j * z[..., 1]).astype(np.complex64)


def nmse(a, b):
    num = np.sum(np.abs(a - b) ** 2, dtype=np.float64)
    den = np.sum(np.abs(b) ** 2, dtype=np.float64)
    return float(num / max(den, 1e-30))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=str(PATHS.round1))
    p.add_argument('--outdir', default=str(PATHS.artifacts / 'pca_experiments'))
    p.add_argument('--mode', default='delay', choices=['freq','delay','angle_delay'])
    p.add_argument('--components', type=int, default=512)
    p.add_argument('--sample', type=int, default=1200)
    p.add_argument('--batch', type=int, default=50)
    p.add_argument('--seed', type=int, default=20260703)
    args = p.parse_args()

    outdir = Path(args.outdir) / f'{args.mode}_c{args.components}_n{args.sample}'
    outdir.mkdir(parents=True, exist_ok=True)

    H = load_h(args.data)
    rng = np.random.default_rng(args.seed)
    ids = np.sort(rng.choice(H.shape[0], size=min(args.sample, H.shape[0]), replace=False))
    val_n = max(100, len(ids)//5)
    val_ids = ids[:val_n]
    train_ids = ids[val_n:]

    # Materialize transformed flattened data for the sampled subset; 1200 samples is manageable on disk/mem.
    rows = []
    for start in range(0, len(ids), args.batch):
        bid = ids[start:start+args.batch]
        x = transform(np.asarray(H[bid]), args.mode)
        rows.append(flatten_complex(x))
        print('loaded', start + len(bid), '/', len(ids), flush=True)
    X = np.vstack(rows)
    del rows
    mean_power = np.mean(X * X, axis=1, dtype=np.float64)
    scale = np.sqrt(np.maximum(mean_power, 1e-30)).astype(np.float32)
    Xn = X / scale[:, None]

    ipca = PCA(n_components=min(args.components, Xn.shape[0], Xn.shape[1]), svd_solver='randomized', random_state=args.seed)
    Z = ipca.fit_transform(Xn)
    Xr = ipca.inverse_transform(Z) * scale[:, None]

    shape = transform(np.asarray(H[ids[:1]]), args.mode).shape
    metrics = []
    for local_idx, global_idx in enumerate(ids):
        y_true = np.asarray(H[global_idx:global_idx+1])
        xt = unflatten_complex(Xr[local_idx:local_idx+1], shape)
        y_pred = inverse_transform(xt, args.mode)
        metrics.append(nmse(y_pred, y_true))
        if (local_idx + 1) % 100 == 0:
            print('reconstructed', local_idx+1, '/', len(ids), flush=True)
    metrics = np.array(metrics)
    result = {
        'mode': args.mode,
        'components': int(ipca.n_components_),
        'sample': int(len(ids)),
        'explained_variance_ratio_sum': float(ipca.explained_variance_ratio_.sum()),
        'recon_nmse_mean': float(metrics.mean()),
        'recon_nmse_median': float(np.median(metrics)),
        'recon_nmse_q90': float(np.quantile(metrics, .9)),
        'recon_nmse_power_weighted': float(np.sum(metrics * np.sum(np.abs(np.asarray(H[ids]))**2, axis=(1,2,3))) / np.sum(np.sum(np.abs(np.asarray(H[ids]))**2, axis=(1,2,3)))),
    }
    with open(outdir / 'result.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
