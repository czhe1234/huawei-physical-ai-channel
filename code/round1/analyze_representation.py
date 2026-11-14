#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import json
import os

import numpy as np


MH, MV, MP = 16, 8, 2
NH, NV, NP = 1, 2, 2


def top_energy_fraction(x, topk):
    flat = np.asarray(x).ravel()
    total = float(flat.sum())
    if total <= 0:
        return 1.0
    if topk >= flat.size:
        return 1.0
    idx = np.argpartition(flat, -topk)[-topk:]
    return float(flat[idx].sum() / total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=str(PATHS.round1))
    parser.add_argument('--out', default=str(PATHS.analysis / 'results/json/representation_stats.json'))
    parser.add_argument('--sample', type=int, default=400)
    parser.add_argument('--seed', type=int, default=20260703)
    args = parser.parse_args()

    h = np.load(os.path.join(args.data, 'Round1_Train_Channel.npy'), mmap_mode='r')
    rng = np.random.default_rng(args.seed)
    ids = np.sort(rng.choice(h.shape[0], size=min(args.sample, h.shape[0]), replace=False))

    stats = []
    global_delay = np.zeros(h.shape[-1], dtype=np.float64)
    global_ad = None

    for c, i in enumerate(ids):
        a = np.asarray(h[i])
        power = float(np.mean(np.abs(a) ** 2))
        cir = np.fft.ifft(a, axis=-1)
        delay_p = np.sum(np.abs(cir) ** 2, axis=(0, 1)).astype(np.float64)
        global_delay += delay_p

        arr = cir.reshape(MH, MV, MP, NH, NV, NP, h.shape[-1])
        ad = np.fft.fftn(arr, axes=(0, 1, 3, 4))
        ad_p = np.sum(np.abs(ad) ** 2, axis=(2, 5)).astype(np.float64)  # MH,MV,NH,NV,S
        if global_ad is None:
            global_ad = np.zeros_like(ad_p, dtype=np.float64)
        global_ad += ad_p

        stats.append({
            'idx': int(i),
            'power': power,
            'delay_top1': top_energy_fraction(delay_p, 1),
            'delay_top2': top_energy_fraction(delay_p, 2),
            'delay_top4': top_energy_fraction(delay_p, 4),
            'delay_top8': top_energy_fraction(delay_p, 8),
            'ad_top1': top_energy_fraction(ad_p, 1),
            'ad_top4': top_energy_fraction(ad_p, 4),
            'ad_top8': top_energy_fraction(ad_p, 8),
            'ad_top16': top_energy_fraction(ad_p, 16),
            'ad_top32': top_energy_fraction(ad_p, 32),
            'ad_top64': top_energy_fraction(ad_p, 64),
        })
        if (c + 1) % 25 == 0 or c + 1 == len(ids):
            print(f'analyzed {c+1}/{len(ids)}', flush=True)

    keys = [k for k in stats[0].keys() if k != 'idx']
    summary = {}
    for k in keys:
        v = np.array([s[k] for s in stats], dtype=np.float64)
        summary[k] = {
            'mean': float(v.mean()),
            'q10': float(np.quantile(v, 0.10)),
            'q25': float(np.quantile(v, 0.25)),
            'median': float(np.median(v)),
            'q75': float(np.quantile(v, 0.75)),
            'q90': float(np.quantile(v, 0.90)),
        }

    summary['global_delay_top'] = {str(k): top_energy_fraction(global_delay, k) for k in [1,2,4,8,16,32]}
    summary['global_ad_top'] = {str(k): top_energy_fraction(global_ad, k) for k in [1,4,8,16,32,64,128,256,512,1024]}
    out = {'sample_count': len(ids), 'summary': summary, 'per_sample_head': stats[:10]}
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
