#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree, Delaunay


MH, MV, MP = 16, 8, 2
NH, NV, NP = 1, 2, 2


@dataclass
class Accumulator:
    name: str
    nmse_num: float = 0.0
    nmse_den: float = 0.0
    pdp_sum: float = 0.0
    pas_sum: float = 0.0
    count: int = 0

    def add(self, pred, truth):
        diff = pred - truth
        self.nmse_num += float(np.vdot(diff.ravel(), diff.ravel()).real)
        self.nmse_den += float(np.vdot(truth.ravel(), truth.ravel()).real)
        self.pdp_sum += cosine(pdp(pred), pdp(truth))
        self.pas_sum += cosine(pas(pred), pas(truth))
        self.count += 1

    def result(self):
        nmse = self.nmse_num / max(self.nmse_den, 1e-30)
        pdp_mean = self.pdp_sum / max(self.count, 1)
        pas_mean = self.pas_sum / max(self.count, 1)
        score = 0.4 * pas_mean + 0.4 * pdp_mean + 0.2 / (1.0 + nmse)
        return {
            "method": self.name,
            "score_proxy": score,
            "pas_cos": pas_mean,
            "pdp_cos": pdp_mean,
            "global_nmse": nmse,
            "inv_nmse": 1.0 / (1.0 + nmse),
        }


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def pdp(h):
    ht = np.fft.ifft(h, axis=-1)
    return np.sum(np.abs(ht) ** 2, axis=(0, 1))


def pas(h):
    arr = h.reshape(MH, MV, MP, NH, NV, NP, h.shape[-1])
    ang = np.fft.fftn(arr, axes=(0, 1, 3, 4))
    return np.sum(np.abs(ang) ** 2, axis=(2, 5, 6)).ravel()


def idw_weights(d, power):
    d = np.asarray(d, dtype=np.float64)
    if d[0] < 1e-12:
        w = np.zeros_like(d)
        w[0] = 1.0
        return w
    w = 1.0 / np.maximum(d, 1e-6) ** power
    return w / np.sum(w)


def gaussian_weights(d, sigma):
    d = np.asarray(d, dtype=np.float64)
    if d[0] < 1e-12:
        w = np.zeros_like(d)
        w[0] = 1.0
        return w
    w = np.exp(-0.5 * (d / sigma) ** 2)
    if float(np.sum(w)) == 0.0:
        w[0] = 1.0
    return w / np.sum(w)


def blend(neigh, weights):
    return np.tensordot(weights.astype(np.float32), neigh, axes=(0, 0)).astype(np.complex64)


def scale_nearest_to_local_power(neigh, weights):
    pred = neigh[0].astype(np.complex64, copy=True)
    powers = np.mean(np.abs(neigh) ** 2, axis=(1, 2, 3), dtype=np.float64)
    target_power = float(np.dot(weights, powers))
    base_power = float(powers[0])
    if base_power > 0.0 and target_power > 0.0:
        pred *= np.float32(np.sqrt(target_power / base_power))
    elif target_power == 0.0:
        pred.fill(0.0)
    return pred


def make_methods(max_k):
    methods = [("nn", 1, lambda neigh, d: neigh[0])]
    for k in (2, 3, 4, 6, 8, 12, 16, 24, 32):
        if k <= max_k:
            methods.append((f"mean{k}", k, lambda neigh, d, k=k: np.mean(neigh[:k], axis=0).astype(np.complex64)))
            for p in (1.0, 1.5, 2.0, 3.0, 4.0):
                methods.append((f"idw{k}_p{p:g}", k, lambda neigh, d, k=k, p=p: blend(neigh[:k], idw_weights(d[:k], p))))
            for p in (1.0, 2.0, 3.0):
                methods.append((f"nn_scaled_idw{k}_p{p:g}", k, lambda neigh, d, k=k, p=p: scale_nearest_to_local_power(neigh[:k], idw_weights(d[:k], p))))
    for sigma in (0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0):
        methods.append((f"gauss32_s{sigma:g}", max_k, lambda neigh, d, sigma=sigma: blend(neigh, gaussian_weights(d, sigma))))
    return methods


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--val-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--max-k", type=int, default=32)
    parser.add_argument("--output", default=str(PATHS.analysis / "results/json/validation_results.json"))
    args = parser.parse_args()

    pos = np.load(os.path.join(args.data, "Round1_Train_Pos.npy"))[:, :2]
    h = np.load(os.path.join(args.data, "Round1_Train_Channel.npy"), mmap_mode="r")
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(pos))
    val_idx = np.sort(perm[: args.val_count])
    train_idx = np.sort(perm[args.val_count :])
    train_pos = pos[train_idx]
    val_pos = pos[val_idx]

    tree = cKDTree(train_pos)
    try:
        dists, local_nbr = tree.query(val_pos, k=args.max_k, workers=-1)
    except TypeError:
        dists, local_nbr = tree.query(val_pos, k=args.max_k)
    nbr_idx = train_idx[local_nbr]

    methods = make_methods(args.max_k)
    accum = {name: Accumulator(name) for name, _, _ in methods}

    tri = Delaunay(train_pos)
    simplex = tri.find_simplex(val_pos)
    linear_acc = Accumulator("delaunay_linear")

    for i, truth_idx in enumerate(val_idx):
        truth = np.asarray(h[truth_idx])
        neigh = np.asarray(h[nbr_idx[i]])
        d = dists[i]
        for name, _, fn in methods:
            accum[name].add(fn(neigh, d), truth)

        s = simplex[i]
        if s >= 0:
            transform = tri.transform[s]
            delta = val_pos[i] - transform[2]
            bary = np.dot(transform[:2], delta)
            weights = np.r_[bary, 1.0 - bary.sum()]
            vertices = train_idx[tri.simplices[s]]
            pred = blend(np.asarray(h[vertices]), weights)
        else:
            pred = neigh[0]
        linear_acc.add(pred, truth)

        if (i + 1) % 25 == 0 or i + 1 == len(val_idx):
            print(f"validated {i + 1}/{len(val_idx)}", flush=True)

    results = [acc.result() for acc in accum.values()]
    results.append(linear_acc.result())
    results.sort(key=lambda x: x["score_proxy"], reverse=True)
    with open(args.output, "w") as f:
        json.dump({"seed": args.seed, "val_count": args.val_count, "results": results}, f, indent=2)
    for row in results[:20]:
        print(
            f"{row['method']:22s} score={row['score_proxy']:.6f} "
            f"pas={row['pas_cos']:.6f} pdp={row['pdp_cos']:.6f} "
            f"nmse={row['global_nmse']:.6g} inv={row['inv_nmse']:.6f}"
        )


if __name__ == "__main__":
    main()
