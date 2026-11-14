#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


MH, MV, MP = 16, 8, 2


def parse_int_list(text):
    return [int(x) for x in str(text).split(",") if x]


def parse_str_list(text):
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
        return w
    w = 1.0 / np.maximum(d, 1e-6) ** power
    return w / np.sum(w)


def cosine_last(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    num = np.sum(a * b, axis=-1)
    den = np.sqrt(np.sum(a * a, axis=-1) * np.sum(b * b, axis=-1))
    both_zero = den <= 1e-30
    out = num / np.maximum(den, 1e-30)
    out[both_zero] = 1.0
    return out


def exact_pdp_scores(pred, truth):
    pred_p = np.abs(np.fft.ifft(pred, axis=-1)) ** 2
    truth_p = np.abs(np.fft.ifft(truth, axis=-1)) ** 2
    scores = cosine_last(pred_p.reshape(-1, pred_p.shape[-1]), truth_p.reshape(-1, truth_p.shape[-1]))
    return scores.reshape(pred.shape[0], -1).mean(axis=1)


def angle_coeff(h):
    arr = h.reshape(h.shape[0], MH, MV, MP, h.shape[2], h.shape[3])
    return np.fft.fftn(arr, axes=(1, 2))


def inv_angle_coeff(coeff):
    h = np.fft.ifftn(coeff, axes=(1, 2))
    return h.reshape(h.shape[0], MH * MV * MP, h.shape[4], h.shape[5]).astype(np.complex64)


def exact_pas_scores(pred, truth, mode):
    pred_a = angle_coeff(pred)
    truth_a = angle_coeff(truth)
    if mode == "sumpol":
        pred_p = np.sum(np.abs(pred_a) ** 2, axis=3)  # B,H,V,N,S
        truth_p = np.sum(np.abs(truth_a) ** 2, axis=3)
        pred_p = np.moveaxis(pred_p, (1, 2), (-2, -1)).reshape(pred.shape[0], pred.shape[2], pred.shape[3], MH * MV)
        truth_p = np.moveaxis(truth_p, (1, 2), (-2, -1)).reshape(pred.shape[0], pred.shape[2], pred.shape[3], MH * MV)
        scores = cosine_last(pred_p.reshape(-1, MH * MV), truth_p.reshape(-1, MH * MV))
        return scores.reshape(pred.shape[0], -1).mean(axis=1)
    if mode == "fullpol":
        pred_p = np.abs(pred_a) ** 2
        truth_p = np.abs(truth_a) ** 2
        pred_p = np.moveaxis(pred_p, (1, 2, 3), (-3, -2, -1)).reshape(pred.shape[0], pred.shape[2], pred.shape[3], MH * MV * MP)
        truth_p = np.moveaxis(truth_p, (1, 2, 3), (-3, -2, -1)).reshape(pred.shape[0], pred.shape[2], pred.shape[3], MH * MV * MP)
        scores = cosine_last(pred_p.reshape(-1, MH * MV * MP), truth_p.reshape(-1, MH * MV * MP))
        return scores.reshape(pred.shape[0], -1).mean(axis=1)
    raise ValueError(f"unknown PAS mode {mode}")


class MetricAccumulator:
    def __init__(self, pas_mode):
        self.pas_mode = pas_mode
        self.nmse_num = 0.0
        self.nmse_den = 0.0
        self.pdp = []
        self.pas = []

    def add(self, pred, truth):
        diff = pred - truth
        self.nmse_num += float(np.sum(np.abs(diff) ** 2, dtype=np.float64))
        self.nmse_den += float(np.sum(np.abs(truth) ** 2, dtype=np.float64))
        self.pdp.append(exact_pdp_scores(pred, truth))
        self.pas.append(exact_pas_scores(pred, truth, self.pas_mode))

    def result(self):
        pdp = np.concatenate(self.pdp)
        pas = np.concatenate(self.pas)
        nmse = self.nmse_num / max(self.nmse_den, 1e-30)
        score = 0.4 * float(np.mean(pas)) + 0.4 * float(np.mean(pdp)) + 0.2 / (1.0 + nmse)
        return {
            "score": score,
            "pas": float(np.mean(pas)),
            "pdp": float(np.mean(pdp)),
            "global_nmse": float(nmse),
            "inv_nmse": float(1.0 / (1.0 + nmse)),
        }


def weighted_sum(weights, values):
    return np.tensordot(weights.astype(np.float32), values, axes=(0, 0)).astype(np.complex64)


def predict_full_idw(neigh, distances, power):
    w = idw_weights(distances, power)
    return weighted_sum(w, neigh)


def predict_nn_scaled(neigh, distances, power):
    w = idw_weights(distances, power)
    pred = neigh[0].astype(np.complex64, copy=True)
    powers = np.sum(np.abs(neigh) ** 2, axis=(1, 2, 3), dtype=np.float64)
    target = float(np.dot(w, powers))
    base = float(powers[0])
    if base > 1e-30 and target > 0.0:
        pred *= np.float32(np.sqrt(target / base))
    elif target <= 0.0:
        pred.fill(0.0)
    return pred


def project_pdp(base_h, neigh, weights):
    base_cir = np.fft.ifft(base_h, axis=-1)
    neigh_cir = np.fft.ifft(neigh, axis=-1)
    target_power = np.tensordot(weights.astype(np.float32), np.abs(neigh_cir) ** 2, axes=(0, 0))
    mag = np.sqrt(np.maximum(target_power, 0.0)).astype(np.float32)
    phase = base_cir / np.maximum(np.abs(base_cir), 1e-20)
    cir = mag * phase
    return np.fft.fft(cir, axis=-1).astype(np.complex64)


def project_pas_fullpol(base_h, neigh, weights):
    base_a = angle_coeff(base_h[None])[0]
    neigh_a = angle_coeff(neigh)
    target_power = np.tensordot(weights.astype(np.float32), np.abs(neigh_a) ** 2, axes=(0, 0))
    mag = np.sqrt(np.maximum(target_power, 0.0)).astype(np.float32)
    phase = base_a / np.maximum(np.abs(base_a), 1e-20)
    coeff = mag * phase
    return inv_angle_coeff(coeff[None])[0]


def project_pas_sumpol(base_h, neigh, weights):
    base_a = angle_coeff(base_h[None])[0]
    neigh_a = angle_coeff(neigh)
    target_power = np.tensordot(weights.astype(np.float32), np.sum(np.abs(neigh_a) ** 2, axis=3), axes=(0, 0))
    base_power = np.sum(np.abs(base_a) ** 2, axis=2)
    scale = np.sqrt(np.maximum(target_power, 0.0) / np.maximum(base_power, 1e-30)).astype(np.float32)
    coeff = base_a * scale[:, :, None, :, :]
    return inv_angle_coeff(coeff[None])[0]


def predict_projected(neigh, distances, method, power, pass_count):
    weights = idw_weights(distances, power)
    if method.startswith("nn"):
        h = neigh[0].astype(np.complex64, copy=True)
    else:
        h = weighted_sum(weights, neigh)
    if method == "full_idw":
        return h
    if method == "nn_scaled":
        return predict_nn_scaled(neigh, distances, power)
    steps = method.split("+")[1:] if "+" in method else []
    if not steps:
        raise ValueError(method)
    for _ in range(pass_count):
        for step in steps:
            if step == "pdp":
                h = project_pdp(h, neigh, weights)
            elif step == "pasfull":
                h = project_pas_fullpol(h, neigh, weights)
            elif step == "passum":
                h = project_pas_sumpol(h, neigh, weights)
            else:
                raise ValueError(f"unknown projection step {step} in {method}")
    return h


def evaluate(args):
    pos = np.load(os.path.join(args.data, "Round1_Train_Pos.npy"))[:, :2]
    channels = np.load(os.path.join(args.data, "Round1_Train_Channel.npy"), mmap_mode="r")
    train_idx, val_idx = build_split(len(pos), args.val_count, args.seed)
    tree = cKDTree(pos[train_idx])
    results = []
    for k in args.k_values:
        distances, local_idx = tree.query(pos[val_idx], k=k, workers=-1)
        if k == 1:
            distances = distances[:, None]
            local_idx = local_idx[:, None]
        neighbor_idx = train_idx[local_idx]
        for power in args.powers:
            for method in args.methods:
                acc = MetricAccumulator(args.pas_mode)
                for start in range(0, len(val_idx), args.eval_batch):
                    end = min(start + args.eval_batch, len(val_idx))
                    truth = np.asarray(channels[val_idx[start:end]])
                    pred = np.empty_like(truth)
                    for j in range(end - start):
                        neigh = np.asarray(channels[neighbor_idx[start + j]])
                        pred[j] = predict_projected(neigh, distances[start + j], method, power, args.pass_count)
                    acc.add(pred, truth)
                    if args.verbose:
                        print(f"method={method} k={k} p={power:g} {end}/{len(val_idx)}", flush=True)
                row = {
                    "method": method,
                    "k": int(k),
                    "power": float(power),
                    "pass_count": int(args.pass_count),
                    "pas_mode": args.pas_mode,
                    **acc.result(),
                }
                results.append(row)
                print(json.dumps(row, indent=2), flush=True)
    results.sort(key=lambda x: x["score"], reverse=True)
    return {"seed": args.seed, "val_count": args.val_count, "results": results}


def generate(args):
    pos = np.load(os.path.join(args.data, "Round1_Train_Pos.npy"))[:, :2]
    test_pos = np.load(os.path.join(args.data, "Round1_Test_Pos.npy"))[:, :2]
    channels = np.load(os.path.join(args.data, "Round1_Train_Channel.npy"), mmap_mode="r")
    tree = cKDTree(pos)
    distances, neighbor_idx = tree.query(test_pos, k=args.k, workers=-1)
    if args.k == 1:
        distances = distances[:, None]
        neighbor_idx = neighbor_idx[:, None]
    output = np.lib.format.open_memmap(
        args.output,
        mode="w+",
        dtype=np.complex64,
        shape=(len(test_pos), channels.shape[1], channels.shape[2], channels.shape[3]),
    )
    for start in range(0, len(test_pos), args.eval_batch):
        end = min(start + args.eval_batch, len(test_pos))
        for i in range(start, end):
            neigh = np.asarray(channels[neighbor_idx[i]])
            output[i] = predict_projected(neigh, distances[i], args.method, args.power, args.pass_count)
        output.flush()
        print(f"generated {end}/{len(test_pos)}", flush=True)
    return {
        "method": args.method,
        "k": int(args.k),
        "power": float(args.power),
        "pass_count": int(args.pass_count),
        "output": args.output,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/spectrum_projection_results.json"))
    parser.add_argument("--val-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--k-values", type=parse_int_list, default="2,3")
    parser.add_argument("--powers", type=lambda s: [float(x) for x in str(s).split(",") if x], default="1.5,2.0,3.0")
    parser.add_argument(
        "--methods",
        type=parse_str_list,
        default="full_idw,idw+pdp,idw+passum,idw+pdp+passum,idw+passum+pdp,nn+pdp,nn+passum,nn+pdp+passum,nn+passum+pdp",
    )
    parser.add_argument("--pass-count", type=int, default=1)
    parser.add_argument("--pas-mode", choices=["sumpol", "fullpol"], default="sumpol")
    parser.add_argument("--eval-batch", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--method", default="idw+pdp+passum")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--output", default=str(PATHS.temporary / "Round1_Test_Channel.npy"))
    args = parser.parse_args()

    if args.generate:
        result = generate(args)
    else:
        result = evaluate(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
