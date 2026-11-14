#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from evaluate_ad_coeff_methods import eval_one
from evaluate_derotated_loo_grid import choose_val_indices
from evaluate_local_ad_paths import BS_POS
from evaluate_physical_local_linear import predict_affine
from generate_geometric_steering_blend import unit_vectors
from spectrum_projection import MetricAccumulator
from train_path_scalar_mlp import (
    clip_scalar,
    load_selected_map_features,
    load_virtual_source_model,
    make_neighbors,
    parse_floats,
    query_components,
    sparse_pred,
)
from train_path_set_transformer import (
    PathSetTransformer,
    apply_normalize_sequence,
    normalize_sequence_features,
)


def build_coeff_sequences(h, pos, dist_all, u_all, query_idx, train_pool, args, label, map_features, virtual_model):
    d, neigh = make_neighbors(pos, query_idx, train_pool, args.k)
    xs, bases, truths, powers = [], [], [], []
    for i, qi in enumerate(query_idx):
        item = query_components(
            h, pos, dist_all, u_all, int(qi), neigh[i], d[i], args, map_features, virtual_model
        )
        xs.append(item["features"].astype(np.float32))
        bases.append(item["pred_coeff"].astype(np.complex64))
        truths.append(item["truth_coeff"].astype(np.complex64))
        powers.append(np.maximum(item["pred_power"], 1e-12).astype(np.float32))
        if (i + 1) % 50 == 0 or i + 1 == len(query_idx):
            print(f"{label} coeff sequences {i + 1}/{len(query_idx)}", flush=True)
    x = np.stack(xs, axis=0).astype(np.float32)
    base = np.stack(bases, axis=0).astype(np.complex64)
    truth = np.stack(truths, axis=0).astype(np.complex64)
    power = np.stack(powers, axis=0).astype(np.float32)
    return x, base, truth, power


def complex_to_ri(x):
    return np.stack([x.real, x.imag], axis=-1).astype(np.float32)


def coeff_loss(pred_residual, base_ri, truth_ri, scalar_l2):
    br, bi = base_ri[..., 0], base_ri[..., 1]
    tr, ti = truth_ri[..., 0], truth_ri[..., 1]
    sr = 1.0 + pred_residual[..., 0]
    si = pred_residual[..., 1]
    hr = br * sr[..., None, None] - bi * si[..., None, None]
    hi = br * si[..., None, None] + bi * sr[..., None, None]
    err = (hr - tr) ** 2 + (hi - ti) ** 2
    den = tr ** 2 + ti ** 2
    loss = err.sum() / torch.clamp(den.sum(), min=1e-12)
    if scalar_l2 > 0:
        loss = loss + scalar_l2 * (pred_residual ** 2).mean()
    return loss


def train_model(x, base, truth, args):
    rng = np.random.default_rng(args.seed)
    x, mean, std = normalize_sequence_features(x)
    base_ri = complex_to_ri(base)
    truth_ri = complex_to_ri(truth)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = PathSetTransformer(
        cont_dim=x.shape[2] - 5,
        component_count=x.shape[1],
        max_sources=args.max_virtual_sources,
        d_model=args.d_model,
        nhead=args.nhead,
        layers=args.layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    xt = torch.from_numpy(x)
    bt = torch.from_numpy(base_ri)
    tt = torch.from_numpy(truth_ri)
    n = x.shape[0]
    for epoch in range(1, args.epochs + 1):
        perm = rng.permutation(n)
        total = 0.0
        seen = 0
        model.train()
        for start in range(0, n, args.batch_size):
            idx = torch.from_numpy(perm[start:start + args.batch_size]).long()
            xb = xt[idx].to(device)
            bb = bt[idx].to(device)
            tb = tt[idx].to(device)
            pred = model(xb)
            loss = coeff_loss(pred, bb, tb, args.scalar_l2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            total += float(loss.detach().cpu()) * len(idx)
            seen += len(idx)
        if epoch == 1 or epoch % max(args.print_every, 1) == 0 or epoch == args.epochs:
            print(f"epoch {epoch}/{args.epochs} coeff_nmse_loss={total / max(seen, 1):.6f}", flush=True)
    return model, mean, std


def predict_residual_scalar(model, mean, std, features, args):
    x = apply_normalize_sequence(features, mean, std)[None, :, :]
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x).to(device))[0].detach().cpu().numpy()
    scalar = (1.0 + pred[:, 0].astype(np.float32)) + 1j * pred[:, 1].astype(np.float32)
    return clip_scalar(scalar.astype(np.complex64), args.pred_clip_scalar)


def evaluate_model(h, pos, dist_all, u_all, val_idx, train_pool, model, mean, std, args, map_features, virtual_model):
    d, neigh = make_neighbors(pos, val_idx, train_pool, args.k)
    names = ["physical_full", "sparse_base", "residual_pred", "scalar_oracle"]
    names.extend([f"residual_{rho:g}" for rho in args.residual_rhos])
    stats = {name: {"num": 0.0, "den": 0.0} for name in names}
    accs = {name: MetricAccumulator("sumpol") for name in names}
    for i, qi in enumerate(val_idx):
        item = query_components(
            h, pos, dist_all, u_all, int(qi), neigh[i], d[i], args, map_features, virtual_model
        )
        pred_scalar = predict_residual_scalar(model, mean, std, item["features"], args)
        coeff_base = item["pred_coeff"]
        coeff_pred = (coeff_base * pred_scalar[:, None, None]).astype(np.complex64)
        coeff_delta = (coeff_pred - coeff_base).astype(np.complex64)
        coeff_oracle = (coeff_base * item["scalar"][:, None, None]).astype(np.complex64)
        pred_sparse_base = sparse_pred(item["component_ids"], coeff_base, args.n_order)
        pred_residual_sparse = sparse_pred(item["component_ids"], coeff_delta, args.n_order)
        pred_oracle_sparse = sparse_pred(item["component_ids"], coeff_oracle, args.n_order)
        pred_full = predict_affine(
            item["neigh_h"],
            item["affine_w"],
            item["dist_delta"],
            u_all,
            neigh[i],
            int(qi),
            args.beta,
            args.gamma,
            args.alpha,
            args.steering_scale,
        )
        truth = item["truth"]
        preds = {
            "physical_full": pred_full[None],
            "sparse_base": pred_sparse_base,
            "residual_pred": (pred_full[None] + pred_residual_sparse).astype(np.complex64),
            "scalar_oracle": pred_oracle_sparse,
        }
        for rho in args.residual_rhos:
            preds[f"residual_{rho:g}"] = (pred_full[None] + rho * pred_residual_sparse).astype(np.complex64)
        for name, pred in preds.items():
            num, den = eval_one(truth, pred)
            stats[name]["num"] += num
            stats[name]["den"] += den
            accs[name].add(pred, truth)
        if (i + 1) % 10 == 0 or i + 1 == len(val_idx):
            print(f"val reconstruct {i + 1}/{len(val_idx)}", flush=True)
    results = []
    for name in names:
        nmse = stats[name]["num"] / max(stats[name]["den"], 1e-30)
        results.append({
            "method": name,
            "global_nmse_direct": float(nmse),
            **accs[name].result(),
        })
    results.sort(key=lambda r: r["global_nmse_direct"])
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/path_residual_transformer.json"))
    parser.add_argument("--map-features", default=str(PATHS.artifacts / "shared_features/map_features.npz"))
    parser.add_argument(
        "--map-feature-indices",
        default="0,1,2,3,5,9,13,17,21,25,28,32,33,34,35,39,40,41",
    )
    parser.add_argument("--virtual-sources", default=str(PATHS.analysis / "results/json/wall_planes.json"))
    parser.add_argument("--max-virtual-sources", type=int, default=80)
    parser.add_argument("--delay-fit", default=str(PATHS.artifacts / "shared_features/delay_length_fit.npz"))
    parser.add_argument("--val-count", type=int, default=100)
    parser.add_argument("--val-mode", choices=["matched-test", "random"], default="matched-test")
    parser.add_argument("--train-query-count", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--component-count", type=int, default=128)
    parser.add_argument("--component-mode", default="weighted")
    parser.add_argument("--per-neighbor", type=int, default=256)
    parser.add_argument("--n-order", default="vp")
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--gamma", type=float, default=0.15)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=0.3)
    parser.add_argument("--steering-scale", type=float, default=4.71238898038)
    parser.add_argument("--clip-scalar", type=float, default=2.0)
    parser.add_argument("--pred-clip-scalar", type=float, default=2.0)
    parser.add_argument("--max-weight", type=float, default=20.0)
    parser.add_argument("--residual-rhos", type=parse_floats, default="0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.75,1.0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--scalar-l2", type=float, default=1e-4)
    parser.add_argument("--print-every", type=int, default=5)
    args = parser.parse_args()

    pos = np.load(os.path.join(args.data, "Round1_Train_Pos.npy"))
    test_pos = np.load(os.path.join(args.data, "Round1_Test_Pos.npy"))
    h = np.load(os.path.join(args.data, "Round1_Train_Channel.npy"), mmap_mode="r")
    val_idx = choose_val_indices(pos[:, :2], test_pos[:, :2], args.val_count, args.val_mode, args.seed)
    train_pool = np.setdiff1d(np.arange(len(pos), dtype=np.int64), val_idx, assume_unique=False)
    rng = np.random.default_rng(args.seed)
    train_queries = np.array(train_pool, copy=True)
    if 0 < args.train_query_count < len(train_queries):
        train_queries = rng.choice(train_queries, size=args.train_query_count, replace=False)
    train_queries = np.asarray(train_queries, dtype=np.int64)
    dist_all = np.linalg.norm(pos - BS_POS[None, :], axis=1)
    u_all = unit_vectors(pos)
    map_train = load_selected_map_features(args.map_features, args.map_feature_indices, "train")
    virtual_model = load_virtual_source_model(args.virtual_sources, args.max_virtual_sources, args.delay_fit)
    print(
        f"train_queries={len(train_queries)} val={len(val_idx)} comp={args.component_count} "
        f"d_model={args.d_model} layers={args.layers}",
        flush=True,
    )
    x, base, truth, power = build_coeff_sequences(
        h, pos, dist_all, u_all, train_queries, train_pool, args, "train", map_train, virtual_model
    )
    print(f"coeff sequences x={x.shape} base={base.shape} truth={truth.shape}", flush=True)
    model, mean, std = train_model(x, base, truth, args)
    results = evaluate_model(h, pos, dist_all, u_all, val_idx, train_pool, model, mean, std, args, map_train, virtual_model)
    payload = {
        "args": vars(args),
        "train_sequences": int(x.shape[0]),
        "component_count": int(x.shape[1]),
        "feature_dim": int(x.shape[2]),
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
