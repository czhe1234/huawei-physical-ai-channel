#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from evaluate_local_ad_paths import BS_POS
from generate_geometric_steering_blend import unit_vectors
from generate_path_scalar_mlp_blend import make_test_components
from train_path_residual_transformer import (
    build_coeff_sequences,
    predict_residual_scalar,
    sparse_pred,
    train_model,
)
from train_path_scalar_mlp import (
    load_selected_map_features,
    load_virtual_source_model,
)


def generate(args):
    pos = np.load(os.path.join(args.data, "Round1_Train_Pos.npy"))
    test_pos = np.load(os.path.join(args.data, "Round1_Test_Pos.npy"))
    train_h = np.load(os.path.join(args.data, "Round1_Train_Channel.npy"), mmap_mode="r")

    train_pool = np.arange(len(pos), dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    train_queries = np.array(train_pool, copy=True)
    if 0 < args.train_query_count < len(train_queries):
        train_queries = rng.choice(train_queries, size=args.train_query_count, replace=False)
    train_queries = np.asarray(train_queries, dtype=np.int64)

    dist_train = np.linalg.norm(pos - BS_POS[None, :], axis=1)
    u_train = unit_vectors(pos)
    map_train = load_selected_map_features(args.map_features, args.map_feature_indices, "train")
    virtual_model = load_virtual_source_model(args.virtual_sources, args.max_virtual_sources, args.delay_fit)

    print(
        f"building residual transformer train set: queries={len(train_queries)} "
        f"k={args.k} components={args.component_count} residual_rho={args.residual_rho}",
        flush=True,
    )
    x, base, truth, _ = build_coeff_sequences(
        train_h,
        pos,
        dist_train,
        u_train,
        train_queries,
        train_pool,
        args,
        "train",
        map_train,
        virtual_model,
    )
    print(f"training residual transformer x={x.shape} base={base.shape}", flush=True)
    model, mean, std = train_model(x, base, truth, args)

    pos_all = np.concatenate([pos, test_pos], axis=0)
    dist_all = np.linalg.norm(pos_all - BS_POS[None, :], axis=1)
    u_all = unit_vectors(pos_all)
    map_test = load_selected_map_features(args.map_features, args.map_feature_indices, "test")
    if map_train is not None and map_test is not None:
        map_all = np.concatenate([map_train, map_test], axis=0).astype(np.float32)
    else:
        map_all = None

    tree = cKDTree(pos[:, :2])
    distances, neighbor_idx = tree.query(test_pos[:, :2], k=args.k, workers=-1)
    if args.k == 1:
        distances = distances[:, None]
        neighbor_idx = neighbor_idx[:, None]

    out = np.lib.format.open_memmap(
        args.output,
        mode="w+",
        dtype=np.complex64,
        shape=(len(test_pos), train_h.shape[1], train_h.shape[2], train_h.shape[3]),
    )
    test_offset = len(pos)
    for start in range(0, len(test_pos), args.batch):
        end = min(start + args.batch, len(test_pos))
        for i in range(start, end):
            qi = test_offset + i
            comp_ids, pred_coeff, feat, pred_full = make_test_components(
                train_h,
                pos_all,
                dist_all,
                u_all,
                qi,
                neighbor_idx[i],
                distances[i],
                args,
                map_all,
                virtual_model,
            )
            scalar = predict_residual_scalar(model, mean, std, feat, args)
            coeff_delta = (pred_coeff * (scalar[:, None, None] - 1.0)).astype(np.complex64)
            pred_residual = sparse_pred(comp_ids, coeff_delta, args.n_order)[0]
            out[i] = (pred_full + args.residual_rho * pred_residual).astype(np.complex64)
        out.flush()
        print(f"generated {end}/{len(test_pos)}", flush=True)

    return {
        "method": "path_residual_transformer_blend",
        "train_query_count": int(len(train_queries)),
        "k": int(args.k),
        "component_count": int(args.component_count),
        "component_mode": args.component_mode,
        "per_neighbor": int(args.per_neighbor),
        "n_order": args.n_order,
        "power": float(args.power),
        "beta": float(args.beta),
        "gamma": float(args.gamma),
        "alpha": float(args.alpha),
        "ridge": float(args.ridge),
        "steering_scale": float(args.steering_scale),
        "residual_rho": float(args.residual_rho),
        "epochs": int(args.epochs),
        "d_model": int(args.d_model),
        "layers": int(args.layers),
        "ff_dim": int(args.ff_dim),
        "dropout": float(args.dropout),
        "scalar_l2": float(args.scalar_l2),
        "map_features": args.map_features,
        "virtual_sources": args.virtual_sources,
        "max_virtual_sources": int(args.max_virtual_sources),
        "output": args.output,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--output", default=str(PATHS.temporary / "Round1_Test_Channel.npy"))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/path_residual_transformer_generate.json"))
    parser.add_argument("--map-features", default=str(PATHS.artifacts / "shared_features/map_features.npz"))
    parser.add_argument(
        "--map-feature-indices",
        default="0,1,2,3,5,9,13,17,21,25,28,32,33,34,35,39,40,41",
    )
    parser.add_argument("--virtual-sources", default=str(PATHS.analysis / "results/json/wall_planes.json"))
    parser.add_argument("--max-virtual-sources", type=int, default=80)
    parser.add_argument("--delay-fit", default=str(PATHS.artifacts / "shared_features/delay_length_fit.npz"))
    parser.add_argument("--train-query-count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--component-count", type=int, default=128)
    parser.add_argument("--component-mode", default="weighted")
    parser.add_argument("--per-neighbor", type=int, default=256)
    parser.add_argument("--n-order", default="vp")
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--gamma", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=0.3)
    parser.add_argument("--steering-scale", type=float, default=4.71238898038)
    parser.add_argument("--clip-scalar", type=float, default=2.0)
    parser.add_argument("--pred-clip-scalar", type=float, default=2.0)
    parser.add_argument("--max-weight", type=float, default=20.0)
    parser.add_argument("--residual-rho", type=float, default=0.15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--scalar-l2", type=float, default=1e-4)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--batch", type=int, default=10)
    args = parser.parse_args()

    result = generate(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
