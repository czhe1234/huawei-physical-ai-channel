#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from evaluate_local_ad_paths import BS_POS, idw_weights
from evaluate_physical_local_linear import affine_coefficients, predict_affine
from evaluate_ue_angle_delay import ad3_transform, gather_coeff3, inverse_ad3, scatter_coeff3, select_components3
from generate_geometric_steering_blend import steering_phase, unit_vectors
from train_path_scalar_mlp import (
    build_examples,
    component_feature_block,
    load_selected_map_features,
    load_virtual_source_model,
    parse_floats,
    predict_scalars,
    radial_phase,
    sparse_pred,
    train_model,
)


def make_test_components(
    train_h,
    pos_all,
    dist_all,
    u_all,
    qi,
    neigh_idx,
    distances,
    args,
    map_features=None,
    virtual_model=None,
):
    neigh_h = np.asarray(train_h[neigh_idx])
    weights_idw = idw_weights(distances, args.power)
    dist_delta = dist_all[neigh_idx] - dist_all[qi]
    radial = radial_phase(dist_delta, args.beta, args.gamma)
    steer = steering_phase(u_all, neigh_idx, qi, args.steering_scale)
    corrected = neigh_h * radial[:, None, None, :] * steer[:, :, None, None]
    corrected_ad = ad3_transform(corrected, args.n_order)
    corrected_power = np.sum(np.abs(corrected_ad) ** 2, axis=(1, 5), dtype=np.float64)
    component_ids = select_components3(
        corrected_power, weights_idw, args.component_mode, args.per_neighbor, args.component_count
    )
    coeff = gather_coeff3(corrected_ad, component_ids)
    rel_xy = pos_all[neigh_idx, :2] - pos_all[qi, :2]
    affine_w = affine_coefficients(rel_xy, distances, args.power, args.ridge)
    pred_coeff = np.tensordot(affine_w.astype(np.float32), coeff, axes=(0, 0)).astype(np.complex64)
    features, _ = component_feature_block(
        pos_all,
        dist_all,
        u_all,
        qi,
        neigh_idx,
        distances,
        weights_idw,
        args.power,
        component_ids,
        pred_coeff,
        coeff,
        map_features,
        virtual_model,
        args.steering_scale,
    )
    pred_full = predict_affine(
        neigh_h,
        affine_w,
        dist_delta,
        u_all,
        neigh_idx,
        qi,
        args.beta,
        args.gamma,
        args.alpha,
        args.steering_scale,
    )
    return component_ids, pred_coeff, features, pred_full


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
        f"building scalar MLP train set: queries={len(train_queries)} k={args.k} "
        f"components={args.component_count} virtual_sources={0 if virtual_model is None else len(virtual_model['sources'])}",
        flush=True,
    )
    x, y, w = build_examples(
        train_h, pos, dist_train, u_train, train_queries, train_pool, args, "train", map_train, virtual_model
    )
    print(f"training scalar MLP examples={x.shape[0]} features={x.shape[1]}", flush=True)
    model, mean, std = train_model(x, y, w, args)

    pos_all = np.concatenate([pos, test_pos], axis=0)
    dist_all = np.linalg.norm(pos_all - BS_POS[None, :], axis=1)
    u_all = unit_vectors(pos_all)
    map_test = load_selected_map_features(args.map_features, args.map_feature_indices, "test")
    if map_train is not None and map_test is not None:
        map_all = np.concatenate([map_train, map_test], axis=0).astype(np.float32)
    else:
        map_all = None
    test_offset = len(pos)
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
            scalar = predict_scalars(model, mean, std, feat, args)
            coeff_scalar = (pred_coeff * scalar[:, None, None]).astype(np.complex64)
            pred_sparse = sparse_pred(comp_ids, coeff_scalar, args.n_order)[0]
            out[i] = ((1.0 - args.blend_rho) * pred_full + args.blend_rho * pred_sparse).astype(np.complex64)
        out.flush()
        print(f"generated {end}/{len(test_pos)}", flush=True)
    return {
        "method": "path_scalar_mlp_blend",
        "train_query_count": int(len(train_queries)),
        "k": int(args.k),
        "component_count": int(args.component_count),
        "n_order": args.n_order,
        "component_mode": args.component_mode,
        "power": float(args.power),
        "beta": float(args.beta),
        "gamma": float(args.gamma),
        "alpha": float(args.alpha),
        "ridge": float(args.ridge),
        "blend_rho": float(args.blend_rho),
        "epochs": int(args.epochs),
        "width": int(args.width),
        "depth": int(args.depth),
        "model_kind": args.model_kind,
        "num_experts": int(args.num_experts),
        "target_mode": args.target_mode,
        "loss": args.loss,
        "map_features": args.map_features,
        "virtual_sources": args.virtual_sources,
        "max_virtual_sources": int(args.max_virtual_sources),
        "output": args.output,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--output", default=str(PATHS.temporary / "Round1_Test_Channel.npy"))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/path_scalar_mlp_generate.json"))
    parser.add_argument("--map-features", default="")
    parser.add_argument(
        "--map-feature-indices",
        default="0,1,2,3,5,9,13,17,21,25,28,32,33,34,35,39,40,41",
    )
    parser.add_argument("--virtual-sources", default="")
    parser.add_argument("--max-virtual-sources", type=int, default=80)
    parser.add_argument(
        "--delay-fit",
        default=str(PATHS.artifacts / "shared_features/delay_length_fit.npz"),
    )
    parser.add_argument("--train-query-count", type=int, default=2000)
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
    parser.add_argument("--blend-rho", type=float, default=0.3)
    parser.add_argument("--steering-scale", type=float, default=4.71238898038)
    parser.add_argument("--clip-scalar", type=float, default=2.0)
    parser.add_argument("--pred-clip-scalar", type=float, default=2.0)
    parser.add_argument("--max-weight", type=float, default=20.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--infer-batch-size", type=int, default=65536)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--model-kind", choices=["mlp", "embed_mlp", "delay_moe", "source_moe"], default="mlp")
    parser.add_argument("--emb-h", type=int, default=8)
    parser.add_argument("--emb-v", type=int, default=8)
    parser.add_argument("--emb-uv", type=int, default=4)
    parser.add_argument("--emb-s", type=int, default=16)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--target-mode", choices=["direct", "residual_identity", "residual_loo_idw", "residual_loo_nearest"], default="direct")
    parser.add_argument("--loss", choices=["mse", "huber"], default="mse")
    parser.add_argument("--huber-delta", type=float, default=0.25)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--blend-rhos", type=parse_floats, default="0.25,0.5,0.75,1.0")
    args = parser.parse_args()
    result = generate(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
