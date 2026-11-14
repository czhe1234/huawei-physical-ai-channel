#!/usr/bin/env python3
"""Build a v98-like train leave-one-out AD base and scalar labels.

v99 learned component scalar labels from the old v36 LOO base and then applied
them to the much stronger v98 test base.  This script reduces that distribution
mismatch.  It cross-fits the v98 graph-attention joint-AD support target on
training points, projects the existing v36 LOO AD base toward that target, and
extracts component-level oracle scalar labels from the resulting base.

The output is not a submission.  It is training infrastructure for the next
joint support/scalar model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from generate_v87_easy_field_projection import MH, MV, S_SC, row_cos, summarize
from generate_v98_path_ensemble_graph_attention_joint_ad import (
    AttentionNet,
    build_neighbor_indices,
    edge_features,
    normalize_support,
    parse_regime_values,
    parse_seeds,
    path_summary,
    set_seed,
    support_loss,
    support_summary,
)
from generate_v99_learned_component_scalar_gate import (
    component_vectors,
    compute_oracle_scalar,
    flat_to_hvt,
    top_ids_from_power,
)
from project_paths import PATHS
from powerfield_common import ASSETS, TINY, regime_of


REGIMES = ("clear", "light", "medium", "heavy")


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def train_attention_subset(
    edge: np.ndarray,
    neighbor_idx: np.ndarray,
    support: np.ndarray,
    query_rows: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[AttentionNet, dict[str, object]]:
    """Train v98-style attention on a subset of query rows.

    `neighbor_idx` always uses global train indices, while `query_rows` maps the
    local edge rows back to global truth-support rows.  This is the piece that
    makes cross-fit LOO possible without target leakage from the held-out fold.
    """
    n, k, f = edge.shape
    model = AttentionNet(f, int(args.hidden), float(args.dropout)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    edge_t = torch.from_numpy(edge.astype(np.float32))
    idx_t = torch.from_numpy(neighbor_idx.astype(np.int64))
    row_t = torch.from_numpy(query_rows.astype(np.int64))
    supp = torch.from_numpy(support.reshape(support.shape[0], -1).astype(np.float32)).to(device)
    log_supp = torch.log(supp.clamp_min(float(args.target_floor)))
    ds = TensorDataset(torch.arange(n, dtype=torch.long))
    loader = DataLoader(ds, batch_size=int(args.batch_size), shuffle=True, drop_last=False)
    history: list[dict[str, float]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        vals = []
        for (local_cpu,) in loader:
            local = local_cpu.to(device)
            e = edge_t[local_cpu].to(device)
            nb = idx_t[local_cpu].to(device)
            truth_rows = row_t[local_cpu].to(device)
            scores = model(e)
            weights = torch.softmax(scores / float(args.temperature), dim=1)
            pred_log = torch.sum(weights[..., None] * log_supp[nb], dim=1)
            pred = torch.exp(pred_log).clamp_min(float(args.target_floor))
            pred = pred / pred.sum(dim=1, keepdim=True).clamp_min(1e-30)
            truth = supp[truth_rows]
            loss, metrics = support_loss(pred, truth)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            vals.append(metrics)
        if epoch == 1 or epoch % int(args.log_every) == 0 or epoch == int(args.epochs):
            avg = {key: float(np.mean([v[key] for v in vals])) for key in vals[0]}
            avg["epoch"] = epoch
            history.append(avg)
            print("subset train", json.dumps(avg, ensure_ascii=False), flush=True)
    return model, {"history": history, "in_dim": f, "rows": int(n), "k": int(k)}


@torch.no_grad()
def predict_attention_rows(
    model: AttentionNet,
    edge: np.ndarray,
    neighbor_idx: np.ndarray,
    support: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    n = edge.shape[0]
    supp = torch.from_numpy(support.reshape(support.shape[0], -1).astype(np.float32)).to(device)
    log_supp = torch.log(supp.clamp_min(float(args.target_floor)))
    out = np.empty((n, MH, MV, S_SC), dtype=np.float32)
    weights_out = np.empty((n, neighbor_idx.shape[1]), dtype=np.float32)
    edge_t = torch.from_numpy(edge.astype(np.float32))
    idx_t = torch.from_numpy(neighbor_idx.astype(np.int64))
    for start in range(0, n, int(args.predict_batch)):
        end = min(start + int(args.predict_batch), n)
        e = edge_t[start:end].to(device)
        nb = idx_t[start:end].to(device)
        scores = model(e)
        weights = torch.softmax(scores / float(args.temperature), dim=1)
        pred_log = torch.sum(weights[..., None] * log_supp[nb], dim=1)
        pred = torch.exp(pred_log).clamp_min(float(args.target_floor))
        pred = pred / pred.sum(dim=1, keepdim=True).clamp_min(1e-30)
        out[start:end] = pred.reshape(end - start, MH, MV, S_SC).cpu().numpy().astype(np.float32)
        weights_out[start:end] = weights.cpu().numpy().astype(np.float32)
    return out, weights_out


def crossfit_targets(
    edge: np.ndarray,
    neighbor_idx: np.ndarray,
    support: np.ndarray,
    query_indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    n = len(query_indices)
    folds = np.array_split(np.arange(n, dtype=np.int64), int(args.folds))
    seeds = parse_seeds(args.seeds, int(args.seed))
    target_acc = np.zeros((n, MH, MV, S_SC), dtype=np.float64)
    weights_acc = np.zeros((n, int(args.k)), dtype=np.float64)
    infos: list[dict[str, object]] = []
    for seed in seeds:
        set_seed(seed)
        for fold_id, valid_local in enumerate(folds):
            train_local = np.setdiff1d(np.arange(n, dtype=np.int64), valid_local, assume_unique=True)
            print(f"crossfit seed={seed} fold={fold_id+1}/{len(folds)} train={len(train_local)} valid={len(valid_local)}", flush=True)
            model, info = train_attention_subset(
                edge[train_local],
                neighbor_idx[train_local],
                support,
                query_indices[train_local],
                args,
                device,
            )
            target_fold, weights_fold = predict_attention_rows(
                model,
                edge[valid_local],
                neighbor_idx[valid_local],
                support,
                args,
                device,
            )
            target_acc[valid_local] += target_fold.astype(np.float64)
            weights_acc[valid_local] += weights_fold.astype(np.float64)
            info["seed"] = int(seed)
            info["fold"] = int(fold_id)
            info["valid_count"] = int(len(valid_local))
            infos.append(info)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    target = (target_acc / len(seeds)).astype(np.float32)
    target = np.maximum(target, float(args.target_floor)).astype(np.float32)
    target /= np.maximum(target.sum(axis=(1, 2, 3), keepdims=True), TINY)
    weights = (weights_acc / len(seeds)).astype(np.float32)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), TINY)
    return target, weights, infos


def project_base_and_make_labels(
    base_ad_path: str,
    residual_ad_path: str,
    query_indices: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    regimes: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, object]:
    base_ad = np.load(base_ad_path, mmap_mode="r")
    residual_ad = np.load(residual_ad_path, mmap_mode="r")
    out_base = np.lib.format.open_memmap(args.out_base_ad, mode="w+", dtype=np.complex64, shape=(len(query_indices),) + base_ad.shape[1:])
    out_residual = np.lib.format.open_memmap(args.out_residual_ad, mode="w+", dtype=np.complex64, shape=(len(query_indices),) + base_ad.shape[1:])

    label_ids = np.empty((len(query_indices), int(args.label_top_components)), dtype=np.int64)
    label_scalar = np.empty(label_ids.shape, dtype=np.complex64)
    label_residual = np.empty(label_ids.shape, dtype=np.complex64)
    label_coherence = np.empty(label_ids.shape, dtype=np.float32)
    label_power = np.empty(label_ids.shape, dtype=np.float32)
    label_frac = np.empty(label_ids.shape, dtype=np.float32)

    eta_base = parse_regime_values(args.eta, 0.0)
    gain_clip = tuple(float(x) for x in args.gain_clip.split(","))
    top_gain_clip = tuple(float(x) for x in args.top_gain_clip.split(","))
    maxw = weights.max(axis=1)
    entropy = -(weights * np.log(np.maximum(weights, 1e-12))).sum(axis=1) / np.log(weights.shape[1])
    confidence = np.clip((maxw - float(args.conf_center)) / max(float(args.conf_width), 1e-6), 0.0, 1.0)
    confidence *= np.clip(
        (float(args.entropy_high) - entropy) / max(float(args.entropy_high) - float(args.entropy_low), 1e-6),
        0.0,
        1.0,
    )
    eta = np.array([eta_base[str(r)] for r in regimes], dtype=np.float32)
    eta *= (float(args.min_gate) + (float(args.max_gate) - float(args.min_gate)) * confidence).astype(np.float32)

    stats: dict[str, object] = {
        "eta": eta.tolist(),
        "attention_max": maxw.tolist(),
        "attention_entropy": entropy.tolist(),
        "confidence": confidence.tolist(),
        "joint_cos_before": [],
        "joint_cos_after": [],
        "gain_min": [],
        "gain_max": [],
        "label_coherence": [],
        "label_residual_abs": [],
        "label_frac": [],
        "delta_num": 0.0,
        "delta_den": 0.0,
    }
    by_regime = {r: {"count": 0, "eta": [], "joint_before": [], "joint_after": [], "coherence": []} for r in REGIMES}

    for start in range(0, len(query_indices), int(args.project_batch)):
        end = min(start + int(args.project_batch), len(query_indices))
        qids = query_indices[start:end]
        n = end - start
        ad0 = np.asarray(base_ad[qids], dtype=np.complex64)
        truth = ad0 + np.asarray(residual_ad[qids], dtype=np.complex64)
        p0 = (np.abs(ad0) ** 2).astype(np.float32)
        coll0 = p0.sum(axis=(1, 4))
        total0 = np.maximum(coll0.sum(axis=(1, 2, 3)), TINY).astype(np.float32)
        cur = coll0 / total0[:, None, None, None]
        tgt = target[start:end].astype(np.float32)
        exponent = eta[start:end].astype(np.float32)
        ratio_hvt = ((tgt + float(args.target_floor)) / (cur + float(args.target_floor))) ** exponent[:, None, None, None]
        p = p0 * ratio_hvt[:, None, :, :, None, :]
        gain = np.sqrt(p / np.maximum(p0, 1e-30)).astype(np.float32)
        gain = np.clip(gain, gain_clip[0], gain_clip[1])

        flat_power = coll0.reshape(n, -1)
        ktop = min(int(args.top_components), flat_power.shape[1])
        top_ids = np.argpartition(flat_power, -ktop, axis=1)[:, -ktop:]
        top_mask = np.zeros_like(flat_power, dtype=bool)
        top_mask[np.arange(n)[:, None], top_ids] = True
        top_mask = top_mask.reshape(n, MH, MV, S_SC)[:, None, :, :, None, :]
        gain = np.where(top_mask, np.clip(gain, top_gain_clip[0], top_gain_clip[1]), gain)

        ad1 = (ad0 * gain).astype(np.complex64)
        total1 = np.maximum(np.sum(np.abs(ad1) ** 2, axis=(1, 2, 3, 4, 5), dtype=np.float64), TINY).astype(np.float32)
        ad1 *= np.sqrt(total0 / total1)[:, None, None, None, None, None].astype(np.float32)
        res1 = (truth - ad1).astype(np.complex64)

        out_base[start:end] = ad1
        out_residual[start:end] = res1
        out_base.flush()
        out_residual.flush()

        p1 = (np.abs(ad1) ** 2).astype(np.float32)
        coll1 = p1.sum(axis=(1, 4))
        total_after = np.maximum(coll1.sum(axis=(1, 2, 3)), TINY).astype(np.float32)
        cur1 = coll1 / total_after[:, None, None, None]
        before = row_cos(cur, tgt)
        after = row_cos(cur1, tgt)
        stats["joint_cos_before"].extend(before.tolist())
        stats["joint_cos_after"].extend(after.tolist())
        stats["gain_min"].extend(np.min(gain.reshape(n, -1), axis=1).tolist())
        stats["gain_max"].extend(np.max(gain.reshape(n, -1), axis=1).tolist())
        diff = ad1 - ad0
        stats["delta_num"] += float(np.sum(np.abs(diff) ** 2, dtype=np.float64))
        stats["delta_den"] += float(np.sum(np.abs(ad0) ** 2, dtype=np.float64))

        power_hvt = np.sum(np.abs(ad1) ** 2, axis=(1, 4), dtype=np.float64).astype(np.float32)
        top_label = top_ids_from_power(power_hvt, int(args.label_top_components))
        for r in range(n):
            comp_ids = top_label[r]
            bv = component_vectors(ad1, r, comp_ids)
            tv = component_vectors(truth, r, comp_ids)
            scalar, coh, cpow = compute_oracle_scalar(bv, tv)
            h, v, t = flat_to_hvt(comp_ids)
            frac = cpow / max(float(power_hvt[r].sum()), TINY)
            label_ids[start + r] = comp_ids
            label_scalar[start + r] = scalar
            label_residual[start + r] = (scalar - 1.0).astype(np.complex64)
            label_coherence[start + r] = coh.astype(np.float32)
            label_power[start + r] = cpow.astype(np.float32)
            label_frac[start + r] = frac.astype(np.float32)
            stats["label_coherence"].extend(coh.tolist())
            stats["label_residual_abs"].extend(np.abs(scalar - 1.0).tolist())
            stats["label_frac"].extend(frac.tolist())
            reg = str(regimes[start + r])
            br = by_regime[reg]
            br["count"] += 1
            br["eta"].append(float(exponent[r]))
            br["joint_before"].append(float(before[r]))
            br["joint_after"].append(float(after[r]))
            br["coherence"].extend(coh.tolist())
        print(f"projected/labelled {end}/{len(query_indices)}", flush=True)

    labels_path = Path(args.out_labels)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        labels_path,
        point_indices=query_indices.astype(np.int64),
        component_ids=label_ids,
        scalar=label_scalar,
        residual=label_residual,
        coherence=label_coherence,
        component_power=label_power,
        power_fraction=label_frac,
        target=target.astype(np.float32),
        attention_weights=weights.astype(np.float32),
        eta=eta.astype(np.float32),
        confidence=confidence.astype(np.float32),
    )
    payload = {
        "eta": summarize(stats["eta"]),
        "attention_max": summarize(stats["attention_max"]),
        "attention_entropy": summarize(stats["attention_entropy"]),
        "confidence": summarize(stats["confidence"]),
        "joint_cos_before": summarize(stats["joint_cos_before"]),
        "joint_cos_after": summarize(stats["joint_cos_after"]),
        "gain_min": summarize(stats["gain_min"]),
        "gain_max": summarize(stats["gain_max"]),
        "label_coherence": summarize(stats["label_coherence"]),
        "label_residual_abs": summarize(stats["label_residual_abs"]),
        "label_frac": summarize(stats["label_frac"]),
        "delta_nmse_to_v36_loo_base_ad": float(stats["delta_num"] / max(stats["delta_den"], TINY)),
        "by_regime": {r: {k: summarize(v) if isinstance(v, list) else v for k, v in vals.items()} for r, vals in by_regime.items()},
    }
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(PATHS.round1))
    ap.add_argument("--geometry-cache", default=ASSETS + "/height_geometry_cache.npz")
    ap.add_argument("--path-candidates", default=ASSETS + "/sionna_path_candidates.npz")
    ap.add_argument("--coll", default=ASSETS + "/truth_coll_power.npy")
    ap.add_argument("--train-base-ad", default=ASSETS + "/train_v36_loo_base_derot_ad.npy")
    ap.add_argument("--train-residual-ad", default=ASSETS + "/train_v36_loo_residual_derot_ad.npy")
    ap.add_argument("--out-base-ad", default=ASSETS + "/train_v98like_loo_base_ad.npy")
    ap.add_argument("--out-residual-ad", default=ASSETS + "/train_v98like_loo_residual_ad.npy")
    ap.add_argument("--out-target", default=ASSETS + "/train_v98like_loo_joint_target.npy")
    ap.add_argument("--out-weights", default=ASSETS + "/train_v98like_loo_attention_weights.npy")
    ap.add_argument("--out-labels", default=ASSETS + "/train_v98like_component_scalar_labels.npz")
    ap.add_argument("--report", default=ASSETS + "/train_v98like_loo_report.json")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=9801)
    ap.add_argument("--seeds", default="9801")
    ap.add_argument("--limit-points", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--predict-batch", type=int, default=128)
    ap.add_argument("--project-batch", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=112)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--log-every", type=int, default=15)
    ap.add_argument("--target-floor", type=float, default=1e-12)
    ap.add_argument("--eta", default="clear:0.025,light:0.05,medium:0.30,heavy:0.36")
    ap.add_argument("--min-gate", type=float, default=0.12)
    ap.add_argument("--max-gate", type=float, default=1.90)
    ap.add_argument("--conf-center", type=float, default=0.38)
    ap.add_argument("--conf-width", type=float, default=0.25)
    ap.add_argument("--entropy-low", type=float, default=0.22)
    ap.add_argument("--entropy-high", type=float, default=0.90)
    ap.add_argument("--gain-clip", default="0.54,2.05")
    ap.add_argument("--top-gain-clip", default="0.68,1.65")
    ap.add_argument("--top-components", type=int, default=320)
    ap.add_argument("--label-top-components", type=int, default=96)
    args = ap.parse_args()

    set_seed(int(args.seed))
    device = choose_device(args.device)
    data_dir = Path(args.data)
    pos_train = np.load(data_dir / "Round1_Train_Pos.npy")
    geo = np.load(args.geometry_cache)
    blocked_train = geo["train_blocked"].astype(np.float64)
    all_indices = np.arange(len(pos_train), dtype=np.int64)
    if int(args.limit_points) > 0:
        all_indices = all_indices[: int(args.limit_points)]
        # Keep smoke-test outputs separate unless the caller explicitly sets paths.
        if args.out_base_ad == ASSETS + "/train_v98like_loo_base_ad.npy":
            args.out_base_ad = ASSETS + f"/train_v98like_loo_base_ad_smoke{len(all_indices)}.npy"
        if args.out_residual_ad == ASSETS + "/train_v98like_loo_residual_ad.npy":
            args.out_residual_ad = ASSETS + f"/train_v98like_loo_residual_ad_smoke{len(all_indices)}.npy"
        if args.out_target == ASSETS + "/train_v98like_loo_joint_target.npy":
            args.out_target = ASSETS + f"/train_v98like_loo_joint_target_smoke{len(all_indices)}.npy"
        if args.out_weights == ASSETS + "/train_v98like_loo_attention_weights.npy":
            args.out_weights = ASSETS + f"/train_v98like_loo_attention_weights_smoke{len(all_indices)}.npy"
        if args.out_labels == ASSETS + "/train_v98like_component_scalar_labels.npz":
            args.out_labels = ASSETS + f"/train_v98like_component_scalar_labels_smoke{len(all_indices)}.npz"
        if args.report == ASSETS + "/train_v98like_loo_report.json":
            args.report = ASSETS + f"/train_v98like_loo_report_smoke{len(all_indices)}.json"

    print(f"device={device} query_points={len(all_indices)} folds={args.folds} seeds={args.seeds}", flush=True)

    coll = np.asarray(np.load(args.coll, mmap_mode="r"), dtype=np.float32)
    support = normalize_support(coll, float(args.target_floor))
    train_summary = support_summary(support)
    path_npz = np.load(args.path_candidates)
    train_path = path_summary(path_npz, "train")

    print("build train-query neighbors", flush=True)
    query_pos = pos_train[all_indices]
    query_blocked = blocked_train[all_indices]
    neighbor_idx, eff = build_neighbor_indices(pos_train, blocked_train, query_pos, query_blocked, int(args.k), True)
    edge = edge_features(
        query_pos,
        query_blocked,
        pos_train,
        blocked_train,
        neighbor_idx,
        eff,
        train_summary,
        query_path=train_path[all_indices],
        train_path=train_path,
    )
    edge_mean = edge.reshape(-1, edge.shape[-1]).mean(axis=0, keepdims=True).astype(np.float32)
    edge_std = np.maximum(edge.reshape(-1, edge.shape[-1]).std(axis=0, keepdims=True), 1e-3).astype(np.float32)
    edge_n = ((edge - edge_mean[None, :, :]) / edge_std[None, :, :]).astype(np.float32)

    target, weights, train_infos = crossfit_targets(edge_n, neighbor_idx, support, all_indices, args, device)
    Path(args.out_target).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_target, target.astype(np.float32))
    np.save(args.out_weights, weights.astype(np.float32))

    regimes = regime_of(query_blocked)
    projection_stats = project_base_and_make_labels(
        args.train_base_ad,
        args.train_residual_ad,
        all_indices,
        target,
        weights,
        regimes,
        args,
    )

    payload = {
        "method": "build_v98like_loo_base",
        "purpose": "cross-fit v98-style graph-attention joint AD target on train points, project v36 LOO AD base, and extract component scalar labels",
        "args": vars(args),
        "query_points": int(len(all_indices)),
        "train": train_infos,
        "projection": projection_stats,
        "outputs": {
            "base_ad": args.out_base_ad,
            "residual_ad": args.out_residual_ad,
            "target": args.out_target,
            "weights": args.out_weights,
            "labels": args.out_labels,
        },
    }
    Path(args.report).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
