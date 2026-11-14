#!/usr/bin/env python3
"""Visualize v98 bad/risky points.

Train points have ground truth, so we can plot true leave-one-out support
prediction errors. Test points do not have ground truth, so their plot is a
risk map based on v98 internal diagnostics: low attention confidence, high
entropy, poor target fit after projection, and large support movement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch

import generate_v98_path_ensemble_graph_attention_joint_ad as v98
from generate_v87_easy_field_projection import (
    BS_POS,
    MH,
    MV,
    S_SC,
    ad_transform,
    beta_subcarriers,
    row_cos,
    summarize,
)
from project_paths import PATHS
from powerfield_common import ASSETS, TINY, regime_of


REGIME_COLORS = {
    "clear": "#2ca02c",
    "light": "#1f77b4",
    "medium": "#ff7f0e",
    "heavy": "#d62728",
}


def support_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, np.ndarray]:
    joint = row_cos(pred, truth)
    pred_beam = pred.sum(axis=-1)
    true_beam = truth.sum(axis=-1)
    pred_delay = pred.sum(axis=(1, 2))
    true_delay = truth.sum(axis=(1, 2))
    beam = row_cos(pred_beam, true_beam)
    delay = row_cos(pred_delay, true_delay)
    score = 0.50 * joint + 0.25 * beam + 0.25 * delay
    return {"score": score, "joint": joint, "beam": beam, "delay": delay}


def rank01(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, len(x), dtype=np.float32)
    return ranks


def load_or_train_loo(args: argparse.Namespace) -> dict[str, np.ndarray]:
    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pred_cache = cache_dir / "train_loo_pred_support.npy"
    metrics_cache = cache_dir / "train_loo_metrics.npz"
    if pred_cache.exists() and metrics_cache.exists() and not args.recompute:
        print(f"load cached train LOO: {metrics_cache}", flush=True)
        m = np.load(metrics_cache)
        return {k: m[k] for k in m.files}

    device = v98.choose_device(args.device)
    print(f"device={device}", flush=True)
    data_dir = Path(args.data)
    pos_train = np.load(data_dir / "Round1_Train_Pos.npy")
    geo = np.load(args.geometry_cache)
    blocked_train = geo["train_blocked"].astype(np.float64)
    coll = np.asarray(np.load(args.coll, mmap_mode="r"), dtype=np.float32)
    support = v98.normalize_support(coll, float(args.target_floor))
    train_summary = v98.support_summary(support)
    path_npz = np.load(args.path_candidates)
    train_path = v98.path_summary(path_npz, "train")

    print("build train neighbors", flush=True)
    train_idx, train_eff = v98.build_neighbor_indices(
        pos_train, blocked_train, pos_train, blocked_train, int(args.k), True
    )
    train_edge = v98.edge_features(
        pos_train,
        blocked_train,
        pos_train,
        blocked_train,
        train_idx,
        train_eff,
        train_summary,
        query_path=train_path,
        train_path=train_path,
    )
    edge_mean = train_edge.reshape(-1, train_edge.shape[-1]).mean(axis=0, keepdims=True).astype(np.float32)
    edge_std = np.maximum(train_edge.reshape(-1, train_edge.shape[-1]).std(axis=0, keepdims=True), 1e-3).astype(np.float32)
    train_edge_n = (train_edge - edge_mean[None, :, :]) / edge_std[None, :, :]

    target_acc = np.zeros_like(support, dtype=np.float64)
    weights_acc = np.zeros((len(pos_train), int(args.k)), dtype=np.float64)
    ns = SimpleNamespace(**vars(args))
    ns.log_every = int(args.log_every)
    for seed in [int(x) for x in args.seeds.split(",") if x.strip()]:
        print(f"train LOO visualization seed={seed}", flush=True)
        v98.set_seed(seed)
        model, info = v98.train_attention(train_edge_n.astype(np.float32), train_idx, support, ns, device)
        print("final", info["history"][-1], flush=True)
        target_i, weights_i = v98.predict_support(model, train_edge_n.astype(np.float32), train_idx, support, ns, device)
        target_acc += target_i.astype(np.float64)
        weights_acc += weights_i.astype(np.float64)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pred = (target_acc / len([x for x in args.seeds.split(",") if x.strip()])).astype(np.float32)
    pred = np.maximum(pred, float(args.target_floor)).astype(np.float32)
    pred /= np.maximum(pred.sum(axis=(1, 2, 3), keepdims=True), TINY)
    weights = (weights_acc / len([x for x in args.seeds.split(",") if x.strip()])).astype(np.float32)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), TINY)
    np.save(pred_cache, pred)
    np.save(cache_dir / "train_loo_weights.npy", weights)

    met = support_metrics(pred, support)
    entropy = -(weights * np.log(np.maximum(weights, 1e-12))).sum(axis=1) / np.log(weights.shape[1])
    met["attention_max"] = weights.max(axis=1)
    met["attention_entropy"] = entropy.astype(np.float32)
    np.savez(metrics_cache, **met)
    return met


def test_risk_diagnostics(args: argparse.Namespace) -> dict[str, np.ndarray]:
    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / "test_risk_metrics.npz"
    if cache.exists() and not args.recompute:
        print(f"load cached test risk: {cache}", flush=True)
        m = np.load(cache)
        return {k: m[k] for k in m.files}

    data_dir = Path(args.data)
    pos_test = np.load(data_dir / "Round1_Test_Pos.npy")
    target = np.load(args.target, mmap_mode="r")
    weights = np.load(args.weights)
    base = np.load(args.base, mmap_mode="r")
    pred = np.load(args.pred, mmap_mode="r")
    beta_s = beta_subcarriers(args.beta, args.gamma)
    d_test = np.linalg.norm(pos_test - BS_POS[None, :], axis=1)

    joint_before: list[float] = []
    joint_after: list[float] = []
    support_move: list[float] = []
    pas_to_base: list[float] = []
    pdp_to_base: list[float] = []
    for start in range(0, len(pos_test), int(args.batch)):
        end = min(start + int(args.batch), len(pos_test))
        n = end - start
        phase = np.exp(1j * d_test[start:end, None, None, None] * beta_s).astype(np.complex64)
        h0 = np.asarray(base[start:end], dtype=np.complex64) * phase
        h1 = np.asarray(pred[start:end], dtype=np.complex64) * phase
        ad0 = ad_transform(h0)
        ad1 = ad_transform(h1)
        p0 = (np.abs(ad0) ** 2).astype(np.float32)
        p1 = (np.abs(ad1) ** 2).astype(np.float32)
        c0 = p0.sum(axis=(1, 4))
        c1 = p1.sum(axis=(1, 4))
        c0 /= np.maximum(c0.sum(axis=(1, 2, 3), keepdims=True), TINY)
        c1 /= np.maximum(c1.sum(axis=(1, 2, 3), keepdims=True), TINY)
        tgt = np.asarray(target[start:end], dtype=np.float32)
        joint_before.extend(row_cos(c0, tgt).tolist())
        joint_after.extend(row_cos(c1, tgt).tolist())
        support_move.extend((1.0 - row_cos(c0, c1)).tolist())
        pas_to_base.extend(row_cos(c1.sum(axis=-1), c0.sum(axis=-1)).tolist())
        pdp_to_base.extend(row_cos(c1.sum(axis=(1, 2)), c0.sum(axis=(1, 2))).tolist())
        print(f"test diagnostics {end}/{len(pos_test)}", flush=True)

    attention_entropy = -(weights * np.log(np.maximum(weights, 1e-12))).sum(axis=1) / np.log(weights.shape[1])
    confidence = rank01(weights.max(axis=1)) * (1.0 - rank01(attention_entropy))
    risk = (
        0.35 * rank01(1.0 - np.asarray(joint_after, dtype=np.float32))
        + 0.25 * rank01(attention_entropy.astype(np.float32))
        + 0.20 * rank01(np.asarray(support_move, dtype=np.float32))
        + 0.20 * rank01(1.0 - confidence.astype(np.float32))
    )
    payload = {
        "joint_before": np.asarray(joint_before, dtype=np.float32),
        "joint_after": np.asarray(joint_after, dtype=np.float32),
        "support_move": np.asarray(support_move, dtype=np.float32),
        "pas_to_base": np.asarray(pas_to_base, dtype=np.float32),
        "pdp_to_base": np.asarray(pdp_to_base, dtype=np.float32),
        "attention_max": weights.max(axis=1).astype(np.float32),
        "attention_entropy": attention_entropy.astype(np.float32),
        "confidence_proxy": confidence.astype(np.float32),
        "risk": risk.astype(np.float32),
    }
    np.savez(cache, **payload)
    return payload


def scatter_map(ax, pos_bg, pos, values, title, label, cmap, bad_idx=None, regimes=None):
    ax.scatter(pos_bg[:, 0], pos_bg[:, 1], s=6, c="#d0d0d0", alpha=0.30, linewidths=0)
    sc = ax.scatter(pos[:, 0], pos[:, 1], s=22, c=values, cmap=cmap, alpha=0.95, linewidths=0.15, edgecolors="#222")
    if bad_idx is not None and len(bad_idx) > 0:
        ax.scatter(pos[bad_idx, 0], pos[bad_idx, 1], s=80, facecolors="none", edgecolors="black", linewidths=1.2)
        for rank, i in enumerate(bad_idx[:12], 1):
            ax.text(pos[i, 0], pos[i, 1], str(i), fontsize=7, color="black")
    if regimes is not None:
        for reg, color in REGIME_COLORS.items():
            mask = regimes == reg
            if np.any(mask):
                ax.scatter([], [], c=color, label=f"{reg}: {int(mask.sum())}")
        ax.legend(loc="best", fontsize=8, frameon=True)
    ax.scatter([BS_POS[0]], [BS_POS[1]], marker="*", s=160, c="#8c00ff", edgecolors="white", linewidths=0.6, label="BS")
    ax.set_title(title)
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_aspect("equal", adjustable="box")
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(label)


def box_by_regime(ax, values, regimes, title, ylabel):
    data = [values[regimes == r] for r in v98.REGIMES]
    ax.boxplot(data, labels=v98.REGIMES, showfliers=False)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(PATHS.analysis / "visualizations/v98_bad_points"))
    ap.add_argument("--data", default=str(PATHS.round1))
    ap.add_argument("--geometry-cache", default=ASSETS + "/height_geometry_cache.npz")
    ap.add_argument("--coll", default=ASSETS + "/truth_coll_power.npy")
    ap.add_argument("--path-candidates", default=ASSETS + "/sionna_path_candidates.npz")
    ap.add_argument("--base", default=ASSETS + "/cache_v93/v93_Round1_Test_Channel.npy")
    ap.add_argument("--pred", default=str(PATHS.cache_dir(98) / "v98_Round1_Test_Channel.npy"))
    ap.add_argument("--target", default=ASSETS + "/cache_v98/v98_path_ensemble_graph_attention_target.npy")
    ap.add_argument("--weights", default=ASSETS + "/cache_v98/v98_path_ensemble_graph_attention_weights.npy")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seeds", default="9801,9802,9803")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--predict-batch", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=112)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--target-floor", type=float, default=1e-12)
    ap.add_argument("--log-every", type=int, default=30)
    ap.add_argument("--beta", type=float, default=140.3)
    ap.add_argument("--gamma", type=float, default=0.10)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    data_dir = out_dir / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    pos_train = np.load(Path(args.data) / "Round1_Train_Pos.npy")
    pos_test = np.load(Path(args.data) / "Round1_Test_Pos.npy")
    geo = np.load(args.geometry_cache)
    reg_train = regime_of(geo["train_blocked"].astype(np.float64))
    reg_test = regime_of(geo["test_blocked"].astype(np.float64))

    train_m = load_or_train_loo(args)
    test_m = test_risk_diagnostics(args)

    train_bad = np.argsort(train_m["score"])[:40]
    test_bad = np.argsort(test_m["risk"])[-40:][::-1]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    scatter_map(
        axes[0],
        pos_test,
        pos_train,
        train_m["score"],
        "Train LOO true bad points: support score",
        "higher is better",
        "viridis",
        bad_idx=train_bad,
    )
    scatter_map(
        axes[1],
        pos_train,
        pos_test,
        test_m["risk"],
        "Test high-risk points: no truth, proxy risk",
        "higher is riskier",
        "magma",
        bad_idx=test_bad,
    )
    fig.savefig(fig_dir / "bad_points_overview.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    scatter_map(axes[0, 0], pos_test, pos_train, train_m["joint"], "Train LOO joint AD cosine", "higher is better", "viridis", bad_idx=np.argsort(train_m["joint"])[:30])
    scatter_map(axes[0, 1], pos_test, pos_train, train_m["delay"], "Train LOO delay cosine", "higher is better", "viridis", bad_idx=np.argsort(train_m["delay"])[:30])
    scatter_map(axes[1, 0], pos_train, pos_test, test_m["joint_after"], "Test target fit after v98 projection", "higher is better", "viridis", bad_idx=np.argsort(test_m["joint_after"])[:30])
    scatter_map(axes[1, 1], pos_train, pos_test, test_m["attention_entropy"], "Test attention entropy", "higher = less decisive", "magma", bad_idx=np.argsort(test_m["attention_entropy"])[-30:][::-1])
    fig.savefig(fig_dir / "bad_point_diagnostics.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    box_by_regime(axes[0, 0], train_m["score"], reg_train, "Train LOO support score by regime", "score")
    box_by_regime(axes[0, 1], train_m["beam"], reg_train, "Train LOO beam cosine by regime", "cosine")
    box_by_regime(axes[0, 2], train_m["delay"], reg_train, "Train LOO delay cosine by regime", "cosine")
    box_by_regime(axes[1, 0], test_m["risk"], reg_test, "Test risk proxy by regime", "risk")
    box_by_regime(axes[1, 1], test_m["joint_after"], reg_test, "Test target fit after projection by regime", "cosine")
    box_by_regime(axes[1, 2], test_m["attention_entropy"], reg_test, "Test attention entropy by regime", "entropy")
    fig.savefig(fig_dir / "regime_error_summary.png", dpi=180)
    plt.close(fig)

    summary = {
        "train_worst_score_indices": train_bad[:20].tolist(),
        "test_highest_risk_indices": test_bad[:20].tolist(),
        "train_score": summarize(train_m["score"]),
        "train_joint": summarize(train_m["joint"]),
        "train_beam": summarize(train_m["beam"]),
        "train_delay": summarize(train_m["delay"]),
        "test_risk": summarize(test_m["risk"]),
        "test_joint_after": summarize(test_m["joint_after"]),
        "test_attention_entropy": summarize(test_m["attention_entropy"]),
        "by_regime": {},
    }
    for r in v98.REGIMES:
        mt = reg_train == r
        ms = reg_test == r
        summary["by_regime"][r] = {
            "train_count": int(mt.sum()),
            "test_count": int(ms.sum()),
            "train_score": summarize(train_m["score"][mt]) if mt.any() else {},
            "test_risk": summarize(test_m["risk"][ms]) if ms.any() else {},
        }
    (data_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md = f"""# v98 Bad / Risky Points Visualization

This directory separates two concepts:

- **Train LOO bad points**: true support-prediction errors on training points. These have ground truth.
- **Test high-risk points**: proxy diagnostics only. Test ground truth is unavailable, so these are not confirmed bad points.

## Figures

- `figures/bad_points_overview.png`: left shows true train LOO support score; right shows test proxy risk.
- `figures/bad_point_diagnostics.png`: decomposes joint/delay/attention diagnostics.
- `figures/regime_error_summary.png`: summarizes error/risk by clear/light/medium/heavy regime.

## Worst / Risky Indices

Train worst support-score indices:

```text
{summary['train_worst_score_indices']}
```

Test highest-risk proxy indices:

```text
{summary['test_highest_risk_indices']}
```

## Important Interpretation

The train plot is the reliable error map. The test plot is a risk map:

```text
risk = low target-fit + high attention entropy + large support movement + low confidence
```

So if a test point is highlighted, it means v98 itself is uncertain or made a large AD-support correction there. It does not prove the online error is high.

"""
    (out_dir / "analysis.md").write_text(md, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
