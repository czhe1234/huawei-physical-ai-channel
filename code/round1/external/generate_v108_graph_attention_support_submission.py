#!/usr/bin/env python3
"""Generate v108 graph-attention support completion submission."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import PATHS  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

PROFILE_LEARNED = (0.10, 0.10, 0.175, 0.25)


def load_external_source(source: str, core_package: str) -> tuple[Any, ...]:
    """Load the unavailable v108 upstream checkout from an explicit location."""
    if not source:
        raise RuntimeError(
            "v108 requires its original upstream source checkout; set "
            "HW_V108_SOURCE or pass --external-source"
        )
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_dir():
        raise RuntimeError(f"v108 upstream source directory does not exist: {source_path}")
    sys.path.insert(0, str(source_path))
    try:
        v084 = importlib.import_module("diagnose_v084_v36_geometry_attention_simplification")
        v100 = importlib.import_module("evaluate_v100_graph_spectral_power_field")
        v108 = importlib.import_module("train_v108_graph_attention_support_oof")
        v101 = importlib.import_module("generate_v101_v55_graph_power_submission")
        config = importlib.import_module(f"{core_package}.config")
        data_module = importlib.import_module(f"{core_package}.data")
        geometry = importlib.import_module(f"{core_package}.physical_reflection")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"incomplete v108 upstream checkout at {source_path}: missing {exc.name}"
        ) from exc
    return v084, v100, v108, v101, config, data_module, geometry


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate v108 graph-attention support submission")
    ap.add_argument("--external-source", default=os.environ.get("HW_V108_SOURCE", ""))
    ap.add_argument("--core-package", default=os.environ.get("HW_V108_CORE_PACKAGE", "project_core"))
    ap.add_argument(
        "--anchor-zip",
        default="",
    )
    ap.add_argument("--anchor-npy", default=str(PATHS.cache_dir(91) / "v91_Round1_Test_Channel.npy"))
    ap.add_argument("--out-dir", default=str(PATHS.artifacts / "external" / "v108_graph_attention_support"))
    ap.add_argument("--attn-k", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=108)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--log-every", type=int, default=60)
    ap.add_argument("--support-cache", default=str(PATHS.artifacts / "external" / "v108_graph_attention_support" / "train_ad_support_power.npy"))
    ap.add_argument("--force-support-cache", action="store_true")
    ap.add_argument("--support-batch", type=int, default=16)
    ap.add_argument("--ply", default="")
    ap.add_argument("--los-clear-dist", type=float, default=0.5)
    ap.add_argument("--support-mode", choices=["log", "linear"], default="log")
    ap.add_argument("--total-mode", choices=["anchor", "propagated"], default="anchor")
    ap.add_argument("--ratio-min", type=float, default=0.25)
    ap.add_argument("--ratio-max", type=float, default=4.0)
    ap.add_argument("--power-floor", type=float, default=1e-18)
    ap.add_argument("--power-ceiling", type=float, default=1e6)
    ap.add_argument("--height-cell", type=float, default=1.0)
    ap.add_argument("--height-z-min", type=float, default=-5.0)
    ap.add_argument("--height-samples", type=int, default=300)
    ap.add_argument("--height-skip-near", type=float, default=4.0)
    ap.add_argument("--height-clear", type=float, default=1.0)
    ap.add_argument("--force-geometry", action="store_true")
    ap.add_argument("--write-zip", action="store_true")
    args = ap.parse_args()

    V084, V100, V108, V101, config, data_module, geometry = load_external_source(
        args.external_source, args.core_package
    )
    DEFAULT_PATHS = config.DEFAULT_PATHS
    load_round1_data = data_module.load_round1_data
    MapGeometry = geometry.MapGeometry
    _extract_anchor = V101._extract_anchor
    _jsonify = V101._jsonify
    _pair_stats = V101._pair_stats
    _zip_prediction = V101._zip_prediction
    if not args.ply:
        args.ply = str(DEFAULT_PATHS.data_dir / "Round1_Map.ply")

    V108._set_reproducible_seed(int(args.seed))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    anchor_zip = Path(args.anchor_zip) if args.anchor_zip else None
    anchor_npy = Path(args.anchor_npy)
    if not anchor_npy.is_file():
        if anchor_zip is None:
            raise FileNotFoundError(f"anchor array does not exist: {anchor_npy}")
        _extract_anchor(anchor_zip, anchor_npy)

    data = load_round1_data(DEFAULT_PATHS, mmap_channel=True)
    geom = MapGeometry.from_ply(args.ply)
    support = V100._load_or_build_support(data, Path(args.support_cache), bool(args.force_support_cache), int(args.support_batch))
    train_blocked, test_blocked = V084.height_geometry_cache(DEFAULT_PATHS, out_dir, args)
    anchor = np.load(anchor_npy, mmap_mode="r")
    if anchor.shape != (data.test_pos.shape[0], 256, 4, 192):
        raise ValueError(f"unexpected anchor shape {anchor.shape}")

    train_pos = data.train_pos
    test_pos = data.test_pos
    train_support = np.asarray(support, dtype=np.float32)
    train_summary = V108._support_summary_features(train_support, data.setup, float(args.power_floor))
    summary_mean = np.mean(train_summary, axis=0, keepdims=True)
    summary_std = np.maximum(np.std(train_summary, axis=0, keepdims=True), 1e-3)
    train_summary_n = ((train_summary - summary_mean) / summary_std).astype(np.float32)

    los_train = geom.los_min_dist(train_pos) >= float(args.los_clear_dist)
    los_test = geom.los_min_dist(test_pos) >= float(args.los_clear_dist)
    train_nb = V108._neighbor_indices(train_pos, train_pos, int(args.attn_k), exclude_self=True)
    test_nb = V108._neighbor_indices(train_pos, test_pos, int(args.attn_k), exclude_self=False)
    train_edge = V108._edge_features(train_pos, train_pos, train_nb, train_blocked, train_blocked, los_train, los_train)
    test_edge = V108._edge_features(test_pos, train_pos, test_nb, test_blocked, train_blocked, los_test, los_train)
    edge_mean = np.mean(train_edge.reshape(-1, train_edge.shape[-1]), axis=0).astype(np.float32)
    edge_std = np.maximum(np.std(train_edge.reshape(-1, train_edge.shape[-1]), axis=0), 1e-3).astype(np.float32)
    train_edge_n = ((train_edge - edge_mean[None, None, :]) / edge_std[None, None, :]).astype(np.float32)

    model, train_info = V108._train_attention(train_edge_n, train_nb, train_summary_n, train_summary_n, args)
    test_weights = V108._predict_attention_weights(model, test_edge, (edge_mean, edge_std), args)
    pred_support = V108._predict_full_support(test_weights, test_nb, train_support, args)
    np.save(out_dir / "graph_attention_pred_support.npy", pred_support)
    np.save(out_dir / "test_attention_weights.npy", test_weights)
    np.save(out_dir / "test_neighbor_idx.npy", test_nb)
    np.save(out_dir / "test_blocked.npy", test_blocked)

    pred_path = out_dir / "Round1_Test_Channel.npy"
    pred = np.lib.format.open_memmap(pred_path, mode="w+", dtype=np.complex64, shape=anchor.shape)
    for start in range(0, anchor.shape[0], 8):
        end = min(start + 8, anchor.shape[0])
        pred[start:end] = V108._reconstruct_profile(
            np.asarray(anchor[start:end], dtype=np.complex64),
            np.asarray(pred_support[start:end], dtype=np.float32),
            data.setup,
            test_blocked[start:end],
            PROFILE_LEARNED,
            args,
        )
        pred.flush()
        print(f"wrote v108 attention prediction {end}/{anchor.shape[0]}", flush=True)
    stats = _pair_stats(pred, anchor, 8)

    payload: dict[str, Any] = {
        "method": "v108_graph_attention_support_completion",
        "anchor_zip": str(anchor_zip) if anchor_zip else None,
        "anchor_npy": str(anchor_npy),
        "prediction": str(pred_path),
        "support_path": str(out_dir / "graph_attention_pred_support.npy"),
        "alpha_profile": {
            "clear": PROFILE_LEARNED[0],
            "light": PROFILE_LEARNED[1],
            "medium": PROFILE_LEARNED[2],
            "heavy": PROFILE_LEARNED[3],
        },
        "attention": {
            "k": int(args.attn_k),
            "hidden": int(args.hidden),
            "epochs": int(args.epochs),
            "train_info": train_info,
            "weight_entropy_mean": float(np.mean(-np.sum(test_weights * np.log(np.maximum(test_weights, 1e-12)), axis=1))),
            "weight_max_mean": float(np.mean(np.max(test_weights, axis=1))),
        },
        "drift_vs_anchor": stats,
        "notes": [
            "Graph attention learns geometry-conditioned neighbor weights from train support summaries.",
            "Full AD support is attention-completed in log-support space.",
            "Complex phase/texture remains locked to the v91 anchor.",
            "No test labels or official feedback are used in training.",
        ],
    }
    if args.write_zip:
        zip_path = out_dir / "submission.zip"
        _zip_prediction(pred_path, zip_path)
        payload["submission_zip"] = str(zip_path)
    (out_dir / "summary.json").write_text(json.dumps(_jsonify(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(_jsonify(payload), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
