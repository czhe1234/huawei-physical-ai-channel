#!/usr/bin/env python3
"""Build consistent train/test features for the explicit-path energy model."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from build_geodesic_corridor_features import build_features as build_geodesic_features
from project_paths import PATHS
from round2.data import load_positions
from run_v6_regime_component_lifecycle import physical_risk_and_regime
from train_explicit_path_pdp_predictor import build_features


TRAIN_CACHE = (
    PATHS.artifacts
    / "round2"
    / "joint_path_token_oracle"
    / "n4000_s24_k128_b096_d064_scatter1_sel106"
)
TEST_CACHE = (
    PATHS.artifacts
    / "round2"
    / "joint_path_token_oracle"
    / "test_n500_s24_k128_b096_d064_scatter1_sel106"
)
ARTIFACT_DIR = PATHS.artifacts / "round2" / "explicit_path_pdp_predictor"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--presence-seed-count", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    _, train_positions, test_positions, train_site, test_site = load_positions()
    train_geodesic, test_geodesic, names = build_geodesic_features()
    full_fit = np.arange(len(train_positions), dtype=np.int64)
    _, _, train_regime, test_regime, _ = physical_risk_and_regime(
        train_geodesic,
        test_geodesic,
        train_site,
        test_site,
        full_fit,
        names,
    )
    if args.split == "train":
        count = len(train_positions)
        path_cache = TRAIN_CACHE
        regime = train_regime
    else:
        count = len(test_positions)
        path_cache = TEST_CACHE
        regime = test_regime
    missing = [
        PATHS.relative(path)
        for path in (path_cache,)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"missing path caches: {missing}")
    point_ids = np.arange(count, dtype=np.int64)
    rows = np.arange(count, dtype=np.int64)
    output = ARTIFACT_DIR / f"path_features_{args.split}_n{count}.npy"
    features = build_features(
        rows,
        point_ids,
        regime,
        output,
        path_cache=path_cache,
        split=args.split,
        presence_seed_count=args.presence_seed_count,
    )
    regime_path = ARTIFACT_DIR / f"path_regime_{args.split}_n{count}.npy"
    np.save(regime_path, regime)
    report = {
        "split": args.split,
        "point_count": count,
        "presence_seed_count": int(args.presence_seed_count),
        "feature_shape": list(features.shape),
        "feature_path": PATHS.relative(output),
        "regime_path": PATHS.relative(regime_path),
        "seconds": float(time.time() - started),
    }
    report_path = ARTIFACT_DIR / f"path_features_{args.split}_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
