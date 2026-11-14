#!/usr/bin/env python3
"""Generate v102: v98-like support-target projection.

v101 showed that same-distribution component scalar labels still do not produce
a net online gain.  This version therefore disables scalar correction and uses
the v98-like LOO cache only for the collapsed joint angle-delay support target.

The generated candidate was produced by reusing
`generate_v98_path_ensemble_graph_attention_joint_ad.py` with:

    --base artifacts/version_cache/v098/v98_Round1_Test_Channel.npy
    --coll artifacts/shared_features/train_v98like_loo_joint_target.npy

That means the base complex phase and per-point total power are inherited from
v98; only the normalized joint AD power support is nudged toward a graph
attention target trained on v98-like LOO support.
"""

from __future__ import annotations

import sys

from generate_v98_path_ensemble_graph_attention_joint_ad import main
from project_paths import PATHS


DEFAULT_ARGS = [
    "--base",
    "artifacts/version_cache/v098/v98_Round1_Test_Channel.npy",
    "--output",
    "Round1_Test_Channel.npy",
    "--coll",
    "artifacts/shared_features/train_v98like_loo_joint_target.npy",
    "--report",
    str(PATHS.version_report_dir(102) / "v102_v98like_support_target_generate.json"),
    "--target-out",
    "artifacts/version_cache/v102/v102_v98like_support_target.npy",
    "--weights-out",
    "artifacts/version_cache/v102/v102_v98like_support_weights.npy",
    "--seeds",
    "10201,10202,10203",
    "--use-path-features",
    "--k",
    "16",
    "--epochs",
    "70",
    "--hidden",
    "112",
    "--eta",
    "clear:0.020,light:0.040,medium:0.240,heavy:0.300",
    "--min-gate",
    "0.10",
    "--max-gate",
    "1.65",
    "--gain-clip",
    "0.60,1.85",
    "--top-gain-clip",
    "0.72,1.50",
]


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(DEFAULT_ARGS)
    main()
