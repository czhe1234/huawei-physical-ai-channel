#!/usr/bin/env python3
"""Generate v129: apex bad-point support residual templates.

v128 validated that train bad-point residual support patches contain
transferable signal.  v129 is the aggressive follow-up:

* use v128 output as the complex-channel base;
* use v128 support target as the current support skeleton;
* mine a broader set of bad train residual templates;
* apply stronger medium/heavy support-only projection.

It still preserves total power and does not learn or rotate complex phase.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

from generate_v128_badpoint_support_residual_templates import main as v128_main
from project_paths import PATHS
from powerfield_common import ASSETS


ZIP_OUT = str(PATHS.round1_versions / "v129_apex_badpoint_support_residual_templates_candidate_pending.zip")
REPORT = ASSETS + "/v129_apex_badpoint_support_residual_templates_generate.json"


DEFAULT_ARGS = [
    "--base",
    ASSETS + "/cache_v128/v128_Round1_Test_Channel.npy",
    "--output",
    ASSETS + "/cache_v129/v129_Round1_Test_Channel.npy",
    "--current-target",
    ASSETS + "/cache_v128/v128_badpoint_support_residual_target.npy",
    "--train-base-support-out",
    ASSETS + "/cache_v128/train_v125eq_collapsed_support.npy",
    "--target-out",
    ASSETS + "/cache_v129/v129_apex_badpoint_support_residual_target.npy",
    "--weights-out",
    ASSETS + "/cache_v129/v129_apex_badpoint_support_residual_weights.npy",
    "--prob-out",
    ASSETS + "/cache_v129/v129_apex_badpoint_support_residual_probs.npy",
    "--label-out",
    ASSETS + "/cache_v129/v129_apex_badpoint_residual_train_labels.npy",
    "--report",
    REPORT,
    "--zip-out",
    ZIP_OUT,
    "--seed",
    "12907",
    "--bad-quantile",
    "0.52",
    "--min-bad-cos",
    "0.975",
    "--clusters",
    "96",
    "--svd-components",
    "96",
    "--ratio-clip",
    "0.16,6.80",
    "--k-bad",
    "46",
    "--bad-power",
    "3.65",
    "--bad-floor",
    "0.34",
    "--proto-temp",
    "0.014",
    "--bad-weight",
    "0.84",
    "--proto-weight",
    "0.16",
    "--eta",
    "clear:0.0001,light:0.006,medium:0.720,heavy:1.020",
    "--min-gate",
    "0.20",
    "--max-gate",
    "3.45",
    "--conf-center",
    "0.16",
    "--conf-width",
    "0.18",
    "--entropy-low",
    "0.18",
    "--entropy-high",
    "0.99",
    "--gain-clip",
    "0.30,4.50",
    "--top-gain-clip",
    "0.40,3.50",
    "--top-components",
    "3584",
]


def _arg_value(name: str, default: str) -> str:
    argv = sys.argv[1:]
    for i, item in enumerate(argv):
        if item == name and i + 1 < len(argv):
            return argv[i + 1]
    return default


def main() -> None:
    if len(sys.argv) == 1:
        sys.argv.extend(DEFAULT_ARGS)

    v128_main()

    report_path = Path(_arg_value("--report", REPORT))
    if report_path.exists():
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        payload["method"] = "v129_apex_badpoint_support_residual_templates"
        payload["parent_method"] = "v128_badpoint_support_residual_templates"
        payload["operation"] = (
            "aggressive v128 follow-up: base=v128, current support=v128 target, "
            "broader bad residual templates, stronger medium/heavy support-only projection"
        )
        report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    zip_out = Path(_arg_value("--zip-out", ZIP_OUT))
    with zipfile.ZipFile(zip_out, "a", compression=zipfile.ZIP_STORED) as zf:
        zf.write(__file__, "generate_v129_apex_badpoint_support_residual_templates.py")
    print(f"appended v129 wrapper source to {zip_out}", flush=True)


if __name__ == "__main__":
    main()
