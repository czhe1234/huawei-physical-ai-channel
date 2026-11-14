#!/usr/bin/env python3
"""Generate v135: exact-zero-only-pruned bad residual templates.

v134 excluded train_total_power <= 1e-12 and dropped online.  v135 isolates the
hypothesis: remove only the 16 exact-zero train points while keeping near-zero
but nonzero samples, because those may still encode valid weak propagation.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

from generate_v128_badpoint_support_residual_templates import main as template_main
from project_paths import PATHS
from powerfield_common import ASSETS


ZIP_OUT = str(PATHS.round1_versions / "v135_exact_zero_pruned_templates_candidate_pending.zip")
REPORT = ASSETS + "/v135_exact_zero_pruned_templates_generate.json"


DEFAULT_ARGS = [
    "--base",
    ASSETS + "/cache_v129/v129_Round1_Test_Channel.npy",
    "--output",
    ASSETS + "/cache_v135/v135_Round1_Test_Channel.npy",
    "--current-target",
    ASSETS + "/cache_v129/v129_apex_badpoint_support_residual_target.npy",
    "--train-base-support-out",
    ASSETS + "/cache_v128/train_v125eq_collapsed_support.npy",
    "--target-out",
    ASSETS + "/cache_v135/v135_exact_zero_pruned_target.npy",
    "--weights-out",
    ASSETS + "/cache_v135/v135_exact_zero_pruned_weights.npy",
    "--prob-out",
    ASSETS + "/cache_v135/v135_exact_zero_pruned_probs.npy",
    "--label-out",
    ASSETS + "/cache_v135/v135_exact_zero_pruned_train_labels.npy",
    "--report",
    REPORT,
    "--zip-out",
    ZIP_OUT,
    "--seed",
    "13507",
    "--bad-quantile",
    "0.52",
    "--min-bad-cos",
    "0.975",
    "--min-train-power",
    "0",
    "--clusters",
    "112",
    "--svd-components",
    "112",
    "--ratio-clip",
    "0.20,6.60",
    "--k-bad",
    "48",
    "--bad-power",
    "3.70",
    "--bad-floor",
    "0.34",
    "--proto-temp",
    "0.014",
    "--bad-weight",
    "0.86",
    "--proto-weight",
    "0.14",
    "--eta",
    "clear:0.012,light:0.024,medium:0.760,heavy:1.060",
    "--min-gate",
    "0.18",
    "--max-gate",
    "3.55",
    "--conf-center",
    "0.15",
    "--conf-width",
    "0.18",
    "--entropy-low",
    "0.16",
    "--entropy-high",
    "0.99",
    "--gain-clip",
    "0.30,4.60",
    "--top-gain-clip",
    "0.40,3.55",
    "--top-components",
    "3840",
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

    template_main()

    report_path = Path(_arg_value("--report", REPORT))
    if report_path.exists():
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        payload["method"] = "v135_exact_zero_pruned_templates"
        payload["parent_method"] = "v134_zero_power_pruned_templates"
        payload["operation"] = (
            "rerun bad residual template mining from v129 base while excluding "
            "only exact-zero train points; keep near-zero nonzero samples"
        )
        report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    zip_out = Path(_arg_value("--zip-out", ZIP_OUT))
    with zipfile.ZipFile(zip_out, "a", compression=zipfile.ZIP_STORED) as zf:
        zf.write(__file__, "generate_v135_exact_zero_pruned_templates.py")
    print(f"appended v135 wrapper source to {zip_out}", flush=True)


if __name__ == "__main__":
    main()
