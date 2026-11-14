#!/usr/bin/env python3
"""v250: fixed electromagnetic path-law marginal direction audit.

This analysis asks one narrow question: does a label-free path target built
from public point-cloud path candidates point from the strict v239 carrier
toward the local diagnostic truth at all?  The target is fixed before reading
the diagnostic truth.  Local truth is used only to evaluate the resulting
alpha curves; no prediction archive is generated from this audit.
"""

from __future__ import annotations

import json

import numpy as np

import analyze_v208_strict_historical_replay as v208
import analyze_v226_official_path_kernel_prior as v226
import analyze_v227_official_path_sh_kernel as v227
import analyze_v247_delaunay_spectral_field as v247
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 250
VERSION = "v250_fixed_em_path_law"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"

BASE = PATHS.cache_dir(239) / "v239_path_on_v237_Round1_Test_Channel.npy"
ALPHAS = np.linspace(0.0, 1.0, 51, dtype=np.float32)


def score_marginals(
    pred_sh: np.ndarray,
    pred_d: np.ndarray,
    truth_sh: np.ndarray,
    truth_d: np.ndarray,
) -> dict[str, float]:
    sh = float(np.mean(v247.row_cos(pred_sh, truth_sh)))
    d = float(np.mean(v247.row_cos(pred_d, truth_d)))
    return {"sh": sh, "d": d, "mean": 0.5 * (sh + d)}


def alpha_curve(
    base_sh: np.ndarray,
    base_d: np.ndarray,
    target_sh: np.ndarray,
    target_d: np.ndarray,
    truth_sh: np.ndarray,
    truth_d: np.ndarray,
    mode: str,
) -> list[dict[str, float]]:
    curve = []
    for alpha in ALPHAS:
        sh_alpha = float(alpha) if mode in {"joint", "sh_only"} else 0.0
        d_alpha = float(alpha) if mode in {"joint", "d_only"} else 0.0
        pred_sh = v247.normalize((1.0 - sh_alpha) * base_sh + sh_alpha * target_sh)
        pred_d = v247.normalize((1.0 - d_alpha) * base_d + d_alpha * target_d)
        curve.append(
            {
                "alpha": float(alpha),
                **score_marginals(pred_sh, pred_d, truth_sh, truth_d),
            }
        )
    return curve


def main() -> None:
    for path in [BASE, v208.TEST_TRUTH, v226.PATH_CACHE]:
        if not path.is_file():
            raise FileNotFoundError(path)

    base_h = np.load(BASE, mmap_mode="r")
    truth_h = np.load(v208.TEST_TRUTH, mmap_mode="r")
    base_sh, base_d = v247.extract_current(np.asarray(base_h, dtype=np.complex64))
    truth_sh, truth_d = v247.extract_current(np.asarray(truth_h, dtype=np.complex64))

    target_d, d_diag = v226.path_kernel_d(
        "test",
        temperature=1.05,
        sigma_los=0.55,
        sigma_reflect=1.05,
        type_bias=(0.42, 0.08, -0.04),
        tail_weight=0.01,
        delay_shift=0.0,
    )
    target_sh, sh_diag = v227.path_sh_kernel(
        orientation_deg=-30.0,
        s_mode="neg_delay",
        h_mode="sin",
        temperature=1.05,
        s_sigma=4.8,
        h_sigma=0.9,
        type_bias=(0.42, 0.08, -0.04),
        prefix="test",
    )

    curves = {
        mode: alpha_curve(
            base_sh,
            base_d,
            target_sh,
            target_d,
            truth_sh,
            truth_d,
            mode,
        )
        for mode in ("joint", "sh_only", "d_only")
    }
    best = {
        mode: max(curve, key=lambda row: row["mean"])
        for mode, curve in curves.items()
    }
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "A fixed label-free electromagnetic path law may provide a positive "
            "S-H/PDP direction even when official-spectrum regressors do not."
        ),
        "compliance_boundary": {
            "target_uses": [
                "official point-cloud multibounce path candidates",
                "fixed free-space/reflection path weighting",
                "fixed S=-delay and ULA steering law",
            ],
            "target_does_not_use": [
                "synthetic/Sionna channel labels",
                "v212/v217 teacher outputs",
                "hidden online labels",
                "local diagnostic truth",
            ],
            "local_truth_usage": (
                "retrospective alpha-direction audit only; no output archive is generated"
            ),
        },
        "fixed_parameters": {
            "temperature": 1.05,
            "sigma_los": 0.55,
            "sigma_reflect": 1.05,
            "type_bias": [0.42, 0.08, -0.04],
            "tail_weight": 0.01,
            "delay_shift": 0.0,
            "orientation_deg": -30.0,
            "s_mode": "neg_delay",
            "h_mode": "sin",
            "s_sigma": 4.8,
            "h_sigma": 0.9,
        },
        "base": score_marginals(base_sh, base_d, truth_sh, truth_d),
        "target": score_marginals(target_sh, target_d, truth_sh, truth_d),
        "best": best,
        "positive_direction": {
            mode: bool(row["alpha"] > 0.0 and row["mean"] > curves[mode][0]["mean"])
            for mode, row in best.items()
        },
        "curves": curves,
        "diagnostics": {
            "d": {key: summarize(value) for key, value in d_diag.items()},
            "sh": {key: summarize(value) for key, value in sh_diag.items()},
        },
        "outputs": {"report": PATHS.relative(REPORT), "zip": None},
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"base": report["base"], "target": report["target"], "best": best}, indent=2))


if __name__ == "__main__":
    main()
