#!/usr/bin/env python3
"""v228 analysis: train-calibrated official path -> leaked S-H map.

v227 showed that hand-guessed path direction to leaked-H mapping is negative.
v228 calibrates that global mapping on released train H first, using only
official train channels and public point-cloud path candidates.  The calibrated
path S-H prior is then tested as a bounded projection on top of the v226 carrier.
"""

from __future__ import annotations

import json

import numpy as np

import analyze_v208_strict_historical_replay as v208
import analyze_v227_official_path_sh_kernel as v227
import generate_v209_adaptive_metric_projection_controller as v209
from generate_v87_easy_field_projection import summarize
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 228
VERSION = "v228_official_calibrated_path_sh"

OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / "v228_official_calibrated_path_sh.json"
ACTION_OUT = OUT_DIR / "v228_official_calibrated_path_sh_actions.npz"

GEOM_CACHE = PATHS.shared_features / "height_geometry_cache.npz"
SOURCE = PATHS.cache_dir(226) / "v226_early_shift_v224_path_heavy_Round1_Test_Channel.npy"

LOCAL_GATE = 0.645604


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = a.reshape(len(a), -1).astype(np.float64)
    y = b.reshape(len(b), -1).astype(np.float64)
    return (
        np.sum(x * y, axis=1)
        / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30)
    ).astype(np.float32)


def select_calibration_ids(valid: np.ndarray, count: int = 640) -> np.ndarray:
    ids = np.flatnonzero(valid)
    # Deterministic spatially spread sample: keep every stride and cap.  This
    # is enough to rank global orientation/mapping candidates cheaply.
    stride = max(len(ids) // count, 1)
    picked = ids[::stride][:count]
    if len(picked) < min(count, len(ids)):
        picked = ids[: min(count, len(ids))]
    return picked.astype(np.int64)


def calibrate_path_sh() -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
    train_m = v209.build_train_leaked_marginals()
    valid = train_m["valid"].astype(bool)
    target = train_m["sh"].astype(np.float32)
    sample_ids = select_calibration_ids(valid, 640)
    regimes = regime_of(np.load(GEOM_CACHE)["train_blocked"].astype(np.float32))[sample_ids]

    specs = []
    for s_mode in ("delay", "neg_delay"):
        for h_mode in ("sin", "cos", "az"):
            for orientation in (-120, -90, -60, -30, 0, 30, 60, 90, 120):
                specs.append((float(orientation), s_mode, h_mode))
    results: list[dict[str, object]] = []
    for orientation, s_mode, h_mode in specs:
        pred, diag = v227.path_sh_kernel(
            orientation,
            s_mode,
            h_mode,
            temperature=0.95,
            s_sigma=5.2,
            h_sigma=1.25,
            type_bias=(0.30, 0.12, 0.02),
            prefix="train",
        )
        score = row_cos(pred[sample_ids], target[sample_ids])
        by_regime = {}
        for regime in ("clear", "light", "medium", "heavy"):
            mask = regimes.astype(str) == regime
            if np.any(mask):
                by_regime[regime] = float(np.mean(score[mask]))
        item = {
            "orientation": orientation,
            "s_mode": s_mode,
            "h_mode": h_mode,
            "mean_cos": float(np.mean(score)),
            "p10_cos": float(np.quantile(score, 0.10)),
            "by_regime": by_regime,
            "diag": {key: summarize(value[sample_ids]) for key, value in diag.items()},
        }
        results.append(item)
        print(f"v228 calibration {orientation} {s_mode} {h_mode} mean={item['mean_cos']:.6f}", flush=True)
    results.sort(key=lambda item: float(item["mean_cos"]), reverse=True)
    return results, target, sample_ids


def run_test_variants(calibration: list[dict[str, object]]) -> list[dict[str, object]]:
    geom = np.load(GEOM_CACHE)
    regimes = regime_of(geom["test_blocked"].astype(np.float32))
    graph = v227.graph_sh()
    current = v227.current_sh(SOURCE)
    variants = []
    for rank, spec in enumerate(calibration[:6], start=1):
        path_sh, _diag = v227.path_sh_kernel(
            float(spec["orientation"]),
            str(spec["s_mode"]),
            str(spec["h_mode"]),
            temperature=0.95,
            s_sigma=5.2,
            h_sigma=1.25,
            type_bias=(0.30, 0.12, 0.02),
            prefix="test",
        )
        for suffix, graph_weight, alpha, gain_scale in [
            ("cal_moderate", 0.76, {"clear": 0.0, "light": 0.025, "medium": 0.18, "heavy": 0.28}, 0.92),
            ("cal_strong", 0.62, {"clear": 0.0, "light": 0.040, "medium": 0.30, "heavy": 0.42}, 1.00),
        ]:
            result = v227.run_variant(
                f"rank{rank}_{spec['s_mode']}_{spec['h_mode']}_{int(float(spec['orientation']))}_{suffix}",
                path_sh,
                graph,
                current,
                regimes,
                graph_weight,
                alpha,
                gain_scale,
            )
            result["calibration_rank"] = rank
            result["calibration_mean_cos"] = spec["mean_cos"]
            variants.append(result)
            print(json.dumps({"name": result["name"], "score": result["score"]}, ensure_ascii=False), flush=True)
    return variants


def main() -> None:
    for path in [GEOM_CACHE, SOURCE, v227.PATH_CACHE, v227.V224_LEAKED_TARGET]:
        if not path.is_file():
            raise FileNotFoundError(path)
    calibration, _target, sample_ids = calibrate_path_sh()
    variants = run_test_variants(calibration)
    best = max(variants, key=lambda item: item["score"]["score_traceback_leaked"])
    np.savez_compressed(ACTION_OUT, sample_ids=sample_ids)
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": "Official train-H can calibrate the global mapping from public path directions to leaked S-H bins.",
        "compliance_boundary": {
            "generation_uses": [
                "official train H leaked S-H marginals for global calibration",
                "official point-cloud multibounce path candidates",
                "v226 official-data carrier",
                "v224 official graph leaked anchor",
            ],
            "generation_does_not_use": [
                "Sionna generated channel labels",
                "v212/v217 teacher outputs",
                "hidden online labels",
                "synthetic replay truth",
            ],
            "note": "Analysis-only calibrated path S-H prior; no ZIP unless local v212 gate is exceeded.",
        },
        "local_gate_required": LOCAL_GATE,
        "calibration_top12": calibration[:12],
        "variants": variants,
        "best_variant": best["name"],
        "best_score_traceback_leaked": best["score"]["score_traceback_leaked"],
        "gate_passed": bool(best["score"]["score_traceback_leaked"] >= LOCAL_GATE),
        "outputs": {
            "actions": PATHS.relative(ACTION_OUT),
            "report": PATHS.relative(REPORT),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
