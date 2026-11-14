#!/usr/bin/env python3
"""v226 analysis: official point-cloud path-kernel PDP prior.

v225 showed that learning a train-H residual on top of v224 graph propagation
does not strengthen the local gate.  This probe switches the information
source: instead of learning residual texture from nearby train labels, it uses
the official point-cloud multi-bounce path candidates as a physically
interpretable delay-kernel prior.

Generation boundary:
* uses official train/test coordinates and official point-cloud-derived path
  candidates;
* uses v224/v209 official-data outputs only as current support carriers;
* does not use Sionna generated channel labels, v212/v217 teacher outputs,
  synthetic replay truth, or hidden online labels for generation.

The probe is deterministic analysis-only.  If it finds a strong signal, the
next step is to convert the kernel parameters/gates into a neural Physical-AI
module trained from official train H.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import analyze_v208_strict_historical_replay as v208
import analyze_v225_official_graph_residual_field as v225
from generate_v87_easy_field_projection import summarize
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 226
VERSION = "v226_official_path_kernel_prior"

OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / "v226_official_path_kernel_prior.json"
ACTION_OUT = OUT_DIR / "v226_official_path_kernel_actions.npz"

PATH_CACHE = PATHS.shared_features / "multibounce_path_candidates.npz"
GEOM_CACHE = PATHS.shared_features / "height_geometry_cache.npz"
BASE_V224 = PATHS.cache_dir(224) / "v224_mix0.58_it8_moderate_Round1_Test_Channel.npy"
BASE_V209 = PATHS.cache_dir(209) / "v209_Round1_Test_Channel.npy"
V224_DELAY_TARGET = PATHS.cache_dir(224) / "v224_delay_mix0.58_it8.npy"

S_SC = 192
FLOOR = 1e-12


def normalize(x: np.ndarray, axis: tuple[int, ...] | int) -> np.ndarray:
    out = np.maximum(np.asarray(x, dtype=np.float32), FLOOR)
    out /= np.maximum(out.sum(axis=axis, keepdims=True), TINY)
    return out.astype(np.float32)


def path_kernel_d(
    prefix: str,
    temperature: float,
    sigma_los: float,
    sigma_reflect: float,
    type_bias: tuple[float, float, float],
    tail_weight: float,
    delay_shift: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    cache = np.load(PATH_CACHE)
    valid = cache[f"{prefix}_valid"].astype(bool)
    path_type = cache[f"{prefix}_path_type"].astype(np.int64)
    delay = cache[f"{prefix}_delay_bin"].astype(np.float32) + np.float32(delay_shift)
    score = np.nan_to_num(cache[f"{prefix}_score"].astype(np.float32), nan=-4.0, posinf=4.0, neginf=-4.0)
    excess = np.nan_to_num(cache[f"{prefix}_excess_length"].astype(np.float32), nan=220.0, posinf=220.0, neginf=0.0)
    inc = np.nan_to_num(cache[f"{prefix}_incidence_cos"].astype(np.float32), nan=1.0, posinf=1.0, neginf=1e-3)
    bins = np.arange(S_SC, dtype=np.float32)
    logits = temperature * score + np.take(np.asarray(type_bias, dtype=np.float32), np.clip(path_type, 0, 2))
    logits += 0.18 * np.log(np.maximum(inc, 1e-3)) - 0.006 * np.maximum(excess, 0.0)
    logits = np.where(valid, logits, -1e9)
    logits -= np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits).astype(np.float32)
    empty = weights.sum(axis=1) <= 1e-20
    if np.any(empty):
        weights[empty] = valid[empty].astype(np.float32)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), TINY)
    sigma = np.where(path_type == 0, sigma_los, sigma_reflect).astype(np.float32)
    kernel = np.exp(-0.5 * ((bins[None, None, :] - delay[:, :, None]) / np.maximum(sigma[:, :, None], 0.15)) ** 2)
    kernel *= valid[:, :, None].astype(np.float32)
    d = np.sum(weights[:, :, None] * kernel.astype(np.float32), axis=1)
    # A tiny causal tail lets long reflected energy survive without flattening
    # the main peaks.  It is generated from public path excess, not labels.
    if tail_weight > 0:
        max_delay = np.nanmax(np.where(valid, delay, 0.0), axis=1)
        tail = np.exp(-np.maximum(bins[None, :] - max_delay[:, None], 0.0) / 18.0)
        tail *= bins[None, :] >= max_delay[:, None]
        d = d + np.float32(tail_weight) * tail.astype(np.float32)
    d = normalize(d, axis=1)
    diag = {
        "path_weight_entropy": -np.sum(weights * np.log(np.maximum(weights, FLOOR)), axis=1).astype(np.float32),
        "weighted_delay": np.sum(weights * delay, axis=1).astype(np.float32),
        "weighted_score": np.sum(weights * score, axis=1).astype(np.float32),
        "top_weight": weights.max(axis=1).astype(np.float32),
    }
    return d, diag


def current_delay(path: Path) -> dict[str, np.ndarray]:
    return v225.current_delay_marginals(path)


def v224_anchor_delay() -> dict[str, np.ndarray]:
    support = np.load(V224_DELAY_TARGET, mmap_mode="r")
    return v225.delay_marginals_from_support(support)


def blend_d_target(
    current: dict[str, np.ndarray],
    graph: dict[str, np.ndarray],
    path_d: np.ndarray,
    regimes: np.ndarray,
    alpha: dict[str, float],
    graph_weight: float,
) -> dict[str, np.ndarray]:
    a = np.asarray([alpha[str(r)] for r in regimes], dtype=np.float32)
    d_phys = normalize(graph_weight * graph["d"] + (1.0 - graph_weight) * path_d, axis=1)
    target_d = normalize((1.0 - a[:, None]) * current["d"] + a[:, None] * d_phys, axis=1)
    # Keep conditional UE/H allocation from the current carrier.  v226 tests
    # path-derived PDP only; mixing conditional slices caused v225 degradation.
    return {
        "d": target_d,
        "du": current["du"],
        "dh": current["dh"],
    }


def run_variant(
    name: str,
    source: Path,
    current: dict[str, np.ndarray],
    graph: dict[str, np.ndarray],
    regimes: np.ndarray,
    path_d: np.ndarray,
    alpha: dict[str, float],
    graph_weight: float,
    gain_clip_scale: float,
) -> dict[str, object]:
    target = blend_d_target(current, graph, path_d, regimes, alpha, graph_weight)
    out = OUT_DIR / f"v226_{name}_Round1_Test_Channel.npy"
    diag = v225.apply_delay_targets(source, out, target, regimes, gain_clip_scale)
    score = v208.score_prediction(out)
    return {
        "name": name,
        "source": PATHS.relative(source),
        "output": PATHS.relative(out),
        "alpha": alpha,
        "graph_weight": graph_weight,
        "gain_clip_scale": gain_clip_scale,
        "score": score,
        "diagnostics": {key: summarize(value) for key, value in diag.items()},
        "target_l1_to_current_d": summarize(v225.l1_rows(target["d"], current["d"])),
        "target_l1_to_graph_d": summarize(v225.l1_rows(target["d"], graph["d"])),
    }


def main() -> None:
    for path in [PATH_CACHE, GEOM_CACHE, BASE_V224, BASE_V209, V224_DELAY_TARGET]:
        if not path.is_file():
            raise FileNotFoundError(path)
    v225.PROJ_BATCH = 10
    geom = np.load(GEOM_CACHE)
    regimes = regime_of(geom["test_blocked"].astype(np.float32))
    current224 = current_delay(BASE_V224)
    current209 = current_delay(BASE_V209)
    graph = v224_anchor_delay()

    kernel_specs = [
        ("sharp_los", 1.05, 0.55, 1.05, (0.42, 0.08, -0.04), 0.010, 0.0),
        ("wide_reflect", 0.85, 0.72, 1.55, (0.25, 0.18, 0.06), 0.024, 0.0),
        ("late_shift", 0.95, 0.68, 1.35, (0.28, 0.18, 0.10), 0.020, 0.42),
        ("early_shift", 1.00, 0.58, 1.20, (0.38, 0.10, 0.00), 0.014, -0.25),
    ]
    variants = []
    kernel_diag: dict[str, object] = {}
    for kname, temp, sig_los, sig_ref, bias, tail, shift in kernel_specs:
        path_d, diag = path_kernel_d("test", temp, sig_los, sig_ref, bias, tail, shift)
        kernel_diag[kname] = {key: summarize(value) for key, value in diag.items()}
        for vname, source, current, alpha, graph_weight, clip in [
            (
                "v224_moderate",
                BASE_V224,
                current224,
                {"clear": 0.0, "light": 0.05, "medium": 0.34, "heavy": 0.48},
                0.64,
                1.0,
            ),
            (
                "v224_strong",
                BASE_V224,
                current224,
                {"clear": 0.0, "light": 0.08, "medium": 0.52, "heavy": 0.72},
                0.48,
                1.08,
            ),
            (
                "v224_path_heavy",
                BASE_V224,
                current224,
                {"clear": 0.0, "light": 0.06, "medium": 0.44, "heavy": 0.88},
                0.24,
                1.14,
            ),
            (
                "v209_path_heavy",
                BASE_V209,
                current209,
                {"clear": 0.0, "light": 0.08, "medium": 0.56, "heavy": 0.92},
                0.28,
                1.14,
            ),
        ]:
            result = run_variant(
                f"{kname}_{vname}",
                source,
                current,
                graph,
                regimes,
                path_d,
                alpha,
                graph_weight,
                clip,
            )
            variants.append(result)
            print(json.dumps({"name": result["name"], "score": result["score"]}, ensure_ascii=False), flush=True)

    best = max(variants, key=lambda item: item["score"]["score_traceback_leaked"])
    np.savez_compressed(ACTION_OUT, regimes=regimes.astype("U8"))
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "Official point-cloud path candidates may provide delay peak locations "
            "that v224 graph propagation smooths away."
        ),
        "compliance_boundary": {
            "generation_uses": [
                "official train/test coordinates",
                "official point-cloud multibounce path candidates",
                "v224/v209 official-data carriers",
            ],
            "generation_does_not_use": [
                "Sionna generated channel labels",
                "v212/v217 teacher outputs",
                "hidden online labels",
                "synthetic replay truth",
            ],
            "note": "Deterministic analysis-only path-kernel prior; no ZIP unless local v212 gate is exceeded.",
        },
        "local_gate_required": 0.645604,
        "kernel_diagnostics": kernel_diag,
        "variants": variants,
        "best_variant": best["name"],
        "best_score_traceback_leaked": best["score"]["score_traceback_leaked"],
        "gate_passed": bool(best["score"]["score_traceback_leaked"] >= 0.645604),
        "outputs": {
            "actions": PATHS.relative(ACTION_OUT),
            "report": PATHS.relative(REPORT),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
