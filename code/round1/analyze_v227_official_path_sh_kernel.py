#!/usr/bin/env python3
"""v227 analysis: official path-kernel leaked S-H prior.

v226 proved that public point-cloud path candidates contain a small positive
delay/PDP signal.  This probe asks whether the same candidates also provide
horizontal leaked-PAS peak locations.  It builds a public-geometry kernel over
the scorer-like leaked S-H grid using candidate delay and TX direction, then
applies only bounded amplitude gains on top of the best v226 carrier.

This remains analysis-only.  It does not use Sionna generated channel labels,
v212/v217 teachers, synthetic truth, or hidden online labels for generation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import analyze_v208_strict_historical_replay as v208
import analyze_v226_official_path_kernel_prior as v226
import generate_v187_leaked_pas_lifecycle_projection as leak
import generate_v209_adaptive_metric_projection_controller as v209
from generate_v87_easy_field_projection import MH, MP, MV, N_ANT, S_SC, summarize
from generate_v171_full_complex_ad_tail_denoising import top_mask
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 227
VERSION = "v227_official_path_sh_kernel"

OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / "v227_official_path_sh_kernel.json"
ACTION_OUT = OUT_DIR / "v227_official_path_sh_kernel_actions.npz"

PATH_CACHE = PATHS.shared_features / "multibounce_path_candidates.npz"
GEOM_CACHE = PATHS.shared_features / "height_geometry_cache.npz"
V224_LEAKED_TARGET = PATHS.cache_dir(224) / "v224_leaked_mix0.58_it8.npy"
SOURCE = PATHS.cache_dir(226) / "v226_early_shift_v224_path_heavy_Round1_Test_Channel.npy"

FLOOR = 1e-12
PROJ_BATCH = 10


def normalize(x: np.ndarray, axis: tuple[int, ...] | int) -> np.ndarray:
    out = np.maximum(np.asarray(x, dtype=np.float32), FLOOR)
    out /= np.maximum(out.sum(axis=axis, keepdims=True), TINY)
    return out.astype(np.float32)


def circular_gaussian(size: int, centers: np.ndarray, sigma: float) -> np.ndarray:
    bins = np.arange(size, dtype=np.float32)
    delta = np.abs(bins[None, :] - centers[:, None])
    delta = np.minimum(delta, size - delta)
    return np.exp(-0.5 * (delta / max(float(sigma), 1e-3)) ** 2).astype(np.float32)


def path_weights(prefix: str, temperature: float, type_bias: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    cache = np.load(PATH_CACHE)
    valid = cache[f"{prefix}_valid"].astype(bool)
    path_type = cache[f"{prefix}_path_type"].astype(np.int64)
    score = np.nan_to_num(cache[f"{prefix}_score"].astype(np.float32), nan=-4.0, posinf=4.0, neginf=-4.0)
    excess = np.nan_to_num(cache[f"{prefix}_excess_length"].astype(np.float32), nan=220.0, posinf=220.0, neginf=0.0)
    inc = np.nan_to_num(cache[f"{prefix}_incidence_cos"].astype(np.float32), nan=1.0, posinf=1.0, neginf=1e-3)
    logits = temperature * score + np.take(np.asarray(type_bias, dtype=np.float32), np.clip(path_type, 0, 2))
    logits += 0.18 * np.log(np.maximum(inc, 1e-3)) - 0.006 * np.maximum(excess, 0.0)
    logits = np.where(valid, logits, -1e9)
    logits -= np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits).astype(np.float32)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), TINY)
    return weights, valid


def path_sh_kernel(
    orientation_deg: float,
    s_mode: str,
    h_mode: str,
    temperature: float,
    s_sigma: float,
    h_sigma: float,
    type_bias: tuple[float, float, float],
    prefix: str = "test",
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    cache = np.load(PATH_CACHE)
    weights, valid = path_weights(prefix, temperature, type_bias)
    delay = cache[f"{prefix}_delay_bin"].astype(np.float32)
    tx = cache[f"{prefix}_tx_dir"].astype(np.float32)
    az = np.arctan2(tx[:, :, 1], tx[:, :, 0]) - math.radians(float(orientation_deg))
    if h_mode == "sin":
        h_center = (np.sin(az) + 1.0) * 0.5 * MH
    elif h_mode == "cos":
        h_center = (np.cos(az) + 1.0) * 0.5 * MH
    else:
        # Direct azimuth winding is a fallback for non-ULA interpretation.
        h_center = ((az + math.pi) / (2.0 * math.pi)) * MH
    h_center = np.mod(h_center.astype(np.float32), float(MH))
    if s_mode == "delay":
        s_center = np.mod(delay, float(S_SC))
    elif s_mode == "neg_delay":
        s_center = np.mod(-delay, float(S_SC))
    else:
        s_center = np.mod(S_SC - delay, float(S_SC))

    out = np.zeros((len(weights), S_SC, MH), dtype=np.float32)
    for i in range(len(weights)):
        valid_i = np.flatnonzero(valid[i])
        if len(valid_i) == 0:
            out[i, 0, 0] = 1.0
            continue
        s_k = circular_gaussian(S_SC, s_center[i, valid_i], s_sigma)
        h_k = circular_gaussian(MH, h_center[i, valid_i], h_sigma)
        field = np.einsum("k,ks,kh->sh", weights[i, valid_i], s_k, h_k, optimize=True)
        out[i] = field.astype(np.float32)
    out = normalize(out, axis=(1, 2))
    diag = {
        "weighted_h": np.sum(weights * h_center, axis=1).astype(np.float32),
        "weighted_s": np.sum(weights * s_center, axis=1).astype(np.float32),
        "top_weight": weights.max(axis=1).astype(np.float32),
    }
    return out, diag


def current_sh(source_path: Path) -> np.ndarray:
    source = np.load(source_path, mmap_mode="r")
    parts = []
    for start in range(0, len(source), PROJ_BATCH):
        end = min(start + PROJ_BATCH, len(source))
        power = leak.leaked_cell_power(leak.leaked_forward(np.asarray(source[start:end], dtype=np.complex64)))
        sh = power.sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32)
        parts.append(normalize(sh, axis=(1, 2)))
    return np.concatenate(parts, axis=0).astype(np.float32)


def graph_sh() -> np.ndarray:
    support = np.load(V224_LEAKED_TARGET, mmap_mode="r")
    sh = np.asarray(support).sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32)
    return normalize(sh, axis=(1, 2))


def apply_sh_target(
    source_path: Path,
    output_path: Path,
    target_sh: np.ndarray,
    regimes: np.ndarray,
    alpha: dict[str, float],
    gain_scale: float,
) -> dict[str, np.ndarray]:
    source = np.load(source_path, mmap_mode="r")
    output = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.complex64, shape=source.shape)
    a = np.asarray([alpha[str(r)] for r in regimes], dtype=np.float32)
    diag: dict[str, list[np.ndarray]] = {
        "sh_l1_actual": [],
        "power_factor": [],
        "gain_min": [],
        "gain_p90": [],
        "gain_max": [],
    }
    for start in range(0, len(source), PROJ_BATCH):
        end = min(start + PROJ_BATCH, len(source))
        h0 = np.asarray(source[start:end], dtype=np.complex64)
        spec0 = leak.leaked_forward(h0)
        power0 = leak.leaked_cell_power(spec0)
        total0 = np.maximum(power0.sum(axis=(1, 2, 3, 4, 5)), TINY).astype(np.float32)
        support = power0 / total0[:, None, None, None, None, None]
        sh0 = normalize(support.sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32), axis=(1, 2))
        desired_sh = normalize(
            (1.0 - a[start:end, None, None]) * sh0
            + a[start:end, None, None] * target_sh[start:end],
            axis=(1, 2),
        )
        work = support.copy()
        for _ in range(5):
            v209.apply_leaked_ratio(
                work,
                np.maximum(desired_sh, FLOOR)
                / np.maximum(work.sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32), FLOOR),
                "sh",
                0.88,
            )
            work = normalize(work.reshape(end - start, -1), axis=1).reshape(support.shape)
        gain = np.sqrt((work * total0[:, None, None, None, None, None]) / np.maximum(power0, 1e-30)).astype(np.float32)
        gain = np.clip(gain, 0.20, 5.60 * gain_scale)
        flat = support.reshape(end - start, -1)
        strong128 = top_mask(flat, 128).reshape(support.shape)
        strong1024 = top_mask(flat, 1024).reshape(support.shape)
        strong8192 = top_mask(flat, 8192).reshape(support.shape)
        gain[strong8192] = np.clip(gain[strong8192], 0.50, 2.35 * gain_scale)
        gain[strong1024] = np.clip(gain[strong1024], 0.64, 1.72 * gain_scale)
        gain[strong128] = np.clip(gain[strong128], 0.80, 1.34 * gain_scale)
        for local, regime in enumerate(regimes[start:end].astype(str)):
            if regime == "clear":
                gain[local] = 1.0
            elif regime == "light":
                gain[local] = 1.0 + 0.28 * (gain[local] - 1.0)
        h1 = leak.leaked_inverse(spec0 * gain)
        p0 = np.maximum(np.sum(np.abs(h0) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY)
        p1 = np.maximum(np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY)
        h1 *= np.sqrt(p0 / p1)[:, None, None, None].astype(np.float32)
        output[start:end] = h1.astype(np.complex64)
        output.flush()
        power1 = leak.leaked_cell_power(leak.leaked_forward(h1.astype(np.complex64)))
        sh1 = normalize(power1.sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32), axis=(1, 2))
        diag["sh_l1_actual"].append(np.sum(np.abs(sh1 - sh0), axis=(1, 2)).astype(np.float32))
        diag["power_factor"].append(
            (np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64) / p0).astype(np.float32)
        )
        gflat = gain.reshape(end - start, -1)
        diag["gain_min"].append(gflat.min(axis=1).astype(np.float32))
        diag["gain_p90"].append(np.quantile(gflat, 0.90, axis=1).astype(np.float32))
        diag["gain_max"].append(gflat.max(axis=1).astype(np.float32))
        if end == len(source) or end % 100 == 0:
            print(f"v227 sh projection {output_path.name} {end}/{len(source)}", flush=True)
    return {key: np.concatenate(value).astype(np.float32) for key, value in diag.items()}


def run_variant(
    name: str,
    path_sh: np.ndarray,
    graph: np.ndarray,
    current: np.ndarray,
    regimes: np.ndarray,
    graph_weight: float,
    alpha: dict[str, float],
    gain_scale: float,
) -> dict[str, object]:
    target = normalize(graph_weight * graph + (1.0 - graph_weight) * path_sh, axis=(1, 2))
    out = OUT_DIR / f"v227_{name}_Round1_Test_Channel.npy"
    diag = apply_sh_target(SOURCE, out, target, regimes, alpha, gain_scale)
    score = v208.score_prediction(out)
    return {
        "name": name,
        "source": PATHS.relative(SOURCE),
        "output": PATHS.relative(out),
        "graph_weight": graph_weight,
        "alpha": alpha,
        "gain_scale": gain_scale,
        "score": score,
        "diagnostics": {key: summarize(value) for key, value in diag.items()},
        "target_l1_to_current_sh": summarize(np.sum(np.abs(target - current), axis=(1, 2))),
    }


def main() -> None:
    for path in [PATH_CACHE, GEOM_CACHE, V224_LEAKED_TARGET, SOURCE]:
        if not path.is_file():
            raise FileNotFoundError(path)
    geom = np.load(GEOM_CACHE)
    regimes = regime_of(geom["test_blocked"].astype(np.float32))
    graph = graph_sh()
    current = current_sh(SOURCE)
    variants = []
    kernel_diag: dict[str, object] = {}
    specs = [
        ("ori_m30_delay_sin", -30.0, "delay", "sin", 1.00, 4.8, 0.90, (0.38, 0.10, 0.00)),
        ("ori_m30_neg_sin", -30.0, "neg_delay", "sin", 1.00, 4.8, 0.90, (0.38, 0.10, 0.00)),
        ("ori_0_delay_sin", 0.0, "delay", "sin", 1.00, 4.8, 0.90, (0.38, 0.10, 0.00)),
        ("ori_30_delay_sin", 30.0, "delay", "sin", 1.00, 4.8, 0.90, (0.38, 0.10, 0.00)),
        ("ori_m30_delay_az", -30.0, "delay", "az", 0.90, 5.4, 1.20, (0.22, 0.14, 0.06)),
        ("ori_0_delay_az", 0.0, "delay", "az", 0.90, 5.4, 1.20, (0.22, 0.14, 0.06)),
    ]
    for kname, ori, smode, hmode, temp, ssig, hsig, bias in specs:
        path_sh, diag = path_sh_kernel(ori, smode, hmode, temp, ssig, hsig, bias)
        kernel_diag[kname] = {key: summarize(value) for key, value in diag.items()}
        for suffix, graph_weight, alpha, gain_scale in [
            ("moderate", 0.64, {"clear": 0.0, "light": 0.04, "medium": 0.26, "heavy": 0.38}, 1.0),
            ("strong", 0.42, {"clear": 0.0, "light": 0.06, "medium": 0.42, "heavy": 0.58}, 1.08),
        ]:
            result = run_variant(f"{kname}_{suffix}", path_sh, graph, current, regimes, graph_weight, alpha, gain_scale)
            variants.append(result)
            print(json.dumps({"name": result["name"], "score": result["score"]}, ensure_ascii=False), flush=True)
    best = max(variants, key=lambda item: item["score"]["score_traceback_leaked"])
    np.savez_compressed(ACTION_OUT, regimes=regimes.astype("U8"))
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": "Official path candidate delay plus TX direction may provide leaked S-H peak positions missing from v224/v226.",
        "compliance_boundary": {
            "generation_uses": [
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
            "note": "Deterministic analysis-only path S-H prior; no ZIP unless local v212 gate is exceeded.",
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
