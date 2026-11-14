#!/usr/bin/env python3
"""v247: strict official-data Delaunay spectral field.

Official train/test points lie almost entirely inside the 2-D training convex
hull.  This experiment predicts evaluator-sensitive leaked S-H and PDP spectra
with barycentric interpolation on the local Delaunay triangle, then projects
both marginals onto the strict v239 carrier.  All method selection uses an
official matched spatial holdout; local replay truth is read only after the
single output has been generated.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay, cKDTree

import analyze_v208_strict_historical_replay as v208
import generate_v187_leaked_pas_lifecycle_projection as leak
from evaluate_derotated_loo_grid import choose_val_indices
from generate_v87_easy_field_projection import MH, MP, MV, N_ANT, S_SC, summarize
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 247
VERSION = "v247_delaunay_spectral_field"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v247_Round1_Test_Channel.npy"
TARGETS = OUT_DIR / "v247_delaunay_spectral_targets.npz"
ACTIONS = OUT_DIR / "v247_delaunay_spectral_actions.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"

BASE = PATHS.cache_dir(239) / "v239_path_on_v237_Round1_Test_Channel.npy"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
GEOM_CACHE = PATHS.shared_features / "height_geometry_cache.npz"

LOCAL_GATE = 0.645604
FLOOR = 1e-11
VAL_COUNT = 320
BATCH = 5
OUTER_ITERS = 3
REGIMES = ("clear", "light", "medium", "heavy")


def normalize(x: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(x, dtype=np.float32), 0.0)
    total = out.reshape(len(out), -1).sum(axis=1)
    valid = total > 1e-20
    out[valid] /= total[valid].reshape((np.sum(valid),) + (1,) * (out.ndim - 1))
    out[~valid] = 0.0
    return out.astype(np.float32)


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=np.float64).reshape(len(a), -1)
    y = np.asarray(b, dtype=np.float64).reshape(len(b), -1)
    return (
        np.sum(x * y, axis=1)
        / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30)
    ).astype(np.float32)


def delaunay_weights(
    bank_pos: np.ndarray, query_pos: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tri = Delaunay(bank_pos[:, :2])
    simplex = tri.find_simplex(query_pos[:, :2])
    vertices = np.empty((len(query_pos), 3), dtype=np.int32)
    weights = np.zeros((len(query_pos), 3), dtype=np.float32)
    inside = simplex >= 0
    for index in np.flatnonzero(inside):
        transform = tri.transform[simplex[index]]
        delta = query_pos[index, :2] - transform[2]
        bary = transform[:2] @ delta
        vertices[index] = tri.simplices[simplex[index]]
        weights[index] = np.asarray([bary[0], bary[1], 1.0 - bary.sum()], dtype=np.float32)
    if np.any(~inside):
        nearest = cKDTree(bank_pos[:, :2]).query(query_pos[~inside, :2], k=1)[1]
        vertices[~inside] = np.repeat(np.asarray(nearest, dtype=np.int32)[:, None], 3, axis=1)
        weights[~inside, 0] = 1.0
    return vertices, weights, inside


def interpolate(bank: np.ndarray, vertices: np.ndarray, weights: np.ndarray) -> np.ndarray:
    output = np.zeros((len(vertices),) + bank.shape[1:], dtype=np.float32)
    shape = (len(vertices),) + (1,) * (bank.ndim - 1)
    for corner in range(3):
        output += weights[:, corner].reshape(shape) * np.asarray(
            bank[vertices[:, corner]], dtype=np.float32
        )
    return normalize(output)


def metrics(
    pred_sh: np.ndarray,
    pred_d: np.ndarray,
    truth_sh: np.ndarray,
    truth_d: np.ndarray,
) -> dict[str, float]:
    sh = float(np.mean(row_cos(pred_sh, truth_sh)))
    d = float(np.mean(row_cos(pred_d, truth_d)))
    return {"sh": sh, "d": d, "weighted": 0.52 * sh + 0.48 * d}


def extract_current(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    leaked_power = leak.leaked_cell_power(leak.leaked_forward(h))
    sh = normalize(leaked_power.sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32))
    cir = np.fft.ifft(h, axis=-1)
    arr = cir.reshape(len(h), MP, MH, MV, N_ANT, S_SC)
    d = normalize(
        (np.abs(arr).astype(np.float32) ** 2).sum(
            axis=(1, 2, 3, 4), dtype=np.float64
        ).astype(np.float32)
    )
    return sh, d


def project(
    target_sh: np.ndarray,
    target_d: np.ndarray,
    output_path: Path = OUTPUT,
) -> dict[str, np.ndarray]:
    source = np.load(BASE, mmap_mode="r")
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.complex64, shape=source.shape
    )
    diagnostics: dict[str, list[np.ndarray]] = {
        "sh_l1": [],
        "d_l1": [],
        "final_sh_cos_target": [],
        "final_d_cos_target": [],
        "power_factor": [],
        "gain_min": [],
        "gain_max": [],
    }
    for start in range(0, len(source), BATCH):
        end = min(start + BATCH, len(source))
        h0 = np.asarray(source[start:end], dtype=np.complex64)
        h = h0.copy()
        original_power = np.maximum(
            np.sum(np.abs(h0) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY
        ).astype(np.float32)
        initial_sh, initial_d = extract_current(h0)
        gain_mins = []
        gain_maxs = []
        for _ in range(OUTER_ITERS):
            spec = leak.leaked_forward(h)
            leaked_power = leak.leaked_cell_power(spec)
            current_sh = normalize(
                leaked_power.sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32)
            )
            ratio_sh = np.maximum(target_sh[start:end], FLOOR) / np.maximum(
                current_sh, FLOOR
            )
            gain_sh = np.clip(np.sqrt(ratio_sh).astype(np.float32), 0.18, 5.5)
            spec *= gain_sh[:, None, :, :, None, None]
            h = leak.leaked_inverse(spec)
            p = np.maximum(
                np.sum(np.abs(h) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY
            ).astype(np.float32)
            h *= np.sqrt(original_power / p)[:, None, None, None].astype(np.float32)

            cir = np.fft.ifft(h, axis=-1)
            arr = cir.reshape(end - start, MP, MH, MV, N_ANT, S_SC)
            arr_power = np.abs(arr).astype(np.float32) ** 2
            current_d = normalize(
                arr_power.sum(axis=(1, 2, 3, 4), dtype=np.float64).astype(np.float32)
            )
            ratio_d = np.maximum(target_d[start:end], FLOOR) / np.maximum(
                current_d, FLOOR
            )
            gain_d = np.clip(np.sqrt(ratio_d).astype(np.float32), 0.20, 5.0)
            arr *= gain_d[:, None, None, None, None, :]
            h = np.fft.fft(
                arr.reshape(end - start, MP * MH * MV, N_ANT, S_SC), axis=-1
            ).astype(np.complex64)
            p = np.maximum(
                np.sum(np.abs(h) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY
            ).astype(np.float32)
            h *= np.sqrt(original_power / p)[:, None, None, None].astype(np.float32)
            gain_mins.append(
                np.minimum(gain_sh.reshape(end - start, -1).min(axis=1), gain_d.min(axis=1))
            )
            gain_maxs.append(
                np.maximum(gain_sh.reshape(end - start, -1).max(axis=1), gain_d.max(axis=1))
            )
        if not np.isfinite(h).all():
            raise FloatingPointError(f"non-finite v247 block {start}:{end}")
        output[start:end] = h.astype(np.complex64)
        output.flush()
        final_sh, final_d = extract_current(h.astype(np.complex64))
        final_power = np.sum(np.abs(h) ** 2, axis=(1, 2, 3), dtype=np.float64)
        diagnostics["sh_l1"].append(
            np.sum(np.abs(final_sh - initial_sh), axis=(1, 2)).astype(np.float32)
        )
        diagnostics["d_l1"].append(
            np.sum(np.abs(final_d - initial_d), axis=1).astype(np.float32)
        )
        diagnostics["final_sh_cos_target"].append(
            row_cos(final_sh, target_sh[start:end])
        )
        diagnostics["final_d_cos_target"].append(row_cos(final_d, target_d[start:end]))
        diagnostics["power_factor"].append((final_power / original_power).astype(np.float32))
        diagnostics["gain_min"].append(np.min(np.stack(gain_mins), axis=0).astype(np.float32))
        diagnostics["gain_max"].append(np.max(np.stack(gain_maxs), axis=0).astype(np.float32))
        print(f"v247 projection {end}/{len(source)}", flush=True)
    return {key: np.concatenate(value).astype(np.float32) for key, value in diagnostics.items()}


def main() -> None:
    for path in [BASE, TARGET_CACHE, GEOM_CACHE]:
        if not path.is_file():
            raise FileNotFoundError(path)
    data = np.load(TARGET_CACHE)
    sh = data["sh"].astype(np.float32)
    d = data["d"].astype(np.float32)
    valid_ids = np.flatnonzero(data["valid"].astype(bool)).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    geom = np.load(GEOM_CACHE)
    train_regimes = regime_of(geom["train_blocked"].astype(np.float32)).astype(str)

    matched = choose_val_indices(
        train_pos[:, :2], test_pos[:, :2], VAL_COUNT, "matched-test", VERSION_NUMBER
    )
    val_ids = np.intersect1d(matched, valid_ids)
    fit_ids = np.setdiff1d(valid_ids, val_ids)
    val_vertices, val_weights, val_inside = delaunay_weights(
        train_pos[fit_ids], train_pos[val_ids]
    )
    val_sh = interpolate(sh[fit_ids], val_vertices, val_weights)
    val_d = interpolate(d[fit_ids], val_vertices, val_weights)
    validation = {
        "overall": metrics(val_sh, val_d, sh[val_ids], d[val_ids]),
        "inside_fraction": float(np.mean(val_inside)),
        "by_regime": {},
    }
    for regime in REGIMES:
        mask = train_regimes[val_ids] == regime
        validation["by_regime"][regime] = {
            "count": int(np.sum(mask)),
            **metrics(val_sh[mask], val_d[mask], sh[val_ids][mask], d[val_ids][mask]),
        }

    test_vertices, test_weights, test_inside = delaunay_weights(
        train_pos[valid_ids], test_pos
    )
    test_sh = interpolate(sh[valid_ids], test_vertices, test_weights)
    test_d = interpolate(d[valid_ids], test_vertices, test_weights)
    np.savez_compressed(
        TARGETS,
        sh=test_sh,
        d=test_d,
        vertices=valid_ids[test_vertices],
        weights=test_weights,
        inside=test_inside,
    )
    actions = project(test_sh, test_d)
    score = v208.score_prediction(OUTPUT)
    gate_passed = bool(float(score["score_traceback_leaked"]) >= LOCAL_GATE)
    np.savez_compressed(ACTIONS, **actions)
    report = {
        "method": VERSION,
        "status": "gate_passed_needs_packaging" if gate_passed else "analysis_only_no_candidate",
        "primary_hypothesis": (
            "The evaluator-sensitive official spectral field is locally affine inside the "
            "Delaunay triangle, and barycentric interpolation avoids the cross-boundary mixing "
            "introduced by isotropic KNN averaging."
        ),
        "compliance_boundary": {
            "generation_uses": [
                "official train H denoised leaked S-H and PDP targets",
                "official train/test coordinates",
                "v239 strict official-data carrier",
            ],
            "generation_does_not_use": [
                "Sionna or synthetic channel labels",
                "v212/v217/v221/v222 teacher output",
                "local replay truth during method selection or generation",
                "hidden labels",
                "raw RT H",
            ],
            "local_truth_usage": "one final diagnostic score after output generation",
        },
        "parameters": {
            "validation_count": int(len(val_ids)),
            "fit_count": int(len(fit_ids)),
            "outer_projection_iterations": OUTER_ITERS,
            "outside_hull_fallback": "nearest official training point",
        },
        "validation": validation,
        "test_geometry": {
            "inside_fraction": float(np.mean(test_inside)),
            "negative_weight_count": int(np.sum(test_weights < -1e-6)),
            "weights": summarize(test_weights.reshape(-1)),
        },
        "diagnostics": {key: summarize(value) for key, value in actions.items()},
        "local_gate": {
            "required_score_traceback_leaked": LOCAL_GATE,
            "score": score,
            "passed": gate_passed,
            "gap": float(score["score_traceback_leaked"]) - LOCAL_GATE,
        },
        "outputs": {
            "prediction": PATHS.relative(OUTPUT),
            "targets": PATHS.relative(TARGETS),
            "actions": PATHS.relative(ACTIONS),
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
