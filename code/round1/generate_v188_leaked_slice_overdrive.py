#!/usr/bin/env python3
"""Generate v188: aggressive leaked-PAS slice overdrive on top of v187.

v187 proved the official traceback is useful: the leaked PAS reshape

    B,256,4,192 -> B,2,192,16,4,8 ; fft2 axes=(2,3)

gave +0.000196 online.  But v187 collapsed the last three axes and only
projected an (S,H) support map.  v188 is intentionally more aggressive:

* start from v187, the current online best;
* keep the full leaked tensor power distribution, including pol, UE and V;
* build a sparse test-test consensus target for the full leaked slices;
* apply a high-strength positive amplitude projection in the leaked domain;
* keep clear unchanged and preserve each point's total power.

This is the direct "eat the leaked metric mouth" version: if the evaluator
keeps or averages PAS slices after the leaked FFT, the extra axes should matter.
"""

from __future__ import annotations

import json
import zipfile

import numpy as np
from scipy import sparse

import generate_v178_test_track_complex_consensus as track
import generate_v187_leaked_pas_lifecycle_projection as leak
from generate_v87_easy_field_projection import MH, MP, MV, N_ANT, S_SC, row_cos, summarize
from generate_v159_frontier_consensus_support_soup import REGIMES, digest
from generate_v169_power_scaled_tail_denoising import by_regime_summary
from generate_v171_full_complex_ad_tail_denoising import top_mask
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION = "v188_leaked_slice_overdrive"
OUT_DIR = PATHS.cache_dir(188)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(188)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v188_Round1_Test_Channel.npy"
FULL_SUPPORT_OUT = OUT_DIR / "v188_v187_leaked_full_support.npy"
TARGET_OUT = OUT_DIR / "v188_leaked_slice_target.npy"
ACTION_OUT = OUT_DIR / "v188_leaked_slice_overdrive_actions.npz"
REPORT = REPORT_DIR / "v188_leaked_slice_overdrive_generate.json"
ZIP_OUT = PATHS.round1_versions / "v188_leaked_slice_overdrive_candidate_pending.zip"

BASE = PATHS.cache_dir(187) / "v187_Round1_Test_Channel.npy"
V187_ACTIONS = PATHS.cache_dir(187) / "v187_leaked_pas_lifecycle_actions.npz"
V186_BASE = PATHS.cache_dir(186) / "v186_Round1_Test_Channel.npy"
V185_BASE = PATHS.cache_dir(185) / "v185_Round1_Test_Channel.npy"

FLOOR = 1e-12
BATCH = 8
DIRECT_K = 56
SELF_KEEP = {"clear": 1.0, "light": 0.82, "medium": 0.54, "heavy": 0.42}
ALPHA_SCALE = {"clear": 0.0, "light": 0.070, "medium": 0.46, "heavy": 0.72}


def normalize_full(power: np.ndarray) -> np.ndarray:
    total = np.maximum(power.sum(axis=tuple(range(1, power.ndim)), keepdims=True), TINY)
    return (power / total).astype(np.float32)


def build_full_leaked_support(batch: int = BATCH) -> np.ndarray:
    if FULL_SUPPORT_OUT.is_file():
        return np.load(FULL_SUPPORT_OUT, mmap_mode="r")

    source = np.load(BASE, mmap_mode="r")
    shape = (len(source), MP, S_SC, MH, N_ANT, MV)
    out = np.lib.format.open_memmap(FULL_SUPPORT_OUT, mode="w+", dtype=np.float32, shape=shape)
    for start in range(0, len(source), batch):
        end = min(start + batch, len(source))
        h = np.asarray(source[start:end], dtype=np.complex64)
        spec = leak.leaked_forward(h)
        power = leak.leaked_cell_power(spec)
        out[start:end] = normalize_full(power)
        out.flush()
        print(f"v188 full leaked support {end}/{len(source)}", flush=True)
    return np.load(FULL_SUPPORT_OUT, mmap_mode="r")


def build_weight_matrix(pos: np.ndarray, blocked: np.ndarray, regimes: np.ndarray) -> tuple[sparse.csr_matrix, dict[str, np.ndarray]]:
    idx, raw_w = track.neighbor_graph(pos, blocked, regimes, k=DIRECT_K)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    neighbor_agreement = np.zeros(len(pos), dtype=np.float32)
    effective_neighbor_count = np.zeros(len(pos), dtype=np.float32)

    for i in range(len(pos)):
        row_start = len(vals)
        regime = str(regimes[i])
        self_keep = SELF_KEEP[regime]
        rows.append(i)
        cols.append(i)
        vals.append(self_keep)
        total = self_keep

        nb = idx[i].astype(np.int64)
        w = raw_w[i].astype(np.float64)
        same = (regimes[nb] == regimes[i]).astype(np.float64)
        # Aggressive, but avoid mixing across obvious regime boundaries.
        w = w * (0.58 + 0.42 * same)
        w = w / max(float(w.sum()), 1e-12)
        neighbor_mass = 1.0 - min(0.92, self_keep)
        if regime == "heavy":
            neighbor_mass = 1.12
        elif regime == "medium":
            neighbor_mass = 0.84
        elif regime == "light":
            neighbor_mass = 0.30
        elif regime == "clear":
            neighbor_mass = 0.0
        for node, weight in zip(nb, w):
            value = float(neighbor_mass * weight)
            if value <= 0:
                continue
            rows.append(i)
            cols.append(int(node))
            vals.append(value)
            total += value
        neighbor_agreement[i] = np.float32(float(np.sum(raw_w[i] * same)))
        effective_neighbor_count[i] = np.float32(1.0 / max(float(np.sum(w * w)), 1e-12))
        # Normalize each row exactly.
        for j in range(row_start, len(vals)):
            vals[j] = vals[j] / total

    matrix = sparse.csr_matrix((vals, (rows, cols)), shape=(len(pos), len(pos)), dtype=np.float32)
    return matrix, {
        "neighbor_idx": idx.astype(np.int32),
        "neighbor_weights": raw_w.astype(np.float32),
        "neighbor_agreement": neighbor_agreement,
        "effective_neighbor_count": effective_neighbor_count,
    }


def build_target(full_support: np.ndarray, regimes: np.ndarray, matrix: sparse.csr_matrix) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if TARGET_OUT.is_file():
        target = np.load(TARGET_OUT, mmap_mode="r")
        actions = np.load(ACTION_OUT) if ACTION_OUT.is_file() else None
        if actions is not None and "alpha" in actions.files:
            return target, {"alpha": actions["alpha"].astype(np.float32)}

    n = full_support.shape[0]
    flat = np.asarray(full_support).reshape(n, -1)
    consensus = matrix @ flat
    consensus = np.maximum(np.asarray(consensus, dtype=np.float32), FLOOR)
    consensus /= np.maximum(consensus.sum(axis=1, keepdims=True), TINY)

    alpha = np.zeros(n, dtype=np.float32)
    v187_actions = np.load(V187_ACTIONS)
    prior_l1 = v187_actions["prior_support_l1"].astype(np.float32)
    support_l1 = v187_actions["support_l1_actual"].astype(np.float32)
    for i, regime in enumerate(regimes):
        regime_s = str(regime)
        scale = ALPHA_SCALE[regime_s]
        if scale <= 0:
            alpha[i] = 0.0
            continue
        novelty = float(np.clip((prior_l1[i] - 0.20) / 0.42, 0.0, 1.0))
        not_overdone = float(np.clip((0.100 - support_l1[i]) / 0.075, 0.55, 1.0))
        alpha[i] = np.float32(scale * (0.62 + 0.38 * novelty) * not_overdone)

    target_flat = ((1.0 - alpha[:, None]) * flat + alpha[:, None] * consensus).astype(np.float32)
    target_flat /= np.maximum(target_flat.sum(axis=1, keepdims=True), TINY)
    target = np.lib.format.open_memmap(TARGET_OUT, mode="w+", dtype=np.float32, shape=full_support.shape)
    target[:] = target_flat.reshape(full_support.shape)
    target.flush()

    diag = {
        "alpha": alpha,
        "target_full_l1": np.sum(np.abs(target_flat - flat), axis=1).astype(np.float32),
        "target_full_cos": row_cos(target_flat, flat).astype(np.float32),
        "v187_prior_l1": prior_l1,
        "v187_support_l1": support_l1,
    }
    return np.load(TARGET_OUT, mmap_mode="r"), diag


def apply_target(target: np.ndarray, regimes: np.ndarray, batch: int = BATCH) -> dict[str, np.ndarray]:
    source = np.load(BASE, mmap_mode="r")
    output = np.lib.format.open_memmap(OUTPUT, mode="w+", dtype=np.complex64, shape=source.shape)
    diag: dict[str, list[np.ndarray]] = {
        "actual_power_factor": [],
        "full_l1_actual": [],
        "collapsed_sh_l1_actual": [],
        "gain_min": [],
        "gain_p10": [],
        "gain_p90": [],
        "gain_max": [],
    }

    for start in range(0, len(source), batch):
        end = min(start + batch, len(source))
        h0 = np.asarray(source[start:end], dtype=np.complex64)
        spec0 = leak.leaked_forward(h0)
        power0 = leak.leaked_cell_power(spec0)
        total0 = np.maximum(power0.sum(axis=(1, 2, 3, 4, 5)), TINY).astype(np.float32)
        support0 = power0 / total0[:, None, None, None, None, None]
        h_power0 = np.maximum(np.sum(np.abs(h0) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY).astype(np.float32)

        target_power = target[start:end] * total0[:, None, None, None, None, None]
        gain = np.sqrt(target_power / np.maximum(power0, 1e-30)).astype(np.float32)
        gain = np.clip(gain, 0.18, 5.60).astype(np.float32)

        flat = support0.reshape(end - start, -1)
        strong512 = top_mask(flat, 512).reshape(support0.shape)
        strong4096 = top_mask(flat, 4096).reshape(support0.shape)
        gain[strong4096] = np.clip(gain[strong4096], 0.48, 2.35)
        gain[strong512] = np.clip(gain[strong512], 0.66, 1.68)

        for local, regime in enumerate(regimes[start:end]):
            regime_s = str(regime)
            if regime_s == "clear":
                gain[local] = 1.0
            elif regime_s == "light":
                gain[local] = 1.0 + 0.62 * (gain[local] - 1.0)

        spec1 = spec0 * gain
        h1 = leak.leaked_inverse(spec1)
        h_power1 = np.maximum(np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY).astype(np.float32)
        h1 *= np.sqrt(h_power0 / h_power1)[:, None, None, None].astype(np.float32)

        if not (np.isfinite(h1.real).all() and np.isfinite(h1.imag).all()):
            raise FloatingPointError(f"non-finite block {start}:{end}")
        output[start:end] = h1.astype(np.complex64)
        output.flush()

        spec_after = leak.leaked_forward(h1.astype(np.complex64))
        power1 = leak.leaked_cell_power(spec_after)
        support1 = power1 / np.maximum(power1.sum(axis=(1, 2, 3, 4, 5), keepdims=True), TINY)
        collapsed0 = support0.sum(axis=(1, 4, 5))
        collapsed1 = support1.sum(axis=(1, 4, 5))
        diag["actual_power_factor"].append(
            (np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64) / h_power0).astype(np.float32)
        )
        diag["full_l1_actual"].append(np.sum(np.abs(support1 - support0), axis=(1, 2, 3, 4, 5)).astype(np.float32))
        diag["collapsed_sh_l1_actual"].append(np.sum(np.abs(collapsed1 - collapsed0), axis=(1, 2)).astype(np.float32))
        gflat = gain.reshape(end - start, -1)
        diag["gain_min"].append(gflat.min(axis=1).astype(np.float32))
        diag["gain_p10"].append(np.quantile(gflat, 0.10, axis=1).astype(np.float32))
        diag["gain_p90"].append(np.quantile(gflat, 0.90, axis=1).astype(np.float32))
        diag["gain_max"].append(gflat.max(axis=1).astype(np.float32))
        print(f"v188 leaked slice projection {end}/{len(source)}", flush=True)

    return {key: np.concatenate(values).astype(np.float32) for key, values in diag.items()}


def compare_output_to(output_path, base_path, regimes: np.ndarray) -> dict[str, object]:
    return track.compare_output_to(output_path, base_path, regimes)


def generate() -> dict[str, object]:
    pos_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    blocked = np.load(PATHS.shared_features / "height_geometry_cache.npz")["test_blocked"].astype(np.float32)
    regimes = regime_of(blocked)
    full_support = build_full_leaked_support()
    matrix, graph_diag = build_weight_matrix(pos_test, blocked, regimes)
    target, target_diag = build_target(full_support, regimes, matrix)
    apply_diag = apply_target(target, regimes)

    np.savez_compressed(
        ACTION_OUT,
        regimes=regimes.astype("U8"),
        **graph_diag,
        **target_diag,
        **apply_diag,
    )

    comparison_to_v187 = compare_output_to(OUTPUT, BASE, regimes)
    comparison_to_v186 = compare_output_to(OUTPUT, V186_BASE, regimes)
    comparison_to_v185 = compare_output_to(OUTPUT, V185_BASE, regimes)
    if comparison_to_v187["shape"] != [500, 256, 4, 192] or comparison_to_v187["dtype"] != "complex64":
        raise ValueError(f"invalid output layout: {comparison_to_v187['shape']} {comparison_to_v187['dtype']}")
    if not comparison_to_v187["finite"]:
        raise ValueError("v188 output contains non-finite values")

    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "v187 proved the leaked PAS SxH scoring axis is useful, but it collapsed "
            "pol/UE/V.  v188 aggressively projects the full leaked tensor slices so "
            "the correction can affect any official PAS averaging/slicing after the "
            "traceback FFT."
        ),
        "new_information_used": {
            "current_best": "v187 online 0.504752",
            "official_traceback": "B,2,192,16,4,8 with fft2 axes=(2,3)",
            "full_slice_target": "test-test sparse consensus over B,2,S,H,UE,V normalized power",
            "difference_from_v187": "keeps pol/UE/V instead of collapsed (S,H) only",
        },
        "parameters": {
            "base": PATHS.relative(BASE),
            "direct_k": DIRECT_K,
            "self_keep": SELF_KEEP,
            "alpha_scale": ALPHA_SCALE,
            "gain_clip": {
                "all": [0.18, 5.60],
                "top4096_full_slices": [0.48, 2.35],
                "top512_full_slices": [0.66, 1.68],
                "light_blend": "62% of raw gain",
                "clear": "identity",
            },
            "power_policy": "preserve v187 per-point total power",
            "phase_policy": "preserve leaked-domain phase; no absolute phase transport",
        },
        "diagnostics": {
            "overall": {
                "weight_nnz": int(matrix.nnz),
                "neighbor_agreement": summarize(graph_diag["neighbor_agreement"]),
                "effective_neighbor_count": summarize(graph_diag["effective_neighbor_count"]),
                **{key: summarize(value) for key, value in target_diag.items()},
                **{key: summarize(value) for key, value in apply_diag.items()},
            },
            "by_regime": by_regime_summary(
                regimes,
                {
                    "alpha": target_diag["alpha"],
                    "target_full_l1": target_diag["target_full_l1"],
                    "target_full_cos": target_diag["target_full_cos"],
                    "full_l1_actual": apply_diag["full_l1_actual"],
                    "collapsed_sh_l1_actual": apply_diag["collapsed_sh_l1_actual"],
                    "actual_power_factor": apply_diag["actual_power_factor"],
                    "gain_max": apply_diag["gain_max"],
                },
            ),
        },
        "comparison_to_v187": comparison_to_v187,
        "comparison_to_v186": comparison_to_v186,
        "comparison_to_v185": comparison_to_v185,
        "preserved": {
            "base_complex": "v187",
            "point_total_power": "v187",
            "absolute_phase": "not transported",
            "clear": "unchanged",
            "train_truth_metric_slice": "not used as target",
        },
        "changed": {
            "full_leaked_slices": "medium/heavy full B,2,S,H,UE,V normalized power moves toward sparse test-test consensus",
            "pol_ue_v_allocation": "can change inside leaked PAS cells, unlike v187",
        },
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
            "full_support": PATHS.relative(FULL_SUPPORT_OUT),
            "target": PATHS.relative(TARGET_OUT),
            "actions": PATHS.relative(ACTION_OUT),
            "report": PATHS.relative(REPORT),
            "zip": PATHS.relative(ZIP_OUT),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def write_zip(report: dict[str, object]) -> dict[str, object]:
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        for src in [
            PATHS.code / "generate_v188_leaked_slice_overdrive.py",
            PATHS.code / "generate_v187_leaked_pas_lifecycle_projection.py",
            PATHS.code / "generate_v178_test_track_complex_consensus.py",
            PATHS.code / "generate_v87_easy_field_projection.py",
            PATHS.code / "generate_v171_full_complex_ad_tail_denoising.py",
            PATHS.code / "generate_v169_power_scaled_tail_denoising.py",
            PATHS.code / "generate_v159_frontier_consensus_support_soup.py",
            PATHS.code / "powerfield_common.py",
            PATHS.code_root / "project_paths.py",
        ]:
            archive.write(src, src.name)
    with zipfile.ZipFile(ZIP_OUT) as archive:
        bad = archive.testzip()
        members = archive.namelist()
    if bad is not None:
        raise ValueError(f"ZIP CRC failure: {bad}")

    output = np.load(OUTPUT, mmap_mode="r")
    finite = True
    for start in range(0, len(output), 20):
        block = np.asarray(output[start : start + 20], dtype=np.complex64)
        finite = finite and bool(np.isfinite(block.real).all() and np.isfinite(block.imag).all())
    report["integrity"] = {
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": finite,
        "zip_crc": True,
        "zip_members": members,
        "sha256": digest(ZIP_OUT, "sha256"),
        "md5": digest(ZIP_OUT, "md5"),
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    report = generate()
    report = write_zip(report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
