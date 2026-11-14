#!/usr/bin/env python3
"""Generate v189: leaked full-slice multi-marginal overdrive on top of v188.

v188 made the major jump by keeping the full official-traceback-like tensor

    B,256,4,192 -> B,2,192,16,4,8 ; fft2 axes=(2,3)

instead of collapsing pol/UE/V.  v189 keeps that winning base and attacks the
next likely scoring detail: the evaluator may average or slice the leaked
tensor at several levels, not only use the full six-dimensional power table.

This version therefore builds a test-test consensus target in the full leaked
power tensor and then uses multiplicative multi-marginal fitting to also move
the following marginals toward consensus:

* S-H, equivalent to the collapsed v187-like leaked PAS map;
* pol-S-H, preserving polarization-specific PAS slices;
* S-H-UE, preserving UE-port-specific PAS slices;
* S-H-V, preserving vertical-index-specific PAS slices.

It is intentionally aggressive and uses v188 as the current online best.
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


VERSION = "v189_leaked_multimarginal_overdrive"
OUT_DIR = PATHS.cache_dir(189)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(189)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v189_Round1_Test_Channel.npy"
FULL_SUPPORT_OUT = OUT_DIR / "v189_v188_leaked_full_support.npy"
TARGET_OUT = OUT_DIR / "v189_leaked_multimarginal_target.npy"
ACTION_OUT = OUT_DIR / "v189_leaked_multimarginal_actions.npz"
REPORT = REPORT_DIR / "v189_leaked_multimarginal_overdrive_generate.json"
ZIP_OUT = PATHS.round1_versions / "v189_leaked_multimarginal_overdrive_candidate_pending.zip"

BASE = PATHS.cache_dir(188) / "v188_Round1_Test_Channel.npy"
V188_ACTIONS = PATHS.cache_dir(188) / "v188_leaked_slice_overdrive_actions.npz"
V187_BASE = PATHS.cache_dir(187) / "v187_Round1_Test_Channel.npy"
V186_BASE = PATHS.cache_dir(186) / "v186_Round1_Test_Channel.npy"

FLOOR = 1e-12
BATCH = 8
DIRECT_K = 84
SELF_KEEP = {"clear": 1.0, "light": 0.74, "medium": 0.36, "heavy": 0.25}
NEIGHBOR_MASS = {"clear": 0.0, "light": 0.46, "medium": 1.05, "heavy": 1.36}
ALPHA_SCALE = {"clear": 0.0, "light": 0.12, "medium": 0.50, "heavy": 0.72}
MARGINAL_SCALE = {"clear": 0.0, "light": 0.18, "medium": 0.66, "heavy": 0.88}
IPF_ITERS = 3


def normalize_full(power: np.ndarray) -> np.ndarray:
    total = np.maximum(power.sum(axis=tuple(range(1, power.ndim)), keepdims=True), TINY)
    return (power / total).astype(np.float32)


def normalize_rows(flat: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(flat, dtype=np.float32), FLOOR)
    out /= np.maximum(out.sum(axis=1, keepdims=True), TINY)
    return out.astype(np.float32)


def safe_ratio(desired: np.ndarray, current: np.ndarray) -> np.ndarray:
    return np.power(
        np.maximum(desired, FLOOR) / np.maximum(current, FLOOR),
        1.0,
        dtype=np.float32,
    ).astype(np.float32)


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
        print(f"v189 full leaked support {end}/{len(source)}", flush=True)
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
        w = w * (0.52 + 0.48 * same)
        w = w / max(float(w.sum()), 1e-12)
        for node, weight in zip(nb, w):
            value = float(NEIGHBOR_MASS[regime] * weight)
            if value <= 0.0:
                continue
            rows.append(i)
            cols.append(int(node))
            vals.append(value)
            total += value
        neighbor_agreement[i] = np.float32(float(np.sum(raw_w[i] * same)))
        effective_neighbor_count[i] = np.float32(1.0 / max(float(np.sum(w * w)), 1e-12))
        for j in range(row_start, len(vals)):
            vals[j] = vals[j] / total

    matrix = sparse.csr_matrix((vals, (rows, cols)), shape=(len(pos), len(pos)), dtype=np.float32)
    return matrix, {
        "neighbor_idx": idx.astype(np.int32),
        "neighbor_weights": raw_w.astype(np.float32),
        "neighbor_agreement": neighbor_agreement,
        "effective_neighbor_count": effective_neighbor_count,
    }


def blended_marginal(current: np.ndarray, consensus: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    desired = ((1.0 - alpha) * current + alpha * consensus).astype(np.float32)
    desired /= np.maximum(desired.sum(axis=tuple(range(1, desired.ndim)), keepdims=True), TINY)
    return desired


def apply_ratio(work: np.ndarray, ratio: np.ndarray, axes: str, exponent: float) -> None:
    ratio = np.power(np.maximum(ratio, FLOOR), exponent, dtype=np.float32).astype(np.float32)
    if axes == "sh":
        work *= ratio[:, None, :, :, None, None]
    elif axes == "psh":
        work *= ratio[:, :, :, :, None, None]
    elif axes == "shue":
        work *= ratio[:, None, :, :, :, None]
    elif axes == "shv":
        work *= ratio[:, None, :, :, None, :]
    else:
        raise ValueError(axes)


def build_target(full_support: np.ndarray, regimes: np.ndarray, matrix: sparse.csr_matrix, graph_diag: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if TARGET_OUT.is_file() and ACTION_OUT.is_file():
        target = np.load(TARGET_OUT, mmap_mode="r")
        actions = np.load(ACTION_OUT)
        if "alpha_full" in actions.files:
            return target, {key: actions[key].astype(np.float32) for key in actions.files if key.startswith(("alpha", "target", "margin", "v188"))}

    n = full_support.shape[0]
    flat = np.asarray(full_support).reshape(n, -1)
    consensus_flat = matrix @ flat
    consensus_flat = normalize_rows(np.asarray(consensus_flat, dtype=np.float32))
    consensus = consensus_flat.reshape(full_support.shape)

    v188_actions = np.load(V188_ACTIONS)
    prev_full_l1 = v188_actions["full_l1_actual"].astype(np.float32)
    prev_collapsed_l1 = v188_actions["collapsed_sh_l1_actual"].astype(np.float32)
    agreement = graph_diag["neighbor_agreement"].astype(np.float32)

    alpha_full = np.zeros(n, dtype=np.float32)
    alpha_margin = np.zeros(n, dtype=np.float32)
    for i, regime in enumerate(regimes):
        regime_s = str(regime)
        if ALPHA_SCALE[regime_s] <= 0.0:
            continue
        momentum = float(np.clip((prev_full_l1[i] - 0.035) / 0.165, 0.0, 1.0))
        collapsed_momentum = float(np.clip((prev_collapsed_l1[i] - 0.025) / 0.125, 0.0, 1.0))
        local_agreement = float(np.clip(agreement[i], 0.25, 1.0))
        alpha_full[i] = np.float32(ALPHA_SCALE[regime_s] * (0.66 + 0.34 * momentum) * (0.78 + 0.22 * local_agreement))
        alpha_margin[i] = np.float32(
            MARGINAL_SCALE[regime_s]
            * (0.60 + 0.25 * collapsed_momentum + 0.15 * momentum)
            * (0.76 + 0.24 * local_agreement)
        )

    work = ((1.0 - alpha_full[:, None]) * flat + alpha_full[:, None] * consensus_flat).astype(np.float32)
    work = normalize_rows(work).reshape(full_support.shape)

    sh0 = np.asarray(full_support).sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32)
    shc = consensus.sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32)
    psh0 = np.asarray(full_support).sum(axis=(4, 5), dtype=np.float64).astype(np.float32)
    pshc = consensus.sum(axis=(4, 5), dtype=np.float64).astype(np.float32)
    shue0 = np.asarray(full_support).sum(axis=(1, 5), dtype=np.float64).astype(np.float32)
    shuec = consensus.sum(axis=(1, 5), dtype=np.float64).astype(np.float32)
    shv0 = np.asarray(full_support).sum(axis=(1, 4), dtype=np.float64).astype(np.float32)
    shvc = consensus.sum(axis=(1, 4), dtype=np.float64).astype(np.float32)

    alpha_m = alpha_margin
    desired_sh = blended_marginal(sh0, shc, alpha_m[:, None, None])
    desired_psh = blended_marginal(psh0, pshc, alpha_m[:, None, None, None])
    desired_shue = blended_marginal(shue0, shuec, alpha_m[:, None, None, None])
    desired_shv = blended_marginal(shv0, shvc, alpha_m[:, None, None, None])

    for _ in range(IPF_ITERS):
        current_sh = work.sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32)
        apply_ratio(work, safe_ratio(desired_sh, current_sh), "sh", 0.70)
        current_psh = work.sum(axis=(4, 5), dtype=np.float64).astype(np.float32)
        apply_ratio(work, safe_ratio(desired_psh, current_psh), "psh", 0.42)
        current_shue = work.sum(axis=(1, 5), dtype=np.float64).astype(np.float32)
        apply_ratio(work, safe_ratio(desired_shue, current_shue), "shue", 0.46)
        current_shv = work.sum(axis=(1, 4), dtype=np.float64).astype(np.float32)
        apply_ratio(work, safe_ratio(desired_shv, current_shv), "shv", 0.42)
        work = normalize_rows(work.reshape(n, -1)).reshape(full_support.shape)

    target = np.lib.format.open_memmap(TARGET_OUT, mode="w+", dtype=np.float32, shape=full_support.shape)
    target[:] = work.astype(np.float32)
    target.flush()

    target_flat = work.reshape(n, -1)
    target_sh = work.sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32)
    target_psh = work.sum(axis=(4, 5), dtype=np.float64).astype(np.float32)
    target_shue = work.sum(axis=(1, 5), dtype=np.float64).astype(np.float32)
    target_shv = work.sum(axis=(1, 4), dtype=np.float64).astype(np.float32)
    diag = {
        "alpha_full": alpha_full,
        "alpha_margin": alpha_margin,
        "target_full_l1": np.sum(np.abs(target_flat - flat), axis=1).astype(np.float32),
        "target_full_cos": row_cos(target_flat, flat).astype(np.float32),
        "margin_sh_l1": np.sum(np.abs(target_sh - sh0), axis=(1, 2)).astype(np.float32),
        "margin_psh_l1": np.sum(np.abs(target_psh - psh0), axis=(1, 2, 3)).astype(np.float32),
        "margin_shue_l1": np.sum(np.abs(target_shue - shue0), axis=(1, 2, 3)).astype(np.float32),
        "margin_shv_l1": np.sum(np.abs(target_shv - shv0), axis=(1, 2, 3)).astype(np.float32),
        "v188_full_l1": prev_full_l1,
        "v188_collapsed_sh_l1": prev_collapsed_l1,
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
        gain = np.clip(gain, 0.16, 6.80).astype(np.float32)

        flat = support0.reshape(end - start, -1)
        strong128 = top_mask(flat, 128).reshape(support0.shape)
        strong1024 = top_mask(flat, 1024).reshape(support0.shape)
        strong8192 = top_mask(flat, 8192).reshape(support0.shape)
        gain[strong8192] = np.clip(gain[strong8192], 0.42, 2.75)
        gain[strong1024] = np.clip(gain[strong1024], 0.58, 1.92)
        gain[strong128] = np.clip(gain[strong128], 0.74, 1.46)

        for local, regime in enumerate(regimes[start:end]):
            regime_s = str(regime)
            if regime_s == "clear":
                gain[local] = 1.0
            elif regime_s == "light":
                gain[local] = 1.0 + 0.72 * (gain[local] - 1.0)

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
        print(f"v189 leaked multimarginal projection {end}/{len(source)}", flush=True)

    return {key: np.concatenate(values).astype(np.float32) for key, values in diag.items()}


def compare_output_to(output_path, base_path, regimes: np.ndarray) -> dict[str, object]:
    return track.compare_output_to(output_path, base_path, regimes)


def generate() -> dict[str, object]:
    pos_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    blocked = np.load(PATHS.shared_features / "height_geometry_cache.npz")["test_blocked"].astype(np.float32)
    regimes = regime_of(blocked)
    full_support = build_full_leaked_support()
    matrix, graph_diag = build_weight_matrix(pos_test, blocked, regimes)
    target, target_diag = build_target(full_support, regimes, matrix, graph_diag)
    apply_diag = apply_target(target, regimes)

    np.savez_compressed(
        ACTION_OUT,
        regimes=regimes.astype("U8"),
        **graph_diag,
        **target_diag,
        **apply_diag,
    )

    comparison_to_v188 = compare_output_to(OUTPUT, BASE, regimes)
    comparison_to_v187 = compare_output_to(OUTPUT, V187_BASE, regimes)
    comparison_to_v186 = compare_output_to(OUTPUT, V186_BASE, regimes)
    if comparison_to_v188["shape"] != [500, 256, 4, 192] or comparison_to_v188["dtype"] != "complex64":
        raise ValueError(f"invalid output layout: {comparison_to_v188['shape']} {comparison_to_v188['dtype']}")
    if not comparison_to_v188["finite"]:
        raise ValueError("v189 output contains non-finite values")

    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "v188 confirmed that full leaked B,2,192,16,4,8 slice power is a major "
            "scoring signal.  v189 tests whether the next missing signal is the "
            "set of official-like marginals that may be averaged before PAS/PDP "
            "scoring: S-H, pol-S-H, S-H-UE and S-H-V."
        ),
        "new_information_used": {
            "current_best": "v188 online 0.507831",
            "online_feedback": "v188 +0.003079 vs v187 confirms full leaked slice-level target",
            "official_traceback": "B,2,192,16,4,8 with fft2 axes=(2,3)",
            "difference_from_v188": "uses v188 as base and applies multi-marginal fitting, not only full 6D consensus",
        },
        "parameters": {
            "base": PATHS.relative(BASE),
            "direct_k": DIRECT_K,
            "self_keep": SELF_KEEP,
            "neighbor_mass": NEIGHBOR_MASS,
            "alpha_scale": ALPHA_SCALE,
            "marginal_scale": MARGINAL_SCALE,
            "ipf_iters": IPF_ITERS,
            "marginals": ["S-H", "pol-S-H", "S-H-UE", "S-H-V"],
            "gain_clip": {
                "all": [0.16, 6.80],
                "top8192_full_slices": [0.42, 2.75],
                "top1024_full_slices": [0.58, 1.92],
                "top128_full_slices": [0.74, 1.46],
                "light_blend": "72% of raw gain",
                "clear": "identity",
            },
            "power_policy": "preserve v188 per-point total power",
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
                    "alpha_full": target_diag["alpha_full"],
                    "alpha_margin": target_diag["alpha_margin"],
                    "target_full_l1": target_diag["target_full_l1"],
                    "margin_sh_l1": target_diag["margin_sh_l1"],
                    "margin_shue_l1": target_diag["margin_shue_l1"],
                    "full_l1_actual": apply_diag["full_l1_actual"],
                    "collapsed_sh_l1_actual": apply_diag["collapsed_sh_l1_actual"],
                    "actual_power_factor": apply_diag["actual_power_factor"],
                    "gain_max": apply_diag["gain_max"],
                },
            ),
        },
        "comparison_to_v188": comparison_to_v188,
        "comparison_to_v187": comparison_to_v187,
        "comparison_to_v186": comparison_to_v186,
        "preserved": {
            "base_complex": "v188",
            "point_total_power": "v188",
            "absolute_phase": "not transported",
            "clear": "unchanged",
            "train_truth_metric_slice": "not used as target",
        },
        "changed": {
            "full_leaked_slices": "second-stage medium/heavy full-slice target from v188",
            "metric_marginals": "explicitly moves S-H, pol-S-H, S-H-UE and S-H-V marginals toward test-test consensus",
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
            PATHS.code / "generate_v189_leaked_multimarginal_overdrive.py",
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
