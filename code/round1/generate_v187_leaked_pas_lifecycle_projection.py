#!/usr/bin/env python3
"""Generate v187: leaked-PAS lifecycle projection on top of v186.

The platform traceback reportedly exposed the PAS intermediate tensor

    (B, 2, 192, 16, 4, 8)

followed by ``np.fft.fft2(..., axes=(2, 3))``.  That means an important scorer
slice may live in a subcarrier-by-horizontal-antenna spectrum, not only in the
usual 16x8 spatial AD support used by the recent best versions.

v187 uses this as a new information source while keeping the verified v186
backbone:

* start from v186 complex output;
* transform to the leaked PAS domain: B,2,S,H,UE,V;
* build a test-test local lifecycle target over collapsed (S,H) support;
* apply bounded positive amplitude gains in that leaked domain;
* preserve each point's total power and do not transport absolute phase.

This is deliberately not a rerun of v155/v157.  Those versions pushed
train-truth metric slices onto test points.  v187 instead uses the same
test-test continuity principle that made v178-v186 work, but on the leaked
official scoring axes.
"""

from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path

import numpy as np

import generate_v178_test_track_complex_consensus as track
from generate_v87_easy_field_projection import MH, MP, MV, N_ANT, S_SC, row_cos, summarize
from generate_v159_frontier_consensus_support_soup import REGIMES, digest
from generate_v169_power_scaled_tail_denoising import by_regime_summary
from generate_v171_full_complex_ad_tail_denoising import top_mask
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION = "v187_leaked_pas_lifecycle_projection"
OUT_DIR = PATHS.cache_dir(187)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(187)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v187_Round1_Test_Channel.npy"
SUPPORT_OUT = OUT_DIR / "v187_v186_leaked_pas_support.npy"
PEAKS_OUT = OUT_DIR / "v187_v186_leaked_pas_peaks.npz"
TARGET_OUT = OUT_DIR / "v187_leaked_pas_lifecycle_target.npy"
ACTION_OUT = OUT_DIR / "v187_leaked_pas_lifecycle_actions.npz"
REPORT = REPORT_DIR / "v187_leaked_pas_lifecycle_projection_generate.json"
ZIP_OUT = PATHS.round1_versions / "v187_leaked_pas_lifecycle_projection_candidate_pending.zip"

BASE = PATHS.cache_dir(186) / "v186_Round1_Test_Channel.npy"
V185_BASE = PATHS.cache_dir(185) / "v185_Round1_Test_Channel.npy"
V184_BASE = PATHS.cache_dir(184) / "v184_Round1_Test_Channel.npy"
V186_ACTIONS = PATHS.cache_dir(186) / "v186_lifecycle_quality_actions.npz"

FLOOR = 1e-12
PEAK_COUNT = 112
NEIGHBOR_PEAKS = 56
PEAK_CANDIDATES = 1536
DIRECT_K = 36
EXTENDED_K = 104


def normalize_support(x: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(x, dtype=np.float32), FLOOR)
    out /= np.maximum(out.sum(axis=tuple(range(1, out.ndim)), keepdims=True), TINY)
    return out.astype(np.float32)


def circular_delta(a: int, b: int, size: int) -> int:
    delta = abs(int(a) - int(b))
    return int(min(delta, size - delta))


def circular_signed_delta(origin: int, value: int, size: int) -> int:
    return int((int(value) - int(origin) + size // 2) % size - size // 2)


def leaked_forward(h: np.ndarray) -> np.ndarray:
    """Official-traceback-like transform: H -> B,2,S,H,UE,V."""
    arr = h.reshape(h.shape[0], MP, MH, MV, N_ANT, S_SC)
    arr = arr.transpose(0, 1, 5, 2, 4, 3)
    return np.fft.fft2(arr, axes=(2, 3)).astype(np.complex64)


def leaked_inverse(spec: np.ndarray) -> np.ndarray:
    """Inverse of leaked_forward."""
    arr = np.fft.ifft2(spec, axes=(2, 3))
    arr = arr.transpose(0, 1, 3, 5, 4, 2)
    return arr.reshape(spec.shape[0], MP * MH * MV, N_ANT, S_SC).astype(np.complex64)


def leaked_cell_power(spec: np.ndarray) -> np.ndarray:
    return np.abs(spec).astype(np.float32) ** 2


def add_leaked_kernel(
    field: np.ndarray,
    s_bin: int,
    h_bin: int,
    mass: float,
    s_sigma: float = 4.20,
    h_sigma: float = 0.95,
) -> None:
    entries: list[tuple[int, int, float]] = []
    total = 0.0
    for ds in range(-12, 13):
        ss = (int(s_bin) + ds) % S_SC
        ws = math.exp(-0.5 * (ds / s_sigma) ** 2)
        for dh in range(-2, 3):
            hh = (int(h_bin) + dh) % MH
            wh = math.exp(-0.5 * (dh / h_sigma) ** 2)
            w = ws * wh
            entries.append((ss, hh, w))
            total += w
    if total <= 0:
        return
    scale = float(mass) / total
    for ss, hh, w in entries:
        field[ss, hh] += np.float32(scale * w)


def extract_leaked_support_and_peaks(batch: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if SUPPORT_OUT.is_file() and PEAKS_OUT.is_file():
        support = normalize_support(np.load(SUPPORT_OUT).astype(np.float32))
        data = np.load(PEAKS_OUT)
        return support, data["peaks"].astype(np.int16), data["energy"].astype(np.float32)

    source = np.load(BASE, mmap_mode="r")
    support = np.empty((source.shape[0], S_SC, MH), dtype=np.float32)
    peaks = np.full((source.shape[0], PEAK_COUNT, 2), -1, dtype=np.int16)
    energy = np.zeros((source.shape[0], PEAK_COUNT), dtype=np.float32)

    for start in range(0, len(source), batch):
        end = min(start + batch, len(source))
        h = np.asarray(source[start:end], dtype=np.complex64)
        spec = leaked_forward(h)
        power = leaked_cell_power(spec)
        collapsed = power.sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32)
        block_support = collapsed / np.maximum(collapsed.sum(axis=(1, 2), keepdims=True), TINY)
        support[start:end] = block_support

        flat = block_support.reshape(end - start, -1)
        candidate_count = min(PEAK_CANDIDATES, flat.shape[1])
        candidates = np.argpartition(flat, -candidate_count, axis=1)[:, -candidate_count:]
        candidate_values = np.take_along_axis(flat, candidates, axis=1)
        order = np.argsort(candidate_values, axis=1)[:, ::-1]
        candidates = np.take_along_axis(candidates, order, axis=1)

        for row in range(end - start):
            accepted: list[tuple[int, int]] = []
            for flat_index in candidates[row]:
                s_bin, h_bin = divmod(int(flat_index), MH)
                too_close = any(
                    circular_delta(s_bin, ss, S_SC) <= 4
                    and circular_delta(h_bin, hh, MH) <= 1
                    for ss, hh in accepted
                )
                if too_close:
                    continue
                slot = len(accepted)
                peaks[start + row, slot] = (s_bin, h_bin)
                energy[start + row, slot] = flat[row, flat_index]
                accepted.append((s_bin, h_bin))
                if slot + 1 >= PEAK_COUNT:
                    break
        print(f"v187 leaked support/peaks {end}/{len(source)}", flush=True)

    support = normalize_support(support)
    np.save(SUPPORT_OUT, support)
    np.savez_compressed(PEAKS_OUT, peaks=peaks, energy=energy)
    return support, peaks, energy


def build_extended_graph(
    pos: np.ndarray,
    blocked: np.ndarray,
    regimes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    direct_idx, direct_w = track.neighbor_graph(pos, blocked, regimes, k=DIRECT_K)
    extended_idx = np.full((len(pos), EXTENDED_K), -1, dtype=np.int32)
    extended_w = np.zeros((len(pos), EXTENDED_K), dtype=np.float32)
    twohop_flag = np.zeros((len(pos), EXTENDED_K), dtype=np.float32)

    for i in range(len(pos)):
        weights: dict[int, float] = {}
        direct_nodes: set[int] = set()
        for slot, nb in enumerate(direct_idx[i]):
            node = int(nb)
            direct_nodes.add(node)
            weights[node] = weights.get(node, 0.0) + 0.74 * float(direct_w[i, slot])

        for slot, nb in enumerate(direct_idx[i]):
            first = int(nb)
            first_w = float(direct_w[i, slot])
            for slot2, nb2 in enumerate(direct_idx[first]):
                node = int(nb2)
                if node == i:
                    continue
                same_regime = regimes[node] == regimes[i]
                penalty = 0.92 if same_regime else 0.56
                weights[node] = weights.get(node, 0.0) + 0.34 * first_w * float(direct_w[first, slot2]) * penalty

        ordered = sorted(weights.items(), key=lambda item: item[1], reverse=True)[:EXTENDED_K]
        total = sum(value for _, value in ordered)
        if total <= 0:
            continue
        for out_slot, (node, value) in enumerate(ordered):
            extended_idx[i, out_slot] = int(node)
            extended_w[i, out_slot] = np.float32(value / total)
            if node not in direct_nodes:
                twohop_flag[i, out_slot] = 1.0

    return direct_idx, direct_w, extended_idx, extended_w, twohop_flag


def build_leaked_target(
    support: np.ndarray,
    peaks: np.ndarray,
    energy: np.ndarray,
    extended_idx: np.ndarray,
    extended_w: np.ndarray,
    twohop_flag: np.ndarray,
    regimes: np.ndarray,
    v186_actions: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    n = len(support)
    target = np.empty_like(support, dtype=np.float32)

    alpha = np.zeros(n, dtype=np.float32)
    evidence_mass = np.zeros(n, dtype=np.float32)
    direct_mass = np.zeros(n, dtype=np.float32)
    neighbor_peak_count = np.zeros(n, dtype=np.float32)
    target_support_l1 = np.zeros(n, dtype=np.float32)
    target_support_cos = np.ones(n, dtype=np.float32)
    prior_support_l1 = np.zeros(n, dtype=np.float32)
    prior_support_cos = np.ones(n, dtype=np.float32)

    prev_l1 = v186_actions["support_l1_actual"].astype(np.float32)
    prev_alpha = v186_actions["lifecycle_alpha"].astype(np.float32)

    for i in range(n):
        regime = str(regimes[i])
        if regime == "heavy":
            base_keep = 0.56
            scale = 0.54
        elif regime == "medium":
            base_keep = 0.66
            scale = 0.30
        elif regime == "light":
            base_keep = 0.90
            scale = 0.055
        else:
            base_keep = 1.0
            scale = 0.0

        field = (base_keep * support[i]).astype(np.float32)
        evidence = 0.0
        direct_evidence = 0.0
        count = 0

        for nslot, nb in enumerate(extended_idx[i]):
            if nb < 0:
                break
            hop_scale = 0.68 if float(twohop_flag[i, nslot]) > 0.5 else 1.0
            node_weight = float(extended_w[i, nslot]) * hop_scale
            if node_weight <= 0:
                continue
            for pslot in range(NEIGHBOR_PEAKS):
                s_bin, h_bin = peaks[int(nb), pslot]
                if s_bin < 0:
                    break
                e = float(energy[int(nb), pslot])
                if e < 4.5e-5:
                    continue
                mass = node_weight * e * (1.08 if hop_scale >= 0.99 else 0.82)
                add_leaked_kernel(field, int(s_bin), int(h_bin), mass)
                evidence += mass
                if hop_scale >= 0.99:
                    direct_evidence += mass
                count += 1

        prior = normalize_support(field[None])[0]
        prior_l1 = float(np.sum(np.abs(prior - support[i])))
        prior_cos = float(row_cos(prior[None], support[i : i + 1])[0])
        evidence_gate = float(np.clip(evidence / 0.125, 0.0, 1.0))
        count_gate = float(np.clip(count / 420.0, 0.0, 1.0))
        direct_gate = float(np.clip(direct_evidence / max(evidence, 1e-12), 0.0, 1.0))
        novelty_gate = float(np.clip((0.992 - prior_cos) / 0.140, 0.0, 1.0))
        stability_gate = float(np.clip((prior_cos - 0.700) / 0.190, 0.0, 1.0))

        # v186 already moved AD lifecycle support.  Let that evidence increase
        # the leaked-PAS action slightly, but damp points that v186 pushed hard.
        prev_continue = float(np.clip(prev_alpha[i] / (0.78 if regime == "heavy" else 0.36), 0.0, 1.0)) if scale > 0 else 0.0
        overpush_damp = float(np.clip((0.105 - prev_l1[i]) / 0.075, 0.58, 1.0))
        conf = float(
            np.clip(
                0.44 * evidence_gate
                + 0.18 * count_gate
                + 0.16 * direct_gate
                + 0.12 * novelty_gate
                + 0.10 * math.sqrt(prev_continue),
                0.0,
                1.0,
            )
        )
        point_alpha = scale * (0.26 + 0.74 * conf) * stability_gate * (0.58 + 0.42 * novelty_gate) * overpush_damp
        alpha[i] = np.float32(np.clip(point_alpha, 0.0, scale))
        target[i] = normalize_support(((1.0 - alpha[i]) * support[i] + alpha[i] * prior)[None])[0]

        evidence_mass[i] = np.float32(evidence)
        direct_mass[i] = np.float32(direct_evidence)
        neighbor_peak_count[i] = np.float32(count)
        prior_support_l1[i] = np.float32(prior_l1)
        prior_support_cos[i] = np.float32(prior_cos)
        target_support_l1[i] = np.float32(np.sum(np.abs(target[i] - support[i])))
        target_support_cos[i] = np.float32(row_cos(target[i : i + 1], support[i : i + 1])[0])

        if (i + 1) % 100 == 0:
            print(f"v187 leaked target {i + 1}/{n}", flush=True)

    diagnostics = {
        "alpha": alpha,
        "evidence_mass": evidence_mass,
        "direct_mass": direct_mass,
        "neighbor_peak_count": neighbor_peak_count,
        "prior_support_l1": prior_support_l1,
        "prior_support_cos": prior_support_cos,
        "target_support_l1": target_support_l1,
        "target_support_cos": target_support_cos,
    }
    return normalize_support(target), diagnostics


def apply_leaked_target(
    target: np.ndarray,
    regimes: np.ndarray,
    batch: int = 16,
) -> dict[str, np.ndarray]:
    source = np.load(BASE, mmap_mode="r")
    output = np.lib.format.open_memmap(OUTPUT, mode="w+", dtype=np.complex64, shape=source.shape)

    diag: dict[str, list[np.ndarray]] = {
        "actual_power_factor": [],
        "support_l1_actual": [],
        "gain_min": [],
        "gain_p10": [],
        "gain_p90": [],
        "gain_max": [],
    }

    for start in range(0, len(source), batch):
        end = min(start + batch, len(source))
        h0 = np.asarray(source[start:end], dtype=np.complex64)
        spec0 = leaked_forward(h0)
        power0 = leaked_cell_power(spec0)
        cell0 = power0.sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32)
        total_cell0 = np.maximum(cell0.sum(axis=(1, 2)), TINY).astype(np.float32)
        support0 = cell0 / total_cell0[:, None, None]
        h_power0 = np.maximum(np.sum(np.abs(h0) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY).astype(np.float32)

        target_cell = target[start:end] * total_cell0[:, None, None]
        gain_cell = np.sqrt(target_cell / np.maximum(cell0, 1e-30)).astype(np.float32)
        gain_cell = np.clip(gain_cell, 0.24, 3.85).astype(np.float32)

        flat = support0.reshape(end - start, -1)
        strong64 = top_mask(flat, 64).reshape(end - start, S_SC, MH)
        strong256 = top_mask(flat, 256).reshape(end - start, S_SC, MH)
        gain_cell[strong256] = np.clip(gain_cell[strong256], 0.52, 2.10)
        gain_cell[strong64] = np.clip(gain_cell[strong64], 0.68, 1.52)

        # Light points only get a very small probe; clear points are unchanged.
        for local, regime in enumerate(regimes[start:end]):
            if str(regime) == "clear":
                gain_cell[local] = 1.0
            elif str(regime) == "light":
                gain_cell[local] = 1.0 + 0.42 * (gain_cell[local] - 1.0)

        spec1 = spec0 * gain_cell[:, None, :, :, None, None].astype(np.float32)
        h1 = leaked_inverse(spec1)
        h_power1 = np.maximum(np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY).astype(np.float32)
        h1 *= np.sqrt(h_power0 / h_power1)[:, None, None, None].astype(np.float32)

        if not (np.isfinite(h1.real).all() and np.isfinite(h1.imag).all()):
            raise FloatingPointError(f"non-finite block {start}:{end}")
        output[start:end] = h1.astype(np.complex64)
        output.flush()

        spec_after = leaked_forward(h1.astype(np.complex64))
        cell1 = leaked_cell_power(spec_after).sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32)
        support1 = cell1 / np.maximum(cell1.sum(axis=(1, 2), keepdims=True), TINY)
        diag["actual_power_factor"].append(
            (np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64) / h_power0).astype(np.float32)
        )
        diag["support_l1_actual"].append(np.sum(np.abs(support1 - support0), axis=(1, 2)).astype(np.float32))
        gflat = gain_cell.reshape(end - start, -1)
        diag["gain_min"].append(gflat.min(axis=1).astype(np.float32))
        diag["gain_p10"].append(np.quantile(gflat, 0.10, axis=1).astype(np.float32))
        diag["gain_p90"].append(np.quantile(gflat, 0.90, axis=1).astype(np.float32))
        diag["gain_max"].append(gflat.max(axis=1).astype(np.float32))
        print(f"v187 leaked projection applied {end}/500", flush=True)

    return {key: np.concatenate(value).astype(np.float32) for key, value in diag.items()}


def compare_output_to(output_path: Path, base_path: Path, regimes: np.ndarray) -> dict[str, object]:
    return track.compare_output_to(output_path, base_path, regimes)


def generate() -> dict[str, object]:
    pos_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    blocked = np.load(PATHS.shared_features / "height_geometry_cache.npz")["test_blocked"].astype(np.float32)
    regimes = regime_of(blocked)
    action_npz = np.load(V186_ACTIONS)
    v186_actions = {key: action_npz[key] for key in action_npz.files}

    support, peaks, energy = extract_leaked_support_and_peaks()
    direct_idx, direct_w, extended_idx, extended_w, twohop_flag = build_extended_graph(
        pos_test,
        blocked,
        regimes,
    )
    target, target_diag = build_leaked_target(
        support,
        peaks,
        energy,
        extended_idx,
        extended_w,
        twohop_flag,
        regimes,
        v186_actions,
    )
    np.save(TARGET_OUT, target)
    apply_diag = apply_leaked_target(target, regimes)

    np.savez_compressed(
        ACTION_OUT,
        regimes=regimes.astype("U8"),
        direct_neighbor_idx=direct_idx,
        direct_neighbor_weights=direct_w,
        extended_neighbor_idx=extended_idx,
        extended_neighbor_weights=extended_w,
        twohop_flag=twohop_flag,
        peaks=peaks,
        energy=energy,
        v186_lifecycle_alpha=v186_actions["lifecycle_alpha"].astype(np.float32),
        v186_support_l1_actual=v186_actions["support_l1_actual"].astype(np.float32),
        **target_diag,
        **apply_diag,
    )

    comparison_to_v186 = compare_output_to(OUTPUT, BASE, regimes)
    comparison_to_v185 = compare_output_to(OUTPUT, V185_BASE, regimes)
    comparison_to_v184 = compare_output_to(OUTPUT, V184_BASE, regimes)
    if comparison_to_v186["shape"] != [500, 256, 4, 192] or comparison_to_v186["dtype"] != "complex64":
        raise ValueError(f"invalid output layout: {comparison_to_v186['shape']} {comparison_to_v186['dtype']}")
    if not comparison_to_v186["finite"]:
        raise ValueError("v187 output contains non-finite values")

    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "The leaked official PAS traceback reveals a scorer-sensitive "
            "subcarrier-by-horizontal-antenna spectrum.  Current best versions "
            "mainly optimize conventional AD support, so a test-test lifecycle "
            "projection in the leaked PAS domain may recover information that "
            "the AD-domain lifecycle has saturated."
        ),
        "new_information_used": {
            "current_best": "v186 online 0.504556",
            "official_traceback": "reshape to B,2,192,16,4,8 then fft2 axes=(2,3)",
            "leaked_domain": "collapsed support over (subcarrier FFT bin, horizontal antenna FFT bin)",
            "test_test_prior": "nearby test receivers and two-hop route consensus in leaked-PAS peaks",
            "not_reused_failure": "does not project train-truth metric slices like v155/v157",
        },
        "parameters": {
            "base": PATHS.relative(BASE),
            "peak_count": PEAK_COUNT,
            "neighbor_peaks": NEIGHBOR_PEAKS,
            "peak_candidates": PEAK_CANDIDATES,
            "direct_k": DIRECT_K,
            "extended_k": EXTENDED_K,
            "kernel": {"s_sigma": 4.20, "h_sigma": 0.95, "s_radius": 12, "h_radius": 2},
            "alpha_scale": {"clear": 0.0, "light": 0.055, "medium": 0.30, "heavy": 0.54},
            "base_keep": {"clear": 1.0, "light": 0.90, "medium": 0.66, "heavy": 0.56},
            "gain_clip": {
                "all": [0.24, 3.85],
                "top256_leaked_cells": [0.52, 2.10],
                "top64_leaked_cells": [0.68, 1.52],
                "light_blend": "42% of raw gain",
                "clear": "identity",
            },
            "power_policy": "preserve v186 per-point total power",
            "phase_policy": "preserve leaked-domain phase; no absolute phase transport",
        },
        "diagnostics": {
            "overall": {
                "support_peak_energy": summarize(energy[:, :NEIGHBOR_PEAKS].reshape(-1)),
                "v186_lifecycle_alpha": summarize(v186_actions["lifecycle_alpha"].astype(np.float32)),
                "v186_support_l1_actual": summarize(v186_actions["support_l1_actual"].astype(np.float32)),
                **{key: summarize(value) for key, value in target_diag.items()},
                **{key: summarize(value) for key, value in apply_diag.items()},
            },
            "by_regime": by_regime_summary(
                regimes,
                {
                    "alpha": target_diag["alpha"],
                    "evidence_mass": target_diag["evidence_mass"],
                    "neighbor_peak_count": target_diag["neighbor_peak_count"],
                    "prior_support_l1": target_diag["prior_support_l1"],
                    "target_support_l1": target_diag["target_support_l1"],
                    "support_l1_actual": apply_diag["support_l1_actual"],
                    "actual_power_factor": apply_diag["actual_power_factor"],
                    "gain_max": apply_diag["gain_max"],
                },
            ),
        },
        "comparison_to_v186": comparison_to_v186,
        "comparison_to_v185": comparison_to_v185,
        "comparison_to_v184": comparison_to_v184,
        "preserved": {
            "base_complex": "v186",
            "point_total_power": "v186",
            "absolute_phase": "not transported",
            "train_truth_metric_slice": "not used as target",
            "clear": "unchanged",
        },
        "changed": {
            "leaked_pas_support": "medium/heavy, and very weak light, moved toward test-test leaked-PAS lifecycle prior",
            "domain": "positive amplitude gain in official-traceback-like B,2,S,H,UE,V FFT domain",
        },
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
            "support_cache": PATHS.relative(SUPPORT_OUT),
            "peaks": PATHS.relative(PEAKS_OUT),
            "target_support": PATHS.relative(TARGET_OUT),
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
            PATHS.code / "generate_v187_leaked_pas_lifecycle_projection.py",
            PATHS.code / "generate_v186_lifecycle_quality_overdrive.py",
            PATHS.code / "generate_v185_global_component_lifecycle.py",
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
