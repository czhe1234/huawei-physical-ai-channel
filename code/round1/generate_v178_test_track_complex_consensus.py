#!/usr/bin/env python3
"""Generate v178: test-test component-track complex consensus on v173.

Hypothesis:
    The next large signal should come from path/component continuity, not from
    another global residual overdrive.  v173 already uses neighbor support as a
    weak-tail reliability signal, but it treats support evidence as a dense
    field.  v178 works at component level:

    * extract top collapsed AD peaks from the current best v173 output;
    * match those peaks across nearby test points to form local component
      tracks;
    * move/reshape support only toward high-persistence track consensus;
    * optionally smooth each tracked cell's 2-pol x 4-UE complex direction by
      phase-aligning neighbor vectors to the current vector.

Preserved:
    v173 point total power, strong cell stability, no learned/global phase
    regression, no train-truth peak prior.
"""

from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path

import numpy as np

from generate_v87_easy_field_projection import (
    BS_POS,
    MH,
    MP,
    MV,
    N_ANT,
    S_SC,
    ad_transform,
    beta_subcarriers,
    inverse_ad,
    row_cos,
    summarize,
)
from generate_v159_frontier_consensus_support_soup import REGIMES, digest
from generate_v169_power_scaled_tail_denoising import by_regime_summary
from generate_v171_full_complex_ad_tail_denoising import top_mask
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION = "v178_test_track_complex_consensus"
OUT_DIR = PATHS.cache_dir(178)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(178)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v178_Round1_Test_Channel.npy"
SUPPORT_OUT = OUT_DIR / "v178_v173_collapsed_support.npy"
PEAKS_OUT = OUT_DIR / "v178_v173_top_peaks.npz"
TARGET_OUT = OUT_DIR / "v178_test_track_target_support.npy"
ACTION_OUT = OUT_DIR / "v178_test_track_complex_consensus_actions.npz"
REPORT = REPORT_DIR / "v178_test_track_complex_consensus_generate.json"
ZIP_OUT = PATHS.round1_versions / "v178_test_track_complex_consensus_candidate_pending.zip"

BASE = PATHS.cache_dir(173) / "v173_Round1_Test_Channel.npy"
V173_ACTIONS = PATHS.cache_dir(173) / "v173_spatial_coherence_tail_denoising_actions.npz"

FLOOR = 1e-12
PEAK_COUNT = 48
TRACK_PEAKS = 20
PEAK_CANDIDATES = 1536
MAX_TRACKS = 20


def normalize_support(x: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(x, dtype=np.float32), FLOOR)
    out /= np.maximum(out.sum(axis=(1, 2, 3), keepdims=True), TINY)
    return out.astype(np.float32)


def circular_delta(a: int, b: int, size: int) -> int:
    delta = abs(int(a) - int(b))
    return int(min(delta, size - delta))


def circular_signed_delta(origin: int, value: int, size: int) -> int:
    return int((int(value) - int(origin) + size // 2) % size - size // 2)


def add_peak_kernel(
    field: np.ndarray,
    h: int,
    v: int,
    delay: int,
    mass: float,
    h_sigma: float = 0.95,
    v_sigma: float = 0.78,
    d_sigma: float = 2.45,
) -> None:
    entries: list[tuple[int, int, int, float]] = []
    total = 0.0
    for dh in range(-2, 3):
        hh = (int(h) + dh) % MH
        wh = math.exp(-0.5 * (dh / h_sigma) ** 2)
        for dv in range(-1, 2):
            vv = (int(v) + dv) % MV
            wv = math.exp(-0.5 * (dv / v_sigma) ** 2)
            for dd in range(-6, 7):
                tau = int(delay) + dd
                if tau < 0 or tau >= S_SC:
                    continue
                wd = math.exp(-0.5 * (dd / d_sigma) ** 2)
                w = wh * wv * wd
                entries.append((hh, vv, tau, w))
                total += w
    if total <= 0:
        return
    scale = float(mass) / total
    for hh, vv, tau, w in entries:
        field[hh, vv, tau] += np.float32(scale * w)


def extract_v173_support_and_peaks(pos_test: np.ndarray, batch: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if SUPPORT_OUT.is_file() and PEAKS_OUT.is_file():
        support = normalize_support(np.load(SUPPORT_OUT).astype(np.float32))
        data = np.load(PEAKS_OUT)
        return (
            support,
            data["peaks"].astype(np.int16),
            data["energy"].astype(np.float32),
            data["vectors"].astype(np.complex64),
        )

    source = np.load(BASE, mmap_mode="r")
    beta_s = beta_subcarriers(140.3, 0.10)
    distance = np.linalg.norm(pos_test - BS_POS[None, :], axis=1)
    support = np.empty((source.shape[0], MH, MV, S_SC), dtype=np.float32)
    peaks = np.full((source.shape[0], PEAK_COUNT, 3), -1, dtype=np.int16)
    energy = np.zeros((source.shape[0], PEAK_COUNT), dtype=np.float32)
    vectors = np.zeros((source.shape[0], PEAK_COUNT, MP, N_ANT), dtype=np.complex64)

    for start in range(0, len(source), batch):
        end = min(start + batch, len(source))
        h = np.asarray(source[start:end], dtype=np.complex64)
        derotated = h * np.exp(
            1j * distance[start:end, None, None, None] * beta_s
        ).astype(np.complex64)
        ad = ad_transform(derotated)
        power = (np.abs(ad) ** 2).astype(np.float32)
        collapsed = power.sum(axis=(1, 4), dtype=np.float64).astype(np.float32)
        block_support = collapsed / np.maximum(collapsed.sum(axis=(1, 2, 3), keepdims=True), TINY)
        support[start:end] = block_support

        flat = block_support.reshape(end - start, -1)
        candidate_count = min(PEAK_CANDIDATES, flat.shape[1])
        candidates = np.argpartition(flat, -candidate_count, axis=1)[:, -candidate_count:]
        candidate_values = np.take_along_axis(flat, candidates, axis=1)
        order = np.argsort(candidate_values, axis=1)[:, ::-1]
        candidates = np.take_along_axis(candidates, order, axis=1)

        for row in range(end - start):
            accepted: list[tuple[int, int, int]] = []
            for flat_index in candidates[row]:
                h_bin, remainder = divmod(int(flat_index), MV * S_SC)
                v_bin, delay = divmod(remainder, S_SC)
                too_close = any(
                    circular_delta(h_bin, ah, MH) <= 1
                    and circular_delta(v_bin, av, MV) <= 1
                    and abs(delay - adelay) <= 2
                    for ah, av, adelay in accepted
                )
                if too_close:
                    continue
                slot = len(accepted)
                peaks[start + row, slot] = (h_bin, v_bin, delay)
                energy[start + row, slot] = flat[row, flat_index]
                vectors[start + row, slot] = ad[row, :, h_bin, v_bin, :, delay]
                accepted.append((h_bin, v_bin, delay))
                if slot + 1 >= PEAK_COUNT:
                    break
        print(f"v173 support/peaks {end}/{len(source)}", flush=True)

    support = normalize_support(support)
    np.save(SUPPORT_OUT, support)
    np.savez_compressed(PEAKS_OUT, peaks=peaks, energy=energy, vectors=vectors)
    return support, peaks, energy, vectors


def neighbor_graph(pos: np.ndarray, blocked: np.ndarray, regimes: np.ndarray, k: int = 18) -> tuple[np.ndarray, np.ndarray]:
    xy = pos[:, :2].astype(np.float32)
    dxy = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2).astype(np.float32)
    db = np.abs(blocked[:, None] - blocked[None, :]).astype(np.float32)
    regime_mismatch = (regimes[:, None] != regimes[None, :]).astype(np.float32)
    effective = dxy + 0.18 * db + 4.0 * regime_mismatch
    np.fill_diagonal(effective, np.inf)
    idx = np.argpartition(effective, k, axis=1)[:, :k]
    rows = np.arange(len(pos))[:, None]
    idx = idx[rows, np.argsort(effective[rows, idx], axis=1)]
    selected = effective[rows, idx]
    weights = np.exp(-selected / 16.0).astype(np.float32)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), TINY)
    return idx.astype(np.int32), weights.astype(np.float32)


def match_neighbor_peak(
    current: tuple[int, int, int],
    neighbor_peaks: np.ndarray,
    neighbor_energy: np.ndarray,
) -> tuple[int, float, tuple[int, int, int]]:
    h0, v0, d0 = current
    best_slot = -1
    best_score = 0.0
    best_delta = (0, 0, 0)
    for slot in range(PEAK_COUNT):
        h, v, delay = neighbor_peaks[slot]
        if h < 0:
            break
        dh_abs = circular_delta(h0, int(h), MH)
        dv_abs = circular_delta(v0, int(v), MV)
        dd_abs = abs(int(delay) - d0)
        if dh_abs > 2 or dv_abs > 1 or dd_abs > 6:
            continue
        score = math.exp(
            -0.5
            * (
                (dh_abs / 1.20) ** 2
                + (dv_abs / 0.82) ** 2
                + (dd_abs / 3.10) ** 2
            )
        )
        score *= float(np.sqrt(max(float(neighbor_energy[slot]), FLOOR)))
        if score > best_score:
            best_score = score
            best_slot = slot
            best_delta = (
                circular_signed_delta(h0, int(h), MH),
                circular_signed_delta(v0, int(v), MV),
                int(delay) - d0,
            )
    return best_slot, best_score, best_delta


def unit_vec(vec: np.ndarray) -> np.ndarray:
    norm = np.sqrt(np.sum(np.abs(vec) ** 2) + 1e-18).astype(np.float32)
    return (vec / norm).astype(np.complex64)


def build_track_target(
    support: np.ndarray,
    peaks: np.ndarray,
    energy: np.ndarray,
    vectors: np.ndarray,
    neighbor_idx: np.ndarray,
    neighbor_weight: np.ndarray,
    regimes: np.ndarray,
    v173_action: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    target = np.empty_like(support, dtype=np.float32)
    track_cells = np.full((len(support), MAX_TRACKS, 3), -1, dtype=np.int16)
    track_dirs = np.zeros((len(support), MAX_TRACKS, MP, N_ANT), dtype=np.complex64)
    track_weights = np.zeros((len(support), MAX_TRACKS), dtype=np.float32)

    track_conf = np.zeros(len(support), dtype=np.float32)
    stable_track_count = np.zeros(len(support), dtype=np.float32)
    stable_energy_fraction = np.zeros(len(support), dtype=np.float32)
    mean_persistence = np.zeros(len(support), dtype=np.float32)
    mean_shift_abs = np.zeros(len(support), dtype=np.float32)
    alpha = np.zeros(len(support), dtype=np.float32)
    prior_cos = np.ones(len(support), dtype=np.float32)

    for i in range(len(support)):
        field = (0.34 * support[i]).astype(np.float32)
        total_top_energy = float(np.sum(energy[i, :16])) + FLOOR
        stable_energy = 0.0
        persistence_acc = 0.0
        shift_acc = 0.0
        track_slot = 0

        for pidx in range(TRACK_PEAKS):
            h0, v0, d0 = peaks[i, pidx]
            if h0 < 0:
                break
            e0 = float(energy[i, pidx])
            if e0 < 1.0e-4:
                continue
            current_dir = unit_vec(vectors[i, pidx])

            weighted_score = 0.0
            weighted_presence = 0.0
            dh_sum = 0.0
            dv_sum = 0.0
            dd_sum = 0.0
            neighbor_energy_sum = 0.0
            dir_sum = (0.55 * current_dir).astype(np.complex64)
            dir_weight = 0.55
            matched_neighbors = 0

            for nslot, nb in enumerate(neighbor_idx[i]):
                qslot, score, delta = match_neighbor_peak(
                    (int(h0), int(v0), int(d0)),
                    peaks[nb],
                    energy[nb],
                )
                if qslot < 0:
                    continue
                w = float(neighbor_weight[i, nslot]) * score / math.sqrt(max(e0, FLOOR))
                if w <= 0:
                    continue
                matched_neighbors += 1
                presence = float(neighbor_weight[i, nslot]) * min(1.0, score / math.sqrt(max(e0, FLOOR)))
                weighted_presence += presence
                weighted_score += w
                dh_sum += w * delta[0]
                dv_sum += w * delta[1]
                dd_sum += w * delta[2]
                neighbor_energy_sum += w * float(energy[nb, qslot])

                ndir = unit_vec(vectors[nb, qslot])
                inner = np.sum(np.conj(ndir) * current_dir)
                phase = inner / max(abs(inner), 1e-12)
                dir_sum += np.complex64(w) * (ndir * phase).astype(np.complex64)
                dir_weight += w

            persistence = float(np.clip(weighted_presence, 0.0, 1.0))
            if matched_neighbors < 5 or persistence < 0.30 or weighted_score <= 0:
                # weak isolated peaks remain represented by the 0.24*support
                # floor, but are not allowed to drive the track prior.
                continue

            mean_dh = float(np.clip(dh_sum / weighted_score, -1.8, 1.8))
            mean_dv = float(np.clip(dv_sum / weighted_score, -1.2, 1.2))
            mean_dd = float(np.clip(dd_sum / weighted_score, -4.0, 4.0))
            h_track = (int(h0) + int(round(0.58 * mean_dh))) % MH
            v_track = (int(v0) + int(round(0.58 * mean_dv))) % MV
            d_track = int(np.clip(int(d0) + int(round(0.58 * mean_dd)), 0, S_SC - 1))

            neighbor_mean_energy = neighbor_energy_sum / weighted_score
            amp_ratio = float(np.clip(math.sqrt(max(neighbor_mean_energy, FLOOR) / max(e0, FLOOR)), 0.72, 1.36))
            mass = e0 * (0.60 + 0.65 * persistence) * amp_ratio
            add_peak_kernel(field, h_track, v_track, d_track, mass)

            stable_energy += e0 * persistence
            persistence_acc += persistence
            shift_acc += abs(mean_dh) + 0.65 * abs(mean_dv) + 0.24 * abs(mean_dd)
            stable_track_count[i] += 1.0

            if track_slot < MAX_TRACKS:
                target_dir = unit_vec(dir_sum / np.complex64(max(dir_weight, 1e-12)))
                track_cells[i, track_slot] = (h_track, v_track, d_track)
                # Direction smoothing is intentionally much weaker than
                # support movement.  It tests phase/polUE continuity without
                # replacing v173's absolute phase.
                track_dirs[i, track_slot] = target_dir
                if str(regimes[i]) == "medium":
                    direction_scale = 0.70
                elif str(regimes[i]) == "heavy":
                    direction_scale = 1.00
                else:
                    direction_scale = 0.00
                track_weights[i, track_slot] = np.float32(
                    direction_scale * min(0.18, 0.035 + 0.18 * persistence)
                )
                track_slot += 1

        if stable_track_count[i] > 0:
            field = np.maximum(field, FLOOR)
            prior = field / np.maximum(field.sum(), TINY)
            stable_energy_fraction[i] = np.float32(np.clip(stable_energy / total_top_energy, 0.0, 1.0))
            mean_persistence[i] = np.float32(persistence_acc / stable_track_count[i])
            mean_shift_abs[i] = np.float32(shift_acc / stable_track_count[i])
            prior_cos[i] = float(row_cos(prior[None], support[i : i + 1])[0])
        else:
            prior = support[i]
            prior_cos[i] = 1.0

        conf = np.clip(
            0.55 * stable_energy_fraction[i]
            + 0.35 * mean_persistence[i]
            + 0.10 * np.clip(stable_track_count[i] / 9.0, 0.0, 1.0),
            0.0,
            1.0,
        )
        track_conf[i] = conf

        regime = str(regimes[i])
        if regime == "clear":
            scale = 0.0
        elif regime == "light":
            scale = 0.0
        elif regime == "medium":
            scale = 0.18
        else:
            scale = 0.32
        # Use v173 action as a reliability mask, not as the target.  The
        # constant floor lets very coherent medium/heavy tracks move even when
        # v173's own spatial action was modest.
        point_alpha = scale * conf * (0.38 + 0.62 * math.sqrt(float(np.clip(v173_action[i], 0.0, 1.0))))
        alpha[i] = np.float32(np.clip(point_alpha, 0.0, scale))
        target[i] = normalize_support(
            ((1.0 - alpha[i]) * support[i] + alpha[i] * prior)[None]
        )[0]

        if (i + 1) % 100 == 0:
            print(f"track target {i + 1}/{len(support)}", flush=True)

    diagnostics = {
        "track_cells": track_cells,
        "track_dirs": track_dirs,
        "track_weights": track_weights,
        "track_conf": track_conf,
        "stable_track_count": stable_track_count,
        "stable_energy_fraction": stable_energy_fraction,
        "mean_persistence": mean_persistence,
        "mean_shift_abs": mean_shift_abs,
        "alpha": alpha,
        "prior_cos": prior_cos,
        "target_support_l1": np.sum(np.abs(target - support), axis=(1, 2, 3)).astype(np.float32),
        "target_support_cos": row_cos(target, support).astype(np.float32),
    }
    return normalize_support(target), diagnostics


def apply_track_target(
    pos_test: np.ndarray,
    target: np.ndarray,
    track_cells: np.ndarray,
    track_dirs: np.ndarray,
    track_weights: np.ndarray,
    batch: int = 20,
) -> dict[str, np.ndarray]:
    source = np.load(BASE, mmap_mode="r")
    output = np.lib.format.open_memmap(OUTPUT, mode="w+", dtype=np.complex64, shape=source.shape)
    beta_s = beta_subcarriers(140.3, 0.10)
    distance = np.linalg.norm(pos_test - BS_POS[None, :], axis=1)

    diag: dict[str, list[np.ndarray]] = {
        "actual_power_factor": [],
        "support_l1_actual": [],
        "gain_min": [],
        "gain_p10": [],
        "gain_p90": [],
        "gain_max": [],
        "direction_update_energy": [],
    }

    for start in range(0, len(source), batch):
        end = min(start + batch, len(source))
        h0 = np.asarray(source[start:end], dtype=np.complex64)
        derotated = h0 * np.exp(
            1j * distance[start:end, None, None, None] * beta_s
        ).astype(np.complex64)
        ad0 = ad_transform(derotated)
        cell0 = (np.abs(ad0) ** 2).astype(np.float32).sum(axis=(1, 4), dtype=np.float64).astype(np.float32)
        total0 = np.maximum(cell0.sum(axis=(1, 2, 3)), TINY).astype(np.float32)
        support0 = cell0 / total0[:, None, None, None]

        target_cell = target[start:end] * total0[:, None, None, None]
        gain_cell = np.sqrt(target_cell / np.maximum(cell0, 1e-30)).astype(np.float32)
        gain_cell = np.clip(gain_cell, 0.22, 4.25).astype(np.float32)
        flat = support0.reshape(end - start, -1)
        strong64 = top_mask(flat, 64).reshape(end - start, MH, MV, S_SC)
        strong256 = top_mask(flat, 256).reshape(end - start, MH, MV, S_SC)
        gain_cell[strong256] = np.clip(gain_cell[strong256], 0.54, 2.05)
        gain_cell[strong64] = np.clip(gain_cell[strong64], 0.70, 1.46)

        ad1 = ad0 * gain_cell[:, None, :, :, None, :]
        total_after = np.maximum(
            np.sum(np.abs(ad1) ** 2, axis=(1, 2, 3, 4, 5), dtype=np.float64),
            TINY,
        ).astype(np.float32)
        ad1 *= np.sqrt(total0 / total_after)[:, None, None, None, None, None].astype(np.float32)

        direction_energy = np.zeros(end - start, dtype=np.float32)
        for local, global_i in enumerate(range(start, end)):
            for slot in range(MAX_TRACKS):
                weight = float(track_weights[global_i, slot])
                if weight <= 0:
                    break
                h_bin, v_bin, delay = track_cells[global_i, slot]
                if h_bin < 0:
                    break
                vec = ad1[local, :, int(h_bin), int(v_bin), :, int(delay)]
                norm = np.sqrt(np.sum(np.abs(vec) ** 2) + 1e-18).astype(np.float32)
                if norm <= 0:
                    continue
                current_dir = (vec / norm).astype(np.complex64)
                target_dir = track_dirs[global_i, slot]
                inner = np.sum(np.conj(target_dir) * current_dir)
                phase = inner / max(abs(inner), 1e-12)
                aligned = (target_dir * phase).astype(np.complex64)
                mixed = (1.0 - weight) * current_dir + weight * aligned
                mixed_norm = np.sqrt(np.sum(np.abs(mixed) ** 2) + 1e-18).astype(np.float32)
                ad1[local, :, int(h_bin), int(v_bin), :, int(delay)] = (mixed / mixed_norm * norm).astype(np.complex64)
                direction_energy[local] += np.float32((weight * weight) * norm * norm)

        # Direction updates preserve each edited cell norm, but do one final
        # total-power normalization to remove numerical drift.
        total_after_dir = np.maximum(
            np.sum(np.abs(ad1) ** 2, axis=(1, 2, 3, 4, 5), dtype=np.float64),
            TINY,
        ).astype(np.float32)
        ad1 *= np.sqrt(total0 / total_after_dir)[:, None, None, None, None, None].astype(np.float32)

        h1 = inverse_ad(ad1).reshape(end - start, 256, 4, 192)
        h1 = h1 * np.exp(
            -1j * distance[start:end, None, None, None] * beta_s
        ).astype(np.complex64)
        if not (np.isfinite(h1.real).all() and np.isfinite(h1.imag).all()):
            raise FloatingPointError(f"non-finite block {start}:{end}")
        output[start:end] = h1.astype(np.complex64)
        output.flush()

        cell1 = (np.abs(ad1) ** 2).astype(np.float32).sum(axis=(1, 4), dtype=np.float64).astype(np.float32)
        support1 = cell1 / np.maximum(cell1.sum(axis=(1, 2, 3), keepdims=True), TINY)
        diag["actual_power_factor"].append((cell1.sum(axis=(1, 2, 3)) / total0).astype(np.float32))
        diag["support_l1_actual"].append(np.sum(np.abs(support1 - support0), axis=(1, 2, 3)).astype(np.float32))
        gflat = gain_cell.reshape(end - start, -1)
        diag["gain_min"].append(gflat.min(axis=1).astype(np.float32))
        diag["gain_p10"].append(np.quantile(gflat, 0.10, axis=1).astype(np.float32))
        diag["gain_p90"].append(np.quantile(gflat, 0.90, axis=1).astype(np.float32))
        diag["gain_max"].append(gflat.max(axis=1).astype(np.float32))
        diag["direction_update_energy"].append((direction_energy / total0).astype(np.float32))
        print(f"track consensus applied {end}/500", flush=True)

    return {key: np.concatenate(value).astype(np.float32) for key, value in diag.items()}


def compare_output_to(output_path: Path, base_path: Path, regimes: np.ndarray) -> dict[str, object]:
    output = np.load(output_path, mmap_mode="r")
    base = np.load(base_path, mmap_mode="r")
    numerator = np.zeros(len(output), dtype=np.float64)
    denominator = np.zeros(len(output), dtype=np.float64)
    out_power = np.zeros(len(output), dtype=np.float64)
    base_power = np.zeros(len(output), dtype=np.float64)
    finite = True
    for start in range(0, len(output), 20):
        end = min(start + 20, len(output))
        current = np.asarray(output[start:end], dtype=np.complex64)
        reference = np.asarray(base[start:end], dtype=np.complex64)
        numerator[start:end] = np.sum(np.abs(current - reference) ** 2, axis=(1, 2, 3), dtype=np.float64)
        denominator[start:end] = np.sum(np.abs(reference) ** 2, axis=(1, 2, 3), dtype=np.float64)
        out_power[start:end] = np.sum(np.abs(current) ** 2, axis=(1, 2, 3), dtype=np.float64)
        base_power[start:end] = denominator[start:end]
        finite = finite and bool(np.isfinite(current.real).all() and np.isfinite(current.imag).all())
    point_nmse = numerator / np.maximum(denominator, TINY)
    return {
        "finite": finite,
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "nmse": float(numerator.sum() / max(denominator.sum(), TINY)),
        "nmse_by_regime": {
            regime: float(numerator[regimes == regime].sum() / max(denominator[regimes == regime].sum(), TINY))
            for regime in REGIMES
            if np.any(regimes == regime)
        },
        "point_nmse": summarize(point_nmse),
        "power_ratio": summarize(out_power / np.maximum(base_power, TINY)),
    }


def generate() -> dict[str, object]:
    pos_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    blocked = np.load(PATHS.shared_features / "height_geometry_cache.npz")["test_blocked"].astype(np.float32)
    regimes = regime_of(blocked)
    action_data = np.load(V173_ACTIONS)
    v173_action = action_data["action"].astype(np.float32)

    support, peaks, energy, vectors = extract_v173_support_and_peaks(pos_test)
    idx, weights = neighbor_graph(pos_test, blocked, regimes)
    target, track = build_track_target(
        support,
        peaks,
        energy,
        vectors,
        idx,
        weights,
        regimes,
        v173_action,
    )
    np.save(TARGET_OUT, target)
    apply_diag = apply_track_target(
        pos_test,
        target,
        track["track_cells"],
        track["track_dirs"],
        track["track_weights"],
    )

    np.savez_compressed(
        ACTION_OUT,
        regimes=regimes.astype("U8"),
        neighbor_idx=idx,
        neighbor_weights=weights,
        v173_action=v173_action,
        **track,
        **apply_diag,
    )

    comparison_to_v173 = compare_output_to(OUTPUT, BASE, regimes)
    comparison_to_v171 = compare_output_to(OUTPUT, PATHS.cache_dir(171) / "v171_Round1_Test_Channel.npy", regimes)
    comparison_to_v170 = compare_output_to(OUTPUT, PATHS.cache_dir(170) / "v170_Round1_Test_Channel.npy", regimes)
    comparison_to_v163 = compare_output_to(OUTPUT, PATHS.cache_dir(163) / "v163_Round1_Test_Channel.npy", regimes)
    if comparison_to_v173["shape"] != [500, 256, 4, 192] or comparison_to_v173["dtype"] != "complex64":
        raise ValueError(f"invalid output layout: {comparison_to_v173['shape']} {comparison_to_v173['dtype']}")
    if not comparison_to_v173["finite"]:
        raise ValueError("v178 output contains non-finite values")

    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "Sionna paths should form locally continuous AD components across "
            "nearby test points.  v178 uses only v173 test outputs to track "
            "high-persistence components, then applies support movement and "
            "phase-aligned cell-internal complex-direction smoothing on those tracks."
        ),
        "new_information_used": {
            "current_best": "v173 online 0.503667",
            "test_test_component_tracks": "top v173 AD peaks matched among nearby test receivers",
            "phase_policy": "no global phase regression; neighbor vectors are phase-aligned to current cell before blending",
            "negative_guardrail": "does not use train-truth peak prior that caused v158 to fail",
        },
        "parameters": {
            "base": PATHS.relative(BASE),
            "peak_count": PEAK_COUNT,
            "track_peaks_per_point": TRACK_PEAKS,
            "peak_candidates": PEAK_CANDIDATES,
            "neighbor_k": 18,
            "neighbor_metric": "XY + 0.18*blocked_delta + 4.0*regime_mismatch",
            "match_window": {"h": 2, "v": 1, "delay": 6},
            "alpha_scale": {"clear": 0.0, "light": 0.0, "medium": 0.18, "heavy": 0.32},
            "support_gain_clip": [0.22, 4.25],
            "top256_gain_clip": [0.54, 2.05],
            "top64_gain_clip": [0.70, 1.46],
            "max_direction_weight": 0.18,
            "power_policy": "preserve v173 per-point total power",
        },
        "diagnostics": {
            "overall": {
                "v173_action": summarize(v173_action),
                **{
                    key: summarize(value)
                    for key, value in track.items()
                    if isinstance(value, np.ndarray)
                    and value.ndim == 1
                    and np.issubdtype(value.dtype, np.number)
                },
                **{key: summarize(value) for key, value in apply_diag.items()},
            },
            "by_regime": by_regime_summary(
                regimes,
                {
                    "v173_action": v173_action,
                    "track_conf": track["track_conf"],
                    "stable_track_count": track["stable_track_count"],
                    "stable_energy_fraction": track["stable_energy_fraction"],
                    "mean_persistence": track["mean_persistence"],
                    "mean_shift_abs": track["mean_shift_abs"],
                    "alpha": track["alpha"],
                    "target_support_l1": track["target_support_l1"],
                    "support_l1_actual": apply_diag["support_l1_actual"],
                    "direction_update_energy": apply_diag["direction_update_energy"],
                    "actual_power_factor": apply_diag["actual_power_factor"],
                },
            ),
        },
        "comparison_to_v173": comparison_to_v173,
        "comparison_to_v171": comparison_to_v171,
        "comparison_to_v170": comparison_to_v170,
        "comparison_to_v163": comparison_to_v163,
        "preserved": {
            "base_complex": "v173",
            "point_total_power": "v173",
            "global_phase": "not learned; not globally regressed",
            "clear": "no support action",
            "randomness": "none",
        },
        "changed": {
            "tracked_support": "medium/heavy support moved toward local test-test component-track consensus",
            "cell_internal_direction": "tracked cells get weak phase-aligned 2x4 complex-direction smoothing",
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
        for source, name in (
            (__file__, "generate_v178_test_track_complex_consensus.py"),
            (PATHS.code / "generate_v173_spatial_coherence_tail_denoising.py", "generate_v173_spatial_coherence_tail_denoising.py"),
            (PATHS.code / "generate_v171_full_complex_ad_tail_denoising.py", "generate_v171_full_complex_ad_tail_denoising.py"),
            (PATHS.code / "generate_v87_easy_field_projection.py", "generate_v87_easy_field_projection.py"),
            (PATHS.code / "generate_v159_frontier_consensus_support_soup.py", "generate_v159_frontier_consensus_support_soup.py"),
            (PATHS.code / "generate_v169_power_scaled_tail_denoising.py", "generate_v169_power_scaled_tail_denoising.py"),
            (PATHS.code / "powerfield_common.py", "powerfield_common.py"),
            (PATHS.code_root / "project_paths.py", "project_paths.py"),
        ):
            archive.write(source, name)

    with zipfile.ZipFile(ZIP_OUT) as archive:
        bad_member = archive.testzip()
        members = archive.namelist()
    if bad_member is not None:
        raise ValueError(f"ZIP CRC failure: {bad_member}")
    output = np.load(OUTPUT, mmap_mode="r")
    integrity = {
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": bool(np.isfinite(output.real).all() and np.isfinite(output.imag).all()),
        "zip_crc": True,
        "zip_members": members,
        "sha256": digest(ZIP_OUT, "sha256"),
        "md5": digest(ZIP_OUT, "md5"),
    }
    report["integrity"] = integrity
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    report = generate()
    report = write_zip(report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
