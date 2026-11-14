#!/usr/bin/env python3
"""v232 analysis: test-test AD component tracks on the v226 carrier.

v226 is the current best strict official-data local result.  It improves PDP
with public point-cloud path-delay kernels, but still misses joint AD support
and leaked S-H structure.  This probe reuses a historically positive idea in a
strict form: if a real propagation component exists, its angle-delay peak
should move smoothly across nearby public test coordinates.

Generation boundary:
* uses official test coordinates and official geometry cache;
* uses v226 official-data output as the current carrier;
* derives all component tracks from the v226 test output itself;
* does not use Sionna generated channel labels, v212/v217 teacher outputs,
  synthetic replay truth, hidden online labels, or raw RT H.

The script is analysis-only.  It scans several raw/derotated AD track actions
and scores them with the local v212 gate.  No ZIP is created unless a later
workflow explicitly packages a passing candidate.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import analyze_v208_strict_historical_replay as v208
from generate_v159_frontier_consensus_support_soup import REGIMES
from generate_v169_power_scaled_tail_denoising import by_regime_summary
from generate_v171_full_complex_ad_tail_denoising import top_mask
from generate_v87_easy_field_projection import (
    BS_POS,
    MH,
    MP,
    MV,
    N_ANT,
    S_SC,
    beta_subcarriers,
    row_cos,
    summarize,
)
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 232
VERSION = "v232_official_v226_component_track"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

BASE = PATHS.cache_dir(226) / "v226_early_shift_v224_path_heavy_Round1_Test_Channel.npy"
GEOM_CACHE = PATHS.shared_features / "height_geometry_cache.npz"
REPORT = REPORT_DIR / "v232_official_v226_component_track.json"
ACTION_OUT = OUT_DIR / "v232_component_track_actions.npz"

FLOOR = 1e-12
PEAK_COUNT = 56
TRACK_PEAKS = 24
PEAK_CANDIDATES = 2048
MAX_TRACKS = 24
LOCAL_BATCH = 16
GATE_SCORE = 0.645604


def normalize_support(x: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(x, dtype=np.float32), FLOOR)
    out /= np.maximum(out.sum(axis=(1, 2, 3), keepdims=True), TINY)
    return out.astype(np.float32)


def circular_delta(a: int, b: int, size: int) -> int:
    delta = abs(int(a) - int(b))
    return int(min(delta, size - delta))


def circular_signed_delta(origin: int, value: int, size: int) -> int:
    return int((int(value) - int(origin) + size // 2) % size - size // 2)


def transform_to_ad(h: np.ndarray, distance: np.ndarray, beta_s: np.ndarray | None) -> np.ndarray:
    work = h
    if beta_s is not None:
        work = h * np.exp(1j * distance[:, None, None, None] * beta_s).astype(np.complex64)
    cir = np.fft.ifft(work, axis=-1)
    arr = cir.reshape(len(h), MP, MH, MV, N_ANT, S_SC)
    return np.fft.fft2(arr, axes=(2, 3)).astype(np.complex64)


def inverse_from_ad(ad: np.ndarray, distance: np.ndarray, beta_s: np.ndarray | None) -> np.ndarray:
    arr = np.fft.ifft2(ad, axes=(2, 3))
    h = np.fft.fft(arr.reshape(len(ad), MP * MH * MV, N_ANT, S_SC), axis=-1).astype(np.complex64)
    if beta_s is not None:
        h = h * np.exp(-1j * distance[:, None, None, None] * beta_s).astype(np.complex64)
    return h.astype(np.complex64)


def unit_vec(vec: np.ndarray) -> np.ndarray:
    norm = np.sqrt(np.sum(np.abs(vec) ** 2) + 1e-18).astype(np.float32)
    return (vec / norm).astype(np.complex64)


def add_peak_kernel(
    field: np.ndarray,
    h_bin: int,
    v_bin: int,
    delay: int,
    mass: float,
    h_sigma: float = 1.05,
    v_sigma: float = 0.82,
    d_sigma: float = 2.85,
) -> None:
    entries: list[tuple[int, int, int, float]] = []
    total = 0.0
    for dh in range(-2, 3):
        hh = (int(h_bin) + dh) % MH
        wh = math.exp(-0.5 * (dh / h_sigma) ** 2)
        for dv in range(-1, 2):
            vv = (int(v_bin) + dv) % MV
            wv = math.exp(-0.5 * (dv / v_sigma) ** 2)
            for dd in range(-7, 8):
                tau = int(delay) + dd
                if tau < 0 or tau >= S_SC:
                    continue
                wd = math.exp(-0.5 * (dd / d_sigma) ** 2)
                weight = wh * wv * wd
                entries.append((hh, vv, tau, weight))
                total += weight
    if total <= 0.0:
        return
    scale = float(mass) / total
    for hh, vv, tau, weight in entries:
        field[hh, vv, tau] += np.float32(scale * weight)


def support_cache_paths(mode: str) -> tuple[Path, Path]:
    return (
        OUT_DIR / f"v232_{mode}_v226_collapsed_support.npy",
        OUT_DIR / f"v232_{mode}_v226_top_peaks.npz",
    )


def extract_support_and_peaks(
    mode: str,
    pos_test: np.ndarray,
    batch: int = LOCAL_BATCH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    support_path, peaks_path = support_cache_paths(mode)
    if support_path.is_file() and peaks_path.is_file():
        support = normalize_support(np.load(support_path).astype(np.float32))
        data = np.load(peaks_path)
        return (
            support,
            data["peaks"].astype(np.int16),
            data["energy"].astype(np.float32),
            data["vectors"].astype(np.complex64),
        )

    source = np.load(BASE, mmap_mode="r")
    distance = np.linalg.norm(pos_test - BS_POS[None, :], axis=1).astype(np.float32)
    beta_s = beta_subcarriers(140.3, 0.10) if mode == "derot" else None
    support = np.empty((len(source), MH, MV, S_SC), dtype=np.float32)
    peaks = np.full((len(source), PEAK_COUNT, 3), -1, dtype=np.int16)
    energy = np.zeros((len(source), PEAK_COUNT), dtype=np.float32)
    vectors = np.zeros((len(source), PEAK_COUNT, MP, N_ANT), dtype=np.complex64)

    for start in range(0, len(source), batch):
        end = min(start + batch, len(source))
        h = np.asarray(source[start:end], dtype=np.complex64)
        ad = transform_to_ad(h, distance[start:end], beta_s)
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
        print(f"v232 {mode} support/peaks {end}/{len(source)}", flush=True)

    support = normalize_support(support)
    np.save(support_path, support)
    np.savez_compressed(peaks_path, peaks=peaks, energy=energy, vectors=vectors)
    return support, peaks, energy, vectors


def neighbor_graph(
    pos: np.ndarray,
    blocked: np.ndarray,
    regimes: np.ndarray,
    k: int,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    xyz = pos.astype(np.float32)
    xy = xyz[:, :2]
    dxy = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2).astype(np.float32)
    dz = np.abs(xyz[:, None, 2] - xyz[None, :, 2]).astype(np.float32)
    db = np.abs(blocked[:, None] - blocked[None, :]).astype(np.float32)
    regime_mismatch = (regimes[:, None] != regimes[None, :]).astype(np.float32)
    effective = dxy + 0.42 * dz + 0.18 * db + 4.25 * regime_mismatch
    np.fill_diagonal(effective, np.inf)
    idx = np.argpartition(effective, k, axis=1)[:, :k]
    rows = np.arange(len(pos))[:, None]
    idx = idx[rows, np.argsort(effective[rows, idx], axis=1)]
    selected = effective[rows, idx]
    weights = np.exp(-selected / float(scale)).astype(np.float32)
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
        h_bin, v_bin, delay = neighbor_peaks[slot]
        if h_bin < 0:
            break
        dh_abs = circular_delta(h0, int(h_bin), MH)
        dv_abs = circular_delta(v0, int(v_bin), MV)
        dd_abs = abs(int(delay) - d0)
        if dh_abs > 2 or dv_abs > 1 or dd_abs > 7:
            continue
        score = math.exp(
            -0.5
            * (
                (dh_abs / 1.25) ** 2
                + (dv_abs / 0.88) ** 2
                + (dd_abs / 3.55) ** 2
            )
        )
        score *= float(np.sqrt(max(float(neighbor_energy[slot]), FLOOR)))
        if score > best_score:
            best_score = score
            best_slot = slot
            best_delta = (
                circular_signed_delta(h0, int(h_bin), MH),
                circular_signed_delta(v0, int(v_bin), MV),
                int(delay) - d0,
            )
    return best_slot, best_score, best_delta


def build_track_prior(
    support: np.ndarray,
    peaks: np.ndarray,
    energy: np.ndarray,
    vectors: np.ndarray,
    neighbor_idx: np.ndarray,
    neighbor_weight: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    prior = np.empty_like(support, dtype=np.float32)
    track_cells = np.full((len(support), MAX_TRACKS, 3), -1, dtype=np.int16)
    track_dirs = np.zeros((len(support), MAX_TRACKS, MP, N_ANT), dtype=np.complex64)
    track_weights = np.zeros((len(support), MAX_TRACKS), dtype=np.float32)

    track_conf = np.zeros(len(support), dtype=np.float32)
    stable_track_count = np.zeros(len(support), dtype=np.float32)
    stable_energy_fraction = np.zeros(len(support), dtype=np.float32)
    mean_persistence = np.zeros(len(support), dtype=np.float32)
    mean_shift_abs = np.zeros(len(support), dtype=np.float32)
    prior_cos = np.ones(len(support), dtype=np.float32)

    for i in range(len(support)):
        field = (0.30 * support[i]).astype(np.float32)
        total_top_energy = float(np.sum(energy[i, :18])) + FLOOR
        stable_energy = 0.0
        persistence_acc = 0.0
        shift_acc = 0.0
        track_slot = 0

        for pidx in range(TRACK_PEAKS):
            h0, v0, d0 = peaks[i, pidx]
            if h0 < 0:
                break
            e0 = float(energy[i, pidx])
            if e0 < 7.5e-5:
                continue
            current_dir = unit_vec(vectors[i, pidx])

            weighted_score = 0.0
            weighted_presence = 0.0
            dh_sum = 0.0
            dv_sum = 0.0
            dd_sum = 0.0
            neighbor_energy_sum = 0.0
            dir_sum = (0.50 * current_dir).astype(np.complex64)
            dir_weight = 0.50
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
                if w <= 0.0:
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
            if matched_neighbors < 6 or persistence < 0.32 or weighted_score <= 0.0:
                continue

            mean_dh = float(np.clip(dh_sum / weighted_score, -1.8, 1.8))
            mean_dv = float(np.clip(dv_sum / weighted_score, -1.2, 1.2))
            mean_dd = float(np.clip(dd_sum / weighted_score, -4.8, 4.8))
            h_track = (int(h0) + int(round(0.55 * mean_dh))) % MH
            v_track = (int(v0) + int(round(0.55 * mean_dv))) % MV
            d_track = int(np.clip(int(d0) + int(round(0.55 * mean_dd)), 0, S_SC - 1))

            neighbor_mean_energy = neighbor_energy_sum / weighted_score
            amp_ratio = float(np.clip(math.sqrt(max(neighbor_mean_energy, FLOOR) / max(e0, FLOOR)), 0.70, 1.42))
            mass = e0 * (0.55 + 0.72 * persistence) * amp_ratio
            add_peak_kernel(field, h_track, v_track, d_track, mass)

            stable_energy += e0 * persistence
            persistence_acc += persistence
            shift_acc += abs(mean_dh) + 0.65 * abs(mean_dv) + 0.22 * abs(mean_dd)
            stable_track_count[i] += 1.0

            if track_slot < MAX_TRACKS:
                track_cells[i, track_slot] = (h_track, v_track, d_track)
                track_dirs[i, track_slot] = unit_vec(dir_sum / np.complex64(max(dir_weight, 1e-12)))
                track_weights[i, track_slot] = np.float32(min(0.15, 0.028 + 0.16 * persistence))
                track_slot += 1

        if stable_track_count[i] > 0:
            prior_i = np.maximum(field, FLOOR)
            prior_i = prior_i / np.maximum(prior_i.sum(), TINY)
            stable_energy_fraction[i] = np.float32(np.clip(stable_energy / total_top_energy, 0.0, 1.0))
            mean_persistence[i] = np.float32(persistence_acc / stable_track_count[i])
            mean_shift_abs[i] = np.float32(shift_acc / stable_track_count[i])
            prior_cos[i] = float(row_cos(prior_i[None], support[i : i + 1])[0])
            prior[i] = prior_i.astype(np.float32)
        else:
            prior[i] = support[i]
            prior_cos[i] = 1.0

        conf = np.clip(
            0.54 * stable_energy_fraction[i]
            + 0.34 * mean_persistence[i]
            + 0.12 * np.clip(stable_track_count[i] / 10.0, 0.0, 1.0),
            0.0,
            1.0,
        )
        track_conf[i] = np.float32(conf)
        if (i + 1) % 100 == 0:
            print(f"v232 track prior {i + 1}/{len(support)}", flush=True)

    diagnostics = {
        "track_cells": track_cells,
        "track_dirs": track_dirs,
        "track_weights": track_weights,
        "track_conf": track_conf,
        "stable_track_count": stable_track_count,
        "stable_energy_fraction": stable_energy_fraction,
        "mean_persistence": mean_persistence,
        "mean_shift_abs": mean_shift_abs,
        "prior_cos": prior_cos,
        "prior_l1_to_support": np.sum(np.abs(prior - support), axis=(1, 2, 3)).astype(np.float32),
    }
    return normalize_support(prior), diagnostics


def target_from_prior(
    support: np.ndarray,
    prior: np.ndarray,
    regimes: np.ndarray,
    track_conf: np.ndarray,
    alpha_by_regime: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    scale = np.asarray([alpha_by_regime[str(r)] for r in regimes], dtype=np.float32)
    alpha = np.clip(scale * track_conf, 0.0, scale).astype(np.float32)
    target = normalize_support((1.0 - alpha[:, None, None, None]) * support + alpha[:, None, None, None] * prior)
    return target, alpha


def apply_track_variant(
    name: str,
    mode: str,
    pos_test: np.ndarray,
    support: np.ndarray,
    target: np.ndarray,
    track: dict[str, np.ndarray],
    direction_scale: float,
    gain_clip_scale: float,
    batch: int = LOCAL_BATCH,
) -> dict[str, np.ndarray]:
    source = np.load(BASE, mmap_mode="r")
    output = np.lib.format.open_memmap(
        OUT_DIR / f"v232_{name}_Round1_Test_Channel.npy",
        mode="w+",
        dtype=np.complex64,
        shape=source.shape,
    )
    distance = np.linalg.norm(pos_test - BS_POS[None, :], axis=1).astype(np.float32)
    beta_s = beta_subcarriers(140.3, 0.10) if mode == "derot" else None

    diag: dict[str, list[np.ndarray]] = {
        "power_factor": [],
        "support_l1_actual": [],
        "gain_min": [],
        "gain_p10": [],
        "gain_p90": [],
        "gain_max": [],
        "direction_update_energy": [],
    }
    track_cells = track["track_cells"]
    track_dirs = track["track_dirs"]
    track_weights = track["track_weights"]

    for start in range(0, len(source), batch):
        end = min(start + batch, len(source))
        h0 = np.asarray(source[start:end], dtype=np.complex64)
        p0 = np.maximum(np.sum(np.abs(h0) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY)
        ad0 = transform_to_ad(h0, distance[start:end], beta_s)
        cell0 = (np.abs(ad0) ** 2).astype(np.float32).sum(axis=(1, 4), dtype=np.float64).astype(np.float32)
        total0 = np.maximum(cell0.sum(axis=(1, 2, 3)), TINY).astype(np.float32)
        support0 = cell0 / total0[:, None, None, None]

        target_cell = target[start:end] * total0[:, None, None, None]
        gain_cell = np.sqrt(target_cell / np.maximum(cell0, 1e-30)).astype(np.float32)
        gain_cell = np.clip(gain_cell, 0.20, 4.60 * gain_clip_scale).astype(np.float32)
        flat = support0.reshape(end - start, -1)
        strong64 = top_mask(flat, 64).reshape(end - start, MH, MV, S_SC)
        strong256 = top_mask(flat, 256).reshape(end - start, MH, MV, S_SC)
        strong1024 = top_mask(flat, 1024).reshape(end - start, MH, MV, S_SC)
        gain_cell[strong1024] = np.clip(gain_cell[strong1024], 0.46, 2.30 * gain_clip_scale)
        gain_cell[strong256] = np.clip(gain_cell[strong256], 0.56, 1.88 * gain_clip_scale)
        gain_cell[strong64] = np.clip(gain_cell[strong64], 0.72, 1.42 * gain_clip_scale)

        ad1 = ad0 * gain_cell[:, None, :, :, None, :]
        total_after = np.maximum(
            np.sum(np.abs(ad1) ** 2, axis=(1, 2, 3, 4, 5), dtype=np.float64),
            TINY,
        ).astype(np.float32)
        ad1 *= np.sqrt(total0 / total_after)[:, None, None, None, None, None].astype(np.float32)

        direction_energy = np.zeros(end - start, dtype=np.float32)
        if direction_scale > 0.0:
            for local, global_i in enumerate(range(start, end)):
                for slot in range(MAX_TRACKS):
                    weight = float(track_weights[global_i, slot]) * direction_scale
                    if weight <= 0.0:
                        break
                    h_bin, v_bin, delay = track_cells[global_i, slot]
                    if h_bin < 0:
                        break
                    vec = ad1[local, :, int(h_bin), int(v_bin), :, int(delay)]
                    norm = np.sqrt(np.sum(np.abs(vec) ** 2) + 1e-18).astype(np.float32)
                    if norm <= 0.0:
                        continue
                    current_dir = (vec / norm).astype(np.complex64)
                    target_dir = track_dirs[global_i, slot]
                    inner = np.sum(np.conj(target_dir) * current_dir)
                    phase = inner / max(abs(inner), 1e-12)
                    aligned = (target_dir * phase).astype(np.complex64)
                    mixed = (1.0 - weight) * current_dir + weight * aligned
                    mixed_norm = np.sqrt(np.sum(np.abs(mixed) ** 2) + 1e-18).astype(np.float32)
                    ad1[local, :, int(h_bin), int(v_bin), :, int(delay)] = (
                        mixed / mixed_norm * norm
                    ).astype(np.complex64)
                    direction_energy[local] += np.float32((weight * weight) * norm * norm)

        total_after_dir = np.maximum(
            np.sum(np.abs(ad1) ** 2, axis=(1, 2, 3, 4, 5), dtype=np.float64),
            TINY,
        ).astype(np.float32)
        ad1 *= np.sqrt(total0 / total_after_dir)[:, None, None, None, None, None].astype(np.float32)
        h1 = inverse_from_ad(ad1, distance[start:end], beta_s)
        p1 = np.maximum(np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY)
        h1 *= np.sqrt(p0 / p1)[:, None, None, None].astype(np.float32)
        if not (np.isfinite(h1.real).all() and np.isfinite(h1.imag).all()):
            raise FloatingPointError(f"non-finite block {start}:{end}")
        output[start:end] = h1.astype(np.complex64)
        output.flush()

        cell1 = (np.abs(ad1) ** 2).astype(np.float32).sum(axis=(1, 4), dtype=np.float64).astype(np.float32)
        support1 = cell1 / np.maximum(cell1.sum(axis=(1, 2, 3), keepdims=True), TINY)
        diag["power_factor"].append((p1 / p0).astype(np.float32))
        diag["support_l1_actual"].append(np.sum(np.abs(support1 - support[start:end]), axis=(1, 2, 3)).astype(np.float32))
        gflat = gain_cell.reshape(end - start, -1)
        diag["gain_min"].append(gflat.min(axis=1).astype(np.float32))
        diag["gain_p10"].append(np.quantile(gflat, 0.10, axis=1).astype(np.float32))
        diag["gain_p90"].append(np.quantile(gflat, 0.90, axis=1).astype(np.float32))
        diag["gain_max"].append(gflat.max(axis=1).astype(np.float32))
        diag["direction_update_energy"].append((direction_energy / np.maximum(total0, TINY)).astype(np.float32))
        if end == len(source) or end % 100 == 0:
            print(f"v232 applied {name} {end}/{len(source)}", flush=True)

    return {key: np.concatenate(value).astype(np.float32) for key, value in diag.items()}


def variant_specs() -> list[dict[str, object]]:
    return [
        {
            "name": "raw_support_light",
            "mode": "raw",
            "alpha": {"clear": 0.0, "light": 0.02, "medium": 0.10, "heavy": 0.20},
            "direction_scale": 0.0,
            "gain_clip_scale": 0.92,
        },
        {
            "name": "raw_support_moderate",
            "mode": "raw",
            "alpha": {"clear": 0.0, "light": 0.03, "medium": 0.18, "heavy": 0.34},
            "direction_scale": 0.0,
            "gain_clip_scale": 1.00,
        },
        {
            "name": "raw_support_aggressive",
            "mode": "raw",
            "alpha": {"clear": 0.0, "light": 0.04, "medium": 0.28, "heavy": 0.52},
            "direction_scale": 0.0,
            "gain_clip_scale": 1.08,
        },
        {
            "name": "raw_direction_moderate",
            "mode": "raw",
            "alpha": {"clear": 0.0, "light": 0.03, "medium": 0.18, "heavy": 0.34},
            "direction_scale": 0.75,
            "gain_clip_scale": 1.00,
        },
        {
            "name": "derot_support_moderate",
            "mode": "derot",
            "alpha": {"clear": 0.0, "light": 0.03, "medium": 0.18, "heavy": 0.34},
            "direction_scale": 0.0,
            "gain_clip_scale": 1.00,
        },
        {
            "name": "derot_direction_aggressive",
            "mode": "derot",
            "alpha": {"clear": 0.0, "light": 0.04, "medium": 0.26, "heavy": 0.48},
            "direction_scale": 1.00,
            "gain_clip_scale": 1.06,
        },
    ]


def main() -> None:
    for path in [BASE, GEOM_CACHE]:
        if not path.is_file():
            raise FileNotFoundError(path)
    pos_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    geom = np.load(GEOM_CACHE)
    blocked = geom["test_blocked"].astype(np.float32)
    regimes = regime_of(blocked)

    specs = variant_specs()
    per_mode: dict[str, dict[str, object]] = {}
    variants: list[dict[str, object]] = []
    for mode in sorted({str(spec["mode"]) for spec in specs}):
        support, peaks, energy, vectors = extract_support_and_peaks(mode, pos_test)
        idx, weights = neighbor_graph(pos_test, blocked, regimes, k=22 if mode == "raw" else 18, scale=15.5 if mode == "raw" else 17.0)
        prior, track = build_track_prior(support, peaks, energy, vectors, idx, weights)
        per_mode[mode] = {
            "support": support,
            "prior": prior,
            "track": track,
            "neighbor_idx": idx,
            "neighbor_weights": weights,
            "diagnostics": {
                "support_prior_l1": summarize(track["prior_l1_to_support"]),
                "support_prior_cos": summarize(track["prior_cos"]),
                "track_conf": summarize(track["track_conf"]),
                "stable_track_count": summarize(track["stable_track_count"]),
                "by_regime": by_regime_summary(
                    regimes,
                    {
                        "track_conf": track["track_conf"],
                        "stable_track_count": track["stable_track_count"],
                        "stable_energy_fraction": track["stable_energy_fraction"],
                        "mean_persistence": track["mean_persistence"],
                        "mean_shift_abs": track["mean_shift_abs"],
                        "prior_l1_to_support": track["prior_l1_to_support"],
                    },
                ),
            },
        }

    action_payload: dict[str, np.ndarray] = {"regimes": regimes.astype("U8"), "blocked": blocked.astype(np.float32)}

    for spec in specs:
        name = str(spec["name"])
        mode = str(spec["mode"])
        alpha_cfg = spec["alpha"]  # type: ignore[assignment]
        data = per_mode[mode]
        support = data["support"]  # type: ignore[assignment]
        prior = data["prior"]  # type: ignore[assignment]
        track = data["track"]  # type: ignore[assignment]
        target, alpha = target_from_prior(
            support,  # type: ignore[arg-type]
            prior,  # type: ignore[arg-type]
            regimes,
            track["track_conf"],  # type: ignore[index]
            alpha_cfg,  # type: ignore[arg-type]
        )
        diag = apply_track_variant(
            name,
            mode,
            pos_test,
            support,  # type: ignore[arg-type]
            target,
            track,  # type: ignore[arg-type]
            direction_scale=float(spec["direction_scale"]),
            gain_clip_scale=float(spec["gain_clip_scale"]),
        )
        out_path = OUT_DIR / f"v232_{name}_Round1_Test_Channel.npy"
        score = v208.score_prediction(out_path)
        variant = {
            "name": name,
            "mode": mode,
            "output": PATHS.relative(out_path),
            "alpha": alpha_cfg,
            "direction_scale": float(spec["direction_scale"]),
            "gain_clip_scale": float(spec["gain_clip_scale"]),
            "score": score,
            "diagnostics": {
                "alpha": summarize(alpha),
                "target_l1_to_support": summarize(np.sum(np.abs(target - support), axis=(1, 2, 3))),
                "target_cos_to_support": summarize(row_cos(target, support)),
                **{key: summarize(value) for key, value in diag.items()},
                "by_regime": by_regime_summary(
                    regimes,
                    {
                        "alpha": alpha,
                        "target_l1_to_support": np.sum(np.abs(target - support), axis=(1, 2, 3)).astype(np.float32),
                        "support_l1_actual": diag["support_l1_actual"],
                        "direction_update_energy": diag["direction_update_energy"],
                        "power_factor": diag["power_factor"],
                    },
                ),
            },
        }
        variants.append(variant)
        action_payload[f"{name}_alpha"] = alpha.astype(np.float32)
        action_payload[f"{name}_support_l1_actual"] = diag["support_l1_actual"].astype(np.float32)
        print(json.dumps({"name": name, "score": score}, ensure_ascii=False), flush=True)

    best = max(variants, key=lambda item: item["score"]["score_traceback_leaked"])  # type: ignore[index]
    action_payload["raw_track_conf"] = per_mode["raw"]["track"]["track_conf"].astype(np.float32)  # type: ignore[index,union-attr]
    if "derot" in per_mode:
        action_payload["derot_track_conf"] = per_mode["derot"]["track"]["track_conf"].astype(np.float32)  # type: ignore[index,union-attr]
    np.savez_compressed(ACTION_OUT, **action_payload)

    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate"
        if float(best["score"]["score_traceback_leaked"]) < GATE_SCORE  # type: ignore[index]
        else "local_gate_passed_needs_packaging_review",
        "primary_hypothesis": (
            "v226 lacks joint angle-delay component continuity.  Public test-test "
            "coordinates should reveal persistent propagation components whose peaks "
            "move smoothly across neighboring receivers."
        ),
        "compliance_boundary": {
            "generation_uses": [
                "official test coordinates",
                "official geometry cache",
                "v226 strict official-data carrier",
                "test-test component tracks derived from v226 output",
            ],
            "generation_does_not_use": [
                "Sionna generated channel labels",
                "v212/v217 teacher outputs",
                "hidden online labels",
                "synthetic replay truth",
                "raw RT H",
            ],
            "note": "Analysis-only scan; no ZIP unless local v212 gate is exceeded and compliance is reviewed.",
        },
        "local_gate": {
            "required_score_traceback_leaked": GATE_SCORE,
            "best_score_traceback_leaked": float(best["score"]["score_traceback_leaked"]),  # type: ignore[index]
            "gap_to_gate": float(float(best["score"]["score_traceback_leaked"]) - GATE_SCORE),  # type: ignore[index]
        },
        "base": {
            "path": PATHS.relative(BASE),
            "known_v226_score_traceback_leaked": 0.5875098194765852,
        },
        "mode_diagnostics": {
            mode: data["diagnostics"] for mode, data in per_mode.items()
        },
        "variants": variants,
        "best": best,
        "outputs": {
            "actions": PATHS.relative(ACTION_OUT),
            "report": PATHS.relative(REPORT),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
