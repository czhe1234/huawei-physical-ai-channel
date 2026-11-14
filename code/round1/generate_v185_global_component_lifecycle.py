#!/usr/bin/env python3
"""Generate v185: global component lifecycle support correction.

v178-v181 proved that local test-test component tracks contain useful
propagation information, but the one-hop/two-hop track chain is saturated.
v184 then showed that post-track weak-tail denoising is still positive but
also near saturation.

v185 changes the information source: instead of only asking whether a current
peak is locally persistent, it clusters v184 top AD components across the
whole test graph.  A cluster is treated as a propagation band with a lifecycle:
birth, stable presence, or death.  Medium/heavy points are nudged toward local
evidence from those global bands:

* components supported by nearby points and a global band can be rescued;
* weak components with little local/global support naturally decay;
* point total power is preserved;
* no absolute phase is transported.
"""

from __future__ import annotations

import json
import math
import zipfile
from dataclasses import dataclass

import numpy as np

import generate_v178_test_track_complex_consensus as track
from generate_v87_easy_field_projection import MH, MP, MV, N_ANT, S_SC, row_cos, summarize
from generate_v159_frontier_consensus_support_soup import REGIMES, digest
from generate_v169_power_scaled_tail_denoising import by_regime_summary
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION = "v185_global_component_lifecycle"
OUT_DIR = PATHS.cache_dir(185)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(185)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v185_Round1_Test_Channel.npy"
SUPPORT_OUT = OUT_DIR / "v185_v184_collapsed_support.npy"
PEAKS_OUT = OUT_DIR / "v185_v184_top_peaks.npz"
TARGET_OUT = OUT_DIR / "v185_global_component_lifecycle_target.npy"
ACTION_OUT = OUT_DIR / "v185_global_component_lifecycle_actions.npz"
REPORT = REPORT_DIR / "v185_global_component_lifecycle_generate.json"
ZIP_OUT = PATHS.round1_versions / "v185_global_component_lifecycle_candidate_pending.zip"

BASE = PATHS.cache_dir(184) / "v184_Round1_Test_Channel.npy"
V183_BASE = PATHS.cache_dir(183) / "v183_Round1_Test_Channel.npy"
V181_BASE = PATHS.cache_dir(181) / "v181_Round1_Test_Channel.npy"
V166_PREDS = PATHS.shared_features / "v166_path_family_predictability_predictions.npz"

FLOOR = 1e-12
PEAK_COUNT = 96
TRACK_PEAKS = 48
NEIGHBOR_PEAKS = 40
PEAK_CANDIDATES = 4096
MAX_LIFECYCLE_CELLS = 48
DIRECT_K = 28
EXTENDED_K = 84


def configure_track_module() -> None:
    """Reuse v178 AD extraction/projection utilities with v185 paths."""
    track.BASE = BASE
    track.OUTPUT = OUTPUT
    track.SUPPORT_OUT = SUPPORT_OUT
    track.PEAKS_OUT = PEAKS_OUT
    track.TARGET_OUT = TARGET_OUT
    track.ACTION_OUT = ACTION_OUT
    track.REPORT = REPORT
    track.ZIP_OUT = ZIP_OUT
    track.PEAK_COUNT = PEAK_COUNT
    track.TRACK_PEAKS = TRACK_PEAKS
    track.PEAK_CANDIDATES = PEAK_CANDIDATES
    track.MAX_TRACKS = MAX_LIFECYCLE_CELLS


class DSU:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        parent = self.parent
        while int(parent[x]) != x:
            parent[x] = parent[int(parent[x])]
            x = int(parent[x])
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


@dataclass
class ClusterStats:
    root_of: np.ndarray
    valid: np.ndarray
    root_good: np.ndarray
    root_point_count: np.ndarray
    root_node_count: np.ndarray
    root_mean_energy: np.ndarray
    root_quality: np.ndarray


def circular_abs_delta_matrix(a: np.ndarray, b: np.ndarray, size: int) -> np.ndarray:
    raw = np.abs(a[:, None].astype(np.int16) - b[None, :].astype(np.int16))
    return np.minimum(raw, size - raw).astype(np.float32)


def build_extended_graph(
    pos: np.ndarray,
    blocked: np.ndarray,
    regimes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    direct_idx, direct_w = track.neighbor_graph(pos, blocked, regimes, k=DIRECT_K)
    extended_idx = np.full((len(pos), EXTENDED_K), -1, dtype=np.int32)
    extended_w = np.zeros((len(pos), EXTENDED_K), dtype=np.float32)
    twohop_flag = np.zeros((len(pos), EXTENDED_K), dtype=np.float32)
    direct_mass = np.zeros(len(pos), dtype=np.float32)
    twohop_mass = np.zeros(len(pos), dtype=np.float32)

    for i in range(len(pos)):
        weights: dict[int, float] = {}
        direct_nodes: set[int] = set()
        for slot, nb in enumerate(direct_idx[i]):
            node = int(nb)
            direct_nodes.add(node)
            weights[node] = weights.get(node, 0.0) + 0.70 * float(direct_w[i, slot])

        for slot, nb in enumerate(direct_idx[i]):
            first = int(nb)
            first_w = float(direct_w[i, slot])
            for slot2, nb2 in enumerate(direct_idx[first]):
                node = int(nb2)
                if node == i:
                    continue
                penalty = 0.92 if regimes[node] == regimes[i] else 0.54
                weights[node] = weights.get(node, 0.0) + 0.38 * first_w * float(direct_w[first, slot2]) * penalty

        ordered = sorted(weights.items(), key=lambda item: item[1], reverse=True)[:EXTENDED_K]
        total = sum(value for _, value in ordered)
        if total <= 0:
            continue
        for out_slot, (node, value) in enumerate(ordered):
            w = float(value / total)
            extended_idx[i, out_slot] = int(node)
            extended_w[i, out_slot] = np.float32(w)
            if node in direct_nodes:
                direct_mass[i] += np.float32(w)
            else:
                twohop_flag[i, out_slot] = 1.0
                twohop_mass[i] += np.float32(w)

    return direct_idx, direct_w, extended_idx, extended_w, twohop_flag, direct_mass, twohop_mass


def normalized_peak_vectors(vectors: np.ndarray) -> np.ndarray:
    flat = vectors[:, :TRACK_PEAKS].reshape(len(vectors), TRACK_PEAKS, MP * N_ANT)
    norm = np.sqrt(np.sum(np.abs(flat) ** 2, axis=2, keepdims=True) + 1e-18).astype(np.float32)
    return (flat / norm).astype(np.complex64)


def cluster_global_components(
    peaks: np.ndarray,
    energy: np.ndarray,
    vectors: np.ndarray,
    direct_idx: np.ndarray,
    regimes: np.ndarray,
) -> ClusterStats:
    n = len(peaks)
    valid = (peaks[:, :TRACK_PEAKS, 0] >= 0) & (energy[:, :TRACK_PEAKS] >= 4.0e-5)
    dsu = DSU(n * TRACK_PEAKS)
    unit = normalized_peak_vectors(vectors)

    for i in range(n):
        p_valid = valid[i]
        if not np.any(p_valid):
            continue
        p = peaks[i, :TRACK_PEAKS]
        p_energy = energy[i, :TRACK_PEAKS]
        p_unit = unit[i]
        for nb in direct_idx[i]:
            nb = int(nb)
            if nb <= i:
                continue
            q_valid = valid[nb]
            if not np.any(q_valid):
                continue
            q = peaks[nb, :TRACK_PEAKS]
            q_energy = energy[nb, :TRACK_PEAKS]
            q_unit = unit[nb]

            dh = circular_abs_delta_matrix(p[:, 0], q[:, 0], MH)
            dv = circular_abs_delta_matrix(p[:, 1], q[:, 1], MV)
            dd = np.abs(p[:, 2].astype(np.float32)[:, None] - q[:, 2].astype(np.float32)[None, :])
            mask = (dh <= 1.0) & (dv <= 1.0) & (dd <= 4.0) & p_valid[:, None] & q_valid[None, :]
            if not np.any(mask):
                continue

            ad_score = np.exp(-0.5 * ((dh / 0.92) ** 2 + (dv / 0.72) ** 2 + (dd / 2.35) ** 2)).astype(np.float32)
            dir_cos = np.abs(p_unit @ np.conj(q_unit).T).astype(np.float32)
            energy_ratio = np.minimum(
                np.sqrt(np.maximum(q_energy[None, :], FLOOR) / np.maximum(p_energy[:, None], FLOOR)),
                np.sqrt(np.maximum(p_energy[:, None], FLOOR) / np.maximum(q_energy[None, :], FLOOR)),
            ).astype(np.float32)
            score = ad_score * (0.70 + 0.30 * dir_cos) * (0.70 + 0.30 * np.clip(energy_ratio, 0.0, 1.0))
            score[~mask] = -1.0
            best_q = np.argmax(score, axis=1)
            best_score = score[np.arange(TRACK_PEAKS), best_q]
            best_p = np.argmax(score, axis=0)
            threshold = 0.66 if regimes[i] != regimes[nb] else 0.58
            for pslot, qslot in enumerate(best_q):
                if best_score[pslot] < threshold:
                    continue
                if int(best_p[int(qslot)]) != pslot:
                    continue
                if dir_cos[pslot, int(qslot)] < 0.38:
                    continue
                dsu.union(i * TRACK_PEAKS + pslot, nb * TRACK_PEAKS + int(qslot))

        if (i + 1) % 100 == 0:
            print(f"global component clustering {i + 1}/{n}", flush=True)

    root_of = np.full((n, TRACK_PEAKS), -1, dtype=np.int32)
    root_node_count = np.zeros(n * TRACK_PEAKS, dtype=np.int32)
    root_energy_sum = np.zeros(n * TRACK_PEAKS, dtype=np.float64)
    root_point_sets: dict[int, set[int]] = {}

    for i in range(n):
        for pslot in range(TRACK_PEAKS):
            if not valid[i, pslot]:
                continue
            root = dsu.find(i * TRACK_PEAKS + pslot)
            root_of[i, pslot] = root
            root_node_count[root] += 1
            root_energy_sum[root] += float(energy[i, pslot])
            root_point_sets.setdefault(root, set()).add(i)

    root_point_count = np.zeros(n * TRACK_PEAKS, dtype=np.int32)
    for root, points in root_point_sets.items():
        root_point_count[root] = len(points)
    root_mean_energy = np.zeros(n * TRACK_PEAKS, dtype=np.float32)
    nonzero = root_node_count > 0
    root_mean_energy[nonzero] = (root_energy_sum[nonzero] / np.maximum(root_node_count[nonzero], 1)).astype(np.float32)

    point_gate = np.clip((root_point_count.astype(np.float32) - 4.0) / 16.0, 0.0, 1.0)
    node_gate = np.clip((root_node_count.astype(np.float32) - 4.0) / 18.0, 0.0, 1.0)
    energy_gate = np.clip((np.log10(np.maximum(root_mean_energy, 1e-8)) + 4.20) / 1.10, 0.0, 1.0)
    root_quality = (0.45 * point_gate + 0.35 * node_gate + 0.20 * energy_gate).astype(np.float32)
    root_good = (root_point_count >= 6) & (root_node_count >= 6) & (root_quality >= 0.26)

    return ClusterStats(
        root_of=root_of,
        valid=valid,
        root_good=root_good,
        root_point_count=root_point_count,
        root_node_count=root_node_count,
        root_mean_energy=root_mean_energy,
        root_quality=root_quality,
    )


def circular_mean_bin(values: list[int], weights: list[float], size: int) -> int:
    if not values:
        return 0
    angles = 2.0 * math.pi * np.asarray(values, dtype=np.float64) / float(size)
    w = np.asarray(weights, dtype=np.float64)
    z = np.sum(w * np.exp(1j * angles))
    if abs(z) < 1e-12:
        return int(values[0])
    value = int(round((math.atan2(z.imag, z.real) % (2.0 * math.pi)) * size / (2.0 * math.pi))) % size
    return value


def local_support_peak(support: np.ndarray, h0: int, v0: int, d0: int) -> float:
    best = 0.0
    for dh in (-1, 0, 1):
        h = (h0 + dh) % MH
        for dv in (-1, 0, 1):
            v = v0 + dv
            if v < 0 or v >= MV:
                continue
            for dd in range(-4, 5):
                d = d0 + dd
                if 0 <= d < S_SC:
                    best = max(best, float(support[h, v, d]))
    return best


def build_lifecycle_target(
    support: np.ndarray,
    peaks: np.ndarray,
    energy: np.ndarray,
    extended_idx: np.ndarray,
    extended_w: np.ndarray,
    twohop_flag: np.ndarray,
    regimes: np.ndarray,
    top1_two: np.ndarray,
    clusters: ClusterStats,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    n = len(support)
    target = np.empty_like(support, dtype=np.float32)
    track_cells = np.full((n, MAX_LIFECYCLE_CELLS, 3), -1, dtype=np.int16)
    track_dirs = np.zeros((n, MAX_LIFECYCLE_CELLS, MP, N_ANT), dtype=np.complex64)
    track_weights = np.zeros((n, MAX_LIFECYCLE_CELLS), dtype=np.float32)

    lifecycle_conf = np.zeros(n, dtype=np.float32)
    lifecycle_alpha = np.zeros(n, dtype=np.float32)
    rescue_count = np.zeros(n, dtype=np.float32)
    stable_count = np.zeros(n, dtype=np.float32)
    decay_count = np.zeros(n, dtype=np.float32)
    neighbor_band_count = np.zeros(n, dtype=np.float32)
    local_presence = np.zeros(n, dtype=np.float32)
    birth_mass = np.zeros(n, dtype=np.float32)
    stable_mass = np.zeros(n, dtype=np.float32)
    decay_mass = np.zeros(n, dtype=np.float32)
    target_support_l1 = np.zeros(n, dtype=np.float32)
    target_support_cos = np.ones(n, dtype=np.float32)

    for i in range(n):
        current_roots = {
            int(clusters.root_of[i, pslot])
            for pslot in range(TRACK_PEAKS)
            if clusters.root_of[i, pslot] >= 0 and clusters.root_good[int(clusters.root_of[i, pslot])]
        }
        field = (0.72 * support[i]).astype(np.float32)
        root_acc: dict[int, dict[str, object]] = {}

        for nslot, nb in enumerate(extended_idx[i]):
            if nb < 0:
                break
            nb = int(nb)
            hop_scale = 0.76 if float(twohop_flag[i, nslot]) > 0.5 else 1.0
            base_w = float(extended_w[i, nslot]) * hop_scale
            if base_w <= 0:
                continue
            seen: dict[int, tuple[int, float]] = {}
            for qslot in range(NEIGHBOR_PEAKS):
                root = int(clusters.root_of[nb, qslot])
                if root < 0 or not clusters.root_good[root]:
                    continue
                e = float(energy[nb, qslot])
                if e < 2.5e-5:
                    continue
                if root not in seen or e > seen[root][1]:
                    seen[root] = (qslot, e)
            for root, (qslot, e) in seen.items():
                quality = float(clusters.root_quality[root])
                w = base_w * (0.72 + 0.28 * quality)
                acc = root_acc.setdefault(
                    root,
                    {
                        "weight": 0.0,
                        "energy": 0.0,
                        "count": 0,
                        "direct": 0,
                        "h": [],
                        "v": [],
                        "d": [],
                        "coord_w": [],
                    },
                )
                acc["weight"] = float(acc["weight"]) + w
                acc["energy"] = float(acc["energy"]) + w * e
                acc["count"] = int(acc["count"]) + 1
                if float(twohop_flag[i, nslot]) <= 0.5:
                    acc["direct"] = int(acc["direct"]) + 1
                h, v, d = peaks[nb, qslot]
                cast_h: list[int] = acc["h"]  # type: ignore[assignment]
                cast_v: list[int] = acc["v"]  # type: ignore[assignment]
                cast_d: list[int] = acc["d"]  # type: ignore[assignment]
                cast_w: list[float] = acc["coord_w"]  # type: ignore[assignment]
                cast_h.append(int(h))
                cast_v.append(int(v))
                cast_d.append(int(d))
                cast_w.append(w * max(e, 1e-8))

        slot = 0
        evidence_sum = 0.0
        for root, acc in sorted(root_acc.items(), key=lambda item: float(item[1]["weight"]), reverse=True)[:MAX_LIFECYCLE_CELLS]:
            weight = float(acc["weight"])
            count = int(acc["count"])
            direct = int(acc["direct"])
            if count < 3 or direct < 1 or weight < 0.045:
                continue
            h_values: list[int] = acc["h"]  # type: ignore[assignment]
            v_values: list[int] = acc["v"]  # type: ignore[assignment]
            d_values: list[int] = acc["d"]  # type: ignore[assignment]
            coord_w: list[float] = acc["coord_w"]  # type: ignore[assignment]
            h0 = circular_mean_bin(h_values, coord_w, MH)
            v0 = circular_mean_bin(v_values, coord_w, MV)
            d0 = int(np.clip(round(float(np.average(np.asarray(d_values), weights=np.asarray(coord_w)))), 0, S_SC - 1))
            consensus_energy = float(acc["energy"]) / max(weight, 1e-12)
            local_peak = local_support_peak(support[i], h0, v0, d0)
            deficit = float(np.clip((consensus_energy - local_peak) / max(consensus_energy, 1e-8), 0.0, 1.0))
            present = root in current_roots or local_peak >= 0.58 * consensus_energy
            quality = float(clusters.root_quality[root])
            point_count = float(clusters.root_point_count[root])
            evidence = float(np.clip(0.55 * weight + 0.25 * min(1.0, count / 8.0) + 0.20 * quality, 0.0, 1.0))

            if present:
                mass = consensus_energy * (0.30 + 0.62 * evidence) * (0.82 + 0.18 * min(1.0, point_count / 24.0))
                stable_count[i] += 1.0
                stable_mass[i] += np.float32(mass)
            elif deficit > 0.24 and weight >= 0.070 and count >= 4:
                mass = consensus_energy * (0.20 + 0.92 * evidence) * deficit
                rescue_count[i] += 1.0
                birth_mass[i] += np.float32(mass)
            else:
                continue

            track.add_peak_kernel(field, h0, v0, d0, mass, h_sigma=0.92, v_sigma=0.76, d_sigma=2.50)
            evidence_sum += evidence
            if slot < MAX_LIFECYCLE_CELLS:
                track_cells[i, slot] = (h0, v0, d0)
                # Direction weights remain zero: v185 tests lifecycle support,
                # not absolute phase or neighbor phase transport.
                slot += 1

        supported_roots = set(root_acc)
        for pslot in range(12, TRACK_PEAKS):
            root = int(clusters.root_of[i, pslot])
            if root < 0:
                continue
            e = float(energy[i, pslot])
            if e <= 0:
                continue
            if root not in supported_roots or not clusters.root_good[root]:
                decay_count[i] += 1.0
                decay_mass[i] += np.float32(e)

        prior = track.normalize_support(np.maximum(field, FLOOR)[None])[0]
        target_support_l1[i] = np.float32(np.sum(np.abs(prior - support[i])))
        target_support_cos[i] = np.float32(row_cos(prior[None], support[i : i + 1])[0])
        neighbor_band_count[i] = np.float32(len(root_acc))
        local_presence[i] = np.float32(min(1.0, evidence_sum / 5.0))

        regime = str(regimes[i])
        if regime == "heavy":
            scale = 0.68
        elif regime == "medium":
            scale = 0.34
        elif regime == "light":
            scale = 0.0
        else:
            scale = 0.0
        two_keep = 1.0 - 0.12 * float(np.clip((top1_two[i] - 0.55) / 0.35, 0.0, 1.0))
        conf = float(
            np.clip(
                0.36 * min(1.0, rescue_count[i] / 4.0)
                + 0.28 * min(1.0, stable_count[i] / 9.0)
                + 0.20 * local_presence[i]
                + 0.16 * min(1.0, decay_count[i] / 14.0),
                0.0,
                1.0,
            )
        )
        lifecycle_conf[i] = np.float32(conf)
        alpha = scale * (0.32 + 0.68 * conf) * two_keep
        # If the prior is too far from the base, damp rather than clipping cells
        # after projection.  This prevents a few dense bands from hijacking heavy
        # points with already calibrated support.
        alpha *= float(np.clip((0.205 - target_support_l1[i]) / 0.120, 0.45, 1.0))
        lifecycle_alpha[i] = np.float32(np.clip(alpha, 0.0, scale))
        target[i] = track.normalize_support(((1.0 - lifecycle_alpha[i]) * support[i] + lifecycle_alpha[i] * prior)[None])[0]

        if (i + 1) % 100 == 0:
            print(f"v185 lifecycle target {i + 1}/{n}", flush=True)

    diag = {
        "track_cells": track_cells,
        "track_dirs": track_dirs,
        "track_weights": track_weights,
        "lifecycle_conf": lifecycle_conf,
        "lifecycle_alpha": lifecycle_alpha,
        "rescue_count": rescue_count,
        "stable_count": stable_count,
        "decay_count": decay_count,
        "neighbor_band_count": neighbor_band_count,
        "local_presence": local_presence,
        "birth_mass": birth_mass,
        "stable_mass": stable_mass,
        "decay_mass": decay_mass,
        "target_support_l1": np.sum(np.abs(target - support), axis=(1, 2, 3)).astype(np.float32),
        "raw_prior_support_l1": target_support_l1,
        "raw_prior_support_cos": target_support_cos,
        "target_support_cos": row_cos(target, support).astype(np.float32),
    }
    return track.normalize_support(target), diag


def generate() -> dict[str, object]:
    configure_track_module()
    pos_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    blocked = np.load(PATHS.shared_features / "height_geometry_cache.npz")["test_blocked"].astype(np.float32)
    regimes = regime_of(blocked)
    top1_two = np.load(V166_PREDS)["test_top1_two"].astype(np.float32)

    support, peaks, energy, vectors = track.extract_v173_support_and_peaks(pos_test)
    direct_idx, direct_w, extended_idx, extended_w, twohop_flag, direct_mass, twohop_mass = build_extended_graph(
        pos_test,
        blocked,
        regimes,
    )
    clusters = cluster_global_components(peaks, energy, vectors, direct_idx, regimes)
    target, lifecycle = build_lifecycle_target(
        support,
        peaks,
        energy,
        extended_idx,
        extended_w,
        twohop_flag,
        regimes,
        top1_two,
        clusters,
    )
    np.save(TARGET_OUT, target)
    apply_diag = track.apply_track_target(
        pos_test,
        target,
        lifecycle["track_cells"],
        lifecycle["track_dirs"],
        lifecycle["track_weights"],
    )

    good_roots = np.flatnonzero(clusters.root_good)
    np.savez_compressed(
        ACTION_OUT,
        regimes=regimes.astype("U8"),
        top1_two=top1_two,
        direct_neighbor_idx=direct_idx,
        direct_neighbor_weights=direct_w,
        extended_neighbor_idx=extended_idx,
        extended_neighbor_weights=extended_w,
        twohop_flag=twohop_flag,
        direct_mass=direct_mass,
        twohop_mass=twohop_mass,
        root_of=clusters.root_of,
        valid_peak=clusters.valid,
        good_roots=good_roots.astype(np.int32),
        root_point_count=clusters.root_point_count[good_roots].astype(np.int32),
        root_node_count=clusters.root_node_count[good_roots].astype(np.int32),
        root_mean_energy=clusters.root_mean_energy[good_roots].astype(np.float32),
        root_quality=clusters.root_quality[good_roots].astype(np.float32),
        **lifecycle,
        **apply_diag,
    )

    comparison_to_v184 = track.compare_output_to(OUTPUT, BASE, regimes)
    comparison_to_v183 = track.compare_output_to(OUTPUT, V183_BASE, regimes)
    comparison_to_v181 = track.compare_output_to(OUTPUT, V181_BASE, regimes)
    if comparison_to_v184["shape"] != [500, 256, 4, 192] or comparison_to_v184["dtype"] != "complex64":
        raise ValueError(f"invalid output layout: {comparison_to_v184['shape']} {comparison_to_v184['dtype']}")
    if not comparison_to_v184["finite"]:
        raise ValueError("v185 output contains non-finite values")

    root_summary = {
        "valid_peak_count": int(np.sum(clusters.valid)),
        "good_root_count": int(len(good_roots)),
        "good_root_point_count": summarize(clusters.root_point_count[good_roots].astype(np.float32)) if len(good_roots) else {},
        "good_root_node_count": summarize(clusters.root_node_count[good_roots].astype(np.float32)) if len(good_roots) else {},
        "good_root_quality": summarize(clusters.root_quality[good_roots].astype(np.float32)) if len(good_roots) else {},
    }

    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "v178-v181 local component tracks were useful but saturated. v185 "
            "uses a different signal: global component clusters across the test "
            "graph. Those clusters define propagation-band lifecycles, allowing "
            "birth/rescue of underpowered components and decay of unsupported "
            "weak components."
        ),
        "new_information_used": {
            "current_best": "v184 online 0.504349",
            "global_component_lifecycle": "DSU clusters of v184 top AD peaks over the whole test graph",
            "birth_rescue": "nearby good global bands can add support even when the current point lacks a strong peak",
            "death_decay": "weak current peaks unsupported by local/global bands decay through the lifecycle prior",
            "guardrails": "preserve point total power; no absolute phase transport; clear unchanged; top1_two dampens over-action",
        },
        "parameters": {
            "base": PATHS.relative(BASE),
            "peak_count": PEAK_COUNT,
            "track_peaks": TRACK_PEAKS,
            "neighbor_peaks": NEIGHBOR_PEAKS,
            "direct_k": DIRECT_K,
            "extended_k": EXTENDED_K,
            "cluster_match_window": {"h": 1, "v": 1, "delay": 4},
            "cluster_threshold": "0.58 same-regime, 0.66 cross-regime, mutual-best, direction_cos>=0.38",
            "good_cluster": "point_count>=6, node_count>=6, quality>=0.26",
            "alpha_scale": {"clear": 0.0, "light": 0.0, "medium": 0.34, "heavy": 0.68},
            "support_gain_clip": "inherited v178 apply: all cells [0.22,4.25], top256 [0.54,2.05], top64 [0.70,1.46]",
            "direction_policy": "track_weights all zero; support-only lifecycle correction",
            "power_policy": "preserve v184 per-point total power",
        },
        "diagnostics": {
            "clusters": root_summary,
            "overall": {
                "top1_two": summarize(top1_two),
                "direct_mass": summarize(direct_mass),
                "twohop_mass": summarize(twohop_mass),
                **{
                    key: summarize(value)
                    for key, value in lifecycle.items()
                    if isinstance(value, np.ndarray)
                    and value.ndim == 1
                    and np.issubdtype(value.dtype, np.number)
                },
                **{key: summarize(value) for key, value in apply_diag.items()},
            },
            "by_regime": by_regime_summary(
                regimes,
                {
                    "lifecycle_conf": lifecycle["lifecycle_conf"],
                    "lifecycle_alpha": lifecycle["lifecycle_alpha"],
                    "rescue_count": lifecycle["rescue_count"],
                    "stable_count": lifecycle["stable_count"],
                    "decay_count": lifecycle["decay_count"],
                    "neighbor_band_count": lifecycle["neighbor_band_count"],
                    "birth_mass": lifecycle["birth_mass"],
                    "stable_mass": lifecycle["stable_mass"],
                    "decay_mass": lifecycle["decay_mass"],
                    "target_support_l1": lifecycle["target_support_l1"],
                    "support_l1_actual": apply_diag["support_l1_actual"],
                    "actual_power_factor": apply_diag["actual_power_factor"],
                },
            ),
        },
        "comparison_to_v184": comparison_to_v184,
        "comparison_to_v183": comparison_to_v183,
        "comparison_to_v181": comparison_to_v181,
        "preserved": {
            "base_complex": "v184",
            "point_total_power": "v184",
            "absolute_phase": "not transported",
            "cell_internal_direction": "not changed",
            "clear": "no lifecycle action",
            "randomness": "none",
        },
        "changed": {
            "support": "medium/heavy support moves toward global component lifecycle prior",
            "birth": "neighbor-supported global bands can rescue weak/missing current cells",
            "death": "unsupported weak cells are not restored by the lifecycle prior",
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
            PATHS.code / "generate_v185_global_component_lifecycle.py",
            PATHS.code / "generate_v178_test_track_complex_consensus.py",
            PATHS.code / "generate_v184_validated_post_denoising_overdrive.py",
            PATHS.code / "generate_v183_track_protected_post_denoising.py",
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
