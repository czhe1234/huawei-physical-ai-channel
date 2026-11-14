#!/usr/bin/env python3
"""Generate v186: lifecycle-quality bifurcation overdrive on v185.

v185 proved that global component lifecycle is a positive information source:
clusters of AD peaks across the whole test graph act like propagation bands
with birth/stable/death states.  v186 keeps that source but makes the action
more discriminative:

* rerun lifecycle extraction on the v185 output itself;
* strengthen birth/rescue only when cluster quality, direct evidence and
  regime agree;
* damp points whose prior is far from the current support, instead of simply
  increasing alpha everywhere;
* preserve per-point total power and avoid absolute phase transport.
"""

from __future__ import annotations

import json
import math
import zipfile

import numpy as np

import generate_v178_test_track_complex_consensus as track
import generate_v185_global_component_lifecycle as life
from generate_v87_easy_field_projection import MH, MP, MV, N_ANT, S_SC, row_cos, summarize
from generate_v159_frontier_consensus_support_soup import REGIMES, digest
from generate_v169_power_scaled_tail_denoising import by_regime_summary
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION = "v186_lifecycle_quality_overdrive"
OUT_DIR = PATHS.cache_dir(186)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(186)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v186_Round1_Test_Channel.npy"
SUPPORT_OUT = OUT_DIR / "v186_v185_collapsed_support.npy"
PEAKS_OUT = OUT_DIR / "v186_v185_top_peaks.npz"
TARGET_OUT = OUT_DIR / "v186_lifecycle_quality_target.npy"
ACTION_OUT = OUT_DIR / "v186_lifecycle_quality_actions.npz"
REPORT = REPORT_DIR / "v186_lifecycle_quality_overdrive_generate.json"
ZIP_OUT = PATHS.round1_versions / "v186_lifecycle_quality_overdrive_candidate_pending.zip"

BASE = PATHS.cache_dir(185) / "v185_Round1_Test_Channel.npy"
V184_BASE = PATHS.cache_dir(184) / "v184_Round1_Test_Channel.npy"
V183_BASE = PATHS.cache_dir(183) / "v183_Round1_Test_Channel.npy"
V185_ACTIONS = PATHS.cache_dir(185) / "v185_global_component_lifecycle_actions.npz"
V166_PREDS = PATHS.shared_features / "v166_path_family_predictability_predictions.npz"

FLOOR = 1e-12
PEAK_COUNT = 112
TRACK_PEAKS = 56
NEIGHBOR_PEAKS = 48
PEAK_CANDIDATES = 4608
MAX_LIFECYCLE_CELLS = 56
DIRECT_K = 32
EXTENDED_K = 96


def configure_modules() -> None:
    """Point reusable v178/v185 helpers at v186 paths and constants."""
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

    life.TRACK_PEAKS = TRACK_PEAKS
    life.NEIGHBOR_PEAKS = NEIGHBOR_PEAKS
    life.DIRECT_K = DIRECT_K
    life.EXTENDED_K = EXTENDED_K
    life.MAX_LIFECYCLE_CELLS = MAX_LIFECYCLE_CELLS


def weighted_delay_average(values: list[int], weights: list[float]) -> int:
    if not values:
        return 0
    return int(np.clip(round(float(np.average(np.asarray(values), weights=np.asarray(weights)))), 0, S_SC - 1))


def build_quality_target(
    support: np.ndarray,
    peaks: np.ndarray,
    energy: np.ndarray,
    extended_idx: np.ndarray,
    extended_w: np.ndarray,
    twohop_flag: np.ndarray,
    regimes: np.ndarray,
    top1_two: np.ndarray,
    clusters: life.ClusterStats,
    v185_actions: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    n = len(support)
    target = np.empty_like(support, dtype=np.float32)
    track_cells = np.full((n, MAX_LIFECYCLE_CELLS, 3), -1, dtype=np.int16)
    track_dirs = np.zeros((n, MAX_LIFECYCLE_CELLS, MP, N_ANT), dtype=np.complex64)
    track_weights = np.zeros((n, MAX_LIFECYCLE_CELLS), dtype=np.float32)

    lifecycle_conf = np.zeros(n, dtype=np.float32)
    lifecycle_alpha = np.zeros(n, dtype=np.float32)
    high_quality_count = np.zeros(n, dtype=np.float32)
    rescue_count = np.zeros(n, dtype=np.float32)
    stable_count = np.zeros(n, dtype=np.float32)
    decay_count = np.zeros(n, dtype=np.float32)
    direct_birth_count = np.zeros(n, dtype=np.float32)
    route_quality = np.zeros(n, dtype=np.float32)
    mean_root_quality = np.zeros(n, dtype=np.float32)
    birth_mass = np.zeros(n, dtype=np.float32)
    stable_mass = np.zeros(n, dtype=np.float32)
    decay_mass = np.zeros(n, dtype=np.float32)
    raw_prior_support_l1 = np.zeros(n, dtype=np.float32)
    raw_prior_support_cos = np.ones(n, dtype=np.float32)

    prev_alpha = v185_actions["lifecycle_alpha"].astype(np.float32)
    prev_l1 = v185_actions["support_l1_actual"].astype(np.float32)
    prev_gain_max = v185_actions["gain_max"].astype(np.float32)

    for i in range(n):
        current_roots = {
            int(clusters.root_of[i, pslot])
            for pslot in range(TRACK_PEAKS)
            if clusters.root_of[i, pslot] >= 0 and clusters.root_good[int(clusters.root_of[i, pslot])]
        }

        regime = str(regimes[i])
        if regime == "heavy":
            base_keep = 0.62
        elif regime == "medium":
            base_keep = 0.70
        else:
            base_keep = 1.0
        field = (base_keep * support[i]).astype(np.float32)
        root_acc: dict[int, dict[str, object]] = {}

        for nslot, nb in enumerate(extended_idx[i]):
            if nb < 0:
                break
            nb = int(nb)
            hop_scale = 0.66 if float(twohop_flag[i, nslot]) > 0.5 else 1.0
            base_w = float(extended_w[i, nslot]) * hop_scale
            if base_w <= 0:
                continue

            seen: dict[int, tuple[int, float]] = {}
            for qslot in range(NEIGHBOR_PEAKS):
                root = int(clusters.root_of[nb, qslot])
                if root < 0 or not clusters.root_good[root]:
                    continue
                e = float(energy[nb, qslot])
                if e < 2.0e-5:
                    continue
                if root not in seen or e > seen[root][1]:
                    seen[root] = (qslot, e)

            for root, (qslot, e) in seen.items():
                quality = float(clusters.root_quality[root])
                point_count = float(clusters.root_point_count[root])
                node_count = float(clusters.root_node_count[root])
                cluster_gate = float(np.clip(0.58 * quality + 0.24 * min(1.0, point_count / 30.0) + 0.18 * min(1.0, node_count / 34.0), 0.0, 1.0))
                w = base_w * (0.60 + 0.54 * cluster_gate)
                acc = root_acc.setdefault(
                    root,
                    {
                        "weight": 0.0,
                        "energy": 0.0,
                        "count": 0,
                        "direct": 0,
                        "quality": quality,
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
                cast_h: list[int] = acc["h"]  # type: ignore[assignment]
                cast_v: list[int] = acc["v"]  # type: ignore[assignment]
                cast_d: list[int] = acc["d"]  # type: ignore[assignment]
                cast_w: list[float] = acc["coord_w"]  # type: ignore[assignment]
                h, v, d = peaks[nb, qslot]
                cast_h.append(int(h))
                cast_v.append(int(v))
                cast_d.append(int(d))
                cast_w.append(w * max(e, 1e-8) * (0.65 + 0.35 * quality))

        slot = 0
        quality_sum = 0.0
        quality_weight = 0.0
        direct_evidence_sum = 0.0
        evidence_sum = 0.0

        def acc_score(item: tuple[int, dict[str, object]]) -> float:
            root, acc = item
            return float(acc["weight"]) * (0.68 + 0.32 * float(clusters.root_quality[root]))

        for root, acc in sorted(root_acc.items(), key=acc_score, reverse=True)[:MAX_LIFECYCLE_CELLS]:
            weight = float(acc["weight"])
            count = int(acc["count"])
            direct = int(acc["direct"])
            quality = float(clusters.root_quality[root])
            point_count = float(clusters.root_point_count[root])
            if count < 3 or direct < 1 or weight < 0.040:
                continue

            h_values: list[int] = acc["h"]  # type: ignore[assignment]
            v_values: list[int] = acc["v"]  # type: ignore[assignment]
            d_values: list[int] = acc["d"]  # type: ignore[assignment]
            coord_w: list[float] = acc["coord_w"]  # type: ignore[assignment]
            h0 = life.circular_mean_bin(h_values, coord_w, MH)
            v0 = life.circular_mean_bin(v_values, coord_w, MV)
            d0 = weighted_delay_average(d_values, coord_w)
            consensus_energy = float(acc["energy"]) / max(weight, 1e-12)
            local_peak = life.local_support_peak(support[i], h0, v0, d0)
            deficit = float(np.clip((consensus_energy - local_peak) / max(consensus_energy, 1e-8), 0.0, 1.0))
            present = root in current_roots or local_peak >= 0.55 * consensus_energy
            direct_gate = min(1.0, direct / 3.0)
            count_gate = min(1.0, count / 9.0)
            span_gate = min(1.0, point_count / 34.0)
            evidence = float(np.clip(0.36 * weight + 0.22 * count_gate + 0.24 * quality + 0.18 * direct_gate, 0.0, 1.0))
            high_quality = quality >= 0.60 and (direct >= 2 or point_count >= 18)

            if present:
                mass = consensus_energy * (0.38 + 0.82 * evidence) * (0.82 + 0.28 * span_gate)
                if high_quality:
                    mass *= 1.18
                    high_quality_count[i] += 1.0
                stable_count[i] += 1.0
                stable_mass[i] += np.float32(mass)
            elif deficit > 0.18 and weight >= 0.055 and (count >= 4 or direct >= 2):
                # Birth/rescue is where v185 made the most new-information
                # contribution.  Make it stronger only for high-quality bands.
                mass = consensus_energy * (0.24 + 1.22 * evidence) * deficit
                mass *= 1.00 + 0.42 * float(high_quality)
                if direct >= 2:
                    mass *= 1.12
                    direct_birth_count[i] += 1.0
                if high_quality:
                    high_quality_count[i] += 1.0
                rescue_count[i] += 1.0
                birth_mass[i] += np.float32(mass)
            else:
                continue

            if regime == "heavy":
                h_sigma, v_sigma, d_sigma = 0.94, 0.78, 2.65
            else:
                h_sigma, v_sigma, d_sigma = 0.86, 0.72, 2.35
            track.add_peak_kernel(field, h0, v0, d0, mass, h_sigma=h_sigma, v_sigma=v_sigma, d_sigma=d_sigma)
            evidence_sum += evidence
            direct_evidence_sum += evidence * direct_gate
            quality_sum += quality * evidence
            quality_weight += evidence

            if slot < MAX_LIFECYCLE_CELLS:
                track_cells[i, slot] = (h0, v0, d0)
                slot += 1

        supported_roots = set(root_acc)
        for pslot in range(10, TRACK_PEAKS):
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
        raw_l1 = float(np.sum(np.abs(prior - support[i])))
        raw_prior_support_l1[i] = np.float32(raw_l1)
        raw_prior_support_cos[i] = np.float32(row_cos(prior[None], support[i : i + 1])[0])
        route_quality[i] = np.float32(np.clip(direct_evidence_sum / max(evidence_sum, 1e-12), 0.0, 1.0))
        mean_root_quality[i] = np.float32(quality_sum / max(quality_weight, 1e-12)) if quality_weight > 0 else 0.0

        if regime == "heavy":
            scale = 0.78
        elif regime == "medium":
            scale = 0.36
        else:
            scale = 0.0
        birth_gate = min(1.0, rescue_count[i] / 3.0)
        stable_gate = min(1.0, stable_count[i] / 12.0)
        high_gate = min(1.0, high_quality_count[i] / 7.0)
        direct_gate = float(route_quality[i])
        quality_gate = float(mean_root_quality[i])
        conf = float(np.clip(0.34 * birth_gate + 0.24 * stable_gate + 0.20 * high_gate + 0.12 * direct_gate + 0.10 * quality_gate, 0.0, 1.0))
        lifecycle_conf[i] = np.float32(conf)

        two_keep = 1.0 - 0.10 * float(np.clip((top1_two[i] - 0.58) / 0.34, 0.0, 1.0))
        prev_continue = float(np.clip(prev_alpha[i] / (0.78 if regime == "heavy" else 0.36), 0.15, 1.0)) if scale > 0 else 0.0
        overpush_damp = float(np.clip((0.095 - prev_l1[i]) / 0.060, 0.62, 1.0))
        if prev_gain_max[i] >= 4.20 and raw_l1 > 0.13:
            overpush_damp *= 0.84
        raw_damp = float(np.clip((0.310 - raw_l1) / 0.220, 0.55, 1.0))
        alpha = scale * (0.31 + 0.69 * conf) * (0.72 + 0.28 * math.sqrt(prev_continue)) * two_keep * overpush_damp * raw_damp
        if high_quality_count[i] >= 6 and route_quality[i] >= 0.28 and regime == "heavy":
            alpha *= 1.14
        lifecycle_alpha[i] = np.float32(np.clip(alpha, 0.0, scale))
        target[i] = track.normalize_support(((1.0 - lifecycle_alpha[i]) * support[i] + lifecycle_alpha[i] * prior)[None])[0]

        if (i + 1) % 100 == 0:
            print(f"v186 quality lifecycle target {i + 1}/{n}", flush=True)

    diag = {
        "track_cells": track_cells,
        "track_dirs": track_dirs,
        "track_weights": track_weights,
        "lifecycle_conf": lifecycle_conf,
        "lifecycle_alpha": lifecycle_alpha,
        "high_quality_count": high_quality_count,
        "rescue_count": rescue_count,
        "stable_count": stable_count,
        "decay_count": decay_count,
        "direct_birth_count": direct_birth_count,
        "route_quality": route_quality,
        "mean_root_quality": mean_root_quality,
        "birth_mass": birth_mass,
        "stable_mass": stable_mass,
        "decay_mass": decay_mass,
        "raw_prior_support_l1": raw_prior_support_l1,
        "raw_prior_support_cos": raw_prior_support_cos,
        "target_support_l1": np.sum(np.abs(target - support), axis=(1, 2, 3)).astype(np.float32),
        "target_support_cos": row_cos(target, support).astype(np.float32),
    }
    return track.normalize_support(target), diag


def generate() -> dict[str, object]:
    configure_modules()
    pos_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    blocked = np.load(PATHS.shared_features / "height_geometry_cache.npz")["test_blocked"].astype(np.float32)
    regimes = regime_of(blocked)
    top1_two = np.load(V166_PREDS)["test_top1_two"].astype(np.float32)
    v185_actions_npz = np.load(V185_ACTIONS)
    v185_actions = {key: v185_actions_npz[key] for key in v185_actions_npz.files}

    support, peaks, energy, vectors = track.extract_v173_support_and_peaks(pos_test)
    direct_idx, direct_w, extended_idx, extended_w, twohop_flag, direct_mass, twohop_mass = life.build_extended_graph(
        pos_test,
        blocked,
        regimes,
    )
    clusters = life.cluster_global_components(peaks, energy, vectors, direct_idx, regimes)
    target, lifecycle = build_quality_target(
        support,
        peaks,
        energy,
        extended_idx,
        extended_w,
        twohop_flag,
        regimes,
        top1_two,
        clusters,
        v185_actions,
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
        v185_lifecycle_alpha=v185_actions["lifecycle_alpha"].astype(np.float32),
        v185_support_l1_actual=v185_actions["support_l1_actual"].astype(np.float32),
        v185_gain_max=v185_actions["gain_max"].astype(np.float32),
        **lifecycle,
        **apply_diag,
    )

    comparison_to_v185 = track.compare_output_to(OUTPUT, BASE, regimes)
    comparison_to_v184 = track.compare_output_to(OUTPUT, V184_BASE, regimes)
    comparison_to_v183 = track.compare_output_to(OUTPUT, V183_BASE, regimes)
    if comparison_to_v185["shape"] != [500, 256, 4, 192] or comparison_to_v185["dtype"] != "complex64":
        raise ValueError(f"invalid output layout: {comparison_to_v185['shape']} {comparison_to_v185['dtype']}")
    if not comparison_to_v185["finite"]:
        raise ValueError("v186 output contains non-finite values")

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
            "v185 confirmed global component lifecycle as a positive source. "
            "v186 reruns lifecycle on the v185 output and makes a quality-gated "
            "second-stage controller: high-quality birth/rescue can continue, "
            "but points with large previous support movement or weak route "
            "evidence are damped."
        ),
        "new_information_used": {
            "current_best": "v185 online 0.504515",
            "second_stage_lifecycle": "DSU clusters of v185 top AD peaks over the whole test graph",
            "cluster_quality_bifurcation": "quality/direct evidence/high-quality count decide stronger rescue vs damped action",
            "guardrails": "preserve point total power; support-only; no absolute phase transport; clear/light unchanged",
        },
        "parameters": {
            "base": PATHS.relative(BASE),
            "peak_count": PEAK_COUNT,
            "track_peaks": TRACK_PEAKS,
            "neighbor_peaks": NEIGHBOR_PEAKS,
            "direct_k": DIRECT_K,
            "extended_k": EXTENDED_K,
            "cluster_match_window": {"h": 1, "v": 1, "delay": 4},
            "cluster_threshold": "same v185 logic: 0.58 same-regime, 0.66 cross-regime, mutual-best, direction_cos>=0.38",
            "good_cluster": "point_count>=6, node_count>=6, quality>=0.26",
            "alpha_scale": {"clear": 0.0, "light": 0.0, "medium": 0.36, "heavy": 0.78},
            "high_quality_rule": "root_quality>=0.60 and (direct>=2 or root_point_count>=18)",
            "damping": "previous v185 support_l1/gain_max and raw prior L1 damp over-pushed points",
            "support_gain_clip": "inherited v178 apply: all cells [0.22,4.25], top256 [0.54,2.05], top64 [0.70,1.46]",
            "direction_policy": "track_weights all zero; support-only lifecycle correction",
            "power_policy": "preserve v185 per-point total power",
        },
        "diagnostics": {
            "clusters": root_summary,
            "overall": {
                "top1_two": summarize(top1_two),
                "direct_mass": summarize(direct_mass),
                "twohop_mass": summarize(twohop_mass),
                "v185_lifecycle_alpha": summarize(v185_actions["lifecycle_alpha"].astype(np.float32)),
                "v185_support_l1_actual": summarize(v185_actions["support_l1_actual"].astype(np.float32)),
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
                    "high_quality_count": lifecycle["high_quality_count"],
                    "rescue_count": lifecycle["rescue_count"],
                    "stable_count": lifecycle["stable_count"],
                    "direct_birth_count": lifecycle["direct_birth_count"],
                    "route_quality": lifecycle["route_quality"],
                    "mean_root_quality": lifecycle["mean_root_quality"],
                    "target_support_l1": lifecycle["target_support_l1"],
                    "support_l1_actual": apply_diag["support_l1_actual"],
                    "actual_power_factor": apply_diag["actual_power_factor"],
                },
            ),
        },
        "comparison_to_v185": comparison_to_v185,
        "comparison_to_v184": comparison_to_v184,
        "comparison_to_v183": comparison_to_v183,
        "preserved": {
            "base_complex": "v185",
            "point_total_power": "v185",
            "absolute_phase": "not transported",
            "cell_internal_direction": "not changed",
            "clear_light": "no lifecycle action",
            "randomness": "none",
        },
        "changed": {
            "support": "medium/heavy support moves toward quality-gated global lifecycle prior",
            "birth_rescue": "stronger only for high-quality and direct-supported clusters",
            "damping": "points with large previous v185 movement or far raw prior are damped",
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
            PATHS.code / "generate_v186_lifecycle_quality_overdrive.py",
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
