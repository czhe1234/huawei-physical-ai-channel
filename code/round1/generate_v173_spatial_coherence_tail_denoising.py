#!/usr/bin/env python3
"""Generate v173: spatial-coherence AD tail denoising on top of v171.

Hypothesis:
    v171 confirmed that weak full-complex AD coefficients can be denoised, but
    it decides reliability from a single test point.  Sionna propagation is
    spatially continuous: a real weak path should usually have nearby support
    in neighboring test points after angle-delay bin pooling, while released
    label noise or unstable tail speckles should be much less coherent across
    space.

    v173 therefore builds a test-test neighborhood support prior from v171
    itself.  It preserves v171 phase and pol/UE allocation, then only applies a
    positive real AD-cell gain:

    * strong ranked cells are protected;
    * weak cells supported by neighbors are protected or mildly completed;
    * weak cells that are high in one point but absent from neighbors are
      attenuated;
    * medium/heavy receive most of the action, clear/light are capped.
"""

from __future__ import annotations

import json
import zipfile

import numpy as np
from scipy.ndimage import maximum_filter

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
from generate_v159_frontier_consensus_support_soup import REGIMES, compare_outputs, digest
from generate_v169_power_scaled_tail_denoising import by_regime_summary
from generate_v171_full_complex_ad_tail_denoising import support_metrics_from_cell, top_mask
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION = "v173_spatial_coherence_tail_denoising"
OUT_DIR = PATHS.cache_dir(173)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(173)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v173_Round1_Test_Channel.npy"
ACTION_OUT = OUT_DIR / "v173_spatial_coherence_tail_denoising_actions.npz"
SUPPORT_OUT = OUT_DIR / "v173_spatial_coherence_target_support.npy"
REPORT = REPORT_DIR / "v173_spatial_coherence_tail_denoising_generate.json"
ZIP_OUT = PATHS.round1_versions / "v173_spatial_coherence_tail_denoising_candidate_pending.zip"

BASE = PATHS.cache_dir(171) / "v171_Round1_Test_Channel.npy"
V171_ACTIONS = PATHS.cache_dir(171) / "v171_full_complex_ad_tail_denoising_actions.npz"
V170_ACTIONS = PATHS.cache_dir(170) / "v170_adaptive_floor_subtraction_actions.npz"

CELL_COUNT = MH * MV * S_SC
FLOOR = 1e-12


def normalize_support(x: np.ndarray) -> np.ndarray:
    y = np.maximum(np.asarray(x, dtype=np.float32), FLOOR)
    y /= np.maximum(y.sum(axis=(1, 2, 3), keepdims=True), TINY)
    return y.astype(np.float32)


def build_v171_support(pos_test: np.ndarray, batch: int = 20) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    support_cache = OUT_DIR / "v171_collapsed_support.npy"
    metrics_cache = OUT_DIR / "v171_support_metrics.npz"
    if support_cache.is_file() and metrics_cache.is_file():
        support = normalize_support(np.load(support_cache).astype(np.float32))
        data = np.load(metrics_cache)
        return support, {key: data[key].astype(np.float32) for key in data.files}

    source = np.load(BASE, mmap_mode="r")
    beta_s = beta_subcarriers(140.3, 0.10)
    distance = np.linalg.norm(pos_test - BS_POS[None, :], axis=1)
    support = np.empty((source.shape[0], MH, MV, S_SC), dtype=np.float32)
    totals = np.empty(source.shape[0], dtype=np.float32)

    for start in range(0, len(source), batch):
        end = min(start + batch, len(source))
        h = np.asarray(source[start:end], dtype=np.complex64)
        derot = h * np.exp(1j * distance[start:end, None, None, None] * beta_s).astype(np.complex64)
        ad = ad_transform(derot)
        power = (np.abs(ad) ** 2).astype(np.float32)
        cell = power.sum(axis=(1, 4), dtype=np.float64).astype(np.float32)
        total = np.maximum(cell.sum(axis=(1, 2, 3)), TINY).astype(np.float32)
        totals[start:end] = total
        support[start:end] = cell / total[:, None, None, None]
        print(f"v171 support {end}/{len(source)}", flush=True)

    support = normalize_support(support)
    np.save(support_cache, support)
    flat = support.reshape(len(support), -1)
    sorted_flat = np.sort(flat, axis=1)
    entropy = -np.sum(flat * np.log(np.maximum(flat, FLOOR)), axis=1) / np.log(CELL_COUNT)
    metrics = {
        "cell_total_power": totals,
        "top64_fraction": sorted_flat[:, -64:].sum(axis=1).astype(np.float32),
        "top128_fraction": sorted_flat[:, -128:].sum(axis=1).astype(np.float32),
        "top512_fraction": sorted_flat[:, -512:].sum(axis=1).astype(np.float32),
        "tail512_fraction": (sorted_flat[:, :-512].mean(axis=1) * CELL_COUNT).astype(np.float32),
        "tail512_floor": sorted_flat[:, :-512].mean(axis=1).astype(np.float32),
        "entropy": entropy.astype(np.float32),
    }
    np.savez_compressed(metrics_cache, **metrics)
    return support, metrics


def neighbor_graph(pos: np.ndarray, blocked: np.ndarray, regimes: np.ndarray, k: int = 18) -> tuple[np.ndarray, np.ndarray]:
    xy = pos[:, :2].astype(np.float32)
    dxy = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2).astype(np.float32)
    db = np.abs(blocked[:, None] - blocked[None, :]).astype(np.float32)
    regime_mismatch = (regimes[:, None] != regimes[None, :]).astype(np.float32)
    effective = dxy + 0.20 * db + 4.5 * regime_mismatch
    np.fill_diagonal(effective, np.inf)
    idx = np.argpartition(effective, k, axis=1)[:, :k]
    row = np.arange(len(pos))[:, None]
    idx = idx[row, np.argsort(effective[row, idx], axis=1)]
    selected = effective[row, idx]
    weights = np.exp(-selected / 18.0).astype(np.float32)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), TINY)
    return idx.astype(np.int32), weights.astype(np.float32)


def neighbor_evidence(support: np.ndarray, idx: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pooled = maximum_filter(
        support,
        size=(1, 3, 3, 5),
        mode=("nearest", "wrap", "wrap", "nearest"),
    ).astype(np.float32)
    evidence = np.zeros_like(support, dtype=np.float32)
    raw = np.zeros_like(support, dtype=np.float32)
    for n in range(idx.shape[1]):
        w = weights[:, n].reshape(-1, 1, 1, 1)
        neighbor = idx[:, n]
        evidence += w * pooled[neighbor]
        raw += w * support[neighbor]
    # Blend pooled support with exact-bin support: pooled support tolerates
    # small angle/delay drift; raw support avoids over-completing broad halos.
    evidence = normalize_support(0.68 * evidence + 0.32 * raw)
    raw = normalize_support(raw)
    return evidence.astype(np.float32), raw.astype(np.float32)


def build_action(regimes: np.ndarray, metrics: dict[str, np.ndarray], evidence: np.ndarray, support: np.ndarray) -> dict[str, np.ndarray]:
    v171 = np.load(V171_ACTIONS)
    v170 = np.load(V170_ACTIONS)
    base = v171["action"].astype(np.float32)

    support_cos = row_cos(support, evidence).astype(np.float32)
    mismatch_gate = np.clip((1.0 - support_cos - 0.020) / 0.115, 0.0, 1.0).astype(np.float32)
    tail_gate = np.clip((metrics["tail512_fraction"] - 0.010) / 0.090, 0.0, 1.0).astype(np.float32)
    entropy_gate = np.clip((metrics["entropy"] - 0.28) / 0.30, 0.0, 1.0).astype(np.float32)

    scale = np.zeros(len(regimes), dtype=np.float32)
    scale[regimes == "clear"] = 0.12
    scale[regimes == "light"] = 0.18
    scale[regimes == "medium"] = 1.10
    scale[regimes == "heavy"] = 1.42

    top1_two = v170["top1_two"].astype(np.float32)
    two_power = v170["two_power_frac"].astype(np.float32)
    assigned = v170["assigned_power_frac"].astype(np.float32)
    multibounce_protect = (
        1.06
        - 0.22 * np.clip((top1_two - 0.70) / 0.25, 0.0, 1.0)
        - 0.14 * np.clip((two_power - 0.44) / 0.32, 0.0, 1.0)
        - 0.08 * np.clip((assigned - 0.80) / 0.14, 0.0, 1.0)
    ).astype(np.float32)
    multibounce_protect = np.clip(multibounce_protect, 0.64, 1.06)

    action = (
        base
        * scale
        * np.clip(0.42 + 0.58 * mismatch_gate, 0.0, 1.0)
        * np.clip(0.52 + 0.28 * tail_gate + 0.20 * entropy_gate, 0.0, 1.0)
        * multibounce_protect
    ).astype(np.float32)
    action = np.clip(action, 0.0, 1.0)
    action[regimes == "clear"] = np.minimum(action[regimes == "clear"], 0.004)
    action[regimes == "light"] = np.minimum(action[regimes == "light"], 0.015)
    return {
        "v171_action": base,
        "action_scale": scale,
        "support_cos_to_neighbor": support_cos,
        "mismatch_gate": mismatch_gate,
        "tail_gate": tail_gate,
        "entropy_gate": entropy_gate,
        "multibounce_protect": multibounce_protect,
        "action": action,
        "top1_two": top1_two,
        "two_power_frac": two_power,
        "assigned_power_frac": assigned,
    }


def build_target_support(
    support: np.ndarray,
    evidence: np.ndarray,
    actions: dict[str, np.ndarray],
    regimes: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    flat = support.reshape(len(support), -1)
    ev = evidence.reshape(len(support), -1)

    mask128 = top_mask(flat, 128)
    mask512 = top_mask(flat, 512)
    mask2048 = top_mask(flat, 2048)
    mask4096 = top_mask(flat, 4096)

    rank_strength = np.ones(flat.shape, dtype=np.float32)
    rank_strength[mask4096] = 0.82
    rank_strength[mask2048] = 0.55
    rank_strength[mask512] = 0.18
    rank_strength[mask128] = 0.0

    min_keep = np.full(flat.shape, 0.030, dtype=np.float32)
    min_keep[mask4096] = 0.080
    min_keep[mask2048] = 0.200
    min_keep[mask512] = 0.650
    min_keep[mask128] = 1.0

    floor = np.maximum(np.take_along_axis(np.sort(flat, axis=1), np.zeros((len(flat), 1), dtype=np.int64), axis=1), FLOOR)
    floor = np.maximum(floor, FLOOR).astype(np.float32)
    over = np.maximum(flat - 1.18 * ev, 0.0).astype(np.float32)
    under = np.maximum(ev - 1.10 * flat, 0.0).astype(np.float32)
    unsupported = np.clip(over / np.maximum(flat + 6.0 * floor, FLOOR), 0.0, 1.0).astype(np.float32)
    coherent_missing = np.clip(under / np.maximum(ev + 6.0 * floor, FLOOR), 0.0, 1.0).astype(np.float32)

    action = actions["action"].astype(np.float32)[:, None]
    shrink = np.clip(1.18 * action * rank_strength * unsupported, 0.0, 0.96).astype(np.float32)
    add = np.clip(0.32 * action * rank_strength * coherent_missing, 0.0, 0.30).astype(np.float32)

    target_flat = flat * (1.0 - shrink) + add * under
    target_flat = np.maximum(target_flat, flat * min_keep)
    target = normalize_support(target_flat.reshape(support.shape))

    removed_mass = np.sum(np.maximum(flat - target_flat, 0.0), axis=1).astype(np.float32)
    added_mass = np.sum(np.maximum(target_flat - flat, 0.0), axis=1).astype(np.float32)
    target_l1 = np.sum(np.abs(target.reshape(len(target), -1) - flat), axis=1).astype(np.float32)

    lower = np.ones(len(regimes), dtype=np.float32)
    lower[regimes == "clear"] = 0.9995
    lower[regimes == "light"] = 0.9980
    lower[regimes == "medium"] = 0.9780
    lower[regimes == "heavy"] = 0.9560
    power_factor = np.clip(1.0 - 0.34 * removed_mass + 0.06 * added_mass, lower, 1.0).astype(np.float32)

    diagnostics = {
        "removed_mass": removed_mass,
        "added_mass": added_mass,
        "target_support_l1": target_l1,
        "power_factor": power_factor,
        "unsupported_mean": unsupported.mean(axis=1).astype(np.float32),
        "coherent_missing_mean": coherent_missing.mean(axis=1).astype(np.float32),
        "shrink_mean": shrink.mean(axis=1).astype(np.float32),
        "add_mean": add.mean(axis=1).astype(np.float32),
    }
    return target, diagnostics


def apply_target_support(
    pos_test: np.ndarray,
    target: np.ndarray,
    power_factor: np.ndarray,
    batch: int = 20,
) -> dict[str, np.ndarray]:
    beta_s = beta_subcarriers(140.3, 0.10)
    distance = np.linalg.norm(pos_test - BS_POS[None, :], axis=1)
    source = np.load(BASE, mmap_mode="r")
    output = np.lib.format.open_memmap(OUTPUT, mode="w+", dtype=np.complex64, shape=source.shape)

    diagnostics: dict[str, list[np.ndarray]] = {
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
        derot = h0 * np.exp(1j * distance[start:end, None, None, None] * beta_s).astype(np.complex64)
        ad0 = ad_transform(derot)
        cell0 = (np.abs(ad0) ** 2).astype(np.float32).sum(axis=(1, 4), dtype=np.float64).astype(np.float32)
        total0 = np.maximum(cell0.sum(axis=(1, 2, 3)), TINY).astype(np.float32)
        support0 = cell0 / total0[:, None, None, None]
        target_cell = target[start:end] * (total0 * power_factor[start:end])[:, None, None, None]
        gain_cell = np.sqrt(target_cell / np.maximum(cell0, 1e-30)).astype(np.float32)
        gain_cell = np.clip(gain_cell, 0.045, 3.40).astype(np.float32)

        # Keep the strongest current cells very stable even if the neighborhood
        # target slightly disagrees.
        flat = support0.reshape(end - start, -1)
        strong = top_mask(flat, 128).reshape(end - start, MH, MV, S_SC)
        gain_cell[strong] = np.clip(gain_cell[strong], 0.82, 1.22)

        ad1 = ad0 * gain_cell[:, None, :, :, None, :]
        total_after = np.maximum(np.sum(np.abs(ad1) ** 2, axis=(1, 2, 3, 4, 5), dtype=np.float64), TINY).astype(np.float32)
        scale = np.sqrt((total0 * power_factor[start:end]) / total_after).astype(np.float32)
        ad1 *= scale[:, None, None, None, None, None]

        h1 = inverse_ad(ad1).reshape(end - start, 256, 4, 192)
        h1 = h1 * np.exp(-1j * distance[start:end, None, None, None] * beta_s).astype(np.complex64)
        if not (np.isfinite(h1.real).all() and np.isfinite(h1.imag).all()):
            raise FloatingPointError(f"non-finite block {start}:{end}")
        output[start:end] = h1.astype(np.complex64)
        output.flush()

        cell1 = (np.abs(ad1) ** 2).astype(np.float32).sum(axis=(1, 4), dtype=np.float64).astype(np.float32)
        support1 = cell1 / np.maximum(cell1.sum(axis=(1, 2, 3), keepdims=True), TINY)
        diagnostics["actual_power_factor"].append((cell1.sum(axis=(1, 2, 3)) / total0).astype(np.float32))
        diagnostics["support_l1_actual"].append(np.sum(np.abs(support1 - support0), axis=(1, 2, 3)).astype(np.float32))
        gflat = gain_cell.reshape(end - start, -1)
        diagnostics["gain_min"].append(gflat.min(axis=1).astype(np.float32))
        diagnostics["gain_p10"].append(np.quantile(gflat, 0.10, axis=1).astype(np.float32))
        diagnostics["gain_p90"].append(np.quantile(gflat, 0.90, axis=1).astype(np.float32))
        diagnostics["gain_max"].append(gflat.max(axis=1).astype(np.float32))
        print(f"spatial coherence denoised {end}/500", flush=True)

    return {key: np.concatenate(value).astype(np.float32) for key, value in diagnostics.items()}


def generate() -> dict[str, object]:
    pos_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    blocked = np.load(PATHS.shared_features / "height_geometry_cache.npz")["test_blocked"].astype(np.float32)
    regimes = regime_of(blocked)

    support, metrics = build_v171_support(pos_test)
    idx, weights = neighbor_graph(pos_test, blocked, regimes)
    evidence, raw_evidence = neighbor_evidence(support, idx, weights)
    actions = build_action(regimes, metrics, evidence, support)
    target, target_diag = build_target_support(support, evidence, actions, regimes)
    np.save(SUPPORT_OUT, target)

    apply_diag = apply_target_support(pos_test, target, target_diag["power_factor"])

    np.savez_compressed(
        ACTION_OUT,
        regimes=regimes.astype("U8"),
        neighbor_idx=idx,
        neighbor_weights=weights,
        evidence_cos=row_cos(support, evidence).astype(np.float32),
        raw_evidence_cos=row_cos(support, raw_evidence).astype(np.float32),
        **metrics,
        **actions,
        **target_diag,
        **apply_diag,
    )

    comparison_to_v171 = compare_outputs(OUTPUT, BASE, regimes)
    comparison_to_v170 = compare_outputs(OUTPUT, PATHS.cache_dir(170) / "v170_Round1_Test_Channel.npy", regimes)
    comparison_to_v163 = compare_outputs(OUTPUT, PATHS.cache_dir(163) / "v163_Round1_Test_Channel.npy", regimes)
    comparison_to_v140 = compare_outputs(OUTPUT, PATHS.cache_dir(140) / "v140_Round1_Test_Channel.npy", regimes)
    if comparison_to_v171["shape"] != [500, 256, 4, 192] or comparison_to_v171["dtype"] != "complex64":
        raise ValueError(f"invalid output layout: {comparison_to_v171['shape']} {comparison_to_v171['dtype']}")
    if not comparison_to_v171["finite"]:
        raise ValueError("v173 output contains non-finite values")

    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "Real Sionna weak paths should be spatially coherent among nearby "
            "test points after small AD-bin pooling, while noisy weak-tail "
            "speckles should be less repeatable.  v173 uses test-test support "
            "coherence to decide which v171 weak AD cells to attenuate or "
            "mildly complete."
        ),
        "new_information_used": {
            "v171_online": "0.503656 confirms weak full-complex AD tail denoising is useful",
            "test_test_spatial_coherence": "neighbor support voting separates coherent weak paths from isolated tail speckles",
            "sionna_continuity_prior": "propagation components change continuously over nearby receiver positions",
        },
        "parameters": {
            "base": PATHS.relative(BASE),
            "neighbors": "18 nearest test points by XY distance + blocked/regime penalty",
            "ad_pooling": "neighbor support max-filter size 3x3x5 over H,V,delay then weighted average",
            "action": "v171 action scaled by regime, mismatch, tail, entropy and multibounce protection",
            "regime_scale": {"clear": 0.12, "light": 0.18, "medium": 1.10, "heavy": 1.42},
            "action_caps": {"clear": 0.004, "light": 0.015},
            "support_target": "shrink self-only weak cells; mildly add coherent neighbor-supported missing cells",
            "rank_strength": {"top128": 0.0, "top512": 0.18, "top2048": 0.55, "top4096": 0.82, "tail": 1.0},
            "power_factor": "clip(1 - .34*removed_mass + .06*added_mass, regime lower bound, 1)",
            "lower_bounds": {"clear": 0.9995, "light": 0.9980, "medium": 0.9780, "heavy": 0.9560},
            "phase_policy": "preserve v171 complex phase; only positive AD-cell gains are applied",
        },
        "diagnostics": {
            "overall": {
                **{key: summarize(value) for key, value in metrics.items() if value.ndim == 1},
                **{key: summarize(value) for key, value in actions.items() if value.ndim == 1},
                **{key: summarize(value) for key, value in target_diag.items()},
                **{key: summarize(value) for key, value in apply_diag.items()},
            },
            "by_regime": by_regime_summary(
                regimes,
                {
                    "action": actions["action"],
                    "support_cos_to_neighbor": actions["support_cos_to_neighbor"],
                    "mismatch_gate": actions["mismatch_gate"],
                    "removed_mass": target_diag["removed_mass"],
                    "added_mass": target_diag["added_mass"],
                    "target_support_l1": target_diag["target_support_l1"],
                    "power_factor": target_diag["power_factor"],
                    "actual_power_factor": apply_diag["actual_power_factor"],
                    "support_l1_actual": apply_diag["support_l1_actual"],
                },
            ),
        },
        "comparison_to_v171": comparison_to_v171,
        "comparison_to_v170": comparison_to_v170,
        "comparison_to_v163": comparison_to_v163,
        "comparison_to_v140": comparison_to_v140,
        "preserved": {
            "base_complex": "v171",
            "strong_cells": "top128 protected; top512 lightly affected",
            "phase": "no learned phase, only positive AD-cell gains",
            "pol_ue_allocation": "preserved inside each AD cell",
            "randomness": "none",
        },
        "changed": {
            "weak_tail_support": "spatially incoherent weak AD cells attenuated",
            "coherent_missing_tail": "neighbor-supported weak cells mildly completed",
            "total_power": "small bounded shrink from isolated removed tail mass",
        },
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
            "target_support": PATHS.relative(SUPPORT_OUT),
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
            (__file__, "generate_v173_spatial_coherence_tail_denoising.py"),
            (PATHS.code / "generate_v171_full_complex_ad_tail_denoising.py", "generate_v171_full_complex_ad_tail_denoising.py"),
            (PATHS.code / "generate_v170_adaptive_floor_subtraction_denoising.py", "generate_v170_adaptive_floor_subtraction_denoising.py"),
            (PATHS.code / "generate_v169_power_scaled_tail_denoising.py", "generate_v169_power_scaled_tail_denoising.py"),
            (PATHS.code / "analyze_v168_channel_awgn_power_audit.py", "analyze_v168_channel_awgn_power_audit.py"),
            (PATHS.code / "analyze_v166_path_family_predictability.py", "analyze_v166_path_family_predictability.py"),
            (PATHS.code / "generate_v159_frontier_consensus_support_soup.py", "generate_v159_frontier_consensus_support_soup.py"),
            (PATHS.code / "generate_v87_easy_field_projection.py", "generate_v87_easy_field_projection.py"),
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
