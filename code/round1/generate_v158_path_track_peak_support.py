#!/usr/bin/env python3
"""Generate v158: path-track peak support boost on top of v140.

Hypothesis:
    The path-track audit showed that strong joint angle-delay peaks in the
    training truth are spatially persistent among nearby same-regime points.
    v154-v157 failed because they transferred broad residuals from a non-
    equivalent train current and often reversed v140's verified heavy support
    diffusion.  This version uses the new information more narrowly: build a
    test-side support prior only from repeated local truth peaks, then inject it
    into the v140 complex output with a bounded support-only projection.

Preserved:
    v140 complex phase, total power, scalar scale, and pol/UE allocation.
Changed:
    medium/heavy collapsed joint AD support around locally persistent peaks.
"""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from generate_v87_easy_field_projection import (
    ad_transform,
    beta_subcarriers,
    row_cos,
    summarize,
)
from generate_v98_path_ensemble_graph_attention_joint_ad import projection
from powerfield_common import BS_POS, TINY, regime_of
from project_paths import PATHS


VERSION = "v158_path_track_peak_support"
OUT_DIR = PATHS.cache_dir(158)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(158)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v158_Round1_Test_Channel.npy"
TARGET_OUT = OUT_DIR / "v158_path_track_peak_support_target.npy"
PRIOR_OUT = OUT_DIR / "v158_path_track_peak_prior.npy"
WEIGHTS_OUT = OUT_DIR / "v158_path_track_peak_support_weights.npy"
ACTION_OUT = OUT_DIR / "v158_path_track_peak_actions.npy"
REPORT = REPORT_DIR / "v158_path_track_peak_support_generate.json"
ZIP_OUT = PATHS.round1_versions / "v158_path_track_peak_support_candidate_pending.zip"

PEAK_COUNT = 8
PEAK_CANDIDATES = 512
TRACK_GROUPS = 12
FLOOR = 1e-12
REGIMES = ("clear", "light", "medium", "heavy")


@dataclass
class PeakGroup:
    h_sum: float
    v_sum: float
    d_sum: float
    weight: float
    neighbor_mask: int

    def center(self) -> tuple[int, int, int]:
        return (
            int(round(self.h_sum / max(self.weight, TINY))) % 16,
            int(round(self.v_sum / max(self.weight, TINY))) % 8,
            int(round(self.d_sum / max(self.weight, TINY))),
        )

    def neighbor_count(self) -> int:
        return int(self.neighbor_mask.bit_count())


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def normalize_support(x: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=np.float32)
    out = np.maximum(out, FLOOR)
    out /= np.maximum(out.sum(axis=(1, 2, 3), keepdims=True), TINY)
    return out.astype(np.float32)


def circular_delta(a: int, b: int, size: int) -> int:
    delta = abs(int(a) - int(b))
    return int(min(delta, size - delta))


def circular_signed_delta(a: int, b: int, size: int) -> int:
    delta = (int(b) - int(a) + size // 2) % size - size // 2
    return int(delta)


def extract_distinct_peaks(support_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = np.asarray(support_raw.sum(axis=(1, 2, 3)), dtype=np.float64)
    valid = total > TINY
    support = support_raw[valid].astype(np.float32)
    support /= np.maximum(support.sum(axis=(1, 2, 3), keepdims=True), TINY)
    coordinates = np.full((len(support), PEAK_COUNT, 3), -1, dtype=np.int16)
    energy = np.zeros((len(support), PEAK_COUNT), dtype=np.float32)

    for start in range(0, len(support), 32):
        end = min(start + 32, len(support))
        block = support[start:end]
        flat = block.reshape(len(block), -1)
        candidate_count = min(PEAK_CANDIDATES, flat.shape[1])
        candidate = np.argpartition(flat, -candidate_count, axis=1)[:, -candidate_count:]
        candidate_value = np.take_along_axis(flat, candidate, axis=1)
        order = np.argsort(candidate_value, axis=1)[:, ::-1]
        candidate = np.take_along_axis(candidate, order, axis=1)

        for row in range(len(block)):
            accepted: list[tuple[int, int, int]] = []
            for flat_index in candidate[row]:
                h, remainder = divmod(int(flat_index), 8 * 192)
                v, delay = divmod(remainder, 192)
                too_close = any(
                    circular_delta(h, ah, 16) <= 1
                    and circular_delta(v, av, 8) <= 1
                    and abs(delay - ad) <= 2
                    for ah, av, ad in accepted
                )
                if too_close:
                    continue
                slot = len(accepted)
                coordinates[start + row, slot] = (h, v, delay)
                energy[start + row, slot] = flat[row, flat_index]
                accepted.append((h, v, delay))
                if len(accepted) == PEAK_COUNT:
                    break
        print(f"extract peaks {end}/{len(support)}", flush=True)
    return coordinates, energy, valid


def current_collapsed_support(base_path: Path, pos: np.ndarray, batch: int = 20) -> np.ndarray:
    base = np.load(base_path, mmap_mode="r")
    beta_s = beta_subcarriers(140.3, 0.10)
    distance = np.linalg.norm(pos - BS_POS[None, :], axis=1)
    current = np.empty((base.shape[0], 16, 8, 192), dtype=np.float32)
    for start in range(0, base.shape[0], batch):
        end = min(start + batch, base.shape[0])
        h0 = np.asarray(base[start:end], dtype=np.complex64)
        derot = h0 * np.exp(1j * distance[start:end, None, None, None] * beta_s).astype(np.complex64)
        ad = ad_transform(derot)
        p = (np.abs(ad) ** 2).astype(np.float32)
        coll = p.sum(axis=(1, 4))
        current[start:end] = coll / np.maximum(coll.sum(axis=(1, 2, 3), keepdims=True), TINY)
        print(f"current support {end}/{base.shape[0]}", flush=True)
    return normalize_support(current)


def same_regime_neighbors(
    train_pos: np.ndarray,
    train_regimes: np.ndarray,
    train_valid: np.ndarray,
    test_pos: np.ndarray,
    test_regimes: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.empty((len(test_pos), k), dtype=np.int32)
    dist = np.empty((len(test_pos), k), dtype=np.float32)
    xy_train = train_pos[:, :2].astype(np.float64)
    xy_test = test_pos[:, :2].astype(np.float64)
    for i in range(len(test_pos)):
        mask = train_valid & (train_regimes == test_regimes[i])
        if np.sum(mask) < k:
            mask = train_valid
        ids = np.flatnonzero(mask)
        d = np.linalg.norm(xy_train[ids] - xy_test[i], axis=1)
        take = np.argpartition(d, k)[:k]
        take = take[np.argsort(d[take])]
        idx[i] = ids[take]
        dist[i] = d[take].astype(np.float32)
    return idx, dist


def add_peak_kernel(field: np.ndarray, h: int, v: int, d: int, weight: float) -> None:
    for dh in (-2, -1, 0, 1, 2):
        hh = (h + dh) % 16
        wh = math.exp(-0.5 * (dh / 0.95) ** 2)
        for dv in (-1, 0, 1):
            vv = (v + dv) % 8
            wv = math.exp(-0.5 * (dv / 0.80) ** 2)
            for dd in range(-6, 7):
                tau = d + dd
                if tau < 0 or tau >= 192:
                    continue
                wd = math.exp(-0.5 * (dd / 2.60) ** 2)
                field[hh, vv, tau] += float(weight) * wh * wv * wd


def build_path_track_prior(
    train_support: np.ndarray,
    train_valid: np.ndarray,
    peaks: np.ndarray,
    peak_energy: np.ndarray,
    neighbor_idx: np.ndarray,
    neighbor_dist: np.ndarray,
    current: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    prior = np.empty_like(current, dtype=np.float32)
    top_persistence = np.zeros(len(current), dtype=np.float32)
    peak_mass = np.zeros(len(current), dtype=np.float32)
    top_group_weight = np.zeros(len(current), dtype=np.float32)
    neighbor_entropy = np.zeros(len(current), dtype=np.float32)

    valid_to_compact = -np.ones(train_support.shape[0], dtype=np.int32)
    compact = np.arange(len(peaks), dtype=np.int32)
    valid_rows = np.flatnonzero(train_valid)
    valid_to_compact[valid_rows] = compact

    for i in range(len(current)):
        ids_full = neighbor_idx[i]
        ids = valid_to_compact[ids_full]
        good = ids >= 0
        ids = ids[good]
        d = neighbor_dist[i, good].astype(np.float64)
        if len(ids) == 0:
            prior[i] = current[i]
            continue

        w_neighbor = np.exp(-d / 9.0) / np.maximum(d, 0.35) ** 2.0
        w_neighbor /= np.maximum(w_neighbor.sum(), TINY)
        neighbor_entropy[i] = float(-(w_neighbor * np.log(np.maximum(w_neighbor, 1e-12))).sum() / math.log(len(w_neighbor)))

        items: list[tuple[float, int, int, int, int]] = []
        for local_j, compact_id in enumerate(ids):
            for pidx in range(PEAK_COUNT):
                h, v, tau = peaks[compact_id, pidx]
                if h < 0:
                    continue
                item_weight = float(w_neighbor[local_j] * peak_energy[compact_id, pidx])
                items.append((item_weight, int(h), int(v), int(tau), local_j))
        items.sort(reverse=True, key=lambda row: row[0])

        groups: list[PeakGroup] = []
        for item_weight, h, v, tau, local_j in items:
            assigned = False
            for group in groups:
                gh, gv, gd = group.center()
                if circular_delta(h, gh, 16) <= 1 and circular_delta(v, gv, 8) <= 1 and abs(tau - gd) <= 4:
                    dh = circular_signed_delta(gh, h, 16)
                    dv = circular_signed_delta(gv, v, 8)
                    group.h_sum += item_weight * (gh + dh)
                    group.v_sum += item_weight * (gv + dv)
                    group.d_sum += item_weight * tau
                    group.weight += item_weight
                    group.neighbor_mask |= 1 << local_j
                    assigned = True
                    break
            if not assigned:
                groups.append(PeakGroup(item_weight * h, item_weight * v, item_weight * tau, item_weight, 1 << local_j))

        groups.sort(key=lambda group: (group.neighbor_count(), group.weight), reverse=True)
        field = np.full((16, 8, 192), FLOOR, dtype=np.float32)
        used_weight = 0.0
        used_neighbor_mass = 0.0
        for group in groups[:TRACK_GROUPS]:
            h, v, tau = group.center()
            tau = int(np.clip(tau, 0, 191))
            persistence = group.neighbor_count() / max(len(ids), 1)
            group_weight = group.weight * (0.25 + 1.20 * persistence)
            add_peak_kernel(field, h, v, tau, group_weight)
            used_weight += group_weight
            used_neighbor_mass = max(used_neighbor_mass, persistence)

        if used_weight <= 0:
            prior[i] = current[i]
        else:
            # Keep the local peak prior broad enough that projection changes
            # support placement without collapsing v140's verified diffusion.
            field = np.maximum(field, FLOOR)
            field /= np.maximum(field.sum(), TINY)
            prior[i] = normalize_support(field[None])[0]
        top_persistence[i] = float(used_neighbor_mass)
        top_group_weight[i] = float(groups[0].weight if groups else 0.0)
        peak_mass[i] = float(used_weight)
        if (i + 1) % 100 == 0:
            print(f"path prior {i + 1}/{len(current)}", flush=True)

    diagnostics = {
        "top_persistence": top_persistence,
        "peak_mass": peak_mass,
        "top_group_weight": top_group_weight,
        "neighbor_entropy": neighbor_entropy,
    }
    return prior.astype(np.float32), diagnostics


def make_weights(action: np.ndarray) -> np.ndarray:
    maxw = np.clip(0.16 + 0.82 * action, 0.16, 0.98).astype(np.float32)
    weights = np.empty((len(action), 8), dtype=np.float32)
    weights[:, 0] = maxw
    weights[:, 1:] = (1.0 - maxw[:, None]) / 7.0
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), TINY)
    return weights


def compare_outputs(output_path: Path, base_path: Path, regimes: np.ndarray) -> dict[str, object]:
    out = np.load(output_path, mmap_mode="r")
    base = np.load(base_path, mmap_mode="r")
    num = np.zeros(len(out), dtype=np.float64)
    den = np.zeros(len(out), dtype=np.float64)
    for start in range(0, len(out), 20):
        end = min(start + 20, len(out))
        h1 = np.asarray(out[start:end], dtype=np.complex64)
        h0 = np.asarray(base[start:end], dtype=np.complex64)
        num[start:end] = np.sum(np.abs(h1 - h0) ** 2, axis=(1, 2, 3), dtype=np.float64)
        den[start:end] = np.sum(np.abs(h0) ** 2, axis=(1, 2, 3), dtype=np.float64)
    ratio = num / np.maximum(den, TINY)
    return {
        "finite": bool(np.isfinite(out.real).all() and np.isfinite(out.imag).all()),
        "shape": list(out.shape),
        "dtype": str(out.dtype),
        "nmse_to_v140": float(num.sum() / max(den.sum(), TINY)),
        "nmse_to_v140_by_regime": {
            regime: float(num[regimes == regime].sum() / max(den[regimes == regime].sum(), TINY))
            for regime in REGIMES
            if np.any(regimes == regime)
        },
        "point_nmse": summarize(ratio),
    }


def main() -> None:
    pos_train = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    pos_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    geo = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    blocked_train = geo["train_blocked"].astype(np.float32)
    blocked_test = geo["test_blocked"].astype(np.float32)
    train_regimes = regime_of(blocked_train)
    test_regimes = regime_of(blocked_test)

    train_truth_raw = np.load(PATHS.shared_features / "truth_coll_power.npy", mmap_mode="r")
    peaks, peak_energy, train_valid_compact = extract_distinct_peaks(train_truth_raw)
    train_valid = np.asarray(train_truth_raw.sum(axis=(1, 2, 3)) > TINY)
    train_truth = normalize_support(np.asarray(train_truth_raw, dtype=np.float32))

    v140_base = PATHS.cache_dir(140) / "v140_Round1_Test_Channel.npy"
    current = current_collapsed_support(v140_base, pos_test)
    neighbor_idx, neighbor_dist = same_regime_neighbors(
        pos_train, train_regimes, train_valid, pos_test, test_regimes, k=8
    )
    prior, prior_diag = build_path_track_prior(
        train_truth, train_valid, peaks, peak_energy, neighbor_idx, neighbor_dist, current
    )

    risk = np.load(PATHS.cache_dir(140) / "v140_geodesic_risk.npy").astype(np.float32)
    local_range = np.clip((18.0 - neighbor_dist[:, 3]) / 14.0, 0.0, 1.0).astype(np.float32)
    persistence = prior_diag["top_persistence"].astype(np.float32)
    peak_strength = np.clip(prior_diag["peak_mass"] / 0.030, 0.0, 1.0).astype(np.float32)
    path_conf = np.clip(0.45 * local_range + 0.40 * persistence + 0.15 * peak_strength, 0.0, 1.0)
    regime_scale = np.zeros(len(test_regimes), dtype=np.float32)
    regime_scale[test_regimes == "medium"] = 0.55
    regime_scale[test_regimes == "heavy"] = 1.00
    action = np.clip(regime_scale * path_conf * np.clip((risk - 0.35) / 0.65, 0.0, 1.0), 0.0, 1.0)

    # Aggressive but bounded: target moves up to 16% for medium and 28% for
    # heavy, and only where path-track confidence and v140 geodesic risk agree.
    alpha = np.zeros_like(action, dtype=np.float32)
    alpha[test_regimes == "medium"] = 0.16 * action[test_regimes == "medium"]
    alpha[test_regimes == "heavy"] = 0.28 * action[test_regimes == "heavy"]
    final_target = (1.0 - alpha[:, None, None, None]) * current + alpha[:, None, None, None] * prior
    final_target = normalize_support(final_target)
    final_target[test_regimes == "clear"] = current[test_regimes == "clear"]
    final_target[test_regimes == "light"] = current[test_regimes == "light"]

    weights = make_weights(action)
    np.save(TARGET_OUT, final_target)
    np.save(PRIOR_OUT, prior)
    np.save(WEIGHTS_OUT, weights)
    np.save(
        ACTION_OUT,
        np.column_stack(
            [
                risk,
                path_conf,
                persistence,
                local_range,
                peak_strength,
                action,
                alpha,
                neighbor_dist[:, 0],
                neighbor_dist[:, 3],
                neighbor_dist[:, -1],
            ]
        ).astype(np.float32),
    )

    args = type("Args", (), {})()
    args.base = str(v140_base)
    args.output = str(OUTPUT)
    args.beta = 140.3
    args.gamma = 0.10
    args.eta = "clear:0.0,light:0.0,medium:0.26,heavy:0.52"
    args.min_gate = 0.10
    args.max_gate = 2.15
    args.conf_center = 0.28
    args.conf_width = 0.42
    args.entropy_low = 0.02
    args.entropy_high = 0.92
    args.gain_clip = "0.48,2.70"
    args.top_gain_clip = "0.58,2.18"
    args.top_components = 5632
    args.target_floor = FLOOR
    args.project_batch = 20
    proj = projection(final_target, weights, test_regimes, pos_test, args)
    comparison = compare_outputs(OUTPUT, v140_base, test_regimes)

    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "local repeated train-truth AD peaks provide a new path-track support signal "
            "that can improve v140 without broad residual transfer"
        ),
        "base": "v140 online_0.501352 complex output",
        "only_primary_change": (
            "bounded medium/heavy support-only peak prior from nearby same-regime "
            "train truth peak tracks; clear/light, phase, total power and pol/UE allocation preserved"
        ),
        "parameters": {
            "neighbor_k": 8,
            "track_groups": TRACK_GROUPS,
            "peak_count": PEAK_COUNT,
            "peak_candidates": PEAK_CANDIDATES,
            "alpha_medium_max": 0.16,
            "alpha_heavy_max": 0.28,
            "eta": args.eta,
            "gain_clip": args.gain_clip,
            "top_gain_clip": args.top_gain_clip,
            "top_components": args.top_components,
        },
        "local_track_diagnostics": {
            "neighbor_d1": summarize(neighbor_dist[:, 0]),
            "neighbor_d4": summarize(neighbor_dist[:, 3]),
            "neighbor_d8": summarize(neighbor_dist[:, -1]),
            "top_persistence": summarize(persistence),
            "peak_mass": summarize(prior_diag["peak_mass"]),
            "path_conf": summarize(path_conf),
            "action": summarize(action),
            "alpha": summarize(alpha),
            "active_fraction": float(np.mean(alpha > 1e-6)),
            "active_by_regime": {
                regime: float(np.mean(alpha[test_regimes == regime] > 1e-6))
                for regime in REGIMES
                if np.any(test_regimes == regime)
            },
        },
        "target_diagnostics": {
            "target_to_current": summarize(row_cos(final_target, current)),
            "prior_to_current": summarize(row_cos(prior, current)),
            "target_h_l1_change": summarize(
                np.sum(
                    np.abs(final_target.sum(axis=-1).reshape(len(final_target), -1) - current.sum(axis=-1).reshape(len(current), -1)),
                    axis=1,
                )
            ),
        },
        "projection": proj,
        "comparison_to_v140": comparison,
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
            "target": PATHS.relative(TARGET_OUT),
            "prior": PATHS.relative(PRIOR_OUT),
            "weights": PATHS.relative(WEIGHTS_OUT),
            "actions": PATHS.relative(ACTION_OUT),
            "report": PATHS.relative(REPORT),
            "zip": PATHS.relative(ZIP_OUT),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        for source, name in (
            (__file__, "generate_v158_path_track_peak_support.py"),
            (PATHS.code / "analyze_path_track_evidence.py", "analyze_path_track_evidence.py"),
            (PATHS.code / "generate_v140_balanced_hminus2_overdrive.py", "generate_v140_balanced_hminus2_overdrive.py"),
            (PATHS.code / "generate_v98_path_ensemble_graph_attention_joint_ad.py", "generate_v98_path_ensemble_graph_attention_joint_ad.py"),
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
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
