#!/usr/bin/env python3
"""Test whether train-truth angle-delay peaks form trackable spatial paths.

This is an analysis-only capacity audit. It does not use a reconstructed model
base or produce a submission candidate. The test compares joint H/V/delay peak
continuity between nearby same-regime points against regime-matched points more
than 80 m apart, then measures how often four local neighbors independently
recover the same held-out peak.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from powerfield_common import TINY, regime_of
from project_paths import PATHS


OUT_DIR = PATHS.analysis / "results" / "path_track_evidence"
REPORT = OUT_DIR / "path_track_evidence_analysis.json"
PAIR_CSV = OUT_DIR / "path_track_pair_distance_curve.csv"

REGIMES = ("clear", "light", "medium", "heavy")
DISTANCE_BINS = (0.0, 2.5, 5.0, 10.0, 15.0, 20.0)
PEAK_COUNT = 8
PEAK_CANDIDATES = 512
RANDOM_STATE = 16101
BOOTSTRAPS = 2000


@dataclass(frozen=True)
class MatchThreshold:
    name: str
    h: int
    v: int
    delay: int


THRESHOLDS = (
    MatchThreshold("strict", 1, 1, 3),
    MatchThreshold("main", 2, 1, 6),
    MatchThreshold("loose", 3, 2, 10),
)


def summarize(values: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    return {
        "count": int(len(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p10": float(np.quantile(x, 0.10)),
        "median": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
    }


def circular_distance(a: np.ndarray, b: np.ndarray, size: int) -> np.ndarray:
    delta = np.abs(a - b)
    return np.minimum(delta, size - delta)


def extract_distinct_peaks(
    support_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = np.asarray(support_raw.sum(axis=(1, 2, 3)), dtype=np.float64)
    valid = total > TINY
    support = support_raw[valid]
    coordinates = np.full((len(support), PEAK_COUNT, 3), -1, dtype=np.int16)
    energy = np.zeros((len(support), PEAK_COUNT), dtype=np.float32)

    for start in range(0, len(support), 32):
        end = min(start + 32, len(support))
        block = np.asarray(support[start:end], dtype=np.float32)
        block /= np.maximum(block.sum(axis=(1, 2, 3), keepdims=True), TINY)
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
                    min(abs(h - ah), 16 - abs(h - ah)) <= 1
                    and min(abs(v - av), 8 - abs(v - av)) <= 1
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


def coordinate_deltas(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dh = circular_distance(a[:, None, 0], b[None, :, 0], 16)
    dv = circular_distance(a[:, None, 1], b[None, :, 1], 8)
    dd = np.abs(a[:, None, 2] - b[None, :, 2])
    return dh, dv, dd


def evaluate_pairs(
    pairs: np.ndarray,
    positions: np.ndarray,
    peaks: np.ndarray,
    energy: np.ndarray,
) -> dict[str, np.ndarray]:
    output: dict[str, list[float]] = {
        "distance": [],
        "assigned_h_shift": [],
        "assigned_v_shift": [],
        "assigned_delay_shift": [],
    }
    for threshold in THRESHOLDS:
        output[f"{threshold.name}_match_fraction"] = []
        output[f"{threshold.name}_energy_overlap"] = []

    for row, (left, right) in enumerate(pairs):
        a = peaks[left]
        b = peaks[right]
        dh, dv, dd = coordinate_deltas(a, b)
        cost = np.sqrt((dh / 2.0) ** 2 + (dv / 1.5) ** 2 + (dd / 6.0) ** 2)
        ai, bi = linear_sum_assignment(cost)
        assigned_h = dh[ai, bi]
        assigned_v = dv[ai, bi]
        assigned_d = dd[ai, bi]
        output["distance"].append(float(np.linalg.norm(positions[left] - positions[right])))

        main = THRESHOLDS[1]
        main_match = (
            (assigned_h <= main.h)
            & (assigned_v <= main.v)
            & (assigned_d <= main.delay)
        )
        output["assigned_h_shift"].append(float(np.mean(assigned_h[main_match])) if np.any(main_match) else np.nan)
        output["assigned_v_shift"].append(float(np.mean(assigned_v[main_match])) if np.any(main_match) else np.nan)
        output["assigned_delay_shift"].append(float(np.mean(assigned_d[main_match])) if np.any(main_match) else np.nan)

        for threshold in THRESHOLDS:
            matched = (
                (assigned_h <= threshold.h)
                & (assigned_v <= threshold.v)
                & (assigned_d <= threshold.delay)
            )
            overlap = np.minimum(energy[left, ai], energy[right, bi])
            denominator = max(
                min(float(energy[left].sum()), float(energy[right].sum())), TINY
            )
            output[f"{threshold.name}_match_fraction"].append(float(np.mean(matched)))
            output[f"{threshold.name}_energy_overlap"].append(
                float(overlap[matched].sum() / denominator)
            )
        if (row + 1) % 10000 == 0:
            print(f"match pairs {row + 1}/{len(pairs)}", flush=True)
    return {name: np.asarray(values, dtype=np.float32) for name, values in output.items()}


def same_regime_local_pairs(positions: np.ndarray, regimes: np.ndarray) -> np.ndarray:
    pairs = np.asarray(list(cKDTree(positions).query_pairs(DISTANCE_BINS[-1])), dtype=np.int32)
    return pairs[regimes[pairs[:, 0]] == regimes[pairs[:, 1]]]


def regime_matched_far_pairs(
    positions: np.ndarray,
    regimes: np.ndarray,
    local_pairs: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    selected = []
    for regime in REGIMES:
        required = int(np.sum(regimes[local_pairs[:, 0]] == regime))
        ids = np.flatnonzero(regimes == regime)
        left, right = np.triu_indices(len(ids), k=1)
        candidate = np.column_stack([ids[left], ids[right]]).astype(np.int32)
        distance = np.linalg.norm(
            positions[candidate[:, 0]] - positions[candidate[:, 1]], axis=1
        )
        candidate = candidate[distance >= 80.0]
        replace = len(candidate) < required
        take = rng.choice(len(candidate), size=required, replace=replace)
        selected.append(candidate[take])
    return np.concatenate(selected, axis=0)


def peak_recovery(
    query_peak: np.ndarray,
    query_energy: np.ndarray,
    neighbor_peaks: np.ndarray,
    threshold: MatchThreshold,
) -> tuple[float, float, float]:
    persistence = np.zeros(PEAK_COUNT, dtype=np.int16)
    for neighbor in neighbor_peaks:
        dh, dv, dd = coordinate_deltas(query_peak, neighbor)
        matched = (dh <= threshold.h) & (dv <= threshold.v) & (dd <= threshold.delay)
        persistence += np.any(matched, axis=1)
    denominator = max(float(query_energy.sum()), TINY)
    return tuple(
        float(query_energy[persistence >= count].sum() / denominator)
        for count in (1, 2, 3)
    )


def neighbor_persistence(
    positions: np.ndarray,
    regimes: np.ndarray,
    peaks: np.ndarray,
    energy: np.ndarray,
    rng: np.random.Generator,
) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray]:
    local_neighbors = np.empty((len(positions), 4), dtype=np.int32)
    fourth_distance = np.empty(len(positions), dtype=np.float32)
    far_pool: dict[str, list[np.ndarray]] = {}
    for regime in REGIMES:
        ids = np.flatnonzero(regimes == regime)
        distance, index = cKDTree(positions[ids]).query(positions[ids], k=5)
        local_neighbors[ids] = ids[index[:, 1:5]]
        fourth_distance[ids] = distance[:, 4]
        pools = []
        for query in ids:
            far = ids[np.linalg.norm(positions[ids] - positions[query], axis=1) >= 80.0]
            pools.append(far)
        far_pool[regime] = pools

    output: dict[str, dict[str, np.ndarray]] = {}
    for threshold in THRESHOLDS:
        local = np.empty((len(positions), 3), dtype=np.float32)
        random_control = np.empty((len(positions), 3), dtype=np.float32)
        regime_offsets = {regime: 0 for regime in REGIMES}
        for query in range(len(positions)):
            regime = str(regimes[query])
            local[query] = peak_recovery(
                peaks[query], energy[query], peaks[local_neighbors[query]], threshold
            )
            pool = far_pool[regime][regime_offsets[regime]]
            regime_offsets[regime] += 1
            chosen = rng.choice(pool, size=4, replace=len(pool) < 4)
            random_control[query] = peak_recovery(
                peaks[query], energy[query], peaks[chosen], threshold
            )
        output[threshold.name] = {"local": local, "far_control": random_control}
    return output, fourth_distance


def bootstrap_difference(
    local: np.ndarray,
    control: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float]:
    difference = np.asarray(local - control, dtype=np.float64)
    boot = np.empty(BOOTSTRAPS, dtype=np.float64)
    for index in range(BOOTSTRAPS):
        sample = rng.integers(0, len(difference), size=len(difference))
        boot[index] = float(np.mean(difference[sample]))
    return {
        "mean_difference": float(np.mean(difference)),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "bootstrap_nonpositive_fraction": float(np.mean(boot <= 0.0)),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)
    support_raw = np.load(PATHS.shared_features / "truth_coll_power.npy", mmap_mode="r")
    positions_all = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)[:, :2]
    blocked_all = np.load(PATHS.shared_features / "height_geometry_cache.npz")[
        "train_blocked"
    ].astype(np.float32)
    peaks, energy, valid = extract_distinct_peaks(support_raw)
    positions = positions_all[valid]
    regimes = regime_of(blocked_all[valid])

    local_pairs = same_regime_local_pairs(positions, regimes)
    far_pairs = regime_matched_far_pairs(positions, regimes, local_pairs, rng)
    local_metrics = evaluate_pairs(local_pairs, positions, peaks, energy)
    far_metrics = evaluate_pairs(far_pairs, positions, peaks, energy)

    pair_rows = []
    for low, high in zip(DISTANCE_BINS[:-1], DISTANCE_BINS[1:]):
        mask = (local_metrics["distance"] >= low) & (local_metrics["distance"] < high)
        row: dict[str, float | int | str] = {
            "source": "local_same_regime",
            "distance_low": low,
            "distance_high": high,
            "pair_count": int(np.sum(mask)),
        }
        for threshold in THRESHOLDS:
            row[f"{threshold.name}_match_fraction"] = float(
                np.mean(local_metrics[f"{threshold.name}_match_fraction"][mask])
            )
            row[f"{threshold.name}_energy_overlap"] = float(
                np.mean(local_metrics[f"{threshold.name}_energy_overlap"][mask])
            )
        for name in ("assigned_h_shift", "assigned_v_shift", "assigned_delay_shift"):
            row[name] = float(np.nanmean(local_metrics[name][mask]))
        pair_rows.append(row)

    far_row: dict[str, float | int | str | None] = {
        "source": "same_regime_far_control",
        "distance_low": 80.0,
        "distance_high": None,
        "pair_count": int(len(far_pairs)),
    }
    for threshold in THRESHOLDS:
        far_row[f"{threshold.name}_match_fraction"] = float(
            np.mean(far_metrics[f"{threshold.name}_match_fraction"])
        )
        far_row[f"{threshold.name}_energy_overlap"] = float(
            np.mean(far_metrics[f"{threshold.name}_energy_overlap"])
        )
    for name in ("assigned_h_shift", "assigned_v_shift", "assigned_delay_shift"):
        far_row[name] = float(np.nanmean(far_metrics[name]))
    pair_rows.append(far_row)

    with PAIR_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)

    persistence, fourth_distance = neighbor_persistence(
        positions, regimes, peaks, energy, rng
    )
    persistence_report = {}
    for threshold in THRESHOLDS:
        values = persistence[threshold.name]
        threshold_report = {}
        for column, label in enumerate(("recovered_by_1plus", "persistent_in_2plus", "persistent_in_3plus")):
            threshold_report[label] = {
                "local": summarize(values["local"][:, column]),
                "far_control": summarize(values["far_control"][:, column]),
                "paired_bootstrap": bootstrap_difference(
                    values["local"][:, column], values["far_control"][:, column], rng
                ),
                "by_regime": {
                    regime: {
                        "local_mean": float(np.mean(values["local"][regimes == regime, column])),
                        "far_control_mean": float(
                            np.mean(values["far_control"][regimes == regime, column])
                        ),
                    }
                    for regime in REGIMES
                },
            }
        persistence_report[threshold.name] = threshold_report

    monotonic_distance_decay = all(
        bool(
            np.all(
                np.diff(
                    [
                        float(row[f"{threshold.name}_{metric}"])
                        for row in pair_rows[:-1]
                    ]
                )
                < 0.0
            )
        )
        for threshold in THRESHOLDS
        for metric in ("match_fraction", "energy_overlap")
    )
    persistence_ci_positive = all(
        persistence_report[threshold.name][label]["paired_bootstrap"]["ci95_low"]
        > 0.0
        for threshold in THRESHOLDS
        for label in ("recovered_by_1plus", "persistent_in_2plus", "persistent_in_3plus")
    )
    report = {
        "method": "train_truth_path_track_evidence",
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "strong joint angle-delay peaks retain a matchable identity across nearby spatial points "
            "more often than across regime-matched distant points"
        ),
        "data": {
            "support": PATHS.relative(PATHS.shared_features / "truth_coll_power.npy"),
            "train_points": int(len(support_raw)),
            "valid_nonzero_points": int(np.sum(valid)),
            "exact_zero_points_excluded_from_peak_statistics": int(np.sum(~valid)),
            "support_shape": list(support_raw.shape),
            "peak_count": PEAK_COUNT,
            "nms_radius_bins": {"h": 1, "v": 1, "delay": 2},
            "thresholds": {threshold.name: threshold.__dict__ for threshold in THRESHOLDS},
        },
        "geometry": {
            "same_regime_local_pair_count_lt20m": int(len(local_pairs)),
            "regime_matched_far_pair_count_ge80m": int(len(far_pairs)),
            "fourth_same_regime_neighbor_distance": summarize(fourth_distance),
        },
        "peak_energy": {
            "distinct_top8_center_energy_fraction": summarize(energy.sum(axis=1)),
        },
        "pair_distance_curve": pair_rows,
        "neighbor_peak_persistence": persistence_report,
        "decision_rule": {
            "supported_if": (
                "all three matching thresholds show monotonic distance decay and local four-neighbor "
                "peak recovery exceeds the far control with a positive paired-bootstrap 95% interval"
            ),
            "pair_csv": PATHS.relative(PAIR_CSV),
        },
        "conclusion": {
            "hypothesis_supported": bool(
                monotonic_distance_decay and persistence_ci_positive
            ),
            "monotonic_distance_decay_all_thresholds": monotonic_distance_decay,
            "positive_bootstrap_ci_all_thresholds": persistence_ci_positive,
            "key_evidence": {
                "strict_pair_match_lt2_5m": pair_rows[0]["strict_match_fraction"],
                "strict_pair_match_far": far_row["strict_match_fraction"],
                "strict_energy_overlap_lt2_5m": pair_rows[0]["strict_energy_overlap"],
                "strict_energy_overlap_far": far_row["strict_energy_overlap"],
                "strict_three_neighbor_persistence_local": persistence_report["strict"][
                    "persistent_in_3plus"
                ]["local"]["mean"],
                "strict_three_neighbor_persistence_far": persistence_report["strict"][
                    "persistent_in_3plus"
                ]["far_control"]["mean"],
            },
            "limitations": [
                "Peak identity is inferred from joint-bin proximity, not assigned to a named physical reflector.",
                "The eight NMS peak centers contain about 40% of total support energy; this audit does not claim full-channel recovery.",
                "Spatial continuity is necessary but not sufficient for online gain; a fold-isolated predictor must still beat static neighbor and v140 support baselines.",
            ],
        },
        "outputs": {"report": PATHS.relative(REPORT), "pair_csv": PATHS.relative(PAIR_CSV)},
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
