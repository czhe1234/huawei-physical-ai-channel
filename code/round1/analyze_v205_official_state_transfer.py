#!/usr/bin/env python3
"""Audit whether the v205 Sionna propagation-state signal transfers to official data."""

from __future__ import annotations

import json

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from project_paths import PATHS


VERSION = "v205_official_state_transfer"
PROFILES = (
    "sector_m30_track_d5_f6p0",
    "sector_m30_track_d5_f2p6",
    "sector_m30_track_d5_f6p7",
)
OUT_DIR = PATHS.cache_dir(205)
REPORT_DIR = PATHS.version_report_dir(205)
REPORT = REPORT_DIR / "v205_official_state_transfer.json"
POINT_METRICS = OUT_DIR / "v205_primary_hybrid_point_metrics.npz"
V200 = PATHS.cache_dir(200) / "v200_Round1_Test_Channel.npy"


def point_power(path) -> np.ndarray:
    h = np.load(path, mmap_mode="r")
    power = np.empty(len(h), dtype=np.float64)
    for start in range(0, len(h), 16):
        end = min(start + 16, len(h))
        block = np.asarray(h[start:end], dtype=np.complex64)
        power[start:end] = np.sum(np.abs(block) ** 2, axis=(1, 2, 3), dtype=np.float64)
    return power


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(a & b) / max(int(np.sum(a | b)), 1))


def quantiles(value: np.ndarray) -> dict[str, float]:
    q = np.quantile(np.asarray(value, dtype=np.float64), [0.0, 0.1, 0.5, 0.9, 1.0])
    return {name: float(item) for name, item in zip(("min", "p10", "p50", "p90", "max"), q)}


def regime_counts(mask: np.ndarray, blocked: np.ndarray) -> dict[str, int]:
    return {
        "clear": int(np.sum(mask & (blocked <= 0))),
        "light": int(np.sum(mask & (blocked >= 1) & (blocked <= 25))),
        "medium": int(np.sum(mask & (blocked >= 26) & (blocked <= 100))),
        "heavy": int(np.sum(mask & (blocked > 100))),
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    train_zero = []
    test_zero = []
    domain_counts = {}
    for profile in PROFILES:
        domain = OUT_DIR / f"benchmark_{profile}"
        train_mask = point_power(domain / "train_clean_H.npy") == 0.0
        test_mask = point_power(domain / "test_clean_H.npy") == 0.0
        train_zero.append(train_mask)
        test_zero.append(test_mask)
        domain_counts[profile] = {
            "train_zero": int(np.sum(train_mask)),
            "test_zero": int(np.sum(test_mask)),
        }

    pairwise = []
    for i in range(len(PROFILES)):
        for j in range(i + 1, len(PROFILES)):
            pairwise.append(
                {
                    "profiles": [PROFILES[i], PROFILES[j]],
                    "train_jaccard": jaccard(train_zero[i], train_zero[j]),
                    "test_jaccard": jaccard(test_zero[i], test_zero[j]),
                }
            )
    train_consensus = np.sum(train_zero, axis=0) >= 2
    test_consensus = np.sum(test_zero, axis=0) >= 2
    train_unanimous = np.all(train_zero, axis=0)
    test_unanimous = np.all(test_zero, axis=0)

    official_power = np.load(PATHS.shared_features / "train_total_power.npy").astype(np.float64)
    official_log = np.log(np.maximum(official_power, 1e-30))
    official_rank = rankdata(official_log) / len(official_log)
    official_exact_zero = official_power == 0.0
    low_power_auc = float(roc_auc_score(train_consensus.astype(np.int32), -official_log))
    geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    train_blocked = geometry["train_blocked"].astype(np.float32)
    test_blocked = geometry["test_blocked"].astype(np.float32)

    learned = np.load(POINT_METRICS)
    nonzero_probability = learned["nonzero_probability"].astype(np.float32)
    learned_zero = nonzero_probability < 0.18
    strict = test_unanimous & (nonzero_probability < 0.10)
    consensus_learned = test_consensus & learned_zero
    v200_power = point_power(V200)
    v200_log = np.log(np.maximum(v200_power, 1e-30))
    v200_rank = rankdata(v200_log) / len(v200_log)

    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "The cross-frequency Sionna unreachable state is transferable only if it is stable across "
            "calibrated domains and independently associated with low official train power."
        ),
        "domain_counts": domain_counts,
        "pairwise_zero_jaccard": pairwise,
        "consensus": {
            "train_majority_count": int(np.sum(train_consensus)),
            "test_majority_count": int(np.sum(test_consensus)),
            "train_unanimous_count": int(np.sum(train_unanimous)),
            "test_unanimous_count": int(np.sum(test_unanimous)),
            "train_regimes": regime_counts(train_consensus, train_blocked),
            "test_regimes": regime_counts(test_consensus, test_blocked),
        },
        "official_train_transfer": {
            "auc_negative_official_log_power_to_sionna_zero": low_power_auc,
            "official_log_power_sionna_zero": quantiles(official_log[train_consensus]),
            "official_log_power_sionna_nonzero": quantiles(official_log[~train_consensus]),
            "official_power_rank_sionna_zero": quantiles(official_rank[train_consensus]),
            "official_exact_zero_count": int(np.sum(official_exact_zero)),
            "exact_zero_overlap_with_sionna_consensus": int(np.sum(official_exact_zero & train_consensus)),
            "interpretation": "The 16 released exact-zero rows are not the same phenomenon as the Sionna unreachable/low-floor state.",
        },
        "test_selector": {
            "learned_zero_count_p_lt_0p18": int(np.sum(learned_zero)),
            "majority_and_learned_count": int(np.sum(consensus_learned)),
            "strict_unanimous_and_p_lt_0p10_count": int(np.sum(strict)),
            "learned_vs_majority_jaccard": jaccard(learned_zero, test_consensus),
            "learned_recall_of_majority": float(np.mean(learned_zero[test_consensus])),
            "nonzero_probability": quantiles(nonzero_probability),
            "v200_log_power_majority": quantiles(v200_log[test_consensus]),
            "v200_power_rank_majority": quantiles(v200_rank[test_consensus]),
            "v200_log_power_strict": quantiles(v200_log[strict]) if np.any(strict) else {},
            "strict_regimes": regime_counts(strict, test_blocked),
        },
        "decision": {
            "positive_evidence": (
                "Cross-frequency zero masks are stable and official train power independently predicts "
                "the synthetic state with AUC above 0.85."
            ),
            "risk": (
                "Official weak nonzero paths can occupy the same low-power region; hard-zeroing all 109-116 "
                "points would over-assume that the restored RT path budget exactly matches hidden truth."
            ),
            "next_candidate_policy": (
                "Use only the strict cross-domain+learned intersection as a new selector, and learn a bounded "
                "downscale from official train low-power residuals rather than setting channels to exact zero."
            ),
        },
        "outputs": {"report": PATHS.relative(REPORT)},
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"report": PATHS.relative(REPORT), "consensus": report["consensus"], "test_selector": report["test_selector"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
