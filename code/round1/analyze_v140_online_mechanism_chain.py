#!/usr/bin/env python3
"""Compare the actual AD-support effects across the v140 online causal chain."""

from __future__ import annotations

import json

import numpy as np
from scipy.stats import pearsonr, spearmanr

from analyze_v140_physical_mechanism import (
    circular_center,
    marginal_h,
    normalize_support,
    signed_circular_delta,
    summarize,
)
from analyze_v140_source_alignment import (
    collapsed_support_from_complex,
    support_change,
    support_summary,
)
from powerfield_common import regime_of
from project_paths import PATHS


REPORT_DIR = PATHS.version_report_dir(140)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / "v140_online_mechanism_chain_analysis.json"

VERSIONS = {
    129: {
        "name": "v129 residual-template base",
        "operator": "none",
        "score": 0.498117,
        "output": "v129_Round1_Test_Channel.npy",
    },
    133: {
        "name": "v133 geodesic h=-1 log overlap",
        "operator": "risk-conditioned geometric log overlap",
        "score": 0.499218,
        "output": "v133_Round1_Test_Channel.npy",
    },
    138: {
        "name": "v138 broader h=-1 plus h=-2 log overlap",
        "operator": "stronger risk-conditioned geometric log overlap",
        "score": 0.500793,
        "output": "v138_Round1_Test_Channel.npy",
    },
    139: {
        "name": "v139 over-aggressive h=-3 prototype",
        "operator": "ultra geometric log overlap; not submitted",
        "score": None,
        "output": "v139_Round1_Test_Channel.npy",
    },
    140: {
        "name": "v140 balanced broader h=-2 log overlap",
        "operator": "balanced risk-conditioned geometric log overlap",
        "score": 0.501352,
        "output": "v140_Round1_Test_Channel.npy",
    },
    142: {
        "name": "v142 linear mass transport",
        "operator": "same v140 risk and projection; linear transport",
        "score": 0.500329,
        "output": "v142_Round1_Test_Channel.npy",
    },
    143: {
        "name": "v143 graph-diffused v140 risk",
        "operator": "v140 log overlap; fixed 50% graph risk diffusion",
        "score": 0.501271,
        "output": "v143_Round1_Test_Channel.npy",
    },
}


def complex_nmse(path, baseline_path) -> float:
    values = np.load(path, mmap_mode="r")
    baseline = np.load(baseline_path, mmap_mode="r")
    numerator = 0.0
    denominator = 0.0
    for start in range(0, len(values), 20):
        end = min(start + 20, len(values))
        current = np.asarray(values[start:end])
        reference = np.asarray(baseline[start:end])
        numerator += float(np.sum(np.abs(current - reference) ** 2, dtype=np.float64))
        denominator += float(np.sum(np.abs(reference) ** 2, dtype=np.float64))
    return numerator / max(denominator, 1e-30)


def target_path(version: int):
    candidates = list(PATHS.cache_dir(version).glob("*target.npy"))
    return candidates[0] if len(candidates) == 1 else None


def scalar_relation(rows: dict[int, dict[str, object]], versions: list[int], key: str) -> dict[str, object]:
    values = np.asarray([rows[version][key] for version in versions], dtype=np.float64)
    scores = np.asarray([VERSIONS[version]["score"] for version in versions], dtype=np.float64)
    return {
        "versions": versions,
        "values": values.tolist(),
        "scores": scores.tolist(),
        "pearson": float(pearsonr(values, scores).statistic),
        "spearman": float(spearmanr(values, scores).statistic),
        "warning": "Only four sequential online points; descriptive causal-chain evidence, not a fitted law.",
    }


def main() -> None:
    positions = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    blocked = np.load(PATHS.shared_features / "height_geometry_cache.npz")["test_blocked"]
    regimes = regime_of(blocked)
    baseline_path = PATHS.cache_dir(129) / VERSIONS[129]["output"]
    support: dict[int, np.ndarray] = {}
    for version, config in VERSIONS.items():
        output_path = PATHS.cache_dir(version) / config["output"]
        support[version] = collapsed_support_from_complex(output_path, positions)

    baseline = support[129]
    baseline_h = marginal_h(baseline)
    baseline_center, _ = circular_center(baseline_h)
    rows: dict[int, dict[str, object]] = {}
    for version, config in VERSIONS.items():
        current = support[version]
        current_h = marginal_h(current)
        center, _ = circular_center(current_h)
        change = support_change(baseline, current, regimes)
        summary = support_summary(current, regimes)
        target = target_path(version)
        target_cos = None
        if target is not None:
            target_values = normalize_support(np.load(target))
            flat_a = target_values.reshape(len(target_values), -1).astype(np.float64)
            flat_b = current.reshape(len(current), -1).astype(np.float64)
            target_cos = summarize(
                np.sum(flat_a * flat_b, axis=1)
                / np.maximum(
                    np.sqrt(np.sum(flat_a * flat_a, axis=1) * np.sum(flat_b * flat_b, axis=1)),
                    1e-30,
                )
            )
        rows[version] = {
            "name": config["name"],
            "operator": config["operator"],
            "online_score": config["score"],
            "online_delta_to_v129": (
                float(config["score"] - VERSIONS[129]["score"])
                if config["score"] is not None
                else None
            ),
            "complex_nmse_to_v129": complex_nmse(
                PATHS.cache_dir(version) / config["output"], baseline_path
            ),
            "support_cosine_to_v129": change["cosine"],
            "h_entropy": summary["h_entropy"],
            "joint_entropy": summary["joint_entropy"],
            "joint_top1": summary["joint_top1"],
            "h_entropy_delta_to_v129_mean": float(change["h_entropy_change"]["mean"]),
            "joint_entropy_delta_to_v129_mean": float(change["joint_entropy_change"]["mean"]),
            "joint_top1_delta_to_v129_mean": float(change["joint_top1_change"]["mean"]),
            "heavy_h_entropy_delta_to_v129_mean": float(
                change["by_regime"]["heavy"]["h_entropy_change"]["mean"]
            ),
            "heavy_joint_entropy_delta_to_v129_mean": float(
                change["by_regime"]["heavy"]["joint_entropy_change"]["mean"]
            ),
            "heavy_joint_top1_delta_to_v129_mean": float(
                change["by_regime"]["heavy"]["joint_top1_change"]["mean"]
            ),
            "h_center_delta_to_v129_mean": float(
                np.mean(signed_circular_delta(center, baseline_center))
            ),
            "h_center_delta_to_v129": summarize(
                signed_circular_delta(center, baseline_center)
            ),
            "target_to_output_cosine": target_cos,
        }

    log_chain = [129, 133, 138, 140]
    report = {
        "method": "v140_online_mechanism_chain_analysis",
        "status": "analysis_only_no_candidate",
        "purpose": (
            "Use actual complex outputs and confirmed online ablations to identify whether v140's "
            "benefit follows directional H-spectrum diffusion rather than literal bin transport."
        ),
        "versions": {str(version): row for version, row in rows.items()},
        "log_family_relations": {
            "heavy_h_entropy_delta_vs_online_score": scalar_relation(
                rows, log_chain, "heavy_h_entropy_delta_to_v129_mean"
            ),
            "heavy_joint_entropy_delta_vs_online_score": scalar_relation(
                rows, log_chain, "heavy_joint_entropy_delta_to_v129_mean"
            ),
            "heavy_top1_delta_vs_online_score": scalar_relation(
                rows, log_chain, "heavy_joint_top1_delta_to_v129_mean"
            ),
            "h_center_delta_vs_online_score": scalar_relation(
                rows, log_chain, "h_center_delta_to_v129_mean"
            ),
        },
        "controlled_online_contrasts": {
            "v140_log_vs_v142_linear": {
                "online_advantage": float(VERSIONS[140]["score"] - VERSIONS[142]["score"]),
                "heavy_h_entropy_delta_difference": float(
                    rows[140]["heavy_h_entropy_delta_to_v129_mean"]
                    - rows[142]["heavy_h_entropy_delta_to_v129_mean"]
                ),
                "heavy_joint_entropy_delta_difference": float(
                    rows[140]["heavy_joint_entropy_delta_to_v129_mean"]
                    - rows[142]["heavy_joint_entropy_delta_to_v129_mean"]
                ),
                "h_center_delta_difference": float(
                    rows[140]["h_center_delta_to_v129_mean"]
                    - rows[142]["h_center_delta_to_v129_mean"]
                ),
                "meaning": (
                    "Risk, thresholds, nominal shift weights and projection are held fixed; the "
                    "remaining online difference isolates geometric log blend versus linear transport."
                ),
            },
            "v140_raw_risk_vs_v143_graph_risk": {
                "online_advantage": float(VERSIONS[140]["score"] - VERSIONS[143]["score"]),
                "heavy_h_entropy_delta_difference": float(
                    rows[140]["heavy_h_entropy_delta_to_v129_mean"]
                    - rows[143]["heavy_h_entropy_delta_to_v129_mean"]
                ),
                "meaning": (
                    "The operator is fixed; the small v140 advantage shows pointwise extreme risk "
                    "contains useful information that fixed graph smoothing slightly erases."
                ),
            },
        },
        "outputs": {"report": PATHS.relative(REPORT)},
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {REPORT}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
