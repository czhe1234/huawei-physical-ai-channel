#!/usr/bin/env python3
"""Audit a full-4000 OOF PAS field as a proxy for the V11 carrier PAS."""

from __future__ import annotations

import json
import time

import numpy as np
import torch

import analyze_e105_transport_projection_mismatch as e105
import analyze_e86_surface_endpoint_v11_blend as e86
import train_surface_interaction_energy_model as e85
from analyze_v4_exact_joint_projection import pas_distribution
from project_paths import PATHS
from run_v9_joint_sionna_migration import endpoint_cosine


VERSION = "Round2-E106-v11-base-pas-proxy-audit"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "v11_base_pas_proxy_audit"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v11_base_pas_proxy_audit"
REPORT = ANALYSIS_DIR / "v11_base_pas_proxy_audit.json"
METRICS = ARTIFACT_DIR / "v11_base_pas_proxy_metrics.npz"
PROXY_PAS = (
    PATHS.artifacts
    / "round2"
    / "full_path_pas_slice_gate"
    / "train_rbf_base_pas_f16.npy"
)
OFFICIAL_PAS = (
    PATHS.artifacts
    / "round2"
    / "sionna_official_calibration"
    / "official_train_pas_f16.npy"
)
OLD_ENDPOINT_PAS = e85.ARTIFACT_DIR / "e86_geometry_sequence_rendered_pas_f16.npy"
NEW_ENDPOINT_PAS = (
    PATHS.artifacts
    / "round2"
    / "locked_pdp_angle_transport"
    / "strict_oof_locked_pdp_transport_pas_f16.npy"
)


def subset_correlation(
    left: np.ndarray, right: np.ndarray, mask: np.ndarray
) -> dict[str, float | int]:
    return e105.finite_correlation(left[mask], right[mask])


def subset_sign(
    left: np.ndarray, right: np.ndarray, mask: np.ndarray
) -> dict[str, float | int]:
    return e105.sign_summary(left[mask], right[mask])


def main() -> None:
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    required = (
        PROXY_PAS,
        OFFICIAL_PAS,
        OLD_ENDPOINT_PAS,
        NEW_ENDPOINT_PAS,
        e86.V11_CHANNEL,
        e85.OOF_META,
        e85.POINT_POWER,
    )
    missing = [PATHS.relative(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")

    metadata = np.load(e85.OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    point_power = np.asarray(np.load(e85.POINT_POWER, mmap_mode="r")[query_ids])
    nonzero = point_power > 0.0

    v11 = torch.as_tensor(
        np.array(np.load(e86.V11_CHANNEL, mmap_mode="r"), copy=True), device=device
    )
    with torch.inference_mode():
        actual_base = pas_distribution(v11)
    proxy_all = np.load(PROXY_PAS, mmap_mode="r")
    truth_all = np.load(OFFICIAL_PAS, mmap_mode="r")
    proxy = torch.as_tensor(
        np.asarray(proxy_all[query_ids], dtype=np.float32), device=device
    )
    truth = torch.as_tensor(
        np.asarray(truth_all[query_ids], dtype=np.float32), device=device
    )
    old_endpoint = torch.as_tensor(
        np.asarray(np.load(OLD_ENDPOINT_PAS, mmap_mode="r"), dtype=np.float32),
        device=device,
    )
    new_collapsed = torch.as_tensor(
        np.asarray(np.load(NEW_ENDPOINT_PAS, mmap_mode="r"), dtype=np.float32),
        device=device,
    )
    new_endpoint = new_collapsed[..., None].expand_as(old_endpoint)

    actual_similarity = endpoint_cosine(proxy, actual_base, "pas")
    actual_truth_score = endpoint_cosine(actual_base, truth, "pas")
    proxy_truth_score = endpoint_cosine(proxy, truth, "pas")
    with torch.inference_mode():
        actual_old = e86.endpoint_target(
            actual_base, old_endpoint, 0.20, "sqrt", (1, 2, 3)
        )
        actual_new = e86.endpoint_target(
            actual_base, new_endpoint, 0.20, "sqrt", (1, 2, 3)
        )
        proxy_old = e86.endpoint_target(
            proxy, old_endpoint, 0.20, "sqrt", (1, 2, 3)
        )
        proxy_new = e86.endpoint_target(
            proxy, new_endpoint, 0.20, "sqrt", (1, 2, 3)
        )
    actual_delta = endpoint_cosine(actual_new, truth, "pas") - endpoint_cosine(
        actual_old, truth, "pas"
    )
    proxy_delta = endpoint_cosine(proxy_new, truth, "pas") - endpoint_cosine(
        proxy_old, truth, "pas"
    )

    fold_rows = []
    sign_matches = 0
    for outer in range(5):
        mask = (fold_id == outer) & nonzero
        actual_mean = float(np.mean(actual_delta[mask]))
        proxy_mean = float(np.mean(proxy_delta[mask]))
        same = bool((actual_mean > 0.0) == (proxy_mean > 0.0))
        sign_matches += int(same)
        fold_rows.append(
            {
                "fold": outer,
                "count": int(np.sum(mask)),
                "actual_action_gain": actual_mean,
                "proxy_action_gain": proxy_mean,
                "same_sign": same,
            }
        )

    label_fidelity = {
        "all": {
            "correlation": e105.finite_correlation(actual_delta, proxy_delta),
            "sign": e105.sign_summary(actual_delta, proxy_delta),
        },
        "nonzero": {
            "correlation": subset_correlation(actual_delta, proxy_delta, nonzero),
            "sign": subset_sign(actual_delta, proxy_delta, nonzero),
        },
        "folds": fold_rows,
        "fold_mean_sign_matches": sign_matches,
    }
    decision = {
        "minimum_nonzero_pearson": 0.80,
        "minimum_nonzero_spearman": 0.75,
        "minimum_nonzero_same_sign_fraction": 0.75,
        "minimum_fold_mean_sign_matches": 4,
        "nonzero_pearson": label_fidelity["nonzero"]["correlation"]["pearson"],
        "nonzero_spearman": label_fidelity["nonzero"]["correlation"]["spearman"],
        "nonzero_same_sign_fraction": label_fidelity["nonzero"]["sign"][
            "same_sign_fraction"
        ],
        "fold_mean_sign_matches": sign_matches,
    }
    decision["use_proxy_for_full4000_training"] = bool(
        decision["nonzero_pearson"] >= decision["minimum_nonzero_pearson"]
        and decision["nonzero_spearman"] >= decision["minimum_nonzero_spearman"]
        and decision["nonzero_same_sign_fraction"]
        >= decision["minimum_nonzero_same_sign_fraction"]
        and decision["fold_mean_sign_matches"]
        >= decision["minimum_fold_mean_sign_matches"]
    )

    np.savez_compressed(
        METRICS,
        query_ids=query_ids.astype(np.int32),
        fold_id=fold_id,
        nonzero=nonzero,
        base_pas_cosine=actual_similarity.astype(np.float32),
        actual_truth_pas=actual_truth_score.astype(np.float32),
        proxy_truth_pas=proxy_truth_score.astype(np.float32),
        actual_action_gain=actual_delta.astype(np.float32),
        proxy_action_gain=proxy_delta.astype(np.float32),
    )
    report = {
        "experiment": VERSION,
        "status": "strict_diagnostic_complete",
        "primary_hypothesis": (
            "The full-4000 broad-fold RBF PAS field preserves the V11-relative "
            "transport action labels well enough to train a base-conditioned model."
        ),
        "inputs": {
            "proxy": PATHS.relative(PROXY_PAS),
            "proxy_shape": list(proxy_all.shape),
            "official_pas": PATHS.relative(OFFICIAL_PAS),
            "official_pas_shape": list(truth_all.shape),
            "canonical_count": int(len(query_ids)),
            "canonical_nonzero": int(np.sum(nonzero)),
        },
        "base_fidelity": {
            "proxy_v11_pas_cosine_all": float(np.mean(actual_similarity)),
            "proxy_v11_pas_cosine_nonzero": float(np.mean(actual_similarity[nonzero])),
            "v11_truth_pas": float(np.mean(actual_truth_score)),
            "proxy_truth_pas": float(np.mean(proxy_truth_score)),
            "proxy_minus_v11_truth_pas": float(
                np.mean(proxy_truth_score - actual_truth_score)
            ),
        },
        "action_label_fidelity": label_fidelity,
        "decision": decision,
        "outputs": {
            "metrics": PATHS.relative(METRICS),
            "finite": bool(
                np.isfinite(actual_similarity).all()
                and np.isfinite(actual_delta).all()
                and np.isfinite(proxy_delta).all()
            ),
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
