#!/usr/bin/env python3
"""Generate v148 from calibrated continuous risk/benefit/uncertainty outputs."""

from __future__ import annotations

import json
import zipfile

import numpy as np

from generate_v143_transductive_geodesic_risk import digest, output_comparison
from generate_v147_continuous_local_action_gate import (
    apply_complex_local_field,
)
from generate_v87_easy_field_projection import summarize
from powerfield_common import regime_of
from project_paths import PATHS


VERSION = "v148_calibrated_continuous_local_gate"
OUT_DIR = PATHS.cache_dir(148)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(148)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v148_Round1_Test_Channel.npy"
DECISION_OUT = OUT_DIR / "v148_calibrated_continuous_decisions.npz"
REPORT = REPORT_DIR / "v148_calibrated_continuous_local_gate_generate.json"
ZIP_OUT = PATHS.round1_versions / "v148_calibrated_continuous_local_gate_candidate_pending.zip"

ACTION_INDEX = 0
ACTION_STRENGTH = 0.15
PROBABILITY_THRESHOLD = 0.70
BENEFIT_THRESHOLD = 0.002
UNCERTAINTY_QUANTILE = 0.85


def gain_summary(values: np.ndarray) -> dict[str, object]:
    return {
        **summarize(values),
        "positive_fraction": float(np.mean(values > 0)),
    }


def main() -> None:
    source = np.load(
        PATHS.cache_dir(147) / "v147_continuous_local_action_decisions.npz"
    )
    probability_train = source["train_oof_probability"].astype(np.float32)
    benefit_train = source["train_oof_benefit"][:, ACTION_INDEX].astype(np.float32)
    uncertainty_train = source["train_oof_uncertainty"][:, ACTION_INDEX].astype(
        np.float32
    )
    gains_train = source["train_oof_action_gains"][:, ACTION_INDEX].astype(
        np.float32
    )
    fold = source["train_oof_fold"].astype(np.int16)
    distance_train = source["train_oof_maximum_distance"].astype(np.float32)
    valid = fold >= 0
    train_uncertainty_threshold = float(
        np.quantile(uncertainty_train[valid], UNCERTAINTY_QUANTILE)
    )
    active_train = (
        valid
        & (probability_train >= PROBABILITY_THRESHOLD)
        & (benefit_train > BENEFIT_THRESHOLD)
        & (uncertainty_train <= train_uncertainty_threshold)
        & (distance_train <= 20.0)
    )
    realized_train = np.where(active_train, gains_train, 0.0).astype(np.float32)

    probability_test = source["test_probability"].astype(np.float32)
    benefit_test = source["test_benefit"][:, ACTION_INDEX].astype(np.float32)
    uncertainty_test = source["test_uncertainty"][:, ACTION_INDEX].astype(
        np.float32
    )
    distance_test = source["test_distance"].astype(np.float32)
    prediction_test = source["test_prediction"].astype(np.float32)
    test_uncertainty_threshold = float(
        np.quantile(uncertainty_test, UNCERTAINTY_QUANTILE)
    )
    active_test = (
        (probability_test >= PROBABILITY_THRESHOLD)
        & (benefit_test > BENEFIT_THRESHOLD)
        & (uncertainty_test <= test_uncertainty_threshold)
        & (distance_test <= 20.0)
    )
    strength_test = np.where(
        active_test, ACTION_STRENGTH, 0.0
    ).astype(np.float32)

    positions_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(
        np.float32
    )
    height = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    regimes_test = regime_of(height["test_blocked"].astype(np.float32))
    application = apply_complex_local_field(
        PATHS.cache_dir(140) / "v140_Round1_Test_Channel.npy",
        OUTPUT,
        positions_test,
        prediction_test,
        strength_test,
    )
    output = np.load(OUTPUT, mmap_mode="r")
    baseline = np.load(
        PATHS.cache_dir(140) / "v140_Round1_Test_Channel.npy", mmap_mode="r"
    )
    comparison = output_comparison(output, baseline, regimes_test)
    if output.shape != baseline.shape or output.dtype != np.complex64:
        raise ValueError(
            f"invalid output layout: shape={output.shape} dtype={output.dtype}"
        )
    if not comparison["finite"]:
        raise ValueError("v148 output contains non-finite values")

    np.savez_compressed(
        DECISION_OUT,
        action_strength=np.asarray(ACTION_STRENGTH, dtype=np.float32),
        probability_threshold=np.asarray(
            PROBABILITY_THRESHOLD, dtype=np.float32
        ),
        benefit_threshold=np.asarray(BENEFIT_THRESHOLD, dtype=np.float32),
        uncertainty_quantile=np.asarray(
            UNCERTAINTY_QUANTILE, dtype=np.float32
        ),
        train_uncertainty_threshold=np.asarray(
            train_uncertainty_threshold, dtype=np.float32
        ),
        test_uncertainty_threshold=np.asarray(
            test_uncertainty_threshold, dtype=np.float32
        ),
        train_active=active_train,
        train_realized_gain=realized_train,
        test_active=active_test,
        test_strength=strength_test,
        test_probability=probability_test,
        test_benefit=benefit_test,
        test_uncertainty=uncertainty_test,
    )

    fold_reports = []
    for fold_index in range(5):
        mask = fold == fold_index
        fold_reports.append(
            {
                "fold": fold_index,
                "count": int(mask.sum()),
                "active_count": int(np.sum(active_train & mask)),
                "active_fraction": float(np.mean(active_train[mask])),
                "gain": gain_summary(realized_train[mask]),
            }
        )
    active_gain = realized_train[active_train]
    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "base": "v140 online 0.501352",
        "primary_hypothesis": (
            "a continuous three-output decision layer can replace an added "
            "hard propagation regime when it deploys only a cross-fold stable action"
        ),
        "only_primary_change": (
            "direct post-v140 local H residual correction at strength 0.15, "
            "selected by continuous probability, benefit, and percentile-calibrated uncertainty"
        ),
        "decision": {
            "continuous_risk_probability_threshold": PROBABILITY_THRESHOLD,
            "action": "local XY k=8 H residual field",
            "action_strength": ACTION_STRENGTH,
            "predicted_benefit_threshold": BENEFIT_THRESHOLD,
            "uncertainty_quantile": UNCERTAINTY_QUANTILE,
            "train_uncertainty_threshold": train_uncertainty_threshold,
            "test_uncertainty_threshold": test_uncertainty_threshold,
            "uncertainty_calibration": (
                "domain-wise percentile because absolute ensemble disagreement "
                "has a documented train/test scale shift"
            ),
            "maximum_neighbor_distance": 20.0,
            "hard_clear_light_medium_heavy_gate": False,
        },
        "nested_spatial_oof": {
            "full_gain": gain_summary(realized_train[valid]),
            "active_fraction": float(np.mean(active_train[valid])),
            "active_count": int(active_train.sum()),
            "active_gain": gain_summary(active_gain),
            "folds": fold_reports,
            "all_folds_positive": bool(
                all(row["gain"]["mean"] > 0 for row in fold_reports)
            ),
        },
        "test_decision": {
            "active_count": int(active_test.sum()),
            "active_fraction": float(np.mean(active_test)),
            "probability": summarize(probability_test),
            "benefit": summarize(benefit_test),
            "uncertainty": summarize(uncertainty_test),
            "active_by_legacy_regime_diagnostic_only": {
                regime: {
                    "count": int(np.sum(active_test & (regimes_test == regime))),
                    "fraction": float(
                        np.mean(active_test[regimes_test == regime])
                    ),
                }
                for regime in ("clear", "light", "medium", "heavy")
            },
        },
        "application": application,
        "comparison_to_v140": comparison,
        "excluded": {
            "v145_fixed_all_point_correction": True,
            "v146_pure_diffusion": True,
            "v147_per_point_action_argmax": True,
            "v144_v_correction": True,
            "new_fixed_h_shift": True,
            "complex_phase_regression": True,
        },
        "limitations": (
            "threshold selection audited on the same outer-OOF predictions; "
            "online score remains final evidence, and no large uplift is claimed locally"
        ),
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
            "decisions": PATHS.relative(DECISION_OUT),
            "source_decisions": PATHS.relative(
                PATHS.cache_dir(147)
                / "v147_continuous_local_action_decisions.npz"
            ),
            "report": PATHS.relative(REPORT),
            "zip": PATHS.relative(ZIP_OUT),
        },
        "integrity": {
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "finite": comparison["finite"],
            "output_bytes": int(output.nbytes),
        },
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with zipfile.ZipFile(
        ZIP_OUT, "w", compression=zipfile.ZIP_STORED
    ) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        for source_path, name in (
            (__file__, "generate_v148_calibrated_continuous_local_gate.py"),
            (
                PATHS.code / "generate_v147_continuous_local_action_gate.py",
                "generate_v147_continuous_local_action_gate.py",
            ),
            (
                PATHS.code / "analyze_v145_gated_local_h_field.py",
                "analyze_v145_gated_local_h_field.py",
            ),
            (
                PATHS.code / "generate_v140_balanced_hminus2_overdrive.py",
                "generate_v140_balanced_hminus2_overdrive.py",
            ),
            (
                PATHS.code
                / "generate_v98_path_ensemble_graph_attention_joint_ad.py",
                "generate_v98_path_ensemble_graph_attention_joint_ad.py",
            ),
            (PATHS.code / "powerfield_common.py", "powerfield_common.py"),
            (PATHS.code_root / "project_paths.py", "project_paths.py"),
        ):
            archive.write(source_path, name)
    with zipfile.ZipFile(ZIP_OUT) as archive:
        bad_member = archive.testzip()
        members = archive.namelist()
    if bad_member is not None:
        raise ValueError(f"ZIP CRC failure: {bad_member}")
    report["integrity"].update(
        {
            "zip_crc": True,
            "zip_members": members,
            "sha256": digest(ZIP_OUT, "sha256"),
            "md5": digest(ZIP_OUT, "md5"),
        }
    )
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {ZIP_OUT}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
