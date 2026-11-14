#!/usr/bin/env python3
"""Generate v150 by adding an LOS-safe physical fallback to v149."""

from __future__ import annotations

import json
import zipfile

import numpy as np

from generate_v143_transductive_geodesic_risk import digest, output_comparison
from generate_v147_continuous_local_action_gate import apply_complex_local_field
from generate_v87_easy_field_projection import summarize
from powerfield_common import regime_of
from project_paths import PATHS


VERSION = "v150_los_safe_continuous_action_controller"
OUT_DIR = PATHS.cache_dir(150)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(150)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v150_Round1_Test_Channel.npy"
REPORT = REPORT_DIR / "v150_los_safe_continuous_action_controller_generate.json"
ZIP_OUT = (
    PATHS.round1_versions
    / "v150_los_safe_continuous_action_controller_candidate_pending.zip"
)


def load_v149_report() -> dict[str, object]:
    path = (
        PATHS.version_report_dir(149)
        / "v149_continuous_action_response_controller_generate.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    positions_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(
        np.float32
    )
    height = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    blocked_test = height["test_blocked"].astype(np.float32)
    regimes_test = regime_of(blocked_test)
    decisions = np.load(
        PATHS.cache_dir(149) / "v149_continuous_action_response_decisions.npz"
    )
    v149_report = load_v149_report()
    prediction = decisions["test_prediction"].astype(np.float32)
    strength_v149 = decisions["test_strength"].astype(np.float32)
    los_mask = blocked_test <= 0
    strength = strength_v149.copy()
    strength[los_mask] = 0.0
    suppressed = (strength_v149 > 0) & los_mask

    application = apply_complex_local_field(
        PATHS.cache_dir(140) / "v140_Round1_Test_Channel.npy",
        OUTPUT,
        positions_test,
        prediction,
        strength,
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
        raise ValueError("v150 output contains non-finite values")

    active = strength > 0
    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "base": "v140 online 0.501352; v149 learned continuous policy",
        "primary_hypothesis": (
            "apply the v149 R/B/U continuous action policy only where the "
            "EM prior permits unresolved NLOS angular residual correction; "
            "LOS/clear points fall back to exact v140"
        ),
        "only_primary_change_from_v149": (
            "set selected local-H strength to zero when official blocked <= 0"
        ),
        "physical_interpretation": (
            "clear LOS points should not receive a learned local NLOS angular "
            "residual field; this removes test-only clear activations that "
            "were absent in v149 spatial OOF"
        ),
        "inherited_oof_from_v149": {
            "reason": (
                "v149 OOF already selected zero clear points, so the LOS "
                "fallback does not inflate local OOF gain"
            ),
            "full_gain_mean": v149_report["nested_spatial_oof"]["full_gain"][
                "mean"
            ],
            "active_fraction": v149_report["nested_spatial_oof"][
                "active_fraction"
            ],
            "active_gain_mean": v149_report["nested_spatial_oof"][
                "active_gain"
            ]["mean"],
            "fold_gain_means": v149_report["nested_spatial_oof"][
                "fold_gain_means"
            ],
            "all_folds_positive": v149_report["nested_spatial_oof"][
                "all_folds_positive"
            ],
            "clear_oof_active_fraction": v149_report["nested_spatial_oof"][
                "by_regime"
            ]["clear"]["active_fraction"],
        },
        "test_decision": {
            "v149_active_fraction": v149_report["test_decision"][
                "active_fraction"
            ],
            "active_fraction": float(np.mean(active)),
            "suppressed_clear_active_count": int(np.sum(suppressed)),
            "selected_strength": summarize(strength),
            "active_by_legacy_regime_diagnostic_only": {
                regime: float(np.mean(active[regimes_test == regime]))
                for regime in ("clear", "light", "medium", "heavy")
            },
        },
        "application": application,
        "comparison_to_v140": comparison,
        "comparison_to_v149": {
            "removed_clear_points": int(np.sum(suppressed)),
            "nmse_v149_to_v140": v149_report["comparison_to_v140"][
                "nmse_to_v140"
            ],
            "nmse_v150_to_v140": comparison["nmse_to_v140"],
        },
        "excluded": {
            "v144_v_correction": True,
            "fixed_v145_strength_everywhere": True,
            "fixed_v148_point15_gate": True,
            "new_hard_clear_light_medium_heavy_complexity_split": True,
            "complex_phase_regression": True,
        },
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
            "source_decisions": PATHS.relative(
                PATHS.cache_dir(149)
                / "v149_continuous_action_response_decisions.npz"
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

    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        for source_path, name in (
            (__file__, "generate_v150_los_safe_continuous_action_controller.py"),
            (
                PATHS.code / "generate_v149_continuous_action_response_controller.py",
                "generate_v149_continuous_action_response_controller.py",
            ),
            (
                PATHS.code / "generate_v147_continuous_local_action_gate.py",
                "generate_v147_continuous_local_action_gate.py",
            ),
            (
                PATHS.code / "generate_v146_continuous_diffusion_gate.py",
                "generate_v146_continuous_diffusion_gate.py",
            ),
            (
                PATHS.code / "analyze_v145_gated_local_h_field.py",
                "analyze_v145_gated_local_h_field.py",
            ),
            (
                PATHS.code / "analyze_post_v143_hidden_information.py",
                "analyze_post_v143_hidden_information.py",
            ),
            (
                PATHS.code / "generate_v140_balanced_hminus2_overdrive.py",
                "generate_v140_balanced_hminus2_overdrive.py",
            ),
            (
                PATHS.code / "generate_v143_transductive_geodesic_risk.py",
                "generate_v143_transductive_geodesic_risk.py",
            ),
            (
                PATHS.code / "generate_v112_kernel_moe_path_support.py",
                "generate_v112_kernel_moe_path_support.py",
            ),
            (
                PATHS.code / "generate_v98_path_ensemble_graph_attention_joint_ad.py",
                "generate_v98_path_ensemble_graph_attention_joint_ad.py",
            ),
            (
                PATHS.code / "generate_v87_easy_field_projection.py",
                "generate_v87_easy_field_projection.py",
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
