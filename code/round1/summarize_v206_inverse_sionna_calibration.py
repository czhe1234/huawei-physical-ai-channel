#!/usr/bin/env python3
"""Summarize the v206 staged inverse-Sionna calibration reports."""

from __future__ import annotations

import json

from project_paths import PATHS


REPORT_DIR = PATHS.version_report_dir(206)
REPORT = REPORT_DIR / "v206_inverse_sionna_calibration_summary.json"


def load(name: str) -> dict:
    return json.loads((REPORT_DIR / name).read_text(encoding="utf-8"))


def mean_split_score(item: dict) -> float:
    return 0.5 * (
        float(item["calibration"]["selection_score"])
        + float(item["validation"]["selection_score"])
    )


def find_config(report: dict, name: str) -> dict:
    return next(item for item in report["calibration_ranking"] if item["config"]["name"] == name)


def best_transferable(report: dict) -> dict:
    return max(report["calibration_ranking"], key=mean_split_score)


def main() -> None:
    system_id = load("v206_official_system_identification.json")
    orientation = load("v206_sionna_orientation_search.json")
    depth = load("v206_sionna_depth_tail_search.json")
    frequency_seed = load("v206_sionna_frequency_seed_search.json")
    final_audit = load("v206_sionna_final_audit_search.json")
    controls = load("v206_sionna_final_controls_search.json")
    noise = load("v206_noise_mechanism_search.json")

    final = find_config(final_audit, "frequency_6.0ghz_seed_42_1m")
    v205 = find_config(controls, "final_control_v205_all_d5_60k")
    default = find_config(controls, "final_control_sionna_default_d3_1m")
    final_score = mean_split_score(final)
    v205_score = mean_split_score(v205)
    default_score = mean_split_score(default)

    report = {
        "method": "v206_inverse_sionna_calibration_summary",
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "Canonical Etoile materials plus same-coordinate official train H can identify a metric-equivalent "
            "Sionna configuration and release-noise family without hidden test labels."
        ),
        "identified_constraints": {
            "tx_port_layout": system_id["fixed_constraints_for_search"]["port_layout"],
            "tx_azimuth_deg": -30.0,
            "rx_orientation": "track_bs",
            "subcarrier_spacing_hz": 30_000,
            "materials": "canonical Sionna Etoile ITU defaults",
            "equivalent_path_solver": {
                "max_depth_plateau": [12, 16],
                "los": True,
                "specular_reflection": True,
                "refraction": True,
                "diffuse_reflection": False,
                "diffraction": False,
                "edge_diffraction": False,
                "samples_per_src": 1_000_000,
                "max_num_paths_per_src": 1_000_000,
            },
            "carrier_equivalence_domain_ghz": [6.0, 6.5],
            "most_robust_audited_configuration": final["config"],
            "noise_family": {
                "model": noise["conclusion"]["best_validation_model"],
                "power_law_slope": noise["conclusion"]["power_law_slope"],
                "fixed_awgn_only_rejected": noise["conclusion"]["fixed_awgn_only_rejected"],
            },
        },
        "final_spatial_audit": {
            "point_count": int(final["calibration"]["count"] + final["validation"]["count"]),
            "new_configuration_mean_score": final_score,
            "v205_control_mean_score": v205_score,
            "sionna_default_control_mean_score": default_score,
            "gain_vs_v205": final_score - v205_score,
            "gain_vs_sionna_default": final_score - default_score,
            "new_configuration_halves": [
                final["calibration"]["selection_score"],
                final["validation"]["selection_score"],
            ],
            "v205_halves": [
                v205["calibration"]["selection_score"],
                v205["validation"]["selection_score"],
            ],
            "heavy_new": {
                "calibration": final["calibration"]["by_regime"]["heavy"],
                "validation": final["validation"]["by_regime"]["heavy"],
            },
            "heavy_v205": {
                "calibration": v205["calibration"]["by_regime"]["heavy"],
                "validation": v205["validation"]["by_regime"]["heavy"],
            },
        },
        "stage_evidence": {
            "orientation_best": best_transferable(orientation)["config"],
            "depth_best_validation": depth["best_by_validation"]["config"],
            "frequency_seed_best_validation": frequency_seed["best_by_validation"]["config"],
            "noise_validation_ranking": noise["validation_ranking"],
        },
        "not_uniquely_identified": [
            "The exact carrier frequency: 6.0 and 6.5 GHz are metric-equivalent within search variance.",
            "The official random ray seed: seed 42 is robust, but several high-budget seeds are close.",
            "Whether max_depth 12-16 is the literal official value or compensates for another path setting.",
            "The exact clean channel and noise realization at hidden test points.",
        ],
        "decision": {
            "positive": (
                "The final untouched spatial audit improves the v205 control by more than 0.078, so high-order "
                "specular/refraction coverage is a real missing information source in the restored scene."
            ),
            "next_model_use": (
                "Generate multi-seed high-budget 6.0/6.5 GHz path-state features and use their consensus as "
                "learned support/power features; do not replace official predictions with raw ray-traced H."
            ),
        },
        "outputs": {"report": PATHS.relative(REPORT)},
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
