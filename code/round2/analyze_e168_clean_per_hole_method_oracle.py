#!/usr/bin/env python3
"""Measure the per-pseudo-hole oracle across historical Clean methods."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import time

import numpy as np
import torch

from analyze_v4_exact_joint_projection import pas_distribution, pdp_distribution
from common.official_metrics import (
    PointMetricStatistics,
    aggregate_statistics,
)
from data import load_positions
from project_paths import PATHS
from run_v9_joint_sionna_migration import point_metrics


VERSION = "Round2-E168-clean-per-hole-method-oracle"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e168_clean_per_hole_method_oracle"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e168_clean_per_hole_method_oracle"
REPORT_JSON = ANALYSIS_DIR / "clean_per_hole_method_oracle.json"
REPORT_MD = ANALYSIS_DIR / "历史Clean方法逐pseudo洞选择上限.md"
ADMISSION_CSV = ANALYSIS_DIR / "clean_method_admission.csv"
OVERALL_CSV = ANALYSIS_DIR / "clean_method_overall_scores.csv"
BY_HOLE_CSV = ANALYSIS_DIR / "clean_method_by_hole_scores.csv"
FORMAL_WINNERS_CSV = ANALYSIS_DIR / "formal_v1_v7_hole_winners.csv"
ALL_WINNERS_CSV = ANALYSIS_DIR / "all_clean_hole_winners.csv"
POINT_METRICS = ARTIFACT_DIR / "clean_method_point_metrics.npz"

OOF_META = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_pre_sionna_oof_foundation_meta.npz"
)
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"
EXPECTED_SHAPE = (500, 256, 4, 192)
IDENTITY_TOLERANCE = 5e-6
REGIME_NAMES = ("clear", "light", "medium", "heavy")


@dataclass(frozen=True)
class MethodRecord:
    method_id: str
    display_name: str
    family: str
    formal: bool
    source: str
    lineage: str
    expected_score: float
    statistics: PointMetricStatistics


@dataclass(frozen=True)
class ChannelMethodSpec:
    method_id: str
    display_name: str
    family: str
    source: str
    lineage: str
    path: object
    expected_score: float
    scale: np.ndarray | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def write_csv(path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def json_compatible(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_compatible(value.tolist())
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return value


def statistics_from_arrays(
    pas: np.ndarray,
    pdp: np.ndarray,
    squared_error: np.ndarray,
    truth_energy: np.ndarray,
) -> PointMetricStatistics:
    arrays = [np.asarray(value, dtype=np.float64) for value in (pas, pdp, squared_error, truth_energy)]
    if any(value.shape != (500,) for value in arrays):
        raise ValueError(f"point statistic shape mismatch: {[value.shape for value in arrays]}")
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("non-finite point statistics")
    return PointMetricStatistics(*arrays)


def selected_scaled_statistics(
    loaded: np.lib.npyio.NpzFile,
    action_index: np.ndarray,
    scale: np.ndarray,
    error_key: str = "error",
    energy_key: str = "truth_energy",
) -> PointMetricStatistics:
    rows = np.arange(500, dtype=np.int64)
    pas = loaded["pas"][action_index, rows]
    pdp = loaded["pdp"][action_index, rows]
    truth_energy = loaded[energy_key][action_index, rows]
    if "predicted_energy" in loaded.files and "cross" in loaded.files:
        predicted_energy = loaded["predicted_energy"][action_index, rows]
        cross = loaded["cross"][action_index, rows]
        squared_error = scale * scale * predicted_energy + truth_energy - 2.0 * scale * cross
    else:
        if not np.allclose(scale, 1.0):
            raise ValueError("cannot apply non-unit scale without energy/cross statistics")
        squared_error = loaded[error_key][action_index, rows]
    return statistics_from_arrays(pas, pdp, squared_error, truth_energy)


def nested_action_indices(
    candidate_names: np.ndarray,
    decisions: list[dict[str, object]],
    fold_id: np.ndarray,
    site: np.ndarray,
    regime: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    names = candidate_names.astype(str)
    identity = np.flatnonzero(names == "identity")
    default = int(identity[0]) if len(identity) else -1
    index = np.full(500, default, dtype=np.int64)
    scale = np.ones(500, dtype=np.float64)
    assigned = np.zeros(500, dtype=bool) if default < 0 else np.ones(500, dtype=bool)
    for decision in decisions:
        mask = (fold_id == int(decision["held_fold"])) & (site == int(decision["site"]))
        if regime is not None:
            regime_value = REGIME_NAMES.index(str(decision["regime"]))
            mask &= regime == regime_value
        matches = np.flatnonzero(names == str(decision["candidate"]))
        if len(matches) != 1:
            raise RuntimeError(f"candidate lookup failed: {decision['candidate']}")
        index[mask] = int(matches[0])
        scale[mask] = float(decision.get("scale", 1.0))
        assigned[mask] = True
    if not np.all(assigned):
        raise RuntimeError(f"nested decisions left {int(np.sum(~assigned))} rows unassigned")
    return index, scale


def load_formal_methods(
    query_ids: np.ndarray,
    fold_id: np.ndarray,
    site: np.ndarray,
) -> tuple[list[MethodRecord], list[dict[str, object]]]:
    methods: list[MethodRecord] = []
    admission: list[dict[str, object]] = []

    def add(
        method_id: str,
        display_name: str,
        family: str,
        expected_score: float,
        statistics: PointMetricStatistics,
        source: str,
        lineage: str,
        formal: bool = True,
    ) -> None:
        methods.append(
            MethodRecord(
                method_id,
                display_name,
                family,
                formal,
                source,
                lineage,
                expected_score,
                statistics,
            )
        )
        admission.append(
            {
                "method_id": method_id,
                "method": display_name,
                "family": family,
                "formal_v1_v7": formal,
                "status": "CLEAN_ACCEPTED",
                "source": source,
                "lineage": lineage,
                "reason": "canonical 500 point statistics are available and trace only to official data",
            }
        )

    v1_path = (
        PATHS.artifacts
        / "round2"
        / "pre_sionna"
        / "round2_pre_sionna_oof_action_metrics.npz"
    )
    with np.load(v1_path, allow_pickle=False) as loaded:
        names = loaded["action_names"].astype(str)
        report_path = (
            PATHS.analysis
            / "results"
            / "round2_pre_sionna_pipeline"
            / "round2_pre_sionna_pipeline.json"
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        selected = np.full(500, -1, dtype=np.int64)
        for decision in report["validation"]["decisions"]:
            name = str(decision["selected_action"])
            matches = np.flatnonzero(names == name)
            if len(matches) != 1:
                raise RuntimeError(f"V1 action lookup failed: {name}")
            mask = (fold_id == int(decision["held_fold"])) & (
                site == int(decision["site"])
            )
            selected[mask] = int(matches[0])
        if np.any(selected < 0):
            raise RuntimeError("V1 nested decisions left canonical rows unassigned")
        rows = np.arange(500, dtype=np.int64)
        statistics = statistics_from_arrays(
            loaded["pas"][selected, rows],
            loaded["pdp"][selected, rows],
            loaded["error"][selected, rows],
            loaded["energy"][selected, rows],
        )
    add(
        "V1",
        "V1 complete pre-Sionna pipeline",
        "formal_version",
        0.5965343913062728,
        statistics,
        f"{PATHS.relative(v1_path)} + {PATHS.relative(report_path)}",
        "Official Train H/coordinates/Map/Setup; held-fold and site nested action",
    )

    for version, relative, expected in (
        (
            "V2",
            "rbf_spectral_field/round2_rbf_oof_metrics.npz",
            0.6285676716291476,
        ),
        (
            "V3",
            "v3_anchored_neural_rbf/oof_projection_metrics.npz",
            0.6345084646755442,
        ),
    ):
        path = PATHS.artifacts / "round2" / relative
        with np.load(path, allow_pickle=False) as loaded:
            if not np.array_equal(loaded["query_ids"].astype(np.int64), query_ids):
                raise RuntimeError(f"{version} query order mismatch")
            statistics = selected_scaled_statistics(
                loaded,
                loaded["selected_action"].astype(np.int64),
                loaded["selected_scale"].astype(np.float64),
            )
        add(
            version,
            f"{version} selected spectral field",
            "formal_version",
            expected,
            statistics,
            PATHS.relative(path),
            "Official Train H/coordinates/Map/Setup; stored nested OOF action and scale",
        )

    v4_path = (
        PATHS.artifacts
        / "round2"
        / "v4_calibrated_exact_projection"
        / "calibrated_exact_projection_metrics.npz"
    )
    v4_selection = (
        PATHS.artifacts
        / "round2"
        / "v4_calibrated_exact_projection"
        / "calibrated_exact_projection_selection.json"
    )
    with np.load(v4_path, allow_pickle=False) as loaded:
        if not np.array_equal(loaded["query_ids"].astype(np.int64), query_ids):
            raise RuntimeError("V4 query order mismatch")
        selection = json.loads(v4_selection.read_text(encoding="utf-8"))
        index, scale = nested_action_indices(
            loaded["candidate_names"], selection["nested_decisions"], fold_id, site
        )
        statistics = selected_scaled_statistics(loaded, index, scale)
    add(
        "V4",
        "V4 calibrated exact projection",
        "formal_version",
        0.6437377881656652,
        statistics,
        f"{PATHS.relative(v4_path)} + {PATHS.relative(v4_selection)}",
        "Official Train H/coordinates/Map/Setup; fold-site nested decisions",
    )

    for version, directory, metrics_name, selection_name, expected in (
        (
            "V6",
            "regime_lifecycle_support",
            "v6_projection_metrics.npz",
            "v6_selection.json",
            0.643928016047628,
        ),
        (
            "V7",
            "complex_carrier_lifecycle_overlay",
            "stacked_lifecycle_metrics.npz",
            "stacked_lifecycle_selection.json",
            0.6444563457343335,
        ),
    ):
        path = PATHS.artifacts / "round2" / directory / metrics_name
        selection_path = PATHS.artifacts / "round2" / directory / selection_name
        with np.load(path, allow_pickle=False) as loaded:
            if not np.array_equal(loaded["query_ids"].astype(np.int64), query_ids):
                raise RuntimeError(f"{version} query order mismatch")
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            index, scale = nested_action_indices(
                loaded["candidate_names"],
                selection["winner"]["nested_decisions"],
                fold_id,
                site,
                loaded["regime"].astype(np.int8),
            )
            statistics = selected_scaled_statistics(
                loaded, index, scale, energy_key="energy"
            )
            identity_index = int(
                np.flatnonzero(loaded["candidate_names"].astype(str) == "identity")[0]
            )
            core = statistics_from_arrays(
                loaded["pas"][identity_index],
                loaded["pdp"][identity_index],
                loaded["error"][identity_index],
                loaded["energy"][identity_index],
            )
        add(
            version,
            f"{version} nested lifecycle",
            "formal_version",
            expected,
            statistics,
            f"{PATHS.relative(path)} + {PATHS.relative(selection_path)}",
            "Official Train H/coordinates/Map/Setup; fold-site-regime nested decisions",
        )
        core_expected = 0.6436892536169081 if version == "V6" else 0.644314057726413
        add(
            f"{version}_core",
            f"{version} pre-lifecycle core",
            "historical_stage",
            core_expected,
            core,
            PATHS.relative(path),
            "Official Train H/coordinates/Map/Setup; identity action before lifecycle",
            formal=False,
        )
    return methods, admission


def add_npz_method(
    records: list[MethodRecord],
    admission: list[dict[str, object]],
    *,
    method_id: str,
    display_name: str,
    family: str,
    source_path,
    query_ids: np.ndarray,
    key_prefix: str,
    expected_score: float,
    lineage: str,
    error_suffix: str = "error",
    energy_suffix: str = "energy",
) -> None:
    with np.load(source_path, allow_pickle=False) as loaded:
        if "query_ids" in loaded.files and not np.array_equal(
            loaded["query_ids"].astype(np.int64), query_ids
        ):
            raise RuntimeError(f"{method_id} query order mismatch")
        statistics = statistics_from_arrays(
            loaded[f"{key_prefix}_pas"],
            loaded[f"{key_prefix}_pdp"],
            loaded[f"{key_prefix}_{error_suffix}"],
            loaded[f"{key_prefix}_{energy_suffix}"],
        )
    records.append(
        MethodRecord(
            method_id,
            display_name,
            family,
            False,
            PATHS.relative(source_path),
            lineage,
            expected_score,
            statistics,
        )
    )
    admission.append(
        {
            "method_id": method_id,
            "method": display_name,
            "family": family,
            "formal_v1_v7": False,
            "status": "CLEAN_ACCEPTED",
            "source": PATHS.relative(source_path),
            "lineage": lineage,
            "reason": "canonical 500 point statistics are available and method-specific flow is official-only",
        }
    )


def load_metric_artifact_methods(
    query_ids: np.ndarray,
) -> tuple[list[MethodRecord], list[dict[str, object]]]:
    records: list[MethodRecord] = []
    admission: list[dict[str, object]] = []

    e130 = (
        PATHS.artifacts
        / "round2"
        / "official_path_identity_system"
        / "official_path_identity_fivefold_metrics.npz"
    )
    for method_id, display, prefix, expected in (
        ("E130_identity", "E130 identity baseline", "identity_baseline", 0.528740),
        ("E130_mass", "E130 mass-only", "mass_only", 0.535224),
        ("E130_coverage", "E130 coverage-only", "coverage_only", 0.535198),
        ("E130_full", "E130 official physics full", "official_physics", 0.541694),
    ):
        add_npz_method(
            records,
            admission,
            method_id=method_id,
            display_name=display,
            family="official_path_generator",
            source_path=e130,
            query_ids=query_ids,
            key_prefix=prefix,
            expected_score=expected,
            lineage="Official Train H donor components, coordinates, Map and Setup; no Sionna/V12 path values",
            error_suffix="squared_error",
            energy_suffix="truth_energy",
        )

    e137 = (
        PATHS.artifacts
        / "round2"
        / "v7_v12_inspired_path_student"
        / "v7_v12_inspired_path_student_metrics.npz"
    )
    for method_id, display, prefix, expected in (
        ("E131_V7_endpoint", "E131 base+V7 path endpoint", "complete_base_plus_v7", 0.645746340),
        ("E137_basic", "E137 official basic student", "complete_official_basic_student", 0.645847769),
        ("E137_full", "E137 official full student", "complete_official_full_student", 0.645850825),
    ):
        add_npz_method(
            records,
            admission,
            method_id=method_id,
            display_name=display,
            family="official_path_endpoint_on_v7",
            source_path=e137,
            query_ids=query_ids,
            key_prefix=prefix,
            expected_score=expected,
            lineage="Official E120 path candidates plus V7/local official features; Sionna teacher branches in the same file are not read",
            error_suffix="squared_error",
            energy_suffix="truth_energy",
        )

    e152 = (
        PATHS.artifacts
        / "round2"
        / "full_official_state_pretraining"
        / "full_official_state_pretraining_metrics.npz"
    )
    add_npz_method(
        records,
        admission,
        method_id="E152_selected",
        display_name="E152 full official state rollout",
        family="official_path_state_rollout",
        source_path=e152,
        query_ids=query_ids,
        key_prefix="selected",
        expected_score=0.6442236080625254,
        lineage="Official Train H, V7 carriers, coordinates and Map candidates; five-fold official state training",
    )

    e154 = (
        PATHS.artifacts
        / "round2"
        / "local_spectral_transport"
        / "local_spectral_transport_metrics.npz"
    )
    for method_id, display, prefix, expected in (
        ("E154_pas", "E154 PAS transport", "pas_transport", 0.6380487976606919),
        ("E154_pdp", "E154 PDP transport", "pdp_transport", 0.637353059377082),
        ("E154_joint", "E154 joint PAS/PDP transport", "joint_pas_pdp", 0.634170274267494),
    ):
        add_npz_method(
            records,
            admission,
            method_id=method_id,
            display_name=display,
            family="local_spectral_transport",
            source_path=e154,
            query_ids=query_ids,
            key_prefix=prefix,
            expected_score=expected,
            lineage="Official V7 source spectrum and fit-only official Train H peak tracks",
        )

    e157 = (
        PATHS.artifacts
        / "round2"
        / "relational_component_transport"
        / "relational_component_transport_metrics.npz"
    )
    add_npz_method(
        records,
        admission,
        method_id="E157_candidate",
        display_name="E157 relational component transport",
        family="component_transport",
        source_path=e157,
        query_ids=query_ids,
        key_prefix="candidate",
        expected_score=0.6440620786412442,
        lineage="Official Train H 24-component relationships, coordinates, Map and V7 carrier",
    )

    e145_dir = PATHS.artifacts / "round2" / "surface_field_inverse_renderer"
    for method_id, display, prefix, expected in (
        ("E145_anchor", "E145 official anchor", "anchor", 0.644483047475815),
        ("E145_model", "E145 surface-field inverse renderer", "model", 0.6461068675937506),
    ):
        arrays = {
            "pas": np.empty(500, dtype=np.float64),
            "pdp": np.empty(500, dtype=np.float64),
            "error": np.empty(500, dtype=np.float64),
            "energy": np.empty(500, dtype=np.float64),
        }
        assigned = np.zeros(500, dtype=bool)
        for fold in range(5):
            path = e145_dir / f"fold{fold}_actual_metrics.npz"
            with np.load(path, allow_pickle=False) as loaded:
                rows = loaded["canonical_rows"].astype(np.int64)
                if not np.array_equal(query_ids[rows], loaded["query_ids"].astype(np.int64)):
                    raise RuntimeError(f"E145 fold{fold} query order mismatch")
                for name in arrays:
                    arrays[name][rows] = loaded[f"{prefix}_{name}"]
                assigned[rows] = True
        if not np.all(assigned):
            raise RuntimeError(f"{method_id} missing canonical rows")
        statistics = statistics_from_arrays(
            arrays["pas"], arrays["pdp"], arrays["error"], arrays["energy"]
        )
        source = " + ".join(
            PATHS.relative(e145_dir / f"fold{fold}_actual_metrics.npz")
            for fold in range(5)
        )
        records.append(
            MethodRecord(
                method_id,
                display,
                "surface_field",
                False,
                source,
                "Official Train H, coordinates, Map surface field and Clean V7; mismatched-surface control excluded",
                expected,
                statistics,
            )
        )
        admission.append(
            {
                "method_id": method_id,
                "method": display,
                "family": "surface_field",
                "formal_v1_v7": False,
                "status": "CLEAN_ACCEPTED",
                "source": source,
                "lineage": "Official Train H, coordinates, Map surface field and Clean V7",
                "reason": "five actual folds cover all canonical rows; mismatched control is not included",
            }
        )
    return records, admission


def e142_nested_scale(kind: str, fold_id: np.ndarray, site: np.ndarray) -> np.ndarray:
    report = json.loads(
        (
            PATHS.analysis
            / "results"
            / "round2_sparse_denoised_v7_carrier"
            / "sparse_denoised_v7_carrier.json"
        ).read_text(encoding="utf-8")
    )
    decisions = report["evaluation"][f"{kind}_score"]["nested"]["scale_decisions"]
    scale = np.ones(500, dtype=np.float64)
    assigned = np.zeros(500, dtype=bool)
    for decision in decisions:
        mask = (fold_id == int(decision["fold"])) & (site == int(decision["site"]))
        scale[mask] = float(decision["scale"])
        assigned[mask] = True
    if not np.all(assigned):
        raise RuntimeError(f"E142 {kind} nested scale incomplete")
    return scale


def channel_specs(fold_id: np.ndarray, site: np.ndarray) -> list[ChannelMethodSpec]:
    root = PATHS.artifacts / "round2"
    return [
        ChannelMethodSpec(
            "E121_base",
            "E121 strict radial IDW carrier",
            "strict_official_carrier",
            "artifacts/round2/standalone_unique_path_carrier/strict_radial_idw_k24_channel.npy",
            "Official Train H, coordinates and Setup; E120 fold assignment only",
            root / "standalone_unique_path_carrier" / "strict_radial_idw_k24_channel.npy",
            0.5292381279506954,
        ),
        ChannelMethodSpec(
            "E121_path",
            "E121 strict unique-path projection",
            "strict_official_carrier",
            "artifacts/round2/standalone_unique_path_carrier/strict_unique_path_projected_channel.npy",
            "E121 official carrier plus official E120 mass/coordinate endpoint",
            root / "standalone_unique_path_carrier" / "strict_unique_path_projected_channel.npy",
            0.5382320257827342,
        ),
        ChannelMethodSpec(
            "E122_stack",
            "E122 strict 17-carrier stack",
            "strict_official_carrier",
            "artifacts/round2/strict_complex_stack_unique_path/strict_stacked_official_channel.npy",
            "Fit-only 17 official Train H carriers and complex ridge fusion",
            root / "strict_complex_stack_unique_path" / "strict_stacked_official_channel.npy",
            0.547299073508247,
        ),
        ChannelMethodSpec(
            "E122_path",
            "E122 strict stack plus unique path",
            "strict_official_carrier",
            "artifacts/round2/strict_complex_stack_unique_path/strict_stacked_unique_path_channel.npy",
            "E122 official carrier plus official E120 mass/coordinate endpoint",
            root / "strict_complex_stack_unique_path" / "strict_stacked_unique_path_channel.npy",
            0.5552047130920094,
        ),
        ChannelMethodSpec(
            "E142_raw",
            "E142 denoised raw complex stack",
            "denoised_carrier",
            "artifacts/round2/sparse_denoised_v7_carrier/oof_denoised_stacked_complex_channel.npy",
            "Official Train H 24-component reconstruction and V7 carrier bank",
            root / "sparse_denoised_v7_carrier" / "oof_denoised_stacked_complex_channel.npy",
            0.5718574121229082,
            e142_nested_scale("raw", fold_id, site),
        ),
        ChannelMethodSpec(
            "E142_projected",
            "E142 denoised projected V7 carrier",
            "denoised_carrier",
            "artifacts/round2/sparse_denoised_v7_carrier/oof_denoised_stacked_v4_projection.npy",
            "Official Train H 24-component reconstruction plus official V4 spectral targets",
            root / "sparse_denoised_v7_carrier" / "oof_denoised_stacked_v4_projection.npy",
            0.6443359107104556,
            e142_nested_scale("projected", fold_id, site),
        ),
        ChannelMethodSpec(
            "E161_student_core",
            "E161 student before lifecycle",
            "correction_student",
            "artifacts/round2/e161_v7_correction_student/canonical_student_pre_lifecycle.npy",
            "Official full4000 V7 correction supervision; student selected zero-action behavior",
            root / "e161_v7_correction_student" / "canonical_student_pre_lifecycle.npy",
            0.6443140569983312,
        ),
        ChannelMethodSpec(
            "E161_student",
            "E161 student complete",
            "correction_student",
            "artifacts/round2/e161_v7_correction_student/canonical_student_complete.npy",
            "Official full4000 V7 correction supervision plus lifecycle",
            root / "e161_v7_correction_student" / "canonical_student_complete.npy",
            0.6444558722006445,
        ),
        ChannelMethodSpec(
            "E161_knn_core",
            "E161 KNN correction before lifecycle",
            "correction_student",
            "artifacts/round2/e161_v7_correction_student/canonical_knn_pre_lifecycle.npy",
            "Official full4000 correction coefficients transferred by spatial KNN",
            root / "e161_v7_correction_student" / "canonical_knn_pre_lifecycle.npy",
            0.5946544175801188,
        ),
        ChannelMethodSpec(
            "E161_knn",
            "E161 KNN correction complete",
            "correction_student",
            "artifacts/round2/e161_v7_correction_student/canonical_knn_complete.npy",
            "Official full4000 correction coefficients transferred by spatial KNN plus lifecycle",
            root / "e161_v7_correction_student" / "canonical_knn_complete.npy",
            0.5985488284555751,
        ),
        ChannelMethodSpec(
            "E162_knn_core",
            "E162 relative-peak KNN before lifecycle",
            "relative_peak_relation",
            "artifacts/round2/e162_v7_relative_peak_relation/canonical_knn_relative_action_pre_lifecycle.npy",
            "Official V7-relative PAS/PDP actions transferred from official Train H neighbors",
            root / "e162_v7_relative_peak_relation" / "canonical_knn_relative_action_pre_lifecycle.npy",
            0.6390007220909576,
        ),
        ChannelMethodSpec(
            "E162_knn",
            "E162 relative-peak KNN complete",
            "relative_peak_relation",
            "artifacts/round2/e162_v7_relative_peak_relation/canonical_knn_relative_action_complete.npy",
            "Official V7-relative PAS/PDP actions transferred from official Train H neighbors plus lifecycle",
            root / "e162_v7_relative_peak_relation" / "canonical_knn_relative_action_complete.npy",
            0.6393158600547713,
        ),
    ]


def score_channel_methods(
    specs: list[ChannelMethodSpec],
    query_ids: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[list[MethodRecord], list[dict[str, object]]]:
    channels = []
    for spec in specs:
        channel = np.load(spec.path, mmap_mode="r")
        if channel.shape != EXPECTED_SHAPE or channel.dtype != np.complex64:
            raise ValueError(f"invalid channel {spec.source}: {channel.shape}/{channel.dtype}")
        channels.append(channel)
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    outputs = {
        spec.method_id: {
            name: np.empty(500, dtype=np.float64)
            for name in ("pas", "pdp", "error", "energy")
        }
        for spec in specs
    }
    with torch.inference_mode():
        for start in range(0, 500, batch_size):
            end = min(start + batch_size, 500)
            truth_np = np.asarray(truth_all[query_ids[start:end]], dtype=np.complex64)
            truth = torch.as_tensor(truth_np, device=device)
            truth_pas = pas_distribution(truth)
            truth_pdp = pdp_distribution(truth)
            for spec, channel in zip(specs, channels):
                value_np = np.array(channel[start:end], dtype=np.complex64, copy=True)
                if not np.isfinite(value_np.real).all() or not np.isfinite(value_np.imag).all():
                    raise ValueError(f"non-finite channel values in {spec.source}")
                value = torch.as_tensor(value_np, device=device)
                if spec.scale is not None:
                    scale = torch.as_tensor(
                        spec.scale[start:end], dtype=value.real.dtype, device=device
                    )
                    value = value * scale[:, None, None, None]
                pas, pdp, error, energy = point_metrics(
                    value, truth, truth_pas, truth_pdp
                )
                for name, array in zip(("pas", "pdp", "error", "energy"), (pas, pdp, error, energy)):
                    outputs[spec.method_id][name][start:end] = array
            print(f"E168 GPU official metrics {end}/500 across {len(specs)} channels", flush=True)

    records: list[MethodRecord] = []
    admission: list[dict[str, object]] = []
    for spec in specs:
        value = outputs[spec.method_id]
        statistics = statistics_from_arrays(
            value["pas"], value["pdp"], value["error"], value["energy"]
        )
        records.append(
            MethodRecord(
                spec.method_id,
                spec.display_name,
                spec.family,
                False,
                spec.source,
                spec.lineage,
                spec.expected_score,
                statistics,
            )
        )
        admission.append(
            {
                "method_id": spec.method_id,
                "method": spec.display_name,
                "family": spec.family,
                "formal_v1_v7": False,
                "status": "CLEAN_ACCEPTED",
                "source": spec.source,
                "lineage": spec.lineage,
                "reason": "canonical 500 complex channel exists, is finite, and is scored with the official metric",
            }
        )
    return records, admission


def excluded_admission_rows() -> list[dict[str, object]]:
    rows = [
        ("V5", "V5 Sionna shared calibration", "DIRTY_EXCLUDED", "uses Sionna-derived spectra"),
        ("V8_V12", "V8-V12", "DIRTY_EXCLUDED", "directly or indirectly use test-position Sionna path results"),
        ("E133", "E133 deformable alignment", "EXCLUDED_INCOMPLETE", "nontrivial complete point statistics or replayable H were not retained"),
        ("E134_E136", "E134-E136", "DIRTY_EXCLUDED", "direct Sionna candidate evidence or Sionna-derived deployment"),
        ("E137_sionna", "E137 direct/distilled Sionna branches", "GRAY_DIRTY_EXCLUDED", "teacher evidence or teacher labels come from Sionna"),
        ("E138_E141", "E138-E141 V12-guided correction", "GRAY_EXCLUDED", "correction basis or targets are derived from V12"),
        ("E143_E147", "E143-E147 geometry diagnostics", "EXCLUDED_INCOMPLETE", "no deployable complete 500-point channel method"),
        ("E148", "E148 mass-only positive variant", "EXCLUDED_INCOMPLETE", "only the V7-reverting primary point arrays were retained; mass-only per-point output is absent"),
        ("E149", "E149 in-loop mass residual", "EXCLUDED_DUPLICATE", "selected output is numerically V7 zero action"),
        ("E150_E151", "E150-E151", "EXCLUDED_SINGLE_FOLD", "only outer0 strict results exist"),
        ("E155_E156", "E155-E156", "GRAY_EXCLUDED", "V12-derived correction bases are used"),
        ("E158_E159", "E158-E159", "EXCLUDED_SINGLE_FOLD", "only a pre-registered single-hole screen was run"),
        ("E160", "E160 correction capacity", "ORACLE_EXCLUDED", "uses held truth coefficients"),
        ("E162_controls", "E162 shuffle/zero/oracle", "CONTROL_ORACLE_EXCLUDED", "shuffle and zero are controls; predictable-subspace output uses held truth"),
        ("E163_plus", "E163 and later V12 transport diagnostics", "GRAY_ORACLE_EXCLUDED", "use V12 transport signatures and/or held truth transport amounts"),
    ]
    return [
        {
            "method_id": method_id,
            "method": method,
            "family": "excluded",
            "formal_v1_v7": False,
            "status": status,
            "source": "",
            "lineage": "",
            "reason": reason,
        }
        for method_id, method, status, reason in rows
    ]


def identity_audit(methods: list[MethodRecord]) -> list[dict[str, object]]:
    rows = []
    for method in methods:
        overall = aggregate_statistics(method.statistics)
        error = abs(float(overall["score"]) - method.expected_score)
        if error > IDENTITY_TOLERANCE:
            raise RuntimeError(
                f"identity gate failed for {method.method_id}: "
                f"actual={overall['score']} expected={method.expected_score} error={error}"
            )
        rows.append(
            {
                "method_id": method.method_id,
                "method": method.display_name,
                "family": method.family,
                "formal_v1_v7": method.formal,
                **overall,
                "historical_expected_score": method.expected_score,
                "identity_abs_error": error,
                "source": method.source,
                "lineage": method.lineage,
            }
        )
    return sorted(rows, key=lambda row: float(row["score"]), reverse=True)


def hole_masks(
    fold_id: np.ndarray,
    site: np.ndarray,
) -> list[tuple[str, int, int, np.ndarray]]:
    output = []
    for site_value in (0, 1):
        for fold_value in range(5):
            mask = (site == site_value) & (fold_id == fold_value)
            output.append(
                (
                    f"P_S{site_value}F{fold_value}",
                    site_value,
                    fold_value,
                    mask,
                )
            )
    if sum(int(np.sum(mask)) for _, _, _, mask in output) != 500:
        raise RuntimeError("canonical hole masks do not cover 500 points")
    return output


def per_hole_scores(
    methods: list[MethodRecord],
    masks: list[tuple[str, int, int, np.ndarray]],
) -> list[dict[str, object]]:
    rows = []
    for method in methods:
        for hole_id, site, fold, mask in masks:
            rows.append(
                {
                    "method_id": method.method_id,
                    "method": method.display_name,
                    "family": method.family,
                    "formal_v1_v7": method.formal,
                    "pseudo_hole_id": hole_id,
                    "site": site,
                    "fold": fold,
                    **aggregate_statistics(method.statistics, mask),
                }
            )
    return rows


def oracle_result(
    methods: list[MethodRecord],
    masks: list[tuple[str, int, int, np.ndarray]],
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, int]]:
    stitched = PointMetricStatistics(
        pas=np.empty(500, dtype=np.float64),
        pdp=np.empty(500, dtype=np.float64),
        squared_error=np.empty(500, dtype=np.float64),
        truth_energy=np.empty(500, dtype=np.float64),
    )
    winner_rows = []
    counts: dict[str, int] = {}
    v7 = next(method for method in methods if method.method_id == "V7")
    for hole_id, site, fold, mask in masks:
        scored = [(aggregate_statistics(method.statistics, mask), method) for method in methods]
        winner_score, winner = max(scored, key=lambda item: float(item[0]["score"]))
        v7_score = aggregate_statistics(v7.statistics, mask)
        for name in ("pas", "pdp", "squared_error", "truth_energy"):
            getattr(stitched, name)[mask] = getattr(winner.statistics, name)[mask]
        counts[winner.method_id] = counts.get(winner.method_id, 0) + 1
        winner_rows.append(
            {
                "pseudo_hole_id": hole_id,
                "site": site,
                "fold": fold,
                "point_count": int(np.sum(mask)),
                "winner_id": winner.method_id,
                "winner": winner.display_name,
                "winner_family": winner.family,
                "winner_pas": winner_score["pas"],
                "winner_pdp": winner_score["pdp"],
                "winner_nmse": winner_score["nmse"],
                "winner_score": winner_score["score"],
                "v7_score": v7_score["score"],
                "delta_vs_v7": float(winner_score["score"]) - float(v7_score["score"]),
            }
        )
    result = aggregate_statistics(stitched)
    result["delta_vs_v7"] = float(result["score"]) - float(aggregate_statistics(v7.statistics)["score"])
    return winner_rows, result, counts


def save_point_metrics(methods: list[MethodRecord], query_ids: np.ndarray, fold_id: np.ndarray, site: np.ndarray) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "query_ids": query_ids.astype(np.int32),
        "fold_id": fold_id.astype(np.int8),
        "site": site.astype(np.int8),
        "method_ids": np.asarray([method.method_id for method in methods]),
    }
    for method in methods:
        prefix = method.method_id
        payload[f"{prefix}_pas"] = method.statistics.pas
        payload[f"{prefix}_pdp"] = method.statistics.pdp
        payload[f"{prefix}_squared_error"] = method.statistics.squared_error
        payload[f"{prefix}_truth_energy"] = method.statistics.truth_energy
    np.savez_compressed(POINT_METRICS, **payload)


def markdown_table(rows: list[dict[str, object]], columns: list[tuple[str, str]], precision: int = 6) -> list[str]:
    output = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                values.append(f"{value:.{precision}f}")
            else:
                values.append(str(value))
        output.append("| " + " | ".join(values) + " |")
    return output


def write_report(
    overall_rows: list[dict[str, object]],
    formal_winners: list[dict[str, object]],
    formal_result: dict[str, object],
    formal_counts: dict[str, int],
    all_winners: list[dict[str, object]],
    all_result: dict[str, object],
    all_counts: dict[str, int],
) -> None:
    best_clean = overall_rows[0]
    lines = [
        "# 历史Clean方法逐pseudo洞选择上限",
        "",
        "## 固定结论",
        "",
        f"全局单一Clean方法最高为{best_clean['method']}，固定500点官方总分={float(best_clean['score']):.6f}。",
        f"只允许正式V1/V2/V3/V4/V6/V7逐洞选方法时，拼接PAS/PDP/NMSE/总分={float(formal_result['pas']):.6f}/{float(formal_result['pdp']):.6f}/{float(formal_result['nmse']):.6f}/{float(formal_result['score']):.6f}，相对V7={float(formal_result['delta_vs_v7']):+.6f}。",
        f"允许全部准入历史Clean方法逐洞选方法时，拼接PAS/PDP/NMSE/总分={float(all_result['pas']):.6f}/{float(all_result['pdp']):.6f}/{float(all_result['nmse']):.6f}/{float(all_result['score']):.6f}，相对V7={float(all_result['delta_vs_v7']):+.6f}。",
        "",
        "该结果使用每个pseudo-hole的held GT选择赢家，是不可部署oracle，只衡量历史方法互补上限。",
        "",
        "## 正式V1-V7逐洞赢家",
        "",
    ]
    lines.extend(
        markdown_table(
            formal_winners,
            [
                ("pseudo_hole_id", "pseudo-hole"),
                ("point_count", "点数"),
                ("winner_id", "赢家"),
                ("winner_score", "洞总分"),
                ("v7_score", "V7洞总分"),
                ("delta_vs_v7", "相对V7"),
            ],
        )
    )
    lines.extend(["", f"赢家洞数：{json.dumps(formal_counts, ensure_ascii=False)}", "", "## 全部Clean方法逐洞赢家", ""])
    lines.extend(
        markdown_table(
            all_winners,
            [
                ("pseudo_hole_id", "pseudo-hole"),
                ("point_count", "点数"),
                ("winner_id", "赢家"),
                ("winner_score", "洞总分"),
                ("v7_score", "V7洞总分"),
                ("delta_vs_v7", "相对V7"),
            ],
        )
    )
    lines.extend(["", f"赢家洞数：{json.dumps(all_counts, ensure_ascii=False)}", "", "## 准入方法整体分", ""])
    lines.extend(
        markdown_table(
            overall_rows,
            [
                ("method_id", "方法"),
                ("pas", "PAS"),
                ("pdp", "PDP"),
                ("nmse", "NMSE"),
                ("score", "总分"),
            ],
        )
    )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "逐洞oracle只回答历史方法的错误是否互补。它没有提供测试时可观测的selector，也不能当作新Clean模型、本地可部署分或线上预期。参数扫描内部动作、打乱控制、单折结果、held真值oracle、V12/Sionna派生分支均未进入赢家池。",
            "",
        ]
    )
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.time()
    device = torch.device(args.device)
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    if len(query_ids) != 500 or len(np.unique(query_ids)) != 500:
        raise RuntimeError("canonical OOF query IDs are invalid")
    _, _, _, train_site, _ = load_positions()
    site = train_site[query_ids].astype(np.int8)

    formal, formal_admission = load_formal_methods(query_ids, fold_id, site)
    artifact_methods, artifact_admission = load_metric_artifact_methods(query_ids)
    large_methods, large_admission = score_channel_methods(
        channel_specs(fold_id, site), query_ids, device, args.batch_size
    )
    methods = formal + artifact_methods + large_methods
    if len({method.method_id for method in methods}) != len(methods):
        raise RuntimeError("duplicate method IDs")

    overall_rows = identity_audit(methods)
    masks = hole_masks(fold_id, site)
    by_hole = per_hole_scores(methods, masks)
    formal_pool = [method for method in methods if method.formal]
    formal_winners, formal_result, formal_counts = oracle_result(formal_pool, masks)
    all_winners, all_result, all_counts = oracle_result(methods, masks)

    admission = formal_admission + artifact_admission + large_admission + excluded_admission_rows()
    write_csv(ADMISSION_CSV, admission)
    write_csv(OVERALL_CSV, overall_rows)
    write_csv(BY_HOLE_CSV, by_hole)
    write_csv(FORMAL_WINNERS_CSV, formal_winners)
    write_csv(ALL_WINNERS_CSV, all_winners)
    save_point_metrics(methods, query_ids, fold_id, site)
    write_report(
        overall_rows,
        formal_winners,
        formal_result,
        formal_counts,
        all_winners,
        all_result,
        all_counts,
    )

    report = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "diagnostic_only": True,
        "official_inputs_only": True,
        "sionna_or_v12_methods_in_winner_pool": False,
        "canonical_query_count": 500,
        "accepted_method_count": len(methods),
        "formal_method_ids": [method.method_id for method in formal_pool],
        "identity_tolerance": IDENTITY_TOLERANCE,
        "overall": overall_rows,
        "formal_v1_v7_oracle": {
            "official": formal_result,
            "winners": formal_winners,
            "winner_hole_counts": formal_counts,
        },
        "all_clean_oracle": {
            "official": all_result,
            "winners": all_winners,
            "winner_hole_counts": all_counts,
        },
        "outputs": {
            "report_md": PATHS.relative(REPORT_MD),
            "admission_csv": PATHS.relative(ADMISSION_CSV),
            "overall_csv": PATHS.relative(OVERALL_CSV),
            "by_hole_csv": PATHS.relative(BY_HOLE_CSV),
            "formal_winners_csv": PATHS.relative(FORMAL_WINNERS_CSV),
            "all_winners_csv": PATHS.relative(ALL_WINNERS_CSV),
            "point_metrics": PATHS.relative(POINT_METRICS),
        },
        "seconds": time.time() - started,
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(
        json.dumps(json_compatible(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_compatible(report["formal_v1_v7_oracle"]), ensure_ascii=False, indent=2))
    print(json.dumps(json_compatible(report["all_clean_oracle"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
