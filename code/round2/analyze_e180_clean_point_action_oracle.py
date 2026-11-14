#!/usr/bin/env python3
"""Measure the point-wise oracle ceiling of historical Clean channel actions."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from analyze_e168_clean_per_hole_method_oracle import e142_nested_scale
from analyze_e173_v7_clean_carrier_oracle import point_metrics_torch
from common.official_metrics import (
    OfficialMetricAccumulator,
    PointMetricStatistics,
    aggregate_statistics,
)
from project_paths import PATHS


VERSION = "Round2-E180-clean-point-action-oracle"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e180_clean_point_action_oracle"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e180_clean_point_action_oracle"

E168_METRICS = (
    PATHS.artifacts
    / "round2"
    / "e168_clean_per_hole_method_oracle"
    / "clean_method_point_metrics.npz"
)
E168_ADMISSION = (
    PATHS.analysis
    / "results"
    / "round2_e168_clean_per_hole_method_oracle"
    / "clean_method_admission.csv"
)
E173_METRICS = (
    PATHS.artifacts
    / "round2"
    / "e173_v7_clean_carrier_oracle"
    / "carrier_point_metrics_and_compact.npz"
)
CARRIER_BANK = (
    PATHS.artifacts / "round2" / "complex_carrier_bank" / "oof_complex_carrier_bank.npy"
)
CARRIER_META = (
    PATHS.artifacts
    / "round2"
    / "complex_carrier_bank"
    / "complex_carrier_bank_metadata.json"
)
PHASE_FACTORS = (
    PATHS.artifacts
    / "round2"
    / "v5_static_link_phase_calibration"
    / "oof_static_phase_factors.npz"
)
PHASE_METRICS = (
    PATHS.artifacts
    / "round2"
    / "v5_static_link_phase_calibration"
    / "oof_static_phase_metrics.npz"
)
OOF_META = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_pre_sionna_oof_foundation_meta.npz"
)
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"
V7_CHANNEL = (
    PATHS.artifacts
    / "round2"
    / "e160_full4000_v7_supervision"
    / "canonical_v7_complete.npy"
)
V4_CHANNEL = (
    PATHS.artifacts
    / "round2"
    / "v5_static_link_phase_calibration"
    / "v4_nested_oof_channel.npy"
)
ADDITIONAL_METRICS = ARTIFACT_DIR / "additional_action_point_metrics.npz"
COMPACT_ORACLE_H = ARTIFACT_DIR / "compact_distinct_oracle_channel.npy"
EXPANDED_ORACLE_H = ARTIFACT_DIR / "expanded_complete_oracle_channel.npy"
REPORT_JSON = ANALYSIS_DIR / "clean_point_action_oracle.json"
REPORT_MD = ANALYSIS_DIR / "历史Clean局部修正动作库逐点上限.md"
ADMISSION_CSV = ANALYSIS_DIR / "clean_action_admission.csv"
ACTION_CSV = ANALYSIS_DIR / "clean_action_local_complementarity.csv"
SELECTION_CSV = ANALYSIS_DIR / "point_oracle_selection_counts.csv"
HOLE_CSV = ANALYSIS_DIR / "point_oracle_hole_scores.csv"
FAMILY_CSV = ANALYSIS_DIR / "action_family_incremental_oracles.csv"
PAIR_CSV = ANALYSIS_DIR / "action_point_error_correlations.csv"

EXPECTED_SHAPE = (500, 256, 4, 192)
EXPECTED_V7_SCORE = 0.6444563457343335
SUCCESS_SCORE = 0.660000
TARGET_LOCAL_SCORE = 0.6775208075
TINY = 1e-30
TIE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class ActionSpec:
    action_id: str
    display_name: str
    family: str
    source: str
    lineage: str
    provider: str
    path: Path | None = None
    carrier_index: int | None = None
    scale: np.ndarray | None = None
    phase_factor: str | None = None
    metric_source: str = "compute"
    metric_key: str | None = None
    primary: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--force-additional-metrics", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def statistics_from_arrays(
    pas: np.ndarray,
    pdp: np.ndarray,
    error: np.ndarray,
    energy: np.ndarray,
) -> PointMetricStatistics:
    return PointMetricStatistics(
        pas=np.asarray(pas, dtype=np.float64),
        pdp=np.asarray(pdp, dtype=np.float64),
        squared_error=np.asarray(error, dtype=np.float64),
        truth_energy=np.asarray(energy, dtype=np.float64),
    )


def local_scores(values: PointMetricStatistics) -> np.ndarray:
    point_nmse = values.squared_error / np.maximum(values.truth_energy, TINY)
    return 0.4 * values.pas + 0.4 * values.pdp + 0.2 / (1.0 + point_nmse)


def check_clean_report(relative_path: str) -> None:
    payload = json.loads((PATHS.root / relative_path).read_text(encoding="utf-8"))
    if not bool(payload.get("official_inputs_only")) or bool(payload.get("sionna_used")):
        raise RuntimeError(f"Clean lineage gate failed: {relative_path}")


def build_actions(fold_id: np.ndarray, site: np.ndarray) -> list[ActionSpec]:
    root = PATHS.artifacts / "round2"
    carrier_metadata = json.loads(CARRIER_META.read_text(encoding="utf-8"))
    if not bool(carrier_metadata.get("official_inputs_only")) or bool(
        carrier_metadata.get("sionna_used")
    ):
        raise RuntimeError("E44 carrier bank failed Clean lineage gate")
    carrier_names = [str(value) for value in carrier_metadata["config_names"]]
    if len(carrier_names) != 17:
        raise RuntimeError(f"unexpected carrier count: {len(carrier_names)}")

    for report in (
        "analysis/results/round2_v4_path_component_transport/path_component_transport.json",
        "analysis/results/round2_v4_structured_complex_field/structured_complex_field.json",
        "analysis/results/round2_v5_static_link_phase_calibration/static_link_phase_calibration.json",
    ):
        check_clean_report(report)

    def file_action(
        action_id: str,
        display_name: str,
        family: str,
        path: Path,
        lineage: str,
        *,
        metric_key: str | None = None,
        metric_source: str = "e168",
        scale: np.ndarray | None = None,
        primary: bool = False,
    ) -> ActionSpec:
        return ActionSpec(
            action_id=action_id,
            display_name=display_name,
            family=family,
            source=PATHS.relative(path),
            lineage=lineage,
            provider="file",
            path=path,
            scale=scale,
            metric_source=metric_source,
            metric_key=metric_key or action_id,
            primary=primary,
        )

    actions = [
        file_action(
            "V7",
            "V7 baseline",
            "baseline",
            V7_CHANNEL,
            "Official Train H/coordinates/Map/Setup; canonical Clean V7",
            primary=True,
        ),
        file_action(
            "V4",
            "V4 calibrated projection",
            "spectral_projection",
            V4_CHANNEL,
            "Official Train H/coordinates/Map/Setup; strict nested V4",
            primary=True,
        ),
        file_action(
            "E121_path",
            "E121 radial carrier plus unique path",
            "unique_path",
            root / "standalone_unique_path_carrier" / "strict_unique_path_projected_channel.npy",
            "Official Train H radial carrier plus official unique-path endpoint",
            primary=True,
        ),
        file_action(
            "E122_path",
            "E122 stack plus unique path",
            "unique_path",
            root / "strict_complex_stack_unique_path" / "strict_stacked_unique_path_channel.npy",
            "Fit-only official carrier stack plus official unique-path endpoint",
            primary=True,
        ),
        file_action(
            "E142_raw",
            "E142 denoised raw carrier",
            "sparse_denoising",
            root / "sparse_denoised_v7_carrier" / "oof_denoised_stacked_complex_channel.npy",
            "Official Train H sparse reconstruction and Clean carrier stack",
            scale=e142_nested_scale("raw", fold_id, site),
            primary=True,
        ),
        file_action(
            "E142_projected",
            "E142 denoised projected carrier",
            "sparse_denoising",
            root / "sparse_denoised_v7_carrier" / "oof_denoised_stacked_v4_projection.npy",
            "Official Train H sparse reconstruction plus official V4 targets",
            scale=e142_nested_scale("projected", fold_id, site),
            primary=True,
        ),
        file_action(
            "E161_knn",
            "E161 coefficient-field KNN",
            "coefficient_field",
            root / "e161_v7_correction_student" / "canonical_knn_complete.npy",
            "Official full4000 correction coefficients transferred by spatial KNN",
            primary=True,
        ),
        file_action(
            "E162_knn",
            "E162 relative-peak KNN",
            "relative_peak_transport",
            root / "e162_v7_relative_peak_relation" / "canonical_knn_relative_action_complete.npy",
            "Official V7-relative PAS/PDP actions transferred from official Train H",
            primary=True,
        ),
    ]

    primary_carriers = {
        0: "nearest_neighbor",
        7: "raw_idw",
        8: "radial_idw",
        11: "radial_affine",
        16: "array_steered_affine",
    }
    for index, family in primary_carriers.items():
        actions.append(
            ActionSpec(
                action_id=f"carrier_{index:02d}_{carrier_names[index]}",
                display_name=f"E44 carrier: {carrier_names[index]}",
                family=family,
                source=f"{PATHS.relative(CARRIER_BANK)}[{index}]",
                lineage="Strict OOF carrier generated only from fold-external official Train H and coordinates",
                provider="carrier",
                carrier_index=index,
                metric_source="e173",
                metric_key=str(index),
                primary=True,
            )
        )

    expanded_files = (
        ("V7_core", "V7 pre-lifecycle core", "v7_stage", root / "complex_carrier_lifecycle_overlay" / "oof_stacked_v4_projection_nested_scaled.npy"),
        ("E121_base", "E121 radial base", "strict_carrier_stage", root / "standalone_unique_path_carrier" / "strict_radial_idw_k24_channel.npy"),
        ("E122_stack", "E122 strict stack", "strict_carrier_stage", root / "strict_complex_stack_unique_path" / "strict_stacked_official_channel.npy"),
        ("E161_student_core", "E161 student core", "coefficient_field_stage", root / "e161_v7_correction_student" / "canonical_student_pre_lifecycle.npy"),
        ("E161_student", "E161 student complete", "coefficient_field_stage", root / "e161_v7_correction_student" / "canonical_student_complete.npy"),
        ("E161_knn_core", "E161 KNN core", "coefficient_field_stage", root / "e161_v7_correction_student" / "canonical_knn_pre_lifecycle.npy"),
        ("E162_knn_core", "E162 relative-peak KNN core", "relative_peak_stage", root / "e162_v7_relative_peak_relation" / "canonical_knn_relative_action_pre_lifecycle.npy"),
    )
    for action_id, display, family, path in expanded_files:
        actions.append(
            file_action(
                action_id,
                display,
                family,
                path,
                "Strict Clean historical complete-H stage; see E168 lineage audit",
            )
        )

    for index, name in enumerate(carrier_names):
        action_id = f"carrier_{index:02d}_{name}"
        if any(value.action_id == action_id for value in actions):
            continue
        actions.append(
            ActionSpec(
                action_id=action_id,
                display_name=f"E44 carrier: {name}",
                family="carrier_parameter_variant",
                source=f"{PATHS.relative(CARRIER_BANK)}[{index}]",
                lineage="Strict OOF carrier generated only from fold-external official Train H and coordinates",
                provider="carrier",
                carrier_index=index,
                metric_source="e173",
                metric_key=str(index),
            )
        )

    for regime in ("clear", "light", "medium", "heavy"):
        path = (
            root
            / "e179_v7_regime_oracle"
            / f"fixed500_v7_forced_{regime}_channel.npy"
        )
        actions.append(
            file_action(
                f"E179_forced_{regime}",
                f"E179 forced {regime}",
                "v7_regime_variant",
                path,
                "V7 with one fold-external Clean lifecycle regime forced at every point",
                metric_source="compute",
            )
        )

    additional_files = (
        (
            "foundation",
            "Pre-Sionna foundation",
            "foundation",
            root / "pre_sionna" / "round2_pre_sionna_oof_foundation.npy",
            "Official Train H and coordinates; original strict spatial-hole foundation",
        ),
        (
            "E45_raw_stack",
            "E45 raw complex stack",
            "raw_stack",
            root / "complex_carrier_stacking" / "oof_stacked_complex_channel.npy",
            "Fit-only 17-carrier complex ridge stack",
        ),
        (
            "E7_path_carrier_base",
            "E7 path carrier base",
            "continuous_path_carrier",
            root / "v4_path_component_transport" / "oof_path_carrier_k16_os2.npy",
            "Official Train H continuous component transport",
        ),
        (
            "E7_path_carrier_fine",
            "E7 path carrier fine direction",
            "continuous_path_carrier_variant",
            root / "v4_path_component_transport" / "oof_path_carrier_k16_os2_n40_dir3600_r1.npy",
            "Official Train H continuous component transport with historical fine direction grid",
        ),
        (
            "E7_path_carrier_refined",
            "E7 path carrier refined components",
            "continuous_path_carrier_variant",
            root / "v4_path_component_transport" / "oof_path_carrier_k16_os2_ref2_n40_dir3600_r1.npy",
            "Official Train H refined continuous component transport",
        ),
        (
            "E6_nmse_oriented_carrier",
            "E6 NMSE-oriented complex carrier",
            "structured_complex_field",
            root / "v4_structured_complex_field" / "oof_nmse_oriented_complex_carrier.npy",
            "Official Train H structured complex-field carrier",
        ),
        (
            "E6_nmse_oriented_projected",
            "E6 NMSE-oriented projected carrier",
            "structured_complex_field",
            root / "v4_structured_complex_field" / "oof_nmse_oriented_v3_projection.npy",
            "Official Train H structured complex field plus official V3 spectral projection",
        ),
    )
    for action_id, display, family, path, lineage in additional_files:
        actions.append(
            file_action(
                action_id,
                display,
                family,
                path,
                lineage,
                metric_source="compute",
            )
        )

    for phase_name, family in (
        ("polue", "static_polue_phase"),
        ("separable", "static_separable_phase"),
        ("full_link", "static_link_phase"),
    ):
        actions.append(
            ActionSpec(
                action_id=f"E26_{phase_name}_a1",
                display_name=f"E26 {phase_name} phase endpoint",
                family=family,
                source=f"{PATHS.relative(V4_CHANNEL)} * {PATHS.relative(PHASE_FACTORS)}[{phase_name}]",
                lineage="Fit-fold-only static phase factor applied to strict Clean V4 H",
                provider="phase",
                path=V4_CHANNEL,
                phase_factor=phase_name,
                metric_source="phase",
                metric_key=f"{phase_name}_a1.00",
            )
        )

    seen: set[str] = set()
    for action in actions:
        if action.action_id in seen:
            raise RuntimeError(f"duplicate action id: {action.action_id}")
        seen.add(action.action_id)
    return actions


class ActionProvider:
    def __init__(self, actions: list[ActionSpec]) -> None:
        self.actions = {action.action_id: action for action in actions}
        self.files: dict[Path, np.ndarray] = {}
        self.carrier_bank = np.load(CARRIER_BANK, mmap_mode="r")
        self.phase = np.load(PHASE_FACTORS, allow_pickle=False)

    def file(self, path: Path) -> np.ndarray:
        if path not in self.files:
            value = np.load(path, mmap_mode="r")
            if value.shape != EXPECTED_SHAPE or value.dtype != np.complex64:
                raise ValueError(f"invalid channel artifact {PATHS.relative(path)}: {value.shape}/{value.dtype}")
            self.files[path] = value
        return self.files[path]

    def block(self, action_id: str, start: int, end: int) -> np.ndarray:
        action = self.actions[action_id]
        if action.provider == "file":
            if action.path is None:
                raise RuntimeError(action.action_id)
            result = np.array(self.file(action.path)[start:end], dtype=np.complex64, copy=True)
        elif action.provider == "carrier":
            if action.carrier_index is None:
                raise RuntimeError(action.action_id)
            result = np.array(
                self.carrier_bank[action.carrier_index, start:end],
                dtype=np.complex64,
                copy=True,
            )
        elif action.provider == "phase":
            if action.path is None or action.phase_factor is None:
                raise RuntimeError(action.action_id)
            source = np.asarray(self.file(action.path)[start:end], dtype=np.complex64)
            factor = np.asarray(
                self.phase[action.phase_factor][start:end], dtype=np.complex64
            ).reshape(end - start, 256, 4, 1)
            result = np.asarray(source * factor, dtype=np.complex64)
        else:
            raise ValueError(f"unknown provider {action.provider}")
        if action.scale is not None:
            result *= np.asarray(action.scale[start:end], dtype=np.float32)[:, None, None, None]
        return result


def load_cached_metrics(actions: list[ActionSpec]) -> dict[str, PointMetricStatistics]:
    e168 = np.load(E168_METRICS, allow_pickle=False)
    e173 = np.load(E173_METRICS, allow_pickle=False)
    phase = np.load(PHASE_METRICS, allow_pickle=False)
    phase_names = phase["variant_names"].astype(str).tolist()
    output: dict[str, PointMetricStatistics] = {}
    for action in actions:
        if action.metric_source == "e168":
            key = str(action.metric_key)
            output[action.action_id] = statistics_from_arrays(
                e168[f"{key}_pas"],
                e168[f"{key}_pdp"],
                e168[f"{key}_squared_error"],
                e168[f"{key}_truth_energy"],
            )
        elif action.metric_source == "e173":
            index = int(str(action.metric_key))
            output[action.action_id] = statistics_from_arrays(
                e173["pas"][index],
                e173["pdp"][index],
                e173["error"][index],
                e173["energy"],
            )
        elif action.metric_source == "phase":
            index = phase_names.index(str(action.metric_key))
            output[action.action_id] = statistics_from_arrays(
                phase["pas"][index],
                phase["pdp"][index],
                phase["error"][index],
                phase["energy"][index],
            )
    return output


def compute_additional_metrics(
    actions: list[ActionSpec],
    provider: ActionProvider,
    query_ids: np.ndarray,
    truth_all: np.ndarray,
    device: torch.device,
    batch_size: int,
    force: bool,
) -> tuple[dict[str, PointMetricStatistics], float]:
    needed = [action for action in actions if action.metric_source == "compute"]
    if ADDITIONAL_METRICS.is_file() and not force:
        loaded = np.load(ADDITIONAL_METRICS, allow_pickle=False)
        names = loaded["action_ids"].astype(str).tolist()
        if names == [action.action_id for action in needed]:
            return (
                {
                    name: statistics_from_arrays(
                        loaded["pas"][index],
                        loaded["pdp"][index],
                        loaded["error"][index],
                        loaded["energy"][index],
                    )
                    for index, name in enumerate(names)
                },
                0.0,
            )

    started = time.perf_counter()
    arrays = {
        name: np.empty((len(needed), 500), dtype=np.float64)
        for name in ("pas", "pdp", "error", "energy")
    }
    with torch.inference_mode():
        for action_index, action in enumerate(needed):
            for start in range(0, 500, batch_size):
                end = min(start + batch_size, 500)
                prediction = torch.as_tensor(
                    provider.block(action.action_id, start, end), device=device
                )
                truth = torch.as_tensor(
                    np.asarray(truth_all[query_ids[start:end]], dtype=np.complex64),
                    device=device,
                )
                pas, pdp, error, energy, _ = point_metrics_torch(prediction, truth)
                arrays["pas"][action_index, start:end] = pas
                arrays["pdp"][action_index, start:end] = pdp
                arrays["error"][action_index, start:end] = error
                arrays["energy"][action_index, start:end] = energy
            print(f"E180 additional metrics {action_index + 1}/{len(needed)}: {action.action_id}", flush=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ADDITIONAL_METRICS,
        action_ids=np.asarray([action.action_id for action in needed]),
        **arrays,
    )
    output = {
        action.action_id: statistics_from_arrays(
            arrays["pas"][index],
            arrays["pdp"][index],
            arrays["error"][index],
            arrays["energy"][index],
        )
        for index, action in enumerate(needed)
    }
    return output, time.perf_counter() - started


def oracle_selection(
    action_ids: list[str], metrics: dict[str, PointMetricStatistics]
) -> tuple[np.ndarray, PointMetricStatistics, dict[str, float | int]]:
    utility = np.stack([local_scores(metrics[action_id]) for action_id in action_ids])
    maximum = np.max(utility, axis=0)
    selection = np.zeros(500, dtype=np.int16)
    for point in range(500):
        selection[point] = int(
            np.flatnonzero(utility[:, point] >= maximum[point] - TIE_TOLERANCE)[0]
        )
    rows = np.arange(500, dtype=np.int64)
    selected = statistics_from_arrays(
        np.stack([metrics[value].pas for value in action_ids])[selection, rows],
        np.stack([metrics[value].pdp for value in action_ids])[selection, rows],
        np.stack([metrics[value].squared_error for value in action_ids])[selection, rows],
        metrics[action_ids[0]].truth_energy,
    )
    return selection, selected, aggregate_statistics(selected)


def assemble_oracle(
    path: Path,
    action_ids: list[str],
    selection: np.ndarray,
    provider: ActionProvider,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.complex64, shape=EXPECTED_SHAPE
    )
    for action_index, action_id in enumerate(action_ids):
        rows = np.flatnonzero(selection == action_index)
        for start in range(0, len(rows), 8):
            local_rows = rows[start : start + 8]
            if len(local_rows) == 0:
                continue
            # Rows are not necessarily contiguous, so read one point at a time from the provider.
            for row in local_rows:
                output[row] = provider.block(action_id, int(row), int(row) + 1)[0]
    output.flush()


def official_score_channel(
    path: Path,
    truth_all: np.ndarray,
    query_ids: np.ndarray,
    batch_size: int,
) -> tuple[PointMetricStatistics, dict[str, float | int]]:
    channel = np.load(path, mmap_mode="r")
    if channel.shape != EXPECTED_SHAPE or channel.dtype != np.complex64:
        raise ValueError(f"invalid oracle H: {channel.shape}/{channel.dtype}")
    accumulator = OfficialMetricAccumulator()
    for start in range(0, 500, batch_size):
        end = min(start + batch_size, 500)
        prediction = np.asarray(channel[start:end], dtype=np.complex64)
        if not np.isfinite(prediction).all():
            raise ValueError(f"non-finite oracle channel rows {start}:{end}")
        accumulator.add(
            prediction,
            np.asarray(truth_all[query_ids[start:end]], dtype=np.complex64),
        )
    values = accumulator.statistics()
    return values, aggregate_statistics(values)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if float(np.std(left)) <= 1e-15 or float(np.std(right)) <= 1e-15:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def channel_change_statistics(
    actions: list[ActionSpec],
    provider: ActionProvider,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, dict[str, float]], float]:
    started = time.perf_counter()
    baseline = torch.as_tensor(np.asarray(np.load(V7_CHANNEL)), device=device)
    baseline_energy = float(baseline.abs().square().sum(dtype=torch.float64).item())
    baseline_norm = math.sqrt(max(baseline_energy, TINY))
    output: dict[str, dict[str, float]] = {}
    with torch.inference_mode():
        for action_index, action in enumerate(actions):
            difference_energy = 0.0
            action_energy = 0.0
            cross = 0.0
            for start in range(0, 500, batch_size):
                end = min(start + batch_size, 500)
                value = torch.as_tensor(
                    provider.block(action.action_id, start, end), device=device
                )
                reference = baseline[start:end]
                difference_energy += float(
                    (value - reference).abs().square().sum(dtype=torch.float64).item()
                )
                action_energy += float(value.abs().square().sum(dtype=torch.float64).item())
                cross += float((value * reference.conj()).real.sum(dtype=torch.float64).item())
            output[action.action_id] = {
                "relative_h_l2_change": math.sqrt(difference_energy / max(baseline_energy, TINY)),
                "real_h_cosine_vs_v7": cross
                / max(math.sqrt(action_energy) * baseline_norm, TINY),
            }
            print(f"E180 H difference {action_index + 1}/{len(actions)}: {action.action_id}", flush=True)
    return output, time.perf_counter() - started


def subset_metrics(
    values: PointMetricStatistics, mask: np.ndarray
) -> dict[str, float | int]:
    return aggregate_statistics(values, mask)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("E180 requires one CUDA GPU for complete-H auditing")
    torch.cuda.reset_peak_memory_stats()

    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    e168 = np.load(E168_METRICS, allow_pickle=False)
    site = e168["site"].astype(np.int8)
    if not np.array_equal(query_ids, e168["query_ids"].astype(np.int64)):
        raise RuntimeError("E168 canonical query order mismatch")
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")

    actions = build_actions(fold_id, site)
    provider = ActionProvider(actions)
    metrics = load_cached_metrics(actions)
    additional, additional_seconds = compute_additional_metrics(
        actions,
        provider,
        query_ids,
        truth_all,
        device,
        args.batch_size,
        args.force_additional_metrics,
    )
    metrics.update(additional)
    if set(metrics) != {action.action_id for action in actions}:
        missing = sorted({action.action_id for action in actions} - set(metrics))
        raise RuntimeError(f"missing action metrics: {missing}")

    baseline = aggregate_statistics(metrics["V7"])
    if abs(float(baseline["score"]) - EXPECTED_V7_SCORE) > 5e-6:
        raise RuntimeError(f"V7 identity gate failed: {baseline}")
    truth_reference = metrics["V7"].truth_energy
    for action in actions:
        if not np.allclose(metrics[action.action_id].truth_energy, truth_reference, rtol=1e-6, atol=1e-8):
            raise RuntimeError(f"truth energy mismatch: {action.action_id}")

    compact_ids = [action.action_id for action in actions if action.primary]
    expanded_ids = [action.action_id for action in actions]
    compact_selection, compact_cached_values, compact_cached = oracle_selection(
        compact_ids, metrics
    )
    expanded_selection, expanded_cached_values, expanded_cached = oracle_selection(
        expanded_ids, metrics
    )
    selection_seconds = time.perf_counter() - started - additional_seconds

    assemble_started = time.perf_counter()
    assemble_oracle(COMPACT_ORACLE_H, compact_ids, compact_selection, provider)
    assemble_oracle(EXPANDED_ORACLE_H, expanded_ids, expanded_selection, provider)
    assembly_seconds = time.perf_counter() - assemble_started

    scoring_started = time.perf_counter()
    compact_values, compact_official = official_score_channel(
        COMPACT_ORACLE_H, truth_all, query_ids, args.batch_size
    )
    expanded_values, expanded_official = official_score_channel(
        EXPANDED_ORACLE_H, truth_all, query_ids, args.batch_size
    )
    scoring_seconds = time.perf_counter() - scoring_started
    if abs(float(compact_official["score"]) - float(compact_cached["score"])) > 2e-5:
        raise RuntimeError(f"compact assembled-H identity failed: {compact_cached} vs {compact_official}")
    if abs(float(expanded_official["score"]) - float(expanded_cached["score"])) > 2e-5:
        raise RuntimeError(f"expanded assembled-H identity failed: {expanded_cached} vs {expanded_official}")

    change_stats, change_seconds = channel_change_statistics(
        actions, provider, device, args.batch_size
    )

    action_by_id = {action.action_id: action for action in actions}
    baseline_local = local_scores(metrics["V7"])
    action_rows: list[dict[str, object]] = []
    for action in actions:
        values = metrics[action.action_id]
        overall = aggregate_statistics(values)
        local = local_scores(values)
        win = local > baseline_local + TIE_TOLERANCE
        action_rows.append(
            {
                "action_id": action.action_id,
                "display_name": action.display_name,
                "family": action.family,
                "primary_compact": action.primary,
                "pas": overall["pas"],
                "pdp": overall["pdp"],
                "nmse": overall["nmse"],
                "score": overall["score"],
                "delta_vs_v7": float(overall["score"]) - float(baseline["score"]),
                "point_local_win_count_vs_v7": int(np.sum(win)),
                "point_local_win_fraction_vs_v7": float(np.mean(win)),
                "mean_positive_local_gain": float(np.mean((local - baseline_local)[win]))
                if np.any(win)
                else 0.0,
                "pas_error_corr_vs_v7": correlation(1.0 - values.pas, 1.0 - metrics["V7"].pas),
                "pdp_error_corr_vs_v7": correlation(1.0 - values.pdp, 1.0 - metrics["V7"].pdp),
                **change_stats[action.action_id],
                "source": action.source,
                "lineage": action.lineage,
            }
        )

    selection_rows: list[dict[str, object]] = []
    for library, ids, selection, values in (
        ("compact_distinct", compact_ids, compact_selection, compact_values),
        ("expanded_complete", expanded_ids, expanded_selection, expanded_values),
    ):
        chosen_local = local_scores(values)
        for index, action_id in enumerate(ids):
            selected = selection == index
            selection_rows.append(
                {
                    "library": library,
                    "action_id": action_id,
                    "family": action_by_id[action_id].family,
                    "selected_count": int(np.sum(selected)),
                    "selected_fraction": float(np.mean(selected)),
                    "diagnostic_local_gain_sum_vs_v7": float(
                        np.sum((chosen_local - baseline_local)[selected])
                    ),
                }
            )

    hole_rows: list[dict[str, object]] = []
    for library, values in (
        ("V7", metrics["V7"]),
        ("compact_distinct", compact_values),
        ("expanded_complete", expanded_values),
    ):
        for fold in range(5):
            for site_value in range(2):
                mask = (fold_id == fold) & (site == site_value)
                result = subset_metrics(values, mask)
                baseline_hole = subset_metrics(metrics["V7"], mask)
                hole_rows.append(
                    {
                        "library": library,
                        "hole_id": f"P_S{site_value}F{fold}",
                        **result,
                        "delta_vs_v7": float(result["score"]) - float(baseline_hole["score"]),
                    }
                )

    primary_families = sorted(
        {action_by_id[action_id].family for action_id in compact_ids if action_id != "V7"}
    )
    family_rows: list[dict[str, object]] = []
    for family in primary_families:
        family_ids = [
            action_id
            for action_id in compact_ids
            if action_id == "V7" or action_by_id[action_id].family == family
        ]
        _, _, family_result = oracle_selection(family_ids, metrics)
        without_ids = [
            action_id
            for action_id in compact_ids
            if action_id == "V7" or action_by_id[action_id].family != family
        ]
        _, _, without_result = oracle_selection(without_ids, metrics)
        family_rows.append(
            {
                "analysis": "family_alone_with_v7",
                "family": family,
                "action_count": len(family_ids) - 1,
                **family_result,
                "delta_vs_v7": float(family_result["score"]) - float(baseline["score"]),
                "compact_leave_out_drop": float(compact_cached["score"])
                - float(without_result["score"]),
            }
        )

    chosen_families: list[str] = []
    remaining = list(primary_families)
    current_score = float(baseline["score"])
    while remaining:
        candidates: list[tuple[float, str, dict[str, float | int]]] = []
        for family in remaining:
            allowed = {"baseline", *chosen_families, family}
            ids = [
                action_id
                for action_id in compact_ids
                if action_by_id[action_id].family in allowed
            ]
            _, _, result = oracle_selection(ids, metrics)
            candidates.append((float(result["score"]), family, result))
        score, family, result = max(candidates, key=lambda value: value[0])
        chosen_families.append(family)
        remaining.remove(family)
        family_rows.append(
            {
                "analysis": "greedy_cumulative",
                "family": family,
                "step": len(chosen_families),
                "action_count": sum(
                    action_by_id[action_id].family in {"baseline", *chosen_families}
                    for action_id in compact_ids
                )
                - 1,
                **result,
                "incremental_gain": score - current_score,
                "delta_vs_v7": score - float(baseline["score"]),
            }
        )
        current_score = score

    pair_rows: list[dict[str, object]] = []
    for left_index, left in enumerate(compact_ids):
        for right in compact_ids[left_index + 1 :]:
            pair_rows.append(
                {
                    "left": left,
                    "right": right,
                    "pas_point_error_corr": correlation(
                        1.0 - metrics[left].pas, 1.0 - metrics[right].pas
                    ),
                    "pdp_point_error_corr": correlation(
                        1.0 - metrics[left].pdp, 1.0 - metrics[right].pdp
                    ),
                }
            )

    e168_ids = e168["method_ids"].astype(str).tolist()
    e168_selection, _, e168_point_oracle = oracle_selection(e168_ids, {
        action_id: statistics_from_arrays(
            e168[f"{action_id}_pas"],
            e168[f"{action_id}_pdp"],
            e168[f"{action_id}_squared_error"],
            e168[f"{action_id}_truth_energy"],
        )
        for action_id in e168_ids
    })
    del e168_selection
    union_metrics = {
        action_id: statistics_from_arrays(
            e168[f"{action_id}_pas"],
            e168[f"{action_id}_pdp"],
            e168[f"{action_id}_squared_error"],
            e168[f"{action_id}_truth_energy"],
        )
        for action_id in e168_ids
    }
    carrier_loaded = np.load(E173_METRICS, allow_pickle=False)
    union_ids = list(e168_ids)
    for index in range(17):
        name = f"supplemental_carrier_{index}"
        union_ids.append(name)
        union_metrics[name] = statistics_from_arrays(
            carrier_loaded["pas"][index],
            carrier_loaded["pdp"][index],
            carrier_loaded["error"][index],
            carrier_loaded["energy"],
        )
    _, _, metric_union_oracle = oracle_selection(union_ids, union_metrics)

    admission_rows: list[dict[str, object]] = []
    for action in actions:
        admission_rows.append(
            {
                "action_id": action.action_id,
                "family": action.family,
                "status": "CLEAN_PRIMARY_COMPLETE_H" if action.primary else "CLEAN_EXPANDED_COMPLETE_H",
                "complete_h": True,
                "primary_compact": action.primary,
                "source": action.source,
                "lineage": action.lineage,
            }
        )
    complete_metric_keys = {
        str(action.metric_key)
        for action in actions
        if action.metric_source == "e168" and action.metric_key is not None
    }
    with E168_ADMISSION.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            method_id = str(row["method_id"])
            if method_id in complete_metric_keys:
                continue
            admission_rows.append(
                {
                    "action_id": method_id,
                    "family": str(row["family"]),
                    "status": "CLEAN_SUPPLEMENTAL_POINT_METRICS_ONLY",
                    "complete_h": False,
                    "primary_compact": False,
                    "source": str(row["source"]),
                    "lineage": str(row["lineage"]),
                }
            )
    for action_id, status, reason in (
        ("E138_held_coefficient_oracle", "EXCLUDED_ORACLE", "uses held GT coefficients"),
        ("E162_predictable_subspace_oracle", "EXCLUDED_ORACLE", "uses held GT action values"),
        ("E163_transport_oracles", "EXCLUDED_DIRTY", "derived from V12 transport states"),
        ("V8_to_V12", "EXCLUDED_DIRTY", "uses Sionna test-position information or derivatives"),
    ):
        admission_rows.append(
            {
                "action_id": action_id,
                "status": status,
                "complete_h": False,
                "primary_compact": False,
                "reason": reason,
            }
        )

    write_csv(ADMISSION_CSV, admission_rows)
    write_csv(ACTION_CSV, action_rows)
    write_csv(SELECTION_CSV, selection_rows)
    write_csv(HOLE_CSV, hole_rows)
    write_csv(FAMILY_CSV, family_rows)
    write_csv(PAIR_CSV, pair_rows)

    elapsed = time.perf_counter() - started
    peak_bytes = int(torch.cuda.max_memory_allocated())
    report = {
        "experiment": VERSION,
        "status": "action_space_sufficient" if float(compact_official["score"]) >= SUCCESS_SCORE else "action_space_insufficient",
        "official_inputs_only": True,
        "sionna_used": False,
        "held_gt_used_for_oracle_selection": True,
        "baseline_v7": baseline,
        "success_score": SUCCESS_SCORE,
        "compact_distinct": {
            "action_ids": compact_ids,
            "cached_point_aggregate": compact_cached,
            "assembled_official": compact_official,
            "delta_vs_v7": float(compact_official["score"]) - float(baseline["score"]),
            "gap_vs_0_66": float(compact_official["score"]) - SUCCESS_SCORE,
            "gap_vs_v12_local": float(compact_official["score"]) - TARGET_LOCAL_SCORE,
            "output": PATHS.relative(COMPACT_ORACLE_H),
            "shape": list(np.load(COMPACT_ORACLE_H, mmap_mode="r").shape),
            "dtype": str(np.load(COMPACT_ORACLE_H, mmap_mode="r").dtype),
            "finite": True,
        },
        "expanded_complete": {
            "action_count": len(expanded_ids),
            "cached_point_aggregate": expanded_cached,
            "assembled_official": expanded_official,
            "delta_vs_v7": float(expanded_official["score"]) - float(baseline["score"]),
            "output": PATHS.relative(EXPANDED_ORACLE_H),
            "shape": list(np.load(EXPANDED_ORACLE_H, mmap_mode="r").shape),
            "dtype": str(np.load(EXPANDED_ORACLE_H, mmap_mode="r").dtype),
            "finite": True,
        },
        "supplemental_metric_only": {
            "e168_34_method_point_oracle": e168_point_oracle,
            "e168_plus_17_carrier_point_oracle": metric_union_oracle,
            "warning": "Exact aggregate of official point statistics; not the primary assembled-H result.",
        },
        "performance": {
            "device": torch.cuda.get_device_name(device),
            "cuda_visible_device": str(device),
            "peak_torch_bytes": peak_bytes,
            "additional_metric_seconds": additional_seconds,
            "selection_and_setup_seconds": selection_seconds,
            "assembly_seconds": assembly_seconds,
            "official_scoring_seconds": scoring_seconds,
            "channel_change_seconds": change_seconds,
            "total_seconds": elapsed,
        },
        "files": {
            "admission_csv": PATHS.relative(ADMISSION_CSV),
            "action_csv": PATHS.relative(ACTION_CSV),
            "selection_csv": PATHS.relative(SELECTION_CSV),
            "hole_csv": PATHS.relative(HOLE_CSV),
            "family_csv": PATHS.relative(FAMILY_CSV),
            "pair_csv": PATHS.relative(PAIR_CSV),
        },
    }
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(
        json.dumps(json_compatible(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    compact_counts = [
        row
        for row in selection_rows
        if row["library"] == "compact_distinct" and int(row["selected_count"]) > 0
    ]
    compact_counts.sort(key=lambda row: int(row["selected_count"]), reverse=True)
    top_actions = "\n".join(
        f"| {row['action_id']} | {row['family']} | {row['selected_count']} | {float(row['diagnostic_local_gain_sum_vs_v7']):.6f} |"
        for row in compact_counts
    )
    hole_compact = [row for row in hole_rows if row["library"] == "compact_distinct"]
    hole_table = "\n".join(
        f"| {row['hole_id']} | {int(row['count'])} | {float(row['pas']):.6f} | {float(row['pdp']):.6f} | {float(row['nmse']):.6f} | {float(row['score']):.6f} | {float(row['delta_vs_v7']):+.6f} |"
        for row in hole_compact
    )
    family_alone = [row for row in family_rows if row["analysis"] == "family_alone_with_v7"]
    family_alone.sort(key=lambda row: float(row["compact_leave_out_drop"]), reverse=True)
    family_table = "\n".join(
        f"| {row['family']} | {float(row['score']):.6f} | {float(row['delta_vs_v7']):+.6f} | {float(row['compact_leave_out_drop']):+.6f} |"
        for row in family_alone
    )
    REPORT_MD.write_text(
        f"""# 历史Clean局部修正动作库逐点上限

## 最终裁决

本轮是不可部署的held-GT逐点动作oracle，不是新模型，也不是可提交分数。严格按机制去重的13动作主库实际拼接完整500点H后，官方PAS/PDP/NMSE/总分为：

```text
PAS   = {float(compact_official['pas']):.9f}
PDP   = {float(compact_official['pdp']):.9f}
NMSE  = {float(compact_official['nmse']):.9f}
score = {float(compact_official['score']):.9f}
```

相对V7提高`{float(compact_official['score']) - float(baseline['score']):+.9f}`，高于预注册`0.66`门槛`{float(compact_official['score']) - SUCCESS_SCORE:+.9f}`，距离V12本地`0.677521`为`{float(compact_official['score']) - TARGET_LOCAL_SCORE:+.9f}`。因此问题的直接答案是：**有。现有严格Clean历史动作已经足够强、足够互补，动作空间本身不是0.66以下的瓶颈。** 但本轮没有证明测试时能只靠Clean信息选对动作。

包含阶段和参数变体的完整H扩展库共有{len(expanded_ids)}个动作，实际拼接官方总分=`{float(expanded_official['score']):.9f}`。E168的34个Clean方法只用已核验逐点官方统计计算时为`{float(e168_point_oracle['score']):.9f}`；再加17载体为`{float(metric_union_oracle['score']):.9f}`。后两者没有作为主结果，因为部分历史方法未保留完整H。

## 主库动作选择

| 动作 | 家族 | 选中点数 | 诊断局部增益和 |
| --- | --- | ---: | ---: |
{top_actions}

## 十个洞

| pseudo-hole | 点数 | PAS | PDP | NMSE | 总分 | 相对V7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{hole_table}

## 家族贡献

`V7+该家族`表示只允许V7和该家族竞争；`完整库移除损失`表示从13动作主库删除该家族后oracle下降多少。

| 家族 | V7+家族总分 | 相对V7 | 完整库移除损失 |
| --- | ---: | ---: | ---: |
{family_table}

## 边界

主库没有使用Sionna、V8-V12、held系数oracle、E162 held动作oracle或E163运输动作。E145/E154等只保留逐点官方统计、没有现成完整H的方法仅进入补充上限。所有主库动作都有canonical `(500,256,4,192) complex64`完整OOF H；两个oracle输出均finite，并由项目官方评分实现重新计算。

本轮不训练gate、不分类、不做soft blend、不调阈值或动作强度、不生成测试H或提交版本。达到`0.66`只证明“可选答案库够用”，没有解决“如何在未知GT时选对答案”。
""",
        encoding="utf-8",
    )

    print(json.dumps(json_compatible(report), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
