#!/usr/bin/env python3
"""Train and audit E183 hole-specific Clean action selectors."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import resource
import time

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer

from common.official_metrics import PointMetricStatistics, aggregate_statistics
from project_paths import PATHS
import score_e183_clean_action_labels as labels_module


VERSION = "Round2-E183-hole-specific-clean-action-selector"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e183_hole_specific_selector"
FEATURE_DIR = ARTIFACT_DIR / "features"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e183_hole_specific_selector"
AUDIT_REPORT = ANALYSIS_DIR / "strict_split_audit.json"
OOF_REPORT = ANALYSIS_DIR / "pseudo_hole_selector_oof.json"
FAMILY_CSV = ANALYSIS_DIR / "six_family_selector_summary.csv"
FOLD_CSV = ANALYSIS_DIR / "strict_oof_fold_results.csv"
HOLE_CSV = ANALYSIS_DIR / "strict_oof_hole_results.csv"
SELECTION_CSV = ANALYSIS_DIR / "strict_oof_action_selections.csv"
OOF_ARTIFACT = ARTIFACT_DIR / "strict_pseudo_hole_oof_predictions.npz"

LABEL_DIR = PATHS.artifacts / "round2" / "e183_strict_action_labels" / "labels"
CANONICAL_META = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_pre_sionna_oof_foundation_meta.npz"
)
SPECIALIZED_FAMILIES = (
    "S0H0",
    "S0H1",
    "S0H2",
    "S0H3",
    "S1H1",
    "S1H3",
)
ACTION_NAMES = labels_module.ACTION_NAMES
TINY = 1e-30
TIE_TOLERANCE = 1e-12
SEED = 20_260_812_183

MODEL_CONFIGS = (
    ("v7_only", None),
    ("shared_gain_regression", 0.005),
    ("multioutput_gain_regression", 0.005),
    ("positive_gain_ranker", 0.15),
)
PRIMARY_MODEL = "shared_gain_regression"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("audit", "oof"), default="audit")
    parser.add_argument("--families", default=",".join(SPECIALIZED_FAMILIES))
    parser.add_argument("--trees", type=int, default=128)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def json_value(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class FamilyData:
    def __init__(self, family: str) -> None:
        self.family = family
        feature_path = FEATURE_DIR / f"{family}_clean_features.npz"
        label_path = LABEL_DIR / f"{family}_action_utility_labels.npz"
        if not feature_path.is_file():
            raise FileNotFoundError(f"missing feature artifact: {feature_path}")
        if not label_path.is_file():
            raise FileNotFoundError(f"missing label artifact: {label_path}")
        feature = np.load(feature_path, allow_pickle=False)
        label = np.load(label_path, allow_pickle=False)
        self.action_names = tuple(str(value) for value in feature["action_names"].tolist())
        label_actions = tuple(str(value) for value in label["action_names"].tolist())
        if self.action_names != ACTION_NAMES or label_actions != ACTION_NAMES:
            raise RuntimeError(f"{family} action identity changed")
        self.hidden_global_ids = np.asarray(feature["hidden_global_ids"], dtype=np.int64)
        if not np.array_equal(
            self.hidden_global_ids,
            np.asarray(label["hidden_global_ids"], dtype=np.int64),
        ):
            raise RuntimeError(f"{family} hidden ID order mismatch")
        self.point_context = np.asarray(feature["point_context"], dtype=np.float32)
        self.point_context_names = tuple(
            str(value) for value in feature["point_context_names"].tolist()
        )
        self.action_descriptors = np.asarray(
            feature["action_descriptors"], dtype=np.float32
        )
        self.action_descriptor_names = tuple(
            str(value) for value in feature["action_descriptor_names"].tolist()
        )
        self.occurrence_point_rows = np.asarray(
            feature["occurrence_point_rows"], dtype=np.int64
        )
        self.occurrence_hole = np.asarray(
            feature["occurrence_hole_index"], dtype=np.int16
        )
        self.occurrence_features = np.asarray(
            feature["occurrence_features"], dtype=np.float32
        )
        self.occurrence_feature_names = tuple(
            str(value) for value in feature["occurrence_feature_names"].tolist()
        )
        self.hole_offsets = np.asarray(feature["hole_offsets"], dtype=np.int64)
        if not np.array_equal(
            self.occurrence_point_rows,
            np.asarray(label["hole_test_rows_flat"], dtype=np.int64),
        ):
            raise RuntimeError(f"{family} feature/label occurrence order mismatch")
        if not np.array_equal(
            self.hole_offsets, np.asarray(label["hole_offsets"], dtype=np.int64)
        ):
            raise RuntimeError(f"{family} feature/label hole offsets changed")
        self.pas = np.asarray(label["pas"], dtype=np.float64)
        self.pdp = np.asarray(label["pdp"], dtype=np.float64)
        self.error = np.asarray(label["squared_error"], dtype=np.float64)
        self.energy = np.asarray(label["truth_energy"], dtype=np.float64)
        self.utility = np.asarray(label["utility"], dtype=np.float64)
        self.gain = self.utility - self.utility[0:1]
        self.base_context = np.column_stack(
            [self.point_context[self.occurrence_point_rows], self.occurrence_features]
        ).astype(np.float32)
        self.hole_fold = np.arange(30, dtype=np.int8) % 5
        self.occurrence_fold = self.hole_fold[self.occurrence_hole]
        self.global_id_by_occurrence = self.hidden_global_ids[
            self.occurrence_point_rows
        ]
        self._validate()

    def _validate(self) -> None:
        point_count = len(self.hidden_global_ids)
        occurrence_count = len(self.occurrence_point_rows)
        expected = {
            "point_context": (point_count, self.point_context.shape[1]),
            "action_descriptors": (
                point_count,
                len(ACTION_NAMES),
                len(self.action_descriptor_names),
            ),
            "occurrence_features": (
                occurrence_count,
                len(self.occurrence_feature_names),
            ),
            "pas": (len(ACTION_NAMES), point_count),
            "pdp": (len(ACTION_NAMES), point_count),
            "error": (len(ACTION_NAMES), point_count),
            "utility": (len(ACTION_NAMES), point_count),
        }
        actual = {
            "point_context": self.point_context.shape,
            "action_descriptors": self.action_descriptors.shape,
            "occurrence_features": self.occurrence_features.shape,
            "pas": self.pas.shape,
            "pdp": self.pdp.shape,
            "error": self.error.shape,
            "utility": self.utility.shape,
        }
        for name, shape in expected.items():
            if actual[name] != shape:
                raise RuntimeError(
                    f"{self.family} invalid {name}: {actual[name]} != {shape}"
                )
        if len(self.hole_offsets) != 31:
            raise RuntimeError(f"{self.family} must contain 30 holes")
        if self.hole_offsets[0] != 0 or self.hole_offsets[-1] != occurrence_count:
            raise RuntimeError(f"{self.family} invalid occurrence offsets")
        for hole in range(30):
            start, end = self.hole_offsets[hole : hole + 2]
            if not np.all(self.occurrence_hole[start:end] == hole):
                raise RuntimeError(f"{self.family} occurrence hole order changed")
        for name, value in (
            ("point_context", self.point_context),
            ("action_descriptors", self.action_descriptors),
            ("occurrence_features", self.occurrence_features),
            ("utility", self.utility),
        ):
            if not np.isfinite(value).all():
                raise RuntimeError(f"{self.family} non-finite {name}")

    def split(self, held_holes: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        held_holes = np.asarray(held_holes, dtype=np.int16)
        valid_occurrence = np.isin(self.occurrence_hole, held_holes)
        valid_ids = np.unique(self.global_id_by_occurrence[valid_occurrence])
        candidate_train = ~valid_occurrence
        train_occurrence = candidate_train & ~np.isin(
            self.global_id_by_occurrence, valid_ids
        )
        train_ids = np.unique(self.global_id_by_occurrence[train_occurrence])
        overlap = np.intersect1d(train_ids, valid_ids)
        if len(overlap):
            raise RuntimeError(f"{self.family} train/valid utility label overlap")
        for hole in held_holes:
            start, end = self.hole_offsets[hole : hole + 2]
            if not np.all(valid_occurrence[start:end]):
                raise RuntimeError(f"{self.family} held hole {hole} is incomplete")
        return train_occurrence, valid_occurrence, {
            "train_occurrence_count": int(np.sum(train_occurrence)),
            "valid_occurrence_count": int(np.sum(valid_occurrence)),
            "train_unique_point_count": len(train_ids),
            "valid_unique_point_count": len(valid_ids),
            "label_point_overlap_count": len(overlap),
        }


def occurrence_weights(point_rows: np.ndarray) -> np.ndarray:
    unique, inverse, counts = np.unique(
        np.asarray(point_rows, dtype=np.int64), return_inverse=True, return_counts=True
    )
    del unique
    return (1.0 / counts[inverse]).astype(np.float64)


def imputed(
    train: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, SimpleImputer]:
    transformer = SimpleImputer(strategy="median", add_indicator=True)
    return (
        transformer.fit_transform(train).astype(np.float32),
        transformer.transform(valid).astype(np.float32),
        transformer,
    )


def model_seed(seed: int, family: str, outer: int, model: str) -> int:
    stable = sum((index + 1) * ord(value) for index, value in enumerate(family + model))
    return (seed + stable * 1009 + outer * 104729) % (2**32)


def shared_design(
    data: FamilyData, occurrence_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.flatnonzero(occurrence_mask)
    point_rows = data.occurrence_point_rows[rows]
    context = data.base_context[rows]
    action_count = len(ACTION_NAMES) - 1
    context_repeated = np.repeat(context, action_count, axis=0)
    descriptors = data.action_descriptors[point_rows, 1:].reshape(
        len(rows) * action_count, -1
    )
    one_hot = np.tile(np.eye(action_count, dtype=np.float32), (len(rows), 1))
    design = np.column_stack([context_repeated, descriptors, one_hot]).astype(np.float32)
    target = data.gain[1:, point_rows].T.reshape(-1).astype(np.float32)
    weights = np.repeat(occurrence_weights(point_rows), action_count)
    return design, target, weights


def multioutput_design(
    data: FamilyData, occurrence_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.flatnonzero(occurrence_mask)
    point_rows = data.occurrence_point_rows[rows]
    descriptor = data.action_descriptors[point_rows, 1:].reshape(len(rows), -1)
    design = np.column_stack([data.base_context[rows], descriptor]).astype(np.float32)
    target = data.gain[1:, point_rows].T.astype(np.float32)
    return design, target, occurrence_weights(point_rows)


def fit_predict_model(
    data: FamilyData,
    model_name: str,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
    trees: int,
    seed: int,
) -> np.ndarray:
    valid_rows = np.flatnonzero(valid_mask)
    prediction = np.zeros((len(valid_rows), len(ACTION_NAMES)), dtype=np.float32)
    if model_name == "v7_only":
        return prediction
    if model_name in {"shared_gain_regression", "positive_gain_ranker"}:
        train_design, train_target, train_weight = shared_design(data, train_mask)
        valid_design, _, _ = shared_design(data, valid_mask)
        train_design, valid_design, _ = imputed(train_design, valid_design)
        action_count = len(ACTION_NAMES) - 1
        if model_name == "shared_gain_regression":
            model = ExtraTreesRegressor(
                n_estimators=trees,
                max_depth=8,
                min_samples_leaf=8,
                max_features="sqrt",
                n_jobs=-1,
                random_state=seed,
            )
            model.fit(train_design, train_target, sample_weight=train_weight)
            local = model.predict(valid_design).reshape(len(valid_rows), action_count)
        else:
            target = train_target > TIE_TOLERANCE
            if len(np.unique(target)) < 2:
                probability = np.full(
                    len(valid_design), float(np.mean(target)), dtype=np.float32
                )
            else:
                model = ExtraTreesClassifier(
                    n_estimators=trees,
                    max_depth=8,
                    min_samples_leaf=8,
                    max_features="sqrt",
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=seed,
                )
                model.fit(train_design, target, sample_weight=train_weight)
                probability = model.predict_proba(valid_design)[:, 1]
            local = probability.reshape(len(valid_rows), action_count) - 0.5
        prediction[:, 1:] = np.asarray(local, dtype=np.float32)
        return prediction
    if model_name == "multioutput_gain_regression":
        train_design, train_target, train_weight = multioutput_design(data, train_mask)
        valid_design, _, _ = multioutput_design(data, valid_mask)
        train_design, valid_design, _ = imputed(train_design, valid_design)
        model = ExtraTreesRegressor(
            n_estimators=trees,
            max_depth=8,
            min_samples_leaf=8,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
        )
        model.fit(train_design, train_target, sample_weight=train_weight)
        prediction[:, 1:] = np.asarray(model.predict(valid_design), dtype=np.float32)
        return prediction
    raise ValueError(model_name)


def select_actions(prediction: np.ndarray, model_name: str) -> np.ndarray:
    threshold = dict(MODEL_CONFIGS)[model_name]
    if threshold is None:
        return np.zeros(len(prediction), dtype=np.int16)
    best_nonzero = np.argmax(prediction[:, 1:], axis=1).astype(np.int16) + 1
    best_value = prediction[np.arange(len(prediction)), best_nonzero]
    return np.where(best_value > float(threshold), best_nonzero, 0).astype(np.int16)


def statistics_for_occurrences(
    data: FamilyData,
    occurrence_rows: np.ndarray,
    selection: np.ndarray,
) -> PointMetricStatistics:
    point_rows = data.occurrence_point_rows[occurrence_rows]
    return PointMetricStatistics(
        pas=data.pas[selection, point_rows],
        pdp=data.pdp[selection, point_rows],
        squared_error=data.error[selection, point_rows],
        truth_energy=data.energy[point_rows],
    )


def hole_result(
    data: FamilyData,
    hole: int,
    occurrence_to_selection: dict[int, int],
) -> tuple[dict[str, float | int], dict[str, float | int], dict[str, float | int]]:
    start, end = data.hole_offsets[hole : hole + 2]
    occurrence_rows = np.arange(start, end, dtype=np.int64)
    selection = np.asarray(
        [occurrence_to_selection[int(row)] for row in occurrence_rows], dtype=np.int16
    )
    baseline = np.zeros(len(occurrence_rows), dtype=np.int16)
    point_rows = data.occurrence_point_rows[occurrence_rows]
    oracle = np.argmax(
        data.utility[:, point_rows]
        >= data.utility[:, point_rows].max(axis=0, keepdims=True) - TIE_TOLERANCE,
        axis=0,
    ).astype(np.int16)
    return (
        aggregate_statistics(statistics_for_occurrences(data, occurrence_rows, selection)),
        aggregate_statistics(statistics_for_occurrences(data, occurrence_rows, baseline)),
        aggregate_statistics(statistics_for_occurrences(data, occurrence_rows, oracle)),
    )


def evaluate_prediction(
    data: FamilyData,
    valid_mask: np.ndarray,
    prediction: np.ndarray,
    model_name: str,
) -> tuple[list[dict[str, object]], dict[str, object], list[dict[str, object]]]:
    valid_rows = np.flatnonzero(valid_mask)
    selection = select_actions(prediction, model_name)
    occurrence_to_selection = {
        int(row): int(action) for row, action in zip(valid_rows, selection, strict=True)
    }
    holes = sorted(np.unique(data.occurrence_hole[valid_rows]).tolist())
    hole_rows: list[dict[str, object]] = []
    for hole in holes:
        selected, baseline, oracle = hole_result(data, hole, occurrence_to_selection)
        hole_rows.append(
            {
                "family": data.family,
                "hole_index": hole,
                "outer_fold": int(data.hole_fold[hole]),
                "model": model_name,
                "V7_pas": baseline["pas"],
                "V7_pdp": baseline["pdp"],
                "V7_nmse": baseline["nmse"],
                "V7_score": baseline["score"],
                "selector_pas": selected["pas"],
                "selector_pdp": selected["pdp"],
                "selector_nmse": selected["nmse"],
                "selector_score": selected["score"],
                "selector_delta": float(selected["score"]) - float(baseline["score"]),
                "oracle_score": oracle["score"],
                "oracle_gain": float(oracle["score"]) - float(baseline["score"]),
            }
        )
    point_rows = data.occurrence_point_rows[valid_rows]
    selected_gain = data.gain[selection, point_rows]
    best_utility = data.utility[:, point_rows].max(axis=0)
    regret = np.maximum(best_utility - data.utility[selection, point_rows], 0.0)
    wrong = (selection != 0) & (selected_gain < -TIE_TOLERANCE)
    selection_rows = [
        {
            "family": data.family,
            "occurrence_row": int(occurrence),
            "hole_index": int(data.occurrence_hole[occurrence]),
            "outer_fold": int(data.occurrence_fold[occurrence]),
            "point_row": int(point),
            "global_point_id": int(data.hidden_global_ids[point]),
            "model": model_name,
            "selected_action": ACTION_NAMES[int(action)],
            "predicted_value": float(prediction[index, action]),
            "true_gain_vs_v7": float(selected_gain[index]),
            "regret_vs_best": float(regret[index]),
            "wrong_negative_switch": bool(wrong[index]),
        }
        for index, (occurrence, point, action) in enumerate(
            zip(valid_rows, point_rows, selection, strict=True)
        )
    ]
    summary = {
        "valid_hole_count": len(holes),
        "valid_occurrence_count": len(valid_rows),
        "switch_occurrence_count": int(np.sum(selection != 0)),
        "fallback_occurrence_count": int(np.sum(selection == 0)),
        "switch_unique_point_count": int(
            len(np.unique(point_rows[selection != 0]))
        ),
        "wrong_negative_switch_count": int(np.sum(wrong)),
        "wrong_negative_switch_loss_sum": float(-np.sum(selected_gain[wrong])),
        "wrong_negative_switch_loss_mean": (
            float(-np.mean(selected_gain[wrong])) if np.any(wrong) else 0.0
        ),
        "mean_regret_vs_best": float(np.mean(regret)),
        "mean_hole_score": float(np.mean([row["selector_score"] for row in hole_rows])),
        "mean_hole_v7_score": float(np.mean([row["V7_score"] for row in hole_rows])),
        "mean_hole_oracle_score": float(np.mean([row["oracle_score"] for row in hole_rows])),
    }
    return hole_rows, summary, selection_rows


def audit_family(data: FamilyData) -> dict[str, object]:
    fold_rows = []
    for outer in range(5):
        held = np.flatnonzero(data.hole_fold == outer)
        train_mask, valid_mask, split = data.split(held)
        fold_rows.append(
            {
                "outer_fold": outer,
                "held_holes": held.tolist(),
                **split,
                "held_holes_complete": bool(
                    all(
                        np.all(valid_mask[data.hole_offsets[hole] : data.hole_offsets[hole + 1]])
                        for hole in held
                    )
                ),
            }
        )
    canonical_ids = set(
        np.load(CANONICAL_META, allow_pickle=False)["query_ids"].astype(int).tolist()
    )
    canonical_overlap = canonical_ids & set(data.hidden_global_ids.tolist())
    return {
        "family": data.family,
        "point_count": len(data.hidden_global_ids),
        "occurrence_count": len(data.occurrence_point_rows),
        "point_context_count": len(data.point_context_names),
        "action_descriptor_count": len(data.action_descriptor_names),
        "occurrence_feature_count": len(data.occurrence_feature_names),
        "folds": fold_rows,
        "all_zero_label_overlap": all(
            row["label_point_overlap_count"] == 0 for row in fold_rows
        ),
        "all_held_holes_complete": all(row["held_holes_complete"] for row in fold_rows),
        "canonical_point_overlap_count": len(canonical_overlap),
        "canonical_strict_rebuild_required": len(canonical_overlap) > 0,
    }


def run_audit(families: tuple[str, ...]) -> dict[str, object]:
    started = time.perf_counter()
    family_rows = [audit_family(FamilyData(family)) for family in families]
    passed = all(
        row["all_zero_label_overlap"] and row["all_held_holes_complete"]
        for row in family_rows
    )
    report = {
        "experiment": VERSION,
        "phase": "strict_split_and_lineage_audit",
        "status": "audit_passed" if passed else "audit_failed",
        "official_inputs_only": True,
        "sionna_used": False,
        "v8_v12_used": False,
        "families": family_rows,
        "gate": {
            "six_family_count": len(family_rows),
            "all_folds_zero_label_overlap": passed,
            "canonical_score_must_use_additional_fold-isolated_rebuild": True,
        },
        "performance": {
            "wall_seconds": time.perf_counter() - started,
            "device": "CPU",
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (1024.0**2),
        },
    }
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_REPORT.write_text(
        json.dumps(json_value(report), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return report


def run_oof(
    families: tuple[str, ...], trees: int, seed: int, smoke: bool
) -> dict[str, object]:
    audit = json.loads(AUDIT_REPORT.read_text(encoding="utf-8"))
    if audit.get("status") != "audit_passed":
        raise RuntimeError("strict split audit has not passed")
    started = time.perf_counter()
    family_summaries: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    hole_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    artifact_family: list[str] = []
    artifact_occurrence: list[int] = []
    artifact_prediction: list[np.ndarray] = []
    artifact_selection: list[int] = []
    for family_index, family in enumerate(families):
        data = FamilyData(family)
        family_holes: list[dict[str, object]] = []
        family_selections: list[dict[str, object]] = []
        outer_count = 1 if smoke else 5
        for outer in range(outer_count):
            held_holes = np.flatnonzero(data.hole_fold == outer)
            train_mask, valid_mask, split = data.split(held_holes)
            for model_name, _ in MODEL_CONFIGS:
                prediction = fit_predict_model(
                    data,
                    model_name,
                    train_mask,
                    valid_mask,
                    trees,
                    model_seed(seed, family, outer, model_name),
                )
                local_holes, summary, local_selection = evaluate_prediction(
                    data, valid_mask, prediction, model_name
                )
                fold_rows.append(
                    {
                        "family": family,
                        "outer_fold": outer,
                        "held_holes": ";".join(str(value) for value in held_holes),
                        "model": model_name,
                        "is_primary_model": model_name == PRIMARY_MODEL,
                        "fallback_threshold": dict(MODEL_CONFIGS)[model_name],
                        **split,
                        **summary,
                        "delta_vs_v7": float(summary["mean_hole_score"])
                        - float(summary["mean_hole_v7_score"]),
                    }
                )
                if model_name != PRIMARY_MODEL:
                    continue
                family_holes.extend(local_holes)
                family_selections.extend(local_selection)
                valid_rows = np.flatnonzero(valid_mask)
                selected = select_actions(prediction, model_name)
                artifact_family.extend([family] * len(valid_rows))
                artifact_occurrence.extend(valid_rows.tolist())
                artifact_prediction.extend(prediction)
                artifact_selection.extend(selected.tolist())
                print(
                    f"E183 selector {family} fold {outer}: model={model_name} "
                    f"delta={float(summary['mean_hole_score']) - float(summary['mean_hole_v7_score']):+.6f}",
                    flush=True,
                )
        hole_rows.extend(family_holes)
        selection_rows.extend(family_selections)
        v7_score = float(np.mean([row["V7_score"] for row in family_holes]))
        selector_score = float(
            np.mean([row["selector_score"] for row in family_holes])
        )
        oracle_score = float(np.mean([row["oracle_score"] for row in family_holes]))
        selector_gain = selector_score - v7_score
        oracle_gain = oracle_score - v7_score
        wrong = [row for row in family_selections if row["wrong_negative_switch"]]
        family_summaries.append(
            {
                "family": family,
                "evaluated_hole_count": len(family_holes),
                "V7_score": v7_score,
                "selector_score": selector_score,
                "selector_gain": selector_gain,
                "oracle_score": oracle_score,
                "oracle_gain": oracle_gain,
                "oracle_gain_recovered_fraction": selector_gain / max(oracle_gain, TINY),
                "positive_hole_count": int(
                    sum(float(row["selector_delta"]) > 0.0 for row in family_holes)
                ),
                "switch_occurrence_count": int(
                    sum(row["selected_action"] != "V7" for row in family_selections)
                ),
                "fallback_occurrence_count": int(
                    sum(row["selected_action"] == "V7" for row in family_selections)
                ),
                "wrong_negative_switch_count": len(wrong),
                "wrong_negative_switch_loss_sum": float(
                    -sum(float(row["true_gain_vs_v7"]) for row in wrong)
                ),
                "mean_regret_vs_best": float(
                    np.mean([row["regret_vs_best"] for row in family_selections])
                ),
                "fold_score_std": float(
                    np.std(
                        [
                            row["mean_hole_score"]
                            for row in fold_rows
                            if row["family"] == family
                            and row["model"] == PRIMARY_MODEL
                        ]
                    )
                ),
                "primary_model": PRIMARY_MODEL,
            }
        )
        if smoke:
            break

    weighted_v7 = float(
        np.average(
            [row["V7_score"] for row in family_summaries],
            weights=[row["evaluated_hole_count"] for row in family_summaries],
        )
    )
    weighted_selector = float(
        np.average(
            [row["selector_score"] for row in family_summaries],
            weights=[row["evaluated_hole_count"] for row in family_summaries],
        )
    )
    weighted_oracle = float(
        np.average(
            [row["oracle_score"] for row in family_summaries],
            weights=[row["evaluated_hole_count"] for row in family_summaries],
        )
    )
    positive_families = int(
        sum(float(row["selector_gain"]) > 0.0 for row in family_summaries)
    )
    oof_pass = (
        not smoke
        and weighted_selector > weighted_v7
        and positive_families >= 4
    )
    report = {
        "experiment": VERSION,
        "phase": "strict_pseudo_hole_oof",
        "status": "pseudo_oof_passed" if oof_pass else "pseudo_oof_failed",
        "smoke": smoke,
        "official_inputs_only": True,
        "sionna_used": False,
        "v8_v12_used": False,
        "strict_validation": {
            "fold_definition": "hole_rank_mod_5; six complete holes held per fold",
            "shared_point_rule": "remove every held global point ID from training occurrences",
            "label_overlap_required": 0,
            "model_choice": "shared_gain_regression frozen before held-fold scoring; other simple models are diagnostic only",
            "primary_model": PRIMARY_MODEL,
            "model_configs": {
                name: {"fallback_threshold": threshold}
                for name, threshold in MODEL_CONFIGS
            },
        },
        "families": family_summaries,
        "aggregate": {
            "V7_mean_hole_score": weighted_v7,
            "selector_mean_hole_score": weighted_selector,
            "selector_gain": weighted_selector - weighted_v7,
            "oracle_mean_hole_score": weighted_oracle,
            "oracle_gain": weighted_oracle - weighted_v7,
            "oracle_gain_recovered_fraction": (
                (weighted_selector - weighted_v7)
                / max(weighted_oracle - weighted_v7, TINY)
            ),
            "positive_family_count": positive_families,
            "required_positive_family_count": 4,
            "pseudo_oof_gate_passed": oof_pass,
        },
        "canonical_boundary": {
            "canonical_scoring_run": False,
            "reason": "requires six action-label workspaces regenerated with the corresponding canonical broad fold removed before selector training",
        },
        "performance": {
            "wall_seconds": time.perf_counter() - started,
            "device": "CPU ExtraTrees",
            "trees": trees,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (1024.0**2),
            "measured_bottleneck": "nested whole-hole ExtraTrees fits",
        },
    }
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(FAMILY_CSV, family_summaries)
    write_csv(FOLD_CSV, fold_rows)
    write_csv(HOLE_CSV, hole_rows)
    write_csv(SELECTION_CSV, selection_rows)
    np.savez_compressed(
        OOF_ARTIFACT,
        family=np.asarray(artifact_family, dtype="U8"),
        occurrence_row=np.asarray(artifact_occurrence, dtype=np.int32),
        predicted_values=np.asarray(artifact_prediction, dtype=np.float32),
        selected_action=np.asarray(artifact_selection, dtype=np.int8),
    )
    OOF_REPORT.write_text(
        json.dumps(json_value(report), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    args = parse_args()
    families = tuple(value for value in args.families.split(",") if value)
    unknown = set(families) - set(SPECIALIZED_FAMILIES)
    if unknown:
        raise ValueError(f"unsupported families: {sorted(unknown)}")
    if args.trees <= 0:
        raise ValueError("trees must be positive")
    if args.phase == "audit":
        report = run_audit(families)
    else:
        report = run_oof(families, args.trees, args.seed, args.smoke)
    print(json.dumps(json_value(report), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
