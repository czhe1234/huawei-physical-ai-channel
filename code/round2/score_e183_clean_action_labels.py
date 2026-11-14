#!/usr/bin/env python3
"""Score strict E183 family actions after all predictions are materialized."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from scipy.stats import rankdata

from official_metrics import PointMetricStatistics, aggregate_statistics
from project_paths import PATHS


VERSION = "Round2-E183-strict-clean-action-label-scoring"
WORKSPACE_ROOT = Path(
    os.environ.get(
        "E183_WORKSPACE_DIR",
        str(PATHS.artifacts / "round2" / "e183_strict_action_labels" / "workspaces"),
    )
).resolve()
RESULT_DIR = PATHS.analysis / "results" / "round2_e183_strict_action_labels"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e183_strict_action_labels" / "labels"
IDENTITY_REPORT = (
    PATHS.analysis
    / "results"
    / "round2_e183_hybrid_specialization"
    / "canonical_13_action_identity_gate.json"
)
SOURCE_TRUTH = PATHS.round2 / "Round2_Train_Channel.npy"
FAMILIES = (
    "S0H0",
    "S0H1",
    "S0H2",
    "S0H3",
    "S0H4",
    "S1H1",
    "S1H2",
    "S1H3",
    "S1H4",
)
ACTION_NAMES = (
    "V7",
    "V4",
    "E121_path",
    "E122_path",
    "E142_raw",
    "E142_projected",
    "E161_knn",
    "E162_knn",
    "carrier_00_nearest",
    "carrier_07_raw_idw_k24",
    "carrier_08_radial_idw_k24",
    "carrier_11_radial_affine_k24_r0.1",
    "carrier_16_affine_k24_steer_axis90_v8pi",
)
CARRIER_INDICES = (0, 7, 8, 11, 16)
TINY = 1e-30
TIE_TOLERANCE = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def json_value(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def ensure_identity_gate() -> dict[str, object]:
    report = json.loads(IDENTITY_REPORT.read_text(encoding="utf-8"))
    if not bool(report.get("gate", {}).get("overall_pass")):
        raise RuntimeError("canonical 13-action identity gate has not passed")
    return report


class ActionProvider:
    def __init__(self, workspace: Path, count: int) -> None:
        self.workspace = workspace
        root = workspace / "artifacts" / "round2"
        action_dir = root / "e183_family_actions"
        self.count = count
        self.files = {
            "V7": root
            / "complex_carrier_lifecycle_overlay"
            / "Round2_V7_Stacked_Lifecycle_Test_Channel.npy",
            "V4": root
            / "v4_calibrated_exact_projection"
            / "Round2_V4_Calibrated_Exact_Projection_Test_Channel.npy",
            "E121_path": action_dir / "E121_path.npy",
            "E122_path": action_dir / "E122_path.npy",
            "E142_raw": action_dir / "E142_raw.npy",
            "E142_projected": action_dir / "E142_projected.npy",
            "E161_knn": action_dir / "E161_knn.npy",
            "E162_knn": action_dir / "E162_knn.npy",
        }
        self.bank_path = root / "complex_carrier_bank" / "test_complex_carrier_bank.npy"
        self.arrays: dict[str, np.ndarray] = {}
        self.bank: np.ndarray | None = None

    def validate_lineage_reports(self) -> dict[str, object]:
        reports = {
            "e120_endpoint": self.workspace
            / "analysis"
            / "results"
            / "round2_fivefold_unique_path_v12"
            / "fivefold_unique_path_v12_test.json",
            "actions_11": self.workspace
            / "analysis"
            / "results"
            / "round2_e183_strict_action_labels"
            / "family_action_generation.json",
            "e160": self.workspace
            / "analysis"
            / "results"
            / "round2_e160_full4000_v7_supervision"
            / "full4000_v7_supervision.json",
            "e161_e162": self.workspace
            / "analysis"
            / "results"
            / "round2_e183_strict_action_labels"
            / "e161_e162_action_generation.json",
        }
        output: dict[str, object] = {}
        for name, path in reports.items():
            if not path.is_file():
                raise FileNotFoundError(f"missing E183 lineage report before GT read: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if bool(payload.get("sionna_used")) or bool(payload.get("v8_v12_used")):
                raise RuntimeError(f"forbidden lineage in {name}")
            if name != "e160" and not bool(payload.get("official_inputs_only")):
                raise RuntimeError(f"official-input lineage missing in {name}")
            if name in ("actions_11", "e161_e162") and bool(
                payload.get("hidden_truth_read")
            ):
                raise RuntimeError(f"hidden truth was read while generating {name}")
            if name == "e120_endpoint" and bool(payload.get("v12_used")):
                raise RuntimeError("E120 endpoint used V12")
            if name == "e160" and not bool(payload.get("strict_family_isolation")):
                raise RuntimeError("E160 report is not the strict family-isolated run")
            if name == "actions_11":
                relative = Path(str(payload["actions"]["V7"]["path"]))
                if relative.is_absolute() or ".." in relative.parts:
                    raise RuntimeError(f"invalid V7 lineage path: {relative}")
                v7_path = self.workspace / relative
                if not v7_path.resolve().is_relative_to(self.workspace.resolve()):
                    raise RuntimeError(f"V7 lineage escapes workspace: {v7_path}")
                self.files["V7"] = v7_path
            output[name] = {"path": PATHS.relative(path), "status": payload.get("status")}
        return output

    def validate_before_truth(self) -> dict[str, object]:
        records: dict[str, object] = {}
        for name, path in self.files.items():
            if not path.is_file():
                raise FileNotFoundError(f"missing E183 action before GT read: {path}")
            value = np.load(path, mmap_mode="r")
            if value.shape != (self.count, 256, 4, 192) or value.dtype != np.complex64:
                raise RuntimeError(f"invalid action {name}: {value.shape}/{value.dtype}")
            for start in range(0, self.count, 8):
                block = np.asarray(value[start : start + 8])
                if not np.isfinite(block.real).all() or not np.isfinite(block.imag).all():
                    raise RuntimeError(f"non-finite action {name}")
            self.arrays[name] = value
            records[name] = {"path": PATHS.relative(path), "shape": list(value.shape)}
        if not self.bank_path.is_file():
            raise FileNotFoundError(f"missing E183 carrier bank before GT read: {self.bank_path}")
        bank = np.load(self.bank_path, mmap_mode="r")
        if bank.shape != (17, self.count, 256, 4, 192) or bank.dtype != np.complex64:
            raise RuntimeError(f"invalid E183 carrier bank: {bank.shape}/{bank.dtype}")
        for index, name in zip(CARRIER_INDICES, ACTION_NAMES[8:], strict=True):
            for start in range(0, self.count, 8):
                block = np.asarray(bank[index, start : start + 8])
                if not np.isfinite(block.real).all() or not np.isfinite(block.imag).all():
                    raise RuntimeError(f"non-finite action {name}")
            records[name] = {
                "path": PATHS.relative(self.bank_path),
                "carrier_index": index,
                "shape": list(bank[index].shape),
            }
        self.bank = bank
        return records

    def block(self, action: int, start: int, end: int) -> np.ndarray:
        name = ACTION_NAMES[action]
        if action < 8:
            return np.asarray(self.arrays[name][start:end], dtype=np.complex64)
        if self.bank is None:
            raise RuntimeError("carrier bank has not been validated")
        index = CARRIER_INDICES[action - 8]
        return np.asarray(self.bank[index, start:end], dtype=np.complex64)


@torch.inference_mode()
def point_metrics(
    prediction: torch.Tensor, truth: torch.Tensor
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = len(prediction)
    pred = prediction.reshape(count, 2, 16, 8, 4, 192)
    target = truth.reshape(count, 2, 16, 8, 4, 192)
    pred_pas = torch.fft.fft2(pred, dim=(2, 3)).abs().square().float()
    truth_pas = torch.fft.fft2(target, dim=(2, 3)).abs().square().float()
    pred_pas = pred_pas.reshape(count, 256, 4, 192)
    truth_pas = truth_pas.reshape(count, 256, 4, 192)
    pas_num = (pred_pas * truth_pas).sum(dim=1, dtype=torch.float64)
    pas_den = torch.sqrt(
        pred_pas.square().sum(dim=1, dtype=torch.float64)
        * truth_pas.square().sum(dim=1, dtype=torch.float64)
    )
    pas_den = torch.where(pas_den == 0.0, torch.full_like(pas_den, 1e-12), pas_den)
    pas = (pas_num / pas_den).mean(dim=(1, 2), dtype=torch.float64)
    pred_pdp = torch.fft.ifft(prediction, dim=-1).abs().square().float()
    truth_pdp = torch.fft.ifft(truth, dim=-1).abs().square().float()
    pdp_num = (pred_pdp * truth_pdp).sum(dim=3, dtype=torch.float64)
    pdp_den = torch.sqrt(
        pred_pdp.square().sum(dim=3, dtype=torch.float64)
        * truth_pdp.square().sum(dim=3, dtype=torch.float64)
    )
    pdp_den = torch.where(pdp_den == 0.0, torch.full_like(pdp_den, 1e-12), pdp_den)
    pdp = (pdp_num / pdp_den).mean(dim=(1, 2), dtype=torch.float64)
    error = (prediction - truth).abs().square().sum(dim=(1, 2, 3), dtype=torch.float64)
    energy = truth.abs().square().sum(dim=(1, 2, 3), dtype=torch.float64)
    return tuple(
        value.cpu().numpy().astype(np.float64)
        for value in (pas, pdp, error, energy)
    )


def statistics(
    pas: np.ndarray,
    pdp: np.ndarray,
    error: np.ndarray,
    energy: np.ndarray,
) -> PointMetricStatistics:
    return PointMetricStatistics(pas, pdp, error, energy)


def point_utility(
    pas: np.ndarray,
    pdp: np.ndarray,
    error: np.ndarray,
    energy: np.ndarray,
) -> np.ndarray:
    nmse = error / np.maximum(energy[None], TINY)
    return 0.4 * pas + 0.4 * pdp + 0.2 / (1.0 + nmse)


def pairwise_consistency(values: np.ndarray, ranking: bool) -> float:
    table = (
        np.stack([rankdata(-row, method="average") for row in values], axis=0)
        if ranking
        else np.asarray(values, dtype=np.float64)
    )
    correlation = np.corrcoef(table)
    upper = correlation[np.triu_indices(len(table), 1)]
    finite = upper[np.isfinite(upper)]
    return float(np.mean(finite)) if len(finite) else 0.0


def score_family(family: str, device: torch.device, batch_size: int) -> dict[str, object]:
    started = time.time()
    identity = ensure_identity_gate()
    workspace = WORKSPACE_ROOT / family
    mapping = np.load(
        workspace / "artifacts" / "round2" / "e183_mapping" / "family_mapping.npz",
        allow_pickle=False,
    )
    hidden_global = mapping["test_global_ids"].astype(np.int64)
    flat_rows = mapping["hole_test_rows_flat"].astype(np.int64)
    offsets = mapping["hole_offsets"].astype(np.int64)
    if len(offsets) != 31:
        raise RuntimeError(f"{family} must have 30 analog holes")
    provider = ActionProvider(workspace, len(hidden_global))
    lineage_reports = provider.validate_lineage_reports()
    action_lineage = provider.validate_before_truth()

    # Hidden GT is deliberately opened only after all 13 predictions pass validation.
    source_truth = np.load(SOURCE_TRUTH, mmap_mode="r")
    truth = np.asarray(source_truth[hidden_global], dtype=np.complex64)
    action_pas = np.empty((len(ACTION_NAMES), len(hidden_global)), dtype=np.float64)
    action_pdp = np.empty_like(action_pas)
    action_error = np.empty_like(action_pas)
    energy = np.empty(len(hidden_global), dtype=np.float64)
    truth_tensor = torch.as_tensor(truth, device=device)
    for action in range(len(ACTION_NAMES)):
        for start in range(0, len(hidden_global), batch_size):
            end = min(start + batch_size, len(hidden_global))
            values = point_metrics(
                torch.as_tensor(provider.block(action, start, end), device=device),
                truth_tensor[start:end],
            )
            action_pas[action, start:end] = values[0]
            action_pdp[action, start:end] = values[1]
            action_error[action, start:end] = values[2]
            energy[start:end] = values[3]
    utility = point_utility(action_pas, action_pdp, action_error, energy)
    point_best = np.argmax(
        utility >= utility.max(axis=0, keepdims=True) - TIE_TOLERANCE,
        axis=0,
    )

    hole_rows: list[dict[str, object]] = []
    action_best_count = np.zeros(len(ACTION_NAMES), dtype=np.int64)
    gain_table = np.empty((30, len(ACTION_NAMES)), dtype=np.float64)
    for hole in range(30):
        rows = flat_rows[offsets[hole] : offsets[hole + 1]]
        action_scores = np.empty(len(ACTION_NAMES), dtype=np.float64)
        action_metrics = []
        for action in range(len(ACTION_NAMES)):
            result = aggregate_statistics(
                statistics(
                    action_pas[action, rows],
                    action_pdp[action, rows],
                    action_error[action, rows],
                    energy[rows],
                )
            )
            action_metrics.append(result)
            action_scores[action] = float(result["score"])
        whole_best = int(
            np.argmax(action_scores >= action_scores.max() - TIE_TOLERANCE)
        )
        action_best_count[whole_best] += 1
        gain_table[hole] = action_scores - action_scores[0]
        selected = point_best[rows]
        point_oracle = aggregate_statistics(
            statistics(
                action_pas[selected, rows],
                action_pdp[selected, rows],
                action_error[selected, rows],
                energy[rows],
            )
        )
        hole_rows.append(
            {
                "family": family,
                "analog_rank": hole,
                "point_count": int(len(rows)),
                "V7_pas": action_metrics[0]["pas"],
                "V7_pdp": action_metrics[0]["pdp"],
                "V7_nmse": action_metrics[0]["nmse"],
                "V7_score": action_metrics[0]["score"],
                "point_oracle_pas": point_oracle["pas"],
                "point_oracle_pdp": point_oracle["pdp"],
                "point_oracle_nmse": point_oracle["nmse"],
                "point_oracle_score": point_oracle["score"],
                "point_oracle_gain": float(point_oracle["score"] - action_scores[0]),
                "hole_oracle_action": ACTION_NAMES[whole_best],
                "hole_oracle_score": float(action_scores[whole_best]),
                "hole_oracle_gain": float(action_scores[whole_best] - action_scores[0]),
                **{
                    f"score_{name}": float(action_scores[index])
                    for index, name in enumerate(ACTION_NAMES)
                },
            }
        )

    order = np.argsort(-action_best_count, kind="stable")
    top1 = float(action_best_count[order[:1]].sum() / 30.0)
    top2 = float(action_best_count[order[:2]].sum() / 30.0)
    top3 = float(action_best_count[order[:3]].sum() / 30.0)
    ranking_consistency = pairwise_consistency(gain_table, ranking=True)
    gain_consistency = pairwise_consistency(gain_table, ranking=False)
    mean_v7 = float(np.mean([row["V7_score"] for row in hole_rows]))
    mean_point_oracle = float(
        np.mean([row["point_oracle_score"] for row in hole_rows])
    )
    mean_hole_oracle = float(
        np.mean([row["hole_oracle_score"] for row in hole_rows])
    )
    oracle_gain = mean_point_oracle - mean_v7
    concentrated = bool(
        top1 >= 0.30 and top3 >= 0.70 and ranking_consistency >= 0.35
    )
    if oracle_gain <= 0.003:
        category = "oracle_gain_small"
    elif oracle_gain >= 0.010 and concentrated:
        category = "oracle_gain_large_action_concentrated"
    elif oracle_gain >= 0.010:
        category = "oracle_gain_large_action_chaotic"
    else:
        category = "oracle_gain_moderate"
    state = json.loads((workspace / "e183_workspace_state.json").read_text(encoding="utf-8"))
    action_counts = [
        {
            "action": ACTION_NAMES[index],
            "best_hole_count": int(action_best_count[index]),
            "best_hole_share": float(action_best_count[index] / 30.0),
            "point_best_count": int(np.sum(point_best == index)),
            "point_best_share": float(np.mean(point_best == index)),
        }
        for index in range(len(ACTION_NAMES))
    ]
    summary = {
        "target_hole": family,
        "n_analogs": 30,
        "hidden_union_count": int(len(hidden_global)),
        "mean_match_cost": float(state["mean_match_cost"]),
        "V7_score": mean_v7,
        "13_action_oracle_score": mean_point_oracle,
        "13_action_hole_oracle_score": mean_hole_oracle,
        "oracle_gain": oracle_gain,
        "hole_oracle_gain": mean_hole_oracle - mean_v7,
        "top1_action": ACTION_NAMES[int(order[0])],
        "top1_action_share": top1,
        "top2_action_coverage": top2,
        "top3_action_coverage": top3,
        "ranking_consistency": ranking_consistency,
        "gain_pearson_consistency": gain_consistency,
        "concentration_gate_passed": concentrated,
        "category": category,
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ARTIFACT_DIR / f"{family}_action_utility_labels.npz",
        action_names=np.asarray(ACTION_NAMES, dtype="U64"),
        hidden_global_ids=hidden_global.astype(np.int32),
        hole_test_rows_flat=flat_rows.astype(np.int32),
        hole_offsets=offsets.astype(np.int32),
        pas=action_pas.astype(np.float32),
        pdp=action_pdp.astype(np.float32),
        squared_error=action_error.astype(np.float64),
        truth_energy=energy.astype(np.float64),
        utility=utility.astype(np.float32),
        point_best_action=point_best.astype(np.int8),
        hole_gain_table=gain_table.astype(np.float32),
        hole_best_counts=action_best_count.astype(np.int16),
    )
    write_csv(RESULT_DIR / f"{family}_hole_action_scores.csv", hole_rows)
    write_csv(RESULT_DIR / f"{family}_action_best_counts.csv", action_counts)
    payload = {
        "experiment": VERSION,
        "status": "family_scored",
        "family": family,
        "canonical_identity": {
            "passed": True,
            "oracle_score": identity["oracle"]["recomputed_point_statistics"]["score"],
        },
        "official_inputs_only_for_prediction": True,
        "hidden_truth_read_after_all_actions": True,
        "action_lineage": action_lineage,
        "lineage_reports": lineage_reports,
        "summary": summary,
        "action_best_counts": action_counts,
        "holes": hole_rows,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "seconds": float(time.time() - started),
    }
    report_path = RESULT_DIR / f"{family}_strict_action_labels.json"
    report_path.write_text(
        json.dumps(json_value(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def summarize() -> dict[str, object]:
    rows = []
    categories: dict[str, list[str]] = {}
    for family in FAMILIES:
        path = RESULT_DIR / f"{family}_strict_action_labels.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing E183 family score: {path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        row = dict(report["summary"])
        rows.append(row)
        categories.setdefault(str(row["category"]), []).append(family)
    pass_count = int(sum(bool(row["concentration_gate_passed"]) for row in rows))
    if pass_count >= 6:
        conclusion = "supported"
    elif pass_count <= 3:
        conclusion = "rejected"
    else:
        conclusion = "mixed"
    write_csv(RESULT_DIR / "e183_nine_family_action_label_summary.csv", rows)
    payload = {
        "experiment": VERSION,
        "status": "nine_family_summary_complete",
        "question": "Do analogous holes in one family prefer similar Clean actions?",
        "gate": {
            "required_families": 6,
            "passed_families": pass_count,
            "conclusion": conclusion,
        },
        "categories": categories,
        "families": rows,
    }
    (RESULT_DIR / "e183_nine_family_action_label_summary.json").write_text(
        json.dumps(json_value(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    args = parse_args()
    if args.summarize:
        payload = summarize()
    else:
        if args.family is None:
            raise SystemExit("--family is required unless --summarize is used")
        payload = score_family(args.family, torch.device(args.device), args.batch_size)
    print(json.dumps(json_value(payload), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
