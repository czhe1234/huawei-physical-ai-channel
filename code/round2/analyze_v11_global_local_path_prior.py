#!/usr/bin/env python3
"""Apply the E79 learned path lifecycle field as a bounded V11 prior."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import time

import numpy as np
import torch

from analyze_v4_exact_joint_projection import (
    normalized,
    pas_distribution,
    pdp_distribution,
    truth_views,
)
from analyze_v4_path_track_spectral_targets import render_noncoherent
from project_paths import PATHS
from run_rbf_spectral_field import statistics
from run_v9_joint_sionna_migration import endpoint_cosine, point_metrics
from train_full_path_pas_slice_gate import project_joint


VERSION = "Round2-E80-V11-global-local-path-prior"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "v11_global_local_path_prior"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v11_global_local_path_prior"
REPORT = ANALYSIS_DIR / "v11_global_local_path_prior.json"

E79_REPORT = (
    PATHS.analysis
    / "results"
    / "round2_global_local_path_lifecycle"
    / "global_local_path_lifecycle.json"
)
E79_PREDICTIONS = (
    PATHS.artifacts
    / "round2"
    / "global_local_path_lifecycle"
    / "oof_path_lifecycle_predictions.npz"
)
OOF_META = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_pre_sionna_oof_foundation_meta.npz"
)
V11_CHANNEL = (
    PATHS.artifacts
    / "round2"
    / "canonical_500_version_replay"
    / "V11_Canonical_500_OOF_Channel.npy"
)
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"
TINY = 1e-30


@dataclass(frozen=True)
class DomainAction:
    mode: str
    alpha: float

    @property
    def name(self) -> str:
        return "identity" if self.alpha == 0.0 else f"{self.mode}_a{self.alpha:.3f}"


@dataclass(frozen=True)
class ProjectionSpec:
    endpoint: str
    pas_action: DomainAction
    pdp_action: DomainAction
    order: str
    cycles: int

    @property
    def name(self) -> str:
        return (
            f"{self.endpoint}__pas-{self.pas_action.name}__"
            f"pdp-{self.pdp_action.name}__{self.order}_c{self.cycles}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--render-batch-size", type=int, default=16)
    parser.add_argument("--top-domain-actions", type=int, default=3)
    return parser.parse_args()


def apply_prior(
    base: torch.Tensor,
    path: torch.Tensor,
    action: DomainAction,
    dims: tuple[int, ...],
) -> torch.Tensor:
    if action.alpha == 0.0:
        return base
    target = normalized(path, dims, fallback=base)
    if action.mode == "sqrt":
        amplitude = (
            (1.0 - action.alpha) * torch.sqrt(base.clamp_min(0.0))
            + action.alpha * torch.sqrt(target.clamp_min(0.0))
        )
        return normalized(amplitude.square(), dims, fallback=base)
    if action.mode == "power":
        return normalized(
            (1.0 - action.alpha) * base + action.alpha * target,
            dims,
            fallback=base,
        )
    raise ValueError(action.mode)


def render_endpoint(
    cache: np.lib.npyio.NpzFile,
    name: str,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    coordinate = cache[f"{name}_coordinates"]
    coefficient = cache[f"{name}_coefficients"]
    pas_parts = []
    pdp_parts = []
    for start in range(0, len(coordinate), batch_size):
        end = min(start + batch_size, len(coordinate))
        _, pas, pdp = render_noncoherent(
            coordinate[start:end], coefficient[start:end], device
        )
        pas_parts.append(torch.as_tensor(pas, device=device))
        pdp_parts.append(torch.as_tensor(pdp, device=device))
    pas_tensor = torch.cat(pas_parts).reshape(-1, 2, 16, 8, 4, 192)
    pdp_tensor = torch.cat(pdp_parts)
    return pas_tensor, pdp_tensor


def domain_rows(
    base: torch.Tensor,
    path: torch.Tensor,
    truth: torch.Tensor,
    fold_id: np.ndarray,
    domain: str,
) -> tuple[list[dict[str, object]], dict[str, torch.Tensor]]:
    dims = (1, 2, 3) if domain == "pas" else (3,)
    base_values = endpoint_cosine(base, truth, domain)
    actions = [DomainAction("identity", 0.0)] + [
        DomainAction(mode, alpha)
        for mode in ("sqrt", "power")
        for alpha in (0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)
    ]
    values: dict[str, torch.Tensor] = {}
    rows = []
    for action in actions:
        desired = apply_prior(base, path, action, dims)
        selected = endpoint_cosine(desired, truth, domain)
        folds = [
            float(np.mean(selected[fold_id == fold]) - np.mean(base_values[fold_id == fold]))
            for fold in range(5)
        ]
        rows.append(
            {
                **asdict(action),
                "name": action.name,
                "score": float(np.mean(selected)),
                "delta": float(np.mean(selected) - np.mean(base_values)),
                "positive_folds": int(sum(value > 0.0 for value in folds)),
                "worst_fold_delta": float(min(folds)),
                "fold_delta": folds,
            }
        )
        values[action.name] = desired
    rows.sort(key=lambda row: float(row["score"]), reverse=True)
    return rows, values


def selected_actions(rows: list[dict[str, object]], limit: int) -> list[DomainAction]:
    output = [DomainAction("identity", 0.0)]
    eligible = [
        row
        for row in rows
        if row["name"] != "identity"
        and float(row["delta"]) > 0.0
        and int(row["positive_folds"]) >= 3
    ]
    for row in eligible[:limit]:
        output.append(DomainAction(str(row["mode"]), float(row["alpha"])))
    return output


def metric_summary(
    values: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    baseline: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    fold_id: np.ndarray,
) -> dict[str, object]:
    selected = statistics(*values)
    anchor = statistics(*baseline)
    folds = []
    for fold in range(5):
        mask = fold_id == fold
        local = statistics(*values, mask)
        local_anchor = statistics(*baseline, mask)
        folds.append(
            {
                "fold": fold,
                "selected": local,
                "baseline": local_anchor,
                "delta": float(local["score"] - local_anchor["score"]),
            }
        )
    return {
        "baseline": anchor,
        "selected": selected,
        "delta": float(selected["score"] - anchor["score"]),
        "positive_folds": int(sum(row["delta"] > 0.0 for row in folds)),
        "worst_fold_delta": float(min(row["delta"] for row in folds)),
        "folds": folds,
    }


def nested_selection(
    candidates: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    baseline: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    fold_id: np.ndarray,
) -> dict[str, object]:
    stitched = tuple(np.array(value, copy=True) for value in baseline)
    decisions = []
    for outer in range(5):
        inner_rows = []
        for name, values in candidates.items():
            deltas = []
            for inner in range(5):
                if inner == outer:
                    continue
                mask = fold_id == inner
                deltas.append(
                    float(
                        statistics(*values, mask)["score"]
                        - statistics(*baseline, mask)["score"]
                    )
                )
            fit_mask = fold_id != outer
            fit_delta = float(
                statistics(*values, fit_mask)["score"]
                - statistics(*baseline, fit_mask)["score"]
            )
            inner_rows.append(
                {
                    "name": name,
                    "fit_delta": fit_delta,
                    "positive_inner_folds": int(sum(value > 0.0 for value in deltas)),
                    "worst_inner_delta": float(min(deltas)),
                }
            )
        eligible = [row for row in inner_rows if row["positive_inner_folds"] == 4]
        winner = max(
            eligible,
            key=lambda row: (row["fit_delta"], row["worst_inner_delta"]),
            default=None,
        )
        held = fold_id == outer
        if winner is not None and winner["fit_delta"] > 0.0:
            for target, source in zip(stitched, candidates[str(winner["name"])]):
                target[held] = source[held]
            name = str(winner["name"])
        else:
            name = "identity"
        held_delta = float(
            statistics(*stitched, held)["score"]
            - statistics(*baseline, held)["score"]
        )
        decisions.append({"fold": outer, "candidate": name, "held_delta": held_delta})
    result = statistics(*stitched)
    anchor = statistics(*baseline)
    return {
        "baseline": anchor,
        "selected": result,
        "delta": float(result["score"] - anchor["score"]),
        "positive_folds": int(sum(row["held_delta"] > 0.0 for row in decisions)),
        "decisions": decisions,
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    lifecycle_report = json.loads(E79_REPORT.read_text(encoding="utf-8"))
    endpoints = {
        "local": str(lifecycle_report["best_local"]["name"]),
        "global_local": str(lifecycle_report["best_global_local"]["name"]),
    }
    cache = np.load(E79_PREDICTIONS, allow_pickle=False)
    base = torch.as_tensor(
        np.asarray(np.load(V11_CHANNEL, mmap_mode="r"), dtype=np.complex64),
        device=device,
    )
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    truth = torch.as_tensor(
        np.asarray(truth_all[query_ids], dtype=np.complex64), device=device
    )
    truth_pas, truth_pdp = truth_views(truth)
    base_pas = pas_distribution(base)
    base_pdp = pdp_distribution(base)
    baseline = point_metrics(base, truth, truth_pas, truth_pdp)

    domain_screen: dict[str, object] = {}
    metric_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    projection_rows = []
    for endpoint_group, endpoint_name in endpoints.items():
        path_pas, path_pdp = render_endpoint(
            cache, endpoint_name, device, args.render_batch_size
        )
        pas_rows, pas_values = domain_rows(
            base_pas, path_pas, truth_pas, fold_id, "pas"
        )
        pdp_rows, pdp_values = domain_rows(
            base_pdp, path_pdp, truth_pdp, fold_id, "pdp"
        )
        pas_actions = selected_actions(pas_rows, args.top_domain_actions)
        pdp_actions = selected_actions(pdp_rows, args.top_domain_actions)
        domain_screen[endpoint_group] = {
            "endpoint": endpoint_name,
            "pas": pas_rows,
            "pdp": pdp_rows,
            "selected_pas_actions": [asdict(row) for row in pas_actions],
            "selected_pdp_actions": [asdict(row) for row in pdp_actions],
        }
        for pas_action in pas_actions:
            for pdp_action in pdp_actions:
                if pas_action.alpha == 0.0 and pdp_action.alpha == 0.0:
                    continue
                for order in ("pas_pdp", "pdp_pas"):
                    for cycles in (2, 4):
                        spec = ProjectionSpec(
                            endpoint_group, pas_action, pdp_action, order, cycles
                        )
                        with torch.inference_mode():
                            prediction = project_joint(
                                base,
                                pas_values[pas_action.name],
                                pdp_values[pdp_action.name],
                                cycles,
                                order,
                            )
                            values = point_metrics(
                                prediction, truth, truth_pas, truth_pdp
                            )
                        metric_cache[spec.name] = values
                        projection_rows.append(
                            {**asdict(spec), "name": spec.name, **metric_summary(values, baseline, fold_id)}
                        )
                        del prediction
        del path_pas, path_pdp
        if device.type == "cuda":
            torch.cuda.empty_cache()

    projection_rows.sort(
        key=lambda row: float(row["selected"]["score"]), reverse=True
    )
    selected = projection_rows[0] if projection_rows else None
    nested = nested_selection(metric_cache, baseline, fold_id) if metric_cache else None
    passed = bool(
        selected is not None
        and float(selected["delta"]) >= 5e-4
        and int(selected["positive_folds"]) >= 4
        and float(selected["worst_fold_delta"]) >= -2e-4
        and nested is not None
        and float(nested["delta"]) >= 3e-4
        and int(nested["positive_folds"]) >= 4
    )
    payload = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "The E79 path selector should act as a bounded support prior on V11, "
            "not replace its Sionna-calibrated spectrum and complex carrier."
        ),
        "validation": "fixed 500-point complete official score",
        "baseline": statistics(*baseline),
        "domain_screen": domain_screen,
        "selected": selected,
        "nested_selection": nested,
        "top_candidates": projection_rows[:30],
        "deployment_gate": {
            "minimum_delta": 5e-4,
            "minimum_positive_folds": 4,
            "minimum_worst_fold_delta": -2e-4,
            "minimum_nested_delta": 3e-4,
            "minimum_nested_positive_folds": 4,
            "passed": passed,
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "baseline": payload["baseline"],
                "selected": selected,
                "nested_selection": nested,
                "deployment_gate": payload["deployment_gate"],
                "seconds": payload["seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
