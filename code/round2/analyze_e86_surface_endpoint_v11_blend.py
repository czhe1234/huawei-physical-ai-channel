#!/usr/bin/env python3
"""Test E85 path endpoints as weak PAS/PDP targets for the V11 channel."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch

from analyze_v4_exact_joint_projection import pas_distribution, pdp_distribution
from analyze_v4_path_track_spectral_targets import render_noncoherent
from project_paths import PATHS
from run_rbf_spectral_field import statistics
from run_v9_joint_sionna_migration import point_metrics
from train_full_path_pas_slice_gate import project_joint


VERSION = "Round2-E86-surface-endpoint-v11-blend"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "surface_interaction_energy_model"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_surface_interaction_energy_model"
REPORT = ANALYSIS_DIR / "e86_surface_endpoint_v11_blend.json"
METRICS_PATH = ARTIFACT_DIR / "e86_candidate_metrics.npz"
NESTED_CHANNEL = ARTIFACT_DIR / "e86_nested_oof_channel.npy"
RENDERED = {
    "geometry_sequence": (
        ARTIFACT_DIR / "e86_geometry_sequence_rendered_pas_f16.npy",
        ARTIFACT_DIR / "e86_geometry_sequence_rendered_pdp_f16.npy",
        ARTIFACT_DIR / "strict_oof_geometry_sequence_path_endpoint.npz",
    ),
    "no_sequence": (
        ARTIFACT_DIR / "e86_no_sequence_rendered_pas_f16.npy",
        ARTIFACT_DIR / "e86_no_sequence_rendered_pdp_f16.npy",
        ARTIFACT_DIR / "strict_oof_no_sequence_path_endpoint.npz",
    ),
}
V11_CHANNEL = (
    PATHS.artifacts
    / "round2"
    / "canonical_500_version_replay"
    / "V11_Canonical_500_OOF_Channel.npy"
)
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"
OOF_META = PATHS.artifacts / "round2" / "pre_sionna" / "round2_pre_sionna_oof_foundation_meta.npz"
TINY = 1e-30


@dataclass(frozen=True)
class Candidate:
    endpoint: str
    mode: str
    pas_beta: float
    pdp_beta: float
    order: str
    cycles: int

    @property
    def name(self) -> str:
        return (
            f"{self.endpoint}_{self.mode}_p{int(round(1000 * self.pas_beta)):03d}_"
            f"d{int(round(1000 * self.pdp_beta)):03d}_{self.order}_c{self.cycles}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--force-render", action="store_true")
    return parser.parse_args()


def normalized(value: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    value = value.clamp_min(0.0)
    total = value.sum(dim=dims, keepdim=True)
    return torch.where(total > TINY, value / total.clamp_min(TINY), torch.zeros_like(value))


def endpoint_target(
    base: torch.Tensor,
    endpoint: torch.Tensor,
    beta: float,
    mode: str,
    dims: tuple[int, ...],
) -> torch.Tensor:
    if beta == 0.0:
        return base
    if mode == "power":
        mixed = (1.0 - beta) * base + beta * endpoint
    elif mode == "sqrt":
        mixed = (
            (1.0 - beta) * torch.sqrt(base.clamp_min(0.0))
            + beta * torch.sqrt(endpoint.clamp_min(0.0))
        ).square()
    else:
        raise ValueError(mode)
    return normalized(mixed, dims)


def render_cache(
    endpoint_path: str,
    pas_path: str,
    pdp_path: str,
    device: torch.device,
    batch_size: int,
    force: bool,
) -> tuple[np.ndarray, np.ndarray]:
    endpoint = np.load(endpoint_path, allow_pickle=False)
    pas_file_path = Path(pas_path).expanduser()
    pdp_file_path = Path(pdp_path).expanduser()
    if not pas_file_path.is_absolute():
        pas_file_path = PATHS.root / pas_file_path
    if not pdp_file_path.is_absolute():
        pdp_file_path = PATHS.root / pdp_file_path
    if not force and pas_file_path.is_file() and pdp_file_path.is_file():
        return (
            np.load(pas_file_path, mmap_mode="r"),
            np.load(pdp_file_path, mmap_mode="r"),
        )

    count = len(endpoint["query_ids"])
    pas = np.empty((count, 2, 16, 8, 4, 192), dtype=np.float32)
    pdp = np.empty((count, 256, 4, 192), dtype=np.float32)
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        mass = np.asarray(endpoint["mass"][start:end], dtype=np.float32)
        port = np.asarray(endpoint["port"][start:end], dtype=np.float32)
        coordinate = np.asarray(endpoint["coordinate"][start:end], dtype=np.float32)
        coefficient = np.sqrt(
            np.maximum(mass[..., None] * port.reshape(len(mass), 128, 8), 0.0)
        ).astype(np.complex64)
        _, local_pas, local_pdp = render_noncoherent(coordinate, coefficient, device)
        pas[start:end] = local_pas.reshape(len(mass), 2, 16, 8, 4, 192)
        pdp[start:end] = local_pdp
        print(f"E86 render {end}/{count}", flush=True)
    np.save(pas_file_path, pas.astype(np.float16))
    np.save(pdp_file_path, pdp.astype(np.float16))
    return np.load(pas_file_path, mmap_mode="r"), np.load(pdp_file_path, mmap_mode="r")


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
        "selected": selected,
        "baseline": anchor,
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
        fit_mask = fold_id != outer
        rows = []
        for name, values in candidates.items():
            inner_deltas = []
            for inner in range(5):
                if inner == outer:
                    continue
                mask = fold_id == inner
                inner_deltas.append(
                    float(
                        statistics(*values, mask)["score"]
                        - statistics(*baseline, mask)["score"]
                    )
                )
            fit = statistics(*values, fit_mask)
            rows.append(
                {
                    "name": name,
                    "fit_delta": float(fit["score"] - statistics(*baseline, fit_mask)["score"]),
                    "positive_inner_folds": int(sum(delta > 0.0 for delta in inner_deltas)),
                    "worst_inner_delta": float(min(inner_deltas)),
                    "inner_deltas": inner_deltas,
                }
            )
        eligible = [row for row in rows if row["positive_inner_folds"] == 4]
        winner = max(
            eligible,
            key=lambda row: (row["fit_delta"], row["worst_inner_delta"]),
            default=None,
        )
        name = "identity" if winner is None else str(winner["name"])
        if winner is not None and winner["fit_delta"] > 0.0:
            for output, source in zip(stitched, candidates[name]):
                output[fold_id == outer] = source[fold_id == outer]
        else:
            name = "identity"
        held = fold_id == outer
        selected = statistics(*stitched, held)
        anchor = statistics(*baseline, held)
        decisions.append(
            {
                "fold": outer,
                "candidate": name,
                "held_delta": float(selected["score"] - anchor["score"]),
                "selection": winner,
            }
        )
    selected = statistics(*stitched)
    anchor = statistics(*baseline)
    return {
        "selected": selected,
        "baseline": anchor,
        "delta": float(selected["score"] - anchor["score"]),
        "positive_folds": int(sum(row["held_delta"] > 0.0 for row in decisions)),
        "worst_fold_delta": float(min(row["held_delta"] for row in decisions)),
        "decisions": decisions,
    }


def build_candidates() -> list[Candidate]:
    betas = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40)
    output = []
    for endpoint in ("geometry_sequence", "no_sequence"):
        for mode in ("power", "sqrt"):
            for pas_beta in betas:
                for pdp_beta in betas:
                    for order in ("pas_pdp", "pdp_pas"):
                        for cycles in (1, 2, 4):
                            candidate = Candidate(
                                endpoint, mode, pas_beta, pdp_beta, order, cycles
                            )
                            if pas_beta == 0.0 and pdp_beta == 0.0:
                                continue
                            output.append(candidate)
    return output


def make_candidate(
    base: torch.Tensor,
    base_pas: torch.Tensor,
    base_pdp: torch.Tensor,
    endpoint_pas: torch.Tensor,
    endpoint_pdp: torch.Tensor,
    spec: Candidate,
) -> torch.Tensor:
    desired_pas = endpoint_target(
        base_pas,
        endpoint_pas,
        spec.pas_beta,
        spec.mode,
        (1, 2, 3),
    )
    desired_pdp = endpoint_target(
        base_pdp,
        endpoint_pdp,
        spec.pdp_beta,
        spec.mode,
        (3,),
    )
    return project_joint(base, desired_pas, desired_pdp, spec.cycles, spec.order)


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    base_np = np.load(V11_CHANNEL, mmap_mode="r")
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    truth_np = np.asarray(truth_all[query_ids], dtype=np.complex64)
    if base_np.shape != truth_np.shape:
        raise ValueError(f"base/truth shape mismatch: {base_np.shape} != {truth_np.shape}")

    endpoint_pas_np = {}
    endpoint_pdp_np = {}
    for name, (pas_path, pdp_path, endpoint_path) in RENDERED.items():
        endpoint = np.load(endpoint_path, allow_pickle=False)
        if not np.array_equal(endpoint["query_ids"].astype(np.int64), query_ids):
            raise ValueError(f"query id mismatch for {name}")
        pas, pdp = render_cache(
            endpoint_path,
            str(pas_path),
            str(pdp_path),
            device,
            args.batch_size,
            args.force_render,
        )
        endpoint_pas_np[name] = pas
        endpoint_pdp_np[name] = pdp

    base = torch.as_tensor(np.array(base_np, copy=True), device=device)
    truth = torch.as_tensor(truth_np, device=device)
    truth_pas, truth_pdp = (
        pas_distribution(truth),
        pdp_distribution(truth),
    )
    base_pas = pas_distribution(base)
    base_pdp = pdp_distribution(base)
    baseline = point_metrics(base, truth, truth_pas, truth_pdp)
    candidates = build_candidates()
    metrics: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    descriptions: dict[str, dict[str, object]] = {}
    for index, spec in enumerate(candidates):
        endpoint_pas = torch.as_tensor(
            np.asarray(endpoint_pas_np[spec.endpoint], dtype=np.float32), device=device
        )
        endpoint_pdp = torch.as_tensor(
            np.asarray(endpoint_pdp_np[spec.endpoint], dtype=np.float32), device=device
        )
        with torch.inference_mode():
            prediction = make_candidate(
                base, base_pas, base_pdp, endpoint_pas, endpoint_pdp, spec
            )
            metrics[spec.name] = point_metrics(
                prediction, truth, truth_pas, truth_pdp
            )
        descriptions[spec.name] = asdict(spec)
        if (index + 1) % 100 == 0 or index + 1 == len(candidates):
            print(f"E86 candidate {index + 1}/{len(candidates)}", flush=True)

    baseline_summary = statistics(*baseline)
    rows = []
    for name, values in metrics.items():
        row = {"name": name, **descriptions[name]}
        row.update(metric_summary(values, baseline, fold_id))
        rows.append(row)
    rows.sort(key=lambda row: float(row["selected"]["score"]), reverse=True)
    nested = nested_selection(metrics, baseline, fold_id)

    stitched = np.empty_like(np.asarray(base_np), dtype=np.complex64)
    decisions = nested["decisions"]
    selected_names = [str(row["candidate"]) for row in decisions]
    unique_names = {name for name in selected_names if name != "identity"}
    rendered_channels: dict[str, np.ndarray] = {}
    for name in unique_names:
        spec = next(spec for spec in candidates if spec.name == name)
        endpoint_pas = torch.as_tensor(
            np.asarray(endpoint_pas_np[spec.endpoint], dtype=np.float32), device=device
        )
        endpoint_pdp = torch.as_tensor(
            np.asarray(endpoint_pdp_np[spec.endpoint], dtype=np.float32), device=device
        )
        with torch.inference_mode():
            rendered_channels[name] = make_candidate(
                base, base_pas, base_pdp, endpoint_pas, endpoint_pdp, spec
            ).cpu().numpy().astype(np.complex64)
    for outer, decision in enumerate(decisions):
        held = fold_id == outer
        name = str(decision["candidate"])
        stitched[held] = (
            np.asarray(base_np[held], dtype=np.complex64)
            if name == "identity"
            else rendered_channels[name][held]
        )
    np.save(NESTED_CHANNEL, stitched)

    nested_metrics = point_metrics(
        torch.as_tensor(stitched, device=device), truth, truth_pas, truth_pdp
    )
    payload = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "validation": (
            "Fixed 500-point complete official score; candidate selection uses only "
            "the other four spatial folds for each held fold."
        ),
        "primary_hypothesis": (
            "E85 path energy/support endpoints can provide complementary PAS/PDP "
            "targets when injected weakly into the V11 complex channel, while V11 "
            "remains the phase and power carrier."
        ),
        "query_count": int(len(query_ids)),
        "query_ids_sha256": __import__("hashlib").sha256(query_ids.tobytes()).hexdigest(),
        "baseline": baseline_summary,
        "candidate_count": len(candidates),
        "top_candidates": rows[:40],
        "nested_selection": nested,
        "nested_metric_recheck": statistics(*nested_metrics),
        "rendered_endpoint_paths": {
            name: {
                "pas": str(PATHS.relative(RENDERED[name][0])),
                "pdp": str(PATHS.relative(RENDERED[name][1])),
            }
            for name in RENDERED
        },
        "deployment_gate": {
            "minimum_nested_delta": 0.001,
            "minimum_nested_positive_folds": 4,
            "minimum_nested_worst_fold_delta": -0.001,
            "passed": bool(
                nested["delta"] >= 0.001
                and nested["positive_folds"] >= 4
                and nested["worst_fold_delta"] >= -0.001
            ),
        },
        "outputs": {
            "metrics": str(PATHS.relative(METRICS_PATH)),
            "nested_channel": str(PATHS.relative(NESTED_CHANNEL)),
        },
        "seconds": float(time.time() - started),
    }
    np.savez_compressed(
        METRICS_PATH,
        candidate_names=np.asarray([row["name"] for row in rows], dtype="U160"),
        score=np.asarray([row["selected"]["score"] for row in rows], dtype=np.float64),
        delta=np.asarray([row["delta"] for row in rows], dtype=np.float64),
        positive_folds=np.asarray([row["positive_folds"] for row in rows], dtype=np.int8),
    )
    REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"baseline": baseline_summary, "top": rows[:5], "nested": nested, "gate": payload["deployment_gate"], "seconds": payload["seconds"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
