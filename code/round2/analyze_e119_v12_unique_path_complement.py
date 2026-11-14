#!/usr/bin/env python3
"""Test whether the E118 unique-path endpoint adds information to V12."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import time

import numpy as np
import torch

import analyze_e86_surface_endpoint_v11_blend as e86
import train_e115_complete_path_state_operator as e115
import train_e118_hierarchical_unique_path_operator as e118
from analyze_v4_exact_joint_projection import pas_distribution, pdp_distribution
from project_paths import PATHS
from run_v9_joint_sionna_migration import point_metrics
from run_rbf_spectral_field import statistics
from train_full_path_pas_slice_gate import project_joint
from train_surface_interaction_energy_model import render_spectra


VERSION = "Round2-E119-v12-unique-path-complement"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v12_unique_path_complement"
REPORT = ANALYSIS_DIR / "v12_unique_path_complement.json"
ENDPOINTS = ANALYSIS_DIR / "strict_outer0_unique_path_endpoints.npz"
V12_CHANNEL = (
    PATHS.artifacts
    / "round2"
    / "surface_interaction_energy_model"
    / "e86_frozen_geometry_sequence_oof_channel.npy"
)
OOF_META = PATHS.artifacts / "round2" / "pre_sionna" / "round2_pre_sionna_oof_foundation_meta.npz"
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"


STATE_COMBINATIONS: dict[str, tuple[str, ...]] = {
    "raw": (),
    "mass": ("mass",),
    "coordinate": ("coordinate",),
    "port": ("port",),
    "mass_coordinate": ("mass", "coordinate"),
    "mass_port": ("mass", "port"),
    "coordinate_port": ("coordinate", "port"),
    "full": ("mass", "coordinate", "port"),
}


@dataclass(frozen=True)
class Action:
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


PRIMARY = Action(
    endpoint="scene_mass_coordinate",
    mode="sqrt",
    pas_beta=0.10,
    pdp_beta=0.10,
    order="pas_pdp",
    cycles=4,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def mixed_output(
    raw: dict[str, torch.Tensor],
    learned: dict[str, torch.Tensor],
    states: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    output = dict(raw)
    if "mass" in states:
        output["mass"] = learned["mass"]
    if "coordinate" in states:
        output["coordinate"] = learned["coordinate"]
    if "port" in states:
        output["port"] = learned["port"]
    return output


def load_model(
    stage: str,
    store: e118.ClusterTensorStore,
    device: torch.device,
) -> tuple[e118.HierarchicalUniquePathOperator, dict[str, torch.Tensor]]:
    checkpoint_path = e118.ARTIFACT_DIR / f"outer0_hierarchical_{stage}.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = e118.Config(**checkpoint["config"])
    variant = e115.Variant(**checkpoint["variant"])
    model = e118.HierarchicalUniquePathOperator(
        store.raw.candidate_feature.shape[-1],
        store.cluster_feature.shape[-1],
        store.raw.context.shape[-1],
        store.graph_feature.shape[-1],
        config,
        variant,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    normalization = {
        key: value.to(device) for key, value in checkpoint["normalization"].items()
    }
    return model, normalization


@torch.inference_mode()
def render_endpoints(
    store: e118.ClusterTensorStore,
    rows: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    endpoints: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}
    for stage in ("core", "scene", "graph"):
        model, normalization = load_model(stage, store, device)
        combinations = STATE_COMBINATIONS if stage == "scene" else {"full": STATE_COMBINATIONS["full"]}
        for state_name in combinations:
            endpoints[f"{stage}_{state_name}"] = ([], [])
        for start in range(0, len(rows), batch_size):
            local = rows[start : start + batch_size]
            batch = store.batch(torch.as_tensor(local, device=device), normalization)
            raw = e115.raw_output(batch)
            learned = model(batch)
            for state_name, states in combinations.items():
                output = mixed_output(raw, learned, states)
                rendered = {
                    "mass": output["mass"],
                    "coordinate": output["coordinate"],
                    "port": output["port"].reshape(
                        len(local), output["mass"].shape[1], e115.MP, e115.N_UE
                    ),
                }
                pas, pdp = render_spectra(rendered)
                endpoints[f"{stage}_{state_name}"][0].append(
                    pas.cpu().numpy().astype(np.float32)
                )
                endpoints[f"{stage}_{state_name}"][1].append(
                    pdp.cpu().numpy().astype(np.float32)
                )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        name: (np.concatenate(values[0]), np.concatenate(values[1]))
        for name, values in endpoints.items()
    }


def expand_endpoint(
    pas: np.ndarray, pdp: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    pas_tensor = torch.as_tensor(pas, device=device)
    pas_tensor = pas_tensor[..., None].expand(-1, -1, -1, -1, -1, e115.S_SC)
    pdp_tensor = torch.as_tensor(pdp, device=device)
    pdp_tensor = pdp_tensor[:, :, None, None, :, :].expand(
        -1, e115.MP, e115.MH, e115.MV, e115.N_UE, e115.S_SC
    )
    return pas_tensor, pdp_tensor.reshape(-1, e115.MP * e115.MH * e115.MV, e115.N_UE, e115.S_SC)


def make_candidate(
    base: torch.Tensor,
    base_pas: torch.Tensor,
    base_pdp: torch.Tensor,
    endpoint_pas: torch.Tensor,
    endpoint_pdp: torch.Tensor,
    action: Action,
) -> torch.Tensor:
    desired_pas = e86.endpoint_target(
        base_pas, endpoint_pas, action.pas_beta, action.mode, (1, 2, 3)
    )
    desired_pdp = e86.endpoint_target(
        base_pdp, endpoint_pdp, action.pdp_beta, action.mode, (3,)
    )
    return project_joint(
        base, desired_pas, desired_pdp, action.cycles, action.order
    )


def diagnostic_actions(endpoint_names: list[str]) -> list[Action]:
    pairs = (
        (0.05, 0.0),
        (0.10, 0.0),
        (0.20, 0.0),
        (0.0, 0.05),
        (0.0, 0.10),
        (0.0, 0.20),
        (0.05, 0.05),
        (0.10, 0.10),
        (0.20, 0.20),
        (0.10, 0.05),
        (0.05, 0.10),
    )
    actions = []
    for endpoint in endpoint_names:
        for pas_beta, pdp_beta in pairs:
            for order in ("pas_pdp", "pdp_pas"):
                actions.append(
                    Action(endpoint, "sqrt", pas_beta, pdp_beta, order, 4)
                )
    for mode in ("power",):
        for pas_beta, pdp_beta in ((0.05, 0.05), (0.10, 0.10), (0.20, 0.20)):
            actions.append(
                Action("scene_mass_coordinate", mode, pas_beta, pdp_beta, "pas_pdp", 4)
            )
    return actions


def metric_row(
    action: Action,
    values: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    baseline: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, object]:
    selected = statistics(*values)
    anchor = statistics(*baseline)
    return {
        "name": action.name,
        **asdict(action),
        "metrics": selected,
        "delta": float(selected["score"] - anchor["score"]),
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    canonical_fold = metadata["fold_id"].astype(np.int8)
    full_fold = np.load(e118.array_path(e118.cache_dir(), "fold_id"))
    selected = full_fold[query_ids] == 0
    selected_query_ids = query_ids[selected]
    selected_canonical_rows = np.flatnonzero(selected)
    store = e118.ClusterTensorStore(e118.cache_dir(), device)
    endpoints = render_endpoints(
        store, selected_query_ids, device, args.batch_size
    )
    np.savez_compressed(
        ENDPOINTS,
        query_ids=selected_query_ids,
        canonical_rows=selected_canonical_rows,
        **{
            f"{name}_{kind}": values[index].astype(np.float16)
            for name, values in endpoints.items()
            for index, kind in enumerate(("pas", "pdp"))
        },
    )

    base_all = np.load(V12_CHANNEL, mmap_mode="r")
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    base = torch.as_tensor(
        np.asarray(base_all[selected_canonical_rows], dtype=np.complex64), device=device
    )
    truth = torch.as_tensor(
        np.asarray(truth_all[selected_query_ids], dtype=np.complex64), device=device
    )
    truth_pas = pas_distribution(truth)
    truth_pdp = pdp_distribution(truth)
    base_pas = pas_distribution(base)
    base_pdp = pdp_distribution(base)
    baseline = point_metrics(base, truth, truth_pas, truth_pdp)
    expanded = {
        name: expand_endpoint(*values, device)
        for name, values in endpoints.items()
    }
    actions = diagnostic_actions(list(endpoints))
    if PRIMARY.name not in {action.name for action in actions}:
        actions.append(PRIMARY)
    rows = []
    primary_metrics = None
    for index, action in enumerate(actions):
        with torch.inference_mode():
            prediction = make_candidate(
                base,
                base_pas,
                base_pdp,
                *expanded[action.endpoint],
                action,
            )
            values = point_metrics(prediction, truth, truth_pas, truth_pdp)
        row = metric_row(action, values, baseline)
        rows.append(row)
        if action.name == PRIMARY.name:
            primary_metrics = row
        if (index + 1) % 50 == 0:
            print(f"evaluated {index + 1}/{len(actions)}", flush=True)
    rows.sort(key=lambda row: float(row["metrics"]["score"]), reverse=True)
    if primary_metrics is None:
        raise RuntimeError("primary action was not evaluated")
    baseline_summary = statistics(*baseline)
    fold_rows = []
    for fold in range(5):
        mask = canonical_fold[selected] == fold
        fold_rows.append(
            {
                "canonical_fold": fold,
                "count": int(np.sum(mask)),
                "baseline": statistics(*baseline, mask) if np.any(mask) else None,
            }
        )
    report = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "The strict E118 unique-path endpoint corrects PAS/PDP errors that remain "
            "in V12 while V12 continues to carry phase and total power."
        ),
        "official_h_only_for_e118": True,
        "sionna_used_by_v12_base": True,
        "protocol": {
            "e118_fit_folds": [2, 3, 4],
            "e118_validation_fold": 1,
            "strict_e118_held_fold": 0,
            "canonical_subset_count": int(len(selected_query_ids)),
            "primary_action_frozen_before_evaluation": asdict(PRIMARY),
            "diagnostic_grid_is_not_a_deployable_selection": True,
        },
        "baseline": baseline_summary,
        "primary": primary_metrics,
        "primary_delta": float(primary_metrics["delta"]),
        "primary_gate": {
            "minimum_delta": 0.001,
            "passed": bool(float(primary_metrics["delta"]) >= 0.001),
        },
        "diagnostic_best": rows[0],
        "diagnostic_top20": rows[:20],
        "canonical_fold_composition": fold_rows,
        "outputs": {"endpoints": PATHS.relative(ENDPOINTS)},
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
