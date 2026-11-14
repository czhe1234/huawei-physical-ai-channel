#!/usr/bin/env python3
"""Train five strict E118 folds and test their stitched endpoint on V12."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import time

import numpy as np
import torch

import analyze_e119_v12_unique_path_complement as e119
import train_e115_complete_path_state_operator as e115
import train_e118_hierarchical_unique_path_operator as e118
from analyze_v4_exact_joint_projection import pas_distribution, pdp_distribution
from project_paths import PATHS
from run_rbf_spectral_field import statistics
from run_v9_joint_sionna_migration import point_metrics
from train_surface_interaction_energy_model import render_spectra


VERSION = "Round2-E120-fivefold-unique-path-v12"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "fivefold_unique_path_v12"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_fivefold_unique_path_v12"
REPORT = ANALYSIS_DIR / "fivefold_unique_path_v12.json"
ENDPOINT = ARTIFACT_DIR / "strict_oof_mass_coordinate_endpoint.npz"
OOF_CHANNEL = ARTIFACT_DIR / "strict_oof_selected_v12_channel.npy"

SCREEN_ACTION = e119.Action(
    endpoint="fivefold_mass_coordinate",
    mode="sqrt",
    pas_beta=0.20,
    pdp_beta=0.0,
    order="pdp_pas",
    cycles=4,
)
PRIMARY_ACTION = e119.Action(
    endpoint="fivefold_mass_coordinate",
    mode="sqrt",
    pas_beta=0.10,
    pdp_beta=0.10,
    order="pas_pdp",
    cycles=4,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("prepare", "train", "screen", "evaluate"),
        default="prepare",
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--core-epochs", type=int, default=24)
    parser.add_argument("--scene-epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def fold_tags(outer: int) -> tuple[int, str, str]:
    inner = (outer + 1) % 5
    raw_tag = f"outer{outer}_block{outer}{inner}"
    unique_tag = f"outer{outer}_block{outer}{inner}"
    return inner, raw_tag, unique_tag


def prepare(config: e118.Config, force: bool) -> dict[str, object]:
    rows = []
    raw_config = e115.Config(batch_size=config.batch_size)
    for outer in range(5):
        inner, raw_tag, unique_tag = fold_tags(outer)
        raw = e115.build_cache(raw_tag, (outer, inner), raw_config, force)
        unique = e118.build_cache(config, force, unique_tag, raw_tag)
        rows.append(
            {
                "outer": outer,
                "inner": inner,
                "raw_tag": raw_tag,
                "unique_tag": unique_tag,
                "raw_seconds": raw["seconds"],
                "unique_seconds": unique["seconds"],
                "cluster_count_median": unique["cluster_count_median"],
                "retained_target_mass_median": unique[
                    "retained_target_mass_median"
                ],
            }
        )
        print(f"prepared outer={outer} inner={inner}", flush=True)
    return {"experiment": VERSION, "status": "fivefold_cache_complete", "folds": rows}


@torch.inference_mode()
def render_mass_coordinate(
    model: e118.HierarchicalUniquePathOperator,
    store: e118.ClusterTensorStore,
    rows: np.ndarray,
    normalization: dict[str, torch.Tensor],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    pas_parts = []
    pdp_parts = []
    model.eval()
    for start in range(0, len(rows), batch_size):
        local = rows[start : start + batch_size]
        batch = store.batch(torch.as_tensor(local, device=store.device), normalization)
        raw = e115.raw_output(batch)
        learned = model(batch)
        rendered = {
            "mass": learned["mass"],
            "coordinate": learned["coordinate"],
            "port": raw["port"].reshape(
                len(local), learned["mass"].shape[1], e115.MP, e115.N_UE
            ),
        }
        pas, pdp = render_spectra(rendered)
        pas_parts.append(pas.cpu().numpy().astype(np.float32))
        pdp_parts.append(pdp.cpu().numpy().astype(np.float32))
    return np.concatenate(pas_parts), np.concatenate(pdp_parts)


def fold_metric_rows(
    values: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    baseline: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    fold_id: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for fold in range(5):
        mask = fold_id == fold
        selected = statistics(*values, mask)
        anchor = statistics(*baseline, mask)
        rows.append(
            {
                "fold": fold,
                "count": int(np.sum(mask)),
                "selected": selected,
                "baseline": anchor,
                "delta": float(selected["score"] - anchor["score"]),
            }
        )
    return rows


def evaluate_endpoint(
    config: e118.Config,
    device: torch.device,
    endpoint_pas: np.ndarray,
    endpoint_pdp: np.ndarray,
    fold_results: list[dict[str, object]],
    started: float,
) -> dict[str, object]:
    metadata = np.load(e119.OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    full_fold = np.load(e115.FULL_FOLD).astype(np.int8)
    base_np = np.load(e119.V12_CHANNEL, mmap_mode="r")
    truth_all = np.load(e119.TRAIN_CHANNEL, mmap_mode="r")
    base = torch.as_tensor(np.asarray(base_np, np.complex64), device=device)
    truth = torch.as_tensor(
        np.asarray(truth_all[query_ids], dtype=np.complex64), device=device
    )
    truth_pas = pas_distribution(truth)
    truth_pdp = pdp_distribution(truth)
    base_pas = pas_distribution(base)
    base_pdp = pdp_distribution(base)
    endpoint_expanded = e119.expand_endpoint(endpoint_pas, endpoint_pdp, device)
    baseline = point_metrics(base, truth, truth_pas, truth_pdp)
    actions = (PRIMARY_ACTION, SCREEN_ACTION)
    action_rows = []
    channels: dict[str, np.ndarray] = {}
    for action in actions:
        with torch.inference_mode():
            prediction = e119.make_candidate(
                base,
                base_pas,
                base_pdp,
                *endpoint_expanded,
                action,
            )
            values = point_metrics(prediction, truth, truth_pas, truth_pdp)
        selected = statistics(*values)
        anchor = statistics(*baseline)
        folds = fold_metric_rows(values, baseline, full_fold[query_ids])
        action_rows.append(
            {
                "name": action.name,
                "action": asdict(action),
                "selected": selected,
                "baseline": anchor,
                "delta": float(selected["score"] - anchor["score"]),
                "positive_folds": int(sum(row["delta"] > 0.0 for row in folds)),
                "worst_fold_delta": float(min(row["delta"] for row in folds)),
                "folds": folds,
            }
        )
        channels[action.name] = prediction.cpu().numpy().astype(np.complex64)
    action_by_name = {str(row["name"]): row for row in action_rows}
    selected = action_by_name[PRIMARY_ACTION.name]
    diagnostic = action_by_name[SCREEN_ACTION.name]
    gate = {
        "minimum_delta": 0.001,
        "minimum_positive_folds": 4,
        "minimum_worst_fold_delta": -0.001,
        "delta": selected["delta"],
        "positive_folds": selected["positive_folds"],
        "worst_fold_delta": selected["worst_fold_delta"],
    }
    gate["passed"] = bool(
        gate["delta"] >= gate["minimum_delta"]
        and gate["positive_folds"] >= gate["minimum_positive_folds"]
        and gate["worst_fold_delta"] >= gate["minimum_worst_fold_delta"]
    )
    if gate["passed"]:
        np.save(OOF_CHANNEL, channels[str(selected["name"])])
    report = {
        "experiment": VERSION,
        "status": "fivefold_screen_complete",
        "primary_hypothesis": (
            "Unique-path mass and coordinate states learned from official H provide a "
            "cross-fold PAS correction that remains complementary to V12."
        ),
        "protocol": {
            "outer_folds": 5,
            "inner_validation_rule": "(outer + 1) % 5",
            "candidate_donors_exclude_outer_and_inner": True,
            "endpoint_state": "learned mass + learned coordinate + raw cluster port",
            "actions_frozen_from_e119": [asdict(action) for action in actions],
        },
        "config": asdict(config),
        "fold_training": fold_results,
        "baseline": statistics(*baseline),
        "actions": action_rows,
        "selected": selected,
        "diagnostic": diagnostic,
        "deployment_gate": gate,
        "outputs": {
            "endpoint": PATHS.relative(ENDPOINT),
            "oof_channel": PATHS.relative(OOF_CHANNEL) if gate["passed"] else None,
        },
        "seconds": float(time.time() - started),
    }
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "baseline": report["baseline"],
                "actions": action_rows,
                "gate": gate,
                "seconds": report["seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return report


def clean_endpoint_report(
    config: e118.Config,
    endpoint_pas: np.ndarray,
    endpoint_pdp: np.ndarray,
    fold_results: list[dict[str, object]],
    started: float,
    reused: bool,
) -> dict[str, object]:
    if endpoint_pas.shape[0] != 500 or endpoint_pdp.shape[0] != 500:
        raise RuntimeError("E120 Clean endpoint must contain 500 internal OOF rows")
    finite = bool(np.isfinite(endpoint_pas).all() and np.isfinite(endpoint_pdp).all())
    if not finite:
        raise RuntimeError("E120 Clean endpoint contains non-finite values")
    report = {
        "experiment": VERSION,
        "status": "clean_endpoint_ready",
        "official_inputs_only": True,
        "sionna_used": False,
        "v12_evaluation_used": False,
        "config": asdict(config),
        "fold_training": fold_results,
        "endpoint": {
            "path": PATHS.relative(ENDPOINT),
            "pas_shape": list(endpoint_pas.shape),
            "pdp_shape": list(endpoint_pdp.shape),
            "finite": finite,
            "reused": reused,
        },
        "seconds": float(time.time() - started),
    }
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def run_screen(
    config: e118.Config,
    args: argparse.Namespace,
    clean_only: bool = False,
) -> dict[str, object]:
    started = time.time()
    if clean_only and ENDPOINT.is_file() and not args.force:
        endpoint = np.load(ENDPOINT, allow_pickle=False)
        return clean_endpoint_report(
            config,
            endpoint["pas"].astype(np.float32),
            endpoint["pdp"].astype(np.float32),
            [],
            started,
            reused=True,
        )
    device = torch.device(args.device)
    metadata = np.load(e119.OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    canonical_fold = metadata["fold_id"].astype(np.int8)
    full_fold = np.load(e115.FULL_FOLD).astype(np.int8)
    endpoint_pas = np.zeros(
        (len(query_ids), e115.MP, e115.MH, e115.MV, e115.N_UE), np.float32
    )
    endpoint_pdp = np.zeros(
        (len(query_ids), e115.MP, e115.N_UE, e115.S_SC), np.float32
    )
    fold_results = []
    for outer in range(5):
        inner, _, unique_tag = fold_tags(outer)
        directory = e118.cache_dir(unique_tag)
        if not (directory / "state.json").is_file():
            raise FileNotFoundError(f"missing cache for outer={outer}: {directory}")
        store = e118.ClusterTensorStore(directory, device)
        fold_id = store.fold_id.cpu().numpy()
        held_rows = np.flatnonzero(fold_id == outer)
        validation_rows = np.flatnonzero(fold_id == inner)
        fit_rows = np.flatnonzero(~np.isin(fold_id, (outer, inner)))
        normalization = store.normalization(fit_rows)
        seed = (config.random_state + 10000 * outer) % (2**32)
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = e118.HierarchicalUniquePathOperator(
            store.raw.candidate_feature.shape[-1],
            store.cluster_feature.shape[-1],
            store.raw.context.shape[-1],
            store.graph_feature.shape[-1],
            config,
            e115.Variant(f"outer{outer}_core", False, False),
        ).to(device)
        core = e118.train_stage(
            model,
            store,
            fit_rows,
            validation_rows,
            held_rows,
            normalization,
            replace(config, epochs=args.core_epochs),
            "core",
            e115.Variant(f"outer{outer}_core", False, False),
            seed,
            checkpoint_prefix=f"e120_outer{outer}_hierarchical",
        )
        scene = e118.train_stage(
            model,
            store,
            fit_rows,
            validation_rows,
            held_rows,
            normalization,
            replace(config, epochs=args.scene_epochs),
            "scene",
            e115.Variant(f"outer{outer}_scene", True, False),
            seed + 1,
            checkpoint_prefix=f"e120_outer{outer}_hierarchical",
        )
        canonical_rows = np.flatnonzero(full_fold[query_ids] == outer)
        local_pas, local_pdp = render_mass_coordinate(
            model,
            store,
            query_ids[canonical_rows],
            normalization,
            args.batch_size,
        )
        endpoint_pas[canonical_rows] = local_pas
        endpoint_pdp[canonical_rows] = local_pdp
        fold_results.append(
            {
                "outer": outer,
                "inner": inner,
                "fit_count": int(len(fit_rows)),
                "held_count": int(len(held_rows)),
                "canonical_count": int(len(canonical_rows)),
                "core": core,
                "scene": scene,
            }
        )
        print(f"finished outer={outer} canonical={len(canonical_rows)}", flush=True)
        del model, store
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if np.any(endpoint_pas.sum(axis=(1, 2, 3, 4)) <= 0.0):
        raise RuntimeError("one or more canonical endpoints were not populated")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ENDPOINT,
        query_ids=query_ids,
        full_fold=full_fold[query_ids],
        canonical_fold=canonical_fold,
        pas=endpoint_pas.astype(np.float16),
        pdp=endpoint_pdp.astype(np.float16),
    )
    if clean_only:
        return clean_endpoint_report(
            config,
            endpoint_pas,
            endpoint_pdp,
            fold_results,
            started,
            reused=False,
        )
    return evaluate_endpoint(
        config, device, endpoint_pas, endpoint_pdp, fold_results, started
    )


def run_evaluate(config: e118.Config, args: argparse.Namespace) -> dict[str, object]:
    if not ENDPOINT.is_file() or not REPORT.is_file():
        raise FileNotFoundError("E120 endpoint and original screen report are required")
    previous = json.loads(REPORT.read_text(encoding="utf-8"))
    endpoint = np.load(ENDPOINT, allow_pickle=False)
    metadata = np.load(e119.OOF_META, allow_pickle=False)
    if not np.array_equal(endpoint["query_ids"], metadata["query_ids"]):
        raise RuntimeError("E120 endpoint query IDs do not match canonical metadata")
    return evaluate_endpoint(
        config,
        torch.device(args.device),
        endpoint["pas"].astype(np.float32),
        endpoint["pdp"].astype(np.float32),
        previous["fold_training"],
        time.time(),
    )


def main() -> None:
    args = parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")
    config = e118.Config(batch_size=args.batch_size)
    if args.stage == "prepare":
        print(json.dumps(prepare(config, args.force), ensure_ascii=False, indent=2))
    elif args.stage == "train":
        run_screen(config, args, clean_only=True)
    elif args.stage == "screen":
        run_screen(config, args)
    else:
        run_evaluate(config, args)


if __name__ == "__main__":
    main()
