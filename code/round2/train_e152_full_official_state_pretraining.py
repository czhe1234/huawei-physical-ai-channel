#!/usr/bin/env python3
"""Pretrain E150 path state on all strict official fit points before V7 rollout."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import time

import numpy as np
import torch

import build_v7_complex_carrier_bank as carrier_bank
from project_paths import PATHS
from run_rbf_spectral_field import statistics
import train_e115_complete_path_state_operator as e115
import train_e150_v7_pathwise_carrier_operator as e150
import train_e151_identity_supervised_action_gate as e151


VERSION = "Round2-E152-full-official-state-pretraining"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "full_official_state_pretraining"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_full_official_state_pretraining"
REPORT = ANALYSIS_DIR / "full_official_state_pretraining.json"
METRICS = ARTIFACT_DIR / "full_official_state_pretraining_metrics.npz"
REFERENCE_CARRIER = "radial_idw_k24"
POINT_LOG_REFERENCE_INDEX = 10


@dataclass(frozen=True)
class Config(e151.Config):
    full_batch_size: int = 32
    cache_batch_size: int = 16
    seed: int = 20_260_810_152


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--outer", default="0")
    parser.add_argument("--phase", choices=("cache", "train", "all"), default="all")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--pretrain-epochs", type=int, default=18)
    parser.add_argument("--action-warmup-epochs", type=int, default=12)
    parser.add_argument("--finetune-epochs", type=int, default=12)
    parser.add_argument("--full-batch-size", type=int, default=32)
    parser.add_argument("--cache-batch-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--metric-batch-size", type=int, default=8)
    return parser.parse_args()


def reference_carrier_index() -> int:
    names = [config.name for config in carrier_bank.carrier_configs()]
    if REFERENCE_CARRIER not in names:
        raise RuntimeError(f"missing reference carrier {REFERENCE_CARRIER}")
    return names.index(REFERENCE_CARRIER)


def full_cache_dir(outer: int) -> Path:
    return ARTIFACT_DIR / f"outer{outer}_full_state_cache"


def checkpoint_path(outer: int) -> Path:
    return ARTIFACT_DIR / f"outer{outer}_full_state_v7_rollout.pt"


def fold_result_path(outer: int) -> Path:
    return ANALYSIS_DIR / f"outer{outer}_result.json"


def fold_metrics_path(outer: int) -> Path:
    return ARTIFACT_DIR / f"outer{outer}_metrics.npz"


def load_completed_fold(
    outer: int,
) -> tuple[
    dict[str, object],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    record = json.loads(fold_result_path(outer).read_text(encoding="utf-8"))
    metrics = np.load(fold_metrics_path(outer), allow_pickle=False)
    selected = tuple(
        metrics[name].astype(np.float64)
        for name in (
            "selected_pas",
            "selected_pdp",
            "selected_error",
            "selected_energy",
        )
    )
    control = tuple(
        metrics[name].astype(np.float64)
        for name in (
            "control_pas",
            "control_pdp",
            "control_error",
            "control_energy",
        )
    )
    if len({len(value) for value in (*selected, *control)}) != 1:
        raise RuntimeError(f"inconsistent cached E152 metrics for outer{outer}")
    return record, selected, control


def build_full_cache(
    outer: int,
    config: Config,
    device: torch.device,
    force: bool,
) -> dict[str, object]:
    directory = full_cache_dir(outer)
    state_path = directory / "state.json"
    if state_path.is_file() and not force:
        return json.loads(state_path.read_text(encoding="utf-8"))
    fold = np.load(e115.FULL_FOLD).astype(np.int8)
    inner = (outer + 1) % 5
    query_ids = np.flatnonzero(fold != outer)
    fit_count = int(np.sum((fold[query_ids] != inner)))
    validation_count = int(np.sum(fold[query_ids] == inner))
    if len(query_ids) + int(np.sum(fold == outer)) != len(fold):
        raise RuntimeError(
            "unexpected E152 split counts: "
            f"query={len(query_ids)} fit={fit_count} inner={validation_count}"
        )
    state = e150.build_cache(
        outer,
        config,
        device,
        force,
        query_ids_override=query_ids,
        directory_override=directory,
        reference_carrier_index=reference_carrier_index(),
        reference_name=REFERENCE_CARRIER,
        carrier_batch_size=config.cache_batch_size,
    )
    state.update(
        {
            "experiment": VERSION,
            "cache_role": "strict_full_official_state_pretraining",
            "full_fit_count": fit_count,
            "full_validation_count": validation_count,
            "outer_excluded_count": int(np.sum(fold == outer)),
        }
    )
    state_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return state


def load_full_cache(outer: int) -> dict[str, object]:
    directory = full_cache_dir(outer)
    state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    arrays = {
        name: np.load(directory / f"{name}.npy", mmap_mode="r")
        for name in state["arrays"]
    }
    return {"state": state, **arrays}


class FullStateStore:
    def __init__(self, outer: int, device: torch.device) -> None:
        raw = load_full_cache(outer)
        self.outer = outer
        self.device = device
        self.state = raw["state"]
        self.query_ids_np = np.asarray(raw["query_ids"], dtype=np.int64)
        self.query_fold_np = np.asarray(raw["query_fold"], dtype=np.int8)
        self.mask = torch.as_tensor(np.asarray(raw["mask"], bool), device=device)
        self.coordinate = torch.as_tensor(
            np.asarray(raw["coordinate"], np.float32), device=device
        )
        self.source_type = torch.as_tensor(
            np.asarray(raw["source_type"], np.int64), device=device
        )
        self.candidate_feature = torch.as_tensor(
            np.asarray(raw["candidate_feature"], np.float32), device=device
        )
        self.point_feature = torch.as_tensor(
            np.asarray(raw["point_feature"], np.float32), device=device
        )
        self.carrier_coefficient = torch.as_tensor(
            np.asarray(raw["carrier_coefficient"], np.complex64), device=device
        )
        self.reference_coefficient = torch.as_tensor(
            np.asarray(raw["v7_coefficient"], np.complex64), device=device
        )
        self.carrier_log_power = torch.as_tensor(
            np.asarray(raw["carrier_log_power"], np.float32), device=device
        )
        self.reference_log_power = torch.as_tensor(
            np.asarray(raw["v7_log_power"], np.float32), device=device
        )
        targets, self.assignment = e150.target_state(raw)
        self.target_positive = torch.as_tensor(targets["positive"], device=device)
        self.target_mass = torch.as_tensor(
            targets["mass_absolute"], device=device
        )
        self.target_coordinate = torch.as_tensor(
            targets["coordinate"], device=device
        )
        self.target_coefficient = torch.as_tensor(
            targets["coefficient"], device=device
        )
        self.target_point_power = torch.as_tensor(
            targets["point_power"], dtype=torch.float64, device=device
        )

    def normalization(self, fit_rows: np.ndarray) -> dict[str, torch.Tensor]:
        rows = torch.as_tensor(fit_rows, device=self.device)
        valid = self.mask[rows]
        candidate_values = self.candidate_feature[rows][valid]
        candidate_mean = candidate_values.mean(dim=0)
        candidate_std = candidate_values.std(dim=0)
        candidate_std = torch.where(
            candidate_std < 1e-4, torch.ones_like(candidate_std), candidate_std
        )
        point_mean = self.point_feature[rows].mean(dim=0)
        point_std = self.point_feature[rows].std(dim=0)
        point_std = torch.where(
            point_std < 1e-4, torch.ones_like(point_std), point_std
        )
        return {
            "candidate_mean": candidate_mean,
            "candidate_std": candidate_std,
            "point_mean": point_mean,
            "point_std": point_std,
        }

    def batch(
        self,
        rows: torch.Tensor,
        normalization: dict[str, torch.Tensor],
        candidate_override: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        candidate = (
            self.candidate_feature[rows]
            if candidate_override is None
            else candidate_override
        )
        candidate = (candidate - normalization["candidate_mean"]) / normalization[
            "candidate_std"
        ]
        candidate = torch.where(
            self.mask[rows, :, None],
            candidate.clamp(-8.0, 8.0),
            torch.zeros_like(candidate),
        )
        point = (self.point_feature[rows] - normalization["point_mean"]) / normalization[
            "point_std"
        ]
        return {
            "rows": rows,
            "mask": self.mask[rows],
            "coordinate": self.coordinate[rows],
            "source_type": self.source_type[rows],
            "candidate_feature": candidate,
            "point_feature": point.clamp(-8.0, 8.0),
            "carrier_coefficient": self.carrier_coefficient[rows],
            "carrier_log_power": self.carrier_log_power[rows],
            "reference_coefficient": self.reference_coefficient[rows],
            "v7_coefficient": self.reference_coefficient[rows],
            "v7_log_power": self.reference_log_power[rows],
            "target_positive": self.target_positive[rows],
            "target_mass": self.target_mass[rows],
            "target_coordinate": self.target_coordinate[rows],
            "target_coefficient": self.target_coefficient[rows],
            "target_point_power": self.target_point_power[rows],
        }


class ReferenceAlignedFoldStore(e150.FoldStore):
    def __init__(self, outer: int, device: torch.device) -> None:
        super().__init__(outer, device)
        index = reference_carrier_index()
        self.reference_coefficient = self.carrier_coefficient[:, :, index]
        self.reference_log_power = self.carrier_log_power[:, index]
        self.point_feature = self.point_feature.clone()
        self.point_feature[:, POINT_LOG_REFERENCE_INDEX] = self.reference_log_power

    def batch(
        self,
        rows: torch.Tensor,
        normalization: dict[str, torch.Tensor],
        candidate_override: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch = super().batch(rows, normalization, candidate_override)
        batch["reference_coefficient"] = self.reference_coefficient[rows]
        batch["v7_log_power"] = self.reference_log_power[rows]
        return batch


def train_fold(
    outer: int,
    config: Config,
    device: torch.device,
) -> tuple[
    dict[str, object],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    inner = (outer + 1) % 5
    full_store = FullStateStore(outer, device)
    full_fit_rows = np.flatnonzero(full_store.query_fold_np != inner)
    full_validation_rows = np.flatnonzero(full_store.query_fold_np == inner)
    if len(full_fit_rows) + len(full_validation_rows) != len(
        full_store.query_ids_np
    ):
        raise RuntimeError("E152 full state split is incomplete")
    full_normalization = full_store.normalization(full_fit_rows)
    fixed_state = json.loads(
        (e150.cache_dir(outer) / "state.json").read_text(encoding="utf-8")
    )
    if full_store.state["candidate_feature_names"] != fixed_state[
        "candidate_feature_names"
    ]:
        raise RuntimeError("full/fixed candidate feature contracts differ")
    if full_store.state["point_feature_names"] != fixed_state["point_feature_names"]:
        raise RuntimeError("full/fixed point feature contracts differ")

    torch.manual_seed(config.seed + outer * 1009)
    model = e150.PathwiseCarrierOperator(
        full_store.candidate_feature.shape[-1],
        full_store.point_feature.shape[-1],
        config,
    ).to(device)
    full_config = replace(config, batch_size=config.full_batch_size)
    full_state, full_pretrain = e150.train_pretrain(
        model,
        full_store,
        full_fit_rows,
        full_validation_rows,
        full_normalization,
        full_config,
        config.seed + outer * 2027,
    )
    assignment = full_store.assignment
    del full_store
    if device.type == "cuda":
        torch.cuda.empty_cache()

    store = ReferenceAlignedFoldStore(outer, device)
    fit_rows = np.flatnonzero(
        (store.query_fold_np != outer) & (store.query_fold_np != inner)
    )
    validation_rows = np.flatnonzero(store.query_fold_np == inner)
    held_rows = np.flatnonzero(store.query_fold_np == outer)
    transfer_inner_state = e150.evaluate_state_rows(
        model, store, validation_rows, full_normalization, config
    )

    _, action_alpha, action = e151.train_action_warmup(
        model,
        store,
        fit_rows,
        validation_rows,
        full_normalization,
        config,
        config.seed + outer * 4099,
    )
    best_state, alpha, finetune = e151.train_finetune(
        model,
        store,
        fit_rows,
        validation_rows,
        full_normalization,
        config,
        action_alpha,
        config.seed + outer * 8191,
    )
    selected_values, selected_diagnostics = e150.evaluate_model_rows(
        model, store, held_rows, full_normalization, alpha, config
    )
    selected = statistics(*selected_values)
    baseline = statistics(
        *(
            value[held_rows].cpu().numpy()
            for value in (
                store.base_pas,
                store.base_pdp,
                store.base_error,
                store.base_energy,
            )
        )
    )
    transfer_held_state = e150.evaluate_state_rows(
        model, store, held_rows, full_normalization, config
    )
    override, control_record = e150.geometry_control_override(
        store,
        held_rows,
        config.seed + outer * 16_381,
        config.geometry_control_strata,
    )
    control_values, control_diagnostics = e150.evaluate_model_rows(
        model,
        store,
        held_rows,
        full_normalization,
        alpha,
        config,
        override,
    )
    control = statistics(*control_values)
    path = checkpoint_path(outer)
    torch.save(
        {
            "experiment": VERSION,
            "outer": outer,
            "inner": inner,
            "config": asdict(config),
            "state_dict": best_state,
            "full_pretrain_state_dict": full_state,
            "normalization": {
                key: value.detach().cpu()
                for key, value in full_normalization.items()
            },
            "selected_alpha": alpha,
            "reference_carrier": REFERENCE_CARRIER,
            "official_h_only": True,
            "sionna_used": False,
            "v12_used": False,
        },
        path,
    )
    record = {
        "outer": outer,
        "inner": inner,
        "full_fit_count": int(len(full_fit_rows)),
        "full_validation_count": int(len(full_validation_rows)),
        "fixed_fit_count": int(len(fit_rows)),
        "fixed_validation_count": int(len(validation_rows)),
        "held_count": int(len(held_rows)),
        "assignment": assignment,
        "full_pretrain": full_pretrain,
        "transfer_inner_state": transfer_inner_state,
        "action_warmup": action,
        "finetune": finetune,
        "transfer_held_state": transfer_held_state,
        "selected_alpha": alpha,
        "held_baseline": baseline,
        "held_selected": selected,
        "held_delta": float(selected["score"] - baseline["score"]),
        "selected_diagnostics": selected_diagnostics,
        "geometry_control": {
            **control_record,
            "metrics": control,
            "diagnostics": control_diagnostics,
        },
        "geometry_margin": float(selected["score"] - control["score"]),
        "outer_screen_passed": bool(selected["score"] > baseline["score"]),
        "checkpoint": PATHS.relative(path),
    }
    np.savez_compressed(
        fold_metrics_path(outer),
        selected_pas=selected_values[0],
        selected_pdp=selected_values[1],
        selected_error=selected_values[2],
        selected_energy=selected_values[3],
        control_pas=control_values[0],
        control_pdp=control_values[1],
        control_error=control_values[2],
        control_energy=control_values[3],
    )
    fold_result_path(outer).write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"E152 outer={outer} held={selected['score']:.6f} "
        f"delta={record['held_delta']:+.6f} "
        f"geometry_margin={record['geometry_margin']:+.6f}",
        flush=True,
    )
    del model, store
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record, selected_values, control_values


def main() -> None:
    args = parse_args()
    started = time.time()
    config = Config(
        pretrain_epochs=args.pretrain_epochs,
        action_warmup_epochs=args.action_warmup_epochs,
        finetune_epochs=args.finetune_epochs,
        full_batch_size=args.full_batch_size,
        cache_batch_size=args.cache_batch_size,
        batch_size=args.batch_size,
        metric_batch_size=args.metric_batch_size,
    )
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed % (2**32 - 1))
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    outers = e150.selected_outers(args.outer)
    for outer in outers:
        if not (e150.cache_dir(outer) / "state.json").is_file():
            e150.build_cache(
                outer,
                config,
                device,
                False,
                carrier_batch_size=config.cache_batch_size,
            )
        build_full_cache(outer, config, device, args.force_cache)
    if args.phase == "cache":
        return

    baseline = e150.full_baseline(config, device)
    fold_all = np.load(e115.FULL_FOLD).astype(np.int8)
    query_fold = fold_all[baseline["query_ids"]]
    selected_values = tuple(np.empty(500, dtype=np.float64) for _ in range(4))
    control_values = tuple(np.empty(500, dtype=np.float64) for _ in range(4))
    fold_records = []
    for outer in outers:
        completed = (
            fold_result_path(outer).is_file()
            and fold_metrics_path(outer).is_file()
            and checkpoint_path(outer).is_file()
        )
        if completed and not args.force_train:
            record, values, control = load_completed_fold(outer)
            print(f"E152 outer={outer} reused completed fold", flush=True)
        else:
            record, values, control = train_fold(outer, config, device)
        held = np.flatnonzero(query_fold == outer)
        for destination, value in zip(selected_values, values, strict=True):
            destination[held] = value
        for destination, value in zip(control_values, control, strict=True):
            destination[held] = value
        fold_records.append(record)

    complete = len(outers) == 5
    selected = statistics(*selected_values) if complete else None
    control = statistics(*control_values) if complete else None
    gate = None
    if complete and selected is not None and control is not None:
        fold_delta = [float(record["held_delta"]) for record in fold_records]
        gate = {
            "required_fixed500_score": config.required_score,
            "required_v7_gain": config.required_v7_gain,
            "required_positive_folds": config.required_positive_folds,
            "required_geometry_margin": config.required_geometry_margin,
            "fixed500_score": float(selected["score"]),
            "v7_gain": float(selected["score"] - baseline["metrics"]["score"]),
            "positive_folds": int(sum(value > 0.0 for value in fold_delta)),
            "minimum_fold_delta": float(min(fold_delta)),
            "geometry_margin": float(selected["score"] - control["score"]),
            "pas_non_worse": bool(selected["pas"] >= baseline["metrics"]["pas"]),
            "pdp_non_worse": bool(selected["pdp"] >= baseline["metrics"]["pdp"]),
            "nmse_non_worse": bool(selected["nmse"] <= baseline["metrics"]["nmse"]),
        }
        gate["passed"] = bool(
            gate["fixed500_score"] > gate["required_fixed500_score"]
            and gate["v7_gain"] >= gate["required_v7_gain"]
            and gate["positive_folds"] >= gate["required_positive_folds"]
            and gate["geometry_margin"] >= gate["required_geometry_margin"]
            and gate["pas_non_worse"]
            and gate["pdp_non_worse"]
            and gate["nmse_non_worse"]
        )
        np.savez_compressed(
            METRICS,
            query_ids=baseline["query_ids"].astype(np.int32),
            baseline_pas=baseline["values"][0],
            baseline_pdp=baseline["values"][1],
            baseline_error=baseline["values"][2],
            baseline_energy=baseline["values"][3],
            selected_pas=selected_values[0],
            selected_pdp=selected_values[1],
            selected_error=selected_values[2],
            selected_energy=selected_values[3],
            control_pas=control_values[0],
            control_pdp=control_values[1],
            control_error=control_values[2],
            control_energy=control_values[3],
        )

    report = {
        "experiment": VERSION,
        "status": (
            "analysis_only_positive"
            if gate is not None and gate["passed"]
            else "analysis_only_negative"
            if gate is not None
            else "analysis_only_positive_outer_screen"
            if fold_records and all(row["outer_screen_passed"] for row in fold_records)
            else "analysis_only_negative_outer0_screen"
        ),
        "single_experimental_goal": (
            "Test whether scaling strict official path-state supervision from the "
            "fixed-500 fit subset to every strict full-data fit point makes the "
            "E151 V7 action "
            "correction transfer across a held spatial hole."
        ),
        "protocol": {
            "full_split_rule": "exclude outer fold; use inner fold only for epoch selection",
            "reference_carrier": REFERENCE_CARRIER,
            "e150_candidates_and_renderer_frozen": True,
            "e151_action_supervision_frozen": True,
            "official_h_only": True,
            "sionna_used": False,
            "v12_used": False,
            "submission_generated": False,
        },
        "config": asdict(config),
        "baseline_v7": baseline["metrics"],
        "folds": fold_records,
        "selected": selected,
        "geometry_control": control,
        "gate": gate,
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "baseline_v7": baseline["metrics"],
                "folds": [
                    {
                        "outer": row["outer"],
                        "full_pretrain_best": row["full_pretrain"][
                            "best_validation_state_loss"
                        ],
                        "transfer_inner_state": row["transfer_inner_state"],
                        "held_selected": row["held_selected"],
                        "held_delta": row["held_delta"],
                        "geometry_margin": row["geometry_margin"],
                        "outer_screen_passed": row["outer_screen_passed"],
                    }
                    for row in fold_records
                ],
                "selected": selected,
                "gate": gate,
                "seconds": report["seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
