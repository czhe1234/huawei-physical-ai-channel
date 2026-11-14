#!/usr/bin/env python3
"""Migrate the full Round 1 Sionna spectral chain onto the Round 2 V7 carrier."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from analyze_v4_exact_joint_projection import (
    exact_pas_projection,
    exact_pdp_projection,
    fixed_blend,
    pas_distribution,
    pdp_distribution,
    preserve_point_power,
    truth_views,
)
from analyze_v7_stacked_lifecycle_overlay import (
    OOF_CURRENT,
    OOF_SCALED,
    SELECTION as V7_SELECTION,
)
from prepare_sionna_calibration_dataset import (
    ARTIFACT_DIR as CALIBRATION_DIR,
    BASE_CONFIG,
    SCATTER_CONFIG,
    running_path,
)
from project_paths import PATHS
from round2.data import load_positions
from run_rbf_spectral_field import OOF_META, statistics
from run_v6_regime_component_lifecycle import (
    REGIME_CACHE,
    TARGET_PATHS,
    _candidate_from_name,
    apply_candidate as apply_lifecycle_candidate,
)


VERSION = "Round2-E56-full-Sionna-migration"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "full_sionna_migration"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_full_sionna_migration"
OOF_V7 = ARTIFACT_DIR / "oof_v7_nested_channel.npy"
METRICS = ARTIFACT_DIR / "staged_candidate_metrics.npz"
REPORT = ANALYSIS_DIR / "full_sionna_migration.json"

V7_TEST = (
    PATHS.artifacts
    / "round2"
    / "complex_carrier_lifecycle_overlay"
    / "Round2_V7_Stacked_Lifecycle_Test_Channel.npy"
)
OOF_PAS_TARGET = CALIBRATION_DIR / "oof_pas_mean_f16.npy"
OOF_PDP_TARGET = CALIBRATION_DIR / "oof_pdp_mean_f16.npy"
TEST_PAS_TARGET = CALIBRATION_DIR / "test_pas_mean_ensemble_f16.npy"
TEST_PDP_TARGET = CALIBRATION_DIR / "test_pdp_mean_ensemble_f16.npy"

TINY = 1e-30
MARGINAL_ITERATIONS = 5


@dataclass(frozen=True)
class Action:
    name: str
    marginal_profile: str = "none"
    physical_mix: float = 0.0
    alpha_pas: float = 0.0
    alpha_pdp: float = 0.0
    order: str = "pas_pdp"
    cycles: int = 4


MARGINAL_PROFILES = {
    "soft": {
        "d": (0.02, 0.08, 0.22, 0.30),
        "du": (0.005, 0.025, 0.075, 0.12),
        "dh": (0.004, 0.020, 0.060, 0.10),
    },
    "balanced": {
        "d": (0.04, 0.14, 0.42, 0.55),
        "du": (0.010, 0.050, 0.150, 0.22),
        "dh": (0.008, 0.040, 0.120, 0.18),
    },
    "strong": {
        "d": (0.06, 0.20, 0.62, 0.78),
        "du": (0.015, 0.080, 0.230, 0.34),
        "dh": (0.012, 0.060, 0.180, 0.28),
    },
}


def actions() -> list[Action]:
    result = [
        Action("identity"),
        Action("marginal_balanced_cal", "balanced"),
        Action("marginal_strong_cal", "strong"),
        Action("marginal_balanced_physical", "balanced", 0.30),
        Action("marginal_strong_physical", "strong", 0.40),
        Action("link_pdp_a030", alpha_pdp=0.30),
        Action("link_pdp_a060", alpha_pdp=0.60),
        Action("link_pdp_a090", alpha_pdp=0.90),
        Action("pas_a030", alpha_pas=0.30),
        Action("pas_a045", alpha_pas=0.45),
        Action(
            "full_soft",
            "soft",
            0.20,
            alpha_pas=0.25,
            alpha_pdp=0.30,
        ),
        Action(
            "full_balanced",
            "balanced",
            0.30,
            alpha_pas=0.35,
            alpha_pdp=0.45,
        ),
        Action(
            "full_balanced_cal",
            "balanced",
            0.0,
            alpha_pas=0.35,
            alpha_pdp=0.45,
        ),
        Action(
            "full_strong",
            "strong",
            0.40,
            alpha_pas=0.45,
            alpha_pdp=0.65,
        ),
        Action(
            "full_strong_pdp",
            "strong",
            0.40,
            alpha_pas=0.40,
            alpha_pdp=0.90,
        ),
        Action(
            "full_balanced_c8",
            "balanced",
            0.30,
            alpha_pas=0.35,
            alpha_pdp=0.45,
            cycles=8,
        ),
        Action(
            "full_balanced_pdp_pas",
            "balanced",
            0.30,
            alpha_pas=0.35,
            alpha_pdp=0.45,
            order="pdp_pas",
        ),
    ]
    for alpha_pas in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45):
        for alpha_pdp in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60):
            result.append(
                Action(
                    f"joint_a{int(round(100 * alpha_pas)):03d}_"
                    f"{int(round(100 * alpha_pdp)):03d}",
                    alpha_pas=alpha_pas,
                    alpha_pdp=alpha_pdp,
                )
            )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--reuse-oof", action="store_true")
    parser.add_argument("--reuse-metrics", action="store_true")
    return parser.parse_args()


def normalize(value: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    value = value.clamp_min(0.0)
    total = value.sum(dim=dims, keepdim=True)
    return torch.where(total > TINY, value / total.clamp_min(TINY), 0.0)


def safe_power_ratio(desired: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    return desired.clamp_min(1e-12) / current.clamp_min(1e-12)


def shifted_ad(value: np.ndarray, site: np.ndarray) -> np.ndarray:
    output = np.asarray(value, dtype=np.float32).copy()
    for site_value, shift in ((0, 2), (1, 1)):
        mask = site == site_value
        output[mask] = np.roll(output[mask], shift=shift, axis=2)
    return output


def ad_target(split: str, ids: np.ndarray, site: np.ndarray) -> np.ndarray:
    values = []
    for config in (BASE_CONFIG, SCATTER_CONFIG):
        source = np.load(running_path(config, split, "ad", "mean"), mmap_mode="r")
        local = np.asarray(source[ids] if split == "train" else source[ids], dtype=np.float32)
        values.append(shifted_ad(local, site))
    target = 0.65 * values[0] + 0.35 * values[1]
    total = target.sum(axis=(1, 2, 3), keepdims=True, dtype=np.float64)
    return np.divide(
        target,
        np.maximum(total, TINY),
        out=np.zeros_like(target),
        where=total > 0.0,
    ).astype(np.float32)


def reconstruct_v7_nested(
    device: torch.device, batch_size: int, reuse: bool
) -> np.ndarray:
    if reuse and OOF_V7.is_file():
        return np.load(OOF_V7, mmap_mode="r")
    metadata = np.load(OOF_META, allow_pickle=False)
    fold_id = metadata["fold_id"].astype(np.int8)
    query_ids = metadata["query_ids"].astype(np.int64)
    _, _, _, train_site, _ = load_positions()
    site = train_site[query_ids].astype(np.int8)
    regime = np.load(REGIME_CACHE, allow_pickle=False)["oof_regime"].astype(np.int8)
    selection = json.loads(V7_SELECTION.read_text(encoding="utf-8"))["winner"]
    if selection is None:
        raise RuntimeError("V7 lifecycle selection has no winner")
    chosen = np.full(len(query_ids), "identity", dtype="U72")
    for row in selection["nested_decisions"]:
        mask = (
            (fold_id == int(row["held_fold"]))
            & (site == int(row["site"]))
            & (regime == int({"clear": 0, "light": 1, "medium": 2, "heavy": 3}.get(str(row.get("regime", "")), row.get("regime_index", -1))))
        )
        chosen[mask] = str(row["candidate"])
    base = np.load(OOF_SCALED, mmap_mode="r")
    current = np.load(OOF_CURRENT, mmap_mode="r")
    target = np.load(TARGET_PATHS[str(selection["stage"])][0], mmap_mode="r")
    output = np.lib.format.open_memmap(
        OOF_V7, mode="w+", dtype=np.complex64, shape=base.shape
    )
    for start in range(0, len(base), batch_size):
        end = min(start + batch_size, len(base))
        block = np.array(base[start:end], dtype=np.complex64, copy=True)
        for name in np.unique(chosen[start:end]):
            rows = np.flatnonzero(chosen[start:end] == name)
            if name == "identity" or len(rows) == 0:
                continue
            spec = _candidate_from_name(str(name))
            with torch.inference_mode():
                prediction = apply_lifecycle_candidate(
                    torch.as_tensor(block[rows], device=device),
                    torch.as_tensor(
                        np.asarray(current[start:end][rows], dtype=np.float32),
                        device=device,
                    ),
                    torch.as_tensor(
                        np.asarray(target[start:end][rows], dtype=np.float32),
                        device=device,
                    ),
                    regime[start:end][rows],
                    site[start:end][rows],
                    spec,
                )
            block[rows] = prediction.cpu().numpy().astype(np.complex64)
        output[start:end] = block
        output.flush()
        print(f"V8 reconstruct V7 OOF {end}/{len(base)}", flush=True)
    return np.load(OOF_V7, mmap_mode="r")


def profile_alpha(
    profile: str, key: str, regime: np.ndarray, device: torch.device
) -> torch.Tensor:
    table = torch.as_tensor(
        MARGINAL_PROFILES[profile][key], dtype=torch.float32, device=device
    )
    return table[torch.as_tensor(regime, dtype=torch.long, device=device)]


def marginal_projection(
    channel: torch.Tensor,
    pdp_target: torch.Tensor,
    physical_ad: torch.Tensor,
    regime: np.ndarray,
    action: Action,
) -> torch.Tensor:
    if action.marginal_profile == "none":
        return channel
    point_power = channel.abs().square().sum(dim=(1, 2, 3))
    cir = torch.fft.ifft(channel, dim=-1)
    shaped = cir.reshape(len(channel), 2, 16, 8, 4, 192)
    power = shaped.abs().square().float()
    support = power / power.sum(dim=(1, 2, 3, 4, 5), keepdim=True).clamp_min(TINY)
    link_energy = support.sum(dim=5, keepdim=True)
    link_target = pdp_target.reshape(len(channel), 2, 16, 8, 4, 192)
    link_target = normalize(link_target, (5,))
    target_support = link_target * link_energy
    target_support = normalize(target_support, (1, 2, 3, 4, 5))

    current_d = normalize(support.sum(dim=(1, 2, 3, 4)), (1,))
    current_du = normalize(support.sum(dim=(1, 2, 3)), (1, 2))
    current_dh = normalize(support.sum(dim=(1, 2, 4)), (1, 2))
    target_d = normalize(target_support.sum(dim=(1, 2, 3, 4)), (1,))
    target_du = normalize(target_support.sum(dim=(1, 2, 3)), (1, 2))
    target_dh = normalize(target_support.sum(dim=(1, 2, 4)), (1, 2))

    ad_d = normalize(physical_ad.sum(dim=(1, 2)), (1,))
    ad_dh = normalize(physical_ad.sum(dim=1), (1, 2))
    target_d = normalize(
        (1.0 - action.physical_mix) * target_d + action.physical_mix * ad_d,
        (1,),
    )
    target_dh = normalize(
        (1.0 - action.physical_mix) * target_dh + action.physical_mix * ad_dh,
        (1, 2),
    )

    alpha_d = profile_alpha(action.marginal_profile, "d", regime, channel.device)
    alpha_du = profile_alpha(action.marginal_profile, "du", regime, channel.device)
    alpha_dh = profile_alpha(action.marginal_profile, "dh", regime, channel.device)
    desired_d = normalize(
        (1.0 - alpha_d[:, None]) * current_d + alpha_d[:, None] * target_d,
        (1,),
    )
    desired_du = normalize(
        (1.0 - alpha_du[:, None, None]) * current_du
        + alpha_du[:, None, None] * target_du,
        (1, 2),
    )
    desired_dh = normalize(
        (1.0 - alpha_dh[:, None, None]) * current_dh
        + alpha_dh[:, None, None] * target_dh,
        (1, 2),
    )

    work = support.clone()
    for _ in range(MARGINAL_ITERATIONS):
        work *= safe_power_ratio(
            desired_d, work.sum(dim=(1, 2, 3, 4))
        )[:, None, None, None, None, :].pow(0.96)
        work *= safe_power_ratio(
            desired_du, work.sum(dim=(1, 2, 3))
        )[:, None, None, None, :, :].pow(0.40)
        work *= safe_power_ratio(
            desired_dh, work.sum(dim=(1, 2, 4))
        )[:, None, None, :, None, :].pow(0.34)
        work = normalize(work, (1, 2, 3, 4, 5))

    gain = torch.sqrt(work / support.clamp_min(1e-30)).clamp(0.18, 5.80)
    flat = support.reshape(len(channel), -1)
    gain_flat = gain.reshape(len(channel), -1)
    for count, lower, upper in (
        (8192, 0.44, 2.48),
        (1024, 0.60, 1.74),
        (128, 0.80, 1.34),
    ):
        indices = torch.topk(flat, k=count, dim=1, largest=True).indices
        selected = gain_flat.gather(1, indices).clamp(lower, upper)
        gain_flat.scatter_(1, indices, selected)
    candidate = shaped * gain_flat.reshape_as(shaped)
    candidate = torch.fft.fft(candidate.reshape_as(channel), dim=-1)
    return preserve_point_power(candidate, point_power)


def apply_action(
    base: torch.Tensor,
    pas_target: torch.Tensor,
    pdp_target: torch.Tensor,
    physical_ad: torch.Tensor,
    regime: np.ndarray,
    action: Action,
) -> torch.Tensor:
    if action.name == "identity":
        return base
    reference_power = base.abs().square().sum(dim=(1, 2, 3))
    candidate = marginal_projection(
        base, pdp_target, physical_ad, regime, action
    )
    desired_pas = None
    desired_pdp = None
    if action.alpha_pas > 0.0:
        desired_pas = fixed_blend(
            pas_distribution(candidate), pas_target, action.alpha_pas, (1, 2, 3)
        )
    if action.alpha_pdp > 0.0:
        desired_pdp = fixed_blend(
            pdp_distribution(candidate), pdp_target, action.alpha_pdp, (3,)
        )
    domains = []
    if desired_pas is not None:
        domains.append("pas")
    if desired_pdp is not None:
        domains.append("pdp")
    if len(domains) == 2:
        domains = action.order.split("_")
    for _ in range(action.cycles if domains else 0):
        for domain in domains:
            if domain == "pas":
                candidate = exact_pas_projection(candidate, desired_pas)
            elif domain == "pdp":
                candidate = exact_pdp_projection(candidate, desired_pdp)
            else:
                raise ValueError(domain)
        candidate = preserve_point_power(candidate, reference_power)
    return candidate


def point_metrics(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    truth_pas: torch.Tensor,
    truth_pdp: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shaped = prediction.reshape(len(prediction), 2, 16, 8, 4, 192)
    prediction_pas = torch.fft.fft2(shaped, dim=(2, 3)).abs().square().float()
    prediction_pdp = torch.fft.ifft(prediction, dim=-1).abs().square().float()
    pas_num = (prediction_pas * truth_pas).sum(dim=(1, 2, 3), dtype=torch.float64)
    pas_den = torch.sqrt(
        prediction_pas.square().sum(dim=(1, 2, 3), dtype=torch.float64)
        * truth_pas.square().sum(dim=(1, 2, 3), dtype=torch.float64)
    )
    pas = (pas_num / torch.where(pas_den == 0.0, 1e-12, pas_den)).mean(
        dim=(1, 2), dtype=torch.float64
    )
    pdp_num = (prediction_pdp * truth_pdp).sum(dim=3, dtype=torch.float64)
    pdp_den = torch.sqrt(
        prediction_pdp.square().sum(dim=3, dtype=torch.float64)
        * truth_pdp.square().sum(dim=3, dtype=torch.float64)
    )
    pdp = (pdp_num / torch.where(pdp_den == 0.0, 1e-12, pdp_den)).mean(
        dim=(1, 2), dtype=torch.float64
    )
    error = (prediction - truth).abs().square().sum(
        dim=(1, 2, 3), dtype=torch.float64
    )
    energy = truth.abs().square().sum(dim=(1, 2, 3), dtype=torch.float64)
    return tuple(value.cpu().numpy() for value in (pas, pdp, error, energy))


def generate_metrics(
    base: np.ndarray, device: torch.device, batch_size: int
) -> dict[str, np.ndarray]:
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    _, _, _, train_site, _ = load_positions()
    site = train_site[query_ids].astype(np.int8)
    regime = np.load(REGIME_CACHE, allow_pickle=False)["oof_regime"].astype(np.int8)
    specs = actions()
    result = {
        key: np.empty((len(specs), len(query_ids)), dtype=np.float64)
        for key in ("pas", "pdp", "error", "energy")
    }
    truth_all = np.load(PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r")
    pas_all = np.load(OOF_PAS_TARGET, mmap_mode="r")
    pdp_all = np.load(OOF_PDP_TARGET, mmap_mode="r")
    ad_all = ad_target("train", query_ids, site)
    for start in range(0, len(query_ids), batch_size):
        end = min(start + batch_size, len(query_ids))
        base_tensor = torch.as_tensor(
            np.array(base[start:end], dtype=np.complex64, copy=True), device=device
        )
        truth = torch.as_tensor(
            np.array(truth_all[query_ids[start:end]], dtype=np.complex64, copy=True),
            device=device,
        )
        pas_target = torch.as_tensor(
            np.asarray(pas_all[start:end], dtype=np.float32), device=device
        )
        pdp_target = torch.as_tensor(
            np.asarray(pdp_all[start:end], dtype=np.float32), device=device
        )
        physical_ad = torch.as_tensor(ad_all[start:end], device=device)
        with torch.inference_mode():
            truth_pas, truth_pdp = truth_views(truth)
            for index, action in enumerate(specs):
                prediction = apply_action(
                    base_tensor,
                    pas_target,
                    pdp_target,
                    physical_ad,
                    regime[start:end],
                    action,
                )
                values = point_metrics(prediction, truth, truth_pas, truth_pdp)
                for key, value in zip(("pas", "pdp", "error", "energy"), values):
                    result[key][index, start:end] = value
        print(f"V8 staged grid {end}/{len(query_ids)}", flush=True)
    result.update(
        {
            "candidate_names": np.asarray([action.name for action in specs], dtype="U48"),
            "query_ids": query_ids,
            "fold_id": fold_id,
            "site": site,
            "regime": regime,
        }
    )
    np.savez_compressed(METRICS, **result)
    return result


def candidate_result(
    metrics: dict[str, np.ndarray], index: int, mask: np.ndarray | None = None
) -> dict[str, float | int]:
    return statistics(
        metrics["pas"][index],
        metrics["pdp"][index],
        metrics["error"][index],
        metrics["energy"][index],
        mask,
    )


def selected_result(
    metrics: dict[str, np.ndarray], selected: np.ndarray, mask: np.ndarray | None = None
) -> dict[str, float | int]:
    if mask is None:
        mask = np.ones(len(selected), dtype=bool)
    point = np.arange(len(selected))[mask]
    chosen = selected[mask]
    return statistics(
        metrics["pas"][chosen, point],
        metrics["pdp"][chosen, point],
        metrics["error"][chosen, point],
        metrics["energy"][chosen, point],
    )


def nested_group_policy(
    metrics: dict[str, np.ndarray], use_regime: bool
) -> tuple[np.ndarray, list[dict[str, object]]]:
    selected = np.zeros(len(metrics["fold_id"]), dtype=np.int64)
    decisions = []
    fold_id = metrics["fold_id"]
    site = metrics["site"]
    regime = metrics["regime"]
    for held_fold in np.unique(fold_id):
        for site_value in np.unique(site):
            regimes = np.unique(regime[site == site_value]) if use_regime else [-1]
            for regime_value in regimes:
                fit = (fold_id != held_fold) & (site == site_value)
                held = (fold_id == held_fold) & (site == site_value)
                if int(regime_value) >= 0:
                    fit &= regime == regime_value
                    held &= regime == regime_value
                if not np.any(held):
                    continue
                if np.sum(fit) < 16:
                    fit = (fold_id != held_fold) & (site == site_value)
                rows = [
                    candidate_result(metrics, index, fit)
                    for index in range(len(metrics["candidate_names"]))
                ]
                best = int(np.argmax([float(row["score"]) for row in rows]))
                selected[held] = best
                decisions.append(
                    {
                        "held_fold": int(held_fold),
                        "site": int(site_value),
                        "regime": int(regime_value),
                        "fit_count": int(np.sum(fit)),
                        "held_count": int(np.sum(held)),
                        "candidate": str(metrics["candidate_names"][best]),
                    }
                )
    return selected, decisions


def oracle_policy(metrics: dict[str, np.ndarray]) -> np.ndarray:
    score = (
        0.4 * metrics["pas"]
        + 0.4 * metrics["pdp"]
        + 0.2
        / (
            1.0
            + metrics["error"] / np.maximum(metrics["energy"], TINY)
        )
    )
    return np.argmax(score, axis=0).astype(np.int64)


def policy_summary(
    metrics: dict[str, np.ndarray], selected: np.ndarray
) -> dict[str, object]:
    identity = np.zeros(len(selected), dtype=np.int64)
    folds = []
    for fold in np.unique(metrics["fold_id"]):
        mask = metrics["fold_id"] == fold
        result = selected_result(metrics, selected, mask)
        base = selected_result(metrics, identity, mask)
        folds.append(
            {
                "fold": int(fold),
                "result": result,
                "delta_score": float(result["score"] - base["score"]),
            }
        )
    return {
        "overall": selected_result(metrics, selected),
        "folds": folds,
        "positive_folds": int(sum(float(row["delta_score"]) > 0.0 for row in folds)),
    }


def analyze(metrics: dict[str, np.ndarray]) -> dict[str, object]:
    rows = [
        {
            "candidate": str(name),
            **candidate_result(metrics, index),
        }
        for index, name in enumerate(metrics["candidate_names"])
    ]
    rows.sort(key=lambda row: float(row["score"]), reverse=True)
    best_index = int(
        np.flatnonzero(metrics["candidate_names"] == rows[0]["candidate"])[0]
    )
    fixed = np.full(len(metrics["fold_id"]), best_index, dtype=np.int64)
    site_selected, site_decisions = nested_group_policy(metrics, False)
    regime_selected, regime_decisions = nested_group_policy(metrics, True)
    oracle = oracle_policy(metrics)
    return {
        "identity": candidate_result(metrics, 0),
        "fixed_best": rows[0],
        "fixed_policy": policy_summary(metrics, fixed),
        "site_nested": policy_summary(metrics, site_selected),
        "site_nested_decisions": site_decisions,
        "site_regime_nested": policy_summary(metrics, regime_selected),
        "site_regime_nested_decisions": regime_decisions,
        "pointwise_oracle": policy_summary(metrics, oracle),
        "top_candidates": rows[:12],
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    base = reconstruct_v7_nested(device, args.batch_size, args.reuse_oof)
    if args.reuse_metrics and METRICS.is_file():
        metrics = {key: value for key, value in np.load(METRICS).items()}
    else:
        metrics = generate_metrics(base, device, args.batch_size)
    analysis = analyze(metrics)
    payload = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "The full Round 1 Sionna chain transfers only when low-order delay "
            "marginals, scorer-aligned linkwise PDP, calibrated PAS, and alternating "
            "joint projection are applied as distinct stages on the V7 carrier."
        ),
        "candidate_count": int(len(metrics["candidate_names"])),
        "actions": [action.__dict__ for action in actions()],
        "analysis": analysis,
        "artifacts": {
            "oof_v7": PATHS.relative(OOF_V7),
            "metrics": PATHS.relative(METRICS),
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
