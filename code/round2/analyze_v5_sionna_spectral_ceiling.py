#!/usr/bin/env python3
"""Measure the Round 2 score ceiling of compact Sionna spectral targets."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import time

import numpy as np
import torch

from analyze_v4_exact_joint_projection import (
    ad_distribution,
    exact_ad_projection,
    exact_pas_projection,
    exact_pdp_projection,
    fixed_blend,
    pas_distribution,
    pdp_distribution,
    preserve_point_power,
    truth_views,
)
from project_paths import PATHS
from round2.data import load_positions
from run_rbf_spectral_field import OOF_META, statistics


VERSION = "Round2-E38-Sionna-spectral-ceiling"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "v5_sionna_upper_bound"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v5_sionna_upper_bound"
METRICS = ARTIFACT_DIR / "sionna_spectral_ceiling_metrics.npz"
REPORT = ANALYSIS_DIR / "sionna_spectral_ceiling.json"
V4_OOF = (
    PATHS.artifacts
    / "round2"
    / "v5_static_link_phase_calibration"
    / "v4_nested_oof_channel.npy"
)
CONFIG_V4 = "d16_base_v4lambda"
CONFIG_SCATTER = "d10_scatter015_v4lambda"
CONFIG_V05 = "d16_base_v05lambda"
TAG = "oof500"
TINY = 1e-30


@dataclass(frozen=True)
class Candidate:
    name: str
    pas_source: str | None = None
    pdp_source: str | None = None
    ad_source: str | None = None
    alpha_pas: float = 0.0
    alpha_pdp: float = 0.0
    alpha_ad: float = 0.0
    order: str = "pas_pdp"
    cycles: int = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--reuse", action="store_true")
    return parser.parse_args()


def compact_path(config: str, kind: str) -> str:
    suffix = {
        "pas": "grouped_pas_f16.npy",
        "pdp": "grouped_pdp_f16.npy",
        "ad": "joint_ad_f16.npy",
        "ids": "ids.npy",
    }[kind]
    return str(ARTIFACT_DIR / f"{config}_train_{TAG}_{suffix}")


def normalized_numpy(value: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    value = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    total = value.sum(axis=axes, keepdims=True, dtype=np.float64)
    return np.divide(
        value,
        np.maximum(total, TINY),
        out=np.zeros_like(value),
        where=total > 0.0,
    )


def shift_by_site(
    value: np.ndarray,
    site: np.ndarray,
    shifts: dict[int, int],
    axis: int,
) -> np.ndarray:
    output = np.empty_like(value)
    for site_value, shift in shifts.items():
        rows = site == site_value
        output[rows] = np.roll(value[rows], shift, axis=axis)
    return output


def load_sources(query_ids: np.ndarray, query_site: np.ndarray) -> dict[str, np.ndarray]:
    for config in (CONFIG_V4, CONFIG_SCATTER, CONFIG_V05):
        ids = np.load(compact_path(config, "ids"))
        if not np.array_equal(ids, query_ids):
            raise ValueError(f"Sionna ids do not match OOF query ids for {config}")

    pas_v4 = np.asarray(np.load(compact_path(CONFIG_V4, "pas")), dtype=np.float32)
    pas_v4 = pas_v4.reshape(len(query_ids), 32, 2, 16, 8)
    pas_v4 = shift_by_site(pas_v4, query_site, {0: 2, 1: 2}, axis=4)
    pas_v4 = pas_v4.reshape(len(query_ids), 32, 256)

    pas_scatter = np.asarray(
        np.load(compact_path(CONFIG_SCATTER, "pas")), dtype=np.float32
    ).reshape(len(query_ids), 32, 2, 16, 8)
    pas_scatter = shift_by_site(pas_scatter, query_site, {0: 2, 1: 2}, axis=4)
    pas_scatter = pas_scatter.reshape(len(query_ids), 32, 256)

    pas_v05 = np.asarray(np.load(compact_path(CONFIG_V05, "pas")), dtype=np.float32)
    pas_v05 = pas_v05.reshape(len(query_ids), 32, 2, 16, 8)
    pas_v05 = shift_by_site(pas_v05, query_site, {0: 2, 1: 1}, axis=4)
    pas_v05 = pas_v05.reshape(len(query_ids), 32, 256)

    pdp_v4 = np.asarray(np.load(compact_path(CONFIG_V4, "pdp")), dtype=np.float32)
    pdp_scatter = np.asarray(
        np.load(compact_path(CONFIG_SCATTER, "pdp")), dtype=np.float32
    )
    pdp_v05 = np.asarray(np.load(compact_path(CONFIG_V05, "pdp")), dtype=np.float32)

    ad_v4 = np.asarray(np.load(compact_path(CONFIG_V4, "ad")), dtype=np.float32)
    ad_v4 = shift_by_site(ad_v4, query_site, {0: 2, 1: 1}, axis=2)
    ad_scatter = np.asarray(
        np.load(compact_path(CONFIG_SCATTER, "ad")), dtype=np.float32
    )
    ad_scatter = shift_by_site(ad_scatter, query_site, {0: 2, 1: 1}, axis=2)

    pas_power = [normalized_numpy(value**2, (2,)) for value in (pas_v4, pas_scatter, pas_v05)]
    pdp_power = [normalized_numpy(value**2, (2,)) for value in (pdp_v4, pdp_scatter, pdp_v05)]
    ad_power = [normalized_numpy(value**2, (1, 2, 3)) for value in (ad_v4, ad_scatter)]

    return {
        "pas_v4": pas_power[0],
        "pas_scatter": pas_power[1],
        "pas_v05": pas_power[2],
        "pas_ensemble": normalized_numpy(sum(pas_power), (2,)),
        "pdp_v4": pdp_power[0],
        "pdp_scatter": pdp_power[1],
        "pdp_v05": pdp_power[2],
        "pdp_ensemble": normalized_numpy(sum(pdp_power), (2,)),
        "ad_v4": ad_power[0],
        "ad_scatter": ad_power[1],
        "ad_ensemble": normalized_numpy(sum(ad_power), (1, 2, 3)),
    }


def expand_pas(value: np.ndarray) -> np.ndarray:
    count = len(value)
    power = value.reshape(count, 4, 8, 2, 16, 8)
    power = normalized_numpy(power, (3, 4, 5))
    power = power.transpose(0, 3, 4, 5, 1, 2)
    return np.repeat(power, 24, axis=5).astype(np.float32)


def expand_pdp(value: np.ndarray) -> np.ndarray:
    count = len(value)
    power = normalized_numpy(value.reshape(count, 2, 4, 192), (3,))
    power = np.broadcast_to(power[:, :, None, None, :, :], (count, 2, 16, 8, 4, 192))
    return power.reshape(count, 256, 4, 192).astype(np.float32)


def build_candidates() -> list[Candidate]:
    candidates = [Candidate(name="identity")]
    single_alphas = (0.025, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00)
    for source in ("pas_v4", "pas_v05", "pas_ensemble"):
        candidates.extend(
            Candidate(name=f"{source}_a{alpha:.3f}", pas_source=source, alpha_pas=alpha)
            for alpha in single_alphas
        )
    for source in ("pdp_v4", "pdp_scatter", "pdp_ensemble"):
        candidates.extend(
            Candidate(name=f"{source}_a{alpha:.3f}", pdp_source=source, alpha_pdp=alpha)
            for alpha in single_alphas
        )
    for source in ("ad_v4", "ad_scatter", "ad_ensemble"):
        candidates.extend(
            Candidate(name=f"{source}_a{alpha:.3f}", ad_source=source, alpha_ad=alpha)
            for alpha in (0.025, 0.05, 0.10, 0.20, 0.35, 0.50)
        )

    for pas_source in ("pas_v4", "pas_ensemble"):
        for pdp_source in ("pdp_scatter", "pdp_ensemble"):
            for alpha_pas in (0.05, 0.10, 0.20):
                for alpha_pdp in (0.10, 0.20, 0.35, 0.50):
                    for order in ("pas_pdp", "pdp_pas"):
                        candidates.append(
                            Candidate(
                                name=(
                                    f"dual_{pas_source}_{pdp_source}_"
                                    f"ap{alpha_pas:.2f}_ad{alpha_pdp:.2f}_{order}"
                                ),
                                pas_source=pas_source,
                                pdp_source=pdp_source,
                                alpha_pas=alpha_pas,
                                alpha_pdp=alpha_pdp,
                                order=order,
                                cycles=4,
                            )
                        )
    for ad_source in ("ad_v4", "ad_ensemble"):
        for pas_source in ("pas_v4", "pas_ensemble"):
            for pdp_source in ("pdp_scatter", "pdp_ensemble"):
                for alpha_ad in (0.05, 0.10):
                    for alpha_pas in (0.10, 0.20):
                        for alpha_pdp in (0.20, 0.35):
                            for order in ("ad_pas_pdp", "ad_pdp_pas"):
                                candidates.append(
                                    Candidate(
                                        name=(
                                            f"triple_{ad_source}_{pas_source}_{pdp_source}_"
                                            f"aa{alpha_ad:.2f}_ap{alpha_pas:.2f}_"
                                            f"ad{alpha_pdp:.2f}_{order}"
                                        ),
                                        ad_source=ad_source,
                                        pas_source=pas_source,
                                        pdp_source=pdp_source,
                                        alpha_ad=alpha_ad,
                                        alpha_pas=alpha_pas,
                                        alpha_pdp=alpha_pdp,
                                        order=order,
                                        cycles=4,
                                    )
                                )
    return candidates


def apply_candidate(
    base: torch.Tensor,
    targets: dict[str, torch.Tensor],
    spec: Candidate,
) -> torch.Tensor:
    if spec.name == "identity":
        return base
    reference_power = base.abs().square().sum(dim=(1, 2, 3))
    candidate = base
    desired_pas = pas_distribution(base)
    desired_pdp = pdp_distribution(base)
    desired_ad = ad_distribution(base)
    if spec.pas_source is not None:
        desired_pas = fixed_blend(
            desired_pas, targets[spec.pas_source], spec.alpha_pas, (1, 2, 3)
        )
    if spec.pdp_source is not None:
        desired_pdp = fixed_blend(
            desired_pdp, targets[spec.pdp_source], spec.alpha_pdp, (3,)
        )
    if spec.ad_source is not None:
        desired_ad = fixed_blend(
            desired_ad, targets[spec.ad_source], spec.alpha_ad, (1, 2, 3)
        )

    domains = []
    if spec.ad_source is not None:
        domains.append("ad")
    if spec.pas_source is not None:
        domains.append("pas")
    if spec.pdp_source is not None:
        domains.append("pdp")
    if len(domains) > 1:
        domains = spec.order.split("_")
    for _ in range(spec.cycles):
        for domain in domains:
            if domain == "ad":
                candidate = exact_ad_projection(candidate, desired_ad)
            elif domain == "pas":
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
    pas_den = torch.where(pas_den == 0.0, torch.full_like(pas_den, 1e-12), pas_den)
    pas = (pas_num / pas_den).mean(dim=(1, 2), dtype=torch.float64)

    pdp_num = (prediction_pdp * truth_pdp).sum(dim=3, dtype=torch.float64)
    pdp_den = torch.sqrt(
        prediction_pdp.square().sum(dim=3, dtype=torch.float64)
        * truth_pdp.square().sum(dim=3, dtype=torch.float64)
    )
    pdp_den = torch.where(pdp_den == 0.0, torch.full_like(pdp_den, 1e-12), pdp_den)
    pdp = (pdp_num / pdp_den).mean(dim=(1, 2), dtype=torch.float64)

    error = (prediction - truth).abs().square().sum(dim=(1, 2, 3), dtype=torch.float64)
    energy = truth.abs().square().sum(dim=(1, 2, 3), dtype=torch.float64)
    return tuple(value.cpu().numpy() for value in (pas, pdp, error, energy))


def generate_metrics(device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
    metadata = np.load(OOF_META)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    _, _, _, train_site, _ = load_positions()
    site = train_site[query_ids].astype(np.int8)
    sources = load_sources(query_ids, site)
    expanded: dict[str, np.ndarray] = {}
    for name, value in sources.items():
        if name.startswith("pas_"):
            expanded[name] = expand_pas(value)
        elif name.startswith("pdp_"):
            expanded[name] = expand_pdp(value)
        else:
            expanded[name] = value.astype(np.float32)

    candidates = build_candidates()
    count = len(query_ids)
    output = {
        "pas": np.empty((len(candidates), count), dtype=np.float64),
        "pdp": np.empty((len(candidates), count), dtype=np.float64),
        "error": np.empty((len(candidates), count), dtype=np.float64),
        "energy": np.empty((len(candidates), count), dtype=np.float64),
    }
    base_all = np.load(V4_OOF, mmap_mode="r")
    truth_all = np.load(PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r")
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        base = torch.as_tensor(
            np.array(base_all[start:end], dtype=np.complex64, copy=True), device=device
        )
        truth = torch.as_tensor(
            np.array(truth_all[query_ids[start:end]], dtype=np.complex64, copy=True),
            device=device,
        )
        with torch.inference_mode():
            truth_pas, truth_pdp = truth_views(truth)
            targets = {
                name: torch.as_tensor(value[start:end], device=device)
                for name, value in expanded.items()
            }
            for index, spec in enumerate(candidates):
                prediction = apply_candidate(base, targets, spec)
                values = point_metrics(prediction, truth, truth_pas, truth_pdp)
                for key, value in zip(("pas", "pdp", "error", "energy"), values):
                    output[key][index, start:end] = value
                del prediction
        print(f"Sionna ceiling {end}/{count}", flush=True)
    output.update(
        {
            "candidate_names": np.asarray([candidate.name for candidate in candidates]),
            "query_ids": query_ids,
            "fold_id": fold_id,
            "site": site,
        }
    )
    return output


def candidate_result(metrics: dict[str, np.ndarray], index: int, mask=None) -> dict[str, float | int]:
    return statistics(
        metrics["pas"][index],
        metrics["pdp"][index],
        metrics["error"][index],
        metrics["energy"][index],
        mask,
    )


def selected_result(metrics: dict[str, np.ndarray], selected: np.ndarray) -> dict[str, float | int]:
    point = np.arange(len(selected))
    return statistics(
        metrics["pas"][selected, point],
        metrics["pdp"][selected, point],
        metrics["error"][selected, point],
        metrics["energy"][selected, point],
    )


def nested_policy(metrics: dict[str, np.ndarray]) -> tuple[np.ndarray, list[dict[str, object]]]:
    names = metrics["candidate_names"]
    fold_id = metrics["fold_id"]
    site = metrics["site"]
    selected = np.zeros(len(fold_id), dtype=np.int64)
    decisions: list[dict[str, object]] = []
    for held_fold in np.unique(fold_id):
        for site_value in np.unique(site):
            fit = (fold_id != held_fold) & (site == site_value)
            held = (fold_id == held_fold) & (site == site_value)
            rows = [candidate_result(metrics, index, fit) for index in range(len(names))]
            best = int(np.argmax([row["score"] for row in rows]))
            selected[held] = best
            decisions.append(
                {
                    "fold": int(held_fold),
                    "site": int(site_value),
                    "candidate": str(names[best]),
                    "fit": rows[best],
                    "held": candidate_result(metrics, best, held),
                }
            )
    return selected, decisions


def oracle_policy(metrics: dict[str, np.ndarray]) -> np.ndarray:
    energy = np.maximum(metrics["energy"], TINY)
    point_score = (
        0.4 * metrics["pas"]
        + 0.4 * metrics["pdp"]
        + 0.2 / (1.0 + metrics["error"] / energy)
    )
    return np.argmax(point_score, axis=0).astype(np.int64)


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    if args.reuse:
        metrics = {key: value for key, value in np.load(METRICS).items()}
    else:
        metrics = generate_metrics(torch.device(args.device), args.batch_size)
        np.savez_compressed(METRICS, **metrics)

    rows = []
    for index, name in enumerate(metrics["candidate_names"]):
        rows.append({"candidate": str(name), **candidate_result(metrics, index)})
    rows.sort(key=lambda row: float(row["score"]), reverse=True)
    identity_index = int(np.flatnonzero(metrics["candidate_names"] == "identity")[0])
    identity = candidate_result(metrics, identity_index)
    nested_selected, decisions = nested_policy(metrics)
    nested = selected_result(metrics, nested_selected)
    oracle_selected = oracle_policy(metrics)
    oracle = selected_result(metrics, oracle_selected)
    fixed_best = rows[0]
    report = {
        "experiment": VERSION,
        "status": "complete",
        "primary_hypothesis": (
            "Same-coordinate Sionna spectra contain complementary delay/angle support that "
            "can improve V4 after train-calibrated vertical shifts and bounded exact projection."
        ),
        "input_policy": {
            "base": PATHS.relative(V4_OOF),
            "sionna_configs": [CONFIG_V4, CONFIG_SCATTER, CONFIG_V05],
            "absolute_sionna_phase_used": False,
            "point_power_preserved": True,
            "vertical_shifts": {
                "pas_v4": {"site0": 2, "site1": 2},
                "pas_v05": {"site0": 2, "site1": 1},
                "ad": {"site0": 2, "site1": 1},
            },
        },
        "candidate_count": int(len(metrics["candidate_names"])),
        "identity": identity,
        "fixed_best": fixed_best,
        "fixed_best_delta": float(fixed_best["score"] - identity["score"]),
        "nested": nested,
        "nested_delta": float(nested["score"] - identity["score"]),
        "nested_decisions": decisions,
        "pointwise_oracle": oracle,
        "pointwise_oracle_delta": float(oracle["score"] - identity["score"]),
        "oracle_changed_fraction": float(np.mean(oracle_selected != identity_index)),
        "top_candidates": rows[:25],
        "metrics": PATHS.relative(METRICS),
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
