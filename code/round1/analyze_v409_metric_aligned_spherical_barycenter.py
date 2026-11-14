#!/usr/bin/env python3
"""Estimate the PAS/PDP Bayes action under a local empirical spectrum law."""

from __future__ import annotations

import json

import numpy as np
import torch

import analyze_v297_continuous_delay_component_transport as v297
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v402_visible_path_residual_factorization as v402
import analyze_v406_neural_propagation_mode_router as v406
from generate_v87_easy_field_projection import summarize
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 409
VERSION = "v409_metric_aligned_spherical_barycenter"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_CACHE = OUT_DIR / "v409_metric_aligned_spherical_barycenter_oof.npz"

TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
GEOMETRY_CACHE = PATHS.shared_features / "height_geometry_cache.npz"

NEIGHBORS = 16
OPTIMIZATION_STEPS = 80
LEARNING_RATE = 0.12
MIN_WEIGHTED_DELTA = 0.01
MIN_FOLD_WINS = 4


def l2_normalize(x: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(x, dtype=np.float64)
    norm = np.sqrt(np.sum(value * value, axis=axes, keepdims=True))
    return (value / np.maximum(norm, 1e-30)).astype(np.float32)


def empirical_moments(
    donor: np.ndarray, weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    donor_sh_unit = l2_normalize(donor, (2, 3))
    donor_pdp = donor.sum(axis=3)
    donor_pdp_unit = l2_normalize(donor_pdp, (2,))
    mu_sh = np.einsum("qk,qksh->qsh", weight, donor_sh_unit, optimize=True)
    mu_pdp = np.einsum("qk,qks->qs", weight, donor_pdp_unit, optimize=True)
    return mu_sh.astype(np.float32), mu_pdp.astype(np.float32)


def sh_spherical_mean(mu_sh: np.ndarray) -> np.ndarray:
    output = np.maximum(mu_sh, 0.0)
    return v406.normalize(output)


def sqrt_barycenter(donor: np.ndarray, weight: np.ndarray) -> np.ndarray:
    root = np.einsum(
        "qk,qksh->qsh", weight, np.sqrt(np.maximum(donor, 0.0)), optimize=True
    )
    return v406.normalize(root * root)


def joint_spherical_barycenter(
    anchor: np.ndarray,
    mu_sh: np.ndarray,
    mu_pdp: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, list[float]]:
    logits = torch.nn.Parameter(
        torch.log(torch.from_numpy(np.maximum(anchor, 1e-10))).to(device)
    )
    sh_moment = torch.from_numpy(mu_sh).to(device)
    pdp_moment = torch.from_numpy(mu_pdp).to(device)
    optimizer = torch.optim.Adam([logits], lr=LEARNING_RATE)
    history: list[float] = []
    for step in range(1, OPTIMIZATION_STEPS + 1):
        spectrum = torch.softmax(logits.flatten(1), dim=1).reshape_as(logits)
        sh_unit = spectrum / torch.clamp(
            torch.linalg.vector_norm(spectrum.flatten(1), dim=1)[:, None, None],
            min=1e-20,
        )
        pdp = spectrum.sum(dim=2)
        pdp_unit = pdp / torch.clamp(
            torch.linalg.vector_norm(pdp, dim=1)[:, None], min=1e-20
        )
        expected_score = 0.5 * torch.sum(
            sh_unit * sh_moment, dim=(1, 2)
        ) + 0.5 * torch.sum(pdp_unit * pdp_moment, dim=1)
        loss = -expected_score.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 10 == 0 or step == OPTIMIZATION_STEPS:
            history.append(float(expected_score.mean().detach().cpu()))
    with torch.no_grad():
        output = torch.softmax(logits.flatten(1), dim=1).reshape_as(logits)
    return output.cpu().numpy().astype(np.float32), history


def evaluate(
    prediction: np.ndarray, anchor: np.ndarray, truth: np.ndarray
) -> tuple[dict[str, object], np.ndarray]:
    base = v402.point_metrics(anchor, truth)
    values = v402.point_metrics(prediction, truth)
    delta = values["weighted"] - base["weighted"]
    return (
        {
            "metrics": v402.metric_summary(prediction, truth),
            "delta": {
                key: float(np.mean(values[key] - base[key])) for key in values
            },
            "point_gain": summarize(delta),
            "point_win_fraction": float(np.mean(delta > 0.0)),
        },
        delta,
    )


def fold_experiment(
    fold: int,
    query_ids: np.ndarray,
    valid_ids: np.ndarray,
    xy: np.ndarray,
    target: np.ndarray,
    blocked: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    fit_ids = np.setdiff1d(valid_ids, query_ids)
    donor_ids, distance = v406.find_neighbors(
        query_ids, fit_ids, xy, NEIGHBORS
    )
    anchor, weight = v406.weighted_anchor(target, donor_ids, distance)
    donor = target[donor_ids]
    mu_sh, mu_pdp = empirical_moments(donor, weight)
    predictions = {
        "sh_spherical": sh_spherical_mean(mu_sh),
        "sqrt_barycenter": sqrt_barycenter(donor, weight),
    }
    predictions["joint_spherical"], optimization_history = (
        joint_spherical_barycenter(anchor, mu_sh, mu_pdp, device)
    )
    truth = target[query_ids]
    methods: dict[str, object] = {}
    deltas: dict[str, np.ndarray] = {}
    regimes = regime_of(blocked[query_ids]).astype(str)
    for name, prediction in predictions.items():
        result, delta = evaluate(prediction, anchor, truth)
        result["by_regime"] = {
            regime: float(np.mean(delta[regimes == regime]))
            for regime in ("clear", "light", "medium", "heavy")
            if np.any(regimes == regime)
        }
        methods[name] = result
        deltas[name] = delta
    row = {
        "fold": fold,
        "fit_count": int(len(fit_ids)),
        "query_count": int(len(query_ids)),
        "baseline": v402.metric_summary(anchor, truth),
        "methods": methods,
        "optimization_expected_score": optimization_history,
        "leakage_assertions": {
            "query_in_fit": int(np.intersect1d(query_ids, fit_ids).size),
            "outer_truth_in_estimator": False,
            "outer_truth_used_for_scoring_only": True,
        },
    }
    arrays = {
        "ids": query_ids,
        "baseline": anchor,
        "truth": truth,
        **predictions,
        **{f"delta_{name}": value for name, value in deltas.items()},
    }
    return row, arrays


def main() -> None:
    for path in (TARGET_CACHE, GEOMETRY_CACHE, v297.COMPONENT_CACHE):
        if not path.is_file():
            raise FileNotFoundError(path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    targets = np.load(TARGET_CACHE)
    target = v406.normalize(targets["sh"].astype(np.float32))
    positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    xy = positions[:, :2]
    blocked = np.load(GEOMETRY_CACHE)["train_blocked"].astype(np.float32)
    component = np.load(v297.COMPONENT_CACHE)
    valid_ids = np.flatnonzero(
        targets["valid"].astype(bool) & component["valid"].any(axis=1)
    ).astype(np.int64)
    folds = v315.disjoint_folds(valid_ids, xy)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, object]] = []
    parts: list[dict[str, np.ndarray]] = []
    for fold, query_ids in enumerate(folds):
        row, arrays = fold_experiment(
            fold, query_ids, valid_ids, xy, target, blocked, device
        )
        rows.append(row)
        parts.append(arrays)
        print(
            f"v409 fold={fold} sh={row['methods']['sh_spherical']['delta']['weighted']:+.5f} "
            f"sqrt={row['methods']['sqrt_barycenter']['delta']['weighted']:+.5f} "
            f"joint={row['methods']['joint_spherical']['delta']['weighted']:+.5f}",
            flush=True,
        )
    aggregate: dict[str, object] = {}
    for name in ("sh_spherical", "sqrt_barycenter", "joint_spherical"):
        delta = np.asarray(
            [row["methods"][name]["delta"]["weighted"] for row in rows]
        )
        aggregate[name] = {
            "weighted_delta": summarize(delta),
            "positive_folds": int(np.sum(delta > 0.0)),
        }
    best_name = max(
        aggregate,
        key=lambda name: aggregate[name]["weighted_delta"]["mean"],
    )
    passed = bool(
        aggregate[best_name]["weighted_delta"]["mean"] >= MIN_WEIGHTED_DELTA
        and aggregate[best_name]["positive_folds"] >= MIN_FOLD_WINS
    )
    ids = np.concatenate([part["ids"] for part in parts])
    if len(ids) != len(np.unique(ids)) or len(ids) != len(valid_ids):
        raise AssertionError("OOF IDs are not a unique cover")
    names = (
        "baseline",
        "truth",
        "sh_spherical",
        "sqrt_barycenter",
        "joint_spherical",
        "delta_sh_spherical",
        "delta_sqrt_barycenter",
        "delta_joint_spherical",
    )
    np.savez_compressed(
        OOF_CACHE,
        ids=ids,
        fold=np.concatenate(
            [
                np.full(len(part["ids"]), fold, dtype=np.int16)
                for fold, part in enumerate(parts)
            ]
        ),
        **{
            name: np.concatenate([part[name] for part in parts])
            for name in names
        },
    )
    report = {
        "method": VERSION,
        "status": (
            "official_oof_passed_ready_for_ai_integration"
            if passed
            else "analysis_only_no_candidate"
        ),
        "primary_hypothesis": (
            "The score-optimal action for a multimodal local spectrum law is a PAS/PDP "
            "spherical barycenter, not an L1 arithmetic average or a guessed donor."
        ),
        "compliance": {
            "uses": ["official train H spectra", "official train coordinates"],
            "does_not_use": [
                "test H or hidden truth",
                "v212/Sionna/RT/synthetic targets or online labels",
            ],
        },
        "parameters": {
            "neighbors": NEIGHBORS,
            "optimization_steps": OPTIMIZATION_STEPS,
            "learning_rate": LEARNING_RATE,
            "objective": "equal-weight expected S-H cosine and PDP cosine",
        },
        "fold_rows": rows,
        "aggregate": aggregate,
        "selection": {
            "best_method": best_name,
            "minimum_weighted_delta": MIN_WEIGHTED_DELTA,
            "minimum_positive_folds": MIN_FOLD_WINS,
            "passed": passed,
            "next_action": (
                "use the metric-aligned barycenter as the output layer of the strict AI carrier"
                if passed
                else "metric mismatch does not explain the remaining unseen-path gap"
            ),
        },
        "outputs": {
            "oof": PATHS.relative(OOF_CACHE),
            "report": PATHS.relative(REPORT),
            "prediction": None,
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["selection"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
