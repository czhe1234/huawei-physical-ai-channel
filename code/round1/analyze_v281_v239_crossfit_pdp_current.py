#!/usr/bin/env python3
"""Reconstruct a strict cross-fit v239-like PDP current on official train H."""

from __future__ import annotations

import json

import numpy as np
import torch

import analyze_v233_official_joint_ad_manifold as v233
import analyze_v236_official_path_pdp_decoder as v236
import analyze_v238_raw_path_conditional_angle as v238
import analyze_v239_nnls_path_energy_decoder as v239
import analyze_v254_route_topology_spectral_field as v254
import generate_v215_pointcloud_physics_neural_field as v215
import generate_v218_official_multispectrum_physics_ai as v218
from generate_v87_easy_field_projection import summarize
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 281
VERSION = "v281_v239_crossfit_pdp_current"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_OUT = OUT_DIR / "v281_v239_crossfit_pdp_current.npz"

OUTER_SEEDS = (281, 1281, 2281, 3281, 4281)
OUTER_COUNT = 320
FROZEN_EPOCHS = 60
FROZEN_ALPHA = {
    "clear": 0.10,
    "light": 0.30,
    "medium": 0.25,
    "heavy": 0.15,
}
MIN_OFFICIAL_DELTA = 5e-4
SUPPORT_BATCH = 16


def weighted_support(
    bank: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    output = np.empty(
        (len(indices),) + tuple(bank.shape[1:]), dtype=np.float32
    )
    for start in range(0, len(indices), SUPPORT_BATCH):
        end = min(start + SUPPORT_BATCH, len(indices))
        block = np.zeros(
            (end - start,) + tuple(bank.shape[1:]), dtype=np.float32
        )
        for rank in range(indices.shape[1]):
            block += weights[start:end, rank, None, None, None] * np.asarray(
                bank[indices[start:end, rank]], dtype=np.float32
            )
        output[start:end] = v238.normalize_support(block)
    return output


def make_anchor(
    query_ids: np.ndarray,
    bank_ids: np.ndarray,
    support: np.ndarray,
    log_power: np.ndarray,
    train_pos: np.ndarray,
    train_blocked: np.ndarray,
    exclude_self: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kwargs = {}
    if exclude_self:
        kwargs = {"query_ids": query_ids, "bank_ids": bank_ids}
    indices, weights = v218.neighbor_index_weights(
        train_pos[query_ids],
        train_blocked[query_ids],
        train_pos[bank_ids],
        train_blocked[bank_ids],
        k=v238.ANCHOR_K,
        **kwargs,
    )
    anchor = weighted_support(support[bank_ids], indices, weights)
    anchor_power = np.sum(
        log_power[bank_ids][indices] * weights, axis=1
    ).astype(np.float32)
    return anchor, anchor_power, weights


def features_for(
    ids: np.ndarray,
    anchor: np.ndarray,
    anchor_power: np.ndarray,
    weights: np.ndarray,
    train_pos: np.ndarray,
    train_blocked: np.ndarray,
    anchor_d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = v233.raw_features(
        "train",
        ids,
        train_pos[ids],
        train_blocked[ids],
        anchor,
        anchor_power,
        weights,
    )
    return v236.augment_raw_features(raw, anchor_d)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean_targets, _raw, _power, valid_delay = v215.load_targets()
    target_d = v239.normalize_d(clean_targets["d"])
    support, valid_support, log_power = v238.build_raw_support()
    valid_ids = np.flatnonzero(valid_delay & valid_support).astype(np.int64)
    nnls_weights, _reconstruction, nnls_report = v239.build_nnls_labels(
        target_d, valid_ids
    )
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    geom = np.load(v238.GEOM_CACHE)
    train_blocked = geom["train_blocked"].astype(np.float32)

    rows: list[dict[str, object]] = []
    ids_parts: list[np.ndarray] = []
    anchor_parts: list[np.ndarray] = []
    model_parts: list[np.ndarray] = []
    current_parts: list[np.ndarray] = []
    truth_parts: list[np.ndarray] = []
    for fold, seed in enumerate(OUTER_SEEDS):
        val_ids = np.intersect1d(
            v254.choose_seeded_matched_indices(
                train_pos[:, :2], test_pos[:, :2], OUTER_COUNT, seed
            ),
            valid_ids,
        )
        fit_ids = np.setdiff1d(valid_ids, val_ids)
        fit_anchor, fit_power, fit_weights = make_anchor(
            fit_ids,
            fit_ids,
            support,
            log_power,
            train_pos,
            train_blocked,
            True,
        )
        val_anchor, val_power, val_weights = make_anchor(
            val_ids,
            fit_ids,
            support,
            log_power,
            train_pos,
            train_blocked,
            False,
        )
        fit_anchor_d = v239.normalize_d(fit_anchor.sum(axis=(1, 2)))
        val_anchor_d = v239.normalize_d(val_anchor.sum(axis=(1, 2)))
        fit_raw = features_for(
            fit_ids,
            fit_anchor,
            fit_power,
            fit_weights,
            train_pos,
            train_blocked,
            fit_anchor_d,
        )
        val_raw = features_for(
            val_ids,
            val_anchor,
            val_power,
            val_weights,
            train_pos,
            train_blocked,
            val_anchor_d,
        )
        fit_features, [val_features], _stats = v233.standardize_feature_sets(
            fit_raw, [val_raw]
        )
        fit_delay = v236.raw_path_delay("train", fit_ids)
        val_delay = v236.raw_path_delay("train", val_ids)
        model, training = v239.train_model(
            fit_features,
            fit_delay,
            target_d[fit_ids],
            nnls_weights[fit_ids],
            seed + 28100,
            FROZEN_EPOCHS,
            device,
            None,
        )
        model_output = v239.predict_model(
            model, val_features, val_delay, device
        ).d
        regimes = regime_of(train_blocked[val_ids]).astype(str)
        alpha = np.asarray(
            [FROZEN_ALPHA[str(value)] for value in regimes], dtype=np.float32
        )
        current = v239.normalize_d(
            (1.0 - alpha[:, None]) * val_anchor_d
            + alpha[:, None] * model_output
        )
        anchor_cos = v239.row_cos(val_anchor_d, target_d[val_ids])
        model_cos = v239.row_cos(model_output, target_d[val_ids])
        current_cos = v239.row_cos(current, target_d[val_ids])
        delta = current_cos - anchor_cos
        rows.append(
            {
                "fold": fold,
                "seed": seed,
                "fit_count": int(len(fit_ids)),
                "validation_count": int(len(val_ids)),
                "training": training,
                "anchor_cos": summarize(anchor_cos),
                "model_cos": summarize(model_cos),
                "current_cos": summarize(current_cos),
                "delta": summarize(delta),
                "positive_fraction": float(np.mean(delta > 0.0)),
                "alpha": summarize(alpha),
            }
        )
        ids_parts.append(val_ids)
        anchor_parts.append(val_anchor_d)
        model_parts.append(model_output)
        current_parts.append(current)
        truth_parts.append(target_d[val_ids])
        print(
            f"v281 outer {fold + 1}/{len(OUTER_SEEDS)} "
            f"anchor={anchor_cos.mean():.6f} current={current_cos.mean():.6f} "
            f"delta={delta.mean():+.6f}",
            flush=True,
        )
        del model, fit_anchor, val_anchor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fold_delta = np.asarray(
        [row["delta"]["mean"] for row in rows], dtype=np.float64
    )
    selected = bool(
        np.mean(fold_delta) >= MIN_OFFICIAL_DELTA
        and np.sum(fold_delta > 0.0) >= 3
    )
    np.savez_compressed(
        OOF_OUT,
        ids=np.concatenate(ids_parts),
        anchor_d=np.concatenate(anchor_parts),
        model_d=np.concatenate(model_parts),
        current_d=np.concatenate(current_parts),
        truth_d=np.concatenate(truth_parts),
        fold=np.concatenate(
            [np.full(len(ids), index, dtype=np.int16) for index, ids in enumerate(ids_parts)]
        ),
    )
    report = {
        "method": VERSION,
        "status": (
            "crossfit_current_validated_next_stage"
            if selected
            else "analysis_only_no_candidate"
        ),
        "primary_hypothesis": (
            "The frozen v239 path-energy decoder and regime blend can reconstruct a "
            "genuinely v239-like PDP current under independent spatial cross-fit, providing "
            "an isomorphic error source for a second-stage residual model."
        ),
        "compliance_boundary": {
            "uses": [
                "official fit H PDP/support labels",
                "official train coordinates and point-cloud path features",
                "v239 epoch and regime alpha frozen before these folds",
            ],
            "does_not_use": [
                "validation H in anchors or model training",
                "synthetic/Sionna labels",
                "v212/v217 teacher",
                "local replay truth",
                "hidden labels or RT H/path coefficients",
            ],
            "deployment": "analysis-only cross-fit current reconstruction",
        },
        "parameters": {
            "outer_seeds": list(OUTER_SEEDS),
            "outer_count": OUTER_COUNT,
            "frozen_epochs": FROZEN_EPOCHS,
            "frozen_alpha": FROZEN_ALPHA,
            "minimum_official_delta": MIN_OFFICIAL_DELTA,
        },
        "nnls": nnls_report,
        "outer_folds": rows,
        "selection": {
            "mean_delta": float(np.mean(fold_delta)),
            "fold_delta": fold_delta.tolist(),
            "wins": int(np.sum(fold_delta > 0.0)),
            "selected": selected,
        },
        "local_gate": None,
        "outputs": {
            "oof": PATHS.relative(OOF_OUT),
            "report": PATHS.relative(REPORT),
            "prediction": None,
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
