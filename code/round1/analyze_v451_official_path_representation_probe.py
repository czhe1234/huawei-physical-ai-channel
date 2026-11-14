#!/usr/bin/env python3
"""Probe whether v445 path representations transfer after output remapping."""

from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

import analyze_v401_dualdomain_pas_preserving_confocal as v401
import analyze_v450_official_geometry_token_transfer as v450
import train_v439_component_route_long_model as v439
import train_v445_candidate_path_token_model as v445
from project_paths import PATHS


VERSION_NUMBER = 451
VERSION = "v451_official_path_representation_probe"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_CACHE = OUT_DIR / "v451_official_path_representation_oof.npz"

V443_CHECKPOINT = (
    PATHS.checkpoints / "v443_dense_component_route_model" / "component_route_best.pt"
)
V445_CHECKPOINT = (
    PATHS.checkpoints / "v445_candidate_path_token_model" / "component_route_best.pt"
)
TARGET_SOURCE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"

RIDGE_GRID = (1.0, 10.0, 100.0, 1000.0)
STRENGTH_GRID = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50)
LATENT_DIM = 64
FLOOR = 1e-7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=45_101)
    return parser.parse_args()


def raw_geometry_features(inputs: dict[str, np.ndarray]) -> np.ndarray:
    token = np.asarray(inputs["path_candidate_feature"], dtype=np.float32)
    valid = np.asarray(inputs["path_candidate_valid"], dtype=bool)
    count = np.sum(valid, axis=1)
    weight = valid[..., None].astype(np.float32)
    denominator = np.maximum(count[:, None], 1)
    mean = np.sum(token * weight, axis=1) / denominator
    variance = np.sum((token - mean[:, None]) ** 2 * weight, axis=1) / denominator
    std = np.sqrt(np.maximum(variance, 0.0))
    minimum = np.zeros((len(token), token.shape[2]), dtype=np.float32)
    maximum = np.zeros_like(minimum)
    for row in range(len(token)):
        if count[row]:
            current = token[row, valid[row]]
            minimum[row] = np.min(current, axis=0)
            maximum[row] = np.max(current, axis=0)
    position = v450.position_features(np.asarray(inputs["positions"]))
    return np.column_stack(
        [
            position,
            count.astype(np.float32) / v450.MAX_PATHS,
            (count == 0).astype(np.float32),
            mean,
            std,
            minimum,
            maximum,
        ]
    ).astype(np.float32)


@torch.no_grad()
def path_embedding(
    model: v445.CandidatePathTokenModel,
    inputs: dict[str, np.ndarray],
    ids: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    parts: list[np.ndarray] = []
    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start : start + batch_size]
        position = torch.from_numpy(
            v450.position_features(np.asarray(inputs["positions"][batch_ids]))
        ).to(device)
        feature = torch.from_numpy(
            np.asarray(inputs["path_candidate_feature"][batch_ids], dtype=np.float32)
        ).to(device)
        valid = torch.from_numpy(
            np.asarray(inputs["path_candidate_valid"][batch_ids], dtype=bool)
        ).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            position_token = model.candidate_position(position)[:, None, :]
            path = model.candidate_encoder(feature) + position_token
            special = (
                model.candidate_special.expand(len(batch_ids), -1, -1)
                + position_token
            )
            memory_valid = torch.cat(
                [
                    torch.ones(len(batch_ids), 1, dtype=torch.bool, device=device),
                    valid,
                ],
                dim=1,
            )
            memory = model.candidate_transformer(
                torch.cat([special, path], dim=1),
                src_key_padding_mask=~memory_valid,
            )
            encoded = memory[:, 1:]
            mass_logits = model.candidate_mass_head(encoded).squeeze(-1)
            mass_logits = mass_logits.masked_fill(~valid, -1e4)
            mass = F.softmax(mass_logits, dim=1) * valid.to(mass_logits.dtype)
            mass = mass / mass.sum(dim=1, keepdim=True).clamp_min(1e-8)
            weighted = torch.sum(encoded * mass[:, :, None], dim=1)
            unweighted = torch.sum(
                encoded * valid[:, :, None].to(encoded.dtype), dim=1
            ) / valid.sum(dim=1, keepdim=True).clamp_min(1).to(encoded.dtype)
            embedding = torch.cat([memory[:, 0], weighted, unweighted], dim=1)
        parts.append(embedding.float().cpu().numpy())
    return np.concatenate(parts).astype(np.float32)


@torch.no_grad()
def base_prediction(
    inputs: dict[str, np.ndarray],
    ids: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    checkpoint = torch.load(V443_CHECKPOINT, map_location="cpu", weights_only=True)
    model = v439.ComponentRouteTransformer(coordinate_dropout=0.0)
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device).eval()
    parts: list[np.ndarray] = []
    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start : start + batch_size]
        route = torch.from_numpy(
            np.asarray(inputs["route_raster"][batch_ids], dtype=np.float32)
        ).to(device)
        position = torch.from_numpy(
            v450.position_features(np.asarray(inputs["positions"][batch_ids]))
        ).to(device)
        component = torch.from_numpy(
            np.asarray(inputs["component_features"][batch_ids], dtype=np.float32)
        ).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(route, position, component)
        parts.append(
            F.softmax(output["joint_logits"].flatten(1), dim=1)
            .reshape(-1, 192, 16)
            .float()
            .cpu()
            .numpy()
        )
    del model
    torch.cuda.empty_cache()
    return np.concatenate(parts).astype(np.float32)


def load_embeddings(
    inputs: dict[str, np.ndarray], args: argparse.Namespace, device: torch.device
) -> dict[str, np.ndarray]:
    ids = np.arange(len(inputs["positions"]), dtype=np.int64)
    raw_path = OUT_DIR / "raw_geometry_features.npy"
    random_path = OUT_DIR / "random_path_embedding.npy"
    pretrained_path = OUT_DIR / "v445_path_embedding.npy"
    if raw_path.is_file():
        raw = np.load(raw_path)
    else:
        raw = raw_geometry_features(inputs)
        np.save(raw_path, raw)
    if random_path.is_file():
        random_embedding = np.load(random_path)
    else:
        torch.manual_seed(args.seed)
        random_model = v445.CandidatePathTokenModel(coordinate_dropout=0.0).to(device)
        random_embedding = path_embedding(
            random_model, inputs, ids, device, args.batch_size
        )
        np.save(random_path, random_embedding)
        del random_model
        torch.cuda.empty_cache()
    if pretrained_path.is_file():
        pretrained = np.load(pretrained_path)
    else:
        checkpoint = torch.load(V445_CHECKPOINT, map_location="cpu", weights_only=True)
        pretrained_model = v445.CandidatePathTokenModel(coordinate_dropout=0.0)
        pretrained_model.load_state_dict(checkpoint["model"], strict=True)
        pretrained_model = pretrained_model.to(device)
        pretrained = path_embedding(
            pretrained_model, inputs, ids, device, args.batch_size
        )
        np.save(pretrained_path, pretrained)
        del pretrained_model
        torch.cuda.empty_cache()
    return {
        "raw_geometry": raw.astype(np.float32),
        "random_path_encoder": random_embedding.astype(np.float32),
        "v445_path_encoder": pretrained.astype(np.float32),
        "raw_plus_v445": np.concatenate([raw, pretrained], axis=1).astype(np.float32),
    }


def standardize_fit(
    feature: np.ndarray, fit_ids: np.ndarray, query_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(feature[fit_ids], axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(feature[fit_ids], axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-4)
    return (
        ((feature[fit_ids] - mean) / std).astype(np.float32),
        ((feature[query_ids] - mean) / std).astype(np.float32),
    )


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    x_mean = np.mean(x, axis=0, dtype=np.float64).astype(np.float32)
    y_mean = np.mean(y, axis=0, dtype=np.float64).astype(np.float32)
    xc = x - x_mean
    yc = y - y_mean
    gram = xc.T @ xc
    gram.flat[:: len(gram) + 1] += alpha
    coefficient = np.linalg.solve(gram, xc.T @ yc).astype(np.float32)
    intercept = (y_mean - x_mean @ coefficient).astype(np.float32)
    return coefficient, intercept


def apply_delta(
    base: np.ndarray, delta: np.ndarray, strength: float
) -> np.ndarray:
    logits = np.log(np.maximum(base.reshape(len(base), -1), FLOOR))
    logits += strength * np.clip(delta, -6.0, 6.0)
    logits -= np.max(logits, axis=1, keepdims=True)
    probability = np.exp(logits)
    probability /= np.maximum(np.sum(probability, axis=1, keepdims=True), 1e-30)
    return probability.reshape(-1, 192, 16).astype(np.float32)


def score_joint(joint: np.ndarray, truth_sh: np.ndarray, truth_d: np.ndarray) -> dict[str, float]:
    reverse = np.concatenate([[0], np.arange(191, 0, -1)])
    pdp = np.sum(joint[:, reverse], axis=2)
    sh = float(np.mean(v450.row_cosine(joint, truth_sh)))
    d = float(np.mean(v450.row_cosine(pdp, truth_d)))
    return {
        "score": 0.60 * sh + 0.40 * d,
        "joint_sh_cosine": sh,
        "pdp_cosine": d,
    }


def delta_metric(model: dict[str, float], base: dict[str, float]) -> dict[str, float]:
    return {key: float(model[key] - base[key]) for key in model}


def prepare_latent_target(
    residual: np.ndarray,
    train_ids: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, PCA]:
    latent_dim = min(LATENT_DIM, len(train_ids) - 1, residual.shape[1])
    pca = PCA(n_components=latent_dim, svd_solver="randomized", random_state=seed)
    target_latent = pca.fit_transform(residual[train_ids]).astype(np.float32)
    return target_latent, pca


def fit_predict_low_rank(
    feature: np.ndarray,
    train_ids: np.ndarray,
    query_ids: np.ndarray,
    target_latent: np.ndarray,
    pca: PCA,
    ridge: float,
) -> np.ndarray:
    fit_x, query_x = standardize_fit(feature, train_ids, query_ids)
    coefficient, intercept = ridge_fit(fit_x, target_latent, ridge)
    latent = query_x @ coefficient + intercept
    return pca.inverse_transform(latent).astype(np.float32)


def build_fold_targets(
    folds: list[np.ndarray],
    valid_ids: np.ndarray,
    residual: np.ndarray,
    seed: int,
) -> list[dict[str, object]]:
    output = []
    for outer, query_ids in enumerate(folds):
        fit_ids = np.setdiff1d(valid_ids, query_ids)
        validation_ids = np.intersect1d(fit_ids, folds[(outer + 1) % len(folds)])
        inner_train_ids = np.setdiff1d(fit_ids, validation_ids)
        inner_latent, inner_pca = prepare_latent_target(
            residual, inner_train_ids, seed + 1000 * outer
        )
        full_latent, full_pca = prepare_latent_target(
            residual, fit_ids, seed + 20_000 + 1000 * outer
        )
        output.append(
            {
                "query_ids": query_ids,
                "fit_ids": fit_ids,
                "validation_ids": validation_ids,
                "inner_train_ids": inner_train_ids,
                "inner_latent": inner_latent,
                "inner_pca": inner_pca,
                "full_latent": full_latent,
                "full_pca": full_pca,
            }
        )
    return output


def probe_method(
    name: str,
    feature: np.ndarray,
    fold_targets: list[dict[str, object]],
    valid_ids: np.ndarray,
    base: np.ndarray,
    truth_sh: np.ndarray,
    truth_d: np.ndarray,
) -> tuple[dict[str, object], np.ndarray]:
    oof = base.copy()
    fold_rows = []
    for outer, fold_target in enumerate(fold_targets):
        query_ids = fold_target["query_ids"]
        fit_ids = fold_target["fit_ids"]
        validation_ids = fold_target["validation_ids"]
        inner_train_ids = fold_target["inner_train_ids"]
        best = None
        for ridge in RIDGE_GRID:
            validation_delta = fit_predict_low_rank(
                feature,
                inner_train_ids,
                validation_ids,
                fold_target["inner_latent"],
                fold_target["inner_pca"],
                ridge,
            )
            for strength in STRENGTH_GRID:
                prediction = apply_delta(
                    base[validation_ids], validation_delta, strength
                )
                metric = score_joint(
                    prediction, truth_sh[validation_ids], truth_d[validation_ids]
                )
                candidate = (metric["score"], -ridge, -strength, ridge, strength)
                if best is None or candidate > best[0]:
                    best = (candidate, metric)
        assert best is not None
        ridge = float(best[0][3])
        strength = float(best[0][4])
        outer_delta = fit_predict_low_rank(
            feature,
            fit_ids,
            query_ids,
            fold_target["full_latent"],
            fold_target["full_pca"],
            ridge,
        )
        oof[query_ids] = apply_delta(base[query_ids], outer_delta, strength)
        base_metric = score_joint(
            base[query_ids], truth_sh[query_ids], truth_d[query_ids]
        )
        model_metric = score_joint(
            oof[query_ids], truth_sh[query_ids], truth_d[query_ids]
        )
        fold_rows.append(
            {
                "fold": outer,
                "fit_count": int(len(fit_ids)),
                "query_count": int(len(query_ids)),
                "selected_ridge": ridge,
                "selected_strength": strength,
                "inner_validation": best[1],
                "base": base_metric,
                "model": model_metric,
                "delta": delta_metric(model_metric, base_metric),
            }
        )
        print(
            f"v451 {name} fold={outer} ridge={ridge:.0f} strength={strength:.2f} "
            f"delta={fold_rows[-1]['delta']['score']:+.6f}",
            flush=True,
        )
    base_metric = score_joint(base[valid_ids], truth_sh[valid_ids], truth_d[valid_ids])
    model_metric = score_joint(oof[valid_ids], truth_sh[valid_ids], truth_d[valid_ids])
    return {
        "feature_dim": int(feature.shape[1]),
        "base": base_metric,
        "model": model_metric,
        "delta": delta_metric(model_metric, base_metric),
        "positive_folds": int(
            sum(row["delta"]["score"] > 0.0 for row in fold_rows)
        ),
        "mean_selected_strength": float(
            np.mean([row["selected_strength"] for row in fold_rows])
        ),
        "folds": fold_rows,
    }, oof


def main() -> None:
    args = parse_args()
    for path in (V443_CHECKPOINT, V445_CHECKPOINT, TARGET_SOURCE):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"v451 requires exactly one visible CUDA GPU, got {torch.cuda.device_count()}"
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    device = torch.device("cuda")
    inputs = v450.load_or_build_inputs()
    embeddings = load_embeddings(inputs, args, device)
    target = np.load(TARGET_SOURCE)
    valid_ids = np.flatnonzero(target["valid"].astype(bool)).astype(np.int64)
    folds = v401.confirmatory_folds(
        valid_ids,
        np.asarray(inputs["positions"][:2000, :2], dtype=np.float64),
    )
    merged = np.concatenate(folds)
    if not np.array_equal(np.sort(merged), np.sort(valid_ids)):
        raise ValueError("v451 folds are not exhaustive and disjoint")
    base_cache = OUT_DIR / "v443_official_joint.npy"
    if base_cache.is_file():
        base = np.load(base_cache)
    else:
        base = base_prediction(
            inputs,
            np.arange(2000, dtype=np.int64),
            device,
            args.batch_size,
        )
        np.save(base_cache, base)
    truth_sh = target["sh"].astype(np.float32)
    truth_d = target["d"].astype(np.float32)
    residual = np.log(np.maximum(truth_sh.reshape(2000, -1), FLOOR)) - np.log(
        np.maximum(base.reshape(2000, -1), FLOOR)
    )
    residual = np.clip(residual, -6.0, 6.0).astype(np.float32)
    fold_targets = build_fold_targets(folds, valid_ids, residual, args.seed)
    rng = np.random.default_rng(args.seed)
    methods = {}
    predictions = {}
    for name, feature in embeddings.items():
        result, prediction = probe_method(
            name,
            feature[:2000],
            fold_targets,
            valid_ids,
            base,
            truth_sh,
            truth_d,
        )
        methods[name] = result
        predictions[name] = prediction
    shuffled = embeddings["v445_path_encoder"][:2000][rng.permutation(2000)]
    shuffled_result, shuffled_prediction = probe_method(
        "shuffled_v445_path_encoder",
        shuffled,
        fold_targets,
        valid_ids,
        base,
        truth_sh,
        truth_d,
    )
    methods["shuffled_v445_path_encoder"] = shuffled_result
    predictions["shuffled_v445_path_encoder"] = shuffled_prediction
    winner = max(methods, key=lambda name: methods[name]["delta"]["score"])
    pretrained_gain = float(methods["v445_path_encoder"]["delta"]["score"])
    random_gain = float(methods["random_path_encoder"]["delta"]["score"])
    raw_gain = float(methods["raw_geometry"]["delta"]["score"])
    passed = bool(
        pretrained_gain > max(random_gain, raw_gain, 0.0) + 5e-4
        and methods["v445_path_encoder"]["positive_folds"] >= 4
    )
    np.savez_compressed(
        OOF_CACHE,
        valid_ids=valid_ids,
        base=base[valid_ids].astype(np.float16),
        **{
            name: prediction[valid_ids].astype(np.float16)
            for name, prediction in predictions.items()
        },
    )
    report = {
        "method": VERSION,
        "status": "strict_official_oof_representation_probe",
        "primary_hypothesis": (
            "Although the v445 synthetic output mapping fails on official geometry-only "
            "candidate sets, its frozen path representation may still encode transferable "
            "path existence, delay, angle, and interaction structure."
        ),
        "known_v445_bias": (
            "Synthetic v445 set membership was selected by path power before power fields "
            "and rank were removed; v291 official candidates use geometry-only selection."
        ),
        "compliance_boundary": {
            "uses": [
                "official point-cloud geometry inputs",
                "official train H-derived joint S-H/PDP targets",
                "strict five-fold official-train evaluation",
                "frozen synthetic weights for representation research",
            ],
            "does_not_use": [
                "official test H",
                "test-coordinate Sionna CFR or path coefficients",
                "v436/v449 teacher predictions",
                "online feedback for fold or hyperparameter selection",
            ],
        },
        "parameters": {
            "ridge_grid": list(RIDGE_GRID),
            "strength_grid": list(STRENGTH_GRID),
            "target_latent_dim": LATENT_DIM,
            "floor": FLOOR,
            "seed": args.seed,
            "single_gpu": torch.cuda.get_device_name(0),
        },
        "methods": methods,
        "winner": winner,
        "decision": {
            "criterion": (
                "v445 frozen representation must beat raw geometry, random encoder, "
                "and zero by >0.0005 with at least 4/5 positive folds"
            ),
            "passed": passed,
            "next_step": (
                "train a low-capacity official adapter on the frozen representation"
                if passed
                else "regenerate synthetic candidates with geometry-only set membership"
            ),
        },
        "seconds": float(time.time() - started),
        "outputs": {
            "oof": PATHS.relative(OOF_CACHE),
            "report": PATHS.relative(REPORT),
            "prediction": None,
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
