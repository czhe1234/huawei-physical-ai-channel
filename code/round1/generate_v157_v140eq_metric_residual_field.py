#!/usr/bin/env python3
"""Generate v157: source-matched official-slice residual field over v140.

v155 interpolated absolute train-truth PAS/PDP slices and failed online.
v156 built a v140-equivalent complex cross-fit base, but merely changing the
audit base reproduced the same v155 test output.  This version uses the new
base correctly: it learns local log-ratio residuals from v140-equivalent
current slices to train truth slices, then transfers only those residuals to
the exact v140 test output.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans

from generate_v155_metric_aligned_pas_pdp_projection import (
    DEVICE,
    Config,
    aggregate_score,
    build_neighbors,
    merge_terms,
    metric_terms,
    normalized_pas,
    normalized_pdp,
    output_comparison,
    project_metric_slices,
    summarize,
)
from generate_v87_easy_field_projection import (
    BS_POS,
    N_ANT,
    S_SC,
    ad_transform,
    beta_subcarriers,
    inverse_ad,
)
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION = "v157_v140eq_metric_residual_field"
OUT_DIR = PATHS.cache_dir(157)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(157)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v157_Round1_Test_Channel.npy"
DECISIONS = OUT_DIR / "v157_metric_residual_decisions.npz"
REPORT = REPORT_DIR / "v157_v140eq_metric_residual_field_generate.json"
ZIP_OUT = PATHS.round1_versions / "v157_v140eq_metric_residual_field_candidate_pending.zip"

TRAIN_BASE = PATHS.cache_dir(156) / "train_v140eq_crossfit_base_ad.npy"
TRAIN_RESIDUAL = PATHS.cache_dir(156) / "train_v140eq_crossfit_residual_ad.npy"
MATCHED_SETS = PATHS.analysis / "results" / "test_regime_analysis" / "test_matched_val_sets.npz"
V140_OUTPUT = PATHS.cache_dir(140) / "v140_Round1_Test_Channel.npy"
V155_OUTPUT = PATHS.cache_dir(155) / "v155_Round1_Test_Channel.npy"

AUDIT_BATCH = 10
GENERATE_BATCH = 8
RANDOM_STATE = 15701
LOG_CLIP = 2.5


@dataclass(frozen=True)
class ResidualConfig:
    name: str
    k: int
    strength: float
    pdp_eta: float


CONFIGS = tuple(
    ResidualConfig(
        f"k{k}_s{strength:.2f}_pdp{pdp_eta:.2f}",
        k,
        strength,
        pdp_eta,
    )
    for k in (4, 8)
    for strength in (0.10, 0.14, 0.20, 0.28)
    for pdp_eta in (0.0, 0.25, 0.50, 1.0)
)


def projector(config: ResidualConfig) -> Config:
    return Config(
        name="residual_target_projection",
        k=0,
        eta_pas=1.0,
        eta_pdp=config.pdp_eta,
        iterations=2,
        gain_low=0.62,
        gain_high=1.62,
    )


def file_digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def predict_log_residual(
    neighbor_indices: np.ndarray,
    neighbor_weights: np.ndarray,
    base_mm: np.ndarray,
    residual_mm: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = len(neighbor_indices)
    log_pas = torch.zeros((batch, 256, N_ANT, S_SC), dtype=torch.float32, device=DEVICE)
    log_pdp = torch.zeros_like(log_pas)
    for rank in range(neighbor_indices.shape[1]):
        ids = neighbor_indices[:, rank]
        current_np = np.asarray(base_mm[ids], dtype=np.complex64)
        truth_np = current_np + np.asarray(residual_mm[ids], dtype=np.complex64)
        current = torch.from_numpy(current_np).to(DEVICE)
        truth = torch.from_numpy(truth_np).to(DEVICE)
        weight = torch.from_numpy(neighbor_weights[:, rank].astype(np.float32)).to(DEVICE)
        residual_pas = (
            torch.log(normalized_pas(truth) + 1e-8)
            - torch.log(normalized_pas(current) + 1e-8)
        ).clamp(-LOG_CLIP, LOG_CLIP)
        residual_pdp = (
            torch.log(normalized_pdp(truth) + 1e-8)
            - torch.log(normalized_pdp(current) + 1e-8)
        ).clamp(-LOG_CLIP, LOG_CLIP)
        log_pas += weight[:, None, None, None] * residual_pas
        log_pdp += weight[:, None, None, None] * residual_pdp
        del current, truth, current_np, truth_np, residual_pas, residual_pdp
    return log_pas, log_pdp


def residual_targets(
    base: torch.Tensor,
    log_pas: torch.Tensor,
    log_pdp: torch.Tensor,
    strength: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_pas = normalized_pas(base) * torch.exp(float(strength) * log_pas)
    target_pas /= target_pas.sum(dim=1, keepdim=True).clamp_min(1e-30)
    target_pdp = normalized_pdp(base) * torch.exp(float(strength) * log_pdp)
    target_pdp /= target_pdp.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    return target_pas, target_pdp


def audit(
    positions: np.ndarray,
    base_mm: np.ndarray,
    residual_mm: np.ndarray,
) -> tuple[ResidualConfig, dict[str, object]]:
    matched = np.load(MATCHED_SETS)["spatial_matched"].astype(np.int64)[:300]
    max_k = max(config.k for config in CONFIGS)
    neighbors, distances, _ = build_neighbors(
        positions, positions[matched], max_k + 1
    )
    filtered_indices = np.empty((len(matched), max_k), dtype=np.int64)
    filtered_distances = np.empty((len(matched), max_k), dtype=np.float32)
    for row, query in enumerate(matched):
        keep = neighbors[row] != query
        filtered_indices[row] = neighbors[row][keep][:max_k]
        filtered_distances[row] = distances[row][keep][:max_k]
    inverse = 1.0 / np.maximum(filtered_distances, 0.25) ** 3.0
    filtered_weights = (inverse / inverse.sum(axis=1, keepdims=True)).astype(np.float32)
    diagnostic_fold = (
        KMeans(n_clusters=25, random_state=RANDOM_STATE, n_init=20)
        .fit_predict(positions[:, :2])[matched]
        % 5
    )

    accum: dict[str, list[dict[str, np.ndarray]]] = {"base": []}
    diagnostics = {
        config.name: {"gain_min": [], "gain_max": []} for config in CONFIGS
    }
    for config in CONFIGS:
        accum[config.name] = []

    for start in range(0, len(matched), AUDIT_BATCH):
        end = min(start + AUDIT_BATCH, len(matched))
        ids = matched[start:end]
        base_np = np.asarray(base_mm[ids], dtype=np.complex64)
        truth_np = base_np + np.asarray(residual_mm[ids], dtype=np.complex64)
        base = torch.from_numpy(base_np).to(DEVICE)
        truth = torch.from_numpy(truth_np).to(DEVICE)
        accum["base"].append(metric_terms(base, truth))
        residual_by_k: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for k in sorted({config.k for config in CONFIGS}):
            weight = filtered_weights[start:end, :k].copy()
            weight /= weight.sum(axis=1, keepdims=True)
            residual_by_k[k] = predict_log_residual(
                filtered_indices[start:end, :k], weight, base_mm, residual_mm
            )
        for config in CONFIGS:
            log_pas, log_pdp = residual_by_k[config.k]
            target_pas, target_pdp = residual_targets(
                base, log_pas, log_pdp, config.strength
            )
            prediction, diag = project_metric_slices(
                base, target_pas, target_pdp, projector(config)
            )
            accum[config.name].append(metric_terms(prediction, truth))
            diagnostics[config.name]["gain_min"].append(diag["gain_min"])
            diagnostics[config.name]["gain_max"].append(diag["gain_max"])
            del prediction, target_pas, target_pdp
        del base, truth, base_np, truth_np, residual_by_k
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"audit residual {end}/{len(matched)}", flush=True)

    merged = {name: merge_terms(chunks) for name, chunks in accum.items()}
    base_score = aggregate_score(merged["base"])
    rows = []
    base_spectral = 0.4 * merged["base"]["pas"] + 0.4 * merged["base"]["pdp"]
    for config in CONFIGS:
        terms = merged[config.name]
        score = aggregate_score(terms)
        spectral = 0.4 * terms["pas"] + 0.4 * terms["pdp"]
        gain = spectral - base_spectral
        folds = [float(gain[diagnostic_fold == fold].mean()) for fold in range(5)]
        rows.append(
            {
                "config": config.name,
                "params": config.__dict__,
                "score_proxy": score,
                "score_gain": float(score - base_score),
                "pas_gain": float(terms["pas"].mean() - merged["base"]["pas"].mean()),
                "pdp_gain": float(terms["pdp"].mean() - merged["base"]["pdp"].mean()),
                "spectral_gain": summarize(gain),
                "fold_means": folds,
                "all_folds_positive": bool(all(value > 0 for value in folds)),
                "gain_min": summarize(np.concatenate(diagnostics[config.name]["gain_min"])),
                "gain_max": summarize(np.concatenate(diagnostics[config.name]["gain_max"])),
            }
        )
    eligible = [
        row for row in rows
        if row["score_gain"] > 0.001 and row["all_folds_positive"]
    ]
    if not eligible:
        raise RuntimeError("No source-matched residual config passed v157 audit")
    selected_row = max(eligible, key=lambda row: row["score_gain"])
    selected = next(
        config for config in CONFIGS if config.name == selected_row["config"]
    )
    return selected, {
        "selection_set": "spatial_matched first 300 points",
        "base": {
            "score_proxy": base_score,
            "pas": summarize(merged["base"]["pas"]),
            "pdp": summarize(merged["base"]["pdp"]),
            "nmse_global": float(
                merged["base"]["error"].sum()
                / max(merged["base"]["truth_energy"].sum(), TINY)
            ),
        },
        "configs": rows,
        "selected": selected_row,
        "nearest_distance": summarize(filtered_distances[:, 0]),
        "k8_distance": summarize(filtered_distances[:, 7]),
    }


def generate(
    config: ResidualConfig,
    positions_train: np.ndarray,
    positions_test: np.ndarray,
    base_mm: np.ndarray,
    residual_mm: np.ndarray,
) -> dict[str, object]:
    indices, distances, weights = build_neighbors(
        positions_train, positions_test, config.k
    )
    v140 = np.load(V140_OUTPUT, mmap_mode="r")
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=v140.shape
    )
    beta_s = beta_subcarriers(140.3, 0.10)
    distance_to_bs = np.linalg.norm(positions_test - BS_POS[None, :], axis=1)
    gain_min = []
    gain_max = []
    log_pas_abs = []
    log_pdp_abs = []
    for start in range(0, len(positions_test), GENERATE_BATCH):
        end = min(start + GENERATE_BATCH, len(positions_test))
        base_h = np.asarray(v140[start:end], dtype=np.complex64)
        derotated = base_h * np.exp(
            1j * distance_to_bs[start:end, None, None, None] * beta_s
        ).astype(np.complex64)
        base = torch.from_numpy(ad_transform(derotated)).to(DEVICE)
        log_pas, log_pdp = predict_log_residual(
            indices[start:end], weights[start:end], base_mm, residual_mm
        )
        target_pas, target_pdp = residual_targets(
            base, log_pas, log_pdp, config.strength
        )
        final_ad, diag = project_metric_slices(
            base, target_pas, target_pdp, projector(config)
        )
        result = inverse_ad(final_ad.detach().cpu().numpy()).reshape(
            end - start, 256, N_ANT, S_SC
        )
        result *= np.exp(
            -1j * distance_to_bs[start:end, None, None, None] * beta_s
        ).astype(np.complex64)
        output[start:end] = result.astype(np.complex64)
        output.flush()
        gain_min.extend(diag["gain_min"].tolist())
        gain_max.extend(diag["gain_max"].tolist())
        log_pas_abs.extend(log_pas.abs().mean(dim=(1, 2, 3)).detach().cpu().tolist())
        log_pdp_abs.extend(log_pdp.abs().mean(dim=(1, 2, 3)).detach().cpu().tolist())
        del base, final_ad, target_pas, target_pdp, log_pas, log_pdp, result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"generate residual {end}/{len(positions_test)}", flush=True)
    np.savez_compressed(
        DECISIONS,
        neighbor_indices=indices.astype(np.int32),
        neighbor_distances=distances.astype(np.float32),
        neighbor_weights=weights.astype(np.float32),
    )
    return {
        "neighbor_distance": summarize(distances),
        "farthest_neighbor_distance": summarize(distances[:, -1]),
        "mean_abs_log_pas_residual": summarize(log_pas_abs),
        "mean_abs_log_pdp_residual": summarize(log_pdp_abs),
        "gain_min": summarize(gain_min),
        "gain_max": summarize(gain_max),
    }


def main() -> None:
    positions_train = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(
        np.float32
    )
    positions_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(
        np.float32
    )
    height = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    regimes_test = regime_of(height["test_blocked"].astype(np.float32))
    base_mm = np.load(TRAIN_BASE, mmap_mode="r")
    residual_mm = np.load(TRAIN_RESIDUAL, mmap_mode="r")
    selected, audit_report = audit(positions_train, base_mm, residual_mm)
    generation = generate(
        selected, positions_train, positions_test, base_mm, residual_mm
    )
    output = np.load(OUTPUT, mmap_mode="r")
    v140 = np.load(V140_OUTPUT, mmap_mode="r")
    v155 = np.load(V155_OUTPUT, mmap_mode="r")
    comparison = output_comparison(output, v140, regimes_test)
    comparison_v155 = output_comparison(output, v155, regimes_test)
    if not comparison["finite"]:
        raise ValueError("v157 output contains non-finite values")
    if comparison["nmse_to_v140"] > 0.010:
        raise RuntimeError(
            f"v157 output change is too large: {comparison['nmse_to_v140']:.6f}"
        )
    if comparison_v155["nmse_to_v140"] < 1e-6:
        raise RuntimeError("v157 output is not materially different from failed v155")

    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "base": "v140 online 0.501352",
        "primary_hypothesis": (
            "local official-slice residuals relative to a v140-equivalent complex current transfer to exact v140"
        ),
        "only_primary_change": (
            "interpolate truth/current PAS and PDP log-ratio residuals instead of absolute train-truth slices"
        ),
        "selected_config": selected.__dict__,
        "audit": audit_report,
        "test_generation": generation,
        "comparison_to_v140": comparison,
        "comparison_to_v155": comparison_v155,
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
            "decisions": PATHS.relative(DECISIONS),
            "report": PATHS.relative(REPORT),
            "zip": PATHS.relative(ZIP_OUT),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    sources = [
        Path(__file__),
        PATHS.code / "generate_v156_v140eq_metric_slice_projection.py",
        PATHS.code / "generate_v155_metric_aligned_pas_pdp_projection.py",
        PATHS.code / "generate_v140_balanced_hminus2_overdrive.py",
        PATHS.code / "generate_v87_easy_field_projection.py",
        PATHS.code / "powerfield_common.py",
        PATHS.code_root / "project_paths.py",
    ]
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        for source in sources:
            archive.write(source, source.name)
    with zipfile.ZipFile(ZIP_OUT) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP CRC failure: {bad}")
        members = archive.namelist()
    report["integrity"] = {
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": comparison["finite"],
        "output_bytes": int(OUTPUT.stat().st_size),
        "zip_crc": True,
        "zip_members": members,
        "sha256": file_digest(ZIP_OUT, "sha256"),
        "md5": file_digest(ZIP_OUT, "md5"),
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"zip": PATHS.relative(ZIP_OUT), "integrity": report["integrity"]}, indent=2))


if __name__ == "__main__":
    main()
