#!/usr/bin/env python3
"""Generate v155 with metric-aligned PAS/PDP slice projection over v140.

The official score gives 0.4 weight to PAS and 0.4 weight to PDP.  Earlier
support models predict a collapsed H/V/delay distribution, while the evaluator
uses much finer slices: one 256-bin BS PAS for every UE/subcarrier pair and one
192-bin PDP for every BS/UE antenna pair.  This experiment predicts those
exact normalized slices from nearby training truth and alternates bounded
positive-amplitude projections in the two unitary domains.

The v140 phase backbone and per-point total power are preserved.  The script
first selects one configuration on a fixed spatially matched train holdout.
It creates a submission only when every diagnostic partition has positive
gain over the v125-equivalent cross-fit base.
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
from sklearn.neighbors import NearestNeighbors

from generate_v87_easy_field_projection import (
    BS_POS,
    MH,
    MP,
    MV,
    N_ANT,
    S_SC,
    ad_transform,
    beta_subcarriers,
    inverse_ad,
)
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION = "v155_metric_aligned_pas_pdp_projection"
OUT_DIR = PATHS.cache_dir(155)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(155)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v155_Round1_Test_Channel.npy"
DECISIONS = OUT_DIR / "v155_metric_aligned_pas_pdp_decisions.npz"
REPORT = REPORT_DIR / "v155_metric_aligned_pas_pdp_projection_generate.json"
ZIP_OUT = PATHS.round1_versions / "v155_metric_aligned_pas_pdp_projection_candidate_pending.zip"

TRAIN_BASE_AD = PATHS.shared_features / "train_v125eq_crossfit_base_ad.npy"
TRAIN_RESIDUAL_AD = PATHS.shared_features / "train_v125eq_crossfit_residual_ad.npy"
MATCHED_SETS = PATHS.analysis / "results" / "test_regime_analysis" / "test_matched_val_sets.npz"
V140_OUTPUT = PATHS.cache_dir(140) / "v140_Round1_Test_Channel.npy"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AUDIT_BATCH = 12
GENERATE_BATCH = 10
DISTANCE_FLOOR = 0.25
NEIGHBOR_POWER = 3.0
RANDOM_STATE = 15501


@dataclass(frozen=True)
class Config:
    name: str
    k: int
    eta_pas: float
    eta_pdp: float
    iterations: int
    gain_low: float
    gain_high: float


CONFIGS = (
    Config("k1_e040_030", 1, 0.040, 0.030, 1, 0.72, 1.38),
    Config("k1_e070_050", 1, 0.070, 0.050, 1, 0.66, 1.52),
    Config("k1_e100_070", 1, 0.100, 0.070, 1, 0.60, 1.67),
    Config("k1_e140_100", 1, 0.140, 0.100, 2, 0.54, 1.85),
    Config("k4_e040_030", 4, 0.040, 0.030, 1, 0.72, 1.38),
    Config("k4_e070_050", 4, 0.070, 0.050, 1, 0.66, 1.52),
    Config("k4_e100_070", 4, 0.100, 0.070, 1, 0.60, 1.67),
    Config("k4_e140_100", 4, 0.140, 0.100, 2, 0.54, 1.85),
)


def summarize(values: np.ndarray | list[float]) -> dict[str, float | int]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"count": 0}
    return {
        "count": int(len(x)),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "p10": float(np.quantile(x, 0.10)),
        "median": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def file_digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def row_cos_torch(a: torch.Tensor, b: torch.Tensor, dim: int) -> torch.Tensor:
    numerator = torch.sum(a * b, dim=dim)
    denominator = torch.linalg.vector_norm(a, dim=dim) * torch.linalg.vector_norm(b, dim=dim)
    return numerator / denominator.clamp_min(1e-30)


def metric_terms(pred: torch.Tensor, truth: torch.Tensor) -> dict[str, np.ndarray]:
    """Official-like metrics for derotated angle-delay tensors."""
    pred_frequency_angle = torch.fft.fft(pred, dim=-1)
    truth_frequency_angle = torch.fft.fft(truth, dim=-1)
    pred_pas = pred_frequency_angle.abs().square().reshape(len(pred), 256, N_ANT, S_SC)
    truth_pas = truth_frequency_angle.abs().square().reshape(len(truth), 256, N_ANT, S_SC)
    pas = row_cos_torch(pred_pas, truth_pas, 1).mean(dim=(1, 2))

    pred_antenna_delay = torch.fft.ifft2(pred, dim=(2, 3))
    truth_antenna_delay = torch.fft.ifft2(truth, dim=(2, 3))
    pred_pdp = pred_antenna_delay.abs().square().reshape(len(pred), 256, N_ANT, S_SC)
    truth_pdp = truth_antenna_delay.abs().square().reshape(len(truth), 256, N_ANT, S_SC)
    pdp = row_cos_torch(pred_pdp, truth_pdp, -1).mean(dim=(1, 2))

    error = (pred - truth).abs().square().sum(dim=(1, 2, 3, 4, 5))
    truth_energy = truth.abs().square().sum(dim=(1, 2, 3, 4, 5))
    return {
        "pas": pas.detach().cpu().numpy().astype(np.float32),
        "pdp": pdp.detach().cpu().numpy().astype(np.float32),
        "error": error.detach().cpu().numpy().astype(np.float64),
        "truth_energy": truth_energy.detach().cpu().numpy().astype(np.float64),
    }


def normalized_pas(ad: torch.Tensor) -> torch.Tensor:
    power = torch.fft.fft(ad, dim=-1).abs().square().reshape(len(ad), 256, N_ANT, S_SC)
    return power / power.sum(dim=1, keepdim=True).clamp_min(1e-30)


def normalized_pdp(ad: torch.Tensor) -> torch.Tensor:
    power = torch.fft.ifft2(ad, dim=(2, 3)).abs().square().reshape(len(ad), 256, N_ANT, S_SC)
    return power / power.sum(dim=-1, keepdim=True).clamp_min(1e-30)


def predict_targets(
    neighbor_indices: np.ndarray,
    neighbor_weights: np.ndarray,
    base_mm: np.ndarray,
    residual_mm: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = len(neighbor_indices)
    pas_target = torch.zeros((batch, 256, N_ANT, S_SC), dtype=torch.float32, device=DEVICE)
    pdp_target = torch.zeros_like(pas_target)
    for rank in range(neighbor_indices.shape[1]):
        ids = neighbor_indices[:, rank]
        truth_np = (
            np.asarray(base_mm[ids], dtype=np.complex64)
            + np.asarray(residual_mm[ids], dtype=np.complex64)
        )
        truth = torch.from_numpy(truth_np).to(DEVICE)
        weight = torch.from_numpy(neighbor_weights[:, rank].astype(np.float32)).to(DEVICE)
        pas_target += weight[:, None, None, None] * normalized_pas(truth)
        pdp_target += weight[:, None, None, None] * normalized_pdp(truth)
        del truth, truth_np
    pas_target /= pas_target.sum(dim=1, keepdim=True).clamp_min(1e-30)
    pdp_target /= pdp_target.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    return pas_target, pdp_target


def project_metric_slices(
    base: torch.Tensor,
    target_pas: torch.Tensor,
    target_pdp: torch.Tensor,
    config: Config,
) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
    original_energy = base.abs().square().sum(dim=(1, 2, 3, 4, 5)).clamp_min(1e-30)
    value = base.clone()
    gain_min = torch.full((len(base),), 1.0, dtype=torch.float32, device=DEVICE)
    gain_max = torch.full((len(base),), 1.0, dtype=torch.float32, device=DEVICE)

    for _ in range(config.iterations):
        frequency_angle = torch.fft.fft(value, dim=-1)
        current_pas = frequency_angle.abs().square().reshape(len(value), 256, N_ANT, S_SC)
        current_pas /= current_pas.sum(dim=1, keepdim=True).clamp_min(1e-30)
        pas_gain = ((target_pas + 1e-10) / (current_pas + 1e-10)).pow(
            config.eta_pas / config.iterations / 2.0
        )
        pas_gain = pas_gain.clamp(config.gain_low, config.gain_high)
        frequency_angle *= pas_gain.reshape(len(value), MP, MH, MV, N_ANT, S_SC)
        value = torch.fft.ifft(frequency_angle, dim=-1)
        gain_min = torch.minimum(gain_min, pas_gain.reshape(len(value), -1).amin(dim=1))
        gain_max = torch.maximum(gain_max, pas_gain.reshape(len(value), -1).amax(dim=1))

        antenna_delay = torch.fft.ifft2(value, dim=(2, 3))
        current_pdp = antenna_delay.abs().square().reshape(len(value), 256, N_ANT, S_SC)
        current_pdp /= current_pdp.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        pdp_gain = ((target_pdp + 1e-10) / (current_pdp + 1e-10)).pow(
            config.eta_pdp / config.iterations / 2.0
        )
        pdp_gain = pdp_gain.clamp(config.gain_low, config.gain_high)
        antenna_delay *= pdp_gain.reshape(len(value), MP, MH, MV, N_ANT, S_SC)
        value = torch.fft.fft2(antenna_delay, dim=(2, 3))
        gain_min = torch.minimum(gain_min, pdp_gain.reshape(len(value), -1).amin(dim=1))
        gain_max = torch.maximum(gain_max, pdp_gain.reshape(len(value), -1).amax(dim=1))

    final_energy = value.abs().square().sum(dim=(1, 2, 3, 4, 5)).clamp_min(1e-30)
    value *= torch.sqrt(original_energy / final_energy)[:, None, None, None, None, None]
    return value.to(torch.complex64), {
        "gain_min": gain_min.detach().cpu().numpy(),
        "gain_max": gain_max.detach().cpu().numpy(),
    }


def build_neighbors(
    reference_positions: np.ndarray,
    query_positions: np.ndarray,
    k: int,
    allowed_reference: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    allowed = (
        np.arange(len(reference_positions), dtype=np.int64)
        if allowed_reference is None
        else np.asarray(allowed_reference, dtype=np.int64)
    )
    model = NearestNeighbors(n_neighbors=k, n_jobs=-1)
    model.fit(reference_positions[allowed, :2])
    distance, local = model.kneighbors(query_positions[:, :2])
    indices = allowed[local]
    inverse = 1.0 / np.maximum(distance, DISTANCE_FLOOR) ** NEIGHBOR_POWER
    weight = inverse / inverse.sum(axis=1, keepdims=True)
    return indices.astype(np.int64), distance.astype(np.float32), weight.astype(np.float32)


def aggregate_score(terms: dict[str, np.ndarray]) -> float:
    pas = float(np.mean(terms["pas"]))
    pdp = float(np.mean(terms["pdp"]))
    nmse = float(np.sum(terms["error"]) / max(np.sum(terms["truth_energy"]), TINY))
    return float(0.4 * pas + 0.4 * pdp + 0.2 / (1.0 + nmse))


def merge_terms(chunks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}


def audit_configs(
    positions: np.ndarray,
    base_mm: np.ndarray,
    residual_mm: np.ndarray,
) -> tuple[Config, dict[str, object], np.ndarray]:
    matched = np.load(MATCHED_SETS)["spatial_matched"].astype(np.int64)[:300]
    groups_all = KMeans(n_clusters=25, random_state=RANDOM_STATE, n_init=20).fit_predict(positions[:, :2])
    diagnostic_fold = groups_all[matched] % 5
    max_k = max(config.k for config in CONFIGS)
    neighbors, distances, _ = build_neighbors(positions, positions[matched], max_k + 1)
    # Remove self and renormalize.  This matches the random official split more
    # closely than excluding a whole spatial district, while fold summaries
    # still expose geographic instability.
    filtered_indices = np.empty((len(matched), max_k), dtype=np.int64)
    filtered_distances = np.empty((len(matched), max_k), dtype=np.float32)
    for row, query in enumerate(matched):
        keep = neighbors[row] != query
        kept_indices = neighbors[row][keep][:max_k]
        kept_distances = distances[row][keep][:max_k]
        filtered_indices[row] = kept_indices
        filtered_distances[row] = kept_distances
    inverse = 1.0 / np.maximum(filtered_distances, DISTANCE_FLOOR) ** NEIGHBOR_POWER
    filtered_weights = (inverse / inverse.sum(axis=1, keepdims=True)).astype(np.float32)

    accum: dict[str, list[dict[str, np.ndarray]]] = {"base": []}
    diagnostics: dict[str, dict[str, list[np.ndarray]]] = {
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

        target_by_k: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for k in sorted({config.k for config in CONFIGS}):
            w = filtered_weights[start:end, :k]
            w = w / w.sum(axis=1, keepdims=True)
            target_by_k[k] = predict_targets(
                filtered_indices[start:end, :k], w, base_mm, residual_mm
            )
        for config in CONFIGS:
            target_pas, target_pdp = target_by_k[config.k]
            prediction, diag = project_metric_slices(base, target_pas, target_pdp, config)
            accum[config.name].append(metric_terms(prediction, truth))
            diagnostics[config.name]["gain_min"].append(diag["gain_min"])
            diagnostics[config.name]["gain_max"].append(diag["gain_max"])
            del prediction
        del base, truth, base_np, truth_np, target_by_k
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"audit {end}/{len(matched)}", flush=True)

    merged = {name: merge_terms(chunks) for name, chunks in accum.items()}
    base_score = aggregate_score(merged["base"])
    rows = []
    for config in CONFIGS:
        terms = merged[config.name]
        score = aggregate_score(terms)
        per_point_base = 0.4 * merged["base"]["pas"] + 0.4 * merged["base"]["pdp"]
        per_point_current = 0.4 * terms["pas"] + 0.4 * terms["pdp"]
        spectral_gain = per_point_current - per_point_base
        fold_means = [float(spectral_gain[diagnostic_fold == fold].mean()) for fold in range(5)]
        rows.append(
            {
                "config": config.name,
                "params": config.__dict__,
                "score_proxy": score,
                "score_gain": float(score - base_score),
                "pas_gain": float(terms["pas"].mean() - merged["base"]["pas"].mean()),
                "pdp_gain": float(terms["pdp"].mean() - merged["base"]["pdp"].mean()),
                "spectral_gain": summarize(spectral_gain),
                "spectral_fold_means": fold_means,
                "all_spectral_folds_positive": bool(all(value > 0 for value in fold_means)),
                "gain_min": summarize(np.concatenate(diagnostics[config.name]["gain_min"])),
                "gain_max": summarize(np.concatenate(diagnostics[config.name]["gain_max"])),
            }
        )
    eligible = [row for row in rows if row["score_gain"] > 0 and row["all_spectral_folds_positive"]]
    if not eligible:
        raise RuntimeError("No v155 configuration has positive score and all-fold spectral gain")
    best_row = max(eligible, key=lambda row: row["score_gain"])
    selected = next(config for config in CONFIGS if config.name == best_row["config"])
    report = {
        "selection_set": "spatial_matched first 300 points",
        "selection_indices": matched.tolist(),
        "diagnostic_fold": diagnostic_fold.tolist(),
        "base": {
            "score_proxy": base_score,
            "pas": summarize(merged["base"]["pas"]),
            "pdp": summarize(merged["base"]["pdp"]),
            "nmse_global": float(
                merged["base"]["error"].sum() / max(merged["base"]["truth_energy"].sum(), TINY)
            ),
        },
        "configs": rows,
        "selected": best_row,
        "nearest_distance": summarize(filtered_distances[:, 0]),
        "k4_distance": summarize(filtered_distances[:, 3]),
    }
    return selected, report, matched


def output_comparison(output: np.ndarray, base: np.ndarray, regimes: np.ndarray) -> dict[str, object]:
    total_num = 0.0
    total_den = 0.0
    by_regime = {}
    power_ratio = []
    for regime in ("clear", "light", "medium", "heavy"):
        ids = np.flatnonzero(regimes == regime)
        num = 0.0
        den = 0.0
        for start in range(0, len(ids), 20):
            rows = ids[start : start + 20]
            current = np.asarray(output[rows], dtype=np.complex64)
            reference = np.asarray(base[rows], dtype=np.complex64)
            num += float(np.sum(np.abs(current - reference) ** 2, dtype=np.float64))
            den += float(np.sum(np.abs(reference) ** 2, dtype=np.float64))
            current_power = np.sum(np.abs(current) ** 2, axis=(1, 2, 3), dtype=np.float64)
            reference_power = np.sum(np.abs(reference) ** 2, axis=(1, 2, 3), dtype=np.float64)
            power_ratio.extend((current_power / np.maximum(reference_power, TINY)).tolist())
        by_regime[regime] = float(num / max(den, TINY))
        total_num += num
        total_den += den
    return {
        "finite": bool(np.isfinite(output.real).all() and np.isfinite(output.imag).all()),
        "nmse_to_v140": float(total_num / max(total_den, TINY)),
        "nmse_to_v140_by_regime": by_regime,
        "power_ratio": summarize(power_ratio),
    }


def generate_test(
    config: Config,
    positions_train: np.ndarray,
    positions_test: np.ndarray,
    base_mm: np.ndarray,
    residual_mm: np.ndarray,
) -> dict[str, object]:
    indices, distances, weights = build_neighbors(positions_train, positions_test, config.k)
    v140 = np.load(V140_OUTPUT, mmap_mode="r")
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=v140.shape
    )
    beta_s = beta_subcarriers(140.3, 0.10)
    distance_to_bs = np.linalg.norm(positions_test - BS_POS[None, :], axis=1)
    gain_min = []
    gain_max = []
    target_pas_cos = []
    target_pdp_cos = []

    for start in range(0, len(positions_test), GENERATE_BATCH):
        end = min(start + GENERATE_BATCH, len(positions_test))
        h0 = np.asarray(v140[start:end], dtype=np.complex64)
        derotated = h0 * np.exp(
            1j * distance_to_bs[start:end, None, None, None] * beta_s
        ).astype(np.complex64)
        ad0_np = ad_transform(derotated)
        ad0 = torch.from_numpy(ad0_np).to(DEVICE)
        target_pas, target_pdp = predict_targets(
            indices[start:end], weights[start:end], base_mm, residual_mm
        )
        target_pas_cos.extend(
            row_cos_torch(normalized_pas(ad0), target_pas, 1).mean(dim=(1, 2)).detach().cpu().tolist()
        )
        target_pdp_cos.extend(
            row_cos_torch(normalized_pdp(ad0), target_pdp, -1).mean(dim=(1, 2)).detach().cpu().tolist()
        )
        ad1, diag = project_metric_slices(ad0, target_pas, target_pdp, config)
        h1 = inverse_ad(ad1.detach().cpu().numpy()).reshape(end - start, 256, N_ANT, S_SC)
        h1 *= np.exp(
            -1j * distance_to_bs[start:end, None, None, None] * beta_s
        ).astype(np.complex64)
        output[start:end] = h1.astype(np.complex64)
        output.flush()
        gain_min.extend(diag["gain_min"].tolist())
        gain_max.extend(diag["gain_max"].tolist())
        del ad0, ad1, target_pas, target_pdp, h0, h1, ad0_np, derotated
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"generate {end}/{len(positions_test)}", flush=True)

    np.savez_compressed(
        DECISIONS,
        neighbor_indices=indices.astype(np.int32),
        neighbor_distances=distances.astype(np.float32),
        neighbor_weights=weights.astype(np.float32),
    )
    return {
        "neighbor_distance": summarize(distances),
        "farthest_neighbor_distance": summarize(distances[:, -1]),
        "target_pas_cos_before": summarize(target_pas_cos),
        "target_pdp_cos_before": summarize(target_pdp_cos),
        "gain_min": summarize(gain_min),
        "gain_max": summarize(gain_max),
    }


def main() -> None:
    positions_train = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    positions_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    base_mm = np.load(TRAIN_BASE_AD, mmap_mode="r")
    residual_mm = np.load(TRAIN_RESIDUAL_AD, mmap_mode="r")

    selected, audit, _ = audit_configs(positions_train, base_mm, residual_mm)
    print("selected", json.dumps(selected.__dict__, ensure_ascii=False), flush=True)
    generation = generate_test(selected, positions_train, positions_test, base_mm, residual_mm)

    output = np.load(OUTPUT, mmap_mode="r")
    v140 = np.load(V140_OUTPUT, mmap_mode="r")
    blocked = np.load(PATHS.shared_features / "height_geometry_cache.npz")["test_blocked"]
    regimes = regime_of(blocked)
    comparison = output_comparison(output, v140, regimes)
    if output.shape != (500, 256, N_ANT, S_SC) or output.dtype != np.complex64:
        raise ValueError(f"invalid output shape/dtype: {output.shape} {output.dtype}")
    if not comparison["finite"]:
        raise ValueError("v155 output contains non-finite values")
    if comparison["nmse_to_v140"] > 0.012:
        raise RuntimeError(
            f"v155 output change is too large after v154 failure: {comparison['nmse_to_v140']:.6f}"
        )

    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "base": "v140 online 0.501352",
        "primary_hypothesis": (
            "predicting the exact evaluator PAS and PDP slices contains metric-aligned information "
            "that is lost by collapsed H/V/delay support targets"
        ),
        "only_primary_change": (
            "bounded alternating positive-amplitude projection toward local train-truth PAS and PDP slices"
        ),
        "preserved": {
            "v140_phase_backbone": True,
            "per_point_total_power": True,
            "v140_support_initialization": True,
        },
        "device": str(DEVICE),
        "selected_config": selected.__dict__,
        "audit": audit,
        "test_generation": generation,
        "comparison_to_v140": comparison,
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
