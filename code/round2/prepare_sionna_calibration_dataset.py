#!/usr/bin/env python3
"""Prepare scorer-aligned official labels for Round 2 Sionna calibration."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from project_paths import PATHS
from round2.data import load_positions


VERSION = "Round2-E40-Sionna-calibration-dataset"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "sionna_official_calibration"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_sionna_official_calibration"
REPORT = ANALYSIS_DIR / "calibration_dataset.json"
TRUTH_PAS = ARTIFACT_DIR / "official_train_pas_f16.npy"
TRUTH_PDP = ARTIFACT_DIR / "official_train_pdp_f16.npy"
TRUTH_AD = ARTIFACT_DIR / "official_train_ad_f16.npy"
TRUTH_POWER = ARTIFACT_DIR / "official_train_point_power_f64.npy"

SIONNA_ROOT = PATHS.artifacts / "round2" / "sionna_multiseed"
BASE_CONFIG = "primary_d16_base_v4lambda_1m"
SCATTER_CONFIG = "primary_d10_scatter015_v4lambda_1m"
SEEDS = 24
TINY = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def normalized(value: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    value = value.clamp_min(0.0)
    total = value.sum(dim=dims, keepdim=True)
    return torch.where(total > TINY, value / total.clamp_min(TINY), 0.0)


def open_output(path: Path, shape: tuple[int, ...], force: bool) -> np.memmap:
    if path.is_file() and not force:
        value = np.load(path, mmap_mode="r+")
        if value.shape != shape:
            raise ValueError(f"shape mismatch for {path}: {value.shape} != {shape}")
        return value
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)


def prepare_truth(device: torch.device, batch_size: int, force: bool) -> dict[str, object]:
    channel = np.load(PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r")
    count = len(channel)
    expected = {
        "pas": (count, 2, 16, 8, 4, 192),
        "pdp": (count, 256, 4, 192),
        "ad": (count, 16, 8, 192),
    }
    if (
        not force
        and TRUTH_PAS.is_file()
        and TRUTH_PDP.is_file()
        and TRUTH_AD.is_file()
        and TRUTH_POWER.is_file()
    ):
        cached = {
            "pas": np.load(TRUTH_PAS, mmap_mode="r"),
            "pdp": np.load(TRUTH_PDP, mmap_mode="r"),
            "ad": np.load(TRUTH_AD, mmap_mode="r"),
            "power": np.load(TRUTH_POWER, mmap_mode="r"),
        }
        if all(cached[name].shape == expected[name] for name in ("pas", "pdp", "ad")):
            return {
                "reused": True,
                "count": count,
                "nonzero": int(np.count_nonzero(cached["power"] > 0.0)),
            }

    pas_out = open_output(TRUTH_PAS, expected["pas"], True)
    pdp_out = open_output(TRUTH_PDP, expected["pdp"], True)
    ad_out = open_output(TRUTH_AD, expected["ad"], True)
    power_out = np.lib.format.open_memmap(
        TRUTH_POWER, mode="w+", dtype=np.float64, shape=(count,)
    )
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        value = torch.as_tensor(
            np.array(channel[start:end], dtype=np.complex64, copy=True), device=device
        )
        with torch.inference_mode():
            shaped = value.reshape(end - start, 2, 16, 8, 4, 192)
            spatial = torch.fft.fft2(shaped, dim=(2, 3))
            pas = normalized(spatial.abs().square().float(), (1, 2, 3))
            cir = torch.fft.ifft(value, dim=-1)
            pdp = normalized(cir.abs().square().float(), (3,))
            joint = torch.fft.ifft(spatial, dim=-1)
            ad = normalized(
                joint.abs().square().sum(dim=(1, 4), dtype=torch.float32),
                (1, 2, 3),
            )
            power = value.abs().square().sum(dim=(1, 2, 3), dtype=torch.float64)
        pas_out[start:end] = pas.cpu().numpy().astype(np.float16)
        pdp_out[start:end] = pdp.cpu().numpy().astype(np.float16)
        ad_out[start:end] = ad.cpu().numpy().astype(np.float16)
        power_out[start:end] = power.cpu().numpy()
        print(f"official spectra {end}/{count}", flush=True)
    for output in (pas_out, pdp_out, ad_out, power_out):
        output.flush()
    return {
        "reused": False,
        "count": count,
        "nonzero": int(np.count_nonzero(power_out > 0.0)),
    }


def running_path(config: str, split: str, kind: str, statistic: str) -> Path:
    return (
        SIONNA_ROOT
        / config
        / "full"
        / "running"
        / f"{split}_{kind}_{statistic}_n{SEEDS}.npy"
    )


def cosine_over_axes(
    prediction: np.ndarray,
    truth: np.ndarray,
    axes: tuple[int, ...],
) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=np.float32)
    truth = np.asarray(truth, dtype=np.float32)
    numerator = np.sum(prediction * truth, axis=axes, dtype=np.float64)
    denominator = np.sqrt(
        np.sum(prediction * prediction, axis=axes, dtype=np.float64)
        * np.sum(truth * truth, axis=axes, dtype=np.float64)
    )
    denominator = np.where(denominator == 0.0, 1e-12, denominator)
    return numerator / denominator


def point_cosine(kind: str, prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    if kind == "pas":
        return cosine_over_axes(prediction, truth, (1, 2, 3)).mean(
            axis=(1, 2), dtype=np.float64
        )
    if kind == "pdp":
        return cosine_over_axes(prediction, truth, (3,)).mean(
            axis=(1, 2), dtype=np.float64
        )
    if kind == "ad":
        return cosine_over_axes(prediction, truth, (1, 2, 3))
    raise ValueError(kind)


def shifted_pas(value: np.ndarray, site: np.ndarray) -> np.ndarray:
    output = np.empty_like(value)
    for site_value, shift in ((0, 2), (1, 2)):
        rows = site == site_value
        output[rows] = np.roll(value[rows], shift=shift, axis=3)
    return output


def shifted_ad(value: np.ndarray, site: np.ndarray) -> np.ndarray:
    output = np.empty_like(value)
    for site_value, shift in ((0, 2), (1, 1)):
        rows = site == site_value
        output[rows] = np.roll(value[rows], shift=shift, axis=2)
    return output


def summarize(values: np.ndarray, valid: np.ndarray) -> dict[str, float | int]:
    selected = np.asarray(values)[valid]
    return {
        "count": int(len(selected)),
        "mean": float(np.mean(selected)) if len(selected) else 0.0,
        "median": float(np.median(selected)) if len(selected) else 0.0,
        "p10": float(np.quantile(selected, 0.10)) if len(selected) else 0.0,
        "p90": float(np.quantile(selected, 0.90)) if len(selected) else 0.0,
    }


def raw_diagnostics(batch_size: int) -> dict[str, object]:
    truth_pas = np.load(TRUTH_PAS, mmap_mode="r")
    truth_pdp = np.load(TRUTH_PDP, mmap_mode="r")
    truth_ad = np.load(TRUTH_AD, mmap_mode="r")
    truth_power = np.load(TRUTH_POWER, mmap_mode="r")
    _, _, _, train_site, _ = load_positions()
    valid = truth_power > 0.0
    count = len(valid)
    metrics: dict[str, list[np.ndarray]] = {
        "base_pas": [],
        "scatter_pas": [],
        "ensemble_pas": [],
        "pas_oracle": [],
        "base_pdp": [],
        "scatter_pdp": [],
        "ensemble_pdp": [],
        "pdp_oracle": [],
        "base_ad": [],
        "scatter_ad": [],
        "ensemble_ad": [],
        "ad_oracle": [],
    }
    uncertainty: dict[str, list[np.ndarray]] = {
        "base_pas": [],
        "scatter_pas": [],
        "base_pdp": [],
        "scatter_pdp": [],
        "base_ad": [],
        "scatter_ad": [],
    }
    sources = {}
    for name, config in (("base", BASE_CONFIG), ("scatter", SCATTER_CONFIG)):
        sources[name] = {
            kind: {
                statistic: np.load(
                    running_path(config, "train", kind, statistic), mmap_mode="r"
                )
                for statistic in ("mean", "m2", "count")
            }
            for kind in ("pas", "pdp", "ad")
        }

    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        site = train_site[start:end]
        truth = {
            "pas": np.asarray(truth_pas[start:end], dtype=np.float32),
            "pdp": np.asarray(truth_pdp[start:end], dtype=np.float32),
            "ad": np.asarray(truth_ad[start:end], dtype=np.float32),
        }
        current = {}
        for source_name in ("base", "scatter"):
            current[source_name] = {}
            for kind in ("pas", "pdp", "ad"):
                raw_mean = np.asarray(
                    sources[source_name][kind]["mean"][start:end], dtype=np.float32
                )
                mean = raw_mean
                if kind == "pas":
                    mean = shifted_pas(mean, site)
                elif kind == "ad":
                    mean = shifted_ad(mean, site)
                axes = tuple(range(1, mean.ndim))
                total = mean.sum(axis=axes, keepdims=True, dtype=np.float64)
                mean = np.divide(
                    mean,
                    np.maximum(total, TINY),
                    out=np.zeros_like(mean),
                    where=total > 0.0,
                )
                current[source_name][kind] = mean
                score = point_cosine(kind, mean, truth[kind])
                metrics[f"{source_name}_{kind}"].append(score)

                m2 = np.asarray(
                    sources[source_name][kind]["m2"][start:end], dtype=np.float32
                )
                count_value = np.asarray(
                    sources[source_name][kind]["count"][start:end], dtype=np.float32
                )
                if kind == "pas":
                    count_value = count_value[:, None, None, None, :, :]
                elif kind == "pdp":
                    count_value = count_value[..., None]
                else:
                    count_value = count_value[:, None, None, None]
                variance = m2 / np.maximum(count_value - 1.0, 1.0)
                relative = np.sqrt(np.maximum(variance, 0.0)) / np.maximum(
                    raw_mean, 1e-8
                )
                uncertainty[f"{source_name}_{kind}"].append(
                    np.mean(np.log1p(np.minimum(relative, 1e4)), axis=axes)
                )

        for kind in ("pas", "pdp", "ad"):
            ensemble = current["base"][kind] + current["scatter"][kind]
            axes = tuple(range(1, ensemble.ndim))
            total = ensemble.sum(axis=axes, keepdims=True, dtype=np.float64)
            ensemble = np.divide(
                ensemble,
                np.maximum(total, TINY),
                out=np.zeros_like(ensemble),
                where=total > 0.0,
            )
            ensemble_score = point_cosine(kind, ensemble, truth[kind])
            metrics[f"ensemble_{kind}"].append(ensemble_score)
            base_score = metrics[f"base_{kind}"][-1]
            scatter_score = metrics[f"scatter_{kind}"][-1]
            metrics[f"{kind}_oracle"].append(
                np.maximum(np.maximum(base_score, scatter_score), ensemble_score)
            )
        print(f"raw diagnostics {end}/{count}", flush=True)

    combined = {name: np.concatenate(parts) for name, parts in metrics.items()}
    combined_uncertainty = {
        name: np.concatenate(parts) for name, parts in uncertainty.items()
    }
    report: dict[str, object] = {
        "overall": {
            name: summarize(value, valid) for name, value in combined.items()
        },
        "by_site": {},
        "uncertainty_error_pearson": {},
    }
    for site_value in (0, 1):
        mask = valid & (train_site == site_value)
        report["by_site"][f"site{site_value}"] = {
            name: summarize(value, mask) for name, value in combined.items()
        }
    for name, value in combined_uncertainty.items():
        score = combined[name]
        mask = valid & np.isfinite(value) & np.isfinite(score)
        correlation = np.corrcoef(value[mask], 1.0 - score[mask])[0, 1]
        report["uncertainty_error_pearson"][name] = float(correlation)
    return report


def validate_outputs() -> dict[str, object]:
    outputs = {
        "pas": TRUTH_PAS,
        "pdp": TRUTH_PDP,
        "ad": TRUTH_AD,
        "power": TRUTH_POWER,
    }
    result = {}
    for name, path in outputs.items():
        value = np.load(path, mmap_mode="r")
        result[name] = {
            "path": PATHS.relative(path),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "finite": bool(np.all(np.isfinite(value))),
            "bytes": int(path.stat().st_size),
        }
    return result


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    preparation = prepare_truth(torch.device(args.device), args.batch_size, args.force)
    diagnostics = raw_diagnostics(args.batch_size)
    report = {
        "experiment": VERSION,
        "status": "complete",
        "truth_preparation": preparation,
        "sionna": {
            "base_config": BASE_CONFIG,
            "scatter_config": SCATTER_CONFIG,
            "seeds": SEEDS,
            "fixed_pas_h_shifts": {"site0": 2, "site1": 2},
            "fixed_ad_h_shifts": {"site0": 2, "site1": 1},
        },
        "raw_diagnostics": diagnostics,
        "outputs": validate_outputs(),
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
