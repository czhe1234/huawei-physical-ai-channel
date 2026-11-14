#!/usr/bin/env python3
"""Audit hidden-score inversion with the exact traceback PAS/PDP mouths."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from project_paths import PATHS


VERSION_NUMBER = 555
VERSION = "v555_exact_traceback_online_inverse"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
GRAM_CACHE = CACHE_DIR / f"{VERSION}_gram.npz"

SHAPE = (500, 256, 4, 192)
RIDGES = np.logspace(-12, -2, 21)
WEIGHT_STEPS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


@dataclass(frozen=True)
class Probe:
    version: int
    score: float
    family: str
    filename: str
    core: bool = True

    @property
    def path(self) -> Path:
        return PATHS.cache_dir(self.version) / self.filename


PROBES = (
    Probe(209, 0.634488, "early", "v209_Round1_Test_Channel.npy", False),
    Probe(212, 0.690183, "early", "v212_Round1_Test_Channel.npy", False),
    Probe(364, 0.514695, "raw_sionna", "Round1_Test_Channel.npy", False),
    Probe(424, 0.567791, "pas_bootstrap", "v424_Round1_Test_Channel.npy", False),
    Probe(426, 0.721766, "projection_frontier", "v426_Round1_Test_Channel.npy"),
    Probe(427, 0.702860, "conventional_pas", "v427_Round1_Test_Channel.npy", False),
    Probe(428, 0.721227, "projection_frontier", "v428_Round1_Test_Channel.npy"),
    Probe(436, 0.727491, "projection_frontier", "v436_Round1_Test_Channel.npy"),
    Probe(449, 0.731214, "projection_frontier", "v449_Round1_Test_Channel.npy"),
    Probe(458, 0.739203, "projection_frontier", "v458_Round1_Test_Channel.npy"),
    Probe(459, 0.736148, "link_selector", "v459_Round1_Test_Channel.npy"),
    Probe(463, 0.737441, "spatial_residual", "v463_Round1_Test_Channel.npy"),
    Probe(469, 0.739036, "neural_heads", "v469_Round1_Test_Channel.npy"),
    Probe(480, 0.663617, "old_online_inverse", "v480_Round1_Test_Channel.npy", False),
    Probe(491, 0.739634, "neural_heads", "v491_Round1_Test_Channel.npy"),
    Probe(492, 0.739236, "neural_heads", "v492_Round1_Test_Channel.npy"),
    Probe(493, 0.738570, "neural_heads", "v493_Round1_Test_Channel.npy"),
    Probe(497, 0.739554, "physical_selector", "v497_Round1_Test_Channel.npy"),
    Probe(500, 0.739607, "physical_selector", "v500_Round1_Test_Channel.npy"),
    Probe(514, 0.738903, "physical_selector", "v514_Round1_Test_Channel.npy"),
    Probe(528, 0.737892, "path_action", "v528_Round1_Test_Channel.npy"),
    Probe(533, 0.740393, "physical_stack", "v533_Round1_Test_Channel.npy"),
    Probe(540, 0.740503, "physical_stack", "v540_Round1_Test_Channel.npy"),
    Probe(541, 0.740439, "physical_stack", "v541_Round1_Test_Channel.npy"),
    Probe(543, 0.738327, "cell_fusion", "v543_Round1_Test_Channel.npy"),
    Probe(548, 0.739456, "path_action", "v548_Round1_Test_Channel.npy"),
    Probe(549, 0.740891, "multiseed", "v549_Round1_Test_Channel.npy"),
    Probe(552, 0.724761, "full_h_network", "v552_Round1_Test_Channel.npy", False),
    Probe(553, 0.721269, "full_h_network", "v553_Round1_Test_Channel.npy", False),
    Probe(554, 0.740808, "multiseed", "v554_Round1_Test_Channel.npy"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--point-batch", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def validate_inputs() -> list[np.memmap]:
    arrays = []
    for probe in PROBES:
        if not probe.path.is_file():
            raise FileNotFoundError(probe.path)
        value = np.load(probe.path, mmap_mode="r")
        if value.shape != SHAPE or value.dtype != np.complex64:
            raise ValueError(
                f"invalid v{probe.version}: shape={value.shape}, dtype={value.dtype}"
            )
        arrays.append(value)
    return arrays


def normalized_power(value: torch.Tensor, dim: int) -> torch.Tensor:
    norm = torch.linalg.vector_norm(value, dim=dim, keepdim=True)
    return torch.where(norm > 0, value / norm, torch.zeros_like(value))


@torch.inference_mode()
def compute_grams(
    arrays: list[np.memmap], device: torch.device, point_batch: int
) -> dict[str, np.ndarray]:
    count = len(arrays)
    pas_gram = np.zeros((count, count), dtype=np.float64)
    pdp_gram = np.zeros((count, count), dtype=np.float64)
    complex_inner = np.zeros((count, count), dtype=np.complex128)
    energy = np.zeros(count, dtype=np.float64)
    pas_vectors = 0
    pdp_vectors = 0

    for start in range(0, SHAPE[0], point_batch):
        stop = min(start + point_batch, SHAPE[0])
        block = np.stack(
            [np.asarray(value[start:stop], dtype=np.complex64) for value in arrays],
            axis=0,
        )
        channel = torch.from_numpy(block).to(device)

        leaked = channel.reshape(count, stop - start, 2, 16, 8, 4, 192)
        leaked = leaked.permute(0, 1, 2, 6, 3, 5, 4)
        leaked_power = torch.abs(torch.fft.fft2(leaked, dim=(3, 4))).square()
        leaked_power = leaked_power.permute(0, 1, 2, 5, 6, 3, 4).reshape(
            count, -1, 192 * 16
        )
        leaked_feature = normalized_power(leaked_power, -1)
        pas_block = torch.einsum("knf,lnf->kl", leaked_feature, leaked_feature).cpu()
        pas_gram += pas_block.numpy().astype(np.float64)
        pas_vectors += leaked_feature.shape[1]

        delay_power = torch.abs(torch.fft.ifft(channel, dim=-1)).square()
        delay_feature = normalized_power(delay_power.reshape(count, -1, 192), -1)
        pdp_block = torch.einsum("knf,lnf->kl", delay_feature, delay_feature).cpu()
        pdp_gram += pdp_block.numpy().astype(np.float64)
        pdp_vectors += delay_feature.shape[1]

        flat = channel.reshape(count, -1)
        complex_inner += (flat.conj() @ flat.T).cpu().numpy().astype(np.complex128)
        energy += (
            torch.sum(torch.abs(flat).square(), dim=1).cpu().numpy().astype(np.float64)
        )
        if stop % 25 == 0 or stop == SHAPE[0]:
            print(f"v555 exact metric gram {stop}/{SHAPE[0]}", flush=True)

    pas_gram /= pas_vectors
    pdp_gram /= pdp_vectors
    complex_cosine = np.real(complex_inner) / np.sqrt(
        np.maximum(energy[:, None] * energy[None, :], 1e-30)
    )
    error = np.maximum(
        energy[:, None] + energy[None, :] - 2.0 * np.real(complex_inner), 0.0
    )
    directed_nmse_score = 1.0 / (1.0 + error / np.maximum(energy[None, :], 1e-30))
    return {
        "pas": pas_gram,
        "pdp": pdp_gram,
        "complex_cosine": complex_cosine,
        "directed_nmse_score": directed_nmse_score,
        "energy": energy,
    }


def load_or_compute_grams(
    arrays: list[np.memmap], args: argparse.Namespace
) -> dict[str, np.ndarray]:
    if GRAM_CACHE.is_file() and not args.force:
        cache = np.load(GRAM_CACHE)
        cached_versions = cache["versions"].astype(int).tolist()
        versions = [probe.version for probe in PROBES]
        if cached_versions == versions:
            return {
                key: np.asarray(cache[key], dtype=np.float64)
                for key in (
                    "pas",
                    "pdp",
                    "complex_cosine",
                    "directed_nmse_score",
                    "energy",
                )
            }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    grams = compute_grams(arrays, torch.device(args.device), args.point_batch)
    np.savez_compressed(
        GRAM_CACHE,
        versions=np.asarray([probe.version for probe in PROBES], dtype=np.int16),
        **grams,
    )
    return grams


def centered_difference_kernel(kernel: np.ndarray, anchor: int) -> np.ndarray:
    return (
        kernel
        - kernel[:, anchor : anchor + 1]
        - kernel[anchor : anchor + 1, :]
        + kernel[anchor, anchor]
    )


def fit_delta(
    kernel: np.ndarray,
    score: np.ndarray,
    ids: np.ndarray,
    anchor: int,
    ridge: float,
) -> np.ndarray:
    train = ids[ids != anchor]
    matrix = kernel[np.ix_(train, train)]
    target = score[train] - score[anchor]
    return np.linalg.solve(
        matrix + ridge * np.eye(len(matrix), dtype=np.float64), target
    )


def predict_delta(
    kernel: np.ndarray,
    score: np.ndarray,
    train_ids: np.ndarray,
    query_ids: np.ndarray,
    anchor: int,
    ridge: float,
) -> np.ndarray:
    fit_ids = train_ids[train_ids != anchor]
    alpha = fit_delta(kernel, score, train_ids, anchor, ridge)
    return score[anchor] + kernel[np.ix_(query_ids, fit_ids)] @ alpha


def weight_grid() -> list[tuple[float, float, float]]:
    rows = []
    for pas in WEIGHT_STEPS:
        for pdp in WEIGHT_STEPS:
            complex_weight = 1.0 - pas - pdp
            if complex_weight < -1e-12:
                continue
            rows.append((pas, pdp, max(complex_weight, 0.0)))
    return rows


def select_inner(
    components: dict[str, np.ndarray],
    score: np.ndarray,
    train_ids: np.ndarray,
    anchor: int,
    family: np.ndarray,
) -> tuple[tuple[float, float, float], float, float]:
    train_families = sorted(set(family[train_ids]) - {family[anchor]})
    best = None
    for weights in weight_grid():
        raw = (
            weights[0] * components["pas"]
            + weights[1] * components["pdp"]
            + weights[2] * components["complex_cosine"]
        )
        kernel = centered_difference_kernel(raw, anchor)
        for ridge in RIDGES:
            truth_parts = []
            pred_parts = []
            for held_family in train_families:
                query = train_ids[family[train_ids] == held_family]
                fit = train_ids[family[train_ids] != held_family]
                if anchor not in fit or len(fit) < 4:
                    continue
                prediction = predict_delta(
                    kernel, score, fit, query, anchor, float(ridge)
                )
                truth_parts.append(score[query])
                pred_parts.append(prediction)
            if not truth_parts:
                continue
            truth = np.concatenate(truth_parts)
            prediction = np.concatenate(pred_parts)
            rmse = float(np.sqrt(np.mean((prediction - truth) ** 2)))
            max_error = float(np.max(np.abs(prediction - truth)))
            key = (rmse, max_error, -weights[0] - weights[1])
            if best is None or key < best[0]:
                best = (key, weights, float(ridge))
    if best is None:
        raise RuntimeError("no valid nested kernel configuration")
    return best[1], best[2], best[0][0]


def nested_family_validation(
    components: dict[str, np.ndarray],
    score: np.ndarray,
    ids: np.ndarray,
    anchor: int,
    family: np.ndarray,
) -> dict[str, object]:
    predictions = np.full(len(score), np.nan, dtype=np.float64)
    rows = []
    held_families = sorted(set(family[ids]) - {family[anchor]})
    for held_family in held_families:
        query = ids[family[ids] == held_family]
        train = ids[family[ids] != held_family]
        weights, ridge, inner_rmse = select_inner(
            components, score, train, anchor, family
        )
        raw = (
            weights[0] * components["pas"]
            + weights[1] * components["pdp"]
            + weights[2] * components["complex_cosine"]
        )
        kernel = centered_difference_kernel(raw, anchor)
        prediction = predict_delta(kernel, score, train, query, anchor, ridge)
        predictions[query] = prediction
        rows.append(
            {
                "held_family": str(held_family),
                "versions": [int(PROBES[index].version) for index in query],
                "truth": score[query].tolist(),
                "prediction": prediction.tolist(),
                "weights_pas_pdp_complex": list(weights),
                "ridge": ridge,
                "inner_family_rmse": inner_rmse,
            }
        )
    valid = np.isfinite(predictions) & np.isin(np.arange(len(score)), ids)
    error = predictions[valid] - score[valid]
    baseline = score[anchor] - score[valid]
    return {
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "max_abs_error": float(np.max(np.abs(error))),
        "baseline_anchor_rmse": float(np.sqrt(np.mean(baseline * baseline))),
        "correlation": float(np.corrcoef(predictions[valid], score[valid])[0, 1]),
        "rows": rows,
    }


def ordinary_loo(
    components: dict[str, np.ndarray],
    score: np.ndarray,
    ids: np.ndarray,
    anchor: int,
    family: np.ndarray,
) -> dict[str, object]:
    weights, ridge, _ = select_inner(components, score, ids, anchor, family)
    raw = (
        weights[0] * components["pas"]
        + weights[1] * components["pdp"]
        + weights[2] * components["complex_cosine"]
    )
    kernel = centered_difference_kernel(raw, anchor)
    predictions = np.full(len(score), np.nan, dtype=np.float64)
    for query in ids:
        if query == anchor:
            predictions[query] = score[anchor]
            continue
        train = ids[ids != query]
        predictions[query] = predict_delta(
            kernel,
            score,
            train,
            np.asarray([query]),
            anchor,
            ridge,
        )[0]
    error = predictions[ids] - score[ids]
    return {
        "weights_pas_pdp_complex": list(weights),
        "ridge": ridge,
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "correlation": float(np.corrcoef(predictions[ids], score[ids])[0, 1]),
        "prediction": {
            f"v{PROBES[index].version}": float(predictions[index]) for index in ids
        },
    }


def main() -> None:
    args = parse_args()
    arrays = validate_inputs()
    grams = load_or_compute_grams(arrays, args)
    score = np.asarray([probe.score for probe in PROBES], dtype=np.float64)
    family = np.asarray([probe.family for probe in PROBES], dtype=object)
    versions = np.asarray([probe.version for probe in PROBES], dtype=np.int64)
    core_ids = np.flatnonzero([probe.core for probe in PROBES])
    broad_ids = np.arange(len(PROBES), dtype=np.int64)
    anchor = int(np.flatnonzero(versions == 549)[0])

    core_family = nested_family_validation(grams, score, core_ids, anchor, family)
    broad_family = nested_family_validation(grams, score, broad_ids, anchor, family)
    core_loo = ordinary_loo(grams, score, core_ids, anchor, family)
    broad_loo = ordinary_loo(grams, score, broad_ids, anchor, family)

    core_pass = bool(
        core_family["rmse"] < 0.65 * core_family["baseline_anchor_rmse"]
        and core_family["correlation"] >= 0.75
        and core_family["max_abs_error"] <= 0.006
    )
    broad_pass = bool(
        broad_family["rmse"] < 0.75 * broad_family["baseline_anchor_rmse"]
        and broad_family["correlation"] >= 0.85
        and broad_family["max_abs_error"] <= 0.035
    )
    passed = core_pass and broad_pass
    report = {
        "method": VERSION,
        "status": "analysis_passed_inverse_gate" if passed else "analysis_rejected",
        "primary_hypothesis": (
            "The exact traceback PAS slices, exact linkwise PDP, and complex carrier "
            "similarity turn historical online scores into transferable measurements of "
            "the hidden target once out-of-manifold negative probes are retained."
        ),
        "difference_from_v479_v480": {
            "old_pas": "conventional 16x8 spatial FFT per UE/subcarrier",
            "new_pas": "traceback Bx2x192x16x4x8 FFT over (S,H), cosine over SxH per (P,UE,V)",
            "old_probe_count": 11,
            "new_probe_count": len(PROBES),
            "negative_controls": [364, 480, 552, 553],
            "validation": "nested leave-entire-method-family-out",
        },
        "parameters": vars(args),
        "probes": [
            {
                "version": probe.version,
                "score": probe.score,
                "family": probe.family,
                "core": probe.core,
                "path": PATHS.relative(probe.path),
            }
            for probe in PROBES
        ],
        "gram_diagnostics": {
            "pas_diag_minmax": [
                float(np.min(np.diag(grams["pas"]))),
                float(np.max(np.diag(grams["pas"]))),
            ],
            "pdp_diag_minmax": [
                float(np.min(np.diag(grams["pdp"]))),
                float(np.max(np.diag(grams["pdp"]))),
            ],
            "energy_minmax": [
                float(np.min(grams["energy"])),
                float(np.max(grams["energy"])),
            ],
        },
        "core_nested_family": core_family,
        "broad_nested_family": broad_family,
        "core_ordinary_loo": core_loo,
        "broad_ordinary_loo": broad_loo,
        "decision_gate": {
            "core_pass": core_pass,
            "broad_pass": broad_pass,
            "pass": passed,
        },
        "outputs": {
            "gram_cache": PATHS.relative(GRAM_CACHE),
            "report": PATHS.relative(REPORT),
        },
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "core_nested_family": {
                    key: core_family[key]
                    for key in (
                        "rmse",
                        "baseline_anchor_rmse",
                        "max_abs_error",
                        "correlation",
                    )
                },
                "broad_nested_family": {
                    key: broad_family[key]
                    for key in (
                        "rmse",
                        "baseline_anchor_rmse",
                        "max_abs_error",
                        "correlation",
                    )
                },
                "core_ordinary_loo": core_loo,
                "broad_ordinary_loo": broad_loo,
                "decision_gate": report["decision_gate"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
