#!/usr/bin/env python3
"""Test analytic LOS component transport on strict spatial OOF carriers.

The experiment changes only the rank-one geometric LOS component. The remaining
NLOS channel is inherited element-for-element from the OOF carrier. It scores the
literal traceback PAS mouth in addition to PDP and global NMSE.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
import torch

import analyze_v401_dualdomain_pas_preserving_confocal as v401
from project_paths import PATHS


VERSION_NUMBER = 556
VERSION = "v556_analytic_los_component_transport"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
OOF_CACHE = CACHE_DIR / "v556_analytic_los_oof.npz"

TRAIN_H = PATHS.round1 / "Round1_Train_Channel.npy"
TRAIN_POS = PATHS.round1 / "Round1_Train_Pos.npy"
OOF_BASE = PATHS.cache_dir(471) / "v471_train_v98like_base_h.npy"
SIONNA_H = PATHS.cache_dir(208) / "v208_sionna_train_clean_H.npy"
GEOMETRY = PATHS.shared_features / "height_geometry_cache.npz"
V534_REPORT = (
    PATHS.version_report_dir(534)
    / "v534_los_official_sionna_configuration_inverse.json"
)

BS_POSITION = np.asarray([50.0, 0.0, 25.0], dtype=np.float64)
LIGHT_SPEED = 299_792_458.0
SUBCARRIER_SPACING_HZ = 30_000.0
SIONNA_FREQUENCY_HZ = 6.0e9
N_SUBCARRIERS = 192
TINY = 1e-30
ALPHAS = (0.25, 0.50, 0.75, 1.00)
DEPLOYABLE_METHODS = (
    "official_knn4",
    "official_knn8",
    "official_knn16",
    "sionna_affine",
    "sionna_kernel",
    "hybrid_knn_sionna",
    "sionna_kernel_port_norm_lock",
)


@dataclass(frozen=True)
class SystemParameters:
    delay_offset_cycles: float
    beta_rad_per_m: float
    horizontal_vector: np.ndarray
    vertical_vector: np.ndarray
    receiver_phase_pi: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def require(path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def load_parameters() -> SystemParameters:
    report = json.loads(V534_REPORT.read_text(encoding="utf-8"))
    array = report["bs_array"]
    receiver = report["ue_array"]
    return SystemParameters(
        delay_offset_cycles=float(
            report["frequency_axis_and_delay"][
                "final_delay_offset_cycles_per_subcarrier"
            ]
        ),
        beta_rad_per_m=float(report["carrier_phase"]["beta_rad_per_m"]),
        horizontal_vector=np.asarray(
            array["horizontal_fitted_vector"], dtype=np.float64
        ),
        vertical_vector=np.asarray(array["vertical_fitted_vector"], dtype=np.float64),
        receiver_phase_pi=float(receiver["observed_abs_phase_median_pi"] * 0.0),
    )


def geometry(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vector = np.asarray(positions, dtype=np.float64) - BS_POSITION[None, :]
    distance = np.linalg.norm(vector, axis=1)
    direction = vector / np.maximum(distance[:, None], TINY)
    delay_cycles = distance * SUBCARRIER_SPACING_HZ / LIGHT_SPEED
    return distance, direction, delay_cycles


def extract_coefficients(
    channel: np.ndarray,
    delay_cycles: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows = len(channel)
    coefficient = np.empty((rows, 2, 16, 8, 2, 2), dtype=np.complex64)
    fraction = np.empty(rows, dtype=np.float64)
    frequency = np.arange(N_SUBCARRIERS, dtype=np.float64)
    for start in range(0, rows, batch_size):
        end = min(start + batch_size, rows)
        value = np.asarray(channel[start:end], dtype=np.complex64)
        steering = np.exp(
            2j * np.pi * delay_cycles[start:end, None] * frequency[None, :]
        )
        current = (
            np.einsum("nurf,nf->nur", value, steering, optimize=True) / N_SUBCARRIERS
        )
        coefficient[start:end] = current.reshape(end - start, 2, 16, 8, 2, 2)
        total = np.mean(
            np.sum(np.abs(value) ** 2, axis=(1, 2), dtype=np.float64),
            axis=-1,
        )
        direct = np.sum(np.abs(current) ** 2, axis=(1, 2), dtype=np.float64)
        fraction[start:end] = direct / np.maximum(total, TINY)
        if end % 200 == 0 or end == rows:
            print(f"v556 extract coefficients {end}/{rows}", flush=True)
    return coefficient, fraction


def adjacent_phase(value: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    left = [slice(None)] * value.ndim
    right = [slice(None)] * value.ndim
    left[axis] = slice(None, -1)
    right[axis] = slice(1, None)
    product = np.conj(value[tuple(left)]) * value[tuple(right)]
    axes = tuple(range(1, value.ndim))
    summed = np.sum(product, axis=axes, dtype=np.complex128)
    denominator = np.sum(np.abs(product), axis=axes, dtype=np.float64)
    return (
        (np.angle(summed) / np.pi).astype(np.float64),
        (np.abs(summed) / np.maximum(denominator, TINY)).astype(np.float64),
    )


def fixed_array_phases(
    direction: np.ndarray, parameters: SystemParameters
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    horizontal = direction @ parameters.horizontal_vector
    vertical = direction @ parameters.vertical_vector
    receiver = np.full(len(direction), parameters.receiver_phase_pi, dtype=np.float64)
    return horizontal, vertical, receiver


def reduce_port(
    coefficient: np.ndarray,
    horizontal_phase: np.ndarray,
    vertical_phase: np.ndarray,
    receiver_phase: np.ndarray,
) -> np.ndarray:
    horizontal = np.arange(16, dtype=np.float64)[None, None, :, None, None, None]
    vertical = np.arange(8, dtype=np.float64)[None, None, None, :, None, None]
    receiver = np.arange(2, dtype=np.float64)[None, None, None, None, None, :]
    correction = np.exp(
        -1j
        * np.pi
        * (
            horizontal_phase[:, None, None, None, None, None] * horizontal
            + vertical_phase[:, None, None, None, None, None] * vertical
            + receiver_phase[:, None, None, None, None, None] * receiver
        )
    )
    return np.mean(
        coefficient * correction, axis=(2, 3, 5), dtype=np.complex128
    ).astype(np.complex64)


def render_port(
    port: np.ndarray,
    delay_cycles: np.ndarray,
    horizontal_phase: np.ndarray,
    vertical_phase: np.ndarray,
    receiver_phase: np.ndarray,
) -> np.ndarray:
    rows = len(port)
    horizontal = np.arange(16, dtype=np.float64)[None, None, :, None, None, None]
    vertical = np.arange(8, dtype=np.float64)[None, None, None, :, None, None]
    receiver = np.arange(2, dtype=np.float64)[None, None, None, None, None, :]
    array = np.exp(
        1j
        * np.pi
        * (
            horizontal_phase[:, None, None, None, None, None] * horizontal
            + vertical_phase[:, None, None, None, None, None] * vertical
            + receiver_phase[:, None, None, None, None, None] * receiver
        )
    )
    coefficient = (
        port[:, :, None, None, :, None].astype(np.complex128) * array
    ).reshape(rows, 256, 4)
    frequency = np.arange(N_SUBCARRIERS, dtype=np.float64)
    steering = np.exp(-2j * np.pi * delay_cycles[:, None] * frequency[None, :])
    return (coefficient[..., None] * steering[:, None, None, :]).astype(np.complex64)


def source_quality(
    coefficient: np.ndarray, direct_fraction: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    h_phase, h_coherence = adjacent_phase(coefficient, 2)
    v_phase, v_coherence = adjacent_phase(coefficient, 3)
    r_phase, r_coherence = adjacent_phase(coefficient, 5)
    quality = (
        np.maximum(direct_fraction, 1e-8)
        * np.sqrt(np.maximum(h_coherence, 1e-8))
        * np.sqrt(np.maximum(v_coherence, 1e-8))
        * np.power(np.maximum(r_coherence, 1e-8), 0.25)
    )
    return quality, {
        "horizontal_phase": h_phase,
        "vertical_phase": v_phase,
        "receiver_phase": r_phase,
        "horizontal_coherence": h_coherence,
        "vertical_coherence": v_coherence,
        "receiver_coherence": r_coherence,
    }


def knn_predict(
    train_xy: np.ndarray,
    source_ids: np.ndarray,
    query_ids: np.ndarray,
    values: np.ndarray,
    k: int,
) -> np.ndarray:
    tree = cKDTree(train_xy[source_ids])
    distance, local = tree.query(train_xy[query_ids], k=min(k, len(source_ids)))
    if np.ndim(distance) == 1:
        distance = np.asarray(distance)[:, None]
        local = np.asarray(local)[:, None]
    weight = 1.0 / np.maximum(distance + 1.0, 1.0) ** 1.5
    weight /= np.maximum(weight.sum(axis=1, keepdims=True), TINY)
    return np.sum(
        weight[:, :, None, None] * values[source_ids[local]],
        axis=1,
        dtype=np.complex128,
    ).astype(np.complex64)


def fit_complex_affine(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    x = source.reshape(len(source), -1).astype(np.complex128)
    y = target.reshape(len(target), -1).astype(np.complex128)
    scale = np.maximum(np.sqrt(np.mean(np.abs(x) ** 2, axis=0)), 1e-12)
    design = np.column_stack([np.ones(len(x), dtype=np.complex128), x / scale])
    gram = design.conj().T @ design
    ridge = 1e-3 * float(np.trace(gram).real) / len(gram)
    gram.flat[:: len(gram) + 1] += ridge
    weight = np.linalg.solve(gram, design.conj().T @ y)
    return np.vstack([scale.astype(np.complex128), weight])


def apply_complex_affine(source: np.ndarray, model: np.ndarray) -> np.ndarray:
    x = source.reshape(len(source), -1).astype(np.complex128)
    scale = model[0].real
    weight = model[1:]
    design = np.column_stack([np.ones(len(x), dtype=np.complex128), x / scale])
    return (design @ weight).reshape(len(source), 2, 2).astype(np.complex64)


def kernel_residual_predict(
    train_xy: np.ndarray,
    source_ids: np.ndarray,
    query_ids: np.ndarray,
    residual: np.ndarray,
    length_scale: float = 45.0,
) -> np.ndarray:
    source_xy = train_xy[source_ids].astype(np.float64)
    query_xy = train_xy[query_ids].astype(np.float64)
    train_delta = source_xy[:, None, :] - source_xy[None, :, :]
    query_delta = query_xy[:, None, :] - source_xy[None, :, :]
    train_kernel = np.exp(
        -np.sum(train_delta * train_delta, axis=2) / (2.0 * length_scale**2)
    )
    query_kernel = np.exp(
        -np.sum(query_delta * query_delta, axis=2) / (2.0 * length_scale**2)
    )
    ridge = 0.03
    coefficient = np.linalg.solve(
        train_kernel + ridge * np.eye(len(source_ids)),
        residual[source_ids].reshape(len(source_ids), -1).astype(np.complex128),
    )
    return (
        (query_kernel @ coefficient).reshape(len(query_ids), 2, 2).astype(np.complex64)
    )


def lock_port_norm(prediction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    pred_norm = np.linalg.norm(prediction.reshape(len(prediction), -1), axis=1)
    ref_norm = np.linalg.norm(reference.reshape(len(reference), -1), axis=1)
    scale = ref_norm / np.maximum(pred_norm, TINY)
    return (prediction * scale[:, None, None]).astype(np.complex64)


def torch_cosine(
    left: torch.Tensor, right: torch.Tensor, axes: tuple[int, ...]
) -> torch.Tensor:
    numerator = torch.sum(left * right, dim=axes)
    denominator = torch.sqrt(
        torch.sum(left * left, dim=axes) * torch.sum(right * right, dim=axes)
    )
    return numerator / torch.where(
        denominator > 0.0,
        denominator,
        torch.full_like(denominator, 1e-12),
    )


@torch.inference_mode()
def point_metrics(
    prediction: np.ndarray, truth: np.ndarray, device: torch.device
) -> dict[str, np.ndarray]:
    rows = len(prediction)
    prediction_tensor = torch.from_numpy(
        np.ascontiguousarray(prediction, dtype=np.complex64)
    ).to(device)
    truth_tensor = torch.from_numpy(np.ascontiguousarray(truth, dtype=np.complex64)).to(
        device
    )
    pred6 = prediction_tensor.reshape(rows, 2, 16, 8, 4, N_SUBCARRIERS)
    truth6 = truth_tensor.reshape(rows, 2, 16, 8, 4, N_SUBCARRIERS)

    pred_trace = pred6.permute(0, 1, 5, 2, 4, 3)
    truth_trace = truth6.permute(0, 1, 5, 2, 4, 3)
    pred_trace = torch.abs(torch.fft.fft2(pred_trace, dim=(2, 3))).square().float()
    truth_trace = torch.abs(torch.fft.fft2(truth_trace, dim=(2, 3))).square().float()
    traceback_pas = torch_cosine(pred_trace, truth_trace, (2, 3)).mean(dim=(1, 2, 3))

    pred_conventional = torch.abs(torch.fft.fft2(pred6, dim=(2, 3))).square().float()
    truth_conventional = torch.abs(torch.fft.fft2(truth6, dim=(2, 3))).square().float()
    conventional_pas = torch_cosine(
        pred_conventional.reshape(rows, 256, 4, N_SUBCARRIERS),
        truth_conventional.reshape(rows, 256, 4, N_SUBCARRIERS),
        (1,),
    ).mean(dim=(1, 2))

    pred_delay = torch.abs(torch.fft.ifft(prediction_tensor, dim=-1)).square().float()
    truth_delay = torch.abs(torch.fft.ifft(truth_tensor, dim=-1)).square().float()
    pdp = torch_cosine(pred_delay, truth_delay, (3,)).mean(dim=(1, 2))
    error = torch.sum(
        torch.abs(prediction_tensor - truth_tensor).square(), dim=(1, 2, 3)
    )
    energy = torch.sum(torch.abs(truth_tensor).square(), dim=(1, 2, 3))
    return {
        "traceback_pas": traceback_pas.cpu().numpy().astype(np.float32),
        "conventional_pas": conventional_pas.cpu().numpy().astype(np.float32),
        "pdp": pdp.cpu().numpy().astype(np.float32),
        "error": error.double().cpu().numpy(),
        "energy": energy.double().cpu().numpy(),
    }


def score(
    metrics: dict[str, np.ndarray], mask: np.ndarray | None = None
) -> dict[str, float]:
    if mask is None:
        mask = np.ones(len(metrics["pdp"]), dtype=bool)
    nmse = float(
        np.sum(metrics["error"][mask])
        / max(float(np.sum(metrics["energy"][mask])), TINY)
    )
    nmse_score = 1.0 / (1.0 + nmse)
    pas = float(np.mean(metrics["traceback_pas"][mask]))
    pdp = float(np.mean(metrics["pdp"][mask]))
    return {
        "count": int(np.sum(mask)),
        "traceback_pas": pas,
        "conventional_pas": float(np.mean(metrics["conventional_pas"][mask])),
        "pdp": pdp,
        "global_nmse": nmse,
        "nmse_score": nmse_score,
        "official_score": 0.4 * pas + 0.4 * pdp + 0.2 * nmse_score,
    }


def score_delta(
    candidate: dict[str, float], base: dict[str, float]
) -> dict[str, float]:
    return {
        key: float(candidate[key] - base[key])
        for key in (
            "traceback_pas",
            "conventional_pas",
            "pdp",
            "global_nmse",
            "nmse_score",
            "official_score",
        )
    }


def compute_base_metrics(
    base: np.ndarray,
    truth: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    output = {
        "traceback_pas": np.empty(len(base), dtype=np.float32),
        "conventional_pas": np.empty(len(base), dtype=np.float32),
        "pdp": np.empty(len(base), dtype=np.float32),
        "error": np.empty(len(base), dtype=np.float64),
        "energy": np.empty(len(base), dtype=np.float64),
    }
    for start in range(0, len(base), batch_size):
        end = min(start + batch_size, len(base))
        current = point_metrics(
            np.asarray(base[start:end], dtype=np.complex64),
            np.asarray(truth[start:end], dtype=np.complex64),
            device,
        )
        for key in output:
            output[key][start:end] = current[key]
        if end % 200 == 0 or end == len(base):
            print(f"v556 base score {end}/{len(base)}", flush=True)
    return output


def candidate_metrics(
    base: np.ndarray,
    truth: np.ndarray,
    query_ids: np.ndarray,
    base_port: np.ndarray,
    target_port: np.ndarray,
    delay_cycles: np.ndarray,
    horizontal_phase: np.ndarray,
    vertical_phase: np.ndarray,
    receiver_phase: np.ndarray,
    alpha: float,
    batch_size: int,
    device: torch.device,
    *,
    delay_shift_bins: float = 0.0,
    horizontal_shift_bins: float = 0.0,
) -> dict[str, np.ndarray]:
    output = {
        "traceback_pas": np.empty(len(query_ids), dtype=np.float32),
        "conventional_pas": np.empty(len(query_ids), dtype=np.float32),
        "pdp": np.empty(len(query_ids), dtype=np.float32),
        "error": np.empty(len(query_ids), dtype=np.float64),
        "energy": np.empty(len(query_ids), dtype=np.float64),
    }
    for start in range(0, len(query_ids), batch_size):
        end = min(start + batch_size, len(query_ids))
        ids = query_ids[start:end]
        old_direct = render_port(
            base_port[ids],
            delay_cycles[ids],
            horizontal_phase[ids],
            vertical_phase[ids],
            receiver_phase[ids],
        )
        new_direct = render_port(
            target_port[start:end],
            delay_cycles[ids] + delay_shift_bins / N_SUBCARRIERS,
            horizontal_phase[ids] + 2.0 * horizontal_shift_bins / 16.0,
            vertical_phase[ids],
            receiver_phase[ids],
        )
        prediction = (
            np.asarray(base[ids], dtype=np.complex64)
            + np.float32(alpha) * (new_direct - old_direct)
        ).astype(np.complex64)
        current = point_metrics(
            prediction, np.asarray(truth[ids], dtype=np.complex64), device
        )
        for key in output:
            output[key][start:end] = current[key]
    return output


def replace_rows(
    base_metrics: dict[str, np.ndarray],
    ids: np.ndarray,
    replacement: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    output = {key: value.copy() for key, value in base_metrics.items()}
    for key in output:
        output[key][ids] = replacement[key]
    return output


def main() -> None:
    args = parse_args()
    for path in (TRAIN_H, TRAIN_POS, OOF_BASE, SIONNA_H, GEOMETRY, V534_REPORT):
        require(path)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    positions = np.load(TRAIN_POS).astype(np.float64)
    truth = np.load(TRAIN_H, mmap_mode="r")
    base = np.load(OOF_BASE, mmap_mode="r")
    sionna = np.load(SIONNA_H, mmap_mode="r")
    blocked = np.load(GEOMETRY)["train_blocked"].astype(np.int32)
    parameters = load_parameters()
    device = torch.device(
        args.device
        if torch.cuda.is_available() and args.device.startswith("cuda")
        else "cpu"
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    print(f"v556 metric device={device}", flush=True)
    distance, direction, geometric_delay = geometry(positions)
    official_delay = geometric_delay + parameters.delay_offset_cycles
    horizontal_phase, vertical_phase, receiver_phase = fixed_array_phases(
        direction, parameters
    )

    truth_coefficient, truth_fraction = extract_coefficients(
        truth, official_delay, args.batch_size
    )
    base_coefficient, base_fraction = extract_coefficients(
        base, official_delay, args.batch_size
    )
    sionna_coefficient, sionna_fraction = extract_coefficients(
        sionna, geometric_delay, args.batch_size
    )
    quality, official_observed = source_quality(truth_coefficient, truth_fraction)
    sionna_quality, sionna_observed = source_quality(
        sionna_coefficient, sionna_fraction
    )

    truth_port = reduce_port(
        truth_coefficient, horizontal_phase, vertical_phase, receiver_phase
    )
    base_port = reduce_port(
        base_coefficient, horizontal_phase, vertical_phase, receiver_phase
    )
    sionna_port = reduce_port(
        sionna_coefficient,
        sionna_observed["horizontal_phase"],
        sionna_observed["vertical_phase"],
        sionna_observed["receiver_phase"],
    )
    truth_canonical = truth_port * np.exp(
        1j * parameters.beta_rad_per_m * distance[:, None, None]
    ).astype(np.complex64)
    sionna_beta = 2.0 * np.pi * SIONNA_FREQUENCY_HZ / LIGHT_SPEED
    sionna_canonical = sionna_port * np.exp(
        1j * sionna_beta * distance[:, None, None]
    ).astype(np.complex64)

    base_metrics = compute_base_metrics(base, truth, args.batch_size, device)
    base_all = score(base_metrics)
    clear = blocked == 0
    nonzero = np.asarray(base_metrics["energy"] > 0.0)
    base_clear = score(base_metrics, clear)
    base_clear_nonzero = score(base_metrics, clear & nonzero)

    folds = v401.confirmatory_folds(np.arange(len(positions)), positions[:, :2])
    oof_targets = {
        method: np.zeros_like(truth_port, dtype=np.complex64)
        for method in ("oracle",) + DEPLOYABLE_METHODS
    }
    fold_source_rows = []
    fold_query_ids: list[np.ndarray] = []
    for fold_id, outer_ids in enumerate(folds):
        fit_ids = np.setdiff1d(np.arange(len(positions)), outer_ids)
        query_ids = outer_ids[clear[outer_ids]]
        fit_clear = fit_ids[clear[fit_ids] & nonzero[fit_ids]]
        threshold = float(np.quantile(quality[fit_clear], 0.25))
        source_ids = fit_clear[
            (quality[fit_clear] >= threshold)
            & (truth_fraction[fit_clear] >= 0.55)
            & (
                sionna_quality[fit_clear]
                >= np.quantile(sionna_quality[fit_clear], 0.10)
            )
        ]
        if len(source_ids) < 32:
            raise RuntimeError(f"fold {fold_id} has only {len(source_ids)} LOS sources")

        canonical_predictions = {
            "official_knn4": knn_predict(
                positions[:, :2], source_ids, query_ids, truth_canonical, 4
            ),
            "official_knn8": knn_predict(
                positions[:, :2], source_ids, query_ids, truth_canonical, 8
            ),
            "official_knn16": knn_predict(
                positions[:, :2], source_ids, query_ids, truth_canonical, 16
            ),
        }
        affine = fit_complex_affine(
            sionna_canonical[source_ids], truth_canonical[source_ids]
        )
        mapped_all = apply_complex_affine(sionna_canonical, affine)
        canonical_predictions["sionna_affine"] = mapped_all[query_ids]
        residual = truth_canonical - mapped_all
        kernel = mapped_all[query_ids] + kernel_residual_predict(
            positions[:, :2], source_ids, query_ids, residual
        )
        canonical_predictions["sionna_kernel"] = kernel.astype(np.complex64)
        canonical_predictions["hybrid_knn_sionna"] = (
            0.5 * canonical_predictions["official_knn8"] + 0.5 * kernel
        ).astype(np.complex64)

        phase = np.exp(
            -1j * parameters.beta_rad_per_m * distance[query_ids, None, None]
        ).astype(np.complex64)
        oof_targets["oracle"][query_ids] = truth_port[query_ids]
        for method, canonical in canonical_predictions.items():
            oof_targets[method][query_ids] = canonical * phase
        oof_targets["sionna_kernel_port_norm_lock"][query_ids] = lock_port_norm(
            oof_targets["sionna_kernel"][query_ids], base_port[query_ids]
        )
        fold_query_ids.append(query_ids)
        fold_source_rows.append(
            {
                "fold": fold_id,
                "fit_count": int(len(fit_ids)),
                "query_clear_count": int(len(query_ids)),
                "source_count": int(len(source_ids)),
                "quality_threshold": threshold,
                "source_direct_fraction_median": float(
                    np.median(truth_fraction[source_ids])
                ),
            }
        )
        print(
            f"v556 fold={fold_id} sources={len(source_ids)} query_clear={len(query_ids)}",
            flush=True,
        )

    clear_ids = np.flatnonzero(clear).astype(np.int64)
    methods = ("oracle",) + DEPLOYABLE_METHODS
    method_rows = []
    fold_rows = []
    candidate_cache: dict[tuple[str, float], dict[str, np.ndarray]] = {}
    for method in methods:
        for alpha in ALPHAS:
            replacement = candidate_metrics(
                base,
                truth,
                clear_ids,
                base_port,
                oof_targets[method][clear_ids],
                official_delay,
                horizontal_phase,
                vertical_phase,
                receiver_phase,
                alpha,
                args.batch_size,
                device,
            )
            candidate_cache[(method, alpha)] = replacement
            merged = replace_rows(base_metrics, clear_ids, replacement)
            all_score = score(merged)
            clear_score = score(merged, clear)
            clear_nonzero_score = score(merged, clear & nonzero)
            fold_delta = []
            for fold_id, query_ids in enumerate(fold_query_ids):
                local = np.isin(clear_ids, query_ids)
                local_base = {key: base_metrics[key][query_ids] for key in base_metrics}
                local_candidate = {key: replacement[key][local] for key in replacement}
                fold_delta.append(
                    score(local_candidate)["official_score"]
                    - score(local_base)["official_score"]
                )
            method_rows.append(
                {
                    "method": method,
                    "alpha": alpha,
                    "all": all_score,
                    "all_delta": score_delta(all_score, base_all),
                    "clear": clear_score,
                    "clear_delta": score_delta(clear_score, base_clear),
                    "clear_nonzero": clear_nonzero_score,
                    "clear_nonzero_delta": score_delta(
                        clear_nonzero_score, base_clear_nonzero
                    ),
                    "fold_clear_score_delta": fold_delta,
                    "fold_wins": int(np.sum(np.asarray(fold_delta) > 0.0)),
                }
            )
            print(
                f"v556 method={method} alpha={alpha:.2f} "
                f"all_delta={method_rows[-1]['all_delta']['official_score']:+.6f} "
                f"fold_wins={method_rows[-1]['fold_wins']}/5",
                flush=True,
            )

    deployable_rows = [
        row
        for row in method_rows
        if row["method"] in DEPLOYABLE_METHODS and row["fold_wins"] >= 4
    ]
    if not deployable_rows:
        deployable_rows = [
            row for row in method_rows if row["method"] in DEPLOYABLE_METHODS
        ]
    best = max(deployable_rows, key=lambda row: row["all"]["official_score"])

    best_replacement = candidate_cache[(best["method"], best["alpha"])]
    controls = {}
    for name, delay_shift, horizontal_shift in (
        ("wrong_delay_plus_2_bins", 2.0, 0.0),
        ("wrong_horizontal_plus_3_bins", 0.0, 3.0),
    ):
        replacement = candidate_metrics(
            base,
            truth,
            clear_ids,
            base_port,
            oof_targets[best["method"]][clear_ids],
            official_delay,
            horizontal_phase,
            vertical_phase,
            receiver_phase,
            float(best["alpha"]),
            args.batch_size,
            device,
            delay_shift_bins=delay_shift,
            horizontal_shift_bins=horizontal_shift,
        )
        merged = replace_rows(base_metrics, clear_ids, replacement)
        current = score(merged)
        controls[name] = {
            "score": current,
            "delta": score_delta(current, base_all),
            "margin_vs_selected": float(
                best["all"]["official_score"] - current["official_score"]
            ),
        }

    oracle_best = max(
        (row for row in method_rows if row["method"] == "oracle"),
        key=lambda row: row["all"]["official_score"],
    )
    selected_gain = float(best["all_delta"]["official_score"])
    oracle_gain = float(oracle_best["all_delta"]["official_score"])
    control_margin = min(
        float(value["margin_vs_selected"]) for value in controls.values()
    )
    passed = bool(
        selected_gain >= 0.002
        and best["fold_wins"] >= 4
        and control_margin >= 0.001
        and best["all_delta"]["pdp"] >= -0.001
        and best["all_delta"]["nmse_score"] >= -0.001
    )

    np.savez_compressed(
        OOF_CACHE,
        clear_ids=clear_ids,
        truth_fraction=truth_fraction,
        base_fraction=base_fraction,
        sionna_fraction=sionna_fraction,
        source_quality=quality,
        sionna_quality=sionna_quality,
        truth_port=truth_port,
        base_port=base_port,
        selected_target_port=oof_targets[best["method"]],
        selected_method=np.asarray(best["method"]),
        selected_alpha=np.asarray(best["alpha"]),
        selected_traceback_pas=best_replacement["traceback_pas"],
        selected_pdp=best_replacement["pdp"],
        selected_error=best_replacement["error"],
    )
    report = {
        "method": VERSION,
        "status": "analysis_passed_deployment_gate" if passed else "analysis_rejected",
        "primary_hypothesis": (
            "The v534-inferred continuous LOS delay, array steering, and carrier phase "
            "allow the dominant direct component to be transported independently of the "
            "NLOS carrier, improving the exact traceback PAS, PDP, and NMSE on unseen clear points."
        ),
        "new_information": {
            "official_los_signal_chain": "v534 delay/array/carrier inverse",
            "same_coordinate_sionna_los": "v208 high-budget 6.0 GHz direct component",
            "interface": "subtract carrier LOS, transport one rank-one LOS component, preserve all NLOS samples",
        },
        "parameters": {
            "alphas": list(ALPHAS),
            "sionna_frequency_hz": SIONNA_FREQUENCY_HZ,
            "official_beta_rad_per_m": parameters.beta_rad_per_m,
            "delay_offset_cycles": parameters.delay_offset_cycles,
            "folds": "v401 mutually exclusive spatial folds",
            "source_quality_quantile": 0.25,
            "kernel_length_scale_m": 45.0,
            "kernel_ridge": 0.03,
        },
        "population": {
            "all_count": int(len(positions)),
            "clear_count": int(np.sum(clear)),
            "clear_nonzero_count": int(np.sum(clear & nonzero)),
            "fold_sources": fold_source_rows,
        },
        "base": {
            "all": base_all,
            "clear": base_clear,
            "clear_nonzero": base_clear_nonzero,
        },
        "method_rows": method_rows,
        "selected": best,
        "oracle_best": oracle_best,
        "negative_controls": controls,
        "decision_gate": {
            "selected_gain_min": 0.002,
            "fold_wins_min": 4,
            "wrong_geometry_margin_min": 0.001,
            "pdp_delta_min": -0.001,
            "nmse_score_delta_min": -0.001,
            "selected_gain": selected_gain,
            "oracle_gain": oracle_gain,
            "wrong_geometry_margin": control_margin,
            "pass": passed,
        },
        "outputs": {
            "report": PATHS.relative(REPORT),
            "oof_cache": PATHS.relative(OOF_CACHE),
            "submission_zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "base": base_all,
                "selected": best,
                "oracle_best": oracle_best,
                "negative_controls": controls,
                "decision_gate": report["decision_gate"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
