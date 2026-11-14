"""Frozen runtime primitives for path delay field."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from project_paths import PATHS
from .exact_projection import pdp_distribution
from .dataset_compat import load_positions
from .path_link_model import load_model, predict_alpha
from .path_link_core import link_coordinates, project

_RUNTIME = Path("__configured_by_pipeline__")
PATH_CACHE = _RUNTIME / "path_tokens"
FEATURES = _RUNTIME / "features.npy"
REGIME = _RUNTIME / "regime.npy"
RELIABILITY = _RUNTIME / "pdp_reliability.npy"
JOINT_TEST = _RUNTIME / "joint_channel.npy"
ARTIFACT_DIR = _RUNTIME / "path_link"
BASE_PDP = ARTIFACT_DIR / "test_joint_base_pdp_f16.npy"
RAW_PDP = ARTIFACT_DIR / "test_raw_path_pdp_f16.npy"
POINT_CONTEXT = ARTIFACT_DIR / "test_point_context.npy"
FOLD_ALPHA = ARTIFACT_DIR / "test_fold_alpha_f16.npy"
ENSEMBLE_ALPHA = ARTIFACT_DIR / "test_ensemble_alpha_f16.npy"
OUTPUT = ARTIFACT_DIR / "channel.npy"
POLICY = {"alpha_scale": 0.5, "order": "pas_pdp", "cycles": 2}
TINY = 1e-30
FREQUENCIES = np.arange(-96, 96, dtype=np.float32) * 30_000.0


def normalized_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    numerator = np.sum(a * b, axis=-1, dtype=np.float64)
    denominator = np.sqrt(
        np.sum(a * a, axis=-1, dtype=np.float64)
        * np.sum(b * b, axis=-1, dtype=np.float64)
    )
    return numerator / np.where(denominator == 0.0, 1e-12, denominator)


def build_base_pdp(device: torch.device, batch_size: int, force: bool) -> np.ndarray:
    if BASE_PDP.is_file() and (not force):
        return np.load(BASE_PDP, mmap_mode="r")
    base = np.load(JOINT_TEST, mmap_mode="r")
    output = np.lib.format.open_memmap(
        BASE_PDP, mode="w+", dtype=np.float16, shape=base.shape
    )
    for start in range(0, len(base), batch_size):
        end = min(start + batch_size, len(base))
        channel = torch.as_tensor(
            np.array(base[start:end], dtype=np.complex64, copy=True), device=device
        )
        with torch.inference_mode():
            output[start:end] = (
                pdp_distribution(channel).cpu().numpy().astype(np.float16)
            )
        if end % 100 == 0 or end == len(base):
            output.flush()
            print(f"PATH_LINK base PDP {end}/{len(base)}", flush=True)
    output.flush()
    return np.load(BASE_PDP, mmap_mode="r")


def build_raw_path_pdp(
    device: torch.device, batch_size: int, force: bool
) -> np.ndarray:
    if RAW_PDP.is_file() and (not force):
        return np.load(RAW_PDP, mmap_mode="r")
    coefficient = np.load(PATH_CACHE / "coefficient.npy", mmap_mode="r")
    tau = np.load(PATH_CACHE / "tau.npy", mmap_mode="r")
    mask = np.load(PATH_CACHE / "mask.npy", mmap_mode="r")
    output = np.lib.format.open_memmap(
        RAW_PDP, mode="w+", dtype=np.float16, shape=(len(coefficient), 256, 4, 192)
    )
    frequency = torch.as_tensor(FREQUENCIES, device=device, dtype=torch.float32)
    for start in range(0, len(coefficient), batch_size):
        end = min(start + batch_size, len(coefficient))
        coefficient_tensor = torch.as_tensor(
            np.array(coefficient[start:end], dtype=np.complex64, copy=True),
            device=device,
        )
        tau_tensor = torch.as_tensor(
            np.array(tau[start:end], dtype=np.float32, copy=True), device=device
        )
        mask_tensor = torch.as_tensor(np.asarray(mask[start:end]), device=device)
        phase = torch.exp(
            -2j * torch.pi * tau_tensor[:, :, None] * frequency[None, None]
        )
        channel = torch.einsum(
            "bk,bkmu,bks->bmus",
            mask_tensor.to(coefficient_tensor.dtype),
            coefficient_tensor,
            phase,
        )
        with torch.inference_mode():
            output[start:end] = (
                pdp_distribution(channel).cpu().numpy().astype(np.float16)
            )
        if end % 100 == 0 or end == len(coefficient):
            output.flush()
            print(f"PATH_LINK raw path PDP {end}/{len(coefficient)}", flush=True)
    output.flush()
    return np.load(RAW_PDP, mmap_mode="r")


def build_point_context(force: bool) -> np.ndarray:
    if POINT_CONTEXT.is_file() and (not force):
        return np.load(POINT_CONTEXT, mmap_mode="r")
    features = np.load(FEATURES, mmap_mode="r")
    mask = np.asarray(np.load(PATH_CACHE / "mask.npy", mmap_mode="r"), dtype=bool)
    token_mean = np.empty((len(features), features.shape[2]), dtype=np.float32)
    for start in range(0, len(features), 64):
        end = min(start + 64, len(features))
        valid = mask[start:end, :, None].astype(np.float32)
        token_mean[start:end] = (
            np.asarray(features[start:end], dtype=np.float32) * valid
        ).sum(axis=1) / valid.sum(axis=1).clip(min=1.0)
    setup, train_positions, test_positions, train_site, test_site = load_positions()
    train_xy = train_positions[:, :2]
    test_xy = test_positions[:, :2]
    center = train_xy.mean(axis=0)
    scale = train_xy.std(axis=0) + 1e-06
    train_relative = train_xy - setup.bs_positions[train_site, :2]
    test_relative = test_xy - setup.bs_positions[test_site, :2]
    relative_scale = train_relative.std(axis=0) + 1e-06
    bearing = np.arctan2(test_relative[:, 1], test_relative[:, 0])
    regime = np.load(REGIME).astype(np.int64)
    reliability = np.asarray(np.load(RELIABILITY, mmap_mode="r"), dtype=np.float32)
    extra = np.concatenate(
        (
            ((test_xy - center) / scale).astype(np.float32),
            (test_relative / relative_scale).astype(np.float32),
            np.sin(bearing)[:, None].astype(np.float32),
            np.cos(bearing)[:, None].astype(np.float32),
            np.eye(2, dtype=np.float32)[test_site],
            np.eye(4, dtype=np.float32)[regime],
            reliability,
        ),
        axis=1,
    )
    context = np.concatenate((token_mean, extra), axis=1).astype(np.float32)
    np.save(POINT_CONTEXT, context)
    return np.load(POINT_CONTEXT, mmap_mode="r")


def predict_ensemble_alpha(
    base_pdp: np.ndarray,
    raw_pdp: np.ndarray,
    context: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    point_ids = np.arange(len(base_pdp), dtype=np.int64)
    link_tensor = torch.as_tensor(link_coordinates(), device=device)
    fold_prediction = np.empty((5, len(base_pdp), 1024), dtype=np.float32)
    for fold in range(5):
        model, checkpoint = load_model(fold, device)
        fold_prediction[fold] = predict_alpha(
            model,
            point_ids,
            base_pdp,
            raw_pdp,
            context,
            np.asarray(checkpoint["context_mean"], dtype=np.float32),
            np.asarray(checkpoint["context_std"], dtype=np.float32),
            link_tensor,
            device,
            batch_size,
        )
        print(f"PATH_LINK alpha model {fold + 1}/5", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    alpha = np.mean(fold_prediction, axis=0, dtype=np.float32)
    np.save(FOLD_ALPHA, fold_prediction.astype(np.float16))
    np.save(ENSEMBLE_ALPHA, alpha.astype(np.float16))
    return (alpha, np.std(fold_prediction, axis=0, dtype=np.float32))


def generate(
    base_pdp: np.ndarray,
    raw_pdp: np.ndarray,
    alpha: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    base = np.load(JOINT_TEST, mmap_mode="r")
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=base.shape
    )
    change = np.empty(len(base), dtype=np.float64)
    power_ratio = np.empty(len(base), dtype=np.float64)
    pas_cosine = np.empty(len(base), dtype=np.float64)
    pdp_cosine = np.empty(len(base), dtype=np.float64)
    for start in range(0, len(base), batch_size):
        end = min(start + batch_size, len(base))
        reference = torch.as_tensor(
            np.array(base[start:end], dtype=np.complex64, copy=True), device=device
        )
        local_base_pdp = torch.as_tensor(
            np.asarray(base_pdp[start:end], dtype=np.float32), device=device
        )
        local_raw_pdp = torch.as_tensor(
            np.asarray(raw_pdp[start:end], dtype=np.float32), device=device
        )
        local_alpha = torch.clamp(
            POLICY["alpha_scale"]
            * torch.as_tensor(alpha[start:end], dtype=torch.float32, device=device),
            0.0,
            1.0,
        ).reshape(end - start, 256, 4)
        desired = (1.0 - local_alpha[..., None]) * local_base_pdp + local_alpha[
            ..., None
        ] * local_raw_pdp
        desired = desired / desired.sum(dim=3, keepdim=True).clamp_min(TINY)
        with torch.inference_mode():
            candidate = project(
                reference, desired, int(POLICY["cycles"]), str(POLICY["order"])
            )
            reference_power = reference.abs().square().sum(dim=(1, 2, 3))
            candidate_power = candidate.abs().square().sum(dim=(1, 2, 3))
            error = (candidate - reference).abs().square().sum(dim=(1, 2, 3))
            change[start:end] = (error / reference_power.clamp_min(TINY)).cpu().numpy()
            power_ratio[start:end] = (
                (candidate_power / reference_power.clamp_min(TINY)).cpu().numpy()
            )
            reference_pas = (
                torch.fft.fft2(
                    reference.reshape(end - start, 2, 16, 8, 4, 192), dim=(2, 3)
                )
                .abs()
                .square()
            )
            candidate_pas = (
                torch.fft.fft2(
                    candidate.reshape(end - start, 2, 16, 8, 4, 192), dim=(2, 3)
                )
                .abs()
                .square()
            )
            reference_pdp = torch.fft.ifft(reference, dim=-1).abs().square()
            candidate_pdp = torch.fft.ifft(candidate, dim=-1).abs().square()
        pas_cosine[start:end] = normalized_cosine(
            candidate_pas.cpu().numpy().reshape(end - start, -1),
            reference_pas.cpu().numpy().reshape(end - start, -1),
        )
        pdp_cosine[start:end] = np.mean(
            normalized_cosine(candidate_pdp.cpu().numpy(), reference_pdp.cpu().numpy()),
            axis=(1, 2),
        )
        output[start:end] = candidate.cpu().numpy().astype(np.complex64)
        if end % 100 == 0 or end == len(base):
            output.flush()
            print(f"PATH_LINK projection {end}/{len(base)}", flush=True)
    output.flush()
    finite = True
    for start in range(0, len(output), 8):
        block = np.asarray(output[start : start + 8])
        finite &= bool(np.isfinite(block.real).all() and np.isfinite(block.imag).all())
    return {
        "path": PATHS.relative(OUTPUT),
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": finite,
        "bytes": int(OUTPUT.stat().st_size),
        "relative_change_vs_joint_mean": float(np.mean(change)),
        "relative_change_vs_joint_median": float(np.median(change)),
        "relative_change_vs_joint_p90": float(np.quantile(change, 0.9)),
        "power_ratio_mean": float(np.mean(power_ratio)),
        "power_ratio_max_abs_error": float(np.max(np.abs(power_ratio - 1.0))),
        "pas_cosine_vs_joint_mean": float(np.mean(pas_cosine)),
        "pdp_cosine_vs_joint_mean": float(np.mean(pdp_cosine)),
    }
