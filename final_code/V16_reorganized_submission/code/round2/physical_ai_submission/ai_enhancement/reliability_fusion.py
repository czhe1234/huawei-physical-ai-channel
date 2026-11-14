from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from project_paths import PATHS
from .angle_residual import normalize_pas, pas_from_slices, pas_slices
from .angle_reliability import percentile_rank
from .angle_consensus import build_consensus
from ..synthesis.spectral_projection import pas_distribution, pdp_distribution
from .angle_projection import project_joint

_RUNTIME = Path("__configured_by_pipeline__")
COMPONENT_LINK_TEST = _RUNTIME / "component_link_channel.npy"
CALIBRATED_PAS = _RUNTIME / "calibrated_pas.npy"
JOINT_ENDPOINT_TEST = _RUNTIME / "joint_pas.npy"
BASE_PHYSICAL_PAS = _RUNTIME / "base_physical_pas.npy"
SCATTER_PHYSICAL_PAS = _RUNTIME / "scatter_physical_pas.npy"
ARTIFACT_DIR = _RUNTIME / "reliable_angle"
RAW_FEATURES = ARTIFACT_DIR / "test_pair_rnorm_raw_features.npy"
GATE_SCORE = ARTIFACT_DIR / "test_pair_rnorm_percentile_score_f16.npy"
GATE = ARTIFACT_DIR / "test_reliable_pas_gate_f16.npy"
OUTPUT = ARTIFACT_DIR / "channel.npy"
POLICY = {
    "gate": "pair_rnorm",
    "quantile": 0.7,
    "alpha": 0.2,
    "order": "pdp_pas",
    "cycles": 4,
}
TINY = 1e-30


def source_arrays() -> tuple[np.ndarray, ...]:
    return tuple(
        (
            np.load(path, mmap_mode="r")
            for path in (
                CALIBRATED_PAS,
                JOINT_ENDPOINT_TEST,
                BASE_PHYSICAL_PAS,
                SCATTER_PHYSICAL_PAS,
            )
        )
    )


def source_amplitudes(
    arrays: tuple[np.ndarray, ...], start: int, end: int, device: torch.device
) -> list[torch.Tensor]:
    values = [np.asarray(array[start:end], dtype=np.float32) for array in arrays]
    values[2] = np.roll(values[2], 2, axis=3)
    values[3] = np.roll(values[3], 2, axis=3)
    output = []
    for value in values:
        tensor = normalize_pas(
            torch.as_tensor(np.array(value, copy=True), device=device)
        )
        output.append(torch.sqrt(pas_slices(tensor).clamp_min(0.0)))
    return output


def base_amplitude(
    base: np.ndarray, start: int, end: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    channel = torch.as_tensor(
        np.array(base[start:end], dtype=np.complex64, copy=True), device=device
    )
    pas = normalize_pas(pas_distribution(channel))
    return (channel, torch.sqrt(pas_slices(pas).clamp_min(0.0)))


def build_gate(
    base: np.ndarray,
    arrays: tuple[np.ndarray, ...],
    device: torch.device,
    batch_size: int,
    force: bool,
) -> np.ndarray:
    if GATE.is_file() and RAW_FEATURES.is_file() and (not force):
        return np.load(GATE, mmap_mode="r")
    raw = np.lib.format.open_memmap(
        RAW_FEATURES, mode="w+", dtype=np.float32, shape=(len(base), 4, 192, 2)
    )
    for start in range(0, len(base), batch_size):
        end = min(start + batch_size, len(base))
        _, local_base = base_amplitude(base, start, end, device)
        sources = source_amplitudes(arrays, start, end, device)
        residual = torch.stack([source - local_base for source in sources])
        residual_norm = residual.square().sum(dim=4).sqrt()
        pair_spread = (
            torch.stack([source.square() for source in sources]).std(dim=0).mean(dim=3)
        )
        raw[start:end, :, :, 0] = pair_spread.cpu().numpy()
        raw[start:end, :, :, 1] = residual_norm.median(dim=0).values.cpu().numpy()
        if end % 100 == 0 or end == len(base):
            raw.flush()
            print(
                f"RELIABLE_ANGLE reliability features {end }/{len (base )}", flush=True
            )
    raw.flush()
    raw_tensor = torch.as_tensor(np.asarray(raw), device=device)
    pair_rank = percentile_rank(raw_tensor[..., 0])
    norm_rank = percentile_rank(raw_tensor[..., 1])
    score = 0.5 * (pair_rank + norm_rank)
    gate = (score <= POLICY["quantile"]).float()
    np.save(GATE_SCORE, score.cpu().numpy().astype(np.float16))
    np.save(GATE, gate.cpu().numpy().astype(np.float16))
    return np.load(GATE, mmap_mode="r")


def normalized_cosine(
    left: torch.Tensor, right: torch.Tensor, dims: tuple[int, ...]
) -> torch.Tensor:
    numerator = (left * right).sum(dim=dims, dtype=torch.float64)
    denominator = torch.sqrt(
        left.square().sum(dim=dims, dtype=torch.float64)
        * right.square().sum(dim=dims, dtype=torch.float64)
    )
    return numerator / torch.where(denominator == 0.0, 1e-12, denominator)


def generate(
    base: np.ndarray,
    arrays: tuple[np.ndarray, ...],
    gate: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=base.shape
    )
    relative_change = np.empty(len(base), dtype=np.float64)
    power_ratio = np.empty(len(base), dtype=np.float64)
    pas_cosine = np.empty(len(base), dtype=np.float64)
    pdp_cosine = np.empty(len(base), dtype=np.float64)
    for start in range(0, len(base), batch_size):
        end = min(start + batch_size, len(base))
        reference, local_base = base_amplitude(base, start, end, device)
        sources = source_amplitudes(arrays, start, end, device)
        consensus = build_consensus(local_base, sources, tuple(range(4)))
        local_gate = torch.as_tensor(
            np.asarray(gate[start:end], dtype=np.float32), device=device
        )
        amplitude = (
            local_base
            + POLICY["alpha"] * local_gate[..., None] * (consensus - local_base)
        ).clamp_min(0.0)
        desired_slices = amplitude.square()
        desired_slices /= desired_slices.sum(dim=3, keepdim=True).clamp_min(TINY)
        desired_pas = pas_from_slices(desired_slices)
        reference_pdp = pdp_distribution(reference)
        with torch.inference_mode():
            candidate = project_joint(
                reference,
                desired_pas,
                reference_pdp,
                int(POLICY["cycles"]),
                str(POLICY["order"]),
            )
            reference_power = reference.abs().square().sum(dim=(1, 2, 3))
            candidate_power = candidate.abs().square().sum(dim=(1, 2, 3))
            relative_change[start:end] = (
                (
                    (candidate - reference).abs().square().sum(dim=(1, 2, 3))
                    / reference_power.clamp_min(TINY)
                )
                .cpu()
                .numpy()
            )
            power_ratio[start:end] = (
                (candidate_power / reference_power.clamp_min(TINY)).cpu().numpy()
            )
            reference_pas = pas_distribution(reference)
            candidate_pas = pas_distribution(candidate)
            candidate_pdp = pdp_distribution(candidate)
            pas_cosine[start:end] = (
                normalized_cosine(candidate_pas, reference_pas, (1, 2, 3))
                .mean(dim=(1, 2))
                .cpu()
                .numpy()
            )
            pdp_cosine[start:end] = (
                normalized_cosine(candidate_pdp, reference_pdp, (3,))
                .mean(dim=(1, 2))
                .cpu()
                .numpy()
            )
        output[start:end] = candidate.cpu().numpy().astype(np.complex64)
        if end % 100 == 0 or end == len(base):
            output.flush()
            print(f"RELIABLE_ANGLE projection {end }/{len (base )}", flush=True)
    output.flush()
    finite = True
    for start in range(0, len(output), 8):
        value = np.asarray(output[start : start + 8])
        finite &= bool(np.isfinite(value.real).all() and np.isfinite(value.imag).all())
    return {
        "path": PATHS.relative(OUTPUT),
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": finite,
        "bytes": int(OUTPUT.stat().st_size),
        "relative_change_vs_component_link_mean": float(np.mean(relative_change)),
        "relative_change_vs_component_link_median": float(np.median(relative_change)),
        "relative_change_vs_component_link_p90": float(np.quantile(relative_change, 0.9)),
        "power_ratio_mean": float(np.mean(power_ratio)),
        "power_ratio_max_abs_error": float(np.max(np.abs(power_ratio - 1.0))),
        "pas_cosine_vs_component_link_mean": float(np.mean(pas_cosine)),
        "pdp_cosine_vs_component_link_mean": float(np.mean(pdp_cosine)),
    }
