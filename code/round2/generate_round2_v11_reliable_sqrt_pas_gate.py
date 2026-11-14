#!/usr/bin/env python3
"""Generate Round 2 V11 with reliability-gated square-root PAS consensus."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from analyze_v11_low_rank_path_pas_residual import (
    normalize_pas,
    pas_from_slices,
    pas_slices,
)
from analyze_v11_reliable_sqrt_pas_gate import percentile_rank
from analyze_v11_sqrt_pas_source_consensus import build_consensus
from analyze_v4_exact_joint_projection import pas_distribution, pdp_distribution
from project_paths import PATHS
from train_full_path_pas_slice_gate import project_joint


VERSION = "V11"
METHOD = "reliable_sqrt_pas_consensus"
V10_TEST = (
    PATHS.artifacts
    / "round2"
    / "full_path_pdp_link_gate"
    / "Round2_V10_Full_Path_PDP_Link_Gate_Test_Channel.npy"
)
E40_TEST = (
    PATHS.artifacts
    / "round2"
    / "sionna_official_calibration"
    / "test_pas_mean_ensemble_f16.npy"
)
V9_ENDPOINT_TEST = (
    PATHS.artifacts
    / "round2"
    / "v9_joint_sionna_endpoints"
    / "test_pas_joint_f16.npy"
)
E39_BASE_TEST = (
    PATHS.artifacts
    / "round2"
    / "sionna_multiseed"
    / "primary_d16_base_v4lambda_1m"
    / "full"
    / "checkpoints"
    / "n24_test_pas_mean.npy"
)
E39_SCATTER_TEST = (
    PATHS.artifacts
    / "round2"
    / "sionna_multiseed"
    / "primary_d10_scatter015_v4lambda_1m"
    / "full"
    / "checkpoints"
    / "n24_test_pas_mean.npy"
)
E76_REPORT = (
    PATHS.analysis
    / "results"
    / "round2_reliable_sqrt_pas_gate"
    / "reliable_sqrt_pas_gate.json"
)
ARTIFACT_DIR = PATHS.artifacts / "round2" / "reliable_sqrt_pas_gate"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_reliable_sqrt_pas_gate"
RAW_FEATURES = ARTIFACT_DIR / "test_pair_rnorm_raw_features.npy"
GATE_SCORE = ARTIFACT_DIR / "test_pair_rnorm_percentile_score_f16.npy"
GATE = ARTIFACT_DIR / "test_reliable_pas_gate_f16.npy"
OUTPUT = ARTIFACT_DIR / "Round2_V11_Reliable_Sqrt_PAS_Consensus_Test_Channel.npy"
REPORT = ANALYSIS_DIR / "v11_generate.json"
POLICY = {
    "gate": "pair_rnorm",
    "quantile": 0.70,
    "alpha": 0.20,
    "order": "pdp_pas",
    "cycles": 4,
}
TINY = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def source_arrays() -> tuple[np.ndarray, ...]:
    return tuple(
        np.load(path, mmap_mode="r")
        for path in (E40_TEST, V9_ENDPOINT_TEST, E39_BASE_TEST, E39_SCATTER_TEST)
    )


def source_amplitudes(
    arrays: tuple[np.ndarray, ...],
    start: int,
    end: int,
    device: torch.device,
) -> list[torch.Tensor]:
    values = [
        np.asarray(array[start:end], dtype=np.float32) for array in arrays
    ]
    # E38 identified the scorer alignment needed by the raw 4-lambda sources.
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
    return channel, torch.sqrt(pas_slices(pas).clamp_min(0.0))


def build_gate(
    base: np.ndarray,
    arrays: tuple[np.ndarray, ...],
    device: torch.device,
    batch_size: int,
    force: bool,
) -> np.ndarray:
    if GATE.is_file() and RAW_FEATURES.is_file() and not force:
        return np.load(GATE, mmap_mode="r")
    raw = np.lib.format.open_memmap(
        RAW_FEATURES,
        mode="w+",
        dtype=np.float32,
        shape=(len(base), 4, 192, 2),
    )
    for start in range(0, len(base), batch_size):
        end = min(start + batch_size, len(base))
        _, local_base = base_amplitude(base, start, end, device)
        sources = source_amplitudes(arrays, start, end, device)
        residual = torch.stack([source - local_base for source in sources])
        residual_norm = residual.square().sum(dim=4).sqrt()
        pair_spread = torch.stack([source.square() for source in sources]).std(
            dim=0
        ).mean(dim=3)
        raw[start:end, :, :, 0] = pair_spread.cpu().numpy()
        raw[start:end, :, :, 1] = residual_norm.median(dim=0).values.cpu().numpy()
        if end % 100 == 0 or end == len(base):
            raw.flush()
            print(f"V11 reliability features {end}/{len(base)}", flush=True)
    raw.flush()
    raw_tensor = torch.as_tensor(np.asarray(raw), device=device)
    pair_rank = percentile_rank(raw_tensor[..., 0])
    norm_rank = percentile_rank(raw_tensor[..., 1])
    score = 0.5 * (pair_rank + norm_rank)
    gate = (score <= POLICY["quantile"]).float()
    np.save(GATE_SCORE, score.cpu().numpy().astype(np.float16))
    np.save(GATE, gate.cpu().numpy().astype(np.float16))
    return np.load(GATE, mmap_mode="r")


def normalized_cosine(left: torch.Tensor, right: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
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
            + POLICY["alpha"]
            * local_gate[..., None]
            * (consensus - local_base)
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
                (candidate - reference).abs().square().sum(dim=(1, 2, 3))
                / reference_power.clamp_min(TINY)
            ).cpu().numpy()
            power_ratio[start:end] = (
                candidate_power / reference_power.clamp_min(TINY)
            ).cpu().numpy()
            reference_pas = pas_distribution(reference)
            candidate_pas = pas_distribution(candidate)
            candidate_pdp = pdp_distribution(candidate)
            pas_cosine[start:end] = normalized_cosine(
                candidate_pas, reference_pas, (1, 2, 3)
            ).mean(dim=(1, 2)).cpu().numpy()
            pdp_cosine[start:end] = normalized_cosine(
                candidate_pdp, reference_pdp, (3,)
            ).mean(dim=(1, 2)).cpu().numpy()
        output[start:end] = candidate.cpu().numpy().astype(np.complex64)
        if end % 100 == 0 or end == len(base):
            output.flush()
            print(f"V11 projection {end}/{len(base)}", flush=True)
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
        "relative_change_vs_v10_mean": float(np.mean(relative_change)),
        "relative_change_vs_v10_median": float(np.median(relative_change)),
        "relative_change_vs_v10_p90": float(np.quantile(relative_change, 0.90)),
        "power_ratio_mean": float(np.mean(power_ratio)),
        "power_ratio_max_abs_error": float(np.max(np.abs(power_ratio - 1.0))),
        "pas_cosine_vs_v10_mean": float(np.mean(pas_cosine)),
        "pdp_cosine_vs_v10_mean": float(np.mean(pdp_cosine)),
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    required = (
        V10_TEST,
        E40_TEST,
        V9_ENDPOINT_TEST,
        E39_BASE_TEST,
        E39_SCATTER_TEST,
        E76_REPORT,
    )
    missing = [PATHS.relative(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing V11 inputs: {missing}")
    evidence = json.loads(E76_REPORT.read_text(encoding="utf-8"))
    frozen = {
        key: evidence["selected"][key]
        for key in ("gate", "quantile", "alpha", "order", "cycles")
    }
    if frozen != POLICY or not evidence["deployment_gate"]["passed"]:
        raise RuntimeError(f"E76 policy/evidence mismatch: {frozen}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    base = np.load(V10_TEST, mmap_mode="r")
    arrays = source_arrays()
    gate = build_gate(base, arrays, device, args.batch_size, args.force)
    output = generate(base, arrays, gate, device, args.batch_size)
    score = np.asarray(np.load(GATE_SCORE, mmap_mode="r"), dtype=np.float32)
    payload = {
        "version": VERSION,
        "method": METHOD,
        "status": "candidate_generated_not_packaged",
        "base": PATHS.relative(V10_TEST),
        "policy": POLICY,
        "endpoint_sources": [
            PATHS.relative(path)
            for path in (E40_TEST, V9_ENDPOINT_TEST, E39_BASE_TEST, E39_SCATTER_TEST)
        ],
        "gate": {
            "active_fraction": float(np.mean(gate)),
            "score_mean": float(np.mean(score)),
            "score_median": float(np.median(score)),
            "score_p90": float(np.quantile(score, 0.90)),
        },
        "absolute_sionna_phase_used": False,
        "point_power_preserved": True,
        "output": output,
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
