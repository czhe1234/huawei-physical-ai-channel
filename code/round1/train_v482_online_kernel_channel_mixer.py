#!/usr/bin/env python3
"""Train per-test-point complex-channel mixtures against the online spectral kernel."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import analyze_v208_strict_historical_replay as v208
import analyze_v479_online_spectral_target_inversion as v479
import analyze_v480_nested_online_pas_pdp_inversion as v480
import train_v469_neural_sionna_calibrator as v469
from project_paths import PATHS


VERSION_NUMBER = 482
VERSION = "v482_online_kernel_channel_mixer"
CACHE = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
REPORT = REPORT_DIR / f"{VERSION}.json"
OUTPUT = CACHE / "v482_Round1_Test_Channel.npy"
WEIGHTS = CACHE / "v482_point_mixture_weights.npy"
CHECKPOINT = CHECKPOINT_DIR / "point_channel_mixer.pt"
POINTS = 500
SCORED_SOURCES = len(v479.VERSIONS)
MIX_SOURCES = SCORED_SOURCES + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=482_001)
    return parser.parse_args()


class PointMixer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Embedding(POINTS, MIX_SOURCES)
        nn.init.constant_(self.logits.weight, -4.0)
        with torch.no_grad():
            self.logits.weight[:, -1] = 4.0

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.logits(ids), dim=1)


def torch_metric_features(channel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    value6 = channel.reshape(len(channel), 2, 16, 8, 4, 192)
    pas = torch.abs(torch.fft.fft2(value6, dim=(2, 3))) ** 2
    pas = pas.permute(0, 4, 5, 1, 2, 3).reshape(len(channel), 4, 192, 256)
    pas = pas / torch.linalg.vector_norm(pas, dim=-1, keepdim=True).clamp_min(1e-20)
    pdp = torch.abs(torch.fft.ifft(channel, dim=-1)) ** 2
    pdp = pdp.reshape(len(channel), 1024, 192)
    pdp = pdp / torch.linalg.vector_norm(pdp, dim=-1, keepdim=True).clamp_min(1e-20)
    return pas, pdp


def mixed_channel(
    sources: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    candidate = torch.einsum(
        "bk,kbauf->bauf", weights.to(sources.dtype), sources
    )
    anchor = sources[-1]
    candidate_energy = torch.sum(torch.abs(candidate) ** 2, dim=-1, keepdim=True)
    anchor_energy = torch.sum(torch.abs(anchor) ** 2, dim=-1, keepdim=True)
    scale = torch.sqrt(anchor_energy.clamp_min(1e-30) / candidate_energy.clamp_min(1e-30))
    return candidate * scale


def objective(
    candidate: torch.Tensor,
    scored_sources: torch.Tensor,
    alpha: torch.Tensor,
    pas_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    candidate_pas, candidate_pdp = torch_metric_features(candidate)
    flat = scored_sources.flatten(0, 1)
    source_pas, source_pdp = torch_metric_features(flat)
    source_pas = source_pas.reshape(
        SCORED_SOURCES, len(candidate), 4, 192, 256
    )
    source_pdp = source_pdp.reshape(
        SCORED_SOURCES, len(candidate), 1024, 192
    )
    pas_similarity = torch.mean(
        torch.sum(source_pas * candidate_pas[None], dim=-1), dim=(1, 2, 3)
    )
    pdp_similarity = torch.mean(
        torch.sum(source_pdp * candidate_pdp[None], dim=-1), dim=(1, 2)
    )
    similarity = pas_weight * pas_similarity + (1.0 - pas_weight) * pdp_similarity
    return torch.sum(alpha * similarity), pas_similarity, pdp_similarity


@torch.no_grad()
def generate_output(
    model: PointMixer,
    arrays: list[np.ndarray],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    output = np.lib.format.open_memmap(
        OUTPUT,
        mode="w+",
        dtype=np.complex64,
        shape=(POINTS, 256, 4, 192),
    )
    all_weights = np.empty((POINTS, MIX_SOURCES), dtype=np.float32)
    model.eval()
    for start in range(0, POINTS, batch_size):
        end = min(start + batch_size, POINTS)
        ids = np.arange(start, end, dtype=np.int64)
        ids_tensor = torch.from_numpy(ids).to(device)
        weights = model(ids_tensor)
        sources = torch.from_numpy(
            np.stack(
                [np.asarray(value[ids], dtype=np.complex64) for value in arrays],
                axis=0,
            )
        ).to(device)
        candidate = mixed_channel(sources, weights)
        output[start:end] = candidate.cpu().numpy().astype(np.complex64)
        all_weights[start:end] = weights.cpu().numpy().astype(np.float32)
    output.flush()
    np.save(WEIGHTS, all_weights)
    return all_weights


def main() -> None:
    args = parse_args()
    source_report = json.loads(v480.REPORT.read_text(encoding="utf-8"))
    selected_kernel = source_report["inherited_full_kernel"]
    pas_weight = float(selected_kernel["pas_weight"])
    alpha_np = np.asarray(source_report["full_fit"]["alpha"], dtype=np.float32)
    intercept = float(source_report["full_fit"]["intercept"])
    scored_arrays = [np.load(path, mmap_mode="r") for _, _, path in v479.VERSIONS]
    arrays = [*scored_arrays, np.load(v469.OUTPUT, mmap_mode="r")]
    device = torch.device(args.device)
    alpha = torch.from_numpy(alpha_np).to(device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = PointMixer().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-5
    )
    history = []
    for step in range(1, args.steps + 1):
        ids = rng.choice(POINTS, size=args.batch_size, replace=False).astype(np.int64)
        ids_tensor = torch.from_numpy(ids).to(device)
        weights = model(ids_tensor)
        sources = torch.from_numpy(
            np.stack(
                [np.asarray(value[ids], dtype=np.complex64) for value in arrays],
                axis=0,
            )
        ).to(device)
        candidate = mixed_channel(sources, weights)
        score_without_intercept, _, _ = objective(
            candidate, sources[:SCORED_SOURCES], alpha, pas_weight
        )
        entropy = -torch.mean(torch.sum(weights * torch.log(weights.clamp_min(1e-12)), dim=1))
        loss = -score_without_intercept + 2e-5 * entropy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 200 == 0:
            row = {
                "step": step,
                "batch_predicted_score": float(score_without_intercept.detach() + intercept),
                "entropy": float(entropy.detach()),
                "anchor_weight": float(torch.mean(weights[:, -1]).detach()),
            }
            history.append(row)
            print(f"v482 {json.dumps(row)}", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "parameters": vars(args),
            "alpha": alpha_np,
            "intercept": intercept,
            "pas_weight": pas_weight,
        },
        CHECKPOINT,
    )
    mixture_weights = generate_output(model, arrays, device, args.batch_size)
    candidate = np.load(OUTPUT, mmap_mode="r")
    pas_similarity, pdp_similarity = v480.similarities_to_versions(
        candidate, scored_arrays, batch_size=1
    )
    kernel_similarity = pas_weight * pas_similarity + (1.0 - pas_weight) * pdp_similarity
    predicted_score = float(kernel_similarity @ alpha_np.astype(np.float64) + intercept)
    replay = v208.score_prediction(OUTPUT)
    baseline_replay = v208.score_prediction(v469.OUTPUT)
    passed = predicted_score > float(
        source_report["selected"]["realized_output_predicted_online_score"]
    )
    report = {
        "method": VERSION,
        "status": "candidate_signal_passed" if passed else "analysis_only_no_candidate",
        "primary_hypothesis": (
            "Direct differentiable mixtures of feasible complex channels can maximize the "
            "nested-validated online spectral kernel without incompatible PAS/PDP targets."
        ),
        "parameters": vars(args),
        "history": history,
        "mixture": {
            "mean_weights": np.mean(mixture_weights, axis=0).tolist(),
            "median_anchor_weight": float(np.median(mixture_weights[:, -1])),
            "mean_entropy": float(
                np.mean(-np.sum(mixture_weights * np.log(np.maximum(mixture_weights, 1e-12)), axis=1))
            ),
        },
        "predicted_online_score": predicted_score,
        "pas_similarity_to_versions": pas_similarity.tolist(),
        "pdp_similarity_to_versions": pdp_similarity.tolist(),
        "replay": replay,
        "baseline_v469_replay": baseline_replay,
        "replay_delta_score": float(
            replay["score_traceback_leaked"] - baseline_replay["score_traceback_leaked"]
        ),
        "decision": passed,
        "outputs": {
            "weights": PATHS.relative(WEIGHTS),
            "checkpoint": PATHS.relative(CHECKPOINT),
            "channel": PATHS.relative(OUTPUT),
            "report": PATHS.relative(REPORT),
        },
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "predicted_online_score": predicted_score,
        "replay_delta_score": report["replay_delta_score"],
        "mixture": report["mixture"],
        "passed": passed,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
