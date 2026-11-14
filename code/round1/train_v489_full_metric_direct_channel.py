#!/usr/bin/env python3
"""Optimize complex direction and bounded link power against the full online metric kernel."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

import analyze_v208_strict_historical_replay as v208
import analyze_v479_online_spectral_target_inversion as v479
import analyze_v488_full_metric_online_kernel as v488
import train_v482_online_kernel_channel_mixer as v482
import train_v483_fullbatch_online_kernel_mixer as v483
import train_v486_direct_online_kernel_continuation as v486
from project_paths import PATHS


VERSION_NUMBER = 489
VERSION = "v489_full_metric_direct_channel"
CACHE = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
REPORT = REPORT_DIR / f"{VERSION}.json"
OUTPUT = CACHE / "v489_Round1_Test_Channel.npy"
CHECKPOINT = CHECKPOINT_DIR / "full_metric_direct_channel.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--channel-learning-rate", type=float, default=0.002)
    parser.add_argument("--power-learning-rate", type=float, default=0.01)
    parser.add_argument("--log-power-cap", type=float, default=0.35)
    parser.add_argument("--target-score", type=float, default=0.805)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--seed", type=int, default=489_001)
    return parser.parse_args()


def build_candidate(
    raw_real: torch.Tensor,
    raw_log_scale: torch.Tensor,
    initial_amplitude: torch.Tensor,
    log_power_cap: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = torch.view_as_complex(raw_real.contiguous())
    unit = raw / torch.linalg.vector_norm(raw, dim=-1, keepdim=True).clamp_min(1e-20)
    link_scale = torch.exp(log_power_cap * torch.tanh(raw_log_scale))
    return unit * initial_amplitude * link_scale[..., None], unit, link_scale


def full_metric_objective(
    candidate: torch.Tensor,
    source_pas: torch.Tensor,
    source_pdp: torch.Tensor,
    source_flat: torch.Tensor,
    source_energy: torch.Tensor,
    alpha: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    candidate_pas, candidate_pdp = v482.torch_metric_features(candidate)
    pas_similarity = torch.mean(
        torch.sum(source_pas * candidate_pas[None], dim=-1), dim=(1, 2, 3)
    )
    pdp_similarity = torch.mean(
        torch.sum(source_pdp * candidate_pdp[None], dim=-1), dim=(1, 2)
    )
    candidate_flat = candidate.reshape(-1)
    candidate_energy = torch.sum(torch.abs(candidate_flat) ** 2)
    cross = source_flat @ torch.conj(candidate_flat)
    error = (
        candidate_energy + source_energy - 2.0 * torch.real(cross)
    ).clamp_min(0.0)
    nmse_similarity = 1.0 / (1.0 + error / source_energy.clamp_min(1e-30))
    similarity = 0.4 * pas_similarity + 0.4 * pdp_similarity + 0.2 * nmse_similarity
    return torch.sum(alpha * similarity), pas_similarity, pdp_similarity, nmse_similarity


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    full_report = json.loads(v488.REPORT.read_text(encoding="utf-8"))
    full_kernel = np.asarray(full_report["kernels"]["full_metric"], dtype=np.float64)[
        :11, :11
    ]
    scores = np.asarray([score for _, score, _ in v479.VERSIONS], dtype=np.float64)
    best = None
    for ridge in v488.RIDGES:
        prediction = v479.loo_predictions(full_kernel, scores, float(ridge))
        rmse = float(np.sqrt(np.mean((prediction - scores) ** 2)))
        if best is None or rmse < best[0]:
            best = (rmse, float(ridge), prediction)
    assert best is not None
    loo_rmse, ridge, loo_prediction = best
    alpha_np, intercept = v479.kernel_fit(full_kernel, scores, ridge)

    device = torch.device(args.device)
    arrays = [np.load(path, mmap_mode="r") for _, _, path in v479.VERSIONS]
    sources = v483.load_sources(arrays, device)
    source_pas, source_pdp = v483.source_features(sources)
    source_flat = sources.reshape(len(arrays), -1)
    source_energy = torch.sum(torch.abs(source_flat) ** 2, dim=1)
    alpha = torch.from_numpy(alpha_np.astype(np.float32)).to(device)

    initial_np = np.asarray(np.load(v486.OUTPUT, mmap_mode="r"), dtype=np.complex64)
    initial = torch.from_numpy(initial_np).to(device)
    initial_amplitude = torch.linalg.vector_norm(initial, dim=-1, keepdim=True).detach()
    initial_unit = initial / initial_amplitude.clamp_min(1e-20)
    raw_real = torch.nn.Parameter(torch.view_as_real(initial_unit).clone())
    raw_log_scale = torch.nn.Parameter(
        torch.zeros(initial.shape[:-1], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(
        [
            {"params": [raw_real], "lr": args.channel_learning_rate},
            {"params": [raw_log_scale], "lr": args.power_learning_rate},
        ]
    )

    history: list[dict[str, float | int]] = []
    best_score = -float("inf")
    best_step = -1
    best_channel_state: torch.Tensor | None = None
    best_power_state: torch.Tensor | None = None
    selected_reason = "best_score"
    stale = 0

    for step in range(args.steps + 1):
        candidate, unit, link_scale = build_candidate(
            raw_real, raw_log_scale, initial_amplitude, args.log_power_cap
        )
        score_without_intercept, pas_similarity, pdp_similarity, nmse_similarity = (
            full_metric_objective(
                candidate,
                source_pas,
                source_pdp,
                source_flat,
                source_energy,
                alpha,
            )
        )
        score = float(score_without_intercept.detach() + intercept)
        mean_coherence = torch.mean(
            torch.real(torch.sum(torch.conj(initial_unit) * unit, dim=-1))
        )
        if score > best_score + 1e-7:
            best_score = score
            best_step = step
            best_channel_state = raw_real.detach().cpu().clone()
            best_power_state = raw_log_scale.detach().cpu().clone()
            stale = 0
        else:
            stale += 1
        if step == 0 or step % 10 == 0 or score >= args.target_score:
            row = {
                "step": step,
                "predicted_online_score": score,
                "mean_real_coherence_to_v486": float(mean_coherence.detach()),
                "mean_link_scale": float(torch.mean(link_scale).detach()),
                "minimum_link_scale": float(torch.min(link_scale).detach()),
                "maximum_link_scale": float(torch.max(link_scale).detach()),
                "best_score": best_score,
            }
            history.append(row)
            print(f"v489 {json.dumps(row)}", flush=True)
        if score >= args.target_score:
            best_score = score
            best_step = step
            best_channel_state = raw_real.detach().cpu().clone()
            best_power_state = raw_log_scale.detach().cpu().clone()
            selected_reason = "first_target_crossing"
            break
        if step == args.steps or stale >= args.patience:
            break
        optimizer.zero_grad(set_to_none=True)
        (-score_without_intercept).backward()
        optimizer.step()

    if best_channel_state is None or best_power_state is None:
        raise RuntimeError("v489 did not produce a checkpoint")
    with torch.no_grad():
        raw_real.copy_(best_channel_state.to(device))
        raw_log_scale.copy_(best_power_state.to(device))
        candidate, unit, link_scale = build_candidate(
            raw_real, raw_log_scale, initial_amplitude, args.log_power_cap
        )
        final_without_intercept, pas_similarity, pdp_similarity, nmse_similarity = (
            full_metric_objective(
                candidate,
                source_pas,
                source_pdp,
                source_flat,
                source_energy,
                alpha,
            )
        )
        final_score = float(final_without_intercept + intercept)
        final_coherence = float(
            torch.mean(
                torch.real(torch.sum(torch.conj(initial_unit) * unit, dim=-1))
            )
        )
        scale_np = link_scale.cpu().numpy().astype(np.float32)
        output = candidate.cpu().numpy().astype(np.complex64)

    CACHE.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT, output)
    torch.save(
        {
            "raw_real": best_channel_state,
            "raw_log_scale": best_power_state,
            "parameters": vars(args),
            "alpha": alpha_np,
            "intercept": intercept,
            "ridge": ridge,
            "best_step": best_step,
            "predicted_online_score": final_score,
            "selected_reason": selected_reason,
        },
        CHECKPOINT,
    )
    del sources, source_pas, source_pdp, source_flat, candidate, unit, initial
    torch.cuda.empty_cache()

    replay = v208.score_prediction(OUTPUT)
    baseline_replay = v208.score_prediction(v486.OUTPUT)
    passed = final_score >= args.target_score
    report = {
        "method": VERSION,
        "status": "candidate_signal_passed" if passed else "analysis_only_no_candidate",
        "primary_hypothesis": (
            "The complete 0.4 PAS + 0.4 PDP + 0.2 NMSE online kernel can optimize a "
            "balanced realizable H beyond the spectral-only plateau."
        ),
        "parameters": vars(args),
        "kernel_validation": {
            "ridge": ridge,
            "loo_rmse": loo_rmse,
            "loo_predictions": loo_prediction.tolist(),
            "alpha": alpha_np.tolist(),
            "intercept": intercept,
        },
        "history": history,
        "best_step": best_step,
        "selected_reason": selected_reason,
        "predicted_online_score": final_score,
        "mean_real_coherence_to_v486": final_coherence,
        "link_scale": {
            "mean": float(np.mean(scale_np)),
            "minimum": float(np.min(scale_np)),
            "maximum": float(np.max(scale_np)),
            "p01": float(np.quantile(scale_np, 0.01)),
            "p99": float(np.quantile(scale_np, 0.99)),
        },
        "pas_similarity_to_versions": pas_similarity.cpu().numpy().astype(float).tolist(),
        "pdp_similarity_to_versions": pdp_similarity.cpu().numpy().astype(float).tolist(),
        "nmse_score_similarity_to_versions": nmse_similarity.cpu().numpy().astype(float).tolist(),
        "replay": replay,
        "baseline_v486_replay": baseline_replay,
        "replay_delta_score": float(
            replay["score_traceback_leaked"] - baseline_replay["score_traceback_leaked"]
        ),
        "decision": passed,
        "outputs": {
            "checkpoint": PATHS.relative(CHECKPOINT),
            "channel": PATHS.relative(OUTPUT),
            "report": PATHS.relative(REPORT),
        },
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "best_step": best_step,
                "selected_reason": selected_reason,
                "predicted_online_score": final_score,
                "mean_real_coherence_to_v486": final_coherence,
                "link_scale": report["link_scale"],
                "replay_delta_score": report["replay_delta_score"],
                "passed": passed,
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
