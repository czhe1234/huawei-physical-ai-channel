#!/usr/bin/env python3
"""Directly optimize a realizable complex channel against the online spectral kernel."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

import analyze_v208_strict_historical_replay as v208
import analyze_v479_online_spectral_target_inversion as v479
import analyze_v480_nested_online_pas_pdp_inversion as v480
import analyze_v481_exact_dual_spectral_projection as v481
import train_v482_online_kernel_channel_mixer as v482
import train_v483_fullbatch_online_kernel_mixer as v483
from project_paths import PATHS


VERSION_NUMBER = 485
VERSION = "v485_direct_online_kernel_channel"
CACHE = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
REPORT = REPORT_DIR / f"{VERSION}.json"
OUTPUT = CACHE / "v485_Round1_Test_Channel.npy"
CHECKPOINT = CHECKPOINT_DIR / "direct_online_kernel_channel.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.004)
    parser.add_argument("--trust-penalty", type=float, default=0.0005)
    parser.add_argument("--target-score", type=float, default=0.801)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--seed", type=int, default=485_001)
    return parser.parse_args()


def normalized_channel(
    raw_real: torch.Tensor,
    target_amplitude: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw = torch.view_as_complex(raw_real.contiguous())
    unit = raw / torch.linalg.vector_norm(raw, dim=-1, keepdim=True).clamp_min(1e-20)
    return unit * target_amplitude, unit


def state_cpu(parameter: torch.Tensor) -> torch.Tensor:
    return parameter.detach().cpu().clone()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    source_report = json.loads(v480.REPORT.read_text(encoding="utf-8"))
    selected_kernel = source_report["inherited_full_kernel"]
    pas_weight = float(selected_kernel["pas_weight"])
    alpha_np = np.asarray(source_report["full_fit"]["alpha"], dtype=np.float32)
    intercept = float(source_report["full_fit"]["intercept"])
    device = torch.device(args.device)

    scored_arrays = [np.load(path, mmap_mode="r") for _, _, path in v479.VERSIONS]
    source_channels = v483.load_sources(scored_arrays, device)
    source_pas, source_pdp = v483.source_features(source_channels)
    del source_channels
    torch.cuda.empty_cache()

    initial_np = np.asarray(np.load(v481.OUTPUT, mmap_mode="r"), dtype=np.complex64)
    initial = torch.from_numpy(initial_np).to(device)
    target_amplitude = torch.linalg.vector_norm(initial, dim=-1, keepdim=True).detach()
    initial_unit = initial / target_amplitude.clamp_min(1e-20)
    raw_real = torch.nn.Parameter(torch.view_as_real(initial_unit).clone())
    alpha = torch.from_numpy(alpha_np).to(device)
    optimizer = torch.optim.Adam([raw_real], lr=args.learning_rate)

    history: list[dict[str, float | int]] = []
    best_score = -float("inf")
    best_step = -1
    best_state: torch.Tensor | None = None
    selected_reason = "best_score"
    stale = 0

    for step in range(args.steps + 1):
        candidate, unit = normalized_channel(raw_real, target_amplitude)
        score_without_intercept, _, _ = v483.exact_objective(
            candidate, source_pas, source_pdp, alpha, pas_weight
        )
        trust_cost = torch.mean(torch.sum(torch.abs(unit - initial_unit) ** 2, dim=-1))
        mean_real_coherence = torch.mean(
            torch.real(torch.sum(torch.conj(initial_unit) * unit, dim=-1))
        )
        score = float(score_without_intercept.detach() + intercept)
        if score > best_score + 1e-7:
            best_score = score
            best_step = step
            best_state = state_cpu(raw_real)
            stale = 0
        else:
            stale += 1
        if step == 0 or step % 10 == 0 or score >= args.target_score:
            row = {
                "step": step,
                "predicted_online_score": score,
                "trust_cost": float(trust_cost.detach()),
                "mean_real_coherence_to_v481": float(mean_real_coherence.detach()),
                "best_score": best_score,
            }
            history.append(row)
            print(f"v485 {json.dumps(row)}", flush=True)
        if score >= args.target_score:
            best_score = score
            best_step = step
            best_state = state_cpu(raw_real)
            selected_reason = "first_target_crossing"
            break
        if step == args.steps or stale >= args.patience:
            break
        loss = -score_without_intercept + args.trust_penalty * trust_cost
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    if best_state is None:
        raise RuntimeError("v485 did not produce a checkpoint")
    with torch.no_grad():
        raw_real.copy_(best_state.to(device))
        candidate, unit = normalized_channel(raw_real, target_amplitude)
        final_without_intercept, pas_similarity, pdp_similarity = v483.exact_objective(
            candidate, source_pas, source_pdp, alpha, pas_weight
        )
        final_score = float(final_without_intercept + intercept)
        final_trust = float(
            torch.mean(torch.sum(torch.abs(unit - initial_unit) ** 2, dim=-1))
        )
        final_coherence = float(
            torch.mean(
                torch.real(torch.sum(torch.conj(initial_unit) * unit, dim=-1))
            )
        )
        output = candidate.cpu().numpy().astype(np.complex64)

    CACHE.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT, output)
    torch.save(
        {
            "raw_real": best_state,
            "parameters": vars(args),
            "alpha": alpha_np,
            "intercept": intercept,
            "pas_weight": pas_weight,
            "best_step": best_step,
            "predicted_online_score": final_score,
            "selected_reason": selected_reason,
        },
        CHECKPOINT,
    )
    del source_pas, source_pdp, candidate, unit, initial
    torch.cuda.empty_cache()

    replay = v208.score_prediction(OUTPUT)
    baseline_replay = v208.score_prediction(v481.OUTPUT)
    passed = final_score >= args.target_score
    report = {
        "method": VERSION,
        "status": "candidate_signal_passed" if passed else "analysis_only_no_candidate",
        "primary_hypothesis": (
            "Direct complex-channel optimization can realize the online spectral direction "
            "that alternating PAS/PDP projections and finite source mixtures cannot reach."
        ),
        "parameters": vars(args),
        "history": history,
        "best_step": best_step,
        "selected_reason": selected_reason,
        "predicted_online_score": final_score,
        "trust_cost_to_v481": final_trust,
        "mean_real_coherence_to_v481": final_coherence,
        "pas_similarity_to_versions": pas_similarity.cpu().numpy().astype(float).tolist(),
        "pdp_similarity_to_versions": pdp_similarity.cpu().numpy().astype(float).tolist(),
        "replay": replay,
        "baseline_v481_replay": baseline_replay,
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
                "trust_cost_to_v481": final_trust,
                "mean_real_coherence_to_v481": final_coherence,
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
