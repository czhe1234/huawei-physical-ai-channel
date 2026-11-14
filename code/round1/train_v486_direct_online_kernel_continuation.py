#!/usr/bin/env python3
"""Continue v485 without soft trust regularization until a buffered score target."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

import analyze_v208_strict_historical_replay as v208
import analyze_v479_online_spectral_target_inversion as v479
import analyze_v480_nested_online_pas_pdp_inversion as v480
import analyze_v481_exact_dual_spectral_projection as v481
import train_v483_fullbatch_online_kernel_mixer as v483
import train_v485_direct_online_kernel_channel as v485
from project_paths import PATHS


VERSION_NUMBER = 486
VERSION = "v486_direct_online_kernel_continuation"
CACHE = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
REPORT = REPORT_DIR / f"{VERSION}.json"
OUTPUT = CACHE / "v486_Round1_Test_Channel.npy"
CHECKPOINT = CHECKPOINT_DIR / "direct_online_kernel_continuation.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--target-score", type=float, default=0.805)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--seed", type=int, default=486_001)
    return parser.parse_args()


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
    prior = torch.load(v485.CHECKPOINT, map_location="cpu", weights_only=False)
    raw_real = torch.nn.Parameter(prior["raw_real"].to(device))
    alpha = torch.from_numpy(alpha_np).to(device)
    optimizer = torch.optim.Adam([raw_real], lr=args.learning_rate)

    history: list[dict[str, float | int]] = []
    best_score = -float("inf")
    best_step = -1
    best_state: torch.Tensor | None = None
    selected_reason = "best_score"
    stale = 0

    for step in range(args.steps + 1):
        candidate, unit = v485.normalized_channel(raw_real, target_amplitude)
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
            best_state = v485.state_cpu(raw_real)
            stale = 0
        else:
            stale += 1
        if step == 0 or step % 10 == 0 or score >= args.target_score:
            row = {
                "step": step,
                "predicted_online_score": score,
                "trust_cost_to_v481": float(trust_cost.detach()),
                "mean_real_coherence_to_v481": float(mean_real_coherence.detach()),
                "best_score": best_score,
            }
            history.append(row)
            print(f"v486 {json.dumps(row)}", flush=True)
        if score >= args.target_score:
            best_score = score
            best_step = step
            best_state = v485.state_cpu(raw_real)
            selected_reason = "first_target_crossing"
            break
        if step == args.steps or stale >= args.patience:
            break
        optimizer.zero_grad(set_to_none=True)
        (-score_without_intercept).backward()
        optimizer.step()

    if best_state is None:
        raise RuntimeError("v486 did not produce a checkpoint")
    with torch.no_grad():
        raw_real.copy_(best_state.to(device))
        candidate, unit = v485.normalized_channel(raw_real, target_amplitude)
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
    baseline_replay = v208.score_prediction(v485.OUTPUT)
    passed = final_score >= args.target_score
    report = {
        "method": VERSION,
        "status": "candidate_signal_passed" if passed else "analysis_only_no_candidate",
        "primary_hypothesis": (
            "The still-rising v485 direct-channel trajectory can reach a buffered score "
            "target when the redundant soft trust penalty is replaced by first-crossing stop."
        ),
        "parameters": vars(args),
        "source_v485_predicted_online_score": float(prior["predicted_online_score"]),
        "history": history,
        "best_step": best_step,
        "selected_reason": selected_reason,
        "predicted_online_score": final_score,
        "trust_cost_to_v481": final_trust,
        "mean_real_coherence_to_v481": final_coherence,
        "pas_similarity_to_versions": pas_similarity.cpu().numpy().astype(float).tolist(),
        "pdp_similarity_to_versions": pdp_similarity.cpu().numpy().astype(float).tolist(),
        "replay": replay,
        "baseline_v485_replay": baseline_replay,
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
