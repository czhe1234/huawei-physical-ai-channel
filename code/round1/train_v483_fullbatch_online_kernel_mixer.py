#!/usr/bin/env python3
"""Full-batch H100 optimization of the online-kernel complex-channel mixer."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

import analyze_v208_strict_historical_replay as v208
import analyze_v479_online_spectral_target_inversion as v479
import analyze_v480_nested_online_pas_pdp_inversion as v480
import train_v469_neural_sionna_calibrator as v469
import train_v482_online_kernel_channel_mixer as v482
from project_paths import PATHS


VERSION_NUMBER = 483
VERSION = "v483_fullbatch_online_kernel_mixer"
CACHE = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
REPORT = REPORT_DIR / f"{VERSION}.json"
OUTPUT = CACHE / "v483_Round1_Test_Channel.npy"
WEIGHTS = CACHE / "v483_point_mixture_weights.npy"
CHECKPOINT = CHECKPOINT_DIR / "fullbatch_point_channel_mixer.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=483_001)
    parser.add_argument("--patience", type=int, default=100)
    return parser.parse_args()


def load_sources(arrays: list[np.ndarray], device: torch.device) -> torch.Tensor:
    shape = (len(arrays), 500, 256, 4, 192)
    output = torch.empty(shape, dtype=torch.complex64, device=device)
    for index, value in enumerate(arrays):
        output[index].copy_(
            torch.from_numpy(np.asarray(value, dtype=np.complex64)).to(device)
        )
        print(f"v483 load source {index + 1}/{len(arrays)}", flush=True)
    return output


@torch.no_grad()
def source_features(
    sources: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pas_rows = []
    pdp_rows = []
    for index in range(v482.SCORED_SOURCES):
        pas, pdp = v482.torch_metric_features(sources[index])
        pas_rows.append(pas)
        pdp_rows.append(pdp)
        print(f"v483 source spectrum {index + 1}/{v482.SCORED_SOURCES}", flush=True)
    return torch.stack(pas_rows), torch.stack(pdp_rows)


def exact_objective(
    candidate: torch.Tensor,
    source_pas: torch.Tensor,
    source_pdp: torch.Tensor,
    alpha: torch.Tensor,
    pas_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    candidate_pas, candidate_pdp = v482.torch_metric_features(candidate)
    pas_similarity = torch.mean(
        torch.sum(source_pas * candidate_pas[None], dim=-1), dim=(1, 2, 3)
    )
    pdp_similarity = torch.mean(
        torch.sum(source_pdp * candidate_pdp[None], dim=-1), dim=(1, 2)
    )
    similarity = pas_weight * pas_similarity + (1.0 - pas_weight) * pdp_similarity
    return torch.sum(alpha * similarity), pas_similarity, pdp_similarity


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
    torch.manual_seed(args.seed)
    sources = load_sources(arrays, device)
    source_pas, source_pdp = source_features(sources)
    alpha = torch.from_numpy(alpha_np).to(device)
    model = v482.PointMixer().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-5
    )
    ids = torch.arange(500, dtype=torch.long, device=device)
    history = []
    best_score = -float("inf")
    best_state = None
    best_step = -1
    stale = 0
    for step in range(1, args.steps + 1):
        weights = model(ids)
        candidate = v482.mixed_channel(sources, weights)
        score_without_intercept, _, _ = exact_objective(
            candidate, source_pas, source_pdp, alpha, pas_weight
        )
        entropy = -torch.mean(
            torch.sum(weights * torch.log(weights.clamp_min(1e-12)), dim=1)
        )
        loss = -score_without_intercept + 2e-5 * entropy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        score = float(score_without_intercept.detach() + intercept)
        if score > best_score + 1e-7:
            best_score = score
            best_step = step
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if step == 1 or step % 20 == 0:
            row = {
                "step": step,
                "predicted_online_score": score,
                "entropy": float(entropy.detach()),
                "anchor_weight": float(torch.mean(weights[:, -1]).detach()),
                "best_score": best_score,
            }
            history.append(row)
            print(f"v483 {json.dumps(row)}", flush=True)
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("v483 did not produce a checkpoint")
    model.load_state_dict(best_state)
    with torch.no_grad():
        weights = model(ids)
        candidate = v482.mixed_channel(sources, weights)
        final_without_intercept, pas_similarity, pdp_similarity = exact_objective(
            candidate, source_pas, source_pdp, alpha, pas_weight
        )
        final_score = float(final_without_intercept + intercept)
        output = candidate.cpu().numpy().astype(np.complex64)
        mixture_weights = weights.cpu().numpy().astype(np.float32)
    CACHE.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT, output)
    np.save(WEIGHTS, mixture_weights)
    torch.save(
        {
            "model": best_state,
            "parameters": vars(args),
            "alpha": alpha_np,
            "intercept": intercept,
            "pas_weight": pas_weight,
            "best_step": best_step,
            "best_score": best_score,
        },
        CHECKPOINT,
    )
    del sources, source_pas, source_pdp, candidate
    torch.cuda.empty_cache()
    replay = v208.score_prediction(OUTPUT)
    baseline_replay = v208.score_prediction(v469.OUTPUT)
    passed = final_score > float(
        source_report["selected"]["realized_output_predicted_online_score"]
    )
    report = {
        "method": VERSION,
        "status": "candidate_signal_passed" if passed else "analysis_only_no_candidate",
        "primary_hypothesis": (
            "Full-batch optimization removes stochastic point under-training and directly "
            "maximizes the exact global online spectral-kernel objective over feasible H mixtures."
        ),
        "parameters": vars(args),
        "history": history,
        "best_step": best_step,
        "predicted_online_score": final_score,
        "pas_similarity_to_versions": pas_similarity.cpu().numpy().astype(float).tolist(),
        "pdp_similarity_to_versions": pdp_similarity.cpu().numpy().astype(float).tolist(),
        "mixture": {
            "mean_weights": np.mean(mixture_weights, axis=0).tolist(),
            "median_anchor_weight": float(np.median(mixture_weights[:, -1])),
            "mean_entropy": float(
                np.mean(-np.sum(mixture_weights * np.log(np.maximum(mixture_weights, 1e-12)), axis=1))
            ),
        },
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
        "best_step": best_step,
        "predicted_online_score": final_score,
        "replay_delta_score": report["replay_delta_score"],
        "mixture": report["mixture"],
        "passed": passed,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
