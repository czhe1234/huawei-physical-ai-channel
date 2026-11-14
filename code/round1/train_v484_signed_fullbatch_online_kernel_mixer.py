#!/usr/bin/env python3
"""Full-batch signed affine channel mixer for the online spectral kernel."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn

import analyze_v208_strict_historical_replay as v208
import analyze_v479_online_spectral_target_inversion as v479
import analyze_v480_nested_online_pas_pdp_inversion as v480
import train_v469_neural_sionna_calibrator as v469
import train_v482_online_kernel_channel_mixer as v482
import train_v483_fullbatch_online_kernel_mixer as v483
from project_paths import PATHS


VERSION_NUMBER = 484
VERSION = "v484_signed_fullbatch_online_kernel_mixer"
CACHE = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
REPORT = REPORT_DIR / f"{VERSION}.json"
OUTPUT = CACHE / "v484_Round1_Test_Channel.npy"
WEIGHTS = CACHE / "v484_point_mixture_weights.npy"
CHECKPOINT = CHECKPOINT_DIR / "signed_point_channel_mixer.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--residual-cap", type=float, default=0.35)
    parser.add_argument("--residual-penalty", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=484_001)
    parser.add_argument("--patience", type=int, default=100)
    return parser.parse_args()


class SignedPointMixer(nn.Module):
    def __init__(self, cap: float) -> None:
        super().__init__()
        self.cap = cap
        self.raw = nn.Embedding(500, v482.MIX_SOURCES)
        nn.init.zeros_(self.raw.weight)

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = self.cap * torch.tanh(self.raw(ids))
        residual = residual - torch.mean(residual, dim=1, keepdim=True)
        base = torch.zeros_like(residual)
        base[:, -1] = 1.0
        return base + residual, residual


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
    sources = v483.load_sources(arrays, device)
    source_pas, source_pdp = v483.source_features(sources)
    alpha = torch.from_numpy(alpha_np).to(device)
    model = SignedPointMixer(args.residual_cap).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-5
    )
    ids = torch.arange(500, dtype=torch.long, device=device)
    history = []
    best_score = -float("inf")
    best_state = None
    best_step = -1
    stale = 0
    for step in range(args.steps + 1):
        weights, residual = model(ids)
        candidate = v482.mixed_channel(sources, weights)
        score_without_intercept, _, _ = v483.exact_objective(
            candidate, source_pas, source_pdp, alpha, pas_weight
        )
        residual_cost = torch.mean(torch.sum(residual * residual, dim=1))
        loss = -score_without_intercept + args.residual_penalty * residual_cost
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
        if step == 0 or step % 20 == 0:
            row = {
                "step": step,
                "predicted_online_score": score,
                "residual_cost": float(residual_cost.detach()),
                "mean_negative_mass": float(
                    torch.mean(torch.sum(torch.relu(-weights), dim=1)).detach()
                ),
                "weight_min": float(torch.min(weights).detach()),
                "weight_max": float(torch.max(weights).detach()),
                "best_score": best_score,
            }
            history.append(row)
            print(f"v484 {json.dumps(row)}", flush=True)
        if step == args.steps or stale >= args.patience:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    if best_state is None:
        raise RuntimeError("v484 did not produce a checkpoint")
    model.load_state_dict(best_state)
    with torch.no_grad():
        weights, residual = model(ids)
        candidate = v482.mixed_channel(sources, weights)
        final_without_intercept, pas_similarity, pdp_similarity = v483.exact_objective(
            candidate, source_pas, source_pdp, alpha, pas_weight
        )
        final_score = float(final_without_intercept + intercept)
        output = candidate.cpu().numpy().astype(np.complex64)
        mixture_weights = weights.cpu().numpy().astype(np.float32)
        residual_values = residual.cpu().numpy().astype(np.float32)
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
            "Bounded signed affine mixtures can enter the negative-coefficient spectral "
            "direction identified by online inversion while preserving each link energy."
        ),
        "parameters": vars(args),
        "history": history,
        "best_step": best_step,
        "predicted_online_score": final_score,
        "pas_similarity_to_versions": pas_similarity.cpu().numpy().astype(float).tolist(),
        "pdp_similarity_to_versions": pdp_similarity.cpu().numpy().astype(float).tolist(),
        "mixture": {
            "mean_weights": np.mean(mixture_weights, axis=0).tolist(),
            "mean_negative_mass": float(
                np.mean(np.sum(np.maximum(-mixture_weights, 0.0), axis=1))
            ),
            "mean_residual_l2": float(
                np.mean(np.sqrt(np.sum(residual_values * residual_values, axis=1)))
            ),
            "weight_min": float(np.min(mixture_weights)),
            "weight_max": float(np.max(mixture_weights)),
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
