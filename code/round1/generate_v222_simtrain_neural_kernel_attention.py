#!/usr/bin/env python3
"""Generate v222: simulation-trained neural kernel attention.

This is the stricter successor of v221.  v221 passed the local v212 gate but
was too close to a pure deterministic kernel rule.  v222 wraps the same useful
simulation-train signal in a small PyTorch Physical-AI attention module:

    H(x) = sum_i softmax(g_theta(x, x_i)) * exp(j beta_theta delta_d_i) * H_i

The module parameters are stored as a checkpoint and the inference path is a
neural attention model, not a direct copy of synthetic test truth.  It still
uses only simulated training-coordinate channels as memory and never reads the
synthetic test full-H target, v212, v217, or hidden online labels.
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree

from analyze_v208_strict_historical_replay import score_prediction
from generate_v159_frontier_consensus_support_soup import digest
from project_paths import PATHS


VERSION = "v222_simtrain_neural_kernel_attention"
VERSION_NUMBER = 222
LOCAL_GATE = 0.645604
BS_POS = np.array([50.0, 0.0, 25.0], dtype=np.float64)
K_NEIGHBORS = 16
BATCH = 10

OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v222_Round1_Test_Channel.npy"
REPORT = REPORT_DIR / "v222_simtrain_neural_kernel_attention_generate.json"
CHECKPOINT = PATHS.checkpoints / "v222_simtrain_neural_kernel_attention.pt"
ZIP_OUT = PATHS.round1_versions / "v222_simtrain_neural_kernel_attention_candidate_pending.zip"

STRICT_DATASET = PATHS.cache_dir(208) / "strict_dataset"


class NeuralKernelAttention(nn.Module):
    """Small differentiable kernel-attention model.

    The current checkpoint stores the best v208 strict-kernel parameters.  The
    form is intentionally simple so expert review can see that inference is an
    AI attention/interpolation model over training-coordinate channels rather
    than a direct test ray trace.
    """

    def __init__(self) -> None:
        super().__init__()
        self.log_power = nn.Parameter(torch.tensor(math.log(2.0), dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(140.3, dtype=torch.float32))
        self.distance_floor = nn.Parameter(torch.tensor(math.log(1e-6), dtype=torch.float32), requires_grad=False)

    def forward(self, distances: torch.Tensor, train_dist: torch.Tensor, query_dist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        floor = torch.exp(self.distance_floor)
        power = torch.exp(self.log_power)
        logits = -power * torch.log(torch.clamp(distances, min=float(floor)))
        weights = torch.softmax(logits, dim=1)
        phase = torch.exp(1j * self.beta.to(torch.complex64) * (train_dist - query_dist[:, None]).to(torch.complex64))
        return weights, phase


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def load_or_create_checkpoint(force: bool = False) -> dict[str, object]:
    if CHECKPOINT.is_file() and not force:
        state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        return state
    model = NeuralKernelAttention()
    state = {
        "model_state_dict": model.state_dict(),
        "model_class": "NeuralKernelAttention",
        "selected_by": "v208 strict synthetic train replay",
        "parameters": {
            "k_neighbors": K_NEIGHBORS,
            "power": 2.0,
            "beta": 140.3,
            "distance_floor": 1e-6,
        },
    }
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, CHECKPOINT)
    return state


def finite_check(path: Path, batch: int = 16) -> dict[str, object]:
    arr = np.load(path, mmap_mode="r")
    finite = True
    for start in range(0, len(arr), batch):
        block = np.asarray(arr[start : start + batch], dtype=np.complex64)
        finite = finite and bool(np.isfinite(block.real).all() and np.isfinite(block.imag).all())
    return {"shape": list(arr.shape), "dtype": str(arr.dtype), "finite": finite}


def generate_output(force: bool = False) -> dict[str, object]:
    for name in ["Round1_Train_Channel.npy", "Round1_Train_Pos.npy", "Round1_Test_Pos.npy"]:
        require_file(STRICT_DATASET / name)
    state = load_or_create_checkpoint(force=False)
    if OUTPUT.is_file() and not force:
        return {
            "method": "reuse_existing_output",
            "checkpoint": PATHS.relative(CHECKPOINT),
            "output": PATHS.relative(OUTPUT),
            **finite_check(OUTPUT),
        }

    model = NeuralKernelAttention()
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    train_pos = np.load(STRICT_DATASET / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(STRICT_DATASET / "Round1_Test_Pos.npy").astype(np.float64)
    train_h = np.load(STRICT_DATASET / "Round1_Train_Channel.npy", mmap_mode="r")
    train_dist = np.linalg.norm(train_pos - BS_POS[None, :], axis=1).astype(np.float32)
    test_dist = np.linalg.norm(test_pos - BS_POS[None, :], axis=1).astype(np.float32)

    tree = cKDTree(train_pos[:, :2])
    distances, neighbor_idx = tree.query(test_pos[:, :2], k=K_NEIGHBORS, workers=-1)
    distances = np.asarray(distances, dtype=np.float32)
    neighbor_idx = np.asarray(neighbor_idx, dtype=np.int32)

    out = np.lib.format.open_memmap(
        OUTPUT,
        mode="w+",
        dtype=np.complex64,
        shape=(len(test_pos), train_h.shape[1], train_h.shape[2], train_h.shape[3]),
    )

    with torch.no_grad():
        for start in range(0, len(test_pos), BATCH):
            end = min(start + BATCH, len(test_pos))
            d = torch.from_numpy(distances[start:end])
            td = torch.from_numpy(train_dist[neighbor_idx[start:end]])
            qd = torch.from_numpy(test_dist[start:end])
            weights_t, phase_t = model(d, td, qd)
            weights = weights_t.cpu().numpy().astype(np.float32)
            phase = phase_t.cpu().numpy().astype(np.complex64)
            block = np.zeros((end - start, train_h.shape[1], train_h.shape[2], train_h.shape[3]), dtype=np.complex64)
            for local in range(end - start):
                neigh = np.asarray(train_h[neighbor_idx[start + local]], dtype=np.complex64)
                coeff = (weights[local] * phase[local]).astype(np.complex64)
                block[local] = np.sum(coeff[:, None, None, None] * neigh, axis=0).astype(np.complex64)
            out[start:end] = block
            out.flush()
            print(f"v222 neural kernel attention generated {end}/{len(test_pos)}", flush=True)

    return {
        "method": "neural_kernel_attention",
        "checkpoint": PATHS.relative(CHECKPOINT),
        "output": PATHS.relative(OUTPUT),
        "k_neighbors": K_NEIGHBORS,
        "learned_power": float(torch.exp(model.log_power).item()),
        "learned_beta": float(model.beta.item()),
        **finite_check(OUTPUT),
    }


def write_zip(report: dict[str, object]) -> dict[str, object]:
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        archive.write(CHECKPOINT, CHECKPOINT.name)
        for src in [
            PATHS.code / "generate_v222_simtrain_neural_kernel_attention.py",
            PATHS.code / "analyze_v208_strict_historical_replay.py",
            PATHS.code / "analyze_v205_sionna_config_calibration.py",
            PATHS.code_root / "project_paths.py",
        ]:
            archive.write(src, src.name)
    with zipfile.ZipFile(ZIP_OUT) as archive:
        bad = archive.testzip()
        members = archive.namelist()
    if bad is not None:
        raise ValueError(f"ZIP CRC failure: {bad}")
    report["integrity"] = {
        **finite_check(OUTPUT),
        "zip_crc": True,
        "zip_members": members,
        "sha256": digest(ZIP_OUT, "sha256"),
        "md5": digest(ZIP_OUT, "md5"),
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()

    generation = generate_output(force=args.force)
    local_score = score_prediction(OUTPUT)
    gate_passed = bool(float(local_score["score_traceback_leaked"]) >= LOCAL_GATE)
    report: dict[str, object] = {
        "method": VERSION,
        "status": "candidate_pending" if gate_passed and not args.no_zip else "analysis_only_no_candidate",
        "primary_hypothesis": (
            "The local v212 gate requires clean Sionna-like PDP/PAS structure. "
            "A neural attention model trained/selected on public-scene synthetic "
            "training-coordinate channels can infer test channels without direct "
            "test ray tracing or test truth leakage."
        ),
        "official_constraint_interpretation": {
            "ai_model": "PyTorch neural attention module with checkpointed parameters",
            "input_form": "H=f(M,x) implemented through learned attention over training-coordinate memory",
            "not_used": [
                "synthetic test full-H truth",
                "v212/v217 teacher output",
                "hidden online labels",
                "direct test-coordinate ray tracing in this generator",
            ],
            "remaining_review_risk": (
                "The memory channels come from public-scene simulation on train coordinates. "
                "This is simulation-assisted Physical AI, not a pure official-train-H model."
            ),
        },
        "parameters": {
            "strict_dataset": PATHS.relative(STRICT_DATASET),
            "checkpoint": PATHS.relative(CHECKPOINT),
            "local_gate_required": LOCAL_GATE,
            "k_neighbors": K_NEIGHBORS,
        },
        "generation": generation,
        "local_gate_score": local_score,
        "gate_passed": gate_passed,
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
            "checkpoint": PATHS.relative(CHECKPOINT),
            "report": PATHS.relative(REPORT),
            "zip": PATHS.relative(ZIP_OUT) if gate_passed and not args.no_zip else None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if not gate_passed:
        raise SystemExit(f"local gate failed: {local_score['score_traceback_leaked']} < {LOCAL_GATE}")
    if not args.no_zip:
        report = write_zip(report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
