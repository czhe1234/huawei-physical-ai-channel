#!/usr/bin/env python3
"""Generate v156 after building a v140-equivalent complex cross-fit base.

v155 selected exact PAS/PDP slice corrections on a v125-equivalent train
complex base and applied them to v140.  Its online failure isolates that base
mismatch as the dominant defect.  This version first projects the fold-isolated
train complex base through test-like v129 support and the exact v140 geodesic
operator.  It then repeats the official-like PAS/PDP audit on that new current
base and only packages a candidate when the gain remains positive in every
diagnostic partition.
"""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import torch

from generate_v112_kernel_moe_path_support import normalize_support
from generate_v121_support_template_manifold import topk_prob
from generate_v140_balanced_hminus2_overdrive import build_risk, row_cos
from generate_v143_transductive_geodesic_risk import apply_risk_shift
from generate_v155_metric_aligned_pas_pdp_projection import (
    DEVICE,
    Config,
    audit_configs,
    build_neighbors,
    output_comparison,
    predict_targets,
    project_metric_slices,
    summarize,
)
from generate_v87_easy_field_projection import (
    BS_POS,
    MH,
    MP,
    MV,
    N_ANT,
    S_SC,
    ad_transform,
    beta_subcarriers,
    inverse_ad,
)
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION = "v156_v140eq_metric_slice_projection"
OUT_DIR = PATHS.cache_dir(156)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(156)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_BASE = OUT_DIR / "train_v140eq_crossfit_base_ad.npy"
TRAIN_RESIDUAL = OUT_DIR / "train_v140eq_crossfit_residual_ad.npy"
TRAIN_TARGET = OUT_DIR / "train_v140eq_crossfit_support_target.npy"
TRAIN_RISK = OUT_DIR / "train_v140eq_crossfit_risk.npy"
OUTPUT = OUT_DIR / "v156_Round1_Test_Channel.npy"
DECISIONS = OUT_DIR / "v156_metric_slice_decisions.npz"
REPORT = REPORT_DIR / "v156_v140eq_metric_slice_projection_generate.json"
ZIP_OUT = PATHS.round1_versions / "v156_v140eq_metric_slice_projection_candidate_pending.zip"

INPUT_BASE = PATHS.shared_features / "train_v125eq_crossfit_base_ad.npy"
INPUT_RESIDUAL = PATHS.shared_features / "train_v125eq_crossfit_residual_ad.npy"
TRAIN_V129_TARGET = PATHS.shared_features / "train_v129_testlike_target.npy"
TRAIN_V129_PROB = PATHS.shared_features / "train_v129_testlike_residual_probs.npy"
V140_OUTPUT = PATHS.cache_dir(140) / "v140_Round1_Test_Channel.npy"

REGIMES = ("clear", "light", "medium", "heavy")
BATCH = 8
GENERATE_BATCH = 10


def file_digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def parse_regime_values(text: str) -> dict[str, float]:
    values = {regime: 0.0 for regime in REGIMES}
    for item in text.split(","):
        key, value = item.split(":")
        values[key.strip()] = float(value)
    return values


def stage_eta(
    weights: np.ndarray,
    regimes: np.ndarray,
    eta_text: str,
    min_gate: float,
    max_gate: float,
    conf_center: float,
    conf_width: float,
    entropy_low: float,
    entropy_high: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    maximum = weights.max(axis=1)
    entropy = -(weights * np.log(np.maximum(weights, 1e-12))).sum(axis=1) / math.log(weights.shape[1])
    confidence = np.clip((maximum - conf_center) / max(conf_width, 1e-6), 0.0, 1.0)
    confidence *= np.clip(
        (entropy_high - entropy) / max(entropy_high - entropy_low, 1e-6), 0.0, 1.0
    )
    base = parse_regime_values(eta_text)
    eta = np.asarray([base[str(regime)] for regime in regimes], dtype=np.float32)
    eta *= (min_gate + (max_gate - min_gate) * confidence).astype(np.float32)
    return eta.astype(np.float32), confidence.astype(np.float32), entropy.astype(np.float32)


def top_mask(collapsed: np.ndarray, count: int) -> np.ndarray:
    flat = collapsed.reshape(len(collapsed), -1)
    k = min(int(count), flat.shape[1])
    indices = np.argpartition(flat, -k, axis=1)[:, -k:]
    mask = np.zeros_like(flat, dtype=bool)
    mask[np.arange(len(flat))[:, None], indices] = True
    return mask.reshape(collapsed.shape)


def support_projection_ad(
    ad: np.ndarray,
    target: np.ndarray,
    eta: np.ndarray,
    gain_clip: tuple[float, float],
    top_gain_clip: tuple[float, float],
    top_components: int,
) -> np.ndarray:
    power = np.abs(ad).astype(np.float32) ** 2
    collapsed = power.sum(axis=(1, 4))
    total = np.maximum(collapsed.sum(axis=(1, 2, 3)), TINY)
    current = collapsed / total[:, None, None, None]
    ratio = ((target + 1e-12) / (current + 1e-12)) ** eta[:, None, None, None]
    gain = np.sqrt(ratio).astype(np.float32)
    gain = np.clip(gain, gain_clip[0], gain_clip[1])
    mask = top_mask(collapsed, top_components)
    gain = np.where(mask, np.clip(gain, top_gain_clip[0], top_gain_clip[1]), gain)
    output = (ad * gain[:, None, :, :, None, :]).astype(np.complex64)
    final_total = np.maximum(np.sum(np.abs(output) ** 2, axis=(1, 2, 3, 4, 5)), TINY)
    output *= np.sqrt(total / final_total)[:, None, None, None, None, None].astype(np.float32)
    return output


def collapsed_support(ad: np.ndarray) -> np.ndarray:
    power = np.abs(ad).astype(np.float32) ** 2
    value = power.sum(axis=(1, 4))
    return (value / np.maximum(value.sum(axis=(1, 2, 3), keepdims=True), TINY)).astype(np.float32)


def build_v140eq_base(
    positions: np.ndarray,
    regimes: np.ndarray,
    geodesic: np.lib.npyio.NpzFile,
    geodesic_names: list[str],
) -> dict[str, object]:
    old_base = np.load(INPUT_BASE, mmap_mode="r")
    old_residual = np.load(INPUT_RESIDUAL, mmap_mode="r")
    v129_target = normalize_support(np.load(TRAIN_V129_TARGET), 1e-12)
    probability = np.load(TRAIN_V129_PROB).astype(np.float32)

    v129_weights = topk_prob(probability, 12).astype(np.float32)
    v129_eta, v129_conf, v129_entropy = stage_eta(
        v129_weights,
        regimes,
        "clear:0.0001,light:0.006,medium:0.720,heavy:1.020",
        0.20,
        3.45,
        0.16,
        0.18,
        0.18,
        0.99,
    )

    risk = build_risk(
        regimes,
        probability,
        geodesic["train_features"].astype(np.float32),
        geodesic_names,
    )
    v140_target, shift_report = apply_risk_shift(v129_target, risk)
    np.save(TRAIN_TARGET, v140_target.astype(np.float32))
    np.save(TRAIN_RISK, risk.astype(np.float32))
    v140_weights = np.empty((len(risk), 8), dtype=np.float32)
    v140_weights[:, 0] = 0.10 + 0.90 * risk
    v140_weights[:, 1:] = (1.0 - v140_weights[:, :1]) / 7.0
    v140_weights /= np.maximum(v140_weights.sum(axis=1, keepdims=True), 1e-12)
    v140_eta, v140_conf, v140_entropy = stage_eta(
        v140_weights,
        regimes,
        "clear:0.0,light:0.028,medium:0.470,heavy:0.780",
        0.12,
        2.95,
        0.22,
        0.38,
        0.02,
        0.90,
    )

    out_base = np.lib.format.open_memmap(
        TRAIN_BASE, mode="w+", dtype=np.complex64, shape=old_base.shape
    )
    out_residual = np.lib.format.open_memmap(
        TRAIN_RESIDUAL, mode="w+", dtype=np.complex64, shape=old_base.shape
    )
    before_cos = []
    after_v129_cos = []
    after_v140_cos = []
    delta_num = 0.0
    delta_den = 0.0
    truth_num = 0.0
    truth_den = 0.0

    for start in range(0, len(positions), BATCH):
        end = min(start + BATCH, len(positions))
        base = np.asarray(old_base[start:end], dtype=np.complex64)
        truth = base + np.asarray(old_residual[start:end], dtype=np.complex64)
        current_support = collapsed_support(base)
        stage_v129 = support_projection_ad(
            base,
            v129_target[start:end],
            v129_eta[start:end],
            (0.30, 4.50),
            (0.40, 3.50),
            3584,
        )
        stage_v140 = support_projection_ad(
            stage_v129,
            v140_target[start:end],
            v140_eta[start:end],
            (0.34, 3.20),
            (0.48, 2.35),
            5632,
        )
        truth_support = collapsed_support(truth)
        before_cos.extend(row_cos(current_support, truth_support).tolist())
        after_v129_cos.extend(row_cos(collapsed_support(stage_v129), truth_support).tolist())
        after_v140_cos.extend(row_cos(collapsed_support(stage_v140), truth_support).tolist())
        out_base[start:end] = stage_v140
        out_residual[start:end] = truth - stage_v140
        out_base.flush()
        out_residual.flush()
        delta_num += float(np.sum(np.abs(stage_v140 - base) ** 2, dtype=np.float64))
        delta_den += float(np.sum(np.abs(base) ** 2, dtype=np.float64))
        truth_num += float(np.sum(np.abs(stage_v140 - truth) ** 2, dtype=np.float64))
        truth_den += float(np.sum(np.abs(truth) ** 2, dtype=np.float64))
        print(f"build v140eq {end}/{len(positions)}", flush=True)

    return {
        "v129_eta": summarize(v129_eta),
        "v129_confidence": summarize(v129_conf),
        "v129_entropy": summarize(v129_entropy),
        "v140_risk": summarize(risk),
        "v140_eta": summarize(v140_eta),
        "v140_confidence": summarize(v140_conf),
        "v140_entropy": summarize(v140_entropy),
        "shift": shift_report,
        "support_cos_to_truth_before": summarize(before_cos),
        "support_cos_to_truth_after_v129": summarize(after_v129_cos),
        "support_cos_to_truth_after_v140": summarize(after_v140_cos),
        "delta_nmse_to_v125eq": float(delta_num / max(delta_den, TINY)),
        "nmse_to_truth": float(truth_num / max(truth_den, TINY)),
    }


def generate_test(
    config: Config,
    positions_train: np.ndarray,
    positions_test: np.ndarray,
    base_mm: np.ndarray,
    residual_mm: np.ndarray,
) -> dict[str, object]:
    indices, distances, weights = build_neighbors(
        positions_train, positions_test, config.k
    )
    v140 = np.load(V140_OUTPUT, mmap_mode="r")
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=v140.shape
    )
    beta_s = beta_subcarriers(140.3, 0.10)
    distance_to_bs = np.linalg.norm(positions_test - BS_POS[None, :], axis=1)
    gain_min = []
    gain_max = []
    for start in range(0, len(positions_test), GENERATE_BATCH):
        end = min(start + GENERATE_BATCH, len(positions_test))
        base_h = np.asarray(v140[start:end], dtype=np.complex64)
        derotated = base_h * np.exp(
            1j * distance_to_bs[start:end, None, None, None] * beta_s
        ).astype(np.complex64)
        base_ad = torch.from_numpy(ad_transform(derotated)).to(DEVICE)
        target_pas, target_pdp = predict_targets(
            indices[start:end], weights[start:end], base_mm, residual_mm
        )
        final_ad, diagnostics = project_metric_slices(
            base_ad, target_pas, target_pdp, config
        )
        result = inverse_ad(final_ad.detach().cpu().numpy()).reshape(
            end - start, 256, N_ANT, S_SC
        )
        result *= np.exp(
            -1j * distance_to_bs[start:end, None, None, None] * beta_s
        ).astype(np.complex64)
        output[start:end] = result.astype(np.complex64)
        output.flush()
        gain_min.extend(diagnostics["gain_min"].tolist())
        gain_max.extend(diagnostics["gain_max"].tolist())
        del base_ad, final_ad, target_pas, target_pdp, result, base_h, derotated
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"generate v156 {end}/{len(positions_test)}", flush=True)
    np.savez_compressed(
        DECISIONS,
        neighbor_indices=indices.astype(np.int32),
        neighbor_distances=distances.astype(np.float32),
        neighbor_weights=weights.astype(np.float32),
    )
    return {
        "neighbor_distance": summarize(distances),
        "farthest_neighbor_distance": summarize(distances[:, -1]),
        "gain_min": summarize(gain_min),
        "gain_max": summarize(gain_max),
    }


def main() -> None:
    positions_train = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(
        np.float32
    )
    positions_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(
        np.float32
    )
    height = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    regimes_train = regime_of(height["train_blocked"].astype(np.float32))
    regimes_test = regime_of(height["test_blocked"].astype(np.float32))
    geodesic = np.load(PATHS.shared_features / "geodesic_support_features.npz")
    geodesic_names = [str(value) for value in geodesic["feature_names"]]

    crossfit = build_v140eq_base(
        positions_train, regimes_train, geodesic, geodesic_names
    )
    base_mm = np.load(TRAIN_BASE, mmap_mode="r")
    residual_mm = np.load(TRAIN_RESIDUAL, mmap_mode="r")
    selected, audit, _ = audit_configs(positions_train, base_mm, residual_mm)
    if float(audit["selected"]["score_gain"]) < 0.003:
        raise RuntimeError(
            f"v140eq exact-metric gain is too small: {audit['selected']['score_gain']:.6f}"
        )
    generation = generate_test(
        selected, positions_train, positions_test, base_mm, residual_mm
    )
    output = np.load(OUTPUT, mmap_mode="r")
    v140 = np.load(V140_OUTPUT, mmap_mode="r")
    comparison = output_comparison(output, v140, regimes_test)
    if not comparison["finite"]:
        raise ValueError("v156 output contains non-finite values")
    if comparison["nmse_to_v140"] > 0.008:
        raise RuntimeError(
            f"v156 change exceeds post-v155 safety bound: {comparison['nmse_to_v140']:.6f}"
        )
    v155 = np.load(
        PATHS.cache_dir(155) / "v155_Round1_Test_Channel.npy", mmap_mode="r"
    )
    comparison_to_v155 = output_comparison(output, v155, regimes_test)
    if comparison_to_v155["nmse_to_v140"] < 1e-10:
        report = {
            "method": VERSION,
            "status": "rejected_local",
            "rejection_reason": (
                "test output is numerically identical to online-failed v155; "
                "changing only the audit base does not change absolute train-truth slice targets"
            ),
            "crossfit_reconstruction": crossfit,
            "selected_config": selected.__dict__,
            "audit": audit,
            "comparison_to_v140": comparison,
            "comparison_to_v155": comparison_to_v155,
            "zip_created": False,
        }
        REPORT.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        ZIP_OUT.unlink(missing_ok=True)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "base": "v140 online 0.501352",
        "primary_hypothesis": (
            "exact PAS/PDP slice correction transfers only when selected on a v140-equivalent complex cross-fit base"
        ),
        "only_primary_change": (
            "replace v155's v125-equivalent audit base with a reconstructed v140-equivalent complex audit base"
        ),
        "device": str(DEVICE),
        "crossfit_reconstruction": crossfit,
        "selected_config": selected.__dict__,
        "audit": audit,
        "test_generation": generation,
        "comparison_to_v140": comparison,
        "comparison_to_v155": comparison_to_v155,
        "outputs": {
            "train_base": PATHS.relative(TRAIN_BASE),
            "train_residual": PATHS.relative(TRAIN_RESIDUAL),
            "train_target": PATHS.relative(TRAIN_TARGET),
            "submission_npy": PATHS.relative(OUTPUT),
            "decisions": PATHS.relative(DECISIONS),
            "report": PATHS.relative(REPORT),
            "zip": PATHS.relative(ZIP_OUT),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    sources = [
        Path(__file__),
        PATHS.code / "generate_v155_metric_aligned_pas_pdp_projection.py",
        PATHS.code / "generate_v140_balanced_hminus2_overdrive.py",
        PATHS.code / "generate_v143_transductive_geodesic_risk.py",
        PATHS.code / "generate_v87_easy_field_projection.py",
        PATHS.code / "powerfield_common.py",
        PATHS.code_root / "project_paths.py",
    ]
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        for source in sources:
            archive.write(source, source.name)
    with zipfile.ZipFile(ZIP_OUT) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP CRC failure: {bad}")
        members = archive.namelist()
    report["integrity"] = {
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": comparison["finite"],
        "output_bytes": int(OUTPUT.stat().st_size),
        "zip_crc": True,
        "zip_members": members,
        "sha256": file_digest(ZIP_OUT, "sha256"),
        "md5": file_digest(ZIP_OUT, "md5"),
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"zip": PATHS.relative(ZIP_OUT), "integrity": report["integrity"]}, indent=2))


if __name__ == "__main__":
    main()
