#!/usr/bin/env python3
"""Build a train-side carrier with the same neural calibration interface as v469."""

from __future__ import annotations

import json

import numpy as np
import torch

import analyze_v205_sionna_config_calibration as metric
import analyze_v434_sionna_joint_sh_calibration as v434
import analyze_v447_sionna_linkwise_pdp_shape_calibration as v447
import generate_v423_joint_sionna_pas_pdp_diagnostic as v423
import generate_v425_sionna_full_slice_pas_completion_diagnostic as v425
import generate_v426_sionna_linkwise_pdp_completion_diagnostic as v426
import generate_v435_train_calibrated_sionna_joint_sh_diagnostic as v435
import generate_v458_sionna_multimouth_extreme_diagnostic as v458
import train_v469_neural_sionna_calibrator as v469
import train_v471_complex_phase_residual_network as v471
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 473
VERSION = "v473_train_v469like_carrier"
CACHE = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CACHE.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

JOINT_TARGET = CACHE / "v473_train_neural_joint_target.npy"
LINK_TARGET = CACHE / "v473_train_neural_link_target.npy"
OUTPUT = CACHE / "v473_train_v469like_H.npy"
REPORT = REPORT_DIR / f"{VERSION}.json"
BATCH = 2


def require(path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def load_models(device: torch.device, feature_dim: int):
    checkpoint = torch.load(v469.CHECKPOINT, map_location=device, weights_only=True)
    joint = v469.JointCalibrator(feature_dim).to(device)
    link = v469.LinkCalibrator(feature_dim).to(device)
    joint.load_state_dict(checkpoint["joint"])
    link.load_state_dict(checkpoint["link"])
    joint.eval()
    link.eval()
    return joint, link


def predict_joint(
    model: v469.JointCalibrator,
    data: dict[str, np.ndarray],
    features: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    if JOINT_TARGET.is_file():
        return np.load(JOINT_TARGET, mmap_mode="r")
    output = np.lib.format.open_memmap(
        JOINT_TARGET, mode="w+", dtype=np.float32, shape=(2000, 3072)
    )
    with torch.no_grad():
        for start in range(0, 2000, 10):
            end = min(start + 10, 2000)
            prediction = model(
                torch.from_numpy(data["train_raw"][start:end].reshape(-1, 192, 16)).to(device),
                torch.from_numpy(data["train_base"][start:end].reshape(-1, 192, 16)).to(device),
                torch.from_numpy(data["train_cell_quality"][start:end].reshape(-1, 192, 16)).to(device),
                torch.from_numpy(features[start:end]).to(device),
            )
            output[start:end] = prediction.cpu().numpy().reshape(end - start, -1)
        output.flush()
    return np.load(JOINT_TARGET, mmap_mode="r")


def predict_links(
    model: v469.LinkCalibrator,
    features: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    if LINK_TARGET.is_file():
        return np.load(LINK_TARGET, mmap_mode="r")
    source = np.load(v447.SOURCE_CACHE, mmap_mode="r")
    quality = np.load(v447.QUALITY_CACHE, mmap_mode="r")
    output = np.lib.format.open_memmap(
        LINK_TARGET, mode="w+", dtype=np.float16, shape=source.shape
    )
    with torch.no_grad():
        for point in range(len(source)):
            for start in range(0, 1024, 256):
                end = min(start + 256, 1024)
                links = np.arange(start, end, dtype=np.int64)
                point_feature = np.repeat(features[point : point + 1], len(links), axis=0)
                prediction = model(
                    torch.from_numpy(np.asarray(source[point, start:end], dtype=np.float32).copy()).to(device),
                    torch.from_numpy(point_feature).to(device),
                    torch.from_numpy(links).to(device),
                    torch.from_numpy(np.asarray(quality[point, start:end], dtype=np.float32).copy()).to(device),
                )
                output[point, start:end] = prediction.cpu().numpy().astype(np.float16)
            if (point + 1) % 100 == 0:
                output.flush()
                print(f"v473 link target {point + 1}/2000", flush=True)
        output.flush()
    return np.load(LINK_TARGET, mmap_mode="r")


def build_carrier(joint_target: np.ndarray, link_target: np.ndarray) -> np.ndarray:
    if OUTPUT.is_file():
        return np.load(OUTPUT, mmap_mode="r")
    base = np.load(v471.TRAIN_BASE, mmap_mode="r")
    link_quality = np.load(v447.QUALITY_CACHE, mmap_mode="r")
    oof = np.load(v434.OOF_CACHE)
    point_quality = oof["point_quality"].astype(np.float32)
    geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    regimes = regime_of(geometry["train_blocked"].astype(np.float32)).astype("U8")
    profile = next(item for item in v458.JOINT_PROFILES if item.name == "tail085")
    alpha_joint = v458.joint_alpha(profile, regimes, point_quality)
    alpha_link = v458.link_alpha(0.40, regimes, np.asarray(link_quality, dtype=np.float32))
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=base.shape
    )
    for start in range(0, len(base), BATCH):
        end = min(start + BATCH, len(base))
        base_h = np.asarray(base[start:end], dtype=np.complex64)
        base_support, _ = v423.leaked_support(base_h)
        desired_support, _, _ = v435.build_desired_support(
            base_support,
            np.asarray(joint_target[start:end], dtype=np.float32),
            alpha_joint[start:end],
        )
        base_link, _ = v426.linkwise_pdp(base_h)
        target_link = np.asarray(link_target[start:end], dtype=np.float32).reshape(
            end - start, 256, 4, 192
        )
        desired_link = v426.desired_linkwise_pdp(
            base_link,
            target_link,
            alpha_link[start:end].reshape(end - start, 256, 4),
        )
        candidate = base_h.copy()
        for _ in range(v458.JOINT_CYCLES):
            candidate = v425.project_full_support(candidate, desired_support)
            candidate = v426.restore_desired_pdp(candidate, base_h, desired_link)
        output[start:end] = candidate
        if end % 100 == 0:
            output.flush()
            print(f"v473 carrier {end}/2000", flush=True)
    output.flush()
    return np.load(OUTPUT, mmap_mode="r")


def score_carrier(path) -> dict[str, float]:
    prediction = np.load(path, mmap_mode="r")
    truth = np.load(v471.TRAIN_TRUTH, mmap_mode="r")
    pas_values = []
    pdp_values = []
    error = 0.0
    energy = 0.0
    for start in range(0, len(prediction), 4):
        end = min(start + 4, len(prediction))
        pred = np.asarray(prediction[start:end], dtype=np.complex64)
        true = np.asarray(truth[start:end], dtype=np.complex64)
        views = metric.spectral_views(true)
        scores = metric.spectral_scores(pred, views)
        pas_values.extend(scores["pas"].astype(float).tolist())
        pdp_values.extend(scores["pdp"].astype(float).tolist())
        error += float(np.sum(np.abs(pred - true) ** 2, dtype=np.float64))
        energy += float(np.sum(views["energy"], dtype=np.float64))
    nmse = error / max(energy, 1e-30)
    nmse_score = 1.0 / (1.0 + nmse)
    pas = float(np.mean(pas_values))
    pdp = float(np.mean(pdp_values))
    return {
        "pas": pas,
        "pdp": pdp,
        "global_nmse": nmse,
        "nmse_score": nmse_score,
        "official_score": 0.4 * pas + 0.4 * pdp + 0.2 * nmse_score,
    }


def main() -> None:
    for path in (
        v469.CHECKPOINT,
        v469.JOINT_CACHE,
        v469.POINT_FEATURE_CACHE,
        v447.SOURCE_CACHE,
        v447.QUALITY_CACHE,
        v471.TRAIN_BASE,
        v471.TRAIN_TRUTH,
    ):
        require(path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    features = np.load(v469.POINT_FEATURE_CACHE).astype(np.float32)
    data = v469.build_joint_dataset()
    joint_model, link_model = load_models(device, features.shape[1])
    joint_target = predict_joint(joint_model, data, features, device)
    link_target = predict_links(link_model, features, device)
    build_carrier(joint_target, link_target)
    base_score = score_carrier(v471.TRAIN_BASE)
    carrier_score = score_carrier(OUTPUT)
    report = {
        "method": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "Applying the frozen v469 neural calibration interface to the official OOF "
            "carrier creates a deployment-isomorphic train base for residual learning."
        ),
        "base_score": base_score,
        "carrier_score": carrier_score,
        "delta": {
            key: float(carrier_score[key] - base_score[key]) for key in carrier_score
        },
        "outputs": {
            "joint_target": PATHS.relative(JOINT_TARGET),
            "link_target": PATHS.relative(LINK_TARGET),
            "carrier": PATHS.relative(OUTPUT),
            "report": PATHS.relative(REPORT),
        },
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
