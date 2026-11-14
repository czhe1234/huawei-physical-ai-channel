#!/usr/bin/env python3
"""Transfer the v445 same-scene path model to aligned official-coordinate paths."""

from __future__ import annotations

import argparse
import json
import math
import numbers

import numpy as np
import torch
import torch.nn.functional as F

import analyze_v208_strict_historical_replay as v208
import build_v415_route_raster_features as route_builder
import build_v439_component_path_supervision as component_builder
import build_v445_geometry_ordered_path_tokens as token_builder
import generate_v423_joint_sionna_pas_pdp_diagnostic as v423
import generate_v425_sionna_full_slice_pas_completion_diagnostic as v425
import generate_v426_sionna_linkwise_pdp_completion_diagnostic as v426
import generate_v435_train_calibrated_sionna_joint_sh_diagnostic as v435
import generate_v458_sionna_multimouth_extreme_diagnostic as v458
import train_v439_component_route_long_model as v439
import train_v445_candidate_path_token_model as v445
import train_v469_neural_sionna_calibrator as v469
from project_paths import PATHS


VERSION_NUMBER = 477
VERSION = "v477_aligned_pretrained_path_transfer"
CACHE = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
ROUTE = CACHE / "v477_route_raster.npy"
COMPONENT = CACHE / "v477_component_features.npy"
PATH_FEATURE = CACHE / "v477_path_candidate_feature.npy"
PATH_VALID = CACHE / "v477_path_candidate_valid.npy"
V443_JOINT = CACHE / "v477_v443_joint.npy"
V445_JOINT = CACHE / "v477_v445_joint.npy"
SHUFFLED_JOINT = CACHE / "v477_v445_shuffled_joint.npy"
BLENDED_JOINT = CACHE / "v477_blended_joint_target.npy"
OUTPUT = CACHE / "v477_Round1_Test_Channel.npy"
COUNT = 2500
TRAIN_COUNT = 2000
MAX_PATHS = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=477_001)
    return parser.parse_args()


def require(path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def build_route() -> np.ndarray:
    if ROUTE.is_file():
        return np.load(ROUTE, mmap_mode="r")
    positions = np.load(PATHS.cache_dir(475) / "positions.npz")["positions"].astype(np.float32)
    scene = np.load(PATHS.shared_features / "official_pointcloud_scene_cache.npz")
    route_builder.BATCH = 128
    value = route_builder.build_raster(positions, scene)
    if value.shape != (COUNT, 12, 160, 41) or not np.isfinite(value).all():
        raise ValueError(f"Unexpected route raster {value.shape}")
    np.save(ROUTE, value.astype(np.float16))
    return np.load(ROUTE, mmap_mode="r")


def build_component() -> np.ndarray:
    if COMPONENT.is_file():
        return np.load(COMPONENT, mmap_mode="r")
    positions = np.load(PATHS.cache_dir(475) / "positions.npz")["positions"].astype(np.float32)
    scene = np.load(PATHS.shared_features / "official_pointcloud_scene_cache.npz")
    geometry = component_builder.component_geometry(scene)
    value = component_builder.build_component_features(positions, geometry)
    if value.shape != (COUNT, 80, 32) or not np.isfinite(value).all():
        raise ValueError(f"Unexpected component features {value.shape}")
    np.save(COMPONENT, value.astype(np.float16))
    return np.load(COMPONENT, mmap_mode="r")


def build_path_tokens() -> tuple[np.ndarray, np.ndarray]:
    if PATH_FEATURE.is_file() and PATH_VALID.is_file():
        return np.load(PATH_FEATURE, mmap_mode="r"), np.load(PATH_VALID, mmap_mode="r")
    positions = np.load(PATHS.cache_dir(475) / "positions.npz")["positions"].astype(np.float32)
    features = np.lib.format.open_memmap(
        PATH_FEATURE.with_name(PATH_FEATURE.stem + ".partial.npy"),
        mode="w+",
        dtype=np.float16,
        shape=(COUNT, MAX_PATHS, token_builder.FEATURES),
    )
    valid_out = np.lib.format.open_memmap(
        PATH_VALID.with_name(PATH_VALID.stem + ".partial.npy"),
        mode="w+",
        dtype=np.bool_,
        shape=(COUNT, MAX_PATHS),
    )
    features[:] = 0
    valid_out[:] = False
    offset = 0
    source = PATHS.cache_dir(475) / "shards"
    for shard in sorted(source.glob("shard_*/path_targets.npz")):
        data = np.load(shard)
        metadata = np.load(shard.parent / "metadata.npz")
        ids = metadata["position_id"].astype(np.int64)
        interactions = data["interactions"].astype(np.int8)
        first, last = token_builder.first_last_vertices(
            data["vertices"].astype(np.float32), interactions
        )
        for local, global_id in enumerate(ids):
            order = token_builder.geometry_order(data, local)
            count = len(order)
            if not count:
                continue
            tau = data["tau"][local, order].astype(np.float32)
            delay = tau * token_builder.DELAY_SCALE
            length = tau * token_builder.LIGHT_SPEED
            ue = positions[global_id]
            direct = float(np.linalg.norm(ue - token_builder.BS))
            depth = data["interaction_count"][local, order].astype(np.float32)
            mask = data["interaction_mask"][local, order].astype(np.int16)
            angles = [
                data[name][local, order].astype(np.float32)
                for name in ("theta_t", "phi_t", "theta_r", "phi_r")
            ]
            angle_feature = np.column_stack(
                [value for angle in angles for value in (np.sin(angle), np.cos(angle))]
            )
            first_value = first[local, order]
            last_value = last[local, order]
            global_vertex = np.column_stack(
                [
                    (first_value[:, 0] - 150.0) / 150.0,
                    (first_value[:, 1] + 50.0) / 200.0,
                    first_value[:, 2] / 50.0,
                    (last_value[:, 0] - 150.0) / 150.0,
                    (last_value[:, 1] + 50.0) / 200.0,
                    last_value[:, 2] / 50.0,
                ]
            )
            relative_vertex = np.column_stack(
                [
                    (first_value[:, 0] - ue[0]) / 300.0,
                    (first_value[:, 1] - ue[1]) / 300.0,
                    (last_value[:, 0] - ue[0]) / 300.0,
                    (last_value[:, 1] - ue[1]) / 300.0,
                ]
            )
            current = np.column_stack(
                [
                    tau * 1e6 / 8.0,
                    delay / 48.0,
                    length / 600.0,
                    np.maximum(length - direct, 0.0) / 500.0,
                    depth / 16.0,
                    (depth == 0).astype(np.float32),
                    *[((mask & bit) != 0).astype(np.float32) for bit in (1, 2, 4, 8)],
                    angle_feature,
                    global_vertex,
                    relative_vertex,
                ]
            ).astype(np.float32)
            if current.shape != (count, token_builder.FEATURES):
                raise ValueError(current.shape)
            features[global_id, :count] = np.clip(current, -4.0, 4.0).astype(np.float16)
            valid_out[global_id, :count] = True
        offset += len(ids)
    if offset != COUNT:
        raise ValueError(f"Expected {COUNT} rows, found {offset}")
    features.flush()
    valid_out.flush()
    del features, valid_out
    PATH_FEATURE.with_name(PATH_FEATURE.stem + ".partial.npy").replace(PATH_FEATURE)
    PATH_VALID.with_name(PATH_VALID.stem + ".partial.npy").replace(PATH_VALID)
    return np.load(PATH_FEATURE, mmap_mode="r"), np.load(PATH_VALID, mmap_mode="r")


def position_features(positions: np.ndarray) -> np.ndarray:
    xy = positions[:, :2].astype(np.float32)
    delta = xy - np.asarray([50.0, 0.0], dtype=np.float32)[None, :]
    distance = np.maximum(np.linalg.norm(delta, axis=1), 1e-4)
    angle = np.arctan2(delta[:, 1], delta[:, 0])
    base = np.column_stack(
        [
            (xy[:, 0] - 150.0) / 120.0,
            (xy[:, 1] + 50.0) / 160.0,
            delta[:, 0] / 250.0,
            delta[:, 1] / 350.0,
            distance / 350.0,
            np.log1p(distance) / 6.0,
            np.sin(angle),
            np.cos(angle),
        ]
    )
    harmonics = []
    for scale in (5.0, 10.0, 20.0, 40.0, 80.0, 160.0):
        phase_x = 2.0 * np.pi * xy[:, 0] / scale
        phase_y = 2.0 * np.pi * xy[:, 1] / scale
        harmonics.extend(
            [np.sin(phase_x), np.cos(phase_x), np.sin(phase_y), np.cos(phase_y)]
        )
    return np.column_stack([base, *harmonics]).astype(np.float32)


@torch.no_grad()
def infer_joint(
    model,
    output_path,
    route: np.ndarray,
    component: np.ndarray,
    path_feature: np.ndarray,
    path_valid: np.ndarray,
    position: np.ndarray,
    device: torch.device,
    batch_size: int,
    *,
    path_geometry_ids: np.ndarray | None = None,
) -> np.ndarray:
    if output_path.is_file():
        cached = np.load(output_path, mmap_mode="r")
        if cached.shape != (COUNT, 192, 16) or cached.dtype != np.float32:
            raise ValueError(f"Invalid cached inference {output_path}: {cached.shape}/{cached.dtype}")
        return cached
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float32, shape=(COUNT, 192, 16)
    )
    model.eval()
    for start in range(0, COUNT, batch_size):
        end = min(start + batch_size, COUNT)
        geometry = (
            np.arange(start, end, dtype=np.int64)
            if path_geometry_ids is None
            else path_geometry_ids[start:end]
        )
        route_tensor = torch.from_numpy(np.asarray(route[start:end], dtype=np.float32)).to(device)
        position_tensor = torch.from_numpy(position[start:end]).to(device)
        component_tensor = torch.from_numpy(
            np.asarray(component[start:end], dtype=np.float32)
        ).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if getattr(model, "uses_path_candidates", False):
                prediction = model(
                    route_tensor,
                    position_tensor,
                    component_tensor,
                    path_candidate_feature=torch.from_numpy(
                        np.asarray(path_feature[geometry], dtype=np.float32)
                    ).to(device),
                    path_candidate_valid=torch.from_numpy(
                        np.asarray(path_valid[geometry], dtype=np.bool_)
                    ).to(device),
                )
            else:
                prediction = model(route_tensor, position_tensor, component_tensor)
        output[start:end] = (
            F.softmax(prediction["joint_logits"].flatten(1), dim=1)
            .reshape(-1, 192, 16)
            .float()
            .cpu()
            .numpy()
        )
        if end % 200 == 0 or end == COUNT:
            print(f"v477 inference {output_path.name} {end}/{COUNT}", flush=True)
    output.flush()
    del output
    return np.load(output_path, mmap_mode="r")


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left.reshape(len(left), -1).astype(np.float64)
    right = right.reshape(len(right), -1).astype(np.float64)
    return np.sum(left * right, axis=1) / np.maximum(
        np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1), 1e-12
    )


def derived_pdp(joint: np.ndarray) -> np.ndarray:
    reverse = np.concatenate([[0], np.arange(191, 0, -1)])
    return np.sum(joint[:, reverse], axis=2)


def geometric_blend(anchor: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    logits = (1.0 - alpha) * np.log(np.maximum(anchor, 1e-12))
    logits += alpha * np.log(np.maximum(target, 1e-12))
    logits -= np.max(logits, axis=(1, 2), keepdims=True)
    value = np.exp(logits)
    return (value / np.maximum(np.sum(value, axis=(1, 2), keepdims=True), 1e-12)).astype(np.float32)


def generate_channel(joint_target: np.ndarray) -> None:
    source = np.load(v469.BASE, mmap_mode="r")
    link_target = np.load(v469.LINK_PRED, mmap_mode="r")
    link_quality = np.load(v469.TEST_LINK_QUALITY, mmap_mode="r")
    power_delta = np.load(v469.POWER_PRED).astype(np.float32)
    point_actions = np.load(v469.V435_ACTIONS)
    regimes = point_actions["regimes"].astype("U8")
    point_quality = point_actions["point_quality"].astype(np.float32)
    profile = next(item for item in v458.JOINT_PROFILES if item.name == "tail085")
    alpha_joint = v458.joint_alpha(profile, regimes, point_quality)
    alpha_link = v458.link_alpha(0.40, regimes, np.asarray(link_quality, dtype=np.float32))
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=source.shape
    )
    for start in range(0, len(source), 2):
        end = min(start + 2, len(source))
        base_h = np.asarray(source[start:end], dtype=np.complex64)
        base_support, _ = v423.leaked_support(base_h)
        desired_support, _, _ = v435.build_desired_support(
            base_support,
            np.asarray(joint_target[start:end]),
            alpha_joint[start:end],
        )
        base_link, _ = v426.linkwise_pdp(base_h)
        desired_link = v426.desired_linkwise_pdp(
            base_link,
            np.asarray(link_target[start:end]).reshape(end - start, 256, 4, 192),
            alpha_link[start:end].reshape(end - start, 256, 4),
        )
        candidate = base_h.copy()
        for _ in range(5):
            candidate = v425.project_full_support(candidate, desired_support)
            candidate = v426.restore_desired_pdp(candidate, base_h, desired_link)
        scale = np.exp(10.0 * power_delta[start:end] / 2.0).astype(np.float32)
        output[start:end] = candidate * scale[:, None, None, None]
    output.flush()
    del output


def main() -> None:
    args = parse_args()
    for path in (
        PATHS.cache_dir(475) / "positions.npz",
        PATHS.checkpoints / "v445_candidate_path_token_model" / "component_route_best.pt",
        PATHS.checkpoints / "v443_dense_component_route_model" / "component_route_best.pt",
        PATHS.cache_dir(473) / "v473_train_neural_joint_target.npy",
        v469.JOINT_PRED,
        v469.LINK_PRED,
        v469.POWER_PRED,
    ):
        require(path)
    CACHE.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    route = build_route()
    component = build_component()
    path_feature, path_valid = build_path_tokens()
    positions = np.load(PATHS.cache_dir(475) / "positions.npz")["positions"].astype(np.float32)
    position = position_features(positions)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    shuffled_ids = np.concatenate(
        [rng.permutation(TRAIN_COUNT), TRAIN_COUNT + rng.permutation(COUNT - TRAIN_COUNT)]
    )
    if V443_JOINT.is_file() and V445_JOINT.is_file() and SHUFFLED_JOINT.is_file():
        v443_joint = np.load(V443_JOINT, mmap_mode="r")
        v445_joint = np.load(V445_JOINT, mmap_mode="r")
        shuffled_joint = np.load(SHUFFLED_JOINT, mmap_mode="r")
    else:
        baseline_model = v445.BaseComponentModel(0.0).to(device)
        baseline_checkpoint = torch.load(
            PATHS.checkpoints / "v443_dense_component_route_model" / "component_route_best.pt",
            map_location=device,
            weights_only=True,
        )
        baseline_model.load_state_dict(baseline_checkpoint["model"])
        v443_joint = infer_joint(
            baseline_model, V443_JOINT, route, component, path_feature, path_valid,
            position, device, args.batch_size
        )
        del baseline_model
        torch.cuda.empty_cache()

        path_model = v445.CandidatePathTokenModel(0.0).to(device)
        path_checkpoint = torch.load(
            PATHS.checkpoints / "v445_candidate_path_token_model" / "component_route_best.pt",
            map_location=device,
            weights_only=True,
        )
        path_model.load_state_dict(path_checkpoint["model"])
        v445_joint = infer_joint(
            path_model, V445_JOINT, route, component, path_feature, path_valid,
            position, device, args.batch_size
        )
        shuffled_joint = infer_joint(
            path_model, SHUFFLED_JOINT, route, component, path_feature, path_valid,
            position, device, args.batch_size, path_geometry_ids=shuffled_ids
        )
        del path_model
        torch.cuda.empty_cache()

    data = v469.build_joint_dataset()
    truth = data["train_target"].reshape(TRAIN_COUNT, 192, 16).astype(np.float32)
    truth_pdp = derived_pdp(truth)
    transfer = {}
    for name, prediction in (
        ("v443", v443_joint[:TRAIN_COUNT]),
        ("v445", v445_joint[:TRAIN_COUNT]),
        ("v445_shuffled_paths", shuffled_joint[:TRAIN_COUNT]),
    ):
        transfer[name] = {
            "joint_sh_cosine": float(np.mean(cosine_rows(prediction, truth))),
            "derived_pdp_cosine": float(np.mean(cosine_rows(derived_pdp(prediction), truth_pdp))),
        }

    train_anchor = np.load(
        PATHS.cache_dir(473) / "v473_train_neural_joint_target.npy", mmap_mode="r"
    ).reshape(TRAIN_COUNT, 192, 16)
    alpha_rows = []
    best_alpha = 0.0
    best_score = -math.inf
    for alpha in np.linspace(0.0, 0.80, 17):
        blended = geometric_blend(train_anchor, v445_joint[:TRAIN_COUNT], float(alpha))
        joint_score = float(np.mean(cosine_rows(blended, truth)))
        pdp_score = float(np.mean(cosine_rows(derived_pdp(blended), truth_pdp)))
        score = 0.5 * (joint_score + pdp_score)
        row = {
            "alpha": float(alpha),
            "joint_sh_cosine": joint_score,
            "derived_pdp_cosine": pdp_score,
            "mean_score": score,
        }
        alpha_rows.append(row)
        if score > best_score:
            best_score = score
            best_alpha = float(alpha)

    test_anchor = np.load(v469.JOINT_PRED, mmap_mode="r").reshape(500, 192, 16)
    blended_test = geometric_blend(test_anchor, v445_joint[TRAIN_COUNT:], best_alpha)
    np.save(BLENDED_JOINT, blended_test)
    generate_channel(blended_test)
    replay = v208.score_prediction(OUTPUT)
    baseline_replay = v208.score_prediction(v469.OUTPUT)
    decision = (
        best_alpha > 0.0
        and transfer["v445"]["joint_sh_cosine"] > transfer["v443"]["joint_sh_cosine"]
        and replay["score_traceback_leaked"] > baseline_replay["score_traceback_leaked"]
    )
    report = {
        "method": VERSION,
        "status": "candidate_signal_passed" if decision else "analysis_only_no_candidate",
        "primary_hypothesis": (
            "The v445 path-composition rule learned from 200k same-scene points transfers "
            "when official coordinates use exactly aligned true-scene Sionna path tokens."
        ),
        "inputs": {
            "path_source": "v475 official-coordinate high-budget paths",
            "pretrained_model": "v445 200k same-scene candidate path token model",
            "anchor": "v469 neural calibration target",
        },
        "cache": {
            "route_shape": list(route.shape),
            "component_shape": list(component.shape),
            "path_feature_shape": list(path_feature.shape),
            "mean_paths": float(np.mean(np.sum(path_valid, axis=1))),
        },
        "official_train_transfer": transfer,
        "blend_search": alpha_rows,
        "selected_alpha": best_alpha,
        "replay": replay,
        "baseline_v469_replay": baseline_replay,
        "replay_delta": {
            key: float(replay[key] - baseline_replay[key])
            for key in replay
            if isinstance(replay[key], numbers.Number)
            and isinstance(baseline_replay.get(key), numbers.Number)
        },
        "decision": decision,
        "outputs": {
            "v443_joint": PATHS.relative(V443_JOINT),
            "v445_joint": PATHS.relative(V445_JOINT),
            "shuffled_joint": PATHS.relative(SHUFFLED_JOINT),
            "blended_joint": PATHS.relative(BLENDED_JOINT),
            "channel": PATHS.relative(OUTPUT),
            "report": PATHS.relative(REPORT),
        },
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
