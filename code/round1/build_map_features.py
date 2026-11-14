#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


BS_POS = np.array([50.0, 0.0, 25.0], dtype=np.float64)


def read_ply_memmap(path):
    path = Path(path)
    with path.open("rb") as f:
        vertex_count = None
        while True:
            line = f.readline()
            if line.startswith(b"element vertex"):
                vertex_count = int(line.split()[-1])
            if line.strip() == b"end_header":
                offset = f.tell()
                break
    if vertex_count is None:
        raise ValueError("PLY vertex count not found")
    dtype = np.dtype([
        ("x", "<f8"),
        ("y", "<f8"),
        ("z", "<f8"),
        ("nx", "<f8"),
        ("ny", "<f8"),
        ("nz", "<f8"),
    ])
    return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(vertex_count,))


def select_wall_points(vertices, x_min, x_max, y_min, y_max):
    x = np.asarray(vertices["x"])
    y = np.asarray(vertices["y"])
    z = np.asarray(vertices["z"])
    nz = np.asarray(vertices["nz"])
    mask = (
        (z > 2.0)
        & (np.abs(nz) < 0.7)
        & (x >= x_min)
        & (x <= x_max)
        & (y >= y_min)
        & (y <= y_max)
    )
    points = np.stack([x[mask], y[mask], z[mask]], axis=1).astype(np.float32)
    normals = np.stack([
        np.asarray(vertices["nx"])[mask],
        np.asarray(vertices["ny"])[mask],
        np.asarray(vertices["nz"])[mask],
    ], axis=1).astype(np.float32)
    return points, normals


def query_line_features(tree3, points3, pos, samples, thresholds, batch):
    all_feats = []
    eps = 1e-6
    for start in range(0, len(pos), batch):
        end = min(start + batch, len(pos))
        ue = pos[start:end].astype(np.float64)
        line = BS_POS[None, None, :] * (1.0 - samples[None, :, None]) + ue[:, None, :] * samples[None, :, None]
        q = line.reshape(-1, 3)
        dist, _ = tree3.query(q, k=1, workers=-1)
        dist = dist.reshape(end - start, samples.size)

        feats = [
            np.min(dist, axis=1),
            np.percentile(dist, 5, axis=1),
            np.percentile(dist, 10, axis=1),
            np.mean(1.0 / np.maximum(dist, eps), axis=1),
        ]
        for radius in thresholds:
            hit = dist < radius
            count = np.sum(hit, axis=1).astype(np.float64)
            any_hit = count > 0
            first = np.ones(end - start, dtype=np.float64)
            last = np.zeros(end - start, dtype=np.float64)
            if np.any(any_hit):
                first_idx = np.argmax(hit[any_hit], axis=1)
                last_idx = hit.shape[1] - 1 - np.argmax(hit[any_hit, ::-1], axis=1)
                first[any_hit] = samples[first_idx]
                last[any_hit] = samples[last_idx]
            feats.extend([
                count,
                count / samples.size,
                first,
                last,
            ])
        all_feats.append(np.stack(feats, axis=1))
    return np.concatenate(all_feats, axis=0).astype(np.float32)


def query_receiver_features(tree3, tree2, points3, normals, pos, batch):
    feats = []
    for start in range(0, len(pos), batch):
        end = min(start + batch, len(pos))
        p = pos[start:end].astype(np.float64)
        dist3, idx3 = tree3.query(p, k=1, workers=-1)
        dist2, idx2 = tree2.query(p[:, :2], k=1, workers=-1)
        nearest3 = points3[idx3]
        nearest2 = points3[idx2]
        normals3 = normals[idx3]
        normals2 = normals[idx2]
        feats.append(np.concatenate([
            dist3[:, None],
            nearest3,
            normals3,
            dist2[:, None],
            nearest2,
            normals2,
        ], axis=1))
    return np.concatenate(feats, axis=0).astype(np.float32)


def make_feature_names(thresholds):
    names = [
        "los_min_dist",
        "los_p05_dist",
        "los_p10_dist",
        "los_mean_inv_dist",
    ]
    for radius in thresholds:
        tag = str(radius).replace(".", "p")
        names.extend([
            f"los_count_r{tag}",
            f"los_frac_r{tag}",
            f"los_first_t_r{tag}",
            f"los_last_t_r{tag}",
        ])
    names.extend([
        "rx_wall_dist3",
        "rx_wall_x3",
        "rx_wall_y3",
        "rx_wall_z3",
        "rx_wall_nx3",
        "rx_wall_ny3",
        "rx_wall_nz3",
        "rx_wall_dist2",
        "rx_wall_x2",
        "rx_wall_y2",
        "rx_wall_z2",
        "rx_wall_nx2",
        "rx_wall_ny2",
        "rx_wall_nz2",
    ])
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--output", default=str(PATHS.artifacts / "shared_features/map_features.npz"))
    parser.add_argument("--summary", default=str(PATHS.analysis / "results/json/map_features_summary.json"))
    parser.add_argument("--x-min", type=float, default=-50.0)
    parser.add_argument("--x-max", type=float, default=300.0)
    parser.add_argument("--y-min", type=float, default=-250.0)
    parser.add_argument("--y-max", type=float, default=150.0)
    parser.add_argument("--line-samples", type=int, default=96)
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()

    data = Path(args.data)
    train_pos = np.load(data / "Round1_Train_Pos.npy")
    test_pos = np.load(data / "Round1_Test_Pos.npy")
    vertices = read_ply_memmap(data / "Round1_Map.ply")
    wall_points, wall_normals = select_wall_points(vertices, args.x_min, args.x_max, args.y_min, args.y_max)
    print(f"selected wall points: {len(wall_points)}", flush=True)

    tree3 = cKDTree(wall_points)
    tree2 = cKDTree(wall_points[:, :2])
    thresholds = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    samples = np.linspace(0.03, 0.97, args.line_samples, dtype=np.float64)
    names = make_feature_names(thresholds)

    train_line = query_line_features(tree3, wall_points, train_pos, samples, thresholds, args.batch)
    test_line = query_line_features(tree3, wall_points, test_pos, samples, thresholds, args.batch)
    train_rx = query_receiver_features(tree3, tree2, wall_points, wall_normals, train_pos, args.batch)
    test_rx = query_receiver_features(tree3, tree2, wall_points, wall_normals, test_pos, args.batch)
    train_features = np.concatenate([train_line, train_rx], axis=1).astype(np.float32)
    test_features = np.concatenate([test_line, test_rx], axis=1).astype(np.float32)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        train_features=train_features,
        test_features=test_features,
        feature_names=np.asarray(names),
        thresholds=np.asarray(thresholds, dtype=np.float32),
        wall_region=np.asarray([args.x_min, args.x_max, args.y_min, args.y_max], dtype=np.float32),
    )
    summary = {
        "wall_points": int(len(wall_points)),
        "wall_region": [args.x_min, args.x_max, args.y_min, args.y_max],
        "line_samples": int(args.line_samples),
        "feature_count": int(len(names)),
        "feature_names": names,
        "train_shape": list(train_features.shape),
        "test_shape": list(test_features.shape),
        "train_min": train_features.min(axis=0).astype(float).tolist(),
        "train_max": train_features.max(axis=0).astype(float).tolist(),
        "train_mean": train_features.mean(axis=0).astype(float).tolist(),
        "test_min": test_features.min(axis=0).astype(float).tolist(),
        "test_max": test_features.max(axis=0).astype(float).tolist(),
        "test_mean": test_features.mean(axis=0).astype(float).tolist(),
    }
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({
        "output": args.output,
        "summary": args.summary,
        "wall_points": int(len(wall_points)),
        "train_shape": list(train_features.shape),
        "test_shape": list(test_features.shape),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
