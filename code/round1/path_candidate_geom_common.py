#!/usr/bin/env python3
"""Path-candidate-aware geometry helpers for Round1 submissions."""

from pathlib import Path

import numpy as np

from build_map_features import BS_POS
from evaluate_s1style_geom_blend import angle_abs_diff


def load_path_cache(path):
    return np.load(Path(path))


def path_signature(cache, split):
    return {
        "plane": cache[f"{split}_plane_id"].astype(np.int32),
        "delay": cache[f"{split}_delay_bin"].astype(np.float64),
        "excess": cache[f"{split}_excess_length"].astype(np.float64),
        "score": cache[f"{split}_score"].astype(np.float64),
        "incidence": cache[f"{split}_incidence_cos"].astype(np.float64),
    }


def _min_pairwise_abs(a, b):
    diff = np.abs(a[:, :, None] - b[None, None, :])
    return np.nanmin(diff, axis=(1, 2))


def _plane_overlap(train_planes, query_planes):
    valid_q = query_planes[query_planes >= 0]
    if valid_q.size == 0:
        return np.zeros(train_planes.shape[0], dtype=bool)
    return np.any(train_planes[:, :, None] == valid_q[None, None, :], axis=(1, 2))


def select_path_geom_neighbors(pos, train_blocked, train_paths, q_pos, q_blocked, q_paths, q_index, cfg):
    """Select train neighbors using both 2D proximity and path-candidate similarity."""

    d_xy = np.linalg.norm(pos[:, :2] - q_pos[:2], axis=1)
    if q_index is not None:
        d_xy[int(q_index)] = np.inf

    los = train_blocked == 0
    q_los = q_blocked == 0
    d = pos - BS_POS[None, :]
    train_r = np.linalg.norm(d, axis=1)
    train_theta = np.arctan2(d[:, 1], d[:, 0])
    q_d = q_pos - BS_POS
    q_r = np.linalg.norm(q_d)
    q_theta = np.arctan2(q_d[1], q_d[0])

    occ_diff = np.abs(train_blocked.astype(np.float64) - float(q_blocked)) / max(cfg["occ_scale"], 1e-6)
    r_diff = np.abs(train_r - q_r) / max(cfg["r_scale"], 1e-6)
    theta_diff = angle_abs_diff(train_theta, q_theta) / np.deg2rad(max(cfg["theta_scale_deg"], 1e-6))

    top = int(cfg["path_top"])
    train_planes = train_paths["plane"][:, 1 : 1 + top]
    q_planes = q_paths["plane"][1 : 1 + top]
    no_plane_overlap = (~_plane_overlap(train_planes, q_planes)).astype(np.float64)

    train_delay = train_paths["delay"][:, 1 : 1 + top]
    q_delay = q_paths["delay"][1 : 1 + top]
    delay_diff = _min_pairwise_abs(train_delay, q_delay) / max(cfg["path_delay_scale"], 1e-6)
    delay_diff = np.nan_to_num(delay_diff, nan=4.0, posinf=4.0, neginf=4.0)

    train_excess = train_paths["excess"][:, 1 : 1 + top]
    q_excess = q_paths["excess"][1 : 1 + top]
    excess_diff = _min_pairwise_abs(train_excess, q_excess) / max(cfg["path_excess_scale"], 1e-6)
    excess_diff = np.nan_to_num(excess_diff, nan=4.0, posinf=4.0, neginf=4.0)

    train_score = train_paths["score"][:, 1]
    q_score = float(q_paths["score"][1])
    score_diff = np.abs(train_score - q_score) / max(cfg["path_score_scale"], 1e-6)
    score_diff = np.nan_to_num(score_diff, nan=2.0, posinf=2.0, neginf=2.0)

    eff = d_xy * (
        1.0
        + cfg["los_penalty"] * (los != q_los).astype(np.float64)
        + cfg["occ_penalty"] * occ_diff
        + cfg["r_penalty"] * r_diff
        + cfg["theta_penalty"] * theta_diff
        + cfg["path_plane_penalty"] * no_plane_overlap
        + cfg["path_delay_penalty"] * delay_diff
        + cfg["path_excess_penalty"] * excess_diff
        + cfg["path_score_penalty"] * score_diff
    )

    if cfg.get("two_stage", False):
        plane_match = no_plane_overlap < 0.5
        if np.sum(plane_match) >= cfg["k"]:
            eff = np.where(plane_match, eff, np.inf)

    k = min(int(cfg["k"]), len(pos))
    idx = np.argpartition(eff, k - 1)[:k]
    idx = idx[np.argsort(eff[idx])]
    return idx, d_xy[idx], eff[idx]


def regime_label(blocked):
    if blocked <= 0:
        return "clear"
    if blocked <= 25:
        return "light"
    if blocked <= 100:
        return "medium"
    return "heavy"


def rho_for_blocked(blocked, clear_rho, light_rho, medium_rho, heavy_rho):
    label = regime_label(blocked)
    if label == "clear":
        return float(clear_rho), label
    if label == "light":
        return float(light_rho), label
    if label == "medium":
        return float(medium_rho), label
    return float(heavy_rho), label
