#!/usr/bin/env python3
"""Build LOS, one-bounce, and two-bounce specular path candidates.

This is a diagnostic geometry cache, not a traditional ray-tracing channel
synthesizer.  It only produces candidate path descriptors that can later be
matched to learned angle-delay components or used as AI model features.
"""

from __future__ import annotations
from project_paths import PATHS

import argparse
import json
from pathlib import Path

import numpy as np

from build_map_features import BS_POS
from build_path_candidates import (
    load_planes,
    reflection_candidates_for_points,
    summarize,
    unit,
)


def mirror_xy(points_xy, normals, c):
    signed = np.sum(points_xy * normals, axis=-1) - c
    return points_xy - 2.0 * signed[..., None] * normals


def pair_arrays(planes, max_pairs=None):
    n = len(planes["c"])
    p1, p2 = np.meshgrid(np.arange(n, dtype=np.int32), np.arange(n, dtype=np.int32), indexing="ij")
    keep = p1.reshape(-1) != p2.reshape(-1)
    p1 = p1.reshape(-1)[keep]
    p2 = p2.reshape(-1)[keep]
    if max_pairs is not None and len(p1) > max_pairs:
        # Prefer larger, cleaner planes when reducing the pair set.
        score = np.log1p(planes["count"][p1]) + np.log1p(planes["count"][p2])
        score -= 0.25 * (planes["residual_p90"][p1] + planes["residual_p90"][p2])
        take = np.argpartition(score, -int(max_pairs))[-int(max_pairs) :]
        take = take[np.argsort(score[take])[::-1]]
        p1 = p1[take]
        p2 = p2[take]
    return p1.astype(np.int32), p2.astype(np.int32)


def in_plane_patch(point, plane_idx, planes):
    t = np.sum(point[..., :2] * planes["tangent"][plane_idx], axis=-1)
    z = point[..., 2]
    in_t = (t >= planes["t_min"][plane_idx]) & (t <= planes["t_max"][plane_idx])
    in_z = (z >= planes["z_min"][plane_idx]) & (z <= planes["z_max"][plane_idx])
    wall_width = np.maximum(planes["t_max"][plane_idx] - planes["t_min"][plane_idx], 1e-6)
    wall_height = np.maximum(planes["z_max"][plane_idx] - planes["z_min"][plane_idx], 1e-6)
    t_margin = np.minimum(t - planes["t_min"][plane_idx], planes["t_max"][plane_idx] - t) / wall_width
    z_margin = np.minimum(z - planes["z_min"][plane_idx], planes["z_max"][plane_idx] - z) / wall_height
    return in_t & in_z, np.minimum(t_margin, z_margin)


def two_bounce_candidates_for_points(pos, planes, top_double, delay_slope, delay_intercept, batch, max_pairs=None):
    """Return top two-bounce specular candidates for each point."""

    bs = BS_POS.astype(np.float64)
    p1, p2 = pair_arrays(planes, max_pairs=max_pairs)
    n_pairs = len(p1)
    k = int(top_double)

    n1 = planes["normal"][p1]
    c1 = planes["c"][p1]
    n2 = planes["normal"][p2]
    c2 = planes["c"][p2]

    bs_xy = np.repeat(bs[None, :2], n_pairs, axis=0)
    img1_xy = mirror_xy(bs_xy, n1, c1)
    img1 = np.empty((n_pairs, 3), dtype=np.float64)
    img1[:, :2] = img1_xy
    img1[:, 2] = bs[2]
    img12_xy = mirror_xy(img1_xy, n2, c2)
    img12 = np.empty((n_pairs, 3), dtype=np.float64)
    img12[:, :2] = img12_xy
    img12[:, 2] = bs[2]

    count_norm = np.log1p(planes["count"]) / max(float(np.log1p(planes["count"]).max()), 1e-12)
    residual = np.minimum(planes["residual_p90"], 2.0)

    total_points = len(pos)
    shape = (total_points, k)
    valid = np.zeros(shape, dtype=np.bool_)
    path_type = np.full(shape, 2, dtype=np.int8)
    plane_id = np.full(shape, -1, dtype=np.int32)
    plane_id_2 = np.full(shape, -1, dtype=np.int32)
    path_length = np.full(shape, np.inf, dtype=np.float32)
    delay_bin = np.full(shape, np.nan, dtype=np.float32)
    score = np.full(shape, -np.inf, dtype=np.float32)
    virtual_source = np.full((total_points, k, 3), np.nan, dtype=np.float32)
    reflect_point = np.full((total_points, k, 3), np.nan, dtype=np.float32)
    reflect_point_2 = np.full((total_points, k, 3), np.nan, dtype=np.float32)
    tx_dir = np.full((total_points, k, 3), np.nan, dtype=np.float32)
    rx_dir = np.full((total_points, k, 3), np.nan, dtype=np.float32)
    excess_length = np.full(shape, np.nan, dtype=np.float32)
    edge_margin = np.full(shape, np.nan, dtype=np.float32)
    incidence_cos = np.full(shape, np.nan, dtype=np.float32)

    for start in range(0, total_points, batch):
        end = min(start + batch, total_points)
        rx = np.asarray(pos[start:end], dtype=np.float64)
        q = end - start

        # First find second reflection p2 by intersecting image-source -> RX
        # with plane2.
        ray2 = rx[:, None, :] - img12[None, :, :]
        denom2 = np.einsum("qpi,pi->qp", ray2[:, :, :2], n2)
        numer2 = c2[None, :] - np.einsum("pi,pi->p", img12[:, :2], n2)[None, :]
        tau2 = numer2 / np.where(np.abs(denom2) < 1e-12, np.nan, denom2)
        r2 = img12[None, :, :] + tau2[:, :, None] * ray2

        ok2 = (tau2 > 1e-5) & (tau2 < 1.0 - 1e-5) & np.isfinite(tau2)
        in2, margin2 = in_plane_patch(r2, p2[None, :], planes)

        # Then find first reflection p1 by intersecting image1 -> p2 with
        # plane1.
        ray1 = r2 - img1[None, :, :]
        denom1 = np.einsum("qpi,pi->qp", ray1[:, :, :2], n1)
        numer1 = c1[None, :] - np.einsum("pi,pi->p", img1[:, :2], n1)[None, :]
        tau1 = numer1 / np.where(np.abs(denom1) < 1e-12, np.nan, denom1)
        r1 = img1[None, :, :] + tau1[:, :, None] * ray1

        ok1 = (tau1 > 1e-5) & (tau1 < 1.0 - 1e-5) & np.isfinite(tau1)
        in1, margin1 = in_plane_patch(r1, p1[None, :], planes)
        ok = ok1 & ok2 & in1 & in2

        len0 = np.linalg.norm(r1 - bs[None, None, :], axis=2)
        len1 = np.linalg.norm(r2 - r1, axis=2)
        len2 = np.linalg.norm(rx[:, None, :] - r2, axis=2)
        total_len = len0 + len1 + len2
        los_len = np.linalg.norm(rx - bs[None, :], axis=1)[:, None]
        excess = total_len - los_len

        d0 = unit(r1 - bs[None, None, :])
        d1_in = unit(r1 - bs[None, None, :])[:, :, :2]
        d1_out = unit(r2 - r1)[:, :, :2]
        inc1 = np.abs(np.einsum("qpi,pi->qp", d1_in, n1))
        inc2 = np.abs(np.einsum("qpi,pi->qp", d1_out, n2))
        margin = np.minimum(margin1, margin2)

        raw_score = (
            3.0 * ok.astype(np.float64)
            + 0.22 * (count_norm[p1][None, :] + count_norm[p2][None, :])
            + 0.22 * np.clip(margin, -1.0, 1.0)
            + 0.08 * (inc1 + inc2)
            - 0.012 * np.maximum(excess, 0.0)
            - 0.12 * (residual[p1][None, :] + residual[p2][None, :])
        )
        raw_score = np.where(ok, raw_score, -np.inf)

        take = min(k, n_pairs)
        top = np.argpartition(raw_score, -take, axis=1)[:, -take:]
        row = np.arange(q)[:, None]
        top = top[row, np.argsort(raw_score[row, top], axis=1)[:, ::-1]]

        for j in range(take):
            pair_idx = top[:, j]
            rows = np.arange(q)
            good = np.isfinite(raw_score[rows, pair_idx])
            valid[start:end, j] = good
            plane_id[start:end, j] = p1[pair_idx].astype(np.int32)
            plane_id_2[start:end, j] = p2[pair_idx].astype(np.int32)
            path_length[start:end, j] = total_len[rows, pair_idx].astype(np.float32)
            delay_bin[start:end, j] = (delay_slope * total_len[rows, pair_idx] + delay_intercept).astype(np.float32)
            score[start:end, j] = raw_score[rows, pair_idx].astype(np.float32)
            virtual_source[start:end, j] = img12[pair_idx].astype(np.float32)
            reflect_point[start:end, j] = r1[rows, pair_idx].astype(np.float32)
            reflect_point_2[start:end, j] = r2[rows, pair_idx].astype(np.float32)
            tx_dir[start:end, j] = d0[rows, pair_idx].astype(np.float32)
            rx_dir[start:end, j] = unit(r2[rows, pair_idx] - rx).astype(np.float32)
            excess_length[start:end, j] = excess[rows, pair_idx].astype(np.float32)
            edge_margin[start:end, j] = margin[rows, pair_idx].astype(np.float32)
            incidence_cos[start:end, j] = np.minimum(inc1[rows, pair_idx], inc2[rows, pair_idx]).astype(np.float32)

        invalid = ~valid[start:end]
        path_length[start:end][invalid] = np.inf
        delay_bin[start:end][invalid] = np.nan
        score[start:end][invalid] = -np.inf
        excess_length[start:end][invalid] = np.nan
        edge_margin[start:end][invalid] = np.nan
        incidence_cos[start:end][invalid] = np.nan
        virtual_source[start:end][invalid] = np.nan
        reflect_point[start:end][invalid] = np.nan
        reflect_point_2[start:end][invalid] = np.nan
        tx_dir[start:end][invalid] = np.nan
        rx_dir[start:end][invalid] = np.nan

        print(f"processed two-bounce candidates {end}/{total_points}", flush=True)

    return {
        "valid": valid,
        "path_type": path_type,
        "plane_id": plane_id,
        "plane_id_2": plane_id_2,
        "path_length": path_length,
        "delay_bin": delay_bin,
        "score": score,
        "virtual_source": virtual_source,
        "reflect_point": reflect_point,
        "reflect_point_2": reflect_point_2,
        "tx_dir": tx_dir,
        "rx_dir": rx_dir,
        "excess_length": excess_length,
        "edge_margin": edge_margin,
        "incidence_cos": incidence_cos,
    }


def concat_candidates(single, double):
    n = single["valid"].shape[0]
    reflect_point_2_single = np.full_like(single["reflect_point"], np.nan, dtype=np.float32)
    plane_id_2_single = np.full_like(single["plane_id"], -1, dtype=np.int32)
    out = {}
    keys = [
        "valid",
        "path_type",
        "plane_id",
        "plane_id_2",
        "path_length",
        "delay_bin",
        "score",
        "virtual_source",
        "reflect_point",
        "reflect_point_2",
        "tx_dir",
        "rx_dir",
        "excess_length",
        "edge_margin",
        "incidence_cos",
    ]
    for key in keys:
        left = single.get(key)
        if left is None:
            if key == "reflect_point_2":
                left = reflect_point_2_single
            elif key == "plane_id_2":
                left = plane_id_2_single
            else:
                raise KeyError(key)
        right = double[key]
        if left.shape[0] != n or right.shape[0] != n:
            raise ValueError(f"shape mismatch for {key}: {left.shape}, {right.shape}")
        out[key] = np.concatenate([left, right], axis=1)
    return out


def summarize_multibounce(name, cand):
    valid = cand["valid"]
    typ = cand["path_type"]
    summary = summarize(name, cand)
    for t, label in [(1, "one_bounce"), (2, "two_bounce")]:
        mask = valid & (typ == t)
        summary[f"{label}_valid_frac"] = float(np.mean(mask))
        summary[f"points_with_{label}"] = float(np.mean(np.any(mask, axis=1)))
        if np.any(mask):
            summary[f"{label}_excess_p50"] = float(np.nanpercentile(np.where(mask, cand["excess_length"], np.nan), 50))
            summary[f"{label}_excess_p90"] = float(np.nanpercentile(np.where(mask, cand["excess_length"], np.nan), 90))
            summary[f"{label}_score_p50"] = float(np.nanpercentile(np.where(mask, cand["score"], np.nan), 50))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--planes", default=str(PATHS.analysis / "results/json/wall_planes.json"))
    parser.add_argument("--delay-fit", default=str(PATHS.artifacts / "shared_features/delay_length_fit.npz"))
    parser.add_argument("--output", default=str(PATHS.artifacts / "shared_features/multibounce_path_candidates.npz"))
    parser.add_argument("--summary", default=str(PATHS.analysis / "results/json/multibounce_path_candidates_summary.json"))
    parser.add_argument("--max-planes", type=int, default=80)
    parser.add_argument("--max-pairs", type=int, default=6400)
    parser.add_argument("--top-reflections", type=int, default=6)
    parser.add_argument("--top-double", type=int, default=8)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    data = Path(args.data)
    train_pos = np.load(data / "Round1_Train_Pos.npy")
    test_pos = np.load(data / "Round1_Test_Pos.npy")
    delay = np.load(args.delay_fit)
    delay_slope = float(delay["slope"])
    delay_intercept = float(delay["intercept"])
    planes = load_planes(args.planes, args.max_planes)

    payload = {}
    summaries = {}
    for prefix, pos in [("train", train_pos), ("test", test_pos)]:
        single = reflection_candidates_for_points(
            pos,
            planes,
            args.top_reflections,
            delay_slope,
            delay_intercept,
            args.batch,
        )
        double = two_bounce_candidates_for_points(
            pos,
            planes,
            args.top_double,
            delay_slope,
            delay_intercept,
            args.batch,
            args.max_pairs,
        )
        cand = concat_candidates(single, double)
        for key, value in cand.items():
            payload[f"{prefix}_{key}"] = value
        summaries[prefix] = summarize_multibounce(prefix, cand)

    payload.update({
        "bs_pos": BS_POS.astype(np.float32),
        "plane_normal": planes["normal"].astype(np.float32),
        "plane_c": planes["c"].astype(np.float32),
        "plane_tangent": planes["tangent"].astype(np.float32),
        "plane_t_min": planes["t_min"].astype(np.float32),
        "plane_t_max": planes["t_max"].astype(np.float32),
        "plane_z_min": planes["z_min"].astype(np.float32),
        "plane_z_max": planes["z_max"].astype(np.float32),
        "plane_count": planes["count"].astype(np.float32),
        "plane_residual_p90": planes["residual_p90"].astype(np.float32),
        "delay_slope": np.asarray(delay_slope, dtype=np.float32),
        "delay_intercept": np.asarray(delay_intercept, dtype=np.float32),
        "max_pairs": np.asarray(args.max_pairs, dtype=np.int32),
    })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)

    summary = {
        "output": str(out),
        "planes": str(args.planes),
        "max_planes": int(args.max_planes),
        "max_pairs": int(args.max_pairs),
        "top_reflections": int(args.top_reflections),
        "top_double": int(args.top_double),
        "delay_slope": delay_slope,
        "delay_intercept": delay_intercept,
        "train": summaries["train"],
        "test": summaries["test"],
    }
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
