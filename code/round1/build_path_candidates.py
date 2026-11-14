#!/usr/bin/env python3
from project_paths import PATHS
"""Build LOS and specular-reflection path candidates from the Round1 point cloud.

The output is a compact geometric cache for later AI/path-component models. It
does not synthesize channels by ray tracing; it only turns the map into
candidate path descriptors that can be matched to learned angle-delay
components.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from build_map_features import BS_POS


def load_planes(path, max_planes):
    with open(path) as f:
        payload = json.load(f)
    planes = payload["planes"][:max_planes]
    normal = np.asarray([p["normal"] for p in planes], dtype=np.float64)
    c = np.asarray([p["c"] for p in planes], dtype=np.float64)
    t_min = np.asarray([p["t_min"] for p in planes], dtype=np.float64)
    t_max = np.asarray([p["t_max"] for p in planes], dtype=np.float64)
    z_min = np.asarray([p["z_min"] for p in planes], dtype=np.float64)
    z_max = np.asarray([p["z_max"] for p in planes], dtype=np.float64)
    count = np.asarray([p["count"] for p in planes], dtype=np.float64)
    residual_p90 = np.asarray([p["residual_p90"] for p in planes], dtype=np.float64)
    tangent = np.stack([-normal[:, 1], normal[:, 0]], axis=1)
    return {
        "normal": normal,
        "c": c,
        "tangent": tangent,
        "t_min": t_min,
        "t_max": t_max,
        "z_min": z_min,
        "z_max": z_max,
        "count": count,
        "residual_p90": residual_p90,
        "raw": planes,
    }


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-30)


def mirror_points_across_vertical_planes(point, planes):
    """Mirror a 3D point across vertical 2D wall planes n.xy dot x = c."""

    p_xy = np.asarray(point[:2], dtype=np.float64)
    signed = p_xy @ planes["normal"].T - planes["c"]
    mirrored_xy = p_xy[None, :] - 2.0 * signed[:, None] * planes["normal"]
    out = np.empty((len(mirrored_xy), 3), dtype=np.float64)
    out[:, :2] = mirrored_xy
    out[:, 2] = point[2]
    return out


def reflection_candidates_for_points(pos, planes, top_reflections, delay_slope, delay_intercept, batch):
    """Return per-point top specular reflection candidates."""

    n_planes = len(planes["c"])
    bs = BS_POS.astype(np.float64)
    vt = mirror_points_across_vertical_planes(bs, planes)
    count_norm = np.log1p(planes["count"]) / max(float(np.log1p(planes["count"]).max()), 1e-12)
    wall_width = np.maximum(planes["t_max"] - planes["t_min"], 1e-6)
    wall_height = np.maximum(planes["z_max"] - planes["z_min"], 1e-6)

    p_total = len(pos)
    k_ref = int(top_reflections)
    shape = (p_total, 1 + k_ref)
    valid = np.zeros(shape, dtype=np.bool_)
    path_type = np.zeros(shape, dtype=np.int8)
    plane_id = np.full(shape, -1, dtype=np.int32)
    path_length = np.full(shape, np.inf, dtype=np.float32)
    delay_bin = np.full(shape, np.nan, dtype=np.float32)
    score = np.full(shape, -np.inf, dtype=np.float32)
    virtual_source = np.full((p_total, 1 + k_ref, 3), np.nan, dtype=np.float32)
    reflect_point = np.full((p_total, 1 + k_ref, 3), np.nan, dtype=np.float32)
    tx_dir = np.full((p_total, 1 + k_ref, 3), np.nan, dtype=np.float32)
    rx_dir = np.full((p_total, 1 + k_ref, 3), np.nan, dtype=np.float32)
    excess_length = np.full(shape, np.nan, dtype=np.float32)
    edge_margin = np.full(shape, np.nan, dtype=np.float32)
    incidence_cos = np.full(shape, np.nan, dtype=np.float32)

    for start in range(0, p_total, batch):
        end = min(start + batch, p_total)
        rx = np.asarray(pos[start:end], dtype=np.float64)
        q = end - start

        los_vec = rx - bs[None, :]
        los_len = np.linalg.norm(los_vec, axis=1)
        valid[start:end, 0] = True
        path_type[start:end, 0] = 0
        plane_id[start:end, 0] = -1
        path_length[start:end, 0] = los_len.astype(np.float32)
        delay_bin[start:end, 0] = (delay_slope * los_len + delay_intercept).astype(np.float32)
        score[start:end, 0] = 1.0
        virtual_source[start:end, 0] = bs.astype(np.float32)
        reflect_point[start:end, 0] = np.nan
        tx_dir[start:end, 0] = unit(los_vec).astype(np.float32)
        rx_dir[start:end, 0] = unit(bs[None, :] - rx).astype(np.float32)
        excess_length[start:end, 0] = 0.0
        edge_margin[start:end, 0] = np.inf

        # Shape: Q x P x 3, ray from mirror source to RX.
        ray = rx[:, None, :] - vt[None, :, :]
        denom = np.einsum("qpi,pi->qp", ray[:, :, :2], planes["normal"])
        numer = planes["c"][None, :] - np.einsum("pi,pi->p", vt[:, :2], planes["normal"])[None, :]
        tau = numer / np.where(np.abs(denom) < 1e-12, np.nan, denom)
        refl = vt[None, :, :] + tau[:, :, None] * ray
        t_coord = np.einsum("qpi,pi->qp", refl[:, :, :2], planes["tangent"])

        in_segment = (tau > 1e-5) & (tau < 1.0 - 1e-5)
        in_t = (t_coord >= planes["t_min"][None, :]) & (t_coord <= planes["t_max"][None, :])
        in_z = (refl[:, :, 2] >= planes["z_min"][None, :]) & (refl[:, :, 2] <= planes["z_max"][None, :])
        ok = in_segment & in_t & in_z & np.isfinite(tau)

        bs_to_ref = refl - bs[None, None, :]
        ref_to_rx = rx[:, None, :] - refl
        len1 = np.linalg.norm(bs_to_ref, axis=2)
        len2 = np.linalg.norm(ref_to_rx, axis=2)
        total_len = len1 + len2
        los_len_q = los_len[:, None]
        excess = total_len - los_len_q

        t_margin = np.minimum(t_coord - planes["t_min"][None, :], planes["t_max"][None, :] - t_coord) / wall_width[None, :]
        z_margin = np.minimum(refl[:, :, 2] - planes["z_min"][None, :], planes["z_max"][None, :] - refl[:, :, 2]) / wall_height[None, :]
        margin = np.minimum(t_margin, z_margin)
        inc = -unit(bs[None, None, :] - refl)[:, :, :2]
        inc_cos = np.abs(np.einsum("qpi,pi->qp", inc, planes["normal"]))

        # Prefer valid, large, clean planes whose reflection point is away from
        # patch borders and does not add implausibly long excess distance.
        raw_score = (
            2.0 * ok.astype(np.float64)
            + 0.45 * count_norm[None, :]
            + 0.35 * np.clip(margin, -1.0, 1.0)
            + 0.10 * inc_cos
            - 0.020 * np.maximum(excess, 0.0)
            - 0.20 * np.minimum(planes["residual_p90"], 2.0)[None, :]
        )
        raw_score = np.where(ok, raw_score, -np.inf)

        take = min(k_ref, n_planes)
        top = np.argpartition(raw_score, -take, axis=1)[:, -take:]
        row = np.arange(q)[:, None]
        top = top[row, np.argsort(raw_score[row, top], axis=1)[:, ::-1]]

        for j in range(take):
            pidx = top[:, j]
            dst_col = 1 + j
            valid[start:end, dst_col] = np.isfinite(raw_score[np.arange(q), pidx])
            path_type[start:end, dst_col] = 1
            plane_id[start:end, dst_col] = pidx.astype(np.int32)
            path_length[start:end, dst_col] = total_len[np.arange(q), pidx].astype(np.float32)
            delay_bin[start:end, dst_col] = (
                delay_slope * total_len[np.arange(q), pidx] + delay_intercept
            ).astype(np.float32)
            score[start:end, dst_col] = raw_score[np.arange(q), pidx].astype(np.float32)
            virtual_source[start:end, dst_col] = vt[pidx].astype(np.float32)
            reflect_point[start:end, dst_col] = refl[np.arange(q), pidx].astype(np.float32)
            tx_dir[start:end, dst_col] = unit(refl[np.arange(q), pidx] - bs[None, :]).astype(np.float32)
            rx_dir[start:end, dst_col] = unit(refl[np.arange(q), pidx] - rx).astype(np.float32)
            excess_length[start:end, dst_col] = excess[np.arange(q), pidx].astype(np.float32)
            edge_margin[start:end, dst_col] = margin[np.arange(q), pidx].astype(np.float32)
            incidence_cos[start:end, dst_col] = inc_cos[np.arange(q), pidx].astype(np.float32)

        print(f"processed path candidates {end}/{p_total}", flush=True)

    # Clean invalid numeric fields.
    invalid = ~valid
    path_length[invalid] = np.inf
    delay_bin[invalid] = np.nan
    score[invalid] = -np.inf
    excess_length[invalid] = np.nan
    edge_margin[invalid] = np.nan
    incidence_cos[invalid] = np.nan
    virtual_source[invalid] = np.nan
    reflect_point[invalid] = np.nan
    tx_dir[invalid] = np.nan
    rx_dir[invalid] = np.nan
    return {
        "valid": valid,
        "path_type": path_type,
        "plane_id": plane_id,
        "path_length": path_length,
        "delay_bin": delay_bin,
        "score": score,
        "virtual_source": virtual_source,
        "reflect_point": reflect_point,
        "tx_dir": tx_dir,
        "rx_dir": rx_dir,
        "excess_length": excess_length,
        "edge_margin": edge_margin,
        "incidence_cos": incidence_cos,
    }


def summarize(name, cand):
    valid = cand["valid"]
    refl_valid = valid[:, 1:]
    path_length = cand["path_length"]
    score = cand["score"]
    excess = cand["excess_length"]
    summary = {
        "name": name,
        "points": int(valid.shape[0]),
        "candidate_count": int(valid.shape[1]),
        "reflection_valid_frac": float(np.mean(refl_valid)),
        "points_with_reflection": float(np.mean(np.any(refl_valid, axis=1))),
        "los_length_p50": float(np.percentile(path_length[:, 0], 50)),
    }
    if np.any(refl_valid):
        first_valid = refl_valid[:, 0]
        summary.update({
            "top_reflection_valid_frac": float(np.mean(first_valid)),
            "top_reflection_score_p50": float(np.nanpercentile(np.where(first_valid, score[:, 1], np.nan), 50)),
            "top_reflection_excess_p50": float(np.nanpercentile(np.where(first_valid, excess[:, 1], np.nan), 50)),
            "top_reflection_excess_p90": float(np.nanpercentile(np.where(first_valid, excess[:, 1], np.nan), 90)),
        })
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--planes", default=str(PATHS.analysis / "results/json/wall_planes.json"))
    parser.add_argument("--delay-fit", default=str(PATHS.artifacts / "shared_features/delay_length_fit.npz"))
    parser.add_argument("--output", default=str(PATHS.artifacts / "shared_features/path_candidates.npz"))
    parser.add_argument("--summary", default=str(PATHS.analysis / "results/json/path_candidates_summary.json"))
    parser.add_argument("--max-planes", type=int, default=120)
    parser.add_argument("--top-reflections", type=int, default=8)
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args()

    data = Path(args.data)
    train_pos = np.load(data / "Round1_Train_Pos.npy")
    test_pos = np.load(data / "Round1_Test_Pos.npy")
    delay = np.load(args.delay_fit)
    delay_slope = float(delay["slope"])
    delay_intercept = float(delay["intercept"])
    planes = load_planes(args.planes, args.max_planes)

    train = reflection_candidates_for_points(
        train_pos,
        planes,
        args.top_reflections,
        delay_slope,
        delay_intercept,
        args.batch,
    )
    test = reflection_candidates_for_points(
        test_pos,
        planes,
        args.top_reflections,
        delay_slope,
        delay_intercept,
        args.batch,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for prefix, cand in [("train", train), ("test", test)]:
        for key, value in cand.items():
            payload[f"{prefix}_{key}"] = value
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
    })
    np.savez_compressed(out, **payload)

    summary = {
        "output": str(out),
        "planes": str(args.planes),
        "max_planes": int(args.max_planes),
        "top_reflections": int(args.top_reflections),
        "delay_slope": delay_slope,
        "delay_intercept": delay_intercept,
        "train": summarize("train", train),
        "test": summarize("test", test),
    }
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
