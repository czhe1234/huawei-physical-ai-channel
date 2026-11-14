#!/usr/bin/env python3
"""Build source-aligned propagation traces and raw PathTokens for E208."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import time

import numpy as np

from project_paths import PATHS
from round2.data import load_positions
from round2.physical_prior.geometry import load_geometry
from round2.physical_prior.pipeline import GEOMETRY_CACHE as BASE_GEOMETRY_CACHE
from round2.physical_prior.source_aligned import (
    ANALYSIS_DIR,
    GEOMETRY_CACHE,
    RAW_TOKEN_CACHE,
    SourceAlignedTracer,
    assemble_token_shards,
    atomic_json,
    build_diffuse_bank,
    build_route_bank,
    build_token_shard,
    run_trace_shard,
)
from round2.physical_prior.source_aligned_config import SourceAlignedConfig
from round2.physical_prior.source_aligned_geometry import (
    build_source_aligned_geometry,
    load_source_aligned_geometry,
    save_source_aligned_geometry,
)


PROFILE_REPORT = ANALYSIS_DIR / "performance_profile.json"
ASSEMBLY_REPORT = ANALYSIS_DIR / "raw_token_assembly.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("geometry", "profile", "trace", "route", "diffuse", "tokens", "assemble"),
        required=True,
    )
    parser.add_argument("--site", type=int, default=0)
    parser.add_argument("--branch", choices=("base_d16", "scatter015_d10"), default="base_d16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--token-shard-size", type=int, default=250)
    parser.add_argument("--profile-rays", type=int, default=65_536)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def select_branch(config: SourceAlignedConfig, name: str):
    for branch in config.branches:
        if branch.name == name:
            return branch
    raise ValueError(name)


def geometry_stage(force: bool) -> dict[str, object]:
    if GEOMETRY_CACHE.is_file() and not force:
        aligned = load_source_aligned_geometry(GEOMETRY_CACHE)
        return {
            "status": "complete",
            "reused": True,
            "planes": aligned.plane_count,
            "digest": aligned.digest,
        }
    started = time.time()
    geometry = load_geometry(BASE_GEOMETRY_CACHE)
    aligned = build_source_aligned_geometry(geometry)
    save_source_aligned_geometry(GEOMETRY_CACHE, aligned)
    return {
        "status": "complete",
        "reused": False,
        "planes": aligned.plane_count,
        "digest": aligned.digest,
        "seconds": float(time.time() - started),
    }


def profile_stage(
    config: SourceAlignedConfig, device: str, rays: int
) -> dict[str, object]:
    setup, _, _, _, _ = load_positions()
    geometry_started = time.time()
    geometry = load_geometry(BASE_GEOMETRY_CACHE)
    aligned = load_source_aligned_geometry(GEOMETRY_CACHE)
    geometry_seconds = time.time() - geometry_started
    bvh_started = time.time()
    tracer = SourceAlignedTracer(geometry, aligned, config, device)
    bvh_seconds = time.time() - bvh_started
    branches = []
    for branch in config.branches:
        local = replace(branch, shard_size=rays)
        offset = max((branch.rays_per_site_seed - rays) * 3 // 4, 0)
        arrays, timing = tracer.trace(
            source=setup.bs_positions[0],
            branch=local,
            seed=config.seeds[0],
            ray_offset=offset,
            ray_count=rays,
        )
        branches.append(
            {
                "branch": branch.name,
                "rays": rays,
                "hit_fraction": float(np.mean(arrays["count"] > 0)),
                "mean_depth": float(np.mean(arrays["count"])),
                "diffuse_events": int(np.sum(arrays["interaction"] == 2)),
                "output_bytes": int(sum(value.nbytes for value in arrays.values())),
                **timing,
                "rays_per_gpu_second": float(
                    rays / max(timing["gpu_trace_seconds"], 1e-9)
                ),
            }
        )
    expected_trace_job_seconds = max(
        row["gpu_trace_seconds"] * config.base.shard_size / rays
        + row["device_to_host_seconds"] * config.base.shard_size / rays
        for row in branches
    )
    report = {
        "status": "complete",
        "device": device,
        "geometry_load_seconds": float(geometry_seconds),
        "bvh_build_seconds": float(bvh_seconds),
        "branches": branches,
        "expected_trace_job_seconds": float(expected_trace_job_seconds),
        "expected_dominant_bottleneck": "target-batched exact image backtracking and diffuse visibility",
        "cpu_preprocessing": "route hashing and atomic cache assembly",
        "host_to_device": "geometry, route banks, and material tables remain resident per worker",
        "scoring": "five frozen E207 folds run after raw token assembly",
    }
    atomic_json(PROFILE_REPORT, report)
    return report


def main() -> None:
    args = parse_args()
    config = SourceAlignedConfig()
    if args.stage == "geometry":
        report = geometry_stage(args.force)
    elif args.stage == "profile":
        report = profile_stage(config, args.device, args.profile_rays)
    else:
        geometry = load_geometry(BASE_GEOMETRY_CACHE)
        aligned = load_source_aligned_geometry(GEOMETRY_CACHE)
        setup, positions, _, site_id, _ = load_positions()
        branch = select_branch(config, args.branch)
        if args.stage == "trace":
            report = run_trace_shard(
                geometry=geometry,
                aligned=aligned,
                bs_positions=setup.bs_positions,
                config=config,
                site=args.site,
                branch=branch,
                seed=args.seed,
                shard=args.shard,
                device=args.device,
                force=args.force,
            )
        elif args.stage == "route":
            report = build_route_bank(
                site=args.site,
                branch=branch,
                seed=args.seed,
                config=config,
                force=args.force,
            )
        elif args.stage == "diffuse":
            report = build_diffuse_bank(
                site=args.site,
                seed=args.seed,
                branch=config.scatter,
                config=config,
                force=args.force,
            )
        elif args.stage == "tokens":
            point_ids = np.flatnonzero(site_id == args.site).astype(np.int64)
            report = build_token_shard(
                positions=positions,
                point_ids=point_ids,
                site=args.site,
                shard=args.shard,
                shard_size=args.token_shard_size,
                bs_positions=setup.bs_positions,
                geometry=geometry,
                aligned=aligned,
                config=config,
                device=args.device,
                force=args.force,
            )
        elif args.stage == "assemble":
            report = assemble_token_shards(
                site_point_ids={
                    int(site): np.flatnonzero(site_id == site).astype(np.int64)
                    for site in np.unique(site_id)
                },
                point_count=len(positions),
                shard_size=args.token_shard_size,
                config=config,
            )
            atomic_json(ASSEMBLY_REPORT, report)
        else:
            raise ValueError(args.stage)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

