"""Build E211 raw PathTokens with complete three-seed complex path state."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import resource
import time
from typing import Any

import numpy as np

from project_paths import PATHS
from round2.data import load_positions
from round2.physical_prior.geometry import GeometryData, load_geometry
from round2.physical_prior.source_port_common import E185_SEEDS
from round2.physical_prior import source_port_tokens as legacy


VERSION = "Round2-E211-full-physical-path-state-residual"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e211_full_path_state_residual"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e211_full_path_state_residual"
SUPPORT_DIR = ARTIFACT_DIR / "support"
TOKEN_DIR = ARTIFACT_DIR / "token_shards"
RAW_TOKEN_CACHE = ARTIFACT_DIR / "raw_tokens_n4000_k512_full_state.npz"
TOKEN_REPORT = ANALYSIS_DIR / "raw_token_report.json"
GEOMETRY_CACHE = legacy.GEOMETRY_CACHE
REFERENCE_RAW_TOKEN_CACHE = (
    PATHS.artifacts
    / "round2"
    / "e210_source_port_full_channel"
    / "raw_tokens_n4000_k512.npz"
)

STATE_COUNT = len(E185_SEEDS)
STATE_COORDINATE_DIM = 5
STATE_PORTS = 8

SUPPORT_FIELDS = (
    *legacy.SUPPORT_FIELDS,
    "absolute_energy",
    "complex_coefficient",
)


def _configure_legacy() -> None:
    legacy.VERSION = VERSION
    legacy.ARTIFACT_DIR = ARTIFACT_DIR
    legacy.ANALYSIS_DIR = ANALYSIS_DIR
    legacy.SUPPORT_DIR = SUPPORT_DIR
    legacy.TOKEN_DIR = TOKEN_DIR
    legacy.RAW_TOKEN_CACHE = RAW_TOKEN_CACHE
    legacy.TOKEN_REPORT = TOKEN_REPORT
    legacy.SUPPORT_FIELDS = SUPPORT_FIELDS


_configure_legacy()

atomic_json = legacy.atomic_json
atomic_npz = legacy.atomic_npz
batch_count = legacy.batch_count
build_support_worker = legacy.build_support_worker
complete = legacy.complete
support_path = legacy.support_path
token_shard_path = legacy.token_shard_path
report_path = legacy.report_path


def _empty_state_arrays(point_count: int) -> dict[str, np.ndarray]:
    shape = (point_count, legacy.RAW_CANDIDATES, STATE_COUNT)
    return {
        "state_mask": np.zeros(shape, dtype=bool),
        "state_coordinate": np.zeros(
            (*shape, STATE_COORDINATE_DIM), dtype=np.float32
        ),
        "state_relative_energy": np.zeros(shape, dtype=np.float32),
        "state_absolute_energy": np.zeros(shape, dtype=np.float32),
        "state_coefficient": np.zeros((*shape, STATE_PORTS), dtype=np.complex64),
    }


def _token_identity(
    arrays: dict[str, np.ndarray], row: int, index: int, site: int
) -> tuple[Any, ...] | None:
    source = int(arrays["source"][row, index])
    if source < 0:
        return None
    config_name = "base" if source == 0 else "scatter"
    delay, horizontal, vertical = legacy._path_coordinate(
        float(arrays["tau"][row, index]),
        float(arrays["theta_t"][row, index]),
        float(arrays["phi_t"][row, index]),
        config_name,
        site,
    )
    depth = min(int(arrays["interaction_count"][row, index]), legacy.MAX_DEPTH)
    sequence = tuple(
        (
            int(arrays["interactions"][row, index, step]),
            int(arrays["surface_family"][row, index, step]),
        )
        for step in range(depth)
    )
    scale = 50.0 if source == 0 else 4.0
    boundary = sequence if source == 0 or depth <= 2 else (sequence[0], sequence[-1])
    return (
        source,
        int(round(delay * scale)),
        int(round(horizontal * scale)),
        int(round(vertical * scale)),
        depth,
        boundary,
    )


def _periodic_distance(left: float, right: float, period: float) -> float:
    delta = abs(left - right) % period
    return min(delta, period - delta)


def _match_cost(
    reference: dict[str, np.ndarray],
    current: dict[str, np.ndarray],
    row: int,
    reference_index: int,
    current_index: int,
) -> float:
    cost = 0.0
    cost += (
        float(reference["tau"][row, reference_index])
        - float(current["tau"][row, current_index])
    ) ** 2 * 1e14
    for name in ("theta_t", "theta_r"):
        cost += (
            float(reference[name][row, reference_index])
            - float(current[name][row, current_index])
        ) ** 2
    for name in ("phi_t", "phi_r"):
        cost += _periodic_distance(
            float(reference[name][row, reference_index]),
            float(current[name][row, current_index]),
            2.0 * np.pi,
        ) ** 2
    reference_depth = int(reference["interaction_count"][row, reference_index])
    current_depth = int(current["interaction_count"][row, current_index])
    depth = min(reference_depth, current_depth, legacy.MAX_DEPTH)
    if depth:
        interaction_difference = np.count_nonzero(
            reference["interactions"][row, reference_index, :depth]
            != current["interactions"][row, current_index, :depth]
        )
        family_difference = np.count_nonzero(
            reference["surface_family"][row, reference_index, :depth]
            != current["surface_family"][row, current_index, :depth]
        )
        cost += 4.0 * float(interaction_difference + family_difference)
    if int(reference["seed"][row, reference_index]) != int(
        current["seed"][row, current_index]
    ):
        cost += 0.05
    return cost


def _align_states_to_reference(
    current: dict[str, np.ndarray], train_site: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not REFERENCE_RAW_TOKEN_CACHE.is_file():
        raise FileNotFoundError(REFERENCE_RAW_TOKEN_CACHE)
    token_dtypes, _ = legacy._token_helpers()
    with np.load(REFERENCE_RAW_TOKEN_CACHE, allow_pickle=False) as loaded:
        reference = {name: np.asarray(loaded[name]) for name in token_dtypes}
    output = {name: value for name, value in reference.items()}
    output.update(_empty_state_arrays(len(train_site)))
    matched = 0
    valid_reference = 0
    matched_reference_energy = 0.0
    total_reference_energy = 0.0
    copied_state_members = 0
    incomplete_rows = 0
    state_names = tuple(_empty_state_arrays(0))

    for row in range(len(train_site)):
        current_bank: dict[tuple[Any, ...], list[int]] = {}
        for index in np.flatnonzero(current["mask"][row]):
            key = _token_identity(current, row, int(index), int(train_site[row]))
            if key is not None:
                current_bank.setdefault(key, []).append(int(index))
        reference_bank: dict[tuple[Any, ...], list[int]] = {}
        for index in np.flatnonzero(reference["mask"][row]):
            index = int(index)
            if int(reference["source"][row, index]) < 0:
                continue
            key = _token_identity(reference, row, index, int(train_site[row]))
            if key is not None:
                reference_bank.setdefault(key, []).append(index)
            valid_reference += 1
            total_reference_energy += float(reference["relative_energy"][row, index])

        row_matched = 0
        for key, reference_indices in reference_bank.items():
            current_indices = current_bank.get(key, [])
            if not current_indices:
                continue
            pairs = sorted(
                (
                    _match_cost(reference, current, row, reference_index, current_index),
                    reference_index,
                    current_index,
                )
                for reference_index in reference_indices
                for current_index in current_indices
            )
            used_reference: set[int] = set()
            used_current: set[int] = set()
            for _, reference_index, current_index in pairs:
                if reference_index in used_reference or current_index in used_current:
                    continue
                for name in state_names:
                    output[name][row, reference_index] = current[name][row, current_index]
                used_reference.add(reference_index)
                used_current.add(current_index)
                row_matched += 1
                matched_reference_energy += float(
                    reference["relative_energy"][row, reference_index]
                )
                copied_state_members += int(
                    current["state_mask"][row, current_index].sum()
                )
        matched += row_matched
        if row_matched < sum(len(value) for value in reference_bank.values()):
            incomplete_rows += 1

    if not np.array_equal(output["mask"], reference["mask"]):
        raise RuntimeError("E211 aligned mask changed from E210")
    record = {
        "reference_candidate_count": int(valid_reference),
        "matched_candidate_count": int(matched),
        "matched_candidate_fraction": float(matched / max(valid_reference, 1)),
        "matched_reference_energy_fraction": float(
            matched_reference_energy / max(total_reference_energy, 1e-30)
        ),
        "copied_state_members": int(copied_state_members),
        "incomplete_point_count": int(incomplete_rows),
        "reference": PATHS.relative(REFERENCE_RAW_TOKEN_CACHE),
    }
    return output, record


def _candidate(support: dict[str, np.ndarray], index: int) -> dict[str, Any]:
    return {
        "support": support,
        "index": int(index),
        "presence": 1,
        "members": {},
    }


def _replace_representative(
    group: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    members = group["members"]
    presence = group["presence"]
    candidate["members"] = members
    candidate["presence"] = presence
    return candidate


def _write_state(
    arrays: dict[str, np.ndarray], row: int, slot: int, record: dict[str, Any]
) -> None:
    legacy._write_record(
        arrays,
        row,
        slot,
        record,
        _write_state.geometry,
        _write_state.plane_lookup,
    )
    for state_index, seed in enumerate(E185_SEEDS):
        member = record["members"].get(seed)
        if member is None:
            continue
        support = member["support"]
        index = member["index"]
        arrays["state_mask"][row, slot, state_index] = True
        arrays["state_coordinate"][row, slot, state_index] = np.asarray(
            (
                support["tau"][index],
                support["angles"][index, 0],
                support["angles"][index, 1],
                support["angles"][index, 2],
                support["angles"][index, 3],
            ),
            dtype=np.float32,
        )
        arrays["state_relative_energy"][row, slot, state_index] = max(
            float(support["relative_energy"][index]), 0.0
        )
        arrays["state_absolute_energy"][row, slot, state_index] = max(
            float(support["absolute_energy"][index]), 0.0
        )
        coefficient = np.asarray(
            support["complex_coefficient"][index], dtype=np.complex64
        ).reshape(-1)
        if coefficient.shape != (STATE_PORTS,):
            raise ValueError(f"unexpected physical coefficient shape {coefficient.shape}")
        arrays["state_coefficient"][row, slot, state_index] = coefficient


def build_token_shard(site: int, batch: int, force: bool = False) -> dict[str, Any]:
    output = token_shard_path(site, batch)
    if complete(output) and not force:
        return json.loads(report_path(output).read_text(encoding="utf-8"))
    supports = {
        (config, seed): legacy._load_support(config, seed, site, batch)
        for config in legacy.CONFIGS
        for seed in E185_SEEDS
    }
    reference_ids = supports[("base", E185_SEEDS[0])]["point_ids"].astype(
        np.int64
    )
    for support in supports.values():
        if not np.array_equal(reference_ids, support["point_ids"]):
            raise RuntimeError(f"E211 support point mismatch site={site} batch={batch}")

    geometry = load_geometry(GEOMETRY_CACHE)
    plane_lookup = legacy._family_plane_keys(geometry)
    _, empty_token_arrays = legacy._token_helpers()
    arrays = empty_token_arrays(
        len(reference_ids), legacy.RAW_CANDIDATES, legacy.MAX_DEPTH
    )
    arrays.update(_empty_state_arrays(len(reference_ids)))
    _write_state.geometry = geometry
    _write_state.plane_lookup = plane_lookup

    started = time.time()
    base_counts: list[int] = []
    scatter_counts: list[int] = []
    for row in range(len(reference_ids)):
        base_bank: dict[tuple[Any, ...], dict[str, Any]] = {}
        for seed in E185_SEEDS:
            support = supports[("base", seed)]
            for index in legacy._point_indices(support, row):
                candidate = _candidate(support, int(index))
                key = legacy._base_identity(support, int(index), site)
                group = base_bank.get(key)
                if group is None:
                    candidate["members"][seed] = candidate
                    base_bank[key] = candidate
                    continue
                group["members"][seed] = candidate
                group["presence"] += 1
                if legacy._energy(candidate) > legacy._energy(group):
                    base_bank[key] = _replace_representative(group, candidate)
        base = sorted(
            base_bank.values(),
            key=lambda value: legacy._energy(value)
            * (1.0 + 0.05 * (value["presence"] - 1)),
            reverse=True,
        )[: legacy.BASE_POOL]

        scatter_members: dict[tuple[Any, ...], dict[int, dict[str, Any]]] = {}
        scatter: list[dict[str, Any]] = []
        for seed in E185_SEEDS:
            support = supports[("scatter", seed)]
            local: dict[tuple[Any, ...], dict[str, Any]] = {}
            for index in legacy._point_indices(support, row):
                candidate = _candidate(support, int(index))
                key = legacy._scatter_identity(support, int(index), site)
                previous = local.get(key)
                if previous is None or legacy._energy(candidate) > legacy._energy(
                    previous
                ):
                    local[key] = candidate
            for key, candidate in local.items():
                scatter_members.setdefault(key, {})[seed] = candidate
            scatter.extend(local.values())
            scatter = sorted(scatter, key=legacy._energy, reverse=True)[
                : legacy.SCATTER_POOL
            ]
        scatter = sorted(scatter, key=legacy._energy, reverse=True)[
            : legacy.SCATTER_POOL
        ]
        for candidate in scatter:
            support = candidate["support"]
            index = candidate["index"]
            key = legacy._scatter_identity(support, index, site)
            candidate["members"] = scatter_members[key]

        base_counts.append(len(base))
        scatter_counts.append(len(scatter))
        selected = [*base, *scatter]
        for slot, record in enumerate(selected[: legacy.RAW_CANDIDATES]):
            _write_state(arrays, row, slot, record)
        valid = arrays["mask"][row]
        total = float(np.sum(arrays["relative_energy"][row, valid], dtype=np.float64))
        if total > 0.0:
            arrays["relative_energy"][row, valid] /= np.float32(total)
        if not np.any(valid):
            arrays["mask"][row, 0] = True
            arrays["source"][row, 0] = -1

    atomic_npz(output, point_ids=reference_ids, **arrays)
    record = {
        "status": "complete",
        "version": VERSION,
        "site": int(site),
        "batch": int(batch),
        "point_count": int(len(reference_ids)),
        "base_candidate_mean": float(np.mean(base_counts)),
        "scatter_candidate_mean": float(np.mean(scatter_counts)),
        "state_member_mean": float(np.mean(arrays["state_mask"].sum(axis=2))),
        "empty_points": int(np.sum(arrays["source"][:, 0] < 0)),
        "seconds": float(time.time() - started),
        "output": PATHS.relative(output),
    }
    atomic_json(report_path(output), record)
    return record


def build_all_token_shards(workers: int, force: bool = False) -> dict[str, Any]:
    jobs = [
        (site, batch)
        for site in range(2)
        for batch in range(batch_count(site))
        if force or not complete(token_shard_path(site, batch))
    ]
    started = time.time()
    records: list[dict[str, Any]] = []
    if jobs:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(build_token_shard, site, batch, force): (site, batch)
                for site, batch in jobs
            }
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                print(
                    f"E211 tokens site={record['site']} batch={record['batch']} "
                    f"seconds={record['seconds']:.2f}",
                    flush=True,
                )
    return {
        "status": "complete",
        "jobs": len(records),
        "workers": int(workers),
        "seconds": float(time.time() - started),
    }


def assemble_tokens(force: bool = False) -> dict[str, Any]:
    if RAW_TOKEN_CACHE.is_file() and TOKEN_REPORT.is_file() and not force:
        return json.loads(TOKEN_REPORT.read_text(encoding="utf-8"))
    _, positions, _, train_site, _ = load_positions()
    token_dtypes, empty_token_arrays = legacy._token_helpers()
    arrays = empty_token_arrays(len(positions), legacy.RAW_CANDIDATES, legacy.MAX_DEPTH)
    arrays.update(_empty_state_arrays(len(positions)))
    assigned = np.zeros(len(positions), dtype=bool)
    started = time.time()
    for site in range(2):
        for batch in range(batch_count(site)):
            path = token_shard_path(site, batch)
            if not complete(path):
                raise FileNotFoundError(f"incomplete E211 token shard: {path}")
            with np.load(path, allow_pickle=False) as cache:
                ids = cache["point_ids"].astype(np.int64)
                if np.any(assigned[ids]) or not np.all(train_site[ids] == site):
                    raise RuntimeError("E211 token shard identity mismatch")
                for name in (*token_dtypes, *_empty_state_arrays(0)):
                    arrays[name][ids] = cache[name]
                assigned[ids] = True
    if not np.all(assigned):
        raise RuntimeError(f"E211 missed {int(np.sum(~assigned))} train points")
    arrays, alignment = _align_states_to_reference(arrays, train_site)
    mask = arrays["mask"]
    state_mask = arrays["state_mask"]
    if not np.isfinite(arrays["state_coefficient"][state_mask]).all():
        raise FloatingPointError("E211 physical coefficients are non-finite")
    if np.any(state_mask & ~mask[..., None]):
        raise RuntimeError("E211 state exists outside a valid candidate")
    atomic_npz(RAW_TOKEN_CACHE, **arrays)
    valid_count = mask.sum(axis=1)
    record = {
        "status": "complete",
        "version": VERSION,
        "point_count": int(len(positions)),
        "shape": list(mask.shape),
        "valid_candidate_mean": float(np.mean(valid_count)),
        "physical_state_member_mean": float(np.mean(state_mask.sum(axis=2))),
        "physical_state_members": int(np.sum(state_mask)),
        "adapter_fallback_points": int(
            np.sum(np.any(mask & (arrays["source"] < 0), axis=1))
        ),
        "reference_alignment": alignment,
        "seconds": float(time.time() - started),
        "peak_rss_gib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        / (1024.0 * 1024.0),
        "output": PATHS.relative(RAW_TOKEN_CACHE),
    }
    atomic_json(TOKEN_REPORT, record)
    return record


__all__ = [
    "ANALYSIS_DIR",
    "ARTIFACT_DIR",
    "RAW_TOKEN_CACHE",
    "TOKEN_REPORT",
    "VERSION",
    "assemble_tokens",
    "batch_count",
    "build_all_token_shards",
    "build_support_worker",
    "build_token_shard",
    "complete",
    "support_path",
    "token_shard_path",
]
