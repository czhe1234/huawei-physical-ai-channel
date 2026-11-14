#!/usr/bin/env python3
"""Fit strict complex ridge stacks over the diverse official-data carrier bank."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import numpy as np
import torch

from build_v7_complex_carrier_bank import METADATA as BANK_METADATA
from build_v7_complex_carrier_bank import OOF_BANK
from data import load_positions
from project_paths import PATHS
from run_rbf_spectral_field import OOF_META
from run_v6_regime_component_lifecycle import REGIME_CACHE


VERSION = "Round2-E45-complex-carrier-ridge-stacking"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "complex_carrier_stacking"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_complex_carrier_stacking"
GRAM_CACHE = ARTIFACT_DIR / "oof_port_gram.npz"
SELECTION = ARTIFACT_DIR / "complex_stack_selection.json"
REPORT = ANALYSIS_DIR / "complex_carrier_stacking.json"


@dataclass(frozen=True)
class StackConfig:
    scope: str
    ports: str
    ridge: float
    sum_penalty: float

    @property
    def name(self) -> str:
        return (
            f"{self.scope}_{self.ports}_ridge{self.ridge:g}_sum{self.sum_penalty:g}"
        )


def candidate_configs() -> list[StackConfig]:
    return [
        StackConfig(scope, ports, ridge, penalty)
        for scope in ("global", "site", "site_regime")
        for ports in ("shared", "port8")
        for ridge in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
        for penalty in (0.0, 0.1, 1.0, 10.0)
    ]


def compute_gram(device: torch.device) -> dict[str, np.ndarray]:
    if GRAM_CACHE.is_file():
        cached = np.load(GRAM_CACHE, allow_pickle=False)
        required = {"gram", "cross", "energy", "query_ids", "fold_id"}
        if required.issubset(cached.files):
            return {key: cached[key] for key in cached.files}
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    bank = np.load(OOF_BANK, mmap_mode="r")
    truth_all = np.load(PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r")
    count = len(query_ids)
    carriers = len(bank)
    gram = np.empty((count, 8, carriers, carriers), dtype=np.complex64)
    cross = np.empty((count, 8, carriers), dtype=np.complex64)
    energy = np.empty((count, 8), dtype=np.float64)
    for row, query_id in enumerate(query_ids):
        candidate = torch.as_tensor(
            np.array(bank[:, row], dtype=np.complex64, copy=True), device=device
        ).reshape(carriers, 2, 16, 8, 4, 192)
        truth = torch.as_tensor(
            np.array(truth_all[query_id], dtype=np.complex64, copy=True), device=device
        ).reshape(2, 16, 8, 4, 192)
        with torch.inference_mode():
            candidate = candidate.permute(1, 4, 0, 2, 3, 5).reshape(
                8, carriers, -1
            )
            truth = truth.permute(0, 3, 1, 2, 4).reshape(8, -1)
            gram[row] = torch.einsum(
                "gcd,ged->gce", candidate.conj(), candidate
            ).cpu().numpy()
            cross[row] = torch.einsum(
                "gcd,gd->gc", candidate.conj(), truth
            ).cpu().numpy()
            energy[row] = torch.sum(truth.abs().square(), dim=1, dtype=torch.float64).cpu().numpy()
        if (row + 1) % 10 == 0 or row + 1 == count:
            print(f"carrier Gram {row + 1}/{count}", flush=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        GRAM_CACHE,
        gram=gram,
        cross=cross,
        energy=energy,
        query_ids=query_ids,
        fold_id=fold_id,
    )
    return {
        "gram": gram,
        "cross": cross,
        "energy": energy,
        "query_ids": query_ids,
        "fold_id": fold_id,
    }


def _scope_labels(
    config: StackConfig,
    site: np.ndarray,
    regime: np.ndarray,
) -> np.ndarray:
    if config.scope == "global":
        return np.zeros(len(site), dtype=np.int8)
    if config.scope == "site":
        return site.astype(np.int8)
    if config.scope == "site_regime":
        return (site * 4 + regime).astype(np.int8)
    raise ValueError(config.scope)


def _solve_weights(
    gram: np.ndarray,
    cross: np.ndarray,
    config: StackConfig,
) -> np.ndarray:
    carriers = gram.shape[-1]
    trace = float(np.trace(gram).real / carriers)
    scale = max(trace, 1e-20)
    system = np.asarray(gram, dtype=np.complex128).copy()
    right = np.asarray(cross, dtype=np.complex128).copy()
    system += np.eye(carriers, dtype=np.complex128) * (config.ridge * scale)
    if config.sum_penalty > 0.0:
        penalty = config.sum_penalty * scale
        system += penalty * np.ones((carriers, carriers), dtype=np.complex128)
        right += penalty * np.ones(carriers, dtype=np.complex128)
    try:
        return np.linalg.solve(system, right).astype(np.complex64)
    except np.linalg.LinAlgError:
        return (np.linalg.pinv(system) @ right).astype(np.complex64)


def fit_weights(
    cache: dict[str, np.ndarray],
    rows: np.ndarray,
    labels: np.ndarray,
    config: StackConfig,
) -> dict[int, np.ndarray]:
    gram = cache["gram"]
    cross = cache["cross"]
    output: dict[int, np.ndarray] = {}
    for label in np.unique(labels):
        local = rows[labels[rows] == label]
        if len(local) == 0 and config.scope == "site_regime":
            site = int(label) // 4
            local = rows[(labels[rows] // 4) == site]
        if len(local) == 0:
            local = rows
        if config.ports == "shared":
            total_gram = gram[local].sum(axis=(0, 1), dtype=np.complex128)
            total_cross = cross[local].sum(axis=(0, 1), dtype=np.complex128)
            output[int(label)] = _solve_weights(total_gram, total_cross, config)[None, :]
        elif config.ports == "port8":
            weights = []
            for port in range(8):
                total_gram = gram[local, port].sum(axis=0, dtype=np.complex128)
                total_cross = cross[local, port].sum(axis=0, dtype=np.complex128)
                weights.append(_solve_weights(total_gram, total_cross, config))
            output[int(label)] = np.stack(weights)
        else:
            raise ValueError(config.ports)
    return output


def point_errors(
    cache: dict[str, np.ndarray],
    rows: np.ndarray,
    labels: np.ndarray,
    weights: dict[int, np.ndarray],
    config: StackConfig,
) -> np.ndarray:
    gram = cache["gram"]
    cross = cache["cross"]
    energy = cache["energy"]
    output = np.zeros(len(rows), dtype=np.float64)
    for destination, row in enumerate(rows):
        local_weights = weights[int(labels[row])]
        if config.ports == "shared":
            local_weights = np.repeat(local_weights, 8, axis=0)
        value = 0.0
        for port in range(8):
            weight = local_weights[port].astype(np.complex128)
            value += float(
                np.vdot(weight, gram[row, port].astype(np.complex128) @ weight).real
                - 2.0 * np.vdot(weight, cross[row, port].astype(np.complex128)).real
                + energy[row, port]
            )
        output[destination] = max(value, 0.0)
    return output


def evaluate_config(
    cache: dict[str, np.ndarray],
    fit_rows: np.ndarray,
    query_rows: np.ndarray,
    labels: np.ndarray,
    config: StackConfig,
) -> dict[str, float]:
    weights = fit_weights(cache, fit_rows, labels, config)
    error = point_errors(cache, query_rows, labels, weights, config)
    truth_energy = cache["energy"][query_rows].sum()
    nmse = float(error.sum() / max(float(truth_energy), 1e-30))
    return {
        "nmse": nmse,
        "nmse_score": float(1.0 / (1.0 + nmse)),
        "error": float(error.sum()),
        "energy": float(truth_energy),
    }


def strict_nested_selection(
    cache: dict[str, np.ndarray],
    site: np.ndarray,
    regime: np.ndarray,
) -> dict[str, object]:
    configs = candidate_configs()
    fold_id = cache["fold_id"].astype(np.int8)
    records = []
    selected_names = []
    total_error = 0.0
    total_energy = 0.0
    for outer in np.unique(fold_id):
        outer_train = np.flatnonzero(fold_id != outer)
        held = np.flatnonzero(fold_id == outer)
        inner_scores = np.zeros(len(configs), dtype=np.float64)
        inner_energy = np.zeros(len(configs), dtype=np.float64)
        for inner in np.unique(fold_id[outer_train]):
            fit = outer_train[fold_id[outer_train] != inner]
            valid = outer_train[fold_id[outer_train] == inner]
            for config_index, config in enumerate(configs):
                labels = _scope_labels(config, site, regime)
                value = evaluate_config(cache, fit, valid, labels, config)
                inner_scores[config_index] += value["error"]
                inner_energy[config_index] += value["energy"]
        inner_nmse = inner_scores / np.maximum(inner_energy, 1e-30)
        selected_index = int(np.argmin(inner_nmse))
        config = configs[selected_index]
        labels = _scope_labels(config, site, regime)
        held_value = evaluate_config(cache, outer_train, held, labels, config)
        total_error += held_value["error"]
        total_energy += held_value["energy"]
        selected_names.append(config.name)
        records.append(
            {
                "fold": int(outer),
                "config": asdict_stack(config),
                "inner_nmse": float(inner_nmse[selected_index]),
                "held": held_value,
            }
        )
        print(
            f"complex stack outer={outer} config={config.name} "
            f"nmse={held_value['nmse']:.6f}",
            flush=True,
        )
    nmse = total_error / max(total_energy, 1e-30)
    return {
        "overall": {
            "nmse": float(nmse),
            "nmse_score": float(1.0 / (1.0 + nmse)),
        },
        "folds": records,
        "selected_names": selected_names,
    }


def asdict_stack(config: StackConfig) -> dict[str, object]:
    return {
        "name": config.name,
        "scope": config.scope,
        "ports": config.ports,
        "ridge": config.ridge,
        "sum_penalty": config.sum_penalty,
    }


def posterior_grid(
    cache: dict[str, np.ndarray],
    site: np.ndarray,
    regime: np.ndarray,
) -> list[dict[str, object]]:
    fold_id = cache["fold_id"].astype(np.int8)
    rows = np.arange(len(fold_id), dtype=np.int64)
    records = []
    for config in candidate_configs():
        labels = _scope_labels(config, site, regime)
        error = 0.0
        energy = 0.0
        fold_nmse = []
        for held_fold in np.unique(fold_id):
            fit = rows[fold_id != held_fold]
            held = rows[fold_id == held_fold]
            value = evaluate_config(cache, fit, held, labels, config)
            error += value["error"]
            energy += value["energy"]
            fold_nmse.append(value["nmse"])
        nmse = error / max(energy, 1e-30)
        records.append(
            {
                "config": asdict_stack(config),
                "nmse": float(nmse),
                "nmse_score": float(1.0 / (1.0 + nmse)),
                "fold_nmse": fold_nmse,
            }
        )
    records.sort(key=lambda row: float(row["nmse"]))
    return records


def single_carrier_controls(cache: dict[str, np.ndarray]) -> dict[str, object]:
    gram = cache["gram"]
    cross = cache["cross"]
    energy = cache["energy"]
    carrier_count = gram.shape[-1]
    errors = np.empty((carrier_count, len(gram)), dtype=np.float64)
    for carrier in range(carrier_count):
        errors[carrier] = (
            gram[:, :, carrier, carrier].real.sum(axis=1)
            - 2.0 * cross[:, :, carrier].real.sum(axis=1)
            + energy.sum(axis=1)
        )
    total_energy = float(energy.sum())
    nmse = errors.sum(axis=1) / max(total_energy, 1e-30)
    oracle = np.min(errors, axis=0).sum() / max(total_energy, 1e-30)
    metadata = json.loads(BANK_METADATA.read_text(encoding="utf-8"))
    names = metadata["config_names"]
    return {
        "single": [
            {
                "name": names[index],
                "nmse": float(nmse[index]),
                "nmse_score": float(1.0 / (1.0 + nmse[index])),
            }
            for index in np.argsort(nmse)
        ],
        "point_oracle_nmse": float(oracle),
        "point_oracle_nmse_score": float(1.0 / (1.0 + oracle)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("gram", "evaluate", "all"), default="all")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    cache = compute_gram(torch.device(args.device))
    if args.phase == "gram":
        print(
            json.dumps(
                {
                    "gram": PATHS.relative(GRAM_CACHE),
                    "shape": list(cache["gram"].shape),
                },
                indent=2,
            ),
            flush=True,
        )
        return
    _, _, _, train_site, _ = load_positions()
    query_ids = cache["query_ids"].astype(np.int64)
    site = train_site[query_ids].astype(np.int8)
    regime = np.load(REGIME_CACHE, allow_pickle=False)["oof_regime"].astype(np.int8)
    controls = single_carrier_controls(cache)
    grid = posterior_grid(cache, site, regime)
    nested = strict_nested_selection(cache, site, regime)
    payload = {
        "experiment": VERSION,
        "status": "analysis_only",
        "official_inputs_only": True,
        "sionna_used": False,
        "primary_hypothesis": (
            "The historical foundation carriers contain complementary complex phase and "
            "array organization; low-dimensional complex ridge stacking can recover a better "
            "carrier before the already-validated V4 PAS/PDP projection."
        ),
        "controls": controls,
        "posterior_top20": grid[:20],
        "nested": nested,
        "outputs": {
            "gram": PATHS.relative(GRAM_CACHE),
            "selection": PATHS.relative(SELECTION),
            "report": PATHS.relative(REPORT),
        },
        "seconds": float(time.time() - started),
    }
    selection = {
        "nested": nested,
        "posterior_best": grid[0],
        "candidate_count": len(candidate_configs()),
    }
    SELECTION.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
