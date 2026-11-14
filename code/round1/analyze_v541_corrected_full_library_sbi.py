#!/usr/bin/env python3
"""Compare old, corrected-array, and union physical SBI libraries."""

from __future__ import annotations

import json

import numpy as np

import analyze_v434_sionna_joint_sh_calibration as v434
import analyze_v494_sionna_configuration_stacking as v494
import analyze_v531_expanded_sbi_configuration_library as v531
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 541
VERSION = "v541_corrected_full_library_sbi"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
SOURCE_DIR = CACHE_DIR / "physical_sources"
OLD_REPORT = PATHS.version_report_dir(531) / "v531_expanded_sbi_configuration_library.json"
OLD_OOF = PATHS.cache_dir(531) / "v531_expanded_sbi_configuration_library_oof.npz"
V537_SOURCE_DIR = PATHS.cache_dir(537) / "physical_sources"

MIN_MEAN_GAIN = 0.001
MIN_FOLD_WINS = 4
MAX_REGIME_LOSS = 0.0005
CORRECTED_DRAWS = 8192
UNION_DRAWS = 32768
RANDOM_SEED = 20260725

CORRECTED_NAMES = (
    "corrected_robust",
    "corrected_depth16_f6p0",
    "corrected_depth5_f2p6",
    "corrected_depth5_f6p0",
    "corrected_depth5_f6p7",
    "corrected_depth5_all_f6p0",
    "corrected_depth10_scatter015_seed_marginal_3",
    "corrected_depth20_scatter015_seed_marginal_2",
    "corrected_depth16_nonsynthetic_edge_heavy",
)


def source_paths(name: str, split: str = "train", selection: str = "all") -> dict[str, object]:
    suffix = "" if selection == "all" else f"_{selection}"
    prefix = f"{name}_{split}{suffix}"
    return {
        "joint": SOURCE_DIR / f"{prefix}_joint_sh.npy",
        "log_power": SOURCE_DIR / f"{prefix}_log_power.npy",
        "done": SOURCE_DIR / f"{prefix}_done.npy",
        "indices": SOURCE_DIR / f"{prefix}_indices.npy",
    }


def load_complete(name: str) -> tuple[np.ndarray, np.ndarray]:
    paths = source_paths(name)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    done = np.load(paths["done"])
    ids = np.load(paths["indices"]).astype(np.int64)
    if not np.all(done) or not np.array_equal(ids, np.arange(2000)):
        raise ValueError(f"incomplete corrected source: {name}")
    value = np.load(paths["joint"], mmap_mode="r").reshape(2000, -1)
    total = np.exp(np.load(paths["log_power"]).astype(np.float64))
    return v494.normalize_rows(value), total.astype(np.float32)


def seed_mean(paths: tuple, fallback: np.ndarray) -> np.ndarray:
    values = []
    for path in paths:
        done_path = path.with_name(path.name.replace("_joint_sh.npy", "_done.npy"))
        if not path.is_file() or not done_path.is_file() or not np.all(np.load(done_path)):
            raise FileNotFoundError(path)
        values.append(v531.clean_source(np.load(path, mmap_mode="r").reshape(2000, -1), fallback))
    return v494.normalize_rows(np.mean(values, axis=0))


def load_corrected_sources() -> tuple[np.ndarray, list[np.ndarray], np.ndarray, list[str]]:
    truth, _ = v434.load_sh(v494.OFFICIAL)
    high, high_total = load_complete("corrected_depth16_f6p0")
    low_loaded = [
        load_complete(name)
        for name in (
            "corrected_depth5_f2p6",
            "corrected_depth5_f6p0",
            "corrected_depth5_f6p7",
        )
    ]
    lows = [item[0] for item in low_loaded]
    low_totals = [item[1] for item in low_loaded]
    robust, quality, _ = v434.robust_source(high, high_total, lows, low_totals)
    all_interaction, _ = load_complete("corrected_depth5_all_f6p0")
    shallow = seed_mean(
        tuple(
            V537_SOURCE_DIR / f"corrected_array_d10_sc015_seed{seed}_train_joint_sh.npy"
            for seed in (42, 206, 420)
        ),
        robust,
    )
    deep = seed_mean(
        tuple(
            SOURCE_DIR / f"corrected_depth20_scatter015_seed{seed}_train_joint_sh.npy"
            for seed in (42, 206)
        ),
        robust,
    )
    edge_paths = source_paths(
        "corrected_depth16_nonsynthetic_edge", selection="heavy"
    )
    for path in edge_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    edge_ids = np.load(edge_paths["indices"]).astype(np.int64)
    edge_done = np.load(edge_paths["done"])
    edge_rows = np.load(edge_paths["joint"], mmap_mode="r").reshape(len(edge_ids), -1)
    if not np.all(edge_done) or len(edge_rows) != len(edge_ids):
        raise ValueError("incomplete corrected heavy-edge source")
    edge = robust.copy()
    edge[edge_ids] = v531.clean_source(edge_rows, edge[edge_ids])
    sources = [robust, high, *lows, all_interaction, shallow, deep, edge]
    sources = [v531.clean_source(value, robust) for value in sources]
    return truth.astype(np.float32), sources, quality.astype(np.float32), list(CORRECTED_NAMES)


def edge_aware_draws(source_names: list[str], total_draws: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED)
    old_ids = np.asarray([index for index, name in enumerate(source_names) if name.startswith("old_")], dtype=np.int64)
    corrected_ids = np.asarray([index for index, name in enumerate(source_names) if name.startswith("corrected_")], dtype=np.int64)
    output: dict[str, np.ndarray] = {}
    for regime in v494.REGIMES:
        active = np.asarray(
            [
                index
                for index, name in enumerate(source_names)
                if regime == "heavy" or not name.endswith("edge_heavy")
            ],
            dtype=np.int64,
        )
        if len(old_ids) == 0 or len(corrected_ids) == 0:
            local = rng.dirichlet(np.full(len(active), 0.5), size=total_draws).astype(np.float32)
            draws = np.zeros((total_draws, len(source_names)), dtype=np.float32)
            draws[:, active] = local
            output[regime] = draws
            continue

        old_active = np.intersect1d(active, old_ids)
        corrected_active = np.intersect1d(active, corrected_ids)
        pure = total_draws // 4
        mixed = total_draws - 2 * pure
        draws = np.zeros((total_draws, len(source_names)), dtype=np.float32)
        draws[:pure, old_active] = rng.dirichlet(
            np.full(len(old_active), 0.5), size=pure
        ).astype(np.float32)
        draws[pure : 2 * pure, corrected_active] = rng.dirichlet(
            np.full(len(corrected_active), 0.5), size=pure
        ).astype(np.float32)
        old_weight = rng.dirichlet(np.full(len(old_active), 0.5), size=mixed)
        corrected_weight = rng.dirichlet(np.full(len(corrected_active), 0.5), size=mixed)
        gate = rng.beta(0.5, 0.5, size=mixed).astype(np.float32)
        draws[2 * pure :, old_active] = old_weight * gate[:, None]
        draws[2 * pure :, corrected_active] = corrected_weight * (1.0 - gate[:, None])
        output[regime] = draws
    return output


def run_v531_variant(
    name: str,
    loader,
    draws: int,
) -> tuple[dict[str, object], object]:
    variant_cache = CACHE_DIR / name
    variant_report = REPORT_DIR / f"{name}.json"
    variant_oof = variant_cache / f"{name}_oof.npz"
    v531.VERSION_NUMBER = VERSION_NUMBER
    v531.VERSION = name
    v531.REPORT_DIR = REPORT_DIR
    v531.REPORT = variant_report
    v531.CACHE_DIR = variant_cache
    v531.OOF_CACHE = variant_oof
    v531.load_expanded_sources = loader
    v531.prior_draws_by_regime = lambda names: edge_aware_draws(names, draws)
    v531.main()
    return json.loads(variant_report.read_text(encoding="utf-8")), variant_oof


def comparison_row(
    label: str,
    oof_path,
    old_score: np.ndarray,
    fold_id: np.ndarray,
    regimes: np.ndarray,
) -> tuple[dict[str, object], np.ndarray]:
    with np.load(oof_path) as cache:
        score = np.asarray(cache[f"{v531.HYBRID_METHOD}_score"], dtype=np.float64)
    gain = score - old_score
    fold_gains = [float(np.mean(gain[fold_id == fold])) for fold in range(5)]
    regime_gains = {
        regime: float(np.mean(gain[regimes == regime])) for regime in v494.REGIMES
    }
    row = {
        "library": label,
        "mean_score": float(np.mean(score)),
        "mean_gain_vs_v531": float(np.mean(gain)),
        "fold_gains": fold_gains,
        "fold_wins": int(sum(value > 0.0 for value in fold_gains)),
        "regime_gains": regime_gains,
    }
    row["pass"] = bool(
        row["mean_gain_vs_v531"] >= MIN_MEAN_GAIN
        and row["fold_wins"] >= MIN_FOLD_WINS
        and min(regime_gains.values()) >= -MAX_REGIME_LOSS
    )
    return row, score


def main() -> None:
    for path in (OLD_REPORT, OLD_OOF, v494.OFFICIAL, v494.POSITIONS, v434.OOF_CACHE):
        if not path.is_file():
            raise FileNotFoundError(path)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    corrected = load_corrected_sources()
    old = v531.load_expanded_sources()
    if not np.allclose(old[0], corrected[0]):
        raise ValueError("old and corrected truth arrays differ")

    corrected_report, corrected_oof = run_v531_variant(
        "v541_corrected_only_sbi",
        lambda include_nonmetal=False: corrected,
        CORRECTED_DRAWS,
    )
    union = (
        old[0],
        list(old[1]) + list(corrected[1]),
        np.maximum(old[2], corrected[2]),
        [f"old_{name}" for name in old[3]] + list(corrected[3]),
    )
    union_report, union_oof = run_v531_variant(
        "v541_old_corrected_union_sbi",
        lambda include_nonmetal=False: union,
        UNION_DRAWS,
    )

    with np.load(OLD_OOF) as cache:
        old_score = np.asarray(
            cache[f"{v531.HYBRID_METHOD}_score"], dtype=np.float64
        )
        fold_id = np.asarray(cache["fold_id"], dtype=np.int16)
    geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    regimes = regime_of(geometry["train_blocked"]).astype("U8")
    corrected_row, _ = comparison_row(
        "corrected_only", corrected_oof, old_score, fold_id, regimes
    )
    union_row, _ = comparison_row(
        "old_corrected_union", union_oof, old_score, fold_id, regimes
    )
    rows = [
        {
            "library": "old_v531",
            "mean_score": float(np.mean(old_score)),
            "mean_gain_vs_v531": 0.0,
            "fold_gains": [0.0] * 5,
            "fold_wins": 0,
            "regime_gains": {regime: 0.0 for regime in v494.REGIMES},
            "pass": False,
        },
        corrected_row,
        union_row,
    ]
    passing = [row for row in rows if row["pass"]]
    selected = (
        max(passing, key=lambda row: row["mean_gain_vs_v531"])
        if passing
        else None
    )
    report = {
        "method": VERSION,
        "status": "analysis_passed_deployment_gate" if selected else "analysis_negative",
        "primary_hypothesis": (
            "The old v531 library shares a systematic 0.5-wavelength vertical-array error. "
            "Rebuilding the complete library at 4 wavelengths with the independently inferred "
            "orientation should improve multiple propagation regimes, while the union posterior "
            "retains old-library modes that remain useful."
        ),
        "evidence_boundary": {
            "outer_split": "same five mutually exclusive spatial folds as v531",
            "test_sionna_used": False,
            "online_feedback_used_for_parameters": False,
            "corrected_quality": "derived from corrected high/three-frequency source consistency",
            "union_quality": "pointwise maximum of independently derived old and corrected quality",
            "union_prior": (
                "25% old-only, 25% corrected-only, 50% hierarchical old/corrected mixtures"
            ),
        },
        "gate": {
            "mean_gain_min": MIN_MEAN_GAIN,
            "fold_wins_min": MIN_FOLD_WINS,
            "each_regime_gain_min": -MAX_REGIME_LOSS,
        },
        "comparisons": rows,
        "selected_library": None if selected is None else selected["library"],
        "quality_summary": {
            "old_mean": float(np.mean(old[2])),
            "corrected_mean": float(np.mean(corrected[2])),
            "union_max_mean": float(np.mean(union[2])),
        },
        "variant_reports": {
            "corrected_only": PATHS.relative(REPORT_DIR / "v541_corrected_only_sbi.json"),
            "old_corrected_union": PATHS.relative(REPORT_DIR / "v541_old_corrected_union_sbi.json"),
        },
        "variant_decisions": {
            "corrected_only": corrected_report["decision_gate"],
            "old_corrected_union": union_report["decision_gate"],
        },
        "outputs": {
            "report": PATHS.relative(REPORT),
            "corrected_oof": PATHS.relative(corrected_oof),
            "union_oof": PATHS.relative(union_oof),
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
