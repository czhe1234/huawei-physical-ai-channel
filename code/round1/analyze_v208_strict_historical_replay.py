#!/usr/bin/env python3
"""Strictly retrain replayable historical models on the v206 Sionna dataset."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingRegressor

import analyze_v203_sionna_like_scene_simulation as v203
import analyze_v205_full_pipeline_benchmark as pipeline
import analyze_v205_sionna_config_calibration as calibration
import generate_derotated_idw as derotated
import generate_geometric_steering_blend as steering
import generate_path_scalar_mlp_blend as path_scalar
import generate_physical_local_linear as physical
import generate_physical_local_linear_v11 as physical_v11
import generate_reflection_steering as reflection
from powerfield_common import regime_of
from project_paths import PATHS


VERSION = "v208_strict_historical_replay"
OUT_DIR = PATHS.cache_dir(208)
DATASET_DIR = OUT_DIR / "strict_dataset"
CLEAN_TRAIN = OUT_DIR / "v208_sionna_train_clean_H.npy"
OBSERVED_TRAIN = DATASET_DIR / "Round1_Train_Channel.npy"
TEST_TRUTH = PATHS.cache_dir(207) / "v207_sionna_test_reference_H.npy"
SCRATCH_OUTPUT = OUT_DIR / "v208_replay_scratch_output.npy"
REPORT_DIR = PATHS.version_report_dir(208)
REPORT = REPORT_DIR / "v208_strict_historical_replay.json"
EXACT_CSV = REPORT_DIR / "v208_exact_replay_scores.csv"
CORRELATION_CSV = REPORT_DIR / "v208_exact_replay_correlations.csv"
FIGURE = REPORT_DIR / "v208_exact_replay_online_correlation.png"
BATCH = 8
ELEMENT_COUNT = 256 * 4 * 192
FIXED_NOISE_POWER_PER_ELEMENT = 1.3337086068966022e-19
NOISE_SCENARIOS = {
    "fitted_tail_upper_bound": {
        "kind": "fitted_relative_coefficient",
        "log_relative_coefficient": -17.045928611805405,
        "seed": 208_021,
    },
    "relative_25db": {
        "kind": "relative_snr_db",
        "snr_db": 25.0,
        "seed": 208_025,
    },
}
ONLINE_SCORES = {
    1: 0.450782,
    2: 0.437548,
    3: 0.454453,
    4: 0.451313,
    5: 0.450868,
    6: 0.458233,
    7: 0.456639,
    9: 0.458443,
    189: 0.510038,
    196: 0.511432,
    200: 0.511854,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Repeat the expensive end-to-end replay even when a complete report exists.",
    )
    return parser.parse_args()


def summarize(values: np.ndarray) -> dict[str, float | int]:
    value = np.asarray(values, dtype=np.float64)
    value = value[np.isfinite(value)]
    if len(value) == 0:
        return {"count": 0}
    return {
        "count": int(len(value)),
        "mean": float(np.mean(value)),
        "std": float(np.std(value)),
        "min": float(np.min(value)),
        "p10": float(np.quantile(value, 0.10)),
        "p50": float(np.quantile(value, 0.50)),
        "p90": float(np.quantile(value, 0.90)),
        "max": float(np.max(value)),
    }


def point_power_chunked(value, batch: int = 16) -> np.ndarray:
    output = np.empty(len(value), dtype=np.float64)
    for start in range(0, len(value), batch):
        block = np.asarray(value[start : start + batch], dtype=np.complex64)
        output[start : start + len(block)] = np.sum(
            np.abs(block) ** 2, axis=(1, 2, 3), dtype=np.float64
        )
    return output


def official_exact_zero_indices() -> np.ndarray:
    official = np.load(PATHS.round1 / "Round1_Train_Channel.npy", mmap_mode="r")
    return np.flatnonzero(point_power_chunked(official) == 0.0).astype(np.int64)


def noise_coefficient(config: dict[str, object]) -> float:
    if config["kind"] == "fitted_relative_coefficient":
        return float(math.exp(float(config["log_relative_coefficient"])))
    snr_db = float(config["snr_db"])
    return float(10.0 ** (-snr_db / 10.0) / ELEMENT_COUNT)


def write_observed_train(name: str, config: dict[str, object]) -> dict[str, object]:
    clean = np.load(CLEAN_TRAIN, mmap_mode="r")
    if clean.shape != (2000, 256, 4, 192) or clean.dtype != np.complex64:
        raise ValueError(f"Invalid clean train: {clean.shape} {clean.dtype}")
    coefficient = noise_coefficient(config)
    expected_total_ratio = coefficient * ELEMENT_COUNT
    expected_snr_db = -10.0 * math.log10(max(expected_total_ratio, 1e-30))
    exact_zero = official_exact_zero_indices()
    exact_zero_mask = np.zeros(len(clean), dtype=bool)
    exact_zero_mask[exact_zero] = True
    output = np.lib.format.open_memmap(
        OBSERVED_TRAIN,
        mode="w+",
        dtype=np.complex64,
        shape=clean.shape,
    )
    rng = np.random.default_rng(int(config["seed"]))
    actual_signal = 0.0
    actual_noise = 0.0
    for start in range(0, len(clean), BATCH):
        end = min(start + BATCH, len(clean))
        block = np.asarray(clean[start:end], dtype=np.complex64)
        power = np.sum(np.abs(block) ** 2, axis=(1, 2, 3), dtype=np.float64)
        sigma = np.sqrt(
            np.maximum(coefficient * power + FIXED_NOISE_POWER_PER_ELEMENT, 0.0)
        ).astype(np.float32)
        real = rng.standard_normal(block.shape, dtype=np.float32)
        imag = rng.standard_normal(block.shape, dtype=np.float32)
        noise = (real + 1j * imag) * (sigma[:, None, None, None] / np.sqrt(2.0))
        observed = (block + noise.astype(np.complex64)).astype(np.complex64)
        local_zero = exact_zero_mask[start:end]
        observed[local_zero] = 0.0
        output[start:end] = observed
        valid = ~local_zero
        actual_signal += float(np.sum(np.abs(block[valid]) ** 2, dtype=np.float64))
        actual_noise += float(np.sum(np.abs(noise[valid]) ** 2, dtype=np.float64))
    output.flush()
    observed_power = point_power_chunked(output)
    report = {
        "scenario": name,
        "seed": int(config["seed"]),
        "per_element_noise_power_over_point_power": coefficient,
        "fixed_noise_power_per_element": FIXED_NOISE_POWER_PER_ELEMENT,
        "expected_total_noise_to_signal_ratio": expected_total_ratio,
        "expected_snr_db": expected_snr_db,
        "actual_snr_db": float(10.0 * np.log10(actual_signal / max(actual_noise, 1e-30))),
        "official_exact_zero_indices_applied": exact_zero.tolist(),
        "observed_zero_count": int(np.sum(observed_power == 0.0)),
        "observed_log_power": summarize(np.log(np.maximum(observed_power, 1e-30))),
    }
    (DATASET_DIR / "active_noise_metadata.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def score_prediction(path: Path) -> dict[str, float | bool | list[int] | str]:
    pred = np.load(path, mmap_mode="r")
    truth = np.load(TEST_TRUTH, mmap_mode="r")
    if pred.shape != truth.shape or pred.dtype != np.complex64:
        raise ValueError(f"Invalid replay output {path}: {pred.shape} {pred.dtype}")
    accum = {key: [] for key in ("pas", "pdp", "leaked_sh", "ad_support", "complex_coherence")}
    pred_power = np.empty(len(pred), dtype=np.float64)
    truth_power = np.empty(len(pred), dtype=np.float64)
    error = 0.0
    energy = 0.0
    finite = True
    for start in range(0, len(pred), BATCH):
        end = min(start + BATCH, len(pred))
        prediction = np.asarray(pred[start:end], dtype=np.complex64)
        target = np.asarray(truth[start:end], dtype=np.complex64)
        finite = finite and bool(np.isfinite(prediction).all())
        target_views = calibration.spectral_views(target)
        scores = calibration.spectral_scores(prediction, target_views)
        p_power = np.sum(np.abs(prediction) ** 2, axis=(1, 2, 3), dtype=np.float64)
        t_power = target_views["energy"]
        pred_power[start:end] = p_power
        truth_power[start:end] = t_power
        true_zero = t_power <= 1e-25
        pred_zero = p_power <= 1e-25
        for key in accum:
            value = scores[key].copy()
            if key != "complex_coherence":
                value[true_zero & pred_zero] = 1.0
                value[true_zero & ~pred_zero] = 0.0
            accum[key].append(value)
        error += float(np.sum(np.abs(prediction - target) ** 2, dtype=np.float64))
        energy += float(np.sum(np.abs(target) ** 2, dtype=np.float64))
    merged = {key: np.concatenate(value) for key, value in accum.items()}
    nmse = error / max(energy, 1e-30)
    nmse_score = 1.0 / (1.0 + nmse)
    pas = float(np.mean(merged["pas"]))
    pdp = float(np.mean(merged["pdp"]))
    leaked = float(np.mean(merged["leaked_sh"]))
    power_spearman = float(spearmanr(np.log(np.maximum(pred_power, 1e-30)), np.log(np.maximum(truth_power, 1e-30))).statistic)
    return {
        "shape": list(pred.shape),
        "dtype": str(pred.dtype),
        "finite": finite,
        "pas": pas,
        "pdp": pdp,
        "leaked_sh": leaked,
        "ad_support": float(np.mean(merged["ad_support"])),
        "complex_coherence": float(np.mean(merged["complex_coherence"])),
        "global_nmse": nmse,
        "nmse_score": nmse_score,
        "power_spearman": power_spearman,
        "score_conventional": 0.4 * pas + 0.4 * pdp + 0.2 * nmse_score,
        "score_traceback_leaked": 0.4 * leaked + 0.4 * pdp + 0.2 * nmse_score,
    }


def exact_method_specs() -> list[dict[str, object]]:
    common_data = str(DATASET_DIR)
    scratch = str(SCRATCH_OUTPUT)
    return [
        {
            "version": 1,
            "name": "derotated_idw_k16_p2",
            "generate": derotated.generate,
            "args": SimpleNamespace(data=common_data, output=scratch, k=16, power=2.0, beta=140.3, batch=10),
        },
        {
            "version": 2,
            "name": "derotated_idw_k4_p3",
            "generate": derotated.generate,
            "args": SimpleNamespace(data=common_data, output=scratch, k=4, power=3.0, beta=140.3, batch=10),
        },
        {
            "version": 3,
            "name": "geometric_steering_k24",
            "generate": steering.generate,
            "args": SimpleNamespace(data=common_data, output=scratch, k=24, power=2.0, beta=140.3, alpha=0.5, steering_scale=4.71238898038, batch=10),
        },
        {
            "version": 4,
            "name": "geometric_steering_k8",
            "generate": steering.generate,
            "args": SimpleNamespace(data=common_data, output=scratch, k=8, power=2.0, beta=140.3, alpha=0.5, steering_scale=4.71238898038, batch=10),
        },
        {
            "version": 5,
            "name": "map_reflection_rho025",
            "generate": reflection.generate,
            "args": SimpleNamespace(
                data=common_data,
                features=str(PATHS.shared_features / "map_features.npz"),
                output=scratch,
                k=24,
                power=2.0,
                beta=140.3,
                reflection_rho=0.25,
                steering_scale=4.71238898038,
                batch=10,
            ),
        },
        {
            "version": 6,
            "name": "physical_local_linear",
            "generate": physical.generate,
            "args": SimpleNamespace(
                data=common_data,
                output=scratch,
                k=24,
                power=2.0,
                beta=140.3,
                gamma=0.15,
                alpha=1.0,
                ridge=0.3,
                steering_scale=4.71238898038,
                batch=10,
            ),
        },
        {
            "version": 7,
            "name": "path_scalar_mlp_c128_rho025",
            "generate": path_scalar.generate,
            "args": SimpleNamespace(
                data=common_data,
                output=scratch,
                map_features="",
                map_feature_indices="0,1,2,3,5,9,13,17,21,25,28,32,33,34,35,39,40,41",
                virtual_sources="",
                max_virtual_sources=80,
                delay_fit=str(PATHS.shared_features / "delay_length_fit.npz"),
                train_query_count=2000,
                seed=20260704,
                k=24,
                component_count=128,
                component_mode="weighted",
                per_neighbor=256,
                n_order="vp",
                power=2.0,
                beta=140.3,
                gamma=0.15,
                alpha=1.0,
                ridge=0.3,
                blend_rho=0.25,
                steering_scale=4.71238898038,
                clip_scalar=2.0,
                pred_clip_scalar=2.0,
                max_weight=20.0,
                device="cuda",
                epochs=20,
                batch_size=8192,
                infer_batch_size=65536,
                width=256,
                depth=3,
                model_kind="mlp",
                emb_h=8,
                emb_v=8,
                emb_uv=4,
                emb_s=16,
                num_experts=8,
                lr=2e-3,
                weight_decay=1e-4,
                target_mode="direct",
                loss="mse",
                huber_delta=0.25,
                print_every=5,
                batch=10,
                blend_rhos=[0.25, 0.5, 0.75, 1.0],
            ),
        },
        {
            "version": 9,
            "name": "physical_local_linear_v11_safe",
            "generate": physical_v11.generate,
            "args": SimpleNamespace(
                data=common_data,
                output=scratch,
                k=24,
                power=2.0,
                beta=140.3,
                gamma=0.10,
                alpha=1.0,
                ridge=0.3,
                steering_scale=4.71238898038,
                fallback_k=4,
                idw_k=16,
                ratio_small_thresh=0.50,
                ratio_idw_thresh=0.60,
                neg_frac_thresh=0.15,
                cos_thresh=0.99,
                gain_cap=1.25,
                severe_ratio_small_thresh=0.30,
                severe_neg_frac_thresh=0.30,
                mode="gain_only",
                blend_affine=0.25,
                batch=10,
            ),
        },
    ]


def run_exact_methods(scenario: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in exact_method_specs():
        version = int(spec["version"])
        print(f"strict replay {scenario} v{version} {spec['name']}", flush=True)
        started = time.time()
        generation = spec["generate"](spec["args"])
        metrics = score_prediction(SCRATCH_OUTPUT)
        row = {
            "scenario": scenario,
            "version": version,
            "method": str(spec["name"]),
            "online_score": ONLINE_SCORES[version],
            "seconds": float(time.time() - started),
            **metrics,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        del generation
        gc.collect()
        if version == 7:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
    return rows


def fit_power_models_streaming(observed_train, features, seed: int) -> dict[str, np.ndarray]:
    train_power = point_power_chunked(observed_train)
    train_log = np.log(np.maximum(train_power, 1e-30)).astype(np.float32)
    threshold, _ = pipeline.infer_low_state_threshold(train_log)
    train_nonzero = train_log > threshold
    x_train = np.concatenate([features.train_basic_features, features.train_path_features_public], axis=1)
    x_test = np.concatenate([features.test_basic_features, features.test_path_features_public], axis=1)
    x_train = np.nan_to_num(x_train, nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32)
    x_test = np.nan_to_num(x_test, nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32)
    classifier = ExtraTreesClassifier(
        n_estimators=500,
        min_samples_leaf=3,
        max_features=0.80,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    classifier.fit(x_train, train_nonzero.astype(np.int32))
    nonzero_probability = classifier.predict_proba(x_test)[:, 1].astype(np.float32)
    extra = ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=3,
        max_features=0.82,
        random_state=seed + 1,
        n_jobs=-1,
    )
    hgb = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.045,
        max_leaf_nodes=31,
        min_samples_leaf=12,
        l2_regularization=0.04,
        random_state=seed + 2,
    )
    extra.fit(x_train[train_nonzero], train_log[train_nonzero])
    hgb.fit(x_train[train_nonzero], train_log[train_nonzero])
    pred_extra = extra.predict(x_test).astype(np.float32)
    pred_hgb = hgb.predict(x_test).astype(np.float32)
    return {
        "train_nonzero": train_nonzero,
        "nonzero_probability": nonzero_probability,
        "predicted_log_power": (0.5 * pred_extra + 0.5 * pred_hgb).astype(np.float32),
        "power_spread": np.abs(pred_extra - pred_hgb).astype(np.float32),
    }


def stage_score(name: str, version: int, value: np.ndarray) -> dict[str, object]:
    np.save(SCRATCH_OUTPUT, np.asarray(value, dtype=np.complex64), allow_pickle=False)
    return {
        "stage": name,
        "mapped_version": version,
        "online_score": ONLINE_SCORES[version],
        **score_prediction(SCRATCH_OUTPUT),
    }


def run_advanced_stage_replay(scenario: str, exact_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if int(exact_rows[-1]["version"]) != 9:
        raise RuntimeError("v9 must remain the final exact replay output")
    base = np.asarray(np.load(SCRATCH_OUTPUT, mmap_mode="r"), dtype=np.complex64)
    stages = [
        {
            "stage": "v9_exact_base",
            "mapped_version": 9,
            "online_score": ONLINE_SCORES[9],
            **score_prediction(SCRATCH_OUTPUT),
        }
    ]
    positions = {
        "train": np.load(DATASET_DIR / "Round1_Train_Pos.npy").astype(np.float64),
        "test": np.load(DATASET_DIR / "Round1_Test_Pos.npy").astype(np.float64),
    }
    geom = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    train_blocked = geom["train_blocked"].astype(np.float32)
    test_blocked = geom["test_blocked"].astype(np.float32)
    train_regimes = regime_of(train_blocked)
    test_regimes = regime_of(test_blocked)
    matrix, graph_diag = pipeline.test_graph_matrix(positions["test"], test_blocked, test_regimes)
    observed_train = np.load(OBSERVED_TRAIN, mmap_mode="r")
    features = v203.load_inputs()

    print(f"advanced replay {scenario}: v189-like leaked", flush=True)
    leaked, leaked_report = pipeline.leaked_multimarginal_projection(
        base, test_regimes, matrix, graph_diag
    )
    stages.append(stage_score("v189_like_leaked_multimarginal", 189, leaked))

    print(f"advanced replay {scenario}: v196-like delay", flush=True)
    model = fit_power_models_streaming(observed_train, features, seed=208_100)
    delay, delay_report = pipeline.delay_pdp_projection(
        leaked,
        observed_train,
        model["train_nonzero"],
        positions["train"],
        positions["test"],
        train_blocked,
        test_blocked,
        train_regimes,
        test_regimes,
        matrix,
        graph_diag,
    )
    stages.append(stage_score("v196_like_train_guided_delay", 196, delay))

    print(f"advanced replay {scenario}: v200-like selective power", flush=True)
    power, power_report, _ = pipeline.apply_selective_power(
        delay,
        model["predicted_log_power"],
        model["nonzero_probability"],
        model["power_spread"],
    )
    stages.append(stage_score("v200_like_selective_power", 200, power))
    for index in range(1, len(stages)):
        stages[index]["delta_conventional_vs_previous"] = float(
            stages[index]["score_conventional"] - stages[index - 1]["score_conventional"]
        )
        stages[index]["delta_leaked_vs_previous"] = float(
            stages[index]["score_traceback_leaked"] - stages[index - 1]["score_traceback_leaked"]
        )
        stages[index]["online_delta_vs_previous"] = float(
            stages[index]["online_score"] - stages[index - 1]["online_score"]
        )
    stages[1]["action_report"] = leaked_report
    stages[2]["action_report"] = delay_report
    stages[3]["action_report"] = power_report
    del base, leaked, delay, power, model
    gc.collect()
    return stages


def safe_correlation(x: np.ndarray, y: np.ndarray, method: str) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 4 or np.ptp(x) <= 1e-15 or np.ptp(y) <= 1e-15:
        return None
    if method == "pearson":
        return float(pearsonr(x, y).statistic)
    if method == "spearman":
        return float(spearmanr(x, y).statistic)
    return float(kendalltau(x, y).statistic)


def pairwise_concordance(local: np.ndarray, online: np.ndarray) -> float:
    agree = 0
    total = 0
    for i in range(len(local)):
        for j in range(i + 1, len(local)):
            a = np.sign(local[i] - local[j])
            b = np.sign(online[i] - online[j])
            if a == 0 or b == 0:
                continue
            agree += int(a == b)
            total += 1
    return float(agree / total) if total else float("nan")


def correlation_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = (
        "score_conventional",
        "score_traceback_leaked",
        "pas",
        "pdp",
        "leaked_sh",
        "nmse_score",
        "power_spearman",
    )
    output = []
    for scenario in NOISE_SCENARIOS:
        subset = [row for row in rows if row["scenario"] == scenario]
        online = np.asarray([row["online_score"] for row in subset], dtype=np.float64)
        for metric in metrics:
            local = np.asarray([row[metric] for row in subset], dtype=np.float64)
            output.append(
                {
                    "scenario": scenario,
                    "count": len(subset),
                    "metric": metric,
                    "pearson": safe_correlation(local, online, "pearson"),
                    "spearman": safe_correlation(local, online, "spearman"),
                    "kendall": safe_correlation(local, online, "kendall"),
                    "pairwise_concordance": pairwise_concordance(local, online),
                }
            )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, len(NOISE_SCENARIOS), figsize=(12, 5), squeeze=False)
    for axis, scenario in zip(axes[0], NOISE_SCENARIOS):
        subset = [row for row in rows if row["scenario"] == scenario]
        x = np.asarray([row["score_traceback_leaked"] for row in subset])
        y = np.asarray([row["online_score"] for row in subset])
        axis.scatter(x, y, s=45)
        for row, xx, yy in zip(subset, x, y):
            axis.annotate(f"v{row['version']}", (xx, yy), xytext=(4, 3), textcoords="offset points")
        axis.set_title(scenario)
        axis.set_xlabel("synthetic traceback score")
        axis.set_ylabel("online score")
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)


def save_checkpoint(
    exact_rows: list[dict[str, object]],
    noise_reports: list[dict[str, object]],
    stage_runs: dict[str, list[dict[str, object]]],
    status: str,
) -> None:
    report = {
        "method": VERSION,
        "status": status,
        "strict_definition": (
            "Each exact-replay method is retrained or regenerated from the synthetic noisy train H and evaluated "
            "against independently generated clean test H at the official coordinates. Existing official-data "
            "submission outputs and learned caches are not used."
        ),
        "exact_versions": [1, 2, 3, 4, 5, 6, 7, 9],
        "noise_scenarios": noise_reports,
        "exact_replay": exact_rows,
        "advanced_stage_replay": stage_runs,
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    if REPORT.is_file() and not args.force:
        existing = json.loads(REPORT.read_text(encoding="utf-8"))
        if existing.get("status") == "analysis_only_no_candidate":
            print(
                json.dumps(
                    {
                        "status": "reused_complete_report",
                        "report": PATHS.relative(REPORT),
                        "hint": "Pass --force to repeat the strict replay.",
                    },
                    indent=2,
                )
            )
            return
    required = [
        CLEAN_TRAIN,
        TEST_TRUTH,
        DATASET_DIR / "Round1_Train_Pos.npy",
        DATASET_DIR / "Round1_Test_Pos.npy",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Run generate_v208_strict_sionna_dataset.py first: {missing}")

    exact_rows: list[dict[str, object]] = []
    noise_reports: list[dict[str, object]] = []
    stage_runs: dict[str, list[dict[str, object]]] = {}
    for scenario, config in NOISE_SCENARIOS.items():
        print(f"prepare observed train: {scenario}", flush=True)
        noise_reports.append(write_observed_train(scenario, config))
        scenario_rows = run_exact_methods(scenario)
        exact_rows.extend(scenario_rows)
        save_checkpoint(exact_rows, noise_reports, stage_runs, "analysis_in_progress")
        stage_runs[scenario] = run_advanced_stage_replay(scenario, scenario_rows)
        save_checkpoint(exact_rows, noise_reports, stage_runs, "analysis_in_progress")

    correlations = correlation_rows(exact_rows)
    write_csv(EXACT_CSV, exact_rows)
    write_csv(CORRELATION_CSV, correlations)
    make_figure(exact_rows)
    final_report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "If the calibrated Sionna configuration is a faithful competition surrogate, independently replayed "
            "historical models should preserve their online ordering, and late PAS/PDP/power stage gains should "
            "have the same signs as their online gains."
        ),
        "strict_definition": (
            "Exact versions are retrained or regenerated only from synthetic noisy train H and public geometry, "
            "then evaluated against independent clean synthetic test H. No historical submission output is scored "
            "or used as a base."
        ),
        "dataset_generation": PATHS.relative(
            REPORT_DIR / "v208_strict_sionna_dataset_generate.json"
        ),
        "exact_versions": [1, 2, 3, 4, 5, 6, 7, 9],
        "exact_replay_limitations": [
            "Later versions are recursive transformations of predecessor outputs and cannot be called independent models.",
            "The advanced v189/v196/v200 replay therefore tests retrained method families and gain signs, not byte-identical historical scripts.",
            "Official PAS/PDP implementation remains partially hidden, so both conventional PAS and traceback-leaked PAS composite scores are reported.",
            "The fitted-tail noise level is an upper bound because the observed official tail also contains weak true paths.",
        ],
        "noise_scenarios": noise_reports,
        "exact_replay": exact_rows,
        "correlations": correlations,
        "advanced_stage_replay": stage_runs,
        "outputs": {
            "report": PATHS.relative(REPORT),
            "exact_scores_csv": PATHS.relative(EXACT_CSV),
            "correlations_csv": PATHS.relative(CORRELATION_CSV),
            "figure": PATHS.relative(FIGURE),
            "clean_train": PATHS.relative(CLEAN_TRAIN),
            "active_observed_train": PATHS.relative(OBSERVED_TRAIN),
        },
    }
    REPORT.write_text(json.dumps(final_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"correlations": correlations, "stage_runs": stage_runs, "outputs": final_report["outputs"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
