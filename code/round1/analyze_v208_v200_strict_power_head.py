#!/usr/bin/env python3
"""Replay the v200 power head on a synthetic retrained spectral backbone."""

from __future__ import annotations

import json
from collections import Counter
from types import SimpleNamespace

import numpy as np

import analyze_v199_power_field_reliability as v199
import analyze_v203_sionna_like_scene_simulation as v203
import analyze_v205_full_pipeline_benchmark as pipeline
import analyze_v208_strict_historical_replay as v208
import generate_physical_local_linear_v11 as physical_v11
import generate_v196_train_guided_delay_pdp_mouth as v196
import generate_v200_selective_twosided_power_gate as v200
from generate_v171_full_complex_ad_tail_denoising import top_mask
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION = "v208_v200_strict_power_head"
REPORT_DIR = PATHS.version_report_dir(208)
REPORT = REPORT_DIR / "v208_v200_strict_power_head.json"
OBSERVED_TRAIN = PATHS.cache_dir(208) / "strict_dataset" / "Round1_Train_Channel.npy"
SCRATCH = PATHS.cache_dir(208) / "v208_replay_scratch_output.npy"
ORIGINAL_V200 = PATHS.cache_dir(200) / "v200_Round1_Test_Channel.npy"
BATCH = 8


def v9_args() -> SimpleNamespace:
    return SimpleNamespace(
        data=str(PATHS.cache_dir(208) / "strict_dataset"),
        output=str(SCRATCH),
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
    )


def save_and_score(value: np.ndarray) -> dict[str, object]:
    np.save(SCRATCH, np.asarray(value, dtype=np.complex64), allow_pickle=False)
    return v208.score_prediction(SCRATCH)


def exact_v196_projection(
    source: np.ndarray,
    observed_train,
    train_pos: np.ndarray,
    test_pos: np.ndarray,
    train_blocked: np.ndarray,
    test_blocked: np.ndarray,
    train_regimes: np.ndarray,
    test_regimes: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    train_total = v208.point_power_chunked(observed_train) / 192.0
    valid = train_total > 1e-20
    train_marginals = pipeline.train_delay_marginals(observed_train, valid)
    train_idx, train_weights, train_conf = v196.geometry_train_weights(
        train_pos,
        test_pos,
        train_blocked,
        test_blocked,
        train_regimes,
        test_regimes,
        train_marginals["valid"],
    )
    train_pred = v196.predict_train_marginals(train_marginals, train_idx, train_weights)

    support = np.empty((len(source), 2, 16, 8, 4, 192), dtype=np.float32)
    for start in range(0, len(source), BATCH):
        end = min(start + BATCH, len(source))
        power = v196.delay_power_from_h(source[start:end])
        support[start:end] = power / np.maximum(
            power.sum(axis=(1, 2, 3, 4, 5), keepdims=True), TINY
        )
    consensus, graph_diag = v196.build_test_consensus(
        support, test_pos, test_blocked, test_regimes
    )

    d0 = support.sum(axis=(1, 2, 3, 4), dtype=np.float64).astype(np.float32)
    du0 = support.sum(axis=(1, 2, 3), dtype=np.float64).astype(np.float32)
    dh0 = support.sum(axis=(1, 3, 4), dtype=np.float64).astype(np.float32)
    dc = consensus.sum(axis=(1, 2, 3, 4), dtype=np.float64).astype(np.float32)
    duc = consensus.sum(axis=(1, 2, 3), dtype=np.float64).astype(np.float32)
    dhc = consensus.sum(axis=(1, 3, 4), dtype=np.float64).astype(np.float32)
    agreement = graph_diag["test_neighbor_agreement"]
    alpha_d = v196.alpha_by_regime(test_regimes, v196.D_ALPHA, train_conf, agreement)
    alpha_du = v196.alpha_by_regime(test_regimes, v196.DU_ALPHA, train_conf, agreement)
    alpha_dh = v196.alpha_by_regime(test_regimes, v196.DH_ALPHA, train_conf, agreement)
    desired_d = v196.mix_target(d0, train_pred["d"], dc, test_regimes, alpha_d)
    desired_du = v196.mix_target(du0, train_pred["du"], duc, test_regimes, alpha_du)
    desired_dh = v196.mix_target(dh0, train_pred["dh"], dhc, test_regimes, alpha_dh)
    work = support.copy()
    for _ in range(v196.IPF_ITERS):
        v196.apply_delay_ratio(
            work,
            v196.safe_ratio(desired_d, work.sum(axis=(1, 2, 3, 4))),
            "d",
            0.92,
        )
        v196.apply_delay_ratio(
            work,
            v196.safe_ratio(desired_du, work.sum(axis=(1, 2, 3))),
            "du",
            0.58,
        )
        v196.apply_delay_ratio(
            work,
            v196.safe_ratio(desired_dh, work.sum(axis=(1, 3, 4))),
            "dh",
            0.44,
        )
        work = v196.normalize_rows(work.reshape(len(source), -1), axis=1).reshape(work.shape)

    output = np.empty_like(source)
    gain_max = np.empty(len(source), dtype=np.float32)
    gain_p90 = np.empty(len(source), dtype=np.float32)
    for start in range(0, len(source), BATCH):
        end = min(start + BATCH, len(source))
        h0 = source[start:end]
        cir0 = np.fft.ifft(h0, axis=-1)
        arr0 = cir0.reshape(end - start, 2, 16, 8, 4, 192)
        power0 = np.abs(arr0).astype(np.float32) ** 2
        total0 = np.maximum(power0.sum(axis=(1, 2, 3, 4, 5)), TINY).astype(np.float32)
        gain = np.sqrt(
            work[start:end] * total0[:, None, None, None, None, None]
            / np.maximum(power0, 1e-30)
        ).astype(np.float32)
        gain = np.clip(gain, 0.16, 6.40)
        flat = support[start:end].reshape(end - start, -1)
        strong128 = top_mask(flat, 128).reshape(power0.shape)
        strong1024 = top_mask(flat, 1024).reshape(power0.shape)
        strong8192 = top_mask(flat, 8192).reshape(power0.shape)
        gain[strong8192] = np.clip(gain[strong8192], 0.42, 2.65)
        gain[strong1024] = np.clip(gain[strong1024], 0.56, 1.88)
        gain[strong128] = np.clip(gain[strong128], 0.74, 1.42)
        for local, regime in enumerate(test_regimes[start:end].astype(str)):
            if regime == "clear":
                gain[local] = 1.0
            elif regime == "light":
                gain[local] = 1.0 + 0.42 * (gain[local] - 1.0)
        h1 = np.fft.fft(
            (arr0 * gain).reshape(end - start, 256, 4, 192), axis=-1
        )
        p0 = v208.point_power_chunked(h0)
        p1 = v208.point_power_chunked(h1)
        h1 *= np.sqrt(p0 / np.maximum(p1, 1e-30))[:, None, None, None].astype(np.float32)
        output[start:end] = h1.astype(np.complex64)
        gflat = gain.reshape(end - start, -1)
        gain_max[start:end] = np.max(gflat, axis=1)
        gain_p90[start:end] = np.quantile(gflat, 0.90, axis=1)
    return output, {
        "test_neighbor_agreement": agreement.astype(np.float32),
        "gain_max": gain_max,
        "gain_p90": gain_p90,
        "train_confidence": train_conf.astype(np.float32),
        "alpha_d": alpha_d,
        "alpha_du": alpha_du,
        "alpha_dh": alpha_dh,
    }


def synthetic_v199_diagnostics(
    base: np.ndarray,
    observed_train,
    train_pos: np.ndarray,
    test_pos: np.ndarray,
    train_blocked: np.ndarray,
    test_blocked: np.ndarray,
    train_regimes: np.ndarray,
    test_regimes: np.ndarray,
) -> dict[str, np.ndarray]:
    train_total = (v208.point_power_chunked(observed_train) / 192.0).astype(np.float32)
    valid_train = np.isfinite(train_total) & (train_total > 1e-20)
    train_log = np.log(np.maximum(train_total, TINY)).astype(np.float32)
    train_neighbors = v199.effective_neighbor_weights(
        test_pos,
        train_pos,
        test_blocked,
        train_blocked,
        test_regimes,
        train_regimes,
        valid_train,
        v199.K_TRAIN,
        exclude_self=False,
    )
    test_knn_log = np.sum(
        train_neighbors.weights * train_log[train_neighbors.idx], axis=1
    ).astype(np.float32)
    x_train = v199.feature_matrix(train_pos, train_blocked)
    x_test = v199.feature_matrix(test_pos, test_blocked)
    model_predictions: dict[str, np.ndarray] = {}
    for name, model in v199.build_models().items():
        model.fit(x_train[valid_train], train_log[valid_train])
        model_predictions[name] = model.predict(x_test).astype(np.float32)
    model_mean = np.mean(
        np.stack([model_predictions["extra"], model_predictions["hgb"]]), axis=0
    ).astype(np.float32)
    model_spread = np.abs(
        model_predictions["extra"] - model_predictions["hgb"]
    ).astype(np.float32)
    base_log = np.log(
        np.maximum(v208.point_power_chunked(base) / 192.0, TINY)
    ).astype(np.float32)
    consensus_log, consensus_diag = v199.test_power_consensus(
        base_log, test_pos, test_blocked, test_regimes
    )
    point = v199.classify_test_points(
        test_regimes,
        base_log,
        test_knn_log,
        model_mean,
        consensus_log,
        train_neighbors.confidence,
        consensus_diag["test_neighbor_agreement"],
        model_spread,
    )
    return {
        "regimes": test_regimes.astype("U8"),
        "test_neighbor_agreement": consensus_diag["test_neighbor_agreement"],
        **point,
    }


def exact_v200_log_delta(
    diag: dict[str, np.ndarray], actions: dict[str, np.ndarray]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    regimes = diag["regimes"].astype(str)
    labels = diag["power_label"].astype(str)
    suggested = diag["suggested_log_delta"].astype(np.float32)
    confidence = diag["power_confidence"].astype(np.float32)
    d_knn = diag["delta_knn"].astype(np.float32)
    d_model = diag["delta_model"].astype(np.float32)
    d_consensus = diag["delta_consensus"].astype(np.float32)
    test_agreement = diag["test_neighbor_agreement"].astype(np.float32)
    v196_agreement = actions["test_neighbor_agreement"].astype(np.float32)
    v196_gain_max = actions["gain_max"].astype(np.float32)
    final = np.zeros(len(regimes), dtype=np.float32)
    strict_gate = np.zeros(len(regimes), dtype=bool)
    for regime in ("clear", "light", "medium", "heavy"):
        mask = regimes == regime
        up = (
            mask
            & (labels == "power_up")
            & (confidence >= v200.CONF_MIN_UP[regime])
            & (d_knn > 0.06)
            & (d_model > 0.06)
            & (d_consensus > -0.85)
        )
        down = (
            mask
            & (labels == "power_down")
            & (confidence >= v200.CONF_MIN_DOWN[regime])
            & (d_knn < -0.06)
            & (d_model < -0.06)
            & (d_consensus < 0.85)
        )
        conf_gain = np.clip(0.75 + 0.40 * (confidence - 0.62) / 0.33, 0.75, 1.15)
        stable = np.clip(0.50 * test_agreement + 0.50 * v196_agreement, 0.0, 1.0)
        support_stress = np.clip((v196_gain_max - 4.50) / 2.20, 0.0, 1.0)
        stability_gain = np.clip(0.84 + 0.30 * stable - 0.14 * support_stress, 0.75, 1.10)
        up_value = np.clip(
            suggested * v200.UP_SCALE[regime] * conf_gain * stability_gain,
            0.0,
            v200.UP_CAP[regime],
        )
        down_value = np.clip(
            suggested * v200.DOWN_SCALE[regime] * conf_gain * stability_gain,
            v200.DOWN_CAP[regime],
            0.0,
        )
        final[up] = up_value[up]
        final[down] = down_value[down]
        strict_gate |= up | down
    active = np.abs(final) > 1e-7
    strong = active & (confidence >= 0.80) & (np.abs(d_knn) >= 1.0) & (np.abs(d_model) >= 1.0)
    strong &= np.sign(d_knn) == np.sign(d_model)
    final[strong] *= 1.08
    for regime in ("clear", "light", "medium", "heavy"):
        mask = regimes == regime
        final[mask & (final > 0)] = np.minimum(
            final[mask & (final > 0)], v200.UP_CAP[regime]
        )
        final[mask & (final < 0)] = np.maximum(
            final[mask & (final < 0)], v200.DOWN_CAP[regime]
        )
    return final, {
        "active": active,
        "strict_gate": strict_gate,
        "strong_agree": strong,
        "labels": labels.astype("U16"),
        "regimes": regimes.astype("U8"),
    }


def summarize_action(
    log_delta: np.ndarray, action: dict[str, np.ndarray], diag: dict[str, np.ndarray]
) -> dict[str, object]:
    active = action["active"].astype(bool)
    labels = action["labels"].astype(str)
    return {
        "active_count": int(np.sum(active)),
        "up_count": int(np.sum(log_delta > 0)),
        "down_count": int(np.sum(log_delta < 0)),
        "label_counts": dict(Counter(labels.tolist())),
        "active_by_label": {
            label: int(np.sum(active & (labels == label)))
            for label in ("power_up", "power_down", "conflict", "no_action")
        },
        "log_delta": pipeline.summarize(log_delta[active]),
        "power_confidence_active": pipeline.summarize(
            diag["power_confidence"][active]
        ),
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    required = (OBSERVED_TRAIN, ORIGINAL_V200)
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    original_score = v208.score_prediction(ORIGINAL_V200)
    physical_v11.generate(v9_args())
    base = np.asarray(np.load(SCRATCH, mmap_mode="r"), dtype=np.complex64)
    geom = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    train_blocked = geom["train_blocked"].astype(np.float32)
    test_blocked = geom["test_blocked"].astype(np.float32)
    train_regimes = regime_of(train_blocked)
    test_regimes = regime_of(test_blocked)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    matrix, graph_diag = pipeline.test_graph_matrix(test_pos, test_blocked, test_regimes)
    leaked, leaked_report = pipeline.leaked_multimarginal_projection(
        base, test_regimes, matrix, graph_diag
    )
    observed = np.load(OBSERVED_TRAIN, mmap_mode="r")
    delay, delay_actions = exact_v196_projection(
        leaked,
        observed,
        train_pos,
        test_pos,
        train_blocked,
        test_blocked,
        train_regimes,
        test_regimes,
    )
    v196_score = save_and_score(delay)
    power_diag = synthetic_v199_diagnostics(
        delay,
        observed,
        train_pos,
        test_pos,
        train_blocked,
        test_blocked,
        train_regimes,
        test_regimes,
    )
    log_delta, action = exact_v200_log_delta(power_diag, delay_actions)
    output = (
        delay * np.exp(0.5 * log_delta)[:, None, None, None].astype(np.float32)
    ).astype(np.complex64)
    v200_score = save_and_score(output)
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "interpretation_boundary": {
            "original_v200_direct_score": "non-strict; original official-data submission scored against synthetic truth",
            "synthetic_v200_power_head": (
                "v199 power models and v200 exact gate retrained on synthetic train; applied to a synthetic "
                "v9 -> v189-like -> exact-family v196 backbone. This isolates the best version's power head, "
                "but is not a byte-identical replay of the entire v1-v200 recursive ancestry."
            ),
        },
        "noise_scenario": "relative_25db with fixed floor and official 16 exact-zero rows",
        "original_v200_direct_to_synthetic_truth": original_score,
        "synthetic_backbone_v196": v196_score,
        "synthetic_v200_power_head": v200_score,
        "delta_vs_synthetic_v196": {
            "conventional": float(
                v200_score["score_conventional"] - v196_score["score_conventional"]
            ),
            "traceback": float(
                v200_score["score_traceback_leaked"]
                - v196_score["score_traceback_leaked"]
            ),
            "nmse_score": float(v200_score["nmse_score"] - v196_score["nmse_score"]),
            "power_spearman": float(
                v200_score["power_spearman"] - v196_score["power_spearman"]
            ),
        },
        "v200_action": summarize_action(log_delta, action, power_diag),
        "backbone_diagnostics": {
            "v189_like": leaked_report,
            "v196_gain_max": pipeline.summarize(delay_actions["gain_max"]),
            "v196_neighbor_agreement": pipeline.summarize(
                delay_actions["test_neighbor_agreement"]
            ),
        },
        "outputs": {
            "report": PATHS.relative(REPORT),
            "scratch_current_v200_output": PATHS.relative(SCRATCH),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
