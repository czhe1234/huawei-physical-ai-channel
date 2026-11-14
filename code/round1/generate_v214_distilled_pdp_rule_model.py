#!/usr/bin/env python3
"""Generate v214: distill v212 PDP rules into an official-feature student.

v212 is high-scoring but depends on Sionna-derived PDP references.  This
candidate treats v212 only as a teacher action trace and trains a student that
does not use Sionna references as input.  The student sees official/test-chain
features only: coordinates, blockage regime, current v209 delay shape, v196
train-guided anchor, v189/v196/v200/v209 diagnostics and power labels.

The target is the low-dimensional rule in v212-v209 PDP movement.  We fit
fold-local PCA bases for D, U-D and H-D teacher deltas and predict their
coefficients with ExtraTrees in out-of-fold mode.  Each test point is therefore
not reconstructed from its own teacher label; it is predicted from rules learned
from the other test points.

No Sionna H, Sionna phase, Sionna total power or Sionna feature array is used by
the student at generation time.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

import generate_v178_test_track_complex_consensus as track
import generate_v196_train_guided_delay_pdp_mouth as v196
from generate_v87_easy_field_projection import MH, MP, N_ANT, S_SC, summarize
from generate_v159_frontier_consensus_support_soup import digest
from generate_v169_power_scaled_tail_denoising import by_regime_summary
from generate_v171_full_complex_ad_tail_denoising import top_mask
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION = "v214_distilled_pdp_rule_model"

OUT_DIR = PATHS.cache_dir(214)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(214)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v214_Round1_Test_Channel.npy"
MARGINALS = OUT_DIR / "v214_distillation_marginals.npz"
ACTION_OUT = OUT_DIR / "v214_distilled_pdp_rule_actions.npz"
REPORT = REPORT_DIR / "v214_distilled_pdp_rule_model_generate.json"
ZIP_OUT = PATHS.round1_versions / "v214_distilled_pdp_rule_model_candidate_pending.zip"

BASE = PATHS.cache_dir(209) / "v209_Round1_Test_Channel.npy"
TEACHER = PATHS.cache_dir(212) / "v212_Round1_Test_Channel.npy"
V200_BASE = PATHS.cache_dir(200) / "v200_Round1_Test_Channel.npy"
V196_TARGET = PATHS.cache_dir(196) / "v196_train_guided_delay_target.npy"
V196_ACTIONS = PATHS.cache_dir(196) / "v196_train_guided_delay_actions.npz"
V199_DIAG = PATHS.cache_dir(199) / "v199_test_power_diagnostics.npz"
V200_ACTIONS = PATHS.cache_dir(200) / "v200_selective_twosided_power_gate_actions.npz"
V209_ACTIONS = PATHS.cache_dir(209) / "v209_adaptive_metric_projection_actions.npz"
V212_ACTIONS = PATHS.cache_dir(212) / "v212_multidomain_sionna_pdp_actions.npz"

FLOOR = 1e-12
BATCH = 6
IPF_ITERS = 7
N_COMPONENTS = {"d": 18, "du": 28, "dh": 36}


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def normalize(x: np.ndarray, axis: tuple[int, ...] | int) -> np.ndarray:
    out = np.maximum(np.asarray(x, dtype=np.float32), FLOOR)
    out /= np.maximum(out.sum(axis=axis, keepdims=True), TINY)
    return out.astype(np.float32)


def safe_ratio(desired: np.ndarray, current: np.ndarray) -> np.ndarray:
    return (np.maximum(desired, FLOOR) / np.maximum(current, FLOOR)).astype(np.float32)


def delay_support_from_h(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    power = v196.delay_power_from_h(h)
    total = power.sum(axis=(1, 2, 3, 4, 5), dtype=np.float64).astype(np.float32)
    support = power / np.maximum(total[:, None, None, None, None, None], TINY)
    return support.astype(np.float32), total


def support_marginals(support: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = support.sum(axis=(1, 2, 3, 4), dtype=np.float64).astype(np.float32)
    du = support.sum(axis=(1, 2, 3), dtype=np.float64).astype(np.float32)
    dh = support.sum(axis=(1, 3, 4), dtype=np.float64).astype(np.float32)
    return normalize(d, axis=1), normalize(du, axis=(1, 2)), normalize(dh, axis=(1, 2))


def l1_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(a.reshape(len(a), -1) - b.reshape(len(b), -1)), axis=1).astype(np.float32)


def d_stats(d: np.ndarray) -> np.ndarray:
    idx = np.arange(d.shape[1], dtype=np.float32)
    peak = np.argmax(d, axis=1).astype(np.float32) / float(d.shape[1] - 1)
    mean = np.sum(d * idx[None, :], axis=1) / float(d.shape[1] - 1)
    raw_mean = mean * float(d.shape[1] - 1)
    var = np.sum(d * (idx[None, :] - raw_mean[:, None]) ** 2, axis=1)
    spread = np.sqrt(np.maximum(var, 0.0)) / float(d.shape[1] - 1)
    entropy = -np.sum(d * np.log(np.maximum(d, FLOOR)), axis=1) / np.log(float(d.shape[1]))
    top1 = np.max(d, axis=1)
    early = d[:, :24].sum(axis=1)
    mid = d[:, 24:72].sum(axis=1)
    late = d[:, 72:].sum(axis=1)
    tail = d[:, 112:].sum(axis=1)
    return np.stack([peak, mean, spread, entropy, top1, early, mid, late, tail], axis=1).astype(np.float32)


def build_marginals() -> dict[str, np.ndarray]:
    if MARGINALS.is_file():
        data = np.load(MARGINALS)
        return {key: data[key] for key in data.files}
    base = np.load(BASE, mmap_mode="r")
    teacher = np.load(TEACHER, mmap_mode="r")
    anchor = np.load(V196_TARGET, mmap_mode="r")
    n = len(base)
    arrays = {
        "current_d": np.empty((n, S_SC), dtype=np.float32),
        "current_du": np.empty((n, N_ANT, S_SC), dtype=np.float32),
        "current_dh": np.empty((n, MH, S_SC), dtype=np.float32),
        "teacher_d": np.empty((n, S_SC), dtype=np.float32),
        "teacher_du": np.empty((n, N_ANT, S_SC), dtype=np.float32),
        "teacher_dh": np.empty((n, MH, S_SC), dtype=np.float32),
        "anchor_d": np.empty((n, S_SC), dtype=np.float32),
        "anchor_du": np.empty((n, N_ANT, S_SC), dtype=np.float32),
        "anchor_dh": np.empty((n, MH, S_SC), dtype=np.float32),
    }
    for start in range(0, n, BATCH):
        end = min(start + BATCH, n)
        current_support, _ = delay_support_from_h(np.asarray(base[start:end], dtype=np.complex64))
        teacher_support, _ = delay_support_from_h(np.asarray(teacher[start:end], dtype=np.complex64))
        anchor_support = np.asarray(anchor[start:end], dtype=np.float32)
        arrays["current_d"][start:end], arrays["current_du"][start:end], arrays["current_dh"][start:end] = support_marginals(current_support)
        arrays["teacher_d"][start:end], arrays["teacher_du"][start:end], arrays["teacher_dh"][start:end] = support_marginals(teacher_support)
        arrays["anchor_d"][start:end], arrays["anchor_du"][start:end], arrays["anchor_dh"][start:end] = support_marginals(anchor_support)
        print(f"v214 distillation marginals {end}/{n}", flush=True)
    np.savez_compressed(MARGINALS, **arrays)
    data = np.load(MARGINALS)
    return {key: data[key] for key in data.files}


def scalar_features(data: np.lib.npyio.NpzFile, keys: list[str]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for key in keys:
        if key in data.files:
            value = data[key]
            if value.ndim == 1 and value.dtype.kind in "fiu?":
                out.append(value.astype(np.float32)[:, None])
    return out


def make_features(regimes: np.ndarray, marginals: dict[str, np.ndarray]) -> np.ndarray:
    geom = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    blocked = geom["test_blocked"].astype(np.float32)
    one_hot = np.stack([(regimes.astype(str) == name).astype(np.float32) for name in ["clear", "light", "medium", "heavy"]], axis=1)

    current = {"d": marginals["current_d"], "du": marginals["current_du"], "dh": marginals["current_dh"]}
    anchor = {"d": marginals["anchor_d"], "du": marginals["anchor_du"], "dh": marginals["anchor_dh"]}
    base = [
        pos,
        blocked[:, None] / 160.0,
        one_hot,
        d_stats(current["d"]),
        d_stats(anchor["d"]),
        np.stack(
            [
                l1_rows(current["d"], anchor["d"]),
                l1_rows(current["du"], anchor["du"]),
                l1_rows(current["dh"], anchor["dh"]),
            ],
            axis=1,
        ),
    ]

    v196_actions = np.load(V196_ACTIONS)
    v199_diag = np.load(V199_DIAG)
    v200_actions = np.load(V200_ACTIONS)
    v209_actions = np.load(V209_ACTIONS)
    base.extend(
        scalar_features(
            v196_actions,
            [
                "test_neighbor_agreement",
                "test_effective_neighbor_count",
                "train_confidence",
                "alpha_d",
                "alpha_du",
                "alpha_dh",
                "target_full_l1",
                "margin_d_l1",
                "train_d_l1_to_current",
                "test_d_l1_to_current",
                "gain_max",
            ],
        )
    )
    base.extend(
        scalar_features(
            v199_diag,
            [
                "test_train_confidence",
                "test_neighbor_agreement",
                "test_effective_neighbor_count",
                "test_local_log_std",
                "test_model_spread",
                "delta_knn",
                "delta_model",
                "delta_consensus",
                "evidence_signal",
                "power_confidence",
                "positive_votes",
                "negative_votes",
                "conflict",
            ],
        )
    )
    base.extend(
        scalar_features(
            v200_actions,
            [
                "suggested_log_delta",
                "final_log_delta",
                "power_confidence",
                "delta_knn",
                "delta_model",
                "delta_consensus",
                "test_neighbor_agreement",
                "v196_test_neighbor_agreement",
                "v196_gain_max",
            ],
        )
    )
    base.extend(
        scalar_features(
            v209_actions,
            [
                "alpha_core",
                "controller_quality",
                "train_mix",
                "train_confidence",
                "test_neighbor_agreement",
                "power_confidence",
                "support_stress",
                "train_test_agree_mean",
                "target_full_l1",
                "delay_alpha",
                "extra_log_delta",
                "leaked_full_l1_actual",
                "delay_full_l1_actual",
            ],
        )
    )

    labels = v199_diag["power_label"].astype(str)
    base.append(np.stack([(labels == name).astype(np.float32) for name in ["power_up", "power_down", "conflict", "no_action"]], axis=1))
    return np.concatenate(base, axis=1).astype(np.float32)


def distill_block(
    name: str,
    current: np.ndarray,
    teacher: np.ndarray,
    features: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, dict[str, object]]:
    n = len(current)
    flat_current = current.reshape(n, -1).astype(np.float32)
    flat_teacher = teacher.reshape(n, -1).astype(np.float32)
    delta = (flat_teacher - flat_current).astype(np.float32)
    pred_delta = np.zeros_like(delta, dtype=np.float32)
    coeff_oof: list[np.ndarray] = []
    coeff_true: list[np.ndarray] = []

    folds = KFold(n_splits=5, shuffle=True, random_state=214)
    for fold, (train_idx, val_idx) in enumerate(folds.split(features)):
        pca = PCA(
            n_components=min(n_components, len(train_idx) - 1, delta.shape[1]),
            svd_solver="randomized",
            random_state=214 + fold,
        )
        train_coeff = pca.fit_transform(delta[train_idx])
        model = ExtraTreesRegressor(
            n_estimators=420,
            min_samples_leaf=4,
            max_features=0.78,
            random_state=214 + fold,
            n_jobs=-1,
        )
        model.fit(features[train_idx], train_coeff)
        pred_coeff = model.predict(features[val_idx]).astype(np.float32)
        pred_delta[val_idx] = pca.inverse_transform(pred_coeff).astype(np.float32)
        coeff_oof.append(pred_coeff)
        coeff_true.append(pca.transform(delta[val_idx]).astype(np.float32))

    target = normalize((flat_current + pred_delta).reshape(current.shape), axis=tuple(range(1, current.ndim)))
    coeff_pred = np.concatenate(coeff_oof, axis=0)
    coeff_truth = np.concatenate(coeff_true, axis=0)
    teacher_l1 = l1_rows(target, teacher)
    current_l1 = l1_rows(current, teacher)
    gain = current_l1 - teacher_l1
    diag = {
        "components": int(n_components),
        "teacher_delta_l1": summarize(l1_rows(current, teacher)),
        "student_to_teacher_l1": summarize(teacher_l1),
        "student_l1_gain_vs_current": summarize(gain),
        "positive_gain_count": int(np.sum(gain > 0.0)),
        "coeff_r2": float(r2_score(coeff_truth, coeff_pred, multioutput="variance_weighted")),
        "delta_l2_relative": float(np.linalg.norm(pred_delta - delta) / max(np.linalg.norm(delta), 1e-12)),
    }
    print(f"v214 distilled {name}: gain mean={float(np.mean(gain)):.6f} positive={int(np.sum(gain>0))}/{n}", flush=True)
    return target.astype(np.float32), diag


def apply_projection(actions: dict[str, np.ndarray], regimes: np.ndarray) -> dict[str, np.ndarray]:
    source = np.load(BASE, mmap_mode="r")
    output = np.lib.format.open_memmap(OUTPUT, mode="w+", dtype=np.complex64, shape=source.shape)
    desired_d = actions["desired_d"].astype(np.float32)
    desired_du = actions["desired_du"].astype(np.float32)
    desired_dh = actions["desired_dh"].astype(np.float32)
    diag: dict[str, list[np.ndarray]] = {
        "actual_power_factor": [],
        "full_l1_actual": [],
        "d_l1_actual": [],
        "du_l1_actual": [],
        "dh_l1_actual": [],
        "gain_min": [],
        "gain_p10": [],
        "gain_p90": [],
        "gain_max": [],
    }
    for start in range(0, len(source), BATCH):
        end = min(start + BATCH, len(source))
        h0 = np.asarray(source[start:end], dtype=np.complex64)
        cir0 = np.fft.ifft(h0, axis=-1)
        arr0 = cir0.reshape(end - start, MP, MH, 8, N_ANT, S_SC)
        power0 = (np.abs(arr0).astype(np.float32) ** 2).astype(np.float32)
        total0 = np.maximum(power0.sum(axis=(1, 2, 3, 4, 5)), TINY).astype(np.float32)
        support0 = power0 / total0[:, None, None, None, None, None]
        h_power0 = np.maximum(np.sum(np.abs(h0) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY).astype(np.float32)

        work = support0.copy()
        for _ in range(IPF_ITERS):
            v196.apply_delay_ratio(
                work,
                safe_ratio(desired_d[start:end], work.sum(axis=(1, 2, 3, 4), dtype=np.float64).astype(np.float32)),
                "d",
                0.98,
            )
            v196.apply_delay_ratio(
                work,
                safe_ratio(desired_du[start:end], work.sum(axis=(1, 2, 3), dtype=np.float64).astype(np.float32)),
                "du",
                0.38,
            )
            v196.apply_delay_ratio(
                work,
                safe_ratio(desired_dh[start:end], work.sum(axis=(1, 3, 4), dtype=np.float64).astype(np.float32)),
                "dh",
                0.34,
            )
            work = normalize(work.reshape(end - start, -1), axis=1).reshape(support0.shape)

        gain = np.sqrt((work * total0[:, None, None, None, None, None]) / np.maximum(power0, 1e-30)).astype(np.float32)
        gain = np.clip(gain, 0.18, 5.80).astype(np.float32)
        flat = support0.reshape(end - start, -1)
        strong128 = top_mask(flat, 128).reshape(support0.shape)
        strong1024 = top_mask(flat, 1024).reshape(support0.shape)
        strong8192 = top_mask(flat, 8192).reshape(support0.shape)
        gain[strong8192] = np.clip(gain[strong8192], 0.44, 2.45)
        gain[strong1024] = np.clip(gain[strong1024], 0.60, 1.70)
        gain[strong128] = np.clip(gain[strong128], 0.82, 1.30)
        for local, regime in enumerate(regimes[start:end].astype(str)):
            if regime == "clear":
                gain[local] = 1.0
            elif regime == "light":
                gain[local] = 1.0 + 0.22 * (gain[local] - 1.0)

        h1 = np.fft.fft((arr0 * gain).reshape(end - start, 256, N_ANT, S_SC), axis=-1)
        h_power1 = np.maximum(np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY).astype(np.float32)
        h1 *= np.sqrt(h_power0 / h_power1)[:, None, None, None].astype(np.float32)
        if not (np.isfinite(h1.real).all() and np.isfinite(h1.imag).all()):
            raise FloatingPointError(f"non-finite block {start}:{end}")
        output[start:end] = h1.astype(np.complex64)
        output.flush()

        power1 = v196.delay_power_from_h(h1.astype(np.complex64))
        support1 = power1 / np.maximum(power1.sum(axis=(1, 2, 3, 4, 5), keepdims=True), TINY)
        diag["actual_power_factor"].append(
            (np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64) / h_power0).astype(np.float32)
        )
        diag["full_l1_actual"].append(np.sum(np.abs(support1 - support0), axis=(1, 2, 3, 4, 5)).astype(np.float32))
        diag["d_l1_actual"].append(
            np.sum(
                np.abs(
                    support1.sum(axis=(1, 2, 3, 4), dtype=np.float64)
                    - support0.sum(axis=(1, 2, 3, 4), dtype=np.float64)
                ),
                axis=1,
            ).astype(np.float32)
        )
        diag["du_l1_actual"].append(
            np.sum(
                np.abs(
                    support1.sum(axis=(1, 2, 3), dtype=np.float64)
                    - support0.sum(axis=(1, 2, 3), dtype=np.float64)
                ),
                axis=(1, 2),
            ).astype(np.float32)
        )
        diag["dh_l1_actual"].append(
            np.sum(
                np.abs(
                    support1.sum(axis=(1, 3, 4), dtype=np.float64)
                    - support0.sum(axis=(1, 3, 4), dtype=np.float64)
                ),
                axis=(1, 2),
            ).astype(np.float32)
        )
        gflat = gain.reshape(end - start, -1)
        diag["gain_min"].append(gflat.min(axis=1).astype(np.float32))
        diag["gain_p10"].append(np.quantile(gflat, 0.10, axis=1).astype(np.float32))
        diag["gain_p90"].append(np.quantile(gflat, 0.90, axis=1).astype(np.float32))
        diag["gain_max"].append(gflat.max(axis=1).astype(np.float32))
        print(f"v214 distilled projection {end}/{len(source)}", flush=True)
    return {key: np.concatenate(value).astype(np.float32) for key, value in diag.items()}


def write_zip(report: dict[str, object]) -> dict[str, object]:
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        for src in [
            PATHS.code / "generate_v214_distilled_pdp_rule_model.py",
            PATHS.code / "generate_v209_adaptive_metric_projection_controller.py",
            PATHS.code / "generate_v212_multidomain_sionna_pdp_ensemble.py",
            PATHS.code / "generate_v196_train_guided_delay_pdp_mouth.py",
            PATHS.code / "generate_v178_test_track_complex_consensus.py",
            PATHS.code / "generate_v87_easy_field_projection.py",
            PATHS.code / "generate_v171_full_complex_ad_tail_denoising.py",
            PATHS.code / "generate_v169_power_scaled_tail_denoising.py",
            PATHS.code / "generate_v159_frontier_consensus_support_soup.py",
            PATHS.code / "powerfield_common.py",
            PATHS.code_root / "project_paths.py",
        ]:
            archive.write(src, src.name)
    with zipfile.ZipFile(ZIP_OUT) as archive:
        bad = archive.testzip()
        members = archive.namelist()
    if bad is not None:
        raise ValueError(f"ZIP CRC failure: {bad}")
    output = np.load(OUTPUT, mmap_mode="r")
    finite = True
    for start in range(0, len(output), 20):
        block = np.asarray(output[start : start + 20], dtype=np.complex64)
        finite = finite and bool(np.isfinite(block.real).all() and np.isfinite(block.imag).all())
    report["integrity"] = {
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": finite,
        "zip_crc": True,
        "zip_members": members,
        "sha256": digest(ZIP_OUT, "sha256"),
        "md5": digest(ZIP_OUT, "md5"),
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    for path in [BASE, TEACHER, V200_BASE, V196_TARGET, V196_ACTIONS, V199_DIAG, V200_ACTIONS, V209_ACTIONS, V212_ACTIONS]:
        require_file(path)
    geom = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    regimes = regime_of(geom["test_blocked"].astype(np.float32))
    marginals = build_marginals()
    features = make_features(regimes, marginals)

    desired_d, diag_d = distill_block("D", marginals["current_d"], marginals["teacher_d"], features, N_COMPONENTS["d"])
    desired_du, diag_du = distill_block("DU", marginals["current_du"], marginals["teacher_du"], features, N_COMPONENTS["du"])
    desired_dh, diag_dh = distill_block("DH", marginals["current_dh"], marginals["teacher_dh"], features, N_COMPONENTS["dh"])

    actions = {
        "regimes": regimes.astype("U8"),
        "desired_d": desired_d,
        "desired_du": desired_du,
        "desired_dh": desired_dh,
        "teacher_d": marginals["teacher_d"].astype(np.float32),
        "teacher_du": marginals["teacher_du"].astype(np.float32),
        "teacher_dh": marginals["teacher_dh"].astype(np.float32),
        "current_d": marginals["current_d"].astype(np.float32),
        "current_du": marginals["current_du"].astype(np.float32),
        "current_dh": marginals["current_dh"].astype(np.float32),
    }
    apply_diag = apply_projection(actions, regimes)
    np.savez_compressed(ACTION_OUT, **actions, **apply_diag)

    comparison_to_v209 = track.compare_output_to(OUTPUT, BASE, regimes)
    comparison_to_v212 = track.compare_output_to(OUTPUT, TEACHER, regimes)
    comparison_to_v200 = track.compare_output_to(OUTPUT, V200_BASE, regimes)
    if comparison_to_v209["shape"] != [500, 256, 4, 192] or comparison_to_v209["dtype"] != "complex64":
        raise ValueError(f"invalid output layout: {comparison_to_v209['shape']} {comparison_to_v209['dtype']}")
    if not comparison_to_v209["finite"]:
        raise ValueError("v214 output contains non-finite values")

    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "The high score of v212 reflects reusable propagation rules, not only "
            "point-specific Sionna answers.  A student that sees only official "
            "geometry/current-model features can learn a low-dimensional version "
            "of the v212-v209 PDP action and apply it without Sionna references."
        ),
        "new_information_used": {
            "teacher_trace": "v212-v209 normalized D/U-D/H-D movement is used as a distillation target",
            "student_inputs": "official coordinates, blocked/regime, v196/v199/v200/v209 diagnostics and current/anchor PDP features only",
            "no_sionna_runtime_features": "student generation does not read Sionna H, Sionna phase, Sionna total power or Sionna feature arrays",
        },
        "parameters": {
            "base": PATHS.relative(BASE),
            "teacher": PATHS.relative(TEACHER),
            "ipf_iters": IPF_ITERS,
            "pca_components": N_COMPONENTS,
            "student": "5-fold OOF PCA-coefficient ExtraTreesRegressor",
            "phase_policy": "preserve v209 delay-domain phase",
            "power_policy": "preserve v209 point total power exactly",
        },
        "distillation_diagnostics": {"d": diag_d, "du": diag_du, "dh": diag_dh},
        "diagnostics": {
            "overall": {
                key: summarize(value)
                for key, value in apply_diag.items()
            },
            "by_regime": by_regime_summary(
                regimes,
                {
                    "full_l1_actual": apply_diag["full_l1_actual"],
                    "d_l1_actual": apply_diag["d_l1_actual"],
                    "du_l1_actual": apply_diag["du_l1_actual"],
                    "dh_l1_actual": apply_diag["dh_l1_actual"],
                    "actual_power_factor": apply_diag["actual_power_factor"],
                },
            ),
        },
        "comparison_to_v209": comparison_to_v209,
        "comparison_to_v212_teacher": comparison_to_v212,
        "comparison_to_v200": comparison_to_v200,
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
            "marginals": PATHS.relative(MARGINALS),
            "actions": PATHS.relative(ACTION_OUT),
            "report": PATHS.relative(REPORT),
            "zip": PATHS.relative(ZIP_OUT),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report = write_zip(report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
