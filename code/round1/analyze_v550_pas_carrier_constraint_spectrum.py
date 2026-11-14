#!/usr/bin/env python3
"""Audit how v549 PAS actions interact with carrier magnitude constraints."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import analyze_v205_sionna_config_calibration as metric
import analyze_v549_multiseed_pas_calibration as calibration
import generate_v423_joint_sionna_pas_pdp_diagnostic as v423
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 550
VERSION = "v550_pas_carrier_constraint_spectrum"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
TRUTH = PATHS.round1 / "Round1_Train_Channel.npy"
OOF_ROOT = PATHS.cache_dir(549) / "pas_calibration" / "n24"
OOF_META = (
    PATHS.cache_dir(549) / "pas_calibration" / "v549_multiseed_pas_calibration_oof.npz"
)
CALIBRATION_REPORT = calibration.REPORT
CARRIERS = {
    "v471_strict_oof": PATHS.cache_dir(471) / "v471_train_v98like_base_h.npy",
    "v473_v469like": PATHS.cache_dir(473) / "v473_train_v469like_H.npy",
}
REGIMES = ("clear", "light", "medium", "heavy")
MARGINALS = ("sh", "psh", "shue", "shv")
FLOOR = 1e-12
TINY = 1e-30


@dataclass(frozen=True)
class Variant:
    name: str
    constraint: str
    lock_fraction: float = 0.0
    terminal_only: bool = False


VARIANTS = (
    Variant("full_cir_lock", "cir_blend", 1.0),
    Variant("terminal_full_cir_lock", "cir_blend", 1.0, True),
    Variant("cir_lock_075", "cir_blend", 0.75),
    Variant("cir_lock_050", "cir_blend", 0.50),
    Variant("cir_lock_025", "cir_blend", 0.25),
    Variant("link_pdp_shape_lock", "link_pdp_shape"),
    Variant("link_energy_lock", "link_energy"),
    Variant("point_power_lock", "point_power"),
    Variant("unlocked", "none"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument(
        "--carrier",
        action="append",
        choices=tuple(CARRIERS),
        help="carrier to evaluate; repeatable; defaults to both",
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=tuple(value.name for value in VARIANTS),
        help="variant to evaluate; repeatable; defaults to all",
    )
    return parser.parse_args()


def require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def normalize(value: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    output = np.maximum(np.asarray(value, dtype=np.float32), FLOOR)
    output /= np.maximum(
        np.sum(output, axis=axes, keepdims=True, dtype=np.float64), TINY
    )
    return output.astype(np.float32)


def selected_action(
    baseline: np.ndarray,
    calibrated: np.ndarray,
    effective: np.ndarray,
) -> np.ndarray:
    target = calibration.normalize_rows(
        baseline * (1.0 - effective[:, None]) + calibrated * effective[:, None]
    )
    return np.clip(
        np.log(np.maximum(target, FLOOR)) - np.log(np.maximum(baseline, FLOOR)),
        -3.0,
        3.0,
    ).astype(np.float32)


def desired_pas(
    current: dict[str, np.ndarray], actions: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    output = {}
    for name, value in current.items():
        axes = (1, 2) if name == "sh" else (1, 2, 3)
        output[name] = normalize(value * np.exp(actions[name]), axes)
    return output


def unit_phase(value: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    amplitude = np.abs(value)
    fallback_amplitude = np.abs(fallback)
    phase = value / np.maximum(amplitude, TINY)
    fallback_phase = fallback / np.maximum(fallback_amplitude, TINY)
    return np.where(amplitude > 1e-20, phase, fallback_phase)


def constrain(
    candidate: np.ndarray,
    base: np.ndarray,
    variant: Variant,
) -> np.ndarray:
    if variant.constraint == "none":
        return candidate.astype(np.complex64)

    candidate_cir = np.fft.ifft(candidate, axis=-1)
    base_cir = np.fft.ifft(base, axis=-1)
    candidate_power = np.abs(candidate_cir).astype(np.float32) ** 2
    base_power = np.abs(base_cir).astype(np.float32) ** 2
    phase = unit_phase(candidate_cir, base_cir)

    if variant.constraint == "cir_blend":
        output_power = (
            np.float32(variant.lock_fraction) * base_power
            + np.float32(1.0 - variant.lock_fraction) * candidate_power
        )
    elif variant.constraint == "link_pdp_shape":
        candidate_energy = np.sum(
            candidate_power, axis=-1, keepdims=True, dtype=np.float64
        ).astype(np.float32)
        base_shape = base_power / np.maximum(
            np.sum(base_power, axis=-1, keepdims=True, dtype=np.float64), TINY
        )
        output_power = base_shape * candidate_energy
    elif variant.constraint == "link_energy":
        base_energy = np.sum(
            base_power, axis=-1, keepdims=True, dtype=np.float64
        ).astype(np.float32)
        candidate_shape = candidate_power / np.maximum(
            np.sum(candidate_power, axis=-1, keepdims=True, dtype=np.float64), TINY
        )
        output_power = candidate_shape * base_energy
    elif variant.constraint == "point_power":
        candidate_energy = np.sum(
            candidate_power, axis=(1, 2, 3), keepdims=True, dtype=np.float64
        )
        base_energy = np.sum(
            base_power, axis=(1, 2, 3), keepdims=True, dtype=np.float64
        )
        output_power = candidate_power * (
            base_energy / np.maximum(candidate_energy, TINY)
        ).astype(np.float32)
    else:
        raise ValueError(variant.constraint)

    return np.fft.fft(np.sqrt(np.maximum(output_power, 0.0)) * phase, axis=-1).astype(
        np.complex64
    )


def render(
    base: np.ndarray,
    desired: dict[str, np.ndarray],
    variant: Variant,
    cycles: int,
    first_projected: np.ndarray,
) -> np.ndarray:
    candidate = first_projected.copy()
    if not variant.terminal_only or cycles == 1:
        candidate = constrain(candidate, base, variant)
    for cycle in range(1, cycles):
        candidate = v423.project_pas(candidate, desired)
        if not variant.terminal_only:
            candidate = constrain(candidate, base, variant)
        elif cycle == cycles - 1:
            candidate = constrain(candidate, base, variant)
    return candidate.astype(np.complex64)


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left.reshape(len(left), -1).astype(np.float32)
    right = right.reshape(len(right), -1).astype(np.float32)
    numerator = np.sum(left * right, axis=1, dtype=np.float64)
    denominator = np.sqrt(
        np.sum(left * left, axis=1, dtype=np.float64)
        * np.sum(right * right, axis=1, dtype=np.float64)
    )
    return (numerator / np.maximum(denominator, TINY)).astype(np.float32)


class ScoreAccumulator:
    def __init__(self, rows: int) -> None:
        self.pas = np.empty(rows, dtype=np.float32)
        self.pdp = np.empty(rows, dtype=np.float32)
        self.leaked = np.empty(rows, dtype=np.float32)
        self.point_nmse_num = np.empty(rows, dtype=np.float64)
        self.point_energy = np.empty(rows, dtype=np.float64)
        self.pred_power = np.empty(rows, dtype=np.float64)
        self.truth_power = np.empty(rows, dtype=np.float64)

    def update(
        self,
        start: int,
        end: int,
        prediction: np.ndarray,
        truth: np.ndarray,
    ) -> None:
        truth_views = metric.spectral_views(truth)
        scores = metric.spectral_scores(prediction, truth_views)
        self.pas[start:end] = scores["pas"]
        self.pdp[start:end] = scores["pdp"]
        self.leaked[start:end] = scores["leaked_sh"]
        self.point_nmse_num[start:end] = np.sum(
            np.abs(prediction - truth) ** 2,
            axis=(1, 2, 3),
            dtype=np.float64,
        )
        self.point_energy[start:end] = truth_views["energy"]
        self.pred_power[start:end] = np.sum(
            np.abs(prediction) ** 2, axis=(1, 2, 3), dtype=np.float64
        )
        self.truth_power[start:end] = truth_views["energy"]

    def summarize(self, mask: np.ndarray | None = None) -> dict[str, float]:
        if mask is None:
            mask = np.ones(len(self.pas), dtype=bool)
        error = float(np.sum(self.point_nmse_num[mask], dtype=np.float64))
        energy = float(np.sum(self.point_energy[mask], dtype=np.float64))
        nmse = error / max(energy, TINY)
        nmse_score = 1.0 / (1.0 + nmse)
        pas = float(np.mean(self.pas[mask]))
        pdp = float(np.mean(self.pdp[mask]))
        leaked = float(np.mean(self.leaked[mask]))
        return {
            "count": int(np.sum(mask)),
            "pas": pas,
            "pdp": pdp,
            "leaked_pas": leaked,
            "global_nmse": nmse,
            "nmse_score": nmse_score,
            "official_score": 0.4 * pas + 0.4 * pdp + 0.2 * nmse_score,
            "point_power_log_mae": float(
                np.mean(
                    np.abs(
                        np.log(np.maximum(self.pred_power[mask], TINY))
                        - np.log(np.maximum(self.truth_power[mask], TINY))
                    )
                )
            ),
        }


def delta(candidate: dict[str, float], base: dict[str, float]) -> dict[str, float]:
    return {
        key: float(candidate[key] - base[key])
        for key in (
            "pas",
            "pdp",
            "leaked_pas",
            "global_nmse",
            "nmse_score",
            "official_score",
            "point_power_log_mae",
        )
    }


def evaluate_carrier(
    carrier_name: str,
    carrier_path: Path,
    variants: tuple[Variant, ...],
    batch_size: int,
    cycles: int,
    actions: dict[str, np.ndarray],
    fold_id: np.ndarray,
    regimes: np.ndarray,
) -> dict[str, object]:
    carrier = np.load(carrier_path, mmap_mode="r")
    truth = np.load(TRUTH, mmap_mode="r")
    rows = len(truth)
    base_scores = ScoreAccumulator(rows)
    candidate_scores = {variant.name: ScoreAccumulator(rows) for variant in variants}
    action_alignment = {
        variant.name: np.empty(rows, dtype=np.float32) for variant in variants
    }

    for start in range(0, rows, batch_size):
        end = min(start + batch_size, rows)
        base_h = np.asarray(carrier[start:end], dtype=np.complex64)
        truth_h = np.asarray(truth[start:end], dtype=np.complex64)
        base_scores.update(start, end, base_h, truth_h)
        current_support, _ = v423.leaked_support(base_h)
        current = v423.leaked_marginals(current_support)
        local_actions = {
            name: actions[name][start:end].reshape(current[name].shape)
            for name in MARGINALS
        }
        desired = desired_pas(current, local_actions)
        first_projected = v423.project_pas(base_h, desired)
        for variant in variants:
            prediction = render(
                base_h,
                desired,
                variant,
                cycles,
                first_projected,
            )
            if not np.isfinite(prediction).all():
                raise FloatingPointError(
                    f"non-finite {carrier_name}/{variant.name} block {start}:{end}"
                )
            candidate_scores[variant.name].update(start, end, prediction, truth_h)
            predicted_support, _ = v423.leaked_support(prediction)
            predicted = v423.leaked_marginals(predicted_support)
            composite = np.zeros(end - start, dtype=np.float32)
            for name, weight in calibration.MARGINAL_WEIGHTS.items():
                composite += np.float32(weight) * cosine_rows(
                    predicted[name], desired[name]
                )
            action_alignment[variant.name][start:end] = composite
        if end == rows or end % 100 == 0:
            print(
                f"v550 carrier={carrier_name} rows={end}/{rows}",
                flush=True,
            )

    base_summary = base_scores.summarize()
    base_folds = {
        str(fold): base_scores.summarize(fold_id == fold) for fold in range(5)
    }
    base_regimes = {
        regime: base_scores.summarize(regimes == regime) for regime in REGIMES
    }
    results = {}
    for variant in variants:
        scores = candidate_scores[variant.name]
        overall = scores.summarize()
        folds = []
        for fold in range(5):
            local = scores.summarize(fold_id == fold)
            folds.append(
                {
                    "fold": fold,
                    "base": base_folds[str(fold)],
                    "candidate": local,
                    "delta": delta(local, base_folds[str(fold)]),
                }
            )
        by_regime = {}
        for regime in REGIMES:
            local = scores.summarize(regimes == regime)
            by_regime[regime] = {
                "base": base_regimes[regime],
                "candidate": local,
                "delta": delta(local, base_regimes[regime]),
            }
        results[variant.name] = {
            "constraint": variant.constraint,
            "lock_fraction": variant.lock_fraction,
            "terminal_only": variant.terminal_only,
            "overall": overall,
            "delta": delta(overall, base_summary),
            "fold_wins": int(
                sum(row["delta"]["official_score"] > 0.0 for row in folds)
            ),
            "folds": folds,
            "by_regime": by_regime,
            "action_alignment": {
                "mean": float(np.mean(action_alignment[variant.name])),
                "p10": float(np.quantile(action_alignment[variant.name], 0.10)),
                "median": float(np.median(action_alignment[variant.name])),
                "p90": float(np.quantile(action_alignment[variant.name], 0.90)),
            },
        }
    return {
        "path": PATHS.relative(carrier_path),
        "base": base_summary,
        "base_by_fold": base_folds,
        "base_by_regime": base_regimes,
        "variants": results,
    }


def main() -> None:
    args = parse_args()
    selected_carriers = tuple(args.carrier or CARRIERS.keys())
    selected_names = set(args.variant or [value.name for value in VARIANTS])
    variants = tuple(value for value in VARIANTS if value.name in selected_names)
    for path in (TRUTH, OOF_META, CALIBRATION_REPORT):
        require(path)
    for name in selected_carriers:
        require(CARRIERS[name])
    for name in MARGINALS:
        require(OOF_ROOT / f"{name}_baseline_oof.npy")
        require(OOF_ROOT / f"{name}_calibrated_oof.npy")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    calibration_report = json.loads(CALIBRATION_REPORT.read_text(encoding="utf-8"))
    selected = calibration_report["selected"]
    if int(selected["checkpoint"]) != 24:
        raise ValueError(f"expected checkpoint 24, got {selected['checkpoint']}")
    meta = np.load(OOF_META)
    fold_id = meta["n24_fold_id"].astype(np.int16)
    instability = meta["n24_instability"].astype(np.float32)
    effective = np.clip(
        np.float32(selected["alpha"])
        * np.exp(-np.float32(selected["quality_scale"]) * instability),
        0.0,
        1.0,
    ).astype(np.float32)
    actions = {}
    for name in MARGINALS:
        baseline = np.load(OOF_ROOT / f"{name}_baseline_oof.npy", mmap_mode="r")
        calibrated = np.load(OOF_ROOT / f"{name}_calibrated_oof.npy", mmap_mode="r")
        actions[name] = selected_action(baseline, calibrated, effective)

    geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    regimes = regime_of(geometry["train_blocked"]).astype("U8")
    carrier_results = {}
    if REPORT.is_file():
        prior = json.loads(REPORT.read_text(encoding="utf-8"))
        if (
            prior.get("method") == VERSION
            and int(prior.get("evidence_boundary", {}).get("cycles", -1)) == args.cycles
        ):
            carrier_results.update(prior.get("carriers", {}))
    for carrier_name in selected_carriers:
        carrier_results[carrier_name] = evaluate_carrier(
            carrier_name,
            CARRIERS[carrier_name],
            variants,
            args.batch_size,
            args.cycles,
            actions,
            fold_id,
            regimes,
        )

    consensus = []
    consensus_variants = tuple(
        variant
        for variant in VARIANTS
        if all(
            variant.name in carrier["variants"] for carrier in carrier_results.values()
        )
    )
    for variant in consensus_variants:
        rows = [
            carrier["variants"][variant.name] for carrier in carrier_results.values()
        ]
        consensus.append(
            {
                "variant": variant.name,
                "mean_score_delta": float(
                    np.mean([row["delta"]["official_score"] for row in rows])
                ),
                "min_score_delta": float(
                    np.min([row["delta"]["official_score"] for row in rows])
                ),
                "carrier_wins": int(
                    sum(row["delta"]["official_score"] > 0.0 for row in rows)
                ),
                "total_fold_wins": int(sum(row["fold_wins"] for row in rows)),
                "mean_pas_delta": float(np.mean([row["delta"]["pas"] for row in rows])),
                "mean_pdp_delta": float(np.mean([row["delta"]["pdp"] for row in rows])),
                "mean_nmse_score_delta": float(
                    np.mean([row["delta"]["nmse_score"] for row in rows])
                ),
            }
        )
    consensus.sort(
        key=lambda row: (
            row["carrier_wins"],
            row["total_fold_wins"],
            row["min_score_delta"],
            row["mean_score_delta"],
        ),
        reverse=True,
    )
    report = {
        "method": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "v549 is limited by applying a KNN-relative PAS action to a different "
            "carrier while hard-locking all CIR magnitudes; a carrier-consistent "
            "constraint spectrum can identify which released magnitude degree of "
            "freedom improves PAS, PDP, and NMSE jointly."
        ),
        "evidence_boundary": {
            "actions": "frozen v549 n24 OOF calibrated/KNN log-ratio",
            "carriers": sorted(carrier_results),
            "selection_uses_online_score": False,
            "cycles": args.cycles,
        },
        "effective_action": {
            "mean": float(np.mean(effective)),
            "p10": float(np.quantile(effective, 0.10)),
            "median": float(np.median(effective)),
            "p90": float(np.quantile(effective, 0.90)),
        },
        "consensus": consensus,
        "carriers": carrier_results,
        "outputs": {"report": PATHS.relative(REPORT)},
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"consensus": consensus}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
