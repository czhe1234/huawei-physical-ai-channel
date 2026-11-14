#!/usr/bin/env python3
"""Audit global array, port, frequency, and phase conventions from Sionna CFR to official H."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import gc
from itertools import permutations
import json
import math
import time

import numpy as np

import analyze_v501_full_official_sionna_inverse_calibration as v501
import analyze_v503_fixed_material_sionna_version_inverse as v503
import analyze_v504_scene_topology_sionna_inverse as v504
from project_paths import PATHS


VERSION_NUMBER = 515
VERSION = "v515_path_to_cfr_renderer_audit"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
METRIC_DIR = CACHE_DIR / "renderer_metrics"
SCENE_XML = PATHS.artifacts / "external" / "restored_sionna_official_etoile" / "etoile.xml"
TRAIN_POS = PATHS.round1 / "Round1_Train_Pos.npy"
TRAIN_H = PATHS.round1 / "Round1_Train_Channel.npy"
V503_REPORT = PATHS.version_report_dir(503) / "v503_refine.json"

FREQUENCY_HZ = 6.693e9
NUM_SUBCARRIERS = 192
SUBCARRIER_SPACING = 30_000.0
MAX_DEPTH = 16
SAMPLES_PER_SRC = 3_000_000
MAX_PATHS_PER_SRC = 100_000
SEED = 42
TINY = 1e-30


def release_memory() -> None:
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


@dataclass(frozen=True)
class RendererConfig:
    tx_order: str = "phv"
    flip_p: bool = False
    flip_h: bool = False
    flip_v: bool = False
    frequency: str = "identity"
    ue_permutation: tuple[int, int, int, int] = (0, 1, 2, 3)

    @property
    def name(self) -> str:
        ue = "".join(str(value) for value in self.ue_permutation)
        return (
            f"tx_{self.tx_order}_fp{int(self.flip_p)}fh{int(self.flip_h)}"
            f"fv{int(self.flip_v)}_f_{self.frequency}_ue_{ue}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("build", "audit", "all"), default="all")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--metric-batch-size", type=int, default=4)
    parser.add_argument("--point-limit", type=int, default=160)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--mitsuba-variant", default="cuda_ad_mono_polarized")
    return parser.parse_args()


def selected_indices(point_limit: int) -> np.ndarray:
    indices = np.asarray(
        json.loads(V503_REPORT.read_text(encoding="utf-8"))["indices"],
        dtype=np.int64,
    )
    return indices[: min(point_limit, len(indices))]


def cfr_cache_path(point_count: int):
    return CACHE_DIR / f"v515_base_cfr_points{point_count}.npy"


def metadata_path(point_count: int):
    return CACHE_DIR / f"v515_base_cfr_points{point_count}_metadata.npz"


def build_cfr_cache(args: argparse.Namespace, indices: np.ndarray) -> None:
    output_path = cfr_cache_path(len(indices))
    meta_path = metadata_path(len(indices))
    if output_path.is_file() and meta_path.is_file() and not args.force:
        print(f"reuse {PATHS.relative(output_path)}", flush=True)
        return

    import drjit as dr
    import mitsuba as mi

    if mi.variant() is None:
        mi.set_variant(args.mitsuba_variant)
    import sionna
    from sionna.phy.channel import subcarrier_frequencies
    from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

    if sionna.__version__ != "2.0.1":
        raise RuntimeError(f"v515 requires Sionna 2.0.1, got {sionna.__version__}")

    positions = np.load(TRAIN_POS).astype(np.float32)
    frequencies = np.asarray(
        subcarrier_frequencies(NUM_SUBCARRIERS, SUBCARRIER_SPACING)
    ).astype(np.float32)
    scene = load_scene(
        str(SCENE_XML), merge_shapes=True, remove_duplicate_vertices=True
    )
    scene.frequency = FREQUENCY_HZ
    materials = v504.material_audit(scene)
    if set(materials) != {"concrete", "marble", "metal", "wood"}:
        raise RuntimeError(f"unexpected materials: {materials}")
    scene.tx_array = PlanarArray(
        num_rows=8,
        num_cols=16,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="tr38901",
        polarization="VH",
    )
    scene.rx_array = PlanarArray(
        num_rows=2,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="dipole",
        polarization="cross",
    )
    search = v501.SearchConfig(
        name="v515_base",
        tx_azimuth_deg=-30.0,
        tx_tilt_deg=-9.53,
        rx_orientation="track_bs",
        synthetic_array=True,
        frequency_hz=FREQUENCY_HZ,
        max_depth=MAX_DEPTH,
        samples_per_src=SAMPLES_PER_SRC,
        max_num_paths_per_src=MAX_PATHS_PER_SRC,
        seed=SEED,
    )
    tx = Transmitter(
        name="tx",
        position=mi.Point3f(*[float(value) for value in v501.BS_POSITION]),
        look_at=v501.tx_target(search, mi),
        power_dbm=44.0,
    )
    scene.add(tx)
    solver = PathSolver()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.complex64,
        shape=(len(indices), 256, 4, NUM_SUBCARRIERS),
    )
    path_counts = np.zeros(len(indices), dtype=np.int32)
    receiver_names: list[str] = []
    started = time.time()
    for batch_index, start in enumerate(range(0, len(indices), args.batch_size)):
        end = min(start + args.batch_size, len(indices))
        for name in receiver_names:
            scene.remove(name)
        receiver_names = []
        batch_ids = indices[start:end]
        for point_id in batch_ids:
            name = f"rx_{int(point_id):05d}"
            scene.add(
                Receiver(
                    name=name,
                    position=mi.Point3f(
                        *[float(value) for value in positions[int(point_id)]]
                    ),
                    look_at=tx,
                )
            )
            receiver_names.append(name)
        paths = solver(
            scene,
            max_depth=MAX_DEPTH,
            max_num_paths_per_src=MAX_PATHS_PER_SRC,
            samples_per_src=SAMPLES_PER_SRC,
            synthetic_array=True,
            los=True,
            specular_reflection=True,
            diffuse_reflection=False,
            refraction=True,
            diffraction=False,
            edge_diffraction=False,
            diffraction_lit_region=True,
            seed=SEED + batch_index,
        )
        valid = np.asarray(paths.valid.numpy(), dtype=bool)
        path_counts[start:end] = valid.reshape(len(batch_ids), -1).sum(axis=1)
        raw = paths.cfr(
            mi.TensorXf(frequencies),
            normalize_delays=False,
            normalize=False,
            out_type="numpy",
        )
        output[start:end] = v501.convert_cfr(np.asarray(raw), len(batch_ids))
        output.flush()
        del paths, valid, raw
        dr.sync_thread()
        if end == len(indices) or (batch_index + 1) % 10 == 0:
            print(
                f"v515 CFR {end}/{len(indices)} elapsed={time.time() - started:.1f}s",
                flush=True,
            )

    for name in receiver_names:
        scene.remove(name)
    output.flush()
    np.savez_compressed(
        meta_path,
        indices=indices,
        regimes=v501.regime_codes()[indices],
        path_counts=path_counts,
        positions=positions[indices],
    )


def apply_renderer(channel: np.ndarray, config: RendererConfig) -> np.ndarray:
    count = len(channel)
    dimensions = {"p": 2, "h": 16, "v": 8}
    source_shape = tuple(dimensions[key] for key in config.tx_order)
    value = np.asarray(channel).reshape(count, *source_shape, 4, NUM_SUBCARRIERS)
    transpose = [0] + [1 + config.tx_order.index(key) for key in "phv"] + [4, 5]
    value = value.transpose(transpose)
    if config.flip_p:
        value = np.flip(value, axis=1)
    if config.flip_h:
        value = np.flip(value, axis=2)
    if config.flip_v:
        value = np.flip(value, axis=3)
    value = value[:, :, :, :, config.ue_permutation, :]
    if config.frequency == "reverse":
        value = value[..., ::-1]
    elif config.frequency == "shift":
        value = np.roll(value, NUM_SUBCARRIERS // 2, axis=-1)
    elif config.frequency == "reverse_shift":
        value = np.roll(value[..., ::-1], NUM_SUBCARRIERS // 2, axis=-1)
    elif config.frequency != "identity":
        raise ValueError(config.frequency)
    return np.ascontiguousarray(value.reshape(count, 256, 4, NUM_SUBCARRIERS))


def evaluate_renderer(
    source: np.ndarray,
    truth: np.ndarray,
    truth_indices: np.ndarray,
    config: RendererConfig,
    batch_size: int,
) -> dict[str, np.ndarray]:
    names = ("pdp", "leaked_full", "complex_coherence")
    output = {name: np.empty(len(source), dtype=np.float32) for name in names}
    for start in range(0, len(source), batch_size):
        end = min(start + batch_size, len(source))
        prediction = apply_renderer(np.asarray(source[start:end]), config)
        metrics = v503.channel_metrics(
            prediction,
            np.asarray(truth[truth_indices[start:end]], dtype=np.complex64),
        )
        for name in names:
            output[name][start:end] = metrics[name]
    output["spectral"] = 0.5 * (output["pdp"] + output["leaked_full"])
    return output


def evaluate_cached(
    source: np.ndarray,
    truth: np.ndarray,
    truth_indices: np.ndarray,
    config: RendererConfig,
    batch_size: int,
) -> dict[str, np.ndarray]:
    path = METRIC_DIR / f"points{len(source)}" / f"{config.name}.npz"
    if path.is_file():
        with np.load(path) as data:
            return {name: np.asarray(data[name]) for name in data.files}
    metrics = evaluate_renderer(source, truth, truth_indices, config, batch_size)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **metrics)
    return metrics


def array_configs() -> list[RendererConfig]:
    return [
        RendererConfig(order, flip_p, flip_h, flip_v)
        for order in ("".join(value) for value in permutations("phv"))
        for flip_p in (False, True)
        for flip_h in (False, True)
        for flip_v in (False, True)
    ]


def row_summary(config: RendererConfig, metrics: dict[str, np.ndarray]) -> dict[str, object]:
    return {
        "name": config.name,
        "config": {
            "tx_order": config.tx_order,
            "flip_p": config.flip_p,
            "flip_h": config.flip_h,
            "flip_v": config.flip_v,
            "frequency": config.frequency,
            "ue_permutation": list(config.ue_permutation),
        },
        **{name: float(np.mean(value)) for name, value in metrics.items()},
    }


def spatial_nested_selection(
    rows: list[dict[str, object]],
    metric_bank: list[dict[str, np.ndarray]],
    fold_codes: np.ndarray,
    baseline_name: str,
) -> dict[str, object]:
    baseline_index = next(i for i, row in enumerate(rows) if row["name"] == baseline_name)
    folds = []
    gains = []
    selected_names = []
    for fold in sorted(np.unique(fold_codes).tolist()):
        train = fold_codes != fold
        test = fold_codes == fold
        means = np.asarray(
            [np.mean(metrics["spectral"][train]) for metrics in metric_bank]
        )
        selected = int(np.argmax(means))
        gain = float(
            np.mean(metric_bank[selected]["spectral"][test])
            - np.mean(metric_bank[baseline_index]["spectral"][test])
        )
        gains.append(gain)
        selected_names.append(str(rows[selected]["name"]))
        folds.append(
            {
                "fold": int(fold),
                "count": int(np.sum(test)),
                "selected": rows[selected]["name"],
                "gain_vs_direct": gain,
                "selected_pdp": float(np.mean(metric_bank[selected]["pdp"][test])),
                "selected_pas": float(
                    np.mean(metric_bank[selected]["leaked_full"][test])
                ),
            }
        )
    return {
        "folds": folds,
        "mean_gain_vs_direct": float(np.mean(gains)),
        "positive_folds": int(np.sum(np.asarray(gains) > 0.0)),
        "selected_names": selected_names,
        "selection_consistent": len(set(selected_names)) == 1,
    }


def phase_reference_audit(
    source: np.ndarray,
    truth: np.ndarray,
    truth_indices: np.ndarray,
    config: RendererConfig,
    fold_codes: np.ndarray,
    batch_size: int,
) -> dict[str, object]:
    frequency = (
        np.arange(NUM_SUBCARRIERS, dtype=np.float64) - NUM_SUBCARRIERS // 2
    ) * SUBCARRIER_SPACING
    delay_bins = np.linspace(-4.0, 4.0, 129, dtype=np.float64)
    delay_seconds = delay_bins / (NUM_SUBCARRIERS * SUBCARRIER_SPACING)
    ramps = np.exp(2j * np.pi * delay_seconds[:, None] * frequency[None, :])
    pred_power = np.empty(len(source), dtype=np.float64)
    truth_power = np.empty(len(source), dtype=np.float64)
    cross = np.empty((len(source), NUM_SUBCARRIERS), dtype=np.complex128)
    cross_conjugated = np.empty_like(cross)
    for start in range(0, len(source), batch_size):
        end = min(start + batch_size, len(source))
        prediction = apply_renderer(np.asarray(source[start:end]), config)
        target = np.asarray(
            truth[truth_indices[start:end]], dtype=np.complex64
        )
        pred_power[start:end] = np.sum(
            np.abs(prediction) ** 2, axis=(1, 2, 3), dtype=np.float64
        )
        truth_power[start:end] = np.sum(
            np.abs(target) ** 2, axis=(1, 2, 3), dtype=np.float64
        )
        cross[start:end] = np.sum(
            np.conj(prediction) * target, axis=(1, 2), dtype=np.complex128
        )
        cross_conjugated[start:end] = np.sum(
            prediction * target, axis=(1, 2), dtype=np.complex128
        )
    denominator = np.maximum(np.sqrt(pred_power * truth_power), TINY)
    coherence = np.empty((2, len(delay_bins), len(prediction)), dtype=np.float64)
    for operation, values in enumerate((cross, cross_conjugated)):
        coherence[operation] = np.abs(ramps @ values.T) / denominator[None, :]

    fold_rows = []
    gains = []
    direct = coherence[0, np.argmin(np.abs(delay_bins))]
    for fold in sorted(np.unique(fold_codes).tolist()):
        train = fold_codes != fold
        test = fold_codes == fold
        train_mean = np.mean(coherence[:, :, train], axis=2)
        operation, delay_id = np.unravel_index(np.argmax(train_mean), train_mean.shape)
        selected = coherence[operation, delay_id]
        gain = float(np.mean(selected[test]) - np.mean(direct[test]))
        gains.append(gain)
        fold_rows.append(
            {
                "fold": int(fold),
                "operation": "identity" if operation == 0 else "conjugate",
                "delay_bins": float(delay_bins[delay_id]),
                "coherence": float(np.mean(selected[test])),
                "gain_vs_no_shift": gain,
            }
        )
    return {
        "grid_delay_bins": [float(delay_bins[0]), float(delay_bins[-1]), len(delay_bins)],
        "folds": fold_rows,
        "mean_gain_vs_no_shift": float(np.mean(gains)),
        "positive_folds": int(np.sum(np.asarray(gains) > 0.0)),
    }


def audit(args: argparse.Namespace, indices: np.ndarray) -> None:
    source = np.load(cfr_cache_path(len(indices)), mmap_mode="r")
    truth_all = np.load(TRAIN_H, mmap_mode="r")
    positions = np.load(TRAIN_POS).astype(np.float32)
    regimes = v501.regime_codes()
    fold_codes = v501.spatial_fold_codes(positions, regimes)[indices]

    direct = RendererConfig()
    stage1_configs = array_configs()
    stage1_metrics = []
    stage1_rows = []
    for index, config in enumerate(stage1_configs):
        metrics = evaluate_cached(
            source, truth_all, indices, config, args.metric_batch_size
        )
        stage1_metrics.append(metrics)
        stage1_rows.append(row_summary(config, metrics))
        release_memory()
        if (index + 1) % 8 == 0:
            print(f"v515 array audit {index + 1}/{len(stage1_configs)}", flush=True)
    stage1_order = np.argsort([row["spectral"] for row in stage1_rows])[::-1]
    top_array = [stage1_configs[int(i)] for i in stage1_order[:8]]
    if direct not in top_array:
        top_array.append(direct)

    stage2_configs = []
    for config in top_array:
        for frequency in ("identity", "reverse", "shift", "reverse_shift"):
            candidate = RendererConfig(
                config.tx_order,
                config.flip_p,
                config.flip_h,
                config.flip_v,
                frequency,
            )
            if candidate not in stage2_configs:
                stage2_configs.append(candidate)
    stage2_metrics = []
    for config in stage2_configs:
        stage2_metrics.append(
            evaluate_cached(
                source, truth_all, indices, config, args.metric_batch_size
            )
        )
        release_memory()
    stage2_rows = [
        row_summary(config, metrics)
        for config, metrics in zip(stage2_configs, stage2_metrics)
    ]
    stage2_order = np.argsort([row["spectral"] for row in stage2_rows])[::-1]
    top_frequency = [stage2_configs[int(i)] for i in stage2_order[:4]]
    if direct not in top_frequency:
        top_frequency.append(direct)

    stage3_configs = []
    for config in top_frequency:
        for ue in permutations(range(4)):
            candidate = RendererConfig(
                config.tx_order,
                config.flip_p,
                config.flip_h,
                config.flip_v,
                config.frequency,
                tuple(int(value) for value in ue),
            )
            if candidate not in stage3_configs:
                stage3_configs.append(candidate)
    stage3_metrics = []
    stage3_rows = []
    for index, config in enumerate(stage3_configs):
        metrics = evaluate_cached(
            source, truth_all, indices, config, args.metric_batch_size
        )
        stage3_metrics.append(metrics)
        stage3_rows.append(row_summary(config, metrics))
        release_memory()
        if (index + 1) % 24 == 0:
            print(f"v515 port audit {index + 1}/{len(stage3_configs)}", flush=True)
    final_order = np.argsort([row["spectral"] for row in stage3_rows])[::-1]
    best_index = int(final_order[0])
    best_config = stage3_configs[best_index]
    baseline_name = direct.name
    report = {
        "method": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "A global Paths-to-CFR convention mismatch in array flattening, port ordering, "
            "subcarrier ordering, conjugation, or delay reference explains part of the "
            "remaining official-vs-Sionna gap."
        ),
        "new_information": (
            "Full complex Sionna CFR at the same 160 balanced official-train coordinates, "
            "rather than aggregated PAS/PDP metrics or power-only path fingerprints."
        ),
        "point_count": int(len(indices)),
        "indices": indices.tolist(),
        "sionna_config": {
            "frequency_hz": FREQUENCY_HZ,
            "depth": MAX_DEPTH,
            "samples_per_src": SAMPLES_PER_SRC,
            "max_paths_per_src": MAX_PATHS_PER_SRC,
            "synthetic_array": True,
            "diffraction": False,
            "normalize_delays": False,
            "normalize": False,
        },
        "search": {
            "array_candidates": len(stage1_configs),
            "array_frequency_candidates": len(stage2_configs),
            "final_port_candidates": len(stage3_configs),
            "selection_metric": "mean of linkwise PDP cosine and leaked full PAS cosine",
        },
        "direct": next(row for row in stage1_rows if row["name"] == baseline_name),
        "best": stage3_rows[best_index],
        "top_array": [stage1_rows[int(i)] for i in stage1_order[:12]],
        "top_final": [stage3_rows[int(i)] for i in final_order[:20]],
        "spatial_nested": spatial_nested_selection(
            stage3_rows, stage3_metrics, fold_codes, baseline_name
        ),
        "phase_reference": phase_reference_audit(
            source,
            truth_all,
            indices,
            best_config,
            fold_codes,
            args.metric_batch_size,
        ),
        "decision": {},
        "outputs": {
            "cfr_cache": PATHS.relative(cfr_cache_path(len(indices))),
            "metadata": PATHS.relative(metadata_path(len(indices))),
            "report": PATHS.relative(REPORT),
            "submission_zip": None,
        },
    }
    nested = report["spatial_nested"]
    phase = report["phase_reference"]
    report["decision"] = {
        "global_renderer_mismatch_supported": bool(
            nested["mean_gain_vs_direct"] >= 0.005
            and nested["positive_folds"] >= 4
        ),
        "phase_reference_mismatch_supported": bool(
            phase["mean_gain_vs_no_shift"] >= 0.01
            and phase["positive_folds"] >= 4
        ),
        "renderer_gate": "spectral gain >= 0.005 and at least 4/5 positive spatial folds",
        "phase_gate": "coherence gain >= 0.01 and at least 4/5 positive spatial folds",
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "direct_spectral": report["direct"]["spectral"],
                "best_name": report["best"]["name"],
                "best_spectral": report["best"]["spectral"],
                "nested_gain": nested["mean_gain_vs_direct"],
                "nested_positive_folds": nested["positive_folds"],
                "phase_gain": phase["mean_gain_vs_no_shift"],
                "report": PATHS.relative(REPORT),
            },
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    for path in (SCENE_XML, TRAIN_POS, TRAIN_H, V503_REPORT):
        if not path.is_file():
            raise FileNotFoundError(path)
    indices = selected_indices(args.point_limit)
    if args.stage in ("build", "all"):
        build_cfr_cache(args, indices)
    if args.stage in ("audit", "all"):
        if not cfr_cache_path(len(indices)).is_file():
            raise FileNotFoundError(cfr_cache_path(len(indices)))
        audit(args, indices)


if __name__ == "__main__":
    main()
