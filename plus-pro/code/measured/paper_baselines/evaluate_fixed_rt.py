"""Fixed-material RT and a separately labelled fit-only global-gain reference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import time
import traceback

import numpy as np

from materials_rt import array_geometry, channel, initialize, paths_from_dicts, target_axes
from study_common import ARTIFACTS, AUTHORS, CODE, PREPARED, REPORTS, atomic_json, atomic_numpy, freeze_identity, save_prediction, sha256, study_data, training_identity


def energy(value):
    return np.sum(np.abs(value.astype(np.complex128))**2, axis=(2, 3, 4))


def run(args):
    folder = ARTIFACTS/args.run_name/f"fold{args.fold}"
    reports = REPORTS/"fixed_rt"
    shared = ARTIFACTS/"shared"
    started = time.time()
    h, meta, ids = study_data(args.fold)
    geometry_report = json.loads((shared/"rt_geometry_report.json").read_text())
    if sha256(shared/"traced_geometry.npz") != geometry_report["geometry_sha256"]:
        raise ValueError("Frozen geometry cache mismatch")
    dependencies = [Path(__file__), CODE/"materials_rt.py", AUTHORS/"neural_materials/code/utils.py",
                    shared/"traced_geometry.npz", shared/"array_geometry.npz"]
    dependencies.extend(sorted((AUTHORS/"neural_materials/scenes/inue_simple").rglob("*.*")))
    dependencies = [path for path in dependencies if path.is_file()]
    identity = training_identity("fixed_material_rt", args.fold, {
        "frequency_hz": 3.438e9, "learned_materials": False, "neural_path_correction": False,
        "max_depth": geometry_report["max_depth"], "samples_per_source": geometry_report["samples_per_source"],
        "synthetic_array": True, "diffraction": True, "scattering": False,
        "batch_positions": args.batch_size,
        "variants": ["native uncalibrated", "one positive amplitude scalar from total fit energy"],
        "gain_formula": "sqrt(sum_fit_abs_H_measured_squared/sum_fit_abs_H_rt_squared)",
        "gain_phase": 0., "gain_per_panel": False, "gain_per_position": False,
        "exact_original_24_seed_candidate_trace": False}, dependencies)
    freeze_identity(folder, identity)
    if (folder/"result.json").exists():
        print(json.dumps({"status": "already_complete", "run_name": args.run_name}), flush=True)
        return
    tf, utils, _, scene = initialize(args.gpu, args.batch_size)
    if scene.radio_material_callable is not None:
        raise ValueError("Pure RT must use fixed scene materials")
    order, antenna_positions = array_geometry(scene, tf)
    saved_array = dict(np.load(shared/"array_geometry.npz", allow_pickle=False))
    if not np.array_equal(order, saved_array["antenna_order"]) or not np.allclose(antenna_positions, saved_array["antenna_position"], atol=1e-7, rtol=0):
        raise ValueError("RT reference array differs from frozen comparison")
    materials = {}
    for name, obj in scene.objects.items():
        material = obj.radio_material
        materials[name] = {"material": material.name,
                           "relative_permittivity": float(material.relative_permittivity.numpy()),
                           "conductivity": float(material.conductivity.numpy())}
    raw = dict(np.load(shared/"traced_geometry.npz", allow_pickle=False))
    native, axes = [], target_axes()
    for index in range(8):
        fields = {}
        for key, value in raw.items():
            if not key.startswith(f"{index}::"):
                continue
            if value.dtype.kind in "iu":
                if value.size and (value.max() > np.iinfo(np.int32).max or value.min() < np.iinfo(np.int32).min):
                    raise ValueError("Geometry integer exceeds native int32")
                value = value.astype(np.int32)
            elif value.dtype == np.float64:
                value = value.astype(np.float32)
            fields[key.split("::", 1)[1]] = tf.constant(value)
        native.append(fields)
    del raw

    def predict(rows, name):
        path = folder/f"{name}_raw_prediction.npy"
        if path.exists():
            value = np.load(path, allow_pickle=False)
        else:
            blocks = []
            for first in range(0, len(rows), args.batch_size):
                take = rows[first:first+args.batch_size].astype(np.int32)
                count = len(take)
                if count < args.batch_size:
                    take = np.pad(take, (0, args.batch_size-count), mode="edge")
                dictionaries = [{key: tf.gather(value, take, axis=axes[key]) if key in axes else value
                                 for key, value in obj.items()} for obj in native]
                utils.set_receiver_positions(scene, tf.constant(meta["position"][take], tf.float32))
                traced = paths_from_dicts(scene, dictionaries)
                blocks.append(channel(scene, tf, traced, order)[0].numpy()[:count])
                if first % 64 == 0:
                    atomic_json(folder/"progress.json", {"status": "evaluating", "partition": name,
                                "positions_done": min(first+count, len(rows)), "positions_total": len(rows),
                                "elapsed_seconds": time.time()-started})
            value = np.concatenate(blocks).astype(np.complex64)
            atomic_numpy(path, value)
        if value.shape != (len(rows), 2, 8, 4, 1) or value.dtype != np.complex64 or not np.isfinite(value).all():
            raise ValueError("Invalid native RT field")
        return value

    fit_prediction = predict(ids["fit"], "fit")
    measured_fit_energy = float(energy(np.array(h[ids["fit"]])).sum())
    simulated_fit_energy = float(energy(fit_prediction).sum())
    if simulated_fit_energy <= 0:
        raise ValueError("No fit RT energy; gain cannot be estimated")
    gain = float(np.sqrt(measured_fit_energy/simulated_fit_energy))
    calibration = {"amplitude_gain": gain, "power_offset_db": float(20*np.log10(gain)),
                   "fit_positions": len(ids["fit"]), "measured_fit_energy": measured_fit_energy,
                   "simulated_fit_energy": simulated_fit_energy, "phase_calibrated": False,
                   "validation_or_test_used": False}
    atomic_json(folder/"global_gain.json", calibration)
    # Freeze the sole gain before producing/scoring held-out predictions.
    test_prediction = predict(ids["test"], "test")
    truth = np.array(h[ids["test"]])
    variants = []
    for name, scalar in [("raw", 1.), ("global_gain", gain)]:
        output = folder/name
        output.mkdir(parents=True, exist_ok=True)
        prediction = (test_prediction.astype(np.complex128)*scalar).astype(np.complex64)
        metrics = save_prediction(output, prediction, meta["source_id"][ids["test"]], truth)
        zero_panels = int((energy(prediction) == 0).sum())
        metrics["power_mae_is_infinite"] = zero_panels > 0
        metrics["numeric_floor_power_mae_db"] = metrics["power_mae_db"]
        if zero_panels:
            metrics["power_mae_db"] = None
        variants.append({"name": name, "amplitude_gain": scalar, "metrics": metrics,
                         "zero_prediction_panels": zero_panels,
                         "prediction_energy_over_reference_energy": float(energy(prediction).sum()/energy(truth).sum()),
                         "prediction_path": str(output.relative_to(ARTIFACTS)/"test_prediction.npy"),
                         "prediction_sha256": sha256(output/"test_prediction.npy")})
    if abs(variants[0]["metrics"]["array_pas_cosine"]-variants[1]["metrics"]["array_pas_cosine"]) > 1e-6 or abs(variants[0]["metrics"]["dft_beam_loss_db"]-variants[1]["metrics"]["dft_beam_loss_db"]) > 1e-6:
        raise ValueError("A global gain unexpectedly changed normalized structure or beam selection")
    try:
        gpu_memory = tf.config.experimental.get_memory_info("GPU:0")["peak"]/1024**3
    except (ValueError, KeyError):
        gpu_memory = None
    result = {"status": "complete", "identity": identity, "scene_materials": materials,
              "calibration": calibration, "variants": variants, "elapsed_seconds": time.time()-started,
              "hardware": tf.config.experimental.get_device_details(tf.config.list_physical_devices("GPU")[0]).get("device_name"),
              "peak_gpu_gib_tensorflow_allocator": gpu_memory,
              "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
              "geometry_preprocessing_seconds_reused": geometry_report["elapsed_seconds"],
              "neural_network_training": False, "fold": args.fold, "test_positions": len(ids["test"]),
              "scope": "fixed-material depth5 RT; not the exact original24-seed depth16/depth10 generator"}
    atomic_json(folder/"result.json", result)
    atomic_json(reports/"evaluation.json", result)
    atomic_json(folder/"progress.json", {"status": "complete"})
    print(json.dumps({"status": "complete", "calibration": calibration, "variants": variants,
                      "elapsed_seconds": result["elapsed_seconds"]}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--run-name", default="fixed_material_rt")
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        atomic_json(ARTIFACTS/args.run_name/f"fold{args.fold}"/"failure.json",
                    {"status": "failed", "traceback": traceback.format_exc()})
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
