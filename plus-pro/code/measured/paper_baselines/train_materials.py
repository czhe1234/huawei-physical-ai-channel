"""Power-supervised single-frequency adaptation of the author's Neural Materials."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
import math
from pathlib import Path
import pickle
import resource
import sys
import time
import traceback

import numpy as np

from materials_rt import channel, initialize, paths_from_dicts, target_axes
from study_common import ARTIFACTS, AUTHORS, CODE, PREPARED, REPORTS, atomic_json, freeze_identity, save_prediction, sha256, study_data, training_identity


def atomic_pickle(path, value):
    temp = path.with_name(path.name + ".partial")
    with temp.open("wb") as stream:
        pickle.dump(value, stream, protocol=4)
    temp.replace(path)


def run(args):
    started = time.time()
    h, meta, ids = study_data(args.fold)
    tf, utils, original, scene = initialize(args.gpu, 8)
    tf.random.set_seed(20261017+args.fold)
    shared = ARTIFACTS / "shared"
    geometry_report = json.loads((shared / "rt_geometry_report.json").read_text())
    if sha256(shared / "traced_geometry.npz") != geometry_report["geometry_sha256"]:
        raise ValueError("Geometry cache checksum mismatch")
    order = np.load(shared / "array_geometry.npz")["antenna_order"]
    raw = dict(np.load(shared / "traced_geometry.npz", allow_pickle=False))
    axes = target_axes()
    native = []
    for i in range(8):
        fields = {}
        for key, value in raw.items():
            if not key.startswith(f"{i}::"):
                continue
            # Python scalar metadata otherwise becomes platform int64/float64
            # in NPZ; Sionna's complex64 scene requires int32/float32 metadata.
            if value.dtype.kind in "iu":
                if value.size and (np.max(value) > np.iinfo(np.int32).max or np.min(value) < np.iinfo(np.int32).min):
                    raise ValueError("Geometry integer does not fit native Sionna int32")
                value = value.astype(np.int32)
            elif value.dtype == np.float64:
                value = value.astype(np.float32)
            fields[key.split("::", 1)[1]] = tf.constant(value)
        native.append(fields)
    del raw
    positions = tf.constant(meta["position"], tf.float32)

    def field(rows):
        dictionaries = [{key: tf.gather(value, rows, axis=axes[key]) if key in axes else value
                         for key, value in obj.items()} for obj in native]
        utils.set_receiver_positions(scene, tf.gather(positions, rows))
        traced = paths_from_dicts(scene, dictionaries)
        return channel(scene, tf, traced, order)[0]

    scene.radio_material_callable = original.NeuralMaterials(scene, pos_encoding_size=10, learn_scattering=False)
    initial_rows = ids["fit"][:8].astype(np.int32)
    eager = field(tf.constant(initial_rows))
    if eager.shape != (8, 2, 8, 4, 1) or not np.isfinite(eager.numpy()).all():
        raise ValueError("Cached native fields have invalid shape or values")
    material = scene.radio_material_callable
    optimizer = tf.keras.optimizers.Adam(1e-3)
    optimizer.build(material.trainable_variables)
    def optimizer_variables():
        values = optimizer.variables
        return values() if callable(values) else values
    alpha = tf.Variable(6e-9, trainable=False, dtype=tf.float32)

    @tf.function(reduce_retracing=True)
    def inference(rows):
        return field(rows)

    teacher = tf.constant(np.array(h[initial_rows]), tf.complex64)
    with tf.GradientTape() as tape:
        value = field(tf.constant(initial_rows))
        scaled = teacher * tf.complex(tf.sqrt(tf.stop_gradient(tf.reduce_sum(tf.abs(value)**2)/tf.reduce_sum(tf.abs(teacher)**2))), 0.)
        loss = utils.power_loss(tf.reshape(value, [-1, 1]), tf.reshape(scaled, [-1, 1]))
    gradients = tape.gradient(loss, material.trainable_variables)
    if any(g is None or not np.isfinite(g.numpy()).all() for g in gradients) or float(tf.linalg.global_norm(gradients)) == 0:
        raise ValueError("Cached material gradients are invalid")
    graph = inference(tf.constant(initial_rows))
    graph_difference = float(tf.reduce_max(tf.abs(graph-eager)))
    if not np.allclose(graph.numpy(), eager.numpy(), atol=1e-7, rtol=5e-5):
        raise ValueError("Graph field computation differs from eager native fields")
    if args.mode == "probe":
        result = {"status": "passed", "cached_geometry_objects": len(native),
                  "shape": list(eager.shape), "gradients_finite_nonzero": True,
                  "gradient_norm": float(tf.linalg.global_norm(gradients)),
                  "graph_vs_eager_max_complex_error": graph_difference,
                  "elapsed_seconds": time.time()-started,
                  "query_csi_used": False, "wideband_measurements_used": False}
        atomic_json(REPORTS / "materials_cache_probe.json", result)
        print(json.dumps(result), flush=True)
        return
    folder = ARTIFACTS / args.run_name / f"fold{args.fold}"
    config = {"frequency_hz": 3.438e9, "batch_positions": 8, "hidden_layers": 4, "hidden_units": 128,
              "position_encoding": 10, "loss": "author power SMAPE, no delay-spread term",
              "optimizer": "Adam", "learning_rate": .001, "maximum_updates": 10000,
              "validation_interval": 250, "patience": None if args.full_budget else 8,
              "minimum_updates": 10000 if args.full_budget else 1000,
              "trained_checkpoints_only": args.trained_checkpoints_only,
              "minimum_validation_improvement": 1e-4, "scale_ema": .999,
              "initial_scale": "fit-only closed-form initial material power calibration",
              "synthetic_array": True, "samples_per_source": geometry_report["samples_per_source"],
              "max_depth": 5, "scattering": False, "diffraction": True,
              "sionna_019_ris_slots_preserved": True}
    identity = training_identity("neural_materials", args.fold, config,
                                 [Path(__file__), CODE / "materials_rt.py",
                                  AUTHORS / "neural_materials/code/neural_materials.py",
                                  shared / "traced_geometry.npz", shared / "array_geometry.npz"])
    freeze_identity(folder, identity)
    if (folder / "result.json").exists():
        print(json.dumps({"status": "already_complete", "method": "neural_materials"}), flush=True)
        return

    @tf.function(reduce_retracing=True)
    def training_step(rows, observation):
        with tf.GradientTape() as tape:
            prediction = field(rows)
            p = tf.reshape(prediction, [-1, 1])
            t = tf.reshape(observation, [-1, 1])
            calibration = utils.mse_power_scaling_factor(p, t)
            alpha.assign(.999*alpha+.001*calibration)
            loss = utils.power_loss(p, t*tf.complex(tf.sqrt(alpha), 0.))
        grads = tape.gradient(loss, material.trainable_variables)
        for grad in grads:
            tf.debugging.assert_all_finite(grad, "Nonfinite material gradient")
        tf.debugging.assert_all_finite(loss, "Nonfinite material objective")
        optimizer.apply_gradients(zip(grads, material.trainable_variables))
        return loss

    rng = np.random.default_rng(20261017+args.fold)
    step, best, best_step, stale, history = 0, float("inf"), 0, 0, []
    best_weights, best_alpha, elapsed_before = None, None, 0.
    if (folder / "resume.pkl").exists():
        with (folder / "resume.pkl").open("rb") as stream:
            state = pickle.load(stream)
        if state["identity"] != identity:
            raise ValueError("Material checkpoint identity mismatch")
        material.set_weights(state["weights"])
        for variable, value in zip(optimizer_variables(), state["optimizer"]):
            variable.assign(value)
        alpha.assign(state["alpha"])
        step, best, best_step, stale, history, best_weights, best_alpha = (state[key] for key in ["step", "best", "best_step", "stale", "history", "best_weights", "best_alpha"])
        rng.bit_generator.state = state["rng"]
        elapsed_before = state["elapsed_seconds"]
    else:
        # A frequency-specific initial scale is necessary after removing the
        # original 1024-carrier normalization. It uses fit observations only.
        numerator, denominator = 0., 0.
        for first in range(0, len(ids["fit"]), 8):
            rows = ids["fit"][first:first+8]
            padded = np.pad(rows, (0, 8-len(rows)), mode="edge").astype(np.int32)
            value = inference(tf.constant(padded)).numpy()[:len(rows)]
            p = np.abs(value.astype(np.complex128))**2
            t = np.abs(np.array(h[rows], dtype=np.complex128))**2
            numerator += float((p*t).sum())
            denominator += float((t*t).sum())
        alpha.assign(numerator/denominator)
        if not math.isfinite(float(alpha)) or float(alpha) <= 0:
            raise ValueError("Invalid fit-only power calibration")

    def save_resume():
        atomic_pickle(folder / "resume.pkl", {"identity": identity, "weights": material.get_weights(),
                      "optimizer": [value.numpy() for value in optimizer_variables()], "alpha": float(alpha),
                      "step": step, "best": best, "best_step": best_step, "stale": stale,
                      "history": history, "best_weights": best_weights, "best_alpha": best_alpha,
                      "rng": rng.bit_generator.state, "elapsed_seconds": elapsed_before+time.time()-started})

    def predict(rows):
        values = []
        for first in range(0, len(rows), 8):
            selected = np.asarray(rows[first:first+8])
            padded = np.pad(selected, (0, 8-len(selected)), mode="edge").astype(np.int32)
            value = inference(tf.constant(padded)).numpy()[:len(selected)]
            values.append(value/np.sqrt(float(alpha)))
        return np.concatenate(values).astype(np.complex64)

    val_truth = np.array(h[ids["val"]], dtype=np.complex128)
    while step <= 10000:
        if step % 250 == 0 and (step > 0 or not args.trained_checkpoints_only):
            value = predict(ids["val"]).astype(np.complex128)
            p, t = np.abs(value)**2, np.abs(val_truth)**2
            score = float(np.mean(np.abs(p-t)/np.maximum(p+t, 1e-30)))
            nmse = float(np.sum(np.abs(value-val_truth)**2)/np.sum(t))
            if score < best-1e-4 or best_weights is None:
                best, best_step, stale = score, step, 0
                best_weights, best_alpha = material.get_weights(), float(alpha)
            else:
                stale += 1
            row = {"step": step, "validation_power_smape": score, "validation_raw_nmse": nmse,
                   "best_step": best_step, "best_validation_power_smape": best,
                   "elapsed_seconds": elapsed_before+time.time()-started}
            history.append(row)
            atomic_json(folder / "progress.json", {"status": "training", "method": "neural_materials", **row})
            print(json.dumps({"method": "neural_materials", **row}), flush=True)
            save_resume()
            if step == 10000 or (not args.full_budget and step >= 1000 and stale >= 8):
                break
        rows = rng.choice(ids["fit"], size=8, replace=True).astype(np.int32)
        loss = training_step(tf.constant(rows), tf.constant(np.array(h[rows]), tf.complex64))
        step += 1
        if step % 100 == 0:
            atomic_json(folder / "progress.json", {"status": "training", "method": "neural_materials",
                        "step": step, "best_step": best_step, "training_power_loss": float(loss),
                        "elapsed_seconds": elapsed_before+time.time()-started})
    material.set_weights(best_weights)
    alpha.assign(best_alpha)
    atomic_pickle(folder / "model.pkl", {"identity": identity, "weights": best_weights,
                  "measured_scale": best_alpha, "selected_step": best_step})
    # Test labels enter only after model and global calibration are frozen.
    value = predict(ids["test"])
    metrics = save_prediction(folder, value, meta["source_id"][ids["test"]], np.array(h[ids["test"]]))
    memory = tf.config.experimental.get_memory_info("GPU:0")
    result = {"status": "complete", "identity": identity, "selected_step": best_step,
              "executed_updates": step, "test": metrics, "history": history,
              "validation_power_smape": best, "fit_only_power_scale": best_alpha,
              "elapsed_seconds": elapsed_before+time.time()-started,
              "hardware": tf.config.experimental.get_device_details(tf.config.list_physical_devices("GPU")[0]).get("device_name"),
              "peak_gpu_gib_tensorflow_allocator": memory["peak"]/1024**3,
              "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
              "geometry_preprocessing_seconds": geometry_report["elapsed_seconds"],
              "bottleneck": "differentiable EM field computation and material-network gradients"}
    atomic_json(folder / "result.json", result)
    atomic_json(folder / "progress.json", {"status": "complete", "method": "neural_materials", "test": metrics})
    print(json.dumps({"status": "complete", "method": "neural_materials", "test": metrics}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["probe", "train"], required=True)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--run-name", default="neural_materials")
    parser.add_argument("--full-budget", action="store_true")
    parser.add_argument("--trained-checkpoints-only", action="store_true")
    args = parser.parse_args()
    if not args.run_name.replace("_", "").isalnum():
        raise ValueError("Invalid run directory name")
    folder = ARTIFACTS / args.run_name / f"fold{args.fold}"
    folder.mkdir(parents=True, exist_ok=True)

    class Tee:
        def __init__(self, terminal, log):
            self.terminal, self.log = terminal, log

        def write(self, text):
            if not self.log.closed:
                self.log.write(text)
            return self.terminal.write(text)

        def flush(self):
            if not self.log.closed:
                self.log.flush()
            self.terminal.flush()

        def close(self):
            pass

    with (folder / f"{args.mode}.log").open("a", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                run(args)
            except Exception:
                atomic_json(folder / "failure.json", {"status": "failed", "traceback": traceback.format_exc()})
                traceback.print_exc()
                raise


if __name__ == "__main__":
    main()
