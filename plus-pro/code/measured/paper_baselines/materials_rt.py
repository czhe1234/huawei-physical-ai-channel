"""Sionna 0.19 bridge using the pinned author's Neural Materials implementation.

Adaptations: calibrated dc01 single-frequency teacher, synthetic array for
tractable caching, power-only loss, and the shared position split. Geometry
files contain no measured channel labels.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import pickle
import time
import traceback

import numpy as np

from study_common import ARTIFACTS, AUTHORS, CODE, PATHS, PREPARED, REPORTS, atomic_json, atomic_numpy, load_author, sha256, study_data


def initialize(gpu, count):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    tensorflow_spec = importlib.util.find_spec("tensorflow")
    cuda_compiler = Path(tensorflow_spec.origin).parent.parent / "nvidia/cuda_nvcc"
    if (cuda_compiler / "bin/ptxas").is_file():
        os.environ["PATH"] = str(cuda_compiler / "bin") + os.pathsep + os.environ.get("PATH", "")
        if "xla_gpu_cuda_data_dir" not in os.environ.get("XLA_FLAGS", ""):
            os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_gpu_cuda_data_dir=" + str(cuda_compiler)).strip()
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    for device in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(device, True)
    tf.random.set_seed(20261017)
    os.chdir(AUTHORS / "neural_materials/code")
    utils = load_author("neural_materials", "code/utils.py", "dc01_original_rt_utils")
    materials = load_author("neural_materials", "code/neural_materials.py", "dc01_original_neural_materials")
    scene = utils.init_scene("inue_simple", use_tx_array=True)
    scene.synthetic_array = True
    utils.place_transmitter_arrays(scene, [1, 2])
    utils.instantiate_receivers(scene, count)
    _, meta, _ = study_data(0)
    for panel in range(2):
        scene.transmitters[f"tx-{panel+1}"].position = meta["panel_position"][panel].astype(np.float32)
    return tf, utils, materials, scene


def target_axes():
    module = ast.parse((AUTHORS / "neural_materials/code/utils.py").read_text())
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "serialize_traced_paths")
    assignment = next(node for node in function.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "target_axis" for t in node.targets))
    axes = ast.literal_eval(assignment.value)
    axes.update(doppler=0, targets_sources_mask=0)
    return axes


def array_geometry(scene, tf):
    from sionna.rt import rotation_matrix
    local = scene.tx_array.positions.numpy()
    horizontal = np.unique(np.round(local[:, 1], 7))
    vertical = np.unique(np.round(local[:, 2], 7))[::-1]
    if len(horizontal) != 8 or len(vertical) != 4:
        raise ValueError("Unexpected physical 4x8 array coordinates")
    order = np.array([int(np.argmin(np.linalg.norm(local - np.array([0., y, z]), axis=1)))
                      for y in horizontal for z in vertical], dtype=np.int64)
    if len(set(order.tolist())) != 32:
        raise ValueError("Antenna permutation is not one-to-one")
    world = []
    for panel in range(2):
        tx = scene.transmitters[f"tx-{panel+1}"]
        rotation = rotation_matrix(tx.orientation).numpy()
        world.append(tx.position.numpy() + local[order] @ rotation.T)
    return order, np.asarray(world, np.float32).reshape(2, 8, 4, 3)


def channel(scene, tf, traced, antenna_order):
    from sionna.channel import cir_to_ofdm_channel
    paths = scene.compute_fields(*traced, scat_random_phases=False, check_scene=False)
    a, tau = paths.cir(scattering=False)
    h = cir_to_ofdm_channel(tf.constant([0.], tf.float32), a, tau)
    h = tf.squeeze(h, axis=[0, 2, 5])
    h = tf.gather(h, antenna_order, axis=2)
    return tf.reshape(h, [-1, 2, 8, 4, 1]), paths


def trace(scene, samples):
    return scene.trace_paths(num_samples=samples, max_depth=5, los=True,
                             reflection=True, diffraction=True, edge_diffraction=False,
                             scattering=False, check_scene=False)


def split_traced(tf, utils, scene, traced, item, axes):
    dictionaries = []
    for obj in traced:
        result = {}
        for key, value in obj.to_dict().items():
            if key.startswith("scat_prefix_"):
                continue
            if key in axes:
                value = tf.gather(value, [item], axis=axes[key])
            result[key] = value
        dictionaries.append(result)
    return paths_from_dicts(scene, dictionaries)


def paths_from_dicts(scene, dictionaries):
    """Preserve Sionna 0.19's additional empty RIS path/tmp slots."""
    from sionna.rt import Paths
    from sionna.rt.solver_paths import PathsTmpData
    if len(dictionaries) != 8:
        raise ValueError("This adapter requires Sionna 0.19's eight traced path objects")
    sources, targets = dictionaries[0]["sources"], dictionaries[0]["targets"]
    result = []
    for index, data in enumerate(dictionaries):
        obj = Paths(sources, targets, scene) if index < 4 else PathsTmpData(sources, targets, scene.dtype)
        obj.from_dict(data)
        result.append(obj)
    return tuple(result)


def probe(args):
    h, meta, ids = study_data(0)
    rows = ids["fit"][:4]
    started = time.time()
    tf, utils, material_code, scene = initialize(args.gpu, len(rows))
    order, antenna_position = array_geometry(scene, tf)
    geometry = ARTIFACTS / "shared"
    geometry.mkdir(parents=True, exist_ok=True)
    temporary = geometry / "array_geometry.npz.partial"
    with temporary.open("wb") as stream:
        np.savez(stream, antenna_order=order, antenna_position=antenna_position)
    os.replace(temporary, geometry / "array_geometry.npz")
    utils.set_receiver_positions(scene, meta["position"][rows].astype(np.float32))
    t = time.time()
    traced = trace(scene, args.samples)
    tracing_seconds = time.time() - t
    summary = [{key: list(value.shape) for key, value in obj.to_dict().items() if key in ["mask", "objects", "vertices", "targets", "sources", "a"]}
               for obj in traced]
    atomic_json(REPORTS / "materials_path_shapes.json", [
        {key: list(np.shape(value)) for key, value in obj.to_dict().items()} for obj in traced])
    axes = target_axes()
    one = split_traced(tf, utils, scene, traced, 0, axes)
    # The original serializer predates RIS. Only its six matching objects enter
    # that round-trip; our production cache preserves all eight native objects.
    serialized = utils.serialize_traced_paths(tf.constant(meta["position"][rows[0]], tf.float32), tf.zeros([2, 32, 1], tf.complex64),
                                              [one[i] for i in [0, 1, 2, 4, 5, 6]], True)
    decoded = utils.deserialize_paths_as_tensor_dicts(tf.constant(serialized))
    if not decoded:
        raise ValueError("Author path round-trip failed")
    initial, _ = channel(scene, tf, traced, order)
    scene.radio_material_callable = material_code.NeuralMaterials(scene, pos_encoding_size=10, learn_scattering=False)
    teacher = tf.constant(np.array(h[rows]), tf.complex64)
    with tf.GradientTape() as tape:
        prediction, _ = channel(scene, tf, traced, order)
        alpha = tf.stop_gradient(tf.reduce_sum(tf.abs(prediction)**2) / tf.reduce_sum(tf.abs(teacher)**2))
        loss = utils.power_loss(tf.reshape(prediction, [-1, 1]), tf.reshape(teacher * tf.complex(tf.sqrt(alpha), 0.), [-1, 1]))
    gradients = tape.gradient(loss, scene.radio_material_callable.trainable_variables)
    if not gradients or any(g is None or not np.isfinite(g.numpy()).all() for g in gradients):
        raise ValueError("Invalid differentiable material gradient")
    result = {"status": "passed", "shape": list(prediction.shape), "finite": bool(np.isfinite(prediction.numpy()).all()),
              "traced_path_shapes": summary, "trace_seconds_four_positions": tracing_seconds,
              "serialized_point_bytes": len(serialized), "gradient_variables": len(gradients),
              "gradient_norm": float(tf.linalg.global_norm(gradients)), "power_loss": float(loss),
              "initial_itu_power": float(tf.reduce_sum(tf.abs(initial)**2)),
              "samples_per_source": args.samples, "synthetic_array": True,
              "antenna_permutation": order.tolist(), "elapsed_seconds": time.time() - started}
    atomic_json(REPORTS / "materials_probe.json", result)
    print(json.dumps(result), flush=True)


def prepare(args):
    shared = ARTIFACTS / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    report_path = shared / "rt_geometry_report.json"
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        if previous["samples_per_source"] != args.samples or sha256(shared / "traced_geometry.npz") != previous["geometry_sha256"]:
            raise ValueError("Existing geometry cache identity mismatch")
        print(json.dumps({"stage": "geometry_reused", "positions": previous["positions"]}), flush=True)
        return
    _, meta, _ = study_data(0)
    started = time.time()
    tf, utils, _, scene = initialize(args.gpu, len(meta["position"]))
    order, antenna_position = array_geometry(scene, tf)
    temporary = shared / "array_geometry.npz.partial"
    with temporary.open("wb") as stream:
        np.savez(stream, antenna_order=order, antenna_position=antenna_position)
    os.replace(temporary, shared / "array_geometry.npz")
    utils.set_receiver_positions(scene, meta["position"].astype(np.float32))
    traced = trace(scene, args.samples)
    arrays = {f"{i}::{key}": (value.numpy() if hasattr(value, "numpy") else np.asarray(value)) for i, obj in enumerate(traced)
              for key, value in obj.to_dict().items() if not key.startswith("scat_prefix_")}
    temporary = shared / "traced_geometry.npz.partial"
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, shared / "traced_geometry.npz")
    result = {"status": "complete", "positions": len(meta["position"]),
              "samples_per_source": args.samples, "max_depth": 5,
              "synthetic_array": True, "scattering": False, "diffraction": True,
              "geometry_contains_measured_channels": False, "target_axes": target_axes(),
              "fields": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in arrays.items()},
              "geometry_sha256": sha256(shared / "traced_geometry.npz"),
              "position_identity": sha256(PREPARED / "selection.npz"),
              "source_implementation": sha256(Path(__file__)),
              "elapsed_seconds": time.time() - started}
    atomic_json(report_path, result)
    print(json.dumps({key: result[key] for key in ["status", "positions", "elapsed_seconds", "geometry_sha256"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["probe", "prepare"], required=True)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--samples", type=int, default=1000000)
    args = parser.parse_args()
    (probe if args.mode == "probe" else prepare)(args)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
