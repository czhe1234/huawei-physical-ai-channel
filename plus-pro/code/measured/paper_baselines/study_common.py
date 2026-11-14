"""Shared, auditable inputs and outputs for dc01 single-frequency adaptations."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE.parent))
from project_paths import PATHS

PREPARED = PATHS.artifacts / "measured/dc01_single_frequency_v1"
ARTIFACTS = PATHS.artifacts / "measured/dc01_paper_baselines_sf_v1"
REPORTS = PATHS.analysis / "results/dc01_paper_baselines_single_frequency"
AUTHORS = PATHS.code_root / "third_party/paper_baselines"


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8388608), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_numpy(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_author(name, file, module_name):
    path = AUTHORS / name / file
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def study_data(fold):
    h = np.load(PREPARED / "channels.npy", mmap_mode="r", allow_pickle=False)
    meta = dict(np.load(PREPARED / "selection.npz", allow_pickle=False))
    splits = json.loads((PREPARED / "split_manifest.json").read_text())["splits"]
    ids = {key: np.asarray(splits[fold][key], dtype=np.int64) for key in ["fit", "val", "test"]}
    if any(set(ids[a]) & set(ids[b]) for a, b in [("fit", "val"), ("fit", "test"), ("val", "test")]):
        raise ValueError("Overlapping position partitions")
    if h.shape[1:] != (2, 8, 4, 1) or h.dtype != np.complex64:
        raise ValueError("Unexpected single-frequency measurement shape")
    return h, meta, ids


def training_identity(method, fold, config, dependencies=()):
    files = [PREPARED / "identity.json", PREPARED / "split_manifest.json",
             CODE / "study_common.py", CODE / "single_frequency_metrics.py", *dependencies]
    result = {"method": method, "adaptation": "dc01 single-frequency, explicitly adapted",
              "fold": fold, "config": config,
              "files": {PATHS.relative(path): sha256(path) for path in files}}
    result["sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return result


def freeze_identity(folder, identity):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "training_identity.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("Run identity changed; use a distinct output directory")
    atomic_json(path, identity)


def save_prediction(folder, prediction, source_ids, truth):
    from single_frequency_metrics import evaluate
    prediction = np.asarray(prediction, dtype=np.complex64)
    if prediction.shape != truth.shape or not np.isfinite(prediction).all():
        raise ValueError("Invalid prediction shape or nonfinite values")
    atomic_numpy(folder / "test_prediction.npy", prediction)
    atomic_numpy(folder / "test_source_ids.npy", source_ids)
    reread = np.load(folder / "test_prediction.npy", allow_pickle=False)
    metrics, point_metrics = evaluate(reread, truth)
    temp = folder / "test_metrics.npz.partial"
    with temp.open("wb") as stream:
        np.savez(stream, source_id=source_ids, **point_metrics)
    os.replace(temp, folder / "test_metrics.npz")
    return metrics
