"""Shared south-only data contract; query inputs never carry measured labels."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
from project_paths import PATHS

SOUTH = PATHS.artifacts / "measured/hall_sw_publisher_v2"
REPORT = PATHS.analysis / "results/hall_south_adapter_correctness_v1"
ARTIFACTS = PATHS.artifacts / "measured/hall_sw_adapter_correctness_v1"
C0 = 299792458.0

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""): h.update(b)
    return h.hexdigest()

def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)

def frozen_array(value, dtype=None):
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result

@dataclass(frozen=True)
class ArraySpec:
    center: np.ndarray
    direction: np.ndarray
    spacing_m: float = .04364
    carrier_hz: float = 3438000000.
    frequency_count: int = 1024
    bandwidth_hz: float = 50e6

    @property
    def frequencies(self):
        return self.carrier_hz + (np.arange(self.frequency_count) - self.frequency_count // 2) * self.bandwidth_hz / self.frequency_count

    @property
    def axes(self):
        forward = self.direction / np.linalg.norm(self.direction, axis=1, keepdims=True)
        horizontal = np.cross(np.broadcast_to([0., 0., 1.], forward.shape), forward)
        horizontal /= np.linalg.norm(horizontal, axis=1, keepdims=True)
        up = np.cross(forward, horizontal)
        return np.stack([forward, horizontal, up], axis=-1)

    @property
    def element_offsets(self):
        h, v = np.meshgrid(np.arange(8) - 3.5, 1.5 - np.arange(4), indexing="ij")
        local = np.stack([np.zeros_like(h), h, v], axis=-1) * self.spacing_m
        return np.einsum("pij,hvj->phvi", self.axes, local)

@dataclass(frozen=True)
class Query:
    row_ids: np.ndarray
    position: np.ndarray
    spec: ArraySpec

    def __post_init__(self):
        if self.position.shape != (len(self.row_ids), 3) or not np.isfinite(self.position).all():
            raise ValueError("Invalid position-only queries")

@dataclass(frozen=True)
class TrainingBank:
    row_ids: np.ndarray
    position: np.ndarray
    channel: np.ndarray
    spec: ArraySpec

    def __post_init__(self):
        expected = (len(self.row_ids), 2, 8, 4, self.spec.frequency_count)
        if self.channel.shape != expected or self.position.shape != (len(self.row_ids), 3):
            raise ValueError("Training bank layout mismatch")
        if len(np.unique(self.row_ids)) != len(self.row_ids) or not len(self.row_ids):
            raise ValueError("Training IDs must be unique and nonempty")
        if not np.isfinite(self.channel).all(): raise ValueError("Nonfinite training data")

class SouthDataset:
    """No north path accepted. Correctness can read FIT rows only."""
    def __init__(self, directory=SOUTH):
        self.directory = Path(directory)
        if self.directory.resolve() != SOUTH.resolve():
            raise ValueError("This adapter stage accepts the prepared south view only")
        self.manifest = json.loads((self.directory / "preparation_manifest.json").read_text())
        self.protocol = json.loads((self.directory / "split_manifest.json").read_text())
        with np.load(self.directory / "selection.npz", allow_pickle=False) as z:
            self.meta = {k: frozen_array(z[k]) for k in z.files}
        if self.manifest["channel_shape"] != [627, 2, 8, 4, 1024]:
            raise ValueError("Unexpected south preparation")
        for key in ["selection.npz", "split_manifest.json", "model_measurement_frame.npz", "model_objects.json"]:
            if sha256(self.directory / key) != self.manifest["outputs"][key]["sha256"]:
                raise ValueError("Prepared metadata identity changed")
        self.spec = ArraySpec(self.meta["panel_position"], self.meta["panel_direction"])
        if not np.array_equal(self.spec.frequencies, self.meta["frequency_hz"]):
            raise ValueError("Frequency axis differs")
        self.channel_reads = []

    @property
    def identity(self):
        return dict(view=self.directory.name,
                    preparation_manifest_sha256=sha256(self.directory / "preparation_manifest.json"),
                    split_sha256=sha256(self.directory / "split_manifest.json"),
                    selection_sha256=sha256(self.directory / "selection.npz"),
                    channels_sha256=self.manifest["outputs"]["channels.npy"]["sha256"])

    def partition(self, fold, role):
        if role not in ["fit", "validation", "test"]: raise ValueError("Unknown role")
        if not 0 <= fold < 5: raise ValueError("Unknown fold")
        return frozen_array(self.protocol["splits"][fold][role], np.int64)

    def query(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        if np.any((rows < 0) | (rows >= len(self.meta["position"]))): raise ValueError("Unknown row")
        return Query(frozen_array(rows), frozen_array(self.meta["position"][rows]), self.spec)

    def read_fit(self, fold, rows=None):
        allowed = self.partition(fold, "fit")
        rows = allowed if rows is None else np.asarray(rows, dtype=np.int64)
        if len(np.unique(rows)) != len(rows) or not set(rows).issubset(set(allowed)):
            raise ValueError("Measured channel access outside this fold's fit partition")
        source = np.load(self.directory / "channels.npy", mmap_mode="r", allow_pickle=False)
        channel = frozen_array(source[rows], np.complex64)
        self.channel_reads.append(dict(fold=fold, role="fit", rows=rows.tolist()))
        return TrainingBank(frozen_array(rows), frozen_array(self.meta["position"][rows]), channel, self.spec)

def check_prediction(value, query):
    if set(value) - {"channel", "power", "method", "source_ids", "diagnostics", "row_ids"}:
        raise ValueError("Unknown prediction fields")
    if "channel" in value:
        h = np.asarray(value["channel"])
        if h.shape != (len(query.row_ids), 2, 8, 4, query.spec.frequency_count):
            raise ValueError("Wrong complex output shape")
        if h.dtype not in [np.complex64, np.complex128] or not np.isfinite(h).all():
            raise ValueError("Invalid complex output")
    if "power" in value:
        p = np.asarray(value["power"])
        if p.shape != (len(query.row_ids), 2, 32) or not np.isfinite(p).all() or np.any(p < 0):
            raise ValueError("Invalid directional power output")
    if "channel" not in value and "power" not in value: raise ValueError("Empty prediction")
    value = dict(value)
    if "row_ids" in value and not np.array_equal(value["row_ids"], query.row_ids):
        raise ValueError("Output/query row identities differ")
    value["row_ids"] = np.array(query.row_ids, copy=True)
    return value
