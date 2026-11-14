#!/usr/bin/env python3
"""Retrain the v472 magnitude residual on the deployment-isomorphic v473 carrier."""

from __future__ import annotations

import zipfile

import numpy as np

import build_v473_train_v469like_carrier as v473
import train_v471_complex_phase_residual_network as v471
import train_v472_cir_magnitude_residual_network as trainer
from generate_v159_frontier_consensus_support_soup import digest
from project_paths import PATHS


VERSION_NUMBER = 474
VERSION = "v474_v469like_magnitude_residual_network"
CACHE = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
CACHE.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def write_package(report: dict[str, object]) -> None:
    with zipfile.ZipFile(trainer.ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(trainer.OUTPUT, "Round1_Test_Channel.npy")
        archive.write(__file__, "train_v474_v469like_magnitude_residual_network.py")
        archive.write(
            PATHS.code / "train_v472_cir_magnitude_residual_network.py",
            "train_v472_cir_magnitude_residual_network.py",
        )
        archive.writestr("model/", b"")
        archive.write(trainer.CHECKPOINT, "model/cir_magnitude_residual.pt")
        archive.write(
            PATHS.code / "train_v471_complex_phase_residual_network.py",
            "train_v471_complex_phase_residual_network.py",
        )
        archive.write(PATHS.code_root / "project_paths.py", "project_paths.py")
    with zipfile.ZipFile(trainer.ZIP_OUT) as archive:
        bad = archive.testzip()
        members = archive.namelist()
    if bad is not None:
        raise ValueError(f"ZIP CRC failure: {bad}")
    prediction = np.load(trainer.OUTPUT, mmap_mode="r")
    finite = all(
        np.isfinite(np.asarray(prediction[start : start + 5])).all()
        for start in range(0, len(prediction), 5)
    )
    report["integrity"] = {
        "shape": list(prediction.shape),
        "dtype": str(prediction.dtype),
        "finite": bool(finite),
        "zip_members": members,
        "sha256": digest(trainer.ZIP_OUT, "sha256"),
        "md5": digest(trainer.ZIP_OUT, "md5"),
    }


def configure() -> None:
    v471.TRAIN_BASE = v473.OUTPUT
    trainer.VERSION_NUMBER = VERSION_NUMBER
    trainer.VERSION = VERSION
    trainer.CACHE = CACHE
    trainer.REPORT_DIR = REPORT_DIR
    trainer.CHECKPOINT_DIR = CHECKPOINT_DIR
    trainer.VAL_SHAPE = CACHE / "v474_validation_link_shape.npy"
    trainer.VAL_ENERGY = CACHE / "v474_validation_link_energy.npy"
    trainer.TEST_SHAPE = CACHE / "v474_test_link_shape.npy"
    trainer.TEST_ENERGY = CACHE / "v474_test_link_energy.npy"
    trainer.OUTPUT = CACHE / "v474_Round1_Test_Channel.npy"
    trainer.CHECKPOINT = CHECKPOINT_DIR / "cir_magnitude_residual.pt"
    trainer.REPORT = REPORT_DIR / f"{VERSION}.json"
    trainer.ZIP_OUT = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"
    trainer.SEED = 47401
    trainer.write_package = write_package


def main() -> None:
    configure()
    trainer.main()


if __name__ == "__main__":
    main()
