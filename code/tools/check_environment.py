#!/usr/bin/env python3
"""Validate the active Python/conda environment before project work."""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys


REQUIRED = {
    "numpy": "numpy",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "torch": "pytorch",
    "matplotlib": "matplotlib",
}
OPTIONAL = {
    "pandas": "pandas",
    "yaml": "pyyaml",
    "tqdm": "tqdm",
    "trimesh": "trimesh",
}


def missing_modules(modules: dict[str, str]) -> list[str]:
    missing: list[str] = []
    for module in modules:
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(module)
    return missing


def install_with_conda(modules: list[str]) -> None:
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        raise RuntimeError("No active conda environment; activate one before installing.")
    packages = sorted({(REQUIRED | OPTIONAL)[module] for module in modules})
    command = ["conda", "install", "--yes", "--prefix", prefix, *packages]
    print("Installing missing packages:", " ".join(packages), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--install-missing", action="store_true")
    args = parser.parse_args()

    modules = dict(REQUIRED)
    if args.include_optional:
        modules.update(OPTIONAL)
    missing = missing_modules(modules)
    if missing and args.install_missing:
        install_with_conda(missing)
        missing = missing_modules(modules)

    print(f"python={sys.executable}")
    print(f"conda_prefix={os.environ.get('CONDA_PREFIX', '<not active>')}")
    if missing:
        print("missing=" + ",".join(missing), file=sys.stderr)
        return 1

    import numpy
    import scipy
    import sklearn
    import torch

    print(f"numpy={numpy.__version__}")
    print(f"scipy={scipy.__version__}")
    print(f"sklearn={sklearn.__version__}")
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

