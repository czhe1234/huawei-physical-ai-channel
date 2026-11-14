"""Validate the complete production model inventory for a retrained submission."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import RuntimePaths
from .io_utils import atomic_json, portable_path


def expected_checkpoints() -> tuple[Path, ...]:
    paths: list[Path] = []
    for component in ("spectral", "joint"):
        for domain in ("pas", "pdp"):
            for fold in range(5):
                paths.append(
                    Path("base_field")
                    / component
                    / f"{domain}_fold_{fold}.pt"
                )
    paths.extend(Path("path_calibrator") / f"fold_{fold}.pt" for fold in range(5))
    paths.extend(
        Path("base_field") / "path_link" / f"fold_{fold}.pt"
        for fold in range(5)
    )
    paths.append(Path("base_field") / "donor_weight_model.pt")
    return tuple(paths)


EXPECTED_ASSETS = (
    Path("base_field") / "deployment_donor_weights.npz",
    Path("base_field") / "carrier_mix_weights.npz",
    Path("base_field") / "group_calibration.npz",
    Path("base_field") / "projection_policy.json",
    Path("base_field") / "lifecycle_policy.json",
)


def _validate_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload is not a dictionary")
    state = payload.get("state_dict")
    config = payload.get("config")
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint has no non-empty state_dict")
    if not isinstance(config, dict):
        raise ValueError("checkpoint has no config dictionary")
    tensor_count = sum(isinstance(value, torch.Tensor) for value in state.values())
    if tensor_count == 0:
        raise ValueError("checkpoint state_dict contains no tensors")
    return {
        "bytes": int(path.stat().st_size),
        "tensor_count": int(tensor_count),
        "epoch": int(payload.get("epoch", payload.get("best_epoch", -1))),
    }


def _require_npz(path: Path, keys: set[str]) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as loaded:
        missing = sorted(keys - set(loaded.files))
        if missing:
            raise ValueError(f"missing arrays: {missing}")
        shapes = {name: list(loaded[name].shape) for name in sorted(keys)}
        for name in keys:
            value = loaded[name]
            if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                raise FloatingPointError(f"non-finite array: {name}")
    return {"bytes": int(path.stat().st_size), "shapes": shapes}


def _validate_asset(path: Path) -> dict[str, Any]:
    if path.name == "deployment_donor_weights.npz":
        report = _require_npz(
            path, {"donor_ids", "base_weights", "pas_weights", "pdp_weights"}
        )
        with np.load(path, allow_pickle=False) as loaded:
            shape = loaded["donor_ids"].shape
            for name in ("base_weights", "pas_weights", "pdp_weights"):
                if loaded[name].shape != shape:
                    raise ValueError(f"{name} shape differs from donor_ids")
        return report
    if path.name == "carrier_mix_weights.npz":
        return _require_npz(path, {"labels", "label_0", "label_1", "ridge"})
    if path.name == "group_calibration.npz":
        required = {
            f"site{site}_{domain}_{name}"
            for site in (0, 1)
            for domain in ("pas", "pdp")
            for name in ("mean", "vectors", "slope")
        }
        return _require_npz(path, required)

    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if path.name == "projection_policy.json":
        for name in ("reference_policy", "final_policy"):
            rows = payload.get(name, payload.get("final_policy"))
            if not isinstance(rows, list) or len(rows) != 2:
                raise ValueError(f"invalid {name}")
            if {int(row["site"]) for row in rows} != {0, 1}:
                raise ValueError(f"{name} does not cover both sites")
    else:
        winner = payload.get("winner")
        rows = winner.get("final_policy") if isinstance(winner, dict) else None
        if not isinstance(rows, list) or len(rows) != 8:
            raise ValueError("lifecycle final_policy must cover 2 sites x 4 regimes")
    return {"bytes": int(path.stat().st_size)}


def validate_production_inventory(
    paths: RuntimePaths,
    require_complete: bool = True,
    write_report: bool = True,
) -> dict[str, Any]:
    expected = expected_checkpoints()
    expected_set = set(expected)
    actual = {
        path.relative_to(paths.model_dir)
        for path in paths.model_dir.rglob("*.pt")
    } if paths.model_dir.is_dir() else set()
    missing = sorted(expected_set - actual)
    unexpected = sorted(actual - expected_set)
    checkpoint_reports: dict[str, Any] = {}
    errors: list[str] = []
    for relative in expected:
        path = paths.model_dir / relative
        if not path.is_file():
            continue
        try:
            checkpoint_reports[str(relative)] = _validate_checkpoint(path)
        except (KeyError, OSError, TypeError, ValueError, FloatingPointError) as error:
            errors.append(f"{relative}: {type(error).__name__}: {error}")

    asset_reports: dict[str, Any] = {}
    missing_assets = []
    for relative in EXPECTED_ASSETS:
        path = paths.model_dir / relative
        if not path.is_file():
            missing_assets.append(relative)
            continue
        try:
            asset_reports[str(relative)] = _validate_asset(path)
        except (KeyError, OSError, TypeError, ValueError, FloatingPointError) as error:
            errors.append(f"{relative}: {type(error).__name__}: {error}")

    complete = not missing and not unexpected and not missing_assets and not errors
    report = {
        "stage": "validate_production_inventory",
        "status": "complete" if complete else "incomplete",
        "model_dir": portable_path(paths.model_dir),
        "expected_checkpoint_count": len(expected),
        "valid_checkpoint_count": len(checkpoint_reports),
        "missing_checkpoints": [str(path) for path in missing],
        "unexpected_checkpoints": [str(path) for path in unexpected],
        "checkpoints": checkpoint_reports,
        "expected_asset_count": len(EXPECTED_ASSETS),
        "valid_asset_count": len(asset_reports),
        "missing_assets": [str(path) for path in missing_assets],
        "assets": asset_reports,
        "errors": errors,
    }
    if write_report:
        atomic_json(paths.report_dir / "production_inventory.json", report)
    if require_complete and not complete:
        raise RuntimeError(
            "production inventory is incomplete: "
            f"checkpoints={len(checkpoint_reports)}/{len(expected)}, "
            f"assets={len(asset_reports)}/{len(EXPECTED_ASSETS)}, "
            f"errors={len(errors)}, unexpected={len(unexpected)}"
        )
    return report
