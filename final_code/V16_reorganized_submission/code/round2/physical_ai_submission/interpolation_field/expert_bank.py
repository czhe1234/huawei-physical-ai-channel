from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import numpy as np
import torch
from scipy.spatial import cKDTree
from .phase_identification import estimate_carrier
from .spatial_experts import (
    BaselineConfig,
    batched_predict_many_torch,
    predict_many,
)
from project_paths import PATHS

ARTIFACT_DIR = PATHS.artifacts / "round2" / "complex_carrier_bank"
TEST_BANK = ARTIFACT_DIR / "test_complex_carrier_bank.npy"


def carrier_configs() -> list[BaselineConfig]:
    configs = [BaselineConfig(name="nearest", method="nearest", k=1)]
    for k in (4, 8, 16, 24):
        configs.append(BaselineConfig(name=f"raw_idw_k{k }", method="raw_idw", k=k))
        configs.append(
            BaselineConfig(name=f"radial_idw_k{k }", method="radial_idw", k=k)
        )
    for k, ridge in ((8, 0.1), (16, 1.0), (24, 0.1), (32, 0.1)):
        configs.append(
            BaselineConfig(
                name=f"radial_affine_k{k }_r{ridge :g}",
                method="radial_affine",
                k=k,
                ridge=ridge,
            )
        )
    for axis in (0.0, 90.0):
        for vertical_scale, tag in ((1.5 * np.pi, "v1p5pi"), (8.0 * np.pi, "v8pi")):
            configs.append(
                BaselineConfig(
                    name=f"affine_k24_steer_axis{int (axis )}_{tag }",
                    method="radial_affine",
                    k=24,
                    ridge=0.3,
                    steering_alpha=1.0,
                    horizontal_axis_deg=axis,
                    vertical_scale=vertical_scale,
                )
            )
    if len({config.name for config in configs}) != len(configs):
        raise RuntimeError("duplicate carrier names")
    return configs


def build_test_bank(
    train_channel: np.ndarray,
    train_positions: np.ndarray,
    test_positions: np.ndarray,
    train_site: np.ndarray,
    test_site: np.ndarray,
    bs_positions: np.ndarray,
    point_power: np.ndarray,
    configs: list[BaselineConfig],
    carrier_pairs: int,
    workers: int,
    device_channel: torch.Tensor | None = None,
    gpu_batch_size: int = 8,
) -> tuple[np.ndarray, dict[str, object]]:
    expected = (len(configs), len(test_positions), *train_channel.shape[1:])
    if TEST_BANK.is_file():
        cached = np.load(TEST_BANK, mmap_mode="r")
        if cached.shape == expected and cached.dtype == np.complex64:
            return (cached, {"reused": True})
    output = np.lib.format.open_memmap(
        TEST_BANK, mode="w+", dtype=np.complex64, shape=expected
    )
    all_indices = np.arange(len(train_positions), dtype=np.int64)
    carrier = estimate_carrier(
        train_channel,
        train_positions,
        train_site,
        bs_positions,
        all_indices,
        point_power,
        max_pairs_per_site=carrier_pairs,
        seed=20260805,
    )
    max_k = max((config.k for config in configs))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for site_value in np.unique(test_site):
            site_value = int(site_value)
            fit = all_indices[train_site == site_value]
            rows = np.flatnonzero(test_site == site_value)
            tree = cKDTree(train_positions[fit, :2])
            distances, local = tree.query(test_positions[rows, :2], k=max_k, workers=-1)
            neighbors = fit[local]
            effective_configs = [
                replace(
                    template,
                    beta=carrier[site_value].beta,
                    gamma=carrier[site_value].gamma,
                )
                for template in configs
            ]

            def predict_row(local_row: int) -> tuple[int, np.ndarray]:
                test_row = int(rows[local_row])
                donor_ids = neighbors[local_row]
                predictions, _ = predict_many(
                    np.asarray(train_channel[donor_ids], dtype=np.complex64),
                    train_positions[donor_ids],
                    test_positions[test_row],
                    distances[local_row],
                    bs_positions[site_value],
                    effective_configs,
                )
                return (test_row, predictions)

            completed = 0
            if device_channel is not None:
                configs_by_site = {site_value: effective_configs}
                for start in range(0, len(rows), gpu_batch_size):
                    end = min(start + gpu_batch_size, len(rows))
                    test_rows = rows[start:end]
                    prediction, _, _ = batched_predict_many_torch(
                        device_channel,
                        neighbors[start:end],
                        train_positions[neighbors[start:end]],
                        test_positions[test_rows],
                        distances[start:end],
                        bs_positions,
                        test_site[test_rows],
                        configs_by_site,
                        point_power,
                    )
                    output[:, test_rows] = (
                        prediction.permute(1, 0, 2, 3, 4).cpu().numpy()
                    )
                    completed = end
                    print(
                        f"carrier bank test site={site_value } {completed }/{len (rows )}",
                        flush=True,
                    )
                    del prediction
            else:
                block_size = max(1, workers) * 2
                for start in range(0, len(rows), block_size):
                    futures = [
                        executor.submit(predict_row, row)
                        for row in range(start, min(start + block_size, len(rows)))
                    ]
                    for future in futures:
                        test_row, predictions = future.result()
                        output[:, test_row] = predictions
                        completed += 1
                        if completed % 10 == 0 or completed == len(rows):
                            print(
                                f"carrier bank test site={site_value } {completed }/{len (rows )}",
                                flush=True,
                            )
            output.flush()
    output.flush()
    return (
        np.load(TEST_BANK, mmap_mode="r"),
        {
            "reused": False,
            "carrier": {str(site): value.to_dict() for site, value in carrier.items()},
        },
    )
