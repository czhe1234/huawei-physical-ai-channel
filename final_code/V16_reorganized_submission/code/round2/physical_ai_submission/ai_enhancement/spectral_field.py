from __future__ import annotations
import math
import numpy as np
import torch
from ..geometry_context.descriptors import GEOMETRY_CACHE
from ..scene_prior.spectral_dataset import BASE_CONFIG, SEEDS, running_path
from project_paths import PATHS
from ..data.dataset_compat import load_positions
from .spectral_calibrator import (
    SpectralCalibrator,
    TrainConfig,
    input_channels,
    model_path,
)

MODE = "mean"
TINY = 1e-12


def source_scalar_matrix(config: str, split: str) -> np.ndarray:
    columns = []
    for kind in ("log_power", "component_count"):
        mean = np.asarray(
            np.load(running_path(config, split, kind, "mean"), mmap_mode="r"),
            dtype=np.float32,
        )
        m2 = np.asarray(
            np.load(running_path(config, split, kind, "m2"), mmap_mode="r"),
            dtype=np.float32,
        )
        count = np.asarray(
            np.load(running_path(config, split, kind, "count"), mmap_mode="r"),
            dtype=np.float32,
        )
        variance = m2 / np.maximum(count - 1.0, 1.0)
        if kind == "component_count":
            mean = np.log1p(np.maximum(mean, 0.0))
            std = np.log1p(np.sqrt(np.maximum(variance, 0.0)))
        else:
            std = np.sqrt(np.maximum(variance, 0.0))
        columns.extend([mean, std, count / float(SEEDS)])
    return np.column_stack(columns).astype(np.float32)


class TestBank:

    def __init__(self, domain: str, device: torch.device) -> None:
        self.domain = domain
        self.device = device
        _, _, _, _, test_site = load_positions()
        self.site = torch.as_tensor(test_site, device=device, dtype=torch.long)
        geometry = np.load(GEOMETRY_CACHE, allow_pickle=False)
        physical = source_scalar_matrix(BASE_CONFIG, "test")[:, [0, 3]]
        point_context = np.concatenate([geometry["test"], physical], axis=1)
        self.point_context = torch.as_tensor(
            point_context, device=device, dtype=torch.float32
        )
        source = np.load(
            running_path(BASE_CONFIG, "test", domain, "mean"), mmap_mode="r"
        )
        mean = torch.as_tensor(source).to(device=device, dtype=torch.bfloat16)
        if domain == "pas":
            mean = (
                mean.permute(0, 4, 5, 1, 2, 3).reshape(500, 768, 2, 16, 8).contiguous()
            )
            self.slice_count = 768
        else:
            mean = mean.reshape(500, 1024, 192).contiguous()
            self.slice_count = 1024
        self.mean = mean
        print(
            f"loaded test domain={domain } gpu_gib={torch .cuda .memory_allocated (device )/2 **30 :.2f}",
            flush=True,
        )

    def slice_context(self, slice_ids: torch.Tensor) -> torch.Tensor:
        if self.domain == "pas":
            ue = torch.div(slice_ids, 192, rounding_mode="floor")
            frequency = slice_ids % 192
            angle = (frequency.float() + 0.5) * (2.0 * math.pi / 192.0)
            return torch.cat(
                [
                    torch.nn.functional.one_hot(ue, num_classes=4).float(),
                    ((frequency.float() + 0.5) / 96.0 - 1.0)[:, None],
                    *[
                        function(harmonic * angle)[:, None]
                        for harmonic in (1.0, 2.0, 4.0)
                        for function in (torch.sin, torch.cos)
                    ],
                ],
                dim=1,
            )
        antenna = torch.div(slice_ids, 4, rounding_mode="floor")
        ue = slice_ids % 4
        polarization = torch.div(antenna, 128, rounding_mode="floor")
        remainder = antenna % 128
        vertical = torch.div(remainder, 8, rounding_mode="floor")
        horizontal = remainder % 8
        vertical_angle = (vertical.float() + 0.5) * (2.0 * math.pi / 16.0)
        horizontal_angle = (horizontal.float() + 0.5) * (2.0 * math.pi / 8.0)
        return torch.cat(
            [
                torch.nn.functional.one_hot(polarization, num_classes=2).float(),
                torch.nn.functional.one_hot(ue, num_classes=4).float(),
                torch.sin(vertical_angle)[:, None],
                torch.cos(vertical_angle)[:, None],
                torch.sin(horizontal_angle)[:, None],
                torch.cos(horizontal_angle)[:, None],
                ((vertical.float() + 0.5) / 8.0 - 1.0)[:, None],
                ((horizontal.float() + 0.5) / 4.0 - 1.0)[:, None],
            ],
            dim=1,
        )

    def sample(
        self,
        point_ids: torch.Tensor,
        slice_ids: torch.Tensor,
        context_mean: torch.Tensor,
        context_std: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.mean[point_ids, slice_ids].float()
        if self.domain == "pas":
            mean = torch.roll(mean, shifts=2, dims=3)
            feature = torch.sqrt(mean.clamp_min(0.0))
        else:
            feature = torch.sqrt(mean.clamp_min(0.0))[:, None]
        dims = tuple(range(1, mean.ndim))
        total = mean.sum(dim=dims, keepdim=True)
        baseline = torch.where(total > TINY, mean / total.clamp_min(TINY), 0.0)
        point = (self.point_context[point_ids] - context_mean) / context_std
        context = torch.cat([point, self.slice_context(slice_ids)], dim=1)
        return (feature, context, baseline)


def instantiate_model(
    checkpoint: dict[str, object], bank: TestBank, device: torch.device
) -> tuple[SpectralCalibrator, torch.Tensor, torch.Tensor]:
    config = TrainConfig(**checkpoint["config"])
    context_mean = checkpoint["context_mean"].to(device=device, dtype=torch.float32)
    context_std = checkpoint["context_std"].to(device=device, dtype=torch.float32)
    sample_point = torch.zeros(1, device=device, dtype=torch.long)
    sample_slice = torch.zeros(1, device=device, dtype=torch.long)
    feature, context, _ = bank.sample(
        sample_point, sample_slice, context_mean, context_std
    )
    model = SpectralCalibrator(
        bank.domain, input_channels(bank.domain, MODE), context.shape[1], config
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return (model, context_mean, context_std)


@torch.inference_mode()
def predict_model(
    model: SpectralCalibrator,
    bank: TestBank,
    context_mean: torch.Tensor,
    context_std: torch.Tensor,
    batch_size: int,
) -> np.ndarray:
    if bank.domain == "pas":
        output = np.empty((500, 768, 2, 16, 8), dtype=np.float32)
    else:
        output = np.empty((500, 1024, 192), dtype=np.float32)
    all_points = torch.arange(500, device=bank.device)
    for point_start in range(0, 500, 8):
        point_end = min(point_start + 8, 500)
        local_points = all_points[point_start:point_end]
        points = local_points[:, None].expand(-1, bank.slice_count).reshape(-1)
        slices = torch.arange(bank.slice_count, device=bank.device).repeat(
            len(local_points)
        )
        local = []
        for start in range(0, len(points), batch_size):
            end = min(start + batch_size, len(points))
            feature, context, baseline = bank.sample(
                points[start:end], slices[start:end], context_mean, context_std
            )
            with torch.autocast(
                device_type=bank.device.type,
                dtype=torch.bfloat16,
                enabled=bank.device.type == "cuda",
            ):
                prediction, _ = model(feature, context, baseline)
            local.append(prediction.float().cpu())
        assembled = torch.cat(local, dim=0).reshape(
            len(local_points), bank.slice_count, *local[0].shape[1:]
        )
        output[point_start:point_end] = assembled.numpy()
        print(f"test target domain={bank .domain } {point_end }/500", flush=True)
    return output


def ensemble_target(domain: str, device: torch.device, batch_size: int) -> np.ndarray:
    bank = TestBank(domain, device)
    total = None
    checkpoints = []
    for fold in range(5):
        path = model_path(domain, MODE, fold)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model, context_mean, context_std = instantiate_model(checkpoint, bank, device)
        prediction = predict_model(model, bank, context_mean, context_std, batch_size)
        if total is None:
            total = prediction
        else:
            total += prediction
        checkpoints.append(
            {
                "fold": fold,
                "path": PATHS.relative(path),
                "epoch": int(checkpoint["epoch"]),
            }
        )
        del model, prediction
        if device.type == "cuda":
            torch.cuda.empty_cache()
    total /= 5.0
    axes = tuple(range(2, total.ndim)) if domain == "pas" else (2,)
    denominator = total.sum(axis=axes, keepdims=True, dtype=np.float64)
    total = np.divide(
        total,
        np.maximum(denominator, TINY),
        out=np.zeros_like(total),
        where=denominator > 0.0,
    )
    del bank
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return (total, checkpoints)
