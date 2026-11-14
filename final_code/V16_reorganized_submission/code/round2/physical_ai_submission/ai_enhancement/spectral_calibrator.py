from __future__ import annotations
import copy
from dataclasses import dataclass
import math
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from ..geometry_context.descriptors import GEOMETRY_CACHE
from ..scene_prior.spectral_dataset import (
    ARTIFACT_DIR,
    BASE_CONFIG,
    SCATTER_CONFIG,
    SEEDS,
    TRUTH_PAS,
    TRUTH_PDP,
    TRUTH_POWER,
    running_path,
)
from project_paths import PATHS
from ..data.dataset_compat import load_positions

FOLD_CACHE = PATHS.artifacts / "round2" / "foundation" / "round2_spatial_hole_folds.npz"
DOMAINS = ("pas", "pdp")
TINY = 1e-12


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 16
    steps_per_epoch: int = 400
    pas_batch_size: int = 1024
    pdp_batch_size: int = 2048
    validation_batch_size: int = 4096
    hidden: int = 64
    context_channels: int = 24
    blocks: int = 5
    residual_scale: float = 6.0
    learning_rate: float = 0.0015
    weight_decay: float = 2e-05
    hellinger_weight: float = 0.05
    residual_weight: float = 0.0001
    seed: int = 20260806


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tensor(path: Path, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    print(f"load {PATHS .relative (path )}", flush=True)
    source = np.load(path, mmap_mode="r")
    value = torch.as_tensor(source).to(device=device, dtype=dtype)
    return value


def normalize_probability(value: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    value = value.clamp_min(0.0)
    total = value.sum(dim=dims, keepdim=True)
    return torch.where(total > TINY, value / total.clamp_min(TINY), 0.0)


def source_scalar_matrix(config: str) -> np.ndarray:
    columns = []
    for kind in ("log_power", "component_count"):
        mean = np.asarray(
            np.load(running_path(config, "train", kind, "mean"), mmap_mode="r"),
            dtype=np.float32,
        )
        m2 = np.asarray(
            np.load(running_path(config, "train", kind, "m2"), mmap_mode="r"),
            dtype=np.float32,
        )
        count = np.asarray(
            np.load(running_path(config, "train", kind, "count"), mmap_mode="r"),
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


def ad_marginals(config: str, domain: str, site: np.ndarray) -> torch.Tensor:
    source = np.load(running_path(config, "train", "ad", "mean"), mmap_mode="r")
    if domain == "pas":
        value = np.asarray(source, dtype=np.float32).sum(axis=3, dtype=np.float32)
        total = value.sum(axis=(1, 2), keepdims=True, dtype=np.float64)
        value = np.divide(
            value, np.maximum(total, TINY), out=np.zeros_like(value), where=total > 0.0
        )
        shifted = np.empty_like(value)
        for site_value, shift in ((0, 2), (1, 1)):
            rows = site == site_value
            shifted[rows] = np.roll(value[rows], shift=shift, axis=2)
        value = shifted[:, None]
    else:
        value = np.asarray(source, dtype=np.float32).sum(axis=(1, 2), dtype=np.float32)
        total = value.sum(axis=1, keepdims=True, dtype=np.float64)
        value = np.divide(
            value, np.maximum(total, TINY), out=np.zeros_like(value), where=total > 0.0
        )
        value = value[:, None]
    return torch.from_numpy(value.astype(np.float32))


class DomainBank:

    def __init__(self, domain: str, device: torch.device) -> None:
        if domain not in DOMAINS:
            raise ValueError(domain)
        self.domain = domain
        self.device = device
        _, _, _, train_site, _ = load_positions()
        self.site = torch.as_tensor(train_site, device=device, dtype=torch.long)
        geometry_cache = np.load(GEOMETRY_CACHE, allow_pickle=False)
        self.geometry_names = tuple(geometry_cache["names"].astype(str).tolist())
        self.geometry = torch.as_tensor(
            geometry_cache["train"], device=device, dtype=torch.float32
        )
        self.power = torch.as_tensor(
            np.load(TRUTH_POWER), device=device, dtype=torch.float64
        )
        self.nonzero = self.power > 0.0
        self.scalar = {
            "base": torch.as_tensor(
                source_scalar_matrix(BASE_CONFIG), device=device, dtype=torch.float32
            ),
            "scatter": torch.as_tensor(
                source_scalar_matrix(SCATTER_CONFIG), device=device, dtype=torch.float32
            ),
        }
        self.ad = {
            "base": ad_marginals(BASE_CONFIG, domain, train_site).to(
                device=device, dtype=torch.bfloat16
            ),
            "scatter": ad_marginals(SCATTER_CONFIG, domain, train_site).to(
                device=device, dtype=torch.bfloat16
            ),
        }
        self.mean = {}
        self.m2 = {}
        self.count = {}
        for name, config in (("base", BASE_CONFIG), ("scatter", SCATTER_CONFIG)):
            mean = load_tensor(
                running_path(config, "train", domain, "mean"), device, torch.bfloat16
            )
            m2 = load_tensor(
                running_path(config, "train", domain, "m2"), device, torch.bfloat16
            )
            count = load_tensor(
                running_path(config, "train", domain, "count"), device, torch.uint8
            )
            if domain == "pas":
                mean = (
                    mean.permute(0, 4, 5, 1, 2, 3)
                    .reshape(4000, 768, 2, 16, 8)
                    .contiguous()
                )
                m2 = (
                    m2.permute(0, 4, 5, 1, 2, 3)
                    .reshape(4000, 768, 2, 16, 8)
                    .contiguous()
                )
                count = count.reshape(4000, 768).contiguous()
            else:
                mean = mean.reshape(4000, 1024, 192).contiguous()
                m2 = m2.reshape(4000, 1024, 192).contiguous()
                count = count.reshape(4000, 1024).contiguous()
            self.mean[name] = mean
            self.m2[name] = m2
            self.count[name] = count
        truth_path = TRUTH_PAS if domain == "pas" else TRUTH_PDP
        truth = load_tensor(truth_path, device, torch.bfloat16)
        if domain == "pas":
            if truth.ndim == 4:
                truth = truth.reshape(4000, 2, 16, 8, 4, 192)
            truth = (
                truth.permute(0, 4, 5, 1, 2, 3)
                .reshape(4000, 768, 2, 16, 8)
                .contiguous()
            )
            self.slice_count = 768
        else:
            truth = truth.reshape(4000, 1024, 192).contiguous()
            self.slice_count = 1024
        self.truth = truth
        self._point_context: dict[str, torch.Tensor] = {}
        print(
            f"loaded domain={domain } gpu_gib={torch .cuda .memory_allocated (device )/2 **30 :.2f}",
            flush=True,
        )

    def point_context(self, mode: str) -> torch.Tensor:
        if mode in self._point_context:
            return self._point_context[mode]
        if mode == "mean":
            physical = self.scalar["base"][:, [0, 3]]
        elif mode == "variance":
            physical = self.scalar["base"]
        elif mode == "dual":
            physical = torch.cat(
                [self.scalar["base"][:, [0, 3]], self.scalar["scatter"][:, [0, 3]]],
                dim=1,
            )
        elif mode == "full":
            physical = torch.cat([self.scalar["base"], self.scalar["scatter"]], dim=1)
        else:
            raise ValueError(mode)
        value = torch.cat([self.geometry, physical], dim=1)
        self._point_context[mode] = value
        return value

    def context_statistics(
        self, mode: str, fit_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value = self.point_context(mode)[fit_ids]
        mean = value.mean(dim=0)
        std = value.std(dim=0).clamp_min(0.0001)
        return (mean, std)

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
        mode: str,
        context_mean: torch.Tensor,
        context_std: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        means = {
            name: self.mean[name][point_ids, slice_ids].float()
            for name in ("base", "scatter")
        }
        m2 = {
            name: self.m2[name][point_ids, slice_ids].float()
            for name in ("base", "scatter")
        }
        counts = {
            name: self.count[name][point_ids, slice_ids].float()
            for name in ("base", "scatter")
        }
        if self.domain == "pas":
            for values in (means, m2):
                for name in values:
                    values[name] = torch.roll(values[name], shifts=2, dims=3)
        uncertainty = {}
        for name in ("base", "scatter"):
            variance = (
                m2[name]
                / (counts[name] - 1.0).clamp_min(1.0)[
                    (...,) + (None,) * (means[name].ndim - 1)
                ]
            )
            relative = torch.sqrt(variance.clamp_min(0.0)) / means[name].clamp_min(
                1e-05
            )
            uncertainty[name] = (
                torch.log1p(relative.clamp_max(10000.0)) / math.log(10000.0 + 1.0)
            ).clamp(0.0, 1.0)

        def feature_map(value: torch.Tensor) -> torch.Tensor:
            return value if value.ndim == 4 else value[:, None, :]

        features = [feature_map(torch.sqrt(means["base"].clamp_min(0.0)))]
        if mode == "variance":
            features.extend(
                [
                    feature_map(uncertainty["base"]),
                    self._broadcast_scalar(
                        counts["base"] / float(SEEDS), means["base"]
                    ),
                ]
            )
        elif mode in ("dual", "full"):
            features.extend(
                [
                    feature_map(torch.sqrt(means["scatter"].clamp_min(0.0))),
                    feature_map(torch.sqrt((means["base"] - means["scatter"]).abs())),
                ]
            )
            if mode == "full":
                features.extend(
                    [
                        feature_map(uncertainty["base"]),
                        feature_map(uncertainty["scatter"]),
                        self._broadcast_scalar(
                            counts["base"] / float(SEEDS), means["base"]
                        ),
                        self._broadcast_scalar(
                            counts["scatter"] / float(SEEDS), means["base"]
                        ),
                        self.ad["base"][point_ids].float(),
                        self.ad["scatter"][point_ids].float(),
                    ]
                )
        feature = torch.cat(features, dim=1)
        if mode in ("dual", "full"):
            baseline = 0.5 * (means["base"] + means["scatter"])
        else:
            baseline = means["base"]
        dims = tuple(range(1, baseline.ndim))
        baseline = normalize_probability(baseline, dims)
        target = self.truth[point_ids, slice_ids].float()
        point = (self.point_context(mode)[point_ids] - context_mean) / context_std
        context = torch.cat([point, self.slice_context(slice_ids)], dim=1)
        return (feature, context, baseline, target)

    @staticmethod
    def _broadcast_scalar(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        spatial_shape = (
            reference.shape[2:] if reference.ndim == 4 else reference.shape[1:]
        )
        shape = (len(value), 1, *spatial_shape)
        return value.reshape(len(value), 1, *[1] * len(spatial_shape)).expand(shape)


class ResidualBlock1d(nn.Module):

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=2, padding_mode="circular"),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 3, padding=1, padding_mode="circular"),
            nn.GroupNorm(8, channels),
        )
        self.activation = nn.SiLU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(value + self.net(value))


class ResidualBlock2d(nn.Module):

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode="circular"),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode="circular"),
            nn.GroupNorm(8, channels),
        )
        self.activation = nn.SiLU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(value + self.net(value))


class SpectralCalibrator(nn.Module):

    def __init__(
        self, domain: str, input_channels: int, context_size: int, config: TrainConfig
    ) -> None:
        super().__init__()
        self.domain = domain
        self.residual_scale = config.residual_scale
        self.context = nn.Sequential(
            nn.Linear(context_size, 96),
            nn.SiLU(),
            nn.LayerNorm(96),
            nn.Linear(96, config.context_channels),
            nn.SiLU(),
        )
        if domain == "pas":
            vertical = torch.arange(16, dtype=torch.float32) * (2.0 * math.pi / 16.0)
            horizontal = torch.arange(8, dtype=torch.float32) * (2.0 * math.pi / 8.0)
            vv, hh = torch.meshgrid(vertical, horizontal, indexing="ij")
            coordinate = torch.stack(
                [torch.sin(vv), torch.cos(vv), torch.sin(hh), torch.cos(hh)], dim=0
            )[None]
            self.register_buffer("coordinate", coordinate)
            total_input = input_channels + 4 + config.context_channels
            self.stem = nn.Conv2d(
                total_input, config.hidden, 3, padding=1, padding_mode="circular"
            )
            self.blocks = nn.Sequential(
                *[ResidualBlock2d(config.hidden) for _ in range(config.blocks)]
            )
            self.output = nn.Conv2d(config.hidden, 2, 1)
        else:
            delay = torch.arange(192, dtype=torch.float32) * (2.0 * math.pi / 192.0)
            coordinate = torch.stack(
                [
                    torch.sin(delay),
                    torch.cos(delay),
                    torch.sin(2.0 * delay),
                    torch.cos(2.0 * delay),
                    torch.sin(4.0 * delay),
                    torch.cos(4.0 * delay),
                ],
                dim=0,
            )[None]
            self.register_buffer("coordinate", coordinate)
            total_input = input_channels + 6 + config.context_channels
            self.stem = nn.Conv1d(
                total_input, config.hidden, 5, padding=2, padding_mode="circular"
            )
            self.blocks = nn.Sequential(
                *[ResidualBlock1d(config.hidden) for _ in range(config.blocks)]
            )
            self.output = nn.Conv1d(config.hidden, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self, feature: torch.Tensor, context: torch.Tensor, baseline: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.context(context)
        spatial_shape = feature.shape[2:]
        encoded = encoded.reshape(len(feature), -1, *[1] * len(spatial_shape)).expand(
            len(feature), -1, *spatial_shape
        )
        coordinate = self.coordinate.expand(len(feature), -1, *spatial_shape)
        hidden = torch.cat([feature, coordinate, encoded], dim=1)
        hidden = torch.nn.functional.silu(self.stem(hidden))
        delta = self.output(self.blocks(hidden))
        if self.domain == "pdp":
            delta = delta[:, 0]
        base = baseline.reshape(len(baseline), -1).clamp_min(0.0)
        base = base / base.sum(dim=1, keepdim=True).clamp_min(TINY)
        base = 0.999 * base + 0.001 / base.shape[1]
        logits = torch.log(base.clamp_min(1e-08)) + self.residual_scale * torch.tanh(
            delta.reshape(len(delta), -1)
        )
        prediction = torch.softmax(logits, dim=1).reshape_as(baseline)
        return (prediction, delta)


def input_channels(domain: str, mode: str) -> int:
    source_channels = 2 if domain == "pas" else 1
    if mode == "mean":
        return source_channels
    if mode == "variance":
        return 2 * source_channels + 1
    if mode == "dual":
        return 3 * source_channels
    if mode == "full":
        return 5 * source_channels + 4
    raise ValueError(mode)


def spectral_cosine(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    prediction = prediction.reshape(len(prediction), -1)
    truth = truth.reshape(len(truth), -1)
    numerator = (prediction * truth).sum(dim=1)
    denominator = torch.sqrt(prediction.square().sum(dim=1) * truth.square().sum(dim=1))
    return numerator / denominator.clamp_min(TINY)


def training_evaluate_direct_visibilitys(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    delta: torch.Tensor,
    config: TrainConfig,
) -> torch.Tensor:
    cosine = spectral_cosine(prediction, truth)
    valid = truth.reshape(len(truth), -1).sum(dim=1) > 0.0
    cosine_evaluate_direct_visibilitys = 1.0 - cosine[valid].mean()
    hellinger = (
        (
            torch.sqrt(prediction.clamp_min(0.0) + 1e-08)
            - torch.sqrt(truth.clamp_min(0.0) + 1e-08)
        )
        .square()
        .reshape(len(truth), -1)
        .mean(dim=1)
    )
    return (
        cosine_evaluate_direct_visibilitys
        + config.hellinger_weight * hellinger[valid].mean()
        + config.residual_weight * delta.square().mean()
    )


def fold_ids(fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache = np.load(FOLD_CACHE)
    outer_fit = cache[f"fold{fold }_fit"].astype(np.int64)
    outer_query = cache[f"fold{fold }_query"].astype(np.int64)
    inner_fold = (fold + 1) % 5
    inner_fit = cache[f"fold{inner_fold }_fit"].astype(np.int64)
    inner_query = cache[f"fold{inner_fold }_query"].astype(np.int64)
    nested_fit = np.intersect1d(outer_fit, inner_fit, assume_unique=False)
    nested_validation = np.intersect1d(outer_fit, inner_query, assume_unique=False)
    return (outer_fit, outer_query, nested_fit, nested_validation)


def model_path(domain: str, mode: str, fold: int) -> Path:
    return ARTIFACT_DIR / f"calibrator_{domain }_{mode }_fold{fold }.pt"


@torch.inference_mode()
def evaluate_ids(
    model: SpectralCalibrator,
    bank: DomainBank,
    point_ids: np.ndarray,
    mode: str,
    context_mean: torch.Tensor,
    context_std: torch.Tensor,
    batch_size: int,
    return_prediction: bool,
) -> tuple[float, np.ndarray | None]:
    model.eval()
    point_ids = np.asarray(point_ids, dtype=np.int64)
    all_points = torch.as_tensor(point_ids, device=bank.device, dtype=torch.long)
    scores = []
    output = None
    if return_prediction:
        if bank.domain == "pas":
            output = np.empty((len(point_ids), 768, 2, 16, 8), dtype=np.float16)
        else:
            output = np.empty((len(point_ids), 1024, 192), dtype=np.float16)
    for point_start in range(0, len(point_ids), 8):
        point_end = min(point_start + 8, len(point_ids))
        local_points = all_points[point_start:point_end]
        points = local_points[:, None].expand(-1, bank.slice_count).reshape(-1)
        slices = torch.arange(bank.slice_count, device=bank.device).repeat(
            len(local_points)
        )
        local_prediction = []
        for start in range(0, len(points), batch_size):
            end = min(start + batch_size, len(points))
            feature, context, baseline, truth = bank.sample(
                points[start:end], slices[start:end], mode, context_mean, context_std
            )
            with torch.autocast(
                device_type=bank.device.type,
                dtype=torch.bfloat16,
                enabled=bank.device.type == "cuda",
            ):
                prediction, _ = model(feature, context, baseline)
            scores.append(spectral_cosine(prediction.float(), truth).cpu().numpy())
            if return_prediction:
                local_prediction.append(prediction.float().cpu())
        if return_prediction:
            assembled = torch.cat(local_prediction, dim=0).reshape(
                len(local_points), bank.slice_count, *local_prediction[0].shape[1:]
            )
            output[point_start:point_end] = assembled.numpy().astype(np.float16)
    return (float(np.mean(np.concatenate(scores))), output)


def train_epochs(
    bank: DomainBank,
    mode: str,
    train_ids: np.ndarray,
    validation_ids: np.ndarray | None,
    epochs: int,
    config: TrainConfig,
    seed: int,
) -> tuple[SpectralCalibrator, int, list[dict[str, float]]]:
    set_seed(seed)
    train_ids = train_ids[np.asarray(bank.nonzero.cpu())[train_ids]]
    if validation_ids is not None:
        validation_ids = validation_ids[np.asarray(bank.nonzero.cpu())[validation_ids]]
    train_tensor = torch.as_tensor(train_ids, device=bank.device, dtype=torch.long)
    context_mean, context_std = bank.context_statistics(mode, train_tensor)
    sample_slice = torch.zeros(1, device=bank.device, dtype=torch.long)
    sample_point = train_tensor[:1]
    feature, context, _, _ = bank.sample(
        sample_point, sample_slice, mode, context_mean, context_std
    )
    model = SpectralCalibrator(
        bank.domain, feature.shape[1], context.shape[1], config
    ).to(bank.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    batch_size = (
        config.pas_batch_size if bank.domain == "pas" else config.pdp_batch_size
    )
    history = []
    best_score = -float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(
        {name: value.detach().cpu() for name, value in model.state_dict().items()}
    )
    if validation_ids is not None:
        best_score, _ = evaluate_ids(
            model,
            bank,
            validation_ids,
            mode,
            context_mean,
            context_std,
            config.validation_batch_size,
            False,
        )
        history.append(
            {"epoch": 0.0, "loss": float("nan"), "validation_cosine": best_score}
        )
        print(
            f"train domain={bank .domain } mode={mode } epoch=0/{epochs } val={best_score :.6f}",
            flush=True,
        )
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for _ in range(config.steps_per_epoch):
            points = train_tensor[
                torch.randint(len(train_tensor), (batch_size,), device=bank.device)
            ]
            slices = torch.randint(bank.slice_count, (batch_size,), device=bank.device)
            feature, context, baseline, truth = bank.sample(
                points, slices, mode, context_mean, context_std
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=bank.device.type,
                dtype=torch.bfloat16,
                enabled=bank.device.type == "cuda",
            ):
                prediction, delta = model(feature, context, baseline)
                loss = training_evaluate_direct_visibilitys(prediction.float(), truth, delta.float(), config)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        if validation_ids is None:
            validation_score = float("nan")
        else:
            validation_score, _ = evaluate_ids(
                model,
                bank,
                validation_ids,
                mode,
                context_mean,
                context_std,
                config.validation_batch_size,
                False,
            )
            if validation_score > best_score:
                best_score = validation_score
                best_epoch = epoch
                best_state = copy.deepcopy(
                    {
                        name: value.detach().cpu()
                        for name, value in model.state_dict().items()
                    }
                )
        history.append(
            {
                "epoch": float(epoch),
                "loss": float(np.mean(losses)),
                "validation_cosine": validation_score,
            }
        )
        print(
            f"train domain={bank .domain } mode={mode } epoch={epoch }/{epochs } loss={np .mean (losses ):.6f} val={validation_score :.6f}",
            flush=True,
        )
    if validation_ids is None:
        best_epoch = epochs
        best_state = copy.deepcopy(
            {name: value.detach().cpu() for name, value in model.state_dict().items()}
        )
    model.load_state_dict(best_state)
    model.context_mean = context_mean
    model.context_std = context_std
    return (model, best_epoch, history)


def reshape_oof_prediction(domain: str, prediction: np.ndarray) -> np.ndarray:
    if domain == "pas":
        count = len(prediction)
        return prediction.reshape(count, 4, 192, 2, 16, 8).transpose(0, 3, 4, 5, 1, 2)
    return prediction.reshape(len(prediction), 256, 4, 192)
