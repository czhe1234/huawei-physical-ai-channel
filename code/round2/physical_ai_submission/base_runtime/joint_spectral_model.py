"""Frozen runtime primitives for joint spectral model."""

from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import torch
from .geometry_descriptors import GEOMETRY_CACHE
from .physical_spectral_dataset import (
    BASE_CONFIG,
    SCATTER_CONFIG,
    SEEDS,
    TRUTH_PAS,
    TRUTH_PDP,
    TRUTH_POWER,
    running_path,
)
from project_paths import PATHS
from .dataset_compat import load_positions
from .spectral_calibrator import (
    SpectralCalibrator,
    TrainConfig,
    evaluate_ids,
    fold_ids,
    reshape_oof_prediction,
    train_epochs,
)

ARTIFACT_DIR = PATHS.artifacts / "round2" / "joint_joint_physical_endpoints"
DOMAINS = ("pas", "pdp")
MODE = "joint"
TINY = 1e-12


def source_scalar_matrix(config: str, split: str) -> np.ndarray:
    columns = []
    for kind in ("log_power", "path_count"):
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
        if kind == "path_count":
            mean = np.log1p(np.maximum(mean, 0.0))
            std = np.log1p(np.sqrt(np.maximum(variance, 0.0)))
        else:
            std = np.sqrt(np.maximum(variance, 0.0))
        columns.extend([mean, std, count / float(SEEDS)])
    return np.column_stack(columns).astype(np.float32)


def reliability_path(domain: str, split: str) -> Path:
    return ARTIFACT_DIR / f"{split}_{domain}_spectral_reliability.npy"


def count_view(domain: str, count: np.ndarray) -> np.ndarray:
    if domain == "pas":
        return count[:, None, None, None, :, :]
    return count[..., None]


def spectral_reliability(domain: str, split: str, force: bool) -> np.ndarray:
    path = reliability_path(domain, split)
    if path.is_file() and (not force):
        return np.load(path, mmap_mode="r")
    paths = {}
    for name, config in (("base", BASE_CONFIG), ("scatter", SCATTER_CONFIG)):
        paths[name] = {
            statistic: np.load(
                running_path(config, split, domain, statistic), mmap_mode="r"
            )
            for statistic in ("mean", "m2", "count")
        }
    rows = len(paths["base"]["mean"])
    output = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(rows, 11)
    )
    for start in range(0, rows, 4):
        end = min(start + 4, rows)
        values = {}
        for name in ("base", "scatter"):
            mean = np.asarray(paths[name]["mean"][start:end], dtype=np.float32)
            m2 = np.asarray(paths[name]["m2"][start:end], dtype=np.float32)
            count = np.asarray(paths[name]["count"][start:end], dtype=np.float32)
            expanded_count = count_view(domain, count)
            variance = m2 / np.maximum(expanded_count - 1.0, 1.0)
            flat = mean.reshape(end - start, -1).astype(np.float64)
            mass = np.maximum(flat.sum(axis=1), TINY)
            probability = flat / mass[:, None]
            entropy = -np.sum(
                probability * np.log(np.maximum(probability, 1e-30)), axis=1
            ) / math.log(flat.shape[1])
            noise_mass = (
                np.sqrt(np.maximum(variance, 0.0))
                .reshape(end - start, -1)
                .sum(axis=1, dtype=np.float64)
                / mass
            )
            values[name] = {
                "flat": flat,
                "mass": mass,
                "entropy": entropy,
                "peak": np.max(probability, axis=1),
                "noise_mass": noise_mass,
                "coverage": count.reshape(end - start, -1).mean(axis=1) / float(SEEDS),
            }
        numerator = np.sum(values["base"]["flat"] * values["scatter"]["flat"], axis=1)
        denominator = np.sqrt(
            np.sum(values["base"]["flat"] ** 2, axis=1)
            * np.sum(values["scatter"]["flat"] ** 2, axis=1)
        )
        agreement = numerator / np.maximum(denominator, TINY)
        log_mass_ratio = np.log(
            np.maximum(values["base"]["mass"], TINY)
            / np.maximum(values["scatter"]["mass"], TINY)
        )
        output[start:end] = np.column_stack(
            [
                agreement,
                log_mass_ratio,
                values["base"]["noise_mass"],
                values["scatter"]["noise_mass"],
                values["base"]["coverage"],
                values["scatter"]["coverage"],
                values["base"]["entropy"],
                values["scatter"]["entropy"],
                values["base"]["peak"],
                values["scatter"]["peak"],
                np.abs(log_mass_ratio),
            ]
        ).astype(np.float32)
        if end == rows or end % 100 == 0:
            output.flush()
            print(f"reliability domain={domain} split={split} {end}/{rows}", flush=True)
    output.flush()
    return np.load(path, mmap_mode="r")


def normalize_probability(value: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    value = value.clamp_min(0.0)
    total = value.sum(dim=dims, keepdim=True)
    return torch.where(total > TINY, value / total.clamp_min(TINY), 0.0)


def shifted_ad(value: torch.Tensor, site: torch.Tensor, domain: str) -> torch.Tensor:
    if domain == "pas":
        result = value.sum(dim=3, dtype=torch.float32)
        for site_value, shift in ((0, 2), (1, 1)):
            mask = site == site_value
            result[mask] = torch.roll(result[mask], shifts=shift, dims=2)
        return normalize_probability(result[:, None], (2, 3))
    result = value.sum(dim=(1, 2), dtype=torch.float32)
    return normalize_probability(result[:, None], (2,))


class JointDomainBank:

    def __init__(
        self,
        domain: str,
        split: str,
        device: torch.device,
        force_reliability: bool = False,
    ) -> None:
        if domain not in DOMAINS or split not in ("train", "test"):
            raise ValueError((domain, split))
        self.domain = domain
        self.split = split
        self.device = device
        train_positions, test_positions, _, train_site, test_site = load_positions()
        del train_positions, test_positions
        site = train_site if split == "train" else test_site
        self.site = torch.as_tensor(site, device=device, dtype=torch.long)
        geometry = np.load(GEOMETRY_CACHE, allow_pickle=False)[split]
        physical = np.concatenate(
            [
                source_scalar_matrix(BASE_CONFIG, split),
                source_scalar_matrix(SCATTER_CONFIG, split),
                np.asarray(
                    spectral_reliability(domain, split, force_reliability),
                    dtype=np.float32,
                ),
            ],
            axis=1,
        )
        point_context = np.concatenate([geometry, physical], axis=1).astype(np.float32)
        self._point_context = torch.as_tensor(
            point_context, device=device, dtype=torch.float32
        )
        self.rows = len(point_context)
        if split == "train":
            power = np.load(TRUTH_POWER, mmap_mode="r")
            self.nonzero = torch.as_tensor(power > 0.0, device=device)
        else:
            self.nonzero = torch.ones(self.rows, dtype=torch.bool, device=device)
        self.mean: dict[str, torch.Tensor] = {}
        for name, config in (("base", BASE_CONFIG), ("scatter", SCATTER_CONFIG)):
            source = np.load(running_path(config, split, domain, "mean"), mmap_mode="r")
            value = torch.as_tensor(source).to(device=device, dtype=torch.bfloat16)
            if domain == "pas":
                value = value.permute(0, 4, 5, 1, 2, 3).reshape(
                    self.rows, 4, 192, 2, 16, 8
                )
                value = torch.roll(value, shifts=2, dims=5).contiguous()
            else:
                value = value.reshape(self.rows, 256, 4, 192).contiguous()
            self.mean[name] = value
        self.summary: dict[str, dict[str, torch.Tensor]] = {}
        for name, value in self.mean.items():
            if domain == "pas":
                normalized = normalize_probability(value.float(), (3, 4, 5)).to(
                    torch.bfloat16
                )
                self.mean[name] = normalized
                self.summary[name] = {
                    "ue": normalize_probability(value.float().sum(dim=2), (2, 3, 4)).to(
                        torch.bfloat16
                    ),
                    "frequency": normalize_probability(
                        value.float().sum(dim=1), (2, 3, 4)
                    ).to(torch.bfloat16),
                    "global": normalize_probability(
                        value.float().sum(dim=(1, 2)), (1, 2, 3)
                    ).to(torch.bfloat16),
                }
            else:
                normalized = normalize_probability(value.float(), (3,)).to(
                    torch.bfloat16
                )
                self.mean[name] = normalized
                shaped = value.reshape(self.rows, 2, 128, 4, 192)
                self.summary[name] = {
                    "antenna": normalize_probability(value.float().sum(dim=2), (2,)).to(
                        torch.bfloat16
                    ),
                    "ue": normalize_probability(value.float().sum(dim=1), (2,)).to(
                        torch.bfloat16
                    ),
                    "pol_ue": normalize_probability(shaped.float().sum(dim=2), (3,)).to(
                        torch.bfloat16
                    ),
                    "polarization": normalize_probability(
                        shaped.float().sum(dim=(2, 3)), (2,)
                    ).to(torch.bfloat16),
                    "global": normalize_probability(
                        value.float().sum(dim=(1, 2)), (1,)
                    ).to(torch.bfloat16),
                }
        self.ad = {}
        for name, config in (("base", BASE_CONFIG), ("scatter", SCATTER_CONFIG)):
            source = np.load(running_path(config, split, "ad", "mean"), mmap_mode="r")
            value = torch.as_tensor(source).to(device=device, dtype=torch.bfloat16)
            self.ad[name] = shifted_ad(value, self.site, domain).to(torch.bfloat16)
        if split == "train":
            truth_path = TRUTH_PAS if domain == "pas" else TRUTH_PDP
            truth = torch.as_tensor(np.load(truth_path, mmap_mode="r")).to(
                device=device, dtype=torch.bfloat16
            )
            if domain == "pas":
                truth = truth.permute(0, 4, 5, 1, 2, 3).reshape(
                    self.rows, 768, 2, 16, 8
                )
                self.slice_count = 768
            else:
                truth = truth.reshape(self.rows, 1024, 192)
                self.slice_count = 1024
            self.truth = truth.contiguous()
        else:
            self.truth = None
            self.slice_count = 768 if domain == "pas" else 1024
        print(
            f"loaded joint bank domain={domain} split={split} gpu_gib={torch.cuda.memory_allocated(device) / 2 ** 30:.2f}",
            flush=True,
        )

    def point_context(self, mode: str) -> torch.Tensor:
        if mode != MODE:
            raise ValueError(mode)
        return self._point_context

    def context_statistics(
        self, mode: str, fit_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value = self.point_context(mode)[fit_ids]
        return (value.mean(dim=0), value.std(dim=0).clamp_min(0.0001))

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
                        for harmonic in (1.0, 2.0, 4.0, 8.0)
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

    @staticmethod
    def sqrt_feature(value: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(value.float().clamp_min(0.0))

    def sample_input(
        self,
        point_ids: torch.Tensor,
        slice_ids: torch.Tensor,
        mode: str,
        context_mean: torch.Tensor,
        context_std: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if mode != MODE:
            raise ValueError(mode)
        base = self.mean["base"]
        scatter = self.mean["scatter"]
        if self.domain == "pas":
            ue = torch.div(slice_ids, 192, rounding_mode="floor")
            frequency = slice_ids % 192
            local_base = base[point_ids, ue, frequency].float()
            local_scatter = scatter[point_ids, ue, frequency].float()
            maps = [
                local_base,
                local_scatter,
                (local_base - local_scatter).abs(),
                self.summary["base"]["ue"][point_ids, ue].float(),
                self.summary["scatter"]["ue"][point_ids, ue].float(),
                self.summary["base"]["frequency"][point_ids, frequency].float(),
                self.summary["scatter"]["frequency"][point_ids, frequency].float(),
                self.summary["base"]["global"][point_ids].float(),
                self.summary["scatter"]["global"][point_ids].float(),
            ]
            feature = torch.cat([self.sqrt_feature(value) for value in maps], dim=1)
            feature = torch.cat(
                [
                    feature,
                    self.sqrt_feature(self.ad["base"][point_ids]),
                    self.sqrt_feature(self.ad["scatter"][point_ids]),
                ],
                dim=1,
            )
            baseline = normalize_probability(local_base + local_scatter, (1, 2, 3))
        else:
            antenna = torch.div(slice_ids, 4, rounding_mode="floor")
            ue = slice_ids % 4
            polarization = torch.div(antenna, 128, rounding_mode="floor")
            local_base = base[point_ids, antenna, ue].float()
            local_scatter = scatter[point_ids, antenna, ue].float()
            vectors = [
                local_base,
                local_scatter,
                (local_base - local_scatter).abs(),
                self.summary["base"]["antenna"][point_ids, antenna].float(),
                self.summary["scatter"]["antenna"][point_ids, antenna].float(),
                self.summary["base"]["ue"][point_ids, ue].float(),
                self.summary["scatter"]["ue"][point_ids, ue].float(),
                self.summary["base"]["pol_ue"][point_ids, polarization, ue].float(),
                self.summary["scatter"]["pol_ue"][point_ids, polarization, ue].float(),
                self.summary["base"]["polarization"][point_ids, polarization].float(),
                self.summary["scatter"]["polarization"][
                    point_ids, polarization
                ].float(),
                self.summary["base"]["global"][point_ids].float(),
                self.summary["scatter"]["global"][point_ids].float(),
            ]
            feature = torch.cat(
                [self.sqrt_feature(value)[:, None] for value in vectors]
                + [
                    self.sqrt_feature(self.ad["base"][point_ids]),
                    self.sqrt_feature(self.ad["scatter"][point_ids]),
                ],
                dim=1,
            )
            baseline = normalize_probability(local_base + local_scatter, (1,))
        point = (self._point_context[point_ids] - context_mean) / context_std
        context = torch.cat([point, self.slice_context(slice_ids)], dim=1)
        return (feature, context, baseline)

    def sample(
        self,
        point_ids: torch.Tensor,
        slice_ids: torch.Tensor,
        mode: str,
        context_mean: torch.Tensor,
        context_std: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.truth is None:
            raise RuntimeError("test bank has no official truth")
        feature, context, baseline = self.sample_input(
            point_ids, slice_ids, mode, context_mean, context_std
        )
        truth = self.truth[point_ids, slice_ids].float()
        return (feature, context, baseline, truth)


def checkpoint_path(domain: str, fold: int) -> Path:
    return ARTIFACT_DIR / f"joint_calibrator_{domain}_fold{fold}.pt"


def output_path(domain: str, split: str) -> Path:
    return ARTIFACT_DIR / f"{split}_{domain}_joint_f16.npy"


@torch.inference_mode()
def predict_model(
    model: SpectralCalibrator,
    bank: JointDomainBank,
    context_mean: torch.Tensor,
    context_std: torch.Tensor,
    batch_size: int,
) -> np.ndarray:
    shape = (
        (bank.rows, bank.slice_count, 2, 16, 8)
        if bank.domain == "pas"
        else (bank.rows, bank.slice_count, 192)
    )
    output = np.empty(shape, dtype=np.float32)
    all_points = torch.arange(bank.rows, device=bank.device)
    for point_start in range(0, bank.rows, 8):
        point_end = min(point_start + 8, bank.rows)
        local_points = all_points[point_start:point_end]
        points = local_points[:, None].expand(-1, bank.slice_count).reshape(-1)
        slices = torch.arange(bank.slice_count, device=bank.device).repeat(
            len(local_points)
        )
        local = []
        for start in range(0, len(points), batch_size):
            end = min(start + batch_size, len(points))
            feature, context, baseline = bank.sample_input(
                points[start:end], slices[start:end], MODE, context_mean, context_std
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
        print(
            f"joint test target domain={bank.domain} {point_end}/{bank.rows}",
            flush=True,
        )
    return output


def instantiate_model(
    checkpoint: dict[str, object], bank: JointDomainBank, device: torch.device
) -> tuple[SpectralCalibrator, torch.Tensor, torch.Tensor]:
    config = TrainConfig(**checkpoint["config"])
    context_mean = checkpoint["context_mean"].to(device=device, dtype=torch.float32)
    context_std = checkpoint["context_std"].to(device=device, dtype=torch.float32)
    point = torch.zeros(1, device=device, dtype=torch.long)
    slice_id = torch.zeros(1, device=device, dtype=torch.long)
    feature, context, _ = bank.sample_input(
        point, slice_id, MODE, context_mean, context_std
    )
    model = SpectralCalibrator(
        bank.domain, feature.shape[1], context.shape[1], config
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return (model, context_mean, context_std)


def generate_test_targets(
    domains: tuple[str, ...], device: torch.device, batch_size: int, force: bool
) -> dict[str, object]:
    result = {}
    for domain in domains:
        path = output_path(domain, "test")
        if path.is_file() and (not force):
            result[domain] = {"output": PATHS.relative(path), "reused": True}
            continue
        bank = JointDomainBank(domain, "test", device, force)
        total = None
        checkpoints = []
        for fold in range(5):
            checkpoint_file = checkpoint_path(domain, fold)
            checkpoint = torch.load(
                checkpoint_file, map_location=device, weights_only=False
            )
            model, context_mean, context_std = instantiate_model(
                checkpoint, bank, device
            )
            prediction = predict_model(
                model, bank, context_mean, context_std, batch_size
            )
            total = prediction if total is None else total + prediction
            checkpoints.append(
                {
                    "fold": fold,
                    "path": PATHS.relative(checkpoint_file),
                    "epoch": int(checkpoint["epoch"]),
                }
            )
            del model, prediction
            if device.type == "cuda":
                torch.cuda.empty_cache()
        total /= 5.0
        axes = tuple(range(2, total.ndim)) if domain == "pas" else (2,)
        total /= np.maximum(total.sum(axis=axes, keepdims=True, dtype=np.float64), TINY)
        shaped = reshape_oof_prediction(domain, total)
        output = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float16, shape=shaped.shape
        )
        output[...] = shaped.astype(np.float16)
        output.flush()
        result[domain] = {
            "output": PATHS.relative(path),
            "reused": False,
            "checkpoints": checkpoints,
        }
        del output, total, shaped, bank
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return result
