"""Position-only, shared-frequency full-H adaptation of the pinned NeRF2 core.

This is an explicit adaptation, not the author's native uplink-CSI experiment.
No training-point channel donor, query CSI, target normalization, magnitude-only
output, checkpoint, or data loader is used here. All spatial coordinates are in
one right-handed Cartesian frame, in meters. Rays start at each actual RX element
and point out into the scene; tx_position supplies the actual transmitter.

The original model.py positional encoder and MLP are imported unmodified.
Frequency is appended to encoded sample position (3 -> 4 channels), preserving
the MLP skip topology. The author's CSI renderer's complex emission, cumulative
attenuation, cumulative residual phase and coherent ray sum are retained.
Explicit changes: actual RX positions, real shared frequency conditioning,
finite metric ray intervals, full-sphere directions, device-neutral tensors,
and default exp(-j*2*pi*f*tau), rather than the author's +j convention.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import ast
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from project_paths import PATHS

SOURCE_COMMIT = "130b55ea3856aa2f5a42f9329cfa2e845c78e243"
SOURCE_SHA256 = {
    "model.py": "c63fa4246d6255035d03ebc1dbd7b5a1d40391b51e98805ed3d9969bc3546ad1",
    "renderer.py": "4f5e5b1ab617de27918e8d2bc540182a110d37eb6ac5e743449c34b3a8535c92",
}
SPEED_OF_LIGHT_M_S = 299792458.0


@dataclass(frozen=True)
class NeRF2FullHConfig:
    depth: int = 8
    width: int = 256
    skips: tuple[int, ...] = (4,)
    position_levels: int = 10
    direction_levels: int = 10
    transmitter_levels: int = 10
    encode_transmitter: bool = True
    n_rays: int = 324
    n_samples: int = 32
    near_m: float = 0.0
    far_m: float = 5.0
    world_center_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    world_scale_m: float = 1.0
    reference_frequency_hz: float = 3438000000.0
    frequency_scale_hz: float = 50000000.0
    propagation_sign: int = -1
    ray_chunk: int = 16
    frequency_chunk: int = 8

    def __post_init__(self):
        object.__setattr__(self, "skips", tuple(self.skips))
        object.__setattr__(self, "world_center_m", tuple(self.world_center_m))
        integers = ("depth", "width", "position_levels", "direction_levels",
                    "transmitter_levels", "n_rays", "n_samples", "ray_chunk",
                    "frequency_chunk")
        for name in integers:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(name + " must be a positive integer")
        if self.width < 4:
            raise ValueError("width must be at least 4")
        if any(isinstance(i, bool) or not isinstance(i, int)
               or i < 0 or i >= self.depth - 1 for i in self.skips):
            raise ValueError("skip indices must precede the final attenuation layer")
        if len(set(self.skips)) != len(self.skips):
            raise ValueError("skip indices must be unique")
        if (not math.isfinite(self.near_m) or not math.isfinite(self.far_m)
                or self.near_m < 0 or self.far_m <= self.near_m):
            raise ValueError("Require finite 0 <= near_m < far_m")
        if len(self.world_center_m) != 3 or not all(
                math.isfinite(v) for v in self.world_center_m):
            raise ValueError("world_center_m must contain three finite values")
        for name in ("world_scale_m", "reference_frequency_hz", "frequency_scale_hz"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name + " must be finite and positive")
        if self.propagation_sign not in (-1, 1):
            raise ValueError("propagation_sign must be -1 or +1")


def _original_model_class():
    root = PATHS.code_root / "third_party/paper_baselines/nerf2"
    for name, expected in SOURCE_SHA256.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Pinned NeRF2 source identity differs: " + name)
    name = "_pinned_nerf2_model_" + SOURCE_COMMIT
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, root / "model.py")
        if spec is None or spec.loader is None:
            raise ImportError("Cannot load the pinned NeRF2 model.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            tree = ast.parse((root / "model.py").read_text())
            # The pinned model never uses rearrange; avoid adding an unused dependency.
            if any(isinstance(node, ast.Name) and node.id == "rearrange"
                   for node in ast.walk(tree)):
                raise RuntimeError("The supposedly unused einops symbol is now referenced")
            tree.body = [node for node in tree.body
                         if not (isinstance(node, ast.ImportFrom)
                                 and node.module == "einops")]
            exec(compile(tree, str(root / "model.py"), "exec"), module.__dict__)
        except Exception:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name].NeRF2


def _frequencies64(frequencies_hz, device):
    # Casting an already-rounded GHz float32 tensor back to float64 cannot repair it.
    if isinstance(frequencies_hz, torch.Tensor) and frequencies_hz.dtype != torch.float64:
        raise TypeError("frequencies_hz must be float64; do not round GHz tones to float32")
    if hasattr(frequencies_hz, "dtype") and not isinstance(frequencies_hz, torch.Tensor):
        if str(frequencies_hz.dtype) != "float64":
            raise TypeError("frequencies_hz arrays must have float64 dtype")
    frequency = torch.as_tensor(frequencies_hz, dtype=torch.float64, device=device)
    if frequency.ndim != 1 or frequency.numel() == 0:
        raise ValueError("frequencies_hz must be a nonempty [F] vector")
    if not bool(torch.isfinite(frequency).all()) or not bool((frequency > 0).all()):
        raise ValueError("frequencies_hz must be finite and positive")
    return frequency


def coherent_nerf2_integral(raw, segment_lengths_m, frequencies_hz,
                           propagation_sign=-1, near_m=0.0):
    """Author CSI complex core, with a finite grid and real-frequency propagation.

    raw: [..., F, S, 4], channels attenuation amplitude/residual phase and
         emitted amplitude/phase. segment_lengths_m: [S], in meters.
    Returns complex [..., F]; no abs(), power spectrum, or label correction.

    At sample j the wave distance is near_m + sum_{i<=j} segment_lengths_m[i].
    Residual attenuation phases remain per-segment radians, as in the author code.
    Geometry/frequency phase is evaluated and reduced in float64 before the
    final complex64 cast when the MLP is float32.
    """
    if raw.ndim < 3 or raw.shape[-1] != 4 or raw.dtype not in (torch.float32, torch.float64):
        raise ValueError("raw must be float32/float64 [..., F, S, 4]")
    if propagation_sign not in (-1, 1) or not math.isfinite(near_m) or near_m < 0:
        raise ValueError("Invalid fixed propagation convention or near bound")
    frequency = _frequencies64(frequencies_hz, raw.device)
    lengths = torch.as_tensor(segment_lengths_m, dtype=torch.float64, device=raw.device)
    if (lengths.ndim != 1 or lengths.numel() != raw.shape[-2]
            or raw.shape[-3] != frequency.numel()):
        raise ValueError("Raw, frequency and finite segment dimensions differ")
    if not bool(torch.isfinite(lengths).all()) or not bool((lengths > 0).all()):
        raise ValueError("Each finite ray segment must have positive metric length")
    attenuation_amp = F.leaky_relu(raw[..., 0]).abs()
    source_amp = F.leaky_relu(raw[..., 2]).abs()
    attenuation_phase = (torch.sigmoid(raw[..., 1]) * (2.0 * math.pi) - math.pi)
    source_phase = (torch.sigmoid(raw[..., 3]) * (2.0 * math.pi) - math.pi)
    transmittance = torch.exp(torch.cumsum(
        -attenuation_amp * lengths.to(raw.dtype), dim=-1))
    propagation = propagation_sign * (2.0 * math.pi / SPEED_OF_LIGHT_M_S) * (
        frequency[:, None] * lengths[None, :])
    phase = torch.cumsum(attenuation_phase.to(torch.float64) + propagation, dim=-1)
    phase = phase + source_phase.to(torch.float64)
    phase = phase + propagation_sign * (2.0 * math.pi / SPEED_OF_LIGHT_M_S) * (
        frequency * near_m)[..., None]
    phase = torch.remainder(phase + math.pi, 2.0 * math.pi) - math.pi
    amplitude = source_amp * transmittance
    unit_phase = torch.complex(torch.cos(phase), torch.sin(phase))
    value = amplitude.to(torch.float64) * unit_phase
    result = value.sum(dim=-1)
    return result.to(torch.complex64 if raw.dtype == torch.float32 else torch.complex128)


class NeRF2FullH(nn.Module):
    """Return full complex [B,P,M,F] from only TX/RX positions and frequencies.

    Constructor accepts NeRF2FullHConfig, a matching mapping, or keyword fields.
    Tiny correctness fixtures should set depth=2,width=16,skips=(),n_rays=3,
    n_samples=4. Defaults preserve the author's depth/width/encoding levels and
    324 directions; these defaults are not a measured accuracy/cost recommendation.

    world_center_m/world_scale_m are fixed input coordinates/encoding scales.
    They never scale physical ray lengths or propagation wavelengths. Their
    values must come from the permitted training/configuration contract.
    """
    def __init__(self, config=None, **kwargs):
        super().__init__()
        if config is None:
            config = NeRF2FullHConfig(**kwargs)
        elif isinstance(config, Mapping):
            if kwargs:
                raise TypeError("Use config or keyword fields, not both")
            config = NeRF2FullHConfig(**dict(config))
        elif not isinstance(config, NeRF2FullHConfig) or kwargs:
            raise TypeError("Expected a NeRF2FullHConfig and no extra keyword fields")
        self.config = config
        original = _original_model_class()
        self.field = original(
            D=config.depth, W=config.width, skips=list(config.skips),
            input_dims={"pts": 4, "view": 3, "tx": 3},
            multires={"pts": config.position_levels,
                      "view": config.direction_levels,
                      "tx": config.transmitter_levels},
            is_embeded={"pts": True, "view": True,
                        "tx": config.encode_transmitter},
            attn_output_dims=2, sig_output_dims=2)
        index = torch.arange(config.n_rays, dtype=torch.float64)
        z = 1.0 - 2.0 * (index + .5) / config.n_rays
        azimuth = index * (math.pi * (3.0 - math.sqrt(5.0)))
        radius = torch.sqrt(torch.clamp(1.0 - z.square(), min=0))
        directions = torch.stack(
            [radius * torch.cos(azimuth), radius * torch.sin(azimuth), z], dim=-1)
        self.register_buffer("ray_directions", directions, persistent=True)

    def metadata(self):
        return {
            "method": "NeRF2_position_only_shared_frequency_full_H_adapter",
            "source_commit": SOURCE_COMMIT,
            "source_files_sha256": dict(SOURCE_SHA256),
            "source_repository": "https://github.com/XPengZhao/NeRF2",
            "source_license": "MIT",
            "config": asdict(self.config),
            "input_signature": ["tx_position[B,3]", "rx_position[P,M,3]",
                                "frequencies_hz[F] float64"],
            "output": "complex[B,P,M,F]; full amplitude and phase, no magnitude reduction",
            "conventions": {
                "coordinates": "common right-handed Cartesian frame, meters",
                "rays": "unit vectors from each actual RX element out into the scene",
                "frequency": "true float64 Hz; centered/scaled only for MLP encoding",
                "geometric_phase": "exp(sign*j*2*pi*f*distance/c); default sign=-1",
                "author_geometric_phase_sign": 1,
                "default_sign_reason": "Canonical H(f)=sum a exp(-j*2*pi*f*tau), ascending tones plus IFFT PDP",
                "attenuation_phase": "learned per-segment radians, author sigmoid mapping [-pi,pi]",
                "emission": "author abs(leaky_relu(amplitude)) and sigmoid phase mapping",
                "finite_grid": "right endpoints of S equal metric segments on [near,far]",
                "direction_reduction": "author coherent unweighted sum, no final abs",
            },
            "retained": ["unaltered author NeRF2 MLP class and positional encoder",
                         "attenuation/signal heads and skip topology",
                         "complex cumulative attenuation/residual phase and coherent sample/ray sum"],
            "changed": ["sample-coordinate input dimension 3->4 for shared continuous frequency",
                        "query uplink CSI replaced by real transmitter position",
                        "per-physical-RX-element ray origins",
                        "global full-sphere Fibonacci directions instead of fixed upper hemisphere",
                        "finite metric segments replace infinite 1e10 tail",
                        "device-neutral tensors replace unconditional cuda calls",
                        "unused einops import omitted at AST load; MLP/encoder definitions unchanged",
                        "explicit canonical negative propagation phase, configurable before fitting"],
            "limits": ["Not a native author CSI-branch reproduction or performance result",
                       "No explicit antenna pattern or polarization beyond supplied element positions",
                       "Chunking limits individual field calls; a full backward graph can retain all chunks",
                       "Ray count/range and encodings need a separately authorized fixed experiment contract"],
        }

    def forward(self, tx_position, rx_position, frequencies_hz):
        parameter = next(self.field.parameters())
        if parameter.dtype not in (torch.float32, torch.float64):
            raise TypeError("Phase-sensitive NeRF2 adapter supports float32/float64 MLPs")
        device, dtype = parameter.device, parameter.dtype
        tx = torch.as_tensor(tx_position, dtype=torch.float64, device=device)
        rx = torch.as_tensor(rx_position, dtype=torch.float64, device=device)
        frequency = _frequencies64(frequencies_hz, device)
        if tx.ndim != 2 or tx.shape[-1] != 3 or tx.shape[0] == 0:
            raise ValueError("tx_position must have nonempty shape [B,3]")
        if rx.ndim != 3 or rx.shape[-1] != 3 or min(rx.shape[:2]) < 1:
            raise ValueError("rx_position must have nonempty shape [P,M,3]")
        if not bool(torch.isfinite(tx).all()) or not bool(torch.isfinite(rx).all()):
            raise ValueError("Coordinates must be finite meters")
        batch, panels, elements = tx.shape[0], rx.shape[0], rx.shape[1]
        ports = panels * elements
        cfg = self.config
        center = torch.tensor(cfg.world_center_m, dtype=torch.float64, device=device)
        directions = self.ray_directions.to(device=device, dtype=torch.float64)
        # Renormalize if nn.Module.float() cast the stored geometry buffer.
        directions = directions / torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
        step = (cfg.far_m - cfg.near_m) / cfg.n_samples
        segments = torch.full((cfg.n_samples,), step, dtype=torch.float64, device=device)
        radii = cfg.near_m + step * torch.arange(
            1, cfg.n_samples + 1, dtype=torch.float64, device=device)
        all_ray_count = batch * ports * cfg.n_rays
        complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        frequency_blocks = []
        rx_flat = rx.reshape(ports, 3)
        for frequency_first in range(0, len(frequency), cfg.frequency_chunk):
            f = frequency[frequency_first:frequency_first + cfg.frequency_chunk]
            result = torch.zeros((batch * ports, len(f)),
                                 dtype=complex_dtype, device=device)
            f_encoded = ((f - cfg.reference_frequency_hz)
                         / cfg.frequency_scale_hz).to(dtype)
            for first in range(0, all_ray_count, cfg.ray_chunk):
                ray_id = torch.arange(first, min(first + cfg.ray_chunk, all_ray_count),
                                      device=device)
                link = torch.div(ray_id, cfg.n_rays, rounding_mode="floor")
                direction = directions[ray_id % cfg.n_rays]
                origin = rx_flat[link % ports]
                transmitter = tx[torch.div(link, ports, rounding_mode="floor")]
                points = origin[:, None, :] + direction[:, None, :] * radii[None, :, None]
                n = len(ray_id)
                points_encoded = ((points - center) / cfg.world_scale_m).to(dtype)
                points_encoded = points_encoded[:, None].expand(n, len(f), cfg.n_samples, 3)
                frequency_encoded = f_encoded[None, :, None, None].expand(
                    n, len(f), cfg.n_samples, 1)
                points_frequency = torch.cat([points_encoded, frequency_encoded], dim=-1)
                view = direction.to(dtype)[:, None, None, :].expand(
                    n, len(f), cfg.n_samples, 3)
                tx_encoded = ((transmitter - center) / cfg.world_scale_m).to(dtype)
                tx_encoded = tx_encoded[:, None, None, :].expand(
                    n, len(f), cfg.n_samples, 3)
                raw = self.field(points_frequency, view, tx_encoded)
                rays = coherent_nerf2_integral(
                    raw, segments, f, cfg.propagation_sign, cfg.near_m)
                result = result.index_add(0, link, rays)
            frequency_blocks.append(result)
        return torch.cat(frequency_blocks, dim=-1).reshape(
            batch, panels, elements, len(frequency))


def synthetic_self_check():
    """CPU synthetic forward/algebra/autograd only; no optimizer or real data."""
    if torch.get_num_threads() > 2:
        torch.set_num_threads(2)
    torch.manual_seed(91031)
    config = dict(depth=2, width=16, skips=(), position_levels=2,
                  direction_levels=2, transmitter_levels=2, n_rays=3,
                  n_samples=4, near_m=.1, far_m=2.0, ray_chunk=2,
                  frequency_chunk=2)
    model = NeRF2FullH(**config).cpu()
    tx = torch.tensor([[.2, .3, .4], [.5, -.1, .7]], dtype=torch.float64,
                      requires_grad=True)
    rx = torch.tensor([[[0., 0., 0.], [.03, .01, 0.]]], dtype=torch.float64)
    frequency = torch.tensor([3438000000., 3438000000.125, 3438500000.],
                             dtype=torch.float64, requires_grad=True)
    value = model(tx, rx, frequency)
    if value.shape != (2, 1, 2, 3) or value.dtype != torch.complex64:
        raise AssertionError("Synthetic shape/dtype mismatch")
    if not bool(torch.isfinite(value).all()):
        raise AssertionError("Non-finite synthetic channel")
    parameters = tuple(model.parameters())
    gradients = torch.autograd.grad(value.abs().square().sum(),
                                    (tx, frequency) + parameters, allow_unused=True)
    if any(g is not None and not bool(torch.isfinite(g).all()) for g in gradients):
        raise AssertionError("Non-finite synthetic derivative")
    if gradients[0] is None or gradients[1] is None:
        raise AssertionError("Position/frequency derivatives are disconnected")
    other = NeRF2FullH(**{**config, "ray_chunk": 100, "frequency_chunk": 100}).cpu()
    other.load_state_dict(model.state_dict())
    torch.testing.assert_close(value, other(tx, rx, frequency), rtol=3e-6, atol=3e-7)
    # One emitting cell with no attenuation/residual phase gives the canonical delay law.
    f = torch.tensor([3438000000., 3438000000.125, 3439000000.], dtype=torch.float64)
    raw = torch.zeros((1, len(f), 3, 4), dtype=torch.float64)
    raw[:, :, 1, 2] = 1.
    lengths = torch.full((3,), 2.3, dtype=torch.float64)
    result = coherent_nerf2_integral(raw, lengths, f)
    phase = -2.0 * math.pi * f * 4.6 / SPEED_OF_LIGHT_M_S
    expected = torch.complex(torch.cos(phase), torch.sin(phase))[None]
    torch.testing.assert_close(result, expected, rtol=1e-11, atol=1e-11)
    if bool(torch.equal(result[..., 0], result[..., 1])):
        raise AssertionError("Fractional-Hz frequency precision was lost")
    try:
        model(tx, rx, frequency.to(torch.float32))
    except TypeError:
        pass
    else:
        raise AssertionError("Rounded float32 GHz tones were accepted")
    return dict(status="passed", device="cpu", real_channel_arrays_read=False,
                optimizer_created=False, trained=False, scored=False,
                output_shape=list(value.shape), output_dtype=str(value.dtype),
                finite_forward_backward=True, chunk_equivalence=True,
                canonical_negative_delay_phase=True, fractional_hz_precision=True,
                float32_frequency_rejected=True, source_sha256=dict(SOURCE_SHA256))


if __name__ == "__main__":
    import json
    print(json.dumps(synthetic_self_check(), indent=2))
