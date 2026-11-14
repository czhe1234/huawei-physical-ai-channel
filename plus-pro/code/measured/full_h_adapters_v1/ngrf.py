"""Frequency-conditioned nGRF adapter with a fixed physical receive-array contract.

Official AttributeNetwork/ContributionDecoderNetwork and the Torch reference
Gaussian weighting are loaded from the pinned, unmodified source snapshot.
Only import resolution is isolated; the official class/function bodies are reused.
This is a broadband task adaptation, not an unchanged nGRF paper reproduction.
No channel data, donors, optimizer, training loop, or measured-data reader is used.

All positions passed to this module use the same physical coordinate frame.
Channel amplitude normalization/restoration belongs to the external FIT-only wrapper.
The complex decoder learns its phase; no truth-based or analytic phase repair is added.
"""
from __future__ import annotations

import ast
from functools import lru_cache
import hashlib
import math
import sys
from types import ModuleType, SimpleNamespace
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from project_paths import PATHS

SOURCE_COMMIT = "3765d6f1e46a223914908e38430b1ab9d2a5f4fb"
SOURCE_ROOT = PATHS.code_root / "measured/full_h_adapters_v1/ngrf_source_3765d6f"
_SOURCE_SHA256 = {
    "utils/pos_encoder.py": "4b722861a53aff9409bb30f5878babd85499a34177f78cd86bf7261b74f9e484",
    "models/networks.py": "000374ea49b8b5b522acada5dd723554d63079dfcb19157f023b0b6b2f8b1730",
    "utils/transform_utils.py": "71b888e5580bb19a8c200d110d75a3049d3752df0d9763edbead9ef6ea6494c1",
    "render/_torch_impl/__init__.py": "d0ab806ed8ba143f5f4b7cc161e9934858dcff29715a51a91db4dc58787ccbed",
}


def _official_module(label, relative_path, omit_imports=(), namespace=None):
    """Load inspected Torch-only source without importing its CUDA/runner package."""
    source_path = SOURCE_ROOT / relative_path
    payload = source_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != _SOURCE_SHA256[relative_path]:
        raise RuntimeError(f"Pinned nGRF source identity differs: {relative_path}")
    name = f"_hw_full_h_ngrf_3765d6f_{label}"
    if name in sys.modules:
        return sys.modules[name]
    tree = ast.parse(payload, filename=str(source_path))
    tree.body = [node for node in tree.body
                 if not (isinstance(node, ast.ImportFrom)
                         and node.module in omit_imports)]
    module = ModuleType(name)
    module.__file__ = str(source_path)
    module.__dict__.update(namespace or {})
    sys.modules[name] = module
    try:
        exec(compile(tree, str(source_path), "exec"), module.__dict__)
    except Exception:
        del sys.modules[name]
        raise
    return module


@lru_cache(maxsize=1)
def _official_components():
    positional = _official_module("positional", "utils/pos_encoder.py")
    networks = _official_module(
        "networks", "models/networks.py", ("utils.pos_encoder",),
        {"PositionalEncoder": positional.PositionalEncoder})
    transforms = _official_module("transforms", "utils/transform_utils.py")
    # The renderer uses nGRF only as an annotation and otherwise duck-types it.
    # Bypass that import, which would import optimizer/config/CUDA runner code.
    renderer = _official_module(
        "torch_reference", "render/_torch_impl/__init__.py",
        ("models.ngrf_model",), {"nGRF": nn.Module})
    return networks, transforms, renderer


def encode_physical_frequencies(frequencies_hz, center_hz=3438e6,
                                scale_hz=50e6, levels=4):
    """Encode the actual Hz axis in FP64; return FP32 network input features.

    No frequency is treated as an antenna index. The shared decoder receives
    normalized Hz offsets plus Fourier features, with no per-tone model copies.
    """
    if not isinstance(frequencies_hz, torch.Tensor):
        raise TypeError("frequencies_hz must be a torch.Tensor")
    if frequencies_hz.ndim != 1 or not frequencies_hz.numel():
        raise ValueError("frequencies_hz must be a nonempty [F] vector")
    if frequencies_hz.dtype != torch.float64:
        raise TypeError("Actual physical frequencies must arrive as FP64 Hz")
    if not bool(torch.isfinite(frequencies_hz).all()) or bool((frequencies_hz <= 0).any()):
        raise ValueError("Frequencies must be finite positive Hz")
    if not math.isfinite(float(center_hz)) or not math.isfinite(float(scale_hz)) or scale_hz <= 0:
        raise ValueError("Frequency reference and positive scale must be finite")
    if not isinstance(levels, int) or levels < 0:
        raise ValueError("Frequency encoding levels must be a nonnegative integer")
    offset = (frequencies_hz - float(center_hz)) / float(scale_hz)
    parts = [offset[:, None]]
    for level in range(levels):
        phase = offset * (2.0 * math.pi * (2.0 ** level))
        parts.extend((phase.sin()[:, None], phase.cos()[:, None]))
    return torch.cat(parts, dim=-1).to(torch.float32)


class NGRFFullH(nn.Module):
    """One fixed-array nGRF with moving TX and a shared wideband complex decoder.

    ``forward([B,3], [P,M,3], [F]) -> complex64[B,P,M,F]``.
    One official decoder per panel has M physical-port heads, ordered exactly as
    rx_position at construction. Gaussian weights are evaluated at that panel's
    centre, as in the original array-head design. Changing the array, its panel
    order, or its port order is rejected; arbitrary-array generalization is not
    claimed. P/M are architecture settings fixed at construction.

    Defaults are an untrained integration reference (32 Gaussians), not the
    paper's 30,000-Gaussian capacity or a selected research configuration.
    """

    def __init__(self, rx_position: torch.Tensor, num_gaussians: int = 32,
                 latent_dim: int = 32, hidden_dim: int = 64,
                 initial_scale: float = 1.0,
                 gaussian_means: Optional[torch.Tensor] = None,
                 frequency_center_hz: float = 3438e6,
                 frequency_scale_hz: float = 50e6,
                 frequency_encoding_levels: int = 4,
                 frequency_chunk_size: int = 32,
                 pos_encoding_freqs: int = 10,
                 attribute_num_layers: int = 3,
                 decoder_num_layers: int = 4,
                 decoder_dropout_p: float = 0.0):
        super().__init__()
        supplied_rx = torch.as_tensor(rx_position)
        if supplied_rx.is_complex():
            raise TypeError("Receiver coordinates must be real")
        rx = supplied_rx.detach().clone().to(dtype=torch.float64)
        if rx.ndim != 3 or rx.shape[-1] != 3 or min(rx.shape[:2]) < 1:
            raise ValueError("Bind rx_position with shape [P,M,3]")
        if not bool(torch.isfinite(rx).all()):
            raise ValueError("Bound receiver positions must be finite")
        for name, value in [("num_gaussians", num_gaussians), ("latent_dim", latent_dim),
                            ("hidden_dim", hidden_dim), ("frequency_chunk_size", frequency_chunk_size)]:
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(initial_scale) or initial_scale <= 0:
            raise ValueError("initial_scale must be a finite positive physical length")
        if frequency_encoding_levels < 0 or not isinstance(frequency_encoding_levels, int):
            raise ValueError("frequency_encoding_levels must be a nonnegative integer")
        if (not math.isfinite(frequency_center_hz) or not math.isfinite(frequency_scale_hz)
                or frequency_center_hz <= 0 or frequency_scale_hz <= 0):
            raise ValueError("Finite positive frequency reference and scale required")
        self.num_panels, self.num_ports = rx.shape[:2]
        self.num_gaussians = num_gaussians
        self.latent_dim = latent_dim
        self.frequency_center_hz = float(frequency_center_hz)
        self.frequency_scale_hz = float(frequency_scale_hz)
        self.frequency_encoding_levels = frequency_encoding_levels
        self.frequency_chunk_size = frequency_chunk_size
        self.configuration = dict(
            num_gaussians=num_gaussians, latent_dim=latent_dim,
            hidden_dim=hidden_dim, initial_scale=initial_scale,
            frequency_center_hz=float(frequency_center_hz),
            frequency_scale_hz=float(frequency_scale_hz),
            frequency_encoding_levels=frequency_encoding_levels,
            frequency_chunk_size=frequency_chunk_size,
            pos_encoding_freqs=pos_encoding_freqs,
            attribute_num_layers=attribute_num_layers,
            decoder_num_layers=decoder_num_layers,
            decoder_dropout_p=decoder_dropout_p,
            gaussian_initialization=("provided geometry-only means" if gaussian_means is not None
                                     else "receiver-centre geometry-only jitter"))
        self.register_buffer("fixed_rx_position", rx)
        self.register_buffer("panel_centres", rx.mean(dim=1).to(torch.float32))
        if gaussian_means is None:
            # Geometry-only initialization, independent of every channel label.
            centres = self.panel_centres[torch.arange(num_gaussians, device=rx.device) % self.num_panels]
            xyz = centres + (2.0 * torch.rand(num_gaussians, 3, device=rx.device) - 1.0) * initial_scale
        else:
            xyz = torch.as_tensor(gaussian_means, device=rx.device, dtype=torch.float32).detach().clone()
            if xyz.shape != (num_gaussians, 3) or not bool(torch.isfinite(xyz).all()):
                raise ValueError("gaussian_means must be finite [num_gaussians,3]")
        self.gaussian_means = nn.Parameter(xyz.to(torch.float32))
        self.gaussian_log_scales = nn.Parameter(torch.full(
            (num_gaussians, 3), math.log(initial_scale), dtype=torch.float32, device=rx.device))
        rotations = torch.zeros(num_gaussians, 4, dtype=torch.float32, device=rx.device)
        rotations[:, 0] = 1.0
        self.gaussian_quaternions = nn.Parameter(rotations)
        networks, _, _ = _official_components()
        self.attribute_network = networks.AttributeNetwork(
            latent_dim=latent_dim, mlp_hidden_dim=hidden_dim,
            mlp_num_layers=attribute_num_layers, pos_encoding_freqs=pos_encoding_freqs,
            dropout_p=0.0).to(device=rx.device, dtype=torch.float32)
        frequency_dim = 1 + 2 * frequency_encoding_levels
        self.contribution_decoders = nn.ModuleList([
            networks.ContributionDecoderNetwork(
                latent_dim=latent_dim + frequency_dim, output_dim=2 * self.num_ports,
                hidden_dim=hidden_dim, num_layers=decoder_num_layers,
                dropout_p=decoder_dropout_p).to(device=rx.device, dtype=torch.float32)
            for _ in range(self.num_panels)])

    def metadata(self):
        """Describe the architecture and source identity without accessing data."""
        configuration = dict(self.configuration)
        configuration["frequency_chunk_size"] = self.frequency_chunk_size
        return dict(
            method="nGRF full-H frequency-conditioned fixed-array adaptation",
            source=dict(repository="https://github.com/muhd-umer/n-grf",
                        commit=SOURCE_COMMIT,
                        snapshot="code/measured/full_h_adapters_v1/ngrf_source_3765d6f",
                        source_manifest="code/measured/full_h_adapters_v1/ngrf_source_3765d6f/source_manifest.json",
                        directly_executed_sha256=dict(_SOURCE_SHA256)),
            configuration=configuration,
            parameters=sum(p.numel() for p in self.parameters()),
            output=dict(shape=["B", self.num_panels, self.num_ports, "F"],
                        dtype="complex64", amplitude_units="external FIT-only wrapper restores physical scale"),
            preserved=["official AttributeNetwork and ContributionDecoderNetwork class bodies",
                       "positive anisotropic scale and quaternion covariance",
                       "official Torch Gaussian weight including maximum Mahalanobis exponent 50",
                       "real Gaussian weight times complex contribution, summed without opacity compositing"],
            adapted=["one mobile TX per query instead of one shared fixed TX",
                     "actual FP64-Hz Fourier features concatenated to decoder latent, then FP32 network",
                     "one shared-across-frequency complex decoder per fixed physical panel",
                     "strict construction-time RX coordinate and panel/port-order binding"],
            fixed_array_contract=dict(
                panels=self.num_panels, ports_per_panel=self.num_ports,
                head_order="exact construction rx_position[p,m] order; no geometry-independent port remapping",
                gaussian_query="each bound physical panel centre",
                geometry_changes="reject changes to receiver coordinates, panel order, or port order",
                arbitrary_array_generalization=False),
            input_information=["TX position", "bound RX physical positions", "actual physical frequency Hz"],
            aggregation="sum_g sigmoid(activation_g(TX))*exp(-0.5*min(d^T*inverse_cov_g*d,50))*complex_decoder_g(TX,f,port)",
            execution=dict(reference="official pure-Torch renderer; no compiled CUDA renderer import",
                           training_or_optimizer_in_module=False,
                           donor_or_query_CSI_input=False,
                           phase_repair=False),
            scope="untrained interface implementation; default capacity and initialization are integration choices, not paper-equivalent training or measured performance")

    def frequency_features(self, frequencies_hz):
        return encode_physical_frequencies(
            frequencies_hz, self.frequency_center_hz, self.frequency_scale_hz,
            self.frequency_encoding_levels)

    def get_covariance(self, return_inverse=False, eps=1e-10):
        _, transforms, _ = _official_components()
        # Match the original activation and covariance order, including its floor.
        scaling = self.gaussian_log_scales.exp().clamp(min=1e-8)
        quaternion = F.normalize(self.gaussian_quaternions, p=2, dim=-1)
        rotation = transforms.build_rotation(quaternion)
        covariance = rotation @ torch.diag_embed(scaling.square()) @ rotation.transpose(1, 2)
        if return_inverse:
            return covariance, transforms.build_covariance_inverse(rotation, scaling, eps)
        return covariance

    def _attributes(self, tx_position):
        b = len(tx_position)
        xyz = self.gaussian_means[None].expand(b, -1, -1).reshape(-1, 3)
        tx = tx_position.to(torch.float32)[:, None].expand(-1, self.num_gaussians, -1).reshape(-1, 3)
        latent, logits = self.attribute_network(xyz, tx)
        return (latent.reshape(b, self.num_gaussians, self.latent_dim),
                logits.sigmoid().reshape(b, self.num_gaussians))

    def _spatial_weights(self, activations):
        _, _, renderer = _official_components()
        b = len(activations)
        delta = (self.panel_centres[:, None] - self.gaussian_means[None])[None].expand(b, -1, -1, -1)
        _, inverse = self.get_covariance(return_inverse=True)
        inverse = inverse[None, None].expand(b, self.num_panels, -1, -1, -1).reshape(-1, 3, 3)
        activation = activations[:, None, :, None].expand(-1, self.num_panels, -1, -1).reshape(-1, 1)
        # Exact official Torch reference: sigmoid(logit) * exp(-0.5*min(d^T Sigma^-1 d,50)).
        weight = renderer.compute_spatial_weight(delta.reshape(-1, 3), inverse, activation)
        return weight.reshape(b, self.num_panels, self.num_gaussians)

    def forward(self, tx_position: torch.Tensor, rx_position: torch.Tensor,
                frequencies_hz: torch.Tensor) -> torch.Tensor:
        if not isinstance(tx_position, torch.Tensor) or not isinstance(rx_position, torch.Tensor):
            raise TypeError("TX/RX positions must be torch tensors")
        if tx_position.ndim != 2 or tx_position.shape[-1] != 3 or not tx_position.is_floating_point():
            raise ValueError("tx_position must be floating [B,3]")
        if rx_position.shape != self.fixed_rx_position.shape or not rx_position.is_floating_point():
            raise ValueError("RX physical panel/port shape differs from the bound array")
        if not isinstance(frequencies_hz, torch.Tensor):
            raise TypeError("frequencies_hz must be a torch.Tensor")
        device = self.gaussian_means.device
        if tx_position.device != device or rx_position.device != device or frequencies_hz.device != device:
            raise ValueError("Positions, frequencies and model must share a device")
        if not bool(torch.isfinite(tx_position).all()) or not bool(torch.isfinite(rx_position).all()):
            raise ValueError("Positions must be finite")
        if not torch.equal(rx_position.to(torch.float64), self.fixed_rx_position):
            raise ValueError("RX coordinates or physical panel/port order changed")
        features = self.frequency_features(frequencies_hz)
        latent, activated = self._attributes(tx_position)
        weights = self._spatial_weights(activated).to(torch.complex64)
        b, g, _ = latent.shape
        pieces = []
        for start in range(0, len(features), self.frequency_chunk_size):
            freq = features[start:start + self.frequency_chunk_size]
            count = len(freq)
            latent_grid = latent[:, :, None, :].expand(-1, -1, count, -1)
            frequency_grid = freq[None, None].expand(b, g, -1, -1)
            decoder_input = torch.cat((latent_grid, frequency_grid), dim=-1)
            panels = []
            for panel, decoder in enumerate(self.contribution_decoders):
                value = decoder(decoder_input).reshape(b, g, count, 2, self.num_ports)
                contribution = torch.complex(value[..., 0, :], value[..., 1, :])
                panels.append(torch.einsum("bg,bgfm->bmf", weights[:, panel], contribution))
            pieces.append(torch.stack(panels, dim=1))
        return torch.cat(pieces, dim=-1).to(torch.complex64)


def official_reference_panel(model: NGRFFullH, tx_position: torch.Tensor,
                             panel: int, frequency_hz: float):
    """One synthetic query/panel/tone through the unmodified official renderer.

    This helper establishes reference arithmetic equivalence, not CUDA parity,
    model quality, or a measured-data result. It does not read any channel data.
    """
    if tx_position.shape != (3,) or not 0 <= panel < model.num_panels:
        raise ValueError("Reference check expects one TX [3] and a valid panel")
    _, _, renderer = _official_components()
    frequency = torch.tensor([frequency_hz], dtype=torch.float64,
                             device=model.gaussian_means.device)
    encoded = model.frequency_features(frequency)

    def attribute(tx):
        latent, activation = model._attributes(tx[None])
        return latent[0], activation[0, :, None]

    def decoder(latent):
        extra = encoded.expand(len(latent), -1)
        return model.contribution_decoders[panel](torch.cat((latent, extra), dim=-1))

    proxy = SimpleNamespace(get_xyz=model.gaussian_means,
                            get_attributes_and_activation=attribute,
                            get_covariance=model.get_covariance,
                            contribution_decoder=decoder)
    return renderer.render_channel(model.panel_centres[panel:panel + 1], proxy,
                                   tx_position.to(torch.float32), nt=1,
                                   nr=model.num_ports)[0, 0]
