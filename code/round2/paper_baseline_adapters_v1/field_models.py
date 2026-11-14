"""CPU-checkable NeRF2/nGRF Round2 position-only full-channel adapters.

No train/query channel, donor, scorer, optimizer, RT, or data loader is imported.
NeRF2 preserves the BS -> UE ray direction. nGRF reuses its fixed-array head
architecture as a regression representation: actual moving UE element positions
condition fixed BS port heads. This is an explicit role remapping, not a proof
of physical reciprocity. Co-located polarization ports use separate parameters,
never copies of the same complex output. Frequency is always a shared FP64-Hz
input, not an antenna index or a repeated middle-tone prediction.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Mapping

import torch
from torch import nn

from project_paths import PATHS

_ADAPTER_SHA256 = {
    "nerf2": "37704fd408178a4286e0b58a92b21a8f6a02ca635dc712d6194ceca7f488f192",
    "ngrf": "756fdd8fa3ab1d73d7f4f8c024d1ba0aa8833bd8e3ea529b88ae2eea86cfe876",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _real64(value, name):
    if isinstance(value, torch.Tensor) and value.is_complex():
        raise TypeError(name + " must be real coordinates")
    tensor = torch.as_tensor(value, dtype=torch.float64)
    _require(bool(torch.isfinite(tensor).all()), name + " must be finite")
    return tensor.detach().clone()


@dataclass(frozen=True)
class CompetitionGeometry:
    """Explicit physical-port geometry; no real-data defaults are invented.

    BS shape [site,2pol,16horizontal,8vertical,3].
    UE LOCAL offsets [site,2pol,1horizontal,2vertical,3], meters. The local
    axes are x=normal, y=horizontal, z=vertical. Its world orientation follows
    look_at_site_bs by Rz(phi)Ry(theta-pi/2), evaluated for every query.

    Root may supply another object with the same public attributes and
    ue_world_positions(position, site_id)->[B,4,3] method. All geometry must
    derive from the explicit experiment contract; this class does not read it.
    """
    bs_positions_m: torch.Tensor
    bs_port_positions_m: torch.Tensor
    ue_port_offsets_m: torch.Tensor
    bs_polarization_labels: tuple[str, str]
    ue_polarization_labels: tuple[str, str]
    provenance: Mapping
    ue_orientation_policy: str

    def __post_init__(self):
        for name in ("bs_positions_m", "bs_port_positions_m", "ue_port_offsets_m"):
            object.__setattr__(self, name, _real64(getattr(self, name), name))
        sites = len(self.bs_port_positions_m)
        _require(sites > 0 and self.bs_positions_m.shape == (sites, 3),
                 "Explicit BS centres [S,3] are required")
        _require(self.bs_port_positions_m.shape == (sites, 2, 16, 8, 3),
                 "BS physical-port order must be [S,2,16,8,3]")
        _require(self.ue_port_offsets_m.shape == (sites, 2, 1, 2, 3),
                 "Explicit local UE offsets must be [S,2,1,2,3]")
        for name in ("bs_polarization_labels", "ue_polarization_labels"):
            labels = tuple(getattr(self, name))
            _require(len(labels) == 2 and len(set(labels)) == 2
                     and all(isinstance(x, str) and x for x in labels),
                     name + " requires two distinct declared labels")
            object.__setattr__(self, name, labels)
        _require(isinstance(self.provenance, Mapping) and bool(self.provenance),
                 "Physical geometry needs explicit provenance")
        object.__setattr__(self, "provenance", dict(self.provenance))
        _require(self.ue_orientation_policy == "look_at_site_bs",
                 "Unsupported or missing UE orientation policy")

    @property
    def geometry_fingerprint(self):
        payload = {
            "bs_positions_m": self.bs_positions_m.detach().cpu().tolist(),
            "bs_port_positions_m": self.bs_port_positions_m.detach().cpu().tolist(),
            "ue_local_port_offsets_m": self.ue_port_offsets_m.detach().cpu().tolist(),
            "bs_polarization_labels": self.bs_polarization_labels,
            "ue_polarization_labels": self.ue_polarization_labels,
            "ue_orientation_policy": self.ue_orientation_policy,
            "bs_flat_order": "p,h,v", "ue_flat_order": "p,h,v",
            "provenance": dict(self.provenance),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def ue_world_positions(self, ue_position, site_id):
        position = torch.as_tensor(ue_position, dtype=torch.float64)
        site = torch.as_tensor(site_id, dtype=torch.int64, device=position.device)
        centre = self.bs_positions_m.to(position.device)[site]
        direction = centre - position
        horizontal = torch.linalg.vector_norm(direction[:, :2], dim=-1)
        _require(bool((horizontal > 0).all()),
                 "Look-at yaw is undefined for a vertically coincident UE/BS")
        phi = torch.atan2(direction[:, 1], direction[:, 0])
        beta = torch.atan2(-direction[:, 2], horizontal)
        cp, sp, cb, sb = phi.cos(), phi.sin(), beta.cos(), beta.sin()
        zero = torch.zeros_like(phi)
        rotation = torch.stack([
            torch.stack([cp * cb, -sp, cp * sb], dim=-1),
            torch.stack([sp * cb, cp, sp * sb], dim=-1),
            torch.stack([-sb, zero, cb], dim=-1),
        ], dim=-2)
        offset = self.ue_port_offsets_m.to(position.device)[site].reshape(-1, 4, 3)
        return position[:, None, :] + torch.einsum("bij,bmj->bmi", rotation, offset)


def _load_previous_adapter(label):
    """Inject only a module-local source PATHS; never alter global project state."""
    if label not in _ADAPTER_SHA256:
        raise ValueError("Unknown prior adapter")
    source_root = PATHS.root / "plus-pro"
    source_path = source_root / "code/measured/full_h_adapters_v1" / (label + ".py")
    payload = source_path.read_bytes()
    _require(hashlib.sha256(payload).hexdigest() == _ADAPTER_SHA256[label],
             "Frozen prior adapter changed: " + label)
    name = "_round2_paper_field_" + label + "_" + _ADAPTER_SHA256[label][:12]
    if name in sys.modules:
        return sys.modules[name]
    tree = ast.parse(payload, filename=str(source_path))
    tree.body = [node for node in tree.body
                 if not (isinstance(node, ast.ImportFrom)
                         and node.module == "project_paths")]
    module = ModuleType(name)
    module.__file__ = str(source_path)
    module.PATHS = SimpleNamespace(root=source_root, code_root=source_root / "code")
    sys.modules[name] = module
    try:
        exec(compile(tree, str(source_path), "exec"), module.__dict__)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


class _CompetitionField(nn.Module):
    def __init__(self, geometry):
        super().__init__()
        required = ("bs_port_positions_m", "ue_port_offsets_m",
                    "bs_polarization_labels", "ue_polarization_labels",
                    "geometry_fingerprint", "provenance", "ue_world_positions")
        missing = [name for name in required if not hasattr(geometry, name)]
        _require(not missing, "Missing physical geometry fields: " + ", ".join(missing))
        _require(callable(geometry.ue_world_positions),
                 "Geometry must provide dynamic UE world positions")
        bs = _real64(geometry.bs_port_positions_m, "BS positions")
        _require(bs.ndim == 5 and bs.shape[1:] == (2, 16, 8, 3),
                 "Expected BS[site,pol,h,v,xyz] physical positions")
        ue = _real64(geometry.ue_port_offsets_m, "UE local offsets")
        _require(ue.shape == (len(bs), 2, 1, 2, 3), "Expected local UE[S,2,1,2,3]")
        for name in ("bs_polarization_labels", "ue_polarization_labels"):
            labels = tuple(getattr(geometry, name))
            _require(len(labels) == 2 and len(set(labels)) == 2,
                     "Distinct polarization labels required")
        _require(isinstance(geometry.geometry_fingerprint, str)
                 and len(geometry.geometry_fingerprint) > 0,
                 "Geometry fingerprint is required")
        self.geometry = geometry
        self.bound_geometry_fingerprint = geometry.geometry_fingerprint
        self.num_sites = len(bs)
        self.register_buffer("bound_bs_ports", bs)
        self.register_buffer("bound_ue_local_offsets", ue)

    def _queries(self, ue_position, site_id, frequencies_hz):
        parameter = next(self.parameters())
        device = parameter.device
        _require(isinstance(ue_position, torch.Tensor)
                 and ue_position.is_floating_point() and ue_position.ndim == 2
                 and ue_position.shape[1] == 3 and len(ue_position) > 0,
                 "UE coordinates must be a nonempty floating [B,3] tensor")
        _require(isinstance(site_id, torch.Tensor) and site_id.ndim == 1
                 and len(site_id) == len(ue_position)
                 and site_id.dtype in (torch.int32, torch.int64),
                 "site_id must be integer [B] with zero-based site slots")
        _require(ue_position.device == device and site_id.device == device,
                 "Model, query positions and site IDs must share device")
        site = site_id.to(torch.int64)
        _require(bool(torch.isfinite(ue_position).all()) and
                 bool(((site >= 0) & (site < self.num_sites)).all()),
                 "Invalid query coordinates or site ID")
        _require(isinstance(frequencies_hz, torch.Tensor)
                 and frequencies_hz.dtype == torch.float64
                 and frequencies_hz.ndim == 1 and frequencies_hz.numel() > 0
                 and frequencies_hz.device == device
                 and bool(torch.isfinite(frequencies_hz).all())
                 and bool((frequencies_hz > 0).all()),
                 "Actual physical frequency must be finite positive float64[F]")
        _require(self.geometry.geometry_fingerprint == self.bound_geometry_fingerprint,
                 "Bound physical geometry changed")
        ports = self.geometry.ue_world_positions(ue_position, site)
        _require(isinstance(ports, torch.Tensor) and ports.shape == (len(site), 4, 3)
                 and ports.device == device and ports.is_floating_point()
                 and bool(torch.isfinite(ports).all()),
                 "Dynamic UE geometry must return finite [B,4,3]")
        return ue_position, site, frequencies_hz, ports

    def _base_metadata(self):
        return {
            "geometry_fingerprint": self.bound_geometry_fingerprint,
            "geometry_provenance": dict(self.geometry.provenance),
            "sites": self.num_sites,
            "port_order": {"BS": "p,h,v: 2x16x8", "UE": "p,h,v: 2x1x2"},
            "UE_orientation": "per-query geometry.ue_world_positions; offsets are LOCAL",
            "BS_polarization_labels": list(self.geometry.bs_polarization_labels),
            "UE_polarization_labels": list(self.geometry.ue_polarization_labels),
            "input": "UE position[B,3], zero-based site_id[B], actual float64 Hz[F]",
            "output": "complex64[B,256,4,F]",
            "prior_adapter_sha256": dict(_ADAPTER_SHA256),
            "source_path_binding": "module-local PATHS to plus-pro only; global project PATHS unchanged",
            "no_query_CSI_or_donor": True,
            "trained_or_scored": False,
        }


class CompetitionNeRF2(_CompetitionField):
    """Actual BS-element transmitters -> dynamically oriented UE-element receivers."""
    def __init__(self, geometry, field_config=None, **kwargs):
        super().__init__(geometry)
        previous = _load_previous_adapter("nerf2")
        if field_config is not None and kwargs:
            raise TypeError("Use field_config or keyword overrides")
        config = dict(field_config or kwargs)
        config.setdefault("reference_frequency_hz", 6690000000.0)
        config.setdefault("frequency_scale_hz", 192 * 30000.0)
        _require(config.get("propagation_sign", -1) == -1,
                 "Round2 shared contract fixes canonical negative propagation phase")
        self.field_config = config
        self.fields = nn.ModuleDict({
            f"s{s}_b{b}_u{u}": previous.NeRF2FullH(**config)
            for s in range(self.num_sites) for b in range(2) for u in range(2)
        })

    def forward(self, ue_position, site_id, frequencies_hz):
        ue, site, frequency, world_ports = self._queries(
            ue_position, site_id, frequencies_hz)
        output = torch.zeros((len(ue), 256, 4, len(frequency)),
                             dtype=torch.complex64, device=ue.device)
        for s in range(self.num_sites):
            rows = torch.nonzero(site == s, as_tuple=False).flatten()
            if rows.numel() == 0:
                continue
            bs_blocks = []
            for bs_pol in range(2):
                ue_blocks = []
                tx = self.bound_bs_ports[s, bs_pol].reshape(128, 3)
                for ue_pol in range(2):
                    rx = world_ports[rows, ue_pol * 2:(ue_pol + 1) * 2]
                    # Prior core output [128 BS transmitters, query, 2 UE elements,F].
                    value = self.fields[f"s{s}_b{bs_pol}_u{ue_pol}"](tx, rx, frequency)
                    ue_blocks.append(value.permute(1, 0, 2, 3))
                bs_blocks.append(torch.cat(ue_blocks, dim=2))
            output = output.index_copy(0, rows, torch.cat(bs_blocks, dim=1))
        return output

    def metadata(self):
        first = next(iter(self.fields.values())).metadata()
        return {**self._base_metadata(),
                "method": "NeRF2 Round2 full-H position-only adaptation",
                "parameterized_fields": self.num_sites * 4,
                "polarization_representation": "independent core per site/BS-pol/UE-pol; no copied H",
                "physical_direction": "actual BS-port TX, actual query UE-port RX ray origins",
                "core": first,
                "limits": ["Polarization identifiability is architectural, not a learned result",
                           "6.69GHz/30kHz are the declared V16 implementation convention, not independently certified Setup fields",
                           "Untrained integration; frequency/ray budgets are not an accepted experiment configuration"]}


class CompetitionNGRF(_CompetitionField):
    """Fixed BS256-head representation queried at each actual UE physical element."""
    def __init__(self, geometry, field_config=None, **kwargs):
        super().__init__(geometry)
        previous = _load_previous_adapter("ngrf")
        if field_config is not None and kwargs:
            raise TypeError("Use field_config or keyword overrides")
        config = dict(field_config or kwargs)
        _require("rx_position" not in config, "BS array is bound by geometry, not config")
        config.setdefault("frequency_center_hz", 6690000000.0)
        config.setdefault("frequency_scale_hz", 192 * 30000.0)
        self.field_config = config
        self.fields = nn.ModuleDict({
            f"s{s}_u{u}": previous.NGRFFullH(
                rx_position=self.bound_bs_ports[s].reshape(1, 256, 3), **config)
            for s in range(self.num_sites) for u in range(2)
        })

    def forward(self, ue_position, site_id, frequencies_hz):
        ue, site, frequency, world_ports = self._queries(
            ue_position, site_id, frequencies_hz)
        output = torch.zeros((len(ue), 256, 4, len(frequency)),
                             dtype=torch.complex64, device=ue.device)
        for s in range(self.num_sites):
            rows = torch.nonzero(site == s, as_tuple=False).flatten()
            if rows.numel() == 0:
                continue
            ue_blocks = []
            for ue_pol in range(2):
                core = self.fields[f"s{s}_u{ue_pol}"]
                query = world_ports[rows, ue_pol * 2:(ue_pol + 1) * 2]
                # A separate complex head exists for each bound BS port, including
                # co-located BS polarizations. Mobile-UE role remapping is explicit.
                value = core(query.reshape(-1, 3), core.fixed_rx_position, frequency)
                value = value[:, 0].reshape(len(rows), 2, 256, len(frequency))
                ue_blocks.append(value.permute(0, 2, 1, 3))
            output = output.index_copy(0, rows, torch.cat(ue_blocks, dim=2))
        return output

    def metadata(self):
        first = next(iter(self.fields.values())).metadata()
        return {**self._base_metadata(),
                "method": "nGRF Round2 fixed-BS-head full-H adaptation",
                "parameterized_fields": self.num_sites * 2,
                "polarization_representation": "256 distinct BS complex heads; independent site/UE-pol cores",
                "role_remapping": "mobile UE-element coordinates condition fixed BS-port heads; output transposed to [BS,UE,F]",
                "physical_reciprocity_verified": False,
                "core": first,
                "limits": ["Original TX/RX coordinate roles are explicitly remapped as a regression representation",
                           "Bound BS head order is fixed; arbitrary-array generalization is not claimed",
                           "Independent heads permit polarization differences; they are not learned until separately authorized training",
                           "No analytic or truth-based phase repair; complex phase remains a learned output",
                           "6.69GHz/30kHz are V16 implementation conventions, not independently certified Setup fields"]}


def synthetic_geometry():
    """An explicitly artificial geometry fixture; never a fallback for real data."""
    bs_centres = torch.tensor([[0., 0., 3.], [6., 0., 3.]], dtype=torch.float64)
    h = (torch.arange(16, dtype=torch.float64) - 7.5) * .06
    v = (torch.arange(8, dtype=torch.float64) - 3.5) * .06
    hh, vv = torch.meshgrid(h, v, indexing="ij")
    local = torch.stack([torch.zeros_like(hh), hh, vv], dim=-1)
    bs = bs_centres[:, None, None, None] + local[None, None]
    bs = bs.expand(2, 2, 16, 8, 3).clone()
    offset = torch.zeros((2, 2, 1, 2, 3), dtype=torch.float64)
    offset[:, :, 0, 0, 2], offset[:, :, 0, 1, 2] = -.02, .02
    return CompetitionGeometry(
        bs_centres, bs, offset, ("fixture_bs0", "fixture_bs1"),
        ("fixture_ue0", "fixture_ue1"),
        {"synthetic_only": True, "not_official_or_V16_geometry": True},
        "look_at_site_bs")


def synthetic_self_check():
    """Minimal CPU-only forward/autograd/chunk/polarization tests; no optimizer."""
    torch.set_num_threads(min(torch.get_num_threads(), 2))
    torch.manual_seed(13055)
    geometry = synthetic_geometry()
    position = torch.tensor([[1., .4, 1.], [4.3, -.6, 1.4]],
                            dtype=torch.float64, requires_grad=True)
    site = torch.tensor([0, 1], dtype=torch.int64)
    frequency = torch.tensor([6690000000., 6690030000.], dtype=torch.float64)
    nerf_config = dict(depth=2, width=8, skips=(), position_levels=1,
                       direction_levels=1, transmitter_levels=1, n_rays=1,
                       n_samples=2, far_m=.4, ray_chunk=64, frequency_chunk=1)
    ngrf_config = dict(num_gaussians=2, latent_dim=4, hidden_dim=8,
                       frequency_encoding_levels=1, frequency_chunk_size=1,
                       pos_encoding_freqs=1, attribute_num_layers=2,
                       decoder_num_layers=2, decoder_dropout_p=0.)
    summary = {}
    for label, cls, config in (("nerf2", CompetitionNeRF2, nerf_config),
                               ("ngrf", CompetitionNGRF, ngrf_config)):
        model = cls(geometry, field_config=config).cpu()
        value = model(position, site, frequency)
        _require(value.shape == (2, 256, 4, 2) and value.dtype == torch.complex64,
                 "Synthetic output shape/dtype")
        _require(bool(torch.isfinite(value).all()), "Synthetic forward finite")
        probe = value.real.sum() + .3 * value.imag.sum()
        gradients = torch.autograd.grad(probe, (position,) + tuple(model.parameters()),
                                        allow_unused=True)
        _require(gradients[0] is not None and bool(torch.isfinite(gradients[0]).all()),
                 "Query position gradient disconnected")
        _require(all(g is None or bool(torch.isfinite(g).all()) for g in gradients),
                 "Synthetic parameter derivative nonfinite")
        _require(not torch.equal(value[:, :128], value[:, 128:]),
                 "Co-located BS polarizations must not be copied")
        _require(not torch.equal(value[:, :, :2], value[:, :, 2:]),
                 "Co-located UE polarizations must not be copied")
        _require(not torch.equal(value[..., 0], value[..., 1]),
                 "Physical frequencies must not repeat one tone")
        changed = dict(config)
        if label == "nerf2":
            changed.update(ray_chunk=10000, frequency_chunk=10)
        else:
            changed.update(frequency_chunk_size=10)
        other = cls(geometry, field_config=changed).cpu()
        other.load_state_dict(model.state_dict())
        torch.testing.assert_close(value, other(position, site, frequency),
                                   rtol=5e-5, atol=2e-7)
        world = geometry.ue_world_positions(position, site)
        _require(not torch.equal(world[0] - position[0], world[1] - position[1]),
                 "Dynamic UE orientation was accidentally held fixed")
        rejected = False
        try:
            model(position, torch.tensor([0, 2]), frequency)
        except ValueError:
            rejected = True
        _require(rejected, "Out-of-range site IDs must fail")
        summary[label] = dict(shape=list(value.shape), finite_forward_backward=True,
                              chunk_equivalence=True,
                              co_located_polarizations_not_copied=True,
                              frequency_axis_not_repeated=True,
                              actual_dynamic_UE_positions=True)
    return dict(status="passed", device="cpu", synthetic_only=True,
                real_positions_predicted=False, channel_arrays_read=False,
                optimizer_created=False, trained=False, scored=False,
                geometry_fingerprint=geometry.geometry_fingerprint, checks=summary)


if __name__ == "__main__":
    print(json.dumps(synthetic_self_check(), indent=2))
