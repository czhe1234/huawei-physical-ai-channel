"""WRF-GS+ Gaussian core with an explicit learned DFT-codebook power readout.

This CPU task adaptation does not implement the paper's CUDA rasterizer or its
query-uplink-CSI case study.  It returns powers, never a complex H estimate.
Frequency and geometry must be supplied explicitly; no carrier is assumed.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

AUTHOR_COMMIT = "e564f3184463717ae08660a8b7232c6318c73ca6"
AUTHOR_HASHES = {
    "utils/pos_utils.py": "a7e14da60145afc6b2d341d861aaca20bc034b2bf7d32c46799b264ca08847ea",
    "utils/sh_utils.py": "7d1ff267546390635e6d1f68c4f88c4a8b052482d5c9be1d32f06bc69e9a96e7",
}
POWER_CONTRACT = "official_fft2_power_unshifted_P2_H16_V8_unnormalized"


def official_pas_power_targets(channel):
    """Exact official target operator; label construction, not a metric/score.

    Port order is [BS polarization, horizontal16, vertical8], and UE port order
    is untouched.  No FFT shift, orthonormal scaling, averaging or phase fit.
    """
    value = torch.as_tensor(channel)
    if value.device.type != "cpu" or value.ndim != 4 or value.shape[1:3] != (256, 4):
        raise ValueError("channel must be CPU complex [batch,256,4,F]")
    if not value.is_complex() or not torch.isfinite(value).all():
        raise ValueError("channel must be finite complex values")
    shaped = value.reshape(len(value), 2, 16, 8, 4, value.shape[-1])
    return torch.fft.fft2(shaped, dim=(2, 3)).abs().square().reshape_as(value).real


def _array(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


def _author_objects(source_root):
    """Load only verified author's definitions, without CUDA/package imports."""
    if source_root is None:
        from project_paths import PATHS
        source_root = PATHS.root / "plus-pro/code/third_party/paper_baselines/wrf_gsplus"
    root = Path(source_root)
    sources = {}
    for relative, expected in AUTHOR_HASHES.items():
        source = (root / relative).read_bytes()
        if hashlib.sha256(source).hexdigest() != expected:
            raise ValueError(f"Author source identity mismatch: {relative}")
        sources[relative] = source.decode()
    namespace = {"torch": torch, "nn": nn, "F": F, "np": np,
                 "__name__": "wrfgs_verified_author_definitions"}
    for relative, source in sources.items():
        tree = ast.parse(source)
        # The unused is_6dof branch needs exp_se3; it is never enabled here.
        nodes = [item for item in tree.body if isinstance(
            item, (ast.FunctionDef, ast.ClassDef, ast.Assign))]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), relative, "exec"), namespace)
    return namespace["DeformNetwork"], namespace["get_embedder"], namespace["eval_sh"]


def _quaternion_matrix(q):
    q = F.normalize(q, dim=-1, eps=1e-8)
    r, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-r*z), 2*(x*z+r*y),
                        2*(x*y+r*z), 1-2*(x*x+z*z), 2*(y*z-r*x),
                        2*(x*z-r*y), 2*(y*z+r*x), 1-2*(x*x+y*y)), -1).reshape(-1, 3, 3)


def _project(means, covariance, center, camera, width, height):
    local = (means-center) @ camera
    x, y, z = local.unbind(-1)
    rho2 = x.square()+z.square()
    rho = rho2.sqrt().clamp_min(1e-6)
    rr = (rho2+y.square()).clamp_min(1e-6)
    px = (torch.atan2(x, z)/torch.pi+1)*width/2-.5
    py = (torch.asin((y/rr.sqrt()).clamp(-.999999, .999999))/torch.pi+.5)*height-.5
    fx, fy = width/(2*torch.pi), height/torch.pi
    jacobian = torch.stack((fx*z/rho2.clamp_min(1e-6), torch.zeros_like(x),
                           -fx*x/rho2.clamp_min(1e-6), -fy*x*y/rho/rr,
                           fy*rho/rr, -fy*z*y/rho/rr), -1).reshape(-1, 2, 3)
    local_cov = camera.T @ covariance @ camera
    cov2 = jacobian @ local_cov @ jacobian.transpose(-1, -2)
    cov2 = cov2+.3*torch.eye(2, dtype=means.dtype, device=means.device)
    return torch.stack((px, py), -1), cov2, rr.sqrt()


def _alpha_render(pixel, covariance, depth, opacity, color, width, height):
    """Dense CPU alpha blending; no tile/pruning parity with official CUDA."""
    yy, xx = torch.meshgrid(torch.arange(height, dtype=pixel.dtype),
                           torch.arange(width, dtype=pixel.dtype), indexing="ij")
    grid = torch.stack((xx, yy), -1).reshape(-1, 2)
    delta = pixel[:, None]-grid[None]
    exponent = -.5*torch.einsum("kpi,kij,kpj->kp", delta,
                                torch.linalg.inv(covariance), delta)
    alpha = (opacity[:, None]*exponent.exp()).clamp(max=.99)
    alpha = torch.where((alpha >= 1/255) & (depth[:, None] > .2), alpha, 0.)
    order = depth.argsort()
    alpha, color = alpha[order], color[order]
    post = torch.cumprod(1-alpha, dim=0)
    pre = torch.cat((torch.ones_like(post[:1]), post[:-1]), 0)
    weights = alpha*pre*(post >= 1e-4)
    # A floating power target replaces the PNG display range. No output clipping.
    return torch.einsum("kp,kc->pc", weights, color)


class _FrequencyCondition(nn.Module):
    def __init__(self, position_encoder, frequency_encoder):
        super().__init__()
        self.position_encoder, self.frequency_encoder = position_encoder, frequency_encoder

    def forward(self, value):
        return torch.cat((self.position_encoder(value[..., :3]),
                          self.frequency_encoder(value[..., 3:])), -1)


class _SiteField(nn.Module):
    def __init__(self, initial_xyz, center, axes, input_center, input_scale,
                 author, width, height):
        super().__init__()
        network_class, get_embedder, self.eval_sh = author
        xyz = torch.tensor(initial_xyz, dtype=torch.float32)
        self.xyz = nn.Parameter(xyz)
        self.log_scale = nn.Parameter(torch.full_like(xyz, np.log(.85)))
        self.quaternion = nn.Parameter(torch.tensor([[1., 0., 0., 0.]]).repeat(len(xyz), 1))
        self.opacity_logit = nn.Parameter(torch.full((len(xyz),), -2.1972246))
        self.sh = nn.Parameter(torch.zeros(len(xyz), 16, 8))
        nn.init.normal_(self.sh[:, 0], std=.1)
        self.deformation = network_class(is_blender=True, is_6dof=False)
        frequency_encoder, frequency_dim = get_embedder(6, 1)
        position_encoder = self.deformation.embed_pos_fn
        old_linear = self.deformation.posnet[0]
        self.deformation.embed_pos_fn = _FrequencyCondition(position_encoder, frequency_encoder)
        self.deformation.posnet[0] = nn.Linear(old_linear.in_features+frequency_dim,
                                              old_linear.out_features)
        # Eight independent channels are [BS polarization, UE port]; none is copied.
        self.deformation.gaussian_signal = nn.Linear(self.deformation.W, 8)
        self.deformation.gaussian_phase = nn.Linear(self.deformation.W, 8)
        for name in ("gaussian_warp", "gaussian_phase"):
            for parameter in getattr(self.deformation, name).parameters():
                parameter.requires_grad_(False)
        for name in ("gaussian_rotation", "gaussian_scaling"):
            nn.init.zeros_(getattr(self.deformation, name).weight)
            nn.init.zeros_(getattr(self.deformation, name).bias)
        self.register_buffer("center", torch.tensor(center, dtype=torch.float32))
        self.register_buffer("camera", torch.tensor(axes[:, [0, 1, 2]], dtype=torch.float32))
        self.register_buffer("input_center", torch.tensor(input_center, dtype=torch.float32))
        self.register_buffer("input_scale", torch.tensor(input_scale, dtype=torch.float32))
        self.width, self.height = width, height
        # Explicit learned power readout, indexed by unshifted [H16,V8] bins.
        # It is not a physical conversion of angular intensity into coherent beams.
        self.readout_logits = nn.Parameter(torch.empty(8, 128, width*height))
        nn.init.normal_(self.readout_logits, std=.08)
        self.channel_log_gain = nn.Parameter(torch.zeros(8))

    def forward(self, position, normalized_frequency):
        xyz_input = (self.xyz.detach()-self.input_center)/self.input_scale
        position_input = (position-self.input_center)/self.input_scale
        condition = torch.cat((position_input, normalized_frequency.reshape(1)))
        condition = condition.expand(len(self.xyz), -1)
        _, rotation, scale, signal = self.deformation(xyz_input, condition)
        q = F.normalize(self.quaternion, dim=-1, eps=1e-8)+rotation
        matrix = _quaternion_matrix(q)
        scales = (self.log_scale.exp()+scale).abs().clamp_min(1e-3)
        covariance = matrix @ torch.diag_embed(scales.square()) @ matrix.transpose(-1, -2)
        pixel, cov2, depth = _project(self.xyz, covariance, self.center,
                                      self.camera, self.width, self.height)
        direction = F.normalize(self.xyz-self.center, dim=-1, eps=1e-8)
        sh = (self.sh+signal[:, None, :]).transpose(1, 2)
        color = F.softplus(self.eval_sh(3, sh, direction)+.5)
        image = _alpha_render(pixel, cov2, depth, self.opacity_logit.sigmoid(),
                              color, self.width, self.height)
        readout = F.softplus(self.readout_logits)
        readout = readout/readout.sum(-1, keepdim=True).clamp_min(1e-12)
        power = torch.einsum("cjp,pc->cj", readout, image)
        power = power*F.softplus(self.channel_log_gain)[:, None]
        # c=[pol2,UE4], j=[horizontal16,vertical8].
        return power.reshape(2, 4, 128).permute(0, 2, 1).reshape(256, 4)


class WRFGSPlusPowerAdapter(nn.Module):
    """Position/frequency-only power predictor with explicit task binding.

    ``bs_axes`` columns are [camera horizontal, up, normal] in world coordinates.
    They organize the Gaussian representation; readout weights are learned in
    the declared DFT-bin order. ``contract_id`` binds this model to independently
    verified target construction. Model construction never reads any CSI.
    """
    def __init__(self, fit_position, bs_positions, bs_axes, initial_xyz_by_site,
                 frequencies_hz, *, source_root=None, width=32, height=16,
                 contract_id=None, geometry_identity=None):
        super().__init__()
        positions = np.asarray(bs_positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3 or not len(positions):
            raise ValueError("bs_positions must have shape [site,3]")
        positions = _array(positions, positions.shape, "bs_positions")
        axes = _array(bs_axes, (len(positions), 3, 3), "bs_axes")
        if not np.allclose(axes.transpose(0, 2, 1)@axes, np.eye(3), atol=1e-5):
            raise ValueError("bs_axes must have orthonormal columns")
        if not np.allclose(np.linalg.det(axes), 1., atol=1e-5):
            raise ValueError("bs_axes must be right handed")
        fit = np.asarray(fit_position, dtype=np.float64)
        if fit.ndim != 2 or fit.shape[1] != 3 or not len(fit) or not np.isfinite(fit).all():
            raise ValueError("fit_position must be finite nonempty [fit,3]")
        frequencies = np.asarray(frequencies_hz, dtype=np.float64)
        if frequencies.ndim != 1 or not len(frequencies) or not np.isfinite(frequencies).all():
            raise ValueError("frequencies_hz must be an explicit finite nonempty vector")
        if (frequencies <= 0).any() or len(np.unique(frequencies)) != len(frequencies):
            raise ValueError("frequencies must be positive and unique")
        if width < 2 or height < 2 or not isinstance(width, int) or not isinstance(height, int):
            raise ValueError("angular render dimensions must be integers >=2")
        if len(initial_xyz_by_site) != len(positions):
            raise ValueError("one explicitly initialized Gaussian field is required per site")
        self.register_buffer("frequencies_hz", torch.tensor(frequencies, dtype=torch.float64))
        frequency_scale = np.ptp(frequencies)
        self.register_buffer("frequency_center", torch.tensor(frequencies.mean(), dtype=torch.float64))
        self.register_buffer("frequency_scale", torch.tensor(frequency_scale if frequency_scale else 1., dtype=torch.float64))
        author = _author_objects(source_root)
        fields = []
        for site, initial in enumerate(initial_xyz_by_site):
            initial = np.asarray(initial, dtype=np.float64)
            if initial.ndim != 2 or initial.shape[1] != 3 or not len(initial) or not np.isfinite(initial).all():
                raise ValueError("initial_xyz_by_site entries must be finite [Gaussian,3]")
            fields.append(_SiteField(initial, positions[site], axes[site],
                                      fit.mean(0), np.maximum(fit.std(0), .1),
                                      author, width, height))
        self.fields = nn.ModuleList(fields)
        self.contract_id, self.geometry_identity = contract_id, geometry_identity

    def forward(self, ue_position, site_id, frequencies_hz):
        if any(parameter.device.type != "cpu" for parameter in self.parameters()):
            raise ValueError("this correctness adapter is CPU-only")
        position = torch.as_tensor(ue_position, dtype=torch.float32)
        site = torch.as_tensor(site_id)
        frequencies = torch.as_tensor(frequencies_hz, dtype=torch.float64)
        if position.device.type != "cpu" or site.device.type != "cpu" or frequencies.device.type != "cpu":
            raise ValueError("CPU inputs are required")
        if position.ndim != 2 or position.shape[1] != 3 or not torch.isfinite(position).all():
            raise ValueError("ue_position must be finite [batch,3]")
        if site.shape != (len(position),) or site.dtype not in (torch.int32, torch.int64):
            raise ValueError("site_id must be an integer vector matching batch")
        if ((site < 0) | (site >= len(self.fields))).any():
            raise ValueError("unknown site_id")
        if frequencies.shape != self.frequencies_hz.shape or not torch.equal(frequencies, self.frequencies_hz):
            raise ValueError("frequency vector/order differs from the bound explicit spec")
        normalized = ((frequencies-self.frequency_center)/self.frequency_scale).float()
        if not len(position):
            return position.new_empty((0, 256, 4, len(frequencies)))
        return torch.stack([torch.stack([self.fields[int(s)](p, f)
                            for f in normalized], -1) for p, s in zip(position, site)])

    def eligibility(self):
        return {
            "output": "DFT_codebook_power[B,256,4,F]",
            "official_PAS": (self.contract_id == POWER_CONTRACT
                             and isinstance(self.geometry_identity, str)
                             and bool(self.geometry_identity)),
            "PAS_requires": POWER_CONTRACT,
            "PAS_binding": "caller_declared_requires_external_operator_identity_check",
            "H": False, "PDP": False, "NMSE": False,
            "original_CUDA_reproduction": False,
            "physical_angle_to_codebook_equivalence_claimed": False,
            "source_commit": AUTHOR_COMMIT,
            "adaptation": "frequency_conditioned_Gaussian_splat_with_learned_codebook_power_readout",
            "query_CSI_required": False,
            "geometry_identity": self.geometry_identity,
        }
