"""Prespecified power baselines; fitting is explicit and uses TrainingBank only."""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist
import torch
from torch import nn
from .common import check_prediction
from .metrics import directional_power, normalized_power, codebook

class RBFPower:
    def __init__(self, bank, length_m=1., ridge=.003):
        if length_m <= 0 or ridge <= 0: raise ValueError("Positive kernel settings required")
        self.position = np.array(bank.position, copy=True)
        self.source_ids = np.array(bank.row_ids, copy=True)
        self.length_m, self.ridge = float(length_m), float(ridge)
        target = np.sqrt(normalized_power(directional_power(bank.channel))).reshape(len(bank.row_ids), -1)
        k = np.exp(-cdist(self.position[:, :2], self.position[:, :2], "sqeuclidean") / (2*length_m**2))
        self.coefficient = np.linalg.solve(k + ridge*np.eye(len(k)), target)

    def predict(self, query):
        k = np.exp(-cdist(query.position[:, :2], self.position[:, :2], "sqeuclidean") / (2*self.length_m**2))
        power = np.maximum(k @ self.coefficient, 0).reshape(len(query.row_ids), 2, 32)**2
        return check_prediction(dict(method="sqrt_direction_power_RBF_kernel", power=power,
                                     source_ids=self.source_ids), query)

class PositionPowerMLP(nn.Module):
    def __init__(self, fit_positions, levels=6, hidden=96):
        super().__init__()
        p = np.asarray(fit_positions, np.float64)
        self.register_buffer("center", torch.tensor(p.mean(0), dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(np.maximum(p.std(0), .1), dtype=torch.float32))
        self.register_buffer("frequencies", 2**torch.arange(levels, dtype=torch.float32)*torch.pi)
        self.network = nn.Sequential(nn.Linear(3+6*levels, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 64))

    def forward(self, position):
        x = (position-self.center)/self.scale
        phase = x[..., None]*self.frequencies
        encoded = torch.cat([x, phase.sin().flatten(-2), phase.cos().flatten(-2)], -1)
        return nn.functional.softplus(self.network(encoded)).reshape(-1, 2, 32)

    @torch.inference_mode()
    def predict(self, query):
        self.eval(); device = self.center.device
        p = self(torch.tensor(np.array(query.position), dtype=torch.float32, device=device)).cpu().numpy()
        return check_prediction(dict(method="position_encoded_power_MLP", power=p), query)

class DirectPointing:
    def predict(self, query):
        delta = query.position[:, None, :] - query.spec.center[None]
        direction = delta / np.linalg.norm(delta, axis=-1, keepdims=True)
        phase = 2*np.pi*query.spec.carrier_hz/299792458. * np.einsum("npi,phvi->nphv", direction, query.spec.element_offsets)
        steering = np.exp(1j*phase).reshape(len(query.row_ids), 2, 32)
        beam = np.abs(np.einsum("ab,npa->npb", codebook().conj(), steering))**2
        # Only a beam-choice geometry reference; no true channel reconstructed.
        return check_prediction(dict(method="direct_pointing_same32beam_reference", power=beam), query)
