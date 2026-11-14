#!/usr/bin/env python3
"""v298: render transported continuous delays with exact Dirichlet atoms."""

from __future__ import annotations

import torch

import analyze_v297_continuous_delay_component_transport as v297
from project_paths import PATHS


VERSION_NUMBER = 298
VERSION = "v298_dirichlet_component_transport"


class ExactDirichletTransport(v297.ContinuousComponentTransport):
    def render_sparse(
        self,
        component_delay: torch.Tensor,
        token_energy: torch.Tensor,
        shift_probability: torch.Tensor,
        anchor: torch.Tensor,
    ) -> torch.Tensor:
        batch = len(component_delay)
        destination = component_delay.reshape(batch, -1, 1) + self.offsets
        with torch.no_grad():
            bins = torch.arange(
                v297.N_DELAY, device=destination.device, dtype=destination.dtype
            )
            delta = bins[None, None, None, :] - destination[:, :, :, None]
            delta = torch.remainder(
                delta + v297.N_DELAY / 2.0, v297.N_DELAY
            ) - v297.N_DELAY / 2.0
            atom = torch.square(
                torch.sinc(delta) / torch.sinc(delta / v297.N_DELAY)
            )
            atom /= atom.sum(dim=3, keepdim=True).clamp_min(1e-30)
        mass = token_energy[:, :, None] * shift_probability
        sparse = torch.einsum("bto,bton->bn", mass, atom)
        sparse /= sparse.sum(dim=1, keepdim=True).clamp_min(1e-30)
        return sparse


def configure() -> None:
    v297.VERSION_NUMBER = VERSION_NUMBER
    v297.VERSION = VERSION
    v297.OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
    v297.REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
    v297.CHECKPOINT_DIR = PATHS.checkpoints / VERSION
    v297.OUT_DIR.mkdir(parents=True, exist_ok=True)
    v297.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    v297.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    v297.REPORT = v297.REPORT_DIR / f"{VERSION}.json"
    v297.TARGETS_OUT = v297.OUT_DIR / "v298_dirichlet_delay_targets.npz"
    v297.OUTPUT = v297.OUT_DIR / "v298_Round1_Test_Channel.npy"
    v297.MODEL_CLASS = ExactDirichletTransport


if __name__ == "__main__":
    configure()
    v297.main()
