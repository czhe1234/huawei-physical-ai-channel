#!/usr/bin/env python3
"""v301: exact continuous-delay transport with path-gain fingerprints."""

from __future__ import annotations

import json
import math

import numpy as np
import torch
import torch.nn as nn

import analyze_v297_continuous_delay_component_transport as v297
import analyze_v298_dirichlet_component_transport as v298
import analyze_v300_path_gain_fingerprint_oracle as v300
from project_paths import PATHS


VERSION_NUMBER = 301
VERSION = "v301_fingerprint_component_transport"
BASE_MODEL_ARRAYS = v297.model_arrays
FINGERPRINT: dict[str, np.ndarray] = {}


class FingerprintDirichletTransport(v298.ExactDirichletTransport):
    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__(node_dim, edge_dim)
        self.component_encoder = nn.Sequential(
            nn.Linear(44, 128), nn.SiLU(), nn.Linear(128, 48)
        )

    def forward(
        self,
        query_node: torch.Tensor,
        neighbor_node: torch.Tensor,
        edge: torch.Tensor,
        base_weights: torch.Tensor,
        component_delay: torch.Tensor,
        component_weight: torch.Tensor,
        component_valid: torch.Tensor,
        neighbor_diffuse: torch.Tensor,
        neighbor_reconstruction: torch.Tensor,
        anchor: torch.Tensor,
        fingerprint_horizontal: torch.Tensor,
        fingerprint_vertical: torch.Tensor,
        fingerprint_polarization: torch.Tensor,
        fingerprint_ue: torch.Tensor,
        fingerprint_gain_power: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, neighbors, components = component_delay.shape
        q = self.query_encoder(query_node)
        n = self.neighbor_encoder(neighbor_node)
        e = self.edge_encoder(edge)
        angle = 2.0 * math.pi * component_delay / v297.N_DELAY
        rank = torch.arange(components, device=component_delay.device).float()
        rank = rank[None, None, :].expand(batch, neighbors, -1)
        scalar = torch.stack(
            [
                torch.sin(angle),
                torch.cos(angle),
                torch.sin(2.0 * angle),
                torch.cos(2.0 * angle),
                torch.sin(4.0 * angle),
                torch.cos(4.0 * angle),
                torch.log(component_weight.clamp_min(1e-12)) / 12.0,
                component_weight,
                rank / max(components - 1, 1),
                component_valid.float(),
                neighbor_diffuse[:, :, None].expand(-1, -1, components),
                neighbor_reconstruction[:, :, None].expand(-1, -1, components),
                torch.log(
                    base_weights[:, :, None]
                    .expand(-1, -1, components)
                    .clamp_min(1e-12)
                )
                / 8.0,
            ],
            dim=3,
        )
        component_feature = torch.cat(
            [
                scalar,
                fingerprint_horizontal,
                fingerprint_vertical,
                fingerprint_polarization,
                fingerprint_ue,
                fingerprint_gain_power[:, :, :, None],
            ],
            dim=3,
        )
        c = self.component_encoder(component_feature)
        q_token = q[:, None, None, :].expand(-1, neighbors, components, -1)
        n_token = n[:, :, None, :].expand(-1, -1, components, -1)
        e_token = e[:, :, None, :].expand(-1, -1, components, -1)
        token = self.token_encoder(
            torch.cat([q_token, n_token, q_token * n_token, e_token, c], dim=3)
        ).reshape(batch, neighbors * components, -1)
        mask = ~component_valid.reshape(batch, -1)
        token = self.transformer(token, src_key_padding_mask=mask)

        prior = torch.log(
            (
                base_weights[:, :, None] * component_weight.clamp_min(1e-12)
            ).reshape(batch, -1)
        )
        token_logits = prior + 2.0 * torch.tanh(self.token_score(token).squeeze(2))
        token_logits = token_logits.masked_fill(mask, -1e9)
        token_energy = torch.softmax(token_logits, dim=1)
        shift_probability = torch.softmax(
            self.shift_prior[None, None, :]
            + 1.75 * torch.tanh(self.shift_score(token)),
            dim=2,
        )
        sparse = self.render_sparse(
            component_delay, token_energy, shift_probability, anchor
        )
        pooled = torch.sum(token_energy[:, :, None] * token, dim=1)
        gate = 0.65 * torch.sigmoid(self.gate(torch.cat([q, pooled], dim=1)))
        prediction = (1.0 - gate) * anchor + gate * sparse
        prediction /= prediction.sum(dim=1, keepdim=True).clamp_min(1e-30)
        return prediction, {
            "gate": gate.squeeze(1),
            "token_energy": token_energy,
            "shift_probability": shift_probability,
        }


def fingerprint_model_arrays(
    data: v297.v274.ModelSet,
    node_standardized: np.ndarray,
    components: v297.ComponentData,
    truth_d: np.ndarray,
) -> dict[str, np.ndarray]:
    arrays = BASE_MODEL_ARRAYS(data, node_standardized, components, truth_d)
    neighbor = data.neighbor_ids
    arrays.update(
        {
            "fingerprint_horizontal": FINGERPRINT["horizontal"][neighbor].astype(np.float32),
            "fingerprint_vertical": FINGERPRINT["vertical"][neighbor].astype(np.float32),
            "fingerprint_polarization": FINGERPRINT["polarization"][neighbor].astype(np.float32),
            "fingerprint_ue": FINGERPRINT["ue_port"][neighbor].astype(np.float32),
            "fingerprint_gain_power": FINGERPRINT["gain_power"][neighbor].astype(np.float32),
        }
    )
    return arrays


def configure() -> None:
    global FINGERPRINT
    if not v300.FINGERPRINT_CACHE.is_file():
        raise FileNotFoundError(v300.FINGERPRINT_CACHE)
    data = np.load(v300.FINGERPRINT_CACHE)
    FINGERPRINT = {key: data[key] for key in data.files}
    v297.VERSION_NUMBER = VERSION_NUMBER
    v297.VERSION = VERSION
    v297.OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
    v297.REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
    v297.CHECKPOINT_DIR = PATHS.checkpoints / VERSION
    v297.OUT_DIR.mkdir(parents=True, exist_ok=True)
    v297.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    v297.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    v297.REPORT = v297.REPORT_DIR / f"{VERSION}.json"
    v297.TARGETS_OUT = v297.OUT_DIR / "v301_fingerprint_delay_targets.npz"
    v297.OUTPUT = v297.OUT_DIR / "v301_Round1_Test_Channel.npy"
    v297.MODEL_CLASS = FingerprintDirichletTransport
    v297.model_arrays = fingerprint_model_arrays


def add_report_context() -> None:
    report = json.loads(v297.REPORT.read_text(encoding="utf-8"))
    report["additional_information_source"] = {
        "path_gain_tensor": "joint complex LS on each official-H continuous delay set",
        "fingerprint_dimensions": {
            "horizontal_angle": 16,
            "vertical_angle": 8,
            "polarization": 2,
            "ue_port": 4,
            "gain_power": 1,
        },
        "fingerprint_cache": PATHS.relative(v300.FINGERPRINT_CACHE),
    }
    v297.REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    configure()
    v297.main()
    add_report_context()
