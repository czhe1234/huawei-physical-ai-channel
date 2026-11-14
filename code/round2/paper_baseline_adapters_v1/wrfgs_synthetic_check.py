"""Synthetic CPU correctness checks only; never load channels or fit a model."""
from __future__ import annotations

import json
import time
import numpy as np
import torch

from .wrfgs import WRFGSPlusPowerAdapter, POWER_CONTRACT, official_pas_power_targets


def run_check():
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        # Artificial values for an input-contract check, not a verified dataset spec.
        frequencies = np.array([6.69e9-30000., 6.69e9, 6.69e9+30000.])
        centers = np.array([[0., 0., 5.], [20., 0., 5.]])
        axes = np.tile(np.eye(3), (2, 1, 1))
        offset = np.array([[2., 0., 1.], [-2., 0., 1.],
                           [0., 2., 1.], [0., -2., 1.]])
        initial = [center+offset for center in centers]
        fit = np.array([[0., 0., 1.5], [10., -2., 1.5], [20., 2., 1.5]])
        model = WRFGSPlusPowerAdapter(fit, centers, axes, initial, frequencies,
                                      width=8, height=4, contract_id=POWER_CONTRACT,
                                      geometry_identity="synthetic-only")
        position = torch.tensor([[3., 1., 1.5], [3., 1., 1.5]])
        start = time.perf_counter()
        prediction = model(position, torch.tensor([0, 1]), frequencies)
        assert prediction.shape == (2, 256, 4, 3)
        assert prediction.dtype == torch.float32
        assert torch.isfinite(prediction).all() and (prediction >= 0).all()
        channels = prediction.reshape(2, 2, 128, 4, 3).permute(0, 1, 3, 2, 4).reshape(2, 8, 128, 3)
        channel_difference = float((channels[:, 1:]-channels[:, :1]).abs().max())
        site_difference = float((prediction[0]-prediction[1]).abs().max())
        frequency_difference = float((prediction[..., 0]-prediction[..., 2]).abs().max())
        assert min(channel_difference, site_difference, frequency_difference) > 1e-7
        prediction.square().mean().backward()
        for field in model.fields:
            for parameter in [field.sh, field.readout_logits,
                              field.deformation.posnet[0].weight,
                              field.xyz, field.log_scale, field.channel_log_gain]:
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                assert parameter.grad.abs().sum() > 0
            assert (field.channel_log_gain.grad != 0).all()
        # No metric/score: two independently written FFT-power equations agree.
        rng = np.random.default_rng(2)
        channel = (rng.normal(size=(1, 256, 4, 3))+
                   1j*rng.normal(size=(1, 256, 4, 3)))
        actual = official_pas_power_targets(torch.tensor(channel, dtype=torch.complex128)).numpy()
        expected = np.abs(np.fft.fft2(channel.reshape(1, 2, 16, 8, 4, 3), axes=(2, 3)))**2
        operator_error = float(np.max(np.abs(actual-expected.reshape(channel.shape))))
        assert operator_error < 1e-9
        rejected = 0
        invalid = [(torch.tensor([0, 2]), frequencies),
                   (torch.tensor([0., 1.]), frequencies),
                   (torch.tensor([0, 1]), frequencies[::-1].copy())]
        for site, frequency in invalid:
            try:
                model(position, site, frequency)
            except ValueError:
                rejected += 1
        assert rejected == len(invalid)
        assert model.eligibility()["official_PAS"] and not model.eligibility()["H"]
        model.contract_id = None
        assert not model.eligibility()["official_PAS"]
        return {
            "scope": "synthetic_CPU_only_no_real_CSI_training_scoring_or_GPU",
            "shape": list(prediction.shape), "finite_nonnegative": True,
            "channel_max_difference": channel_difference,
            "site_max_difference": site_difference,
            "frequency_max_difference": frequency_difference,
            "selected_core_gradients_finite_nonzero": True,
            "all_eight_channel_gain_gradients_nonzero": True,
            "fft_power_operator_numpy_max_abs_error": operator_error,
            "invalid_input_rejections": rejected,
            "unbound_PAS_rejected": True,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "seconds": time.perf_counter()-start,
        }


if __name__ == "__main__":
    print(json.dumps(run_check(), indent=2))
