"""One registered fit-health repair of the dc01 NeWRF adaptation.

The native hard-ReLU extinction model collapsed to exactly zero on fit data.
Use a smooth approximation plus a small positive floor. All original network
bodies, coarse/fine sampling and coherent synthesis remain in neural_fields.
This is an explicitly disclosed stability adaptation, not the native paper.
"""
from pathlib import Path
import shutil

import torch
import torch.nn.functional as F

import neural_fields as original


def install():
    original_root = original.ARTIFACTS
    repaired_root = original_root / "newrf_stability"
    (repaired_root / "shared").mkdir(parents=True, exist_ok=True)
    source = original_root / "shared/array_geometry.npz"
    target = repaired_root / "shared/array_geometry.npz"
    if target.exists() and target.read_bytes() != source.read_bytes():
        raise ValueError("Stability-run array geometry identity mismatch")
    if not target.exists():
        shutil.copyfile(source, target)
    original.ARTIFACTS = repaired_root
    original.REPORTS = original.REPORTS / "newrf_stability"
    original_configure = original.configure

    def configure(method, geometry, meta, ids, device):
        if method != "newrf":
            raise ValueError("This registered repair applies only to NeWRF")
        model = original_configure(method, geometry, meta, ids, device)
        native_synthesis = model.synthesis.synthesize

        def synthesize(raw, z_vals, fc, ray_batches=None):
            density = F.softplus(raw[..., -1], beta=20.)+1e-4
            adapted = torch.cat([raw[..., :2], density[..., None]], -1)
            return native_synthesis(adapted, z_vals, fc, ray_batches)

        model.synthesis.synthesize = synthesize
        return model

    original.configure = configure
    native_identity = original.training_identity

    def identity(method, fold, config, dependencies=()):
        config = dict(config, extinction_density="softplus(beta=20)+1e-4; fit-health repair of dead hard ReLU",
                      repair_selected_from="fit-only exact-zero-output check; no NeWRF test observations")
        return native_identity(method, fold, config, [*dependencies, Path(__file__)])

    original.training_identity = identity


if __name__ == "__main__":
    install()
    original.main()
