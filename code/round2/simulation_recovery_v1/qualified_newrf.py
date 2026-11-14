"""Versioned NeWRF bridge for canonical-FIT geometric direction recovery.

The earlier synthetic adapter stays unchanged. This bridge requires a complete
FIT geometry bank. It uses cached FIT rays for FIT inputs, and an explicit
FIT-derived image-source prior for new-position inputs. It never turns on the
synthetic-only switch. No CSI loader, optimizer, or score is implemented here.
"""
from __future__ import annotations
import numpy as np
import torch
from round2.paper_baseline_adapters_v1.newrf import NeWRFFullH, _frequency64
from .ray_provider import FitGeometryBank, FitVirtualSourceRayProvider, TrainingGeometryRayProvider


def direction_readiness(bank):
    if type(bank) is not FitGeometryBank:
        raise TypeError("The reviewed FIT-limited geometry capability is required")
    m = bank.metadata()
    complete = bool(m["all_FIT_geometry_available"])
    empty = [int(r) for r in bank.source_fit_ids if bank.manifest["points"][str(int(r))]["path_count"] == 0]
    return dict(direction_contract_qualified=complete, full_canonical_FIT_covered=complete,
                remaining=[] if complete else ["Generate directions for the remaining canonical FIT rows"],
                input_identity=bank.inputs.identity, **m,
                zero_cached_path_FIT_count=len(empty),
                empty_path_policy="Keep each FIT row; use the same FIT-derived image-source/fixed-sphere provider as query inference",
                channel_quality_validated=False, training_executed=False,
                source="new approximate V16-geometry trace; not publisher path truth",
                unchanged_original_paper_reproduction=False)


class QualifiedNeWRFFullH(NeWRFFullH):
    """Same complex field/synthesis, with separately qualified FIT/query rays.

    All calls remain raw complex H in the original 256 x 4 x F port order.
    The path cap is an explicit compute budget. It is not a measured path count.
    """
    def __init__(self, bank, config=None, *, max_query_rays=64, max_training_rays=64):
        q = direction_readiness(bank)
        if not q["direction_contract_qualified"]:
            raise PermissionError("Partial FIT geometry is a correctness prototype, not a training-ready NeWRF adapter")
        if isinstance(max_training_rays, bool) or not isinstance(max_training_rays, int) or max_training_rays < 1:
            raise ValueError("Positive integer FIT-ray cap required")
        self.bank = bank
        self.fit_ray_cap = max_training_rays
        self.query_ray_cap = int(max_query_rays)
        provider = FitVirtualSourceRayProvider(bank, max_rays=max_query_rays)
        super().__init__(bank.inputs.geometry, provider, config, synthetic_only=False)
        self.training_rays = TrainingGeometryRayProvider(bank)
        self._source_fingerprint = bank.source_fingerprint
        self._source_ids = bank.source_fit_ids.copy()
        self._bank_manifest = q["manifest_sha256"]

    def _qualify(self):
        if self.synthetic_only:
            raise PermissionError("Qualified recovery must not use synthetic bypass")
        q = direction_readiness(self.bank)
        if not q["direction_contract_qualified"] or self.bank.source_fingerprint != self._source_fingerprint:
            raise PermissionError("FIT provenance changed")
        if not np.array_equal(self.bank.source_fit_ids, self._source_ids) or q["manifest_sha256"] != self._bank_manifest:
            raise PermissionError("FIT manifest or role changed after construction")
        return q

    def metadata(self):
        m = super().metadata()
        m.update(mode="canonical_FIT_geometry_qualified_adapter",
                 qualification=self._qualify(),
                 training_directions=self.training_rays.metadata(),
                 fit_ray_cap=self.fit_ray_cap, query_ray_cap=self.query_ray_cap,
                 fit_ray_selection="shortest delay first, stable source order; capped explicitly",
                 limits=["geometry-assisted NeWRF adaptation, not an unchanged author experiment",
                         "V16 physical convention; array parameters are not publisher-certified",
                         "FIT ray directions are approximate centre-array traces with actual-port corrections",
                         "query prior does not certify visibility or path birth/death",
                         "direction qualification does not establish convergence or prediction accuracy"])
        return m

    def _forward_provider(self, ue_position, site_id, frequencies_hz, provider, return_coarse):
        # Port packing and author field/synthesis are identical to the earlier adapter.
        device = self.bs_ports.device
        pos = torch.as_tensor(ue_position, dtype=torch.float64, device=device)
        sites = torch.as_tensor(site_id, device=device)
        if pos.ndim != 2 or pos.shape[1] != 3 or not len(pos) or not bool(torch.isfinite(pos).all()):
            raise ValueError("Finite nonempty UE[B,3] required")
        if sites.shape != (len(pos),) or sites.dtype not in (torch.int32, torch.int64) or bool(((sites < 0) | (sites >= len(self.bs_ports))).any()):
            raise ValueError("Valid integer site IDs required")
        f = _frequency64(self.physical_frequencies_hz if frequencies_hz is None else frequencies_hz, device)
        origins = torch.as_tensor(self.spec.ue_world_positions(pos, sites), dtype=torch.float64, device=device)
        if origins.shape != (len(pos), 4, 3) or not bool(torch.isfinite(origins).all()):
            raise ValueError("Dynamic UE port positions changed")
        tx = self.bs_ports[sites]
        d = torch.as_tensor(provider.directions(origins, sites, tx), dtype=torch.float64, device=device)
        if d.ndim != 6 or d.shape[:4] != (len(pos), 2, 128, 4) or d.shape[-1] != 3 or d.shape[-2] < 1 or not bool(torch.isfinite(d).all()):
            raise ValueError("Expected finite[B,2,128,4,R,3] ray directions")
        if not torch.allclose(torch.linalg.vector_norm(d, dim=-1), torch.ones(d.shape[:-1], device=device, dtype=torch.float64), atol=1e-6, rtol=1e-6):
            raise ValueError("Directions must be unit vectors")
        outputs = []; coarse_outputs = []
        for p in range(2):
            values = []; coarse_values = []
            for u in range(4):
                h, hc = self._scalar_links(p*4+u, origins[:,u], d[:,p,:,u], tx[:,p], f)
                values.append(h); coarse_values.append(hc)
            outputs.append(torch.stack(values, dim=2))
            coarse_outputs.append(torch.stack(coarse_values, dim=2))
        h = torch.stack(outputs, dim=1).reshape(len(pos), 256, 4, len(f))
        hc = torch.stack(coarse_outputs, dim=1).reshape_as(h)
        if not bool(torch.isfinite(h).all()) or not bool(torch.isfinite(hc).all()):
            raise FloatingPointError("Nonfinite recovered NeWRF output")
        return (h, hc) if return_coarse else h

    def forward(self, ue_position, site_id, frequencies_hz=None, *, return_coarse=False):
        """New-position inference: only FIT-derived provider; no query cache reads."""
        self._qualify()
        return self._forward_provider(ue_position, site_id, frequencies_hz, self.ray_provider, return_coarse)

    def forward_fit(self, row_ids, frequencies_hz=None, *, return_coarse=False):
        """FIT supervision interface. Reject a held/validation/guard ID before I/O."""
        self._qualify()
        rows = np.asarray(row_ids)
        if rows.ndim != 1 or rows.dtype.kind not in "iu" or not len(rows):
            raise ValueError("Nonempty integer FIT row IDs required")
        if not set(map(int, rows)).issubset(self.bank.allowed):
            raise PermissionError("Requested FIT rays outside this canonical FIT role")
        outputs = []; coarse_outputs = []
        model_device = self.bs_ports.device
        for row in rows:
            point = self.bank.get(int(row))
            if not len(point["tau_seconds"]):
                result = self._forward_provider(self.bank.inputs.positions[[row]], self.bank.inputs.site_id[[row]],
                                                frequencies_hz, self.ray_provider, return_coarse)
                if return_coarse:
                    outputs.append(result[0]); coarse_outputs.append(result[1])
                else:
                    outputs.append(result)
                continue
            record = self.training_rays.for_rows([row], self.spec.bs_port_positions_m)[0]
            chosen = np.argsort(point["tau_seconds"], kind="stable")[:self.fit_ray_cap]
            directions = torch.tensor(record["directions"][chosen], dtype=torch.float64, device=model_device)
            value = directions.reshape(len(chosen),2,128,4,3).permute(1,2,3,0,4).unsqueeze(0)
            class RowProvider:
                def directions(self, *args): return value
            result = self._forward_provider(self.bank.inputs.positions[[row]], self.bank.inputs.site_id[[row]],
                                            frequencies_hz, RowProvider(), return_coarse)
            if return_coarse:
                outputs.append(result[0]); coarse_outputs.append(result[1])
            else:
                outputs.append(result)
        h = torch.cat(outputs, dim=0)
        return (h, torch.cat(coarse_outputs, dim=0)) if return_coarse else h
