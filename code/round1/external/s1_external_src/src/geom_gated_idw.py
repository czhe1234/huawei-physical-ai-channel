"""Geometry-gated IDW for the Round1 map task.

This module keeps the PhaseAlignedIDW signal-combining path intact, but
replaces plain 2D kNN with geometry-aware neighbor ranking:

* LoS/NLoS mismatch can be hard-gated or softly penalized.
* Occlusion count, BS range, and azimuth differences can change the effective
  neighbor distance used for selection and IDW weights.
* A separate deep-blocked shrink is available for heavily occluded query points.

The implementation is intentionally small and local to this module so the
existing baseline remains a clean point of comparison.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import SceneGeometry
from .phase_aligned_idw import C_LIGHT, IDWConfig, PhaseAlignedIDW


@dataclass
class GeometryFeatures:
    """Per-point geometry features used by the gate."""

    los: np.ndarray
    blocked: np.ndarray
    r: np.ndarray
    theta: np.ndarray


@dataclass
class GeomGatedIDWConfig(IDWConfig):
    """IDWConfig plus geometry-gating hyperparameters.

    gate_mode:
        "none"      plain 2D kNN, useful for parity checks.
        "hard"      prefer same LoS/NLoS class; optionally require close
                    occlusion count.
        "soft"      rank by an effective distance with geometry penalties.
        "two_stage" first filter by LoS/occlusion, then soft-rank.
    """

    gate_mode: str = "soft"
    use_effective_distance_for_weights: bool = True

    los_penalty: float = 1.0
    occ_penalty: float = 0.5
    r_penalty: float = 0.0
    theta_penalty: float = 0.0

    occ_scale: float = 80.0
    r_scale: float = 80.0
    theta_scale_deg: float = 30.0

    hard_occ_window: float = -1.0
    two_stage_occ_window: float = 50.0

    deep_block_threshold: int = -1
    deep_block_shrink: float = 1.0

    def label(self) -> str:
        base = super().label()
        if self.gate_mode == "none":
            return f"geom-none {base}"
        parts = [self.gate_mode]
        if self.gate_mode in {"soft", "two_stage"}:
            parts.append(f"L{self.los_penalty:g}")
            parts.append(f"O{self.occ_penalty:g}")
            if self.r_penalty:
                parts.append(f"R{self.r_penalty:g}")
            if self.theta_penalty:
                parts.append(f"T{self.theta_penalty:g}")
        if self.deep_block_threshold >= 0 and self.deep_block_shrink != 1.0:
            parts.append(f"deep{self.deep_block_threshold}:{self.deep_block_shrink:g}")
        return f"geom-{','.join(parts)} {base}"


def make_geometry_features(
    pos: np.ndarray,
    bs_xyz,
    scene: SceneGeometry | None = None,
    blocked: np.ndarray | None = None,
    los: np.ndarray | None = None,
) -> GeometryFeatures:
    """Build LoS/occlusion/r/theta features for positions.

    If blocked is omitted, a SceneGeometry is required. r is the 3D BS distance;
    theta is azimuth in radians around the BS in the xy plane.
    """

    pts = np.asarray(pos, dtype=np.float64)
    bs = np.asarray(bs_xyz, dtype=np.float64)
    if blocked is None:
        if scene is None:
            raise ValueError("scene is required when blocked is not provided")
        blocked = scene.occlusion(pts)
    blocked = np.asarray(blocked, dtype=np.float64)
    if los is None:
        los = blocked == 0
    los = np.asarray(los, dtype=bool)

    d = pts - bs[None, :]
    r = np.linalg.norm(d, axis=1)
    theta = np.arctan2(d[:, 1], d[:, 0])
    return GeometryFeatures(los=los, blocked=blocked, r=r, theta=theta)


def _angle_abs_diff(a: np.ndarray, b: float) -> np.ndarray:
    """Wrapped absolute angular difference in radians."""

    return np.abs(np.angle(np.exp(1j * (a - b))))


class GeomGatedIDW(PhaseAlignedIDW):
    """PhaseAlignedIDW with geometry-aware neighbor selection."""

    def __init__(
        self,
        bs_xyz,
        scene: SceneGeometry | None = None,
        config: GeomGatedIDWConfig | None = None,
    ):
        super().__init__(bs_xyz, config or GeomGatedIDWConfig())
        self.scene = scene
        self._geom: GeometryFeatures | None = None

    def fit(
        self,
        lib_pos: np.ndarray,
        lib_H,
        lib_features: GeometryFeatures | None = None,
    ) -> "GeomGatedIDW":
        super().fit(lib_pos, lib_H)
        if lib_features is None:
            lib_features = make_geometry_features(self._pos, self.bs, self.scene)
        self._geom = lib_features
        return self

    def _query_features(
        self,
        query_pos: np.ndarray,
        query_features: GeometryFeatures | None,
    ) -> GeometryFeatures:
        if query_features is not None:
            return query_features
        return make_geometry_features(query_pos, self.bs, self.scene)

    def _effective_distance(
        self,
        d_xy: np.ndarray,
        q_los: bool,
        q_blocked: float,
        q_r: float,
        q_theta: float,
    ) -> np.ndarray:
        assert self._geom is not None
        cfg = self.cfg
        assert isinstance(cfg, GeomGatedIDWConfig)

        los_mismatch = self._geom.los != q_los
        occ_diff = np.abs(self._geom.blocked - q_blocked) / max(cfg.occ_scale, cfg.eps)
        r_diff = np.abs(self._geom.r - q_r) / max(cfg.r_scale, cfg.eps)
        theta_scale = np.deg2rad(max(cfg.theta_scale_deg, cfg.eps))
        theta_diff = _angle_abs_diff(self._geom.theta, q_theta) / theta_scale

        multiplier = (
            1.0
            + cfg.los_penalty * los_mismatch.astype(np.float64)
            + cfg.occ_penalty * occ_diff
            + cfg.r_penalty * r_diff
            + cfg.theta_penalty * theta_diff
        )
        return d_xy * np.maximum(multiplier, cfg.eps)

    def _candidate_mask(
        self,
        q_los: bool,
        q_blocked: float,
        mode: str,
    ) -> np.ndarray:
        assert self._geom is not None
        cfg = self.cfg
        assert isinstance(cfg, GeomGatedIDWConfig)

        same_los = self._geom.los == q_los
        if mode == "hard":
            mask = same_los.copy()
            if cfg.hard_occ_window >= 0:
                mask &= np.abs(self._geom.blocked - q_blocked) <= cfg.hard_occ_window
            if mask.sum() >= min(cfg.k, len(mask)):
                return mask
            if same_los.sum() >= min(cfg.k, len(mask)):
                return same_los
            return np.ones_like(same_los, dtype=bool)

        if mode == "two_stage":
            mask = same_los & (np.abs(self._geom.blocked - q_blocked) <= cfg.two_stage_occ_window)
            if mask.sum() >= min(cfg.k, len(mask)):
                return mask
            if same_los.sum() >= min(cfg.k, len(mask)):
                return same_los
            return np.ones_like(same_los, dtype=bool)

        return np.ones_like(same_los, dtype=bool)

    def _neighbors_geom(
        self,
        q_pos: np.ndarray,
        q_los: bool,
        q_blocked: float,
        q_r: float,
        q_theta: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return neighbor indices, raw 2D distances, and effective distances."""

        assert self._pos is not None and self._geom is not None
        cfg = self.cfg
        assert isinstance(cfg, GeomGatedIDWConfig)
        d_xy = np.linalg.norm(self._pos[:, :2] - q_pos[:2], axis=1)
        k = min(cfg.k, len(d_xy))
        mode = cfg.gate_mode.lower()

        if mode == "none":
            rank_dist = d_xy
            mask = np.ones(len(d_xy), dtype=bool)
            eff_dist = d_xy
        elif mode == "hard":
            mask = self._candidate_mask(q_los, q_blocked, mode)
            rank_dist = d_xy.copy()
            eff_dist = d_xy
        elif mode in {"soft", "two_stage"}:
            mask = self._candidate_mask(q_los, q_blocked, mode)
            eff_dist = self._effective_distance(d_xy, q_los, q_blocked, q_r, q_theta)
            rank_dist = eff_dist
        else:
            raise ValueError(f"unknown gate_mode: {cfg.gate_mode!r}")

        cand = np.flatnonzero(mask)
        if len(cand) <= k:
            idx = cand[np.argsort(rank_dist[cand])]
        else:
            part = np.argpartition(rank_dist[cand], k - 1)[:k]
            idx = cand[part]
            idx = idx[np.argsort(rank_dist[idx])]
        return idx, d_xy[idx], eff_dist[idx]

    def predict(
        self,
        query_pos: np.ndarray,
        query_features: GeometryFeatures | None = None,
    ) -> np.ndarray:
        """Predict channels for query_pos."""

        assert self._H is not None, "call fit() first"
        assert self._geom is not None, "geometry features are missing"
        query_pos = np.asarray(query_pos, dtype=np.float64)
        qgeom = self._query_features(query_pos, query_features)
        Q = query_pos.shape[0]
        M, N, S = self._H.shape[1], self._H.shape[2], self._H.shape[3]
        out = np.empty((Q, M, N, S), dtype=np.complex64)
        cfg = self.cfg
        assert isinstance(cfg, GeomGatedIDWConfig)
        dq_all = self._dist_to_bs(query_pos)

        s_axis = (np.arange(S) - (S - 1) / 2.0) * cfg.scs if cfg.per_subcarrier else None
        for qi in range(Q):
            idx, raw_dist, eff_dist = self._neighbors_geom(
                query_pos[qi],
                bool(qgeom.los[qi]),
                float(qgeom.blocked[qi]),
                float(qgeom.r[qi]),
                float(qgeom.theta[qi]),
            )
            dist_for_w = eff_dist if cfg.use_effective_distance_for_weights else raw_dist
            w = 1.0 / np.maximum(dist_for_w, cfg.eps) ** cfg.p
            w = w / w.sum()
            dq = dq_all[qi]

            Hk = np.asarray(self._H[idx]).astype(np.complex128)
            if cfg.align_phase:
                dd = self._d_lib[idx] - dq
                if cfg.per_subcarrier:
                    phase = cfg.beta * dd[:, None] + (2 * np.pi * dd[:, None] * s_axis[None, :] / C_LIGHT)
                    Hk = Hk * np.exp(1j * phase)[:, None, None, :]
                else:
                    Hk = Hk * np.exp(1j * cfg.beta * dd)[:, None, None, None]
            acc = np.tensordot(w, Hk, axes=(0, 0))

            if cfg.align_phase and cfg.power_separation:
                cur_pw = float((np.abs(acc) ** 2).sum())
                if cur_pw > 1e-30:
                    pred_logpw = float((w * self._logpw[idx]).sum())
                    scale = np.sqrt(np.exp(pred_logpw) / cur_pw)
                    acc = acc * scale
            if cfg.shrink != 1.0:
                acc = acc * cfg.shrink
            if cfg.deep_block_threshold >= 0 and qgeom.blocked[qi] >= cfg.deep_block_threshold:
                acc = acc * cfg.deep_block_shrink
            out[qi] = acc.astype(np.complex64)
        return out

    def neighbor_diagnostics(
        self,
        query_pos: np.ndarray,
        query_features: GeometryFeatures | None = None,
    ) -> dict[str, float]:
        """Summarize selected-neighbor geometry for reporting."""

        assert self._geom is not None
        query_pos = np.asarray(query_pos, dtype=np.float64)
        qgeom = self._query_features(query_pos, query_features)

        raw_means = []
        raw_nearest = []
        eff_means = []
        mismatch = []
        occ_diffs = []
        for qi in range(len(query_pos)):
            idx, raw, eff = self._neighbors_geom(
                query_pos[qi],
                bool(qgeom.los[qi]),
                float(qgeom.blocked[qi]),
                float(qgeom.r[qi]),
                float(qgeom.theta[qi]),
            )
            raw_means.append(float(raw.mean()))
            raw_nearest.append(float(raw[0]))
            eff_means.append(float(eff.mean()))
            mismatch.append(float((self._geom.los[idx] != qgeom.los[qi]).mean()))
            occ_diffs.append(float(np.abs(self._geom.blocked[idx] - qgeom.blocked[qi]).mean()))

        return {
            "nn_raw_mean": float(np.mean(raw_nearest)),
            "raw_k_mean": float(np.mean(raw_means)),
            "eff_k_mean": float(np.mean(eff_means)),
            "los_mismatch": float(np.mean(mismatch)),
            "occ_abs_diff": float(np.mean(occ_diffs)),
            "query_los_frac": float(np.mean(qgeom.los)),
            "query_blocked_mean": float(np.mean(qgeom.blocked)),
        }
