"""Physical phase alignment plus local linear interpolation.

This module is the in-repo implementation of the current physical baseline
direction:

* radial carrier phase, optionally with a small OFDM frequency slope;
* BS array steering phase on the (polarization, horizontal, vertical) layout;
* either constant IDW or a weighted local affine fit whose intercept is the
  query prediction.

The public shape mirrors ``PhaseAlignedIDW``: call ``fit`` with library
positions/channels, then ``predict`` with query positions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

try:  # scipy is present in the project venv, but keep a small fallback.
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - exercised only on minimal envs.
    cKDTree = None


M_P_DEFAULT = 2
M_H_DEFAULT = 16
M_V_DEFAULT = 8
STEERING_SCALE_DEFAULT = 4.71238898038


@dataclass
class PhysicalLocalLinearConfig:
    """Configuration for physical local interpolation.

    ``local_linear=False`` turns the model into a physical IDW blender. With
    ``alpha=0`` it is derotated IDW; with ``alpha>0`` it is geometric steering
    blend. ``local_linear=True`` uses the same corrected neighbor tensors but
    takes the intercept of a weighted local affine model.
    """

    k: int = 24
    p: float = 2.0
    beta: float = 140.3
    gamma: float = 0.15
    alpha: float = 1.0
    ridge: float = 0.3
    local_linear: bool = True
    align_phase: bool = True
    use_steering: bool = True
    shrink: float = 1.0
    steering_scale: float = STEERING_SCALE_DEFAULT
    eps: float = 1e-6
    m_p: int = M_P_DEFAULT
    m_h: int = M_H_DEFAULT
    m_v: int = M_V_DEFAULT

    def label(self) -> str:
        if self.local_linear:
            name = "physical_local_linear"
        elif self.use_steering and self.alpha:
            name = "geometric_steering_blend"
        elif self.align_phase:
            name = "derotated_idw"
        else:
            name = "plain_idw"
        parts = [
            f"k={self.k}",
            f"p={self.p:g}",
            f"beta={self.beta:g}",
        ]
        if self.gamma:
            parts.append(f"gamma={self.gamma:g}")
        if self.use_steering:
            parts.append(f"alpha={self.alpha:g}")
        if self.local_linear:
            parts.append(f"ridge={self.ridge:g}")
        if self.shrink != 1.0:
            parts.append(f"shrink={self.shrink:g}")
        return f"{name}(" + ",".join(parts) + ")"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _unit_vectors(pos: np.ndarray, bs: np.ndarray) -> np.ndarray:
    vec = np.asarray(pos, dtype=np.float64) - bs[None, :]
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    return vec / np.maximum(norm, 1e-30)


def _antenna_hv_coordinates(m_p: int, m_h: int, m_v: int) -> tuple[np.ndarray, np.ndarray]:
    """Return flattened H/V coordinates for M=(P,H,V) C-order layout."""

    h = np.arange(m_h, dtype=np.float64) - (m_h - 1) / 2.0
    v = np.arange(m_v, dtype=np.float64) - (m_v - 1) / 2.0
    hh, vv = np.meshgrid(h, v, indexing="ij")
    coords = np.empty((m_p, m_h, m_v, 2), dtype=np.float64)
    coords[..., 0] = hh[None, :, :]
    coords[..., 1] = vv[None, :, :]
    flat = coords.reshape(m_p * m_h * m_v, 2)
    return flat[:, 0], flat[:, 1]


class PhysicalLocalLinear:
    """Physical local interpolator for Round1 channels."""

    def __init__(self, bs_xyz, config: PhysicalLocalLinearConfig | None = None):
        self.bs = np.asarray(bs_xyz, dtype=np.float64)
        self.cfg = config or PhysicalLocalLinearConfig()
        self._pos: np.ndarray | None = None
        self._H = None
        self._d_lib: np.ndarray | None = None
        self._u_lib: np.ndarray | None = None
        self._tree = None
        self._h_coord: np.ndarray | None = None
        self._v_coord: np.ndarray | None = None

    def fit(self, lib_pos: np.ndarray, lib_H) -> "PhysicalLocalLinear":
        pos = np.asarray(lib_pos, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[1] < 3:
            raise ValueError(f"lib_pos must be (P,3), got {pos.shape}")
        if len(pos) == 0:
            raise ValueError("empty library")

        self._pos = pos
        self._H = lib_H
        self._d_lib = self._dist_to_bs(pos)
        self._u_lib = _unit_vectors(pos, self.bs)
        self._tree = cKDTree(pos[:, :2]) if cKDTree is not None else None

        cfg = self.cfg
        expected_m = cfg.m_p * cfg.m_h * cfg.m_v
        if lib_H.shape[1] != expected_m:
            raise ValueError(f"channel M={lib_H.shape[1]} does not match {cfg.m_p}*{cfg.m_h}*{cfg.m_v}")
        self._h_coord, self._v_coord = _antenna_hv_coordinates(cfg.m_p, cfg.m_h, cfg.m_v)
        return self

    def _dist_to_bs(self, pos: np.ndarray) -> np.ndarray:
        return np.linalg.norm(np.asarray(pos, dtype=np.float64) - self.bs[None, :], axis=1)

    def _query_neighbors(self, query_pos: np.ndarray, k: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        assert self._pos is not None
        q = np.asarray(query_pos, dtype=np.float64)
        if q.ndim == 1:
            q = q[None, :]
        kk = min(int(k or self.cfg.k), len(self._pos))
        if kk <= 0:
            raise ValueError(f"k must be positive, got {kk}")

        if self._tree is not None:
            try:
                dist, idx = self._tree.query(q[:, :2], k=kk, workers=-1)
            except TypeError:  # older scipy
                dist, idx = self._tree.query(q[:, :2], k=kk)
            if kk == 1:
                dist = dist[:, None]
                idx = idx[:, None]
            return np.asarray(dist, dtype=np.float64), np.asarray(idx, dtype=np.int64)

        dist = np.empty((len(q), kk), dtype=np.float64)
        idx = np.empty((len(q), kk), dtype=np.int64)
        for i, qi in enumerate(q):
            d = np.linalg.norm(self._pos[:, :2] - qi[:2], axis=1)
            part = np.argpartition(d, kk - 1)[:kk]
            part = part[np.argsort(d[part])]
            dist[i] = d[part]
            idx[i] = part
        return dist, idx

    def _idw_weights(self, distances: np.ndarray) -> np.ndarray:
        d = np.asarray(distances, dtype=np.float64)
        zero = np.flatnonzero(d <= self.cfg.eps)
        if len(zero):
            w = np.zeros_like(d, dtype=np.float64)
            w[int(zero[0])] = 1.0
            return w.astype(np.float32)
        w = 1.0 / np.maximum(d, self.cfg.eps) ** float(self.cfg.p)
        w /= np.sum(w)
        return w.astype(np.float32)

    def _affine_intercept_coefficients(self, rel_xy: np.ndarray, distances: np.ndarray) -> np.ndarray:
        """Return coefficients c so that intercept a = sum_i c_i y_i."""

        if len(distances) == 1 or np.min(distances) <= self.cfg.eps:
            return self._idw_weights(distances)

        w = self._idw_weights(distances).astype(np.float64)
        x = np.column_stack([np.ones(len(rel_xy), dtype=np.float64), np.asarray(rel_xy, dtype=np.float64)])
        xw = x * w[:, None]
        xtwx = x.T @ xw

        scale = max(float(np.trace(xtwx)) / xtwx.shape[0], 1e-12)
        reg = np.eye(3, dtype=np.float64) * (float(self.cfg.ridge) * scale)
        reg[0, 0] = 0.0
        mat = xtwx + reg
        e0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        try:
            row0 = np.linalg.solve(mat.T, e0)
        except np.linalg.LinAlgError:
            row0 = np.linalg.pinv(mat)[0]
        coeff = row0 @ xw.T
        return coeff.astype(np.float32)

    def _radial_phase(self, dist_delta: np.ndarray, n_subcarriers: int) -> np.ndarray:
        cfg = self.cfg
        sub = np.linspace(-0.5, 0.5, n_subcarriers, dtype=np.float32)
        beta_s = (float(cfg.beta) + float(cfg.gamma) * sub).astype(np.float32)
        arg = dist_delta.astype(np.float32)[:, None] * beta_s[None, :]
        return np.exp(1j * arg).astype(np.complex64)

    def _steering_phase(self, idx: np.ndarray, q_unit: np.ndarray) -> np.ndarray:
        assert self._u_lib is not None and self._h_coord is not None and self._v_coord is not None
        lib_u = self._u_lib[idx]
        q_h = float(q_unit[1])
        q_v = -float(q_unit[2])
        lib_h = lib_u[:, 1]
        lib_v = -lib_u[:, 2]
        dh = q_h - lib_h
        dv = q_v - lib_v
        arg = float(self.cfg.steering_scale) * (
            dh[:, None] * self._h_coord[None, :] + dv[:, None] * self._v_coord[None, :]
        )
        return np.exp(1j * arg).astype(np.complex64)

    def predict(self, query_pos: np.ndarray) -> np.ndarray:
        """Predict ``(Q,M,N,S)`` complex64 channels for query positions."""

        assert self._H is not None and self._pos is not None and self._d_lib is not None
        query_pos = np.asarray(query_pos, dtype=np.float64)
        if query_pos.ndim == 1:
            query_pos = query_pos[None, :]

        distances, neighbor_idx = self._query_neighbors(query_pos)
        q_dist = self._dist_to_bs(query_pos)
        q_unit = _unit_vectors(query_pos, self.bs)
        q_count = len(query_pos)
        m, n, s = self._H.shape[1], self._H.shape[2], self._H.shape[3]
        out = np.empty((q_count, m, n, s), dtype=np.complex64)
        cfg = self.cfg

        for qi in range(q_count):
            idx = neighbor_idx[qi]
            rel_xy = self._pos[idx, :2] - query_pos[qi, :2]
            if cfg.local_linear:
                coeff = self._affine_intercept_coefficients(rel_xy, distances[qi])
            else:
                coeff = self._idw_weights(distances[qi])

            terms = np.asarray(self._H[idx], dtype=np.complex64)
            if cfg.align_phase:
                dd = self._d_lib[idx] - q_dist[qi]
                radial = self._radial_phase(dd, s)
                terms = terms * radial[:, None, None, :]

            if cfg.use_steering and cfg.alpha != 0.0:
                steering = self._steering_phase(idx, q_unit[qi])
                if cfg.alpha == 1.0:
                    blend = steering
                else:
                    blend = (1.0 - float(cfg.alpha)) + float(cfg.alpha) * steering
                terms = terms * blend[:, :, None, None].astype(np.complex64)

            acc = np.tensordot(coeff, terms, axes=(0, 0))
            if cfg.shrink != 1.0:
                acc = acc * np.float32(cfg.shrink)
            out[qi] = acc.astype(np.complex64)
        return out

    def neighbor_diagnostics(self, query_pos: np.ndarray) -> dict[str, float]:
        distances, _ = self._query_neighbors(query_pos)
        nn = distances[:, 0]
        kth = distances[:, -1]
        mean_k = distances.mean(axis=1)
        pct = [0, 10, 25, 50, 75, 90, 100]
        nn_pct = np.percentile(nn, pct)
        kth_pct = np.percentile(kth, pct)
        return {
            "nn_min": float(nn_pct[0]),
            "nn_p10": float(nn_pct[1]),
            "nn_p25": float(nn_pct[2]),
            "nn_p50": float(nn_pct[3]),
            "nn_p75": float(nn_pct[4]),
            "nn_p90": float(nn_pct[5]),
            "nn_max": float(nn_pct[6]),
            "nn_mean": float(np.mean(nn)),
            "kth_p50": float(kth_pct[3]),
            "kth_p90": float(kth_pct[5]),
            "kth_mean": float(np.mean(kth)),
            "k_mean": float(np.mean(mean_k)),
        }
