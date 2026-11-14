# （本改动由Claude fable5）
"""Shared utilities for the marginal power-field model.

Data objects (all in the derotated angle-delay domain, layout (P,H,V,N,tau)):
  truth collapsed power  coll[i]  = sum_{pol,ue} |ad_truth|^2   (16,8,192)
  base  collapsed power  bcoll[i] = sum_{pol,ue} |ad_base |^2   (16,8,192)

Marginal targets per point:
  beam map  B[i]  = coll[i].sum(-1)        (16,8) -> 128 flat
  delay pdp D[i]  = coll[i].sum((0,1))     (192,)
  total     T[i]  = coll[i].sum()
"""

import numpy as np

from project_paths import PATHS

ASSETS = PATHS.assets
BS_POS = np.array([50.0, 0.0, 25.0])
TINY = 1e-30


def regime_of(blocked):
    r = np.full(blocked.shape, "medium", dtype=object)
    r[blocked <= 0] = "clear"
    r[(blocked >= 1) & (blocked <= 25)] = "light"
    r[(blocked > 25) & (blocked <= 100)] = "medium"
    r[blocked > 100] = "heavy"
    return r


def build_collapsed_caches(force=False):
    """Build/load collapsed truth and base power caches for the 2000 train points."""
    import os
    tf = ASSETS + "/truth_coll_power.npy"
    bf = ASSETS + "/base_coll_power.npy"
    if not force and os.path.exists(tf) and os.path.exists(bf):
        return np.load(tf), np.load(bf)
    base_mm = np.load(ASSETS + "/train_v36_loo_base_derot_ad.npy", mmap_mode="r")
    res_mm = np.load(ASSETS + "/train_v36_loo_residual_derot_ad.npy", mmap_mode="r")
    coll = np.empty((2000, 16, 8, 192), dtype=np.float32)
    bcoll = np.empty((2000, 16, 8, 192), dtype=np.float32)
    for s in range(0, 2000, 100):
        e = min(s + 100, 2000)
        b = np.asarray(base_mm[s:e])
        t = b + np.asarray(res_mm[s:e])
        coll[s:e] = (np.abs(t) ** 2).sum(axis=(1, 4))
        bcoll[s:e] = (np.abs(b) ** 2).sum(axis=(1, 4))
    np.save(tf, coll)
    np.save(bf, bcoll)
    return coll, bcoll


def geom_effective_dist(pos_train, blocked_train, q_xy, q_blocked):
    """Geometry-gated effective distance from query to every train point."""
    d = np.hypot(pos_train[:, 0] - q_xy[0], pos_train[:, 1] - q_xy[1])
    los_q = q_blocked <= 0
    los_i = blocked_train <= 0
    return d * (1.0 + 1.0 * (los_i != los_q) + 0.5 * np.abs(blocked_train - q_blocked) / 80.0)


def knn_power_marginals(coll, pos_train, blocked_train, q_xy, q_blocked, k=32,
                        exclude_idx=None):
    """Geometry-gated IDW average of neighbor collapsed powers -> marginals.

    Returns (beam(16,8), delay(192), total, nn1_beam, nn1_delay, nn1_total,
             nn_dist, weights_entropy).
    """
    eff = geom_effective_dist(pos_train, blocked_train, q_xy, q_blocked)
    if exclude_idx is not None:
        eff = eff.copy()
        eff[exclude_idx] = np.inf
    idx = np.argpartition(eff, k)[:k]
    idx = idx[np.argsort(eff[idx])]
    w = 1.0 / np.maximum(eff[idx], 0.25) ** 2
    w = w / w.sum()
    nbr = coll[idx]  # (k,16,8,192)
    avg = (w[:, None, None, None] * nbr).sum(axis=0)
    beam = avg.sum(-1)
    delay = avg.sum((0, 1))
    total = float(avg.sum())
    nn1 = nbr[0]
    ent = float(-(w * np.log(w + 1e-12)).sum())
    return beam, delay, total, nn1.sum(-1), nn1.sum((0, 1)), float(nn1.sum()), float(eff[idx[0]]), ent


def sharp_marginals(coll_norm_beam, coll_norm_delay, T_train, pos_train,
                    blocked_train, q_xy, q_blocked, k=4, p=3, exclude_idx=None):
    """Sharp local weighted average of neighbor NORMALIZED marginal shapes.

    Returns (beam_shape(128,), delay_shape(192,), agree_beam, agree_delay,
             std_logT, eff_dists(k,)).
    """
    eff = geom_effective_dist(pos_train, blocked_train, q_xy, q_blocked).copy()
    eff[T_train <= 0] = np.inf
    if exclude_idx is not None:
        eff[exclude_idx] = np.inf
    idx = np.argpartition(eff, k)[:k]
    idx = idx[np.argsort(eff[idx])]
    w = 1.0 / np.maximum(eff[idx], 0.3) ** p
    w = w / w.sum()
    nb = coll_norm_beam[idx]
    nd = coll_norm_delay[idx]
    beam = (w[:, None] * nb).sum(0)
    delay = (w[:, None] * nd).sum(0)
    # neighbor agreement: mean pairwise cosine among the k shapes
    def mean_pair_cos(V):
        Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + TINY)
        G = Vn @ Vn.T
        iu = np.triu_indices(len(V), 1)
        return float(G[iu].mean()) if len(iu[0]) else 1.0
    agree_b = mean_pair_cos(nb)
    agree_d = mean_pair_cos(nd)
    std_logT = float(np.std(np.log(T_train[idx] + TINY)))
    return beam, delay, agree_b, agree_d, std_logT, eff[idx]


def build_features(pos, blocked, coll_train, pos_train, blocked_train,
                   base_beam, base_delay, base_total, exclude_self=False, k=32):
    """Feature matrix + marginal feature blocks for a set of query points.

    pos: (P,3) query positions. base_*: base-prediction marginals for the
    queries (log-space handled here). coll_train/pos_train/blocked_train: the
    2000-point training bank used for neighbor features.
    """
    P = len(pos)
    scal = []
    agree = []
    knn_beam = np.empty((P, 128), dtype=np.float32)
    knn_delay = np.empty((P, 192), dtype=np.float32)
    nn1_beam = np.empty((P, 128), dtype=np.float32)
    nn1_delay = np.empty((P, 192), dtype=np.float32)
    sharp_beam = np.empty((P, 128), dtype=np.float32)
    sharp_delay = np.empty((P, 192), dtype=np.float32)
    knn_total = np.empty((P,), dtype=np.float32)
    nn1_total = np.empty((P,), dtype=np.float32)

    T_train = coll_train.sum((1, 2, 3))
    Bn_train = (coll_train.sum(-1).reshape(len(coll_train), 128)
                / np.maximum(T_train[:, None], TINY))
    Dn_train = coll_train.sum((1, 2)) / np.maximum(T_train[:, None], TINY)

    d_bs = np.linalg.norm(pos - BS_POS, axis=1)
    az = np.arctan2(pos[:, 1] - BS_POS[1], pos[:, 0] - BS_POS[0])

    for i in range(P):
        excl = None
        if exclude_self:
            d_all = np.hypot(pos_train[:, 0] - pos[i, 0], pos_train[:, 1] - pos[i, 1])
            excl = d_all < 1e-6
        b, dl, t, b1, d1, t1, nnd, ent = knn_power_marginals(
            coll_train, pos_train, blocked_train, pos[i, :2], blocked[i], k=k,
            exclude_idx=excl)
        knn_beam[i] = np.log(b + TINY).ravel()
        knn_delay[i] = np.log(dl + TINY)
        knn_total[i] = np.log(t + TINY)
        nn1_beam[i] = np.log(b1 + TINY).ravel()
        nn1_delay[i] = np.log(d1 + TINY)
        nn1_total[i] = np.log(t1 + TINY)
        sb, sd, ab, ad, sT, effk = sharp_marginals(
            Bn_train, Dn_train, T_train, pos_train, blocked_train,
            pos[i, :2], blocked[i], k=4, p=3, exclude_idx=excl)
        sharp_beam[i] = np.log(sb + TINY)
        sharp_delay[i] = np.log(sd + TINY)
        agree.append([ab, ad, sT, min(effk[0], 20.0) / 5.0,
                      min(effk[-1], 30.0) / 10.0])
        scal.append([nnd, ent])

    scal = np.asarray(scal, dtype=np.float32)
    agree = np.asarray(agree, dtype=np.float32)

    x = (pos[:, 0] - 150.0) / 100.0
    y = (pos[:, 1] + 75.0) / 100.0
    ff = [x, y]
    for f in (1.0, 2.0, 4.0, 8.0, 16.0):
        for v in (x, y):
            ff.append(np.sin(np.pi * f * v))
            ff.append(np.cos(np.pi * f * v))
    ff = np.stack(ff, axis=1).astype(np.float32)

    blocked_f = np.stack([
        blocked / 80.0,
        np.log1p(np.maximum(blocked, 0.0)),
        (blocked <= 0).astype(float),
        ((blocked >= 1) & (blocked <= 25)).astype(float),
        ((blocked > 25) & (blocked <= 100)).astype(float),
        (blocked > 100).astype(float),
    ], axis=1).astype(np.float32)

    geo = np.stack([
        d_bs / 100.0, np.log(d_bs) - 4.0, np.sin(az), np.cos(az),
        scal[:, 0] / 5.0, scal[:, 1],
    ], axis=1).astype(np.float32)

    lb_beam = np.log(base_beam + TINY).reshape(P, 128).astype(np.float32)
    lb_delay = np.log(base_delay + TINY).reshape(P, 192).astype(np.float32)
    lb_total = np.log(base_total + TINY).reshape(P, 1).astype(np.float32)

    feats = {
        "ff": ff, "blocked": blocked_f, "geo": geo, "agree": agree,
        "base_beam": lb_beam, "base_delay": lb_delay, "base_total": lb_total,
        "knn_beam": knn_beam, "knn_delay": knn_delay,
        "knn_total": knn_total.reshape(P, 1),
        "nn1_beam": nn1_beam, "nn1_delay": nn1_delay,
        "nn1_total": nn1_total.reshape(P, 1),
        "sharp_beam": sharp_beam, "sharp_delay": sharp_delay,
    }
    return feats


def assemble_matrix(feats, keys=None):
    if keys is None:
        keys = ["ff", "blocked", "geo", "agree",
                "base_beam", "base_delay", "base_total",
                "knn_beam", "knn_delay", "knn_total", "nn1_beam", "nn1_delay",
                "nn1_total", "sharp_beam", "sharp_delay"]
    return np.concatenate([feats[k] for k in keys], axis=1)
