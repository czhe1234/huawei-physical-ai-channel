#!/usr/bin/env python3
"""Phase diagnostics for v98 train LOO bad points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from generate_v87_easy_field_projection import BS_POS, MH, MV, S_SC, ad_transform, beta_subcarriers, summarize
from project_paths import PATHS
from powerfield_common import ASSETS, TINY, geom_effective_dist, regime_of


def top_component_vectors(ad: np.ndarray, flat_ids: np.ndarray) -> np.ndarray:
    h = flat_ids // (MV * S_SC)
    rem = flat_ids % (MV * S_SC)
    v = rem // S_SC
    t = rem % S_SC
    # ad: MP,MH,MV,N,S -> K,MP,N
    vec = ad[:, h, v, :, t]  # MP,K,N
    return np.moveaxis(vec, 1, 0).reshape(len(flat_ids), -1)


def circular_abs(angle: np.ndarray) -> np.ndarray:
    return np.abs(np.angle(np.exp(1j * angle)))


def query_phase_metrics(qid: int, nids: np.ndarray, channel, pos, beta_s, top_k: int) -> dict[str, float]:
    ids = np.concatenate([[qid], nids])
    h = np.asarray(channel[ids], dtype=np.complex64)
    d = np.linalg.norm(pos[ids] - BS_POS[None, :], axis=1)
    h = h * np.exp(1j * d[:, None, None, None] * beta_s).astype(np.complex64)
    ad = ad_transform(h)
    q = ad[0]
    neigh = ad[1:]
    q_coll = (np.abs(q) ** 2).sum(axis=(0, 3))  # MH,MV,S
    flat = q_coll.reshape(-1)
    top_ids = np.argpartition(flat, -top_k)[-top_k:]
    top_ids = top_ids[np.argsort(flat[top_ids])[::-1]]
    weights_c = flat[top_ids].astype(np.float64)
    weights_c /= max(float(weights_c.sum()), TINY)
    qv = top_component_vectors(q, top_ids)
    qnorm = np.sqrt(np.sum(np.abs(qv) ** 2, axis=1)).astype(np.float64)

    all_angles = []
    all_coh = []
    all_w = []
    for j in range(len(nids)):
        nv = top_component_vectors(neigh[j], top_ids)
        nnorm = np.sqrt(np.sum(np.abs(nv) ** 2, axis=1)).astype(np.float64)
        ip = np.sum(np.conj(qv) * nv, axis=1)
        coh = np.abs(ip) / np.maximum(qnorm * nnorm, 1e-30)
        ang = np.angle(ip)
        all_angles.append(ang)
        all_coh.append(coh)
        all_w.append(weights_c)
    angles = np.stack(all_angles, axis=0)
    coh = np.stack(all_coh, axis=0)
    wc = np.stack(all_w, axis=0)
    # Distance weights are handled outside by neighbor order; here report NN and K-average.
    nn_abs_phase = float(np.sum(weights_c * circular_abs(angles[0])))
    nn_coh = float(np.sum(weights_c * coh[0]))
    k_abs_phase = float(np.sum(wc * circular_abs(angles)) / np.sum(wc))
    k_coh = float(np.sum(wc * coh) / np.sum(wc))
    resultant = np.abs(np.sum(wc * np.exp(1j * angles)) / np.sum(wc))
    phase_dispersion = float(1.0 - resultant)

    # Common phase over all top components: low residual means a single scalar can align neighbor.
    nv0 = top_component_vectors(neigh[0], top_ids)
    common_ip = np.sum(np.conj(qv) * nv0)
    common_phase = float(abs(np.angle(common_ip)))
    common_coh = float(np.abs(common_ip) / np.maximum(np.sqrt(np.sum(np.abs(qv) ** 2) * np.sum(np.abs(nv0) ** 2)), 1e-30))

    return {
        "nn_abs_phase_rad": nn_abs_phase,
        "nn_component_coh": nn_coh,
        "k_abs_phase_rad": k_abs_phase,
        "k_component_coh": k_coh,
        "k_phase_dispersion": phase_dispersion,
        "nn_common_phase_rad": common_phase,
        "nn_common_coh": common_coh,
        "top1_fraction": float(weights_c[0]),
        "top4_fraction": float(weights_c[:4].sum()),
    }


def group_summary(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: summarize(np.array([r[k] for r in rows], dtype=np.float64)) for k in keys}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(PATHS.round1))
    ap.add_argument("--geometry-cache", default=ASSETS + "/height_geometry_cache.npz")
    ap.add_argument("--metrics", default=str(PATHS.analysis / "visualizations/v98_bad_points/data/train_loo_metrics.npz"))
    ap.add_argument("--out", default=ASSETS + "/train_bad_phase_diagnostic.json")
    ap.add_argument("--top-count", type=int, default=80)
    ap.add_argument("--good-count", type=int, default=80)
    ap.add_argument("--neighbor-k", type=int, default=8)
    ap.add_argument("--top-components", type=int, default=16)
    ap.add_argument("--beta", type=float, default=140.3)
    ap.add_argument("--gamma", type=float, default=0.10)
    args = ap.parse_args()

    data_dir = Path(args.data)
    pos = np.load(data_dir / "Round1_Train_Pos.npy")
    channel = np.load(data_dir / "Round1_Train_Channel.npy", mmap_mode="r")
    geo = np.load(args.geometry_cache)
    blocked = geo["train_blocked"].astype(np.float64)
    regimes = regime_of(blocked)
    score = np.load(args.metrics)["score"]
    bad_ids = np.argsort(score)[: args.top_count]
    good_ids = np.argsort(score)[-args.good_count:][::-1]
    beta_s = beta_subcarriers(args.beta, args.gamma)

    def neighbors(qid: int) -> np.ndarray:
        eff = geom_effective_dist(pos, blocked, pos[qid, :2], blocked[qid]).astype(np.float64)
        eff[qid] = np.inf
        ids = np.argpartition(eff, args.neighbor_k)[: args.neighbor_k]
        return ids[np.argsort(eff[ids])]

    out_rows = {}
    for name, ids in [("bad", bad_ids), ("good", good_ids)]:
        rows = []
        for c, qid in enumerate(ids, 1):
            nids = neighbors(int(qid))
            m = query_phase_metrics(int(qid), nids, channel, pos, beta_s, int(args.top_components))
            m["id"] = int(qid)
            m["score"] = float(score[qid])
            m["blocked"] = float(blocked[qid])
            m["regime"] = str(regimes[qid])
            rows.append(m)
            if c % 20 == 0:
                print(f"{name} {c}/{len(ids)}", flush=True)
        out_rows[name] = rows

    by_regime = {}
    for name, rows in out_rows.items():
        by_regime[name] = {}
        for reg in ("clear", "light", "medium", "heavy"):
            rr = [r for r in rows if r["regime"] == reg]
            by_regime[name][reg] = group_summary([{k: v for k, v in r.items() if isinstance(v, float)} for r in rr])

    payload = {
        "definition": {
            "derotation": "H * exp(j * ||rx-bs|| * (beta + gamma * subcarrier_axis)) before AD transform",
            "component_phase": "phase of complex vector inner product between query true top AD component and neighbor same component",
            "nn_abs_phase_rad": "power-weighted absolute residual phase for nearest neighbor over query top components",
            "k_phase_dispersion": "1 - circular resultant over K neighbors and top components; larger means phase field is less coherent",
        },
        "bad_ids": bad_ids.tolist(),
        "good_ids": good_ids.tolist(),
        "summary": {
            "bad": group_summary([{k: v for k, v in r.items() if isinstance(v, float)} for r in out_rows["bad"]]),
            "good": group_summary([{k: v for k, v in r.items() if isinstance(v, float)} for r in out_rows["good"]]),
        },
        "by_regime": by_regime,
        "rows": out_rows,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
