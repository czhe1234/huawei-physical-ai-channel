#!/usr/bin/env python3
"""Generate v91: confidence-gated component phase calibration on v87.

v87 already preserves the hard coherent structure and improves easy power
marginals.  This script keeps all AD-domain powers unchanged and applies only a
small phase rotation to selected dominant PHV angle-delay components.

The phase table is estimated from train leave-one-out v36 AD predictions:
for each dominant component we measure angle(sum(conj(pred) * truth)).  A group
is usable only when its phase residual is concentrated across many samples.
Heavy/unstable groups are left untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from generate_v87_easy_field_projection import (
    BS_POS,
    MH,
    MP,
    MV,
    N_ANT,
    S_SC,
    ad_transform,
    beta_subcarriers,
    inverse_ad,
    summarize,
)
from project_paths import PATHS
from powerfield_common import ASSETS, TINY, regime_of


REGIMES = ("clear", "light", "medium", "heavy")


def parse_regime_values(text: str, default: float) -> dict[str, float]:
    out = {r: float(default) for r in REGIMES}
    if not text:
        return out
    for item in text.split(","):
        if not item:
            continue
        k, v = item.split(":")
        k = k.strip()
        if k not in out:
            raise ValueError(f"unknown regime {k!r}")
        out[k] = float(v)
    return out


def top_ids_from_power(power_hvt: np.ndarray, k: int) -> np.ndarray:
    flat = power_hvt.reshape(power_hvt.shape[0], -1)
    kk = min(int(k), flat.shape[1])
    ids = np.argpartition(flat, -kk, axis=1)[:, -kk:]
    vals = np.take_along_axis(flat, ids, axis=1)
    order = np.argsort(-vals, axis=1)
    return np.take_along_axis(ids, order, axis=1)


def component_hvt(flat_id: int) -> tuple[int, int, int]:
    h = int(flat_id // (MV * S_SC))
    rem = int(flat_id % (MV * S_SC))
    v = int(rem // S_SC)
    t = int(rem % S_SC)
    return h, v, t


def add_group(groups: dict[tuple, list], key: tuple, z: complex, den: float) -> None:
    g = groups.setdefault(key, [0.0 + 0.0j, 0.0, 0.0, 0])
    g[0] += z
    g[1] += abs(z)
    g[2] += den
    g[3] += 1


def usable_group(
    z: complex,
    absz: float,
    den: float,
    count: int,
    min_count: int,
    min_concentration: float,
    min_coherence: float,
    max_abs_phase_deg: float,
) -> dict[str, float] | None:
    if count < min_count:
        return None
    concentration = abs(z) / max(absz, TINY)
    coherence = abs(z) / max(den, TINY)
    phase_deg = float(np.angle(z, deg=True))
    if concentration < min_concentration or coherence < min_coherence:
        return None
    if abs(phase_deg) > max_abs_phase_deg:
        return None
    return {
        "phase_deg": phase_deg,
        "concentration": float(concentration),
        "coherence": float(coherence),
        "count": int(count),
    }


def build_phase_table(args: argparse.Namespace, blocked_train: np.ndarray) -> dict[str, dict[str, float]]:
    cached = Path(args.phase_table)
    if cached.exists() and not args.rebuild_phase_table:
        return json.loads(cached.read_text(encoding="utf-8"))

    base = np.load(args.train_base_ad, mmap_mode="r")
    residual = np.load(args.train_residual_ad, mmap_mode="r")
    if base.shape != residual.shape:
        raise ValueError(f"train AD shape mismatch: {base.shape} vs {residual.shape}")
    if base.shape[1:] != (MP, MH, MV, N_ANT, S_SC):
        raise ValueError(f"unexpected train AD shape: {base.shape}")

    regimes = regime_of(blocked_train)
    groups: dict[tuple, list] = {}
    n = base.shape[0]
    for start in range(0, n, args.train_batch):
        end = min(start + args.train_batch, n)
        b = np.asarray(base[start:end], dtype=np.complex64)
        truth = b + np.asarray(residual[start:end], dtype=np.complex64)
        power = np.sum(np.abs(b) ** 2, axis=(1, 4), dtype=np.float64)
        ids = top_ids_from_power(power, args.train_top_components)
        for row in range(end - start):
            i = start + row
            reg = str(regimes[i])
            for rank, flat_id in enumerate(ids[row]):
                h, v, t = component_hvt(int(flat_id))
                bv = b[row, :, h, v, :, t].reshape(-1)
                tv = truth[row, :, h, v, :, t].reshape(-1)
                z = np.vdot(bv, tv)
                den = float(np.sqrt(np.sum(np.abs(bv) ** 2) * np.sum(np.abs(tv) ** 2)) + TINY)
                if den <= 0:
                    continue
                rank_bin = "top4" if rank < 4 else "top16"
                add_group(groups, ("reg", reg), z, den)
                add_group(groups, ("reg_rank", reg, rank_bin), z, den)
                add_group(groups, ("reg_hv", reg, h // args.hv_h_bin, v // args.hv_v_bin), z, den)
                add_group(groups, ("reg_delay4", reg, int(t // 4)), z, den)
        print(f"phase table train {end}/{n}", flush=True)

    table: dict[str, dict[str, float]] = {}
    all_groups = []
    for key, (z, absz, den, count) in groups.items():
        entry = usable_group(
            z,
            absz,
            den,
            count,
            args.min_group_count,
            args.min_concentration,
            args.min_coherence,
            args.max_table_phase_deg,
        )
        raw = {
            "phase_deg": float(np.angle(z, deg=True)),
            "concentration": float(abs(z) / max(absz, TINY)),
            "coherence": float(abs(z) / max(den, TINY)),
            "count": int(count),
            "usable": entry is not None,
        }
        all_groups.append({"key": "|".join(map(str, key)), **raw})
        if entry is not None:
            table["|".join(map(str, key))] = entry

    all_groups.sort(key=lambda x: (x["usable"], x["concentration"], x["coherence"]), reverse=True)
    payload = {
        "method": "v91_phase_table_from_v36_loo",
        "usable_table": table,
        "top_groups": all_groups[:300],
        "filters": {
            "min_group_count": int(args.min_group_count),
            "min_concentration": float(args.min_concentration),
            "min_coherence": float(args.min_coherence),
            "max_table_phase_deg": float(args.max_table_phase_deg),
        },
    }
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return table


def phase_lookup(table: dict[str, dict[str, float]], reg: str, h: int, v: int, t: int, rank: int, args: argparse.Namespace) -> tuple[float, str | None, dict[str, float] | None]:
    rank_bin = "top4" if rank < 4 else "top16"
    candidates = [
        ("reg_hv", reg, h // args.hv_h_bin, v // args.hv_v_bin),
        ("reg_rank", reg, rank_bin),
        ("reg_delay4", reg, int(t // 4)),
        ("reg", reg),
    ]
    for key in candidates:
        skey = "|".join(map(str, key))
        info = table.get(skey)
        if info is not None:
            return float(info["phase_deg"]), skey, info
    return 0.0, None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=ASSETS + "/cache_v87/v87_Round1_Test_Channel.npy")
    parser.add_argument("--output", default=str(PATHS.cache_dir(91) / "v91_Round1_Test_Channel.npy"))
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--geometry-cache", default=ASSETS + "/height_geometry_cache.npz")
    parser.add_argument("--report", default=ASSETS + "/v91_phase_calibrated_components_generate.json")
    parser.add_argument("--phase-table", default=ASSETS + "/v91_phase_calibration_table.json")
    parser.add_argument("--train-base-ad", default=ASSETS + "/train_v36_loo_base_derot_ad.npy")
    parser.add_argument("--train-residual-ad", default=ASSETS + "/train_v36_loo_residual_derot_ad.npy")
    parser.add_argument("--rebuild-phase-table", action="store_true")
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--gamma", type=float, default=0.10)
    parser.add_argument("--top-components", type=int, default=16)
    parser.add_argument("--train-top-components", type=int, default=16)
    parser.add_argument("--top4-mult", type=float, default=1.0)
    parser.add_argument("--top16-mult", type=float, default=0.45)
    parser.add_argument("--phase-scale", default="clear:0.50,light:0.50,medium:0.25,heavy:0.00")
    parser.add_argument("--max-apply-phase-deg", type=float, default=4.0)
    parser.add_argument("--min-group-count", type=int, default=180)
    parser.add_argument("--min-concentration", type=float, default=0.70)
    parser.add_argument("--min-coherence", type=float, default=0.60)
    parser.add_argument("--max-table-phase-deg", type=float, default=20.0)
    parser.add_argument("--hv-h-bin", type=int, default=4)
    parser.add_argument("--hv-v-bin", type=int, default=2)
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--train-batch", type=int, default=25)
    args = parser.parse_args()

    data_dir = Path(args.data)
    pos_test = np.load(data_dir / "Round1_Test_Pos.npy")
    geo = np.load(args.geometry_cache)
    blocked_train = geo["train_blocked"].astype(np.float64)
    blocked_test = geo["test_blocked"].astype(np.float64)
    regimes = regime_of(blocked_test)
    phase_scale = parse_regime_values(args.phase_scale, 0.0)
    beta_s = beta_subcarriers(args.beta, args.gamma)

    table = build_phase_table(args, blocked_train)
    print(f"loaded {len(table)} usable phase groups", flush=True)

    base = np.load(args.base, mmap_mode="r")
    if base.shape != (500, 256, 4, 192) or base.dtype != np.complex64:
        raise ValueError(f"unexpected base shape/dtype {base.shape} {base.dtype}")

    d_test = np.linalg.norm(pos_test - BS_POS[None, :], axis=1)
    out = np.lib.format.open_memmap(args.output, mode="w+", dtype=np.complex64, shape=base.shape)

    stats = {
        "corrected_count": [],
        "corrected_energy_frac": [],
        "abs_phase_deg": [],
        "signed_phase_deg": [],
        "power_ratio": [],
    }
    by_regime = {r: {"count": 0, "corrected": 0, "energy": 0.0, "corrected_energy": 0.0} for r in REGIMES}
    used_groups: dict[str, int] = {}

    for start in range(0, 500, args.batch):
        end = min(start + args.batch, 500)
        n = end - start
        h0 = np.asarray(base[start:end], dtype=np.complex64)
        derot = h0 * np.exp(1j * d_test[start:end, None, None, None] * beta_s).astype(np.complex64)
        ad0 = ad_transform(derot)
        component_power = np.sum(np.abs(ad0) ** 2, axis=(1, 4), dtype=np.float64)
        ids = top_ids_from_power(component_power, args.top_components)
        phase_hvt = np.zeros((n, MH, MV, S_SC), dtype=np.float32)

        for row in range(n):
            reg = str(regimes[start + row])
            total_energy = float(np.sum(component_power[row]))
            corr_energy = 0.0
            corr_count = 0
            for rank, flat_id in enumerate(ids[row]):
                h, v, t = component_hvt(int(flat_id))
                raw_phase, skey, _info = phase_lookup(table, reg, h, v, t, rank, args)
                if skey is None:
                    continue
                mult = args.top4_mult if rank < 4 else args.top16_mult
                applied = raw_phase * phase_scale[reg] * mult
                applied = float(np.clip(applied, -args.max_apply_phase_deg, args.max_apply_phase_deg))
                if abs(applied) < 1e-6:
                    continue
                phase_hvt[row, h, v, t] = np.deg2rad(applied)
                used_groups[skey] = used_groups.get(skey, 0) + 1
                corr_count += 1
                corr_energy += float(component_power[row, h, v, t])
                stats["abs_phase_deg"].append(abs(applied))
                stats["signed_phase_deg"].append(applied)
            by_regime[reg]["count"] += 1
            by_regime[reg]["corrected"] += corr_count
            by_regime[reg]["energy"] += total_energy
            by_regime[reg]["corrected_energy"] += corr_energy
            stats["corrected_count"].append(corr_count)
            stats["corrected_energy_frac"].append(corr_energy / max(total_energy, TINY))

        factor = np.exp(1j * phase_hvt).astype(np.complex64)
        ad1 = ad0 * factor[:, None, :, :, None, :]
        h1 = inverse_ad(ad1).reshape(n, 256, 4, 192)
        h1 = h1 * np.exp(-1j * d_test[start:end, None, None, None] * beta_s).astype(np.complex64)
        if not (np.isfinite(h1.real).all() and np.isfinite(h1.imag).all()):
            raise FloatingPointError(f"non-finite block {start}:{end}")
        out[start:end] = h1.astype(np.complex64)
        out.flush()

        p0 = np.sum(np.abs(h0) ** 2, axis=(1, 2, 3), dtype=np.float64)
        p1 = np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64)
        stats["power_ratio"].extend((p1 / np.maximum(p0, TINY)).tolist())
        print(f"generated {end}/500", flush=True)

    payload = {
        "method": "v91_phase_calibrated_components",
        "base": args.base,
        "operation": "phase-only PHV AD top-component correction on v87; powers preserved in AD domain",
        "phase_table": args.phase_table,
        "phase_scale": phase_scale,
        "top_components": int(args.top_components),
        "top4_mult": float(args.top4_mult),
        "top16_mult": float(args.top16_mult),
        "max_apply_phase_deg": float(args.max_apply_phase_deg),
        "table_group_count": int(len(table)),
        "overall": {k: summarize(v) for k, v in stats.items()},
        "by_regime": {
            r: {
                **vals,
                "corrected_energy_frac": vals["corrected_energy"] / max(vals["energy"], TINY),
            }
            for r, vals in by_regime.items()
        },
        "used_groups_top": sorted(
            ({"key": k, "uses": v} for k, v in used_groups.items()),
            key=lambda x: x["uses"],
            reverse=True,
        )[:80],
        "output": args.output,
    }
    Path(args.report).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
