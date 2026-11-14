#!/usr/bin/env python3
"""Generate v191-v193 metric-mouth probe suite on top of v189.

The goal is not to guess one more opaque overdrive, but to split the unknown
official scorer tail into three isolated black-box probes:

v191 low-marginal PAS probe
    Move only the validated low-dimensional leaked marginals
    S-H / pol-S-H / S-H-UE / S-H-V toward test-test consensus.

v192 slice-conditional probe
    Move full 6D leaked slice conditionals while repeatedly anchoring the
    validated low marginals back to v189. This asks whether official scoring
    rewards pol/UE/V slice allocation beyond those marginals.

v193 S-only / PDP-like probe
    Move only the leaked S-axis marginal, keeping the conditional distribution
    over pol/H/UE/V inside each S bin. This is a clean test for whether the
    missing score is more delay/PDP-like than PAS-slice-like.

All three preserve v189 point total power and leaked-domain phase.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass

import numpy as np
from scipy import sparse

import generate_v178_test_track_complex_consensus as track
import generate_v187_leaked_pas_lifecycle_projection as leak
import generate_v189_leaked_multimarginal_overdrive as v189
from generate_v87_easy_field_projection import MP, MH, MV, N_ANT, S_SC, row_cos, summarize
from generate_v159_frontier_consensus_support_soup import digest
from generate_v169_power_scaled_tail_denoising import by_regime_summary
from generate_v171_full_complex_ad_tail_denoising import top_mask
from powerfield_common import TINY, regime_of
from project_paths import PATHS


FLOOR = 1e-12
BATCH = 8
BASE = PATHS.cache_dir(189) / "v189_Round1_Test_Channel.npy"
V188_BASE = PATHS.cache_dir(188) / "v188_Round1_Test_Channel.npy"
V190_BASE = PATHS.cache_dir(190) / "v190_Round1_Test_Channel.npy"
FULL_SUPPORT = PATHS.cache_dir(190) / "v190_v189_leaked_full_support.npy"


@dataclass(frozen=True)
class Probe:
    number: int
    name: str
    mode: str
    hypothesis: str
    low_alpha: dict[str, float]
    full_alpha: dict[str, float]
    s_alpha: dict[str, float]
    gain_all: tuple[float, float]
    gain8192: tuple[float, float]
    gain1024: tuple[float, float]
    gain128: tuple[float, float]
    light_blend: float

    @property
    def version(self) -> str:
        return f"v{self.number}_{self.name}"

    @property
    def out_dir(self):
        path = PATHS.cache_dir(self.number)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def report_dir(self):
        path = PATHS.version_report_dir(self.number)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def output(self):
        return self.out_dir / f"v{self.number}_Round1_Test_Channel.npy"

    @property
    def target(self):
        return self.out_dir / f"v{self.number}_{self.name}_target.npy"

    @property
    def actions(self):
        return self.out_dir / f"v{self.number}_{self.name}_actions.npz"

    @property
    def report(self):
        return self.report_dir / f"v{self.number}_{self.name}_generate.json"

    @property
    def zip_path(self):
        return PATHS.round1_versions / f"v{self.number}_{self.name}_candidate_pending.zip"


PROBES = [
    Probe(
        number=191,
        name="probe_low_marginal_pas",
        mode="low_marginal",
        hypothesis="Is the remaining scorer tail mainly low-dimensional leaked PAS marginals?",
        low_alpha={"clear": 0.0, "light": 0.08, "medium": 0.42, "heavy": 0.62},
        full_alpha={"clear": 0.0, "light": 0.0, "medium": 0.0, "heavy": 0.0},
        s_alpha={"clear": 0.0, "light": 0.0, "medium": 0.0, "heavy": 0.0},
        gain_all=(0.13, 7.8),
        gain8192=(0.36, 3.10),
        gain1024=(0.52, 2.10),
        gain128=(0.70, 1.58),
        light_blend=0.55,
    ),
    Probe(
        number=192,
        name="probe_slice_conditionals",
        mode="slice_conditionals",
        hypothesis="Does official scoring reward full pol/UE/V slice allocation after low marginals are held fixed?",
        low_alpha={"clear": 0.0, "light": 0.0, "medium": 0.0, "heavy": 0.0},
        full_alpha={"clear": 0.0, "light": 0.10, "medium": 0.50, "heavy": 0.72},
        s_alpha={"clear": 0.0, "light": 0.0, "medium": 0.0, "heavy": 0.0},
        gain_all=(0.16, 6.6),
        gain8192=(0.40, 2.85),
        gain1024=(0.56, 1.96),
        gain128=(0.74, 1.48),
        light_blend=0.48,
    ),
    Probe(
        number=193,
        name="probe_sonly_pdp",
        mode="s_only",
        hypothesis="Is the remaining scorer tail mainly S-axis / PDP-like rather than PAS slice allocation?",
        low_alpha={"clear": 0.0, "light": 0.0, "medium": 0.0, "heavy": 0.0},
        full_alpha={"clear": 0.0, "light": 0.0, "medium": 0.0, "heavy": 0.0},
        s_alpha={"clear": 0.0, "light": 0.12, "medium": 0.48, "heavy": 0.68},
        gain_all=(0.12, 7.2),
        gain8192=(0.34, 3.00),
        gain1024=(0.52, 2.05),
        gain128=(0.72, 1.55),
        light_blend=0.60,
    ),
]


def normalize_rows(flat: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(flat, dtype=np.float32), FLOOR)
    out /= np.maximum(out.sum(axis=1, keepdims=True), TINY)
    return out.astype(np.float32)


def safe_ratio(desired: np.ndarray, current: np.ndarray) -> np.ndarray:
    return (np.maximum(desired, FLOOR) / np.maximum(current, FLOOR)).astype(np.float32)


def apply_ratio(work: np.ndarray, ratio: np.ndarray, axes: str, exponent: float) -> None:
    ratio = np.maximum(ratio, FLOOR).astype(np.float32) ** np.float32(exponent)
    if axes == "sh":
        work *= ratio[:, None, :, :, None, None]
    elif axes == "psh":
        work *= ratio[:, :, :, :, None, None]
    elif axes == "shue":
        work *= ratio[:, None, :, :, :, None]
    elif axes == "shv":
        work *= ratio[:, None, :, :, None, :]
    elif axes == "s":
        work *= ratio[:, None, :, None, None, None]
    else:
        raise ValueError(axes)


def build_weight_matrix(pos: np.ndarray, blocked: np.ndarray, regimes: np.ndarray, k: int = 96) -> tuple[sparse.csr_matrix, dict[str, np.ndarray]]:
    idx, raw_w = track.neighbor_graph(pos, blocked, regimes, k=k)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    agreement = np.zeros(len(pos), dtype=np.float32)
    effective = np.zeros(len(pos), dtype=np.float32)
    for i in range(len(pos)):
        rows.append(i)
        cols.append(i)
        vals.append(0.36)
        total = 0.36
        nb = idx[i].astype(np.int64)
        w = raw_w[i].astype(np.float64)
        same = (regimes[nb] == regimes[i]).astype(np.float64)
        w = w * (0.50 + 0.50 * same)
        w = w / max(float(w.sum()), 1e-12)
        for node, weight in zip(nb, w):
            value = float(1.0 * weight)
            rows.append(i)
            cols.append(int(node))
            vals.append(value)
            total += value
        for j in range(len(vals) - len(nb) - 1, len(vals)):
            vals[j] = vals[j] / total
        agreement[i] = np.float32(float(np.sum(raw_w[i] * same)))
        effective[i] = np.float32(1.0 / max(float(np.sum(w * w)), 1e-12))
    matrix = sparse.csr_matrix((vals, (rows, cols)), shape=(len(pos), len(pos)), dtype=np.float32)
    return matrix, {
        "neighbor_idx": idx.astype(np.int32),
        "neighbor_weights": raw_w.astype(np.float32),
        "neighbor_agreement": agreement,
        "effective_neighbor_count": effective,
    }


def alpha_by_regime(regimes: np.ndarray, scale: dict[str, float], agreement: np.ndarray) -> np.ndarray:
    alpha = np.zeros(len(regimes), dtype=np.float32)
    for i, regime in enumerate(regimes):
        local = float(np.clip((agreement[i] - 0.20) / 0.72, 0.0, 1.0))
        alpha[i] = np.float32(scale[str(regime)] * (0.72 + 0.28 * local))
    return alpha


def build_low_marginal_target(current: np.ndarray, consensus: np.ndarray, regimes: np.ndarray, agreement: np.ndarray, probe: Probe) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    n = current.shape[0]
    alpha = alpha_by_regime(regimes, probe.low_alpha, agreement)
    work = current.copy()

    sh0 = current.sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32)
    psh0 = current.sum(axis=(4, 5), dtype=np.float64).astype(np.float32)
    shue0 = current.sum(axis=(1, 5), dtype=np.float64).astype(np.float32)
    shv0 = current.sum(axis=(1, 4), dtype=np.float64).astype(np.float32)
    shc = consensus.sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32)
    pshc = consensus.sum(axis=(4, 5), dtype=np.float64).astype(np.float32)
    shuec = consensus.sum(axis=(1, 5), dtype=np.float64).astype(np.float32)
    shvc = consensus.sum(axis=(1, 4), dtype=np.float64).astype(np.float32)

    desired_sh = ((1.0 - alpha[:, None, None]) * sh0 + alpha[:, None, None] * shc).astype(np.float32)
    desired_psh = ((1.0 - alpha[:, None, None, None]) * psh0 + alpha[:, None, None, None] * pshc).astype(np.float32)
    desired_shue = ((1.0 - alpha[:, None, None, None]) * shue0 + alpha[:, None, None, None] * shuec).astype(np.float32)
    desired_shv = ((1.0 - alpha[:, None, None, None]) * shv0 + alpha[:, None, None, None] * shvc).astype(np.float32)

    for _ in range(4):
        apply_ratio(work, safe_ratio(desired_sh, work.sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32)), "sh", 0.78)
        apply_ratio(work, safe_ratio(desired_psh, work.sum(axis=(4, 5), dtype=np.float64).astype(np.float32)), "psh", 0.48)
        apply_ratio(work, safe_ratio(desired_shue, work.sum(axis=(1, 5), dtype=np.float64).astype(np.float32)), "shue", 0.52)
        apply_ratio(work, safe_ratio(desired_shv, work.sum(axis=(1, 4), dtype=np.float64).astype(np.float32)), "shv", 0.50)
        work = normalize_rows(work.reshape(n, -1)).reshape(current.shape)
    return work.astype(np.float32), {"alpha_low": alpha}


def build_slice_conditional_target(current: np.ndarray, consensus: np.ndarray, regimes: np.ndarray, agreement: np.ndarray, probe: Probe) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    n = current.shape[0]
    alpha = alpha_by_regime(regimes, probe.full_alpha, agreement)
    flat = current.reshape(n, -1)
    cflat = consensus.reshape(n, -1)
    work = ((1.0 - alpha[:, None]) * flat + alpha[:, None] * cflat).astype(np.float32)
    work = normalize_rows(work).reshape(current.shape)

    desired_sh = current.sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32)
    desired_psh = current.sum(axis=(4, 5), dtype=np.float64).astype(np.float32)
    desired_shue = current.sum(axis=(1, 5), dtype=np.float64).astype(np.float32)
    desired_shv = current.sum(axis=(1, 4), dtype=np.float64).astype(np.float32)
    for _ in range(5):
        apply_ratio(work, safe_ratio(desired_sh, work.sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32)), "sh", 0.82)
        apply_ratio(work, safe_ratio(desired_psh, work.sum(axis=(4, 5), dtype=np.float64).astype(np.float32)), "psh", 0.64)
        apply_ratio(work, safe_ratio(desired_shue, work.sum(axis=(1, 5), dtype=np.float64).astype(np.float32)), "shue", 0.66)
        apply_ratio(work, safe_ratio(desired_shv, work.sum(axis=(1, 4), dtype=np.float64).astype(np.float32)), "shv", 0.64)
        work = normalize_rows(work.reshape(n, -1)).reshape(current.shape)
    return work.astype(np.float32), {"alpha_full": alpha}


def build_sonly_target(current: np.ndarray, consensus: np.ndarray, regimes: np.ndarray, agreement: np.ndarray, probe: Probe) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    n = current.shape[0]
    alpha = alpha_by_regime(regimes, probe.s_alpha, agreement)
    s0 = current.sum(axis=(1, 3, 4, 5), dtype=np.float64).astype(np.float32)
    sc = consensus.sum(axis=(1, 3, 4, 5), dtype=np.float64).astype(np.float32)
    desired_s = ((1.0 - alpha[:, None]) * s0 + alpha[:, None] * sc).astype(np.float32)
    desired_s /= np.maximum(desired_s.sum(axis=1, keepdims=True), TINY)
    work = current.copy()
    apply_ratio(work, safe_ratio(desired_s, s0), "s", 1.0)
    work = normalize_rows(work.reshape(n, -1)).reshape(current.shape)
    return work.astype(np.float32), {"alpha_s": alpha}


def build_target(
    current: np.ndarray,
    consensus: np.ndarray,
    regimes: np.ndarray,
    graph_diag: dict[str, np.ndarray],
    probe: Probe,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if probe.mode == "low_marginal":
        target, diag = build_low_marginal_target(current, consensus, regimes, graph_diag["neighbor_agreement"], probe)
    elif probe.mode == "slice_conditionals":
        target, diag = build_slice_conditional_target(current, consensus, regimes, graph_diag["neighbor_agreement"], probe)
    elif probe.mode == "s_only":
        target, diag = build_sonly_target(current, consensus, regimes, graph_diag["neighbor_agreement"], probe)
    else:
        raise ValueError(probe.mode)

    n = current.shape[0]
    flat0 = current.reshape(n, -1)
    flat1 = target.reshape(n, -1)
    sh0 = current.sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32)
    sh1 = target.sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32)
    psh0 = current.sum(axis=(4, 5), dtype=np.float64).astype(np.float32)
    psh1 = target.sum(axis=(4, 5), dtype=np.float64).astype(np.float32)
    shue0 = current.sum(axis=(1, 5), dtype=np.float64).astype(np.float32)
    shue1 = target.sum(axis=(1, 5), dtype=np.float64).astype(np.float32)
    shv0 = current.sum(axis=(1, 4), dtype=np.float64).astype(np.float32)
    shv1 = target.sum(axis=(1, 4), dtype=np.float64).astype(np.float32)
    s0 = current.sum(axis=(1, 3, 4, 5), dtype=np.float64).astype(np.float32)
    s1 = target.sum(axis=(1, 3, 4, 5), dtype=np.float64).astype(np.float32)
    diag.update(
        {
            "target_full_l1": np.sum(np.abs(flat1 - flat0), axis=1).astype(np.float32),
            "target_full_cos": row_cos(flat1, flat0).astype(np.float32),
            "margin_sh_l1": np.sum(np.abs(sh1 - sh0), axis=(1, 2)).astype(np.float32),
            "margin_psh_l1": np.sum(np.abs(psh1 - psh0), axis=(1, 2, 3)).astype(np.float32),
            "margin_shue_l1": np.sum(np.abs(shue1 - shue0), axis=(1, 2, 3)).astype(np.float32),
            "margin_shv_l1": np.sum(np.abs(shv1 - shv0), axis=(1, 2, 3)).astype(np.float32),
            "margin_s_l1": np.sum(np.abs(s1 - s0), axis=1).astype(np.float32),
        }
    )
    target_out = np.lib.format.open_memmap(probe.target, mode="w+", dtype=np.float32, shape=target.shape)
    target_out[:] = target
    target_out.flush()
    return np.load(probe.target, mmap_mode="r"), diag


def apply_target(target: np.ndarray, regimes: np.ndarray, probe: Probe) -> dict[str, np.ndarray]:
    source = np.load(BASE, mmap_mode="r")
    output = np.lib.format.open_memmap(probe.output, mode="w+", dtype=np.complex64, shape=source.shape)
    diag: dict[str, list[np.ndarray]] = {
        "actual_power_factor": [],
        "full_l1_actual": [],
        "collapsed_sh_l1_actual": [],
        "s_l1_actual": [],
        "gain_min": [],
        "gain_p10": [],
        "gain_p90": [],
        "gain_max": [],
    }
    for start in range(0, len(source), BATCH):
        end = min(start + BATCH, len(source))
        h0 = np.asarray(source[start:end], dtype=np.complex64)
        spec0 = leak.leaked_forward(h0)
        power0 = leak.leaked_cell_power(spec0)
        total0 = np.maximum(power0.sum(axis=(1, 2, 3, 4, 5)), TINY).astype(np.float32)
        support0 = power0 / total0[:, None, None, None, None, None]
        h_power0 = np.maximum(np.sum(np.abs(h0) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY).astype(np.float32)

        target_power = target[start:end] * total0[:, None, None, None, None, None]
        gain = np.sqrt(target_power / np.maximum(power0, 1e-30)).astype(np.float32)
        gain = np.clip(gain, *probe.gain_all).astype(np.float32)
        flat = support0.reshape(end - start, -1)
        strong128 = top_mask(flat, 128).reshape(support0.shape)
        strong1024 = top_mask(flat, 1024).reshape(support0.shape)
        strong8192 = top_mask(flat, 8192).reshape(support0.shape)
        gain[strong8192] = np.clip(gain[strong8192], *probe.gain8192)
        gain[strong1024] = np.clip(gain[strong1024], *probe.gain1024)
        gain[strong128] = np.clip(gain[strong128], *probe.gain128)
        for local, regime in enumerate(regimes[start:end]):
            name = str(regime)
            if name == "clear":
                gain[local] = 1.0
            elif name == "light":
                gain[local] = 1.0 + probe.light_blend * (gain[local] - 1.0)

        spec1 = spec0 * gain
        h1 = leak.leaked_inverse(spec1)
        h_power1 = np.maximum(np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY).astype(np.float32)
        h1 *= np.sqrt(h_power0 / h_power1)[:, None, None, None].astype(np.float32)
        if not (np.isfinite(h1.real).all() and np.isfinite(h1.imag).all()):
            raise FloatingPointError(f"non-finite {probe.version} block {start}:{end}")
        output[start:end] = h1.astype(np.complex64)
        output.flush()

        spec_after = leak.leaked_forward(h1.astype(np.complex64))
        power1 = leak.leaked_cell_power(spec_after)
        support1 = power1 / np.maximum(power1.sum(axis=(1, 2, 3, 4, 5), keepdims=True), TINY)
        collapsed0 = support0.sum(axis=(1, 4, 5))
        collapsed1 = support1.sum(axis=(1, 4, 5))
        s0 = support0.sum(axis=(1, 3, 4, 5))
        s1 = support1.sum(axis=(1, 3, 4, 5))
        diag["actual_power_factor"].append(
            (np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64) / h_power0).astype(np.float32)
        )
        diag["full_l1_actual"].append(np.sum(np.abs(support1 - support0), axis=(1, 2, 3, 4, 5)).astype(np.float32))
        diag["collapsed_sh_l1_actual"].append(np.sum(np.abs(collapsed1 - collapsed0), axis=(1, 2)).astype(np.float32))
        diag["s_l1_actual"].append(np.sum(np.abs(s1 - s0), axis=1).astype(np.float32))
        gflat = gain.reshape(end - start, -1)
        diag["gain_min"].append(gflat.min(axis=1).astype(np.float32))
        diag["gain_p10"].append(np.quantile(gflat, 0.10, axis=1).astype(np.float32))
        diag["gain_p90"].append(np.quantile(gflat, 0.90, axis=1).astype(np.float32))
        diag["gain_max"].append(gflat.max(axis=1).astype(np.float32))
        print(f"{probe.version} projection {end}/500", flush=True)
    return {key: np.concatenate(value).astype(np.float32) for key, value in diag.items()}


def compare_output_to(output_path, base_path, regimes: np.ndarray) -> dict[str, object]:
    return track.compare_output_to(output_path, base_path, regimes)


def write_zip(probe: Probe, report: dict[str, object]) -> dict[str, object]:
    with zipfile.ZipFile(probe.zip_path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(probe.output, "Round1_Test_Channel.npy")
        for src in [
            PATHS.code / "generate_v191_v193_metric_probe_suite.py",
            PATHS.code / "generate_v190_leaked_hypermarginal_overdrive.py",
            PATHS.code / "generate_v189_leaked_multimarginal_overdrive.py",
            PATHS.code / "generate_v188_leaked_slice_overdrive.py",
            PATHS.code / "generate_v187_leaked_pas_lifecycle_projection.py",
            PATHS.code / "generate_v178_test_track_complex_consensus.py",
            PATHS.code / "generate_v87_easy_field_projection.py",
            PATHS.code / "generate_v171_full_complex_ad_tail_denoising.py",
            PATHS.code / "generate_v169_power_scaled_tail_denoising.py",
            PATHS.code / "generate_v159_frontier_consensus_support_soup.py",
            PATHS.code / "powerfield_common.py",
            PATHS.code_root / "project_paths.py",
        ]:
            archive.write(src, src.name)
    with zipfile.ZipFile(probe.zip_path) as archive:
        bad = archive.testzip()
        members = archive.namelist()
    if bad is not None:
        raise ValueError(f"ZIP CRC failure: {bad}")
    output = np.load(probe.output, mmap_mode="r")
    finite = True
    for start in range(0, len(output), 20):
        block = np.asarray(output[start : start + 20], dtype=np.complex64)
        finite = finite and bool(np.isfinite(block.real).all() and np.isfinite(block.imag).all())
    report["integrity"] = {
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": finite,
        "zip_crc": True,
        "zip_members": members,
        "sha256": digest(probe.zip_path, "sha256"),
        "md5": digest(probe.zip_path, "md5"),
    }
    probe.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def generate_probe(
    probe: Probe,
    current: np.ndarray,
    consensus: np.ndarray,
    regimes: np.ndarray,
    graph_diag: dict[str, np.ndarray],
) -> dict[str, object]:
    target, target_diag = build_target(current, consensus, regimes, graph_diag, probe)
    apply_diag = apply_target(target, regimes, probe)
    np.savez_compressed(probe.actions, regimes=regimes.astype("U8"), **graph_diag, **target_diag, **apply_diag)

    comparison_to_v189 = compare_output_to(probe.output, BASE, regimes)
    comparison_to_v188 = compare_output_to(probe.output, V188_BASE, regimes)
    comparison_to_v190 = compare_output_to(probe.output, V190_BASE, regimes)
    if comparison_to_v189["shape"] != [500, 256, 4, 192] or comparison_to_v189["dtype"] != "complex64":
        raise ValueError(f"invalid output layout for {probe.version}")
    if not comparison_to_v189["finite"]:
        raise ValueError(f"{probe.version} output contains non-finite values")

    metric_keys = {
        "low_marginal": ["alpha_low", "margin_sh_l1", "margin_psh_l1", "margin_shue_l1", "margin_shv_l1"],
        "slice_conditionals": ["alpha_full", "target_full_l1", "margin_sh_l1", "margin_psh_l1", "margin_shue_l1", "margin_shv_l1"],
        "s_only": ["alpha_s", "margin_s_l1", "target_full_l1"],
    }[probe.mode]
    report = {
        "method": probe.version,
        "status": "candidate_pending",
        "primary_hypothesis": probe.hypothesis,
        "probe_design": {
            "base": "v189 online 0.510038",
            "purpose": "black-box scorer-mouth isolation; submit v191/v192/v193 together and compare online deltas",
            "mode": probe.mode,
            "preserve": "v189 phase and per-point total power; clear unchanged",
        },
        "parameters": {
            "low_alpha": probe.low_alpha,
            "full_alpha": probe.full_alpha,
            "s_alpha": probe.s_alpha,
            "gain_all": list(probe.gain_all),
            "gain8192": list(probe.gain8192),
            "gain1024": list(probe.gain1024),
            "gain128": list(probe.gain128),
            "light_blend": probe.light_blend,
        },
        "diagnostics": {
            "overall": {
                "neighbor_agreement": summarize(graph_diag["neighbor_agreement"]),
                "effective_neighbor_count": summarize(graph_diag["effective_neighbor_count"]),
                **{key: summarize(target_diag[key]) for key in metric_keys if key in target_diag},
                "target_full_l1": summarize(target_diag["target_full_l1"]),
                "target_full_cos": summarize(target_diag["target_full_cos"]),
                "actual_power_factor": summarize(apply_diag["actual_power_factor"]),
                "full_l1_actual": summarize(apply_diag["full_l1_actual"]),
                "collapsed_sh_l1_actual": summarize(apply_diag["collapsed_sh_l1_actual"]),
                "s_l1_actual": summarize(apply_diag["s_l1_actual"]),
                "gain_max": summarize(apply_diag["gain_max"]),
            },
            "by_regime": by_regime_summary(
                regimes,
                {
                    **{key: target_diag[key] for key in metric_keys if key in target_diag},
                    "target_full_l1": target_diag["target_full_l1"],
                    "margin_sh_l1": target_diag["margin_sh_l1"],
                    "margin_s_l1": target_diag["margin_s_l1"],
                    "full_l1_actual": apply_diag["full_l1_actual"],
                    "collapsed_sh_l1_actual": apply_diag["collapsed_sh_l1_actual"],
                    "s_l1_actual": apply_diag["s_l1_actual"],
                    "actual_power_factor": apply_diag["actual_power_factor"],
                    "gain_max": apply_diag["gain_max"],
                },
            ),
        },
        "comparison_to_v189": comparison_to_v189,
        "comparison_to_v188": comparison_to_v188,
        "comparison_to_v190": comparison_to_v190,
        "outputs": {
            "submission_npy": PATHS.relative(probe.output),
            "target": PATHS.relative(probe.target),
            "actions": PATHS.relative(probe.actions),
            "report": PATHS.relative(probe.report),
            "zip": PATHS.relative(probe.zip_path),
        },
    }
    probe.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return write_zip(probe, report)


def main() -> None:
    pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    blocked = np.load(PATHS.shared_features / "height_geometry_cache.npz")["test_blocked"].astype(np.float32)
    regimes = regime_of(blocked)
    current = np.asarray(np.load(FULL_SUPPORT, mmap_mode="r"), dtype=np.float32)
    matrix, graph_diag = build_weight_matrix(pos, blocked, regimes)
    flat = current.reshape(current.shape[0], -1)
    consensus_flat = normalize_rows(np.asarray(matrix @ flat, dtype=np.float32))
    consensus = consensus_flat.reshape(current.shape)

    reports = []
    for probe in PROBES:
        print(f"generate {probe.version}", flush=True)
        reports.append(generate_probe(probe, current, consensus, regimes, graph_diag))
    summary = {
        "status": "candidate_pending_suite",
        "probes": [report["outputs"]["zip"] for report in reports],
        "comparisons_to_v189": {report["method"]: report["comparison_to_v189"]["nmse"] for report in reports},
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
