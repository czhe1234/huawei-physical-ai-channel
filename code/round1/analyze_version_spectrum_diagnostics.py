#!/usr/bin/env python3
"""Diagnose what changed across submitted channel versions.

This script does not need test ground truth. It compares output-level
power/PDP/PAS/AD-component statistics across key online versions and regimes.
The goal is to identify whether previous gains mostly attacked hard coherent
phase reconstruction while easier power-spectrum calibration remains underused.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from project_paths import PATHS


VERSIONS = {
    "v9": {
        "path": PATHS.artifacts / "legacy_inputs/v9.npy",
        "online": 0.458443,
        "note": "physical local linear k24 g0.10",
    },
    "v10_s1": {
        "path": PATHS.version_cache / "legacy_versions/v10_s1_Round1_Test_Channel.npy",
        "online": 0.460859,
        "note": "external s1 reference",
    },
    "v36": {
        "path": PATHS.artifacts / "legacy_inputs/v36.npy",
        "online": 0.468074,
        "note": "regime geometry heavy=4",
    },
    "v48": {
        "path": PATHS.version_cache / "legacy_versions/v48_Round1_Test_Channel.npy",
        "online": 0.472438,
        "note": "v36 + s1-v9 residual",
    },
    "v55": {
        "path": PATHS.version_cache / "legacy_versions/v55_Round1_Test_Channel.npy",
        "online": 0.475165,
        "note": "s1 residual + ADShift + top4 component gate",
    },
    "v83": {
        "path": PATHS.cache_dir(83) / "v83_Round1_Test_Channel.npy",
        "online": 0.475244,
        "note": "v55 + Sionna path delay confidence gate",
    },
    "v85": {
        "path": PATHS.cache_dir(85) / "v85_Round1_Test_Channel.npy",
        "online": None,
        "note": "v83 + Geo-ADTS sparse residual candidate",
    },
}


def regime_labels(blocked: np.ndarray) -> np.ndarray:
    labels = np.empty(blocked.shape, dtype=object)
    labels[blocked <= 0] = "clear"
    labels[(blocked > 0) & (blocked <= 25)] = "light"
    labels[(blocked > 25) & (blocked <= 100)] = "medium"
    labels[blocked > 100] = "heavy"
    return labels


def safe_entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.maximum(p, 0.0)
    s = p.sum(axis=axis, keepdims=True)
    q = p / np.maximum(s, 1e-30)
    return -np.sum(q * np.log(np.maximum(q, 1e-30)), axis=axis)


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = np.sum(a * b, axis=1)
    den = np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1))
    return num / np.maximum(den, 1e-30)


def summarize_vector(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {"count": 0}
    return {
        "count": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p05": float(np.quantile(x, 0.05)),
        "p10": float(np.quantile(x, 0.10)),
        "p25": float(np.quantile(x, 0.25)),
        "median": float(np.quantile(x, 0.50)),
        "p75": float(np.quantile(x, 0.75)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def compute_spectra(path: Path, chunk: int = 10) -> dict[str, np.ndarray]:
    h = np.load(path, mmap_mode="r")
    n = h.shape[0]
    power = np.zeros(n, dtype=np.float64)
    pdp = np.zeros((n, 192), dtype=np.float64)
    pas = np.zeros((n, 16, 8), dtype=np.float64)
    top_fracs = {1: np.zeros(n), 4: np.zeros(n), 16: np.zeros(n), 64: np.zeros(n)}
    ad_entropy = np.zeros(n, dtype=np.float64)
    pdp_entropy = np.zeros(n, dtype=np.float64)
    pas_entropy = np.zeros(n, dtype=np.float64)
    delay_centroid = np.zeros(n, dtype=np.float64)
    delay_peak = np.zeros(n, dtype=np.int16)
    angle_peak_h = np.zeros(n, dtype=np.int16)
    angle_peak_v = np.zeros(n, dtype=np.int16)

    delay_idx = np.arange(192, dtype=np.float64)

    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        x = np.asarray(h[start:end], dtype=np.complex64)
        b = end - start
        power[start:end] = np.sum(np.abs(x) ** 2, axis=(1, 2, 3))

        cir = np.fft.ifft(x, axis=-1)
        pdp_b = np.sum(np.abs(cir) ** 2, axis=(1, 2)).real.astype(np.float64)
        pdp[start:end] = pdp_b
        pdp_sum = np.maximum(pdp_b.sum(axis=1), 1e-30)
        pdp_entropy[start:end] = safe_entropy(pdp_b, axis=1)
        delay_centroid[start:end] = (pdp_b * delay_idx[None, :]).sum(axis=1) / pdp_sum
        delay_peak[start:end] = np.argmax(pdp_b, axis=1).astype(np.int16)

        phv = x.reshape(b, 2, 16, 8, 4, 192)
        ang = np.fft.fft2(phv, axes=(2, 3))
        pas_b = np.sum(np.abs(ang) ** 2, axis=(1, 4, 5)).real.astype(np.float64)
        pas[start:end] = pas_b
        pas_entropy[start:end] = safe_entropy(pas_b.reshape(b, -1), axis=1)
        ap = np.argmax(pas_b.reshape(b, -1), axis=1)
        angle_peak_h[start:end] = (ap // 8).astype(np.int16)
        angle_peak_v[start:end] = (ap % 8).astype(np.int16)

        ad = np.fft.fft2(cir.reshape(b, 2, 16, 8, 4, 192), axes=(2, 3))
        ad_power = np.sum(np.abs(ad) ** 2, axis=(1, 4)).real.reshape(b, -1).astype(np.float64)
        total = np.maximum(ad_power.sum(axis=1), 1e-30)
        part = np.sort(ad_power, axis=1)[:, ::-1]
        for k, out in top_fracs.items():
            top_fracs[k][start:end] = part[:, :k].sum(axis=1) / total
        ad_entropy[start:end] = safe_entropy(ad_power, axis=1)

    return {
        "power": power,
        "pdp": pdp,
        "pas": pas.reshape(n, -1),
        "top1_frac": top_fracs[1],
        "top4_frac": top_fracs[4],
        "top16_frac": top_fracs[16],
        "top64_frac": top_fracs[64],
        "ad_entropy": ad_entropy,
        "pdp_entropy": pdp_entropy,
        "pas_entropy": pas_entropy,
        "delay_centroid": delay_centroid,
        "delay_peak": delay_peak,
        "angle_peak_h": angle_peak_h,
        "angle_peak_v": angle_peak_v,
    }


def summarize_version(name: str, data: dict[str, np.ndarray], labels: np.ndarray) -> dict:
    keys = [
        "power",
        "top1_frac",
        "top4_frac",
        "top16_frac",
        "top64_frac",
        "ad_entropy",
        "pdp_entropy",
        "pas_entropy",
        "delay_centroid",
    ]
    out = {"overall": {}, "by_regime": {}}
    for k in keys:
        out["overall"][k] = summarize_vector(data[k])
    for reg in ["clear", "light", "medium", "heavy"]:
        mask = labels == reg
        out["by_regime"][reg] = {k: summarize_vector(data[k][mask]) for k in keys}
    return out


def compare_versions(base: dict[str, np.ndarray], other: dict[str, np.ndarray], labels: np.ndarray) -> dict:
    out = {"overall": {}, "by_regime": {}}
    power_ratio = other["power"] / np.maximum(base["power"], 1e-30)
    pdp_cos = cosine_rows(other["pdp"], base["pdp"])
    pas_cos = cosine_rows(other["pas"], base["pas"])
    delay_shift = other["delay_centroid"] - base["delay_centroid"]
    top4_delta = other["top4_frac"] - base["top4_frac"]
    fields = {
        "power_ratio": power_ratio,
        "pdp_cos_to_base": pdp_cos,
        "pas_cos_to_base": pas_cos,
        "delay_centroid_shift": delay_shift,
        "top4_frac_delta": top4_delta,
    }
    out["overall"] = {k: summarize_vector(v) for k, v in fields.items()}
    for reg in ["clear", "light", "medium", "heavy"]:
        mask = labels == reg
        out["by_regime"][reg] = {k: summarize_vector(v[mask]) for k, v in fields.items()}
    return out


def train_true_reference(chunk: int = 10) -> dict[str, dict]:
    path = ROOT / "datasets/round1/Round1_Train_Channel.npy"
    data = compute_spectra(path, chunk=chunk)
    blocked = np.load(ROOT / "artifacts/shared_features/height_geometry_cache.npz")["train_blocked"]
    labels = regime_labels(blocked)
    return summarize_version("train_true", data, labels)


def write_markdown(out: dict, path: Path) -> None:
    lines = []
    lines.append("# Version Spectrum Diagnostics")
    lines.append("")
    lines.append("目的：分析历史提分主要改变了哪些输出结构，判断是否存在未充分优化的功率谱/PAS/PDP低垂果实。")
    lines.append("")
    lines.append("## Online Scores")
    lines.append("")
    lines.append("| version | online | note |")
    lines.append("| --- | ---: | --- |")
    for name, meta in out["versions_meta"].items():
        online = meta["online"]
        lines.append(f"| {name} | {online if online is not None else 'pending'} | {meta['note']} |")

    lines.append("")
    lines.append("## Key Overall Statistics")
    lines.append("")
    lines.append("| version | power median | power p10 | top4 AD frac median | PDP entropy median | PAS entropy median | delay centroid median |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name in out["versions_meta"]:
        s = out["summary"][name]["overall"]
        lines.append(
            f"| {name} | {s['power']['median']:.4e} | {s['power']['p10']:.4e} | "
            f"{s['top4_frac']['median']:.4f} | {s['pdp_entropy']['median']:.3f} | "
            f"{s['pas_entropy']['median']:.3f} | {s['delay_centroid']['median']:.2f} |"
        )

    lines.append("")
    lines.append("## Changes Relative To v36")
    lines.append("")
    lines.append("| version | online delta vs v36 | power ratio median | PDP cos median | PAS cos median | top4 frac delta median | delay shift median |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    base_online = out["versions_meta"]["v36"]["online"]
    for name in out["versions_meta"]:
        if name == "v36":
            continue
        c = out["compare_to_v36"][name]["overall"]
        online = out["versions_meta"][name]["online"]
        od = "" if online is None else f"{online - base_online:+.6f}"
        lines.append(
            f"| {name} | {od} | {c['power_ratio']['median']:.3f} | "
            f"{c['pdp_cos_to_base']['median']:.5f} | {c['pas_cos_to_base']['median']:.5f} | "
            f"{c['top4_frac_delta']['median']:+.4f} | {c['delay_centroid_shift']['median']:+.3f} |"
        )

    lines.append("")
    lines.append("## Heavy-Regime Changes Relative To v36")
    lines.append("")
    lines.append("| version | power ratio median | PDP cos median | PAS cos median | top4 frac delta median |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for name in out["versions_meta"]:
        if name == "v36":
            continue
        c = out["compare_to_v36"][name]["by_regime"]["heavy"]
        lines.append(
            f"| {name} | {c['power_ratio']['median']:.3f} | "
            f"{c['pdp_cos_to_base']['median']:.5f} | {c['pas_cos_to_base']['median']:.5f} | "
            f"{c['top4_frac_delta']['median']:+.4f} |"
        )

    lines.append("")
    lines.append("## Initial Reading")
    lines.append("")
    lines.append("1. 线上从 v36 到 v55/v83 的提升，未伴随 PAS/PDP 形状的大幅整体变化；PDP/PAS cosine 对 v36 仍通常很高。这说明后期很多收益来自少量强 AD component 的校准，而不是全局谱形重建。")
    lines.append("2. 如果 top4 AD 能量占比和功率分布仍与训练真值或高分版本存在系统差异，说明还有一条更容易的路线：先优化功率谱/能量尺度，再处理相位。")
    lines.append("3. heavy 区域是最值得单独看的区域，因为 v36 之前的大部分线上提升都来自 heavy/regime 外推；如果 heavy 的功率比、top component 集中度或 delay centroid 仍有明显漂移，应优先在该区域做 power/PDP/PAS 模型。")
    lines.append("")
    lines.append("完整数值见 `spectrum_diagnostics.json`。")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    out_dir = ROOT / "artifacts/spectrum_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    blocked = np.load(ROOT / "artifacts/shared_features/height_geometry_cache.npz")["test_blocked"]
    labels = regime_labels(blocked)
    label_counts = {r: int(np.sum(labels == r)) for r in ["clear", "light", "medium", "heavy"]}

    spectra: dict[str, dict[str, np.ndarray]] = {}
    summary = {}
    versions_meta = {}
    for name, meta in VERSIONS.items():
        if not meta["path"].exists():
            print(f"skip missing {name}: {meta['path']}")
            continue
        print(f"compute {name}: {meta['path']}")
        data = compute_spectra(meta["path"], chunk=10)
        spectra[name] = data
        summary[name] = summarize_version(name, data, labels)
        versions_meta[name] = {"online": meta["online"], "note": meta["note"], "path": str(meta["path"])}

    compare_to_v36 = {}
    compare_to_v55 = {}
    for name, data in spectra.items():
        if name != "v36" and "v36" in spectra:
            compare_to_v36[name] = compare_versions(spectra["v36"], data, labels)
        if name != "v55" and "v55" in spectra:
            compare_to_v55[name] = compare_versions(spectra["v55"], data, labels)

    print("compute train true reference")
    train_ref = train_true_reference(chunk=10)

    result = {
        "label_counts": label_counts,
        "versions_meta": versions_meta,
        "summary": summary,
        "compare_to_v36": compare_to_v36,
        "compare_to_v55": compare_to_v55,
        "train_true_reference": train_ref,
    }
    json_path = out_dir / "spectrum_diagnostics.json"
    md_path = out_dir / "spectrum_diagnostics.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    write_markdown(result, md_path)
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
