#!/usr/bin/env python3
"""Reverse-infer scorer aggregation mouths from historical online versions.

This is an analysis-only experiment.  Hidden test truth is unavailable, so the
script cannot compute official PAS/PDP.  Instead it asks a narrower question:
which mathematically possible power-spectrum aggregation mouths make historical
online versions look most ordered around the current best v189?

For each retained online version with a cached test output, the script computes
cosine similarity to v189 under exhaustive physically meaningful aggregations:

* leaked traceback PAS mouth: vector includes leaked S and H axes;
* leaked S-only / PDP-like mouth: vector includes leaked S;
* standard spatial PAS mouth: vector includes H and V;
* delay-domain PDP mouth: vector includes delay D.

Every non-anchor axis is assigned one of three roles:

* ``sum``: aggregate power before cosine;
* ``vector``: include this axis in the cosine vector;
* ``slice``: keep it outside the vector and average slice-wise cosine.

The resulting feature table is correlated with online scores over all available
versions and over the controlled v187-v194 scorer-mouth probe window.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from project_paths import PATHS


MP = 2
MH = 16
MV = 8
N_ANT = 4
S_SC = 192
FLOOR = 1e-30


@dataclass(frozen=True)
class Mouth:
    name: str
    domain: str
    labels: tuple[str, ...]
    vector: tuple[str, ...]
    summed: tuple[str, ...]


def parse_online_scores() -> dict[int, dict[str, str | float | Path]]:
    pattern = re.compile(r"^v(\d+)_.*_online_([0-9.]+)\.zip$")
    out: dict[int, dict[str, str | float | Path]] = {}
    for path in sorted(PATHS.round1_versions.glob("v*_online_*.zip")):
        match = pattern.match(path.name)
        if not match:
            continue
        version = int(match.group(1))
        score = float(match.group(2))
        out[version] = {"version": version, "online": score, "zip": path}
    return out


def parse_score_csv(path: Path) -> dict[int, dict[str, str | float | Path]]:
    out: dict[int, dict[str, str | float | Path]] = {}
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                version = int(row["version"])
                score = float(row["score"])
            except (KeyError, TypeError, ValueError):
                continue
            out[version] = {
                "version": version,
                "online": score,
                "zip": Path(row.get("file") or f"v{version}_score_csv"),
            }
    return out


def cached_output(version: int) -> Path | None:
    path = PATHS.cache_dir(version) / f"v{version}_Round1_Test_Channel.npy"
    return path if path.exists() else None


def make_mouths() -> list[Mouth]:
    mouths: list[Mouth] = []

    def add_family(domain: str, labels: tuple[str, ...], anchor: tuple[str, ...]) -> None:
        optional = [label for label in labels if label not in anchor]
        for roles in itertools.product(("slice", "sum", "vector"), repeat=len(optional)):
            vector = list(anchor)
            summed = []
            name_parts = [domain, "vec-" + "".join(anchor)]
            for label, role in zip(optional, roles):
                if role == "vector":
                    vector.append(label)
                elif role == "sum":
                    summed.append(label)
                role_name = {"slice": "slc", "sum": "sum", "vector": "vec"}[role]
                name_parts.append(f"{label}:{role_name}")
            mouths.append(
                Mouth(
                    name="__".join(name_parts),
                    domain=domain,
                    labels=labels,
                    vector=tuple(vector),
                    summed=tuple(summed),
                )
            )

    add_family("leaked_sh", ("P", "S", "H", "U", "V"), ("S", "H"))
    add_family("leaked_s", ("P", "S", "H", "U", "V"), ("S",))
    add_family("standard_hv", ("P", "H", "V", "U", "S"), ("H", "V"))
    add_family("delay_d", ("P", "H", "V", "U", "D"), ("D",))
    return mouths


def leaked_power(h: np.ndarray) -> np.ndarray:
    arr = h.reshape(h.shape[0], MP, MH, MV, N_ANT, S_SC)
    arr = arr.transpose(0, 1, 5, 2, 4, 3)
    spec = np.fft.fft2(arr, axes=(2, 3))
    return np.abs(spec).astype(np.float32) ** 2


def standard_pas_power(h: np.ndarray) -> np.ndarray:
    arr = h.reshape(h.shape[0], MP, MH, MV, N_ANT, S_SC)
    spec = np.fft.fft2(arr, axes=(2, 3))
    return np.abs(spec).astype(np.float32) ** 2


def delay_power(h: np.ndarray) -> np.ndarray:
    cir = np.fft.ifft(h, axis=-1)
    arr = cir.reshape(h.shape[0], MP, MH, MV, N_ANT, S_SC)
    return np.abs(arr).astype(np.float32) ** 2


def domain_power(h: np.ndarray, domain: str) -> np.ndarray:
    if domain == "leaked":
        return leaked_power(h)
    if domain == "standard_hv":
        return standard_pas_power(h)
    if domain == "delay_d":
        return delay_power(h)
    raise ValueError(domain)


def domain_cache_key(domain: str) -> str:
    return "leaked" if domain in {"leaked_sh", "leaked_s"} else domain


def sum_for_mouth(
    pred: np.ndarray,
    ref: np.ndarray,
    labels: tuple[str, ...],
    summed: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    if not summed:
        return pred, ref
    axes = tuple(labels.index(label) + 1 for label in summed)
    return pred.sum(axis=axes, keepdims=True), ref.sum(axis=axes, keepdims=True)


def cosine_mean_presummed(
    pred_work: np.ndarray,
    ref_work: np.ndarray,
    labels: tuple[str, ...],
    vector: tuple[str, ...],
) -> tuple[float, int]:
    vector_axes = tuple(labels.index(label) + 1 for label in vector)
    keep_axes = [axis for axis in range(pred_work.ndim) if axis not in vector_axes]
    order = keep_axes + list(vector_axes)
    p = np.transpose(pred_work, order)
    r = np.transpose(ref_work, order)
    vector_len = int(np.prod([pred_work.shape[axis] for axis in vector_axes]))
    p = p.reshape(-1, vector_len).astype(np.float32, copy=False)
    r = r.reshape(-1, vector_len).astype(np.float32, copy=False)
    numerator = np.einsum("ij,ij->i", p, r, optimize=True)
    denom = np.sqrt(
        np.einsum("ij,ij->i", p, p, optimize=True)
        * np.einsum("ij,ij->i", r, r, optimize=True)
    )
    cos = numerator / np.maximum(denom, FLOOR)
    cos = np.clip(cos, 0.0, 1.0)
    return float(np.sum(cos, dtype=np.float64)), int(cos.size)


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def corr(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
    keep = np.isfinite(x) & np.isfinite(y)
    if int(keep.sum()) < 3:
        return None, None
    xx = x[keep]
    yy = y[keep]
    if np.std(xx) <= 0 or np.std(yy) <= 0:
        return None, None
    pearson = float(np.corrcoef(xx, yy)[0, 1])
    rx = rankdata(xx)
    ry = rankdata(yy)
    spearman = float(np.corrcoef(rx, ry)[0, 1]) if np.std(rx) > 0 and np.std(ry) > 0 else None
    return pearson, spearman


def summarize_correlations(rows: list[dict[str, float | int | str]], mouths: list[Mouth]) -> list[dict[str, float | str | int | None]]:
    online = np.asarray([float(row["online"]) for row in rows], dtype=np.float64)
    version = np.asarray([int(row["version"]) for row in rows], dtype=np.int64)
    controlled = (version >= 187) & (version <= 194)
    broad = version >= 138
    result = []
    for mouth in mouths:
        x = np.asarray([float(row[mouth.name]) for row in rows], dtype=np.float64)
        pearson_all, spearman_all = corr(x, online)
        pearson_ctrl, spearman_ctrl = corr(x[controlled], online[controlled])
        pearson_broad, spearman_broad = corr(x[broad], online[broad])
        result.append(
            {
                "mouth": mouth.name,
                "domain": mouth.domain,
                "vector": "+".join(mouth.vector),
                "summed": "+".join(mouth.summed) if mouth.summed else "-",
                "n_all": int(len(rows)),
                "pearson_all": pearson_all,
                "spearman_all": spearman_all,
                "pearson_ge138": pearson_broad,
                "spearman_ge138": spearman_broad,
                "pearson_v187_v194": pearson_ctrl,
                "spearman_v187_v194": spearman_ctrl,
            }
        )
    return result


def fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key, "")) for key in fieldnames})


def write_markdown(path: Path, payload: dict[str, object]) -> None:
    correlations = payload["correlations"]
    assert isinstance(correlations, list)
    top_all = sorted(
        correlations,
        key=lambda row: -abs(float(row["spearman_all"] or 0.0)),
    )[:20]
    top_ctrl = sorted(
        correlations,
        key=lambda row: -abs(float(row["spearman_v187_v194"] or 0.0)),
    )[:20]
    top_positive_ctrl = sorted(
        [row for row in correlations if row["spearman_v187_v194"] is not None],
        key=lambda row: -float(row["spearman_v187_v194"] or -99.0),
    )[:20]
    rows = payload["rows"]
    assert isinstance(rows, list)
    lines = [
        "# v195 Scorer Mouth Reverse Inference",
        "",
        "分析目标：用历史线上分数反推 PAS/PDP 后半段更像哪种聚合口径。",
        "",
        "重要限制：隐藏测试真值不可见，因此这里不是官方 C1/C2 复算；它衡量的是历史输出在不同候选口径下与当前最高 v189 的接近程度，哪些接近程度最能解释线上分数。",
        "",
        "枚举方式：每个非锚定轴穷举 `sum`、`vector`、`slice` 三种角色。`sum` 表示先聚合功率再算余弦；`vector` 表示该轴进入余弦向量；`slice` 表示该轴单独切片算余弦后平均。",
        "",
        f"版本数：{len(rows)}；候选 mouth 数：{len(correlations)}。",
        "",
        "## Top By All-Version Spearman",
        "",
        "| rank | mouth | domain | vector | summed | spearman_all | pearson_all | spearman_187_194 |",
        "| ---: | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for idx, row in enumerate(top_all, 1):
        lines.append(
            f"| {idx} | `{row['mouth']}` | `{row['domain']}` | `{row['vector']}` | `{row['summed']}` | "
            f"{float(row['spearman_all'] or 0.0):.4f} | {float(row['pearson_all'] or 0.0):.4f} | "
            f"{float(row['spearman_v187_v194'] or 0.0):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Top By Controlled v187-v194 Spearman",
            "",
            "| rank | mouth | domain | vector | summed | spearman_187_194 | pearson_187_194 | spearman_all |",
            "| ---: | --- | --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for idx, row in enumerate(top_ctrl, 1):
        lines.append(
            f"| {idx} | `{row['mouth']}` | `{row['domain']}` | `{row['vector']}` | `{row['summed']}` | "
            f"{float(row['spearman_v187_v194'] or 0.0):.4f} | {float(row['pearson_v187_v194'] or 0.0):.4f} | "
            f"{float(row['spearman_all'] or 0.0):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Positive Controlled Mouths",
            "",
            "| rank | mouth | domain | vector | summed | spearman_187_194 | pearson_187_194 |",
            "| ---: | --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    for idx, row in enumerate(top_positive_ctrl, 1):
        lines.append(
            f"| {idx} | `{row['mouth']}` | `{row['domain']}` | `{row['vector']}` | `{row['summed']}` | "
            f"{float(row['spearman_v187_v194'] or 0.0):.4f} | {float(row['pearson_v187_v194'] or 0.0):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Version Scores Used",
            "",
            "| version | online | zip |",
            "| ---: | ---: | --- |",
        ]
    )
    for row in rows:
        lines.append(f"| v{int(row['version'])} | {float(row['online']):.6f} | `{row['zip']}` |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze(args: argparse.Namespace) -> dict[str, object]:
    scores = parse_online_scores()
    if args.score_csv:
        scores.update(parse_score_csv(Path(args.score_csv)))
    versions = []
    for version, meta in sorted(scores.items()):
        if args.min_version is not None and version < args.min_version:
            continue
        output = cached_output(version)
        if output is None:
            continue
        versions.append((version, float(meta["online"]), output, Path(meta["zip"])))
    if args.max_versions is not None:
        versions = versions[-args.max_versions :]
    if not versions:
        raise RuntimeError("no online versions with cached outputs found")

    ref_path = cached_output(args.reference_version)
    if ref_path is None:
        raise FileNotFoundError(f"missing reference cache for v{args.reference_version}")
    ref = np.load(ref_path, mmap_mode="r")
    mouths = make_mouths()
    if args.domains:
        allowed = {item.strip() for item in args.domains.split(",") if item.strip()}
        mouths = [mouth for mouth in mouths if mouth.domain in allowed]
        if not mouths:
            raise ValueError(f"no mouths left after --domains={args.domains!r}")
    mouths_by_domain: dict[str, list[Mouth]] = {}
    for mouth in mouths:
        mouths_by_domain.setdefault(mouth.domain, []).append(mouth)

    rows: list[dict[str, float | int | str]] = []
    total = len(versions)
    for version_idx, (version, online, output_path, zip_path) in enumerate(versions, 1):
        current = np.load(output_path, mmap_mode="r")
        if current.shape != ref.shape:
            print(f"skip v{version}: shape {current.shape} != ref {ref.shape}", flush=True)
            continue
        accum = {mouth.name: 0.0 for mouth in mouths}
        counts = {mouth.name: 0 for mouth in mouths}
        nmse_num = 0.0
        nmse_den = 0.0
        dot_real = 0.0
        norm_cur = 0.0
        norm_ref = 0.0

        for start in range(0, ref.shape[0], args.batch):
            end = min(start + args.batch, ref.shape[0])
            h_cur = np.asarray(current[start:end], dtype=np.complex64)
            h_ref = np.asarray(ref[start:end], dtype=np.complex64)
            diff = h_cur - h_ref
            nmse_num += float(np.sum(np.abs(diff) ** 2, dtype=np.float64))
            nmse_den += float(np.sum(np.abs(h_ref) ** 2, dtype=np.float64))
            dot_real += float(np.sum((h_cur.real * h_ref.real + h_cur.imag * h_ref.imag), dtype=np.float64))
            norm_cur += float(np.sum(np.abs(h_cur) ** 2, dtype=np.float64))
            norm_ref += float(np.sum(np.abs(h_ref) ** 2, dtype=np.float64))

            domain_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for domain, domain_mouths in mouths_by_domain.items():
                cache_key = domain_cache_key(domain)
                if cache_key not in domain_cache:
                    domain_cache[cache_key] = (
                        domain_power(h_cur, cache_key),
                        domain_power(h_ref, cache_key),
                    )
                p_cur, p_ref = domain_cache[cache_key]
                summed_cache: dict[tuple[str, ...], tuple[np.ndarray, np.ndarray]] = {}
                for mouth in domain_mouths:
                    if mouth.summed not in summed_cache:
                        summed_cache[mouth.summed] = sum_for_mouth(
                            p_cur, p_ref, mouth.labels, mouth.summed
                        )
                    p_sum, r_sum = summed_cache[mouth.summed]
                    total_cos, count = cosine_mean_presummed(
                        p_sum, r_sum, mouth.labels, mouth.vector
                    )
                    accum[mouth.name] += total_cos
                    counts[mouth.name] += count

        row: dict[str, float | int | str] = {
            "version": version,
            "online": online,
            "zip": zip_path.name,
            "output": PATHS.relative(output_path),
            "complex_nmse_to_ref": nmse_num / max(nmse_den, FLOOR),
            "complex_cos_to_ref": dot_real / max(np.sqrt(norm_cur * norm_ref), FLOOR),
            "power_ratio_to_ref": norm_cur / max(norm_ref, FLOOR),
        }
        for mouth in mouths:
            row[mouth.name] = accum[mouth.name] / max(counts[mouth.name], 1)
        rows.append(row)
        print(f"v{version} done ({version_idx}/{total}) online={online:.6f}", flush=True)

    correlations = summarize_correlations(rows, mouths)
    return {
        "method": "analyze_v195_scorer_mouth_reverse_inference",
        "reference_version": args.reference_version,
        "reference_path": PATHS.relative(ref_path),
        "min_version": args.min_version,
        "batch": args.batch,
        "mouth_count": len(mouths),
        "domains": args.domains or "all",
        "version_count": len(rows),
        "rows": rows,
        "correlations": correlations,
        "limitations": [
            "Hidden test truth is unavailable; features are closeness-to-v189 mouths, not official C1/C2.",
            "High correlation can indicate a scorer mouth or simply that v189 is a good reference under that representation.",
            "Controlled v187-v194 correlations are more causal but have small sample size.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-version", type=int, default=189)
    parser.add_argument("--min-version", type=int, default=122)
    parser.add_argument("--max-versions", type=int, default=None)
    parser.add_argument("--batch", type=int, default=25)
    parser.add_argument(
        "--score-csv",
        default=str(PATHS.analysis / "visualizations/version_scores/online_scores_by_version.csv"),
        help="optional version,score CSV used in addition to online zip filenames",
    )
    parser.add_argument(
        "--domains",
        default="",
        help="comma-separated mouth domains to evaluate: leaked_sh,leaked_s,standard_hv,delay_d",
    )
    parser.add_argument(
        "--out-dir",
        default=str(PATHS.analysis / "results/scorer_mouth_reverse_inference_v195"),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = analyze(args)

    json_path = out_dir / "scorer_mouth_reverse_inference.json"
    rows_path = out_dir / "feature_table.csv"
    corr_path = out_dir / "correlations.csv"
    md_path = out_dir / "analysis.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = payload["rows"]
    correlations = payload["correlations"]
    assert isinstance(rows, list)
    assert isinstance(correlations, list)
    if rows:
        row_fields = list(rows[0].keys())
        write_csv(rows_path, rows, row_fields)
    if correlations:
        corr_fields = list(correlations[0].keys())
        write_csv(corr_path, correlations, corr_fields)
    write_markdown(md_path, payload)

    print(json.dumps({
        "json": str(json_path),
        "rows": str(rows_path),
        "correlations": str(corr_path),
        "markdown": str(md_path),
        "version_count": payload["version_count"],
        "mouth_count": payload["mouth_count"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
