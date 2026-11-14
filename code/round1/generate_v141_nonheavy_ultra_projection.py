#!/usr/bin/env python3
"""Generate v141 by isolating v139 ultra projection to non-heavy regimes.

Online evidence shows progressively stronger geodesic support projection from
v133 through v140 improved the score.  v139 was not submitted because its
offline diagnostics collapsed in the heavy regime, while its light and medium
joint-support diagnostics continued to improve.  v141 tests one hypothesis:
v139's strength is useful outside heavy propagation, and its failure came from
applying that strength to heavy points.

Preserved:
    v140 for every clear/heavy point, including phase, power, and allocation.
Changed:
    light/medium points use the existing v139 output and support target.
"""

from __future__ import annotations

import hashlib
import json
import zipfile

import numpy as np

from generate_v87_easy_field_projection import summarize
from powerfield_common import regime_of
from project_paths import PATHS


VERSION = "v141_nonheavy_ultra_projection"
OUT_DIR = PATHS.cache_dir(141)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(141)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v141_Round1_Test_Channel.npy"
TARGET_OUT = OUT_DIR / "v141_nonheavy_ultra_projection_target.npy"
WEIGHTS_OUT = OUT_DIR / "v141_nonheavy_ultra_projection_weights.npy"
SELECTOR_OUT = OUT_DIR / "v141_nonheavy_ultra_selector.npy"
REPORT = REPORT_DIR / "v141_nonheavy_ultra_projection_generate.json"
ZIP_OUT = PATHS.round1_versions / "v141_nonheavy_ultra_projection_candidate_pending.zip"

V140_OUTPUT = PATHS.cache_dir(140) / "v140_Round1_Test_Channel.npy"
V139_OUTPUT = PATHS.cache_dir(139) / "v139_Round1_Test_Channel.npy"
V140_TARGET = PATHS.cache_dir(140) / "v140_balanced_hminus2_overdrive_target.npy"
V139_TARGET = PATHS.cache_dir(139) / "v139_ultra_geodesic_multishift_target.npy"
V140_WEIGHTS = PATHS.cache_dir(140) / "v140_balanced_hminus2_overdrive_weights.npy"


def digest(path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def complex_nmse(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    numerator = 0.0
    denominator = 0.0
    for index in np.flatnonzero(mask):
        aa = np.asarray(a[index], dtype=np.complex64)
        bb = np.asarray(b[index], dtype=np.complex64)
        numerator += float(np.sum(np.abs(aa - bb) ** 2, dtype=np.float64))
        denominator += float(np.sum(np.abs(bb) ** 2, dtype=np.float64))
    return numerator / max(denominator, 1e-30)


def row_complex_cos(a: np.ndarray, b: np.ndarray, batch: int = 10) -> np.ndarray:
    result = np.empty(len(a), dtype=np.float32)
    for start in range(0, len(a), batch):
        end = min(start + batch, len(a))
        aa = np.asarray(a[start:end]).reshape(end - start, -1).astype(np.complex128)
        bb = np.asarray(b[start:end]).reshape(end - start, -1).astype(np.complex128)
        numerator = np.abs(np.sum(aa * np.conj(bb), axis=1))
        denominator = np.sqrt(
            np.sum(np.abs(aa) ** 2, axis=1) * np.sum(np.abs(bb) ** 2, axis=1)
        )
        result[start:end] = (numerator / np.maximum(denominator, 1e-30)).astype(
            np.float32
        )
    return result


def power_ratio(a: np.ndarray, b: np.ndarray, batch: int = 10) -> np.ndarray:
    result = np.empty(len(a), dtype=np.float32)
    for start in range(0, len(a), batch):
        end = min(start + batch, len(a))
        aa = np.asarray(a[start:end])
        bb = np.asarray(b[start:end])
        pa = np.sum(np.abs(aa) ** 2, axis=(1, 2, 3), dtype=np.float64)
        pb = np.sum(np.abs(bb) ** 2, axis=(1, 2, 3), dtype=np.float64)
        result[start:end] = (pa / np.maximum(pb, 1e-30)).astype(np.float32)
    return result


def main() -> None:
    v140 = np.load(V140_OUTPUT, mmap_mode="r")
    v139 = np.load(V139_OUTPUT, mmap_mode="r")
    if v140.shape != v139.shape or v140.dtype != v139.dtype:
        raise ValueError("v139/v140 output layout mismatch")

    blocked = np.load(PATHS.shared_features / "height_geometry_cache.npz")[
        "test_blocked"
    ]
    regimes = regime_of(blocked)
    use_ultra = np.isin(regimes, ["light", "medium"])

    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=v140.dtype, shape=v140.shape
    )
    for start in range(0, len(v140), 10):
        end = min(start + 10, len(v140))
        output[start:end] = v140[start:end]
        local = use_ultra[start:end]
        if np.any(local):
            block = np.asarray(output[start:end])
            block[local] = v139[start:end][local]
            output[start:end] = block
    output.flush()

    target140 = np.load(V140_TARGET)
    target139 = np.load(V139_TARGET)
    target = np.array(target140, copy=True)
    target[use_ultra] = target139[use_ultra]
    np.save(TARGET_OUT, target.astype(np.float32))

    weights = np.load(V140_WEIGHTS).astype(np.float32)
    np.save(WEIGHTS_OUT, weights)
    np.save(SELECTOR_OUT, use_ultra.astype(np.uint8))

    finite = True
    for start in range(0, len(output), 10):
        block = np.asarray(output[start : start + 10])
        finite = finite and bool(
            np.isfinite(block.real).all() and np.isfinite(block.imag).all()
        )
    if not finite:
        raise ValueError("v141 output contains non-finite values")

    all_points = np.ones(len(output), dtype=bool)
    complex_cos_v140 = row_complex_cos(output, v140)
    complex_cos_v139 = row_complex_cos(output, v139)
    ratio_v140 = power_ratio(output, v140)
    by_regime = {}
    for regime in ("clear", "light", "medium", "heavy"):
        mask = regimes == regime
        by_regime[regime] = {
            "count": int(np.sum(mask)),
            "source": "v139" if regime in {"light", "medium"} else "v140",
            "nmse_to_v140": complex_nmse(output, v140, mask),
            "nmse_to_v139": complex_nmse(output, v139, mask),
            "cos_to_v140": summarize(complex_cos_v140[mask]),
            "cos_to_v139": summarize(complex_cos_v139[mask]),
            "power_ratio_to_v140": summarize(ratio_v140[mask]),
        }

    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "base": "v140 online 0.501352",
        "hypothesis": (
            "v139 ultra support projection remains beneficial for light/medium; "
            "its offline failure came from heavy-regime overdrive"
        ),
        "selector": {
            "ultra_source": "v139",
            "ultra_regimes": ["light", "medium"],
            "protected_source": "v140",
            "protected_regimes": ["clear", "heavy"],
            "ultra_count": int(np.sum(use_ultra)),
            "protected_count": int(np.sum(~use_ultra)),
        },
        "evidence": {
            "v140_online": 0.501352,
            "v138_online": 0.500793,
            "v139_medium_joint_after": 0.9978813959023172,
            "v140_medium_joint_after": 0.9977917564656619,
            "v139_light_joint_after": 0.7664403249356053,
            "v140_light_joint_after": 0.766358513780442,
            "v139_heavy_joint_after": 0.9196656765654277,
            "v140_heavy_joint_after": 0.9662256346469655,
            "rejected_selector": (
                "continuous shift-gain ExtraTrees spatial OOF Pearson <= 0.025; "
                "not used in candidate"
            ),
        },
        "diagnostics": {
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "finite": finite,
            "nmse_to_v140": complex_nmse(output, v140, all_points),
            "nmse_to_v139": complex_nmse(output, v139, all_points),
            "cos_to_v140": summarize(complex_cos_v140),
            "cos_to_v139": summarize(complex_cos_v139),
            "power_ratio_to_v140": summarize(ratio_v140),
            "by_regime": by_regime,
        },
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
            "target": PATHS.relative(TARGET_OUT),
            "weights": PATHS.relative(WEIGHTS_OUT),
            "selector": PATHS.relative(SELECTOR_OUT),
            "report": PATHS.relative(REPORT),
            "zip": PATHS.relative(ZIP_OUT),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        for source, name in (
            (__file__, "generate_v141_nonheavy_ultra_projection.py"),
            (
                PATHS.code / "analyze_v141_shift_selector.py",
                "analyze_v141_shift_selector.py",
            ),
            (
                PATHS.code / "generate_v140_balanced_hminus2_overdrive.py",
                "generate_v140_balanced_hminus2_overdrive.py",
            ),
            (
                PATHS.code / "generate_v139_ultra_geodesic_multishift.py",
                "generate_v139_ultra_geodesic_multishift.py",
            ),
            (PATHS.code_root / "project_paths.py", "project_paths.py"),
        ):
            archive.write(source, name)

    report["integrity"] = {
        "sha256": digest(ZIP_OUT, "sha256"),
        "md5": digest(ZIP_OUT, "md5"),
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {ZIP_OUT}")
    print(json.dumps(report["selector"], indent=2))
    print(json.dumps(report["diagnostics"], indent=2))
    print(json.dumps(report["integrity"], indent=2))


if __name__ == "__main__":
    main()
