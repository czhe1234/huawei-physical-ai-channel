#!/usr/bin/env python3
"""Calibrate a real Sionna Etoile setup against official train channels.

The official train positions exist in both domains: they have released channel
labels and can also be inserted into the restored Sionna Etoile scene.  This
script uses that correspondence to select plausible simulator settings before
building a large synthetic validation set.  It compares normalized spectral
objects, so an unknown transmit-power scale does not dominate the result.

Run this script in the ``sionna1`` environment because it imports Sionna RT and
Mitsuba.  It is analysis-only and does not create a submission archive.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import time

import numpy as np

from powerfield_common import regime_of
from project_paths import PATHS


VERSION = "v205_sionna_config_calibration"
REPORT_DIR = PATHS.version_report_dir(205)
REPORT = REPORT_DIR / "v205_sionna_config_calibration.json"
SCENE_XML = PATHS.artifacts / "external" / "restored_sionna_official_etoile" / "etoile.xml"
BS_POS = np.asarray([50.0, 0.0, 25.0], dtype=np.float32)
SPACINGS = (15_000.0, 30_000.0, 60_000.0, 120_000.0)


@dataclass(frozen=True)
class PathProfile:
    name: str
    tx_azimuth_deg: float
    rx_orientation: str
    max_depth: int
    diffuse_reflection: bool = True
    refraction: bool = True
    diffraction: bool = True
    samples_per_src: int = 40_000
    max_num_paths_per_src: int = 40_000
    v204_west_target: bool = False
    tx_target_range_m: float = 140.0
    frequency_hz: float | None = None


PROFILES = (
    PathProfile("v204_west_track_d3", 180.0, "track_bs", 3, v204_west_target=True),
    PathProfile("sector_m30_track_d3", -30.0, "track_bs", 3),
    PathProfile("sector_m40_track_d3", -40.0, "track_bs", 3),
    PathProfile("sector_m50_track_d3", -50.0, "track_bs", 3),
    PathProfile("sector_m40_fixed_d3", -40.0, "fixed", 3),
    PathProfile("sector_m40_track_d5", -40.0, "track_bs", 5, samples_per_src=60_000, max_num_paths_per_src=60_000),
    PathProfile("sector_m40_fixed_d5", -40.0, "fixed", 5, samples_per_src=60_000, max_num_paths_per_src=60_000),
    PathProfile(
        "sector_m40_track_specular_d5",
        -40.0,
        "track_bs",
        5,
        diffuse_reflection=False,
        refraction=False,
        diffraction=False,
        samples_per_src=60_000,
        max_num_paths_per_src=60_000,
    ),
    PathProfile("sector_m24_track_d3", -24.0, "track_bs", 3),
    PathProfile("sector_m27_track_d3", -27.0, "track_bs", 3),
    PathProfile("sector_m33_track_d3", -33.0, "track_bs", 3),
    PathProfile("sector_m36_track_d3", -36.0, "track_bs", 3),
    PathProfile("sector_m30_r80_track_d3", -30.0, "track_bs", 3, tx_target_range_m=80.0),
    PathProfile("sector_m30_r100_track_d3", -30.0, "track_bs", 3, tx_target_range_m=100.0),
    PathProfile("sector_m30_r200_track_d3", -30.0, "track_bs", 3, tx_target_range_m=200.0),
    PathProfile("sector_m30_r300_track_d3", -30.0, "track_bs", 3, tx_target_range_m=300.0),
    PathProfile("sector_m30_fixed_d3", -30.0, "fixed", 3),
    PathProfile("sector_m30_track_d5", -30.0, "track_bs", 5, samples_per_src=60_000, max_num_paths_per_src=60_000),
    PathProfile("sector_m30_fixed_d5", -30.0, "fixed", 5, samples_per_src=60_000, max_num_paths_per_src=60_000),
    PathProfile(
        "sector_m30_track_specular_d5",
        -30.0,
        "track_bs",
        5,
        diffuse_reflection=False,
        refraction=False,
        diffraction=False,
        samples_per_src=60_000,
        max_num_paths_per_src=60_000,
    ),
    PathProfile("sector_m30_track_d5_f2p6", -30.0, "track_bs", 5, samples_per_src=60_000, max_num_paths_per_src=60_000, frequency_hz=2.6e9),
    PathProfile("sector_m30_track_d5_f4p9", -30.0, "track_bs", 5, samples_per_src=60_000, max_num_paths_per_src=60_000, frequency_hz=4.9e9),
    PathProfile("sector_m30_track_d5_f6p0", -30.0, "track_bs", 5, samples_per_src=60_000, max_num_paths_per_src=60_000, frequency_hz=6.0e9),
    PathProfile("sector_m30_track_d5_f6p7", -30.0, "track_bs", 5, samples_per_src=60_000, max_num_paths_per_src=60_000, frequency_hz=6.7e9),
    PathProfile("sector_m30_track_d5_f7p0", -30.0, "track_bs", 5, samples_per_src=60_000, max_num_paths_per_src=60_000, frequency_hz=7.0e9),
    PathProfile("sector_m30_r300_track_d3_f6p7", -30.0, "track_bs", 3, tx_target_range_m=300.0, frequency_hz=6.7e9),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-count", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=205)
    parser.add_argument("--frequency", type=float, default=3.5e9)
    parser.add_argument("--num-subcarriers", type=int, default=192)
    parser.add_argument("--tx-power-dbm", type=float, default=44.0)
    parser.add_argument("--mitsuba-variant", default="llvm_ad_mono_polarized")
    parser.add_argument("--profile", action="append", default=None, help="Run only a named profile; repeatable")
    parser.add_argument("--spacing", action="append", type=float, default=None, help="Subcarrier spacing to evaluate; repeatable")
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="Keep candidates from an existing report when adding a profile refinement batch",
    )
    return parser.parse_args()


def stratified_indices(regimes: np.ndarray, count: int, seed: int) -> np.ndarray:
    if count >= len(regimes):
        return np.arange(len(regimes), dtype=np.int32)
    labels = np.asarray(regimes).astype(str)
    rng = np.random.default_rng(seed)
    names = ("clear", "light", "medium", "heavy")
    raw = {name: count * float(np.mean(labels == name)) for name in names}
    allocation = {name: int(math.floor(raw[name])) for name in names}
    for name in sorted(names, key=lambda value: raw[value] - allocation[value], reverse=True):
        if sum(allocation.values()) >= count:
            break
        allocation[name] += 1
    selected: list[int] = []
    for name in names:
        candidates = np.flatnonzero(labels == name)
        selected.extend(rng.choice(candidates, size=allocation[name], replace=False).tolist())
    out = np.asarray(selected, dtype=np.int32)
    out.sort()
    return out


def convert_cfr(h: np.ndarray) -> np.ndarray:
    expected = (4, 1, 256, 1, 192)
    if h.ndim != 6 or tuple(h.shape[1:]) != expected:
        raise ValueError(f"Unexpected Sionna CFR shape {h.shape}; expected [B,{','.join(map(str, expected))}]")
    return np.transpose(h[:, :, 0, :, 0, :], (0, 2, 1, 3)).astype(np.complex64)


def map_tx_axes(h: np.ndarray, mapping: str) -> np.ndarray:
    if mapping == "identity":
        return h
    value = h.reshape(len(h), 2, 16, 8, 4, 192)
    if mapping == "flip_hv":
        value = value[:, :, ::-1, ::-1]
    elif mapping == "flip_h":
        value = value[:, :, ::-1]
    elif mapping == "flip_v":
        value = value[:, :, :, ::-1]
    else:
        raise ValueError(mapping)
    return np.ascontiguousarray(value).reshape(h.shape)


def cosine(a: np.ndarray, b: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    numerator = np.sum(a * b, axis=axes, dtype=np.float64)
    aa = np.sum(a * a, axis=axes, dtype=np.float64)
    bb = np.sum(b * b, axis=axes, dtype=np.float64)
    return (numerator / np.maximum(np.sqrt(aa * bb), 1e-30)).astype(np.float32)


def spectral_views(h: np.ndarray) -> dict[str, np.ndarray]:
    value6 = h.reshape(len(h), 2, 16, 8, 4, 192)
    pas = np.abs(np.fft.fft2(value6, axes=(2, 3))).astype(np.float32) ** 2
    delay = np.abs(np.fft.ifft(h, axis=-1)).astype(np.float32) ** 2
    leaked = value6.transpose(0, 1, 5, 2, 4, 3)
    leaked = np.abs(np.fft.fft2(leaked, axes=(2, 3))).astype(np.float32) ** 2
    leaked = leaked.transpose(0, 1, 4, 5, 2, 3).reshape(len(h), 2, 4, 8, -1)
    ad = np.fft.ifft(np.fft.fft2(value6, axes=(2, 3)), axis=-1)
    ad = np.sum(np.abs(ad).astype(np.float32) ** 2, axis=(1, 4))
    flat = h.reshape(len(h), -1)
    energy = np.sum(np.abs(flat) ** 2, axis=1, dtype=np.float64)
    return {
        "pas": pas.reshape(len(h), 256, 4, 192),
        "delay": delay,
        "leaked": leaked,
        "ad": ad,
        "flat": flat,
        "energy": energy,
    }


def spectral_scores(pred: np.ndarray, truth_views: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    pred_views = spectral_views(pred)
    pas = cosine(
        pred_views["pas"],
        truth_views["pas"],
        (1,),
    ).mean(axis=(1, 2))
    pdp = cosine(pred_views["delay"], truth_views["delay"], (3,)).mean(axis=(1, 2))
    leaked_sh = cosine(pred_views["leaked"], truth_views["leaked"], (4,)).mean(axis=(1, 2, 3))
    ad_support = cosine(pred_views["ad"], truth_views["ad"], (1, 2, 3))

    dot = np.sum(
        np.conj(pred_views["flat"]) * truth_views["flat"],
        axis=1,
        dtype=np.complex128,
    )
    pred_energy = pred_views["energy"]
    truth_energy = truth_views["energy"]
    coherence = (np.abs(dot) / np.maximum(np.sqrt(pred_energy * truth_energy), 1e-30)).astype(np.float32)

    return {
        "pas": pas.astype(np.float32),
        "pdp": pdp.astype(np.float32),
        "leaked_sh": leaked_sh.astype(np.float32),
        "ad_support": ad_support.astype(np.float32),
        "complex_coherence": coherence,
        "pred_log_power": np.log(np.maximum(pred_energy, 1e-30)).astype(np.float32),
        "truth_log_power": np.log(np.maximum(truth_energy, 1e-30)).astype(np.float32),
    }


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import rankdata

    x = rankdata(np.asarray(a, dtype=np.float64))
    y = rankdata(np.asarray(b, dtype=np.float64))
    if np.std(x) <= 0 or np.std(y) <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def summarize_candidate(values: dict[str, list[np.ndarray]]) -> dict[str, float | int]:
    merged = {key: np.concatenate(parts) for key, parts in values.items()}
    valid_power = np.isfinite(merged["pred_log_power"]) & np.isfinite(merged["truth_log_power"])
    power_spearman = rank_correlation(
        merged["pred_log_power"][valid_power], merged["truth_log_power"][valid_power]
    )
    report: dict[str, float | int] = {
        "count": int(len(merged["pas"])),
        "pas_mean": float(np.mean(merged["pas"])),
        "pdp_mean": float(np.mean(merged["pdp"])),
        "leaked_sh_mean": float(np.mean(merged["leaked_sh"])),
        "ad_support_mean": float(np.mean(merged["ad_support"])),
        "complex_coherence_mean": float(np.mean(merged["complex_coherence"])),
        "power_spearman": power_spearman,
        "power_offset_median": float(np.median(merged["truth_log_power"] - merged["pred_log_power"])),
    }
    report["calibration_score"] = float(
        0.30 * report["pas_mean"]
        + 0.30 * report["pdp_mean"]
        + 0.20 * report["leaked_sh_mean"]
        + 0.15 * report["ad_support_mean"]
        + 0.05 * (power_spearman + 1.0) / 2.0
    )
    return report


def tx_target(profile: PathProfile, mi):
    if profile.v204_west_target:
        return mi.Point3f(0.0, 0.0, 1.5)
    radius = profile.tx_target_range_m
    azimuth = math.radians(profile.tx_azimuth_deg)
    return mi.Point3f(
        float(BS_POS[0] + radius * math.cos(azimuth)),
        float(BS_POS[1] + radius * math.sin(azimuth)),
        1.5,
    )


def run_profile(
    profile: PathProfile,
    args: argparse.Namespace,
    positions: np.ndarray,
    official_h,
    mi,
    load_scene,
    Transmitter,
    Receiver,
    PlanarArray,
    PathSolver,
    subcarrier_frequencies,
    spacings: tuple[float, ...],
) -> tuple[dict[str, object], dict[str, dict[str, list[np.ndarray]]]]:
    scene = load_scene(str(SCENE_XML), merge_shapes=True)
    effective_frequency = profile.frequency_hz or args.frequency
    scene.frequency = effective_frequency
    scene.tx_array = PlanarArray(
        num_rows=8,
        num_cols=16,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="tr38901",
        polarization="VH",
    )
    scene.rx_array = PlanarArray(
        num_rows=2,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="dipole",
        polarization="cross",
    )
    tx = Transmitter(
        name=f"tx_{profile.name}",
        position=mi.Point3f(*[float(v) for v in BS_POS]),
        look_at=tx_target(profile, mi),
        power_dbm=args.tx_power_dbm,
    )
    scene.add(tx)
    solver = PathSolver()
    frequencies = {
        spacing: subcarrier_frequencies(args.num_subcarriers, spacing).cpu().numpy().astype(np.float32)
        for spacing in spacings
    }
    mappings = ("identity",)
    candidate_values = {
        f"spacing_{int(spacing)}__{mapping}": {name: [] for name in (
            "pas", "pdp", "leaked_sh", "ad_support", "complex_coherence", "pred_log_power", "truth_log_power"
        )}
        for spacing in spacings
        for mapping in mappings
    }
    receiver_names: list[str] = []
    solve_seconds = 0.0
    cfr_seconds = 0.0
    started = time.time()

    for start in range(0, len(positions), args.batch_size):
        end = min(start + args.batch_size, len(positions))
        for name in receiver_names:
            scene.remove(name)
        receiver_names = []
        for local, pos in enumerate(positions[start:end]):
            name = f"rx_{start + local:05d}"
            kwargs = {
                "name": name,
                "position": mi.Point3f(float(pos[0]), float(pos[1]), float(pos[2])),
            }
            if profile.rx_orientation == "track_bs":
                kwargs["look_at"] = tx
            else:
                kwargs["orientation"] = mi.Point3f(0.0, 0.0, 0.0)
            scene.add(Receiver(**kwargs))
            receiver_names.append(name)

        t0 = time.time()
        paths = solver(
            scene,
            max_depth=profile.max_depth,
            max_num_paths_per_src=profile.max_num_paths_per_src,
            samples_per_src=profile.samples_per_src,
            synthetic_array=True,
            los=True,
            specular_reflection=True,
            diffuse_reflection=profile.diffuse_reflection,
            refraction=profile.refraction,
            diffraction=profile.diffraction,
            edge_diffraction=False,
            diffraction_lit_region=True,
            seed=args.seed + start,
        )
        solve_seconds += time.time() - t0
        truth = np.asarray(official_h[start:end], dtype=np.complex64)
        truth_views = spectral_views(truth)
        for spacing in spacings:
            t1 = time.time()
            raw = paths.cfr(
                mi.TensorXf(frequencies[spacing]),
                normalize_delays=False,
                normalize=False,
                out_type="numpy",
            )
            cfr_seconds += time.time() - t1
            sim = convert_cfr(raw)
            for mapping in mappings:
                key = f"spacing_{int(spacing)}__{mapping}"
                scores = spectral_scores(map_tx_axes(sim, mapping), truth_views)
                for name, value in scores.items():
                    candidate_values[key][name].append(value)
        print(
            f"{profile.name} {end}/{len(positions)} solve={solve_seconds:.1f}s cfr={cfr_seconds:.1f}s",
            flush=True,
        )

    for name in receiver_names:
        scene.remove(name)
    profile_report = {
        "profile": asdict(profile),
        "effective_frequency_hz": float(effective_frequency),
        "seconds": float(time.time() - started),
        "solve_seconds": float(solve_seconds),
        "cfr_seconds": float(cfr_seconds),
    }
    return profile_report, candidate_values


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    import mitsuba as mi

    mi.set_variant(args.mitsuba_variant)
    from sionna.phy.channel import subcarrier_frequencies
    from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    geom = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    regimes_all = regime_of(geom["train_blocked"].astype(np.float32))
    selected = stratified_indices(regimes_all, args.calibration_count, args.seed)
    positions = train_pos[selected]
    official_all = np.load(PATHS.round1 / "Round1_Train_Channel.npy", mmap_mode="r")
    official_h = np.asarray(official_all[selected], dtype=np.complex64)
    requested = set(args.profile or [profile.name for profile in PROFILES])
    profiles = [profile for profile in PROFILES if profile.name in requested]
    missing = requested - {profile.name for profile in profiles}
    if missing:
        raise ValueError(f"Unknown profiles: {sorted(missing)}")
    spacings = tuple(args.spacing) if args.spacing else SPACINGS

    candidates: list[dict[str, object]] = []
    profile_reports: list[dict[str, object]] = []
    started = time.time()
    for profile in profiles:
        profile_report, values = run_profile(
            profile,
            args,
            positions,
            official_h,
            mi,
            load_scene,
            Transmitter,
            Receiver,
            PlanarArray,
            PathSolver,
            subcarrier_frequencies,
            spacings,
        )
        profile_reports.append(profile_report)
        for key, parts in values.items():
            spacing_text, mapping = key.split("__")
            item = summarize_candidate(parts)
            item.update(
                {
                    "profile": profile.name,
                    "frequency_hz": float(profile.frequency_hz or args.frequency),
                    "subcarrier_spacing": int(spacing_text.removeprefix("spacing_")),
                    "tx_axis_mapping": mapping,
                }
            )
            candidates.append(item)

    if args.merge_existing and REPORT.is_file():
        previous = json.loads(REPORT.read_text(encoding="utf-8"))
        replaced_profiles = {profile.name for profile in profiles}
        candidates.extend(
            item for item in previous.get("candidates", []) if item.get("profile") not in replaced_profiles
        )
        profile_reports.extend(
            item
            for item in previous.get("profiles", [])
            if item.get("profile", {}).get("name") not in replaced_profiles
        )
    candidates.sort(key=lambda item: float(item["calibration_score"]), reverse=True)
    regime_counts = {
        name: int(np.sum(regimes_all[selected].astype(str) == name))
        for name in ("clear", "light", "medium", "heavy")
    }
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "A same-coordinate comparison between released train H and real Sionna H can calibrate "
            "the simulator setup well enough to make a full synthetic benchmark informative."
        ),
        "scene": PATHS.relative(SCENE_XML),
        "mitsuba_variant": mi.variant(),
        "frequency": args.frequency,
        "calibration_count": int(len(selected)),
        "calibration_indices": selected.tolist(),
        "regime_counts": regime_counts,
        "antenna_order_evidence": {
            "sionna": "pattern first, then column-first planar index; within each horizontal column rows run top-to-bottom",
            "competition_mapping": "TX [polarization, horizontal, vertical], RX [polarization, vertical]",
            "source": "Sionna AntennaArray.num_ant, PlanarArray numbering, and Paths._fuse_pattern_array_dims",
        },
        "score_formula": "0.30 PAS + 0.30 PDP + 0.20 leaked-SH + 0.15 joint-AD + 0.05 rescaled power Spearman",
        "profiles": profile_reports,
        "candidates": candidates,
        "best_candidate": candidates[0],
        "total_seconds": float(time.time() - started),
        "outputs": {"report": PATHS.relative(REPORT)},
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"report": PATHS.relative(REPORT), "best_candidate": candidates[0]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
