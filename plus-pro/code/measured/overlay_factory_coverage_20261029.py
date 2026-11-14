"""Position-only extraction and scan overlay for factory scene qualification."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import mmap
import os
import struct
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import PATHS
from dc01_data import fields, only, tensor, masked_crc

ROOT = PATHS.root
RAW = ROOT / "datasets/dichasus/dichasus-adxx"
OUT = ROOT / "analysis/factory_coverage_20261029"
ART = ROOT / "artifacts/measured/factory_coverage_20261029"
SPEC_PATH = ROOT / "analysis/audits/factory_scene_20261023/v3_spec.json"
SPEC = json.loads(SPEC_PATH.read_text())
MANIFEST = json.loads((ROOT / "analysis/dichasus_acquisition_20261016/catalog_manifest.json").read_text())
CATALOG = {Path(f["relative_path"]).name: f for f in MANIFEST["files"]
           if "dichasus-adxx" in f.get("groups", []) and f["filename"].endswith(".tfrecords")}
ENTRIES = [e for e in SPEC["tfrecords"] if e["configurationVariant"].startswith("no-wall")
           and (RAW / Path(e["filename"]).name).is_file()]
OUT.mkdir(parents=True, exist_ok=True)
ART.mkdir(parents=True, exist_ok=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    os.replace(tmp, path)


def identity(path):
    st = path.stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "inode": st.st_ino}


def metadata(payload):
    result = {}
    for key, _, entry in fields(only(payload, 1)):
        if key != 1:
            continue
        name = bytes(only(entry, 1)).decode("utf-8")
        if name not in ("pos-tachy", "time", "gt-interp-age-tachy"):
            continue
        kind, _, value = next(fields(only(entry, 2)))
        if kind == 1:
            result[name] = tensor(only(value, 1)).copy()
        elif kind == 2:
            result[name] = float(np.frombuffer(only(value, 1), dtype="<f4")[0])
        else:
            raise ValueError((name, kind))
    if set(result) != {"pos-tachy", "time", "gt-interp-age-tachy"}:
        raise ValueError("Missing required position metadata")
    if result["pos-tachy"].shape != (3,):
        raise ValueError("Position tensor is not XYZ")
    return result


def inspect_entry(entry):
    name = Path(entry["filename"]).stem
    path = RAW / (name + ".tfrecords")
    output = ART / (name + ".npz")
    receipt_path = ART / (name + ".json")
    before = identity(path)
    expected = CATALOG[path.name]
    if before["size"] != expected["size"]:
        raise ValueError(f"Source differs from frozen download size: {name}")
    if output.exists() and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["source_identity"] == before and receipt["npz_sha256"] == sha(output):
            print("reuse", name, receipt["frames"], flush=True)
            return receipt
        raise ValueError(f"Existing extraction identity mismatch: {name}")
    started = time.monotonic()
    positions, timestamps, ages = [], [], []
    record_offsets = []
    crc_cache = {}
    with path.open("rb") as stream:
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as raw:
            if hasattr(raw, "madvise"):
                raw.madvise(mmap.MADV_RANDOM)
            offset = 0
            while offset < len(raw):
                if offset + 16 > len(raw):
                    raise ValueError("Truncated record framing")
                length, header_crc = struct.unpack_from("<QI", raw, offset)
                end = offset + 16 + length
                if end > len(raw):
                    raise ValueError("Truncated record payload")
                if length not in crc_cache:
                    crc_cache[length] = masked_crc(struct.pack("<Q", length))
                if crc_cache[length] != header_crc:
                    raise ValueError("Record length CRC mismatch")
                payload = memoryview(raw)[offset+12:offset+12+length]
                item = metadata(payload)
                del payload
                positions.append(item["pos-tachy"])
                timestamps.append(item["time"])
                ages.append(item["gt-interp-age-tachy"])
                record_offsets.append(offset)
                offset = end
            assert offset == len(raw)
    xyz = np.asarray(positions, dtype=np.float64)
    times = np.asarray(timestamps, dtype=np.float64)
    age = np.asarray(ages, dtype=np.float64)
    finite = np.isfinite(xyz).all(axis=1) & np.isfinite(times) & np.isfinite(age)
    if not finite.any():
        raise ValueError("No finite positions")
    folder = SPEC["configuration"]["variants"][entry["configurationVariant"]]["potree"]["folder"]
    day = folder.rsplit("-", 1)[-1]
    assert day in ("day1", "day2")
    temp = output.with_suffix(".npz.tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, position=xyz, time=times, interpolation_age=age,
                            source_offset=np.array(record_offsets, dtype=np.int64), finite=finite)
    os.replace(temp, output)
    valid = xyz[finite]
    cells = np.floor(valid[:, :2] / .5).astype(np.int64)
    previous_view = ((valid[:, 0] >= -30) & (valid[:, 0] <= 16)
                     & (valid[:, 1] >= -34) & (valid[:, 1] <= 12))
    receipt = {"record": name, "day": day, "variant": entry["configurationVariant"],
        "description": entry["description"], "source_path": str(path.relative_to(ROOT)),
        "source_identity": before, "source_catalog_md5": expected.get("checksum"),
        "frames": len(xyz), "finite_frames": int(finite.sum()), "nonfinite_frames": int((~finite).sum()),
        "position_min": valid.min(axis=0).tolist(), "position_max": valid.max(axis=0).tolist(),
        "unique_half_metre_cells": len(np.unique(cells, axis=0)),
        "outside_previous_atlas_crop": int((~previous_view).sum()),
        "time_nonincreasing_steps": int(np.count_nonzero(np.diff(times) <= 0)),
        "interpolation_age_quantiles_s": np.quantile(age[finite], [0, .5, .9, .99, 1]).tolist(),
        "time_range_s": [float(np.nanmin(times)), float(np.nanmax(times))],
        "position_tensor_decode": "reused audited TensorProto decoder; CSI feature skipped before tensor decoding",
        "header_crc_all_passed": True, "full_payload_crc_checked": False,
        "elapsed_s": time.monotonic() - started, "npz_sha256": sha(output),
        "source_metadata_unchanged": identity(path) == before}
    assert receipt["source_metadata_unchanged"]
    atomic_json(receipt_path, receipt)
    print("extracted", name, len(xyz), "bounds", receipt["position_min"][:2], receipt["position_max"][:2],
          "seconds", round(receipt["elapsed_s"], 2), flush=True)
    return receipt


def extract():
    began = time.monotonic()
    sources = [RAW / Path(e["filename"]).name for e in ENTRIES]
    initial = {str(f.relative_to(ROOT)): identity(f) for f in sources}
    receipts = [inspect_entry(e) for e in ENTRIES]
    # Independent comparison with previously CRC-checked position-only values.
    previous = json.loads((ROOT / "analysis/audits/factory_open_20261024/local_checks.json").read_text())
    comparison = []
    for short, old in previous["recordings"].items():
        data = np.load(ART / ("dichasus-" + short + ".npz"))
        for sample in old["sampled_records"]:
            i = sample["index"]
            error = float(np.max(np.abs(data["position"][i] - sample["position"])))
            time_error = abs(float(data["time"][i]) - sample["time"])
            age_error = abs(float(data["interpolation_age"][i]) - sample["position_interpolation_age_s"])
            comparison.append({"record": short, "index": i, "position_max_abs_error_m": error,
                               "time_abs_error_s": time_error, "age_abs_error_s": age_error})
            assert max(error, time_error, age_error) < 1e-10
    final = {str(f.relative_to(ROOT)): identity(f) for f in sources}
    result = {"scope": "All locally complete no-wall factory recordings; position/time/interpolation age only",
        "recordings": receipts, "frames": sum(r["frames"] for r in receipts),
        "source_stat_unchanged": initial == final, "raw_source_version": "historical V2 download",
        "configuration_source": str(SPEC_PATH.relative_to(ROOT)), "configuration_spec_sha256": sha(SPEC_PATH),
        "configuration_version_note": "V3 spec supplies file/day/configuration association only. Historical V2 position labels retained; no V3 correction applied.",
        "all_header_crc_passed": all(r["header_crc_all_passed"] for r in receipts),
        "prior_crc_checked_metadata_comparison": comparison,
        "csi_tensors_decoded": 0, "calibration_applied": False, "training_scoring_rt": False,
        "elapsed_s": time.monotonic() - began}
    assert result["source_stat_unchanged"]
    atomic_json(OUT / "extraction.json", result)
    with (OUT / "records.csv").open("w") as f:
        columns = ["record", "day", "variant", "frames", "nonfinite_frames", "unique_half_metre_cells", "outside_previous_atlas_crop"]
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: r[k] for k in columns} for r in receipts)
    print("position extraction complete", result["frames"], round(result["elapsed_s"], 2), flush=True)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.collections import LineCollection
    from scipy.spatial import cKDTree

    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    font_manager.fontManager.addfont(str(font))
    family = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"font.family": family, "font.size": 11, "axes.unicode_minus": False})
    extraction = json.loads((OUT / "extraction.json").read_text())
    samples = np.load(ROOT / "artifacts/measured/scene_atlas_20261029/display_samples.npz")
    arrays = np.array([a["location"] for a in SPEC["antennas"]])
    records = {r["record"]: r for r in extraction["recordings"]}
    data = {name: dict(np.load(ART / (name + ".npz"))) for name in records}
    colors = {"day1": "#1169B0", "day2": "#D16319"}
    summary = {"display_pointcloud_source": "fixed-stride samples of original day1/day2 scans, no wall configurations",
        "pointcloud_xy_background_z_m": [.2, 7.5], "low_surface_proxy_z_m": [.3, 2.5],
        "coordinate_transform": "identity: published factory measurement XYZ, unlike hallway GLB",
        "boundary_proxy_note": "XY nearest fixed-stride sampled low-height scan point. Sampling affects distance. Frame weighted, not spatial area or path length. Large distance does not establish open space; small distance does not establish equipment or blockage. Not 3D LoS/NLoS or reflection evidence.",
        "days": {}, "records": {}}
    no_wall = [e for e in SPEC["tfrecords"] if e["configurationVariant"].startswith("no-wall")]
    summary["scope_counts"] = {"spec_no_wall_records": len(no_wall), "included_no_wall_records": len(records),
        "missing_no_wall_files": [Path(e["filename"]).name for e in no_wall if not (RAW/Path(e["filename"]).name).exists()],
        "excluded_wall_records": [Path(e["filename"]).stem for e in SPEC["tfrecords"] if e["configurationVariant"].startswith("wall")]}
    for name, r in records.items():
        folder = SPEC["configuration"]["variants"][r["variant"]]["potree"]["folder"]
        assert r["day"] == folder.rsplit("-", 1)[-1]
        pos = data[name]["position"]
        dt = np.diff(data[name]["time"])
        step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        summary["records"][name] = {"pointcloud_mapping_source": folder,
            "nonfinite_position_frames": int(np.sum(~np.isfinite(pos).all(axis=1))),
            "time_gaps_above_0_5s": int(np.sum(dt > .5)),
            "position_steps_above_1m": int(np.sum(step > 1)),
            "max_positive_time_gap_s": float(np.max(dt)),
            "max_adjacent_position_step_m": float(np.max(step)),
            "note": "Finite means numerically finite only; no localization quality acceptance or filtering performed."}

    for day in ["day1", "day2"]:
        names = [name for name, r in records.items() if r["day"] == day]
        pts = np.concatenate([data[n]["position"][data[n]["finite"]] for n in names])
        cells = np.unique(np.floor(pts[:, :2]/.5).astype(np.int64), axis=0)
        cloud = samples[day]
        # This is a display-derived proximity proxy, never a surface mesh.
        low = cloud[(cloud[:, 2] >= .3) & (cloud[:, 2] <= 2.5), :2]
        tree = cKDTree(low)
        distances = tree.query(pts[:, :2], workers=2)[0]
        limits = [[float(pts[:, 0].min()-3), float(pts[:, 0].max()+3)],
                  [float(pts[:, 1].min()-3), float(pts[:, 1].max()+3)]]
        limits[0] = [min(limits[0][0], arrays[:, 0].min()-3), max(limits[0][1], arrays[:, 0].max()+3)]
        limits[1] = [min(limits[1][0], arrays[:, 1].min()-3), max(limits[1][1], arrays[:, 1].max()+3)]
        summary["days"][day] = {"records": names, "frames": len(pts), "position_min": pts.min(axis=0).tolist(),
            "position_max": pts.max(axis=0).tolist(), "display_limits": limits, "unique_half_metre_cells": len(cells),
            "xy_low_scan_distance_quantiles_m": np.quantile(distances, [0, .25, .5, .75, .9, .99, 1]).tolist(),
            "within_1m_low_scan_fraction": float(np.mean(distances <= 1)), "within_2m_low_scan_fraction": float(np.mean(distances <= 2))}
        for name in names:
            xyz = data[name]["position"][data[name]["finite"]]
            d = tree.query(xyz[:, :2], workers=2)[0]
            summary["records"][name].update({"xy_low_scan_distance_median_m": float(np.median(d)),
                "within_1m_low_scan_fraction": float(np.mean(d <= 1)), "within_2m_low_scan_fraction": float(np.mean(d <= 2))})

    def base(ax, day, limits=None):
        cloud = samples[day]
        points = cloud[(cloud[:, 2] >= .2) & (cloud[:, 2] <= 7.5)]
        if limits is None:
            limits = summary["days"][day]["display_limits"]
        mask = ((points[:, 0] >= limits[0][0]) & (points[:, 0] <= limits[0][1])
                & (points[:, 1] >= limits[1][0]) & (points[:, 1] <= limits[1][1]))
        points = points[mask]
        rgb = np.clip(points[:, 4:7]/255, 0, 1)
        rgb = .15 + .65*rgb
        ax.scatter(points[:, 0], points[:, 1], c=rgb, s=.7, linewidths=0, alpha=.55, rasterized=True, zorder=1)
        ax.scatter(arrays[:, 0], arrays[:, 1], marker="^", s=75, c="#A83128", edgecolors="white", linewidths=.6,
                   zorder=5, label="接收阵列")
        for i, a in enumerate(arrays):
            if limits[0][0] <= a[0] <= limits[0][1] and limits[1][0] <= a[1] <= limits[1][1]:
                ax.annotate(f"A{i+1}", a[:2], (5, 4), textcoords="offset points", fontsize=10, color="#862A23", zorder=6)
        ax.set_xlim(*limits[0]); ax.set_ylim(*limits[1]); ax.set_aspect("equal")
        ax.set_xlabel("X（米）"); ax.set_ylabel("Y（米）")
        ax.grid(alpha=.15)

    def coverage(ax, names, color, alpha=.16):
        for name in names:
            pos = data[name]["position"][data[name]["finite"]]
            ax.scatter(pos[:, 0], pos[:, 1], s=.55, color=color, alpha=alpha, linewidths=0, rasterized=True, zorder=3)

    def route(ax, name, color):
        item = data[name]
        xy = item["position"][:, :2]
        finite = item["finite"]
        dt = np.diff(item["time"])
        step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        ok = finite[:-1] & finite[1:] & (dt > 0) & (dt <= .5) & (step <= 1)
        segments = np.stack([xy[:-1], xy[1:]], axis=1)[ok]
        ax.add_collection(LineCollection(segments, colors=color, linewidths=.55, alpha=.65, zorder=4, rasterized=True))
        valid = xy[finite]
        ax.scatter(valid[0, 0], valid[0, 1], s=35, marker="o", c=color, edgecolors="white", zorder=8)
        return int((~ok).sum())

    shared_limits = [[min(v["display_limits"][0][0] for v in summary["days"].values()),
                      max(v["display_limits"][0][1] for v in summary["days"].values())],
                     [min(v["display_limits"][1][0] for v in summary["days"].values()),
                      max(v["display_limits"][1][1] for v in summary["days"].values())]]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    for ax, day, label in zip(axes, ["day1", "day2"], ["第一天", "第二天"]):
        base(ax, day, shared_limits)
        info = summary["days"][day]
        coverage(ax, info["records"], colors[day])
        ax.set_title(f'{label}｜{len(info["records"])} 份记录，{info["frames"]:,} 个位置帧', pad=12)
    fig.suptitle("工厂：实际测量位置与当天点云", fontsize=21, y=.97)
    fig.text(.5, .075, "蓝色 / 橙色：全部数值有限的位置帧，尚未作定位质量筛选。红色三角：四副接收阵列。", ha="center", fontsize=11)
    fig.text(.5, .047, "灰色：当天点云在 0.2–7.5 米高度内的俯视投影；不是材料标签或完整表面模型。", ha="center", fontsize=10, color="#465665")
    fig.text(.5, .02, "包含所有 20 份已下载无墙记录。历史 V2 位置未作版本校正；未读取信道系数、训练或评分。", ha="center", fontsize=10, color="#465665")
    fig.subplots_adjust(left=.07, right=.97, bottom=.2, top=.82, wspace=.2)
    fig.savefig(OUT / "coverage_all.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    choices = [("dichasus-ad09", "第一天：全区域随机路线", "#126AA4"),
               ("dichasus-ad10", "第二天：东北—西南往返", "#A24D10"),
               ("dichasus-ad11", "第二天：西北—东南往返", "#6446A0")]
    main_limits = [[-25, 13], [-35, 5]]
    for ax, (name, title, color) in zip(axes, choices):
        base(ax, records[name]["day"], main_limits)
        breaks = route(ax, name, color)
        ax.set_title(title + f'\n{records[name]["frames"]:,} 帧', pad=12)
        summary["records"][name]["display_line_breaks"] = breaks
    fig.suptitle("中央区域：三条覆盖互补的路线｜相同接收增益 67 dB", fontsize=21, y=.97)
    fig.text(.5, .045, "每条路线按时间连接；超过 0.5 秒的缺口或相邻位置跳变超过 1 米时断开。圆点是起点。", ha="center", fontsize=11)
    fig.text(.5, .017, "轨迹贴近点云结构，只能说明几何位置接近；不能据此认定发生了直射遮挡切换。", ha="center", fontsize=11, color="#465665")
    fig.subplots_adjust(left=.05, right=.98, bottom=.15, top=.82, wspace=.27)
    fig.savefig(OUT / "routes_main.png", dpi=175); plt.close(fig)

    # Data-driven full bounds for routes that leave the central measurement area.
    names = ["dichasus-ad07", "dichasus-ad08"]
    xyz = np.concatenate([data[n]["position"][data[n]["finite"]] for n in names])
    limits = [[min(xyz[:, 0].min()-4, -30), max(xyz[:, 0].max()+4, 16)],
              [min(xyz[:, 1].min()-4, -34), max(xyz[:, 1].max()+4, 12)]]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    for ax, name, color, title in zip(axes, names, ["#126AA4", "#A24D10"], ["深入工厂路线 · 64 dB", "深入工厂路线 · 70 dB"]):
        base(ax, "day1", limits)
        route(ax, name, color)
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((-30, -34), 46, 46, fill=False, edgecolor="#A33128", linewidth=1.2, linestyle="--", zorder=7))
        ax.set_title(title + f'\n{records[name]["frames"]:,} 帧；此前图册范围外 {records[name]["outside_previous_atlas_crop"]:,} 帧', pad=10)
    fig.suptitle("额外发现：两条路线离开中央开放区", fontsize=21, y=.98)
    fig.text(.5, .048, "红色虚线框：此前图册的固定显示范围。两条路线都使用第一天点云，但接收增益不同。", ha="center", fontsize=11)
    fig.text(.5, .021, "不能因为更深入工厂就称它为设备密集区；需要结合该位置的扫描细节判断。", ha="center", fontsize=11, color="#465665")
    fig.subplots_adjust(left=.06, right=.98, bottom=.17, top=.82, wspace=.2)
    fig.savefig(OUT / "routes_extended.png", dpi=180); plt.close(fig)

    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    fig = plt.figure(figsize=(16, 7))
    for i, (xb, yb, title) in enumerate([([-72, 12], [-35, 5], "中央区域与延伸路线"),
                                         ([-72, -28], [-27, -6], "延伸区域局部")]):
        ax = fig.add_subplot(1, 2, i+1, projection="3d")
        cloud = samples["day1"]
        keep = ((cloud[:, 0] >= xb[0]) & (cloud[:, 0] <= xb[1]) & (cloud[:, 1] >= yb[0])
                & (cloud[:, 1] <= yb[1]) & (cloud[:, 2] >= .2) & (cloud[:, 2] <= 7.5))
        selected = cloud[keep]
        rgb = .1 + .8 * np.clip(selected[:, 4:7]/255, 0, 1)
        ax.scatter(selected[:, 0], selected[:, 1], selected[:, 2], c=rgb, s=.55,
                   linewidths=0, depthshade=False, rasterized=True)
        for name, color in zip(names, ["#126AA4", "#A24D10"]):
            item = data[name]
            xyz = item["position"]
            dt = np.diff(item["time"])
            step = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
            ok = item["finite"][:-1] & item["finite"][1:] & (dt > 0) & (dt <= .5) & (step <= 1)
            ax.add_collection3d(Line3DCollection(np.stack([xyz[:-1], xyz[1:]], axis=1)[ok],
                                                colors=color, linewidths=1.2, alpha=.9))
        ax.set_xlim(*xb); ax.set_ylim(*yb); ax.set_zlim(0, 8)
        ax.set_box_aspect((xb[1]-xb[0], yb[1]-yb[0], 8))
        ax.view_init(elev=34, azim=-65)
        ax.set_xlabel("X（米）"); ax.set_ylabel("Y（米）"); ax.set_zlabel("高度（米）")
        ax.set_title(title, pad=12); ax.tick_params(labelsize=9)
    fig.suptitle("深入工厂：实际轨迹叠加第一天真实点云", fontsize=21, y=.97)
    fig.text(.5, .04, "蓝 / 橙线分别为两份实际路线；点云显示高度 0.2–7.5 米，坐标轴保持相同长度比例。", ha="center", fontsize=11)
    fig.text(.5, .014, "这是扫描点与位置的叠加，没有重建表面、拟合遮挡体或根据测试信道补几何。", ha="center", fontsize=10, color="#465665")
    fig.subplots_adjust(left=.02, right=.98, bottom=.12, top=.85, wspace=.04)
    fig.savefig(OUT / "routes_extended_3d.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    for ax, day, title in zip(axes, ["day1", "day2"], ["第一天", "第二天"]):
        base(ax, day, [[-25, 13], [-35, 5]])
        names = [n for n, r in records.items() if r["day"] == day]
        xyz = np.concatenate([data[n]["position"][data[n]["finite"]] for n in names])
        cells = np.unique(np.floor(xyz[:, :2]/.5).astype(np.int64), axis=0)
        centers = (cells+.5)*.5
        ax.scatter(centers[:, 0], centers[:, 1], marker="s", s=6, c=colors[day], alpha=.45, linewidths=0, zorder=3)
        ax.set_title(title + "｜每个 0.5 米空间格只显示一次", pad=10)
    fig.suptitle("中央区域局部：空间覆盖｜避免停留时间造成视觉偏差", fontsize=20, y=.98)
    fig.text(.5, .035, "显示空间格不是训练 / 验证 / 测试划分；没有根据任何信道性能选择位置。", ha="center", fontsize=11)
    fig.subplots_adjust(left=.07, right=.98, bottom=.13, top=.87, wspace=.22)
    fig.savefig(OUT / "coverage_cells.png", dpi=180); plt.close(fig)
    atomic_json(OUT / "coverage_summary.json", summary)
    print("coverage plots complete", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["extract", "plot", "all"], default="all")
    args = parser.parse_args()
    try:
        if args.stage in ("extract", "all"):
            extract()
        if args.stage in ("plot", "all"):
            plot()
    except Exception:
        (OUT / "failure_traceback.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
