"""Render existing scene geometry for visual selection; never read CSI tensors."""
from __future__ import annotations

import concurrent.futures
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.collections import PolyCollection
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import PATHS

ROOT = PATHS.root
OUT = ROOT / "analysis/scene_atlas_20261029"
ART = ROOT / "artifacts/measured/scene_atlas_20261029"
BASE = ROOT / "datasets/dichasus"
ACQ = ROOT / "analysis/dichasus_acquisition_20261016"
OUT.mkdir(parents=True, exist_ok=True)
ART.mkdir(parents=True, exist_ok=True)
(OUT / "sources").mkdir(exist_ok=True)
FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FP = font_manager.FontProperties(fname=str(FONT))
font_manager.fontManager.addfont(str(FONT))
plt.rcParams.update({"font.family": FP.get_name(), "axes.unicode_minus": False,
                     "font.size": 11, "figure.facecolor": "white", "axes.facecolor": "white"})


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


manifest = json.loads((ACQ / "catalog_manifest.json").read_text())
files = [f for f in manifest["files"] if f["filename"].endswith(".tfrecords")]
counts = {}
for g in manifest["groups"]:
    group = g["group"]
    target = [f for f in files if group in f.get("groups", [])]
    present = [f for f in target if (BASE / f["relative_path"]).is_file()
               and (BASE / f["relative_path"]).stat().st_size == f["size"]]
    counts[group] = {"target": len(target), "present": len(present)}

archive = BASE / "dichasus-adxx/context/pointcloud/pointclouds.tar.xz"
glbs = sorted((BASE / "dichasus-dcxx/context/3dmodel").glob("*.glb"))
protected = [archive, *glbs]
before = {str(p.relative_to(ROOT)): (p.stat().st_size, p.stat().st_mtime_ns) for p in protected}
record = {"scope": "visualization only; no CSI tensor reads, training, scoring or ray tracing",
          "client_date": "2026-10-29", "counts": counts, "pointclouds": {}, "geometry": {}, "photos": []}


def sample_clouds():
    cache = ART / "display_samples.npz"
    if cache.exists():
        data = np.load(cache)
        record["pointclouds"] = json.loads((OUT / "pointcloud_display.json").read_text())
        return {k: data[k] for k in data.files}
    result = {}
    with tarfile.open(archive, mode="r|xz") as tar:
        for member in tar:
            if not member.name.endswith(".pts"):
                continue
            f = tar.extractfile(member)
            declared = int(f.readline())
            stride = 12
            rows = []
            seen = 0
            for i, line in enumerate(f):
                seen += 1
                if i % stride == 0:
                    rows.append(line)
            cloud = np.loadtxt(io.BytesIO(b"".join(rows)), dtype=np.float32)
            name = Path(member.name).stem
            result[name] = cloud
            record["pointclouds"][name] = {"declared": declared, "seen": seen,
                "display_sample": len(cloud), "stride": stride,
                "crop_xyz_m": [[-30, 16], [-34, 12], [0.15, 7.5]],
                "interpretation": "display crop only; no raw points deleted; both scans correspond to no-wall configurations"}
            assert declared == seen
            print("sampled", name, seen, len(cloud), flush=True)
    np.savez_compressed(cache, **result)
    (OUT / "pointcloud_display.json").write_text(json.dumps(record["pointclouds"], ensure_ascii=False, indent=2))
    return result


clouds = sample_clouds()
spec = json.loads((ROOT / "analysis/audits/factory_scene_20261023/v3_spec.json").read_text())
arrays = np.array([a["location"] for a in spec["antennas"]])
factory_variant_counts = {}
for entry in spec["tfrecords"]:
    filename = Path(entry["filename"]).name
    if (BASE / "dichasus-adxx" / filename).exists():
        variant = entry["configurationVariant"]
        factory_variant_counts[variant] = factory_variant_counts.get(variant, 0) + 1
record["factory_variant_counts"] = factory_variant_counts


def factory(ax, key, top=False):
    data = clouds[key]
    xyz = data[:, :3]
    keep = ((xyz[:, 0] >= -30) & (xyz[:, 0] <= 16) & (xyz[:, 1] >= -34)
            & (xyz[:, 1] <= 12) & (xyz[:, 2] >= .15) & (xyz[:, 2] <= (3.5 if top else 7.5)))
    pts = xyz[keep]
    rgb = np.clip(data[keep, 4:7] / 255, 0, 1)
    rgb = np.minimum(.15 + .85 * rgb, 1)
    if top:
        order = np.argsort(pts[:, 2])
        pts, rgb = pts[order], rgb[order]
        ax.scatter(pts[:, 0], pts[:, 1], c=rgb, s=.65, linewidths=0, rasterized=True)
        ax.scatter(arrays[:, 0], arrays[:, 1], marker="^", c="#D94828", s=85, edgecolors="white", linewidths=.6)
        for i, v in enumerate(arrays):
            ax.annotate(f"A{i+1}", v[:2], xytext=(5, 5), textcoords="offset points", color="#96351C", fontsize=11)
        ax.set_aspect("equal")
        ax.set_xlabel("X（米）"); ax.set_ylabel("Y（米）")
        ax.grid(alpha=.12)
    else:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=rgb, s=.45, linewidths=0, depthshade=False, rasterized=True)
        ax.scatter(arrays[:, 0], arrays[:, 1], arrays[:, 2], marker="^", c="#D94828", s=65, depthshade=False)
        ax.set_zlim(0, 8)
        ax.set_box_aspect((46, 46, 8))
        ax.view_init(elev=35, azim=-63)
        ax.set_xlabel("X（米）", labelpad=4); ax.set_ylabel("Y（米）", labelpad=4)
        ax.set_zlabel("高度（米）", labelpad=3)
        ax.tick_params(labelsize=9, pad=0)
    ax.set_xlim(-30, 16); ax.set_ylim(-34, 12)
    ax.set_title(("第一天" if key == "day1" else "第二天") + ("：低层结构俯视" if top else "：真实扫描点云"), pad=10)


models = {}
for filename, label in [("inue-no-walls.glb", "无挡板"), ("inue-southwest-wall.glb", "南侧挡板"), ("inue-northwest-wall.glb", "北侧挡板")]:
    scene = trimesh.load(BASE / "dichasus-dcxx/context/3dmodel" / filename)
    objs = []
    for node in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node]
        mesh = scene.geometry[geom_name].copy()
        mesh.apply_transform(transform)
        v = np.asarray(mesh.vertices)
        v = v[:, [0, 2, 1]] * [1, -1, 1]
        objs.append((geom_name, v, np.asarray(mesh.faces)))
    models[label] = objs
    record["geometry"][label] = {"source": "datasets/dichasus/dichasus-dcxx/context/3dmodel/" + filename,
        "sha256": sha(BASE / "dichasus-dcxx/context/3dmodel" / filename),
        "type": "official triangle mesh, not scanned point cloud", "transform": "(x_gltf, -z_gltf, y_gltf)",
        "visualization_only_hidden_objects": [n for n, _, _ in objs if "ceiling" in n.lower()]}


def hall(ax, label, top=False):
    for name, v, faces in models[label]:
        if "ceiling" in name.lower():
            continue
        is_blocker = "blocker" in name
        floor = "floor" in name
        color = "#DE5538" if is_blocker else ("#E8EBED" if floor else "#7E929D")
        alpha = .98 if is_blocker else (.25 if floor else .42)
        poly = Poly3DCollection(v[faces], facecolors=color, edgecolors="#B94730" if is_blocker else "#89979E",
                                linewidths=.35 if is_blocker else .12, alpha=alpha)
        ax.add_collection3d(poly)
        if is_blocker and "metal" in name:
            center = v.mean(axis=0)
            ax.text(center[0], center[1], 3.6, "金属挡板", color="#A53720", fontsize=10)
    ax.set_xlim(-9, 29); ax.set_ylim(-25, 28); ax.set_zlim(-.3, 5)
    ax.set_box_aspect((38, 53, 5.3))
    ax.view_init(elev=90 if top else 38, azim=-90 if top else -50)
    ax.set_title(label + (" · 俯视" if top else " · 官方三维模型"), pad=10)
    if top:
        ax.set_axis_off()
    else:
        ax.set_xlabel("X（米）", labelpad=3); ax.set_ylabel("Y（米）", labelpad=3)
        ax.set_zticks([]); ax.tick_params(labelsize=8, pad=0)


def hall_lobby(ax, label):
    for name, v, faces in models[label]:
        if "ceiling" in name.lower():
            continue
        blocker = "blocker" in name
        floor = "floor" in name
        ax.add_collection(PolyCollection(v[faces][:, :, :2],
            facecolors="#DE5538" if blocker else ("#F0F2F3" if floor else "#C2CDD3"),
            edgecolors="#B94730" if blocker else "#8498A3", linewidths=1.3 if blocker else .35,
            alpha=1 if blocker else .35, zorder=10 if blocker else (0 if floor else 1)))
    ax.set_xlim(-8, 6); ax.set_ylim(-9, 9); ax.set_aspect("equal")
    ax.set_title(label + " · 大厅局部俯视", pad=10)
    ax.set_xlabel("X（米）"); ax.set_ylabel("Y（米）")
    ax.grid(alpha=.1); ax.tick_params(labelsize=9)


for key in ["day1", "day2"]:
    fig = plt.figure(figsize=(14, 7))
    factory(fig.add_subplot(121, projection="3d"), key)
    factory(fig.add_subplot(122), key, top=True)
    fig.suptitle("工厂开放区域｜" + ("第一天" if key == "day1" else "第二天"), fontsize=20, y=.97)
    fig.text(.5, .055, "红色三角为四个接收阵列。斜视显示高度 0.15–7.5 米；俯视显示高度 0.15–3.5 米。", ha="center", fontsize=11)
    fig.text(.5, .025, "显示范围不代表信道采样覆盖范围。原始点云完整保留。", ha="center", fontsize=11)
    fig.subplots_adjust(top=.85, bottom=.14, wspace=.17)
    fig.savefig(OUT / f"factory_{key}.png", dpi=180)
    plt.close(fig)

fig = plt.figure(figsize=(17, 10))
for i, label in enumerate(models):
    hall(fig.add_subplot(2, 3, i+1, projection="3d"), label)
    hall_lobby(fig.add_subplot(2, 3, i+4), label)
fig.suptitle("走廊—大厅｜同一环境的三种配置", fontsize=21, y=.98)
fig.text(.5, .025, "这是官方网格模型，不是扫描点云。为看清内部，显示时隐藏了天花板；红色为挡板。", ha="center", fontsize=12)
fig.subplots_adjust(top=.9, bottom=.13, wspace=.02, hspace=.12)
fig.savefig(OUT / "hall_models.png", dpi=170); plt.close(fig)

fig = plt.figure(figsize=(8, 5))
hall(fig.add_subplot(111, projection="3d"), "南侧挡板")
fig.subplots_adjust(left=0, right=1, bottom=.02, top=.91)
fig.savefig(OUT / "sources/hall_model_preview.png", dpi=150); plt.close(fig)

fig = plt.figure(figsize=(17, 12))
gs = fig.add_gridspec(2, 6)
factory(fig.add_subplot(gs[0, :3], projection="3d"), "day1")
factory(fig.add_subplot(gs[0, 3:], projection="3d"), "day2")
for i, label in enumerate(models):
    hall(fig.add_subplot(gs[1, i*2:i*2+2], projection="3d"), label)
fig.suptitle("已有信道场景：可直接查看的三维几何", fontsize=23, y=.985)
fig.text(.5, .055, "上排：工厂开放区域的两天扫描（同一场景）。下排：走廊—大厅的三种模型配置（同一场景）。", ha="center", fontsize=12)
fig.text(.5, .027, "工厂显示高度 0.15–7.5 米，显示范围不等于信道采样覆盖；走廊隐藏天花板。未补画设备或替代其他区域几何。", ha="center", fontsize=11, color="#53616B")
fig.subplots_adjust(left=.02, right=.98, top=.92, bottom=.1, hspace=.08, wspace=.05)
fig.savefig(OUT / "geometry_overview.png", dpi=175); plt.close(fig)
print("geometry rendered", flush=True)

photo_specs = [
    ("dichasus-adxx", "工厂区域（开放区及带墙记录）", "photosAll/df-view-antennas.jpg", "无墙状态有真实点云；另有带金属反射墙记录", "factory.jpg"),
    ("dichasus-dcxx", "走廊—大厅", "", "图为官方三维模型：南侧挡板配置", "hall_model_preview.png"),
    ("dichasus-005x", "金属遮挡的单房间", "photosAll/room.jpg", "未找到配套三维点云；图为官方实景照片", "blocked_room.jpg"),
    ("dichasus-015x", "单阵列实验室", "photosAll/room.jpg", "有官方二维地图；未找到配套三维点云", "lab_single.jpg"),
    ("dichasus-0d5x", "双阵列实验室", "photosAll/map.png", "图为官方二维地图；未找到配套三维点云", "lab_distributed_map.png"),
    ("dichasus-000x", "直视走廊", "photosAll/hallway.jpg", "有官方二维地图；未找到配套三维点云", "corridor.jpg"),
    ("dichasus-0c5x", "校园楼间街道", "photosAll/street.jpg", "有官方位置地图；未找到配套三维点云", "street.jpg"),
]


def get_photo(item):
    group, title, relative, note, filename = item
    url = f"https://dichasus.inue.uni-stuttgart.de/datasets/data/{group}/{relative}"
    dest = OUT / "sources" / filename
    if not dest.exists():
        result = subprocess.run(["curl", "-q", "--noproxy", "*", "--cacert", str(ACQ / "tls/dichasus-ca-bundle.pem"),
            "--connect-timeout", "6", "--max-time", "30", "--fail", "--silent", "--show-error", "--location", url, "-o", str(dest)],
            capture_output=True, text=True, timeout=35)
        error = result.stderr[-300:]
    else:
        error = ""
    try:
        with Image.open(dest) as im:
            im.load()
        status = "ok"
    except Exception:
        status = "unavailable"
    return {"group": group, "title": title, "caption": note, "url": url, "filename": filename,
            "status": status, "error": error, "sha256": sha(dest) if status == "ok" else None}


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    record["photos"] = list(pool.map(get_photo, photo_specs))
print("photos", [(p["title"], p["status"]) for p in record["photos"]], flush=True)

canvas = Image.new("RGB", (2200, 2530), "#F6F7F8")
draw = ImageDraw.Draw(canvas)
fonts = {size: ImageFont.truetype(str(FONT), size) for size in [23, 25, 28, 34, 49]}
draw.text((65, 35), "已下载信道场景｜看图选场景", fill="#172832", font=fonts[49])
draw.text((65, 105), "照片、二维地图和三维模型放在一起；图的类型已在旁边标明。", fill="#405563", font=fonts[28])
for i, photo in enumerate(record["photos"]):
    x = 55 + (i % 2) * 1090
    y = 175 + (i // 2) * 575
    draw.rectangle((x, y, x+1040, y+545), fill="white")
    if photo["status"] == "ok":
        with Image.open(OUT / "sources" / photo["filename"]) as im:
            im = ImageOps.contain(im.convert("RGB"), (1020, 388))
            canvas.paste(im, (x+10+(1020-im.width)//2, y+10+(388-im.height)//2))
    else:
        draw.text((x+40, y+160), "官方照片暂未取得", fill="#405563", font=fonts[34])
    c = counts[photo["group"]]
    draw.text((x+22, y+412), f'{i+1}. {photo["title"]}', fill="#172832", font=fonts[34])
    draw.text((x+22, y+462), photo["caption"], fill="#405563", font=fonts[25])
    draw.text((x+22, y+502), f'已有完整信道文件 {c["present"]} / 清单目标 {c["target"]}', fill="#405563", font=fonts[23])
x, y = 1145, 1900
draw.text((x+24, y+36), "工厂另有带金属反射墙的记录", fill="#172832", font=fonts[34])
for j, text in enumerate([f'已有 {factory_variant_counts.get("wall-67db", 0)} 份完整信道文件', "缺少匹配的墙体位置和三维模型", "这部分不能用无墙点云代替", "现有照片、扫描不代表全部采样范围"]):
    draw.text((x+24, y+110+j*66), text, fill="#405563", font=fonts[28])
draw.text((65, 2490), "工厂 L 形集装箱区尚未在项目内找到完整信道文件，因此未列入本页。", fill="#405563", font=fonts[25])
canvas.save(OUT / "all_scenes.png")

after = {str(p.relative_to(ROOT)): (p.stat().st_size, p.stat().st_mtime_ns) for p in protected}
assert before == after
record["protected_geometry_metadata_unchanged"] = True
record["pointcloud_archive_sha256"] = sha(archive)
record["display_note"] = "Factory display bounds intentionally omit distant/outlier candidates and high ceiling; no input data changed. Hall ceiling hidden for visibility. No measured coverage inferred from full scan."
(OUT / "atlas_manifest.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
(OUT / "atlas_zh.md").write_text("""# 已有场景看图汇总

本图册用于选择场景。没有读取信道系数、训练、评分或运行射线追踪。

- `all_scenes.png`：有完整信道文件的七个测量组。包含官方照片、双阵列实验室的二维地图和走廊的模型视图。照片不代表该组每个文件都处于相同状态。单阵列与双阵列实验室等组也不必然是独立物理环境。
- `geometry_overview.png`：工厂两天的真实点云，以及同一走廊—大厅的三种官方模型。
- `factory_day1.png`、`factory_day2.png`：工厂斜视和低层结构俯视。两天属于同一工厂开放区域，均对应官方无墙配置。红色三角是发布的四个接收阵列位置。
- `hall_models.png`：走廊—大厅斜视和俯视。三种配置是同一物理环境；南侧和北侧挡板用红色显示。

工厂原始点云分别有 4,447,022 和 4,361,829 点。显示按固定步长抽样，并裁剪到 X=-30..16 m、Y=-34..12 m。斜视显示高度 0.15..7.5 m，俯视显示高度 0.15..3.5 m。这样便于看清测量区域周边的结构。原始点云没有修改。图中未叠加实测轨迹，扫描覆盖不能当成信道测量覆盖。

走廊模型按已核验规则 `(x_glb,-z_glb,y_glb)` 转到测量坐标。为看清内部，显示时隐藏天花板。它是三角网格，不是扫描点云。

工厂带临时金属墙的状态，没有已核验匹配的墙体几何。这里没有用无墙扫描替代带墙地图。工厂 L 形集装箱区没有列入已有完整信道数据的场景。

工厂已有文件来自历史下载版本；本图册不宣布它们已经完成当前发布版的一致性准备。各文件大小、图片来源、几何身份和显示裁剪写入 `atlas_manifest.json`。

场景外观只能帮助筛选候选。它不能证明哪个场景的多径最丰富、最难预测或最能体现算法优势。

官方场景页：
""" + "\n".join(f'- [{p["title"]}](https://dichasus.inue.uni-stuttgart.de/datasets/data/{p["group"]}/)' for p in record["photos"]) + "\n")
print("atlas complete", OUT.relative_to(ROOT), flush=True)
