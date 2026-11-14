"""Generate real-data figures for the Round2 introductory PPT page.

The PAS/PDP transforms intentionally follow ``code/common/official_metrics.py``.
The output directory is an analysis visualization directory, not a submission
artifact.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import fft as scipy_fft
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import ProjectPaths


PHASE_PAIR = (489, 1830)
SPECTRUM_POINT = 1624
USER_PORT = 3  # zero-based; displayed as port 4


def _set_fonts() -> None:
    """Prefer a CJK font when one is installed, while remaining portable."""
    from matplotlib import font_manager

    # ``NotoSansCJK-Regular.ttc`` is a collection font.  Matplotlib may only
    # register its first face (Japanese) when loading the TTC directly, so
    # also register the standalone Droid fallback font when it is available.
    # The latter contains the Chinese glyphs used in the plot annotations.
    for cjk_file in (
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ):
        if cjk_file.is_file():
            font_manager.fontManager.addfont(str(cjk_file))
    candidates = (
        # The TTC is registered by Matplotlib under its first face name on
        # this system.  That face still contains the unified CJK glyphs and
        # also covers the Latin/math labels used below.
        "Noto Sans CJK JP",
        "Noto Sans CJK SC",
        "Droid Sans Fallback",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
        "DejaVu Sans",
    )
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def _load(paths: ProjectPaths) -> tuple[np.ndarray, np.ndarray]:
    data_dir = paths.round2
    positions = np.load(data_dir / "Round2_Train_Pos.npy")
    channels = np.load(data_dir / "Round2_Train_Channel.npy", mmap_mode="r")
    if positions.shape != (4000, 3):
        raise ValueError(f"unexpected position shape: {positions.shape}")
    if channels.shape != (4000, 256, 4, 192):
        raise ValueError(f"unexpected channel shape: {channels.shape}")
    if channels.dtype != np.complex64:
        raise ValueError(f"unexpected channel dtype: {channels.dtype}")
    return positions, channels


def _phase_fig(out: Path, channels: np.ndarray, positions: np.ndarray) -> dict:
    i, j = PHASE_PAIR
    q1, q2 = positions[i], positions[j]
    distance = float(np.linalg.norm(q2 - q1))
    phase1 = np.angle(np.asarray(channels[i, :, USER_PORT, :]))
    phase2 = np.angle(np.asarray(channels[j, :, USER_PORT, :]))
    phase_delta = np.angle(
        np.asarray(channels[j, :, USER_PORT, :])
        * np.conj(np.asarray(channels[i, :, USER_PORT, :]))
    )

    cmap = "twilight"
    extent = (0.5, 192.5, 256.5, 0.5)
    ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    tick_labels = [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"]

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.5), constrained_layout=True)
    images = []
    for ax, value, label in zip(
        axes,
        (phase1, phase2),
        ("点 1", "点 2"),
    ):
        image = ax.imshow(
            value,
            aspect="auto",
            cmap=cmap,
            vmin=-np.pi,
            vmax=np.pi,
            extent=extent,
            interpolation="nearest",
        )
        images.append(image)
        ax.set_xlabel("子载波编号")
        ax.set_ylabel("基站天线编号")
        ax.set_title(label, fontsize=14, pad=8)
        ax.set_xticks([1, 48, 96, 144, 192])
        ax.set_yticks([1, 64, 128, 192, 256])
    cbar = fig.colorbar(images[0], ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("相位 / rad")
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels)
    fig.suptitle(f"两点距离 = {distance:.6f} m；用户天线端口 = {USER_PORT + 1}", fontsize=15)
    fig.savefig(out / "邻近点相位对比.png", dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 6.5), constrained_layout=True)
    image = ax.imshow(
        phase_delta,
        aspect="auto",
        cmap=cmap,
        vmin=-np.pi,
        vmax=np.pi,
        extent=extent,
        interpolation="nearest",
    )
    ax.set_xlabel("子载波编号")
    ax.set_ylabel("基站天线编号")
    ax.set_xticks([1, 48, 96, 144, 192])
    ax.set_yticks([1, 64, 128, 192, 256])
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("相位差 / rad")
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels)
    ax.text(
        0.5,
        1.02,
        f"两点距离 = {distance:.6f} m；用户天线端口 = {USER_PORT + 1}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=13,
    )
    fig.savefig(out / "邻近点相位差.png", dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return {
        "point_1_index_zero_based": i,
        "point_2_index_zero_based": j,
        "point_1_index_one_based": i + 1,
        "point_2_index_one_based": j + 1,
        "point_1_coordinate": q1.tolist(),
        "point_2_coordinate": q2.tolist(),
        "euclidean_distance_m": distance,
        "user_port_zero_based": USER_PORT,
        "user_port_one_based": USER_PORT + 1,
        "point_1_total_energy": float(np.sum(np.abs(channels[i]) ** 2)),
        "point_2_total_energy": float(np.sum(np.abs(channels[j]) ** 2)),
    }


def _official_pas_pdp(channel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return official-transform PAS and PDP powers for one point."""
    shaped = channel.reshape(2, 16, 8, 4, 192)
    pas = np.square(np.abs(scipy_fft.fft2(shaped, axes=(1, 2))), dtype=np.float32)
    pdp = np.square(np.abs(scipy_fft.ifft(channel, axis=-1)), dtype=np.float32)
    return pas, pdp


def _spectrum_fig(out: Path, channels: np.ndarray, positions: np.ndarray) -> dict:
    point = SPECTRUM_POINT
    channel = np.asarray(channels[point])
    pas, pdp = _official_pas_pdp(channel)

    # Aggregate only after the official 2-D array transform.  This preserves
    # the same PAS definition used by the competition metric.
    pas_2d = pas.sum(axis=(0, 3, 4))
    pas_2d /= max(float(pas_2d.sum()), np.finfo(np.float32).tiny)
    pdp_1d = pdp.sum(axis=(0, 1))
    pdp_1d /= max(float(pdp_1d.sum()), np.finfo(np.float32).tiny)

    fig, ax = plt.subplots(figsize=(8.2, 5.6), constrained_layout=True)
    image = ax.imshow(
        pas_2d,
        origin="lower",
        aspect="auto",
        cmap="magma",
        interpolation="nearest",
        extent=(0.5, 16.5, 0.5, 8.5),
    )
    ax.set_xlabel("水平波束编号")
    ax.set_ylabel("垂直波束编号")
    ax.set_xticks([1, 4, 8, 12, 16])
    ax.set_yticks([1, 2, 4, 6, 8])
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("归一化功率")
    fig.savefig(out / "真实PAS.png", dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    eps = 1e-12
    pdp_db = 10.0 * np.log10(np.maximum(pdp_1d, eps))
    pdp_db -= float(np.max(pdp_db))
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    ax.plot(np.arange(1, 193), pdp_db, color="#1f5a9d", linewidth=1.4)
    ax.set_xlabel("时延 bin")
    ax.set_ylabel("归一化功率 / dB")
    ax.set_xlim(1, 192)
    ax.set_xticks([1, 32, 64, 96, 128, 160, 192])
    ax.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.8)
    fig.savefig(out / "真实PDP.png", dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    return {
        "point_index_zero_based": point,
        "point_index_one_based": point + 1,
        "coordinate": positions[point].tolist(),
        "total_energy": float(np.sum(np.abs(channel) ** 2)),
        "pas_peak_fraction": float(np.max(pas_2d)),
        "pdp_peak_fraction": float(np.max(pdp_1d)),
        "pas_definition": "reshape [2,16,8,4,192], 2-D FFT over [16,8], power, sum over polarization/RX/subcarrier",
        "pdp_definition": "IFFT over 192 subcarriers, power, sum over BS antenna/RX port",
    }


def main() -> None:
    _set_fonts()
    paths = ProjectPaths.load()
    out = paths.analysis / "visualizations" / "ppt_page2_real_data"
    out.mkdir(parents=True, exist_ok=True)
    positions, channels = _load(paths)
    metadata = {
        "data": {
            "positions": "datasets/round2/Round2_Train_Pos.npy",
            "channels": "datasets/round2/Round2_Train_Channel.npy",
            "channel_shape": list(channels.shape),
            "channel_dtype": str(channels.dtype),
        },
        "phase": _phase_fig(out, channels, positions),
        "spectrum": _spectrum_fig(out, channels, positions),
    }
    (out / "图像元数据.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"输出目录: {out}")


if __name__ == "__main__":
    main()
