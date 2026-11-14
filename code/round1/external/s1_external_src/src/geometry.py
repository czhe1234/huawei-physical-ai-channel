"""点云几何特征（LoS/NLOS 判定 + 遮挡程度），供门控 IDW / 几何特征回归用。

物理动机
--------
信道的角度谱(PAS)/时延谱(PDP)由传播几何决定：LoS 点有直达主径→PAS 单主瓣、
PDP 首径主导；NLOS 点只有反射/散射径→PAS 弥散、PDP 拖尾。IDW 若把 LoS 与 NLOS
邻居的谱糊在一起，就破坏了这个结构（PAS/PDP 占评分 0.8）。故先用点云判定每个
位置的 LoS/NLOS，再让 IDW 只在同类邻居内插值。

实现：2.5D 高度图 + 射线遮挡
---------------------------
BS 在 25m 高、UE 在 1.5m。遮挡由建筑立面决定，用俯视高度图（每格存最高建筑 z）。
从 BS 到查询点连 3D 线段，沿途在每格比较"建筑高度 vs 射线高度"，建筑高出射线
(含余隙) 即判遮挡。这比纯 2D 投影正确：射线可从矮楼上方飞过。

实测验证（scripts/geom_probe.py，2000 训练点）：
- LoS 21% / NLOS 79%，与前人 Sionna 文档测试集 zero-path 21.6% 独立吻合；
- LoS 中位功率比 NLOS 高 2.59 dec（≈390×）；PDP 首径占比 0.72 vs 0.63；
- 遮挡体素数 vs log 功率 r=−0.27。→ 几何确为 PAS/PDP 的强预测因子。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SceneGeometry:
    """场景高度图 + BS 坐标，提供任意位置的 LoS/遮挡查询。"""
    hmap: np.ndarray          # (W,H) 每格最高建筑 z
    origin: np.ndarray        # (2,) 高度图 (0,0) 格对应的世界 xy
    cell: float               # 格边长 (m)
    bs: np.ndarray            # (3,) 基站坐标

    @classmethod
    def build(cls, scene_xyz: np.ndarray, bs_xyz, cell: float = 2.0,
              z_min: float = 1.0) -> "SceneGeometry":
        b = scene_xyz[scene_xyz[:, 2] >= z_min]
        xy = b[:, :2].astype(np.float64); z = b[:, 2].astype(np.float32)
        origin = xy.min(0)
        ij = np.floor((xy - origin) / cell).astype(np.int64)
        W = int(ij[:, 0].max()) + 1; H = int(ij[:, 1].max()) + 1
        hmap = np.zeros((W, H), dtype=np.float32)
        np.maximum.at(hmap.reshape(-1), ij[:, 0] * H + ij[:, 1], z)
        return cls(hmap=hmap, origin=np.asarray(origin, np.float64),
                   cell=float(cell), bs=np.asarray(bs_xyz, np.float64))

    def occlusion(self, pts_xyz: np.ndarray, n_samp: int = 300,
                  skip_near: float = 4.0, clear: float = 1.0) -> np.ndarray:
        """返回每点被遮挡的采样数 (越大遮挡越重, 0=LoS)。pts_xyz: (Q,3)。"""
        pts = np.asarray(pts_xyz, np.float64)
        if pts.ndim == 1:
            pts = pts[None, :]
        W, H = self.hmap.shape
        blocked = np.zeros(len(pts), dtype=np.int32)
        t = np.linspace(0.0, 1.0, n_samp)
        for i, p in enumerate(pts):
            d = np.linalg.norm(p[:2] - self.bs[:2])
            if d < 1e-6:
                continue
            seg = self.bs[None, :] + t[:, None] * (p - self.bs)[None, :]
            keep = (t * d > skip_near) & (t * d < d - skip_near)
            seg = seg[keep]
            if not len(seg):
                continue
            ij = np.floor((seg[:, :2] - self.origin) / self.cell).astype(np.int64)
            ok = (ij[:, 0] >= 0) & (ij[:, 0] < W) & (ij[:, 1] >= 0) & (ij[:, 1] < H)
            ij = ij[ok]; ray_z = seg[ok, 2]
            if len(ij):
                bh = self.hmap[ij[:, 0], ij[:, 1]]
                blocked[i] = int((bh > (ray_z + clear)).sum())
        return blocked

    def is_los(self, pts_xyz: np.ndarray, **kw) -> np.ndarray:
        """LoS 布尔数组 (True=直达可见)。"""
        return self.occlusion(pts_xyz, **kw) == 0
