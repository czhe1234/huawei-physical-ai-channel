"""相位对齐的 IDW 复信道插值器（本项目核心算法，放 src/）。

====================================================================
物理动机（先物理，后插值）——每个设计选择的来由
====================================================================

复信道的构成与相位来源
----------------------
某位置 x 的 MIMO-OFDM 信道 H[m,n,s] 是若干传播路径的叠加：
    H ≈ Σ_l a_l · exp(-j·2π·f_s·τ_l) · steer_BS(θ_l,φ_l) · steer_UE(...)
- 主径（LoS 或最强反射）贡献一个总体相位 ≈ exp(-j·k·d)，k=2π/λ，d 为传播长度；
- 各多径有各自到达角(θ,φ)与时延 τ，决定天线维/子载波维的“结构”。
载频 fc≈4.9GHz → λ≈6.1cm。**位置移动几厘米，主径传播距离 d 变化几厘米，
主径相位 k·d 就转过 O(1)~O(几) 圈**。

为什么相邻点直接复数平均会相消
------------------------------
两个相距 ~2m 的训练点（实测最近邻 median 2.17m），其主径相位差 ≈ k·Δd，
Δd 可达几十厘米 → 相位差跨越多个 2π，方向近似随机。直接对复数做 IDW 平均，
这些近似随机相位的向量相互抵消（相干抵消），把本该相干叠加的主径“糊平”，
幅度被压低、角度/时延结构被破坏。这就是普通复数 IDW（对照 B）会崩的原因。

相位对齐补的是哪一项
--------------------
我们给每个邻居 i 乘一个**标量**相位因子
    H_i' = H_i · exp(+j·β·(d_i − d_q))          (d = 到基站三维距离)
把邻居“从它自己的主径相位倒推到查询点的主径相位”。β 理论值 = 2π/λ = 2π·fc/c
（fc=4.9GHz → β≈102.6 rad/m）。对齐后邻居间主径相位一致 → **相干叠加而非抵消**，
恢复幅度与主瓣。注意：
  * 这是**全局标量**相位，不改变任何单个邻居自身的 PAS/PDP（角度/时延结构不变），
    它只决定多个邻居在 IDW 求和时是加强还是抵消 → 直接改善 NMSE(C3)，
    并因避免糊平而间接保住 PAS(C1)/PDP(C2)。
  * 对单个邻居(k=1)，全局相位在 |corr|/PAS/PDP 上不可见（可被吸收），
    所以对齐只有在“合并 k≥2 个不同距离邻居”时才有意义。

哪些量随位置平滑（可插值）、哪些不可插值
----------------------------------------
- **可插值**：功率（幅度）随位置平滑（相关性随距离缓慢衰减）；角度谱/时延谱的
  “结构”在小邻域内相近。→ 幅度与谱结构适合 IDW 空间平滑。
- **不可插值**：主径的**绝对相位** k·d 随位置以波长尺度快速旋转，是“不可插值”的高频量。
  → 必须先用 exp(+jβ(d_i−d_q)) 显式补掉这一确定性快变项，剩下的慢变部分才可平均。

LoS vs NLOS / 死点
------------------
- LoS 点：主径长度≈几何直线距离 d，对齐准确。
- NLOS 点：主径经反射，真实传播长度 > 几何 d，用几何 d 对齐会有偏差 → 这是
  β 的 CV 最优值可能偏离理论 102.6 的物理原因之一（有效“平均”传播/载频）。
- 死点/深遮挡：功率≈噪声底，无稳定主径，既不该做邻居也难被预测，训练中剔除。

功率与相位分离（power/phase separation）
---------------------------------------
逐点总功率动态范围 >80dB。直接复数 IDW 时，个别强邻居会主导，且“对齐后求和”的
模长不等于目标模长 → NMSE 爆炸。故可选：用 IDW 在 **log 功率**域插值出查询点的
目标总功率，再把“对齐 IDW 得到的复方向”归一化后乘上该功率。这把“insensitive 的
功率标定”与“相位相干的方向重建”解耦，显著改善 NMSE。

进阶：逐子载波（频率相关）相位对齐
----------------------------------
单一 β 只补主径的“载波中心”相位。多径时延 τ 使不同子载波 f_s 有额外相位
exp(-j·2π·f_s·τ)。把对齐相位写成随子载波线性变化：
    φ_i(s) = β0·(d_i−d_q) + Δf·(s−s0)/c·2π·(d_i−d_q)
即对齐“载波中心相位 + 时延斜率”。可在 CV 上验证是否再涨（见 per_subcarrier 选项）。

====================================================================
实现（全部超参可调，交由 scripts/ 的 CV 选优，不写死）
====================================================================
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

C_LIGHT = 299_792_458.0
FC_THEORY = 4.9e9                     # 推断载频（数据分析.md 第10节），仅作理论 β 参照
BETA_THEORY = 2 * np.pi * FC_THEORY / C_LIGHT   # ≈102.6 rad/m
SUBCARRIER_SPACING = 60e3            # 推断子载波间隔，用于逐子载波时延对齐


@dataclass
class IDWConfig:
    """相位对齐 IDW 的超参（用 CV 挑最优，勿写死）。"""
    k: int = 8                         # 邻居数
    p: float = 2.0                     # IDW 幂次 w=1/dist^p
    beta: float = BETA_THEORY          # 相位对齐系数 rad/m（默认理论值，实际由 CV 扫）
    align_phase: bool = True           # 是否做相位对齐（关=普通复数 IDW，对照 B）
    power_separation: bool = False     # 是否 log 功率域插值总功率再乘回（幅相分离）
    per_subcarrier: bool = False       # 进阶：逐子载波时延对齐
    shrink: float = 1.0                # 幅度收缩系数 α（见下）：最终输出整体乘 α
    fc: float = FC_THEORY              # 逐子载波对齐用的中心载频
    scs: float = SUBCARRIER_SPACING    # 子载波间隔
    eps: float = 1e-3                  # 距离下限，避免除零

    # 幅度收缩 α 的物理动机（实测结论）：
    # 邻近点的“绝对复相位”在本数据里空间不可预测（k=1 扫 β 实测 mean Re(ρ)≈0，
    # 无以理论 β 为中心的峰；估计与真实的复相关≈0）。当相位不可恢复时，NMSE 的
    # 最优全局幅度不是“满幅度”而是收缩后的幅度（极限 α→0 时 NMSE→1，等于零估计）。
    # 因 C1/C2 尺度不变、只有 C3 随 α 变，用 CV 选 α 平衡 C3 增益与 C2 噪底衰减，
    # 通常最优 α≈0.3~0.5。这才是本题真正把总分 C 抬起来的杠杆，而非相位对齐。

    def label(self) -> str:
        tag = []
        if self.align_phase:
            tag.append(f"β={self.beta:.1f}")
            if self.power_separation:
                tag.append("pwsep")
            if self.per_subcarrier:
                tag.append("perSC")
        if self.shrink != 1.0:
            tag.append(f"α={self.shrink:.2f}")
        base = "alignIDW" if self.align_phase else "naiveIDW"
        return f"{base}(k={self.k},p={self.p}{',' + ','.join(tag) if tag else ''})"


class PhaseAlignedIDW:
    """相位对齐的 IDW 复信道插值器。

    fit(lib_pos, lib_H)：登记库点坐标与信道（信道保留在内存/映射，按邻居惰性取用）。
    predict(query_pos)：对每个查询点做 kNN + 相位对齐 + IDW，返回 (Q,M,N,S) complex64。
    """

    def __init__(self, bs_xyz, config: IDWConfig | None = None):
        self.bs = np.asarray(bs_xyz, dtype=np.float64)
        self.cfg = config or IDWConfig()
        self._pos = None
        self._H = None
        self._d_lib = None      # 库点到基站 3D 距离
        self._logpw = None      # 库点 log 总功率（幂相分离用）

    # ---- 几何 ----
    def _dist_to_bs(self, pos: np.ndarray) -> np.ndarray:
        return np.linalg.norm(pos - self.bs, axis=1)

    def fit(self, lib_pos: np.ndarray, lib_H) -> "PhaseAlignedIDW":
        self._pos = np.asarray(lib_pos, dtype=np.float64)
        self._H = lib_H                                   # 可为 ndarray 或 mmap
        self._d_lib = self._dist_to_bs(self._pos)
        # 始终预计算库点 log 功率（幂相分离用；与超参无关，便于 CV 扫描时复用同一 fit）
        P = self._H.shape[0]
        pw = np.empty(P)
        for a in range(0, P, 100):
            b = min(a + 100, P)
            blk = np.asarray(self._H[a:b])
            pw[a:b] = (np.abs(blk) ** 2).sum(axis=(1, 2, 3))
        self._logpw = np.log(np.maximum(pw, 1e-30))
        return self

    def _neighbors(self, q_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """返回 (邻居下标 [k], 2D 距离 [k])，按距离升序。"""
        d2 = np.linalg.norm(self._pos[:, :2] - q_xy, axis=1)
        k = min(self.cfg.k, len(d2))
        idx = np.argpartition(d2, k - 1)[:k]
        idx = idx[np.argsort(d2[idx])]
        return idx, d2[idx]

    def predict(self, query_pos: np.ndarray) -> np.ndarray:
        """对 (Q,3) 查询点生成 (Q,M,N,S) complex64。"""
        assert self._H is not None, "先 fit()"
        query_pos = np.asarray(query_pos, dtype=np.float64)
        Q = query_pos.shape[0]
        M, N, S = self._H.shape[1], self._H.shape[2], self._H.shape[3]
        out = np.empty((Q, M, N, S), dtype=np.complex64)
        cfg = self.cfg
        dq_all = self._dist_to_bs(query_pos)

        # 逐子载波对齐用的频率轴（相对中心）：f_s = fc + (s - (S-1)/2)*scs
        s_axis = (np.arange(S) - (S - 1) / 2.0) * cfg.scs if cfg.per_subcarrier else None
        for qi in range(Q):
            idx, dist2d = self._neighbors(query_pos[qi, :2])
            w = 1.0 / np.maximum(dist2d, cfg.eps) ** cfg.p
            w = w / w.sum()
            dq = dq_all[qi]

            # 向量化合并 k 个邻居：一次取出 (k,M,N,S)，相位与权重广播后求和
            Hk = np.asarray(self._H[idx]).astype(np.complex128)   # (k,M,N,S)
            if cfg.align_phase:
                dd = self._d_lib[idx] - dq                        # (k,)
                if cfg.per_subcarrier:
                    # 载波中心相位(β·Δd) + 时延斜率(2π·f_s·Δd/c)，随子载波变化 → (k,S)
                    phase = cfg.beta * dd[:, None] + (2 * np.pi * dd[:, None] * s_axis[None, :] / C_LIGHT)
                    Hk = Hk * np.exp(1j * phase)[:, None, None, :]
                else:
                    Hk = Hk * np.exp(1j * cfg.beta * dd)[:, None, None, None]
            acc = np.tensordot(w, Hk, axes=(0, 0))                # (M,N,S)

            if cfg.align_phase and cfg.power_separation:
                # 幅相分离：方向来自对齐 IDW，模长来自 log 功率 IDW 预测
                cur_pw = float((np.abs(acc) ** 2).sum())
                if cur_pw > 1e-30:
                    pred_logpw = float((w * self._logpw[idx]).sum())
                    scale = np.sqrt(np.exp(pred_logpw) / cur_pw)
                    acc = acc * scale
            if cfg.shrink != 1.0:
                acc = acc * cfg.shrink        # 幅度收缩（相位不可恢复时降低 NMSE）
            out[qi] = acc.astype(np.complex64)
        return out
