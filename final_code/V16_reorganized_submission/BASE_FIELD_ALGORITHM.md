# 官方 H 基础场算法详解

## 1. 这一部分在完整算法中的位置

基础场模块的任务不是直接猜出所有传播路径，而是先生成一个数值稳定、相位连续、端口结构完整的复信道，再用 PAS、PDP 和 angle-delay 规律修正它的频谱结构。

```text
官方 Train H + Train/Test 坐标 + BS位置
                    |
                    v
          相位辨识与17专家插值
                    |
                    v
             复数岭回归融合
                    |
                    v
       donor / lifecycle / 频谱校准
                    |
                    v
         LinkGate与可靠性门控投影
                    |
                    v
               基础复信道 H_base
```

基础场输出仍然是完整复数张量：

```text
H_base.shape = (N, 256, 4, 192)
H_base.dtype = complex64
```

其中 `256` 是基站阵列端口，`4` 是 UE 端口，`192` 是频率采样点。后续 ComponentState 模型会继续使用物理传播结构修正该基础场，因此基础场不是最终 V16 输出，但它决定了最终结果的复相位、连续背景、端口关系和大部分总功率。

## 2. 为什么不能直接插值原始复信道

复信道可以写成：

$$
H = A\,\mathrm{e}^{j\phi}
$$

位置只变化几厘米，传播距离造成的相位 $\phi$ 就可能旋转很多。假设两个邻居都具有正确的单位幅度：

$$
H_1 = 1, \qquad H_2 = \mathrm{e}^{j\pi} = -1
$$

如果直接平均：

$$
\frac{H_1 + H_2}{2} = 0
$$

算法会把两个都正确的强信号平均成零。这个问题不是普通噪声，而是不同空间位置使用了不同的相位参考系。

基础场首先估计相位随空间传播距离和频率的变化，把所有邻居搬运到目标点的统一相位参考系，再做复数插值。

## 3. 从官方 Train H 辨识空间载波相位

### 3.1 邻点复相关

对同一站点的附近 Train 点对，代码计算每个频点上的归一化复相关。归一化会去掉绝对功率尺度，主要保留两点之间的相位差及相关强度：

```text
C_ij(f) = normalized_complex_correlation(H_i(f), H_j(f))
```

同时计算两点到对应 BS 的径向距离差：

```text
Delta_r_ij = r_i - r_j
```

相位模型为：

```text
beta(f) = beta + gamma * f
```

其中：

- `beta` 表示中心频率附近每米传播距离对应的相位旋转量。
- `gamma` 表示这个旋转量随子载波位置的线性变化。
- `f` 是从 `-0.5` 到 `0.5` 的归一化子载波坐标。

代码先在 `beta=40~180` 的范围粗搜索，再搜索 `gamma=-0.3~0.3`，最后细化 `beta`。目标是让大量邻点对经过距离补偿后的复相关尽可能同相叠加。每个站点独立得到一组 `beta/gamma`。

实现见 [phase_identification.py](code/round2/physical_ai_submission/interpolation_field/phase_identification.py)。

### 3.2 把邻居搬运到目标点

对目标位置 `q` 和邻居 `i`：

```text
r_q = distance(query, BS)
r_i = distance(neighbor_i, BS)
Delta_r_i = r_i - r_q

phase_i(f) = exp(j * Delta_r_i * (beta + gamma*f))
H_i_aligned(f) = H_i(f) * phase_i(f)
```

`H_i_aligned` 不再表示邻居原位置的相位，而是邻居信道被搬运到目标距离后的候选值。随后对这些对齐结果做 IDW 或局部仿射拟合，复数相消会显著减少。

实际计算位于 [spatial_experts.py](code/round2/physical_ai_submission/interpolation_field/spatial_experts.py#L135)。

## 4. 17个复信道专家

算法不押注单一空间假设，而是为每个目标点生成17个完整复信道候选。

| 序号 | 专家 | 数量 | 核心假设 |
| --- | --- | ---: | --- |
| 1 | 最近邻 | 1 | 最近 Train H 已足够接近，而且不应与其他点平均 |
| 2 | 原始 IDW | 4 | 原始复信道在极局部范围可直接按距离平滑 |
| 3 | 径向相位对齐 IDW | 4 | 去掉主要径向相位旋转后，复信道在局部平滑 |
| 4 | 径向局部仿射场 | 4 | 对齐后的复信道随二维位置近似线性变化 |
| 5 | 阵列方向补偿仿射场 | 4 | 除径向相位外，还存在与阵列方向有关的空间相位变化 |

### 4.1 最近邻

```text
H_pred = H_nearest
```

它完全避免复数平均。在传播状态突变边界上，最近邻有时比平滑模型更可靠。

### 4.2 四个原始 IDW

分别使用：

```text
k = 4, 8, 16, 24
```

一般形式为：

```text
w_i proportional_to 1 / distance_i^p
H_pred = sum_i w_i * H_i
```

这些专家保留为对照和补充。虽然它们可能发生相消，但在极近邻、相位变化较小或相位模型不准确的区域仍可能有效。

### 4.3 四个径向相位对齐 IDW

同样使用 `k=4/8/16/24`，但先执行：

```text
H_i -> H_i_aligned
```

再进行距离加权：

```text
H_pred = sum_i w_i * H_i_aligned
```

这是基础场最直接的物理修正，解决传播距离不同造成的快速相位旋转。

### 4.4 四个径向局部仿射场

配置为：

```text
(k, ridge) = (8, 0.1), (16, 1.0), (24, 0.1), (32, 0.1)
```

它不只做常数平均，而是在目标点附近拟合：

```text
H(x, y) approximately a + b_x * Delta_x + b_y * Delta_y
```

正则项用于抑制邻点几何分布不良时的数值爆炸。不同 `k/ridge` 对应不同的空间尺度和偏差方差折中。

### 4.5 四个阵列方向补偿仿射场

使用：

```text
horizontal_axis = 0 degrees or 90 degrees
vertical_scale = 1.5*pi or 8*pi
k = 24
ridge = 0.3
```

它在径向补偿之外，根据目标与邻点相对于 BS 的方向变化，对阵列维添加 steering phase。其目的是描述同一传播分量在空间移动时沿水平、垂直阵列发生的相位斜率变化。

17个专家的配置定义见 [expert_bank.py](code/round2/physical_ai_submission/interpolation_field/expert_bank.py#L19)。

## 5. 为什么需要17个专家

不同区域可能遵循不同的局部规律：

| 空间状态 | 可能更合适的专家 |
| --- | --- |
| 极近邻、传播状态稳定 | 小 `k` 相位对齐 IDW |
| 信道变化平滑、邻点充足 | 大 `k` 相位对齐 IDW |
| 存在明显二维梯度 | 局部仿射场 |
| 阵列相位随方向快速变化 | steering 仿射场 |
| 传播边界或邻点来自不同状态 | 最近邻或小邻域专家 |
| 相位辨识局部失配 | 原始 IDW 或最近邻可能成为回退 |

因此17个专家不是重复计算同一种结果，而是构造一个有限但互补的复信道候选空间。

## 6. 复数岭回归融合

### 6.1 为什么不能手工平均17个专家

每个专家在不同站点、极化和 UE 端口上的误差并不相同。固定平均会让错误专家污染本来正确的候选。因此训练阶段使用官方 Train H 学习复数融合权重：

```text
H_mix = sum_e w_e * H_e
```

这里 `w_e` 是复数，既可以调整幅度，也可以调整相位。

### 6.2 严格 OOF 训练

训练点不能用自己的 H 生成自己的专家输入。代码先按空间折隐藏一批 Train 点，只用其余点生成这批点的17个 OOF 候选，然后用隐藏点真实官方 H 拟合融合权重。

这样学习到的是“面对未见空间区域时怎样融合专家”，而不是记住训练点。

### 6.3 分站点和端口组学习

权重按以下粒度独立学习：

```text
2个站点 x 8个收发端口组 x 17个专家
```

8个组来自：

```text
2个BS极化组 x 4个UE端口
```

一个权重控制该组中完整的阵列、频率复信道，因此仍保留结构共享，不是为1024条链路各自无约束拟合。

推理时使用 `einsum` 一次完成复数融合。训练实现见 [carrier.py](code/round2/physical_ai_submission/training/carrier.py)，部署融合见 [base_channel_builder.py](code/round2/physical_ai_submission/ai_enhancement/base_channel_builder.py#L454)。

## 7. 为什么复信道融合后还需要频谱修正

相位对齐与复数岭回归擅长恢复连续复场，但空间插值会自然地趋向平滑。真实传播却存在离散状态变化：

- 一条反射路径越过边界后突然消失。
- 新路径在某个区域突然出现。
- angle 或 delay 峰发生移动，而不是连续展宽。
- 相邻区域的总功率相似，但能量分配到不同端口或不同峰。

所以算法把 `H_mix` 看作“复数载体”，再用频谱模块校正它的路径结构。频谱模块主要决定能量应该落在哪里，原始复场主要提供相位和端口连续性。

## 8. 第一级：从官方 Train H 提取频谱监督

`build_official_spectra()` 从官方 Train H 计算：

| 表示 | 含义 |
| --- | --- |
| full PAS | 信道能量在阵列角度格子上的分布 |
| full PDP | 信道能量在 delay 格子上的分布 |
| collapsed angle-delay | 合并端口后的角度与时延联合能量 |
| grouped PAS | `32 x 256` 的分组角度谱 |
| grouped PDP | `8 x 192` 的分组时延谱 |
| total power | 每个点的信道总能量 |
| PAS/PDP entropy | 频谱是集中还是分散 |

分组频谱使用平方根能量保存。平方根空间能降低极强峰对损失的统治程度，也更接近最终谱相似度关注的相对形状。

实现见 [base_channel_builder.py](code/round2/physical_ai_submission/ai_enhancement/base_channel_builder.py#L267)。

## 9. 第二级：donor选择与分组频谱校准

### 9.1 donor不是简单最近邻

目标点附近的几何近邻未必处于相同传播状态。因此 donor 模块分别为 PAS 和 PDP 学习训练点权重：

```text
PAS_target = sum_i w_i_pas * PAS_train_i
PDP_target = sum_i w_i_pdp * PDP_train_i
```

PAS 和 PDP 使用不同权重，因为“角度结构相似的点”不一定也是“时延结构最相似的点”。部署权重保存在 `deployment_donor_weights.npz`。

### 9.2 分站点低维校准

donor 组合后的 grouped PAS/PDP 还会经过按站点训练的低维校准器，修正系统性的峰宽、能量偏差和端口组偏差。

随后目标频谱以有界比例投影到当前复信道，而不是直接替换完整 H。donor 逻辑见 [donor.py](code/round2/physical_ai_submission/training/donor.py)，频谱组合见 [grouped_spectrum.py](code/round2/physical_ai_submission/interpolation_field/grouped_spectrum.py)。

## 10. 第三级：地图风险与路径生命周期

地图和坐标首先被转换为连续风险量，主要包括：

- 路径穿墙比例。
- 路径代价。
- geodesic ratio，即可行路线长度与直线距离之比。
- 局部占用和遮挡。
- 周向视野被遮挡比例。
- 平均净空。

这些特征在每个站点内标准化并加权形成 `physical_risk`。代码使用站点内训练风险的 `25%/50%/75%` 分位点，将连续风险量化为：

```text
clear / light / medium / heavy
```

在每个 `site x regime` 内，官方 Train angle-delay 支撑被聚成传播原型。对于目标点，模型结合几何状态和当前插值场，判断各原型的概率，从而描述：

- 当前峰继续存在。
- 当前峰应该消失。
- 新峰应该出现。
- 峰应该移动到相邻 angle-delay 区域。

这就是“生命周期”的含义。它不是给每条峰人工写规则，而是先按传播环境组织官方 Train 中观察到的频谱状态，再生成保守的目标支撑。

实现见 [lifecycle_core.py](code/round2/physical_ai_submission/ai_enhancement/lifecycle_core.py)。

## 11. 第四级：周期频谱校准器

频谱校准器读取：

- 两套物理先验分支的 PAS/PDP 均值。
- 波动、有效次数和候选数量。
- 几何上下文。
- 站点、链路组和频谱切片身份。
- 当前基础场频谱。

网络不从零生成任意频谱，而是在平方根概率空间预测有界残差：

```text
sqrt(P_target) = sqrt(P_prior) + bounded_residual
```

角度格子本身是周期坐标；由离散傅里叶变换得到的 delay 格子也按周期索引处理。因此网络使用周期 padding/卷积，避免 `bin 0` 和最后一个 bin 在边界处被错误视为完全不相邻。

实现见 [spectral_calibrator.py](code/round2/physical_ai_submission/ai_enhancement/spectral_calibrator.py) 和 [spectral_field.py](code/round2/physical_ai_submission/ai_enhancement/spectral_field.py)。

## 12. 第五级：联合频谱模型

单独校准 PAS 和 PDP 可能产生冲突。例如某个修改提高了 PAS，却要求与现有 PDP 不兼容的完整 H。

联合模型同时读取：

- Base 与 Scatter 两种频谱解释。
- 两种解释的一致程度和差异。
- 不确定性、覆盖度和候选数。
- 频谱熵和峰集中度。
- 几何上下文。
- 前一级校准后的 PAS/PDP 端点。

输出是联合调整后的 PAS/PDP 端点和可靠性描述。它的作用是判断不同传播解释应该如何协同，而不是简单把两套谱平均。

实现见 [joint_spectral_model.py](code/round2/physical_ai_submission/ai_enhancement/joint_spectral_model.py)。

## 13. 第六级：逐链路 LinkGate

每个点包含：

```text
256 x 4 = 1024条复信道链路
```

不同链路对 PDP 修正的接受程度不同。LinkGate 为每条链路读取：

- 当前基础 PDP。
- 候选物理/component PDP。
- 二者的平方根差值和乘积。
- 熵与余弦一致度。
- 点级上下文。
- 阵列极化、H/V坐标和 UE 端口身份。

MLP 输出：

```text
alpha_link in [0, 1]
PDP_target = (1-alpha_link)*PDP_base + alpha_link*PDP_component
```

模型的初始偏置为 `-3`，默认倾向较小动作。只有训练证据充分时，某条链路才会明显采用候选 PDP。

得到目标后，精确 PDP 投影器修改完整复信道，同时保持当前 PAS 和点总功率。实现见 [link_gate.py](code/round2/physical_ai_submission/ai_enhancement/link_gate.py)、[link_gate_model.py](code/round2/physical_ai_submission/ai_enhancement/link_gate_model.py) 和 [component_delay_field.py](code/round2/physical_ai_submission/ai_enhancement/component_delay_field.py)。

## 14. 第七级：PAS可靠性门

最终 PAS 修正同时参考四个来源：

```text
calibrated PAS
joint PAS
Base physical PAS
Scatter physical PAS
```

对每个点、UE端口和频率切片计算：

1. 四个来源之间的 spread。
2. 各来源相对当前基础 PAS 的残差范数中位数。

二者转换为全体样本中的百分位排名，形成连续可靠性分数。固定策略为：

```text
只接受可靠性更高的70%
接受后只向共识PAS移动20%
执行4轮PDP -> PAS交替投影
保持每个点总功率
```

这个门解决的是“有些复杂点物理频谱有帮助，但简单点原始插值已经很好”。如果多个来源彼此冲突，算法宁愿保持当前复信道。

实现见 [reliability_fusion.py](code/round2/physical_ai_submission/ai_enhancement/reliability_fusion.py)。

## 15. 为什么需要交替投影

同一个完整 H 同时决定 PAS 和 PDP：

```text
H --阵列域变换--> PAS
H --频率域变换--> PDP
```

只修改 H 使其满足 PAS，通常会改变 PDP；只满足 PDP，也通常会改变 PAS。因此代码交替执行：

```text
当前H
 -> 投影到目标PDP
 -> 投影到目标PAS
 -> 再检查PDP
 -> 重复有限轮
```

每次投影主要修改对应变换域中的幅度，并尽量保留当前复相位；每轮还恢复点总功率。这样可以把频谱目标写回完整复信道，而不需要让频谱网络直接生成 `256x4x192` 个复数。

## 16. 一个目标点的完整处理示例

假设要预测目标点 `q`：

1. 根据位置确定所属站点和对应 BS。
2. 在同站 Train 点中查询最近32个邻居。
3. 用该站点的 `beta/gamma` 把每个邻居 H 搬运到 `q` 的径向相位参考系。
4. 生成最近邻、4组原始 IDW、4组相位对齐 IDW、4组径向仿射和4组 steering 仿射，共17个完整 H。
5. 按站点和8个端口组，用训练好的复数岭权重融合为 `H_mix(q)`。
6. 从 donor 和分组校准器得到官方 Train H 支持的 PAS/PDP 目标。
7. 根据地图风险确定 `q` 的 lifecycle regime，并调整 angle-delay 峰的持续、消失、出生和移动倾向。
8. 频谱校准器根据传播先验和不确定性生成校准端点。
9. 联合频谱模型协调多种 PAS/PDP 解释。
10. LinkGate 对1024条链路分别决定 PDP 修正强度，并写回完整 H。
11. PAS可靠性门只接受来源一致的保守修正。
12. 多轮投影后得到 `H_base(q)`，保持 `complex64`、完整端口结构和点总功率。

完整编排入口见 [base_channel_builder.py](code/round2/physical_ai_submission/ai_enhancement/base_channel_builder.py#L1149)。

## 17. 代码中的真实调用顺序

`build_base_field()` 的主顺序为：

```text
build_official_spectra
build_foundation_field
project_official_targets(reference)
build_carrier_bank
combine_carriers
project_official_targets(carrier mixture)
build_lifecycle_field
build_spectral_endpoints
build_joint_spectral_field
build_component_link_field
build_reliable_base_field
validate_channel
```

可以把这些阶段归纳为三层：

```text
复数层：foundation + 17 carriers + complex mixture
频谱层：donor + lifecycle + calibrator + joint model
写回层：LinkGate + reliability gate + alternating projection
```

## 18. 哪些是确定性计算，哪些是学习得到的

| 类型 | 内容 |
| --- | --- |
| 确定性计算 | KNN查询、距离、相位搬运、IDW、局部仿射系数、FFT/IFFT频谱变换、地图几何量、PAS/PDP投影、总功率恢复 |
| 从官方 Train H 学习 | 每站点 `beta/gamma`、17专家复数融合权重、donor权重、分组频谱校准、lifecycle原型和映射 |
| 神经网络学习 | 周期频谱残差、联合频谱融合、逐链路 LinkGate |
| 固定策略 | 17种专家配置、风险四分位分档、可靠性百分位计算、LinkGate有界输出、可靠性70%门、PAS移动20%、4轮交替投影 |

## 19. 这一模块解决了什么，没有解决什么

它重点解决：

- 原始复信道直接平均造成的相位抵消。
- 单一邻域尺度无法适应所有空间区域。
- 连续插值导致的 PAS/PDP 过度平滑。
- 频谱修正对不同链路和不同点可靠性不同。
- 只预测频谱却无法还原完整复信道的问题。

它不能单独完全解决：

- 未见空间洞中所有新路径的精确出生与消失。
- 每条路径的精确 angle、delay、能量和端口状态。
- 传播先验中不存在的全新候选结构。

因此后续 ComponentState 模型继续预测传播成分状态，基础场负责提供可信的完整复数载体和保守频谱锚点。

## 20. 快速推理与完整重建的区别

`infer` 为快速复现路径。它校验并读取冻结的基础场和模型输入资产，然后运行后续 ComponentState 推理与合成，不会现场重新训练17专家或重建整条基础场。

`rebuild-all` 才会从官方数据和完整训练资产执行本文件描述的基础场构建流程。两种入口使用的是同一算法资产，区别是是否重新计算已经冻结的中间结果。

## 21. 阅读代码的建议顺序

1. [phase_identification.py](code/round2/physical_ai_submission/interpolation_field/phase_identification.py)：理解 `beta/gamma` 如何从官方 H 辨识。
2. [spatial_experts.py](code/round2/physical_ai_submission/interpolation_field/spatial_experts.py#L135)：理解一个目标点如何生成一个专家结果。
3. [expert_bank.py](code/round2/physical_ai_submission/interpolation_field/expert_bank.py#L19)：核对17种专家配置。
4. [carrier.py](code/round2/physical_ai_submission/training/carrier.py)：理解 OOF 载体生成和复数融合训练。
5. [base_channel_builder.py](code/round2/physical_ai_submission/ai_enhancement/base_channel_builder.py#L1149)：查看整条基础场调用链。
6. [lifecycle_core.py](code/round2/physical_ai_submission/ai_enhancement/lifecycle_core.py)：理解地图风险与路径生命周期。
7. [spectral_calibrator.py](code/round2/physical_ai_submission/ai_enhancement/spectral_calibrator.py)：理解周期频谱残差。
8. [joint_spectral_model.py](code/round2/physical_ai_submission/ai_enhancement/joint_spectral_model.py)：理解多种频谱解释如何联合。
9. [link_gate_model.py](code/round2/physical_ai_submission/ai_enhancement/link_gate_model.py)：理解逐链路 PDP 修正强度。
10. [reliability_fusion.py](code/round2/physical_ai_submission/ai_enhancement/reliability_fusion.py)：理解最后的保守 PAS 接受机制。

## 22. 最核心的理解

这套基础场不是一种单纯插值法，也不是让神经网络直接生成完整复信道。它把不同任务交给最合适的表示：

```text
官方复信道邻域
    -> 提供复相位、端口结构和连续背景

17个空间专家
    -> 提供不同空间规律下的候选答案

官方PAS/PDP与传播先验
    -> 提供路径峰应该位于哪里的结构约束

频谱神经网络与门控
    -> 判断哪些约束可信、每条链路应该采用多少

交替投影
    -> 把频谱修正重新写回完整complex H
```

因此它的基本思想是：

> 不让插值器独自决定传播结构，也不让频谱网络从零生成相位；用官方 H 建立可信的复数连续场，再让受物理和可靠性约束的模型修正其中的路径频谱结构。
