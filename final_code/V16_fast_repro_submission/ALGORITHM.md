# Algorithm And Data Flow

## 1. Design Boundary

本提交采用“离线物理-AI先验构建 + 在线神经推理”的两阶段结构。离线阶段输出
测试场景对应的冻结部署资产；在线阶段加载这些资产和预训练权重，重新计算最终
信道，而不是读取已有的 `Round2_Test_Channel.npy`。

## 2. Offline Prior Construction

### 2.1 Official-channel foundation

`base_field.py` 从官方训练信道和坐标构建 17 个空间复信道候选，包括近邻、IDW
和局部仿射候选。候选混合策略和频谱投影用于保持复数相位结构、PAS、PDP 与
点功率的一致性。

### 2.2 Geometry-conditioned propagation support

`physical_prior/geometry_prior_engine/` 根据官方三维地图和固定传播配置构建
场景条件传播支持。该模块执行场景相交检测、多跳候选采样和镜像几何构造，支持
LoS、镜面反射、漫反射和折射。它不直接产生最终提交信道，其输出被压缩为：

```text
multi-seed PAS/PDP/angle-delay statistics
128 path tokens per test point
path coefficient, delay, angle, interaction and geometry features
```

### 2.3 Learned offline calibration

五折频谱校准网络和联合频谱网络根据官方训练信道、位置特征、几何可见性和传播
统计预测目标 PAS/PDP。LinkGate MLP 学习基础场 PDP 与路径先验 PDP 的融合权重。

这些模块的输出共同形成：

```text
base_channel.npy
model_inputs/test/features.npy
model_inputs/test/auxiliary.npz
```

`base_channel.npy` 因此是混合物理-AI资产，不是传统传播求解器直接输出的信道。

## 3. Online Inference

快速 `infer` 依次执行：

1. 根据 `deployment_manifest.json` 校验官方输入、冻结资产和模型权重。
2. 加载五个 PathState Transformer checkpoint。
3. 对每个测试点的 128 个路径 token 预测质量、端口分布和 angle-delay 坐标。
4. 对五折输出执行集成和周期坐标圆均值。
5. 将预测端点渲染为 PAS/PDP，并对基础信道执行四轮约束投影。
6. 写出新的 `Round2_Test_Channel.npy` 并验证 shape、dtype 和有限性。

## 4. Frozen Numerical Invariants

描述和模块命名重构不改变以下数值定义：

```text
seed count = 24
paths per point = 128
subcarriers = 192
PAS blend = 0.20
PDP blend = 0.20
projection cycles = 4
PathState folds = 5
```

冻结基础场：

```text
shape = (500, 256, 4, 192)
dtype = complex64
SHA256 = 2f525878a9865fca15175ad147a99e406c7505f3dae320c7267eb1eb7f309d84
```

V16 参考输出：

```text
SHA256 = a0ad87f5ca1d1d8bd05d86829a36121a6bca4c309413739da2c53597cd4a2ca9
online score = 0.686754
```

## 5. Source Naming

高层和低层名称按模块职责组织：

```text
prepare-prior                    offline prior preparation
geometry_prior_engine            geometry-conditioned prior engine
propagation_support              propagation support construction
PropagationSupportBuilder        support builder
SpecularPathConstructor          mirrored-source specular construction
MultiBounceCandidateGenerator    multi-bounce candidate sampling
scene_intersections              scene intersection utilities
```

这些名称只改变代码组织和职责表达，不改变底层公式、场景查询或模型计算。
