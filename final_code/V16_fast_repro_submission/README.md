# Round 2 Physical-AI Fast Reproduction

本项目从比赛官方输入快速生成完整的 `Round2_Test_Channel.npy`。默认推理会校验
官方数据、离线物理-AI基础信道先验和五折模型权重，然后实际运行 PathState
Transformer 与最终频谱约束渲染器。默认推理不读取包内已有的最终预测文件。

## 快速复现

将以下五个官方文件放入 `datasets/round2/`：

```text
Round2_Train_Channel.npy
Round2_Train_Pos.npy
Round2_Test_Pos.npy
Round2_Map.ply
Round2_Setup.json
```

安装推理环境：

```bash
conda env create -f requirements.yml
conda activate physical-ai-round2-infer
```

或者使用 pip：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

在项目根目录执行：

```bash
export PYTHONPATH="$PWD/code"
python -m round2.physical_ai_submission infer \
  --gpus 0 --device cuda:0 --batch-size 64 \
  --output Round2_Test_Channel.npy
```

推理输出满足：

```text
shape = (500, 256, 4, 192)
dtype = complex64
```

只验证结果格式：

```bash
python -m round2.physical_ai_submission validate \
  --output Round2_Test_Channel.npy
```

## 方法结构

算法分为离线先验构建与在线神经推理两个阶段。

离线阶段首先使用官方训练信道构建空间复信道基础场，并根据官方三维地图执行
确定性的场景相交检测、多跳传播候选构造和镜面路径构造。LoS、镜面反射、
漫反射和折射支持被压缩为多 seed 频谱统计与固定长度路径特征。频谱校准网络、
联合频谱模型和 LinkGate 再将这些信息与官方训练信道结合，形成冻结的
`base_channel.npy` 与 PathState 模型输入。

在线阶段执行：

```text
部署资产校验
  -> 五折 PathState Transformer
  -> 路径质量、端口分布和 angle-delay 坐标预测
  -> PAS/PDP 约束投影
  -> Round2_Test_Channel.npy
```

`base_channel.npy` 不是纯物理仿真输出。它同时包含官方训练信道的空间重构、
固定物理先验、神经频谱校准和 LinkGate 结果，因此在本项目中称为
“离线构建的物理-AI基础信道先验”。详细数据流见 `ALGORITHM.md`。

## 部署资产

快速推理使用以下预生成资产：

```text
artifacts/round2/physical_ai_submission/
  deployment_manifest.json
  base_field/base_channel.npy
  model_inputs/test/features.npy
  model_inputs/test/auxiliary.npz
  models/path_calibrator/fold_0.pt ... fold_4.pt
```

`deployment_manifest.json` 记录 V16 基线、官方小型输入文件、部署资产和五折权重
的大小、shape、dtype 与 SHA256。训练信道约 6.3 GB，快速推理只检查其文件大小、
shape 和 dtype，避免每次额外执行全文件哈希。

基线信息：

```text
V16 online score = 0.686754
reference output SHA256 = a0ad87f5ca1d1d8bd05d86829a36121a6bca4c309413739da2c53597cd4a2ca9
```

不同 GPU、驱动或 FFT 实现可能产生末位浮点差异，结果应以官方评分和数值误差
检查为准，不要求跨硬件逐字节一致。

## 完整先验重建

完整重建不是默认复现路径。该流程保留用于源码审查和重新生成离线资产，依赖
Linux、NVIDIA GPU、CUDA 12.1、OptiX、Dr.Jit 和 Mitsuba，并建议预留至少
180 GB 可用磁盘空间。

安装完整环境：

```bash
conda env create -f requirements-prior.yml
conda activate physical-ai-round2-prior
```

或在已有推理环境中安装：

```bash
python -m pip install -r requirements-prior.txt
```

完整执行：

```bash
python -m round2.physical_ai_submission rebuild-all \
  --gpus 0,1,2,3,4,5,6,7 \
  --device cuda:0 --workers-per-gpu 2 \
  --output Round2_Test_Channel.npy
```

也可以分阶段执行：

```text
prepare-prior -> spectra -> tokens -> base -> state -> predict
```

离线几何传播支持位于：

```text
code/round2/physical_prior/geometry_prior_engine/
```

其中 `propagation_support/` 构建多跳传播支持，
`specular_path_constructor.py` 使用镜像几何构造镜面传播路径，
`scene_intersections.py` 提供场景相交计算。完整重建会重新生成这些支持；快速
`infer` 不导入或执行这些模块，也不要求 OptiX。

## 训练频谱模型

源码保留频谱校准器的训练流程：

```bash
python -m round2.physical_ai_submission prepare-train-prior \
  --gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 2
python -m round2.physical_ai_submission train-spectra \
  --gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 2
python -m round2.physical_ai_submission train-calibrators \
  --gpus 0 --device cuda:0 --force
```

快速复现不需要重新训练模型。
