# Round 2 Physical-AI Submission

本项目从比赛官方输入生成完整的 `Round2_Test_Channel.npy`。推理过程只读取
官方数据和当前模型目录，不读取历史版本的预测信道。仓库同时保留全部模型和
生产策略资产的训练入口。

## 输入与输出

准备以下官方输入文件：

```text
Round2_Train_Channel.npy
Round2_Train_Pos.npy
Round2_Test_Pos.npy
Round2_Map.ply
Round2_Setup.json
```

默认放置目录为 `datasets/round2/`，也可以通过 `--data-dir` 指定其他目录。
最终输出满足：

```text
shape = (500, 256, 4, 192)
dtype = complex64
```

预训练权重已包含在：

```text
artifacts/round2/physical_ai_submission/models/
```

## 环境

需要 Linux、NVIDIA GPU、可用的 OptiX 驱动，以及与 CUDA 12.1 兼容的驱动。
完整冷启动建议预留至少 `180 GB` 可用磁盘空间。

使用 conda：

```bash
conda env create -f requirements.yml
conda activate physical-ai-round2
```

或使用 pip：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

如果系统未把 `libnvoptix.so.1` 放入动态链接器搜索路径，需要设置：

```bash
export LD_LIBRARY_PATH="/path/to/nvidia/driver/libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export DRJIT_LIBOPTIX_PATH="/path/to/libnvoptix.so.1"
```

路径取决于实际驱动安装位置，源码中不包含机器专用路径。

## 运行

在项目根目录执行：

```bash
export PYTHONPATH="$PWD/code"
python -m round2.physical_ai_submission all \
  --gpus 0,1,2,3,4,5,6,7 \
  --device cuda:0 \
  --workers-per-gpu 2 \
  --receiver-batch-size 16 \
  --batch-size 64 \
  --output Round2_Test_Channel.npy
```

单 GPU 运行：

```bash
python -m round2.physical_ai_submission all \
  --gpus 0 --device cuda:0 --workers-per-gpu 1 \
  --output Round2_Test_Channel.npy
```

完整流程依次执行 `trace -> spectra -> tokens -> base -> state -> predict`。
所有长阶段都保存可恢复缓存；中断后重新执行同一命令即可继续，不要在正常续跑时
添加 `--force`。

## 完整重新训练

以下命令从官方 Train 数据、坐标、地图和 Setup 开始，重新生成 Train 物理缓存，
训练 31 个生产 checkpoint，并重建 donor、carrier、group calibration、projection
和 lifecycle 资产：

```bash
python -m round2.physical_ai_submission train-all \
  --gpus 0,1,2,3,4,5,6,7 \
  --device cuda:0 \
  --workers-per-gpu 2 \
  --receiver-batch-size 16 \
  --batch-size 64
```

训练阶段默认使用隔离目录：

```text
artifacts/round2/physical_ai_retrained/
artifacts/round2/physical_ai_retrained/models/
```

流程按阶段原子保存；中断后重新执行同一命令会从已有完整缓存继续。正常续跑不要
添加 `--force`。最终会生成 `reports/evaluate_retrained.json`，其中包含 canonical
fixed500 的 PAS、PDP、NMSE 和总分，以及 `0.670000` 本地质量门判断。

训练完成后，使用新模型从 Test 输入生成最终结果：

```bash
python -m round2.physical_ai_submission all \
  --retrained \
  --gpus 0,1,2,3,4,5,6,7 \
  --device cuda:0 \
  --workers-per-gpu 2 \
  --output Round2_Test_Channel_retrained.npy
```

核验模型资产和输出：

```bash
python -m round2.physical_ai_submission verify-retrained-assets
python -m round2.physical_ai_submission validate \
  --retrained --output Round2_Test_Channel_retrained.npy
```

完整 Train 物理路径规模显著大于 Test 推理，需要预留较长运行时间和至少
`1.2 TB` 可用磁盘空间。这里包含可恢复的完整复数路径缓存和训练频谱分片；
只运行上面的 Test 冷启动推理仍按环境一节的 `180 GB` 准备即可。

只验证已有输出时不需要 GPU：

```bash
python -m round2.physical_ai_submission validate \
  --output Round2_Test_Channel.npy
```
