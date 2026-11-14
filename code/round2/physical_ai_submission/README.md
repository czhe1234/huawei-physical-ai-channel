# Round 2 Physical-AI Submission

本目录保留研发源代码。Round2最终提交、复现和后续分析的唯一入口已经整理到
项目根目录的 `final_code/`；历史实验脚本和版本归档只用于证据对照。

本目录是线上 V15 方法的独立生产实现。它从官方训练信道、坐标、地图和
Setup 出发，一次生成完整的 `Round2_Test_Channel.npy`，不读取任何历史版本
的预测信道。

## 算法流程

```text
Official Train H/Pos + Test Pos + Map + Setup
  -> project-local physical propagation solver (24 seeds)
  -> physical spectra and 128 path tokens per test point
  -> official-H complex base field and 17 carrier fields
  -> five-fold path-state model ensemble
  -> constrained PAS/PDP projection and complete complex H
```

物理求解采用两套固定配置：

- `base`: depth 16，LOS、镜面反射和折射；
- `scatter`: depth 10，在相同主传播机制上加入散射系数 `0.15`；
- 两套配置都使用固定的 24 个 seed、synthetic array、每个 source 一百万个
  候选样本和最多一百万条候选路径。

物理阶段输出路径几何、delay、角度、交互序列、逐 seed 状态和复路径系数，
但不直接输出最终信道。完整信道由官方 Train H 构成的复基础场、预训练
Physical-AI 模型和统一解析器联合生成。最终投影采用 PAS/PDP 各 `0.20` 的
平方根概率融合、四轮 `PAS -> PDP` 交替投影，并保持逐点总功率。

运行时不安装或导入 `sionna` Python 包。项目内物理求解内核的第三方源码
来源和 Apache-2.0 许可证见 [NOTICE.md](NOTICE.md) 及
`../physical_prior/physical_rt/`。

## 官方输入

把以下文件放在 `datasets/round2/`：

```text
Round2_Train_Channel.npy
Round2_Train_Pos.npy
Round2_Test_Pos.npy
Round2_Map.ply
Round2_Setup.json
```

预训练权重位于：

```text
artifacts/round2/physical_ai_submission/models/
```

权重只包含从官方 Train H 和固定物理传播数据训练得到的参数，不包含官方
Test H。推理不会读取 `V7` 到 `V15` 的任何历史完整信道文件。

## 环境

推荐使用 conda：

```bash
conda env create -f code/round2/physical_ai_submission/requirements.yml
conda activate physical-ai-round2
```

也可以使用 pip 或 uv：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r code/round2/physical_ai_submission/requirements.txt
```

需要 NVIDIA GPU、可用的 OptiX 驱动，以及与 PyTorch CUDA 12.1 兼容的驱动。
H100/A800 已验证。完整冷启动建议预留至少 `180 GB` 磁盘空间。

## 一条命令生成

在项目根目录运行：

```bash
export PYTHONPATH="$PWD/code:$PWD/code/common:$PWD/code/round2"
python -m round2.physical_ai_submission all \
  --gpus 0,1,2,3,4,5,6,7 \
  --device cuda:0 \
  --workers-per-gpu 2 \
  --receiver-batch-size 16 \
  --batch-size 64 \
  --output Round2_Test_Channel.npy
```

任务按 `trace -> spectra -> tokens -> base -> state -> predict` 执行。传播任务、
谱 shard 和后续阶段都保存原子缓存；命令中断后直接重跑即可续算。不要在正常
续算时加 `--force`。

单 GPU 可使用：

```bash
python -m round2.physical_ai_submission all \
  --gpus 0 --device cuda:0 --workers-per-gpu 1 \
  --output Round2_Test_Channel.npy
```

## 分阶段运行与验证

```bash
python -m round2.physical_ai_submission trace   --gpus 0,1,2,3
python -m round2.physical_ai_submission spectra --gpus 0,1,2,3
python -m round2.physical_ai_submission tokens  --gpus 0
python -m round2.physical_ai_submission base    --gpus 0 --device cuda:0
python -m round2.physical_ai_submission state   --gpus 0 --device cuda:0
python -m round2.physical_ai_submission predict --gpus 0 --device cuda:0
python -m round2.physical_ai_submission validate --gpus 0
```

最终文件必须满足：

```text
shape  = (500, 256, 4, 192)
dtype  = complex64
finite = true
```

当前生产重建文件的 SHA-256 为
`74b50e9773afcc965d174c10532155e34201899a62bbf32a8f31dbde681aa635`。
由于官方 PLY 中存在完全重叠三角面，不同 OptiX/BVH 构建可能产生非字节级
差异；本机重建与原线上 V15 的完整复信道余弦为 `0.999906`，PAS/PDP 全局
余弦为 `0.999801/0.999870`。原 V15 的 fixed500 本地完整分为 `0.686375`，
线上分为 `0.686591`；线上分属于原提交，不能当作重新生成文件的现场评分。
