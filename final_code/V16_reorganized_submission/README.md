# Round 2 Physical-AI Fast Reproduction

## 项目简介

本方法构建了一套由真实信道数据锚定、几何物理知识约束并由神经网络自适应校准的复信道生成框架，其核心是相位解耦的多专家复信道插值场：首先利用相邻训练信道的归一化复相关，从官方真实数据中辨识不同基站下空间载波相位及其随频率变化的规律，将邻域信道按照传播距离和子载波位置搬运到目标点的统一相位参考系，并进一步补偿水平与垂直阵列的方向相位，从而解决复信道快速振荡、直接插值容易发生相消的问题；随后在多个空间尺度上构造最近邻、距离加权、相位对齐、正则局部仿射和阵列方向补偿等17个复信道场专家，并通过严格五折OOF训练的复数岭回归，针对不同基站和8个收发端口组学习融合权重，使模型能够自适应处理平滑区域、传播边界和局部结构突变区域。在此基础上，算法提取PAS、PDP和angle-delay结构，通过独立的频谱donor权重、分组低维校准以及保持频谱相位和点总功率的交替投影，将角度、时延和功率规律重新约束到复信道插值场中。场景地图同时被转换为可见性、路线遮挡、空间占用、周边净空、局部表面和传播状态等几何特征，两套互补传播配置经过24个seed形成PAS、PDP、angle-delay、功率和候选数量的均值、波动及有效次数，并被压缩为每个位置128个带有时延、角度、复系数、能量、稳定出现次数、交互序列和几何轨迹的结构token，使模型不仅获得传播结构，还能判断其稳定性与可信度。随后，场景状态模块根据几何环境修正angle-delay演化，采用周期卷积残差网络的频谱校准器在物理频谱基线上学习有界残差，联合频谱模型融合多种传播解释，LinkGate MLP针对1024条收发链路逐链路学习插值场PDP与传播先验PDP的融合比例，可靠性门控则依据多个PAS来源的一致程度决定是否接受增强，避免不可靠先验破坏高质量插值结果。最终，五折ComponentState Transformer先编码每条候选的多步几何序列，再在128个候选之间进行集合注意力建模，联合预测传播成分的能量贡献权重、周期angle-delay坐标和端口分布，并将预测结果以受控比例写入基础复信道；整个生成过程通过多轮PAS/PDP交替投影，在保留复数相位载体和阵列结构的同时严格保持点总功率。因此，该方法兼具数值准确性、物理知识的可解释约束、神经网络的非线性校准能力、多源先验的自适应可信度控制以及冻结中间资产后的快速可复现推理能力。

基础复信道的相位辨识、17专家融合和七级频谱修正链详见 [官方 H 基础场算法详解](BASE_FIELD_ALGORITHM.md)。

## 环境安装

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

## 官方数据放置

将以下五个官方文件放入 `datasets/round2/`：

```text
Round2_Train_Channel.npy
Round2_Train_Pos.npy
Round2_Test_Pos.npy
Round2_Map.ply
Round2_Setup.json
```

## 快速推理

在项目根目录执行：

```bash
export PYTHONPATH="$PWD/code"
python -m round2.physical_ai_submission infer \
  --gpus 0 --device cuda:0 --batch-size 64 \
  --output Round2_Test_Channel.npy
```

## 输出格式

推理输出满足：

```text
shape = (500, 256, 4, 192)
dtype = complex64
```

## 格式验证

只验证结果格式：

```bash
python -m round2.physical_ai_submission validate \
  --output Round2_Test_Channel.npy
```

## 完整训练与训练后生成结果

安装并激活完整训练环境：

```bash
conda env create -f requirements-full-training.yml
conda activate physical-ai-round2-full-training
export PYTHONPATH="$PWD/code"
```

在独立目录中执行完整训练：

```bash
python -m round2.physical_ai_submission train-all \
  --work-dir artifacts/round2/full_training \
  --model-dir artifacts/round2/full_training/models
```

训练完成后，使用同一组训练资产和权重生成比赛结果：

```bash
python -m round2.physical_ai_submission rebuild-all \
  --work-dir artifacts/round2/full_training \
  --model-dir artifacts/round2/full_training/models \
  --output Round2_Test_Channel.npy
```
