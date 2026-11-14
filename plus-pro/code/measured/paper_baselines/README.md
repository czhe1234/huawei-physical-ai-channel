# dc01 单频论文方法适配

本目录比较 Neural Materials、NeWRF、NeRF² 和本项目 M3 信息流的**明确标注单频适配版本**。它不宣称复现原论文原场景、原输入和原指标数值。

固定场景 dc01、3.438 GHz、1023 代表位置；首轮使用已有空间第 0 折的 544/251/183 训练/验证/测试位置，两个 8×4 阵列绑定位置，继承 0.30 m 保护距离。输出 `[183,2,8,4,1] complex64`。所有实测训练信息来自一个频率点，推理不读取查询 CSI、查询 DoA、相位、功率或其他实测子载波。

## 实现

- `neural_fields.py`：复用作者 NeWRF、NeRF² 网络主体和复数传播合成；NeWRF 采用 RT 最后交互点的移动端方向候选和固定源位置上下文；NeRF² 采用三维位置条件、单个复数输出和实际频率。纯可选且未使用的 einops 导入在模型 AST 加载时略去。
- `newrf_stability.py`：针对训练集全零输出，单独登记的一次正密度参数化修复。消光为 `softplus(beta=20)+1e-4`；记录与原硬 ReLU 适配版的区别。
- `materials_rt.py`、`train_materials.py`：运行作者神经材料网络和 Sionna 0.19.2 可微场；保留八对象接口中的空 RIS 槽位、native int32/float32 类型和一一对应的天线排列。单频去掉时延损失，保留作者功率 SMAPE、EMA 和 fit-only 标度校准。主表使用完整 10000 更新预算和已训练检查点选择。
- `our_single_frequency.py`：从单频重训独立功率场、角度谱和有界序列/集合修正；移除实测 PDP 上下文，使用常量占位维持原网络接口。不加载历史训练权重。
- `single_frequency_metrics.py`、`report_study.py`：NumPy complex128 评价，并由独立 CPU/PyTorch complex128 公式复算。单频不定义 PDP、时延扩展或原三项综合分数。

## 复现

从 plus-pro 根目录执行；先阅读唯一实验记录 Measured-M5。原始作者源码和许可证位于 `code/third_party/paper_baselines/`，由完整 source manifest 固定。

PyTorch 使用已验证的 base 环境（Torch 2.5.1+cu121、NumPy 1.26.4）；材料方法使用已有独立 `sionna019` 环境（Python 3.11、TensorFlow 2.15.1、Sionna 0.19.2）。不修改系统 CUDA 或全局环境；材料脚本按环境中已安装的 CUDA 编译器路径配置当前进程。

```bash
# 验证已有单频数据和源码；不会训练
./run.sh python code/measured/paper_baselines/prepare_single_frequency.py --workspace .

# 一次性无标签场景缓存
HW_SKIP_ENV_CHECK=1 HW_CONDA_ENV=sionna019 ./run.sh python code/measured/paper_baselines/materials_rt.py --mode prepare --gpu 6

# 三篇方法的主比较版本
HW_SKIP_ENV_CHECK=1 ./run.sh python code/measured/paper_baselines/neural_fields.py --method nerf2 --mode train --device cuda:0 --batch-size 512 --prediction-batch 512
HW_SKIP_ENV_CHECK=1 ./run.sh python code/measured/paper_baselines/newrf_stability.py --method newrf --mode train --device cuda:0 --batch-size 256 --prediction-batch 256
HW_SKIP_ENV_CHECK=1 HW_CONDA_ENV=sionna019 ./run.sh python code/measured/paper_baselines/train_materials.py --mode train --gpu 6 --run-name neural_materials_full_budget --full-budget --trained-checkpoints-only

# 本项目单频版本及独立比较
HW_SKIP_ENV_CHECK=1 ./run.sh python code/measured/paper_baselines/our_single_frequency.py --device cuda:2
HW_SKIP_ENV_CHECK=1 ./run.sh python code/measured/paper_baselines/report_study.py
```

示例设备号按实际空闲 GPU 调整。每种训练都有源码/数据/配置身份和原子 checkpoint；同一输出目录身份改变会拒绝恢复，应使用新的运行目录。

模型与预测位于 `artifacts/measured/dc01_paper_baselines_sf_v1/`，正式报告位于 `analysis/results/dc01_paper_baselines_single_frequency/`。硬 ReLU 全零训练和材料 0 步模型选择保留为负面记录，不混入三个已训练论文方法的主表。记录的是本次适配设置的性能，不保证每个方法优于全零预测或原论文数值。

## 固定材料 RT 参考

`evaluate_fixed_rt.py` 直接计算固定场景材料的电磁场，分别保存原始输出和仅用训练集总能量确定一个正实数增益的输出；不训练材料或路径网络，不标定相位。复用材料方法的深度 5 几何缓存，因此与原 24 种子、深度 16/10 的候选生成器配置不同。`verify_fixed_rt.py` 独立复算预测，并与冻结的 M5 结果生成比较报告。

```bash
HW_SKIP_ENV_CHECK=1 HW_CONDA_ENV=sionna019 ./run.sh python code/measured/paper_baselines/evaluate_fixed_rt.py --gpu 6 --batch-size 16 --run-name fixed_material_rt_v2
HW_SKIP_ENV_CHECK=1 ./run.sh python code/measured/paper_baselines/verify_fixed_rt.py
```

固定材料参考的资产位于 `artifacts/measured/dc01_paper_baselines_sf_v1/fixed_material_rt_v2/fold0/`，报告位于 `analysis/results/dc01_paper_baselines_single_frequency/fixed_rt/`。完整实验合同和接口修正记录见唯一实验记录 Measured-M6。
