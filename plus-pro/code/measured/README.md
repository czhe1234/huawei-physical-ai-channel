# dc01 实测信道评估

本目录把项目最终 V16 的复载体、谱校准 CNN、LinkGate、ComponentState 和谱约束合成迁移到 DICHASUS dc01。参数在实测数据上重新训练；没有使用竞赛 checkpoint。完整实验合同和结论只记录在 `experiment_records/EXPERIMENT_RECORD.md` 的 Measured-M1/M2。

## 数据与评价单位

- 原始来源：[DICHASUS dcxx](https://dichasus.inue.uni-stuttgart.de/datasets/data/dichasus-dcxx/)，DOI `10.18419/darus-3831`，文件 ID `269445`。
- 完整原始文件为 16,870,531,926 bytes、32,133 帧；先下载到本地，再校验并上传。
- 发布方 MD5：`7934b83963491b493a37556484d6ca42`；本次原始文件 SHA256：`631fa66bb82405f28926b4e6524ab02ce10e0b1b35232b153054f18d9c72b16d`。
- 阵列映射使用数据发布方 spec；CPO/STO、注册场景与坐标来自 [NVlabs/diff-rt-calibration](https://github.com/NVlabs/diff-rt-calibration)，固定 commit `cc70d813a5ac8cd976c0b43ca811712dd5a1629a`。相关文件与许可证保存在原始数据目录的 `context/`。
- 只根据位置选择 0.20m 网格代表点，得到 1,023 个位置。工作张量为 `[位置, 两个真实面板, 水平8, 垂直4, 频率256] complex64`。每第四个原始子载波构成工作视图，保留原始文件。
- 两面板始终绑定同一位置划分；不把面板伪装成 UE 或极化端口。单移动发射端不能检验原算法多 UE 端口的全部机制。

## 固定协议

场景 XY 划为 4m 方格，颜色为 `(tile_x + 2*tile_y) mod 5`。每个外层折用一个颜色测试、下一个颜色验证、其余三个颜色拟合；移除距任一留出点不足 0.30m 的拟合点。测试五折覆盖全部代表位置一次。所有归一化、权重拟合、训练轮次选择和模块接纳仅依赖拟合/验证集。

基础方法包含 17 个复载体专家、复 ridge，以及平方根谱 RBF，共 19 个方法；验证集选最强底座。最终方法在该底座上加入四个原始谱 CNN、原始 LinkGate 和序列/集合 ComponentState。原始网络主体通过导入或 AST 提取复用；输入输出尺寸按真实面板、频率数和单 TX 调整。竞赛专用 donor/PCA/lifecycle 数值资产不迁移。

物理预处理固定 Base 深度16、Scatter 深度10/散射系数0.15，每个源一百万采样、24 个随机种子。单次命令必须明确传入合同中的24个种子；生成脚本的默认值是3个种子。`generation_report.json` 的 `identity.seeds` 是实际运行依据。合并时对全字典的路径身份去重，保留128个候选。

主指标为 PAS/PDP 余弦均值，并报告实测信道上的 DFT 波束损失。预注册通过条件：相对验证选定底座的结构分增量，其 4m 空间块 bootstrap 95% 置信区间下界大于0，且平均波束损失不增加。原始 NMSE、公共相位对齐 NMSE、每面板公共相位/时延对齐 NMSE 分开报告；真值辅助对齐只用于诊断。附带的竞赛形式分数不等价于原比赛榜单分。

## 入口与产物

所有服务器命令从项目根目录运行，使用已验证的 `code/tools/run_in_env.sh`。环境初始化见根目录 AGENTS。

| 文件 | 作用 |
|---|---|
| `download_dc01.py` | 本地分段下载、续传和原始文件校验 |
| `upload_verified_data.py` | 本地校验后分段 SCP，服务器重组并复核 MD5/SHA256 |
| `acquire_dichasus_server.py` | 全部 14 组数据在服务器上直接下载、分段续传、完整核验和原子发布 |
| `follow_dichasus_server.py` | 通过 SSH 将服务器进度元数据同步到本机，不接收原始数据 |
| `show_dichasus_progress.py` | 生成每秒刷新的本地静态进度页，区分远程进程状态与进度连接中断 |
| `dc01_data.py` | TFRecord 审计、校准、坐标注册与代表位置选择 |
| `dc01_baselines.py` | 固定空间划分、19种基础方法与验证选择 |
| `dc01_geometry.py` | 调用现有物理运行时，保存可恢复的路径候选 |
| `dc01_spectral.py` | 四个谱 CNN 和 LinkGate 的实测重训 |
| `dc01_state.py` | 候选合并、ComponentState 重训和最终复信道 |
| `measured_runtime.py` | 动态尺寸 PAS/PDP、四轮投影和独立评价指标 |
| `dc01_report.py` | 配对统计、完整比较表和科研图 |
| `dc01_verify.py` | 独立双精度复算、输出完整性与空间保护带检查 |
| `run_dc01_evaluation.py` | 等待预处理验收并顺序运行训练、最终评价 |

已有的原始数据、准备结果、基线与物理先验验收通过后，运行：

```bash
code/tools/run_in_env.sh python code/measured/run_dc01_evaluation.py --device cuda:1
```

原始数据在 `datasets/dichasus/dc01/`；生成资产在 `artifacts/measured/`；最终机器报告在 `analysis/results/measured_dc01/`。训练保存输入身份、随机数状态、模型与优化器 checkpoint，身份改变时必须使用新输出目录。报告阶段读取已经冻结的测试结果，不执行超参数选择。

全量 DICHASUS 清单及下载验证位于 `analysis/dichasus_acquisition_20261016/`。当前使用本地直连下载、SSH 上传、服务器完整核验后删除对应本地原件；已完成服务器原件复用，优先续传未完成数据。服务器直接下载进程已停止，其断点保留。完整命令、证书链补全、校验回执和恢复说明见该目录的 `README_ZH.md`。

本实验只支持一个实测场景内的空间泛化判断。它不证明跨场景迁移、绝对移动 TX 相位可预测性，或原竞赛全部多端口机制的有效性。

## M3：从实测规律重建

M2保留为模块迁移初验。根据用户要求，M3先在每折拟合分区辨识功率、相位与谱结构规律，再建立实测模型；完整合同和裁决仍以唯一实验记录为准。

- `dc01_law_audit.py`：在互不重叠的拟合点子集检验径向相位律与log功率场，比较正确/打乱几何，并检查近位置、远时间的原始重复观测。权威审计输出为 `analysis/results/measured_dc01_m3_audit_v2/`。
- `dc01_factorized_field.py`：独立预测每面板能量，插值归一化PAS/PDP/联合角度—时延功率；实测邻居提供明确声明的相位参考，避免把复数相消后的能量当成接收功率。验证集选择功率模型与结构重建方式，拟合样本排除自身donor。
- `dc01_angular_state.py`：保留交互序列编码及同点候选竞争思想，学习相对路径能量、角度修正和有界可靠性。去除绝对RT相位特征，不注入未经数据支持的几何时延。输出保持数据驱动PDP及独立预测功率。`--control`使用打乱几何，保留真实查询位置及同等模型/训练预算。
- `dc01_m3_report.py`：输出固定对照、空间块置信区间、科研图和独立NumPy双精度复算。明确区分有物理能量的重建与条件均值收缩诊断；后者不用于声称功率和绝对相位同时恢复。

顺序运行以下服务器命令，第二个几何训练可放在另一张空闲GPU：

```bash
code/tools/run_in_env.sh python code/measured/dc01_law_audit.py --device cuda:1
code/tools/run_in_env.sh python code/measured/dc01_factorized_field.py --device cuda:1
code/tools/run_in_env.sh python code/measured/dc01_angular_state.py --device cuda:1
code/tools/run_in_env.sh python code/measured/dc01_angular_state.py --device cuda:6 --control
code/tools/run_in_env.sh python code/measured/dc01_m3_report.py
```

M3生成资产在 `artifacts/measured/dc01_m3_foundation_v1/`、`dc01_m3_angular_v1/`、`dc01_m3_angular_shuffle_v1/`，比较报告在 `analysis/results/measured_dc01_m3/`。既有dc01结果曾被观察，因此M3是开发比较，不能当作全新采集批次的盲测确认。
