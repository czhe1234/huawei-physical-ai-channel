# Physical AI 无线数字孪生信道生成

此仓库保存2026年11月14日的项目源码和已有实验记录。服务器源项目为 transformer 上的 /home/chenzhe/work/hw。此次上传没有训练、推理或重新评分。

任务是根据场景地图、已知位置及其信道，生成新位置的 MIMO-OFDM 复信道。比赛输出形状为 (500, 256, 4, 192)，类型为 complex64。

## 代码入口

- code/：当前开发源码、物理传播支持、训练、评分及工具。
- final_code/code/ 与 final_code/README.md：完整生产链和原部署说明。
- final_code/V16_fast_repro_submission/：V16最终快速推理源码及原算法说明。
- final_code/V16_reorganized_submission/：便于阅读的重组织源码；基线接口也从此处读取V16频点和阵列约定。
- code/round2/baseline_freqminibatch_20261112/：NeRF²与NeWRF最新频点小批量实现。
- code/round2/baseline_formal_20261111/ 与 code/round2/paper_baseline_training_v1/：冻结训练器和完整H适配。
- plus-pro/code/：实测适配、共享接口及原作者代码依赖。

## 最新已保存的仿真比较

比较范围是500个本地空间留出位置，不是官方500个盲测位置。所有频点及端口统一评价；31个零真值位置保留。

| 方法 | PAS ↑ | PDP ↑ | NMSE ↓ | 预测SNR dB ↑ | 综合分 ↑ |
|---|---:|---:|---:|---:|---:|
| V16（历史版本） | 0.676942 | 0.780347 | 0.933125 | 0.300601 | 0.686375 |
| NeRF²（频点小批量续训） | 0.045773 | 0.252688 | 1.000000 | -1.828649e-07 | 0.219384 |
| NeWRF（频点小批量续训） | 0.046560 | 0.031204 | 1.000000 | -3.980008e-08 | 0.131106 |
| nGRF（已核验修复版） | 0.118423 | 0.026204 | 1.016580 | -0.071417 | 0.157029 |
| WRF-GS+（完整H适配） | 0.433131 | 0.659455 | 1.001089 | -0.004728 | 0.536980 |

完整精度原始来源：
[comparison.json](analysis/results/round2_paper_four_baselines_v1/freqminibatch_20261112/evaluation/comparison.json)。
[增加预测SNR的CSV](results/latest_comparison_with_snr.csv)从该JSON直接计算，公式为 prediction_SNR_dB = -10 log10(global NMSE)。没有先舍入NMSE。这里的SNR不是逐位置SNR的平均数或中位数。
综合分仍为 0.4 PAS + 0.4 PDP + 0.2/(1+NMSE)。

### 结果的适用范围

- V16沿用用户已验证的本地历史汇总，没有重训或重评分。它的历史训练覆盖与四个基线仍有差异，不能把此表写成严格相同训练覆盖下的正式排名。
- 用户已验证的V16线上综合分0.686754单独保留；没有据此补造线上PAS、PDP或NMSE。
- NeRF²和NeWRF的完整500点输出已评分核验，但当前输出能量很小，NMSE接近1；全场景拟合和泛化问题尚未解决。小规模拟合通过不代表原文献方法能力上限已被充分复现。
- 四个文献方法均包含任务适配，不能把这些数值表述为原论文原生任务的结果。
- V16当前本地历史NMSE约0.933，主要收益体现在PAS/PDP；不据此主张高精度完整CSI恢复。

### 结果和训练记录

- [最新四基线结果](analysis/results/round2_paper_four_baselines_v1/freqminibatch_20261112/)
- [各方法、各折报告索引](analysis/results/round2_paper_four_baselines_v1/freqminibatch_20261112/evaluation/training_report_index.json)
- [保留折与nGRF记录](analysis/results/round2_paper_four_baselines_v1/formal_20261111/)
- [WRF-GS+完整H适配及旧成绩](analysis/results/round2_paper_four_baselines_v1/wrfgs_full_h_v1_20261110/)
- [小规模训练修复核验](analysis/results/round2_paper_four_baselines_v1/repair_20261111/)
- [V16原始本地汇总](analysis/results/round2_official_sionna_path_calibrator/fixed500_report.json)
- [V16训练来源与恢复条件](analysis/audits/v16_project_lineage_20261106/recovery_matrix.json)
- [路径模块内部对照](analysis/results/round2_e216_v16_rt_information_ablation/summary.json)

逐点统计NPZ仅包含指标、行号等统计量，不是完整预测H。配置、检查和汇总保留原始文件内容。持续运行日志、重复源码快照和大型保护清单未纳入此快照。

## 南侧实测开发

[已有结果](plus-pro/analysis/results/hall_south_metric_reevaluation_20261025/)与[指标协议](plus-pro/analysis/audits/hall_south_metric_protocol_20261025/)单独保留。
数据范围为南侧96个训练位置和48个开发验证位置。现有默认适配版尚未整体超过强插值，几何开关差异微小且混合。
实测使用逐频点方向谱、相对PDP、接收通道能量分配与接收通道—时延联合分布。它们不冒充原比赛完整H官方综合分。
这些是开发结果；本次未使用南侧正式测试或北侧确认数据。

## 环境、数据与权重

根 environment.yml 为基础环境说明；CUDA依赖见 environment.cuda.yml。V16完整链、快速推理及第三方方法另有各自 requirements 文件，不能把这些环境视为完全相同。按具体入口和原运行配置准备环境。

此GitHub快照不含原始信道、地图、模型权重、完整预测H和大型传播缓存。原部署README中“权重已包含”等表述针对服务器完整包，此快照只保留源码与记录。仅克隆仓库不能直接重现含权重的推理。

官方数据按原文件放入 datasets/round2/：
- Round2_Setup.json
- Round2_Train_Pos.npy
- Round2_Train_Channel.npy
- Round2_Test_Pos.npy
- Round2_Map.ply

固定本地划分的两个小索引文件已保留：
- artifacts/round2/foundation/round2_spatial_hole_folds.npz
- artifacts/round2/pre_sionna/round2_pre_sionna_oof_foundation_meta.npz

现有服务器权重、完整预测及缓存按各配置与manifest引用的相对路径保留。部分历史链依赖尚未找回，详见恢复条件记录；不能把重新生成的输出当作原历史成绩对应的预测。

在本项目服务器执行GPU计算时，只允许使用以下完整UUID，所有子进程继承 CUDA_VISIBLE_DEVICES：
- GPU-6b53c3c2-3071-5dfb-f306-2f2f142f8009
- GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0
- GPU-70b6fceb-c7c4-9467-3c81-097c9284c160

原部署文档中的旧数字选卡示例不覆盖此约定。

## 许可与文件核对

第三方代码的原许可和NOTICE保持原位。NeRF²、NeWRF、WRF-GS+顶层和nGRF快照含MIT许可；部分高斯光栅化子模块采用研究/非商业许可，不能将整个仓库统一视作MIT授权。本仓库未另行授予项目代码统一开源许可。

UPLOAD_MANIFEST.json 记录每个源文件的相对路径、大小和SHA256，并区分原文件与本次生成的README、SNR表。未修改实验源码、模型或原始成绩。
