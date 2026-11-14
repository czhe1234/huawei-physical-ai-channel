# HW 交接
核验日期：2026-11-14。根目录：`/home/chenzhe/work/hw`。

## 当前目标与状态
研究无线信道建模、空间插值及基线比较。主要实现说明在 `final_code/README.md`，开发代码在 `code/`。
最近已核验的四基线增量比较在 `analysis/results/round2_paper_four_baselines_v1/freqminibatch_20261112/`。`evaluation/completion.json` 为 `scored_and_verified`，各方法查询数均为 500。NeRF² 与 NeWRF 的频率小批量补充已完成。没有本项目训练进程正在运行。
这些是冻结评估下的本地结果，不是新的线上成绩。

## 环境与有效约定
服务器执行。`code/tools/run_in_env.sh` 按 `.conda-env` 选环境，本次读取到值为 `base`。该脚本可能创建或修改环境，查看状态时不要直接调用。具体实验按原启动配置核对依赖。
保留原始数据、正式预测、评估器和已提交结果。比较使用同一冻结的 500 个空间留出查询。测试信道不能用于拟合或校准。
GPU 和持续实验监控按个人指令及项目实际配置处理。本轮未启动训练。

## 证据与下一步
读取上述目录的 `evaluation/completion.json`、`evaluation/comparison.json`，以及 `code/round2/` 的相关实现。按用户当前问题继续，不自动执行旧计划。
根目录与 `plus-pro/` 下的 `experiment_records/EXPERIMENT_RECORD.md` 被旧校验器用作哈希或前缀校验输入，故保留原位，不再作为任务日记维护。任务包、模板及必要复现输入也保留。

## GitHub 代码与结果存档
仓库：https://github.com/czhe1234/huawei-physical-ai-channel ，私有，所属账号 czhe1234。
源快照已上传并核验：main 分支，提交 63a8da0b63fb84348ed43c938ab1213be9bd045e；2633个文件的GitHub对象哈希与本地提交一致。该提交保留本次代码与结果快照，后续交接元数据更新不改变该源快照。
本次范围为当前源码、环境与许可文件、固定500点划分索引、V16及四基线的配置和结果、训练修复核验与南侧实测开发记录。原始信道、地图、模型权重、完整预测和大型传播缓存保留在服务器。
GitHub 根 README 标明源码入口、结果性质和外部依赖；UPLOAD_MANIFEST.json 记录源文件路径及SHA256。SNR仅从已保存的完整精度总体NMSE换算，没有重训、推理或重新评分。

## 唯一任务入口与历史
只更新本文件作为 AI 任务交接入口。旧记录在 `/home/chenzhe/archives/ai_records/20261114/hw/`，通过上级 `manifest.csv` 按原路径定向查找。默认不读归档或冻结交付包，旧任务和旧指令不自动生效。
必要的代码说明、校验输入、模型卡、原始材料和数据保留原位。旧调度器或报告脚本复用前，先核对已归档的文档路径。
