# 南侧挡板接口与正确性检查

从项目根目录运行：

    env CUDA_VISIBLE_DEVICES="" HW_SKIP_ENV_CHECK=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./run.sh python -u -m measured.hall_adapters.check

此命令只执行小型数学检查和南侧 fit 分区的有限通路检查。没有训练循环，没有优化器更新，不读取验证、测试或北侧信道标签。RBF 的四点闭式求解是接口检查，不产生正式方法比较。

输入只接受 artifacts/measured/hall_sw_publisher_v2。使用已有 split_manifest.json 中的 fit、validation、test，不重分数据。查询对象仅含 row_ids、position 和阵列参数。TrainingBank 的测量数据必须通过 SouthDataset.read_fit 读取。

common.py 管理数据与输出约定；metrics.py 定义共同的全频带功率和评价算子；foundation.py 提供 V16 的17类复数载体专家家族；core.py 直接继承冻结 V16 的序列/集合编码器并做单发射端口适配；geometry.py 使用对应南侧模型及原物理运行时；baselines.py 提供核回归、位置MLP和直接指向参考；wrf.py 保留作者网络、提供标量功率适配与可微CPU参考渲染；test_correctness.py 是独立小型检查。

WRF-GS+ 适配不能标为作者代码的原样复现。当前保留作者变形网络，替换示例RGB头，并将角度功率通过真实阵列的全频带响应映射到32波束。未核验原CUDA核的数值等价和正式吞吐，未完成正式fit/val收敛健康检查。不能用该阶段的未训练输出宣布先进基线被击败。

主方法也不是比赛冻结权重在新数据上的直接推理。真实测量的频点、阵列和单TX任务不同，竞争专用端口头及冻结资产不能照搬。可迁移核心及输入/输出已接通；后续须在南侧训练分区从头拟合，再按南侧验证分区冻结选择。

详细证据在 analysis/results/hall_south_adapter_correctness_v1/。本模块没有启动正式训练、正式空间比较或北侧确认。
