# 复赛代码

复赛的数据审计、空间洞验证、基础复信道构造、Physical AI 训练与提交生成代码全部放在本目录。
复赛必须从 `datasets/round2/` 官方原始输入重新执行，不得把初赛某个历史输出当作起点。

V1-V4与V6不使用Sionna信道或路径。V1-V4基础执行顺序为：

```bash
code/tools/bootstrap_environment.sh
code/tools/run_in_env.sh python -u code/round2/run_foundation_benchmark.py
code/tools/run_in_env.sh python -u code/round2/run_pre_sionna_pipeline.py
code/tools/run_in_env.sh python -u code/round2/run_rbf_spectral_field.py
code/tools/run_in_env.sh python -u code/round2/run_v3_anchored_neural_rbf.py --folds 0,1,2,3,4 --epochs 6 --correction-l1-bound 0.8 --correction-regularization 0.003
code/tools/run_in_env.sh python -u code/round2/evaluate_v3_anchored_full_targets.py
code/tools/run_in_env.sh python -u code/round2/evaluate_v3_anchored_projection.py
code/tools/run_in_env.sh python -u code/round2/build_v3_anchored_neural_rbf.py
code/tools/run_in_env.sh python -u code/round2/analyze_v4_predictability_calibrated_groups.py
code/tools/run_in_env.sh python -u code/round2/run_v4_calibrated_exact_projection.py
```

V6从V4继续迁移初赛前半段的传播分区、稀疏component support、测试点路径生命周期和官方地图visibility/geodesic信息：

```bash
code/tools/run_in_env.sh python -u code/round2/build_route_topology_features.py
code/tools/run_in_env.sh python -u code/round2/build_geodesic_corridor_features.py
code/tools/run_in_env.sh python -u code/round2/run_v6_regime_component_lifecycle.py --phase all
code/tools/run_in_env.sh python -u code/round2/package_round2_v6_regime_component_lifecycle.py
```

V6只使用官方训练H、训练/测试坐标和官方地图。严格嵌套空间洞由V4重放`0.643689`提高到`0.643928`，4/5折为正；全OOF最终分区策略为`0.644400`，提高`+0.000710`。原始模板和生命周期support都不能直接替换V4，最终只部署geodesic约束的有界component shift，并保留V4复相位与逐点总功率。完整证据以实验记录第641节为准。

V5在V4复信道骨架上增加Sionna谱校准，但不使用Sionna复相位。它先从官方地图和全部公开坐标生成24-seed物理谱，再只用官方训练H监督PAS/PDP共享校准器：

```bash
HW_CONDA_ENV=sionna1 HW_SKIP_ENV_CHECK=1 code/tools/run_in_env.sh python -u code/round2/generate_multiseed_sionna_dataset.py --config primary --split all --max-seeds 24 --selection-tag full --batch-size 16 --cuda-device 0
code/tools/run_in_env.sh python -u code/round2/validate_multiseed_sionna_dataset.py
code/tools/run_in_env.sh python -u code/round2/prepare_sionna_calibration_dataset.py
code/tools/run_in_env.sh python -u code/round2/train_sionna_spectral_calibrator.py --phase screen --epochs 8 --steps-per-epoch 200
code/tools/run_in_env.sh python -u code/round2/train_sionna_spectral_calibrator.py --phase oof --modes mean --epochs 8 --steps-per-epoch 200
code/tools/run_in_env.sh python -u code/round2/evaluate_sionna_spectral_calibrator.py
code/tools/run_in_env.sh python -u code/round2/generate_round2_v5_sionna_calibrated_projection.py
```

V5的固定部署动作、五折证据和产物校验只以`experiment_records/EXPERIMENT_RECORD.md`第640节为准。

`run_foundation_benchmark.py`从官方复赛训练信道建立双基站条件化基础复信道。`run_pre_sionna_pipeline.py`在同一严格空间洞五折上实现并选择以下五个模块：

1. `pre_sionna_spectral.py`：提取angle-delay、PAS和PDP稀疏支撑，并执行保留复相位骨架的谱投影。
2. `geometry_features.py`：从官方三角网格提取五射线扇形遮挡、局部墙面和基站相对几何特征。
3. `pre_sionna_targets.py`：按基站学习传播状态、可见性条件模板、功率和零信道概率。
4. `run_pre_sionna_pipeline.py`：严格OOF枚举联合投影动作，并训练逐点收益控制器。
5. 同一入口按验证结果自动在固定分站策略和逐点策略之间选择，生成最终测试信道。

当前严格500点空间洞分数为`0.596534`，基础复信道为`0.571679`。最终选择固定分站策略，因为逐点控制策略为`0.596011`，没有超过固定策略。V1线上=`0.594822`，归档为`versions/round2/V1_complete_pre_sionna_pipeline_online_0.594822.zip`。

`run_rbf_spectral_field.py`只使用官方训练信道和坐标，在训练功率谱的平方根空间用64邻居线性RBF重建连续谱场，再把目标angle-delay、PAS和PDP投影到E1基础复信道。严格嵌套空间洞总分为`0.628568`，线上为`0.628977`，相对V1线上提高`+0.034155`；该方法不使用Sionna。

V3保持V2的RBF权重作为精确锚点，只学习有界、零和的donor权重修正。训练使用官方训练谱、坐标和官方地图派生特征，并用2至16米伪空间洞模拟测试缺口；完整严格空间洞总分为`0.634508`，比V2提高`+0.005941`且五折全部提高。线上=`0.629603`，仅比V2提高`+0.000626`，归档为`V3_anchored_neural_rbf_online_0.629603.zip`；该反馈说明大部分本地神经修正增益没有迁移。
