<!-- Historical protocol retained for opt-in use; edit this canonical reference only. -->
# Sionna 官方生成配置反演任务

## 一、任务目标

使用包内恢复的 Etoile Sionna 场景、官方训练坐标和空间分离的官方训练复信道子集，系统反演与官方数据最接近的 **等效 Sionna RT 配置**。

这里的“等效”是指：在未参与选择的空间区域中，同时复现官方训练 H 的时延功率谱、泄露角度功率谱、总功率排序、路径覆盖和已知路径相位规律。不要声称能唯一恢复官方内部配置。

本任务只做物理系统辨识和特征生成：

- 不生成比赛提交包。
- 不把纯射线追踪输出当作最终比赛答案。
- 不使用线上隐藏标签或用户反馈分数选择参数。
- 最终返回的路径和谱特征将供另一个由官方数据训练的 AI/Physical-AI 模型使用。

## 二、开始前必须执行

1. 阅读本文件和 `artifacts/task_inputs/task_config.json`。
2. 运行：

   ```bash
   python code/validate_sionna_calibration_task.py
   ```

3. 枚举可用 conda 环境，不要只检查系统 Python。选择已经能导入 `sionna`、`mitsuba`、`numpy` 和 `scipy` 的环境。
4. 保存以下信息到 `analysis/task_results/environment_report.json`：
   - Python、Sionna、TensorFlow、Dr.Jit、Mitsuba、CUDA 版本；
   - GPU 型号、显存和 Mitsuba variant；
   - `PathSolver.__call__`、`Paths.cfr` 的实际函数签名；
   - 当前版本中 `max_depth` 是否有默认值及其值。
5. 先运行 8 个点的冒烟测试：

   ```bash
   conda run --no-capture-output -n <可用环境> \
     python code/run_sionna_calibration_smoke.py --count 8
   ```

如果 Sionna API 与参考脚本不同，应在包内修正脚本并记录差异，不能用简化传播公式代替真正的 Sionna PathSolver。

## 三、固定条件：禁止扫描

以下条件已有较强证据，必须固定：

| 配置 | 固定值 |
| --- | --- |
| 场景 | `artifacts/external/restored_sionna_official_etoile/etoile.xml` |
| 材料 | XML 中官方 Etoile ITU 材料；不得修改介电常数、电导率、厚度或材料分配 |
| BS 位置 | `[50, 0, 25]` 米 |
| BS 阵列 | `8x16`、半波长间距、`tr38901`、`VH` 双极化 |
| UE 阵列 | `2x1`、半波长间距、`dipole`、`cross` 双极化，共 4 端口 |
| 子载波 | 192 |
| 子载波间隔 | 30 kHz；15/60 kHz 只允许作为一次性负控制，不进入主搜索 |
| CFR | `normalize_delays=False`、`normalize=False` |
| 镜面反射 | 开启 |
| 训练目标 | 只能使用包内官方 H 子集，不得使用任何 synthetic teacher 或 v212 输出 |

已知物理证据：

- 官方 H 的阵列相位和 FFT 峰都指向 BS 水平朝向约 `-30°`。
- UE 端口相位支持接收机朝向 BS。
- 官方 LOS 时延支持 `30 kHz` 子载波间隔。
- 路径级相位运输反推出等效载频约 `6.692~6.694 GHz`。
- 旧扫描显示 `max_depth=12~16`、高射线预算、LOS+specular+refraction 明显优于默认深度 3。

## 四、允许扫描的参数

完整参数网格在 `task_config.json` 中。必须按阶段扫描，禁止一次性笛卡尔积穷举后只汇报一个偶然最优点。

### 阶段一：基准控制

至少运行四个基准：

1. Sionna 默认深度 3；
2. 旧 v205 控制：深度 5、60k、全机制但无 edge；
3. 旧 v206 最佳：6.0 GHz、深度 16、1m、specular+refraction；
4. 新物理基准：6.693 GHz、深度 16、1m、specular+refraction。

### 阶段二：路径深度和传播机制

- `max_depth = 3, 6, 8, 10, 12, 14, 16, 20`
- 先用 `200k` 射线预算和 seed 42。
- 传播组合：
  - LOS + specular；
  - LOS + specular + refraction；
  - 上一组合 + diffraction；
  - 上一组合 + diffuse；
  - refraction + diffraction + diffuse。
- edge diffraction 只对阶段二前 3 名做一次开关验证。

保留校准和验证都稳定的前 5 个组合，不得只按校准集选择。

### 阶段三：天线朝向和阵列求解

在阶段二最优路径组合上扫描：

- BS azimuth：`-31, -30.5, -30, -29.5, -29, -28.5` 度；
- BS tilt：`-12, -9.53, -7.5, -5, -2.5, 0` 度；
- UE orientation：`track_bs` 与固定朝向控制；
- `synthetic_array=True/False`。

先扫描 azimuth/tilt，再只在前 3 个朝向上测试 UE orientation 和 synthetic array，避免无意义组合爆炸。

### 阶段四：载频微调

在前述最佳配置上扫描：

`6.690, 6.692, 6.693, 6.694, 6.696 GHz`

额外保留 `6.0` 和 `6.5 GHz` 作为旧结论控制。频率必须同时检查：

- PDP/PAS；
- 材料频率响应引起的功率变化；
- 路径级相位随长度变化是否接近包内 `6.692~6.694 GHz` 证据。

### 阶段五：预算与随机种子

对前 3 个配置扫描：

- paired budget：`200k, 1m, 3m`，即 `samples_per_src=max_num_paths_per_src`；
- seed：`42, 206, 420`。

最终审计再使用 `42, 206, 420, 1337, 2026` 五个 seed。随机种子是稳定性变量，不得把某个偶然高分 seed 宣称为官方 seed。

### 阶段六：噪声模型

只在 clean RT 配置确定后拟合官方噪声族。比较：

- 固定方差复高斯噪声；
- 方差与信道功率成正比；
- `sigma^2 = a * P^gamma + b`。

拟合 `gamma` 和相对噪声强度，但不得通过给仿真结果反复加随机噪声来提高单次分数。应比较期望统计、AD tail floor、real/imag 方差比和 circularity。材料参数在这一阶段仍然固定。

## 五、数据划分和防止过拟合

`official_subset_metadata.npz` 包含三个按 55 米空间 tile 完全互斥的 split：

- `0 = calibration`：允许选择参数；
- `1 = validation`：用于阶段排名和提前停止；
- `2 = audit`：在全部参数冻结前禁止查看结果。

每组 96 点，按无遮挡、轻遮挡、中遮挡、重遮挡分层。16 个官方 exact-zero 点已排除，但其索引保留在 metadata 中。

禁止根据 audit 结果继续调参。若 audit 失败，只能报告失败和重新提出新的物理假设，不能回头微调阈值。

## 六、评分口径

必须复用 `code/sionna_calibration_scoring.py` 的口径，并逐项报告：

1. 256x4 linkwise normalized PDP cosine；
2. traceback reshape 后 full leaked-PAS cosine；
3. collapsed `(delay, BS-horizontal-angle)` PAS cosine；
4. 总功率 Spearman；
5. 校准全局偏移后的 log-power MAE；
6. complex coherence，仅作诊断；
7. 模拟零路径比例；
8. clear/light/medium/heavy 分区结果。

同时保留两个排序分数：

- `legacy_selection = 0.45*leaked_full + 0.45*PDP + 0.10*power_rank`
- `online_informed = 0.60*PDP + 0.25*leaked_full + 0.10*power_rank + 0.05*coverage`

其中 `power_rank=(Spearman+1)/2`。最终配置必须在两个分数上都位于 Pareto 前沿，并满足：

- validation 相对四个基准均有改善；
- audit 没有反转；
- 五 seed 标准差可接受；
- heavy 改善不能以 clear/light 明显退化为代价；
- calibration-validation 差距不能异常扩大。

## 七、必须完成的最终运行

参数冻结后，在全部 2000 个训练坐标和 500 个测试坐标上运行最佳配置的多 seed 共识，但只保存紧凑特征，不返回 full-H。

必须生成：

1. top-128 路径：valid、delay、总长度、功率、AoD/AoA、interaction depth/type；
2. 多 seed 路径存在频率和功率均值/方差；
3. normalized global PDP；
4. collapsed leaked `(delay,H-angle)` 谱；
5. 每点总功率和有效路径数。

如果当前 Sionna API 无法直接导出某个 interaction 字段，应保留能获得的字段，并在报告中准确说明，不得伪造。

## 八、返回文件格式

所有结果写入 `analysis/task_results/`，最后压缩为 `sionna_inverse_calibration_results.zip`。必须包含：

```text
environment_report.json
all_runs.jsonl
ranking.csv
best_config.json
audit_report.json
noise_report.json
findings.md
logs/
features/best_train_path_features.npz
features/best_test_path_features.npz
features/best_train_spectral_features.npz
features/best_test_spectral_features.npz
result_manifest.json
checksums.sha256
```

`findings.md` 必须用中文回答：

1. 哪些参数被可靠识别；
2. 哪些参数只能确定范围；
3. 哪些配置存在不可辨识退化；
4. 最佳配置相对默认、v205、v206 和 6.693GHz 基准分别提高多少；
5. 分数改善来自 PDP、PAS、功率还是覆盖；
6. 对后续官方数据 AI 模型最有价值的物理特征是什么。

## 九、完成标准

不要在只完成环境检查、冒烟测试或校准集搜索时停止。只有满足以下条件才算完成：

- 所有阶段执行完毕；
- audit 只使用一次；
- top 配置完成多 seed 复验；
- 2000/500 坐标紧凑特征生成完成；
- 返回 ZIP 可解压且 checksum 正确；
- 报告明确区分“已识别”“等效范围”“无法唯一确定”。

