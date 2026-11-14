# 初赛代码

本目录保留初赛从基线到 `v578` 的算法、分析、训练脚本及历史辅助源码。
脚本的参数与结论以 `experiment_records/EXPERIMENT_RECORD.md` 为准。

运行时仍使用统一环境入口，例如：

```bash
code/tools/run_in_env.sh python code/round1/generate_v209_adaptive_metric_projection_controller.py
```

`PATHS.code` 为保持历史打包脚本可复现，仍是 `PATHS.round1_code` 的兼容别名；新代码必须使用显式的轮次路径。
