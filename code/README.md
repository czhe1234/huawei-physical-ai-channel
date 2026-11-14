# 代码目录

- `round1/`：初赛全部算法、历史实验、外部参考和早期 probe。
- `round2/`：复赛数据审计、验证、训练和候选生成代码。
- `common/`：两轮共享且不携带特定场景假设的评分、阵列和复信道工具。
- `project_paths.py`：唯一项目路径配置入口。
- `tools/`：环境、硬编码、跨设备接管检查和版本档案工具。

新代码必须使用：

```python
from project_paths import PATHS
```

初赛/复赛数据分别使用 `PATHS.round1`/`PATHS.round2`，代码分别使用
`PATHS.round1_code`/`PATHS.round2_code`，版本缓存使用 `PATHS.cache_dir(N)`，提交包使用
`PATHS.round1_versions`或`PATHS.round2_versions`，运行报告使用 `PATHS.version_report_dir(N)`，共享逻辑资产使用 `PATHS.assets`。不得恢复
`BASE_DIR`/`PROJECT_ROOT` 字符串拼接接口。

禁止写机器专属的用户目录、安装前缀或 Windows 盘符路径。提交前运行：

```bash
code/tools/run_in_env.sh python code/tools/check_hardcoded_paths.py
```

复制到新设备后，在安装项目依赖前可直接使用系统 Python 验证项目自足性：

```bash
python code/tools/check_workspace_portability.py --full
```
