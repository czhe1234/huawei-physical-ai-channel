"""Independently verify frozen predictions and report the matched dc01 holdout."""
from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from study_common import ARTIFACTS, CODE, PATHS, PREPARED, REPORTS, atomic_json, sha256, study_data


METHODS = [
    ("neural_materials", "Neural Materials-SF", "neural_materials_full_budget"),
    ("newrf", "NeWRF-SF（稳定性适配）", "newrf_stability/newrf"),
    ("nerf2", "NeRF²-SF", "nerf2"),
    ("ours", "我们的单频版", "ours"),
]


def independent_metrics(prediction, reference):
    """Independent CPU complex128 Torch formulas, not the NumPy evaluator."""
    p = torch.from_numpy(np.array(prediction)).to(torch.complex128)
    t = torch.from_numpy(np.array(reference)).to(torch.complex128)
    pe = p.abs().square().sum((2, 3, 4))
    te = t.abs().square().sum((2, 3, 4))
    error = (p-t).abs().square().sum((2, 3, 4))
    raw = float(error.sum()/te.sum())
    zero = pe == 0
    floor = torch.finfo(torch.float64).tiny
    power_db = 10*torch.log10((pe/te).clamp_min(floor))
    spectrum_p = torch.fft.fft2(p, dim=(2, 3)).abs().square().flatten(2)
    spectrum_t = torch.fft.fft2(t, dim=(2, 3)).abs().square().flatten(2)
    denominator = (spectrum_p.square().sum(-1)*spectrum_t.square().sum(-1)).sqrt()
    cos = torch.where(denominator > 0, (spectrum_p*spectrum_t).sum(-1)/denominator.clamp_min(floor), 0.)
    beam = spectrum_p.argmax(-1)
    received = spectrum_t.gather(-1, beam[..., None])[..., 0]
    best = spectrum_t.amax(-1)
    beam_loss = -10*torch.log10((received/best).clamp_min(floor))
    dot = (p.conj()*t).sum((2, 3, 4))
    aligned = (pe+te-2*dot.abs()).clamp_min(0).sum()/te.sum()
    points = error.sum(1)/te.sum(1)
    return {"raw_global_energy_ratio_nmse": raw, "mean_position_nmse": float(points.mean()),
            "median_position_nmse": float(points.quantile(.5)), "power_mae_db": float(power_db.abs().mean()),
            "array_pas_cosine": float(cos.mean()), "dft_beam_loss_db": float(beam_loss.mean()),
            "diagnostic_panel_phase_aligned_nmse": float(aligned),
            "prediction_energy_over_reference_energy": float(pe.sum()/te.sum()),
            "zero_prediction_panels": int(zero.sum()),
            "exact_zero_coefficient_fraction": float((p == 0).double().mean()),
            "finite_mean_power_bias_db": float(power_db[~zero].mean()) if bool((~zero).any()) else None}


def verify_sources():
    manifest = json.loads((REPORTS/"source_manifest.json").read_text())
    checked = 0
    for source in manifest["sources"]:
        for item in source["files"]:
            path = PATHS.root/item["path"]
            if sha256(path) != item["sha256"]:
                raise ValueError(f"Pinned author's source was changed: {item['path']}")
            checked += 1
    rule_spec = importlib.util.spec_from_file_location("baseline_path_rules", PATHS.code_root/"tools/check_hardcoded_paths.py")
    rules = importlib.util.module_from_spec(rule_spec)
    rule_spec.loader.exec_module(rules)
    findings = []
    for path in sorted(CODE.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        ast.parse(text, filename=path.name)
        for number, line in enumerate(text.splitlines(), 1):
            for kind, pattern in rules.PATTERNS:
                if pattern.search(line):
                    findings.append({"file": PATHS.relative(path), "line": number, "kind": kind})
    if findings:
        raise ValueError(json.dumps(findings))
    preparation = json.loads((REPORTS/"preflight.json").read_text())
    prepared_files = []
    for name, expected in preparation["files"].items():
        path = PREPARED/name
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Frozen single-frequency preparation changed: {name}")
        prepared_files.append(name)
    return {"author_files_verified": checked, "maintained_python_files_parsed": len(list(CODE.glob("*.py"))),
            "prepared_files_verified": prepared_files,
            "scoped_path_findings": findings,
            "global_path_scan": "historical global findings remain; only maintained study sources are scoped clean"}


def compare_result(folder, reference, expected_ids, strict_source=True):
    result = json.loads((folder/"result.json").read_text())
    if result["status"] != "complete":
        raise ValueError("Model result is not complete")
    prediction = np.load(folder/"test_prediction.npy", allow_pickle=False)
    ids = np.load(folder/"test_source_ids.npy", allow_pickle=False)
    if prediction.dtype != np.complex64 or prediction.shape != reference.shape or not np.isfinite(prediction).all():
        raise ValueError("Invalid saved prediction")
    if not np.array_equal(ids, expected_ids) or len(np.unique(ids)) != len(ids):
        raise ValueError("Test position identity/order differs between methods")
    if strict_source:
        for file, expected in result["identity"]["files"].items():
            if sha256(PATHS.root/file) != expected:
                raise ValueError(f"Frozen run dependency changed: {file}")
    metrics = independent_metrics(prediction, reference)
    discrepancies = {key: abs(metrics[key]-result["test"][key]) for key in [
        "raw_global_energy_ratio_nmse", "mean_position_nmse", "median_position_nmse", "power_mae_db",
        "array_pas_cosine", "dft_beam_loss_db", "diagnostic_panel_phase_aligned_nmse"]}
    if max(discrepancies.values()) > 1e-8:
        raise ValueError(f"Independent saved-prediction verification disagrees: {discrepancies}")
    metrics["numeric_floor_power_mae_db"] = metrics["power_mae_db"]
    metrics["power_mae_is_infinite"] = metrics["zero_prediction_panels"] > 0
    if metrics["power_mae_is_infinite"]:
        metrics["power_mae_db"] = None
    fit_health = None
    if (folder/"fit_health.json").exists():
        fit_health = json.loads((folder/"fit_health.json").read_text())
        if fit_health["model_sha256"] != sha256(folder/"model.pt") or fit_health["used_to_retrain_or_select"]:
            raise ValueError("Fit diagnostic does not refer to the frozen model")
    return {"metrics": metrics, "selected_step": result.get("selected_step"),
            "selected_epoch": result.get("selected_epoch"), "executed_updates": result.get("executed_updates"),
            "elapsed_seconds": result["elapsed_seconds"], "hardware": result["hardware"],
            "peak_gpu_gib": result.get("peak_gpu_gib", result.get("peak_gpu_gib_tensorflow_allocator")),
            "memory_scope": "TensorFlow allocator only" if "peak_gpu_gib_tensorflow_allocator" in result else "PyTorch allocated memory",
            "prediction_sha256": sha256(folder/"test_prediction.npy"),
            "result_file": PATHS.relative(folder/"result.json"),
            "independent_max_metric_discrepancy": max(discrepancies.values()),
            "frozen_fit_health": fit_health,
            "source_identity": result["identity"]["sha256"]}


def number(value, digits=4):
    return "∞" if value is None else f"{value:.{digits}f}"


def write_report(study):
    methods = study["methods"]
    rows = []
    for item in methods:
        m = item["metrics"]
        selected = item["selected_epoch"] if item["selected_epoch"] is not None else item["selected_step"]
        rows.append(f"| {item['label']} | {m['raw_global_energy_ratio_nmse']:.6f} | {number(m['power_mae_db'], 3)} | {m['array_pas_cosine']:.6f} | {m['dft_beam_loss_db']:.3f} | {selected} |")
    report = """# dc01 单频论文算法适配对比

本报告评价明确标注的 dc01 单频适配版本。它不是原论文原始场景和全部输入条件下的逐项复现成绩，也不是 DICHASUS 全场景或旧五折平均结果。

## 固定任务

- 场景：DICHASUS dc01；固定 3.438 GHz；1023 个代表位置中的空间第 0 折。
- 划分：544 个训练位置、251 个验证位置、183 个测试位置；另外 45 个位置因保护距离被排除。两个真实 8×4 阵列按位置绑定；继承 0.30 m 保护距离。
- 每个测试位置输出两个阵列共 64 个复数信道系数，保存为 `[183,2,8,4,1] complex64`。
- 训练测量信息仅来自这一个频率点。查询输入是位置、天线几何和场景，查询 CSI、测得的查询到达角、查询相位和功率均不作为预测输入。
- 各方法采用自己的训练目标，并在同一验证集选择模型。测试数据只在选择冻结后参与评价。dc01 已被用于此前开发，因此不称为新采集盲测。

## 测试结果

NMSE、功率误差和波束损失越低越好，PAS 相似度越高越好。NMSE 是整个测试集误差能量与真值能量之比，原始相位保留。

| 方法 | 原始全局 NMSE ↓ | 功率 MAE / dB ↓ | 阵列 PAS ↑ | 波束损失 / dB ↓ | 验证选中更新步 / 我们的轮次 |
|---|---:|---:|---:|---:|---:|
""" + "\n".join(rows) + """

PDP、RMS 时延扩展和原 PAS/PDP/NMSE 综合分数在本轮均不定义。把一个频率的长度 1 IFFT 当成完整时延谱并给出高分，会失去物理意义。

## 实现与原论文的差异

1. **Neural Materials-SF**：执行作者的 4×128 神经材料网络和 Sionna 0.19.2 可微电磁场计算；使用一个频率的功率 SMAPE，去掉时延扩展项。射线深度 5、每源一百万采样、合成阵列、无漫散射、有绕射；这些设置及完整 10000 次更新预算均记录在模型身份中。原始显式阵列和四百万采样的预处理配置没有被冒充为本轮配置。只从已训练检查点选择。
2. **NeWRF-SF**：复用作者粗/细网络、层次采样和复数相干合成；加入固定阵列天线位置作为源条件。移动端反向射线使用场景 RT 的最后交互点方向，代替缺失的移动端测量到达角与原查询射线搜索；这是场景辅助适配。首次硬 ReLU 消光训练退化为全零，单独保留失败记录；主表采用 `softplus(beta=20)+1e-4` 正密度稳定性适配，并显式记录这项额外改动。
3. **NeRF²-SF**：保留作者衰减/辐射网络与复数 CSI 合成公式，改为 3D 位置条件和单个复数输出，将传播频率设为 3.438 GHz。没有给它原 MIMO 配置中的已知当前 CSI。单频合成公式已与作者 26 载波重复输出的等价情形核对。
4. **我们的单频版**：仅用单频测量重训独立功率场、归一化角度结构、最近训练点相位参考与有界序列/集合角度修正。移除实测时延上下文和 PDP 损失，保留原网络接口中的对应位置为常量；不加载旧宽带训练权重。

原作者资料与实现：[Neural Materials / 可微射线追踪论文](https://research.nvidia.com/publication/2024-10_learning-radio-environments-differentiable-ray-tracing)、[代码](https://github.com/NVlabs/diff-rt-calibration)；[NeWRF / ICML 2024](https://proceedings.mlr.press/v235/lu24j.html)、[代码](https://github.com/LuHaofan/NeWRF)；[NeRF² 作者实现与论文入口](https://github.com/XPengZhao/NeRF2)。本轮作者代码提交号和逐文件哈希见 `source_manifest.json`。

NeWRF/NeRF² 的几何候选及我们方法复用此前仅由场景生成的 M3 射线支撑，原生成成本约 1279.464 秒；Neural Materials 新缓存生成耗时约 39.489 秒。复用几何缓存不等于测得了查询位置信道或移动端到达角。

## 判断范围

全零预测的原始 NMSE 为 1。神经场输出能量收缩时，NMSE 接近 1 不代表功率或信道结构已经恢复；必须一起检查功率、PAS 与波束结果。相反，保留物理能量但公共相位或阵列复结构不准的预测，也可能有大于 1 的 NMSE。

本实验不能证明位置条件下的绝对测量相位可辨识，也不能从该表断言某篇原论文方法在其原协议下无效。空间保护划分、单频训练、样本数量和额外方向信息均与论文原基准不同。

`evaluation.json` 另存真值辅助的每面板公共相位对齐误差，仅作诊断；该对齐不参与模型训练、选择或实际输出。初次全零 NeWRF 和 0 步材料模型列为诊断记录，不列入主表的已训练方法。

## 复核与文件

所有主表预测均由独立 CPU complex128/PyTorch 公式复算，并核对同一测试源记录顺序、complex64 文件形状、有限值、源代码身份和校验和。机器报告保存逐方法最大复核差异、能量比、零输出比例和训练资源。

- 模型与预测：`artifacts/measured/dc01_paper_baselines_sf_v1/`。
- 完整数据与划分：`artifacts/measured/dc01_single_frequency_v1/`。
- 比较表：`analysis/results/dc01_paper_baselines_single_frequency/comparison.csv`。
- 唯一实验过程记录：`experiment_records/EXPERIMENT_RECORD.md` 的 Measured-M5。

训练时间与显存来自各运行程序；神经场记录不含初始化前的几何加载，TensorFlow 显存仅为其分配器口径，硬件也不同，因此本报告不作严格速度或显存排名。
"""
    energy_rows = []
    for item in methods:
        fit_health = item["frozen_fit_health"]
        fit_nmse = f"{fit_health['raw_global_energy_ratio_nmse']:.6f}" if fit_health else "未复算"
        fit_energy = f"{100*fit_health['prediction_energy_over_reference_energy']:.6f}%" if fit_health else "未复算"
        test_energy = 100*item["metrics"]["prediction_energy_over_reference_energy"]
        energy_rows.append(f"| {item['label']} | {test_energy:.6f}% | {fit_energy} | {fit_nmse} |")
    diagnostics = """## 冻结模型的拟合质量诊断

总能量比是预测系数总能量除以真值总能量，100% 表示总体能量相同，不代表每个位置的功率或相位正确。训练集诊断在全部模型冻结后补做，只检查已保存权重，没有据此重新训练或选择模型。

| 方法 | 测试总能量比 | 训练总能量比 | 训练原始 NMSE |
|---|---:|---:|---:|
""" + "\n".join(energy_rows) + """

NeWRF 与 NeRF² 的幅度退化在训练位置上就已存在。本轮没有建立这两个适配版本的有效复信道拟合，不能把它们接近 1 的测试 NMSE 解释为重建成功，也不能据此宣称我们的算法超过了原论文强基线。后续应先解决适配后的拟合与优化问题；当前证据不足以把失败归结为测试空间泛化或测量相位不可辨识。

我们的单频版在本次划分中的功率、PAS 和波束指标较好，但原始 NMSE 仍大于全零预测的 1。Neural Materials 也没有解决原始复信道误差。单折结果不提供统计显著性或跨场景结论。

"""
    report = report.replace("## 判断范围", diagnostics+"## 判断范围")
    (REPORTS/"evaluation_report_zh.md").write_text(report, encoding="utf-8")
    columns = ["method", "raw_nmse", "power_mae_db", "array_pas_cosine", "beam_loss_db",
               "phase_aligned_nmse_diagnostic", "prediction_energy_ratio", "zero_coefficient_fraction",
               "fit_raw_nmse", "fit_energy_ratio", "fitting_quality_note",
               "selected_step", "selected_epoch", "executed_updates", "elapsed_seconds", "hardware", "prediction_sha256"]
    with (REPORTS/"comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for item in methods:
            m = item["metrics"]
            fit = item["frozen_fit_health"] or {}
            writer.writerow({"method": item["label"], "raw_nmse": m["raw_global_energy_ratio_nmse"],
                             "power_mae_db": m["power_mae_db"], "array_pas_cosine": m["array_pas_cosine"],
                             "beam_loss_db": m["dft_beam_loss_db"],
                             "phase_aligned_nmse_diagnostic": m["diagnostic_panel_phase_aligned_nmse"],
                             "prediction_energy_ratio": m["prediction_energy_over_reference_energy"],
                             "zero_coefficient_fraction": m["exact_zero_coefficient_fraction"],
                             "fit_raw_nmse": fit.get("raw_global_energy_ratio_nmse"),
                             "fit_energy_ratio": fit.get("prediction_energy_over_reference_energy"),
                             "fitting_quality_note": "amplitude degeneration on fit and test; strong baseline not established" if fit else "single-fold adapted result; raw complex-channel NMSE remains above 1",
                             **{key: item[key] for key in ["selected_step", "selected_epoch", "executed_updates", "elapsed_seconds", "hardware", "prediction_sha256"]}})


def plot(study):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = ["Neural Materials-SF", "NeWRF-SF (stable)", "NeRF2-SF", "Ours-SF"]
    colors = ["#667fa0", "#789d8a", "#bc9667", "#315a9b"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), layout="constrained")
    definitions = [("raw_global_energy_ratio_nmse", "Raw complex-channel NMSE (lower is better)"),
                   ("power_mae_db", "Power MAE / dB (lower is better)"),
                   ("array_pas_cosine", "Array PAS cosine (higher is better)"),
                   ("dft_beam_loss_db", "DFT beam loss / dB (lower is better)")]
    for ax, (key, title) in zip(axes.flat, definitions):
        values = [item["metrics"][key] for item in study["methods"]]
        if any(value is None for value in values):
            raise ValueError("A primary model has undefined power; do not plot a false finite bar")
        bars = ax.bar(np.arange(4), values, color=colors, width=.65)
        ax.set_xticks(np.arange(4), names, fontsize=8, rotation=12)
        ax.set_title(title, fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(values)*1.18 if key != "array_pas_cosine" else 1.05)
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:.3f}", (bar.get_x()+bar.get_width()/2, value), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=9)
        if key == "raw_global_energy_ratio_nmse":
            ax.axhline(1, color="#555555", linestyle="--", linewidth=.9)
            ax.text(.02, .95, "Dashed line: all-zero predictor NMSE = 1", transform=ax.transAxes,
                    va="top", fontsize=8, color="#555555")
    fig.suptitle("DICHASUS dc01 | 3.438 GHz | 183 held-out positions | explicitly adapted methods", fontsize=13)
    fig.supxlabel("NeWRF-SF and NeRF2-SF show amplitude degeneration on both fit and test positions.",
                  fontsize=10, color="#555555")
    fig.savefig(REPORTS/"comparison.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()
    h, meta, ids = study_data(args.fold)
    missing = [label for _, label, relative in METHODS if not (ARTIFACTS/relative/f"fold{args.fold}/result.json").is_file()]
    if missing:
        atomic_json(REPORTS/"study_status.json", {"status": "running", "missing_methods": missing,
                                                "input_adaptation_authorized": True})
        print(json.dumps({"status": "running", "missing_methods": missing}, ensure_ascii=False), flush=True)
        return
    source_verification = verify_sources()
    truth = np.array(h[ids["test"]])
    expected_ids = meta["source_id"][ids["test"]]
    methods = []
    for method, label, relative in METHODS:
        folder = ARTIFACTS/relative/f"fold{args.fold}"
        result = compare_result(folder, truth, expected_ids)
        if method != "ours" and (result["selected_step"] is None or result["selected_step"] <= 0):
            raise ValueError("The primary published-method table requires a trained checkpoint")
        if result["metrics"]["exact_zero_coefficient_fraction"] >= .99:
            raise ValueError("A collapsed model cannot be presented as a valid trained baseline")
        methods.append(dict(method=method, label=label, **result))
    negative = []
    for name, label in [("newrf", "NeWRF-SF hard-ReLU collapsed fit-health attempt"),
                        ("neural_materials", "Neural Materials-SF early-stop initial-model selection")]:
        folder = ARTIFACTS/name/f"fold{args.fold}"
        if (folder/"result.json").exists():
            negative.append(dict(label=label, principal_table=False,
                                 **compare_result(folder, truth, expected_ids, strict_source=False)))
    fit = meta["position"][ids["fit"], :2]
    held = meta["position"][np.r_[ids["val"], ids["test"]], :2]
    minimum_guard = float(np.sqrt(((fit[:, None]-held[None])**2).sum(-1)).min())
    if minimum_guard < .30:
        raise ValueError("Spatial guard was violated")
    study = {"status": "complete", "date": "2026-10-17", "scene": "dichasus-dc01",
             "frequency_hz": 3.438e9, "fold": args.fold,
             "fit_positions": len(ids["fit"]), "validation_positions": len(ids["val"]), "test_positions": len(ids["test"]),
             "guard_excluded_positions": len(h)-sum(len(part) for part in ids.values()),
             "shape": list(truth.shape), "dtype": "complex64", "minimum_xy_guard_m": minimum_guard,
             "methods": methods, "negative_or_degenerate_attempts": negative,
             "source_verification": source_verification,
             "single_frequency_only_measured_training": True,
             "original_pdp_composite_score": None,
             "published_numbers_reproduced_under_original_protocol": False,
             "limitations": ["one preselected spatial fold, not a five-fold mean", "explicitly adapted published mechanisms",
                              "NeWRF density stability parameterization and geometry proposals differ from the native paper",
                              "NeWRF/NeRF2 amplitudes degenerate even on fit positions; strong-baseline fitting is not established",
                              "no measured-query inputs; absolute measured phase remains a hard target",
                              "dc01 has been used in prior development; not untouched-acquisition confirmation"]}
    atomic_json(REPORTS/"evaluation.json", study)
    write_report(study)
    plot(study)
    atomic_json(REPORTS/"study_status.json", {"status": "complete", "methods": [item["label"] for item in methods],
                                            "trained_main_methods_verified": True, "independent_complex128_verification": True})
    print(json.dumps({"status": "complete", "test_positions": len(truth),
                      "independent_max_metric_discrepancy": max(item["independent_max_metric_discrepancy"] for item in methods),
                      "comparison": [{"method": item["label"], **item["metrics"]} for item in methods]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
