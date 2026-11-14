"""Independently verify the fixed-material reference and write its comparison."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from report_study import independent_metrics
from study_common import ARTIFACTS, PATHS, REPORTS, atomic_json, sha256, study_data


def main():
    torch.set_num_threads(4)
    reports = REPORTS/"fixed_rt"
    result = json.loads((reports/"evaluation.json").read_text())
    folder = ARTIFACTS/Path(result["variants"][0]["prediction_path"]).parents[1]
    if result["status"] != "complete" or result["neural_network_training"]:
        raise ValueError("Expected completed fixed-material evaluation without network training")
    for relative, expected in result["identity"]["files"].items():
        if sha256(PATHS.root/relative) != expected:
            raise ValueError(f"Frozen RT dependency changed: {relative}")
    h, meta, ids = study_data(0)
    truth = np.array(h[ids["test"]])
    expected_ids = meta["source_id"][ids["test"]]
    fit = np.load(folder/"fit_raw_prediction.npy", allow_pickle=False)
    fit_truth = np.array(h[ids["fit"]])
    gain = np.sqrt(np.sum(np.abs(fit_truth.astype(np.complex128))**2)/np.sum(np.abs(fit.astype(np.complex128))**2))
    if not np.isclose(gain, result["calibration"]["amplitude_gain"], rtol=1e-14, atol=0):
        raise ValueError("Gain does not match the registered fit-only formula")
    rows, discrepancies = [], []
    keys = ["raw_global_energy_ratio_nmse", "mean_position_nmse", "median_position_nmse",
            "power_mae_db", "array_pas_cosine", "dft_beam_loss_db", "diagnostic_panel_phase_aligned_nmse"]
    for variant in result["variants"]:
        prediction_file = ARTIFACTS/variant["prediction_path"]
        prediction = np.load(prediction_file, allow_pickle=False)
        source_ids = np.load(prediction_file.parent/"test_source_ids.npy", allow_pickle=False)
        if prediction.dtype != np.complex64 or prediction.shape != truth.shape or not np.isfinite(prediction).all():
            raise ValueError("Invalid frozen RT prediction")
        if not np.array_equal(source_ids, expected_ids) or sha256(prediction_file) != variant["prediction_sha256"]:
            raise ValueError("RT test identity or prediction checksum mismatch")
        independent = independent_metrics(prediction, truth)
        saved = variant["metrics"]
        for key in keys:
            expected = saved["numeric_floor_power_mae_db"] if key == "power_mae_db" else saved[key]
            discrepancies.append(abs(independent[key]-expected))
        rows.append({"method": "固定材料 RT：原始尺度" if variant["name"] == "raw" else "固定材料 RT：仅全局功率标度校准",
                     "raw_nmse": saved["raw_global_energy_ratio_nmse"], "power_mae_db": saved["power_mae_db"],
                     "array_pas_cosine": saved["array_pas_cosine"], "beam_loss_db": saved["dft_beam_loss_db"],
                     "energy_ratio": variant["prediction_energy_over_reference_energy"]})
    if max(discrepancies) > 1e-8:
        raise ValueError(f"Independent RT metric verification failed: {max(discrepancies)}")
    study = json.loads((REPORTS/"evaluation.json").read_text())
    for item in study["methods"]:
        if item["method"] in ["neural_materials", "ours"]:
            m = item["metrics"]
            rows.append({"method": item["label"], "raw_nmse": m["raw_global_energy_ratio_nmse"],
                         "power_mae_db": m["power_mae_db"], "array_pas_cosine": m["array_pas_cosine"],
                         "beam_loss_db": m["dft_beam_loss_db"], "energy_ratio": m["prediction_energy_over_reference_energy"]})
    verification = {"status": "passed", "independent_cpu_complex128_torch": True,
                    "maximum_metric_difference": max(discrepancies), "test_positions": len(truth),
                    "source_dependencies_verified": len(result["identity"]["files"]),
                    "fit_only_gain_recomputed": float(gain), "neural_network_training": False,
                    "comparison": rows}
    atomic_json(reports/"verification.json", verification)
    lines = []
    for row in rows:
        power = "∞" if row["power_mae_db"] is None else f"{row['power_mae_db']:.3f}"
        lines.append(f"| {row['method']} | {row['raw_nmse']:.6f} | {power} | {row['array_pas_cosine']:.6f} | {row['beam_loss_db']:.3f} |")
    calibrated = rows[1]
    ours = next(row for row in rows if row["method"] == "我们的单频版")
    text = """# dc01 固定材料射线追踪参考

本轮直接计算固定场景材料的电磁场，没有材料网络、路径修正网络或邻近实测信道合成。与 M5 相同：3.438GHz、空间第0折、544个训练位置和183个测试位置、两个8×4阵列。

| 方法 | 原始全局 NMSE ↓ | 功率 MAE / dB ↓ | 阵列 PAS ↑ | 波束损失 / dB ↓ |
|---|---:|---:|---:|---:|
"""+"\n".join(lines)+f"""

第一行完全保留物理RT输出尺度。第二行只乘一个正实数增益 {gain:.9g}，对应功率标度偏置 {result['calibration']['power_offset_db']:.3f}dB；增益由全部544个训练位置的总实测能量与总RT能量之比确定。它不使用验证/测试标签，不修正相位，不按位置或面板分别标定，不改变材料、路径、归一化角度谱或波束选择。第二行应明确标成“全局标度校准RT”，不能冒充完全未校准RT。

原始RT幅度与实测存储信道的整体尺度并未预先统一，因此第一行的功率误差和接近全零基准的NMSE需要结合全局增益一起解释。全局标度校准只能处理整体尺度差异，不能处理逐位置功率偏差、天线模型误差、材料误差或相位误差。

本轮全局标度校准RT的功率MAE比我们低 {ours['power_mae_db']-calibrated['power_mae_db']:.3f}dB，说明目前不能宣称我们的功率预测优于该物理参考。我们的PAS相似度高 {ours['array_pas_cosine']-calibrated['array_pas_cosine']:.6f}，波束损失低 {calibrated['beam_loss_db']-ours['beam_loss_db']:.3f}dB，原始NMSE也较低；但二者原始NMSE均大于全零预测基准1，完整复信道恢复仍未解决。该比较同时包含实测基础场与路径网络，不能把改善全部归因于学习路径修正。

配置复用Neural Materials对照的Sionna0.19缓存：简化场景、固定材料、深度5、每源一百万采样、合成阵列、直达/反射/绕射、无漫散射。它是一个明确的纯物理RT参考，但并非我们原depth16/depth10、24种子候选缓存的完整相干场重建；本轮也不能量化候选路径网络自身的独立贡献。

独立CPU complex128/PyTorch复算最大指标差异为 {max(discrepancies):.3g}。预测形状为[183,2,8,4,1] complex64，并核对测试源记录顺序、有限值、预测校验和、冻结源码/几何依赖以及训练集增益公式。单频PDP和原综合分数不定义。结果属于一个已用于开发的场景和空间划分。

计算耗时约 {result['elapsed_seconds']:.2f}秒，硬件 {result['hardware']}；复用了先前约 {result['geometry_preprocessing_seconds_reused']:.2f}秒生成的几何缓存。此时间不是端到端速度排名。

对应预测与身份记录位于 `{folder.relative_to(PATHS.root).as_posix()}/`；唯一实验过程记录见 `experiment_records/EXPERIMENT_RECORD.md` 的 Measured-M6。
"""
    (reports/"evaluation_report_zh.md").write_text(text, encoding="utf-8")
    with (reports/"comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(verification, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
