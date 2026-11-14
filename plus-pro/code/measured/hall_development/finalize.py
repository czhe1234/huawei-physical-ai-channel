"""Consolidate completed development evidence; no model updates or new scoring."""
from __future__ import annotations
import json,runpy
from pathlib import Path
import numpy as np
from .common import *
def load(name):return json.loads((REPORT/(name+".json")).read_text())
def run():
    acceptance=load("acceptance")
    if not acceptance["passed"]:raise RuntimeError("Verification has not passed")
    names=["mlp_levels3","mlp_levels6","wrf_tiny_g64","wrf_g64","wrf_g128","ours_tiny_v2","ours_config0_v2","ours_config1_v2","ours_phase_safe"]
    runs={name:load(name) for name in names}
    selected_models=["ours_config0_v2","ours_config1_v2","ours_phase_safe","mlp_levels3","mlp_levels6","wrf_g64","wrf_g128"]
    table=[dict(name=name,best_epoch=runs[name]["best_epoch"],stop_epoch=runs[name]["epochs"],fit=runs[name]["best_fit"],validation=runs[name]["best_validation"],
       last_fit=runs[name]["curve"][-1]["fit"],last_validation=runs[name]["curve"][-1]["validation"],seconds=runs[name]["seconds"]) for name in selected_models]
    kernel=load("kernel_grid")["candidates"][0]
    rt=load("rt_8m_convergence");rh=load("rt_high_capacity_convergence");contract=load("contract");data,_=init()
    fit=np.array(contract["fit_rows"]);val=np.array(contract["validation_rows"])
    fit_blocks=np.unique(np.floor(data.meta["position"][fit,:2]/4).astype(int),axis=0)
    val_blocks=np.unique(np.floor(data.meta["position"][val,:2]/4).astype(int),axis=0)
    # Persisted recipe is a reproduction bundle, explicitly not a promotion to confirmation.
    recipe=dict(status="Development candidate only; no formal recipe freeze or north promotion.",
        south_data_identity=data.identity,fit_rows=fit.tolist(),validation_rows=val.tolist(),fold=0,
        candidate="ours_phase_safe",checkpoint="artifacts/measured/hall_sw_development_v1/models/ours_phase_safe/best.pt",
        foundation="Phase-safe sqrt-power expert interpolation; NN measured carrier phase, angular magnitude writeback.",
        foundation_weights=load("phase_safe_fusion")["selected"]["weights"],core_config=runs["ours_phase_safe"]["config"],
        geometry_cache=dict(samples_per_source=1048576,max_depth=16,candidate_capacity=1048576,seeds=[SEED,SEED+1],support_cap=128,batch=12,
            numerical_convergence_certified=False,route_presence_weight="mean energy times sqrt(seed presence fraction)"),
        common_metrics="All1024 frequencies integrated before32-beam normalization; no query phase/delay alignment.",
        north_allowed=False,south_test_allowed=False)
    save_json(REPORT/"development_recipe.json",recipe)
    successful_steps=sum(r["steps"] for r in runs.values())
    summary=dict(status="Completed bounded development; optimization health passes, overall advantage and RT convergence are not established.",
        data=dict(fit_points=len(fit),validation_points=len(val),fit_blocks=fit_blocks.tolist(),validation_blocks=val_blocks.tolist(),seed=SEED,
                  formal_test_scores=False,north_channels_used=False),
        methods=table,kernel=kernel,direct_reference=load("direct_reference"),rt_development=load("rt_development_scores"),
        adaptation=dict(original_complex_fusion_validation=load("fusion")["selected"]["validation"],
                        phase_safe_fusion_validation=load("phase_safe_fusion")["selected"]["validation"],
                        phase_safe_core_validation=runs["ours_phase_safe"]["best_validation"]),
        health=dict(wrf_tiny_cosine=runs["wrf_tiny_g64"]["best_fit"]["mean_direction_cosine"],
            ours_tiny_cosine=runs["ours_tiny_v2"]["best_fit"]["mean_direction_cosine"],
            wrf_active_tensors=62,wrf_tensors_updated=62,ours_active_tensors=68,ours_tensors_updated=68,
            successful_neural_optimizer_steps=successful_steps,failed_check_attempt_extra_neural_steps=1560,
            CPU_complex_fusion_steps=1000,CPU_phase_safe_fusion_steps=1000,
            all_nn_gradient_tensors_finite=True,successful_training_wall_seconds=sum(r["seconds"] for r in runs.values()),
            maximum_reported_training_gpu_gib=max(r["peak_gpu_gib"] for r in runs.values())),
        rt=dict(initial_candidate_capacity_saturated=True,high_capacity=4194304,
            highest_checked_rays=8388608,highest_checked_depth=16,depth_energy_checks_pass_at4M=True,
            ray_and_seed_power_distribution_convergence=False,coherent_channel_convergence=False,
            main_beam_stable_in_high_ray_incoherent_probes=True,thresholds_unchanged=True,
            latest_comparisons=rt["comparisons"]),
        acceptance=acceptance,preservation=load("preservation_after"),
        limitations=["Single development seed and one validation4m block; repeatedly used for tuning, no formal inference.",
            "WRF is the explicitly modified location-to-power adaptation with64/128 fixed Gaussians; original CUDA/densification/full-paper replication unverified.",
            "Geometry cache is a provisional1M-ray two-seed input, not a numerically certified RT reference.",
            "Added model capacity, geometry and training are not isolated by this health study; no independent geometry-causal claim.",
            "No raw absolute complex-H reconstruction claim; predicted carrier phase is borrowed from permitted fit measurements."],
        failures_preserved=["ours_tiny.log","ours.log","ours_tiny_failure.json","ours_failure.json","unit_checks.log"],
        global_path_check="Non-clean historical workspace and preserved raw tracebacks. New maintained module scope passes.")
    save_json(REPORT/"development_summary.json",summary)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":180})
    fig,axes=plt.subplots(2,2,figsize=(10.8,7.3),layout="constrained")
    ax=axes[0,0]
    for name,label,color in [("wrf_tiny_g64","WRF-GS+ adaptation","#d47422"),("ours_tiny_v2","Our bounded core","#236f9d")]:
        rows=runs[name]["curve"];ax.plot([r["steps"] for r in rows],[r["fit"]["mean_direction_cosine"] for r in rows],label=label,color=color)
    ax.axhline(.98,color="gray",ls=":",lw=1);ax.set(title="Eight-position fitting check",xlabel="Optimizer updates",ylabel="Training cosine",ylim=(.2,1.01));ax.legend(fontsize=8)
    ax=axes[0,1]
    for name,label,color in [("wrf_g64","64 Gaussians","#d47422"),("wrf_g128","128 Gaussians","#5a798e")]:
        rows=runs[name]["curve"];ep=[r["epoch"] for r in rows]
        ax.plot(ep,[r["fit"]["mean_direction_cosine"] for r in rows],color=color,ls="--",label=label+" fit")
        ax.plot(ep,[r["validation"]["mean_direction_cosine"] for r in rows],color=color,label=label+" validation")
    ax.set(title="WRF-GS+ learns; validation peaks early",xlabel="Epoch",ylabel="Cosine",ylim=(.2,1.01));ax.legend(fontsize=8,ncol=2)
    ax=axes[1,0]
    for name,label,color in [("ours_config0_v2","Complex foundation, cap0.3","#367797"),("ours_config1_v2","Complex foundation, cap0.6","#aaaaaa"),("ours_phase_safe","Power-first foundation, cap0.3","#cf7a22")]:
        rows=runs[name]["curve"];ax.plot([r["epoch"] for r in rows],[r["validation"]["mean_direction_cosine"] for r in rows],label=label,color=color)
    ax.axhline(runs["wrf_g128"]["best_validation"]["mean_direction_cosine"],color="black",ls=":",lw=1,label="WRF best development")
    ax.set(title="Targeted adaptation; no established lead",xlabel="Epoch",ylabel="Validation cosine",ylim=(.78,.91));ax.legend(fontsize=8)
    ax=axes[1,1]
    first=[v for v in rh["comparisons"] if v["axis"]=="rays" and v["first"]==[1048576,16]]
    last=[v for v in rt["comparisons"] if v["axis"]=="rays_4M_to_8M"]
    for index,rs in enumerate([SEED,SEED+1]):
        a=next(v for v in first if v["seed"]==rs);b=next(v for v in last if v["seed"]==rs)
        ax.plot([4.194304,8.388608],[a["incoherent"]["p90_normalized_power_L1"],b["incoherent"]["p90_normalized_power_L1"]],marker="o",label="Seed"+str(index+1))
    ax.axhline(.05,color="black",ls="--",lw=1,label="Frozen threshold")
    ax.set(title="RT ray increments still fail the power criterion",xlabel="Higher ray budget (millions)",ylabel="P90 normalized-power L1 change",ylim=(0,.27));ax.legend(fontsize=8)
    for ax in axes.ravel():ax.grid(alpha=.15)
    fig.savefig(REPORT/"development_curves.png");fig.savefig(REPORT/"development_curves.pdf");plt.close(fig)
    rows=[]
    labels={"ours_config0_v2":"原复数融合适配 + 路径模块","ours_config1_v2":"放大修正范围的对照","ours_phase_safe":"先融合功率的针对性修改 + 路径模块","mlp_levels3":"位置 MLP（3级位置编码）","mlp_levels6":"位置 MLP（6级位置编码）","wrf_g64":"WRF-GS+ 适配版，64个高斯","wrf_g128":"WRF-GS+ 适配版，128个高斯"}
    for r in table:
        m=r["validation"];rows.append("| "+labels[r["name"]]+" | %.4f | %.3f | %d / %d |"%(m["mean_direction_cosine"],m["mean_beam_loss_db"],r["best_epoch"],r["stop_epoch"]))
    rows.append("| 方向功率核回归 | %.4f | %.3f | 闭式拟合 |"%(kernel["validation"]["mean_direction_cosine"],kernel["validation"]["mean_beam_loss_db"]))
    text="""**南侧的小规模训练和验证已经完成。训练通路和拟合能力通过检查，但当前还不能宣布算法领先，也不能冻结正式 RT 参数。**

本轮固定使用同一组96个训练位置、48个验证位置，保留原有空间划分和至少0.30m隔离。每个位置有两个阵列面板，使用全部1024个频点。样本按坐标覆盖选择。训练位置覆盖6个4m空间块，验证位置来自1个4m空间块。所有神经模型只用一个初始化种子。这是反复用于选参的开发验证，不能当作独立论文测试，96个验证面板也不是96个独立空间样本。

比较的是全频带方向功率结构和相同32波束码本的波束选择。方向功率余弦相似度越高越好，1表示形状完全一致；波束损失越低越好，0dB表示选中了该测量中的最优波束。先在全部频点累积能量，再做方向归一化；没有读取查询信道来校正预测相位或时延。

| 方法 | 验证相似度 ↑ | 验证平均波束损失/dB ↓ | 最佳轮次 / 停止轮次 |
| --- | ---: | ---: | ---: |
"""+ "\n".join(rows)+"""

下述64/128个高斯均指每个阵列面板的数量。WRF-GS+ 的小样本拟合检查达到0.9960，16个面板都选中最优波束。实际96个训练位置上，两种规模的最后训练相似度均接近0.98；验证在第5或10轮达到峰值，之后下降。62个可学习参数张量均有有限梯度且发生更新。训练后的批量渲染与逐高斯参考实现也做了输出及梯度对照，所检参数组的相对梯度误差小于6e-7，保存模型可重新生成相同预测。因此，这版小规模适配实现具备学习能力，后期下降表现为过拟合。它仍是明确改过任务头、输出算子和渲染执行方式的适配版，未核验原作者CUDA核、动态增密及完整论文规模复现，不能把这里的成绩标成原论文复现成绩。

我们的路径模块在8个训练位置上达到0.9849，68个可学习参数张量都有有限梯度并更新。保留了最终算法的路径序列与集合编码器。针对实测任务，改用方向功率损失拟合载体融合；限制方向修正幅度；学习每个位置采用多大几何修正；不监督缺少统一参考的绝对时延，并只向复数载体写回可识别的方向能量。复数输出独立DFT复算与训练功率的最大差异小于3e-5，阵列能量保持检查通过。

第一轮复数底座的验证相似度为0.8239，加入路径模块后为0.8572。扩大方向修正范围和几何权重上限没有改善验证。之后只增加一个有物理依据的修改：不同位置的测量先在方向功率域融合，保留允许训练邻点的复数相位结构。其底座提高到0.8597，加入路径模块后为0.8742。这一修改改善了当前开发结果，但尚未超过可学习基线，波束损失仍偏高。模块前后差异同时包含模型容量、训练和几何信息，不能单独解释为正确几何的因果贡献。本轮也没有证明原始完整复信道恢复。

RT 检查使用12个按位置覆盖选定的训练/验证位置、两个随机种子，完全不使用信道真值选预算。射线数均按每个阵列发射源计算。最初每发射源32768个候选路径的容量确实填满，会截断候选。提高到4194304后，本轮最大约195万个候选，没有填满；该容量还影响原运行时的路径去重哈希表大小。逐步检查了4096到8388608条发射射线、2到16层传播，阈值保持不变。

在约419万条射线下，按各路径能量相加的方向分布，从8层加到12层、再加到16层，均通过本轮增深检查，主波束也稳定。因此8层可作为这一功率模型的深度候选，16层是保守检查上限。这个判断不能迁移为相干复信道已经收敛。

射线数仍不能冻结。约419万增加到839万后，两种子下方向分布变化的90%分位L1距离为0.0850和0.2365，均超过预设0.05。最高预算的种子间差异为0.1741。主波束在这24条按路径能量计算的链路上没有改变，但整条方向分布仍不稳定；相干叠加的波动更大。当前训练采用的约105万射线、16层、两个种子的几何缓存，只是开发输入，未取得正式RT收敛资格。128条保留路径与候选路径容量是两个不同限制，完整保留率记录在几何缓存和射线检查JSON中。上述检查也不验证固定材料、未建模散射/绕射或模型几何的物理准确性。

6项新增功能检查通过，已保存模型的预测与独立评价复算通过。旧源代码、准备数据与结果共4933个文件身份检查通过，实验记录原有字节前缀保留。只新增了开发版本和结果，未替换原始文件。全工作区路径检查仍不干净，包含历史路径及为保留诊断而保存的原始报错；新维护模块的路径与语法检查通过。

检查复数写回时曾有一个评估维度广播错误，误报输出功率不一致，已经修复并用独立DFT及已保存模型复算。首次报错、源码副本和轨迹保留，未将失败检查掩盖为成功。另补齐了新模型的预测接口及零能量载体处理。修改后重新生成的预测与训练保存输出一致。

本轮没有取出南侧测试信道用于训练或评分，没有读取北侧信道、生成北侧预测或用北侧选参。原始数据、旧数据和已有结果保留。现在可确认实现能够学习、早停有效、针对性改动有开发集收益；仍缺独立空间验证的总体性能优势、正式基线规模核验和可靠RT方向分布预算，不能据此直接写论文优越性结论。

![训练和RT检查曲线](development_curves.png)

可复现的开发配方见development_recipe.json，完整轨迹见各模型JSON，独立核验见acceptance.json，射线检查见rt_8m_convergence.json及前序结果。开发配方明确禁止直接用于北侧确认。
"""
    (REPORT/"development_zh.md").write_text(text,encoding="utf-8")
    source_module=runpy.run_path(str(PATHS.code_root/"tools/check_hardcoded_paths.py"))
    bad=[]
    for path in Path(__file__).parent.glob("*.py"):
        compile(path.read_text(),str(path),"exec")
        for i,line in enumerate(path.read_text().splitlines(),1):
          for label,pattern in source_module["PATTERNS"]:
            if pattern.search(line):bad.append([path.name,i,label])
    if bad:raise ValueError(bad)
    save_json(REPORT/"final_source_identity.json",dict(source_sha256={str(p.relative_to(PATHS.root)):sha256(p) for p in Path(__file__).parent.glob("*.py")},
        new_source_path_check_passed=True,source_compiles=True,training_source_changes_after_checkpoints="Inference-only method/WRF entry points and zero-carrier guard; saved predictions independently regenerated unchanged."))
    ledger=PATHS.root/"experiment_records/EXPERIMENT_RECORD.md"
    if "hall_south_development_v1/development_zh.md" not in ledger.read_text():
      with ledger.open("a") as f:
        f.write("""
### Completion
- Completed common fold0 south96-fit/48-validation development, all1024 bins, one seed. Validation covers only one4m block and was reused for tuning; no formal south-test scoring or north channel access.
- Learnable optimization health passes. WRF adaptation tiny8-position cosine0.9960; both64/128 Gaussian versions end near0.98 fit cosine, with early validation peaks. All62 active parameter tensors update with finite gradients. Batched-versus-reference trained forward/gradient checks pass, original CUDA and full-paper reproduction remain unverified.
- Measured final-core tiny-fit cosine0.9849. All68 active tensors update with finite gradients. Bounded angular corrections, learned geometry trust, phase-invariant fusion supervision and angle-only complex writeback are explicit measured adaptations, not unchanged competition inference.
- Complex-foundation core validation cosine0.8572. Enlarged correction range/cap does not improve it. One physics-motivated power-first foundation adaptation improves core validation to0.8742 with1.803dB beam loss. MLP best0.8756; WRF best0.8803 with1.111dB. No established overall advantage or isolated geometry-causal claim.
- RT candidate capacity32768 saturates; larger capacity also changes deduplication hash-table size. Capacity4194304 remains unsaturated through8M rays/depth16 on12 geometry-selected probes/two seeds. Path-energy depth8-to12-to16 checks pass at4M rays, but4M-to8M and8M seed power-distribution L1 checks fail frozen0.05 criterion. Main beam is stable in the incoherent high-ray probes; coherent predictions remain unstable. Formal RT rays/depth recipe is not frozen;1M-ray two-seed geometry cache remains provisional development input.
-6 new functional checks,5 saved-model inference/metric recomputations, trained WRF reference parity and new-source path/syntax checks pass. Preserved4933 old file identities and ledger prefix. Full workspace path check remains non-clean from history and preserved tracebacks. No original-source/data/result replacement.
- Preserved an evaluation singleton-axis broadcasting failure and its first-attempt trajectories/source, then fixed and verified independently. Added correct position-only inference entry points and zero-carrier handling; checkpoint outputs regenerate unchanged.
- Evidence: analysis/results/hall_south_development_v1/development_zh.md,development_summary.json,acceptance.json,development_recipe.json,final_source_identity.json and full curves/checkpoints under artifacts/measured/hall_sw_development_v1. This completes bounded development, not formal test, baseline-paper replication or confirmation promotion.
""")
    print("REPORT_COMPLETE",REPORT.name,successful_steps,flush=True)
if __name__=="__main__":run()
