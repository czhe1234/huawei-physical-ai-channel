"""Build the bounded-development result document directly from saved values."""
from __future__ import annotations
import json,csv
import numpy as np
from .common import *
def main():
    v=json.loads((REPORT/"verification.json").read_text());assert v["passed"]
    names=["tiny_default","tiny_delay","ours_default","ours_delay","ours_trust","ours_frequency","no_geometry_default","spectral_mlp_w1","spectral_mlp_w4"]
    runs={n:json.loads((REPORT/(n+".json")).read_text()) for n in names};assert all(r["complete"] for r in runs.values())
    base=json.loads((REPORT/"baseline_comparison.json").read_text())["results"]
    selected=v["selected_ours"];selected_mlp=v["selected_spectral_mlp"]
    rows=[]
    for name in ["nearest_FIT","uniform_frequency_experts","frequency_RBF","previous_angular_ours"]:
        rows.append(dict(name=name,**base[name]["metrics"]))
    for name in ["ours_default","ours_delay","ours_trust","ours_frequency","no_geometry_default","spectral_mlp_w1","spectral_mlp_w4"]:
        rows.append(dict(name=name,**runs[name]["best_validation"]))
    for name in ["wrf_g64","wrf_g128","mlp_levels3","mlp_levels6"]:
        m=dict(base[name]["metrics"]);m["direction_cosine"]=m["mean_direction_cosine"];rows.append(dict(name=name,**m))
    columns=["name","structural_selection","direction_cosine","covariance_NMSE","frequency_correlation_MAE","beam_frequency_correlation_MAE","mean_beam_loss_db","p90_beam_loss_db","beam_loss_over3_fraction","relative_CSI_power_MAE_db"]
    with (REPORT/"comparison.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction="ignore");w.writeheader();w.writerows(rows)
    with (REPORT/"curves.csv").open("w",newline="") as f:
        cols=["name","epoch","steps","split","structural","angle_loss","spatial_loss","frequency_loss","power_loss","direction_cosine","covariance_NMSE","beam_frequency_MAE","beam_loss_db"]
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for name,r in runs.items():
            for row in r["curve"]:
                for split in ["fit","validation"]:
                    m=row[split]
                    if m is None:continue
                    terms=m["loss_terms"]
                    w.writerow(dict(name=name,epoch=row["epoch"],steps=row["steps"],split=split,
                        structural=m["structural_selection"],angle_loss=terms["angle"],spatial_loss=terms["spatial"],
                        frequency_loss=terms["frequency"],power_loss=terms["power"],direction_cosine=m["direction_cosine"],
                        covariance_NMSE=m["covariance_NMSE"],beam_frequency_MAE=m["beam_frequency_correlation_MAE"],beam_loss_db=m["mean_beam_loss_db"]))
    summary=dict(complete=True,scope="South fold0 small-scale development; not formal testing or confirmation",
        selected_ours=selected,selected_spectral_MLP=selected_mlp,comparison=rows,
        run_health={n:dict(epochs=r["epochs"],steps=r["steps"],best_epoch=r["best_epoch"],stop_reason=r["stop_reason"],
            structural_fit_reduction=r["health"]["structural_fit_reduction"],updated_tensors=r["health"]["updated_tensors"],
            parameter_tensors=r["health"]["active_tensors"],nonzero_gradient_tensors=sum(x>0 for x in r["health"]["gradient_max"].values()),
            seconds=r["seconds"],peak_GPU_GiB=r["peak_GPU_GiB"],trainable_parameters=r["trainable_parameters"],
            device=r["contract"]["environment"]["device"],UUID=r["contract"]["environment"]["visible_UUID"]) for n,r in runs.items()},
        total_model_process_minutes=sum(r["seconds"] for r in runs.values())/60,
        claims=dict(learning="supported within bounded health/development scope",validation_convergence="predeclared early-stop plateaus, not a global optimum or full training-loss convergence",
          overall_advantage="not established; mixed output domains",geometry_benefit="not observed; same-model no-geometry control is better in this development comparison",
          absolute_complex_channel="not evaluated; references remain insufficient",formal_confirmation="not run"),
        tuning_decisions=["tuning_decisions_01.json","tuning_decisions_02.json"],verification=v,source_sha256=source_hashes(),north=False,test=False)
    save_json(REPORT/"development_summary.json",summary)
    make_figure(runs,base,selected,selected_mlp)
    write_report(runs,base,selected,selected_mlp,summary)
    print("PACKAGED",selected,selected_mlp,summary["total_model_process_minutes"],flush=True)
def make_figure(runs,base,selected,selected_mlp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    assert "Arial" in font_manager.findfont("Arial",fallback_to_default=False)
    plt.rcParams.update({"font.family":"Arial","font.size":6,"axes.labelsize":7,"xtick.labelsize":5.5,"ytick.labelsize":5.5,
      "axes.linewidth":.5,"lines.linewidth":.8,"pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"})
    fig,axes=plt.subplots(2,2,figsize=(183/25.4,112/25.4))
    fig.subplots_adjust(left=.085,right=.98,bottom=.15,top=.95,wspace=.31,hspace=.48)
    for label,ax in zip("abcd",axes.flat):
        ax.spines["top"].set_visible(False);ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out",width=.5,length=2)
        ax.text(-.17,1.07,label,transform=ax.transAxes,fontweight="bold",fontsize=8,va="top")
    ax=axes[0,0];c=runs["tiny_default"]["curve"];x=[r["steps"] for r in c]
    for key,color,label,dy in [("angle","#0072B2","Angle/frequency",4),("spatial","#D55E00","Spatial",7),("frequency","#009E73","Frequency",-9)]:
        y=[r["fit"]["loss_terms"][key] for r in c];ax.plot(x,y,color=color,marker={"angle":"o","spatial":"^","frequency":"s"}[key],ms=2.2,markevery=2)
        ax.annotate(label,(x[-1],y[-1]),xytext=(-3,dy),textcoords="offset points",ha="right",fontsize=5.5,color=color)
    ax.set(xlabel="Optimizer updates",ylabel="Target error (dimensionless)",yscale="log",ylim=(.015,.55))
    ax=axes[0,1];c=runs[selected]["curve"];x=[r["steps"] for r in c]
    for split,color,style,dy in [("fit","#555555","--",6),("validation","#0072B2","-",5)]:
        y=[r[split]["structural_selection"] for r in c];ax.plot(x,y,color=color,ls=style)
        ax.annotate("FIT" if split=="fit" else "Validation",(x[-1],y[-1]),xytext=(-3,dy),textcoords="offset points",ha="right",fontsize=5.5,color=color)
    ax.axvline(runs[selected]["best_epoch"]*3,color="#999999",ls=":",lw=.6)
    ax.set(xlabel="Optimizer updates",ylabel="Structural error (dimensionless)")
    ax=axes[1,0]
    pts=[("Ours",runs[selected]["best_validation"],"#0072B2","*",5,5),
         ("No geometry",runs["no_geometry_default"]["best_validation"],"#D55E00","D",4,-10),
         ("MLP",runs[selected_mlp]["best_validation"],"#009E73","s",5,4),
         ("RBF",base["frequency_RBF"]["metrics"],"#555555","o",-20,6),
         ("Old angular",base["previous_angular_ours"]["metrics"],"#CC79A7","^",-39,-10)]
    for name,m,color,marker,dx,dy in pts:
        xx,yy=m["covariance_NMSE"],m["beam_frequency_correlation_MAE"]
        ax.plot(xx,yy,marker=marker,color=color,ms=5 if marker=="*" else 3,ls="")
        ax.annotate(name,(xx,yy),xytext=(dx,dy),textcoords="offset points",fontsize=5.5,color=color)
    ax.set(xlabel="Covariance NMSE (dimensionless)",ylabel="Per-beam frequency MAE",xlim=(.88,1.19),ylim=(.069,.14))
    ax=axes[1,1]
    names=["Ours","No geom.","RBF","MLP","WRF"]
    values=[runs[selected]["best_validation"]["mean_beam_loss_db"],runs["no_geometry_default"]["best_validation"]["mean_beam_loss_db"],
        base["frequency_RBF"]["metrics"]["mean_beam_loss_db"],runs[selected_mlp]["best_validation"]["mean_beam_loss_db"],
        base["wrf_g128"]["metrics"]["mean_beam_loss_db"]]
    bars=ax.bar(np.arange(5),values,color=["#0072B2"]+["#BBBBBB"]*4,width=.6)
    for bar,value in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,value+.04,f"{value:.2f}",ha="center",va="bottom",fontsize=5.5)
    ax.set_xticks(np.arange(5),names,rotation=15);ax.set(ylabel="Selected-beam loss (dB)",ylim=(0,max(values)+.45))
    for suffix in ["png","pdf","svg","eps"]:
        fig.savefig(REPORT/("learning_and_comparison."+suffix),dpi=400)
    plt.close(fig)
def write_report(runs,base,selected,selected_mlp,summary):
    full=runs[selected]["best_validation"];control=runs["no_geometry_default"]["best_validation"]
    kernel=base["frequency_RBF"]["metrics"];tiny=runs["tiny_default"]
    labels=[("最近训练点",base["nearest_FIT"]["metrics"]),("频率核插值",kernel),
        ("位置MLP复信道适配",runs[selected_mlp]["best_validation"]),("上版角度方法",base["previous_angular_ours"]["metrics"]),
        ("新版：含几何",full),("新版：去掉几何",control),("WRF-GS+现有适配",base["wrf_g128"]["metrics"])]
    table=["| 方法 | 综合结构误差↓ | 空间相关误差↓ | 逐波束频率误差↓ | 方向余弦↑ | 波束损失/dB↓ | 相对CSI功率误差/dB↓ |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for name,m in labels:
        keys=["structural_selection","covariance_NMSE","beam_frequency_correlation_MAE","direction_cosine","mean_beam_loss_db","relative_CSI_power_MAE_db"]
        vals=[m.get(k,m.get("mean_direction_cosine") if k=="direction_cosine" else None) for k in keys]
        table.append("| "+name+" | "+" | ".join("—" if x is None else f"{x:.4f}" for x in vals)+" |")
    tuning=["| 设置 | 最佳验证轮次 | 综合结构误差↓ | 逐波束频率误差↓ | 波束损失/dB↓ |","|---|---:|---:|---:|---:|"]
    for name in ["ours_default","ours_delay","ours_trust","ours_frequency"]:
        r=runs[name];m=r["best_validation"];tuning.append(f'| {name} | {r["best_epoch"]} | {m["structural_selection"]:.4f} | {m["beam_frequency_correlation_MAE"]:.4f} | {m["mean_beam_loss_db"]:.4f} |')
    lines=[
      "南侧挡板场景：新结构版的小规模训练与基线比较（2026-10-24至25）",
      "",
      "新版本已经能正常学习，训练过程未出现非有限损失或梯度；验证误差按预设标准进入平台并提前停止。但本轮尚未建立整体优于基线的结论，加入几何也没有带来正收益。",
      "",
      f'8个训练位置的小样本拟合中，综合结构误差从{tiny["curve"][0]["fit"]["structural_selection"]:.4f}降到{tiny["best_fit"]["structural_selection"]:.4f}，下降{100*tiny["health"]["structural_fit_reduction"]:.1f}%，方向余弦达到{tiny["best_fit"]["direction_cosine"]:.4f}。85组可训练参数全部更新并出现非零梯度。方向—频率联合能量和空间结构误差下降，频率相关损失却从{tiny["curve"][0]["fit"]["loss_terms"]["frequency"]:.4f}升到{tiny["best_fit"]["loss_terms"]["frequency"]:.4f}；因此拟合检查通过不等于每一类结构都学得更好。',
      "",
      "下面是相同南侧96个训练位置、48个验证位置上的结果。综合结构误差是多尺度角度—频率Hellinger、归一化协方差半Frobenius误差、频率相关均方误差的等权平均。表中的空间相关误差另用相对Frobenius协方差NMSE表示；逐波束频率误差为按参考方向能量加权的相关幅度MAE。这些都从最终生成的信道计算。功率仅表示相对于训练参考的CSI功率，不是dBm。",
      "",
      *table,
      "",
      f'按预先规定的综合结构指标，四个含几何候选中选择{selected}。它相对频率核插值的综合结构误差降低{100*(1-full["structural_selection"]/kernel["structural_selection"]):.1f}%，协方差NMSE降低{100*(1-full["covariance_NMSE"]/kernel["covariance_NMSE"]):.1f}%。但方向余弦、频率相关、波束损失和相对功率均未超过该插值基线；位置MLP的协方差NMSE也低于含几何版本。不能据综合指标一项改善写成全面优越。',
      "",
      f'去掉几何而保留其余结构的对照，综合误差为{control["structural_selection"]:.4f}，优于含几何版本的{full["structural_selection"]:.4f}，频率和波束指标也略好。因此当前数据准备、暂用射线缓存和小样本训练条件下，没有观察到几何模块的额外收益。这不是“几何永远无效”的结论，也不能把原因直接归于射线缓存，因为本轮没有独立检验这种因果解释。',
      "",
      "调参按观察到的问题进行。默认小样本拟合中，时延、频谱和接收能量修正大量触及限幅，于是测试了更宽修正范围、原训练数据学得的专家初始化及较低几何权重。专家初始化的历史检查点曾用同一验证集选取，属于已反复使用的开发信息。又因为频率误差在优化中退步，增加一组训练频率权重1与4的对照；验证定义、综合选轮标准和数据划分保持不变。对应位置MLP也各训练权重1与4两个候选。所有尝试均保留，不只展示最好一次。",
      "",
      *tuning,
      "",
      f'在其余配置相同的trust设置上，提高训练频率权重使逐波束频率MAE从{runs["ours_trust"]["best_validation"]["beam_frequency_correlation_MAE"]:.4f}降到{runs["ours_frequency"]["best_validation"]["beam_frequency_correlation_MAE"]:.4f}，但综合指标没有改善。这支持“当前优化存在结构目标取舍”的观察，尚不支持频率优势已经恢复。',
      "",
      "![训练轨迹与开发比较](learning_and_comparison.png)",
      "",
      "图注：a，8个训练位置的小样本拟合，各分项均为实际保存轮次上的最终信道误差；纵轴为对数尺度。b，按综合结构误差选出的含几何版本，显示训练与验证轨迹，虚线标出所选检查点。c，不同方法的协方差与逐波束频率误差，左下方较好。d，相同32个接收波束上的平均选择损失，含WRF-GS+。验证集为48个位置、每位置2副阵列，只有1个4米空间块；单个随机种子。没有把阵列、训练轮次或超参数候选当作独立重复，也没有据此绘制置信区间。",
      "",
      "实际完成范围：2个小样本健康设置、4个含几何开发设置、1个去几何对照、2个位置MLP复信道适配，另有12个频率核插值候选。WRF-GS+64/128及原功率MLP复用了已有健康训练的预测并重新计算相同方向、波束指标；本轮没有重新训练WRF，也没有赋予它不具备的复数信道输出。原始作者CUDA实现及完整论文复现的既有边界仍保留。",
      "",
      "所有新模型使用同一随机种子和同一空间划分。完整信道仍有2副8×4接收阵列和1024频点。训练查询排除本身所在的4米块，验证只使用96个训练邻点，未按验证真值修正预测相位、时延或幅度。几何仍使用旧的基础传播、2种子暂用缓存，没有新RT，没有伪造散射或波动统计。",
      "",
      f'完整开发运行采用批次32，批次计时显示吞吐约276样本/秒，高于批次16的165；最多320轮对应最多960次更新，避免只增大批次而减少原定更新预算。实际各模型在验证平台后停止，完整含几何设置运行45至60轮。模型进程累计约{summary["total_model_process_minutes"]:.1f}分钟，包括准备和评价；含几何、对照使用允许的两张H100，新增MLP使用允许的A800。全部为FP32/complex64且关闭TF32。不同设备的耗时不用于推断算法速度优势。',
      "",
      "这里的“平台”指连续25轮未取得至少0.0002的验证改善；它是预设的开发停止标准，不是训练损失达到全局最优的证明。各轮曲线和各分项误差均已保存。",
      "",
      f'最佳含几何模型、去几何对照和最佳新增MLP均从检查点重新生成全部验证预测；生成阶段只读取训练信道及验证位置，之后才开放验证标签作复算。预测与保存输出一致；另以NumPy独立计算DFT、协方差和频率相关，核对主要损失。保存性检查覆盖{summary["verification"]["preservation"]["files"]}个既有文件，未发现修改、删除或新区域以外的新增文件，旧实验记录前缀保持不变。',
      "",
      "本轮能支撑的是：结构恢复版可训练，能在训练位置改善主要结构目标，并在这一个反复用于开发的空间块上，相对若干基线取得综合结构改善。还不能支撑整体优越、稳定独立的几何贡献、完整绝对复信道恢复或TWC主结论的确认验证。主要待解决问题是频率结构与其他目标的取舍、几何输入的真实增益以及相对功率泛化误差；不能靠继续扩大同一验证集上的搜索来替代证据。",
      "",
      "南侧正式测试和北侧确认数据均未使用。原始数据、准备数据、旧实现、检查点及成绩全部保留。建议先针对这些已观察到的问题讨论下一轮修改，再决定是否扩展实验；本轮不启动正式测试。",
      "",
      "复查材料：development_summary.json为汇总；comparison.csv为全部比较；curves.csv为完整已记录轨迹；各模型同名JSON为原始训练与验证结果；verification.json为复算和保存性检查。源代码位于code/measured/hall_structured_development；旧实现保持不变，训练频率权重调整前的运行源码另保存在新工作区的runtime_revisions/before_frequency_balance。"
    ]
    (REPORT/"development_zh.md").write_text("\n".join(lines)+"\n")
if __name__=="__main__":main()
