"""Paper-readable summaries of fixed-output new-metric reevaluation."""
from __future__ import annotations
import json,csv
import numpy as np
from .common import *
def main():
    r=json.loads((REPORT/"evaluation.json").read_text());assert r["complete"]
    records=r["full_channel_records"];lookup={x["name"]:x for x in records}
    main_names=["nearest_FIT","frequency_RBF","ours_phase_safe","ours_default","no_geometry_default","spectral_mlp_w1"]
    fields=["name","label","family","best_epoch",*KEYS,"broadband_direction_cosine","beam_loss_db"]
    with (REPORT/"comparison.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for x in records:
            w.writerow(dict(name=x["name"],label=x["label"],family=x["family"],best_epoch=x["best_epoch"],
                **{k:x["scores"][k] for k in KEYS},broadband_direction_cosine=x["auxiliary"]["mean_direction_cosine"],beam_loss_db=x["auxiliary"]["mean_beam_loss_db"]))
        for x in r["broadband_only_records"]:
            w.writerow(dict(name=x["name"],label=x["label"],family="broadband_only",
                broadband_direction_cosine=x["auxiliary"]["mean_direction_cosine"],beam_loss_db=x["auxiliary"]["mean_beam_loss_db"]))
        for x in r["receive_energy_controls"]:
            w.writerow(dict(name=x["name"],label="均分参照" if x["name"].startswith("uniform") else "训练平均占比参照",family="energy_only",receive_energy_TV=x["receive_energy_TV"]))
    with (REPORT/"per_panel.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["method","panel",*KEYS]);w.writeheader()
        for x in records:
            for p in range(2):w.writerow(dict(method=x["name"],panel=p,**{k:x["scores"]["by_panel"][k][p] for k in KEYS}))
    ours=lookup["ours_default"]["scores"];old=lookup["ours_phase_safe"]["scores"];rbf=lookup["frequency_RBF"]["scores"];none=lookup["no_geometry_default"]["scores"]
    contrasts=dict(new_minus_old={k:ours[k]-old[k] for k in KEYS},
        new_minus_rbf={k:ours[k]-rbf[k] for k in KEYS},
        geometry_minus_no_geometry={k:ours[k]-none[k] for k in KEYS})
    save_json(REPORT/"metric_contrasts.json",contrasts)
    plot(r,lookup,main_names)
    table=["| 固定输出 | 逐频点方向谱↑ | 相对PDP↑ | 接收能量占比误差↓ | 通道—时延联合误差↓ |",
           "|---|---:|---:|---:|---:|"]
    for name in main_names:
        x=lookup[name];s=x["scores"];table.append("| "+x["label"]+" | "+" | ".join(f"{s[k]:.4f}" for k in KEYS)+" |")
    for x in r["receive_energy_controls"]:
        table.append("| "+("通道均分参照" if x["name"].startswith("uniform") else "训练平均占比参照")+f' | — | — | {x["receive_energy_TV"]:.4f} | — |')
    appendix=["| 固定输出 | 原选轮次 | 逐频点方向谱↑ | 相对PDP↑ | 接收能量占比误差↓ | 联合误差↓ |",
              "|---|---:|---:|---:|---:|---:|"]
    for x in records:
        s=x["scores"];epoch="固定公式" if x["best_epoch"] is None else str(x["best_epoch"])
        appendix.append("| "+x["label"]+" | "+epoch+" | "+" | ".join(f"{s[k]:.4f}" for k in KEYS)+" |")
    power=["| 原宽带输出 | 宽带方向余弦↑ | 平均波束损失/dB↓ | 新四项指标 |","|---|---:|---:|---|"]
    for x in r["broadband_only_records"]:
        a=x["auxiliary"];power.append(f'| {x["label"]} | {a["mean_direction_cosine"]:.4f} | {a["mean_beam_loss_db"]:.4f} | 不适用 |')
    text=[
      "南侧已有预测的新口径补评（2026-10-25）",
      "",
      "新口径揭示了具体的取舍：相对旧功率融合版，新版默认检查点的逐频点方向谱和接收通道能量占比改善，但相对PDP和接收通道—时延联合分布退步。它优于已有位置MLP的四项指标，却没有超过更强的插值或简单能量参照，当前仍不能写成整体优势已经建立。",
      "",
      "这轮只补评已有固定预测。保持南侧96个训练位置、48个验证位置和旧检查点选择，未重新训练、调参或按新指标挑选轮次。13份复信道输出、4份原宽带功率输出，以及2个能量占比固定参照全部列出。主图中的“New”固定指旧综合口径选择的默认新版，“Old”指旧口径选择的功率融合版，“MLP”指原综合口径选中的标准损失位置MLP；没有看完新成绩再改变这些名称。",
      "",
      "主比较如下。前两项是余弦相似度，越大越好；后两项是归一化能量总变差距离，越小越好。能量占比误差不评价总接收功率，联合误差还检查能量是否位于正确接收通道的正确相对时延。各项分别报告，不合成新总分。",
      "",
      *table,
      "",
      f'方向谱：新版{ours["pas_cosine"]:.4f}高于旧版{old["pas_cosine"]:.4f}，48个验证位置中有32个更好；但低于频率核插值{rbf["pas_cosine"]:.4f}，只有12/48个位置超过它。新的逐频点评价与旧宽带方向余弦不是同一个数，不能把数值跨口径直接比较。',
      "",
      f'相对PDP：新版{ours["relative_pdp_cosine"]:.4f}低于旧版{old["relative_pdp_cosine"]:.4f}，也低于最近训练点和频率核插值。这说明当前恢复的时延自由度尚未转化为这批验证位置上更好的相对时延能量形状。它不是绝对传播时间误差，也不能据此识别退步的唯一原因。',
      "",
      f'接收能量占比：新版误差{ours["receive_energy_TV"]:.4f}低于旧版{old["receive_energy_TV"]:.4f}，43/48个位置改善；相对核插值也有39/48个位置改善。然而，均分参照{r["receive_energy_controls"][0]["receive_energy_TV"]:.4f}和训练平均占比参照{r["receive_energy_controls"][1]["receive_energy_TV"]:.4f}仍更低。所以只能说新版比这些旧输出更接近正确能量占比，不能说它已经超过简单统计规律，或已单独证实某个能量头的机制作用。',
      "",
      f'联合能量结构：新版误差{ours["receive_delay_joint_TV"]:.4f}高于旧版{old["receive_delay_joint_TV"]:.4f}。这正说明“各通道总能量占比更准”不等于“各通道里的回波能量也放对了时间”。不能只报告占比改善而遗漏联合分布退步。',
      "",
      f'几何对照：含几何相对去几何，方向谱平均仅增加{ours["pas_cosine"]-none["pas_cosine"]:.4f}、相对PDP仅增加{ours["relative_pdp_cosine"]-none["relative_pdp_cosine"]:.4f}；能量占比误差反而增加{ours["receive_energy_TV"]-none["receive_energy_TV"]:.4f}，联合误差变化约{ours["receive_delay_joint_TV"]-none["receive_delay_joint_TV"]:.5f}。方向谱在28/48个位置提高，相对PDP在22/48个位置提高；两副阵列的相对PDP变化方向相反。这是微小、混合的开发差异，尚不能认定稳定额外贡献。也不应把旧口径结论扩大成“几何在所有指标上都没有正差”。',
      "",
      "![新口径比较](new_metric_comparison.png)",
      "",
      "图注：a、b为逐频点方向谱及相对PDP相似度，越高越好；c、d为接收能量占比与通道—时延联合总变差误差，越低越好。NN为最近训练点，RBF为原先选定的频率核插值，Old/New分别为旧功率融合版/新版默认检查点，No geom.为同配置去几何对照，MLP为原先选定的位置MLP，U和Avg为均分及训练平均占比参照，只用于c。48个验证位置，每位置2副阵列，来自1个4米空间块，单种子且反复用于开发。无置信区间或显著性宣称。",
      "",
      "其余已有版本完整列出如下。扩大修正范围、降低几何权重或提高频率监督权重的现有版本，并没有在两个新主指标上同时超过默认新版。这个观察不等于已经按新目标充分调参，因为本轮所有模型仍按旧目标训练、按旧规则选轮。",
      "",
      *appendix,
      "",
      "WRF-GS+64/128及原宽带功率MLP只有32个方向的全带功率。它们没有逐频点复信道，也不定义真实接收单元功率分配，因此新四项指标标为不适用。保留并复算其适用的宽带方向、波束结果如下；没有把宽带功率复制成1024个频点或附加任意相位。",
      "",
      *power,
      "",
      "相对参考与数据有效性",
      "",
      "严格沿用先前核验并固定的指标代码。预测和实测各自用自身信道确定一个公共时间参考，同一副阵列的32个通道共同变换。没有根据实测真值搜索预测平移、相位或幅度，没有逐通道各自对齐。当前48个验证位置的96份阵列记录都具有数值有效参考，相邻频点参考相关强度最小约0.9579；参考接收通道均非零，所有13份预测也未触发未定义参考或零预测失败。所有验证位置均保留，指标按位置及面板的固定层级汇总。",
      "",
      "原始绝对复信道参考仍不充分。此处的相对PDP形状比较不证明绝对时延准确，也不证明每条真实物理路径被恢复。共享接收增益与比赛逐路径端口分配的已知实现差别仍然存在，本轮没有修改它。",
      "",
      "来源和独立核验",
      "",
      "已有NPZ直接读取原来保存的验证预测。带行号的文件逐项核对行序；旧文件未保存行号时，依据旧开发合同，同时复算旧方向余弦和波束损失，核对与原成绩一致。最近训练点和固定13专家平均融合没有存下独立信道文件，因此按原固定公式、仅用训练信道与查询位置重生成，未更新参数。均分和训练平均占比只是能量分配参照，不具备复信道输出，也不参加其他三项评分。",
      "",
      "对新版、去几何、频率核插值、旧功率融合及位置MLP的前3个位置，另用显式DFT矩阵和逐通道运算独立复算四项指标；最大差异约4.4e-16。逐位置、逐阵列结果还能重构整体汇总。固定参考掩码对所有方法一致，没有删除对某个方法不利的样本。",
      "",
      f'本轮CPU计算约{r["seconds"]:.1f}秒，峰值内存约{r["peak_RSS_MiB"]:.0f}MiB，没有加载训练检查点、优化器更新或新RT。只有允许的训练信道与这48个验证位置的参考信道参与计算；南侧正式测试和北侧确认信道均未使用。原始数据、准备数据、旧模型、预测和成绩保持不变。',
      "",
      "当前结论应分开写：新版相对旧版改善了方向谱和边缘接收能量分配；相对时延与联合能量结构没有改善；相对强插值和简单统计参照尚未建立整体优势；几何有极小的正向主指标差异，但没有稳定机制证据。这是补充性的开发评价，不是按新口径重新训练后的潜力上限，也不是独立确认结果。",
      "",
      "复查入口：evaluation.json为完整数值、来源哈希和有效性计数；comparison.csv为总表；per_position_panel.csv为逐位置、逐面板分数；per_panel.csv为面板汇总；metric_contrasts.json为固定方法间差异；evaluation.log保留运行过程。指标本体仍使用code/measured/hall_metric_protocol_v1，未改动。"
    ]
    (REPORT/"reevaluation_zh.md").write_text("\n".join(text)+"\n")
    print("Report and figure generated from evaluation.json")
def plot(r,lookup,names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    assert "Arial" in font_manager.findfont("Arial",fallback_to_default=False)
    plt.rcParams.update({"font.family":"Arial","font.size":6,"axes.labelsize":7,"xtick.labelsize":5.5,"ytick.labelsize":5.5,
       "axes.linewidth":.5,"pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"})
    fig,axes=plt.subplots(2,2,figsize=(183/25.4,112/25.4))
    fig.subplots_adjust(left=.075,right=.98,top=.95,bottom=.16,wspace=.29,hspace=.42)
    labels=["NN","RBF","Old","New","No geom.","MLP"]
    ylabels=["Per-frequency PAS cosine","Relative PDP cosine","Receive-energy TV","Receive/delay joint TV"]
    for letter,ax,key,ylabel in zip("abcd",axes.flat,KEYS,ylabels):
        values=[lookup[n]["scores"][key] for n in names];ticks=list(labels)
        colors=["#BBBBBB","#BBBBBB","#BBBBBB","#0072B2","#D55E00","#BBBBBB"]
        if key=="receive_energy_TV":
            values+= [x["receive_energy_TV"] for x in r["receive_energy_controls"]];ticks+=["U","Avg"];colors+=["white","white"]
        bars=ax.bar(np.arange(len(values)),values,color=colors,width=.65,edgecolor="#555555",linewidth=.3)
        for j,(bar,value) in enumerate(zip(bars,values)):
            if j>=6:bar.set_hatch("//")
            ax.text(bar.get_x()+bar.get_width()/2,value+(.014 if key in KEYS[:2] else .008),
                f"{value:.3f}",ha="center",va="bottom",fontsize=5.2)
        ax.set_xticks(np.arange(len(values)),ticks,rotation=20)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0,1. if key in KEYS[:2] else (.34 if key=="receive_energy_TV" else .8))
        ax.spines["top"].set_visible(False);ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out",width=.5,length=2)
        ax.text(-.14,1.07,letter,transform=ax.transAxes,fontweight="bold",fontsize=8,va="top")
    for suffix in ["png","pdf","svg","eps"]:fig.savefig(REPORT/("new_metric_comparison."+suffix),dpi=400)
    plt.close(fig)
if __name__=="__main__":main()
