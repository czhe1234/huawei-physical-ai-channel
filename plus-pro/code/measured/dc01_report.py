"""Auditable paired evaluation and scientific plots; no model selection."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from measured_runtime import aggregate


def points(root,name):
    collections=[]
    ids=[]
    for fold in range(5):
        with np.load(root/f'fold{fold}'/name,allow_pickle=False) as z:
            ids.append(z['source_id'])
            collections.append({k:z[k] for k in z.files if k!='source_id'})
    values={k:np.concatenate([d[k] for d in collections]) for k in collections[0]}
    source_id=np.concatenate(ids)
    assert len(np.unique(source_id))==len(source_id)
    order=np.argsort(source_id)
    return source_id[order],{k:v[order] for k,v in values.items()}


def bootstrap(delta,group,seed=20261015,draws=2000):
    keys=np.unique(group)
    counts=np.array([np.sum(group==k) for k in keys])
    sums=np.array([delta[group==k].sum() for k in keys])
    selected=np.random.default_rng(seed).integers(len(keys),size=(draws,len(keys)))
    estimates=sums[selected].sum(1)/counts[selected].sum(1)
    return dict(mean=float(delta.mean()),ci95=np.quantile(estimates,[.025,.975]).tolist(),
                blocks=len(keys),draws=draws,unit='4m spatial tile; both panels bound to position')


def data_plot(prepared,data,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import trimesh
    metadata=np.load(prepared/'selection.npz')
    position=metadata['position']
    tiles=np.floor(position[:,:2]/4).astype(int)
    color=(tiles[:,0]+2*tiles[:,1])%5
    fig,ax=plt.subplots(figsize=(7,8))
    mesh=trimesh.load(data/'context/scenes/inue_simple/meshes/walls.ply',process=False)
    vertices=np.asarray(mesh.vertices);faces=np.asarray(mesh.faces)
    segments=[]
    for face in faces:
        for i in range(3):segments.append(vertices[face[[i,(i+1)%3]],:2])
    ax.add_collection(LineCollection(segments,colors='0.5',linewidths=1,alpha=.35))
    cloud=ax.scatter(position[:,0],position[:,1],c=color,s=7,cmap='tab10',vmin=0,vmax=9)
    panel=metadata['panel_position'];direction=metadata['panel_direction']
    ax.scatter(panel[:,0],panel[:,1],marker='^',c='black',s=70,label='Physical RX panels')
    ax.quiver(panel[:,0],panel[:,1],direction[:,0],direction[:,1],angles='xy',scale_units='xy',scale=.4,color='black')
    ax.set_aspect('equal');ax.autoscale();ax.set_xlabel('Scene x (m)');ax.set_ylabel('Scene y (m)')
    ax.set_title('dc01 registered scene and spatial split')
    ax.legend(loc='upper right');fig.tight_layout();fig.savefig(out/'data_geometry.png',dpi=180);plt.close(fig)


def chinese_report(result,out):
    metrics=result['metrics'];base=metrics['selected_foundation'];final=metrics['final_measured_v16']
    ci=result['structural_gain'];beam=result['beam_loss_reduction_db']
    verdict='通过预注册主指标条件' if result['hypothesis_supported'] else '未通过预注册主指标条件'
    rows=[]
    for key,label in [('nearest','最近邻'),('idw16','IDW-16'),('selected_foundation','验证选定的最强底座'),
                      ('spectral_link_foundation','底座 + 谱CNN + LinkGate'),('final_measured_v16','最终实测适配方法')]:
        m=metrics[key]
        rows.append(f"| {label} | {m['pas']:.6f} | {m['pdp']:.6f} | {m['structural_score']:.6f} | {m['beam_loss_db']:.4f} | {m['nmse']:.6f} |")
    folds=[f"| {f['fold']} | {f['fit']} | {f['validation']} | {f['test']} | {f['baseline_structural']:.6f} | {f['final_structural']:.6f} | {f['total_gain']:+.6f} | {f['state_epoch']} |"
           for f in result['folds']]
    if final['nmse']>=1:
        complex_conclusion='原始NMSE仍不低于1；在本指标下，全零预测的NMSE为1。因此目前不能声称已经实现高精度的绝对复信道重建。'
    else:
        complex_conclusion='原始NMSE低于全零预测的1，但这不消除移动TX公共相位和时延可辨识性的限制。'
    text=f"""# DICHASUS dc01 实测评估

已完成五组空间隔离评估，覆盖全部1,023个代表位置及其两个真实接收面板。本实验{verdict}。相对验证集选定底座，结构分变化为 **{ci['mean']:+.6f}**，空间块bootstrap 95%置信区间为 **[{ci['ci95'][0]:+.6f}, {ci['ci95'][1]:+.6f}]**；平均DFT波束损失减少 **{beam['mean']:+.4f} dB**，对应95%区间 **[{beam['ci95'][0]:+.4f}, {beam['ci95'][1]:+.4f}] dB**。置信区间以{ci['blocks']}个4m空间块为重采样单位，重复{ci['draws']}次。

数据选自[斯图加特大学DICHASUS dcxx发布页](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-3831)。它同时提供真实复CSI、位置标签、时间戳及已知场景的分布式阵列观测，适合检验本项目的地图和路径谱校准。注册场景与CPO/STO辅助文件来自[NVlabs的配套研究代码](https://github.com/NVlabs/diff-rt-calibration)，固定commit为`cc70d813a5ac8cd976c0b43ca811712dd5a1629a`。本实验不使用该仓库训练好的网络。

完整原始文件16,870,531,926 bytes、32,133帧，先下载到本地，再上传服务器，两端MD5/SHA256一致。原始MD5为`7934b83963491b493a37556484d6ca42`；SHA256为`631fa66bb82405f28926b4e6524ab02ce10e0b1b35232b153054f18d9c72b16d`。全部原始CSI通过finite检查。只依据0.20m位置网格选择1,023个代表点，避免将同一区域的重复帧当作独立测试样本。

原始观测有64个接收天线、1,024个子载波。工作视图按两个真实4×8面板分别组织，并每四个原始子载波取一个，形成`[1023,2,8,4,256] complex64`。载频3.438GHz，原始频带跨度50MHz；子载波抽取减少了无模糊时延范围。原始文件保留。阵列内按发布的CPO/STO校准，不能据此认定移动TX跨位置的绝对相位已经可辨识。

场景按4m XY方格的五种颜色划分。每折使用一种颜色测试、下一种验证、其余三种拟合；移除距任一留出点不足0.30m的拟合位置。同一位置的两个面板绑定划分。模型选择只读取拟合/验证标签，测试标签只用于冻结后的评价。19种基础方法及全部指标见`comparison.csv`。五折验证均选择平方根谱RBF作为底座。

| 方法 | PAS↑ | PDP↑ | 两者均值↑ | 波束损失dB↓ | 原始NMSE↓ |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

按冻结阶段比较，谱CNN与LinkGate相对底座贡献结构分{metrics['spectral_link_foundation']['structural_score']-base['structural_score']:+.6f}，随后ComponentState再贡献{final['structural_score']-metrics['spectral_link_foundation']['structural_score']:+.6f}。但加入ComponentState后，波束损失比前一级增加{final['beam_loss_db']-metrics['spectral_link_foundation']['beam_loss_db']:+.4f}dB。这说明结构分与波束选择存在取舍，不能把每个模块都描述为对所有指标有益。

PAS是逐频率面板二维角谱的余弦；PDP是每条真实天线链路的时延谱余弦。结构分是两者等权均值。DFT波束由预测选择，再用实测H计算相对于最优DFT波束的功率损失。NMSE使用全部留出观测的误差总能量除以真值总能量，不是逐点比例平均。附带的竞赛形式分数不与原榜单比较。

| 折 | 拟合位置 | 验证位置 | 测试位置 | 底座结构分 | 最终结构分 | 增量 | State选定epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(folds)}

本次迁移保留最终V16的17种复载体专家思路、四个原始谱CNN主体、原始LinkGate和96维序列/集合ComponentState编码器；按实际阵列、频率数和单TX调整尺寸，并重新训练。物理候选固定Base深度16、Scatter深度10/散射系数0.15、每源一百万采样和24个种子，跨种子去重后最多保留128条路径。CNN和gate固定16轮，State固定48轮，内部验证决定checkpoint及是否保留模块；无收益可以退回前级。竞赛专用donor/PCA/lifecycle数值资产没有复制到实测域，所以这是最终框架的实测适配评估，不是竞赛权重直接推理的等价复现。

{complex_conclusion}最终公共相位对齐NMSE为{final['phase_aligned_nmse']:.6f}，每面板相位对齐为{final['panel_phase_aligned_nmse']:.6f}，每面板公共相位/时延对齐为{final['panel_phase_delay_aligned_nmse']:.6f}。这些对齐都使用测试真值，只用于诊断，不能计为可部署补偿。最终功率MAE为{final['power_mae_db']:.4f}dB，归一化阵列协方差误差为{final['covariance_error']:.6f}。各谱投影保持底座面板总功率，本轮没有另行训练功率校准器；结构谱改善并不等于功率或绝对相位重建问题已经解决。

结论仅限于dc01同一场景、同一增益配置中的未测位置泛化。一个移动TX端口不能验证原多UE/极化端口机制；本轮也没有跨场景、跨硬件或跨采集批次测试。空间块置信区间只反映本次固定模型与该布局下的空间变异，不代表多次重训的不确定性。

谱CNN/LinkGate五折累计训练与评价墙钟为{sum(f['spectral_seconds'] for f in result['folds']):.1f}s，State为{sum(f['state_seconds'] for f in result['folds']):.1f}s。这些是各折耗时之和，部分折并行运行，不应相加解释为实际等待时间。逐折GPU/显存、环境、预处理耗时与退出码保存在对应JSON及日志中。复现入口见`code/measured/README.md`，唯一实验记录为`experiment_records/EXPERIMENT_RECORD.md`的Measured-M1/M2。

机器结果：`evaluation.json`；完整比较：`comparison.csv`；配对逐位置结果：`paired_point_results.npz`。独立NumPy双精度复核结果另存为`independent_verification.json`。

![实测性能比较](performance.png)

![注册场景与空间划分](data_geometry.png)
"""
    (out/'evaluation_report_zh.md').write_text(text,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepared-dir',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--baseline-dir',type=Path)
    parser.add_argument('--spectral-dir',type=Path)
    parser.add_argument('--state-dir',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--data-only',action='store_true')
    args=parser.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    data_plot(args.prepared_dir,args.data_dir,out)
    if args.data_only:return
    manifest=json.loads((args.baseline_dir/'split_manifest.json').read_text())
    audit=json.loads((args.prepared_dir/'data_audit.json').read_text())
    for fold in range(5):
        state=json.loads((args.state_dir/f'fold{fold}'/'state_report.json').read_text())
        spectral=json.loads((args.spectral_dir/f'fold{fold}'/'baseline_report.json').read_text())
        assert state['split_sha256']==manifest['sha256']
        assert spectral['identity']['split']==manifest['sha256']
    comparison={};arrays={};source=None
    names=['nearest','idw4','idw8','idw16','idw32','radial4','radial8','radial16','radial32',
           'affine8','affine16','affine32','affine64','array_affine8','array_affine16',
           'array_affine32','array_affine64','carrier_ridge','sqrt_rbf']
    sources=[(n,args.baseline_dir,f'{n}_metrics.npz') for n in names]
    sources += [('selected_foundation',args.baseline_dir,'frozen_base_metrics.npz'),
                ('spectral_link_foundation',args.spectral_dir,'frozen_base_metrics.npz'),
                ('final_measured_v16',args.state_dir,'test_metrics.npz'),
                ('uncalibrated_paths_on_enhanced_foundation',args.state_dir,'physical_prior_metrics.npz')]
    for name,root,file in sources:
        ids,values=points(root,file)
        if source is None:source=ids
        else:assert np.array_equal(ids,source)
        arrays[name]=values;comparison[name]=aggregate(values)
    metadata=np.load(args.prepared_dir/'selection.npz')
    row={int(k):i for i,k in enumerate(metadata['source_id'])}
    position=metadata['position'][[row[int(k)] for k in source]]
    tile=np.floor(position[:,:2]/4).astype(np.int64)
    _,group=np.unique(tile,axis=0,return_inverse=True)
    base=arrays['selected_foundation'];final=arrays['final_measured_v16']
    structural=lambda p:0.5*(p['pas'].mean(1)+p['pdp'].mean(1))
    delta=structural(final)-structural(base)
    ci=bootstrap(delta,group)
    beam_gain=base['beam_loss_db'].mean(1)-final['beam_loss_db'].mean(1)
    beam_ci=bootstrap(beam_gain,group,seed=20261016)
    success=ci['ci95'][0]>0 and beam_gain.mean()>=0
    folds=[]
    for fold in range(5):
        s=json.loads((args.state_dir/f'fold{fold}'/'state_report.json').read_text())
        b=json.loads((args.baseline_dir/f'fold{fold}'/'baseline_report.json').read_text())
        c=json.loads((args.spectral_dir/f'fold{fold}'/'baseline_report.json').read_text())
        folds.append(dict(fold=fold,fit=b['fit_count'],validation=b['val_count'],test=b['test_count'],
                          baseline=b['selected_on_validation'],state_epoch=s['selected_epoch'],
                          spectral_accepted=c['spectral_accepted'],gate_accepted=c['gate_accepted'],
                          baseline_structural=b['frozen_base_test']['structural_score'],
                          final_structural=s['test']['structural_score'],
                          total_gain=s['test']['structural_score']-b['frozen_base_test']['structural_score'],
                          state_seconds=s['elapsed_s'],spectral_seconds=c['elapsed_s']))
    result=dict(status='complete',dataset=audit,split_sha256=manifest['sha256'],
                metrics=comparison,structural_gain=ci,beam_loss_reduction_db=beam_ci,
                hypothesis_supported=bool(success),folds=folds,
                limitations=['One measured scene and one gain/configuration',
                             '1023 position representatives from 32133 repeated snapshots',
                             'Single TX port cannot validate the original multi-UE/polarization action',
                             'Published RX calibration does not prove absolute mobile-TX coherence',
                             'Domain-adapted architecture retrained from scratch, not unchanged competition weights',
                             'Bootstrap is conditional on these fitted models and this spatial layout',
                             'Frequency view uses 256 regularly subsampled original tones',
                             'Phase/time alignment is ground-truth-assisted evaluation only'])
    (out/'evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    columns=['method','positions','pas','pdp','relative_pdp','structural_score','beam_loss_db',
             'covariance_error','power_mae_db','nmse','phase_aligned_nmse',
             'panel_phase_aligned_nmse','panel_phase_delay_aligned_nmse','adapted_competition_score']
    with (out/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for name,metrics in comparison.items():writer.writerow({'method':name,**metrics})
    np.savez(out/'paired_point_results.npz',source_id=source,position=position,
             structural_gain=delta,beam_gain_db=beam_gain,spatial_block=group)
    import matplotlib.pyplot as plt
    keep=['nearest','idw16','sqrt_rbf','spectral_link_foundation','final_measured_v16']
    labels=['Nearest','IDW-16','sqrt-RBF','CNN + LinkGate','V16 measured']
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    x=np.arange(len(keep));width=.35
    axes[0].bar(x-width/2,[comparison[k]['pas'] for k in keep],width,label='PAS')
    axes[0].bar(x+width/2,[comparison[k]['pdp'] for k in keep],width,label='PDP')
    axes[0].set_xticks(x,labels,rotation=25,ha='right');axes[0].set_ylim(0,1);axes[0].legend();axes[0].set_title('Spatial holdout spectral cosine')
    for k,label in zip(keep,labels):
        v=np.sort(arrays[k]['beam_loss_db'].ravel());axes[1].plot(v,np.arange(1,len(v)+1)/len(v),label=label)
    axes[1].set_xlabel('DFT beam loss (dB)');axes[1].set_ylabel('Empirical CDF');axes[1].set_xlim(0,10);axes[1].legend(fontsize=8)
    map=axes[2].scatter(position[:,0],position[:,1],c=delta,cmap='coolwarm',vmin=-.1,vmax=.1,s=8)
    axes[2].set_aspect('equal');axes[2].set_title('Paired structural-score gain');axes[2].set_xlabel('x (m)');axes[2].set_ylabel('y (m)');fig.colorbar(map,ax=axes[2])
    axes[2].set_title('Paired structural-score\ngain',fontsize=10)
    axes[2].set_xlim(-6,6);axes[2].set_xticks([-4,0,4]);axes[2].tick_params(axis='x',labelsize=8)
    fig.tight_layout();fig.savefig(out/'performance.png',dpi=180);plt.close(fig)
    chinese_report(result,out)
    print(json.dumps(dict(status='complete',primary=ci,beam=beam_ci,hypothesis_supported=bool(success),
                          baseline=comparison['selected_foundation'],final=comparison['final_measured_v16'])),flush=True)


if __name__=='__main__':main()
