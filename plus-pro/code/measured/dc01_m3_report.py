"""Frozen M3 evaluation, independent FP64 verification and measured-law report."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from project_paths import PATHS
from dc01_report import points,bootstrap
from measured_runtime import aggregate


def normalized(x,axes):return x/np.maximum(x.sum(axis=axes,keepdims=True),1e-30)
def cosine(x,y,axes):return (x*y).sum(axis=axes)/np.maximum(np.sqrt((x*x).sum(axis=axes)*(y*y).sum(axis=axes)),1e-30)


def verify(root,foundation,truth,metadata,split,reference):
    started=time.time();ps=[];ds=[];beams=[];error=energy=0.;power_change=0.;hashes={};seen=[]
    for fold in range(5):
        ids=np.asarray(split['splits'][fold]['test']);seen.extend(ids.tolist())
        path=root/f'fold{fold}'/('test_base.npy' if root==foundation else 'test_prediction.npy')
        pred=np.load(path,mmap_mode='r');expected_power=np.load(foundation/f'fold{fold}/test_log_power.npy')
        assert pred.shape==(len(ids),2,8,4,256) and pred.dtype==np.complex64 and np.isfinite(pred).all()
        if root!=foundation:
            r=json.loads((root/f'fold{fold}/angular_report.json').read_text())
            if r['selected_epoch']==0:assert np.array_equal(pred,np.load(foundation/f'fold{fold}/test_base.npy',mmap_mode='r'))
        hashes[f'fold{fold}']=hashlib.sha256(path.read_bytes()).hexdigest()
        for first in range(0,len(ids),24):
            h=np.asarray(pred[first:first+24],dtype=np.complex128);t=np.asarray(truth[ids[first:first+24]],dtype=np.complex128)
            a=np.abs(np.fft.fft2(h,axes=(2,3)))**2;b=np.abs(np.fft.fft2(t,axes=(2,3)))**2
            ps.append(cosine(normalized(a,(2,3)),normalized(b,(2,3)),(2,3)).mean(-1))
            d=normalized(np.abs(np.fft.ifft(h,axis=-1))**2,(-1,));td=normalized(np.abs(np.fft.ifft(t,axis=-1))**2,(-1,))
            ds.append(cosine(d,td,(-1,)).mean((2,3)))
            chosen=a.mean(-1).reshape(len(h),2,32).argmax(-1);actual=b.mean(-1).reshape(len(h),2,32)
            selected=np.take_along_axis(actual,chosen[...,None],axis=-1)[...,0]
            beams.append(-10*np.log10(np.maximum(selected/actual.max(-1),1e-30)))
            error+=float(np.sum(np.abs(h-t)**2));energy+=float(np.sum(np.abs(t)**2))
            panel_energy=np.sum(np.abs(h)**2,axis=(2,3,4));target=10**(expected_power[first:first+24]/10)
            power_change=max(power_change,float(np.max(np.abs(panel_energy-target)/target)))
    assert np.array_equal(np.sort(seen),np.arange(len(truth)))
    computed=dict(pas=float(np.concatenate(ps).mean()),pdp=float(np.concatenate(ds).mean()),
                  beam_loss_db=float(np.concatenate(beams).mean()),nmse=error/energy)
    computed['structural_score']=.5*(computed['pas']+computed['pdp'])
    differences={k:computed[k]-reference[k] for k in computed}
    assert max(abs(x) for x in differences.values())<3e-6,differences
    assert power_change<1e-5,power_change
    return dict(status='passed',backend='NumPy complex128/FP64',metrics=computed,
                differences=differences,maximum_energy_relative_error=power_change,sha256=hashes,elapsed_s=time.time()-started)


def main():
    started=time.time();art=PATHS.artifacts/'measured';out=PATHS.analysis/'results/measured_dc01_m3'
    prepared=art/'dc01_v1';foundation=art/'dc01_m3_foundation_v1';actual=art/'dc01_m3_angular_v1';control=art/'dc01_m3_angular_shuffle_v1'
    split=json.loads((foundation/'split_manifest.json').read_text());metadata=np.load(prepared/'selection.npz')
    truth=np.load(prepared/'channels.npy',mmap_mode='r');collections={};metrics={};source=None
    files=[('nearest',art/'dc01_baselines_v1','nearest_metrics.npz'),('M2_ported',art/'dc01_state_v1','test_metrics.npz'),
           ('factorized_data',foundation,'test_metrics.npz'),('M3_geometry',actual,'test_metrics.npz'),
           ('shuffled_geometry',control,'test_metrics.npz'),('conditional_mean_diagnostic',actual,'conditional_mean_metrics.npz')]
    for name,root,file in files:
        ids,value=points(root,file)
        if source is None:source=ids
        else:assert np.array_equal(source,ids)
        collections[name]=value;metrics[name]=aggregate(value)
    row={int(k):i for i,k in enumerate(metadata['source_id'])};order=[row[int(k)] for k in source]
    position=metadata['position'][order];_,blocks=np.unique(np.floor(position[:,:2]/4),axis=0,return_inverse=True)
    structure=lambda x:.5*(x['pas'].mean(1)+x['pdp'].mean(1))
    gain=bootstrap(structure(collections['M3_geometry'])-structure(collections['factorized_data']),blocks)
    shuffle=bootstrap(structure(collections['M3_geometry'])-structure(collections['shuffled_geometry']),blocks)
    beam=bootstrap(collections['factorized_data']['beam_loss_db'].mean(1)-collections['M3_geometry']['beam_loss_db'].mean(1),blocks)
    power_gain=bootstrap(np.abs(collections['M2_ported']['power_error_db']).mean(1)-np.abs(collections['M3_geometry']['power_error_db']).mean(1),blocks)
    power_vs_nearest=bootstrap(np.abs(collections['nearest']['power_error_db']).mean(1)-np.abs(collections['M3_geometry']['power_error_db']).mean(1),blocks)
    versus_m2=bootstrap(structure(collections['M3_geometry'])-structure(collections['M2_ported']),blocks)
    power_success=metrics['M3_geometry']['power_mae_db']<metrics['nearest']['power_mae_db'] and metrics['M3_geometry']['power_mae_db']<=.5*metrics['M2_ported']['power_mae_db']
    geometry_success=gain['ci95'][0]>0 and beam['mean']>=0
    independent={name:verify(root,foundation,truth,metadata,split,metrics[name]) for name,root in [('factorized_data',foundation),('M3_geometry',actual),('shuffled_geometry',control)]}
    folds=[]
    for fold in range(5):
        b=json.loads((foundation/f'fold{fold}/foundation_report.json').read_text())
        a=json.loads((actual/f'fold{fold}/angular_report.json').read_text());s=json.loads((control/f'fold{fold}/angular_report.json').read_text())
        folds.append(dict(fold=fold,power_model=b['selected_power'],spectral_model=b['selected_spectral'],actual_epoch=a['selected_epoch'],control_epoch=s['selected_epoch'],
                          geometry_structural=a['test']['structural_score'],base_structural=b['test']['structural_score'],shuffle_structural=s['test']['structural_score'],
                          gate_mean=a['gate_mean'],conditional_coefficient=a['conditional_coefficient'],
                          foundation_seconds=b['elapsed_s'],actual_seconds=a['elapsed_s'],control_seconds=s['elapsed_s'],
                          actual_gpu=a['gpu'],control_gpu=s['gpu'],actual_peak_gpu_GiB=a['peak_gpu_GiB'],control_peak_gpu_GiB=s['peak_gpu_GiB']))
    result=dict(status='complete',interpretation='Development comparison on previously observed dc01; not untouched acquisition confirmation',
                metrics=metrics,geometry_gain=gain,geometry_vs_shuffle=shuffle,beam_loss_reduction_db=beam,power_mae_reduction_db=power_gain,
                power_criterion=bool(power_success),geometry_criterion=bool(geometry_success),shuffle_criterion=bool(shuffle['ci95'][0]>0),
                power_vs_nearest=power_vs_nearest,structure_vs_m2=versus_m2,
                folds=folds,independent_verification=independent,elapsed_s=time.time()-started,
                limitations=['Same observed dc01 development benchmark','Near-position recurrences do not isolate measurement-reference causes',
                             'Failure of tested radial phase law does not prove all phase predictors impossible',
                             'Energy-normalized reconstruction and conditional mean are different outputs',
                             'Conditional-mean shrinkage is not simultaneous accurate power and absolute phase',
                             'One model seed per fold; bootstrap is conditional on these fitted models'])
    (out/'evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    import csv
    with (out/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['method',*metrics['nearest']]);writer.writeheader()
        for name,m in metrics.items():writer.writerow(dict(method=name,**m))
    np.savez(out/'paired_results.npz',source_id=source,position=position,spatial_block=blocks,
             geometry_gain=structure(collections['M3_geometry'])-structure(collections['factorized_data']),
             shuffle_gain=structure(collections['M3_geometry'])-structure(collections['shuffled_geometry']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['nearest','M2_ported','factorized_data','M3_geometry','shuffled_geometry']
    labels=['Nearest','M2 ported','Measured field','M3 geometry','Shuffled geom.']
    fig,axes=plt.subplots(1,3,figsize=(13,4));x=np.arange(len(names))
    axes[0].bar(x,[metrics[k]['power_mae_db'] for k in names]);axes[0].set_ylabel('Power MAE (dB)');axes[0].set_title('Independent measured-energy field')
    axes[1].bar(x-.18,[metrics[k]['pas'] for k in names],.36,label='PAS');axes[1].bar(x+.18,[metrics[k]['pdp'] for k in names],.36,label='PDP');axes[1].set_ylim(0,1);axes[1].legend();axes[1].set_title('Normalized spectral structure')
    for name,label in zip(names,labels):
        y=np.sort(collections[name]['beam_loss_db'].ravel());axes[2].plot(y,np.arange(1,len(y)+1)/len(y),label=label)
    axes[2].set_xlabel('DFT beam loss (dB)');axes[2].set_ylabel('Empirical CDF');axes[2].set_xlim(0,10);axes[2].legend(fontsize=8)
    for ax in axes[:2]:ax.set_xticks(x,labels,rotation=25,ha='right')
    fig.tight_layout();fig.savefig(out/'performance.png',dpi=180);plt.close(fig)
    audit=json.loads((PATHS.analysis/'results/measured_dc01_m3_audit_v2/audit.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(13,4));repeat=audit['recurrence']['metrics'];x=np.arange(4)
    for panel in range(2):axes[0].bar(x+(panel-.5)*.34,[repeat[k][panel] for k in ['correlation','phase_invariant','pas','pdp']],.34,label=f'Panel {panel}')
    axes[0].set_xticks(x,['Real complex','Phase invariant','PAS','PDP'],rotation=20,ha='right');axes[0].set_ylim(0,1);axes[0].legend();axes[0].set_title('Near-position repeats, separated in time')
    x=np.arange(5)
    axes[1].bar(x-.17,[f['law']['power']['global_median_mae_db'] for f in audit['folds']],.34,label='Global panel median')
    axes[1].bar(x+.17,[f['law']['power']['log_idw8_mae_db'] for f in audit['folds']],.34,label='Spatial log-power field')
    axes[1].set_xticks(x);axes[1].set_xlabel('Internal fit-only split');axes[1].set_ylabel('Power MAE (dB)');axes[1].legend(fontsize=8);axes[1].set_title('Power is spatially predictable')
    axes[1].set_ylim(0,6.5)
    for key,offset,label in [('pas_gain',-.12,'PAS'),('pdp_gain',.12,'PDP')]:
        means=np.array([f['geometry'][key]['mean'] for f in audit['folds']]);ci=np.array([f['geometry'][key]['ci95'] for f in audit['folds']])
        axes[2].errorbar(means,x+offset,xerr=np.maximum(np.stack([means-ci[:,0],ci[:,1]-means]),0),fmt='o',capsize=3,label=label)
    axes[2].axvline(0,color='0.5',linestyle='--');axes[2].set_yticks(x);axes[2].set_ylabel('Fold');axes[2].set_xlabel('Registered minus shuffled cosine');axes[2].legend();axes[2].set_title('Geometry information audit, 95% CI')
    axes[2].set_ylim(-.5,5.5);axes[2].legend(loc='upper right')
    fig.tight_layout();fig.savefig(out/'measurement_laws.png',dpi=180);plt.close(fig)
    rows=[]
    for name,label in zip(names,labels):
        m=metrics[name];rows.append(f"| {label} | {m['power_mae_db']:.4f} | {m['pas']:.6f} | {m['pdp']:.6f} | {m['structural_score']:.6f} | {m['beam_loss_db']:.4f} | {m['nmse']:.6f} |")
    f=metrics['M3_geometry'];cm=metrics['conditional_mean_diagnostic']
    text=f"""# 从测量规律重建算法：dc01 M3

本轮从原方法的信息流出发，先辨识观测规律，再建立能量与结构分解，并只在数据支持的角谱方向学习几何修正。五折结果与独立双精度验算均已完成。这是已观察过的dc01上的开发比较，不能解释为独立采集批次的盲测确认。

训练分区内部审计显示：512对相距不超过2cm、时间间隔超过10秒的观测，两个面板的平均功率差为0.480/0.525dB，PAS相似度0.907/0.843，PDP相似度0.933/0.902；复相关实部仅0.025/0.052，而允许公共相位对齐后的相关幅度达到0.805/0.698。它支持分别处理能量、谱结构和相位参考，但不能单独归因于某一硬件因素。所测试径向相位律仅1/10个面板/折通过内部检验，没有建立稳定的位置到公共相位规律。

几何候选对角谱有明确位置相关信息：相对打乱候选，五折训练内部角谱余弦增益为0.084..0.114，置信区间均为正；时延谱增益的区间均跨零。因此本轮保留数据驱动时延谱，几何网络只修正角度及相对路径能量，没有强行写入几何时延。

新底座独立预测每面板log功率，并插值平方根归一化PAS、PDP及联合角度—时延功率。验证集在固定候选中选择能量场和结构重建方式。实测邻居的变换域相位是明确声明的参考，没有把相干平均后的能量当作接收功率。路径网络沿用原方法的有序交互序列和候选集合竞争思想，去除绝对RT相位特征，学习有界角度/质量修正及不超过0.5的可靠性门。训练目标直接评价投影后的角谱和协方差，恢复预测能量并保持数据驱动PDP。打乱几何对照保持真实查询坐标、模型规模、训练轮数和随机种子。

| 方法 | 功率MAE dB↓ | PAS↑ | PDP↑ | 结构分↑ | 波束损失dB↓ | 原始NMSE↓ |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

本轮并未在全部指标上优于M2。功率MAE较M2降低{100*(1-f['power_mae_db']/metrics['M2_ported']['power_mae_db']):.2f}%，结构分变化为{versus_m2['mean']:+.6f}，区间[{versus_m2['ci95'][0]:+.6f},{versus_m2['ci95'][1]:+.6f}]；原始NMSE和平均波束损失更差。相对最近邻的功率MAE改善均值为{power_vs_nearest['mean']:+.4f}dB，区间[{power_vs_nearest['ci95'][0]:+.4f},{power_vs_nearest['ci95'][1]:+.4f}]，均值达标与统计显著性应区分。

功率验收（优于最近邻且较M2至少下降50%）：**{power_success}**。几何验收（相对新底座结构增益置信下界大于0、平均波束损失不增加）：**{geometry_success}**。这两个条件分别记录，不隐藏未通过项。

几何相对新底座的结构分增益为{gain['mean']:+.6f}，4m空间块bootstrap95%区间[{gain['ci95'][0]:+.6f},{gain['ci95'][1]:+.6f}]；波束损失减少{beam['mean']:+.4f}dB，区间[{beam['ci95'][0]:+.4f},{beam['ci95'][1]:+.4f}]。实际几何相对打乱几何的结构分变化为{shuffle['mean']:+.6f}，区间[{shuffle['ci95'][0]:+.6f},{shuffle['ci95'][1]:+.6f}]。完整逐折选择、控制结果和所有指标见evaluation.json及comparison.csv。

绝对相位问题仍需单独看待。能量归一化后的最终重建原始NMSE为{f['nmse']:.6f}，公共相位对齐NMSE为{f['phase_aligned_nmse']:.6f}，每面板相位/时延对齐NMSE为{f['panel_phase_delay_aligned_nmse']:.6f}。对齐使用测试真值，只是诊断。若参考相位难以由输入预测，恢复正确能量并不保证原始复数NMSE变小：两者对应不同要求。本轮另用拟合分区学习每面板复数条件均值系数，得到诊断NMSE {cm['nmse']:.6f}，但其功率MAE为{cm['power_mae_db']:.4f}dB；收缩输出不能解释为同时解决了功率和绝对相位。本轮主输出始终是具有独立预测能量的重建H。

公共相位对齐后误差仍较高，因此不能把剩余误差全部归因于公共相位参考；阵列与多径的相对复结构也未充分恢复。真实几何对打乱几何的增益区间跨零，当前对照未证实复杂路径网络有额外机制价值。它不否定原始几何角谱包含信息，而是说明这些信息进入当前底座后，新增收益尚弱。当前证据支持能量场的独立建模，尚不支持“完整复信道问题已经解决”。

三个新增输出阶段均通过NumPy complex128/FP64独立复算，包括形状、finite、全部1023位置恰好覆盖一次、零动作回退身份和预测功率守恒。所有原始数据、M2预测和最终竞赛代码保留。原始数据来源为[斯图加特大学DICHASUS dcxx](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-3831)；唯一实验记录为experiment_records/EXPERIMENT_RECORD.md的M3A/M3B。单场景、单次模型种子和开发集复用限制仍然存在。

![能量、谱结构与波束比较](performance.png)

![决定模型设计的拟合分区测量规律](measurement_laws.png)
"""
    (out/'evaluation_report_zh.md').write_text(text,encoding='utf-8')
    print(json.dumps(dict(status='complete',power_criterion=power_success,geometry_criterion=geometry_success,geometry_gain=gain,shuffle_gain=shuffle,metrics=metrics)),flush=True)


if __name__=='__main__':main()
