"""Build reviewable, non-destructive cleanup options from an inventory snapshot."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import PATHS


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def build(output):
    load=lambda name:json.loads((output/name).read_text(encoding='utf-8'))
    summary,metadata,duplicates,archives=map(load,('summary.json','metadata.json','duplicates.json','archives.json'))
    files=[r for r in metadata if r['kind']=='file']
    indexed={r['path']:r for r in files}
    directory_rows=list(csv.DictReader((output/'directories.csv').open(encoding='utf-8-sig')))
    directories={r['path']:{k:int(v) if k!='path' else v for k,v in r.items()} for r in directory_rows}
    options=[]
    selected_paths={}

    def matches(path, prefix):
        return path==prefix or path.startswith(prefix+'/')

    def add(identifier,tier,targets,title,impact,evidence,keep=None,exclusions=None):
        exclusions=exclusions or []
        members=[r for r in files if any(matches(r['path'],p) for p in targets) and not any(matches(r['path'],p) for p in exclusions)]
        for row in members:
            if row['path'] in selected_paths:
                raise ValueError('Overlapping cleanup options: '+row['path'])
            selected_paths[row['path']]=identifier
        inode_groups={}
        for row in members:
            inode_groups.setdefault((row['device'],row['inode']),[]).append(row)
        reclaim=sum(group[0]['allocated_bytes'] for group in inode_groups.values() if len(group)>=group[0]['nlink'])
        options.append({'id':identifier,'tier':tier,'title':title,'paths':targets,'exclude_paths':exclusions,
                        'file_count':len(members),'logical_file_GB':sum(r['size_bytes'] for r in members)/1e9,
                        'reclaimable_file_GB':reclaim/1e9,'reclaimable_file_bytes':reclaim,
                        'impact':impact,'evidence':evidence,'must_retain':keep or [],'executed':False})

    add('A01','优先候选',['plus-pro/datasets/dichasus/dc01/upload_parts'],'dc01 已完成传输的 32 个分片',
        '只失去现成的分片续传材料；完整原始测量文件、标定和实验结果保留。',
        '32 个文件名/大小/连续偏移已对齐上传清单，总字节数与完整原始文件一致；已保存整文件上传校验。此次没有重新对整文件计算 SHA256。',
        ['plus-pro/datasets/dichasus/dc01/dichasus-dc01.tfrecords','plus-pro/datasets/dichasus/dc01/upload_verified.json'])
    add('A02','可再生成历史输出',['artifacts/temporary'],'18 组早期参数扫描预测',
        '失去 tmp_v26 至 tmp_v43 各参数组合的完整预测数组；若再比较需重新生成。当前实测实验产物位于独立的 plus-pro。',
        '目录含 18 个历史预测文件；不得因文件大小相同而判为重复内容。')
    source_copies=['versions/round2/Round2_Final_Source_Submission.zip','versions/round2/V16_source_online_0.686754.zip']
    add('A03','SHA256 确认的副本',source_copies,'两份跨目录重复的 Round2 源码归档',
        '保留 final_code 下各自对应归档时，不损失这两份归档的字节内容。两个不同名称的归档彼此并不相同。',
        '每份归档都与 final_code 下同名归档通过完整 SHA256 一致性检查；是两组重复对，而非四个包全部相同。',
        ['final_code/Round2_Final_Source_Submission.zip','final_code/V16_source_online_0.686754.zip'])
    visual_copy='analysis/visualizations/sionna_path_candidates/v395_six_seed_high_order_pas_projection_action075_complement050_candidate_pending.zip'
    add('A04','SHA256 确认的副本',[visual_copy],'可视化目录中的 v395 提交包副本',
        '保留 versions/round1 中对应包时，不损失该归档内容。若同时选择删除初赛全部历史包，意味着主动放弃此版本。',
        '完整 SHA256 与 versions/round1 下同名包一致。',
        ['versions/round1/'+Path(visual_copy).name])
    add('B01','放弃历史版本回放后再选',['versions/round1'],'初赛 133 个历史提交包',
        '失去旧版本已封装的预测、包内源码或资产，无法直接回放对应提交；不能默认将所有历史版本视为无用。',
        '按文件逐项列在 submission_archives.csv；v122 两个文件名实际共用一个硬链接 inode，空间已去重。')
    add('B02','放弃旧特征复用后再选',['artifacts/shared_features'],'历史共享特征与交叉验证大数组',
        '历史算法可能需要重新生成 LOO/cross-fit/angle-delay 特征；精确重建仍依赖历史代码和环境。',
        '主要是 6 个约 3.146 GB 的复数训练特征数组及若干功率、标签和小型特征。')
    add('B03','放弃旧模型回放后再选',['artifacts/checkpoints'],'原项目历史模型权重',
        '旧模型直接推理能力可能丢失，重训练成本较高；plus-pro 内 M2/M3 模型和 final_code 发布模型不在此候选中。',
        '模型目录与当前实测模型目录分离；本清单没有声称这些历史权重均可低成本重建。')
    add('B04','放弃旧监督数据复用后再选',['artifacts/round2/e160_full4000_v7_supervision'],'E160 的 4000 点教师与频谱监督数组',
        '失去该历史分支的复信道教师、PAS/PDP 监督缓存；再训练该分支需重建。',
        '两个约 6.291 GB 的复数数组及四个约 1.573 GB 的半精度频谱数组占主要空间。')
    add('B05','放弃旧载波候选比较后再选',['artifacts/round2/complex_carrier_bank'],'17 路完整载波候选 bank',
        '失去旧载波候选集合和相关堆叠诊断输入。',
        '主数组 shape=(17,500,256,4,192)，complex64，约 13.369 GB。')
    add('B06','放弃旧版本回放后再选',['versions/round2'],'其余复赛历史归档（排除 A03）',
        '失去相应复赛历史或待提交包；保留 final_code 完整冻结交付仍是前提。',
        '逐包名称、版本、文件名所载分数与包内最大条目见 submission_archives.csv；分数仅按文件名转录。',
        ['final_code'],source_copies)
    add('B07','可再生成历史缓存',['artifacts/round2/pre_sionna'],'复赛早期基础特征与预测',
        '旧 pre-Sionna 分支重跑需重新准备相关支撑、频谱或预测。','按当前文件头读取数据类型及 shape，详见 file_inventory.csv。')
    add('B08','保留实验结论后再选',['artifacts/round2/e217_angle_delay_support_only'],'E217 及其 stale 缓存、变体输出',
        '失去该实验与 stale_v7_rebuild_cache 的现成中间结果；stale 名称不作为自动删除依据。','子目录占用与完整文件清单均已列出。')
    excluded=['artifacts/round2/'+name for name in ('e160_full4000_v7_supervision','complex_carrier_bank','pre_sionna','e217_angle_delay_support_only')]
    add('B09','放弃旧实验缓存复用后再选',['artifacts/round2'],'其余复赛生成物（排除 B04/B05/B07/B08）',
        '影响旧 foundation、RBF、lifecycle、stacking 等分支复用。','以排除规则保证与其他候选无重复计数。',exclusions=excluded)
    add('B10','旧外部参考缓存',['artifacts/external'],'旧外部输出和恢复场景缓存',
        '失去旧外部参考及其缓存；当前 plus-pro 内所需材料和运行库另存。','详见目录清单；没有把外部数据来源自动判断为可再下载。')
    add('B11','独立任务交付材料',['artifacts/portable_tasks'],'独立便携任务目录和归档',
        '可能失去独立任务交付及自带输入；选择前应确认还需不需要这些任务。','其中含 original Round1 地图副本；完整目录不能仅凭地图重复而整目录删除。')
    add('B12','历史无版本预测',['artifacts/legacy_outputs'],'旧无版本输出',
        '失去对应历史预测，不能再直接比较该数组。','与当前 plus-pro 保存预测分离。')
    add('B13','旧训练运行材料',['artifacts/training_runs'],'旧训练运行目录',
        '失去旧训练日志或运行资产；具体内容需按文件清单选择。','只列为候选，不自动判定为无用。')
    add('B14','旧诊断结果',['analysis/probes'],'两个历史 probe 预测数组',
        '失去现成诊断预测；分析报告和源码保留。','两个大数组，不是原始训练输入。')
    add('C01','原始输入备份，谨慎选择',['Round2_Map.zip'],'复赛原始输入压缩包',
        '删除后失去这一份原始打包备份；datasets/round2 内解压输入必须保留。包内不仅有地图，还有训练信道、坐标、Setup。',
        '压缩包五个条目与解压文件的大小/CRC32 校对结果见 source_archive_checks.json。', ['datasets/round2'])
    add('C02','原始输入备份，谨慎选择',['datasets/source_archives/Round1_Map.zip'],'初赛原始输入压缩包',
        '删除后失去这一份原始打包备份；datasets/round1 内解压输入必须保留。','五个条目的大小/CRC32 校对见 source_archive_checks.json。',['datasets/round1'])

    for option in options:
        if option['file_count']==0:
            raise ValueError('Empty selection: '+option['id'])
    (output/'cleanup_options.json').write_text(json.dumps(options,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    write_csv(output/'cleanup_options.csv',options,['id','tier','title','reclaimable_file_GB','logical_file_GB','file_count','paths','exclude_paths','impact','evidence','must_retain','executed'])
    file_rows=[]
    for row in sorted(files,key=lambda r:r['size_bytes'],reverse=True):
        file_rows.append(dict(row,size_GB=row['size_bytes']/1e9,size_GiB=row['size_bytes']/1024**3,
                              allocated_GB=row['allocated_bytes']/1e9,
                              standalone_reclaimable_GB=row['allocated_bytes']/1e9 if row['nlink']==1 else 0,
                              cleanup_id=selected_paths.get(row['path'],''),
                              handling='仅为候选，未删除' if row['path'] in selected_paths else '未列为清理候选；默认保留'))
    write_csv(output/'file_inventory.csv',file_rows,['path','role','size_GB','size_GiB','allocated_GB','standalone_reclaimable_GB','nlink','numpy_shape','numpy_dtype','mtime_cst','cleanup_id','handling'])
    inputs=[]
    for row in file_rows:
        name=row['path']
        official=name.startswith(('datasets/round1/','datasets/round2/')) and row['extension'] in ('.npy','.ply','.json')
        measured=name.startswith('plus-pro/datasets/dichasus/') and row['extension'] in ('.tfrecords','.bin') and '/upload_parts/' not in name
        archive=name in ('Round2_Map.zip','datasets/source_archives/Round1_Map.zip')
        if not (official or measured or archive):
            continue
        note='原始比赛输入；默认保留' if official else '完整 dc01；当前实测主输入' if name.endswith('dichasus-dc01.tfrecords') else '原始输入打包备份；含训练信道' if archive else '早期格式核验样本；不可当作完整实验数据集'
        if 'first1MiB' in name:
            note='仅截取前 1 MiB，不是完整文件'
        inputs.append(dict(row,content_note=note))
    write_csv(output/'data_inputs.csv',inputs,['path','size_GB','size_GiB','numpy_shape','numpy_dtype','content_note','cleanup_id'])
    enriched=[dict(r,allocated_GB=r['allocated_unique_bytes']/1e9,reclaimable_GB=r['reclaimable_bytes']/1e9,logical_GB=r['logical_file_bytes']/1e9) for r in directories.values()]
    write_csv(output/'directory_inventory.csv',sorted(enriched,key=lambda r:r['allocated_unique_bytes'],reverse=True),['path','allocated_GB','reclaimable_GB','logical_GB','files','directories','symlinks'])
    duplicate_rows=[]
    for i,group in enumerate(sorted(duplicates['confirmed'],key=lambda g:g['redundant_logical_bytes'],reverse=True),1):
        duplicate_rows.append(dict(group,group_id=f'D{i:03d}',redundant_GB=group['redundant_logical_bytes']/1e9,
                                   warning='相同内容可能用于不同独立交付；先保留每个所需交付的目录完整性。'))
    write_csv(output/'confirmed_duplicates.csv',duplicate_rows,['group_id','file_bytes','redundant_GB','sha256','paths','canonical_suggestion','warning'])
    submissions=[]
    for archive in archives:
        path=archive['path']
        if not path.startswith(('versions/round1/','versions/round2/')) or not path.endswith('.zip'):
            continue
        row=dict(archive)
        row['round']=path.split('/')[1]
        match=re.search(r'(?:^|/)([vV]\d+)',path)
        row['version']=match.group(1) if match else ''
        match=re.search(r'_online_([0-9]+\.[0-9]+)',path)
        row['score_from_filename']=match.group(1) if match else ''
        row['state_from_filename']='candidate_pending' if 'candidate_pending' in path else 'online_named' if match else 'other_archive'
        row['archive_GB']=row['archive_bytes']/1e9
        row['uncompressed_GB']=row.get('uncompressed_bytes',0)/1e9
        row['cleanup_id']=selected_paths.get(path,'')
        submissions.append(row)
    write_csv(output/'submission_archives.csv',sorted(submissions,key=lambda r:(r['round'],r['path'])),['path','round','version','score_from_filename','state_from_filename','archive_GB','uncompressed_GB','entry_count','top_entries','cleanup_id'])

    lines=['# hw 数据与磁盘占用清单','',f"扫描标签：{summary['label_cst']}；快照时间：{summary['snapshot_cst']}。",'',
           '本次只清点并写入清单，没有删除、移动、重命名或解压任何被扫描数据。所有路径相对于 hw 根目录。新增清单目录自身未计入快照，扫描后新增的小型整理工具另计。','',
           f"实际磁盘占用 **{summary['totals']['allocated_unique_bytes']/1e9:.2f} GB / {summary['totals']['allocated_unique_bytes']/1024**3:.2f} GiB**；清点 **{len(files):,} 个普通文件**、**{len(directories):,} 个目录（含根目录）**。",'',
           'GB 按 10^9 字节，GiB 按 2^30 字节。占用按 inode 去重；父目录已包含子目录，不能把父子行相加。候选项之间的文件集合互不重叠；释放量按文件已分配磁盘块估计，不包含少量目录元数据。文件表按路径列行，硬链接会有多行；汇总占用请使用目录表/summary，单文件释放量为零表示仅删这个硬链接名不释放数据。','',
           '## 1. 按用途看主要占用','',
           '| 路径 | 实际占用 GB | 文件数 | 内容与处理建议 |','|---|---:|---:|---|']
    descriptions={'versions':'历史提交与源码归档；按版本选择，保留所需版本。','artifacts':'原项目模型、缓存和历史预测；细分见候选清单。','plus-pro':'当前实测研究；含完整 dc01、上传分片、物理缓存、M2/M3 模型和报告。','datasets':'初复赛已解压原始输入及原始压缩包；默认保留原始输入。','final_code':'冻结的 V16 代码、模型、预测与完整交付；默认保留。','analysis':'历史报告、图和 probe 输出；优先保留报告。','code':'原项目研发源码，体积很小；默认保留。','experiment_records':'唯一实验记录；保留。','.runtime':'原位置运行目录。'}
    for row in sorted(summary['top_level'],key=lambda r:r['allocated_unique_bytes'],reverse=True):
        lines.append(f"| `{row['path']}/` | {row['allocated_unique_bytes']/1e9:.3f} | {row['files']} | {descriptions.get(row['path'],'其他目录')} |")
    lines+=['',f"根目录的 `Round2_Map.zip` 另占 **{indexed['Round2_Map.zip']['allocated_bytes']/1e9:.3f} GB**，已作为 C01 单列，不能从上表遗漏。",'',
            '## 2. 可按编号选择的清理候选','',
            '**这些是供选择的候选，不是已授权执行的删除列表。** A 组包含重复传输材料、历史临时预测与已证实的归档副本；其中 A02 删除会失去历史预测，风险高于单纯的传输分片。B 组会影响历史回放或复用；C 组是原始输入备份。','',
            '| 编号 | 内容 | 可释放 GB | 文件数 | 删除影响 |','|---|---|---:|---:|---|']
    for row in options:
        lines.append(f"| {row['id']} | {row['title']} | {row['reclaimable_file_GB']:.3f} | {row['file_count']} | {row['impact']} |")
    by_id={r['id']:r for r in options}
    lines+=['','候选的完整路径、排除项、保留前提和证据见 [cleanup_options.csv](cleanup_options.csv)。',
            f"- 仅 A01 + A02：约 **{sum(by_id[k]['reclaimable_file_GB'] for k in ('A01','A02')):.2f} GB**。",
            f"- A01 至 A04：约 **{sum(by_id[k]['reclaimable_file_GB'] for k in ('A01','A02','A03','A04')):.2f} GB**。",
            '- B01 是整个初赛历史归档集合。若只想删部分版本，请按 submission_archives.csv 指定要保留的版本，不把整组大小当成默认可删空间。',
            '- A04 以保留对应 versions/round1 归档为前提；若同时放弃 B01 全部初赛归档，代表主动放弃 v395 版本。','',
            '## 3. 当前实测研究中应保留的数据','',
            '| 数据 | 规模 | 用途 |','|---|---:|---|',
            '| `plus-pro/datasets/dichasus/dc01/dichasus-dc01.tfrecords` | 16.871 GB | 完整 32,133 条实测记录；保留原始输入。 |',
            '| `plus-pro/datasets/dichasus/dc01/context/`、`spec.json` 和校验/来源记录 | 小型文件 | 阵列映射、相位/时延标定、注册场景与来源证据。 |',
            '| `plus-pro/artifacts/measured/dc01_v1/channels.npy` | 0.134 GB | shape=(1023,2,8,4,256)，当前抽样工作视图。 |',
            f"| `plus-pro/artifacts/measured/dc01_geometry_v1/` | {directories['plus-pro/artifacts/measured/dc01_geometry_v1']['allocated_unique_bytes']/1e9:.3f} GB | 已生成的物理路径缓存，重建成本较高。 |",
            f"| `plus-pro/artifacts/measured/` 全部 | {directories['plus-pro/artifacts/measured']['allocated_unique_bytes']/1e9:.3f} GB | 包含上面工作视图/路径缓存及 M2/M3 五折权重、预测和对照；父子项不可重复加总。 |",
            f"| `plus-pro/.runtime/` | {directories['plus-pro/.runtime']['allocated_unique_bytes']/1e9:.3f} GB | 匹配 OptiX/RTCore 的运行库与安装材料，当前先保留。 |",
            '| `plus-pro/code/`、`plus-pro/final_code/`、报告和实验记录 | 体积较小 | 当前实测流程源码、原方法依赖和结果证据。 |',
            '| `final_code/`、`datasets/round1/`、`datasets/round2/` | 见目录清单 | 保留冻结交付与原始比赛输入，避免同时丢失模型和训练来源。 |','',
            '## 4. 证实重复与硬链接','',
            f"对至少 1 MiB 的同大小文件先做分段筛查，再做完整 SHA256。确认 **{len(duplicates['confirmed'])} 组独立副本重复**，按每组保留一份估算冗余文件内容约 **{summary['confirmed_duplicate_redundant_logical_bytes']/1e9:.3f} GB**；这只是已核实范围，不代表扫描了全部小文件内容。",'',
            '**这 5.7 GB 不能全部当作可立即删除的空间。** 相当一部分位于独立的冻结交付目录内，例如两套 V16 发布目录所需的同内容模型、base_field 与预测。目录按独立交付设计时，即便字节相同也可能各自需要保留。A03/A04 是本轮单独列出的外部归档副本。', '',
            '- `final_code/Round2_Final_Source_Submission.zip` 与 versions/round2 下同名文件是一组重复。',
            '- `final_code/V16_source_online_0.686754.zip` 与 versions/round2 下同名文件是另一组重复；前后两组的 SHA256 不同，不能把四个包视作相同。',
            '- `final_code/Round2_Test_Channel.npy` 与两套 V16 发布目录里的同名预测字节一致，合计三份；仍按冻结交付保留。',
            '- `versions/round1/v122_multiscale_template_apex_candidate_pending.zip` 与 `_online_0.496774.zip` 是同一个 inode 的硬链接；只删一个文件名不会释放约 0.787 GB 的底层数据。',
            '- plus-pro 的两个 libnvoptix 库文件名也共用 inode，应保留运行库别名。', '',
            '完整 SHA256 和路径组见 [confirmed_duplicates.csv](confirmed_duplicates.csv)，硬链接见 hardlinks.json。','',
            '## 5. 原始压缩包与解压文件','',
            '`Round2_Map.zip` 含 Round2_Train_Channel.npy、Map.ply、Train_Pos.npy、Test_Pos.npy、Setup.json 五项；`datasets/source_archives/Round1_Map.zip` 也含五项完整初赛输入。',
            'source_archive_checks.json 记录解压文件对压缩包条目的长度与 CRC32 校验。CRC32 用于校对输入备份，不能与“压缩包本身 SHA256 完全相同”的重复结论混为一谈。保留已解压输入可以继续使用，删除原始包会减少一份来源备份。','',
            '## 6. 清单入口与核验范围','',
            '- [file_inventory.csv](file_inventory.csv)：全部普通文件，含 GB/GiB、shape/dtype、硬链接及候选编号。',
            '- [data_inputs.csv](data_inputs.csv)：单列原始比赛输入、dc01 原始测量、早期微型/截断样本及原始压缩包，避免把小样本或传输分片误当作完整数据集。',
            '- [directory_inventory.csv](directory_inventory.csv)：全部目录的占用与文件数。',
            '- [cleanup_options.csv](cleanup_options.csv)：可选择的、不重复计数的清理项目。',
            '- [submission_archives.csv](submission_archives.csv)：初复赛提交包逐项内容；线上分数仅从文件名转录，不代表已复核评分，也不跨评分口径排名。',
            '- [confirmed_duplicates.csv](confirmed_duplicates.csv)：完整 SHA256 证实的重复组。',
            '- summary.json、validation.json、source_archive_checks.json、live_references.json：机器检查证据。',
            '- 没有跟随符号链接，未跨其他文件系统；本次没有扫描错误。进程检查仅覆盖同用户可见 cwd/文件描述符的瞬时状态，不能证明其他用户或未来运行均不依赖某目录。',
            '- 清单没有执行任何删除命令；后续应按选中的编号/路径明确列出删除范围，并在执行前重查对应文件状态和保留前提。','']
    (output/'README_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'options':len(options),'files':len(file_rows),'directories':len(directories),'submission_archives':len(submissions),'A_group_reclaimable_GB':sum(r['reclaimable_file_GB'] for r in options if r['id'].startswith('A'))},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    arguments=parser.parse_args()
    build(PATHS.resolve(arguments.output))
