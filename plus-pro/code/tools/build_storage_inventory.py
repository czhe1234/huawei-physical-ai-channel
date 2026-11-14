"""Read-only storage inventory, duplicate evidence and archive contents.

Only writes to the explicitly selected report directory. It never deletes,
moves, extracts, or changes any scanned input. Paths in reports are relative
to the selected scope. Large-file hashes are budgeted and resumable.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import stat
import struct
import sys
import time
import traceback
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import PATHS

CST = timezone(timedelta(hours=8))
MIB = 1024 ** 2


def atomic_json(path, obj):
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def csv_write(path, rows, fields):
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})
    temporary.replace(path)


def parents(rel, include_self=False):
    path = Path(rel)
    current = path if include_self else path.parent
    while True:
        yield '' if str(current) == '.' else current.as_posix()
        if str(current) == '.':
            break
        current = current.parent


def role_for(rel, kind):
    if kind == 'symlink':
        return 'symbolic_link'
    if rel.startswith('plus-pro/datasets/dichasus/dc01/upload_parts/'):
        return 'completed_transfer_chunks'
    if rel.startswith('plus-pro/datasets/'):
        return 'measured_raw_and_scene'
    if rel.startswith('plus-pro/artifacts/'):
        if rel.endswith(('.pt', '.pth', '.ckpt')):
            return 'measured_checkpoint'
        return 'measured_prepared_cache_and_prediction'
    if rel.startswith('plus-pro/.runtime/'):
        return 'measured_runtime_dependency'
    if rel.startswith('plus-pro/'):
        return 'measured_code_reports_and_history'
    if rel.startswith('final_code/'):
        return 'frozen_final_delivery'
    if rel.startswith('versions/round1/'):
        return 'round1_historical_submission'
    if rel.startswith('versions/round2/'):
        return 'round2_submission'
    if rel.startswith('artifacts/temporary/'):
        return 'historical_parameter_sweep'
    if rel.startswith('artifacts/shared_features/'):
        return 'historical_shared_features'
    if rel.startswith('artifacts/checkpoints/'):
        return 'historical_checkpoint'
    if rel.startswith('artifacts/round2/'):
        return 'round2_generated_artifact'
    if rel.startswith('artifacts/'):
        return 'other_generated_artifact'
    if rel.startswith('datasets/'):
        return 'original_competition_data_and_documents'
    if rel.startswith('analysis/'):
        return 'historical_analysis'
    if rel.startswith(('code/', 'experiment_records/')):
        return 'source_and_experiment_record'
    if rel.endswith(('.zip', '.tar', '.gz')):
        return 'root_source_archive'
    return 'root_document_or_configuration'


def npy_header(path):
    with path.open('rb') as stream:
        if stream.read(6) != b'\x93NUMPY':
            return {}
        version = stream.read(2)
        length = struct.unpack('<H' if version[0] == 1 else '<I', stream.read(2 if version[0] == 1 else 4))[0]
        if length > MIB:
            return {'npy_error': 'header larger than 1 MiB'}
        header = ast.literal_eval(stream.read(length).decode('latin-1' if version[0] < 3 else 'utf-8').strip())
        return {'numpy_shape': header.get('shape'), 'numpy_dtype': header.get('descr')}


def scan(scope, output):
    records, errors, mounts = [], [], []
    device = scope.stat().st_dev
    stack = [scope]
    while stack:
        directory = stack.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            errors.append({'path': str(directory.relative_to(scope)), 'error': str(exc)})
            continue
        for entry in entries:
            path = Path(entry.path)
            if path == output:
                continue
            rel = path.relative_to(scope).as_posix()
            try:
                info = entry.stat(follow_symlinks=False)
                kind = 'file' if stat.S_ISREG(info.st_mode) else 'dir' if stat.S_ISDIR(info.st_mode) else 'symlink' if stat.S_ISLNK(info.st_mode) else 'other'
                record = {'path': rel, 'kind': kind, 'size_bytes': info.st_size, 'allocated_bytes': getattr(info, 'st_blocks', 0) * 512,
                          'device': info.st_dev, 'inode': info.st_ino, 'nlink': info.st_nlink,
                          'mtime_ns': info.st_mtime_ns, 'mtime_cst': datetime.fromtimestamp(info.st_mtime, CST).isoformat(),
                          'role': role_for(rel, kind), 'extension': path.suffix.lower()}
                if kind == 'dir':
                    if info.st_dev == device:
                        stack.append(path)
                    else:
                        mounts.append(rel)
                elif kind == 'symlink':
                    record['symlink_target'] = os.readlink(path)
                elif kind == 'file' and path.suffix == '.npy':
                    try:
                        record.update(npy_header(path))
                    except Exception as exc:
                        record['npy_error'] = str(exc)
                records.append(record)
            except OSError as exc:
                errors.append({'path': rel, 'error': str(exc)})
    info = scope.stat()
    records.append({'path': '', 'kind': 'dir', 'size_bytes': info.st_size, 'allocated_bytes': info.st_blocks * 512,
                    'device': info.st_dev, 'inode': info.st_ino, 'nlink': info.st_nlink, 'role': 'scope_root'})
    return sorted(records, key=lambda r: r['path']), errors, mounts


def aggregate(records):
    groups = defaultdict(list)
    totals = defaultdict(lambda: {'files': 0, 'directories': 0, 'symlinks': 0, 'logical_file_bytes': 0, 'allocated_unique_bytes': 0, 'reclaimable_bytes': 0})
    seen = defaultdict(set)
    for row in records:
        identity = (row['device'], row['inode'])
        groups[identity].append(row)
        for ancestor in parents(row['path'], include_self=row['kind'] == 'dir'):
            result = totals[ancestor]
            result['files'] += row['kind'] == 'file'
            result['directories'] += row['kind'] == 'dir'
            result['symlinks'] += row['kind'] == 'symlink'
            if row['kind'] == 'file':
                result['logical_file_bytes'] += row['size_bytes']
            if identity not in seen[ancestor]:
                seen[ancestor].add(identity)
                result['allocated_unique_bytes'] += row['allocated_bytes']
    for items in groups.values():
        first = items[0]
        if first['kind'] == 'file' and first['nlink'] > len(items):
            continue
        common = set(parents(first['path'], include_self=first['kind'] == 'dir'))
        for row in items[1:]:
            common.intersection_update(parents(row['path'], include_self=row['kind'] == 'dir'))
        for ancestor in common:
            totals[ancestor]['reclaimable_bytes'] += first['allocated_bytes']
    return [dict(path=p or '.', **value) for p, value in sorted(totals.items())], groups


def archive_inventory(scope, files):
    rows = []
    for row in files:
        if row['extension'] not in ('.zip', '.npz'):
            continue
        item = {'path': row['path'], 'archive_bytes': row['size_bytes'], 'extension': row['extension']}
        try:
            with zipfile.ZipFile(scope / row['path']) as archive:
                entries = archive.infolist()
                item.update(entry_count=len(entries), uncompressed_bytes=sum(e.file_size for e in entries),
                            top_entries=[{'name': e.filename, 'bytes': e.file_size, 'crc32': f'{e.CRC:08x}'} for e in sorted(entries, key=lambda e: e.file_size, reverse=True)[:8]])
        except Exception as exc:
            item['error'] = str(exc)
        rows.append(item)
    return rows


def signature(record):
    return [record['size_bytes'], record['mtime_ns'], record['device'], record['inode']]


def confirm_duplicates(scope, files, output, budget_bytes):
    cache_path = output / 'hash_cache.json'
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    by_size = defaultdict(list)
    inode_seen = set()
    for row in files:
        if row['size_bytes'] < MIB:
            continue
        identity = (row['device'], row['inode'])
        if identity in inode_seen:
            continue
        inode_seen.add(identity)
        by_size[row['size_bytes']].append(row)
    sampled = defaultdict(list)
    failures, bytes_sampled = [], 0
    for size, items in by_size.items():
        if len(items) < 2:
            continue
        for row in items:
            try:
                dig = hashlib.sha256()
                with (scope / row['path']).open('rb') as stream:
                    for offset in sorted({0, size // 4, size // 2, 3 * size // 4, max(0, size - 65536)}):
                        stream.seek(offset)
                        block = stream.read(min(65536, size - offset))
                        dig.update(block)
                        bytes_sampled += len(block)
                sampled[(size, dig.hexdigest())].append(row)
            except OSError as exc:
                failures.append({'path': row['path'], 'error': str(exc)})
    candidate_groups = [v for v in sampled.values() if len(v) > 1]
    candidate_groups.sort(key=lambda rows: (not any(r['path'].endswith('.zip') for r in rows), sum(r['size_bytes'] for r in rows)))
    hashed_bytes, confirmed, unresolved = 0, [], []
    for group_index, group in enumerate(candidate_groups):
        hashes = defaultdict(list)
        for row in group:
            name = row['path']
            entry = cache.get(name, {})
            if entry.get('signature') == signature(row):
                digest = entry['sha256']
            elif hashed_bytes + row['size_bytes'] <= budget_bytes:
                path = scope / name
                start = path.stat()
                dig = hashlib.sha256()
                with path.open('rb') as stream:
                    while True:
                        block = stream.read(8 * MIB)
                        if not block:
                            break
                        dig.update(block)
                end = path.stat()
                if (start.st_size, start.st_mtime_ns) != (end.st_size, end.st_mtime_ns):
                    failures.append({'path': name, 'error': 'changed during hashing'})
                    continue
                digest = dig.hexdigest()
                hashed_bytes += row['size_bytes']
                cache[name] = {'signature': signature(row), 'sha256': digest}
                atomic_json(cache_path, cache)
            else:
                unresolved.append({'path': name, 'reason': 'full SHA256 budget reached; sample equality is not duplicate proof'})
                continue
            hashes[digest].append(row)
        for digest, equal in hashes.items():
            if len(equal) > 1:
                preferred = sorted(equal, key=lambda r: (not r['path'].startswith('final_code/'), not r['path'].startswith('datasets/'), r['path']))
                confirmed.append({'sha256': digest, 'file_bytes': equal[0]['size_bytes'], 'paths': [r['path'] for r in preferred],
                                  'independent_inode_count': len(equal), 'redundant_logical_bytes': (len(equal)-1)*equal[0]['size_bytes'],
                                  'redundant_reclaimable_bytes_upper_bound': sum(r['allocated_bytes'] for r in preferred[1:] if r['nlink'] == 1),
                                  'canonical_suggestion': preferred[0]['path']})
        if group_index == 0 or (group_index + 1) % 10 == 0 or group_index + 1 == len(candidate_groups):
            print(json.dumps({'phase': 'duplicates', 'group': group_index + 1, 'groups': len(candidate_groups), 'hashed_GB': round(hashed_bytes/1e9, 3)}, ensure_ascii=False), flush=True)
    return {'confirmed': confirmed, 'unresolved': unresolved, 'failures': failures, 'sampled_bytes': bytes_sampled,
            'full_hash_bytes_this_run': hashed_bytes, 'minimum_file_bytes': MIB, 'hash_budget_bytes': budget_bytes}


def live_references(scope):
    observed, inaccessible = [], 0
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            command = (proc / 'comm').read_text().strip()
            cwd = Path(os.readlink(proc / 'cwd'))
            references = []
            try:
                descriptors = list((proc / 'fd').iterdir())
            except OSError:
                descriptors = []
                inaccessible += 1
            for fd in descriptors:
                try:
                    target = os.readlink(fd)
                    target_path = Path(target[:-10] if target.endswith(' (deleted)') else target)
                    if target_path == scope or scope in target_path.parents:
                        references.append(str(target_path.relative_to(scope)))
                except (OSError, ValueError):
                    pass
            if references or cwd == scope or scope in cwd.parents:
                observed.append({'pid': int(proc.name), 'command': command,
                                 'cwd': str(cwd.relative_to(scope)) if cwd == scope or scope in cwd.parents else '(outside scope)',
                                 'open_files': sorted(set(references))})
        except (OSError, ValueError):
            inaccessible += 1
    return {'own_user_processes': observed, 'inaccessible_process_checks': inaccessible,
            'limitation': 'Own-user cwd/open-descriptor snapshot only; not proof that another user, mmap-only file, or future job has no dependency.'}


def run(args):
    started = time.perf_counter()
    scope, output = PATHS.resolve(args.scope), PATHS.resolve(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if output == scope or scope not in output.parents:
        raise ValueError('Report output must be a proper child of the scanned scope.')
    atomic_json(output / 'phase_state.json', {'phase': 'metadata', 'label': args.label})
    records, errors, mounts = scan(scope, output)
    files = [r for r in records if r['kind'] == 'file']
    atomic_json(output / 'metadata.json', records)
    directories, inode_groups = aggregate(records)
    csv_write(output / 'files.csv', sorted(files, key=lambda r: r['size_bytes'], reverse=True),
              ['path','role','size_bytes','allocated_bytes','nlink','mtime_cst','numpy_shape','numpy_dtype','extension','device','inode'])
    csv_write(output / 'directories.csv', sorted(directories, key=lambda r:r['allocated_unique_bytes'], reverse=True),
              ['path','files','directories','symlinks','logical_file_bytes','allocated_unique_bytes','reclaimable_bytes'])
    print(json.dumps({'phase':'metadata_complete','files':len(files),'directories':len(directories)},ensure_ascii=False),flush=True)
    atomic_json(output / 'phase_state.json', {'phase': 'archive_metadata'})
    archives = archive_inventory(scope, files)
    atomic_json(output / 'archives.json', archives)
    csv_write(output / 'archives.csv', archives, ['path','archive_bytes','uncompressed_bytes','entry_count','top_entries','error'])
    atomic_json(output / 'phase_state.json', {'phase': 'duplicate_verification'})
    duplicates = confirm_duplicates(scope, files, output, int(args.hash_budget_gib*1024**3))
    atomic_json(output / 'duplicates.json', duplicates)
    atomic_json(output / 'live_references.json', live_references(scope))
    hardlinks = [{'paths':[r['path'] for r in items],'nlink':items[0]['nlink'],'size_bytes':items[0]['size_bytes'],
                  'allocated_bytes':items[0]['allocated_bytes']} for items in inode_groups.values() if items[0]['kind']=='file' and (len(items)>1 or items[0]['nlink']>1)]
    atomic_json(output / 'hardlinks.json', hardlinks)
    extensions = defaultdict(lambda:{'files':0,'logical_bytes':0})
    for row in files:
        extensions[row['extension']]['files'] += 1
        extensions[row['extension']]['logical_bytes'] += row['size_bytes']
    root_total = next(d for d in directories if d['path']=='.')
    summary = {'label_cst':args.label,'snapshot_cst':datetime.now(CST).isoformat(),'scope_label':scope.name,
               'excluded_report_directory':output.relative_to(scope).as_posix(),'elapsed_seconds':time.perf_counter()-started,
               'totals':root_total,'scan_errors':errors,'skipped_other_filesystems':mounts,'extensions':dict(extensions),
               'confirmed_duplicate_groups':len(duplicates['confirmed']),
               'confirmed_duplicate_redundant_logical_bytes':sum(g['redundant_logical_bytes'] for g in duplicates['confirmed']),
               'unresolved_full_hash_files':len(duplicates['unresolved']),
               'largest_files':sorted(files,key=lambda r:r['size_bytes'],reverse=True)[:80],
               'top_level':[d for d in directories if d['path']!='.' and '/' not in d['path']],
               'disk_space':dict(zip(('total','used','free'),__import__('shutil').disk_usage(scope))),
               'read_only_input_policy':True,'followed_symlinks':False,'deleted_files':0}
    atomic_json(output / 'summary.json',summary)
    atomic_json(output / 'phase_state.json',{'phase':'complete','elapsed_seconds':summary['elapsed_seconds']})
    print(json.dumps({'phase':'complete','files':len(files),'allocated_GB':root_total['allocated_unique_bytes']/1e9,
                      'duplicate_groups':summary['confirmed_duplicate_groups'],'elapsed_seconds':summary['elapsed_seconds']},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope',default='..')
    parser.add_argument('--output',required=True)
    parser.add_argument('--label',required=True)
    parser.add_argument('--hash-budget-gib',type=float,default=20)
    options=parser.parse_args()
    try:
        run(options)
    except Exception:
        target=PATHS.resolve(options.output)
        target.mkdir(parents=True,exist_ok=True)
        (target/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
        raise
