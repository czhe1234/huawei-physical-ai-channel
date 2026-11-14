#!/usr/bin/env python3
"""Move the measured study into its own workspace without copying large data."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import time
import traceback


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    target = source / 'plus-pro'
    if json.loads((target / '.plus-pro-package.json').read_text())['kind'] != 'measured-dc01-study':
        raise ValueError('Unrecognized destination')
    if source.stat().st_dev != target.stat().st_dev:
        raise ValueError('Atomic directory moves require the same filesystem')
    evidence = target / 'analysis/consolidation'
    staging = target / '.organizing/copy_attempt'
    ledger_path = evidence / 'move_ledger.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(status='moving', entries=[])
    completed = {entry['path'] for entry in ledger['entries']}
    relatives = ['code/measured', 'artifacts/measured', 'datasets/dichasus', '.runtime/nvidia-550.135']
    results = source / 'analysis/results'
    relatives += ['analysis/results/' + path.name for path in sorted(results.iterdir())
                  if path.name.startswith('measured') or path.name == 'project_code_analysis_20261015.md']
    # Also recognize earlier completed moves when resuming this organizer.
    relatives += [rel for rel in sorted(completed) if rel not in relatives]
    for relative in relatives:
        old, new = source / relative, target / relative
        if relative in completed:
            if old.exists() or not new.exists():
                raise RuntimeError('Unexpected state for a completed move: ' + relative)
            continue
        before = old.stat()
        paths = list(old.rglob('*')) if old.is_dir() else [old]
        files = [path for path in paths if path.is_file()]
        entry = dict(path=relative, files=len(files), bytes=sum(path.stat().st_size for path in files),
                     source_device=before.st_dev, source_inode=before.st_ino)
        if new.exists():
            saved_copy = staging / relative
            saved_copy.parent.mkdir(parents=True, exist_ok=True)
            if saved_copy.exists():
                raise FileExistsError('Unexpected duplicate staging directory')
            os.rename(new, saved_copy)
        new.parent.mkdir(parents=True, exist_ok=True)
        os.rename(old, new)
        after = new.stat()
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise RuntimeError('Move did not preserve the original directory inode')
        entry.update(destination_inode=after.st_ino, inode_preserved=True)
        ledger['entries'].append(entry)
        atomic_json(ledger_path, ledger)
        print(json.dumps(dict(moved=relative, bytes=entry['bytes'], inode_preserved=True)), flush=True)
    # Turn the driver-library alias into an internal hard link so the package
    # contains no symbolic links and does not duplicate the library's disk blocks.
    converted = []
    for link in (target / '.runtime').rglob('*'):
        if not link.is_symlink():
            continue
        resolved = link.resolve()
        resolved.relative_to(target)
        temporary = link.with_name(link.name + '.portable')
        os.link(resolved, temporary)
        os.replace(temporary, link)
        converted.append(str(link.relative_to(target)))
    ledger.update(status='moved', runtime_aliases_converted_to_internal_hard_links=converted)
    atomic_json(ledger_path, ledger)
    # These directories were created only by the interrupted copy operation.
    # Validate every staged path against the originals now at their final paths
    # before discarding the incomplete copies. No original is deleted here.
    redundant = []
    if staging.exists():
        staging.resolve().relative_to(target.resolve())
        for path in staging.rglob('*'):
            if path.is_symlink():
                raise RuntimeError('Unexpected staging symlink')
            if not path.is_file():
                continue
            relative = path.relative_to(staging)
            original = target / relative
            if not original.is_file() or path.stat().st_size > original.stat().st_size:
                raise RuntimeError('Unrecognized staging file; retaining it: ' + str(relative))
            redundant.append(dict(path=str(relative), bytes=path.stat().st_size))
        atomic_json(evidence / 'redundant_copy_inventory.json', dict(
            kind='Files created by this organizer before the user requested moves',
            original_inodes_preserved=True, originals_at_final_paths=True, files=redundant))
        shutil.rmtree(staging)
        if not any(staging.parent.iterdir()):
            staging.parent.rmdir()
    ledger.update(status='complete', discarded_copy_files=len(redundant),
                  discarded_copy_bytes=sum(row['bytes'] for row in redundant), completed_unix=time.time())
    atomic_json(ledger_path, ledger)
    atomic_json(evidence / 'progress.json', dict(stage='moves_complete',
        moved_entries=len(ledger['entries']), bytes=sum(row['bytes'] for row in ledger['entries']),
        discarded_copy_bytes=ledger['discarded_copy_bytes']))
    shutil.copy2(__file__, target / 'code/tools/relocate_plus_pro.py')
    print(json.dumps({key: value for key, value in ledger.items() if key != 'entries'}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
