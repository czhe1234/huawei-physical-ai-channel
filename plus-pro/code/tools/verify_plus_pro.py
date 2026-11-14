#!/usr/bin/env python3
"""Check relocated assets and optionally write the complete content checksum list."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/measured'))
from project_paths import PATHS


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, path)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full-hash', action='store_true')
    args = parser.parse_args()
    started = time.monotonic()
    evidence = ROOT / 'analysis/consolidation'
    assert PATHS.root == ROOT, 'Project root escaped the new workspace'
    ledger = json.loads((evidence / 'move_ledger.json').read_text())
    assert ledger['status'] == 'complete'
    for item in ledger['entries']:
        final = ROOT / item['path']
        assert final.stat().st_ino == item['source_inode'], item['path']
        assert not (ROOT.parent / item['path']).exists(), item['path']
    assert not (ROOT / '.organizing').exists()
    assert not any(path.is_symlink() for path in ROOT.rglob('*'))
    from dc01_state import ComponentStateModel
    from dc01_angular_state import AngularState
    from dc01_spectral import original_classes
    import torch
    import numpy as np
    torch.set_num_threads(4)
    original_source = Path(inspect.getfile(ComponentStateModel)).resolve()
    original_source.relative_to(ROOT)
    _, spectral_source_hashes = original_classes()
    artifacts = PATHS.artifacts / 'measured'
    foundation = artifacts / 'dc01_m3_foundation_v1'
    actual = artifacts / 'dc01_m3_angular_v1'
    control = artifacts / 'dc01_m3_angular_shuffle_v1'
    for fold in range(5):
        assert (foundation / f'fold{fold}/foundation_report.json').is_file()
        for folder in [actual, control]:
            assert (folder / f'fold{fold}/angular_report.json').is_file()
            assert (folder / f'fold{fold}/angular_model.pt').is_file()
    for folder in [actual, control]:
        checkpoint = torch.load(folder / 'fold0/angular_model.pt', map_location='cpu', weights_only=False)
        model = AngularState()
        model.load_state_dict(checkpoint['model'], strict=True)
        assert checkpoint['identity']['code'] == digest(ROOT / 'code/measured/dc01_angular_state.py')
        del model, checkpoint
    # Recompute from frozen predictions; this does not fit or tune any model.
    from dc01_m3_report import verify
    report = json.loads((PATHS.analysis / 'results/measured_dc01_m3/evaluation.json').read_text())
    truth = np.load(artifacts / 'dc01_v1/channels.npy', mmap_mode='r')
    metadata = np.load(artifacts / 'dc01_v1/selection.npz')
    split = json.loads((foundation / 'split_manifest.json').read_text())
    numerical = {name: verify(folder, foundation, truth, metadata, split, report['metrics'][name])
                 for name, folder in [('factorized_data', foundation), ('M3_geometry', actual), ('shuffled_geometry', control)]}
    # Use a separate process because the frozen V16 package also names itself round2.
    physics = subprocess.run([sys.executable, '-c',
        'import json; from pathlib import Path; from project_paths import PATHS; '
        'import round2.physical_prior.physical_rt as rt; '
        'p=Path(rt.__file__).resolve(); p.relative_to(PATHS.root); '
        'print(json.dumps({"module":str(p.relative_to(PATHS.root))}))'],
        cwd=ROOT, text=True, capture_output=True, check=True)
    library = ROOT / '.runtime/nvidia-550.135/extracted/libnvoptix.so.550.135'
    ctypes.CDLL(str(library))
    result = dict(status='passed', primary_moved_entries=len(ledger['entries']),
        directory_inodes_preserved=True, old_measured_paths_absent=True, symbolic_links=0,
        original_component_source=str(original_source.relative_to(ROOT)),
        original_spectral_sources=spectral_source_hashes, physics_import_output=physics.stdout.strip(),
        checkpoint_loads='Actual and shuffled fold0 strict CPU loads passed; all 10 checkpoints present',
        independent_prediction_verification=numerical, elapsed_s=time.monotonic() - started)
    atomic_json(evidence / 'verification.json', result)
    print(json.dumps(dict(stage='runtime_and_prediction_checks', status='passed', elapsed_s=result['elapsed_s'])), flush=True)
    if args.full_hash:
        # Mutable consolidation diagnostics and this checksum file are excluded.
        files = sorted(path for path in ROOT.rglob('*') if path.is_file()
                       and path.name != 'checksums.sha256'
                       and '__pycache__' not in path.parts
                       and evidence not in path.parents)
        rows = []
        last = time.monotonic()
        def hash_file(path):
            return path, digest(path)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for path, value in pool.map(hash_file, files):
                rows.append((path, value))
                if time.monotonic() - last > 15:
                    progress = dict(stage='hashing', completed=len(rows), total=len(files), elapsed_s=time.monotonic() - started)
                    atomic_json(evidence / 'hash_progress.json', progress)
                    print(json.dumps(progress), flush=True)
                    last = time.monotonic()
        hashes = {str(path.relative_to(ROOT)): value for path, value in rows}
        raw = 'datasets/dichasus/dc01/dichasus-dc01.tfrecords'
        assert hashes[raw] == '631fa66bb82405f28926b4e6524ab02ce10e0b1b35232b153054f18d9c72b16d'
        temporary = ROOT / 'checksums.sha256.tmp'
        temporary.write_text(''.join(value + '  ' + str(path.relative_to(ROOT)) + '\n' for path, value in rows), encoding='utf-8')
        os.replace(temporary, ROOT / 'checksums.sha256')
        summary = dict(status='complete', files=len(rows), bytes=sum(path.stat().st_size for path, _ in rows),
            raw_sha256=hashes[raw], elapsed_s=time.monotonic() - started,
            excludes=['checksums.sha256', '**/__pycache__/**', 'analysis/consolidation/**'])
        atomic_json(evidence / 'checksum_summary.json', summary)
        atomic_json(evidence / 'hash_progress.json', summary)
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
