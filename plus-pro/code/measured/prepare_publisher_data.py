"""Prepare publisher-calibrated dc01/dc02 views on CPU; never fit or score models."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import struct
import sys
import time
import traceback
import numpy as np
from scipy.spatial import cKDTree
from project_paths import PATHS
if __package__:
    from .dc01_data import read_record, coordinate_frame
else:
    from dc01_data import read_record, coordinate_frame

DATASETS = {
    'dc01': dict(raw='datasets/dichasus/dc01/dichasus-dc01.tfrecords',
        receipt='datasets/dichasus/dc01/upload_verified.json', frames=32133,
        md5='7934b83963491b493a37556484d6ca42',
        sha256='631fa66bb82405f28926b4e6524ab02ce10e0b1b35232b153054f18d9c72b16d',
        offsets_sha256='0dec00182bb605bbcdbe78d0953c76519d8b68a3266f785bef2aae75b09c7760'),
    'dc02': dict(raw='datasets/dichasus/dichasus-dcxx/dichasus-dc02.tfrecords',
        receipt='datasets/dichasus/.verification/darus-269446.json', frames=31376,
        md5='51d762df6a95bbb5a9ad7dc1274f8038',
        sha256='43de9576a4650c75f0be09e5fa8ac0f7e3f1a1281a51aafc556ea76e2a444d2b',
        offsets_sha256='326ddd5cf216fc38ccd7e1a8db8523097e679adc1545cc9efbd676ab3d3f9201'),
}
RECORD_BYTES = 525022
CONFIG = dict(voxel_m=0.20, maximum_points=6000, selection_seed=20261015,
              frequency_stride=4, block_m=4.0, guard_m=0.30)

def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''): value.update(block)
    return value.hexdigest()

def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)

def atomic_npz(path, **arrays):
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('wb') as stream: np.savez(stream, **arrays)
    os.replace(temporary, path)

def describe(path, digest=None):
    return dict(path=PATHS.relative(path), bytes=path.stat().st_size,
                sha256=digest if digest is not None else sha256(path))

def select_positions(position, config=CONFIG):
    cells = np.floor(position[:, :2] / config['voxel_m']).astype(np.int64)
    _, groups = np.unique(cells, axis=0, return_inverse=True)
    centers = (cells + 0.5) * config['voxel_m']
    distance = np.sum((position[:, :2] - centers) ** 2, axis=1)
    order = np.lexsort((np.arange(len(position)), distance, groups))
    chosen = order[np.r_[True, groups[order][1:] != groups[order][:-1]]]
    available = len(chosen)
    if available > config['maximum_points']:
        chosen = np.random.default_rng(config['selection_seed']).choice(chosen, config['maximum_points'], replace=False)
    return np.sort(chosen), groups, available

def spatial_split(position, config=CONFIG):
    tiles = np.floor(position[:, :2] / config['block_m']).astype(np.int64)
    colors = (tiles[:, 0] + 2 * tiles[:, 1]) % 5
    splits = []
    for outer in range(5):
        test = np.flatnonzero(colors == outer)
        val = np.flatnonzero(colors == (outer + 1) % 5)
        possible = np.flatnonzero((colors != outer) & (colors != (outer + 1) % 5))
        held = np.r_[test, val]
        if len(held) == 0: raise ValueError('Empty held partition')
        distance = cKDTree(position[held, :2]).query(position[possible, :2], k=1)[0]
        fit = possible[distance >= config['guard_m']]
        if min(len(fit), len(val), len(test)) < 16:
            raise ValueError('Spatial split lacks the registered minimum support')
        assert not (set(fit) & set(test) or set(fit) & set(val) or set(val) & set(test))
        nearest = cKDTree(position[fit, :2]).query(position[test, :2], k=1)[0]
        splits.append(dict(outer=outer, fit=fit.tolist(), val=val.tolist(), test=test.tolist(),
                           min_test_fit_distance=float(nearest.min()),
                           test_fit_distance_quantiles=np.quantile(nearest, [0, .25, .5, .75, 1]).tolist()))
    digest = hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest()
    return dict(block_m=config['block_m'], guard_m=config['guard_m'], fold=colors.tolist(), splits=splits, sha256=digest)

def source_scan(dataset, paths, output, contract_hash, resume):
    metadata_path = output / 'all_metadata.npz'
    scan_path = output / 'source_scan.json'
    if resume and metadata_path.exists() and scan_path.exists():
        scan = json.loads(scan_path.read_text())
        if scan['contract_hash'] != contract_hash or sha256(metadata_path) != scan['metadata_sha256']:
            raise ValueError('Metadata checkpoint identity changed')
        source_stat = paths['raw'].stat()
        if [source_stat.st_size, source_stat.st_mtime_ns] != scan['raw_stat']:
            raise ValueError('Source changed since its verified scan')
        return scan
    count = DATASETS[dataset]['frames']
    positions = np.empty((count, 3), np.float64)
    times = np.empty(count, np.float64)
    ages = np.empty(count, np.float64)
    snr = np.empty((count, 64), np.float32)
    cfo = np.empty_like(snr)
    md5, sha = hashlib.md5(), hashlib.sha256()
    start = time.monotonic()
    initial = paths['raw'].stat()
    with paths['raw'].open('rb') as stream:
        for index in range(count):
            block = stream.read(RECORD_BYTES)
            if len(block) != RECORD_BYTES: raise ValueError(f'Truncated frame {index}')
            if struct.unpack('<Q', block[:8])[0] != RECORD_BYTES - 16: raise ValueError(f'Frame length {index}')
            md5.update(block); sha.update(block)
            record = read_record(block, 0, verify_crc=index in (0, count // 2, count - 1))
            if record['csi'].shape != (64, 1024, 2) or record['csi'].dtype != np.float32:
                raise ValueError(f'Unexpected CSI tensor at {index}')
            if not np.isfinite(record['csi']).all(): raise ValueError(f'Nonfinite CSI at {index}')
            positions[index], times[index], ages[index] = record['pos-tachy'], record['time'], record['gt-interp-age-tachy']
            snr[index], cfo[index] = record['snr'], record['cfo']
            if (index + 1) % 4096 == 0 or index + 1 == count:
                print(json.dumps(dict(dataset=dataset, stage='source_identity_metadata', frames=index+1,
                      total=count, elapsed_s=round(time.monotonic()-start, 3))), flush=True)
        if stream.read(1): raise ValueError('Unexpected extra frames')
    final = paths['raw'].stat()
    if (initial.st_size, initial.st_mtime_ns) != (final.st_size, final.st_mtime_ns): raise ValueError('Source changed while reading')
    if md5.hexdigest() != DATASETS[dataset]['md5'] or sha.hexdigest() != DATASETS[dataset]['sha256']:
        raise ValueError('Original file does not match the pinned source identity')
    for key, array in dict(position=positions, time=times, age=ages, snr=snr, cfo=cfo).items():
        if not np.isfinite(array).all(): raise ValueError(f'Nonfinite metadata {key}')
    atomic_npz(metadata_path, position=positions, time=times, age=ages, snr=snr, cfo=cfo)
    scan = dict(contract_hash=contract_hash, frames=count, bytes=initial.st_size, md5=md5.hexdigest(),
                sha256=sha.hexdigest(), all_source_csi_finite=True, metadata_sha256=sha256(metadata_path),
                raw_stat=[final.st_size, final.st_mtime_ns], elapsed_s=time.monotonic()-start)
    atomic_json(scan_path, scan)
    return scan

def prepare(dataset, resume=False):
    started = time.monotonic()
    cfg = DATASETS[dataset]
    common = PATHS.datasets / 'dichasus/dichasus-dcxx'
    paths = dict(raw=PATHS.root/cfg['raw'], spec=common/'spec.json',
        offsets=common/'context'/f'reftx-offsets-dichasus-{dataset}.json',
        coordinates=PATHS.datasets/'dichasus/dc01/context/data/coordinates.csv')
    receipt = PATHS.root/cfg['receipt']
    registered = json.loads(receipt.read_text())
    if registered['sha256'] != cfg['sha256'] or registered['md5'] != cfg['md5']:
        raise ValueError('Source receipt differs from pinned identity')
    if paths['raw'].stat().st_size != RECORD_BYTES*cfg['frames']: raise ValueError('Source size mismatch')
    if sha256(paths['offsets']) != cfg['offsets_sha256']: raise ValueError('Publisher calibration snapshot changed')
    if sha256(paths['coordinates']) != '227cb5707af19cc8a68ae06b0baf2c5a76c1cbb49e236bb8e516ee20e9076d8f':
        raise ValueError('Registered coordinate frame changed')
    spec = json.loads(paths['spec'].read_text())
    if float(spec['frequency']) != 3.438e9 or float(spec['bandwidth']) != 50e6: raise ValueError('Frequency definition changed')
    if spec != json.loads((PATHS.datasets/'dichasus/dc01/spec.json').read_text()): raise ValueError('dcxx antenna spec is inconsistent')
    assignments = np.asarray([a['assignments'] for a in spec['antennas']], dtype=np.int64)
    if assignments.shape != (2, 4, 8) or sorted(assignments.ravel().tolist()) != list(range(64)):
        raise ValueError('Antenna mapping is not a 64-port bijection')
    offsets = json.loads(paths['offsets'].read_text())
    cpo, sto = np.asarray(offsets['cpo'], np.float64), np.asarray(offsets['sto'], np.float64)
    if cpo.shape != (64,) or sto.shape != (64,) or not np.isfinite(cpo).all() or not np.isfinite(sto).all():
        raise ValueError('Calibration must contain 64 finite rad/sample values')
    if cpo[0] != 0 or sto[0] != 0: raise ValueError('Reference port changed')
    code = dict(prepare_sha256=sha256(Path(__file__)), reader_sha256=sha256(Path(__file__).with_name('dc01_data.py')))
    inputs = {key: describe(path, cfg['sha256'] if key == 'raw' else None) for key, path in paths.items()}
    inputs['raw'].update(md5=cfg['md5'], frames=cfg['frames'], record_bytes=RECORD_BYTES,
                         verified_receipt_sha256=sha256(receipt))
    contract = dict(schema='dichasus-publisher-preparation-v1', dataset=dataset, version=f'{dataset}_publisher_v1',
                    inputs=inputs, config=CONFIG, code=code)
    contract_hash = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    output = PATHS.artifacts/'measured'/f'{dataset}_publisher_v1'
    if output.exists():
        if not resume: raise FileExistsError(f'Refusing to overwrite {PATHS.relative(output)}; explicit --resume required')
        state = json.loads((output/'run_state.json').read_text())
        if state['contract_hash'] != contract_hash: raise ValueError('Existing output belongs to another preparation contract')
        if (output/'preparation_manifest.json').exists():
            print(json.dumps(dict(dataset=dataset, status='already_complete', directory=PATHS.relative(output))), flush=True)
            return
    else: output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'run_state.json', dict(contract_hash=contract_hash, status='running', contract=contract))
    try:
        scan = source_scan(dataset, paths, output, contract_hash, resume)
        with np.load(output/'all_metadata.npz', allow_pickle=False) as meta:
            positions, times = meta['position'], meta['time']
        selected, groups, available = select_positions(positions)
        origin, rotation = coordinate_frame(PATHS.datasets/'dichasus/dc01')
        converted = (positions[selected]-origin) @ rotation.T
        panels = (np.asarray([a['location'] for a in spec['antennas']])-origin) @ rotation.T
        directions = np.asarray([a['direction'] for a in spec['antennas']]) @ rotation.T
        indices = np.arange(0, 1024, CONFIG['frequency_stride'], dtype=np.int64)
        physical_frequency = float(spec['frequency']) + (indices-512)*float(spec['bandwidth'])/1024
        selection = dict(source_id=selected, position=converted, original_position=positions[selected], time=times[selected],
            group=groups[selected], panel_position=panels, panel_direction=directions, frequency_indices=indices,
            origin=origin, rotation=rotation, raw_position=positions[selected], directions=directions,
            frequency_index=indices, frequency_hz=physical_frequency, antenna_assignments=assignments)
        split = spatial_split(converted)
        legacy = None
        if dataset == 'dc01':
            old = PATHS.artifacts/'measured/dc01_v1/selection.npz'
            with np.load(old, allow_pickle=False) as previous:
                for key in previous.files:
                    if key not in selection or previous[key].dtype != selection[key].dtype or not np.array_equal(previous[key], selection[key]):
                        raise ValueError(f'dc01 legacy selection changed: {key}')
            old_split = PATHS.artifacts/'measured/dc01_baselines_v1/split_manifest.json'
            if split != json.loads(old_split.read_text()): raise ValueError('dc01 legacy split changed')
            legacy = dict(selection=describe(old), split=describe(old_split), old_keys_exact=True)
            shutil.copyfile(old_split, output/'split_manifest.json')
        else: atomic_json(output/'split_manifest.json', split)
        atomic_npz(output/'selection.npz', **selection)
        correction = np.exp(1j*(cpo[:,None]+2*np.pi*sto[:,None]*np.arange(1024)/1024)).astype(np.complex64)
        temporary = output/'channels.partial.npy'
        channel = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.complex64, shape=(len(selected),2,8,4,256))
        max_power_change = 0.0
        with paths['raw'].open('rb') as stream:
            for row, source_id in enumerate(selected):
                stream.seek(int(source_id)*RECORD_BYTES)
                record = read_record(stream.read(RECORD_BYTES), 0)
                raw = record['csi'][...,0]+1j*record['csi'][...,1]
                shifted = np.fft.fftshift(raw, axes=1).astype(np.complex64)
                calibrated = shifted*correction
                selected_raw = shifted[:, indices]
                selected_calibrated = calibrated[:, indices]
                before = np.sum(np.abs(selected_raw.astype(np.complex128))**2)
                after = np.sum(np.abs(selected_calibrated.astype(np.complex128))**2)
                relative = abs(after-before)/before if before else abs(after-before)
                max_power_change = max(max_power_change, float(relative))
                if relative > 1e-6: raise ValueError('Unit-modulus calibration changed source energy')
                value = calibrated[assignments][:,:,:,indices].transpose(0,2,1,3)
                if not np.isfinite(value).all(): raise ValueError('Nonfinite prepared H')
                channel[row] = value
                if (row+1)%256 == 0 or row+1 == len(selected):
                    print(json.dumps(dict(dataset=dataset, stage='publisher_calibration', positions=row+1, total=len(selected))), flush=True)
        channel.flush(); del channel
        os.replace(temporary, output/'channels.npy')
        audit = dict(status='complete', dataset=dataset, source_records=cfg['frames'], selected_points=len(selected),
            unique_voxels=available, channel_shape=[len(selected),2,8,4,256], dtype='complex64',
            all_source_csi_finite=scan['all_source_csi_finite'], panel_assignments_unique=True,
            max_calibration_power_relative_change=max_power_change,
            timestamp_range=[float(times.min()),float(times.max())],
            calibration='publisher per-file/per-antenna CPO/STO, raw port 0 reference; no fitted gain or phase alignment',
            formula='fftshift(raw)[a,k] * exp(+1j*(cpo[a]+2*pi*sto[a]*k/1024)); k=0..1023',
            cpo_units='rad', sto_units='samples', layout=['position','panel','horizontal','vertical','frequency'],
            limits=['Publisher offsets do not establish inter-panel or mobile-TX absolute phase/time coherence',
                    'dc02 is a same-scene trajectory record, not established independent acquisition',
                    'No model training, prediction or performance scoring',
                    'Spatial guard is fit-to-held only; cross-record transfer requires its own split'],
            nominal_coordinate_registration='fixed NVlabs scene coordinates; shared dcxx spec, not a new registration accuracy test')
        atomic_json(output/'data_audit.json', audit)
        names=['channels.npy','selection.npz','all_metadata.npz','split_manifest.json','source_scan.json','data_audit.json']
        outputs={name:describe(output/name) for name in names}
        manifest=dict(contract, contract_hash=contract_hash, status='complete', outputs=outputs,
            selection_sha256=outputs['selection.npz']['sha256'], split_sha256=outputs['split_manifest.json']['sha256'],
            split_protocol_sha256=split['sha256'], legacy_dc01=legacy,
            created_utc=dt.datetime.now(dt.timezone.utc).isoformat(), elapsed_s=time.monotonic()-started,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            runtime=dict(python=sys.version.split()[0], numpy=np.__version__, device='CPU'))
        atomic_json(output/'preparation_manifest.json', manifest)
        atomic_json(output/'run_state.json', dict(contract_hash=contract_hash, status='complete'))
        print(json.dumps(dict(dataset=dataset, status='complete', directory=PATHS.relative(output),
              points=len(selected), elapsed_s=manifest['elapsed_s'])), flush=True)
    except BaseException:
        (output/'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        atomic_json(output/'run_state.json', dict(contract_hash=contract_hash, status='failed'))
        raise

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['dc01','dc02','both'], default='both')
    parser.add_argument('--resume', action='store_true')
    args=parser.parse_args()
    for dataset in ('dc01','dc02') if args.dataset=='both' else (args.dataset,): prepare(dataset,args.resume)

if __name__ == '__main__': main()
