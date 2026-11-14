"""Independent FP64 audit of saved final channels and the spatial protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree


def normalized(x, axes):
    return x / np.maximum(x.sum(axis=axes, keepdims=True), 1e-30)


def cosine(x, y, axes):
    return (x*y).sum(axis=axes) / np.maximum(
        np.sqrt((x*x).sum(axis=axes)*(y*y).sum(axis=axes)), 1e-30)


def main():
    parser = argparse.ArgumentParser()
    for name in ['prepared', 'baseline', 'spectral', 'state', 'output']:
        parser.add_argument(f'--{name}-dir', type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    expected = json.loads((args.output_dir/'evaluation.json').read_text())
    split = json.loads((args.baseline_dir/'split_manifest.json').read_text())
    metadata = np.load(args.prepared_dir/'selection.npz')
    truth = np.load(args.prepared_dir/'channels.npy', mmap_mode='r')
    all_rows = []
    pas_scores, pdp_scores, beam_losses = [], [], []
    energy = error = 0.
    minimum_guard = float('inf')
    power_relative_error = 0.
    hashes = {}
    for fold in range(5):
        ids = {k:np.asarray(split['splits'][fold][k]) for k in ['fit','val','test']}
        assert not set(ids['fit']) & (set(ids['val']) | set(ids['test']))
        assert not set(ids['val']) & set(ids['test'])
        hold = np.concatenate([ids['val'], ids['test']])
        distance = cKDTree(metadata['position'][hold,:2]).query(metadata['position'][ids['fit'],:2])[0]
        minimum_guard = min(minimum_guard, float(distance.min()))
        assert distance.min() >= .30 - 1e-7
        all_rows.extend(ids['test'].tolist())
        folder = args.state_dir/f'fold{fold}'
        state = json.loads((folder/'state_report.json').read_text())
        pred_path = folder/'test_prediction.npy'
        prediction = np.load(pred_path, mmap_mode='r')
        assert prediction.dtype == np.complex64
        assert prediction.shape == (len(ids['test']),2,8,4,256)
        assert np.isfinite(prediction).all()
        source = np.load(folder/'test_metrics.npz')['source_id']
        assert np.array_equal(source, metadata['source_id'][ids['test']])
        foundation = np.load(args.spectral_dir/f'fold{fold}'/'test_base.npy', mmap_mode='r')
        if state['selected_epoch'] == 0:
            assert np.array_equal(prediction, foundation)
        hashes[f'fold{fold}/test_prediction.npy'] = hashlib.sha256(pred_path.read_bytes()).hexdigest()
        for first in range(0,len(prediction),32):
            h = np.asarray(prediction[first:first+32], dtype=np.complex128)
            t = np.asarray(truth[ids['test'][first:first+32]], dtype=np.complex128)
            fa = np.abs(np.fft.fft2(h,axes=(2,3)))**2
            ft = np.abs(np.fft.fft2(t,axes=(2,3)))**2
            pas_scores.append(cosine(normalized(fa,(2,3)),normalized(ft,(2,3)),(2,3)).mean(-1))
            d = normalized(np.abs(np.fft.ifft(h,axis=-1))**2,(-1,))
            td = normalized(np.abs(np.fft.ifft(t,axis=-1))**2,(-1,))
            pdp_scores.append(cosine(d,td,(-1,)).mean((2,3)))
            chosen = fa.mean(-1).reshape(len(h),2,32).argmax(-1)
            actual = ft.mean(-1).reshape(len(h),2,32)
            selected = np.take_along_axis(actual,chosen[...,None],axis=-1)[...,0]
            beam_losses.append(-10*np.log10(np.maximum(selected/actual.max(-1),1e-30)))
            error += float(np.sum(np.abs(h-t)**2))
            energy += float(np.sum(np.abs(t)**2))
            b = np.asarray(foundation[first:first+32],dtype=np.complex128)
            base_power = np.sum(np.abs(b)**2,axis=(2,3,4))
            pred_power = np.sum(np.abs(h)**2,axis=(2,3,4))
            power_relative_error = max(power_relative_error,float(np.max(
                np.abs(pred_power-base_power)/np.maximum(base_power,1e-30))))
    assert np.array_equal(np.sort(all_rows),np.arange(len(truth)))
    recomputed = dict(pas=float(np.concatenate(pas_scores).mean()),
                      pdp=float(np.concatenate(pdp_scores).mean()),
                      beam_loss_db=float(np.concatenate(beam_losses).mean()),nmse=error/energy)
    recomputed['structural_score'] = .5*(recomputed['pas']+recomputed['pdp'])
    reference = expected['metrics']['final_measured_v16']
    differences = {k:recomputed[k]-reference[k] for k in recomputed}
    assert max(abs(d) for d in differences.values()) < 3e-6, differences
    assert power_relative_error < 3e-5
    result = dict(status='passed',independent_backend='NumPy complex128 / FP64',
                  positions=len(all_rows),minimum_fit_to_held_distance_m=minimum_guard,
                  maximum_panel_power_relative_change=power_relative_error,
                  metrics=recomputed,difference_from_torch_report=differences,
                  prediction_sha256=hashes,elapsed_s=time.time()-started)
    (args.output_dir/'independent_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
