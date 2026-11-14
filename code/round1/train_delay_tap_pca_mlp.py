#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


MH, MV, MP = 16, 8, 2
NH, NV, NP = 1, 2, 2
BS_POS = np.array([50.0, 0.0, 25.0], dtype=np.float64)


def load_arrays(data_dir):
    return (
        np.load(os.path.join(data_dir, 'Round1_Train_Pos.npy')),
        np.load(os.path.join(data_dir, 'Round1_Test_Pos.npy')),
        np.load(os.path.join(data_dir, 'Round1_Train_Channel.npy'), mmap_mode='r'),
    )


def build_split(n, val_count, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return np.sort(perm[val_count:]), np.sort(perm[:val_count])


def delay_transform(h):
    return np.fft.ifft(h, axis=-1).astype(np.complex64)


def fit_feature_context(pos):
    xy = pos[:, :2]
    rel = pos - BS_POS[None, :]
    dist = np.linalg.norm(rel, axis=1, keepdims=True)
    return {'xy_min': xy.min(axis=0), 'xy_max': xy.max(axis=0), 'dist_min': float(dist.min()), 'dist_max': float(dist.max())}


def position_features(pos, ctx, fourier_bands):
    pos = np.asarray(pos, dtype=np.float64)
    rel = pos - BS_POS[None, :]
    dist = np.linalg.norm(rel, axis=1, keepdims=True)
    horiz = np.linalg.norm(rel[:, :2], axis=1, keepdims=True)
    az = np.arctan2(rel[:, 1:2], rel[:, 0:1])
    el = np.arctan2(rel[:, 2:3], np.maximum(horiz, 1e-9))
    xy_span = np.maximum(ctx['xy_max'] - ctx['xy_min'], 1e-9)
    xy_unit = 2.0 * (pos[:, :2] - ctx['xy_min'][None, :]) / xy_span[None, :] - 1.0
    dist_unit = 2.0 * (dist - ctx['dist_min']) / max(ctx['dist_max'] - ctx['dist_min'], 1e-9) - 1.0
    coords = np.concatenate([xy_unit, dist_unit], axis=1)
    parts = [pos, rel, dist, np.log1p(dist), 1.0 / np.maximum(dist, 1e-3), np.sin(az), np.cos(az), np.sin(el), np.cos(el), coords]
    if fourier_bands > 0:
        for f in (2.0 ** np.arange(fourier_bands, dtype=np.float64)) * np.pi:
            parts.append(np.sin(f * coords))
            parts.append(np.cos(f * coords))
    return np.concatenate(parts, axis=1).astype(np.float32)


def select_taps(train_h, train_idx, tap_count, batch):
    energy = np.zeros(train_h.shape[-1], dtype=np.float64)
    for start in range(0, len(train_idx), batch):
        idx = train_idx[start:start + batch]
        cir = delay_transform(np.asarray(train_h[idx]))
        energy += np.sum(np.abs(cir) ** 2, axis=(0, 1, 2), dtype=np.float64)
        print(f'tap energy {start + len(idx)}/{len(train_idx)}', flush=True)
    taps = np.sort(np.argpartition(energy, -tap_count)[-tap_count:])
    return taps, float(energy[taps].sum() / max(energy.sum(), 1e-30))


def materialize_taps(train_h, indices, taps, batch):
    rows = []
    scales = []
    for start in range(0, len(indices), batch):
        idx = indices[start:start + batch]
        cir = delay_transform(np.asarray(train_h[idx]))[..., taps]
        scale = np.sqrt(np.maximum(np.mean(np.abs(cir) ** 2, axis=(1, 2, 3), dtype=np.float64), 1e-30)).astype(np.float32)
        flat = np.stack([cir.real, cir.imag], axis=-1).reshape(len(idx), -1).astype(np.float32)
        rows.append(flat / scale[:, None])
        scales.append(np.log(scale).reshape(-1, 1).astype(np.float32))
        print(f'tap matrix {start + len(idx)}/{len(indices)}', flush=True)
    return np.vstack(rows), np.vstack(scales)


def reconstruct(flat_norm, log_scale, taps):
    flat = flat_norm * np.exp(log_scale.astype(np.float32)).reshape(-1, 1)
    selected = flat.reshape(flat.shape[0], 256, 4, len(taps), 2)
    selected = (selected[..., 0] + 1j * selected[..., 1]).astype(np.complex64)
    cir = np.zeros((flat.shape[0], 256, 4, 192), dtype=np.complex64)
    cir[..., taps] = selected
    return np.fft.fft(cir, axis=-1).astype(np.complex64)


def nmse_batch(pred, truth):
    diff = pred - truth
    return np.sum(np.abs(diff) ** 2, axis=(1, 2, 3), dtype=np.float64) / np.maximum(np.sum(np.abs(truth) ** 2, axis=(1, 2, 3), dtype=np.float64), 1e-30)


def cosine_rows(a, b):
    a = a.reshape(a.shape[0], -1).astype(np.float64, copy=False)
    b = b.reshape(b.shape[0], -1).astype(np.float64, copy=False)
    return np.sum(a * b, axis=1) / np.maximum(np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1)), 1e-30)


def pdp_batch(h):
    return np.sum(np.abs(np.fft.ifft(h, axis=-1)) ** 2, axis=(1, 2))


def pas_batch(h):
    arr = h.reshape(h.shape[0], MH, MV, MP, NH, NV, NP, h.shape[-1])
    ang = np.fft.fftn(arr, axes=(1, 2, 4, 5))
    return np.sum(np.abs(ang) ** 2, axis=(3, 6, 7)).reshape(h.shape[0], -1)


def summarize(nmse, pdp, pas, power):
    score = 0.4 * pas + 0.4 * pdp + 0.2 / (1.0 + nmse)
    return {
        'score_mean': float(np.mean(score)),
        'pas_mean': float(np.mean(pas)),
        'pdp_mean': float(np.mean(pdp)),
        'nmse_mean': float(np.mean(nmse)),
        'nmse_median': float(np.median(nmse)),
        'nmse_q90': float(np.quantile(nmse, 0.9)),
        'nmse_power_weighted': float(np.sum(nmse * power) / np.sum(power)),
    }


def evaluate(train_h, indices, flat_norm, log_scale, taps, batch):
    nmse_all, pdp_all, pas_all, power_all = [], [], [], []
    for start in range(0, len(indices), batch):
        end = min(start + batch, len(indices))
        pred = reconstruct(flat_norm[start:end], log_scale[start:end], taps)
        truth = np.asarray(train_h[indices[start:end]])
        nmse_all.append(nmse_batch(pred, truth))
        pdp_all.append(cosine_rows(pdp_batch(pred), pdp_batch(truth)))
        pas_all.append(cosine_rows(pas_batch(pred), pas_batch(truth)))
        power_all.append(np.sum(np.abs(truth) ** 2, axis=(1, 2, 3), dtype=np.float64))
        print(f'evaluated {end}/{len(indices)}', flush=True)
    return summarize(np.concatenate(nmse_all), np.concatenate(pdp_all), np.concatenate(pas_all), np.concatenate(power_all))


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, width, depth, dropout):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.SiLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_model(x_train, y_train, x_val, y_val, args, outdir):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    model = MLP(x_train.shape[1], y_train.shape[1], args.width, args.depth, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1), eta_min=args.lr * 0.03)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)), batch_size=args.train_batch, shuffle=True)
    xv = torch.from_numpy(x_val).to(device)
    yv = torch.from_numpy(y_val).to(device)
    best = {'epoch': -1, 'val_mse': float('inf')}
    best_path = outdir / 'best_tap_pca_mlp.pt'
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.item()) * xb.shape[0]
            count += xb.shape[0]
        sched.step()
        model.eval()
        with torch.no_grad():
            val_mse = float(loss_fn(model(xv), yv).item())
        train_mse = total / max(count, 1)
        if val_mse < best['val_mse']:
            best = {'epoch': epoch, 'val_mse': val_mse, 'train_mse': train_mse}
            torch.save({'model': model.state_dict(), 'args': vars(args)}, best_path)
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f'epoch {epoch:04d} train_mse={train_mse:.6g} val_mse={val_mse:.6g} best={best["val_mse"]:.6g}@{best["epoch"]}', flush=True)
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, best, device


def predict(model, x, batch, device):
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            out.append(model(torch.from_numpy(x[start:start + batch]).to(device)).cpu().numpy().astype(np.float32))
    return np.vstack(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=str(PATHS.round1))
    parser.add_argument('--outdir', default=str(PATHS.artifacts / 'training_runs/delay_tap_pca_runs'))
    parser.add_argument('--tap-count', type=int, default=8)
    parser.add_argument('--components', type=int, default=512)
    parser.add_argument('--val-count', type=int, default=400)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--train-batch', type=int, default=256)
    parser.add_argument('--seed', type=int, default=20260703)
    parser.add_argument('--fourier-bands', type=int, default=10)
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--depth', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--log-every', type=int, default=25)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    run_name = f'taps{args.tap_count}_c{args.components}_v{args.val_count}_fb{args.fourier_bands}_{time.strftime("%Y%m%d_%H%M%S")}'
    outdir = Path(args.outdir) / run_name
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    train_pos, test_pos, train_h = load_arrays(args.data)
    train_idx, val_idx = build_split(len(train_pos), args.val_count, args.seed)
    ctx = fit_feature_context(train_pos[train_idx])
    feat_scaler = StandardScaler()
    feat_train = feat_scaler.fit_transform(position_features(train_pos[train_idx], ctx, args.fourier_bands)).astype(np.float32)
    feat_val = feat_scaler.transform(position_features(train_pos[val_idx], ctx, args.fourier_bands)).astype(np.float32)

    taps, tap_frac = select_taps(train_h, train_idx, args.tap_count, args.batch)
    print('selected taps', taps.tolist(), flush=True)
    train_flat, train_log_scale = materialize_taps(train_h, train_idx, taps, args.batch)
    val_flat, val_log_scale = materialize_taps(train_h, val_idx, taps, args.batch)
    flat_scaler = StandardScaler()
    train_flat_scaled = flat_scaler.fit_transform(train_flat).astype(np.float32)
    val_flat_scaled = flat_scaler.transform(val_flat).astype(np.float32)
    n_components = min(args.components, train_flat_scaled.shape[0], train_flat_scaled.shape[1])
    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=args.seed)
    train_coef = pca.fit_transform(train_flat_scaled).astype(np.float32)
    val_coef = pca.transform(val_flat_scaled).astype(np.float32)
    coef_scaler = StandardScaler()
    train_y = np.concatenate([coef_scaler.fit_transform(train_coef), train_log_scale], axis=1).astype(np.float32)
    val_y = np.concatenate([coef_scaler.transform(val_coef), val_log_scale], axis=1).astype(np.float32)

    model, best, device = train_model(feat_train, train_y, feat_val, val_y, args, outdir)
    pred_y = predict(model, feat_val, args.train_batch, device)
    pred_coef = coef_scaler.inverse_transform(pred_y[:, :n_components]).astype(np.float32)
    pred_flat = flat_scaler.inverse_transform(pca.inverse_transform(pred_coef)).astype(np.float32)
    pred_log_scale = pred_y[:, n_components:n_components + 1]
    oracle_flat = flat_scaler.inverse_transform(pca.inverse_transform(val_coef)).astype(np.float32)
    oracle_metrics = evaluate(train_h, val_idx, oracle_flat, val_log_scale, taps, args.batch)
    mlp_metrics = evaluate(train_h, val_idx, pred_flat, pred_log_scale, taps, args.batch)
    result = {
        'run_name': run_name,
        'split': {'train_count': int(len(train_idx)), 'val_count': int(len(val_idx)), 'seed': int(args.seed)},
        'taps': taps.tolist(),
        'tap_energy_fraction_train': tap_frac,
        'pca': {'components': int(n_components), 'explained_variance_ratio_sum': float(pca.explained_variance_ratio_.sum())},
        'best_training': best,
        'oracle_val_metrics': oracle_metrics,
        'mlp_val_metrics': mlp_metrics,
    }
    with open(outdir / 'result.json', 'w') as f:
        json.dump(result, f, indent=2)
    joblib.dump({'feat_scaler': feat_scaler, 'flat_scaler': flat_scaler, 'coef_scaler': coef_scaler, 'pca': pca, 'feature_ctx': ctx, 'taps': taps, 'args': vars(args)}, outdir / 'preprocess.joblib')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
