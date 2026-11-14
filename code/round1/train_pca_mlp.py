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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


MH, MV, MP = 16, 8, 2
NH, NV, NP = 1, 2, 2
BS_POS = np.array([50.0, 0.0, 25.0], dtype=np.float64)


def load_arrays(data_dir):
    train_pos = np.load(os.path.join(data_dir, 'Round1_Train_Pos.npy'))
    test_pos = np.load(os.path.join(data_dir, 'Round1_Test_Pos.npy'))
    train_h = np.load(os.path.join(data_dir, 'Round1_Train_Channel.npy'), mmap_mode='r')
    return train_pos, test_pos, train_h


def transform_channel(h, mode):
    if mode == 'freq':
        return h.astype(np.complex64, copy=False)
    if mode == 'delay':
        return np.fft.ifft(h, axis=-1).astype(np.complex64)
    if mode == 'angle_delay':
        cir = np.fft.ifft(h, axis=-1)
        arr = cir.reshape(h.shape[0], MH, MV, MP, NH, NV, NP, h.shape[-1])
        return np.fft.fftn(arr, axes=(1, 2, 4, 5)).astype(np.complex64)
    raise ValueError(mode)


def inverse_transform_channel(x, mode):
    if mode == 'freq':
        return x.astype(np.complex64, copy=False)
    if mode == 'delay':
        return np.fft.fft(x, axis=-1).astype(np.complex64)
    if mode == 'angle_delay':
        arr = np.fft.ifftn(x, axes=(1, 2, 4, 5))
        arr = arr.reshape(x.shape[0], 256, 4, 192)
        return np.fft.fft(arr, axis=-1).astype(np.complex64)
    raise ValueError(mode)


def flatten_complex(x):
    return np.stack([x.real, x.imag], axis=-1).reshape(x.shape[0], -1).astype(np.float32)


def unflatten_complex(z, shape):
    z = z.reshape((z.shape[0],) + shape[1:] + (2,))
    return (z[..., 0] + 1j * z[..., 1]).astype(np.complex64)


def build_split(n, val_count, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    val_idx = np.sort(perm[:val_count])
    train_idx = np.sort(perm[val_count:])
    return train_idx, val_idx


def fit_feature_context(pos):
    xy = pos[:, :2]
    rel = pos - BS_POS[None, :]
    dist = np.linalg.norm(rel, axis=1, keepdims=True)
    return {
        'xy_min': xy.min(axis=0),
        'xy_max': xy.max(axis=0),
        'dist_min': float(dist.min()),
        'dist_max': float(dist.max()),
    }


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
    base = [
        pos,
        rel,
        dist,
        np.log1p(dist),
        1.0 / np.maximum(dist, 1e-3),
        np.sin(az), np.cos(az), np.sin(el), np.cos(el),
        coords,
    ]
    if fourier_bands > 0:
        freqs = (2.0 ** np.arange(fourier_bands, dtype=np.float64)) * np.pi
        for f in freqs:
            base.append(np.sin(f * coords))
            base.append(np.cos(f * coords))
    return np.concatenate(base, axis=1).astype(np.float32)


def materialize_transformed(train_h, indices, mode, batch):
    rows = []
    shape = None
    for start in range(0, len(indices), batch):
        idx = indices[start:start + batch]
        x = transform_channel(np.asarray(train_h[idx]), mode)
        if shape is None:
            shape = x[:1].shape
        rows.append(flatten_complex(x))
        print(f'materialized {start + len(idx)}/{len(indices)}', flush=True)
    return np.vstack(rows), shape


def nmse_batch(pred, truth):
    diff = pred - truth
    num = np.sum(np.abs(diff) ** 2, axis=(1, 2, 3), dtype=np.float64)
    den = np.sum(np.abs(truth) ** 2, axis=(1, 2, 3), dtype=np.float64)
    return num / np.maximum(den, 1e-30)


def cosine_rows(a, b):
    a = a.reshape(a.shape[0], -1).astype(np.float64, copy=False)
    b = b.reshape(b.shape[0], -1).astype(np.float64, copy=False)
    num = np.sum(a * b, axis=1)
    den = np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1))
    return num / np.maximum(den, 1e-30)


def pdp_batch(h):
    ht = np.fft.ifft(h, axis=-1)
    return np.sum(np.abs(ht) ** 2, axis=(1, 2))


def pas_batch(h):
    arr = h.reshape(h.shape[0], MH, MV, MP, NH, NV, NP, h.shape[-1])
    ang = np.fft.fftn(arr, axes=(1, 2, 4, 5))
    return np.sum(np.abs(ang) ** 2, axis=(3, 6, 7)).reshape(h.shape[0], -1)


def score_summary(nmse, pdp_cos, pas_cos):
    score = 0.4 * pas_cos + 0.4 * pdp_cos + 0.2 / (1.0 + nmse)
    return {
        'score_mean': float(np.mean(score)),
        'pas_mean': float(np.mean(pas_cos)),
        'pdp_mean': float(np.mean(pdp_cos)),
        'nmse_mean': float(np.mean(nmse)),
        'nmse_median': float(np.median(nmse)),
        'nmse_q90': float(np.quantile(nmse, 0.9)),
        'nmse_power_weighted': None,
    }


def evaluate_reconstruction(train_h, indices, coef_pred, pca, x_scaler, scales, shape, mode, batch):
    nmse_all = []
    pdp_all = []
    pas_all = []
    power_all = []
    for start in range(0, len(indices), batch):
        end = min(start + batch, len(indices))
        idx = indices[start:end]
        z = pca.inverse_transform(coef_pred[start:end]).astype(np.float32)
        z = x_scaler.inverse_transform(z).astype(np.float32)
        z *= scales[start:end, None]
        xt = unflatten_complex(z, shape)
        pred = inverse_transform_channel(xt, mode)
        truth = np.asarray(train_h[idx])
        nm = nmse_batch(pred, truth)
        nmse_all.append(nm)
        pdp_all.append(cosine_rows(pdp_batch(pred), pdp_batch(truth)))
        pas_all.append(cosine_rows(pas_batch(pred), pas_batch(truth)))
        power_all.append(np.sum(np.abs(truth) ** 2, axis=(1, 2, 3), dtype=np.float64))
        print(f'evaluated {end}/{len(indices)}', flush=True)
    nmse = np.concatenate(nmse_all)
    pdp = np.concatenate(pdp_all)
    pas = np.concatenate(pas_all)
    power = np.concatenate(power_all)
    out = score_summary(nmse, pdp, pas)
    out['nmse_power_weighted'] = float(np.sum(nmse * power) / np.sum(power))
    return out


class ResidualMLP(nn.Module):
    def __init__(self, in_dim, out_dim, width, depth, dropout):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.SiLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_model(x_train, y_train, x_val, y_val, args, outdir):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    model = ResidualMLP(x_train.shape[1], y_train.shape[1], args.width, args.depth, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1), eta_min=args.lr * 0.03)
    loss_fn = nn.MSELoss()
    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=args.train_batch, shuffle=True, drop_last=False, num_workers=0)
    xv = torch.from_numpy(x_val).to(device)
    yv = torch.from_numpy(y_val).to(device)
    best = {'epoch': -1, 'val_mse': float('inf')}
    best_path = outdir / 'best_mlp.pt'
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
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
            pred = model(xv)
            val_mse = float(loss_fn(pred, yv).item())
        train_mse = total / max(count, 1)
        if val_mse < best['val_mse']:
            best = {'epoch': epoch, 'val_mse': val_mse, 'train_mse': train_mse}
            torch.save({'model': model.state_dict(), 'args': vars(args), 'in_dim': x_train.shape[1], 'out_dim': y_train.shape[1]}, best_path)
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f'epoch {epoch:04d} train_mse={train_mse:.6g} val_mse={val_mse:.6g} best={best["val_mse"]:.6g}@{best["epoch"]}', flush=True)
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    with torch.no_grad():
        val_pred = model(xv).cpu().numpy().astype(np.float32)
    return model, val_pred, best


def predict_coefficients(model, x, batch, device):
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch):
            xb = torch.from_numpy(x[start:start + batch]).to(device)
            preds.append(model(xb).cpu().numpy().astype(np.float32))
    return np.vstack(preds)


def generate_test_output(output_path, model, x_test, pca, x_scaler, scale_pred, shape, mode, batch, device):
    out = np.lib.format.open_memmap(output_path, mode='w+', dtype=np.complex64, shape=(len(x_test), 256, 4, 192))
    coef = predict_coefficients(model, x_test, batch, device)
    for start in range(0, len(x_test), batch):
        end = min(start + batch, len(x_test))
        z = pca.inverse_transform(coef[start:end]).astype(np.float32)
        z = x_scaler.inverse_transform(z).astype(np.float32)
        z *= scale_pred[start:end, None]
        xt = unflatten_complex(z, shape)
        out[start:end] = inverse_transform_channel(xt, mode)
        print(f'wrote test {end}/{len(x_test)}', flush=True)
    out.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=str(PATHS.round1))
    parser.add_argument('--outdir', default=str(PATHS.artifacts / 'training_runs/pca_mlp_runs'))
    parser.add_argument('--mode', default='angle_delay', choices=['freq', 'delay', 'angle_delay'])
    parser.add_argument('--components', type=int, default=512)
    parser.add_argument('--val-count', type=int, default=400)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--train-batch', type=int, default=256)
    parser.add_argument('--seed', type=int, default=20260703)
    parser.add_argument('--fourier-bands', type=int, default=8)
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--depth', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.02)
    parser.add_argument('--epochs', type=int, default=1200)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--log-every', type=int, default=25)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--generate-test', action='store_true')
    parser.add_argument('--output', default=str(PATHS.temporary / "Round1_Test_Channel.npy"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run_name = f'{args.mode}_c{args.components}_v{args.val_count}_fb{args.fourier_bands}_{time.strftime("%Y%m%d_%H%M%S")}'
    outdir = Path(args.outdir) / run_name
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    train_pos, test_pos, train_h = load_arrays(args.data)
    train_idx, val_idx = build_split(len(train_pos), args.val_count, args.seed)
    feature_ctx = fit_feature_context(train_pos[train_idx])
    feat_train_raw = position_features(train_pos[train_idx], feature_ctx, args.fourier_bands)
    feat_val_raw = position_features(train_pos[val_idx], feature_ctx, args.fourier_bands)
    feat_test_raw = position_features(test_pos, feature_ctx, args.fourier_bands)
    feat_scaler = StandardScaler()
    feat_train = feat_scaler.fit_transform(feat_train_raw).astype(np.float32)
    feat_val = feat_scaler.transform(feat_val_raw).astype(np.float32)
    feat_test = feat_scaler.transform(feat_test_raw).astype(np.float32)

    all_idx = np.concatenate([train_idx, val_idx])
    x_all, shape = materialize_transformed(train_h, all_idx, args.mode, args.batch)
    n_train = len(train_idx)
    x_train_raw = x_all[:n_train]
    x_val_raw = x_all[n_train:]
    train_scales = np.sqrt(np.maximum(np.mean(x_train_raw * x_train_raw, axis=1, dtype=np.float64), 1e-30)).astype(np.float32)
    val_scales = np.sqrt(np.maximum(np.mean(x_val_raw * x_val_raw, axis=1, dtype=np.float64), 1e-30)).astype(np.float32)
    x_train_norm = x_train_raw / train_scales[:, None]
    x_val_norm = x_val_raw / val_scales[:, None]
    x_scaler = StandardScaler(with_mean=True, with_std=True)
    x_train_scaled = x_scaler.fit_transform(x_train_norm).astype(np.float32)
    x_val_scaled = x_scaler.transform(x_val_norm).astype(np.float32)

    n_components = min(args.components, x_train_scaled.shape[0], x_train_scaled.shape[1])
    print(f'fitting PCA n_components={n_components} train_input={x_train_scaled.shape}', flush=True)
    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=args.seed)
    coef_train_raw = pca.fit_transform(x_train_scaled).astype(np.float32)
    coef_val_raw = pca.transform(x_val_scaled).astype(np.float32)
    coef_scaler = StandardScaler()
    coef_train = coef_scaler.fit_transform(coef_train_raw).astype(np.float32)
    coef_val = coef_scaler.transform(coef_val_raw).astype(np.float32)
    scale_val = val_scales

    model, val_pred_scaled, best = train_model(feat_train, coef_train, feat_val, coef_val, args, outdir)
    val_pred = coef_scaler.inverse_transform(val_pred_scaled).astype(np.float32)
    oracle_val = coef_scaler.inverse_transform(coef_val).astype(np.float32)

    print('evaluating PCA oracle on validation split', flush=True)
    oracle_metrics = evaluate_reconstruction(train_h, val_idx, oracle_val, pca, x_scaler, scale_val, shape, args.mode, args.batch)
    print('evaluating MLP prediction on validation split', flush=True)
    val_metrics = evaluate_reconstruction(train_h, val_idx, val_pred, pca, x_scaler, scale_val, shape, args.mode, args.batch)
    result = {
        'run_name': run_name,
        'split': {'train_count': int(len(train_idx)), 'val_count': int(len(val_idx)), 'seed': int(args.seed)},
        'pca': {
            'mode': args.mode,
            'components': int(n_components),
            'explained_variance_ratio_sum': float(pca.explained_variance_ratio_.sum()),
        },
        'best_training': best,
        'oracle_val_metrics': oracle_metrics,
        'mlp_val_metrics': val_metrics,
    }
    with open(outdir / 'result.json', 'w') as f:
        json.dump(result, f, indent=2)
    joblib.dump({
        'pca': pca,
        'x_scaler': x_scaler,
        'coef_scaler': coef_scaler,
        'feat_scaler': feat_scaler,
        'feature_ctx': feature_ctx,
        'shape': shape,
        'args': vars(args),
        'train_idx': train_idx,
        'val_idx': val_idx,
    }, outdir / 'preprocess.joblib')
    print(json.dumps(result, indent=2), flush=True)

    if args.generate_test:
        scale_model = ResidualMLP(feat_train.shape[1], 1, max(128, args.width // 4), max(2, args.depth - 2), args.dropout)
        # For test generation use nearest train transformed scale. The scale is weakly varying and this avoids
        # leaking validation labels into a second target model before we know the coefficient model is useful.
        from scipy.spatial import cKDTree
        tree = cKDTree(train_pos[all_idx, :2])
        _, nn_idx = tree.query(test_pos[:, :2], k=1)
        known_scales = np.concatenate([train_scales, val_scales])
        test_scales = known_scales[nn_idx].astype(np.float32)
        device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
        generate_test_output(args.output, model, feat_test, pca, x_scaler, test_scales, shape, args.mode, args.batch, device)


if __name__ == '__main__':
    main()
