#!/usr/bin/env python3
"""Train/generate a Geo-ADTS style big residual model.

This is the deliberately larger-model branch inspired by the Geo-ADTS proposal.
It keeps the AD math that has already been validated for Round1:

    M flatten order: (P,H,V) = (2,16,8)
    AD tensor layout: (P,H,V,N,tau)

The model does not predict the full dense channel directly.  Instead it learns a
large sparse complex residual in angle-delay space over the global top-K
components.  The training loss is the differentiable score proxy:

    0.4 * PAS cosine + 0.4 * PDP cosine + 0.2 / (1 + AD-NMSE)

Inputs are position features, Sionna mesh scene features and exact path
candidates.  The path candidates are encoded as Transformer tokens.
"""

from __future__ import annotations
from project_paths import PATHS

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ASSETS = PATHS.assets
DATA = PATHS.round1
BS_POS = np.array([50.0, 0.0, 25.0], dtype=np.float32)
MP, MH, MV, N_ANT, S_SC = 2, 16, 8, 4, 192
COMP_COUNT = MH * MV * S_SC
TINY = 1e-30


def beta_subcarriers(beta=140.3, gamma=0.10):
    sub = np.linspace(-0.5, 0.5, S_SC, dtype=np.float32)
    return (beta + gamma * sub).astype(np.float32)


def ad_transform(h: np.ndarray) -> np.ndarray:
    cir = np.fft.ifft(h, axis=-1)
    arr = cir.reshape(h.shape[0], MP, MH, MV, N_ANT, S_SC)
    return np.fft.fft2(arr, axes=(2, 3)).astype(np.complex64)


def inverse_ad(ad: np.ndarray) -> np.ndarray:
    arr = np.fft.ifft2(ad, axes=(2, 3))
    cir = arr.reshape(ad.shape[0], MP * MH * MV, N_ANT, S_SC)
    return np.fft.fft(cir, axis=-1).astype(np.complex64)


def flatten_ad_np(ad: np.ndarray) -> np.ndarray:
    # (B,MP,H,V,N,S) -> (B,MP,N,HVS)
    return ad.transpose(0, 1, 4, 2, 3, 5).reshape(ad.shape[0], MP, N_ANT, COMP_COUNT)


def unflatten_ad_np(flat: np.ndarray) -> np.ndarray:
    # (B,MP,N,HVS) -> (B,MP,H,V,N,S)
    return flat.reshape(flat.shape[0], MP, N_ANT, MH, MV, S_SC).transpose(0, 1, 3, 4, 2, 5)


def flat_to_hvs(ids: np.ndarray):
    h = ids // (MV * S_SC)
    rem = ids % (MV * S_SC)
    v = rem // S_SC
    s = rem % S_SC
    return h.astype(np.int64), v.astype(np.int64), s.astype(np.int64)


def make_query_features(pos: np.ndarray, scene_feats: np.ndarray, blocked: np.ndarray) -> np.ndarray:
    pos = pos.astype(np.float32)
    rel = pos - BS_POS[None, :]
    dist = np.linalg.norm(rel, axis=1).astype(np.float32)
    az = np.arctan2(rel[:, 1], rel[:, 0]).astype(np.float32)
    el = np.arcsin(np.clip(rel[:, 2] / np.maximum(dist, 1e-6), -1.0, 1.0)).astype(np.float32)
    x = (pos[:, 0] - 150.0) / 120.0
    y = (pos[:, 1] + 75.0) / 120.0
    z = (pos[:, 2] - 1.5) / 10.0
    ff = [x, y, z, dist / 250.0, np.log1p(dist) / 6.0, np.sin(az), np.cos(az), np.sin(el), np.cos(el)]
    for f in (1.0, 2.0, 4.0, 8.0, 16.0):
        for v in (x, y):
            ff.append(np.sin(np.pi * f * v))
            ff.append(np.cos(np.pi * f * v))
    blocked = blocked.astype(np.float32)
    regime = np.stack([
        blocked / 80.0,
        np.log1p(np.maximum(blocked, 0.0)) / 5.0,
        (blocked <= 0).astype(np.float32),
        ((blocked > 0) & (blocked <= 25)).astype(np.float32),
        ((blocked > 25) & (blocked <= 100)).astype(np.float32),
        (blocked > 100).astype(np.float32),
    ], axis=1)
    base = np.stack(ff, axis=1).astype(np.float32)
    return np.concatenate([base, regime, scene_feats.astype(np.float32)], axis=1)


def make_path_features(paths: np.lib.npyio.NpzFile, prefix: str):
    valid = paths[f"{prefix}_valid"].astype(np.float32)
    ptype = paths[f"{prefix}_path_type"].astype(np.int64)
    mat = paths[f"{prefix}_material_id"].astype(np.int64)
    delay = np.nan_to_num(paths[f"{prefix}_delay_bin"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    score = np.nan_to_num(paths[f"{prefix}_score"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    excess = np.nan_to_num(paths[f"{prefix}_excess_length"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    edge = np.nan_to_num(paths[f"{prefix}_edge_margin"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    inc = np.nan_to_num(paths[f"{prefix}_incidence_cos"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    length = np.nan_to_num(paths[f"{prefix}_path_length"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    los_hit = paths[f"{prefix}_los_hit_count"].astype(np.float32)
    los_clear = paths[f"{prefix}_los_clear"].astype(np.float32)
    first_mat = paths[f"{prefix}_los_first_material_id"].astype(np.float32)

    numeric = np.stack([
        valid,
        np.where(valid > 0, delay / 16.0, 0.0),
        np.where(valid > 0, score, 0.0),
        np.where(valid > 0, excess / 80.0, 0.0),
        np.where(valid > 0, edge / 30.0, 0.0),
        np.where(valid > 0, inc, 0.0),
        np.where(valid > 0, length / 450.0, 0.0),
    ], axis=-1).astype(np.float32)
    global_path = np.stack([
        los_hit / 15.0,
        los_clear,
        np.maximum(first_mat, 0) / 4.0,
    ], axis=1).astype(np.float32)
    # Reserve 0 for invalid; shift real ids by +1.
    type_id = np.where(valid > 0, np.maximum(ptype, 0) + 1, 0).astype(np.int64)
    mat_id = np.where(valid > 0, np.maximum(mat, 0) + 1, 0).astype(np.int64)
    type_id = np.clip(type_id, 0, 4)
    mat_id = np.clip(mat_id, 0, 8)
    return numeric, type_id, mat_id, global_path


def standardize(train: np.ndarray, test: np.ndarray):
    mean = train.mean(axis=0, keepdims=True).astype(np.float32)
    std = train.std(axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return (train - mean) / std, (test - mean) / std, mean, std


class ADResidualDataset(Dataset):
    def __init__(self, indices, query, path_num, path_type, path_mat, base_ad_path, res_ad_path):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.query = query.astype(np.float32)
        self.path_num = path_num.astype(np.float32)
        self.path_type = path_type.astype(np.int64)
        self.path_mat = path_mat.astype(np.int64)
        self.base = np.load(base_ad_path, mmap_mode="r")
        self.res = np.load(res_ad_path, mmap_mode="r")

    def __len__(self):
        return int(self.indices.size)

    def __getitem__(self, item):
        idx = int(self.indices[item])
        base_ad = np.asarray(self.base[idx:idx + 1], dtype=np.complex64)
        res_ad = np.asarray(self.res[idx:idx + 1], dtype=np.complex64)
        base_flat = flatten_ad_np(base_ad)[0]
        res_flat = flatten_ad_np(res_ad)[0]
        return {
            "idx": idx,
            "query": torch.from_numpy(self.query[idx]),
            "path_num": torch.from_numpy(self.path_num[idx]),
            "path_type": torch.from_numpy(self.path_type[idx]),
            "path_mat": torch.from_numpy(self.path_mat[idx]),
            "base_real": torch.from_numpy(base_flat.real.astype(np.float32)),
            "base_imag": torch.from_numpy(base_flat.imag.astype(np.float32)),
            "res_real": torch.from_numpy(res_flat.real.astype(np.float32)),
            "res_imag": torch.from_numpy(res_flat.imag.astype(np.float32)),
        }


class GeoADTSBig(nn.Module):
    def __init__(self, query_dim, path_num_dim, component_count, d_model=256, layers=6, heads=8, dropout=0.05):
        super().__init__()
        self.component_count = int(component_count)
        self.query_proj = nn.Sequential(
            nn.Linear(query_dim, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
        self.type_emb = nn.Embedding(5, 32)
        self.mat_emb = nn.Embedding(9, 32)
        self.path_proj = nn.Sequential(
            nn.Linear(path_num_dim + 64, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, component_count * MP * N_ANT * 2),
        )
        final = self.head[-1]
        nn.init.normal_(final.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(final.bias)
        self.log_scale = nn.Parameter(torch.tensor(-4.0, dtype=torch.float32))

    def forward(self, query, path_num, path_type, path_mat):
        q = self.query_proj(query).unsqueeze(1)
        te = self.type_emb(path_type.clamp(0, 4))
        me = self.mat_emb(path_mat.clamp(0, 8))
        p = self.path_proj(torch.cat([path_num, te, me], dim=-1))
        tok = torch.cat([q, p], dim=1)
        enc = self.encoder(tok)
        pooled = enc[:, 0] + 0.25 * enc[:, 1:].mean(dim=1)
        out = self.head(pooled)
        out = out.view(query.shape[0], self.component_count, MP, N_ANT, 2)
        return out * torch.exp(self.log_scale)


def scatter_components(coeff, comp_ids, device):
    b = coeff.shape[0]
    flat = torch.zeros((b, MP, N_ANT, COMP_COUNT), dtype=torch.complex64, device=device)
    c = torch.complex(coeff[..., 0], coeff[..., 1]).permute(0, 2, 3, 1).contiguous()
    ids = comp_ids.view(1, 1, 1, -1).expand(b, MP, N_ANT, -1)
    flat.scatter_(3, ids, c)
    return flat


def score_loss(pred_flat, true_flat):
    pred_power = torch.abs(pred_flat) ** 2
    true_power = torch.abs(true_flat) ** 2
    pred_coll = pred_power.sum(dim=(1, 2)).view(-1, MH, MV, S_SC)
    true_coll = true_power.sum(dim=(1, 2)).view(-1, MH, MV, S_SC)
    pred_beam = pred_coll.sum(dim=-1).reshape(pred_coll.shape[0], -1)
    true_beam = true_coll.sum(dim=-1).reshape(true_coll.shape[0], -1)
    pred_delay = pred_coll.sum(dim=(1, 2))
    true_delay = true_coll.sum(dim=(1, 2))
    c1 = F.cosine_similarity(pred_beam + 1e-20, true_beam + 1e-20, dim=-1).mean()
    c2 = F.cosine_similarity(pred_delay + 1e-20, true_delay + 1e-20, dim=-1).mean()
    c3 = (torch.abs(pred_flat - true_flat) ** 2).sum() / (
        torch.abs(true_flat) ** 2).sum().clamp_min(1e-20)
    score = 0.4 * c1 + 0.4 * c2 + 0.2 / (1.0 + c3)
    return -score, c1.detach(), c2.detach(), c3.detach()


def coeff_mse(pred_coeff, true_res_flat, comp_ids):
    target = true_res_flat[:, :, :, comp_ids]  # B,MP,N,K
    pred = torch.complex(pred_coeff[..., 0], pred_coeff[..., 1]).permute(0, 2, 3, 1)
    return (torch.abs(pred - target) ** 2).mean()


def prepare_features():
    pos_train = np.load(DATA / "Round1_Train_Pos.npy")
    pos_test = np.load(DATA / "Round1_Test_Pos.npy")
    scene = np.load(ASSETS / "sionna_etoile_features.npz")
    train_scene = scene["train_features"].astype(np.float32)
    test_scene = scene["test_features"].astype(np.float32)
    geo = np.load(ASSETS / "height_geometry_cache.npz")
    train_blocked = geo["train_blocked"]
    test_blocked = geo["test_blocked"]
    paths = np.load(ASSETS / "sionna_path_candidates.npz")
    q_train = make_query_features(pos_train, train_scene, train_blocked)
    q_test = make_query_features(pos_test, test_scene, test_blocked)
    pnum_train, ptype_train, pmat_train, gpath_train = make_path_features(paths, "train")
    pnum_test, ptype_test, pmat_test, gpath_test = make_path_features(paths, "test")
    q_train = np.concatenate([q_train, gpath_train], axis=1)
    q_test = np.concatenate([q_test, gpath_test], axis=1)
    q_train, q_test, q_mean, q_std = standardize(q_train, q_test)
    p_flat_tr = pnum_train.reshape(-1, pnum_train.shape[-1])
    p_flat_te = pnum_test.reshape(-1, pnum_test.shape[-1])
    p_flat_tr, p_flat_te, p_mean, p_std = standardize(p_flat_tr, p_flat_te)
    pnum_train = p_flat_tr.reshape(pnum_train.shape)
    pnum_test = p_flat_te.reshape(pnum_test.shape)
    return {
        "q_train": q_train,
        "q_test": q_test,
        "pnum_train": pnum_train,
        "pnum_test": pnum_test,
        "ptype_train": ptype_train,
        "ptype_test": ptype_test,
        "pmat_train": pmat_train,
        "pmat_test": pmat_test,
        "q_mean": q_mean,
        "q_std": q_std,
        "p_mean": p_mean,
        "p_std": p_std,
    }


def select_global_components(count):
    p = np.load(ASSETS / "train_v36_loo_residual_derot_ad_power.npy", mmap_mode="r")
    score = np.asarray(p).sum(axis=0).reshape(-1)
    ids = np.argpartition(score, -count)[-count:]
    ids = ids[np.argsort(score[ids])[::-1]]
    return ids.astype(np.int64)


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    feats = prepare_features()
    comp_ids_np = select_global_components(args.component_count)
    comp_ids = torch.tensor(comp_ids_np, dtype=torch.long, device=device)

    rng = np.random.default_rng(args.seed)
    all_idx = rng.permutation(2000)
    val_idx = np.sort(all_idx[:args.val_count])
    train_idx = np.sort(all_idx[args.val_count:])
    ds_train = ADResidualDataset(
        train_idx,
        feats["q_train"],
        feats["pnum_train"],
        feats["ptype_train"],
        feats["pmat_train"],
        ASSETS / "train_v36_loo_base_derot_ad.npy",
        ASSETS / "train_v36_loo_residual_derot_ad.npy",
    )
    ds_val = ADResidualDataset(
        val_idx,
        feats["q_train"],
        feats["pnum_train"],
        feats["ptype_train"],
        feats["pmat_train"],
        ASSETS / "train_v36_loo_base_derot_ad.npy",
        ASSETS / "train_v36_loo_residual_derot_ad.npy",
    )
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = GeoADTSBig(
        query_dim=feats["q_train"].shape[1],
        path_num_dim=feats["pnum_train"].shape[-1],
        component_count=args.component_count,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    best = {"select": -1e9, "score": -1e9, "epoch": -1}
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in dl_train:
            query = batch["query"].to(device, non_blocking=True)
            path_num = batch["path_num"].to(device, non_blocking=True)
            path_type = batch["path_type"].to(device, non_blocking=True)
            path_mat = batch["path_mat"].to(device, non_blocking=True)
            base = torch.complex(
                batch["base_real"].to(device, non_blocking=True),
                batch["base_imag"].to(device, non_blocking=True),
            )
            res = torch.complex(
                batch["res_real"].to(device, non_blocking=True),
                batch["res_imag"].to(device, non_blocking=True),
            )
            coeff = model(query, path_num, path_type, path_mat)
            sparse = scatter_components(coeff, comp_ids, device)
            pred = base + sparse
            truth = base + res
            loss_score, c1, c2, c3 = score_loss(pred, truth)
            loss_coeff = coeff_mse(coeff, res, comp_ids)
            loss = loss_score + args.coeff_weight * loss_coeff
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += float(loss.detach().cpu())
        sched.step()

        val = evaluate_model(model, dl_val, comp_ids, device, args.coeff_weight)
        item = {"epoch": epoch, "train_loss": train_loss / max(len(dl_train), 1), **val}
        history.append(item)
        print(json.dumps(item), flush=True)
        select = -val["loss"]
        if select > best["select"]:
            best = {"select": select, "score": val["score"], "epoch": epoch}
            save_checkpoint(args, model, feats, comp_ids_np, history, best)

    save_checkpoint(args, model, feats, comp_ids_np, history, best, final=True)
    return best


@torch.no_grad()
def evaluate_model(model, loader, comp_ids, device, coeff_weight):
    model.eval()
    total = {"loss": 0.0, "score": 0.0, "c1": 0.0, "c2": 0.0, "nmse": 0.0, "n": 0}
    for batch in loader:
        query = batch["query"].to(device, non_blocking=True)
        path_num = batch["path_num"].to(device, non_blocking=True)
        path_type = batch["path_type"].to(device, non_blocking=True)
        path_mat = batch["path_mat"].to(device, non_blocking=True)
        base = torch.complex(batch["base_real"].to(device), batch["base_imag"].to(device))
        res = torch.complex(batch["res_real"].to(device), batch["res_imag"].to(device))
        coeff = model(query, path_num, path_type, path_mat)
        sparse = scatter_components(coeff, comp_ids, device)
        pred = base + sparse
        truth = base + res
        loss_score, c1, c2, c3 = score_loss(pred, truth)
        loss_coeff = coeff_mse(coeff, res, comp_ids)
        b = query.shape[0]
        score = float((-loss_score).cpu())
        total["loss"] += float((loss_score + coeff_weight * loss_coeff).cpu()) * b
        total["score"] += score * b
        total["c1"] += float(c1.cpu()) * b
        total["c2"] += float(c2.cpu()) * b
        total["nmse"] += float(c3.cpu()) * b
        total["n"] += b
    n = max(total.pop("n"), 1)
    return {k: v / n for k, v in total.items()}


def save_checkpoint(args, model, feats, comp_ids, history, best, final=False):
    ckpt_path = Path(args.checkpoint_final if final else args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "args": vars(args),
        "query_dim": feats["q_train"].shape[1],
        "path_num_dim": feats["pnum_train"].shape[-1],
        "component_ids": comp_ids,
        "history": history,
        "best": best,
    }, ckpt_path)


@torch.no_grad()
def generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    feats = prepare_features()
    comp_ids_np = ckpt["component_ids"].astype(np.int64)
    comp_ids = torch.tensor(comp_ids_np, dtype=torch.long, device=device)
    model = GeoADTSBig(
        query_dim=ckpt["query_dim"],
        path_num_dim=ckpt["path_num_dim"],
        component_count=len(comp_ids_np),
        d_model=ckpt["args"]["d_model"],
        layers=ckpt["args"]["layers"],
        heads=ckpt["args"]["heads"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    pos_test = np.load(DATA / "Round1_Test_Pos.npy")
    d_te = np.linalg.norm(pos_test - BS_POS[None, :], axis=1).astype(np.float32)
    beta_s = beta_subcarriers()
    base_freq = np.load(args.generate_base, mmap_mode="r")
    submit_base = np.load(args.submit_base, mmap_mode="r") if args.submit_base else None
    out = np.lib.format.open_memmap(args.output, mode="w+", dtype=np.complex64, shape=(500, 256, 4, 192))

    q = feats["q_test"]
    pn = feats["pnum_test"]
    pt = feats["ptype_test"]
    pm = feats["pmat_test"]
    stats = {"model_delta_power": 0.0, "base_power": 0.0, "out_power": 0.0}
    for s in range(0, 500, args.gen_batch):
        e = min(s + args.gen_batch, 500)
        h = np.asarray(base_freq[s:e], dtype=np.complex64)
        h_derot = h * np.exp(1j * d_te[s:e, None, None, None] * beta_s).astype(np.complex64)
        base_ad = ad_transform(h_derot)
        base_flat = flatten_ad_np(base_ad)
        query = torch.from_numpy(q[s:e]).to(device)
        path_num = torch.from_numpy(pn[s:e]).to(device)
        path_type = torch.from_numpy(pt[s:e]).to(device)
        path_mat = torch.from_numpy(pm[s:e]).to(device)
        coeff = model(query, path_num, path_type, path_mat)
        sparse = scatter_components(coeff, comp_ids, device).cpu().numpy()
        pred_flat = base_flat + args.model_scale * sparse
        pred_ad = unflatten_ad_np(pred_flat)
        pred_freq = inverse_ad(pred_ad)
        pred_freq = pred_freq * np.exp(-1j * d_te[s:e, None, None, None] * beta_s).astype(np.complex64)
        if submit_base is not None and args.submit_blend < 1.0:
            sb = np.asarray(submit_base[s:e], dtype=np.complex64)
            pred_freq = ((1.0 - args.submit_blend) * sb + args.submit_blend * pred_freq).astype(np.complex64)
        out[s:e] = pred_freq.astype(np.complex64)
        out.flush()
        stats["model_delta_power"] += float(np.sum(np.abs(sparse) ** 2, dtype=np.float64))
        stats["base_power"] += float(np.sum(np.abs(h) ** 2, dtype=np.float64))
        stats["out_power"] += float(np.sum(np.abs(pred_freq) ** 2, dtype=np.float64))
        print(f"generated {e}/500", flush=True)

    report = {
        "method": "geoadts_big_sparse_residual",
        "checkpoint": args.checkpoint,
        "component_count": int(len(comp_ids_np)),
        "generate_base": args.generate_base,
        "submit_base": args.submit_base,
        "submit_blend": float(args.submit_blend),
        "model_scale": float(args.model_scale),
        "stats": {
            **stats,
            "out_to_base_power": stats["out_power"] / max(stats["base_power"], 1e-30),
        },
        "output": args.output,
    }
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "generate", "train_generate"], default="train_generate")
    p.add_argument("--seed", type=int, default=20260707)
    p.add_argument("--val-count", type=int, default=300)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--component-count", type=int, default=1024)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--coeff-weight", type=float, default=5.0)
    p.add_argument("--checkpoint", default=str(ASSETS / "geoadts_big_residual_best.pt"))
    p.add_argument("--checkpoint-final", default=str(ASSETS / "geoadts_big_residual_final.pt"))
    p.add_argument("--generate-base", default=str(PATHS.artifacts / "legacy_inputs/v36.npy"))
    p.add_argument("--submit-base", default=str(PATHS.cache_dir(83) / "v83_Round1_Test_Channel.npy"))
    p.add_argument("--submit-base-npy", default="")
    p.add_argument("--submit-blend", type=float, default=0.35)
    p.add_argument("--model-scale", type=float, default=1.0)
    p.add_argument("--gen-batch", type=int, default=16)
    p.add_argument("--output", default=str(PATHS.cache_dir(85) / "v85_Round1_Test_Channel.npy"))
    p.add_argument("--report", default=str(ASSETS / "geoadts_big_residual_generate.json"))
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    if args.submit_base.endswith(".zip"):
        # Caller should pass an extracted npy for generation.  Keep a hard error
        # to avoid silently treating a zip as a numpy array.
        if args.submit_base_npy:
            args.submit_base = args.submit_base_npy
        else:
            raise ValueError("--submit-base points to a zip; pass --submit-base-npy with an extracted .npy")
    return args


def main():
    args = parse_args()
    if args.mode in ("train", "train_generate"):
        train(args)
    if args.mode in ("generate", "train_generate"):
        generate(args)


if __name__ == "__main__":
    main()
