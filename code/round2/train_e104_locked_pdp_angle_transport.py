#!/usr/bin/env python3
"""Train a strict locked-PDP soft angle transport model on E85 candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import time

import numpy as np
import torch
from torch import nn

import analyze_e100_locked_energy_angle_correction as e100
import analyze_e102_locked_pdp_candidate_reassignment_oracle as e102
import analyze_e86_surface_endpoint_v11_blend as e86
import train_surface_interaction_energy_model as e85
from analyze_v4_exact_joint_projection import pas_distribution, pdp_distribution
from project_paths import PATHS
from run_v9_joint_sionna_migration import point_metrics


VERSION = "Round2-E104-locked-pdp-angle-transport"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "locked_pdp_angle_transport"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_locked_pdp_angle_transport"
REPORT = ANALYSIS_DIR / "locked_pdp_angle_transport.json"
OOF_PAS = ARTIFACT_DIR / "strict_oof_locked_pdp_transport_pas_f16.npy"
METRICS = ARTIFACT_DIR / "strict_oof_locked_pdp_transport_metrics.npz"
FULL_ENDPOINT = (
    e85.ARTIFACT_DIR / "oof_geometry_sequence_path_endpoint.npz"
)
STRICT_ENDPOINT = e100.OLD_ENDPOINT
VALID_MASK = e85.PATH_CACHE / "mask.npy"
TINY = 1e-30


@dataclass(frozen=True)
class Config:
    hidden: int = 96
    heads: int = 4
    layers: int = 1
    feedforward: int = 192
    rank: int = 32
    dropout: float = 0.05
    max_transport: float = 0.50
    epochs: int = 32
    validation_interval: int = 4
    batch_size: int = 64
    learning_rate: float = 4e-4
    weight_decay: float = 2e-4
    gate_penalty: float = 0.001
    random_state: int = 20260808104


class EndpointData:
    def __init__(self, path: object, strict: bool) -> None:
        data = np.load(path, allow_pickle=False)
        self.mass = data["mass"].astype(np.float32)
        self.coordinate = data["coordinate"].astype(np.float32)
        self.port = data["port"].astype(np.float32)
        self.pas = data["pas"].astype(np.float64)
        self.pdp = data["pdp"].astype(np.float64)
        self.strict = strict
        if strict:
            self.query_ids = data["query_ids"].astype(np.int64)
            self.lookup = np.full(4000, -1, dtype=np.int32)
            self.lookup[self.query_ids] = np.arange(len(self.query_ids), dtype=np.int32)
        else:
            self.query_ids = np.arange(len(self.mass), dtype=np.int64)
            self.lookup = np.arange(len(self.mass), dtype=np.int32)

    def local_rows(self, global_rows: np.ndarray) -> np.ndarray:
        rows = self.lookup[global_rows]
        if np.any(rows < 0):
            raise ValueError("endpoint does not contain all requested rows")
        return rows.astype(np.int64)


class LockedPdpTransport(nn.Module):
    def __init__(self, input_dim: int, config: Config) -> None:
        super().__init__()
        self.config = config
        self.input = nn.Sequential(
            nn.Linear(input_dim, config.hidden),
            nn.GELU(),
            nn.LayerNorm(config.hidden),
            nn.Linear(config.hidden, config.hidden),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden,
            nhead=config.heads,
            dim_feedforward=config.feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(layer, num_layers=config.layers)
        self.query = nn.Linear(config.hidden, config.rank, bias=False)
        self.key = nn.Linear(config.hidden, config.rank, bias=False)
        self.gate = nn.Linear(config.hidden, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.5)

    def forward(
        self,
        features: torch.Tensor,
        coordinate: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.context(self.input(features))
        query = self.query(hidden)
        key = self.key(hidden)
        score = torch.einsum("bir,bjr->bij", query, key) / math.sqrt(
            self.config.rank
        )
        dh = e102.periodic_delta(
            coordinate[:, None, :, 0] - coordinate[:, :, None, 0], 16.0
        )
        dv = e102.periodic_delta(
            coordinate[:, None, :, 1] - coordinate[:, :, None, 1], 8.0
        )
        score = score - 0.05 * ((dh / 2.0).square() + (dv / 1.5).square())
        score = score.masked_fill(~valid[:, None, :], -1e4)
        transport = torch.softmax(score, dim=2)
        alpha = self.config.max_transport * torch.sigmoid(self.gate(hidden)[..., 0])
        identity = torch.eye(
            coordinate.shape[1], dtype=transport.dtype, device=transport.device
        )[None]
        assignment = (1.0 - alpha[..., None]) * identity + alpha[..., None] * transport
        return assignment, alpha


def normalization(store: e85.DataStore, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return store.normalization(rows)


def endpoint_features(
    store: e85.DataStore,
    endpoint: EndpointData,
    global_rows: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    valid_all: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    local = endpoint.local_rows(global_rows)
    mass_np = endpoint.mass[local]
    coordinate_np = endpoint.coordinate[local]
    port_np = endpoint.port[local]
    base_np = (np.asarray(store.base[global_rows], dtype=np.float32) - mean) / std
    angle = np.empty((len(local), e85.PATHS_PER_POINT, 6), dtype=np.float32)
    for axis, period in enumerate((16.0, 8.0, 192.0)):
        phase = 2.0 * np.pi * coordinate_np[..., axis] / period
        angle[..., 2 * axis] = np.sin(phase)
        angle[..., 2 * axis + 1] = np.cos(phase)
    site = np.zeros((len(local), e85.PATHS_PER_POINT, 2), dtype=np.float32)
    site_id = np.asarray(store.site[global_rows], dtype=np.int64)
    site[..., 0] = (site_id[:, None] == 0).astype(np.float32)
    site[..., 1] = (site_id[:, None] == 1).astype(np.float32)
    valid_np = np.asarray(valid_all[global_rows], dtype=bool)
    feature_np = np.concatenate(
        (
            base_np,
            np.log(np.maximum(mass_np, 1e-12))[..., None],
            angle,
            port_np.reshape(len(local), e85.PATHS_PER_POINT, 8),
            valid_np[..., None].astype(np.float32),
            site,
        ),
        axis=2,
    )
    return (
        torch.as_tensor(feature_np, device=device),
        torch.as_tensor(mass_np, device=device),
        torch.as_tensor(coordinate_np, device=device),
        torch.as_tensor(port_np, device=device),
        torch.as_tensor(valid_np, device=device),
        torch.as_tensor(
            np.asarray(store.target_pas[global_rows], dtype=np.float32), device=device
        ),
    )


def predict_batch(
    model: LockedPdpTransport,
    batch: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features, mass, coordinate, port, valid, target = batch
    assignment, alpha = model(features, coordinate, valid)
    energy = mass[..., None, None] * port
    kernel = e102.candidate_kernels(coordinate)
    pas = e102.assignment_pas(energy, assignment, kernel)
    score = e102.pas_score(pas, target)
    return pas, score, alpha


@torch.inference_mode()
def evaluate(
    model: LockedPdpTransport | None,
    store: e85.DataStore,
    endpoint: EndpointData,
    rows: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    valid: np.ndarray,
    device: torch.device,
    batch_size: int,
    collect: bool,
) -> dict[str, object]:
    if model is not None:
        model.eval()
    scores = []
    gates = []
    spectra = []
    for start in range(0, len(rows), batch_size):
        local_rows = rows[start : start + batch_size]
        batch = endpoint_features(
            store, endpoint, local_rows, mean, std, valid, device
        )
        if model is None:
            _, mass, coordinate, port, _, target = batch
            output = {
                "mass": mass,
                "coordinate": coordinate,
                "port": port,
            }
            pas, _ = e85.render_spectra(output)
            score = e102.pas_score(pas, target)
            alpha = torch.zeros_like(mass)
        else:
            pas, score, alpha = predict_batch(model, batch)
        scores.append(score.cpu().numpy())
        gates.append(alpha.mean(dim=1).cpu().numpy())
        if collect:
            spectra.append(pas.cpu().numpy().astype(np.float32))
    return {
        "score": np.concatenate(scores),
        "mean": float(np.mean(np.concatenate(scores))),
        "gate": np.concatenate(gates),
        "pas": np.concatenate(spectra) if collect else None,
    }


def train_epoch(
    model: LockedPdpTransport,
    optimizer: torch.optim.Optimizer,
    store: e85.DataStore,
    endpoint: EndpointData,
    rows: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    valid: np.ndarray,
    config: Config,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, float]:
    model.train()
    order = rng.permutation(rows)
    score_total = 0.0
    gate_total = 0.0
    count = 0
    for start in range(0, len(order), config.batch_size):
        local_rows = order[start : start + config.batch_size]
        batch = endpoint_features(
            store, endpoint, local_rows, mean, std, valid, device
        )
        _, score, alpha = predict_batch(model, batch)
        loss = 1.0 - score.mean() + config.gate_penalty * alpha.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise FloatingPointError("non-finite E104 gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        score_total += float(score.detach().sum().item())
        gate_total += float(alpha.detach().mean(dim=1).sum().item())
        count += len(local_rows)
    return {
        "pas": score_total / max(count, 1),
        "mean_gate": gate_total / max(count, 1),
    }


def train_model(
    store: e85.DataStore,
    endpoint: EndpointData,
    fit_rows: np.ndarray,
    validation_rows: np.ndarray | None,
    validation_endpoint: EndpointData | None,
    mean: np.ndarray,
    std: np.ndarray,
    valid: np.ndarray,
    config: Config,
    epochs: int,
    seed: int,
    device: torch.device,
) -> tuple[LockedPdpTransport, list[dict[str, object]], int]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    input_dim = store.base.shape[-1] + 1 + 6 + 8 + 1 + 2
    model = LockedPdpTransport(input_dim, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    rng = np.random.default_rng(seed)
    history = []
    best_epoch = 0
    best_value = -math.inf
    if validation_rows is not None and validation_endpoint is not None:
        baseline = evaluate(
            None,
            store,
            validation_endpoint,
            validation_rows,
            mean,
            std,
            valid,
            device,
            config.batch_size,
            False,
        )
        best_value = float(baseline["mean"])
        history.append(
            {
                "epoch": 0,
                "validation_pas": best_value,
                "validation_gate": 0.0,
            }
        )
    for epoch in range(1, epochs + 1):
        train = train_epoch(
            model,
            optimizer,
            store,
            endpoint,
            fit_rows,
            mean,
            std,
            valid,
            config,
            device,
            rng,
        )
        row: dict[str, object] = {"epoch": epoch, "train": train}
        if (
            validation_rows is not None
            and validation_endpoint is not None
            and epoch % config.validation_interval == 0
        ):
            validation = evaluate(
                model,
                store,
                validation_endpoint,
                validation_rows,
                mean,
                std,
                valid,
                device,
                config.batch_size,
                False,
            )
            row["validation_pas"] = validation["mean"]
            row["validation_gate"] = float(np.mean(validation["gate"]))
            if validation["mean"] > best_value:
                best_value = float(validation["mean"])
                best_epoch = epoch
        history.append(row)
        if epoch % config.validation_interval == 0 or epoch == epochs:
            print(
                f"E104 epoch={epoch}/{epochs} train={train['pas']:.6f} "
                f"gate={train['mean_gate']:.4f} "
                f"validation={row.get('validation_pas', float('nan')):.6f}",
                flush=True,
            )
    return model, history, best_epoch


def fold_summary(values: np.ndarray, fold_id: np.ndarray) -> list[dict[str, object]]:
    return [
        {
            "fold": outer,
            "count": int(np.sum(fold_id == outer)),
            "mean": float(np.mean(values[fold_id == outer])),
        }
        for outer in range(5)
    ]


def main() -> None:
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = Config()
    required = (
        FULL_ENDPOINT,
        STRICT_ENDPOINT,
        VALID_MASK,
        e86.V11_CHANNEL,
        e86.TRAIN_CHANNEL,
        e100.V12_CHANNEL,
    )
    missing = [PATHS.relative(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")

    store = e85.DataStore()
    full = EndpointData(FULL_ENDPOINT, strict=False)
    strict = EndpointData(STRICT_ENDPOINT, strict=True)
    query_ids = store.strict_ids
    fold_id = store.strict_fold
    if not np.array_equal(strict.query_ids, query_ids):
        raise ValueError("strict endpoint order mismatch")
    valid = np.asarray(np.load(VALID_MASK, mmap_mode="r"), dtype=bool)
    all_rows = np.arange(len(store.base), dtype=np.int64)
    strict_pas = np.zeros(
        (len(query_ids), e85.MP, e85.MH, e85.MV, e85.N_UE), dtype=np.float32
    )
    strict_score = np.zeros(len(query_ids), dtype=np.float64)
    strict_gate = np.zeros(len(query_ids), dtype=np.float64)
    training = []

    for outer in range(5):
        validation_fold = (outer + 1) % 5
        held = query_ids[fold_id == outer]
        validation = query_ids[fold_id == validation_fold]
        held_mask = np.zeros(len(store.base), dtype=bool)
        validation_mask = np.zeros(len(store.base), dtype=bool)
        held_mask[held] = True
        validation_mask[validation] = True
        screen_fit = all_rows[~held_mask & ~validation_mask]
        refit = all_rows[~held_mask]
        mean_screen, std_screen = normalization(store, screen_fit)
        screen_model, screen_history, best_epoch = train_model(
            store,
            full,
            screen_fit,
            validation,
            strict,
            mean_screen,
            std_screen,
            valid,
            config,
            config.epochs,
            config.random_state + 10000 * outer,
            device,
        )
        del screen_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        rows = np.flatnonzero(fold_id == outer)
        if best_epoch == 0:
            prediction = evaluate(
                None,
                store,
                strict,
                held,
                mean_screen,
                std_screen,
                valid,
                device,
                config.batch_size,
                True,
            )
            refit_history: list[dict[str, object]] = []
            checkpoint = None
        else:
            mean_refit, std_refit = normalization(store, refit)
            model, refit_history, _ = train_model(
                store,
                full,
                refit,
                None,
                None,
                mean_refit,
                std_refit,
                valid,
                config,
                best_epoch,
                config.random_state + 100000 + 10000 * outer,
                device,
            )
            prediction = evaluate(
                model,
                store,
                strict,
                held,
                mean_refit,
                std_refit,
                valid,
                device,
                config.batch_size,
                True,
            )
            checkpoint_path = ARTIFACT_DIR / f"strict_fold{outer}_transport.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": asdict(config),
                    "feature_mean": mean_refit,
                    "feature_std": std_refit,
                    "best_epoch": best_epoch,
                    "outer": outer,
                },
                checkpoint_path,
            )
            checkpoint = PATHS.relative(checkpoint_path)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        strict_pas[rows] = prediction["pas"]
        strict_score[rows] = prediction["score"]
        strict_gate[rows] = prediction["gate"]
        training.append(
            {
                "outer": outer,
                "validation_fold": validation_fold,
                "screen_fit_count": int(len(screen_fit)),
                "refit_count": int(len(refit)),
                "held_count": int(len(held)),
                "best_epoch": best_epoch,
                "screen_history": screen_history,
                "refit_history": refit_history,
                "checkpoint": checkpoint,
                "held_pas": float(prediction["mean"]),
                "held_gate": float(np.mean(prediction["gate"])),
            }
        )
        print(
            f"E104 outer={outer} best_epoch={best_epoch} "
            f"held_pas={prediction['mean']:.6f} "
            f"gate={np.mean(prediction['gate']):.4f}",
            flush=True,
        )

    np.save(OOF_PAS, strict_pas.astype(np.float16))
    baseline_score = strict.pas.astype(np.float64)
    endpoint_delta = strict_score - baseline_score

    old_output = {
        "mass": torch.as_tensor(strict.mass, device=device),
        "coordinate": torch.as_tensor(strict.coordinate, device=device),
        "port": torch.as_tensor(strict.port, device=device),
    }
    with torch.inference_mode():
        _, endpoint_pdp_small = e85.render_spectra(old_output)
        endpoint_pas = (
            torch.as_tensor(strict_pas, device=device)[..., None]
            .expand(-1, -1, -1, -1, -1, 192)
        )
        endpoint_pdp = (
            endpoint_pdp_small[:, :, None, None, :, :]
            .expand(-1, -1, 16, 8, -1, -1)
            .reshape(-1, 256, 4, 192)
        )
        base = torch.as_tensor(
            np.array(np.load(e86.V11_CHANNEL, mmap_mode="r"), copy=True), device=device
        )
        truth = torch.as_tensor(
            np.asarray(
                np.load(e86.TRAIN_CHANNEL, mmap_mode="r")[query_ids],
                dtype=np.complex64,
            ),
            device=device,
        )
        base_pas = pas_distribution(base)
        base_pdp = pdp_distribution(base)
        truth_pas = pas_distribution(truth)
        truth_pdp = pdp_distribution(truth)
        spec = e86.Candidate(
            endpoint="locked_pdp_transport",
            mode="sqrt",
            pas_beta=0.20,
            pdp_beta=0.20,
            order="pas_pdp",
            cycles=4,
        )
        channel = e86.make_candidate(
            base, base_pas, base_pdp, endpoint_pas, endpoint_pdp, spec
        )
        selected_metrics = point_metrics(channel, truth, truth_pas, truth_pdp)
        v12 = torch.as_tensor(
            np.array(np.load(e100.V12_CHANNEL, mmap_mode="r"), copy=True), device=device
        )
        v12_metrics = point_metrics(v12, truth, truth_pas, truth_pdp)
    complete = e100.metric_comparison(selected_metrics, v12_metrics, fold_id)
    decision = {
        "minimum_endpoint_gain": 0.002,
        "minimum_complete_gain": 0.001,
        "minimum_positive_folds": 4,
        "minimum_worst_fold_delta": -0.001,
        "endpoint_gain": float(np.mean(endpoint_delta)),
        "endpoint_positive_folds": int(
            sum(np.mean(endpoint_delta[fold_id == outer]) > 0.0 for outer in range(5))
        ),
        "complete_gain": complete["delta"],
        "complete_positive_folds": complete["positive_folds"],
        "worst_fold_delta": complete["worst_fold_delta"],
    }
    decision["passed"] = bool(
        decision["endpoint_gain"] >= decision["minimum_endpoint_gain"]
        and decision["endpoint_positive_folds"] >= decision["minimum_positive_folds"]
        and decision["complete_gain"] >= decision["minimum_complete_gain"]
        and decision["complete_positive_folds"]
        >= decision["minimum_positive_folds"]
        and decision["worst_fold_delta"] >= decision["minimum_worst_fold_delta"]
    )
    np.savez_compressed(
        METRICS,
        query_ids=query_ids.astype(np.int32),
        fold_id=fold_id,
        baseline_pas=baseline_score.astype(np.float32),
        transport_pas=strict_score.astype(np.float32),
        transport_gate=strict_gate.astype(np.float32),
        endpoint_delta=endpoint_delta.astype(np.float32),
        point_pas=selected_metrics[0].astype(np.float32),
        point_pdp=selected_metrics[1].astype(np.float32),
        point_error=selected_metrics[2].astype(np.float32),
        point_energy=selected_metrics[3].astype(np.float32),
    )
    report = {
        "experiment": VERSION,
        "status": "strict_complete",
        "primary_hypothesis": (
            "A learned row-stochastic transport within the old E85 candidate set can "
            "improve PAS while preserving each path's mass, port vector, and delay."
        ),
        "validation": (
            "Canonical 500 contiguous holes. A different strict hole selects epoch; "
            "the held hole is excluded from transport training and selection."
        ),
        "config": asdict(config),
        "training": training,
        "endpoint": {
            "baseline_pas": float(np.mean(baseline_score)),
            "transport_pas": float(np.mean(strict_score)),
            "gain": float(np.mean(endpoint_delta)),
            "baseline_fold": fold_summary(baseline_score, fold_id),
            "transport_fold": fold_summary(strict_score, fold_id),
            "delta_fold": fold_summary(endpoint_delta, fold_id),
            "mean_gate": float(np.mean(strict_gate)),
            "gate_fold": fold_summary(strict_gate, fold_id),
        },
        "fixed_policy": {
            **spec.__dict__,
            "mass_locked": True,
            "port_locked": True,
            "delay_locked": True,
            "pdp_locked_by_construction": True,
            "selection": "strict alternate-hole epoch selection; no action scan",
        },
        "complete_score": {"versus_v12": complete},
        "decision": {**decision, "generate_v13": decision["passed"]},
        "outputs": {
            "oof_pas": PATHS.relative(OOF_PAS),
            "metrics": PATHS.relative(METRICS),
            "pas_shape": list(strict_pas.shape),
            "pas_dtype": "float16",
            "finite": bool(
                np.isfinite(strict_pas).all()
                and np.isfinite(strict_score).all()
                and np.isfinite(strict_gate).all()
            ),
        },
        "known_limitations": [
            "Transport training uses the existing all-4000 E85 OOF endpoint cache; "
            "held labels are excluded from the transport model but may have weakly "
            "influenced some upstream OOF endpoint checkpoints.",
            "Passing the local gate would require a separately nested endpoint replay "
            "before V13 packaging.",
        ],
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
