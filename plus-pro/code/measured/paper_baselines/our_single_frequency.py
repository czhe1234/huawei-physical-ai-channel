"""Single-frequency retraining of M3's measured-energy/angular-state method."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import time
import traceback

import numpy as np
import torch

from study_common import ARTIFACTS, CODE, PATHS, REPORTS, atomic_json, atomic_numpy, freeze_identity, save_prediction, sha256, study_data, training_identity
from dc01_angular_state import AngularState, angular_render, covariance_error
from dc01_factorized_field import MeasuredField
from measured_runtime import cosine, normalize, pas, safe_sqrt, unit_complex
from single_frequency_metrics import evaluate


def utility(metrics):
    return metrics["array_pas_cosine"]-.05*metrics["covariance_error"]-.02*metrics["dft_beam_loss_db"]


def covariance_numpy(prediction, truth):
    p = prediction[..., 0].reshape(len(prediction), 2, 32).astype(np.complex128)
    t = truth[..., 0].reshape(len(truth), 2, 32).astype(np.complex128)
    rp = p[..., :, None] * p.conj()[..., None, :]
    rt = t[..., :, None] * t.conj()[..., None, :]
    rp /= np.maximum((np.abs(p)**2).sum(-1)[..., None, None], 1e-30)
    rt /= np.maximum((np.abs(t)**2).sum(-1)[..., None, None], 1e-30)
    return float(np.mean((np.abs(rp-rt)**2).sum((-1, -2))/np.maximum((np.abs(rt)**2).sum((-1, -2)), 1e-30)))


def field_evaluation(prediction, truth):
    result, _ = evaluate(prediction, truth)
    result["covariance_error"] = covariance_numpy(prediction, truth)
    return result


def render_field(model, rows, spectral, powers, exclude=False):
    values = []
    for first in range(0, len(rows), 32):
        value, _ = model.prediction(rows[first:first+32], spectral, powers, "joint", exclude=exclude)
        values.append(value.cpu().numpy())
    return np.concatenate(values)


class SingleFrequencyState(AngularState):
    def forward(self, batch):
        state = self.path(batch)
        mass = normalize(batch["base_log_mass"].exp()*batch["mask"], (-1,))
        context = (mass[..., None]*batch["base"]).sum(1)
        alpha = .5*torch.sigmoid(self.gate(context))[:, None, None, None]
        target = angular_render(state["mass"], state["coordinate"])
        channel = batch["channel"]
        desired = normalize(((1-alpha)*safe_sqrt(pas(channel))+alpha*safe_sqrt(target)).square(), (2, 3))
        angle = torch.fft.fft2(channel, dim=(2, 3))
        energy = angle.abs().square().sum((2, 3), keepdim=True)
        value = torch.fft.ifft2(safe_sqrt(desired*energy)*unit_complex(angle), dim=(2, 3))
        original_energy = channel.abs().square().sum((2, 3, 4), keepdim=True)
        value = value*safe_sqrt(original_energy/value.abs().square().sum((2, 3, 4), keepdim=True).clamp_min(1e-30))
        value = torch.where(batch["mask"].any(-1)[:, None, None, None, None], value, channel)
        return value, state, alpha


def input_bank(rows, channel, raw, meta, device):
    value = torch.as_tensor(channel, device=device)
    coordinate = np.array(raw["coordinate"][rows])
    coordinate[..., 2] = 0
    features = np.array(raw["features"][rows])
    mass, mask = np.array(raw["mass"][rows]), np.array(raw["mask"][rows])
    phase = 2*np.pi*coordinate/np.array([8., 4., 1.])
    features[..., :3], features[..., 3:6], features[..., 6:9] = np.sin(phase), np.cos(phase), coordinate/np.array([8., 4., 1.])
    features[..., 9] = np.log(np.maximum(mass, 1e-30))/10
    features[..., 16] = np.log(np.maximum(mass, 1e-30))/20
    features[..., 17:19] = 0
    features[..., 19:22] = meta["position"][rows, None, None, :]/30
    features[..., 15] = np.linalg.norm(meta["position"][rows, None, :]-meta["panel_position"][None], axis=-1)[..., None]/100
    angular = pas(value).mean(-1).flatten(2).clamp_min(1e-8).log().cpu().numpy()
    # Fixed zeros carry no hidden wideband measurement information. They retain
    # the original state-network interface while the delay context is removed.
    context = np.concatenate([angular, np.zeros((len(rows), 2, 64), np.float32)], -1)
    features = np.concatenate([features, np.broadcast_to(context[:, :, None, :], (len(rows), 2, 128, 96))], -1)
    sequence = np.array(raw["sequence"][rows])
    depth = np.array(raw["depth"][rows])
    active = np.arange(16) < depth[..., None]
    sequence[..., 18:21] = (meta["position"][rows, None, None, None, :]/30-sequence[..., :3])*active[..., None]
    return {"base": torch.as_tensor(features.reshape(-1, 128, 120), dtype=torch.float32, device=device),
            "sequence_geometry": torch.as_tensor(sequence.reshape(-1, 128, 16, 21), device=device),
            "interactions": torch.as_tensor(np.array(raw["interaction"][rows]).reshape(-1, 128, 16), device=device),
            "depth": torch.as_tensor(depth.reshape(-1, 128), device=device),
            "coordinate": torch.as_tensor(coordinate.reshape(-1, 128, 3), device=device),
            "base_log_mass": torch.as_tensor(np.log(np.maximum(mass, 1e-30)).reshape(-1, 128), device=device),
            "mask": torch.as_tensor(mask.reshape(-1, 128), device=device),
            "channel": value.reshape(-1, 1, 8, 4, 1)}


@torch.inference_mode()
def predict_state(model, bank, use_model=True):
    model.eval()
    values = []
    for first in range(0, len(bank["channel"]), 64):
        batch = {key: value[first:first+64] for key, value in bank.items()}
        value = model(batch)[0] if use_model else batch["channel"]
        values.append(value[:, 0].cpu().numpy())
    return np.concatenate(values).reshape(-1, 2, 8, 4, 1)


def run(args):
    started = time.time()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(20261017+args.fold)
    h, meta, ids = study_data(args.fold)
    support = PATHS.artifacts / "measured/dc01_spectral_v1/merged_support"
    raw = {key: np.load(support/f"{key}.npy", mmap_mode="r") for key in ["coordinate", "features", "mass", "mask", "sequence", "depth", "interaction"]}
    folder = ARTIFACTS / "ours" / f"fold{args.fold}"
    config = {"frequency_hz": 3.438e9, "epochs": 48, "batch_panels": 64,
              "measured_delay_context": False, "loss": "PAS plus array covariance and bounded residual regularization",
              "phase_reference": "nearest measured fit donor", "power": "independent fit-only energy field",
              "selection": "validation angular utility with exact epoch0 fallback",
              "geometry_support": sha256(support/"support_manifest.json"),
              "old_trained_checkpoints_used": False}
    identity = training_identity("ours", args.fold, config,
                                 [Path(__file__), CODE.parent/"dc01_factorized_field.py", CODE.parent/"dc01_angular_state.py", CODE.parent/"dc01_state.py"])
    freeze_identity(folder, identity)
    if (folder/"result.json").exists():
        print(json.dumps({"status": "already_complete", "method": "ours"}), flush=True)
        return
    torch.cuda.reset_peak_memory_stats(device)
    field = MeasuredField(h, meta["position"], meta["panel_position"], ids["fit"], device)
    power_candidates = [("nearest", 1)]+[("idw", k) for k in (4, 8, 16, 32)]+[("rbf", k) for k in (16, 32, 64)]+[("pathloss", k) for k in (8, 16)]
    observed_power = 10*np.log10(np.sum(np.abs(np.array(h[ids["val"]], dtype=np.complex128))**2, axis=(2, 3, 4)))
    errors = np.array([np.abs(field.power_prediction(ids["val"], candidate)-observed_power).mean(0) for candidate in power_candidates])
    selected_power = [power_candidates[int(np.argmin(errors[:, panel]))] for panel in range(2)]
    spectral_candidates = [("idw", k) for k in (4, 8, 16, 32)]+[("rbf", k) for k in (16, 32, 64)]
    selection = []
    for candidate in spectral_candidates:
        value = render_field(field, ids["val"], candidate, selected_power)
        metrics = field_evaluation(value, np.array(h[ids["val"]]))
        selection.append({"candidate": candidate, "metrics": metrics, "utility": utility(metrics)})
    selected_spectral = max(selection, key=lambda value: value["utility"])["candidate"]
    atomic_json(folder/"foundation_selection.json", {"selected_power": selected_power, "selected_spectral": selected_spectral,
                                                     "validation": selection, "power_candidate_errors": errors.tolist()})
    bases = {name: render_field(field, rows, selected_spectral, selected_power, exclude=name == "fit") for name, rows in ids.items()}
    del field
    banks = {name: input_bank(rows, bases[name], raw, meta, device) for name, rows in ids.items()}
    valid = banks["fit"]["base"][banks["fit"]["mask"]]
    mean, std = valid.mean(0), valid.std(0).clamp_min(1e-3)
    for bank in banks.values():
        bank["base"] = (bank["base"]-mean)/std
    train_truth = torch.as_tensor(np.array(h[ids["fit"]]), device=device).reshape(-1, 1, 8, 4, 1)
    val_truth = np.array(h[ids["val"]])
    model = SingleFrequencyState().to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.001, weight_decay=.0001)
    foundation_validation = field_evaluation(bases["val"], val_truth)
    best, best_epoch, best_state, history, first_epoch = utility(foundation_validation), 0, None, [], 1
    if (folder/"resume.pt").exists():
        saved = torch.load(folder/"resume.pt", map_location=device, weights_only=False)
        if saved["identity"] != identity:
            raise ValueError("Our-SF checkpoint identity mismatch")
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        best, best_epoch, best_state, history = (saved[k] for k in ["best", "best_epoch", "best_state", "history"])
        first_epoch = saved["epoch"]+1
        torch.set_rng_state(saved["cpu_rng"].cpu()); torch.cuda.set_rng_state(saved["cuda_rng"].cpu(), device)
    for epoch in range(first_epoch, 49):
        model.train()
        order = torch.randperm(len(train_truth), device=device)
        losses = []
        for first in range(0, len(order), 64):
            take = order[first:first+64]
            batch = {key: value[take] for key, value in banks["fit"].items()}
            value, state, _ = model(batch)
            target = train_truth[take]
            loss = 1-cosine(pas(value), pas(target), (2, 3)).mean()+.25*covariance_error(value, target).mean()+.001*state["score_residual"].square().mean()+.001*state["coordinate_residual"][..., :2].square().mean()
            optimizer.zero_grad(set_to_none=True); loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(loss) or not torch.isfinite(grad):
                raise ValueError("Our-SF nonfinite training")
            optimizer.step(); losses.append(float(loss.detach()))
        if epoch == 1 or epoch % 4 == 0:
            value = predict_state(model, banks["val"])
            score = utility(field_evaluation(value, val_truth))
            if score > best+1e-7:
                best, best_epoch = score, epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            row = {"epoch": epoch, "training_loss": float(np.mean(losses)), "validation_utility": score, "best_epoch": best_epoch}
            history.append(row)
            atomic_json(folder/"progress.json", {"status": "training", "method": "ours", **row})
            temporary = folder/"resume.pt.partial"
            torch.save({"identity": identity, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "best": best, "best_epoch": best_epoch, "best_state": best_state, "history": history,
                        "epoch": epoch, "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(device)}, temporary)
            temporary.replace(folder/"resume.pt")
            print(json.dumps({"method": "ours", **row}), flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    value = predict_state(model, banks["test"], use_model=best_epoch != 0)
    # The test measurements have not been used for field/state selection.
    truth = np.array(h[ids["test"]])
    metrics = save_prediction(folder, value, meta["source_id"][ids["test"]], truth)
    original_energy = (np.abs(bases["test"].astype(np.complex128))**2).sum((2, 3, 4))
    final_energy = (np.abs(value.astype(np.complex128))**2).sum((2, 3, 4))
    energy_error = float(np.max(np.abs(final_energy-original_energy)/np.maximum(original_energy, 1e-30)))
    if energy_error > 3e-6:
        raise ValueError("Our-SF angular correction altered independent predicted energy")
    temporary = folder/"model.pt.partial"
    torch.save({"identity": identity, "selected_epoch": best_epoch, "state_dict": best_state,
                "feature_mean": mean.cpu(), "feature_std": std.cpu(),
                "selected_power": selected_power, "selected_spectral": selected_spectral}, temporary)
    temporary.replace(folder/"model.pt")
    result = {"status": "complete", "identity": identity, "selected_epoch": best_epoch, "history": history,
              "selected_power": selected_power, "selected_spectral": selected_spectral, "test": metrics,
              "maximum_energy_preservation_relative_error": energy_error,
              "elapsed_seconds": time.time()-started, "hardware": torch.cuda.get_device_name(device),
              "peak_gpu_gib": torch.cuda.max_memory_allocated(device)/1024**3,
              "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
              "bottleneck": "ordered path and candidate-set encoding; geometry support reused with historical cost disclosed"}
    atomic_json(folder/"result.json", result)
    atomic_json(folder/"progress.json", {"status": "complete", "method": "ours", "test": metrics})
    print(json.dumps({"method": "ours", "status": "complete", "test": metrics}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        folder = ARTIFACTS/"ours"/f"fold{args.fold}"
        atomic_json(folder/"failure.json", {"status": "failed", "traceback": traceback.format_exc()})
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
