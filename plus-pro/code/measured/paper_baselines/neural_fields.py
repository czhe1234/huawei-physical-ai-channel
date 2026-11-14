"""Explicit dc01 single-frequency adaptations of NeWRF and NeRF2.

The published network bodies are loaded from the pinned author sources. The
adaptations are endpoint conditioning and scene-only direction proposals for
NeWRF, and position conditioning / one-carrier rendering for NeRF2. There are
no measured query-CSI or query-DoA inputs.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stderr, redirect_stdout
import json
import math
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from study_common import ARTIFACTS, AUTHORS, CODE, PATHS, PREPARED, REPORTS, atomic_json, freeze_identity, load_author, save_prediction, sha256, study_data, training_identity


FREQUENCY_HZ = 3438000000.


def atomic_torch(path, value):
    temporary = path.with_name(path.name + ".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def original_nerf2():
    """Load unchanged model AST; omit its unused optional einops import."""
    source = AUTHORS / "nerf2/model.py"
    tree = ast.parse(source.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
             and node.name in {"Embedder", "get_embedder", "NeRF2"}]
    namespace = {"torch": torch, "nn": nn, "F": F}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(source), "exec"), namespace)
    return namespace["NeRF2"]


def unit(vector):
    return vector / np.maximum(np.linalg.norm(vector, axis=-1, keepdims=True), 1e-12)


def fibonacci(count):
    z = 1 - 2 * (np.arange(count) + .5) / count
    phi = np.arange(count) * math.pi * (3 - math.sqrt(5))
    return np.stack([np.sqrt(1-z*z) * np.cos(phi), np.sqrt(1-z*z) * np.sin(phi), z], -1).astype(np.float32)


class Geometry:
    """Geometry-only lookup. No measurement arrays are kept by this object."""
    def __init__(self, meta, device, proposals=8, extra=10):
        shared = dict(np.load(ARTIFACTS / "shared/array_geometry.npz"))
        self.position = torch.as_tensor(meta["position"], dtype=torch.float32, device=device)
        self.antenna = torch.as_tensor(shared["antenna_position"].reshape(64, 3), device=device)
        support = PATHS.artifacts / "measured/dc01_spectral_v1/merged_support"
        mass = np.load(support / "mass.npy", mmap_mode="r")
        mask = np.load(support / "mask.npy", mmap_mode="r")
        sequence = np.load(support / "sequence.npy", mmap_mode="r")
        depth = np.load(support / "depth.npy", mmap_mode="r")
        first = np.zeros((len(self.position), 2, proposals, 3), np.float32)
        last = np.zeros_like(first)
        valid = np.zeros(first.shape[:-1], bool)
        los = np.zeros_like(valid)
        for point in range(len(self.position)):
            for panel in range(2):
                used = []
                for candidate in np.argsort(-np.asarray(mass[point, panel]), kind="stable"):
                    if not mask[point, panel, candidate]:
                        continue
                    d = int(depth[point, panel, candidate])
                    route = np.asarray(sequence[point, panel, candidate, :, :3]) * 30
                    a = route[0] if d else meta["position"][point]
                    b = route[d-1] if d else meta["panel_position"][panel]
                    direction = unit(b-meta["position"][point])
                    if np.linalg.norm(b-meta["position"][point]) < .01:
                        continue
                    if any(float(direction @ previous) > math.cos(math.radians(.5)) for previous in used):
                        continue
                    k = len(used)
                    first[point, panel, k] = a
                    last[point, panel, k] = b
                    valid[point, panel, k] = True
                    los[point, panel, k] = d == 0
                    used.append(direction)
                    if len(used) == proposals:
                        break
        self.first = torch.as_tensor(first, device=device)
        self.last = torch.as_tensor(last, device=device)
        self.valid = torch.as_tensor(valid, device=device)
        self.los = torch.as_tensor(los, device=device)
        self.proposals, self.rays = proposals, proposals+extra
        self.uniform = torch.as_tensor(fibonacci(self.rays), device=device)
        extent = np.ptp(np.concatenate([meta["position"], meta["panel_position"]]), axis=0)
        self.far = max(24., float(np.linalg.norm(extent) * 1.25))
        self.support_identity = sha256(support / "support_manifest.json")
        self.valid_proposals_mean = float(valid.sum(-1).mean())

    def rays_for(self, points, antennas, reverse):
        query = self.position[points]
        anchor = self.antenna[antennas]
        panel = antennas // 32
        targets = self.last[points, panel] if reverse else self.first[points, panel]
        targets = torch.where(self.los[points, panel][..., None],
                              (anchor if reverse else query)[:, None, :], targets)
        origin = query if reverse else anchor
        direction = F.normalize(targets-origin[:, None, :], dim=-1)
        direction = torch.where(self.valid[points, panel][..., None], direction,
                                self.uniform[:self.proposals][None])
        direction = torch.cat([direction, self.uniform[self.proposals:][None].expand(len(points), -1, -1)], dim=1)
        return origin[:, None, :].expand(-1, self.rays, -1), direction, anchor, query


class NeWRFSingleFrequency(nn.Module):
    def __init__(self, geometry, fit_position):
        super().__init__()
        model = load_author("newrf", "models.py", "dc01_newrf_original_model")
        encoding = load_author("newrf", "encoders.py", "dc01_newrf_original_encoding")
        self.sampling = load_author("newrf", "samplers.py", "dc01_newrf_original_sampling")
        self.synthesis = load_author("newrf", "synthesizers.py", "dc01_newrf_original_synthesis").Synthesizer()
        self.encoding = encoding.PositionalEncoder(3, 10, log_space=True)
        self.views = encoding.PositionalEncoder(3, 4, log_space=True)
        self.coarse = model.NeWRF(d_input=2*self.encoding.d_output, d_viewdirs=self.views.d_output)
        self.fine = model.NeWRF(d_input=2*self.encoding.d_output, d_viewdirs=self.views.d_output)
        self.geometry = geometry
        self.register_buffer("center", fit_position.mean(0))
        self.register_buffer("frequency_ghz", torch.tensor(FREQUENCY_HZ/1e9))
        ranges = torch.linalg.vector_norm(fit_position[:, None] - geometry.antenna[None], dim=-1)
        self.render_gain = float(4*math.pi*ranges.median()/(299792458/FREQUENCY_HZ))

    def render(self, network, samples, distance, directions, anchor, batch):
        source = self.encoding((anchor-self.center)/30)
        source = source[:, None, :].expand(-1, samples.shape[1], -1)
        points = self.encoding((samples-self.center)/30)
        views = self.views(directions)[:, None, :].expand(-1, samples.shape[1], -1)
        raw = network(torch.cat([points, source], -1), viewdirs=views)
        value, _, _, weights = self.synthesis.synthesize(raw, distance, self.frequency_ghz,
                                                         [self.geometry.rays]*batch)
        return value*self.render_gain, weights

    def forward(self, points, antennas):
        origin, directions, anchor, _ = self.geometry.rays_for(points, antennas, reverse=True)
        batch = len(points)
        origin, directions = origin.reshape(-1, 3), directions.reshape(-1, 3)
        anchor = anchor[:, None, :].expand(-1, self.geometry.rays, -1).reshape(-1, 3)
        samples, distance = self.sampling.sample_stratified(origin, directions, .01, self.geometry.far, 128,
                                                           perturb=self.training)
        coarse, weights = self.render(self.coarse, samples, distance, directions, anchor, batch)
        fine_samples, fine_distance, _, _ = self.sampling.sample_hierarchical(origin, directions, distance,
                                                                            weights, 128, perturb=self.training)
        fine, _ = self.render(self.fine, fine_samples, fine_distance, directions, anchor, batch)
        return fine, coarse


def nerf2_complex_render(raw, distances, directions, frequency=FREQUENCY_HZ):
    """Author CSI rendering equations, parameterized for a scalar carrier."""
    increments = distances[..., 1:] - distances[..., :-1]
    increments = torch.cat([increments, torch.full_like(increments[..., :1], 1e10)], -1)
    increments = increments * torch.linalg.vector_norm(directions, dim=-1)[..., None]
    attenuation_amplitude = F.leaky_relu(raw[..., 0]).abs()
    attenuation_phase = torch.sigmoid(raw[..., 1])*(2*math.pi)-math.pi
    signal_amplitude = F.leaky_relu(raw[..., 2]).abs()
    signal_phase = torch.sigmoid(raw[..., 3])*(2*math.pi)-math.pi
    cumulative_amplitude = torch.exp(torch.cumsum(-attenuation_amplitude*increments, -1))
    cumulative_phase = torch.exp(1j*torch.cumsum(attenuation_phase + 2*math.pi*increments/(299792458/frequency), -1))
    rays = (signal_amplitude * torch.exp(1j*signal_phase) * cumulative_amplitude * cumulative_phase).sum(-1)
    return rays.sum(-1)


class NeRF2SingleFrequency(nn.Module):
    def __init__(self, geometry, fit_position):
        super().__init__()
        self.network = original_nerf2()(D=8, W=256, skips=[4],
                                      input_dims={"pts": 3, "view": 3, "tx": 3},
                                      multires={"pts": 10, "view": 10, "tx": 10},
                                      is_embeded={"pts": True, "view": True, "tx": True},
                                      attn_output_dims=2, sig_output_dims=2)
        self.geometry = geometry
        self.register_buffer("center", fit_position.mean(0))

    def forward(self, points, antennas):
        origin, directions, _, query = self.geometry.rays_for(points, antennas, reverse=False)
        z = torch.linspace(.01, self.geometry.far, 32, device=points.device)
        distance = z[None, None, :].expand(len(points), self.geometry.rays, -1)
        samples = origin[..., None, :] + directions[..., None, :]*distance[..., None]
        source = ((query-self.center)/30)[:, None, None, :].expand_as(samples)
        view = directions[..., None, :].expand_as(samples)
        raw = self.network((samples-self.center)/30, view, source)
        return nerf2_complex_render(raw, distance, directions), None


@torch.inference_mode()
def prediction(model, rows, batch, device):
    model.eval()
    points = np.repeat(np.asarray(rows), 64)
    antennas = np.tile(np.arange(64), len(rows))
    values = []
    for first in range(0, len(points), batch):
        p = torch.as_tensor(points[first:first+batch], device=device)
        a = torch.as_tensor(antennas[first:first+batch], device=device)
        values.append(model(p, a)[0].cpu().numpy())
    return np.concatenate(values).reshape(len(rows), 2, 8, 4, 1)


def configure(method, geometry, meta, ids, device):
    fit_position = torch.as_tensor(meta["position"][ids["fit"]], dtype=torch.float32, device=device)
    return (NeWRFSingleFrequency if method == "newrf" else NeRF2SingleFrequency)(geometry, fit_position).to(device)


def run(args):
    torch.set_num_threads(4)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(20261017+args.fold+1000*int(args.method == "nerf2"))
    h, meta, ids = study_data(args.fold)
    geometry = Geometry(meta, device)
    model = configure(args.method, geometry, meta, ids, device)
    fit = np.array(h[ids["fit"]])
    scale = float(np.sqrt(np.mean(np.abs(fit.astype(np.complex128))**2))) if args.method == "newrf" else float(np.abs(fit).max())
    labels = torch.as_tensor(fit.reshape(-1)/scale, device=device)
    point_ids = torch.as_tensor(np.repeat(ids["fit"], 64), device=device)
    antenna_ids = torch.arange(64, device=device).repeat(len(ids["fit"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4 if args.method == "newrf" else 1e-3,
                                 weight_decay=0 if args.method == "newrf" else 5e-5)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.9, patience=10, min_lr=1e-5)
                 if args.method == "newrf" else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000, eta_min=1e-6))
    batch = args.batch_size or (32 if args.method == "newrf" else 64)
    folder = ARTIFACTS / args.method / f"fold{args.fold}"
    config = {"frequency_hz": FREQUENCY_HZ, "batch_links": batch, "prediction_batch": args.prediction_batch,
              "coarse_samples": 128 if args.method == "newrf" else 32,
              "fine_extra_samples": 128 if args.method == "newrf" else 0,
              "geometric_proposals": geometry.proposals, "extra_spherical_rays": geometry.rays-geometry.proposals,
              "far_m": geometry.far, "normalization_fit_only": scale,
              "maximum_updates": 100000 if args.method == "newrf" else 70000,
              "validation_interval": 1000, "early_stop_patience": 4, "minimum_fit_link_exposures": 3,
              "minimum_validation_improvement": 1e-4, "support_manifest": geometry.support_identity,
              "network_body": "pinned author's original classes", "unused_optional_import_omitted": args.method == "nerf2"}
    identity = training_identity(args.method, args.fold, config,
                                 [Path(__file__), AUTHORS / args.method / ("models.py" if args.method == "newrf" else "model.py"),
                                  ARTIFACTS / "shared/array_geometry.npz"])
    if args.mode == "train":
        freeze_identity(folder, identity)
        if (folder / "result.json").exists():
            print(json.dumps({"status": "already_complete", "method": args.method}), flush=True)
            return
    torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    gradients = []
    for index in range(3 if args.mode == "probe" else 0):
        model.train()
        take = torch.arange(batch, device=device)
        fine, coarse = model(point_ids[take], antenna_ids[take])
        target = labels[take]
        loss = ((fine-target).abs().square().sum() / target.abs().square().sum()) if args.method == "newrf" else F.mse_loss(torch.view_as_real(fine), torch.view_as_real(target))
        if coarse is not None:
            loss = .9*loss+.1*((coarse-target).abs().square().sum()/target.abs().square().sum())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        if not torch.isfinite(grad) or not torch.isfinite(loss) or float(grad) == 0:
            raise ValueError("Neural-field gradient/finite preflight failed")
        optimizer.step()
        gradients.append(float(grad))
    if args.mode == "probe":
        torch.cuda.synchronize(device)
        result = {"status": "passed", "method": args.method, "fit_only_updates": 3,
                  "gradients_finite_nonzero": gradients, "batch_links": batch,
                  "elapsed_seconds": time.time()-started,
                  "peak_gpu_gib": torch.cuda.max_memory_allocated(device)/1024**3,
                  "valid_geometric_proposals_mean": geometry.valid_proposals_mean,
                  "rays": geometry.rays, "far_m": geometry.far,
                  "author_network_reused": True, "query_csi_used": False}
        atomic_json(REPORTS / f"{args.method}_probe.json", result)
        print(json.dumps(result), flush=True)
        return
    best = float("inf")
    best_step, best_state, stale = 0, None, 0
    history = []
    step, cursor, order = 0, 0, torch.randperm(len(labels), device=device)
    elapsed_before = 0.
    if (folder / "resume.pt").exists():
        state = torch.load(folder / "resume.pt", map_location=device, weights_only=False)
        if state["identity"] != identity:
            raise ValueError("Checkpoint identity mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step, best, best_step, best_state, stale, history = (state[k] for k in ["step", "best", "best_step", "best_state", "stale", "history"])
        order, cursor = state["order"].to(device), state["cursor"]
        torch.set_rng_state(state["cpu_rng"].cpu())
        torch.cuda.set_rng_state(state["cuda_rng"].cpu(), device)
        elapsed_before = state["elapsed_seconds"]

    def save_resume():
        atomic_torch(folder / "resume.pt", {"identity": identity, "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "step": step, "best": best, "best_step": best_step, "best_state": best_state,
                     "stale": stale, "history": history, "order": order, "cursor": cursor,
                     "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(device),
                     "elapsed_seconds": elapsed_before+time.time()-started})

    validation_truth = np.array(h[ids["val"]], dtype=np.complex128)
    validation_energy = float(np.sum(np.abs(validation_truth)**2))
    minimum_steps = math.ceil(3*len(labels)/batch)
    limit = config["maximum_updates"]
    while step <= limit:
        if step == 0 or step % 1000 == 0 or step == limit:
            value = prediction(model, ids["val"], args.prediction_batch, device)*scale
            score = float(np.sum(np.abs(value.astype(np.complex128)-validation_truth)**2)/validation_energy)
            if score < best-1e-4 or best_state is None:
                best, best_step, stale = score, step, 0
                best_state = {key: val.detach().cpu().clone() for key, val in model.state_dict().items()}
            else:
                stale += 1
            if args.method == "newrf":
                scheduler.step(score)
            row = {"step": step, "validation_raw_nmse": score, "best_step": best_step,
                   "best_validation_raw_nmse": best, "stale": stale,
                   "elapsed_seconds": elapsed_before+time.time()-started}
            history.append(row)
            atomic_json(folder / "progress.json", {"status": "training", "method": args.method, **row})
            print(json.dumps({"method": args.method, **row}), flush=True)
            save_resume()
            if step >= limit or (step >= minimum_steps and stale >= 4):
                break
        model.train()
        if cursor+batch > len(order):
            order, cursor = torch.randperm(len(labels), device=device), 0
        take = order[cursor:cursor+batch]
        cursor += batch
        fine, coarse = model(point_ids[take], antenna_ids[take])
        target = labels[take]
        if args.method == "newrf":
            denominator = target.abs().square().sum().clamp_min(1e-20)
            loss = .9*(fine-target).abs().square().sum()/denominator+.1*(coarse-target).abs().square().sum()/denominator
        else:
            loss = F.mse_loss(torch.view_as_real(fine), torch.view_as_real(target))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        if not torch.isfinite(loss) or not torch.isfinite(gradient):
            raise ValueError("Nonfinite neural-field training")
        optimizer.step()
        if args.method == "nerf2":
            scheduler.step()
        step += 1
        if step % 100 == 0:
            atomic_json(folder / "progress.json", {"status": "training", "method": args.method,
                        "step": step, "maximum_updates": limit, "best_step": best_step,
                        "best_validation_raw_nmse": best, "training_loss": float(loss.detach()),
                        "elapsed_seconds": elapsed_before+time.time()-started})
        if step % 250 == 0:
            save_resume()
    model.load_state_dict(best_state)
    atomic_torch(folder / "model.pt", {"identity": identity, "state_dict": best_state,
                                       "normalization_scale": scale, "selected_step": best_step})
    # Model choice is now frozen. Only this final evaluation reads test H.
    value = (prediction(model, ids["test"], args.prediction_batch, device)*scale).astype(np.complex64)
    metrics = save_prediction(folder, value, meta["source_id"][ids["test"]], np.array(h[ids["test"]]))
    torch.cuda.synchronize(device)
    result = {"status": "complete", "identity": identity, "selected_step": best_step,
              "executed_updates": step, "fit_link_exposures": step*batch/len(labels),
              "validation_raw_nmse": best, "test": metrics, "history": history,
              "elapsed_seconds": elapsed_before+time.time()-started,
              "hardware": torch.cuda.get_device_name(device),
              "peak_gpu_gib": torch.cuda.max_memory_allocated(device)/1024**3,
              "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
              "bottleneck": "coarse/fine neural ray evaluation" if args.method == "newrf" else "neural attenuation and radiance field evaluation"}
    atomic_json(folder / "result.json", result)
    atomic_json(folder / "progress.json", {"status": "complete", "method": args.method, "test": metrics})
    print(json.dumps({"method": args.method, "status": "complete", "test": metrics}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["newrf", "nerf2"], required=True)
    parser.add_argument("--mode", choices=["probe", "train"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--prediction-batch", type=int, default=128)
    args = parser.parse_args()
    if args.mode != "train":
        run(args)
        return
    folder = ARTIFACTS / args.method / f"fold{args.fold}"
    folder.mkdir(parents=True, exist_ok=True)

    class Tee:
        def __init__(self, terminal, log):
            self.terminal, self.log = terminal, log

        def write(self, value):
            self.log.write(value)
            return self.terminal.write(value)

        def flush(self):
            self.log.flush()
            self.terminal.flush()

    with (folder / "training.log").open("a", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                run(args)
            except Exception:
                atomic_json(folder / "failure.json", {"status": "failed", "traceback": traceback.format_exc()})
                traceback.print_exc()
                raise


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
