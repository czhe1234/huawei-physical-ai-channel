"""Isolated NeRF2 FIT repair development; no outer labels or official scoring.

Architecture, renderer, scalar FIT RMS, and 0.5 complex MSE remain unchanged.
Each independent field receives its own balanced update/coverage budget.
The default entrypoint prepares a plan. Training requires --stage run and an
explicit frozen JSON configuration. All new output stays in this dev directory.
"""
from __future__ import annotations

import argparse
import copy
from collections import deque
import hashlib
import json
import math
import random
import signal
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from project_paths import PATHS
from . import common as C
from . import nerf2_runner_v2 as P
from . import nerf2_runner_v3 as V

ART = C.ART / "fit_repair_dev_20261101/nerf2"
REPORT = C.REPORT / "fit_repair_dev_20261101/nerf2"
FIELDS = [(s, b, u) for s in (0, 1) for b in (0, 1) for u in (0, 1)]
SCHEMA = "nerf2-balanced-fit-repair-dev-v1"
DEFAULT_CONFIG = dict(
    schema=SCHEMA, freeze_approved=False, outer=0, seed=42,
    points_per_site=8, positions_per_batch=2, bs_per_batch=8,
    frequencies_per_batch=4, per_field_updates=3072,
    monitor_every_per_field=384, checkpoint_every_per_field=384,
    recent_batches_per_field=4, max_seconds=1800, train_wall_seconds=1500,
    final_reserve_seconds=300, warmup_updates=8,
    rlp_patience=1, rlp_factor=0.5, rlp_relative_threshold=0.001,
    rlp_min_lr=1e-6, enable_plateau_early_stop=False,
    optimizer="one Adam; eight field groups plus single shared-gain group",
    learning_rate="independent field FIT plateau states; shared gain has one Adam state",
    training_loss="0.5*mean(abs(pred-target)**2)",
    role="development FIT repair, no outer/test prediction or score",
)
COMMON_DEV = REPORT.parent



def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_config(path):
    contract = json.loads((COMMON_DEV / "contract.json").read_text())
    limits = contract["budgets"]["nerf2"]
    cfg = {**DEFAULT_CONFIG,
           "per_field_updates": limits["per_field_target"],
           "max_seconds": limits["max_wall_seconds"],
           "train_wall_seconds": limits["train_wall_seconds"],
           **(json.loads(path.read_text()) if path else {})}
    if cfg["schema"] != SCHEMA or cfg["outer"] != 0 or cfg["seed"] != contract["seed"]:
        raise ValueError("This development runner uses the frozen outer0 seed42 contract")
    if cfg["points_per_site"] != 8 or cfg["positions_per_batch"] != 2:
        raise ValueError("Frozen fixture is eight points/site, two points/batch")
    if 128 % cfg["bs_per_batch"] or 192 % cfg["frequencies_per_batch"]:
        raise ValueError("BS/tone batch sizes must divide complete axes")
    for key in ("per_field_updates", "monitor_every_per_field", "checkpoint_every_per_field",
                "recent_batches_per_field", "max_seconds", "train_wall_seconds"):
        if cfg[key] <= 0:
            raise ValueError(key + " must be positive")
    if cfg["per_field_updates"] != limits["per_field_target"]:
        raise ValueError("Per-field target must match common contract")
    if cfg["train_wall_seconds"] != limits["train_wall_seconds"] or cfg["max_seconds"] != limits["max_wall_seconds"]:
        raise ValueError("Wall budgets must match common contract")
    if cfg["training_loss"] != DEFAULT_CONFIG["training_loss"] or cfg["optimizer"] != DEFAULT_CONFIG["optimizer"]:
        raise ValueError("Unexpected loss/optimizer change")
    if cfg["enable_plateau_early_stop"]:
        raise ValueError("This bounded runner uses target/hard-wall stopping only")
    return cfg

def source_identity(config_path):
    source = {**V.identity(), PATHS.relative(Path(__file__)): C.sha256(Path(__file__))}
    extra = [
        PATHS.code_root / "round2/paper_baseline_adapters_v1/field_models.py",
        PATHS.code_root / "round2/paper_baseline_adapters_v1/contract.py",
        PATHS.root / "plus-pro/code/measured/full_h_adapters_v1/nerf2.py",
        PATHS.root / "plus-pro/code/third_party/paper_baselines/nerf2/model.py",
        PATHS.root / "plus-pro/code/third_party/paper_baselines/nerf2/renderer.py",
        COMMON_DEV / "contract.json", COMMON_DEV / "panel.json",
    ]
    if config_path:
        extra.append(config_path)
    source.update({PATHS.relative(p): C.sha256(p) for p in extra})
    return source

def check_sources(source):
    for key, digest in source.items():
        path = PATHS.root / key if "/" in key else Path(__file__).parent / key
        if C.sha256(path) != digest:
            raise RuntimeError("Frozen numerical source/config changed: " + key)


def load_data(cfg):
    data = C.FoldData(0, "screen", "nerf2")
    # Never create or rewrite the shared historical scalar-RMS cache/lock.
    path = C.ART / "fit_scales/outer0_screen.json"
    info = json.loads(path.read_text())
    want = dict(schema="FIT-channel-scalar-RMS-v1", input_identity=data.inputs.identity,
                fit_ids_sha256=C.digest_ids(data.fit_ids), raw=data.raw_identity,
                outer=0, role="screen")
    if info["contract"] != want or not math.isfinite(info["scale"]) or info["scale"] <= 0:
        raise RuntimeError("Existing FIT RMS identity mismatch")
    data._scale_info = info
    return data


def spread_select(ids, positions, count):
    ids = np.asarray(ids, np.int64)
    if len(ids) < count:
        raise ValueError("Insufficient candidates for fixed fixture")
    p = positions[ids]
    centre = p.mean(0)
    selected = [int(np.argmin(np.linalg.norm(p - centre, axis=1)))]
    distance = np.linalg.norm(p - p[selected[0]], axis=1)
    for _ in range(1, count):
        distance[selected] = -1
        index = int(np.argmax(distance))
        selected.append(index)
        distance = np.minimum(distance, np.linalg.norm(p - p[index], axis=1))
    return ids[selected]


def make_plan(data, cfg):
    panel = json.loads((COMMON_DEV / "panel.json").read_text())
    contract = json.loads((COMMON_DEV / "contract.json").read_text())
    fixture = np.asarray(panel["train_ids"], np.int64)
    val = np.asarray(panel["validation_ids"], np.int64)
    zero = np.asarray(panel["zero_probe_ids"], np.int64)
    if panel["input_identity"] != data.inputs.identity or panel["parent_fit_ids_sha256"] != C.digest_ids(data.fit_ids):
        raise ValueError("Common panel/parent FIT identity mismatch")
    if len(fixture) != 16 or len(np.unique(fixture)) != 16 or not set(fixture).issubset(set(data.fit_ids)):
        raise ValueError("Invalid development training fixture")
    if not np.array_equal(np.bincount(data.sites(fixture)), [8, 8]):
        raise ValueError("Need eight fixture positions per site")
    if not set(val).issubset(set(data.val_ids)) or set(fixture) & set(val):
        raise ValueError("Invalid validation role")
    if not set(zero).issubset(set(fixture)) or set(panel["zero_train_ids"]) != set(zero):
        raise ValueError("Zero training/probe membership mismatch")
    if panel["parent_FIT_RMS"] != data.scale or panel["train_ids_sha256"] != C.digest_ids(fixture):
        raise ValueError("Frozen RMS/order mismatch")
    descriptor = PATHS.artifacts / "round2/pre_sionna/round2_train_spectral_descriptors.npz"
    with np.load(descriptor, allow_pickle=False) as z:
        power = np.asarray(z["total_power"], np.float64)
    if not np.all(power[zero] == 0) or not np.all(power[np.asarray(panel["nonzero_train_ids"])] > 0):
        raise ValueError("Panel energy category mismatch")
    p = 8 // cfg["positions_per_batch"]
    b = 128 // cfg["bs_per_batch"]
    f = 192 // cfg["frequencies_per_batch"]
    return dict(
        schema=SCHEMA, config=cfg, config_sha256=canonical_hash(cfg), contract=contract,
        panel_sha256=C.sha256(COMMON_DEV / "panel.json"),
        input_identity=data.inputs.identity, fit_ids_sha256=C.digest_ids(data.fit_ids),
        raw_channel_identity=data.raw_identity, fit_RMS=data.scale,
        fixture_ids=fixture.tolist(), fixture_positions_m=data.positions(fixture).tolist(),
        fixture_site_ids=data.sites(fixture).tolist(), fixture_energies=power[fixture].tolist(),
        selection=panel["selection"], zero_probe_ids=zero.tolist(),
        validation_diagnostic_ids=val.tolist(), zero_rows_are_trained=True,
        training_axes=dict(BS_spatial=list(range(128)), UE_vertical=[0, 1],
                           frequency_ids=list(range(192)), fields=[list(x) for x in FIELDS]),
        position_pair_groups=p, BS_groups=b, frequency_groups=f,
        updates_per_field_complete_joint_cycle=p * b * f,
        complete_joint_cycle="each fixture position x128BS spatial x192tone for every field",
        schedule="strict eight-field round robin; shuffled axis partitions, Latin pair ordering",
        monitor_BS_spatial=P.BS_SPATIAL.tolist(), monitor_frequency_ids=P.FREQUENCY_IDS.tolist(),
        outer_truth_read=False, outer_predictions_written=False, scoring=False,
        previous_results_changed=False, V16_changed=False,
    )

class BalancedSampler:
    """All joint axis blocks once/cycle, with early balanced marginal coverage."""
    def __init__(self, plan):
        self.plan = plan
        self.cfg = plan["config"]
        self.rng = np.random.default_rng(self.cfg["seed"])
        self.updates = np.zeros(8, np.int64)
        self.orders = {}
        self.recent = [deque(maxlen=self.cfg["recent_batches_per_field"]) for _ in FIELDS]

    def draw(self, global_step):
        k = global_step % len(FIELDS)
        s, b, u = FIELDS[k]
        q = int(self.updates[k])
        cycle_size = self.plan["updates_per_field_complete_joint_cycle"]
        cycle, local = divmod(q, cycle_size)
        if k not in self.orders or self.orders[k]["cycle"] != cycle:
            rows = np.asarray(self.plan["fixture_ids"], np.int64)
            sites = np.asarray(self.plan["fixture_site_ids"], np.int64)
            self.orders[k] = dict(
                cycle=cycle,
                rows=self.rng.permutation(rows[sites == s]).reshape(-1, self.cfg["positions_per_batch"]),
                bs=self.rng.permutation(128).reshape(-1, self.cfg["bs_per_batch"]),
                frequencies=self.rng.permutation(192).reshape(-1, self.cfg["frequencies_per_batch"]))
        order = self.orders[k]
        p = len(order["rows"])
        nb, nf = len(order["bs"]), len(order["frequencies"])
        pair, pi = divmod(local, p)
        fi = pair % nf
        bi = (pair // nf + fi) % nb
        task = dict(field=k, site=s, BS_pol=b, UE_pol=u,
                    rows=order["rows"][pi].tolist(),
                    BS_spatial_ids=np.sort(order["bs"][bi]).tolist(),
                    frequency_ids=np.sort(order["frequencies"][fi]).tolist(),
                    local_update=q + 1, joint_cycle=cycle, joint_block=[pi, bi, fi])
        self.updates[k] += 1
        self.recent[k].append(copy.deepcopy(task))
        return task

    def state_dict(self):
        return dict(rng_state=copy.deepcopy(self.rng.bit_generator.state),
                    updates=self.updates.copy(), orders=copy.deepcopy(self.orders),
                    recent=[list(x) for x in self.recent])

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state["rng_state"]
        self.updates = np.asarray(state["updates"], np.int64)
        self.orders = state["orders"]
        self.recent = [deque(x, maxlen=self.cfg["recent_batches_per_field"]) for x in state["recent"]]

    def coverage(self):
        cycle_size = self.plan["updates_per_field_complete_joint_cycle"]
        p, nb, nf = (self.plan[k] for k in ("position_pair_groups", "BS_groups", "frequency_groups"))
        result = []
        for k, update in enumerate(self.updates):
            cycles, partial = divmod(int(update), cycle_size)
            seen_p, seen_b, seen_f = set(), set(), set()
            for j in range(partial):
                pair, pi = divmod(j, p)
                fi = pair % nf
                bi = (pair // nf + fi) % nb
                seen_p.add(pi); seen_b.add(bi); seen_f.add(fi)
            result.append(dict(
                field=list(FIELDS[k]), updates=int(update), complete_joint_cycles=cycles,
                partial_joint_blocks=partial, total_joint_blocks_per_cycle=cycle_size,
                current_cycle_position_pair_groups_seen=len(seen_p),
                current_cycle_BS_groups_seen=len(seen_b), current_cycle_frequency_groups_seen=len(seen_f),
                all_positions_ever_seen=cycles > 0 or len(seen_p) == p,
                all_BS_ever_seen=cycles > 0 or len(seen_b) == nb,
                all_frequencies_ever_seen=cycles > 0 or len(seen_f) == nf))
        return result


class FixtureCache:
    def __init__(self, data, plan):
        ids = np.asarray(plan["fixture_ids"], np.int64)
        self.slot = {int(row): i for i, row in enumerate(ids)}
        self.value = torch.tensor(data.read_fit(ids), device="cuda:0", dtype=torch.complex64)

    def target(self, task):
        slots = torch.tensor([self.slot[int(i)] for i in task["rows"]], device="cuda:0")
        mm = torch.tensor(task["BS_spatial_ids"], device="cuda:0") + task["BS_pol"] * 128
        ff = torch.tensor(task["frequency_ids"], device="cuda:0")
        u = task["UE_pol"]
        return self.value.index_select(0, slots).index_select(1, mm)[:, :, u * 2:(u + 1) * 2].index_select(-1, ff)


def forward_task(model, data, task):
    ids = np.asarray(task["rows"], np.int64)
    p = torch.tensor(data.positions(ids), device="cuda:0", dtype=torch.float64)
    s = torch.tensor(data.sites(ids), device="cuda:0", dtype=torch.int64)
    u = task["UE_pol"]
    world = data.geometry.ue_world_positions(p, s)[:, u * 2:(u + 1) * 2]
    mm = torch.tensor(task["BS_spatial_ids"], device="cuda:0")
    tx = model.bound_bs_ports[task["site"], task["BS_pol"]].reshape(128, 3)[mm]
    f = torch.tensor(data.frequencies(task["frequency_ids"]), device="cuda:0", dtype=torch.float64)
    core = model.fields[f"s{task['site']}_b{task['BS_pol']}_u{u}"]
    return core(tx, world, f).permute(1, 0, 2, 3) * model.log_output_gain.exp()


def sums(pred, target):
    return dict(squared_error=float((pred - target).abs().square().sum(dtype=torch.float64)),
                truth_energy=float(target.abs().square().sum(dtype=torch.float64)),
                prediction_energy=float(pred.abs().square().sum(dtype=torch.float64)),
                signed_cross=float((pred.conj() * target).real.sum(dtype=torch.float64)),
                complex_scalar_count=target.numel())


def finish_stats(parts):
    total = {k: sum(x[k] for x in parts) for k in
             ("squared_error", "truth_energy", "prediction_energy", "signed_cross", "complex_scalar_count")}
    te, pe = total["truth_energy"], total["prediction_energy"]
    total.update(nmse=total["squared_error"] / te if te > 0 else None,
                 energy_ratio=pe / te if te > 0 else None,
                 signed_correlation=total["signed_cross"] / math.sqrt(pe * te) if pe > 0 and te > 0 else None)
    return total


@torch.no_grad()
def diagnostic(model, data, cache, plan, sampler):
    model.eval()
    output = dict(per_field=[], outer_truth_read=False, scoring=False)
    fixture = np.asarray(plan["fixture_ids"], np.int64)
    for k, (s, b, u) in enumerate(FIELDS):
        tasks = []
        rows = fixture[data.sites(fixture) == s]
        for pair in np.array_split(rows, len(rows) // 2):
            tasks.append(dict(site=s, BS_pol=b, UE_pol=u, rows=pair.tolist(),
                              BS_spatial_ids=plan["monitor_BS_spatial"],
                              frequency_ids=plan["monitor_frequency_ids"]))
        fixed = finish_stats([sums(forward_task(model, data, t), cache.target(t)) for t in tasks])
        recent = finish_stats([sums(forward_task(model, data, t), cache.target(t)) for t in sampler.recent[k]]) if sampler.recent[k] else None
        output["per_field"].append(dict(field=[s, b, u], updates=int(sampler.updates[k]),
            fixed_FIT=fixed, recent_actual_training_batches=recent, recent_tasks=list(sampler.recent[k])))
    output["coverage"] = sampler.coverage()
    output["output_gain"] = float(model.log_output_gain.exp())
    output["aggregate_fixed_FIT"] = finish_stats([x["fixed_FIT"] for x in output["per_field"]])
    return output

def verdict(full_statistics, plan, sampler):
    gates = {}
    for name, limits in plan["contract"]["acceptance"].items():
        fields = full_statistics["per_field"]
        pooled = full_statistics["pooled"]["nmse"]
        passes = pooled is not None and pooled <= limits["pooled_nmse_max"]
        passes = passes and all(
            item["nmse"] is not None and item["nmse"] <= limits["all_field_nmse_max"]
            and item["energy_ratio"] is not None
            and limits["all_field_energy_ratio_min"] <= item["energy_ratio"] <= limits["all_field_energy_ratio_max"]
            for item in fields)
        gates[name] = dict(passed=bool(passes), limits=limits)
    status = "good_fit" if gates["good_fit"]["passed"] else "learning_recovered" if gates["learning_recovered"]["passed"] else "underfit"
    return dict(status=status, gates=gates,
                target_budget_complete=bool(np.all(sampler.updates >= plan["config"]["per_field_updates"])),
                complete_joint_cycles=min(x["complete_joint_cycles"] for x in sampler.coverage()),
                decision_basis="complete raw H on all16 FIT rows; pooled/per-field gates; per-position and zero leakage reported separately",
                does_not_establish_generalization_or_paper_performance=True)

def snapshot(model, opt, sampler, step, plan, source, elapsed, history, plateau):
    return dict(model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                optimizer=opt.state_dict(), sampler=sampler.state_dict(),
                plateau=copy.deepcopy(plateau),
                rng_state=copy.deepcopy(sampler.rng.bit_generator.state),
                numpy_global_rng=np.random.get_state(), python_rng=random.getstate(),
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                step=step, plan=plan, source_identity=source, elapsed_seconds=elapsed,
                history=history, outer_truth_read=False, V16_changed=False)

def append_event(event):
    with (REPORT / "raw_log.jsonl").open("a") as f:
        f.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
    print(json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)



def make_optimizer(model):
    # A single Adam owns the shared gain. Each inactive field has grad=None.
    groups = [dict(params=list(model.fields[f"s{s}_b{b}_u{u}"].parameters()),
                   lr=1e-4, initial_lr=1e-4, weight_decay=5e-5, name=f"field{k}")
              for k, (s, b, u) in enumerate(FIELDS)]
    groups.append(dict(params=[model.log_output_gain], lr=1e-3, initial_lr=1e-3,
                       weight_decay=0.0, name="shared_global_gain"))
    opt = torch.optim.Adam(groups)
    identifiers = [id(p) for group in groups for p in group["params"]]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("An optimizer parameter is repeated")
    return opt


def new_plateau():
    return [dict(numerical_best=None, anchor=None, bad_observations=0,
                 reductions=0, last_reduction_update=None, observations=0,
                 lr_scale=1.0) for _ in FIELDS]


def update_plateau(states, diag, opt, cfg):
    for k, entry in enumerate(diag["per_field"]):
        score = entry["fixed_FIT"]["nmse"]
        if score is None or not math.isfinite(score):
            raise FloatingPointError("Invalid field FIT monitor")
        state = states[k]
        state["observations"] += 1
        # Numerical best is independent from the plateau significance threshold.
        if state["numerical_best"] is None or score < state["numerical_best"]:
            state["numerical_best"] = score
        if state["anchor"] is None or score < state["anchor"] * (1 - cfg["rlp_relative_threshold"]):
            state["anchor"] = score
            state["bad_observations"] = 0
        else:
            state["bad_observations"] += 1
        if state["bad_observations"] > cfg["rlp_patience"]:
            old_lr = opt.param_groups[k]["lr"]
            new_lr = max(cfg["rlp_min_lr"], old_lr * cfg["rlp_factor"])
            if new_lr < old_lr:
                opt.param_groups[k]["lr"] = new_lr
                state["lr_scale"] = new_lr / opt.param_groups[k]["initial_lr"]
                state["reductions"] += 1
                state["last_reduction_update"] = entry["updates"]
            state["bad_observations"] = 0
    return copy.deepcopy(states)


def set_active_gain_lr(opt, field, cfg):
    # Field schedulers select the current step size; gain Adam moments stay shared.
    scale = opt.param_groups[field]["lr"] / opt.param_groups[field]["initial_lr"]
    opt.param_groups[-1]["lr"] = max(cfg["rlp_min_lr"], 1e-3 * scale)


def do_update(model, opt, data, cache, task, cfg):
    set_active_gain_lr(opt, task["field"], cfg)
    model.train(); opt.zero_grad(set_to_none=True)
    pred = forward_task(model, data, task)
    truth = cache.target(task)
    loss = 0.5 * (pred - truth).abs().square().mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite native complex MSE")
    loss.backward()
    if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise FloatingPointError("Nonfinite gradient")
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
    opt.step()
    return dict(native_loss=float(loss.detach()), gradient_norm=float(grad),
                statistics=sums(pred.detach(), truth),
                output_gain=float(model.log_output_gain.exp()),
                field_lr=opt.param_groups[task["field"]]["lr"], gain_lr=opt.param_groups[-1]["lr"])


@torch.no_grad()
def first_batch_qa(model, data, cache, plan):
    sampler = BalancedSampler(plan)
    task = sampler.draw(0)
    model.train()
    train_value = forward_task(model, data, task)
    model.eval()
    eval_value = forward_task(model, data, task)
    relative = float(torch.linalg.vector_norm(train_value - eval_value) /
                     torch.linalg.vector_norm(train_value).clamp_min(1e-30))
    raw = data.read_fit(np.asarray(task["rows"]), task["frequency_ids"], normalized=False)
    b, u = task["BS_pol"], task["UE_pol"]
    expected = raw[:, b * 128 + np.asarray(task["BS_spatial_ids"]), u * 2:(u + 1) * 2] / data.scale
    target = cache.target(task)
    target_error = float(torch.max(torch.abs(target - torch.tensor(expected, device="cuda:0"))))
    if relative > 1e-6 or target_error > 1e-6:
        raise RuntimeError("Training/evaluation or normalization/index QA failed")
    return dict(task=task, train_eval_relative_L2=relative, normalized_target_max_error=target_error,
                prediction_statistics=sums(train_value, target), parameter_updates=False)


def array_statistics(pred, truth, plan, row_ids):
    def calc(a, b):
        a, b = a.astype(np.complex128), b.astype(np.complex128)
        te = float(np.sum(np.abs(b) ** 2))
        pe = float(np.sum(np.abs(a) ** 2))
        se = float(np.sum(np.abs(a - b) ** 2))
        return dict(squared_error=se, truth_energy=te, prediction_energy=pe,
                    nmse=se / te if te > 0 else None, energy_ratio=pe / te if te > 0 else None,
                    complex_scalar_count=a.size,
                    normalized_prediction_mean_energy=pe / a.size / plan["fit_RMS"] ** 2)
    site_lookup = dict(zip(plan["fixture_ids"], plan["fixture_site_ids"]))
    sites = np.asarray([site_lookup[int(i)] for i in row_ids])
    result = dict(pooled=calc(pred, truth), per_field=[], per_position=[])
    for s, b, u in FIELDS:
        st = calc(pred[sites == s, b * 128:(b + 1) * 128, u * 2:(u + 1) * 2],
                  truth[sites == s, b * 128:(b + 1) * 128, u * 2:(u + 1) * 2])
        result["per_field"].append(dict(field=[s, b, u], **st))
    for i, row in enumerate(row_ids):
        result["per_position"].append(dict(row_id=int(row), site=int(sites[i]), **calc(pred[i], truth[i])))
    zero = np.asarray([x["truth_energy"] == 0 for x in result["per_position"]])
    zero_energy = float(np.sum(np.abs(pred[zero].astype(np.complex128)) ** 2))
    total_truth = result["pooled"]["truth_energy"]
    result["zero_leakage"] = dict(count=int(zero.sum()),
        row_ids=np.asarray(row_ids)[zero].tolist(), raw_total_prediction_energy=zero_energy,
        relative_to_all_nonzero_truth_energy=zero_energy / total_truth if total_truth > 0 else None,
        normalized_mean_prediction_energy=zero_energy / max(int(zero.sum()) * 256 * 4 * 192, 1) / plan["fit_RMS"] ** 2)
    return result


@torch.no_grad()
def final_fit_fullH(model, data, cache, plan):
    model.eval()
    ids = np.asarray(plan["fixture_ids"], np.int64)
    output = np.zeros((len(ids), 256, 4, 192), np.complex64)
    # Use the same core/port geometry as training, with both complete axes.
    for i, row in enumerate(ids):
        s = int(data.sites(np.asarray([row]))[0])
        for b in (0, 1):
            for u in (0, 1):
                task = dict(site=s, BS_pol=b, UE_pol=u, rows=[int(row)],
                            BS_spatial_ids=list(range(128)), frequency_ids=list(range(192)))
                value = forward_task(model, data, task)[0]
                output[i, b * 128:(b + 1) * 128, u * 2:(u + 1) * 2] = (value * data.scale).cpu().numpy()
        print(json.dumps(dict(stage="final_fullH_FIT", completed=i + 1, total=len(ids))), flush=True)
    if not np.isfinite(output).all():
        raise FloatingPointError("Nonfinite final full-H FIT prediction")
    C.atomic_npy(ART / "final_fit_prediction.npy", output, replace=True)
    C.atomic_npy(ART / "fit_row_ids.npy", ids, replace=True)
    raw_truth = data.read_fit(ids, normalized=False)
    return array_statistics(output, raw_truth, plan, ids)


@torch.no_grad()
def final_validation_subset(model, data, plan):
    ids = np.asarray(plan["validation_diagnostic_ids"], np.int64)
    bs = np.linspace(0, 127, 2, dtype=np.int64)
    ff = np.asarray(plan["monitor_frequency_ids"], np.int64)
    ports = np.concatenate([bs, 128 + bs])
    model.eval()
    output = np.zeros((len(ids), 4, 4, len(ff)), np.complex64)
    # BS4 means two spatial elements per polarization, all four UE ports.
    for i, row in enumerate(ids):
        s = int(data.sites(np.asarray([row]))[0])
        for b in (0, 1):
            for u in (0, 1):
                task = dict(site=s, BS_pol=b, UE_pol=u, rows=[int(row)],
                            BS_spatial_ids=bs.tolist(), frequency_ids=ff.tolist())
                value = forward_task(model, data, task)[0]
                output[i, b * 2:(b + 1) * 2, u * 2:(u + 1) * 2] = (value * data.scale).cpu().numpy()
    truth = data.read_validation(ids, ff, normalized=False)[:, ports]
    C.atomic_npy(ART / "final_validation_prediction.npy", output, replace=True)
    C.atomic_json(ART / "validation_axes.json", dict(BS_ids=ports.tolist(), UE_ids=list(range(4)), frequency_ids=ff.tolist(), shape=list(output.shape), raw_H_units=True))
    C.atomic_npy(ART / "validation_row_ids.npy", ids, replace=True)
    return dict(shape=list(output.shape), BS_port_ids=ports.tolist(), frequency_ids=ff.tolist(),
                squared_error=float(np.sum(np.abs(output.astype(np.complex128) - truth) ** 2)),
                truth_energy=float(np.sum(np.abs(truth.astype(np.complex128)) ** 2)),
                prediction_energy=float(np.sum(np.abs(output.astype(np.complex128)) ** 2)),
                diagnostic_only=True, used_for_selection=False, outer_truth_read=False)


def sampler_cpu_check(plan, data):
    sampler = BalancedSampler(plan)
    n = plan["updates_per_field_complete_joint_cycle"]
    counts = np.zeros((8, 8, 128, 192), np.uint8)
    site_rows = [np.asarray(plan["fixture_ids"])[np.asarray(plan["fixture_site_ids"]) == s] for s in (0, 1)]
    lookup = [{int(v): i for i, v in enumerate(rows)} for rows in site_rows]
    first_tasks = []
    for j in range(n * 8):
        task = sampler.draw(j)
        if j < 8:
            first_tasks.append(task)
        pi = [lookup[task["site"]][int(v)] for v in task["rows"]]
        ix = np.ix_(np.asarray([task["field"]]), np.asarray(pi),
                    np.asarray(task["BS_spatial_ids"]), np.asarray(task["frequency_ids"]))
        counts[ix] += 1
    if not np.all(counts == 1) or not np.all(sampler.updates == n):
        raise RuntimeError("Exact joint coverage check failed")
    # Validate resumed sampling task identity and CPU target slicing algebra.
    saved = sampler.state_dict()
    sampler2 = BalancedSampler(plan); sampler2.load_state_dict(saved)
    t1, t2 = sampler.draw(n * 8), sampler2.draw(n * 8)
    if t1 != t2:
        raise RuntimeError("Sampler resume identity failed")
    task = first_tasks[0]
    raw = data.read_fit(np.asarray(task["rows"]), normalized=False)
    b, u = task["BS_pol"], task["UE_pol"]
    via_full = np.take(raw[:, b * 128 + np.asarray(task["BS_spatial_ids"]), u * 2:(u + 1) * 2],
                       task["frequency_ids"], axis=-1) / data.scale
    via_freq = data.read_fit(np.asarray(task["rows"]), task["frequency_ids"])[:, b * 128 + np.asarray(task["BS_spatial_ids"]), u * 2:(u + 1) * 2]
    diff = float(np.max(np.abs(via_full - via_freq)))
    if diff != 0:
        raise RuntimeError("CPU target frequency/port normalization mismatch")
    return dict(status="passed", complete_joint_cycle_updates_per_field=n,
        joint_axis_cell_count_excluding_UE_vertical=int(counts.size),
        final_complex_values_including_UE_vertical=int(counts.size * 2), min_cell_count=int(counts.min()),
        max_cell_count=int(counts.max()), resume_identical=True,
        normalized_target_max_error=diff, first_round_tasks=first_tasks,
        outer_truth_read=False, parameter_updates=False, GPU_used=False)



def benchmark_reserved_seconds(qa, cfg):
    reserve = max(300.0, 16.0 * qa["full_H_one_FIT_row_seconds"] * 1.25 + 50.0)
    effective = min(float(cfg["train_wall_seconds"]), float(cfg["max_seconds"]) - reserve)
    if effective <= 0:
        raise RuntimeError("Measured full-H diagnostic leaves no training wall budget")
    return dict(full_FIT_estimated_seconds=16.0 * qa["full_H_one_FIT_row_seconds"],
                reserved_final_seconds=reserve, effective_train_wall_seconds=effective,
                maximum_total_wall_seconds=cfg["max_seconds"],
                policy="training cap only shrinks; total1800s never increases")


def gpu_check(cfg, config_path, plan, source):
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("GPU QA hard120s limit")))
    signal.setitimer(signal.ITIMER_REAL, 120)
    started = time.monotonic()
    gpu = C.device_guard(P.GPU_UUID)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    C.seed_all(cfg["seed"])
    data = load_data(cfg); cache = FixtureCache(data, plan)
    model = V.build(data); opt = make_optimizer(model)
    train_eval_qa = first_batch_qa(model, data, cache, plan)
    sampler = BalancedSampler(plan)
    timings = []
    for j in range(cfg["warmup_updates"]):
        task = sampler.draw(j); tick = time.monotonic()
        do_update(model, opt, data, cache, task, cfg)
        torch.cuda.synchronize()
        timings.append(time.monotonic() - tick)
        if time.monotonic() - started > 90:
            raise TimeoutError("GPU QA exhausted warmup allowance")
    first_row = int(plan["fixture_ids"][0])
    site = int(data.sites(np.asarray([first_row]))[0])
    torch.cuda.synchronize(); tick = time.monotonic()
    shapes = []
    with torch.no_grad():
        model.eval()
        for b in (0, 1):
            for u in (0, 1):
                task = dict(site=site, BS_pol=b, UE_pol=u, rows=[first_row],
                            BS_spatial_ids=list(range(128)), frequency_ids=list(range(192)))
                value = forward_task(model, data, task)
                if not bool(torch.isfinite(value).all()) or tuple(value.shape) != (1, 128, 2, 192):
                    raise RuntimeError("Full-axis benchmark shape/finite failure")
                shapes.append(list(value.shape)); del value
    torch.cuda.synchronize()
    full_seconds = time.monotonic() - tick
    check_sources(source)
    result = dict(schema="nerf2-bounded-GPU-QA-v1", status="passed", source_identity=source,
        panel_sha256=plan["panel_sha256"], config=cfg, GPU=gpu,
        warmup_updates=len(timings), warmup_step_seconds=timings,
        warmup_median_step_seconds=float(np.median(timings)),
        full_H_one_FIT_row_id=first_row, full_H_one_FIT_row_seconds=full_seconds,
        full_H_field_shapes=shapes, first_batch_qa=train_eval_qa,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        elapsed_seconds=time.monotonic() - started, weights_optimizer_and_RNG_discarded=True,
        formal_run_reseeds_fresh_model=True, outer_truth_read=False,
        predictions_saved=False, official_scoring=False)
    result["wall_plan"] = benchmark_reserved_seconds(result, cfg)
    if result["elapsed_seconds"] > 120:
        raise TimeoutError("GPU QA exceeded120-second bound")
    C.atomic_json(REPORT / "gpu_bounded_QA.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    del model, opt, cache
    torch.cuda.empty_cache()
    signal.setitimer(signal.ITIMER_REAL, 0)


def run(cfg, config_path, plan, source):
    if cfg["freeze_approved"] is not True or config_path is None:
        raise ValueError("Training requires an explicit frozen config")
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Development hard wall limit")))
    signal.setitimer(signal.ITIMER_REAL, cfg["max_seconds"])
    run_clock = time.monotonic()
    gpu = C.device_guard(P.GPU_UUID)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = load_data(cfg)
    cache = FixtureCache(data, plan)
    step, elapsed_prior, history, plateau = 0, 0.0, [], new_plateau()
    last = ART / "last.pt"
    if (REPORT / "result.json").exists():
        raise FileExistsError("Completed development run is preserved")
    qa_path = REPORT / "gpu_bounded_QA.json"
    warmup = json.loads(qa_path.read_text())
    if any(source.get(k) != v for k, v in warmup["source_identity"].items()):
        raise RuntimeError("Bounded GPU QA numerical source changed")
    if warmup["panel_sha256"] != plan["panel_sha256"]:
        raise RuntimeError("Bounded GPU QA panel identity changed")
    for key in ("positions_per_batch", "bs_per_batch", "frequencies_per_batch"):
        if warmup["config"][key] != cfg[key]:
            raise RuntimeError("Bounded GPU QA batch configuration changed")
    wall_plan = benchmark_reserved_seconds(warmup, cfg)
    C.seed_all(cfg["seed"])
    model = V.build(data); opt = make_optimizer(model); sampler = BalancedSampler(plan)
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        if ck["source_identity"] != source or ck["plan"] != plan:
            raise RuntimeError("Resume source/config/input mismatch")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        for state in opt.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cuda:0")
        sampler.load_state_dict(ck["sampler"])
        np.random.set_state(ck["numpy_global_rng"]); random.setstate(ck["python_rng"])
        torch.set_rng_state(ck["torch_rng"]); torch.cuda.set_rng_state_all(ck["cuda_rng"])
        step, elapsed_prior, history, plateau = ck["step"], ck["elapsed_seconds"], ck["history"], ck["plateau"]
        del ck
    elapsed = lambda: elapsed_prior + time.monotonic() - run_clock
    signal.setitimer(signal.ITIMER_REAL, max(0.001, cfg["max_seconds"] - elapsed()))
    C.atomic_json(REPORT / "frozen_run.json", dict(plan=plan, source_identity=source, GPU=gpu,
        optimizer="one Adam,8 field groups+1 shared gain; gain moments unique",
        early_stop="disabled; target or train wall only; validation never selects"))
    qa = first_batch_qa(model, data, cache, plan)
    C.atomic_json(REPORT / "first_batch_qa.json", qa)
    if step == 0:
        initial = diagnostic(model, data, cache, plan, sampler)
        update_plateau(plateau, initial, opt, cfg)
        event = dict(stage="initial", step=0, elapsed_seconds=elapsed(), diagnostic=initial,
                     plateau=copy.deepcopy(plateau))
        history.append(event); append_event(event)
    target_steps = cfg["per_field_updates"] * len(FIELDS)
    wall_exhausted, train_seconds = False, 0.0
    invocation_start_step = step
    numerical_best = min([x["diagnostic"]["aggregate_fixed_FIT"]["nmse"] for x in history], default=math.inf)
    while step < target_steps:
        if elapsed() >= wall_plan["effective_train_wall_seconds"]:
            wall_exhausted = True
            break
        task = sampler.draw(step)
        tick = time.monotonic()
        detail = do_update(model, opt, data, cache, task, cfg)
        step += 1
        train_seconds += time.monotonic() - tick
        append_event(dict(stage="train", step=step, task=task,
                          elapsed_seconds=elapsed(), **detail))
        monitor_due = step % (len(FIELDS) * cfg["monitor_every_per_field"]) == 0
        checkpoint_due = step % (len(FIELDS) * cfg["checkpoint_every_per_field"]) == 0
        if monitor_due:
            check_sources(source)
            diag = diagnostic(model, data, cache, plan, sampler)
            update_plateau(plateau, diag, opt, cfg)
            event = dict(stage="monitor", step=step, elapsed_seconds=elapsed(), diagnostic=diag,
                         plateau=copy.deepcopy(plateau))
            history.append(event); append_event(event)
            C.atomic_json(REPORT / "progress.json", event)
            score = diag["aggregate_fixed_FIT"]["nmse"]
            if score < numerical_best:
                numerical_best = score
                C.atomic_torch_save(ART / "best_numerical_FIT.pt",
                    snapshot(model, opt, sampler, step, plan, source, elapsed(), history, plateau))
        if checkpoint_due or monitor_due:
            C.atomic_torch_save(last, snapshot(model, opt, sampler, step, plan, source, elapsed(), history, plateau))
    check_sources(source)
    C.atomic_torch_save(last, snapshot(model, opt, sampler, step, plan, source, elapsed(), history, plateau))
    final_subset = diagnostic(model, data, cache, plan, sampler)
    full_tick = time.monotonic()
    full_fit = final_fit_fullH(model, data, cache, plan)
    full_fit_seconds = time.monotonic() - full_tick
    # The mandatory small validation diagnostic is much cheaper than full validation H.
    validation = final_validation_subset(model, data, plan)
    result = dict(schema=SCHEMA, status="complete", stop_reason="train_wall_budget" if wall_exhausted else "per_field_budget",
        updates=step, elapsed_seconds=elapsed(), training_seconds=train_seconds,
        mandatory_full_FIT_seconds=full_fit_seconds, hard_wall_exceeded=elapsed() > cfg["max_seconds"],
        train_step_seconds_current_invocation=train_seconds / max(step - invocation_start_step, 1),
        final_subset=final_subset, final_fullH_FIT=full_fit, final_validation_subset=validation,
        verdict=verdict(full_fit, plan, sampler), plan=plan, source_identity=source,
        plateau_states=plateau, coverage=sampler.coverage(), warmup=warmup, wall_plan=wall_plan,
        selection="final checkpoint; numerical best saved separately, no validation selection",
        final_prediction=dict(path=PATHS.relative(ART / "final_fit_prediction.npy"),
            sha256=C.sha256(ART / "final_fit_prediction.npy"),
            row_ids=plan["fixture_ids"], fit_RMS=data.scale, stored_units="raw H", shape=[16,256,4,192]),
        GPU=gpu, outer_truth_read=False, outer_predictions_written=False,
        official_metrics_scored=False, previous_results_changed=False, V16_changed=False)
    C.atomic_torch_save(ART / "final.pt", snapshot(model, opt, sampler, step, plan, source, elapsed(), history, plateau))
    C.atomic_json(REPORT / "result.json", result)
    append_event(dict(stage="complete", updates=step, elapsed_seconds=elapsed(), verdict=result["verdict"],
                      stop_reason=result["stop_reason"], hard_wall_exceeded=result["hard_wall_exceeded"]))
    signal.setitimer(signal.ITIMER_REAL, 0)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["plan", "check", "gpu-check", "run"], default="plan")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    cfg = load_config(args.config)
    data = load_data(cfg)
    plan = make_plan(data, cfg)
    source = source_identity(args.config)
    ART.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    if args.stage == "gpu-check":
        gpu_check(cfg, args.config, plan, source)
        return
    if args.stage == "check":
        result = sampler_cpu_check(plan, data)
        C.atomic_json(REPORT / "cpu_correctness_check.json", dict(result=result, source_identity=source,
                                                                 plan=plan))
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return
    if args.stage == "plan":
        C.atomic_json(REPORT / "plan.json", dict(plan=plan, source_identity=source))
        print(json.dumps(dict(stage="plan", plan=plan, source_identity=source), ensure_ascii=False), flush=True)
        return
    run(cfg, args.config, plan, source)

if __name__ == "__main__":
    try:
        main()
    except BaseException:
        if REPORT.exists():
            C.atomic_json(REPORT / "failure.json", dict(traceback=traceback.format_exc()))
        raise
