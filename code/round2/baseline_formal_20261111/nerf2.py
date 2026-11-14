"""Full eight-field, native192-head NeRF2: screen -> fresh refit -> raw outer H.

Task adaptation: position/geometry only, no query CSI. One loss is the mean
of per-position, FIT-power-normalized complex errors. Zero rows use a fixed
training floor. Raw evaluation applies only the permitted FIT RMS.
"""
from __future__ import annotations
import argparse, copy, fcntl, hashlib, json, math, os, random, signal, time, traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from project_paths import PATHS
from round2.paper_baseline_training_v1 import common as C, nerf2_runner_v2 as P
from round2.paper_baseline_training_v1 import nerf2_runner_v3 as V, field_fast as FF
from round2.paper_baseline_training_v1 import nerf2_native_frequency_heads_v4 as N
from round2.paper_baseline_training_v1 import nerf2_native_heads_joint_v5 as J

GPU = "GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0"
REPORT = C.REPORT / "formal_20261111/nerf2"
ART = C.ART / "formal_20261111/nerf2"
FIELDS = tuple(f"s{s}_b{b}_u{u}" for s in range(2) for b in range(2) for u in range(2))
STOP = False

def parts(key):
    return tuple(int(x[1:]) for x in key.split("_"))

def locations(fold=None, role=None):
    a, r = ART, REPORT
    if fold is not None:
        a, r = a/f"outer{fold}", r/f"outer{fold}"
    if role is not None:
        a, r = a/role, r/role
    a.mkdir(parents=True, exist_ok=True); r.mkdir(parents=True, exist_ok=True)
    return a, r

def emit(report, item):
    item = dict(utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **item)
    with (report/"events.jsonl").open("a") as f:
        f.write(json.dumps(item, allow_nan=False)+"\n")
    C.atomic_json(report/"progress.json", item)
    print(json.dumps(item, allow_nan=False), flush=True)

def identity(config_path):
    files = [Path(__file__), config_path, Path(C.__file__), Path(P.__file__),
             Path(V.__file__), Path(FF.__file__), Path(N.__file__), Path(J.__file__),
             PATHS.code_root/"round2/paper_baseline_adapters_v1/contract.py",
             PATHS.code_root/"round2/paper_baseline_adapters_v1/field_models.py",
             PATHS.code_root/"round2/paper_baseline_training_v1/field_runner.py",
             PATHS.root/"plus-pro/code/measured/full_h_adapters_v1/nerf2.py",
             PATHS.root/"plus-pro/code/third_party/paper_baselines/nerf2/model.py",
             PATHS.root/"plus-pro/code/third_party/paper_baselines/nerf2/renderer.py"]
    return {PATHS.relative(p): C.sha256(p) for p in files}

def verify_identity(source):
    for rel, digest in source.items():
        if C.sha256(PATHS.root/rel) != digest:
            raise RuntimeError("Frozen source/config changed: "+rel)

def cpu_tree(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu", copy=True)
    if isinstance(x, dict):
        return {k: cpu_tree(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return type(x)(cpu_tree(v) for v in x)
    return copy.deepcopy(x)

def optimizer_fingerprint(state):
    h = hashlib.sha256(); steps = []; tensors = 0
    h.update(json.dumps(state["param_groups"], sort_keys=True).encode())
    for pid in sorted(state["state"]):
        for name, value in sorted(state["state"][pid].items()):
            h.update(str((pid, name)).encode())
            if isinstance(value, torch.Tensor):
                z = value.detach().cpu().contiguous()
                h.update(str((str(z.dtype), list(z.shape))).encode())
                h.update(z.numpy().tobytes()); tensors += 1
                if name == "step":
                    steps.append(int(z.item()))
            else:
                h.update(repr(value).encode())
    return dict(sha256=h.hexdigest(), tensor_count=tensors, steps=steps)

def rng_state():
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(),
                numpy=np.random.get_state(), python=random.getstate())

def restore_rng(d):
    torch.set_rng_state(d["torch"]); torch.cuda.set_rng_state(d["cuda"])
    np.random.set_state(d["numpy"]); random.setstate(d["python"])

def build(data, seed):
    C.seed_all(seed)
    model = V.build(data)
    for key in FIELDS:
        core = model.fields[key]; field = core.field
        original = field.embed_pts_fn
        def fixed(x, fn=original):
            return fn(torch.cat([x[..., :3], torch.zeros_like(x[..., 3:4])], -1))
        field.embed_pts_fn = fixed
        old_a, old_s = field.attenuation_output, field.signal_output
        a = N.MIMOExtinction(old_a.in_features, 192).to(old_a.weight.device)
        s = nn.Linear(old_s.in_features, 384).to(old_s.weight.device)
        with torch.no_grad():
            for new, old in [(a, old_a), (s, old_s)]:
                new.weight[:192].copy_(old.weight[0:1].expand(192, -1))
                new.weight[192:].copy_(old.weight[1:2].expand(192, -1))
                new.bias[:192].fill_(old.bias[0]); new.bias[192:].fill_(old.bias[1])
        field.attenuation_output, field.signal_output = a, s
    if any(model.fields[k].field.signal_output.out_features != 384 for k in FIELDS):
        raise RuntimeError("All eight native192 heads required")
    return model

class Renderer:
    """Generalization of the verified JointRenderer over same-site positions.

    Position, UE and ray dimensions are independent. Training gradients are
    equivalent to the mean of sequential position renders. All actual192Hz
    propagation values,324rays and32samples remain present.
    """
    def __init__(self, data, rows, key):
        self.key = key; self.rows = np.asarray(rows, np.int64)
        self.site, self.bp, self.up = parts(key)
        if np.any(data.sites(self.rows) != self.site):
            raise ValueError("Field/site mapping mismatch")
        p = torch.tensor(data.positions(self.rows), device="cuda:0", dtype=torch.float64)
        site = torch.full((len(p),), self.site, device="cuda:0", dtype=torch.int64)
        self.world = data.geometry.ue_world_positions(p, site)[:, self.up*2:self.up*2+2]
        self.f = torch.tensor(data.frequencies(), device="cuda:0", dtype=torch.float64)

    def prepare(self, model):
        core = model.fields[self.key]; field = core.field; cfg = core.config
        nr, ns, nf = cfg.n_rays, cfg.n_samples, len(self.f)
        if (nr, ns, cfg.far_m, cfg.near_m, nf) != (324, 32, 810., 0., 192):
            raise RuntimeError("Physical renderer convention changed")
        centre = torch.tensor(cfg.world_center_m, device="cuda:0", dtype=torch.float64)
        directions = core.ray_directions.to(dtype=torch.float64)
        directions = directions/torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
        nports = len(self.rows)*2; ray_count = nports*nr
        segment = (cfg.far_m-cfg.near_m)/ns
        radii = cfg.near_m+segment*torch.arange(1, ns+1, device="cuda:0", dtype=torch.float64)
        points = (self.world.reshape(nports, 3)[:, None, None, :]
                  + directions[None, :, None, :]*radii[None, None, :, None]).reshape(ray_count, ns, 3)
        xyz = ((points-centre)/cfg.world_scale_m).float()
        frequency0 = torch.zeros((*xyz.shape[:-1], 1), device="cuda:0", dtype=torch.float32)
        view = directions.float()[None, :, None, :].expand(nports, nr, ns, 3).reshape(ray_count, ns, 3)
        attenuation, common, _ = FF._medium(field, torch.cat([xyz, frequency0], -1), view)
        tx = model.bound_bs_ports[self.site, self.bp].reshape(128, 3)
        encoded = field.embed_tx_fn(((tx-centre)/cfg.world_scale_m).float())
        first = field.signal_linears[0]; td = encoded.shape[-1]; cd = first.in_features-td
        shared = F.linear(common, first.weight[:, :cd], first.bias)
        txpart = F.linear(encoded, first.weight[:, cd:], None)
        if attenuation.shape[-1] != 384:
            raise RuntimeError("Native attenuation192 head missing")
        att = attenuation.reshape(ray_count, ns, 2, nf).permute(0, 3, 1, 2)
        shape = (ray_count, nf, ns)
        segments = torch.full((ns,), segment, device="cuda:0", dtype=torch.float64)
        trans, carrier = FF._shared_transport(att.reshape(-1, 2), shape, segments,
                                              self.f, cfg.propagation_sign, cfg.near_m)
        return shared, txpart, shape, trans, carrier

    def __call__(self, model, bs, prepared=None):
        shared, txpart, shape, trans, carrier = self.prepare(model) if prepared is None else prepared
        field = model.fields[self.key].field
        ix = torch.as_tensor(bs, device="cuda:0", dtype=torch.int64)
        x = F.relu(shared[None]+txpart.index_select(0, ix)[:, None])
        for layer in field.signal_linears[1:]:
            x = F.relu(layer(x))
        signal = field.signal_output(x)
        nr, ns, nf = 324, 32, 192
        sig = signal.reshape(len(bs), shape[0], ns, 2, nf).permute(0, 1, 4, 2, 3)
        rays = FF._factorized_integral(sig.reshape(len(bs), -1, 2), shape, trans, carrier)
        value = rays.reshape(len(bs), len(self.rows), 2, nr, nf).sum(3).permute(1, 0, 2, 3)
        return value*model.log_output_gain.exp()

class FitCache:
    def __init__(self, data, out):
        self.ids = np.asarray(data.fit_ids, np.int64); self.slot = np.full(4000, -1, np.int64)
        self.slot[self.ids] = np.arange(len(self.ids))
        self.value = torch.empty((len(self.ids), 256, 4, 192), device="cuda:0", dtype=torch.complex64)
        self.power = torch.empty((len(self.ids), 2, 2), device="cuda:0", dtype=torch.float64)
        for start in range(0, len(self.ids), 16):
            rows = self.ids[start:start+16]
            h = torch.tensor(data.read_fit(rows), device="cuda:0", dtype=torch.complex64)
            self.value[start:start+len(rows)] = h
            for b in range(2):
                for u in range(2):
                    self.power[start:start+len(rows), b, u] = h[:, b*128:b*128+128, u*2:u*2+2].abs().square().mean((1, 2, 3), dtype=torch.float64)
        power = self.power.cpu().numpy()
        C.atomic_npy(out/"FIT_field_mean_powers.npy", power, replace=True)
        C.atomic_npy(out/"FIT_row_ids.npy", self.ids, replace=True)
        self.power_sha256 = C.sha256(out/"FIT_field_mean_powers.npy")

    def target(self, rows, key, bs):
        _, b, u = parts(key)
        slot = torch.tensor(self.slot[np.asarray(rows)], device="cuda:0")
        if bool((slot < 0).any()):
            raise PermissionError("Training cache requested non-FIT labels")
        h = self.value.index_select(0, slot)
        bs = torch.as_tensor(bs, device="cuda:0", dtype=torch.int64)+b*128
        return h.index_select(1, bs)[:, :, u*2:u*2+2], self.power[slot, b, u]

class Schedule:
    """Balanced eight fields; complete shuffled position x BS coverage sweeps."""
    def __init__(self, data, batch, bs, seed):
        if batch < 2 or 128 % bs or batch % 2:
            raise ValueError("Even position batch>=2 and BS block dividing128 required")
        self.batch, self.bs = int(batch), int(bs)
        self.rng = np.random.default_rng(seed)
        self.available = {k: data.fit_ids[data.sites(data.fit_ids)==parts(k)[0]].copy() for k in FIELDS}
        self.counts = {k: 0 for k in FIELDS}; self.states = {}
        self.row_gradient_counts = {k: np.zeros(4000, np.int32) for k in FIELDS}
        self.BS_supervision_counts = {k: np.zeros((4000, 128), np.uint32) for k in FIELDS}
        for key in FIELDS:
            if len(self.available[key]) < batch:
                raise ValueError("Not enough same-site FIT positions")
        self.epoch_steps = {k: math.ceil(len(self.available[k])/batch)*(128//bs) for k in FIELDS}

    def peek(self, key):
        count = self.counts[key]; epoch = count//self.epoch_steps[key]
        if key not in self.states or self.states[key]["epoch"] != epoch:
            rows = self.rng.permutation(self.available[key])
            pairs = math.ceil(len(rows)/self.batch)
            rows = np.resize(rows, pairs*self.batch).reshape(pairs, self.batch)
            blocks = self.rng.permutation(128).reshape(128//self.bs, self.bs)
            self.states[key] = dict(epoch=epoch, rows=rows, blocks=blocks)
        state = self.states[key]; pairs = len(state["rows"])
        within = count % self.epoch_steps[key]
        return state["rows"][within % pairs].tolist(), sorted(state["blocks"][within//pairs].tolist())

    def commit(self, key, rows, bs):
        self.counts[key] += 1
        rr, bb = np.asarray(rows, np.int64), np.asarray(bs, np.int64)
        np.add.at(self.row_gradient_counts[key], rr, 1)
        np.add.at(self.BS_supervision_counts[key], (rr[:, None], bb[None, :]), 1)

    def coverage(self):
        result = {}
        for key in FIELDS:
            rows = self.available[key]; c = self.BS_supervision_counts[key][rows]
            positive = c > 0; unique = int(positive.sum())
            total = int(c.sum(dtype=np.uint64)); required = len(rows)*128
            result[key] = dict(actual_optimizer_updates=self.counts[key],
                unique_gradient_positions=int((self.row_gradient_counts[key][rows] > 0).sum()),
                gradient_position_draws=int(self.row_gradient_counts[key][rows].sum()),
                permitted_FIT_positions=len(rows), unique_position_BS_links=unique,
                required_position_BS_links=required, completely_covered_positions=int(positive.all(1).sum()),
                total_position_BS_draws=total, repeated_position_BS_draws=total-unique,
                minimum_link_repetitions=int(c.min()), maximum_link_repetitions=int(c.max()),
                equivalent_complete_BS_coverage=total/required,
                supervised_complex_elements=total*2*192)
        return result

    def state_dict(self):
        return copy.deepcopy(dict(batch=self.batch, bs=self.bs, rng=self.rng.bit_generator.state,
                                  counts=self.counts, states=self.states, epoch_steps=self.epoch_steps,
                                  row_gradient_counts=self.row_gradient_counts,
                                  BS_supervision_counts=self.BS_supervision_counts))

    def load_state_dict(self, state):
        if (state["batch"], state["bs"], state["epoch_steps"]) != (self.batch, self.bs, self.epoch_steps):
            raise ValueError("Resume batch/coverage differs")
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])
        self.counts = copy.deepcopy(state["counts"]); self.states = copy.deepcopy(state["states"])
        self.row_gradient_counts = copy.deepcopy(state["row_gradient_counts"])
        self.BS_supervision_counts = copy.deepcopy(state["BS_supervision_counts"])

def train_step(model, opt, data, cache, rows, key, bs, floor, objective="relative", audit_components=False):
    model.train(); opt.zero_grad(set_to_none=True)
    pred = Renderer(data, rows, key)(model, bs)
    truth, power = cache.target(rows, key, bs)
    native = .5*(pred-truth).abs().square().mean((1, 2, 3))
    vector = native if objective=="native_warmup" else native/power.clamp_min(floor)
    loss = vector.mean()
    zero = power==0
    component_gradients = {}; component_seconds = 0.
    if audit_components and bool(zero.any()) and bool((~zero).any()):
        torch.cuda.synchronize(); component_start = time.monotonic()
        params = list(model.fields[key].parameters())+[model.log_output_gain]
        for label, mask in [("zero", zero), ("nonzero", ~zero)]:
            component = vector[mask].sum()/len(rows)
            gradients = torch.autograd.grad(component, params, retain_graph=True, allow_unused=True)
            norm = math.sqrt(sum(float(g.detach().double().square().sum()) for g in gradients if g is not None))
            component_gradients[label] = dict(norm=norm, finite=math.isfinite(norm))
        torch.cuda.synchronize(); component_seconds = time.monotonic()-component_start
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite relative complex loss")
    loss.backward()
    active = [p for p in model.parameters() if p.grad is not None]
    grad = float(torch.nn.utils.clip_grad_norm_(active, 10., error_if_nonfinite=True))
    opt.step()
    return dict(loss=float(loss.detach()), gradient_norm=grad, scalar_labels=pred.numel(),
                zero_field_positions=int(zero.sum()), field=key, objective=objective,
                zero_loss_contribution=float(vector[zero].sum().detach()/len(rows)) if bool(zero.any()) else 0.,
                nonzero_loss_contribution=float(vector[~zero].sum().detach()/len(rows)) if bool((~zero).any()) else 0.,
                component_gradient_QA=component_gradients,
                component_gradient_QA_seconds=component_seconds)

@torch.no_grad()
def predict_full(model, data, rows, bs_chunk=8):
    model.eval(); outputs = []
    for row in np.asarray(rows, np.int64):
        s = int(data.sites([row])[0])
        out = torch.empty((1, 256, 4, 192), device="cuda:0", dtype=torch.complex64)
        for b in range(2):
            for u in range(2):
                r = Renderer(data, [row], f"s{s}_b{b}_u{u}"); prepared = r.prepare(model)
                blocks = [r(model, list(range(start, start+bs_chunk)), prepared)
                          for start in range(0, 128, bs_chunk)]
                out[:, b*128:b*128+128, u*2:u*2+2] = torch.cat(blocks, 1)
                del prepared, blocks
        if not bool(torch.isfinite(out).all()):
            raise FloatingPointError("Nonfinite full-H prediction")
        outputs.append(out)
    return torch.cat(outputs)

@torch.no_grad()
def validation(model, data, ids, bs_chunk=8):
    ids = np.asarray(ids, np.int64)
    if set(ids) & set(data.outer_ids):
        raise PermissionError("Outer truth cannot enter selection")
    truth = data.read_validation(ids)
    se = te = pe = 0.; points = []
    for i, row in enumerate(ids):
        pred = predict_full(model, data, [row], bs_chunk).cpu().numpy().astype(np.complex128)
        target = truth[i:i+1].astype(np.complex128)
        error = float(np.sum(np.abs(pred-target)**2))
        energy = float(np.sum(np.abs(target)**2)); predicted = float(np.sum(np.abs(pred)**2))
        points.append(dict(row_id=int(row), site=int(data.sites([row])[0]),
                           nmse=error/energy if energy else None,
                           energy_ratio=predicted/energy if energy else None, zero_truth=energy==0))
        se += error; te += energy; pe += predicted
    if not te:
        raise ValueError("Validation panel has zero total energy")
    return dict(nmse=se/te, energy_ratio=pe/te, per_position=points,
                shape=[len(ids), 256, 4, 192], all192frequencies=True, all128BS_per_field=True,
                squared_error=se, truth_energy=te, row_ids=ids.tolist())

def snapshot(model, opt, schedule, step, meta, source, data, cfg, active_seconds):
    return dict(schema="nerf2-formal192-eightfield-checkpoint-v1",
                model=cpu_tree(model.state_dict()), optimizer=cpu_tree(opt.state_dict()),
                schedule=schedule.state_dict(), rng=cpu_tree(rng_state()), step=step,
                per_field_updates=copy.deepcopy(schedule.counts), meta=copy.deepcopy(meta),
                source_identity=source, config=cfg, data=data.info(), active_seconds=active_seconds,
                all8fields_native192=True, original_paper_exact_reproduction=False)

def restore(ck, model, opt, schedule, source, data):
    if ck["source_identity"] != source or ck["data"]["fit_ids_sha256"] != C.digest_ids(data.fit_ids):
        raise RuntimeError("Resume source/FIT mismatch")
    expected = optimizer_fingerprint(ck["optimizer"])
    model.load_state_dict(ck["model"]); opt.load_state_dict(copy.deepcopy(ck["optimizer"]))
    schedule.load_state_dict(ck["schedule"]); restore_rng(ck["rng"])
    actual = optimizer_fingerprint(opt.state_dict())
    if actual != expected or optimizer_fingerprint(ck["optimizer"]) != expected:
        raise RuntimeError("Full optimizer state restore differs")
    for pid, st in opt.state_dict()["state"].items():
        for name, value in st.items():
            old = ck["optimizer"]["state"][pid][name]
            if isinstance(value, torch.Tensor) and value.device.type == "cpu" and value.data_ptr() == old.data_ptr():
                raise RuntimeError("CPU optimizer state aliases immutable parent")
    if sum(schedule.counts.values()) != ck["step"]:
        raise RuntimeError("Actual joint update counts inconsistent")
    return int(ck["step"]), copy.deepcopy(ck["meta"]), float(ck["active_seconds"])

def set_lr(opt, cfg, field_updates, warmup_steps):
    fraction = min(1., max(0., field_updates/max(1, warmup_steps)))
    factor = cfg["low_lr_factor"]+(1-cfg["low_lr_factor"])*.5*(1+math.cos(math.pi*fraction))
    for group, initial in zip(opt.param_groups, cfg["initial_lrs"]):
        group["lr"] = initial*factor

def training_coverage(schedule, cache):
    result = schedule.coverage()
    for key in FIELDS:
        _, b, u = parts(key); rows = schedule.available[key]
        slots = torch.tensor(cache.slot[rows], device="cuda:0")
        zero = (cache.power[slots, b, u]==0).cpu().numpy()
        draws = schedule.row_gradient_counts[key][rows]
        result[key].update(zero_field_positions_available=int(zero.sum()),
            zero_field_positions_updated=int((draws[zero]>0).sum()),
            zero_position_gradient_draws=int(draws[zero].sum()))
    return result

def bounds(schedule, cfg):
    epoch = max(schedule.epoch_steps.values())
    return (math.ceil(epoch*cfg["min_coverage_epochs"]),
            math.ceil(epoch*cfg["max_coverage_epochs"]),
            max(1, math.ceil(epoch*cfg["validation_coverage_interval"])))

def stage(fold, role, cfg, source, selection=None):
    a, report = locations(fold, role); result_path = report/"result.json"
    if result_path.exists():
        d = json.loads(result_path.read_text())
        if d.get("status") != role+"_complete" or d["source_identity"] != source:
            raise RuntimeError("Incomplete/incompatible preserved stage result")
        if C.sha256(PATHS.root/d["checkpoint"]) != d["checkpoint_sha256"]:
            raise RuntimeError("Frozen checkpoint changed")
        return d
    data = C.FoldData(fold, role, "nerf2")
    seed = cfg["seed"]+1009*fold
    model = build(data, seed); opt = P.optimizer(model)
    schedule = Schedule(data, cfg["position_batch"], cfg["BS_batch"], seed+97)
    minimum, maximum, interval = bounds(schedule, cfg)
    if role == "refit":
        if selection is None:
            raise ValueError("Refit requires frozen screen selection")
        maximum = max(max(schedule.epoch_steps.values()),
                      math.ceil(selection["selected_per_field_updates"]*len(data.fit_ids)/selection["FIT_count"]))
        minimum = maximum
    cache = FitCache(data, a)
    meta = dict(history=[], best_nmse=None, best_step=None, best_per_field=None,
                anchor=None, stale=0, stop_reason=None)
    step = 0; previous_seconds = 0.
    last = a/"last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        step, meta, previous_seconds = restore(ck, model, opt, schedule, source, data)
        if meta["stop_reason"] in ("signal", "registered_wall_cap"):
            meta.setdefault("interruptions", []).append(dict(step=step, reason=meta["stop_reason"], seconds=previous_seconds))
            meta["stop_reason"] = None
        del ck
    tick = time.monotonic(); last_progress = tick; last_save = tick
    def elapsed():
        return previous_seconds+time.monotonic()-tick
    def save(path=last):
        C.atomic_torch_save(path, snapshot(model, opt, schedule, step, meta, source, data, cfg, elapsed()))
    emit(report, dict(event="stage_start_or_resume", fold=fold, role=role, step=step,
         min_updates_per_field=minimum, max_updates_per_field=maximum,
         validation_every_per_field=interval, updates_per_coverage_epoch=schedule.epoch_steps,
         actual_per_field_updates=schedule.counts, actual_coverage=training_coverage(schedule, cache), FIT_count=len(data.fit_ids), validation_ids=data.probe_ids.tolist() if role=="screen" else []))
    try:
        while step < 8*maximum:
            if STOP or elapsed() > cfg["stage_wall_cap_seconds"]:
                meta["stop_reason"] = "signal" if STOP else "registered_wall_cap"
                save(); emit(report, dict(event="paused", reason=meta["stop_reason"], step=step))
                raise InterruptedError(meta["stop_reason"])
            key = FIELDS[step % 8]; rows, bs = schedule.peek(key)
            warm = math.ceil(schedule.epoch_steps[key]*cfg["native_warmup_coverage_epochs"])
            set_lr(opt, cfg, schedule.counts[key], warm)
            objective = "native_warmup" if schedule.counts[key] < warm else "relative"
            detail = train_step(model, opt, data, cache, rows, key, bs, cfg["power_floor"], objective)
            schedule.commit(key, rows, bs); step += 1
            balanced = step % 8 == 0; per_field = step//8
            if role == "screen" and balanced and (per_field % interval == 0 or per_field == maximum):
                torch.cuda.synchronize(); vt = time.monotonic()
                stats = validation(model, data, data.probe_ids, cfg["inference_BS_chunk"])
                score = stats["nmse"]; eligible = per_field >= minimum
                if eligible and any(v["completely_covered_positions"] != v["permitted_FIT_positions"]
                                    for v in schedule.coverage().values()):
                    raise RuntimeError("Selection before every field has complete FIT x128BS coverage")
                if eligible:
                    if meta["best_nmse"] is None or score < meta["best_nmse"]:
                        meta["best_nmse"], meta["best_step"], meta["best_per_field"] = score, step, per_field
                        save(a/"best.pt")
                    if meta["anchor"] is None or score < meta["anchor"]-cfg["min_delta"]:
                        meta["anchor"], meta["stale"] = score, 0
                    else:
                        meta["stale"] += 1
                event = dict(event="validation", step=step, per_field_updates=per_field,
                             eligible_for_selection=eligible, statistics=stats, best_step=meta["best_step"],
                             stale=meta["stale"], seconds=elapsed(), validation_seconds=time.monotonic()-vt)
                meta["history"].append(event); emit(report, event); save()
                last_save = time.monotonic()
                if eligible and meta["stale"] >= cfg["patience"]:
                    meta["stop_reason"] = "registered_validation_patience"; break
            now = time.monotonic()
            if now-last_progress >= 30:
                emit(report, dict(event="training", fold=fold, role=role, step=step,
                    per_field_updates=schedule.counts, actual_coverage=training_coverage(schedule, cache), seconds=elapsed(), last_batch=detail,
                    lrs=[g["lr"] for g in opt.param_groups],
                    peak_cuda_bytes=torch.cuda.max_memory_allocated()))
                last_progress = now
            if now-last_save >= cfg["checkpoint_seconds"]:
                save(); last_save = time.monotonic()
        meta["stop_reason"] = meta["stop_reason"] or "registered_max_coverage"
        if role == "screen" and meta["best_step"] is None:
            raise RuntimeError("No eligible full-panel validation checkpoint")
        coverage = training_coverage(schedule, cache)
        if any(v["unique_position_BS_links"] != v["required_position_BS_links"] for v in coverage.values()):
            raise RuntimeError("Final stage lacks complete FIT x128BS coverage")
        actual_clocks = {}
        for key in FIELDS:
            clocks = sorted({int(opt.state[p]["step"]) for p in model.fields[key].parameters() if "step" in opt.state[p]})
            if clocks != [schedule.counts[key]]:
                raise RuntimeError(("Adam time disagrees with actual per-field updates", key, clocks, schedule.counts[key]))
            actual_clocks[key] = clocks
        if int(opt.state[model.log_output_gain]["step"]) != step:
            raise RuntimeError("Shared gain Adam clock differs from global joint updates")
        final = a/"final.pt"; save(final); save()
        chosen = a/"best.pt" if role == "screen" else final
        d = dict(status=role+"_complete", outer=fold, role=role, source_identity=source,
                 method_config_sha256=source[cfg["_config_rel"]], config=cfg,
                 actual_updates=step, per_field_updates=schedule.counts, actual_coverage=coverage,
                 verified_Adam_field_step_values=actual_clocks, verified_Adam_gain_steps=step,
                 selected_step=meta["best_step"] if role=="screen" else None,
                 selected_per_field_updates=meta["best_per_field"] if role=="screen" else None,
                 selected_nmse=meta["best_nmse"] if role=="screen" else None,
                 FIT_count=len(data.fit_ids), validation_panel=data.probe_ids.tolist() if role=="screen" else [],
                 checkpoint=PATHS.relative(chosen), checkpoint_sha256=C.sha256(chosen),
                 final_checkpoint=PATHS.relative(final), final_checkpoint_sha256=C.sha256(final),
                 stop_reason=meta["stop_reason"], seconds=elapsed(), history=meta["history"],
                 peak_cuda_bytes=torch.cuda.max_memory_allocated(), data=data.info(),
                 zero_power_floor=cfg["power_floor"], FIT_powers_sha256=cache.power_sha256,
                 no_outer_truth_reads=True, truth_based_alignment=False,
                 selected_screen=selection, geometry=V.metadata())
        C.atomic_json(result_path, d); emit(report, dict(event="stage_complete", fold=fold, role=role,
                actual_updates=step, seconds=d["seconds"], checkpoint=d["checkpoint"]))
        return d
    except BaseException:
        save()
        with (report/"failure.log").open("a") as f:
            f.write(traceback.format_exc()+"\n")
        raise
    finally:
        del cache, model, opt
        torch.cuda.empty_cache()

@torch.no_grad()
def outer_prediction(fold, cfg, source, selected, refit):
    a, report = locations(fold, "prediction"); mp = a/"manifest.json"
    if mp.exists():
        d = json.loads(mp.read_text())
        if d["source_identity"] != source or d["checkpoint_sha256"] != refit["checkpoint_sha256"]:
            raise RuntimeError("Existing prediction lineage differs")
        if C.sha256(a/"channels.npy") != d["sha256"]:
            raise RuntimeError("Existing raw prediction hash differs")
        return d
    data = C.FoldData(fold, "refit", "nerf2")
    model = build(data, cfg["seed"]+1009*fold)
    ck = torch.load(PATHS.root/refit["checkpoint"], map_location="cpu", weights_only=False)
    if ck["source_identity"] != source:
        raise RuntimeError("Refit source/config mismatch")
    model.load_state_dict(ck["model"]); del ck; model.eval()
    if len(data.outer_ids) != (103, 71, 63, 110, 153)[fold]:
        raise ValueError("Original outer fold changed")
    work = a/"channels_work.npy"; progress_path = report/"prediction_work.json"
    shape = (len(data.outer_ids), 256, 4, 192)
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress["source_identity"] != source or progress["checkpoint_sha256"] != refit["checkpoint_sha256"]:
            raise RuntimeError("Partial prediction source/parent differs")
        array = np.load(work, mmap_mode="r+", allow_pickle=False)
    else:
        if work.exists():
            raise FileExistsError(work)
        array = np.lib.format.open_memmap(work, mode="w+", dtype=np.complex64, shape=shape)
        progress = dict(completed=0, source_identity=source, checkpoint_sha256=refit["checkpoint_sha256"])
    if array.shape != shape or array.dtype != np.complex64:
        raise ValueError("Partial prediction header differs")
    if (data.inputs.identity != refit["data"]["input_identity"]
        or C.digest_ids(data.fit_ids) != refit["data"]["fit_ids_sha256"]
        or data.scale != refit["data"]["scale"]):
        raise RuntimeError("Inference context differs from actual refit data")
    first = int(progress["completed"]); tick = time.monotonic()
    for i in range(first, len(data.outer_ids)):
        if STOP:
            raise InterruptedError("Signal during outer prediction")
        value = predict_full(model, data, data.outer_ids[i:i+1], cfg["inference_BS_chunk"])*data.scale
        array[i] = value[0].cpu().numpy(); array.flush()
        progress["completed"] = i+1; C.atomic_json(progress_path, progress)
        if (i+1) % 5 == 0:
            emit(report, dict(event="outer_prediction_only", completed=i+1, total=len(array),
                              seconds=time.monotonic()-tick, query_truth_read=False))
    del array
    # Atomic publication. The work progress is retained as execution evidence.
    work.replace(a/"channels.npy")
    C.atomic_npy(a/"row_ids.npy", data.outer_ids)
    d = dict(schema="canonical-baseline-prediction-v1", method="nerf2", outer=fold, role="refit",
             input_identity=data.inputs.identity, row_ids=data.outer_ids.tolist(),
             row_ids_sha256=C.digest_ids(data.outer_ids), shape=list(shape), dtype="complex64",
             kind="channel", path=PATHS.relative(a/"channels.npy"), sha256=C.sha256(a/"channels.npy"),
             prediction_namespace="canonical_train_row_ids", trained=True, query_truth_read=False,
             source_identity=source, method_config_sha256=source[cfg["_config_rel"]],
             checkpoint=refit["checkpoint"], checkpoint_sha256=refit["checkpoint_sha256"],
             screen_selection=selected, per_field_updates=refit["per_field_updates"],
             fit_data=refit["data"], query_context=data.info(), phase_delay_amplitude_alignment=False,
             metadata=dict(source_identity=source, adapter_version="formal_native192_eight_fields",
                           config=cfg, original_paper_exact_reproduction=False),
             output_scale="one permitted refit FIT-RMS; no query correction", all8fields_trained=True)
    C.atomic_json(mp, d); C.atomic_json(report/"result.json", dict(status="complete", manifest=d))
    del model; torch.cuda.empty_cache()
    return d

def qa_gradients(model, data, cache, floor):
    """Batched vs independently rendered sequential positions for all8 fields."""
    records = []; bs = [0, 17, 30, 47, 64, 81, 98, 127]
    for key in FIELDS:
        s, b, u = parts(key); allowed = data.fit_ids[data.sites(data.fit_ids)==s]
        rows = allowed[:2].tolist(); params = list(model.fields[key].parameters())+[model.log_output_gain]
        truth, power = cache.target(rows, key, bs)
        renderer = Renderer(data, rows, key)
        batched = renderer(model, bs)
        loss = .5*((batched-truth).abs().square().mean((1, 2, 3))/power.clamp_min(floor)).mean()
        bg = torch.autograd.grad(loss, params, allow_unused=True)
        seq = []; sg = [torch.zeros_like(p) for p in params]
        for i, row in enumerate(rows):
            old = J.JointRenderer(data, dict(row_id=row))
            # Old verified implementation is bound to site0/pol0. Substitute
            # only the explicit actual field, TX block and UE world ports.
            old.world = Renderer(data, [row], key).world
            view = SimpleNamespace(fields={"s0_b0_u0": model.fields[key]},
                 bound_bs_ports=model.bound_bs_ports[s:s+1, b:b+1], log_output_gain=model.log_output_gain)
            val = old(view, bs, [0, 1]); seq.append(val.detach())
            part = .5*(val-truth[i:i+1]).abs().square().mean()/power[i].clamp_min(floor)/len(rows)
            gs = torch.autograd.grad(part, params, allow_unused=True)
            sg = [a+(g if g is not None else torch.zeros_like(p)) for a, g, p in zip(sg, gs, params)]
        expected = torch.cat(seq)
        forward = float((batched.detach()-expected).norm()/expected.norm().clamp_min(1e-30))
        num = sum(float((a-(g if g is not None else torch.zeros_like(a))).double().square().sum()) for a, g in zip(sg, bg))
        den = sum(float(a.double().square().sum()) for a in sg)
        gradient = math.sqrt(num/max(den, 1e-60))
        finite = all(g is None or bool(torch.isfinite(g).all()) for g in bg)
        if forward > 1e-4 or gradient > 1e-3 or not finite:
            raise AssertionError((key, forward, gradient, finite))
        records.append(dict(field=key, rows=rows, forward_relative_L2=forward,
                            gradient_relative_L2=gradient, finite=finite,
                            frequency_count=192, UE_count=2, BS_ids=bs))
    return records

def smoke(cfg, source, family):
    a, report = locations(None, "smoke_"+family)
    if (report/"result.json").exists():
        raise FileExistsError("Smoke already preserved; use its result")
    data = C.FoldData(0, "screen", "nerf2"); cache = FitCache(data, a)
    model = build(data, cfg["seed"]); opt = P.optimizer(model)
    schedule = Schedule(data, cfg["position_batch"], cfg["BS_batch"], cfg["seed"]+97)
    qa = qa_gradients(model, data, cache, cfg["power_floor"])
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    tick = time.monotonic(); elements = 0; detail = None; step_records = []
    for step in range(20):
        key = FIELDS[step % 8]; rows, bs = schedule.peek(key)
        warm = math.ceil(schedule.epoch_steps[key]*cfg["native_warmup_coverage_epochs"])
        set_lr(opt, cfg, schedule.counts[key], warm)
        detail = train_step(model, opt, data, cache, rows, key, bs, cfg["power_floor"],
                            "native_warmup", audit_components=True)
        step_records.append(detail)
        elements += detail["scalar_labels"]; schedule.commit(key, rows, bs)
    torch.cuda.synchronize(); seconds = time.monotonic()-tick
    trained_peak = torch.cuda.max_memory_allocated()
    audit_seconds = sum(z["component_gradient_QA_seconds"] for z in step_records)
    net_seconds = seconds-audit_seconds
    saved = a/"smoke.pt"; meta = dict(smoke=True)
    C.atomic_torch_save(saved, snapshot(model, opt, schedule, 20, meta, source, data, cfg, seconds))
    before = {}
    with torch.no_grad():
        for key in FIELDS:
            row = data.fit_ids[data.sites(data.fit_ids)==parts(key)[0]][:1]
            before[key] = Renderer(data, row, key)(model, [0, 17, 64, 127]).cpu()
    expected_fp = optimizer_fingerprint(opt.state_dict())
    del model, opt; torch.cuda.empty_cache()
    model = build(data, cfg["seed"]); opt = P.optimizer(model)
    ck = torch.load(saved, map_location="cpu", weights_only=False)
    _, _, _ = restore(ck, model, opt, schedule, source, data)
    delta = 0.
    with torch.no_grad():
        for key in FIELDS:
            row = data.fit_ids[data.sites(data.fit_ids)==parts(key)[0]][:1]
            now = Renderer(data, row, key)(model, [0, 17, 64, 127]).cpu()
            delta = max(delta, float((now-before[key]).abs().max()))
    if delta != 0 or optimizer_fingerprint(opt.state_dict()) != expected_fp:
        raise AssertionError("Clean reload changed outputs/optimizer")
    full = predict_full(model, data, [data.fit_ids[0]], cfg["inference_BS_chunk"])
    if full.shape != (1, 256, 4, 192):
        raise AssertionError("Full-H output shape")
    d = dict(status="passed", source_identity=source, config=cfg, actual_optimizer_updates=20,
             all8field_updates=schedule.counts, batched_vs_sequential_QA=qa, step_records=step_records,
             seconds=seconds, component_gradient_QA_seconds=audit_seconds,
             optimizer_path_seconds_excluding_extra_component_QA=net_seconds,
             complex_elements=elements, complex_elements_per_second=elements/seconds,
             complex_elements_per_second_without_extra_component_QA=elements/net_seconds,
             full_field_matrices_per_second=elements/(128*2*192)/seconds,
             full_field_matrices_per_second_without_extra_component_QA=elements/(128*2*192)/net_seconds,
             peak_cuda_bytes=trained_peak, clean_reload_max_abs=delta,
             all_optimizer_tensors_restored=True, optimizer_fingerprint=expected_fp,
             smoke_model_discarded_for_formal_fresh_initialization=True,
             full_H_shape=list(full.shape), formal_outer_prediction=False, outer_truth_read=False,
             data=data.info())
    C.atomic_json(report/"result.json", d); print(json.dumps(d), flush=True)
    return d

def cpu_check(cfg, source):
    import ast
    ast.parse(Path(__file__).read_text())
    data = C.FoldData(0, "screen", "nerf2")
    sched = Schedule(data, cfg["position_batch"], cfg["BS_batch"], cfg["seed"]+97)
    for i in range(24):
        key = FIELDS[i % 8]; rows, bs = sched.peek(key)
        if set(rows)-set(data.fit_ids) or len(rows)!=cfg["position_batch"] or len(bs)!=cfg["BS_batch"]:
            raise AssertionError("Training sampler isolation/shape")
        if not np.all(data.sites(rows)==parts(key)[0]):
            raise AssertionError("Training sampler/site")
        sched.commit(key, rows, bs)
    state = sched.state_dict(); other = Schedule(data, cfg["position_batch"], cfg["BS_batch"], 999)
    other.load_state_dict(state)
    for key in FIELDS:
        if sched.peek(key) != other.peek(key):
            raise AssertionError("CPU sampler restoration")
    # An actual CPU Adam roundtrip verifies deep copied scalar clocks and moments.
    m = nn.Linear(2, 2); opt = torch.optim.Adam(m.parameters())
    opt.zero_grad(); m(torch.ones(1, 2)).square().sum().backward(); opt.step()
    saved = cpu_tree(opt.state_dict()); fp = optimizer_fingerprint(saved)
    n = nn.Linear(2, 2); no = torch.optim.Adam(n.parameters()); no.load_state_dict(copy.deepcopy(saved))
    if optimizer_fingerprint(no.state_dict()) != fp:
        raise AssertionError("CPU optimizer roundtrip")
    no.zero_grad(); n(torch.ones(1, 2)).square().sum().backward(); no.step()
    if optimizer_fingerprint(saved) != fp:
        raise AssertionError("Immutable parent CPU state changed")
    a, r = locations()
    result = dict(status="passed", source_identity=source, config=cfg, sampler_restored=True,
                  full_optimizer_roundtrip=True, parent_state_immutable=True,
                  all_fields=list(FIELDS), step_bounds_per_field=bounds(sched, cfg),
                  canonical_outer_counts=[len(C.FoldData(k, "refit", "nerf2").outer_ids) for k in range(5)],
                  CUDA_initialized=torch.cuda.is_initialized(), no_training_data_labels_read=True)
    if result["canonical_outer_counts"] != [103, 71, 63, 110, 153]:
        raise AssertionError("Original canonical split changed")
    C.atomic_json(r/"cpu_check.json", result); print(json.dumps(result), flush=True)
    return result

def default_config():
    return dict(schema="nerf2-formal192-eightfields-v1", seed=42,
       position_batch=8, BS_batch=8, inference_BS_chunk=8, power_floor=1e-6, native_warmup_coverage_epochs=.5,
       initial_lrs=[1e-4, 1e-3], low_lr_factor=.2, min_coverage_epochs=1.,
       max_coverage_epochs=2., validation_coverage_interval=.25,
       validation_selection="global complex NMSE on original fixed16 development locations; all256BS x4UE x192F",
       min_delta=.001, patience=6, checkpoint_seconds=300,
       stage_wall_cap_seconds=172800, min_free_GiB=48,
       initialization="fresh same-seed all8native192; no old field/model stitching",
       training_loss=".5*mean_i[MSE_complex_i/max(FIT_field_mean_power_i,1e-6)]",
       sampling="round-robin8fields; withinfield complete shuffled position x128BS coverage sweeps",
       refit="fresh initialization; fixed selected screen per-field update budget times refit/screen FIT count ratio",
       LR="cosine initial_lrs to factor.2 during first .5 coverage native warmup; fixed factor.2 for relative loss thereafter",
       outer_truth_use="none; root scoring only after all5 predictions frozen")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--mode", choices=["cpu-check", "smoke", "run", "prediction-only"], required=True)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--family", default="batch8_bs8")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text()); cfg["_config_rel"] = PATHS.relative(args.config)
    if cfg["schema"] != "nerf2-formal192-eightfields-v1":
        raise ValueError("Frozen config schema required")
    if args.mode in ("run", "prediction-only") and cfg.get("formal_frozen") is not True:
        raise ValueError("Long stages require the registered formal_frozen config")
    source = identity(args.config)
    if args.mode == "cpu-check":
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("CPU checks need empty CUDA mask")
        return cpu_check(cfg, source)
    global STOP
    def stopping(_signal, _frame):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, stopping); signal.signal(signal.SIGINT, stopping)
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    gpu = C.device_guard(GPU)
    free, _ = torch.cuda.mem_get_info()
    if free < cfg["min_free_GiB"]*(1 << 30):
        raise RuntimeError("Assigned H100 has less than registered free memory")
    _, report = locations()
    C.atomic_json(report/"device.json", dict(gpu, free_bytes_before=free))
    with (report/"execution.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.mode == "smoke":
            return smoke(cfg, source, args.family)
        completed = []
        for fold in map(int, args.folds.split(",")):
            verify_identity(source)
            selected = stage(fold, "screen", cfg, source) if args.mode=="run" else json.loads((REPORT/f"outer{fold}/screen/result.json").read_text())
            refit = stage(fold, "refit", cfg, source, selected) if args.mode=="run" else json.loads((REPORT/f"outer{fold}/refit/result.json").read_text())
            manifest = outer_prediction(fold, cfg, source, selected, refit)
            completed.append(dict(outer=fold, manifest=PATHS.relative(ART/f"outer{fold}/prediction/manifest.json"),
                                  sha256=manifest["sha256"]))
            C.atomic_json(report/"completed_folds.json", dict(status="running", completed=completed))
        if {x["outer"] for x in completed} == set(range(5)):
            C.atomic_json(report/"all_folds_complete.json", dict(status="complete", completed=completed,
                          formal500_scored=False, outer_truth_read=False))
if __name__ == "__main__":
    try:
        main()
    except BaseException:
        REPORT.mkdir(parents=True, exist_ok=True)
        with (REPORT/"failure.log").open("a") as f:
            f.write(traceback.format_exc()+"\n")
        raise

