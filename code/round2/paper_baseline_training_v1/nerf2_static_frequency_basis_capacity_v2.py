"""FIT-only CPU linear capacity diagnostic for a static-frequency NeRF2 network.

No network, optimizer, GPU, validation labels, or query labels are accessed.
Free complex coefficients are a relaxed representation diagnostic. They do not
show that the physical neural renderer can learn the fitted coefficients.
"""
from __future__ import annotations
import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("This CPU diagnostic requires an empty CUDA device mask")
import numpy as np
from project_paths import PATHS

EVIDENCE = PATHS.analysis / "results/round2_paper_four_baselines_v1/fit_repair_paper_v2_20261101"
OUT = EVIDENCE / "nerf2_static_frequency_basis_capacity"
RCONDS = (1e-6, 1e-8, 1e-10, 1e-12)
C0 = 299792458.0


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(8 << 20), b""):
            h.update(part)
    return h.hexdigest()


def literal_assignment(path, name):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("Missing literal: " + name)


def field_configuration(path):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "NERF2_CONFIG" for t in node.targets):
            return {k.arg: ast.literal_eval(k.value) for k in node.value.keywords}
    raise ValueError("NERF2_CONFIG not found")


def config_overrides(path):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CONFIG" for t in node.targets):
            return {ast.literal_eval(k): ast.literal_eval(v) for k, v in zip(node.value.keys, node.value.values) if k is not None}
    raise ValueError("CONFIG overrides not found")


def pipeline_frequency(path):
    wanted = {"physical_carrier_frequency_hz", "subcarrier_spacing_hz", "subcarriers"}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.ClassDef) and node.name == "PipelineConfig":
            d = {n.target.id: ast.literal_eval(n.value) for n in node.body
                 if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id in wanted}
            if set(d) == wanted:
                return d
    raise ValueError("Physical frequency configuration not found")


def file_stat(path):
    st = path.stat()
    return dict(bytes=st.st_size, mtime_ns=st.st_mtime_ns, inode=st.st_ino)


def fit_one(label, radii, frequencies, sign, reference, name):
    # This is the actual-frequency basis with a fixed unitary column rotation:
    # A_actual = A_centered @ diag(exp(sign*j*2*pi*fc*r/c)).
    # No target-dependent phase alignment is performed.
    centered = np.exp(sign * 2j * np.pi * (frequencies - reference)[:, None] * radii[None, :] / C0)
    carrier = np.exp(sign * 2j * np.pi * reference * radii / C0)
    actual = np.exp(sign * 2j * np.pi * frequencies[:, None] * radii[None, :] / C0)
    u, singular, vh = np.linalg.svd(centered, full_matrices=False)
    eps = np.finfo(np.float64).eps
    default_threshold = singular[0] * max(centered.shape) * eps
    default_rank = int(np.count_nonzero(singular > default_threshold))
    energy = float(np.vdot(label, label).real)
    target_l2 = math.sqrt(energy)
    target_rms = target_l2 / math.sqrt(len(label))
    norms = np.linalg.norm(centered, axis=0)
    report = dict(
        basis=name, frequency_count=len(frequencies), column_count=len(radii),
        radii_m=radii.tolist(), singular_values=singular.tolist(),
        numerical_rank_default=default_rank, default_rank_rcond=max(centered.shape)*eps,
        full_condition_number=float(singular[0] / singular[-1]),
        column_norm_min=float(norms.min()), column_norm_max=float(norms.max()),
        centered_actual_basis_max_abs_difference=float(np.max(np.abs(centered * carrier[None, :] - actual))),
        solves=[])
    coefficients = []
    for rcond in RCONDS:
        kept = singular > singular[0] * rcond
        left = u[:, kept].conj().T @ label
        centered_coefficients = vh[kept, :].conj().T @ (left / singular[kept])
        physical_coefficients = centered_coefficients / carrier
        pred = centered @ centered_coefficients
        error = pred - label
        residual = float(np.vdot(error, error).real) / energy
        pred_energy = float(np.vdot(pred, pred).real)
        norm_c = float(np.linalg.norm(physical_coefficients))
        # An independent LAPACK least-squares entry point checks the SVD solve.
        c_ls, _, rank_ls, s_ls = np.linalg.lstsq(centered, label, rcond=rcond)
        pred_ls = centered @ c_ls
        residual_ls = float(np.vdot(pred_ls-label, pred_ls-label).real) / energy
        direct = actual @ physical_coefficients
        direct_residual = float(np.vdot(direct-label, direct-label).real) / energy
        cross_difference = float(np.linalg.norm(pred-pred_ls) / target_l2)
        retained_condition = float(singular[0] / singular[kept][-1]) if np.any(kept) else None
        if int(rank_ls) != int(kept.sum()):
            raise AssertionError("Rank mismatch between SVD and LAPACK least squares")
        if not np.isfinite(pred).all() or not math.isfinite(residual):
            raise FloatingPointError("Nonfinite capacity diagnostic")
        row = dict(
            rcond=rcond, retained_rank=int(kept.sum()), retained_condition_number=retained_condition,
            residual_nmse=residual, prediction_energy_ratio=pred_energy/energy,
            coefficient_l2_raw_H_units=norm_c,
            coefficient_l2_over_target_l2=norm_c/target_l2,
            coefficient_l2_over_target_rms=norm_c/target_rms,
            coefficient_l1_over_target_rms=float(np.sum(np.abs(physical_coefficients))/target_rms),
            coefficient_max_abs_raw_H_units=float(np.max(np.abs(physical_coefficients))),
            lapack_lstsq_rank=int(rank_ls), lapack_lstsq_residual_nmse=residual_ls,
            svd_lstsq_prediction_difference_over_target_l2=cross_difference,
            actual_unrecentered_basis_residual_nmse=direct_residual,
            actual_vs_centered_prediction_difference_over_target_l2=float(np.linalg.norm(direct-pred)/target_l2))
        report["solves"].append(row)
        for radius, coefficient in zip(radii, physical_coefficients):
            coefficients.append(dict(basis=name, rcond=rcond, radius_m=float(radius),
                real_raw_H_units=float(coefficient.real), imag_raw_H_units=float(coefficient.imag),
                abs_raw_H_units=float(abs(coefficient))))
    return report, coefficients


def main():
    started = time.monotonic()
    panel_path = EVIDENCE / "panel_v2.json"
    frozen_path = EVIDENCE / "nerf2/frozen_unit.json"
    panel = json.loads(panel_path.read_text())
    frozen = json.loads(frozen_path.read_text())
    row, bs, ue = panel["primary_row"], panel["primary_BS_spatial_id"], panel["primary_UE_port"]
    if (row, bs, ue) != (2666, 30, 0) or panel["role"] != "screen" or panel["outer"] != 0:
        raise ValueError("Only the parent-authorized FIT scalar link is enabled")
    source_dir = PATHS.code_root / "round2/paper_baseline_training_v1"
    field_path = source_dir / "field_runner.py"
    runner_path = source_dir / "nerf2_runner_v2.py"
    integral_path = PATHS.root / "plus-pro/code/measured/full_h_adapters_v1/nerf2.py"
    v16 = PATHS.root / "final_code/V16_reorganized_submission/code/round2"
    scene_path = v16 / "physical_prior/scene_prior_data.py"
    pipeline_path = v16 / "physical_ai_submission/config.py"
    contract_path = PATHS.code_root / "round2/paper_baseline_adapters_v1/contract.py"
    folds_path = PATHS.artifacts / "round2/foundation/round2_spatial_hole_folds.npz"
    meta_path = PATHS.artifacts / "round2/pre_sionna/round2_pre_sionna_oof_foundation_meta.npz"
    raw_path = PATHS.round2 / "Round2_Train_Channel.npy"
    for path, key in ((field_path, "field_runner.py"), (runner_path, "nerf2_runner_v2.py"),
                      (integral_path, PATHS.relative(integral_path)), (contract_path, PATHS.relative(contract_path))):
        if sha(path) != frozen["source_identity"][key]:
            raise RuntimeError("Previously frozen numerical source changed: " + key)
    cfg = {**field_configuration(field_path), **config_overrides(runner_path)}
    if (cfg["n_samples"], cfg["near_m"], cfg["far_m"], cfg["propagation_sign"]) != (32, 0., 810., -1):
        raise ValueError("Unexpected parent NeRF2 physical configuration")
    with np.load(folds_path, allow_pickle=False) as z:
        fit = np.intersect1d(z["fold0_fit"], z["fold1_fit"])
        outer = np.asarray(z["fold0_query"])
        validation = np.asarray(z["fold1_query"])
    if row not in fit or row in outer or row in validation:
        raise PermissionError("The selected channel is not exclusively permitted screen FIT")
    with np.load(meta_path, allow_pickle=False) as z:
        query = np.asarray(z["query_ids"])
    # A row may be used by a different outer fold. Only this outer0 screen role is authorized here.
    pcfg = pipeline_frequency(pipeline_path)
    carrier = literal_assignment(scene_path, "SCENE_FREQUENCY_HZ")
    if carrier != pcfg["physical_carrier_frequency_hz"] or pcfg["subcarriers"] != 192:
        raise ValueError("Frequency source mismatch")
    frequencies = carrier + (np.arange(192, dtype=np.float64)-96)*pcfg["subcarrier_spacing_hz"]
    step = (cfg["far_m"]-cfg["near_m"])/cfg["n_samples"]
    radii = cfg["near_m"] + step*np.arange(1, cfg["n_samples"]+1, dtype=np.float64)
    # Exact source grid identity: both the raw adapter and the shared-medium implementation use this.
    if "cfg.near_m + step * torch.arange(" not in integral_path.read_text():
        raise RuntimeError("The inspected finite-grid source expression changed")
    raw_before = file_stat(raw_path)
    h = np.load(raw_path, mmap_mode="r", allow_pickle=False)
    if h.shape != (4000,256,4,192) or h.dtype != np.complex64:
        raise ValueError("Original channel header mismatch")
    target = np.array(h[row, bs, ue, :], copy=True)
    del h
    if target.shape != (192,) or not np.isfinite(target).all() or not np.any(np.abs(target)>0):
        raise ValueError("Invalid or zero FIT link")
    if complex(target[96]) != complex(panel["site_details"][0]["raw_target_real"], panel["site_details"][0]["raw_target_imag"]):
        raise RuntimeError("F96 target no longer matches the frozen FIT panel")
    target128 = target.astype(np.complex128)
    bases = []
    coefficient_rows = []
    for name, rr in (("native_32_positive_radii", radii),
                     ("diagnostic_plus_zero_delay", np.concatenate(([0.], radii)))):
        result, coefficients = fit_one(target128, rr, frequencies, cfg["propagation_sign"], carrier, name)
        bases.append(result); coefficient_rows.extend(coefficients)
    if file_stat(raw_path) != raw_before:
        raise RuntimeError("Original channel source identity changed during CPU diagnostic")
    source_paths = [Path(__file__),panel_path,frozen_path,field_path,runner_path,integral_path,
                    source_dir/"field_fast.py",contract_path,scene_path,pipeline_path,folds_path,meta_path]
    result = dict(schema="nerf2-static-frequency-linear-basis-capacity-v2",
        diagnostic_only=True, baseline_score=False, model_trained=False, optimizer_created=False,
        GPU_used=False, query_truth_read=False, validation_truth_read=False,
        target_role="outer0 screen FIT", target_row=row, BS_port=bs, UE_port=ue,
        target_complex_values_read=192, target_dtype=str(target.dtype),
        target_sha256=hashlib.sha256(target.tobytes()).hexdigest(),
        target_energy=float(np.vdot(target128,target128).real),
        target_rms=float(np.linalg.norm(target128)/math.sqrt(192)),
        row_is_in_this_outer_query=False, row_is_in_this_outer_validation=False,
        row_in_all_outer_query_union=bool(row in query),
        parent_fit_ids_sha256=hashlib.sha256(fit.astype(np.int64).tobytes()).hexdigest(),
        raw_source=dict(path=PATHS.relative(raw_path),**raw_before), source_unchanged=True,
        physical_configuration=dict(n_rays=cfg["n_rays"],n_samples=cfg["n_samples"],
            near_m=cfg["near_m"],far_m=cfg["far_m"],segment_length_m=step,
            propagation_sign=cfg["propagation_sign"],speed_of_light_m_per_s=C0,
            frequency_source="f=fc+(k-96)*df; frozen V16 convention",
            actual_frequencies_hz=frequencies.tolist(), frequency_reference_hz=carrier),
        coefficient_convention="H(f)=sum C_j*exp(sign*j*2*pi*f*r_j/c); each C_j free complex, independent of f",
        numerical_solve="complex128 SVD on an exactly column-unitarily-rotated centered-frequency basis; target phase unchanged",
        limitations=[
            "Free coefficients ignore neural parameter sharing and physical renderer coefficient constraints.",
            "A small least-squares residual with huge coefficients and ill conditioning is not evidence of learnability.",
            "The zero-delay column is absent from the current native32-sample renderer and is diagnostic only.",
            "Single FIT link only; no spatial generalization, multiport fit, or literature-baseline performance conclusion.",
            "No target-dependent correction is fitted to query or validation predictions."],
        sources={PATHS.relative(p):sha(p) for p in source_paths},bases=bases,
        environment=dict(python=sys.version,numpy=np.__version__,platform=platform.platform(),
                         CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),
                         OPENBLAS_NUM_THREADS=os.environ.get("OPENBLAS_NUM_THREADS"),
                         OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS")),
        elapsed_seconds=time.monotonic()-started)
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    flat=[dict(basis=b["basis"], **r) for b in bases for r in b["solves"]]
    with (OUT/"comparison.csv").open("x") as f:
        w=csv.DictWriter(f, fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    with (OUT/"coefficients.csv").open("x") as f:
        w=csv.DictWriter(f, fieldnames=list(coefficient_rows[0]));w.writeheader();w.writerows(coefficient_rows)
    print(json.dumps(dict(output=PATHS.relative(OUT),target_energy=result["target_energy"],
        target_rms=result["target_rms"], elapsed_seconds=result["elapsed_seconds"],bases=bases),ensure_ascii=False))
if __name__ == "__main__":
    main()
