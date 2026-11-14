#!/usr/bin/env python3
"""Validate that a copied workspace can be resumed without machine-local state."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_DIRECTORIES = {
    "analysis",
    "artifacts",
    "code",
    "datasets",
    "experiment_records",
    "versions",
}
REQUIRED_FILES = (
    ".hw-project",
    "README.md",
    "AGENTS.md",
    "environment.yml",
    "environment.cuda.yml",
    "experiment_records/EXPERIMENT_RECORD.md",
    "code/project_paths.py",
    "code/round1/generate_v140_balanced_hminus2_overdrive.py",
    "code/round1/generate_v142_mass_transport_geodesic_shift.py",
    "code/round1/generate_v143_transductive_geodesic_risk.py",
    "code/round1/analyze_post_v143_hidden_information.py",
    "code/round1/analyze_v143_matched_residual_probability.py",
    "code/round1/analyze_v143_marginal_residual_learnability.py",
    "code/round1/generate_v144_v_marginal_residual.py",
    "code/round1/analyze_v144_multidimensional_noise_correlation.py",
    "code/round1/analyze_v145_gated_local_h_field.py",
    "code/round1/generate_v145_local_h_residual_field.py",
    "code/round2/analyze_round2_initial_data.py",
    "code/round1/__init__.py",
    "code/round2/__init__.py",
    "code/common/__init__.py",
    "datasets/round1/Round1_Train_Pos.npy",
    "datasets/round1/Round1_Test_Pos.npy",
    "artifacts/shared_features/official_pointcloud_scene_cache.npz",
    "artifacts/shared_features/geodesic_support_features.npz",
    "artifacts/shared_features/train_v125eq_crossfit_template_probs.npy",
    "artifacts/shared_features/height_geometry_cache.npz",
    "artifacts/shared_features/truth_coll_power.npy",
    "artifacts/shared_features/train_v129_crossfit_residual_probs.npy",
    "artifacts/version_cache/v128/train_v125eq_collapsed_support.npy",
    "artifacts/version_cache/v129/v129_apex_badpoint_support_residual_target.npy",
    "artifacts/version_cache/v129/v129_apex_badpoint_support_residual_probs.npy",
    "artifacts/version_cache/v129/v129_Round1_Test_Channel.npy",
    "artifacts/version_cache/v140/v140_Round1_Test_Channel.npy",
    "artifacts/version_cache/v142/v142_Round1_Test_Channel.npy",
    "artifacts/version_cache/v143/v143_Round1_Test_Channel.npy",
    "artifacts/version_cache/v144/v144_Round1_Test_Channel.npy",
    "artifacts/version_cache/v144/v144_v_marginal_residual_target.npy",
    "artifacts/version_cache/v144/v144_v_marginal_log_ratio.npy",
    "artifacts/version_cache/v145/v145_Round1_Test_Channel.npy",
    "artifacts/version_cache/v145/v145_local_h_residual_field_target.npy",
    "artifacts/version_cache/v145/v145_local_h_residual_field.npz",
    "analysis/results/version_reports/v140/v140_balanced_hminus2_overdrive_generate.json",
    "analysis/results/version_reports/v142/v142_mass_transport_geodesic_shift_generate.json",
    "analysis/results/version_reports/v143/v143_transductive_geodesic_risk_generate.json",
    "analysis/results/version_reports/v143/v143_post_online_hidden_information_analysis.json",
    "analysis/results/version_reports/v143/v143_matched_residual_probability_analysis.json",
    "analysis/results/version_reports/v143/v143_marginal_residual_learnability_analysis.json",
    "analysis/results/version_reports/v144/v144_v_marginal_residual_generate.json",
    "analysis/results/version_reports/v144/v144_multidimensional_noise_correlation_analysis.json",
    "analysis/results/version_reports/v144/v144_noise_point_metrics.csv",
    "analysis/results/version_reports/v144/v144_noise_distance_curve.csv",
    "analysis/results/version_reports/v144/v144_noise_neighbor_spaces.csv",
    "analysis/results/version_reports/v144/v144_noise_component_modes.npz",
    "analysis/results/version_reports/v145/v145_gated_local_h_field_audit.json",
    "analysis/results/version_reports/v145/v145_gated_local_h_field_nested_oof.npz",
    "analysis/results/version_reports/v145/v145_local_h_residual_field_generate.json",
    "versions/round1/v140_balanced_hminus2_overdrive_online_0.501352.zip",
    "versions/round1/v142_mass_transport_geodesic_shift_online_0.500329.zip",
    "versions/round1/v143_transductive_geodesic_risk_online_0.501271.zip",
    "versions/round1/v144_v_marginal_residual_online_0.500586.zip",
    "versions/round1/v145_local_h_residual_field_candidate_pending.zip",
    "versions/checksums.sha256",
    "versions/checksums.md5",
    "artifacts/artifact_registry.json",
)
ALLOWED_VERSION_METADATA = {"checksums.sha256", "checksums.md5"}


def checksum_rows(path: Path, digest_length: int, failures: list[str]) -> dict[str, str]:
    rows: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except OSError as exc:
        failures.append(f"cannot read {path.relative_to(ROOT)}: {exc}")
        return rows

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or len(fields[0]) != digest_length:
            failures.append(f"invalid checksum row: {path.name}:{line_number}")
            continue
        digest, relative = fields[0].lower(), fields[1].lstrip("*")
        relative_path = Path(relative)
        if (
            any(char not in "0123456789abcdef" for char in digest)
            or relative_path.is_absolute()
            or len(relative_path.parts) != 3
            or relative_path.parts[0] != "versions"
            or relative_path.parts[1] not in {"round1", "round2"}
            or relative_path.suffix != ".zip"
        ):
            failures.append(f"non-portable checksum row: {path.name}:{line_number}")
            continue
        if relative in rows:
            failures.append(f"duplicate checksum target: {path.name}:{relative}")
        rows[relative] = digest
    return rows


def validate_registry(failures: list[str]) -> int:
    path = ROOT / "artifacts" / "artifact_registry.json"
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"invalid artifact registry: {exc}")
        return 0

    for key, value in registry.items():
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            failures.append(f"non-relative registry target: {key}={value}")
        elif not (ROOT / relative).exists():
            failures.append(f"missing registry target: {key}={value}")
    return len(registry)


def validate_reports(failures: list[str]) -> int:
    paths = sorted((ROOT / "analysis" / "results" / "version_reports").glob("*/*.json"))
    for path in paths:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"invalid version report {path.relative_to(ROOT)}: {exc}")
    return len(paths)


def validate_full_hashes(
    sha_rows: dict[str, str], md5_rows: dict[str, str], failures: list[str]
) -> None:
    for relative in sorted(sha_rows):
        path = ROOT / relative
        sha = hashlib.sha256()
        md5 = hashlib.md5()
        try:
            with path.open("rb") as handle:
                while chunk := handle.read(8 * 1024 * 1024):
                    sha.update(chunk)
                    md5.update(chunk)
        except OSError as exc:
            failures.append(f"cannot hash {relative}: {exc}")
            continue
        if sha.hexdigest() != sha_rows[relative]:
            failures.append(f"SHA-256 mismatch: {relative}")
        if md5.hexdigest() != md5_rows[relative]:
            failures.append(f"MD5 mismatch: {relative}")

    current = ROOT / "versions" / "v143_transductive_geodesic_risk_online_0.501271.zip"
    try:
        with zipfile.ZipFile(current) as archive:
            bad_member = archive.testzip()
        if bad_member is not None:
            failures.append(f"current ZIP CRC failure: {bad_member}")
    except (OSError, zipfile.BadZipFile) as exc:
        failures.append(f"cannot validate current ZIP: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="read every retained ZIP and validate SHA-256, MD5, and the current ZIP CRC",
    )
    args = parser.parse_args()
    failures: list[str] = []

    for relative in REQUIRED_FILES:
        if not (ROOT / relative).is_file():
            failures.append(f"required handoff file is missing: {relative}")

    top_directories = {
        path.name for path in ROOT.iterdir() if path.is_dir() and not path.name.startswith(".")
    }
    if top_directories != EXPECTED_DIRECTORIES:
        failures.append(
            "top-level directories differ: "
            f"expected={sorted(EXPECTED_DIRECTORIES)} actual={sorted(top_directories)}"
        )

    record_docs = sorted((ROOT / "experiment_records").glob("*.md"))
    expected_record = ROOT / "experiment_records" / "EXPERIMENT_RECORD.md"
    if record_docs != [expected_record]:
        failures.append("experiment_records must contain only EXPERIMENT_RECORD.md")

    symlinks = [path.relative_to(ROOT) for path in ROOT.rglob("*") if path.is_symlink()]
    if symlinks:
        failures.extend(f"symbolic link is not cross-device safe: {path}" for path in symlinks)

    misplaced_python = [
        path.relative_to(ROOT)
        for path in ROOT.rglob("*.py")
        if ROOT / "code" not in path.parents
    ]
    if misplaced_python:
        failures.extend(f"Python source outside code/: {path}" for path in misplaced_python)

    versions = ROOT / "versions"
    archives = {str(path.relative_to(ROOT)) for path in versions.rglob("*.zip")}
    for path in versions.iterdir():
        if path.is_dir():
            if path.name not in {"round1", "round2"}:
                failures.append(f"versions contains an unsupported directory: {path.name}")
        elif path.suffix != ".zip" and path.name not in ALLOWED_VERSION_METADATA:
            failures.append(f"versions contains unsupported metadata: {path.name}")
    for round_name in ("round1", "round2"):
        round_dir = versions / round_name
        if not round_dir.is_dir():
            failures.append(f"versions is missing required directory: {round_name}")
            continue
        for path in round_dir.iterdir():
            if not path.is_file() or path.suffix != ".zip":
                failures.append(f"versions/{round_name} contains non-ZIP entry: {path.name}")

    sha_rows = checksum_rows(versions / "checksums.sha256", 64, failures)
    md5_rows = checksum_rows(versions / "checksums.md5", 32, failures)
    if set(sha_rows) != archives:
        failures.append("SHA-256 targets do not exactly match retained ZIP files")
    if set(md5_rows) != archives:
        failures.append("MD5 targets do not exactly match retained ZIP files")
    if args.full and set(sha_rows) == archives and set(md5_rows) == archives:
        validate_full_hashes(sha_rows, md5_rows, failures)

    registry_count = validate_registry(failures)
    report_count = validate_reports(failures)

    hardcoded = subprocess.run(
        [sys.executable, str(ROOT / "code" / "tools" / "check_hardcoded_paths.py")],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if hardcoded.returncode:
        failures.append(hardcoded.stderr.strip() or hardcoded.stdout.strip())

    if failures:
        print("Workspace portability check failed:", file=sys.stderr)
        print("\n".join(f"- {failure}" for failure in failures), file=sys.stderr)
        return 1

    mode = "full" if args.full else "metadata"
    print(
        f"workspace_portable=yes mode={mode} zips={len(archives)} "
        f"reports={report_count} registry_entries={registry_count} symlinks=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
