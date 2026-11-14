#!/usr/bin/env python3
"""Run the five E215 folds in parallel and produce the fixed500 report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from project_paths import PATHS
import train_e215_relative_energy as e215


SCRIPT = Path(__file__).resolve().with_name("train_e215_relative_energy.py")
RUNNER_REPORT = e215.ANALYSIS_DIR / "parallel_run.json"
SUMMARY_REPORT = e215.ANALYSIS_DIR / "summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,2,3,4")
    parser.add_argument("--epochs", type=int, default=e215.e214.Config.epochs)
    parser.add_argument("--batch-size", type=int, default=e215.e214.Config.batch_size)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def command(
    stage: str,
    epochs: int,
    batch_size: int,
    outer: int | None = None,
    force: bool = False,
) -> list[str]:
    result = [
        sys.executable,
        "-u",
        str(SCRIPT),
        "--stage",
        stage,
        "--device",
        "cuda:0",
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
    ]
    if outer is not None:
        result.extend(("--outer", str(outer)))
    if force:
        result.append("--force")
    return result


def gpu_environment(gpu: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    required = (str(PATHS.code_root), str(PATHS.common_code))
    current = tuple(value for value in environment.get("PYTHONPATH", "").split(":") if value)
    environment["PYTHONPATH"] = ":".join(dict.fromkeys((*required, *current)))
    return environment


def run_prepare(args: argparse.Namespace, gpu: str) -> dict[str, object]:
    log_path = e215.ANALYSIS_DIR / "prepare.log"
    with log_path.open("wb") as handle:
        exit_code = subprocess.call(
            command("prepare", args.epochs, args.batch_size, force=args.force),
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=gpu_environment(gpu),
        )
    return {
        "gpu": gpu,
        "exit_code": exit_code,
        "log": PATHS.relative(log_path),
    }


def run_folds(args: argparse.Namespace, gpus: list[str]) -> list[dict[str, object]]:
    pending = list(range(5))
    active: dict[str, tuple[int, subprocess.Popen[bytes], object, Path]] = {}
    results: list[dict[str, object]] = []
    directory = e215.e214.variant_analysis_dir(e215.VARIANT)
    directory.mkdir(parents=True, exist_ok=True)
    while pending or active:
        for gpu in gpus:
            if gpu in active or not pending:
                continue
            outer = pending.pop(0)
            log_path = directory / f"outer{outer}.log"
            handle = log_path.open("wb")
            process = subprocess.Popen(
                command(
                    "train",
                    args.epochs,
                    args.batch_size,
                    outer=outer,
                    force=args.force,
                ),
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=gpu_environment(gpu),
            )
            active[gpu] = (outer, process, handle, log_path)
            print(f"E215 launched outer={outer} gpu={gpu} pid={process.pid}", flush=True)
        finished: list[str] = []
        for gpu, (outer, process, handle, log_path) in active.items():
            exit_code = process.poll()
            if exit_code is None:
                continue
            handle.close()
            (directory / f"outer{outer}.exitcode").write_text(
                f"{exit_code}\n", encoding="ascii"
            )
            results.append(
                {
                    "outer": outer,
                    "gpu": gpu,
                    "pid": process.pid,
                    "exit_code": exit_code,
                    "log": PATHS.relative(log_path),
                }
            )
            print(f"E215 finished outer={outer} gpu={gpu} exit={exit_code}", flush=True)
            finished.append(gpu)
        for gpu in finished:
            del active[gpu]
        if active and not finished:
            time.sleep(1.0)
    return sorted(results, key=lambda item: int(item["outer"]))


def aggregate(args: argparse.Namespace, gpu: str, folds: list[dict[str, object]]) -> dict[str, object]:
    if any(int(item["exit_code"]) != 0 for item in folds):
        return {"status": "skipped_failed_fold", "exit_code": None}
    directory = e215.e214.variant_analysis_dir(e215.VARIANT)
    log_path = directory / "aggregate.log"
    with log_path.open("wb") as handle:
        exit_code = subprocess.call(
            command("aggregate", args.epochs, args.batch_size),
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=gpu_environment(gpu),
        )
    print(f"E215 aggregate exit={exit_code}", flush=True)
    return {
        "status": "complete" if exit_code == 0 else "failed",
        "gpu": gpu,
        "exit_code": exit_code,
        "log": PATHS.relative(log_path),
    }


def summarize() -> dict[str, object]:
    report_path = e215.e214.final_report_path(e215.VARIANT)
    if not report_path.is_file():
        summary = {"experiment": e215.VERSION, "status": "incomplete"}
    else:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        candidate = report["candidate"]
        control_path = (
            PATHS.analysis
            / "results"
            / "round2_e214_geometry_path_topology"
            / "base_s1"
            / "fixed500_report.json"
        )
        control = json.loads(control_path.read_text(encoding="utf-8"))["candidate"]
        summary = {
            "experiment": e215.VERSION,
            "status": "complete",
            "variant": e215.VARIANT,
            "candidate": candidate,
            "control_e214_base_s1": control,
            "delta_vs_e214_base_s1": {
                key: float(candidate[key] - control[key])
                for key in ("pas", "pdp", "nmse_score", "score")
            },
            "delta_vs_v16": report["delta_vs_v16"],
            "success_threshold": 0.680000,
            "passed": bool(candidate["score"] >= 0.680000),
            "report": PATHS.relative(report_path),
        }
    atomic_json(SUMMARY_REPORT, summary)
    return summary


def main() -> None:
    args = parse_args()
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("at least one GPU is required")
    e215.configure_e214()
    e215.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    prepare = (
        {"status": "skipped", "reason": "--skip-prepare"}
        if args.skip_prepare
        else run_prepare(args, gpus[0])
    )
    if prepare.get("exit_code", 0) != 0:
        report = {
            "experiment": e215.VERSION,
            "status": "failed_prepare",
            "prepare": prepare,
            "wall_seconds": float(time.time() - started),
        }
        atomic_json(RUNNER_REPORT, report)
        raise SystemExit(1)
    folds = run_folds(args, gpus)
    aggregate_result = aggregate(args, gpus[0], folds)
    summary = summarize()
    failed = any(int(item["exit_code"]) != 0 for item in folds) or aggregate_result.get(
        "exit_code"
    ) != 0
    report = {
        "experiment": e215.VERSION,
        "status": "failed" if failed else "complete",
        "prepare": prepare,
        "folds": folds,
        "aggregate": aggregate_result,
        "summary": PATHS.relative(SUMMARY_REPORT),
        "wall_seconds": float(time.time() - started),
    }
    atomic_json(RUNNER_REPORT, report)
    print(json.dumps({"runner": report, "summary": summary}, indent=2), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
