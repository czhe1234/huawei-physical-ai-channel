#!/usr/bin/env python3
"""Run all 20 E214 folds across idle GPUs and aggregate four variants."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from project_paths import PATHS
import train_e214_geometry_path_topology as e214


SCRIPT = Path(__file__).resolve().with_name("train_e214_geometry_path_topology.py")
RUNNER_REPORT = e214.ANALYSIS_DIR / "parallel_run.json"
SUMMARY_REPORT = e214.ANALYSIS_DIR / "four_variant_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--epochs", type=int, default=e214.Config.epochs)
    parser.add_argument("--batch-size", type=int, default=e214.Config.batch_size)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--finalize-existing", action="store_true")
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
    variant: str,
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
        "--variant",
        variant,
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
    return environment


def run_prepare(args: argparse.Namespace, gpu: str) -> dict[str, object]:
    log_path = e214.ANALYSIS_DIR / "prepare.log"
    with log_path.open("wb") as handle:
        exit_code = subprocess.call(
            command("prepare", "all", args.epochs, args.batch_size, force=args.force),
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=gpu_environment(gpu),
        )
    (e214.ANALYSIS_DIR / "prepare.exitcode").write_text(
        f"{exit_code}\n", encoding="ascii"
    )
    return {
        "gpu": gpu,
        "exit_code": exit_code,
        "log": PATHS.relative(log_path),
    }


def run_folds(args: argparse.Namespace, gpus: list[str]) -> list[dict[str, object]]:
    tasks = [(variant, outer) for variant in e214.VARIANTS for outer in range(5)]
    pending = list(tasks)
    active: dict[str, tuple[str, int, subprocess.Popen[bytes], object, Path]] = {}
    results: list[dict[str, object]] = []
    while pending or active:
        for gpu in gpus:
            if gpu in active or not pending:
                continue
            variant, outer = pending.pop(0)
            directory = e214.variant_analysis_dir(variant)
            directory.mkdir(parents=True, exist_ok=True)
            log_path = directory / f"outer{outer}.log"
            handle = log_path.open("wb")
            process = subprocess.Popen(
                command(
                    "train",
                    variant,
                    args.epochs,
                    args.batch_size,
                    outer=outer,
                    force=args.force,
                ),
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=gpu_environment(gpu),
            )
            active[gpu] = (variant, outer, process, handle, log_path)
            print(
                f"E214 launched variant={variant} outer={outer} gpu={gpu} pid={process.pid}",
                flush=True,
            )
        finished = []
        for gpu, (variant, outer, process, handle, log_path) in active.items():
            exit_code = process.poll()
            if exit_code is None:
                continue
            handle.close()
            exit_path = e214.variant_analysis_dir(variant) / f"outer{outer}.exitcode"
            exit_path.write_text(f"{exit_code}\n", encoding="ascii")
            results.append(
                {
                    "variant": variant,
                    "outer": outer,
                    "gpu": gpu,
                    "pid": process.pid,
                    "exit_code": exit_code,
                    "log": PATHS.relative(log_path),
                }
            )
            print(
                f"E214 finished variant={variant} outer={outer} gpu={gpu} exit={exit_code}",
                flush=True,
            )
            finished.append(gpu)
        for gpu in finished:
            del active[gpu]
        if active and not finished:
            time.sleep(1.0)
    return sorted(results, key=lambda item: (str(item["variant"]), int(item["outer"])))


def aggregate_variants(
    args: argparse.Namespace,
    gpu: str,
    fold_results: list[dict[str, object]],
) -> list[dict[str, object]]:
    output = []
    for variant in e214.VARIANTS:
        failed_folds = [
            item
            for item in fold_results
            if item["variant"] == variant and int(item["exit_code"]) != 0
        ]
        if failed_folds:
            output.append(
                {
                    "variant": variant,
                    "gpu": gpu,
                    "exit_code": None,
                    "status": "skipped_failed_fold",
                }
            )
            continue
        log_path = e214.variant_analysis_dir(variant) / "aggregate.log"
        with log_path.open("wb") as handle:
            exit_code = subprocess.call(
                command("aggregate", variant, args.epochs, args.batch_size),
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=gpu_environment(gpu),
            )
        (e214.variant_analysis_dir(variant) / "aggregate.exitcode").write_text(
            f"{exit_code}\n", encoding="ascii"
        )
        output.append(
            {
                "variant": variant,
                "gpu": gpu,
                "exit_code": exit_code,
                "status": "complete" if exit_code == 0 else "failed",
                "log": PATHS.relative(log_path),
            }
        )
        print(f"E214 aggregate variant={variant} exit={exit_code}", flush=True)
    return output


def summarize() -> dict[str, object]:
    rows = []
    for variant in e214.VARIANTS:
        path = e214.final_report_path(variant)
        if not path.is_file():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        candidate = report["candidate"]
        rows.append(
            {
                "variant": variant,
                "pas": float(candidate["pas"]),
                "pdp": float(candidate["pdp"]),
                "nmse": float(candidate["nmse"]),
                "nmse_score": float(candidate["nmse_score"]),
                "score": float(candidate["score"]),
                "delta_vs_v7": float(report["delta_vs_v7"]["score"]),
                "delta_vs_v16": float(report["delta_vs_v16"]["score"]),
                "selected_epochs": report["selected_epochs"],
                "passed": bool(report["passed"]),
                "report": PATHS.relative(path),
            }
        )
    rows.sort(key=lambda item: float(item["score"]), reverse=True)
    summary = {
        "experiment": e214.VERSION,
        "status": "complete" if len(rows) == len(e214.VARIANTS) else "incomplete",
        "success_threshold": 0.680000,
        "variants": rows,
        "best": rows[0] if rows else None,
        "passed": bool(rows and float(rows[0]["score"]) >= 0.680000),
    }
    atomic_json(SUMMARY_REPORT, summary)
    return summary


def finalize_existing() -> dict[str, object]:
    if not RUNNER_REPORT.is_file():
        raise FileNotFoundError(RUNNER_REPORT)
    previous = json.loads(RUNNER_REPORT.read_text(encoding="utf-8"))
    folds = previous.get("folds", [])
    if len(folds) != 20 or any(int(item["exit_code"]) != 0 for item in folds):
        raise RuntimeError("cannot finalize: the 20 fold jobs are not all successful")
    aggregates = []
    for variant in e214.VARIANTS:
        path = e214.final_report_path(variant)
        if not path.is_file():
            raise FileNotFoundError(path)
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "complete":
            raise RuntimeError(f"incomplete aggregate report: {variant}")
        aggregates.append(
            {
                "variant": variant,
                "exit_code": 0,
                "status": "complete_after_config_serialization_fix",
                "report": PATHS.relative(path),
            }
        )
    summary = summarize()
    previous["status"] = "complete_after_aggregate_retry"
    previous["aggregates"] = aggregates
    previous["summary"] = PATHS.relative(SUMMARY_REPORT)
    previous["aggregate_retry"] = {
        "status": "complete",
        "cause": "JSON converted coordinate_limits tuple to list; the original aggregate resume check compared the two representations directly",
        "training_repeated": False,
        "numeric_model_outputs_changed": False,
    }
    atomic_json(RUNNER_REPORT, previous)
    return {"runner": previous, "summary": summary}


def main() -> None:
    args = parse_args()
    if args.finalize_existing:
        result = finalize_existing()
        print(json.dumps(result, indent=2), flush=True)
        return
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("at least one GPU is required")
    e214.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    prepare = (
        {"status": "skipped", "reason": "--skip-prepare"}
        if args.skip_prepare
        else run_prepare(args, gpus[0])
    )
    if prepare.get("exit_code", 0) != 0:
        report = {
            "experiment": e214.VERSION,
            "status": "failed_prepare",
            "prepare": prepare,
            "wall_seconds": float(time.time() - started),
        }
        atomic_json(RUNNER_REPORT, report)
        raise SystemExit(1)
    folds = run_folds(args, gpus)
    aggregates = aggregate_variants(args, gpus[0], folds)
    summary = summarize()
    failed = any(int(item["exit_code"]) != 0 for item in folds) or any(
        item.get("exit_code") != 0 for item in aggregates
    )
    report = {
        "experiment": e214.VERSION,
        "status": "failed" if failed else "complete",
        "prepare": prepare,
        "folds": folds,
        "aggregates": aggregates,
        "summary": PATHS.relative(SUMMARY_REPORT),
        "wall_seconds": float(time.time() - started),
    }
    atomic_json(RUNNER_REPORT, report)
    print(json.dumps({"runner": report, "summary": summary}, indent=2), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
