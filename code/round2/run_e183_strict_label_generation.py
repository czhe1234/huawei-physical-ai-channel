#!/usr/bin/env python3
"""Run the remaining strict E183 family label pipelines sequentially."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from project_paths import PATHS


VERSION = "Round2-E183-strict-label-generation-runner"
FAMILIES = (
    "S0H0",
    "S0H1",
    "S0H2",
    "S0H3",
    "S0H4",
    "S1H1",
    "S1H2",
    "S1H3",
    "S1H4",
)
REPORT = (
    PATHS.analysis
    / "results"
    / "round2_e183_strict_action_labels"
    / "strict_label_generation_runner.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--physical-gpu", default="4")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def run(command: list[str], env: dict[str, str]) -> float:
    started = time.time()
    subprocess.run(command, cwd=PATHS.root, env=env, check=True)
    return float(time.time() - started)


def main() -> None:
    args = parse_args()
    families = tuple(value.strip() for value in args.families.split(",") if value.strip())
    unknown = sorted(set(families) - set(FAMILIES))
    if unknown:
        raise ValueError(f"unknown E183 families: {unknown}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.physical_gpu
    env["HW_SKIP_ENV_CHECK"] = "1"
    python = sys.executable
    generator = PATHS.round2_code / "generate_e183_clean_action_labels.py"
    scorer = PATHS.round2_code / "score_e183_clean_action_labels.py"
    existing = (
        json.loads(REPORT.read_text(encoding="utf-8"))
        if REPORT.is_file()
        else {"experiment": VERSION, "families": []}
    )
    completed = {
        str(row["family"]): row
        for row in existing.get("families", [])
        if row.get("status") == "complete"
    }
    records = list(existing.get("families", []))
    overall_started = time.time()
    for family in families:
        score_report = (
            PATHS.analysis
            / "results"
            / "round2_e183_strict_action_labels"
            / f"{family}_strict_action_labels.json"
        )
        if family in completed and score_report.is_file():
            print(f"E183 runner {family}: complete, reused", flush=True)
            continue
        family_started = time.time()
        record: dict[str, object] = {"family": family, "status": "running"}
        try:
            print(f"E183 runner {family}: prepare", flush=True)
            record["prepare_seconds"] = run(
                [python, "-u", str(generator), "--phase", "prepare", "--family", family],
                env,
            )
            print(f"E183 runner {family}: strict pipeline", flush=True)
            record["pipeline_seconds"] = run(
                [
                    python,
                    "-u",
                    str(generator),
                    "--phase",
                    "pipeline",
                    "--family",
                    family,
                    "--device",
                    args.device,
                    "--physical-gpu",
                    args.physical_gpu,
                    "--start-step",
                    "0",
                    "--stop-step",
                    "23",
                ],
                env,
            )
            print(f"E183 runner {family}: score labels", flush=True)
            record["score_seconds"] = run(
                [
                    python,
                    "-u",
                    str(scorer),
                    "--family",
                    family,
                    "--device",
                    args.device,
                    "--batch-size",
                    str(args.batch_size),
                ],
                env,
            )
            record["status"] = "complete"
        except Exception as error:
            record["status"] = "failed"
            record["error"] = str(error)
            raise
        finally:
            record["seconds"] = float(time.time() - family_started)
            records = [row for row in records if row.get("family") != family]
            records.append(record)
            payload = {
                "experiment": VERSION,
                "status": "running",
                "physical_gpu": args.physical_gpu,
                "device": args.device,
                "families": records,
                "seconds_this_invocation": float(time.time() - overall_started),
            }
            REPORT.parent.mkdir(parents=True, exist_ok=True)
            REPORT.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    payload = {
        "experiment": VERSION,
        "status": "complete",
        "physical_gpu": args.physical_gpu,
        "device": args.device,
        "families": records,
        "seconds_this_invocation": float(time.time() - overall_started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
