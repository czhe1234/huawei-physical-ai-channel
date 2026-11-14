#!/usr/bin/env python3
"""Run resumable E211 full-state physical support and token stages."""

from __future__ import annotations

import argparse
import json
import traceback

from round2.physical_prior.source_port_common import CONFIGS, E185_SEEDS
from round2.physical_prior.source_port_full_state_tokens import (
    assemble_tokens,
    build_all_token_shards,
    build_support_worker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("support", "tokens", "assemble"), required=True
    )
    parser.add_argument("--config", choices=tuple(CONFIGS), default="base")
    parser.add_argument("--seed", type=int, choices=E185_SEEDS, default=42)
    parser.add_argument("--site", type=int, choices=(0, 1), default=0)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.stage == "support":
            report = build_support_worker(
                config_name=args.config,
                seed=args.seed,
                site=args.site,
                cuda_device=args.cuda_device,
                only_batch=args.batch,
                force=args.force,
            )
        elif args.stage == "tokens":
            report = build_all_token_shards(args.workers, args.force)
        else:
            report = assemble_tokens(args.force)
    except Exception:
        traceback.print_exc()
        raise
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
