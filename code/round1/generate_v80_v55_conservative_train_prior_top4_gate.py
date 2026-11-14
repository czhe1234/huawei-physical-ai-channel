#!/usr/bin/env python3
"""Generate v80: conservative train-prior confidence gate on top of v55.

This is a low-risk refinement of the current best v55 submission.  It keeps
the v55 residual source, regime gains, residual phases, AD shift, and power
preservation unchanged.  The only change is a narrow confidence adjustment on
the four strongest angle-delay residual components:

    top4 rho = 0.75 + small train-prior confidence correction
    clipped to [0.68, 0.82]

No extra components are promoted.  This avoids the broader v59 gate, which
slightly underperformed v55 online.
"""

from __future__ import annotations

import sys

from generate_s1_residual_train_prior_gate import main


DEFAULT_ARGS = [
    "--output",
    "Round1_Test_Channel.npy",
    "--out",
    "analysis/results/json/s1_residual_train_prior_gate_v80_conservative_top4_generate.json",
    "--clear-lambda",
    "1.0",
    "--light-lambda",
    "1.0",
    "--medium-lambda",
    "1.0",
    "--heavy-lambda",
    "4.0",
    "--clear-phase-deg",
    "0.0",
    "--light-phase-deg",
    "10.0",
    "--medium-phase-deg",
    "3.0",
    "--heavy-phase-deg",
    "2.0",
    "--top-components",
    "4",
    "--extra-components",
    "4",
    "--weak-rho",
    "0.10",
    "--top-rho",
    "0.75",
    "--top-boost",
    "0.05",
    "--top-conflict-penalty",
    "0.08",
    "--top-min-rho",
    "0.68",
    "--top-max-rho",
    "0.82",
    "--extra-rho",
    "0.10",
    "--extra-boost",
    "0.0",
    "--extra-align-thresh",
    "0.99",
    "--prior-power-frac",
    "0.02",
    "--delay-shift",
    "-1",
    "--h-shift",
    "-1",
    "--v-shift",
    "1",
    "--preserve-delta-power",
    "--residual-k",
    "32",
    "--residual-power",
    "2.0",
    "--components",
    "16",
    "--beta",
    "140.3",
    "--gamma",
    "0.10",
    "--batch",
    "2",
]


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(DEFAULT_ARGS)
    main()
