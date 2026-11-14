#!/usr/bin/env python3
"""Plot full online history and the strict v208 replay coverage."""

from __future__ import annotations

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from analyze_v195_scorer_mouth_reverse_inference import parse_online_scores, parse_score_csv
from project_paths import PATHS


REPORT_DIR = PATHS.version_report_dir(208)
V208_REPORT = REPORT_DIR / "v208_strict_historical_replay.json"
V207_FEATURE_CSV = PATHS.version_report_dir(207) / "v207_historical_sionna_reference_features.csv"
SCORE_CSV = PATHS.analysis / "visualizations" / "version_scores" / "online_scores_by_version.csv"
FIGURE_DIR = REPORT_DIR / "figures"
FIGURE = FIGURE_DIR / "v208_full_online_and_strict_replay_curve.png"
COVERAGE_CSV = REPORT_DIR / "v208_full_online_strict_coverage.csv"
LANDMARKS = (1, 6, 55, 83, 112, 129, 140, 163, 169, 181, 187, 189, 196, 200)


def normalized(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    span = float(np.ptp(value))
    return (value - np.min(value)) / max(span, 1e-30)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    scores = parse_online_scores()
    scores.update(parse_score_csv(SCORE_CSV))
    history = sorted((version, float(item["online"])) for version, item in scores.items())
    version = np.asarray([item[0] for item in history], dtype=np.int32)
    online = np.asarray([item[1] for item in history], dtype=np.float64)

    replay = json.loads(V208_REPORT.read_text(encoding="utf-8"))
    exact = replay["exact_replay"]
    fitted = sorted(
        (row for row in exact if row["scenario"] == "fitted_tail_upper_bound"),
        key=lambda row: int(row["version"]),
    )
    relative = {
        int(row["version"]): row
        for row in exact
        if row["scenario"] == "relative_25db"
    }
    exact_version = np.asarray([int(row["version"]) for row in fitted], dtype=np.int32)
    exact_online = np.asarray([float(row["online_score"]) for row in fitted], dtype=np.float64)
    exact_fitted = np.asarray(
        [float(row["score_traceback_leaked"]) for row in fitted], dtype=np.float64
    )
    exact_25db = np.asarray(
        [float(relative[int(row["version"])]["score_traceback_leaked"]) for row in fitted],
        dtype=np.float64,
    )
    with V207_FEATURE_CSV.open("r", encoding="utf-8", newline="") as handle:
        proxy_rows = list(csv.DictReader(handle))
    proxy_version = np.asarray([int(row["version"]) for row in proxy_rows], dtype=np.int32)
    proxy_online = np.asarray([float(row["online"]) for row in proxy_rows], dtype=np.float64)
    proxy_v206_pdp = np.asarray([float(row["v206_pdp_all"]) for row in proxy_rows], dtype=np.float64)
    proxy_v205_pdp = np.asarray([float(row["v205_pdp_all"]) for row in proxy_rows], dtype=np.float64)

    with COVERAGE_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "version",
                "online_score",
                "strict_replay_available",
                "strict_fitted_traceback_score",
                "strict_25db_traceback_score",
            ),
        )
        writer.writeheader()
        fitted_by_version = {int(row["version"]): row for row in fitted}
        for current_version, current_online in history:
            row = fitted_by_version.get(current_version)
            writer.writerow(
                {
                    "version": current_version,
                    "online_score": current_online,
                    "strict_replay_available": int(row is not None),
                    "strict_fitted_traceback_score": ""
                    if row is None
                    else row["score_traceback_leaked"],
                    "strict_25db_traceback_score": ""
                    if row is None
                    else relative[current_version]["score_traceback_leaked"],
                }
            )

    fig = plt.figure(figsize=(14, 13))
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=(1.2, 0.82, 1.0),
        hspace=0.34,
        wspace=0.24,
    )
    top = fig.add_subplot(grid[0, :])
    proxy = fig.add_subplot(grid[1, :])
    shape = fig.add_subplot(grid[2, 0])
    scatter = fig.add_subplot(grid[2, 1])

    top.plot(version, online, color="#687078", linewidth=1.2, alpha=0.80)
    top.scatter(version, online, color="#687078", s=18, alpha=0.80, label="online history")
    strict_online = np.asarray([scores[int(v)]["online"] for v in exact_version], dtype=np.float64)
    top.scatter(
        exact_version,
        strict_online,
        color="#d95f02",
        edgecolor="white",
        linewidth=0.8,
        s=62,
        zorder=4,
        label="strictly replayed in v208",
    )
    best_index = int(np.argmax(online))
    top.scatter(
        [version[best_index]],
        [online[best_index]],
        color="#b2182b",
        marker="*",
        s=150,
        zorder=5,
        label="current online best",
    )
    score_by_version = {item[0]: item[1] for item in history}
    for landmark in LANDMARKS:
        if landmark not in score_by_version:
            continue
        top.annotate(
            f"v{landmark}",
            (landmark, score_by_version[landmark]),
            xytext=(3, 5),
            textcoords="offset points",
            fontsize=8,
        )
    top.set_title(f"Full online history ({len(history)} scored versions) and strict v208 coverage")
    top.set_xlabel("version")
    top.set_ylabel("online score")
    top.grid(alpha=0.18)
    top.legend(loc="lower right")

    proxy.plot(
        proxy_version,
        normalized(proxy_online),
        color="#1b6ca8",
        linewidth=1.7,
        label="online score",
    )
    proxy.plot(
        proxy_version,
        normalized(proxy_v206_pdp),
        color="#d95f02",
        linewidth=1.5,
        label="v206 PDP similarity",
    )
    proxy.plot(
        proxy_version,
        normalized(proxy_v205_pdp),
        color="#7b6ba8",
        linewidth=1.2,
        alpha=0.85,
        label="v205 PDP similarity control",
    )
    proxy_rho = float(spearmanr(proxy_v206_pdp, proxy_online).statistic)
    proxy.set_title(
        f"Non-strict v207 proxy: {len(proxy_rows)} retained submission outputs, not retrained "
        f"(v206 PDP Spearman {proxy_rho:.3f})"
    )
    proxy.set_xlabel("version")
    proxy.set_ylabel("within-series min-max value")
    proxy.grid(alpha=0.18)
    proxy.legend(loc="lower right", ncol=3, fontsize=9)

    shape.plot(
        exact_version,
        normalized(exact_online),
        color="#1b6ca8",
        marker="o",
        linewidth=2.0,
        label="online, min-max normalized",
    )
    shape.plot(
        exact_version,
        normalized(exact_fitted),
        color="#d95f02",
        marker="s",
        linewidth=1.7,
        label="strict synthetic, 21.09 dB",
    )
    shape.plot(
        exact_version,
        normalized(exact_25db),
        color="#4d9221",
        marker="^",
        linewidth=1.4,
        linestyle="--",
        label="strict synthetic, 25 dB",
    )
    shape.set_xticks(exact_version, [f"v{value}" for value in exact_version])
    shape.set_title("Ordering shape on the 8 exact-replay versions")
    shape.set_xlabel("version")
    shape.set_ylabel("within-series min-max value")
    shape.grid(alpha=0.18)
    shape.legend(fontsize=8)

    scatter.scatter(exact_fitted, exact_online, color="#d95f02", s=58)
    for current_version, x, y in zip(exact_version, exact_fitted, exact_online):
        scatter.annotate(
            f"v{current_version}",
            (x, y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=9,
        )
    rho = float(spearmanr(exact_fitted, exact_online).statistic)
    scatter.set_title(f"Strict synthetic vs online (Spearman {rho:.3f})")
    scatter.set_xlabel("strict synthetic traceback score")
    scatter.set_ylabel("online score")
    scatter.grid(alpha=0.18)

    fig.suptitle(
        "v208 strict replay: full online context, exact coverage, and ordering mismatch",
        fontsize=15,
    )
    fig.savefig(FIGURE, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(
        json.dumps(
            {
                "online_version_count": len(history),
                "strict_version_count": len(fitted),
                "strict_versions": exact_version.tolist(),
                "strict_spearman": rho,
                "v207_proxy_version_count": len(proxy_rows),
                "v207_proxy_pdp_spearman": proxy_rho,
                "figure": PATHS.relative(FIGURE),
                "coverage_csv": PATHS.relative(COVERAGE_CSV),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
