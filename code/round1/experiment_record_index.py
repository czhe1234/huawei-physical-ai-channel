"""Read machine-usable online results from the single experiment record."""

from __future__ import annotations

import re
from pathlib import Path

from project_paths import PATHS


ONLINE_ZIP_RE = re.compile(
    r"(?<![A-Za-z0-9])v(\d+)_([A-Za-z0-9_.+%-]+?)_online_([0-9]+\.[0-9]+)\.zip"
)
TABLE_ONLINE_RE = re.compile(
    r"^\|\s*v(\d+)\s*\|.*?online\s*`?([0-9]+\.[0-9]+)`?", re.MULTILINE
)
SCORE_ASSIGN_RE = re.compile(r"^v(\d+)\s*=\s*([0-9]+\.[0-9]+)\s*$", re.MULTILINE)
REGIME_CODE_RE = re.compile(r"(?:^|_)c(\d{3})_l(\d{3})_m(\d{3})_h(\d{3})(?:_|$)")
REGIME_EXPLICIT_RE = re.compile(r"clear([0-9.]+)_else([0-9.]+)")


def _regime_parameters(method: str) -> tuple[str, str]:
    coded = REGIME_CODE_RE.search(method)
    if coded:
        values = [int(coded.group(i)) / 100.0 for i in range(1, 5)]
        params = "clear={:.2f},light={:.2f},medium={:.2f},heavy={:.2f}".format(*values)
        return "regime_s1style_geom_blend", params

    explicit = REGIME_EXPLICIT_RE.search(method)
    if "regime_s1style_geom_blend" in method and explicit:
        clear = float(explicit.group(1))
        other = float(explicit.group(2))
        params = f"clear={clear:.2f},light={other:.2f},medium={other:.2f},heavy={other:.2f}"
        return "regime_s1style_geom_blend", params

    return method, ""


def load_online_rows(record_path: str | Path | None = None) -> list[dict[str, object]]:
    path = Path(record_path) if record_path else PATHS.experiment_records / "EXPERIMENT_RECORD.md"
    text = path.read_text(encoding="utf-8")
    rows: dict[str, dict[str, object]] = {}

    for match in ONLINE_ZIP_RE.finditer(text):
        version = f"v{int(match.group(1))}"
        method, params = _regime_parameters(match.group(2))
        rows[version] = {
            "version": version,
            "method": method,
            "params": params,
            "online": float(match.group(3)),
            "conclusion": "See EXPERIMENT_RECORD.md",
        }

    for match in TABLE_ONLINE_RE.finditer(text):
        version = f"v{int(match.group(1))}"
        row = rows.setdefault(
            version,
            {
                "version": version,
                "method": "",
                "params": "",
                "online": None,
                "conclusion": "See EXPERIMENT_RECORD.md",
            },
        )
        row["online"] = float(match.group(2))

    for match in SCORE_ASSIGN_RE.finditer(text):
        version = f"v{int(match.group(1))}"
        row = rows.setdefault(
            version,
            {
                "version": version,
                "method": "",
                "params": "",
                "online": None,
                "conclusion": "See EXPERIMENT_RECORD.md",
            },
        )
        row["online"] = float(match.group(2))

    return sorted(rows.values(), key=lambda row: int(str(row["version"])[1:]))
