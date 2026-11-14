#!/usr/bin/env python3
from project_paths import PATHS
"""Build a local/online scoreboard from Round1 evaluation JSON files.

The purpose of this script is not to invent a new hidden metric. It puts the
available local validation signals and known online scores in one place so new
candidate decisions are easier to audit.
"""

import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

from experiment_record_index import load_online_rows

KEY_SETS = ("feature_matched", "stress_feature_boundary", "clear_los", "heavy_blocked")


def parse_float(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def param_key(clear, light, medium, heavy):
    vals = (clear, light, medium, heavy)
    if any(v is None for v in vals):
        return None
    return tuple(round(float(v), 6) for v in vals)


def parse_regime_params(text):
    """Parse clear/light/medium/heavy rhos from markdown parameter text."""

    if not text:
        return None
    normalized = text.replace(" ", "")
    full = re.search(
        r"clear=([0-9.]+),light=([0-9.]+),medium=([0-9.]+),heavy=([0-9.]+)",
        normalized,
    )
    if full:
        return param_key(*(float(full.group(i)) for i in range(1, 5)))

    else_match = re.search(r"clear=([0-9.]+),else=([0-9.]+)", normalized)
    if else_match:
        clear = float(else_match.group(1))
        other = float(else_match.group(2))
        return param_key(clear, other, other, other)

    global_rho = re.search(r"rho=([0-9.]+)", normalized)
    if global_rho:
        rho = float(global_rho.group(1))
        return param_key(rho, rho, rho, rho)

    return None


def load_online_table(record_path):
    by_params = {}
    by_version = {}
    rows = []
    for source in load_online_rows(record_path):
        version = str(source["version"])
        method = str(source["method"])
        params = str(source["params"])
        online = parse_float(source["online"])
        key = parse_regime_params(params)
        row = {
            "version": version,
            "method": method,
            "params": params,
            "online": online,
            "param_key": key,
            "conclusion": str(source.get("conclusion") or ""),
        }
        rows.append(row)
        if online is not None:
            by_version[version] = online
            if method == "regime_s1style_geom_blend" and key is not None:
                by_params[key] = max(online, by_params.get(key, -math.inf))
    return by_params, by_version, rows


def load_eval_json(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    combos = {}
    for c in payload.get("combos", []):
        key = param_key(c.get("clear"), c.get("light"), c.get("medium"), c.get("heavy"))
        combos[c["name"]] = {
            "clear": c.get("clear"),
            "light": c.get("light"),
            "medium": c.get("medium"),
            "heavy": c.get("heavy"),
            "param_key": key,
        }

    records = {}
    for set_obj in payload.get("sets", []):
        set_name = set_obj.get("name")
        for rank, row in enumerate(set_obj.get("results", []), start=1):
            name = row["name"]
            rec = records.setdefault(
                name,
                {
                    "name": name,
                    "source": os.path.basename(path),
                    "params": combos.get(name, {}),
                    "sets": {},
                },
            )
            rec["sets"][set_name] = {
                "rank": rank,
                "score": row.get("score"),
                "pas": row.get("pas"),
                "pdp": row.get("pdp"),
                "nmse": row.get("global_nmse", row.get("global_nmse_direct")),
                "inv_nmse": row.get("inv_nmse"),
                "count": row.get("count"),
            }
    return records


def enrich_record(rec, online_by_params):
    sets = rec["sets"]
    params = rec.get("params") or {}
    key = params.get("param_key")
    rec["online"] = online_by_params.get(key) if key is not None else None
    scores = []
    ranks = []
    for set_name in KEY_SETS:
        if set_name in sets:
            val = sets[set_name].get("score")
            if val is not None:
                scores.append(float(val))
            rank = sets[set_name].get("rank")
            if rank is not None:
                ranks.append(float(rank))
    rec["local_mean_score"] = sum(scores) / len(scores) if scores else None
    rec["local_mean_rank"] = sum(ranks) / len(ranks) if ranks else None
    rec["available_sets"] = len(scores)
    rec["risk_notes"] = risk_notes(rec)
    return rec


def risk_notes(rec):
    params = rec.get("params") or {}
    c = params.get("clear")
    l = params.get("light")
    m = params.get("medium")
    h = params.get("heavy")
    notes = []
    if l is not None and l >= 1.0:
        notes.append("light=1.0 已被 v20 线上证伪")
    if m is not None and h is not None and m >= 1.0 and h >= 1.0:
        notes.append("medium/heavy pure geometry 属于当前有效主线")
    if c is not None and c > 0.10:
        notes.append("clear rho 偏高，历史上 clear 区应保守")
    if not notes:
        notes.append("无显著规则风险")
    return "; ".join(notes)


def fmt(x, digits=6):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def params_text(params):
    if not params:
        return "-"
    vals = [params.get(k) for k in ("clear", "light", "medium", "heavy")]
    if any(v is None for v in vals):
        return "-"
    return f"{vals[0]:.2f}/{vals[1]:.2f}/{vals[2]:.2f}/{vals[3]:.2f}"


def record_sort_key(rec):
    online = rec.get("online")
    pending_bonus = 0 if online is not None else 1
    rank = rec.get("local_mean_rank")
    score = rec.get("local_mean_score")
    return (pending_bonus, -(online or -1.0), rank if rank is not None else 999.0, -(score or -1.0))


def build_markdown(records_by_source, online_rows, out_path):
    lines = []
    lines.append("# Round1 Local Scoreboard")
    lines.append("")
    lines.append("本文件由 `code/local_scoreboard.py` 生成。它汇总本地验证集和线上反馈，用来辅助提交决策。")
    lines.append("")
    lines.append("## Interpreting This")
    lines.append("")
    lines.append("- `local_mean_rank` 越低越好，只在该 eval JSON 内比较。")
    lines.append("- `local_mean_score` 是可用验证集 C 的简单平均，不等价于线上分数。")
    lines.append("- `online` 是已知平台反馈；它优先级高于任何本地代理。")
    lines.append("- 当前经验规则：medium/heavy 加强 geometry 有效；light 推到 1.00 已被 v20 证伪；clear 应保持保守。")
    lines.append("")

    lines.append("## Known Online Regime Versions")
    lines.append("")
    lines.append("| version | params clear/light/medium/heavy | online | conclusion |")
    lines.append("| --- | --- | ---: | --- |")
    for row in online_rows:
        if row["method"] != "regime_s1style_geom_blend":
            continue
        key = row["param_key"]
        if key is None:
            continue
        lines.append(
            f"| {row['version']} | {key[0]:.2f}/{key[1]:.2f}/{key[2]:.2f}/{key[3]:.2f} | "
            f"{fmt(row['online'], 6)} | {row['conclusion']} |"
        )
    lines.append("")

    for source in sorted(records_by_source):
        records = sorted(records_by_source[source], key=record_sort_key)
        lines.append(f"## {source}")
        lines.append("")
        lines.append("| candidate | params c/l/m/h | online | local_mean_score | local_mean_rank | feature | stress | clear | heavy | risk notes |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for rec in records:
            sets = rec["sets"]
            lines.append(
                "| {name} | {params} | {online} | {mean_score} | {mean_rank} | "
                "{feature} | {stress} | {clear} | {heavy} | {risk} |".format(
                    name=rec["name"],
                    params=params_text(rec.get("params")),
                    online=fmt(rec.get("online"), 6),
                    mean_score=fmt(rec.get("local_mean_score"), 6),
                    mean_rank=fmt(rec.get("local_mean_rank"), 2),
                    feature=fmt((sets.get("feature_matched") or {}).get("score"), 6),
                    stress=fmt((sets.get("stress_feature_boundary") or {}).get("score"), 6),
                    clear=fmt((sets.get("clear_los") or {}).get("score"), 6),
                    heavy=fmt((sets.get("heavy_blocked") or {}).get("score"), 6),
                    risk=rec.get("risk_notes", ""),
                )
            )
        lines.append("")

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-glob", default=str(PATHS.analysis / "results/json/regime_rho_combo_eval*.json"))
    parser.add_argument("--experiment-record", default=str(PATHS.experiment_records / "EXPERIMENT_RECORD.md"))
    parser.add_argument("--out-json", default=str(PATHS.analysis / "results/json/local_scoreboard.json"))
    parser.add_argument("--out-md", default=str(PATHS.analysis / "results/local_scoreboard.md"))
    args = parser.parse_args()

    online_by_params, _online_by_version, online_rows = load_online_table(args.experiment_record)
    records_by_source = defaultdict(list)
    for path in sorted(glob.glob(args.eval_glob)):
        for rec in load_eval_json(path).values():
            records_by_source[os.path.basename(path)].append(enrich_record(rec, online_by_params))

    json_payload = {
        "key_sets": KEY_SETS,
        "online_regime_rows": online_rows,
        "sources": records_by_source,
    }
    Path(args.out_json).write_text(json.dumps(json_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    build_markdown(records_by_source, online_rows, args.out_md)
    print(json.dumps({"out_json": args.out_json, "out_md": args.out_md, "sources": len(records_by_source)}, indent=2))


if __name__ == "__main__":
    main()
