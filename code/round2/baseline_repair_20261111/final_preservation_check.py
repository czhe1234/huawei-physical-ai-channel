"""Final read-only preservation and new-source checks for experiment 832."""
from __future__ import annotations
import ast
import importlib.util
import hashlib
import json
import re
from pathlib import Path
from project_paths import PATHS

R = PATHS.analysis / "results/round2_paper_four_baselines_v1/repair_20261111"
D = PATHS.code_root / "round2/baseline_repair_20261111"

def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def main():
    before = json.loads((R/"preservation_before.json").read_text())
    root = PATHS.code_root.parent
    errors = []
    checked_hashes = {}
    for rel, expected in before["hashes"].items():
        p = root/rel
        actual = sha(p) if p.is_file() else None
        checked_hashes[rel] = {"expected": expected, "actual": actual, "match": actual == expected}
        if actual != expected:
            errors.append("old_hash:"+rel)
    checks = {}
    for category in ("stats", "raw_data_stats"):
        values = []
        for entry in before[category]:
            p = root/entry["path"]
            st = p.stat() if p.exists() else None
            actual = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "inode": st.st_ino} if st else None
            expected = {k:entry[k] for k in ("size","mtime_ns","inode")}
            same = actual == expected
            values.append({"path": entry["path"], "expected": expected, "actual": actual, "match": same})
            if not same:
                errors.append(category+":"+entry["path"])
        checks[category] = values
    with (root/"experiment_records/EXPERIMENT_RECORD.md").open("rb") as f:
        prefix = f.read(before["record_prefix_bytes"])
    prefix_hash = hashlib.sha256(prefix).hexdigest()
    prefix_ok = prefix_hash == before["record_prefix_sha256"]
    if not prefix_ok:
        errors.append("record_prefix_changed")
    spec = importlib.util.spec_from_file_location("repository_path_checker", PATHS.code_root/"tools/check_hardcoded_paths.py")
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    sources = {}
    for p in sorted(D.glob("*.py")):
        rel = str(p.relative_to(root))
        text = p.read_text()
        ast.parse(text,filename=rel)
        # A targeted check supplements the repository path checker.
        absolute_literals = []
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if re.match(r"^/(?:home|Users|workspace)/[^/]+/", node.value):
                    absolute_literals.append(node.lineno)
        pattern_hits = [{"line":i,"pattern":label} for i,line in enumerate(text.splitlines(),1)
                        for label,pattern in checker.PATTERNS if pattern.search(line)]
        sources[rel] = {"sha256": sha(p), "AST": True, "machine_absolute_literal_lines": absolute_literals,
                        "repository_pattern_hits":pattern_hits}
        if pattern_hits:
            errors.append("new_source_repository_path_pattern:"+rel)
        if absolute_literals:
            errors.append("new_source_absolute_path:"+rel)
    frozen_config_sha = sha(R/"config.json")
    expected_config_sha = "d572d97d0c61d8a3789efb09d56e6feab709a3d2566f8d9f32ec9bcc50e2a258"
    if frozen_config_sha != expected_config_sha:
        errors.append("root_config_changed")
    out = {
        "status": "passed" if not errors else "failed",
        "old_hashed_files": checked_hashes,
        "old_model_stat_checks": checks["stats"],
        "raw_dataset_stat_checks": checks["raw_data_stats"],
        "record_prefix": {"bytes": before["record_prefix_bytes"],"sha256":prefix_hash,"match":prefix_ok},
        "new_sources": sources,
        "frozen_root_config_sha256": frozen_config_sha,
        "errors": errors,
        "limits": ["256 old model files and 3 raw arrays checked by stat, not full content hash.",
                   "New source AST and machine absolute literal scan do not substitute for numerical method audits."],
        "formal_test_rerun": False,
        "V16_retrained": False
    }
    target=R/"final_preservation_and_source_check.json"
    target.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"status":out["status"],"old_hashes":len(checked_hashes),
                      "old_model_stats":len(checks["stats"]),"raw_stats":len(checks["raw_data_stats"]),
                      "new_sources":len(sources),"errors":errors},ensure_ascii=False))
    if errors:
        raise SystemExit(1)

if __name__ == "__main__":
    main()

