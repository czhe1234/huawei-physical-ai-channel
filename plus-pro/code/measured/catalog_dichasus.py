"""Freeze the official DICHASUS catalog and DaRUS released file identities.

Uses native curl with certificate verification, including on Windows. Dataset
pages are snapshots, not executable instructions. No channel data is fetched.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import time
from urllib.parse import urljoin, urlparse, parse_qs

BASE = "https://dichasus.inue.uni-stuttgart.de"
CATALOG = BASE + "/datasets/"


class References(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.refs = set()
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in {"href", "src", "data-gltf", "data-potree"} and value:
                self.refs.add(unescape(value))


def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path.read_text(encoding="utf-8-sig")
    tmp = path.with_suffix(path.suffix + ".partial")
    result = subprocess.run([
        "curl.exe" if __import__("os").name == "nt" else "curl",
        "--fail", "--location", "--silent", "--show-error", "--retry", "3",
        "--connect-timeout", "20", "--max-time", "90", "--output", str(tmp), url,
    ], capture_output=True)
    if result.returncode:
        raise RuntimeError("Public metadata fetch failed for " + url)
    tmp.replace(path)
    return path.read_text(encoding="utf-8-sig")


def safe_name(value):
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or "\\" in value or ":" in value:
        raise ValueError("Unsafe source file name")
    return p.as_posix()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.report_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    catalog = fetch(CATALOG, root / "official_catalog.html")
    pages = sorted({urljoin(CATALOG, r) for r in References(catalog).refs
                    if re.search(r"/datasets/data/dichasus-[^/]+/$", urljoin(CATALOG, r))})
    if len(pages) != 14:
        raise RuntimeError("Official catalog changed; review the group count before proceeding")

    def group(url):
        name = url.rstrip("/").split("/")[-1]
        text = fetch(url, root / (name + ".html"))
        dois = sorted(set(x.lower() for x in re.findall(r"10\.18419/[dD][aA][rR][uU][sS]-[0-9]+", text)))
        if len(dois) != 1:
            raise RuntimeError("Review dataset DOI mapping for " + name)
        refs = References(text).refs
        pids = set()
        assets = set()
        for ref in refs:
            absolute = urljoin(url, ref)
            parsed = urlparse(absolute)
            if parsed.netloc == "darus.uni-stuttgart.de" and "/api/access/datafile/" in parsed.path:
                pid = parse_qs(parsed.query).get("persistentId", [None])[0]
                if pid:
                    pids.add(pid.lower())
            if parsed.netloc == "dichasus.inue.uni-stuttgart.de" and absolute.startswith(url):
                if Path(parsed.path).suffix.lower() in {".json", ".xml", ".zip", ".glb", ".gltf", ".obj", ".ply", ".pcd", ".gpx", ".py", ".ipynb", ".csv", ".yaml", ".yml", ".xz", ".gz", ".tar", ".pdf", ".txt"}:
                    assets.add(absolute)
        return {"group": name, "page": url, "doi": dois[0],
                "linked_persistent_ids": sorted(pids), "assets": sorted(assets)}

    with ThreadPoolExecutor(max_workers=4) as pool:
        groups = list(pool.map(group, pages))
    dois = sorted({g["doi"] for g in groups})

    def metadata(doi):
        url = "https://darus.uni-stuttgart.de/api/datasets/:persistentId/?persistentId=doi:" + doi
        data = json.loads(fetch(url, root / (doi.split("/")[-1] + ".json")))
        if data.get("status") != "OK":
            raise RuntimeError("DaRUS metadata not OK: " + doi)
        version = data["data"]["latestVersion"]
        if version.get("versionState") != "RELEASED":
            raise RuntimeError("Require a public released version: " + doi)
        return doi, version

    with ThreadPoolExecutor(max_workers=4) as pool:
        versions = dict(pool.map(metadata, dois))
    entries = {}
    for g in groups:
        version = versions[g["doi"]]
        group_count = 0
        seen_pids = set()
        for item in version["files"]:
            f = item["dataFile"]
            pid = f.get("persistentId", "").lower()
            # Some catalog groups share a DOI. Keep their complete released
            # repository, and mark memberships using the official page links.
            seen_pids.add(pid)
            if item.get("restricted"):
                raise RuntimeError("Restricted file requires a scope review")
            checksum = f["checksum"]
            if checksum["type"].upper() not in {"MD5", "SHA-256", "SHA256"}:
                raise RuntimeError("Unsupported official checksum algorithm")
            fid = str(f["id"])
            filename = safe_name(f["filename"])
            directory = safe_name(item.get("directoryLabel", ""))
            repository_name = str(PurePosixPath(directory) / filename)
            if fid not in entries:
                # Existing dc01 is kept at its canonical, already used path.
                relative = "dc01/" + filename if fid == "269445" else g["group"] + "/" + repository_name
                entries[fid] = {
                    "key": "darus-" + fid, "file_id": int(fid), "filename": filename,
                    "relative_path": relative, "size": int(f["filesize"]),
                    "checksum_type": checksum["type"], "checksum": checksum["value"].lower(),
                    "source": "https://darus.uni-stuttgart.de/api/access/datafile/" + fid,
                    "persistent_id": f.get("persistentId"), "doi": g["doi"],
                    "version": str(version.get("versionNumber")) + "." + str(version.get("versionMinorNumber")),
                    "license": version.get("license", {}), "groups": [],
                    "linked_from_groups": [], "description": item.get("description", ""),
                }
            entries[fid]["groups"].append(g["group"])
            if pid in g["linked_persistent_ids"]:
                entries[fid]["linked_from_groups"].append(g["group"])
            group_count += 1
        missing = set(g["linked_persistent_ids"]) - seen_pids
        if missing:
            raise RuntimeError("Official linked files missing from released repository: " + str(missing))
        g["repository_file_count"] = group_count
        g["version"] = str(version.get("versionNumber")) + "." + str(version.get("versionMinorNumber"))

    def asset_entry(pair):
        g, url = pair
        import hashlib
        cache = root / "asset_headers" / (hashlib.sha256(url.encode()).hexdigest() + ".txt")
        cache.parent.mkdir(exist_ok=True)
        if cache.exists():
            headers = cache.read_bytes()
        else:
            for attempt in range(4):
                result = subprocess.run([
                    "curl.exe" if __import__("os").name == "nt" else "curl",
                    "--head", "--location", "--fail", "--silent", "--show-error", "--retry", "3", "--retry-all-errors",
                    "--connect-timeout", "20", "--max-time", "90", url,
                ], capture_output=True)
                if result.returncode == 0:
                    break
                time.sleep(2 * (attempt + 1))
            if result.returncode:
                raise RuntimeError("Cannot inspect official asset: " + url + ": " + result.stderr.decode("utf-8", "replace"))
            headers = result.stdout
            cache.write_bytes(headers)
        lengths = re.findall(rb"(?im)^content-length:\s*(\d+)\s*$", headers)
        if not lengths:
            raise RuntimeError("Official asset has no verifiable length: " + url)
        rel = safe_name(url[len(g["page"]):])
        return {"key": "web-" + hashlib.sha256(url.encode()).hexdigest()[:20],
                "filename": PurePosixPath(rel).name, "relative_path": g["group"] + "/context/" + rel,
                "size": int(lengths[-1]), "checksum_type": None, "checksum": None,
                "checksum_provenance": "publisher supplies no checksum; verified TLS, HTTP length and local-to-server SHA256",
                "source": url, "doi": g["doi"], "version": g["version"],
                "groups": [g["group"]], "linked_from_groups": [g["group"]]}

    with ThreadPoolExecutor(max_workers=6) as pool:
        assets = list(pool.map(asset_entry, [(g, u) for g in groups for u in g["assets"]]))
    paths = [x["relative_path"] for x in entries.values()] + [x["relative_path"] for x in assets]
    if len(paths) != len(set(paths)):
        raise RuntimeError("Destination path collision")
    files = sorted(entries.values(), key=lambda e: (e["file_id"] != 269445, not e["relative_path"].startswith("dichasus-dcxx/"), e["relative_path"]))
    repository_bytes = sum(e["size"] for e in files)
    # Start with the existing canonical dc01 and the next hallway recording;
    # keep auxiliary files in the same frozen full-catalog scope.
    files = files[:2] + sorted(assets, key=lambda e: e["size"]) + files[2:]
    manifest = {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "catalog_url": CATALOG, "scope": "all 14 official DICHASUS groups, complete released DaRUS repositories",
                "groups": groups, "files": files, "total_files": len(files),
                "repository_files": len(entries), "repository_bytes": repository_bytes,
                "website_asset_files": len(assets), "website_asset_bytes": sum(e["size"] for e in assets),
                "total_bytes": sum(e["size"] for e in files)}
    (root / "catalog_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"groups": len(groups), "repositories": len(dois), "files": len(files),
                      "total_bytes": manifest["total_bytes"], "total_GB": manifest["total_bytes"] / 1e9,
                      "repository_files": len(entries), "website_asset_files": len(assets),
                      "per_group": [{"group": g["group"], "version": g["version"], "files": g["repository_file_count"]} for g in groups]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
