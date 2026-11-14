"""Central path configuration for the wireless-channel workspace.

All maintained code must resolve paths through this module.  The project can
be relocated by moving the root marker or setting ``HW_PROJECT_ROOT``.
"""

from __future__ import annotations
import os
import json
import re
from dataclasses import dataclass
from pathlib import Path


ROOT_MARKER = ".hw-project"
CACHE_RE = re.compile(r"^cache_v(\d+)(?:/(.*))?$")
VERSION_RE = re.compile(r"(?:^|_)v(\d+)(?:_|\.|$)")


def find_project_root(start: Path | None = None) -> Path:
    configured = os.environ.get("HW_PROJECT_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
        if not (root / ROOT_MARKER).is_file():
            raise RuntimeError(f"HW_PROJECT_ROOT does not contain {ROOT_MARKER}: {root}")
        return root

    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / ROOT_MARKER).is_file():
            return candidate
    raise RuntimeError(
        f"Cannot locate {ROOT_MARKER}. Set HW_PROJECT_ROOT to the workspace root."
    )


class ArtifactRouter:
    """Resolve logical artifact names into the structured layout."""

    def __init__(self, paths: "ProjectPaths") -> None:
        self.paths = paths
        self._registry: dict[str, str] | None = None

    def _load_registry(self) -> dict[str, str]:
        if self._registry is None:
            registry_path = self.paths.artifacts / "artifact_registry.json"
            if registry_path.is_file():
                self._registry = json.loads(registry_path.read_text(encoding="utf-8"))
            else:
                self._registry = {}
        return self._registry

    def resolve(self, suffix: str | Path) -> Path:
        key = str(suffix).lstrip("/")
        registry = self._load_registry()
        if key in registry:
            return self.paths.root / registry[key]

        first, separator, remainder = key.partition("/")
        if first in registry:
            base = self.paths.root / registry[first]
            return base / remainder if separator else base

        cache_match = CACHE_RE.match(key)
        if cache_match:
            base = self.paths.cache_dir(cache_match.group(1))
            return base / cache_match.group(2) if cache_match.group(2) else base

        if key.endswith(".py"):
            return self.paths.code / key
        if key.endswith(".pt"):
            return self.paths.checkpoints / key
        if key.endswith((".npy", ".npz", ".obj")):
            return self.paths.shared_features / key
        if key.endswith(".json"):
            versions = VERSION_RE.findall(Path(key).name)
            if versions:
                return self.paths.version_report_dir(versions[-1]) / key
            return self.paths.analysis / "results" / "json" / key
        if key.endswith(".zip"):
            if re.match(r"^V\d+", Path(key).name):
                return self.paths.round2_versions / key
            return self.paths.round1_versions / key
        if key.endswith(".md"):
            return self.paths.analysis / "results" / key
        return self.paths.artifacts / key

    def __add__(self, suffix: str) -> str:
        return str(self.resolve(suffix))

    def __truediv__(self, suffix: str | Path) -> Path:
        return self.resolve(suffix)


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    datasets: Path
    round1: Path
    round2: Path
    analysis: Path
    experiment_records: Path
    versions: Path
    round1_versions: Path
    round2_versions: Path
    code_root: Path
    round1_code: Path
    round2_code: Path
    common_code: Path
    # Historical scripts use PATHS.code for source packaging. Keep it as the
    # Round 1 code directory while new code uses the explicit properties.
    code: Path
    artifacts: Path
    version_cache: Path
    shared_features: Path
    checkpoints: Path
    temporary: Path

    @classmethod
    def load(cls) -> "ProjectPaths":
        root = find_project_root()
        artifacts = root / "artifacts"
        code_root = root / "code"
        round1_code = code_root / "round1"
        versions = root / "versions"
        return cls(
            root=root,
            datasets=root / "datasets",
            round1=root / "datasets" / "round1",
            round2=root / "datasets" / "round2",
            analysis=root / "analysis",
            experiment_records=root / "experiment_records",
            versions=versions,
            round1_versions=versions / "round1",
            round2_versions=versions / "round2",
            code_root=code_root,
            round1_code=round1_code,
            round2_code=code_root / "round2",
            common_code=code_root / "common",
            code=round1_code,
            artifacts=artifacts,
            version_cache=artifacts / "version_cache",
            shared_features=artifacts / "shared_features",
            checkpoints=artifacts / "checkpoints",
            temporary=artifacts / "temporary",
        )

    def version_report_dir(self, version: int | str) -> Path:
        number = str(version).lower().removeprefix("v")
        return self.analysis / "results" / "version_reports" / f"v{int(number)}"

    def cache_dir(self, version: int | str) -> Path:
        number = str(version).lower().removeprefix("v")
        return self.version_cache / f"v{int(number):03d}"

    def resolve(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.root / path).resolve()

    def relative(self, value: str | Path) -> str:
        path = self.resolve(value)
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)

    @property
    def assets(self) -> ArtifactRouter:
        return ArtifactRouter(self)


PATHS = ProjectPaths.load()
