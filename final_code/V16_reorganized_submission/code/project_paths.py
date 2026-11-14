from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


ROOT_MARKER = ".hw-project"


def find_project_root(start: Path | None = None) -> Path:
    configured = os.environ.get("HW_PROJECT_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
        if not (root / ROOT_MARKER).is_file():
            raise RuntimeError(
                f"HW_PROJECT_ROOT does not contain {ROOT_MARKER }: {root }"
            )
        return root

    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / ROOT_MARKER).is_file():
            return candidate
    raise RuntimeError(
        f"Cannot locate {ROOT_MARKER }. Set HW_PROJECT_ROOT to the submission root."
    )


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    datasets: Path
    round2: Path
    artifacts: Path

    @classmethod
    def load(cls) -> "ProjectPaths":
        root = find_project_root()
        datasets = root / "datasets"
        return cls(
            root=root,
            datasets=datasets,
            round2=datasets / "round2",
            artifacts=root / "artifacts",
        )

    def relative(self, value: str | Path) -> str:
        path = Path(value).expanduser().resolve()
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)


PATHS = ProjectPaths.load()
