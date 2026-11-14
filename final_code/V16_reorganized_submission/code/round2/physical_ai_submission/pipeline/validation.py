from typing import Any

from ..config import RuntimePaths
from ..data.artifact_io import validate_channel


def run_validation(paths: RuntimePaths) -> dict[str, Any]:
    return validate_channel(paths.output, 500)
