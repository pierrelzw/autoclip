"""Analysis result persistence."""

from __future__ import annotations

import os

from autoclip.models import AnalysisResult


def write_analysis_json(result: AnalysisResult, path: str) -> None:
    """Serialize AnalysisResult to JSON and write to disk.

    Args:
        result: The analysis result to persist.
        path: File path to write the JSON output.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(result.model_dump_json(indent=2))
