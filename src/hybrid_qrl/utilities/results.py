"""Consistent, compatibility-preserving result serialization."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ResultWriter:
    """Write experiment artifacts with their established text conventions."""

    json_trailing_newline: bool = True

    def json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = "\n" if self.json_trailing_newline else ""
        path.write_text(
            json.dumps(payload, indent=2) + suffix,
            encoding="utf-8",
        )

    def text(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def artifacts(
        self,
        *,
        json_path: Path,
        report_path: Path,
        payload: Any,
        render_report: Callable[[Any], str],
    ) -> None:
        """Render first, then persist JSON and Markdown in a stable order."""

        report = render_report(payload)
        self.json(json_path, payload)
        self.text(report_path, report)
