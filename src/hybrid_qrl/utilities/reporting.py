"""Reusable Markdown primitives for detailed experiment reports."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def format_mean_ci(
    mean: float,
    interval: float | None = None,
    digits: int = 3,
    separator: str = " ± ",
) -> str:
    """Format a mean and optional confidence interval consistently."""

    if interval is None:
        return f"{mean:.{digits}f}"
    return (
        f"{mean:.{digits}f}"
        f"{separator}"
        f"{interval:.{digits}f}"
    )


def markdown_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    padded_divider: bool = False,
) -> str:
    """Render a compact GitHub-flavored Markdown table."""

    header = "| " + " | ".join(headers) + " |"
    if padded_divider:
        divider = "| " + " | ".join("---" for _ in headers) + " |"
    else:
        divider = "|" + "|".join("---" for _ in headers) + "|"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def aligned_markdown_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> str:
    """Render a source-aligned Markdown table for wide audit reports."""

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    header = "| " + " | ".join(
        value.ljust(widths[index]) for index, value in enumerate(headers)
    ) + " |"
    divider = "|" + "|".join("-" * (width + 2) for width in widths) + "|"
    body = [
        "| " + " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(row)
        ) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def find_summary_row(
    summary: Sequence[dict[str, Any]],
    **matching: Any,
) -> dict[str, Any]:
    """Return the first summary row matching all requested dimensions."""

    return next(
        row
        for row in summary
        if all(row.get(key) == value for key, value in matching.items())
    )
