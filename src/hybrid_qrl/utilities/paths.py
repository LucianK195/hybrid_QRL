"""Stable project paths shared by experiment applications."""

from __future__ import annotations

from pathlib import Path


def repository_root() -> Path:
    """Return the standalone ``hybrid_qrl`` repository root."""

    return Path(__file__).resolve().parents[3]


def workspace_root() -> Path:
    """Return the parent workspace containing datasets, figures, and results."""

    return repository_root().parent
