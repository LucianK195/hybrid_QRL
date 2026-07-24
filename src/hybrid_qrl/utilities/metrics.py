"""Small statistical metrics shared by experiments and report renderers."""

from __future__ import annotations

from math import ceil, log


def shots_for_95_percent(success_probability: float) -> int | None:
    """Return iid shots required for 95% chance of at least one success."""

    if success_probability <= 0.0:
        return None
    if success_probability >= 1.0:
        return 1
    return int(ceil(log(0.05) / log(1.0 - success_probability)))
