"""Create publication-ready SVG figures from dispatch benchmark records.

The script uses only the Python standard library, so plotting remains
reproducible in the minimal NumPy/SciPy benchmark environment. It reads raw
per-seed JSON rather than transcribing report tables and writes editable SVGs.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from html import escape
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METHODS = (
    "beam_search",
    "local_search",
    "simulated_annealing",
    "milp",
    "rydberg_surrogate",
    "mcmc",
    "greedy",
    "autoregressive",
)
LABELS = {
    "beam_search": "Beam",
    "local_search": "Local search",
    "simulated_annealing": "Annealing",
    "milp": "MILP",
    "rydberg_surrogate": "Rydberg surrogate",
    "mcmc": "MCMC",
    "greedy": "Greedy",
    "autoregressive": "Autoregressive",
}
COLORS = {
    "beam_search": "#0072B2",
    "local_search": "#009E73",
    "simulated_annealing": "#E69F00",
    "milp": "#CC79A7",
    "rydberg_surrogate": "#D55E00",
    "mcmc": "#56B4E9",
    "greedy": "#8C8C8C",
    "autoregressive": "#6A51A3",
}
FONT = "Inter,Segoe UI,Arial,sans-serif"


def _mean_ci(values: Iterable[float]) -> tuple[float, float]:
    """Return the sample mean and normal 95% confidence half-width."""

    items = list(values)
    center = mean(items)
    interval = 0.0 if len(items) < 2 else 1.96 * stdev(items) / math.sqrt(len(items))
    return center, interval


def _text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 13,
    anchor: str = "middle",
    weight: int = 400,
    fill: str = "#25313C",
    rotate: int | None = None,
) -> str:
    transform = "" if rotate is None else f' transform="rotate({rotate} {x} {y})"'
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}"{transform}>{escape(value)}</text>'
    )


def _svg(title: str, width: int, height: int, body: str) -> str:
    """Wrap chart primitives in an accessible standalone SVG document."""

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
        f'aria-labelledby="title desc">\n<title id="title">{escape(title)}</title>\n'
        '<desc id="desc">Aggregate dispatch benchmark results with 95 percent '
        'confidence intervals across 20 held-out seeds.</desc>\n'
        f'<rect width="100%" height="100%" fill="#FFFFFF"/>\n{body}\n</svg>\n'
    )


def _axes(
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    x_ticks: list[tuple[float, str]],
    y_ticks: list[tuple[float, str]],
    x_label: str,
    y_label: str,
) -> str:
    parts: list[str] = []
    for y, label in y_ticks:
        parts.append(
            f'<line x1="{left}" x2="{left + width}" y1="{y}" y2="{y}" '
            'stroke="#DFE5EA" stroke-width="1"/>'
        )
        parts.append(_text(left - 10, y + 4, label, anchor="end", fill="#56636F"))
    for x, label in x_ticks:
        parts.append(_text(x, top + height + 22, label, fill="#56636F"))
    parts.append(
        f'<line x1="{left}" x2="{left}" y1="{top}" y2="{top + height}" '
        'stroke="#7C8994"/>'
    )
    parts.append(
        f'<line x1="{left}" x2="{left + width}" y1="{top + height}" '
        f'y2="{top + height}" stroke="#7C8994"/>'
    )
    parts.append(_text(left + width / 2, top + height + 52, x_label, size=14))
    parts.append(
        _text(left - 56, top + height / 2, y_label, size=14, rotate=-90)
    )
    return "\n".join(parts)


def _group_mean(
    rows: Iterable[dict[str, Any]],
    key: Callable[[dict[str, Any]], tuple[Any, ...]],
    metric: str,
) -> dict[tuple[Any, ...], tuple[float, float]]:
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        groups[key(row)].append(float(row[metric]))
    return {name: _mean_ci(values) for name, values in groups.items()}


def scaling_figure(results: dict[str, Any]) -> str:
    """Plot equal-K quality scaling and end-to-end latency scaling."""

    rows = [row for row in results["scaling_records"] if row["mode"] == "equal_k"]
    reward = _group_mean(
        rows, lambda row: (row["method"], row["n_jobs"]), "reward_ratio"
    )
    latency = _group_mean(
        rows, lambda row: (row["method"], row["n_jobs"]), "end_to_end_latency_ms"
    )
    sizes = [20, 40, 60, 100]
    parts = [_text(500, 35, "Equal-K scaling (K = 16)", size=22, weight=650)]
    panels = (
        (70, 75, 385, 390, reward, 0.3, 1.0, "Reward / MILP reference", False),
        (565, 75, 385, 390, latency, 0.4, 1200.0, "End-to-end latency (ms)", True),
    )
    for left, top, width, height, values, low, high, ylabel, logarithmic in panels:
        x = lambda n: left + (n - 20) / 80 * width
        if logarithmic:
            y = lambda value: top + height * (
                1 - (math.log10(value) - math.log10(low))
                / (math.log10(high) - math.log10(low))
            )
            yt = [0.5, 1, 10, 100, 1000]
        else:
            y = lambda value: top + height * (1 - (value - low) / (high - low))
            yt = [0.4, 0.6, 0.8, 1.0]
        parts.append(
            _axes(
                left=left,
                top=top,
                width=width,
                height=height,
                x_ticks=[(x(size), str(size)) for size in sizes],
                y_ticks=[(y(value), f"{value:g}") for value in yt],
                x_label="Binary decisions per state",
                y_label=ylabel,
            )
        )
        for method in METHODS:
            points = [(x(size), y(values[(method, size)][0])) for size in sizes]
            path = " ".join(
                ("M" if index == 0 else "L") + f" {px:.1f} {py:.1f}"
                for index, (px, py) in enumerate(points)
            )
            dash = ' stroke-dasharray="7 4"' if method == "rydberg_surrogate" else ""
            parts.append(
                f'<path d="{path}" fill="none" stroke="{COLORS[method]}" '
                f'stroke-width="{3 if method == "rydberg_surrogate" else 2}"{dash}/>'
            )
            for px, py in points:
                parts.append(
                    f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.5" '
                    f'fill="{COLORS[method]}"/>'
                )
    for index, method in enumerate(METHODS):
        row, column = divmod(index, 4)
        x0 = 120 + column * 220
        y0 = 525 + row * 28
        parts.append(
            f'<line x1="{x0}" x2="{x0 + 28}" y1="{y0}" y2="{y0}" '
            f'stroke="{COLORS[method]}" stroke-width="3"/>'
        )
        parts.append(_text(x0 + 36, y0 + 4, LABELS[method], anchor="start", size=12))
    return _svg("Equal-K dispatch scaling", 1000, 595, "\n".join(parts))


def pareto_figure(results: dict[str, Any]) -> str:
    """Plot equal-latency reward versus measured decision latency at n=100."""

    rows = [
        row
        for row in results["scaling_records"]
        if row["mode"] == "equal_latency" and row["n_jobs"] == 100
    ]
    reward = _group_mean(rows, lambda row: (row["method"],), "reward_ratio")
    latency = _group_mean(
        rows, lambda row: (row["method"],), "end_to_end_latency_ms"
    )
    left, top, width, height = 90, 70, 690, 440
    x = lambda value: left + value / 20.0 * width
    y = lambda value: top + height * (1 - (value - 0.3) / 0.7)
    parts = [
        _text(
            450,
            35,
            "Quality-latency trade-off at 100 decisions",
            size=22,
            weight=650,
        ),
        _axes(
            left=left,
            top=top,
            width=width,
            height=height,
            x_ticks=[(x(v), str(v)) for v in (0, 5, 10, 15, 20)],
            y_ticks=[(y(v), f"{v:.1f}") for v in (0.4, 0.6, 0.8, 1.0)],
            x_label="Measured end-to-end latency (ms)",
            y_label="Reward / MILP reference",
        ),
        f'<line x1="{x(20)}" x2="{x(20)}" y1="{top}" y2="{top + height}" '
        'stroke="#B8C1C9" stroke-dasharray="5 5"/>',
    ]
    label_offsets = {
        "beam_search": (12, -10),
        "local_search": (12, -8),
        "simulated_annealing": (-12, -10),
        "milp": (12, 18),
        "rydberg_surrogate": (12, 18),
        "mcmc": (-12, -10),
        "greedy": (12, -8),
        "autoregressive": (-12, 18),
    }
    for method in METHODS:
        px, py = x(latency[(method,)][0]), y(reward[(method,)][0])
        parts.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="7" fill="{COLORS[method]}" '
            'stroke="#FFFFFF" stroke-width="2"/>'
        )
        dx, dy = label_offsets[method]
        anchor = "end" if dx < 0 else "start"
        parts.append(
            _text(px + dx, py + dy, LABELS[method], anchor=anchor, size=12, weight=550)
        )
    parts.append(
        _text(
            790,
            top + height + 4,
            "20 ms budget",
            anchor="start",
            size=12,
            fill="#56636F",
        )
    )
    return _svg("Dispatch quality-latency Pareto view", 900, 590, "\n".join(parts))


def rollout_figure(results: dict[str, Any]) -> str:
    """Plot 12-step episode return with per-method confidence intervals."""

    grouped = _group_mean(
        results["rollout_records"], lambda row: (row["method"],), "episode_return"
    )
    ordered = sorted(METHODS, key=lambda method: grouped[(method,)][0], reverse=True)
    left, top, width, row_height = 185, 75, 650, 54
    low, high = 1.8, 2.55
    x = lambda value: left + (value - low) / (high - low) * width
    parts = [
        _text(
            460,
            35,
            "Dynamic rollout return (12 decisions)",
            size=22,
            weight=650,
        )
    ]
    for tick in (1.8, 2.0, 2.2, 2.4):
        px = x(tick)
        parts.append(
            f'<line x1="{px}" x2="{px}" y1="{top - 12}" '
            f'y2="{top + row_height * len(ordered)}" stroke="#DFE5EA"/>'
        )
        parts.append(_text(px, top + row_height * len(ordered) + 26, f"{tick:.1f}"))
    for index, method in enumerate(ordered):
        center, interval = grouped[(method,)]
        py = top + index * row_height + 18
        bar_x = x(low)
        bar_width = x(center) - bar_x
        parts.append(
            f'<rect x="{bar_x:.1f}" y="{py - 11:.1f}" width="{bar_width:.1f}" '
            f'height="22" rx="3" fill="{COLORS[method]}" opacity="0.82"/>'
        )
        parts.append(
            f'<line x1="{x(center - interval):.1f}" x2="{x(center + interval):.1f}" '
            f'y1="{py}" y2="{py}" stroke="#17212B" stroke-width="2"/>'
        )
        for cap in (center - interval, center + interval):
            parts.append(
                f'<line x1="{x(cap):.1f}" x2="{x(cap):.1f}" '
                f'y1="{py - 5}" y2="{py + 5}" stroke="#17212B" stroke-width="2"/>'
            )
        parts.append(_text(left - 12, py + 4, LABELS[method], anchor="end", size=13))
        parts.append(
            _text(x(center) + 9, py + 4, f"{center:.3f}", anchor="start", size=12)
        )
    parts.append(
        _text(
            left + width / 2,
            top + row_height * len(ordered) + 56,
            "Episode return",
            size=14,
        )
    )
    return _svg("Dynamic dispatch rollout returns", 920, 590, "\n".join(parts))


def robustness_figure(results: dict[str, Any]) -> str:
    """Plot the Rydberg surrogate's principal robustness sensitivities."""

    rows = [
        row
        for row in results["robustness_records"]
        if row["method"] == "rydberg_surrogate"
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["axis"], str(row["level"]))].append(row)

    facets = (
        ("density", ["0.05", "0.12", "0.25"], "Reward ratio", "reward_ratio"),
        (
            "geometry_error",
            ["0.0", "0.03", "0.08"],
            "Raw feasibility",
            "raw_feasible_rate",
        ),
        (
            "readout_noise",
            ["0.0", "0.01", "0.05"],
            "Raw feasibility",
            "raw_feasible_rate",
        ),
        (
            "pulse_schedule",
            ["short", "balanced", "adiabatic"],
            "Reward ratio",
            "reward_ratio",
        ),
    )
    titles = {
        "density": "Graph density",
        "geometry_error": "Geometry error",
        "readout_noise": "Readout noise",
        "pulse_schedule": "Pulse schedule",
    }
    parts = [_text(500, 35, "Rydberg-surrogate robustness", size=22, weight=650)]
    for index, (axis, levels, ylabel, metric) in enumerate(facets):
        left = 65 + (index % 2) * 500
        top = 80 + (index // 2) * 290
        width, height = 385, 185
        x = lambda item: left + (item + 0.5) / len(levels) * width
        y = lambda value: top + height * (1 - value)
        parts.append(
            _text(
                left + width / 2,
                top - 18,
                titles[axis],
                size=16,
                weight=600,
            )
        )
        parts.append(
            _axes(
                left=left,
                top=top,
                width=width,
                height=height,
                x_ticks=[(x(i), level) for i, level in enumerate(levels)],
                y_ticks=[(y(v), f"{v:.1f}") for v in (0.0, 0.5, 1.0)],
                x_label="Setting",
                y_label=ylabel,
            )
        )
        for item, level in enumerate(levels):
            center, interval = _mean_ci(
                float(row[metric]) for row in grouped[(axis, level)]
            )
            px, py = x(item), y(center)
            parts.append(
                f'<line x1="{px}" x2="{px}" y1="{y(min(1, center + interval))}" '
                f'y2="{y(max(0, center - interval))}" stroke="#D55E00" '
                'stroke-width="2"/>'
            )
            parts.append(f'<circle cx="{px}" cy="{py}" r="6" fill="#D55E00"/>')
            parts.append(_text(px, py - 12, f"{center:.2f}", size=11, weight=550))
    return _svg("Rydberg surrogate robustness", 1000, 660, "\n".join(parts))


def graph_gallery(records: list[dict[str, Any]]) -> str:
    """Show one physical test graph at each benchmark decision dimension."""

    selected = {
        size: next(row for row in records if int(row["n_jobs"]) == size)
        for size in (20, 40, 60, 100)
    }
    parts = [
        _text(
            500,
            34,
            "Held-out neutral-atom-compatible test graphs",
            size=22,
            weight=650,
        )
    ]
    for index, (size, record) in enumerate(selected.items()):
        left = 40 + index * 245
        top = 80
        plot = 205
        positions = record["positions"]
        weights = record["objective_weights"]
        low, high = min(weights), max(weights)
        px = lambda value: left + 10 + float(value) * (plot - 20)
        py = lambda value: top + 10 + (1 - float(value)) * (plot - 20)
        parts.append(
            _text(
                left + plot / 2,
                top - 14,
                f"n = {size}",
                size=16,
                weight=600,
            )
        )
        parts.append(
            f'<rect x="{left}" y="{top}" width="{plot}" height="{plot}" '
            'fill="#F8FAFC" stroke="#CCD5DD"/>'
        )
        for a, b in record["edges"]:
            parts.append(
                f'<line x1="{px(positions[a][0]):.1f}" y1="{py(positions[a][1]):.1f}" '
                f'x2="{px(positions[b][0]):.1f}" y2="{py(positions[b][1]):.1f}" '
                'stroke="#AAB5BE" stroke-width="0.8" opacity="0.55"/>'
            )
        for node, position in enumerate(positions):
            normalized = (weights[node] - low) / max(high - low, 1e-12)
            red = int(48 + 190 * normalized)
            blue = int(210 - 145 * normalized)
            color = f"rgb({red},90,{blue})"
            parts.append(
                f'<circle cx="{px(position[0]):.1f}" cy="{py(position[1]):.1f}" '
                f'r="{5 if size <= 40 else 3.8}" fill="{color}" stroke="#FFFFFF" '
                'stroke-width="1"/>'
            )
        parts.append(
            _text(
                left + plot / 2,
                top + plot + 24,
                f"{len(record['edges'])} conflict edges",
                size=12,
                fill="#56636F",
            )
        )
    parts.append(
        _text(
            500,
            350,
            "Node color: low objective weight (blue) to high (red)",
            size=13,
        )
    )
    return _svg("Held-out dispatch graph examples", 1020, 380, "\n".join(parts))


def parse_args() -> argparse.Namespace:
    """Parse benchmark, dataset, and output paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=PROJECT_ROOT / "results" / "dispatch_benchmark_results.json",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "dispatch_test_v1.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "figures" / "dispatch_benchmark",
    )
    return parser.parse_args()


def main() -> None:
    """Aggregate raw trials and write the benchmark figure set."""

    args = parse_args()
    results = json.loads(args.results.read_text(encoding="utf-8"))
    graph_records = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    figures = {
        "01_equal_k_scaling.svg": scaling_figure(results),
        "02_quality_latency_n100.svg": pareto_figure(results),
        "03_dynamic_rollout_return.svg": rollout_figure(results),
        "04_rydberg_robustness.svg": robustness_figure(results),
        "05_test_graph_examples.svg": graph_gallery(graph_records),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in figures.items():
        path = args.output_dir / filename
        path.write_text(content, encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
