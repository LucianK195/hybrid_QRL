"""Publication-ready figures for the public Wi-Fi MWIS benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


COLORS = {
    "ideal_rydberg": "#C23B3B",
    "randomized_greedy": "#3677A8",
    "one_swap_local_search": "#4C9A61",
    "simulated_annealing": "#7A5AA6",
    "beam_width_16": "#C78D2A",
    "exact_enumeration": "#333333",
}

LABELS = {
    "ideal_rydberg": "里德堡采样器",
    "randomized_greedy": "随机加权贪心",
    "one_swap_local_search": "单交换局部搜索",
    "simulated_annealing": "模拟退火",
    "beam_width_16": "宽度为 16 的束搜索",
    "exact_enumeration": "精确枚举",
}

FAMILY_ORDER = ("bottleneck", "random", "crowded", "corridor")
FAMILY_LABELS = {
    "bottleneck": "中心干扰瓶颈",
    "random": "随机热点",
    "crowded": "高密拥挤热点",
    "corridor": "走廊链式干扰",
}


def _find(rows: list[dict[str, Any]], **matching: Any) -> dict[str, Any]:
    return next(
        row
        for row in rows
        if all(row.get(key) == value for key, value in matching.items())
    )


def _style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "svg.fonttype": "none",
            "mathtext.fontset": "dejavusans",
        }
    )


def _save(fig, output_dir: Path, stem: str, *, tight: bool = True) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_options = {"bbox_inches": "tight"} if tight else {}
    fig.savefig(output_dir / f"{stem}.png", facecolor="white", **save_options)
    fig.savefig(output_dir / f"{stem}.svg", facecolor="white", **save_options)


def _draw_network(
    axis,
    positions: np.ndarray,
    edges: list[list[int]],
    action: np.ndarray,
    utilities: np.ndarray,
    *,
    atom_view: bool,
    radius: float,
) -> None:
    import matplotlib.patches as patches

    for left, right in edges:
        axis.plot(
            [positions[left, 0], positions[right, 0]],
            [positions[left, 1], positions[right, 1]],
            color="#B6BDC5",
            linewidth=1.0,
            zorder=1,
        )
    if atom_view:
        for point in positions:
            axis.add_patch(
                patches.Circle(
                    point,
                    radius=radius,
                    facecolor="#D95F5F",
                    edgecolor="none",
                    alpha=0.055,
                    zorder=0,
                )
            )
    colors = np.where(action > 0, "#C23B3B" if atom_view else "#2A9D68", "#DCE1E6")
    edges_color = np.where(action > 0, "#7A1616" if atom_view else "#17613E", "#6F7780")
    sizes = 230.0 + 180.0 * (utilities - utilities.min()) / (
        np.ptp(utilities) + 1e-12
    )
    axis.scatter(
        positions[:, 0],
        positions[:, 1],
        s=sizes,
        c=colors,
        edgecolors=edges_color,
        linewidths=1.1,
        zorder=3,
    )
    for node, point in enumerate(positions):
        label = (
            f"{node}"
            if not atom_view
            else (r"$|r\rangle$" if action[node] else r"$|0\rangle$")
        )
        axis.text(
            point[0],
            point[1],
            label,
            ha="center",
            va="center",
            fontsize=7.7,
            color="white" if action[node] else "#30363B",
            fontweight="bold",
            zorder=4,
        )
    axis.set_aspect("equal")
    axis.grid(False)
    axis.set_xticks([])
    axis.set_yticks([])


def _plot_test_scenarios_overview(output_dir: Path) -> None:
    """Rebuild the scenario overview as native vector artwork."""

    import matplotlib.lines as mlines
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11.733333, 6.6), facecolor="white")

    def rounded_panel(
        bounds: tuple[float, float, float, float],
        *,
        edge: str = "#D6DEE8",
        face: str = "white",
        linewidth: float = 0.9,
        radius: float = 0.012,
    ) -> None:
        x, y, width, height = bounds
        fig.add_artist(
            patches.FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle=f"round,pad=0.007,rounding_size={radius}",
                transform=fig.transFigure,
                facecolor=face,
                edgecolor=edge,
                linewidth=linewidth,
                zorder=-10,
                clip_on=False,
            )
        )

    def draw_mini_network(
        bounds: tuple[float, float, float, float],
        positions: np.ndarray,
        edges: list[tuple[int, int]],
        colors: list[str],
        *,
        edge_color: str,
        labels: list[str] | None = None,
    ) -> None:
        axis = fig.add_axes(bounds)
        for left, right in edges:
            axis.plot(
                [positions[left, 0], positions[right, 0]],
                [positions[left, 1], positions[right, 1]],
                color=edge_color,
                linewidth=0.75,
                alpha=0.62,
                zorder=1,
            )
        axis.scatter(
            positions[:, 0],
            positions[:, 1],
            s=72,
            c=colors,
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )
        if labels is not None:
            for point, label in zip(positions, labels):
                if label:
                    axis.text(
                        point[0],
                        point[1],
                        label,
                        ha="center",
                        va="center",
                        fontsize=5.9,
                        color="white",
                        zorder=4,
                    )
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_aspect("equal")
        axis.axis("off")

    # Header.
    fig.text(0.035, 0.955, "测试场景与设备优先程度说明", fontsize=18, color="#182236")
    fig.text(
        0.035,
        0.915,
        "根据 12 个设备的位置和通信需求，构造三类典型干扰场景",
        fontsize=10.5,
        color="#647286",
    )
    fig.text(0.965, 0.945, "测试场景总览", ha="right", fontsize=7.5, color="#91A0B2")
    fig.add_artist(
        mlines.Line2D(
            [0.035, 0.965],
            [0.895, 0.895],
            transform=fig.transFigure,
            color="#D4DCE6",
            linewidth=1.0,
        )
    )

    # Left: one weighted test scenario.
    rounded_panel((0.03, 0.105, 0.39, 0.745))
    fig.text(0.055, 0.815, "—  一个测试场景", fontsize=13.5, color="#182236")
    fig.text(
        0.055,
        0.782,
        "12 个待传输设备 · 平面位置 · 设备重要程度",
        fontsize=8.6,
        color="#647286",
    )

    weighted_positions = np.asarray(
        [
            (0.20, 0.77),
            (0.42, 0.88),
            (0.39, 0.74),
            (0.67, 0.86),
            (0.83, 0.72),
            (0.28, 0.48),
            (0.53, 0.49),
            (0.70, 0.61),
            (0.78, 0.44),
            (0.23, 0.16),
            (0.61, 0.17),
            (0.76, 0.15),
        ],
        dtype=float,
    )
    weighted_values = np.asarray(
        [0.69, 0.56, 0.85, 0.32, 0.68, 0.88, 0.30, 0.85, 0.53, 0.76, 0.20, 0.61]
    )
    weighted_edges = [
        (0, 1),
        (1, 2),
        (2, 5),
        (2, 6),
        (2, 7),
        (3, 4),
        (4, 7),
        (4, 8),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 11),
    ]
    weighted_axis = fig.add_axes((0.085, 0.405, 0.275, 0.315))
    for left, right in weighted_edges:
        weighted_axis.plot(
            [weighted_positions[left, 0], weighted_positions[right, 0]],
            [weighted_positions[left, 1], weighted_positions[right, 1]],
            color="#9AA8BA",
            linestyle=(0, (3, 2)),
            linewidth=0.85,
            alpha=0.75,
            zorder=1,
        )
    weighted_colors = plt.get_cmap("YlOrRd")(
        0.34 + 0.56 * (weighted_values - weighted_values.min()) / np.ptp(weighted_values)
    )
    weighted_axis.scatter(
        weighted_positions[:, 0],
        weighted_positions[:, 1],
        s=265,
        c=weighted_colors,
        edgecolors="white",
        linewidths=1.0,
        zorder=3,
    )
    value_offsets = (
        (0.00, -0.095),
        (0.085, -0.010),
        (0.00, -0.095),
        (0.00, -0.095),
        (0.00, -0.095),
        (0.00, -0.095),
        (0.00, -0.095),
        (0.00, -0.095),
        (0.00, -0.095),
        (0.00, -0.095),
        (0.00, -0.095),
        (0.00, -0.095),
    )
    for index, (point, value, offset) in enumerate(
        zip(weighted_positions, weighted_values, value_offsets), start=1
    ):
        weighted_axis.text(
            point[0],
            point[1],
            f"D{index}",
            ha="center",
            va="center",
            fontsize=6.2,
            color="white" if value >= 0.60 else "#182236",
            zorder=4,
        )
        weighted_axis.text(
            point[0] + offset[0],
            point[1] + offset[1],
            rf"$\omega={value:.2f}$",
            ha="center",
            va="top",
            fontsize=5.7,
            color="#263143",
        )
    weighted_axis.text(
        0.5,
        1.02,
        "距离较近 → 两台设备会互相干扰",
        transform=weighted_axis.transAxes,
        ha="center",
        fontsize=7.0,
        color="#647286",
    )
    weighted_axis.set_xlim(0.0, 1.0)
    weighted_axis.set_ylim(0.0, 1.0)
    weighted_axis.set_aspect("equal")
    weighted_axis.axis("off")

    legend_axis = fig.add_axes((0.12, 0.355, 0.22, 0.035))
    legend_axis.axis("off")
    legend_axis.scatter([0.03, 0.39], [0.5, 0.5], s=[36, 58], c=["#FDBB43", "#D7191C"])
    legend_axis.text(0.10, 0.5, r"$\omega$ 较小", va="center", fontsize=6.6, color="#263143")
    legend_axis.text(0.47, 0.5, r"$\omega$ 较大", va="center", fontsize=6.6, color="#263143")
    legend_axis.plot([0.73, 0.83], [0.5, 0.5], color="#8795A8", linestyle=(0, (3, 2)), linewidth=1.0)
    legend_axis.text(0.88, 0.5, "不能同时传输", va="center", fontsize=6.6, color="#263143")

    rounded_panel((0.055, 0.145, 0.335, 0.17), edge="#E1E7EE")
    fig.text(0.078, 0.285, r"设备被优先安排的程度（$\omega$）", fontsize=9.5, color="#182236")
    fig.text(
        0.078,
        0.245,
        r"$\omega_i=0.45\hat{q}_i+0.35\hat{d}_i+0.20p_i$",
        fontsize=12,
        color="#182236",
    )
    bar_y = 0.205
    for x, width, color in (
        (0.078, 0.145, "#326BFF"),
        (0.223, 0.112, "#18A89F"),
        (0.335, 0.070, "#F59E0B"),
    ):
        fig.add_artist(
            patches.Rectangle(
                (x, bar_y),
                width,
                0.022,
                transform=fig.transFigure,
                facecolor=color,
                edgecolor="none",
            )
        )
    fig.text(0.078, 0.170, "●  待发送数据量 45%", fontsize=6.7, color="#326BFF")
    fig.text(0.210, 0.170, "●  等待紧迫程度 35%", fontsize=6.7, color="#18A89F")
    fig.text(0.346, 0.170, "●  业务优先级 20%", fontsize=6.7, color="#F59E0B")

    # Right: the three reference interference families.
    fig.text(0.455, 0.815, "—  三类典型干扰场景", fontsize=13.5, color="#182236")
    fig.text(
        0.455,
        0.782,
        "设备的位置决定谁会互相干扰，不同分布方式带来不同安排难度",
        fontsize=8.6,
        color="#647286",
    )
    fig.add_artist(
        patches.Rectangle(
            (0.455, 0.725), 0.004, 0.030, transform=fig.transFigure, facecolor="#326BFF", edgecolor="none"
        )
    )
    fig.text(
        0.468,
        0.737,
        "两台设备距离小于设定范围时视为互相干扰，由此得到三类典型测试场景。",
        fontsize=8.1,
        color="#263143",
    )

    card_bounds = (
        (0.455, 0.25, 0.16, 0.43),
        (0.645, 0.25, 0.16, 0.43),
        (0.835, 0.25, 0.15, 0.43),
    )
    rounded_panel(card_bounds[0])
    rounded_panel(card_bounds[1])
    rounded_panel(card_bounds[2], edge="#F2B33D", face="#FFF8E8", linewidth=1.15)
    titles = ("1  随机分布场景", "2  密集拥挤场景", "3  中心阻塞场景")
    subtitles = ("自然散布 · 干扰程度中等", "设备集中 · 干扰关系较多", "1 个中心阻塞 5 个外围设备")
    for bounds, title, subtitle in zip(card_bounds, titles, subtitles):
        x, y, _, h = bounds
        fig.text(x + 0.018, y + h - 0.045, title, fontsize=9.1, color="#182236")
        fig.text(x + 0.018, y + h - 0.080, subtitle, fontsize=6.8, color="#647286")

    random_positions = np.asarray(
        [(0.16, 0.80), (0.34, 0.77), (0.52, 0.79), (0.72, 0.71), (0.75, 0.74), (0.12, 0.49),
         (0.24, 0.24), (0.32, 0.19), (0.39, 0.22), (0.47, 0.31), (0.55, 0.40), (0.70, 0.15)]
    )
    random_edges = [
        (0, 1), (1, 2), (2, 3), (2, 4), (3, 4), (5, 6), (6, 7), (6, 8), (6, 9),
        (7, 8), (7, 9), (8, 9), (8, 10), (9, 10), (9, 11), (10, 11), (6, 10),
    ]
    draw_mini_network(
        (0.478, 0.355, 0.115, 0.205),
        random_positions,
        random_edges,
        ["#326BFF"] * 12,
        edge_color="#AAB5C3",
    )

    crowded_positions = np.asarray(
        [(0.50, 0.83), (0.60, 0.62), (0.43, 0.45), (0.53, 0.49), (0.64, 0.47), (0.37, 0.35),
         (0.49, 0.34), (0.59, 0.36), (0.69, 0.34), (0.39, 0.22), (0.56, 0.19), (0.65, 0.25)]
    )
    all_crowded_edges = [(left, right) for left in range(12) for right in range(left + 1, 12)]
    crowded_edges = all_crowded_edges[:55]
    draw_mini_network(
        (0.668, 0.355, 0.115, 0.205),
        crowded_positions,
        crowded_edges,
        ["#F59E0B"] * 12,
        edge_color="#F4C77B",
    )

    bottleneck_positions = np.asarray(
        [(0.50, 0.50), (0.50, 0.75), (0.74, 0.63), (0.72, 0.35), (0.50, 0.20), (0.27, 0.35),
         (0.26, 0.63), (0.14, 0.86), (0.85, 0.84), (0.14, 0.10), (0.33, 0.07), (0.70, 0.08)]
    )
    bottleneck_edges = [(0, outer) for outer in range(1, 6)]
    bottleneck_colors = ["#E64B3C"] + ["#19A974"] * 5 + ["#AAB6C8"] * 6
    bottleneck_labels = ["D1", "D3", "D2", "D6", "D5", "D4"] + [""] * 6
    draw_mini_network(
        (0.858, 0.355, 0.115, 0.205),
        bottleneck_positions,
        bottleneck_edges,
        bottleneck_colors,
        edge_color="#EF6D5D",
        labels=bottleneck_labels,
    )

    metadata = (
        "12 个设备 · 17 对相互干扰 · 干扰占比=0.26",
        "12 个设备 · 55 对相互干扰 · 干扰占比=0.83",
        "12 个设备 · 5 对相互干扰 · 干扰占比=0.08",
    )
    for bounds, line in zip(card_bounds, metadata):
        fig.text(bounds[0] + bounds[2] / 2, 0.285, line, ha="center", fontsize=6.4, color="#647286")

    rounded_panel((0.455, 0.105, 0.53, 0.095), edge="#AEE4CE", face="#EEF9F4")
    fig.text(0.472, 0.170, "最值得关注的场景", fontsize=9.2, color="#0DAA72")
    fig.text(0.472, 0.127, "中心设备会干扰 5 个外围设备，但外围设备之间互不干扰：", fontsize=7.2, color="#263143")
    fig.text(0.765, 0.170, "选择 1 个中心", fontsize=7.5, color="#E64B3C")
    fig.text(0.842, 0.170, "对比", fontsize=7.3, color="#647286")
    fig.text(0.880, 0.170, "同时调度 5 个外围", fontsize=7.5, color="#0DAA72")
    fig.text(
        0.965,
        0.127,
        "系统需要权衡：优先安排一个重要设备，还是同时安排多个外围设备",
        ha="right",
        fontsize=6.8,
        color="#647286",
    )

    fig.text(
        0.035,
        0.052,
        r"怎么看这张图：圆点是待传输设备；连线表示两台设备不能同时传输；圆点越大、颜色越深，$\omega$ 越大。",
        fontsize=7.0,
        color="#647286",
    )
    fig.text(
        0.965,
        0.052,
        "中心阻塞场景中的 6 个灰色设备只用于补充周围环境。",
        ha="right",
        fontsize=6.8,
        color="#647286",
    )

    _save(fig, output_dir, "00_test_scenarios_white_background", tight=False)
    plt.close(fig)


def _set_mapping_view(axis, positions: np.ndarray, radius: float) -> None:
    """Use the same physical plot window for the Wi-Fi and atom encodings."""

    lower = positions.min(axis=0) - radius
    upper = positions.max(axis=0) + radius
    padding = 0.05 * (upper - lower)
    axis.set_xlim(float(lower[0] - padding[0]), float(upper[0] + padding[0]))
    axis.set_ylim(float(lower[1] - padding[1]), float(upper[1] + padding[1]))


def _plot_mapping(results: dict[str, Any], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    record = next(
        row for row in results["records"] if row["family"] == "bottleneck"
    )
    positions = np.asarray(record["positions"], dtype=float)
    utilities = np.asarray(record["utilities"], dtype=float)
    optimum = np.asarray(record["exact_reference"]["optimal_action"], dtype=int)
    edges = record["graph_edges"]
    radius = float(record["interference_radius"])

    figure_size = (6.8, 5.2)
    fig_a, axis_a = plt.subplots(figsize=figure_size)
    _draw_network(
        axis_a,
        positions,
        edges,
        optimum,
        utilities,
        atom_view=False,
        radius=radius,
    )
    _set_mapping_view(axis_a, positions, radius)
    axis_a.set_title("A  公共 Wi-Fi 干扰图", loc="left", fontweight="bold")
    axis_a.set_xlabel(
        "顶点：排队中的传输；边：不能共享同一时隙\n"
        "绿色节点：被选入最大加权独立集的传输"
    )
    fig_a.subplots_adjust(left=0.125, right=0.90, top=0.88, bottom=0.25)
    _save(fig_a, output_dir, "01A_wifi_interference_graph", tight=False)
    plt.close(fig_a)

    scale = 45.0
    atom_positions = positions * scale
    fig_b, axis_b = plt.subplots(figsize=figure_size)
    _draw_network(
        axis_b,
        atom_positions,
        edges,
        optimum,
        utilities,
        atom_view=True,
        radius=radius * scale,
    )
    _set_mapping_view(axis_b, atom_positions, radius * scale)
    axis_b.set_title("B  中性原子里德堡编码", loc="left", fontweight="bold")
    axis_b.set_xlabel(
        "每个决策对应一个原子；里德堡阻塞实现冲突约束\n"
        r"红色 $|r\rangle$ 原子表示 $x_i=1$",
        labelpad=10,
    )
    axis_b.text(
        0.5,
        -0.24,
        r"$H(t)=\frac{\Omega(t)}{2}\sum_i X_i"
        r"-\Delta(t)\sum_i w_i n_i"
        r"+U\sum_{(i,j)\in E}n_i n_j$",
        transform=axis_b.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        color="#333333",
    )
    fig_b.subplots_adjust(left=0.125, right=0.90, top=0.88, bottom=0.25)
    _save(fig_b, output_dir, "01B_rydberg_encoding", tight=False)
    plt.close(fig_b)


def _draw_candidate_budget_panel(
    axis,
    summary: list[dict[str, Any]],
    budgets: list[int],
    methods: tuple[str, ...],
    *,
    metric: str,
    uncertainty: str,
    title: str,
    ylabel: str,
    ylim: tuple[float, float],
    show_legend: bool,
) -> None:
    for method in methods:
        rows = [
            _find(
                summary,
                family="bottleneck",
                method=method,
                candidates_k=budget,
            )
            for budget in budgets
        ]
        axis.errorbar(
            budgets,
            [row[metric] for row in rows],
            yerr=[row[uncertainty] for row in rows],
            marker="o",
            markersize=4.5,
            capsize=2.5,
            linewidth=1.6,
            color=COLORS[method],
            label=LABELS[method],
        )
    axis.set_title(title, loc="left", fontweight="bold")
    axis.set_xlabel("候选预算 $K$")
    axis.set_ylabel(ylabel)
    axis.set_xscale("log", base=2)
    axis.set_xticks(budgets, labels=[str(value) for value in budgets])
    axis.set_ylim(*ylim)
    if show_legend:
        axis.legend(ncol=2, fontsize=8)


def _draw_paired_difference_panel(
    axis,
    paired: list[dict[str, Any]],
    advantage_k: int,
    *,
    annotate_values: bool,
) -> None:
    families = FAMILY_ORDER
    family_labels = tuple(FAMILY_LABELS[family] for family in families)
    rows = [
        _find(
            paired,
            family=family,
            comparator="randomized_greedy",
            candidates_k=advantage_k,
        )
        for family in families
    ]
    x = np.arange(len(families))
    deltas = [row["quantum_minus_classical_mean"] for row in rows]
    intervals = [row["quantum_minus_classical_ci95"] for row in rows]
    colors = ["#C23B3B" if value > 0 else "#78828C" for value in deltas]
    axis.bar(
        x,
        deltas,
        yerr=intervals,
        capsize=4,
        color=colors,
        width=0.62,
    )
    axis.axhline(0.0, color="#222222", linewidth=1.0)
    axis.set_xticks(x, labels=family_labels)
    lower = min(value - interval for value, interval in zip(deltas, intervals, strict=True))
    upper = max(value + interval for value, interval in zip(deltas, intervals, strict=True))
    padding = 0.12 * max(upper - lower, 1e-6)
    axis.set_ylim(min(0.0, lower) - padding, max(0.0, upper) + padding)
    axis.set_ylabel("差值（里德堡采样 − 随机贪心）")
    axis.set_title(
        f"C  不同测试族的配对性能差值（$K={advantage_k}$）",
        loc="left",
        fontweight="bold",
    )
    if not annotate_values:
        return

    for index, (value, interval) in enumerate(zip(deltas, intervals, strict=True)):
        offset = 0.00035 if value >= 0 else -0.00035
        label_y = value + interval + offset if value >= 0 else value - interval + offset
        label = f"{value:+.4f} ± {interval:.4f}".replace("-", "−")
        axis.text(
            index,
            label_y,
            label,
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
            fontweight="bold",
            color="#30363B",
        )
    axis.text(
        0.98,
        0.96,
        "正值表示里德堡采样表现更好",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color="#4F565D",
    )


def _draw_paired_difference_heatmap(
    fig,
    axis,
    colorbar_axis,
    paired: list[dict[str, Any]],
    candidate_budgets: list[int],
) -> None:
    from matplotlib.colors import TwoSlopeNorm

    families = FAMILY_ORDER
    family_labels = tuple(FAMILY_LABELS[family] for family in families)
    rows = [
        [
            _find(
                paired,
                family=family,
                comparator="randomized_greedy",
                candidates_k=budget,
            )
            for family in families
        ]
        for budget in candidate_budgets
    ]
    deltas = np.asarray(
        [
            [row["quantum_minus_classical_mean"] for row in budget_rows]
            for budget_rows in rows
        ],
        dtype=float,
    )
    intervals = np.asarray(
        [
            [row["quantum_minus_classical_ci95"] for row in budget_rows]
            for budget_rows in rows
        ],
        dtype=float,
    )
    color_limit = float(np.max(np.abs(deltas)))
    normalization = TwoSlopeNorm(
        vmin=-color_limit,
        vcenter=0.0,
        vmax=color_limit,
    )
    heatmap = axis.imshow(
        deltas,
        cmap="RdBu_r",
        norm=normalization,
        aspect="auto",
        interpolation="nearest",
    )
    axis.set_xticks(np.arange(len(families)), labels=family_labels)
    axis.set_yticks(
        np.arange(len(candidate_budgets)),
        labels=[str(budget) for budget in candidate_budgets],
    )
    axis.set_xlabel("测试场景")
    axis.set_ylabel(r"候选预算 $K$")
    axis.tick_params(axis="both", length=0)
    axis.set_xticks(np.arange(-0.5, len(families), 1.0), minor=True)
    axis.set_yticks(np.arange(-0.5, len(candidate_budgets), 1.0), minor=True)
    axis.grid(which="major", visible=False)
    axis.grid(which="minor", color="white", linewidth=1.5)
    axis.tick_params(which="minor", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_title(
        "C  候选预算与测试族的配对性能差值矩阵",
        loc="left",
        fontweight="bold",
        pad=12,
    )

    for row_index, budget_rows in enumerate(deltas):
        for column, value in enumerate(budget_rows):
            interval = intervals[row_index, column]
            label = f"{value:+.4f}\n± {interval:.4f}".replace("-", "−")
            axis.text(
                column,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=8.2,
                fontweight="bold",
                color="white" if abs(value) >= 0.45 * color_limit else "#25313F",
            )

    colorbar = fig.colorbar(
        heatmap,
        cax=colorbar_axis,
        orientation="vertical",
    )
    colorbar.ax.set_title(
        "配对性能差值\n里德堡采样 −\n随机加权贪心",
        fontsize=8.5,
        pad=9,
    )
    colorbar.outline.set_edgecolor("#30363B")
    colorbar.outline.set_linewidth(0.7)
    colorbar.ax.tick_params(length=3, width=0.7)
    colorbar.ax.yaxis.set_major_formatter(lambda value, _: f"{value:+.2f}")


def _plot_comparison(results: dict[str, Any], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    summary = results["summary"]
    paired = results["paired_evidence"]
    budgets = list(results["config"]["candidate_budgets"])
    advantage_k = int(results["config"]["advantage_budget"])
    methods = (
        "ideal_rydberg",
        "randomized_greedy",
        "one_swap_local_search",
        "simulated_annealing",
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.4), constrained_layout=True)

    _draw_candidate_budget_panel(
        axes[0, 0],
        summary,
        budgets,
        methods,
        metric="expected_best_ratio_mean",
        uncertainty="expected_best_ratio_ci95",
        title="A  中心干扰瓶颈测试集：平均调度质量",
        ylabel="平均调度质量（期望近似比）",
        ylim=(0.72, 1.015),
        show_legend=True,
    )
    _draw_candidate_budget_panel(
        axes[0, 1],
        summary,
        budgets,
        methods,
        metric="near_optimal_hit_mean",
        uncertainty="near_optimal_hit_ci95",
        title="B  中心干扰瓶颈测试集：近优解命中率",
        ylabel="近优解命中率\n（至少一个候选达到 99% 以上全局最优效用的概率）",
        ylim=(-0.02, 1.04),
        show_legend=False,
    )

    _draw_paired_difference_panel(
        axes[1, 0],
        paired,
        advantage_k,
        annotate_values=False,
    )

    stochastic_rows = {
        method: _find(
            summary,
            family="bottleneck",
            method=method,
            candidates_k=advantage_k,
        )
        for method in methods
    }
    deterministic = {
        method: _find(
            results["deterministic_summary"],
            family="bottleneck",
            method=method,
        )
        for method in ("beam_width_16", "exact_enumeration")
    }
    bar_methods = (
        "ideal_rydberg",
        "randomized_greedy",
        "simulated_annealing",
        "beam_width_16",
        "exact_enumeration",
    )
    means = [
        (
            stochastic_rows[method]["expected_best_ratio_mean"]
            if method in stochastic_rows
            else deterministic[method]["ratio_mean"]
        )
        for method in bar_methods
    ]
    errors = [
        (
            stochastic_rows[method]["expected_best_ratio_ci95"]
            if method in stochastic_rows
            else deterministic[method]["ratio_ci95"]
        )
        for method in bar_methods
    ]
    x = np.arange(len(bar_methods))
    axes[1, 1].bar(
        x,
        means,
        yerr=errors,
        capsize=2.5,
        color=[COLORS[method] for method in bar_methods],
        width=0.68,
    )
    axes[1, 1].set_xticks(
        x,
        labels=("里德堡", "随机\n贪心", "模拟\n退火", "束搜索", "精确解"),
    )
    axes[1, 1].set_ylim(0.74, 1.015)
    axes[1, 1].set_ylabel("平均调度质量（近似比）")
    axes[1, 1].set_title(
        f"D  与强经典对照的比较（$K={advantage_k}$）",
        loc="left",
        fontweight="bold",
    )
    axes[1, 1].text(
        0.02,
        0.04,
        "束搜索和精确解是确定性对照；\n"
        "其计算预算未与量子采样次数匹配。",
        transform=axes[1, 1].transAxes,
        fontsize=7.6,
        color="#4F565D",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.88,
        },
    )

    fig.suptitle(
        "公共 Wi-Fi 最大加权独立集：理想采样器的条件优势",
        fontsize=13,
        fontweight="bold",
    )
    _save(fig, output_dir, "02_classical_quantum_comparison")
    plt.close(fig)

    standalone_panels = (
        (
            "02A_best_of_k_quality",
            "expected_best_ratio_mean",
            "expected_best_ratio_ci95",
            "A  中心干扰瓶颈测试集：平均调度质量",
            "平均调度质量（期望近似比）",
            (0.72, 1.015),
        ),
        (
            "02B_near_optimal_hit",
            "near_optimal_hit_mean",
            "near_optimal_hit_ci95",
            "B  中心干扰瓶颈测试集：近优解命中率",
            "近优解命中率\n（至少一个候选达到 99% 以上全局最优效用的概率）",
            (-0.02, 1.04),
        ),
    )
    for stem, metric, uncertainty, title, ylabel, ylim in standalone_panels:
        panel_fig, panel_axis = plt.subplots(
            figsize=(6.8, 5.2), constrained_layout=True
        )
        _draw_candidate_budget_panel(
            panel_axis,
            summary,
            budgets,
            methods,
            metric=metric,
            uncertainty=uncertainty,
            title=title,
            ylabel=ylabel,
            ylim=ylim,
            show_legend=True,
        )
        _save(panel_fig, output_dir, stem, tight=False)
        plt.close(panel_fig)

    difference_fig, (difference_axis, difference_colorbar_axis) = plt.subplots(
        ncols=2,
        figsize=(8.8, 5.2),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (24, 1), "wspace": 0.06},
    )
    _draw_paired_difference_heatmap(
        difference_fig,
        difference_axis,
        difference_colorbar_axis,
        paired,
        budgets,
    )
    _save(
        difference_fig,
        output_dir,
        "02C_paired_performance_heatmap",
        tight=False,
    )
    plt.close(difference_fig)


def plot_wifi_mis_figures(results: dict[str, Any], output_dir: Path) -> None:
    """Write deterministic PNG and SVG figures from one result payload."""

    _style()
    _plot_test_scenarios_overview(output_dir)
    _plot_mapping(results, output_dir)
    _plot_comparison(results, output_dir)


__all__ = ["plot_wifi_mis_figures"]
