"""Shallow-QAOA versus direct Rydberg mapping on the Wi-Fi MWIS benchmark.

Both methods are evaluated as ideal classical statevector simulations.  The
direct method reuses the frozen QuTiP Rydberg results from ``wifi_mis_results``;
the QAOA baseline uses a standard penalty-QUBO cost Hamiltonian and transverse-X
mixer.  Raw measurements from both methods pass through the same authoritative
MWIS repair rule before candidate quality is evaluated.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import pi
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np

from ..core import Action, ConflictGraph
from ..utilities.results import ResultWriter
from .baselines import repair_action
from .wifi_mis import (
    FAMILIES,
    WifiSlotInstance,
    action_value,
    exact_mwis,
    make_wifi_instance,
)


METHODS = ("direct_rydberg", "qaoa_p1", "qaoa_p2")
FAMILY_LABELS = {
    "bottleneck": "中心干扰瓶颈",
    "random": "随机热点",
    "crowded": "高密拥挤热点",
    "corridor": "走廊链式干扰",
}


@dataclass(frozen=True)
class QAOAParameters:
    """Frozen shallow-QAOA angles shared by all held-out instances."""

    depth: int
    gammas: tuple[float, ...]
    betas: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.depth not in (1, 2):
            raise ValueError("this comparison only supports QAOA depth p=1 or p=2")
        if len(self.gammas) != self.depth or len(self.betas) != self.depth:
            raise ValueError("one gamma and beta are required per QAOA layer")


@dataclass(frozen=True)
class PreparedQAOAProblem:
    """Statevector-ready QUBO data plus deterministic repair lookup tables."""

    instance: WifiSlotInstance
    optimum: float
    penalty: float
    score_scale: float
    normalized_scores: np.ndarray
    repaired_bits: np.ndarray
    repaired_ratios: np.ndarray
    unique_ratios: np.ndarray
    ratio_inverse: np.ndarray
    raw_feasible: np.ndarray


def _basis_bits(nodes: int) -> np.ndarray:
    basis = np.arange(1 << nodes, dtype=np.uint32)
    shifts = np.arange(nodes - 1, -1, -1, dtype=np.uint32)
    return ((basis[:, None] >> shifts[None, :]) & 1).astype(np.uint8)


def _prepare_problem(instance: WifiSlotInstance) -> PreparedQAOAProblem:
    bits = _basis_bits(instance.graph.nodes)
    penalty = 1.25 * float(np.max(instance.utilities))
    scores = bits @ instance.utilities
    for left, right in instance.graph.edges:
        scores -= penalty * bits[:, left] * bits[:, right]
    scale = max(float(np.max(np.abs(scores))), 1e-12)
    normalized_scores = np.asarray(scores / scale, dtype=float)

    optimum, _, _ = exact_mwis(instance)
    repaired_bits = np.empty_like(bits)
    raw_feasible = np.zeros(len(bits), dtype=bool)
    for basis, row in enumerate(bits):
        raw: Action = tuple(int(bit) for bit in row)
        raw_feasible[basis] = instance.graph.is_feasible(raw)
        repaired_bits[basis] = np.asarray(
            repair_action(raw, instance.graph, instance.utilities), dtype=np.uint8
        )
    repaired_ratios = (repaired_bits @ instance.utilities) / max(optimum, 1e-12)
    unique_ratios, ratio_inverse = np.unique(
        np.round(repaired_ratios, 12), return_inverse=True
    )
    return PreparedQAOAProblem(
        instance=instance,
        optimum=optimum,
        penalty=penalty,
        score_scale=scale,
        normalized_scores=normalized_scores,
        repaired_bits=repaired_bits,
        repaired_ratios=np.asarray(repaired_ratios, dtype=float),
        unique_ratios=np.asarray(unique_ratios, dtype=float),
        ratio_inverse=np.asarray(ratio_inverse, dtype=np.int32),
        raw_feasible=raw_feasible,
    )


def prepare_qaoa_problem(instance: WifiSlotInstance) -> PreparedQAOAProblem:
    """Compile one Wi-Fi MWIS instance into the shallow-QAOA SDK format.

    The returned object contains the normalized penalty-QUBO spectrum together
    with the deterministic post-measurement repair lookup used by the benchmark.
    Keeping this preparation step public lets tutorials and downstream SDK users
    reproduce the same encoding without importing a private helper.
    """

    return _prepare_problem(instance)


def pennylane_qaoa_qnode(
    problem: PreparedQAOAProblem,
    parameters: QAOAParameters,
    *,
    shots: int | None = None,
    device_name: str = "default.qubit",
):
    """Build an executable PennyLane QNode for the SDK's shallow QAOA model.

    The cost layer is decomposed into ``RZ`` and ``IsingZZ`` gates and the
    transverse-X mixer into ``RX`` gates.  With ``shots=None`` the QNode returns
    exact basis probabilities in the same node order as :func:`qaoa_probabilities`;
    with a positive shot count it returns computational-basis samples.

    PennyLane is an optional tutorial dependency and is imported lazily.
    """

    try:
        import pennylane as qml
    except (ImportError, OSError) as error:
        raise ImportError(
            "pennylane_qaoa_qnode requires PennyLane; install the tutorial "
            "environment with the PennyLane dependency."
        ) from error

    if shots is not None and shots <= 0:
        raise ValueError("shots must be positive or None")

    nodes = problem.instance.graph.nodes
    device = qml.device(device_name, wires=nodes)
    linear = np.asarray(problem.instance.utilities, dtype=float) / problem.score_scale
    quadratic = -problem.penalty / problem.score_scale
    z_coefficients = -0.5 * linear
    for left, right in problem.instance.graph.edges:
        z_coefficients[left] -= 0.25 * quadratic
        z_coefficients[right] -= 0.25 * quadratic

    @qml.qnode(device)
    def circuit():
        for wire in range(nodes):
            qml.Hadamard(wires=wire)
        for gamma, beta in zip(parameters.gammas, parameters.betas):
            for wire, coefficient in enumerate(z_coefficients):
                qml.RZ(2.0 * gamma * float(coefficient), wires=wire)
            for left, right in problem.instance.graph.edges:
                qml.IsingZZ(
                    gamma * quadratic / 2.0,
                    wires=(left, right),
                )
            for wire in range(nodes):
                qml.RX(2.0 * beta, wires=wire)
        if shots is None:
            return qml.probs(wires=range(nodes))
        return qml.sample(wires=range(nodes))

    if shots is not None:
        return qml.set_shots(circuit, shots=shots)
    return circuit


def _apply_x_mixer(state: np.ndarray, beta: float, nodes: int) -> None:
    cosine = float(np.cos(beta))
    sine = -1j * float(np.sin(beta))
    for qubit in range(nodes):
        stride = 1 << (nodes - qubit - 1)
        block = 2 * stride
        view = state.reshape(-1, block)
        zero = view[:, :stride].copy()
        one = view[:, stride:].copy()
        view[:, :stride] = cosine * zero + sine * one
        view[:, stride:] = sine * zero + cosine * one


def qaoa_probabilities(
    problem: PreparedQAOAProblem,
    parameters: QAOAParameters,
) -> np.ndarray:
    """Return the ideal computational-basis distribution for shallow QAOA."""

    dimension = len(problem.normalized_scores)
    state = np.full(dimension, 1.0 / np.sqrt(dimension), dtype=np.complex128)
    for gamma, beta in zip(parameters.gammas, parameters.betas):
        state *= np.exp(-1j * gamma * problem.normalized_scores)
        _apply_x_mixer(state, beta, problem.instance.graph.nodes)
    probabilities = np.abs(state) ** 2
    probabilities /= float(np.sum(probabilities))
    return probabilities


def _best_of_k(
    problem: PreparedQAOAProblem,
    probabilities: np.ndarray,
    budget: int,
) -> float:
    grouped = np.bincount(
        problem.ratio_inverse,
        weights=probabilities,
        minlength=len(problem.unique_ratios),
    )
    cumulative = np.cumsum(grouped)
    increments = cumulative**budget - np.concatenate(
        ([0.0], cumulative[:-1] ** budget)
    )
    return float(np.dot(problem.unique_ratios, increments))


def qaoa_metrics(
    problem: PreparedQAOAProblem,
    probabilities: np.ndarray,
    budgets: Iterable[int],
    epsilon: float,
) -> dict[str, Any]:
    """Evaluate repaired candidate quality using the Wi-Fi benchmark metrics."""

    near_probability = float(
        np.sum(probabilities[problem.repaired_ratios >= 1.0 - epsilon])
    )
    budget_rows = [
        {
            "candidates_k": int(budget),
            "expected_best_ratio": _best_of_k(problem, probabilities, int(budget)),
            "near_optimal_hit_probability": float(
                1.0 - (1.0 - near_probability) ** int(budget)
            ),
        }
        for budget in budgets
    ]
    marginals = probabilities @ problem.repaired_bits
    repaired_ids = problem.repaired_bits @ (
        1 << np.arange(problem.instance.graph.nodes - 1, -1, -1)
    )
    support = int(len(np.unique(repaired_ids[probabilities > 1e-14])))
    return {
        "expected_one_shot_ratio": float(
            np.dot(probabilities, problem.repaired_ratios)
        ),
        "single_shot_near_optimal_probability": near_probability,
        "expected_hamming_diversity": float(
            np.mean(2.0 * marginals * (1.0 - marginals))
        ),
        "support_size": support,
        "budgets": budget_rows,
        "raw_feasible_probability": float(
            np.sum(probabilities[problem.raw_feasible])
        ),
    }


def _training_score(
    problems: list[PreparedQAOAProblem],
    parameters: QAOAParameters,
    budget: int,
) -> float:
    return float(
        np.mean(
            [
                _best_of_k(problem, qaoa_probabilities(problem, parameters), budget)
                for problem in problems
            ]
        )
    )


def _select_parameters(
    problems: list[PreparedQAOAProblem],
    budget: int,
    seed: int,
) -> tuple[QAOAParameters, QAOAParameters, list[dict[str, Any]]]:
    """Select global p=1 and p=2 angles on bottleneck training instances only."""

    from scipy.optimize import differential_evolution

    search_rows: list[dict[str, Any]] = []

    def optimize_depth(
        depth: int,
        *,
        optimizer_seed: int,
        maxiter: int,
    ) -> tuple[QAOAParameters, float, int, float]:
        def objective(values: np.ndarray) -> float:
            parameters = QAOAParameters(
                depth,
                tuple(float(value) for value in values[:depth]),
                tuple(float(value) for value in values[depth:]),
            )
            return -_training_score(problems, parameters, budget)

        bounds = [(-20.0, 20.0)] * depth + [(0.0, pi)] * depth
        started = perf_counter()
        result = differential_evolution(
            objective,
            bounds,
            seed=optimizer_seed,
            popsize=8,
            maxiter=maxiter,
            polish=True,
            tol=1e-5,
            workers=1,
            updating="immediate",
        )
        parameters = QAOAParameters(
            depth,
            tuple(float(value) for value in result.x[:depth]),
            tuple(float(value) for value in result.x[depth:]),
        )
        return (
            parameters,
            float(-result.fun),
            int(result.nfev),
            float(perf_counter() - started),
        )

    best_p1, best_p1_score, p1_evaluations, p1_elapsed = optimize_depth(
        1, optimizer_seed=seed, maxiter=20
    )
    search_rows.append(
        {
            "depth": 1,
            "optimizer": "differential_evolution",
            "angle_bounds": {"gamma": [-20.0, 20.0], "beta": [0.0, pi]},
            "candidates_evaluated": p1_evaluations,
            "selection_objective_mean": float(best_p1_score),
            "elapsed_seconds": p1_elapsed,
            "chosen": asdict(best_p1),
        }
    )

    best_p2, best_p2_score, p2_evaluations, p2_elapsed = optimize_depth(
        2, optimizer_seed=seed + 77_777, maxiter=24
    )
    embedded_p1 = QAOAParameters(
        2,
        (best_p1.gammas[0], 0.0),
        (best_p1.betas[0], 0.0),
    )
    embedded_score = _training_score(problems, embedded_p1, budget)
    if embedded_score > best_p2_score:
        best_p2 = embedded_p1
        best_p2_score = embedded_score
    start = perf_counter()
    search_rows.append(
        {
            "depth": 2,
            "optimizer": "differential_evolution",
            "angle_bounds": {"gamma": [-20.0, 20.0], "beta": [0.0, pi]},
            "candidates_evaluated": p2_evaluations + 1,
            "selection_objective_mean": float(best_p2_score),
            "elapsed_seconds": p2_elapsed + float(perf_counter() - start),
            "included_p1_embedding": True,
            "chosen": asdict(best_p2),
        }
    )
    return best_p1, best_p2, search_rows


def _instance_from_record(record: dict[str, Any]) -> WifiSlotInstance:
    return WifiSlotInstance(
        family=record["family"],
        seed=int(record["seed"]),
        positions=np.asarray(record["positions"], dtype=float),
        interference_radius=float(record["interference_radius"]),
        graph=ConflictGraph(
            nodes=int(record["nodes"]),
            edges=tuple(tuple(int(node) for node in edge) for edge in record["graph_edges"]),
        ),
        queue_packets=np.asarray(record["queue_packets"], dtype=float),
        latency_slack_ms=np.asarray(record["latency_slack_ms"], dtype=float),
        service_priority=np.asarray(record["service_priority"], dtype=float),
        utilities=np.asarray(record["utilities"], dtype=float),
    )


def _mean_ci(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    mean = float(np.mean(array)) if len(array) else 0.0
    if len(array) < 2:
        return mean, 0.0
    return mean, float(1.96 * np.std(array, ddof=1) / np.sqrt(len(array)))


def _budget_row(method_record: dict[str, Any], budget: int) -> dict[str, Any]:
    return next(
        row for row in method_record["budgets"] if row["candidates_k"] == budget
    )


def _summaries(
    records: list[dict[str, Any]], budgets: tuple[int, ...]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    for family in FAMILIES:
        family_records = [record for record in records if record["family"] == family]
        for method in METHODS:
            for budget in budgets:
                rows = [
                    _budget_row(record["methods"][method], budget)
                    for record in family_records
                ]
                ratio_mean, ratio_ci = _mean_ci(
                    row["expected_best_ratio"] for row in rows
                )
                hit_mean, hit_ci = _mean_ci(
                    row["near_optimal_hit_probability"] for row in rows
                )
                feasible_mean, feasible_ci = _mean_ci(
                    record["methods"][method]["raw_feasible_probability"]
                    for record in family_records
                )
                diversity_mean, diversity_ci = _mean_ci(
                    record["methods"][method]["expected_hamming_diversity"]
                    for record in family_records
                )
                summary.append(
                    {
                        "family": family,
                        "method": method,
                        "candidates_k": budget,
                        "expected_best_ratio_mean": ratio_mean,
                        "expected_best_ratio_ci95": ratio_ci,
                        "near_optimal_hit_mean": hit_mean,
                        "near_optimal_hit_ci95": hit_ci,
                        "raw_feasible_mean": feasible_mean,
                        "raw_feasible_ci95": feasible_ci,
                        "expected_hamming_diversity_mean": diversity_mean,
                        "expected_hamming_diversity_ci95": diversity_ci,
                    }
                )

        for method in ("qaoa_p1", "qaoa_p2"):
            for budget in budgets:
                differences = [
                    _budget_row(record["methods"]["direct_rydberg"], budget)[
                        "expected_best_ratio"
                    ]
                    - _budget_row(record["methods"][method], budget)[
                        "expected_best_ratio"
                    ]
                    for record in family_records
                ]
                mean, ci = _mean_ci(differences)
                paired.append(
                    {
                        "family": family,
                        "qaoa_method": method,
                        "candidates_k": budget,
                        "direct_rydberg_minus_qaoa_mean": mean,
                        "direct_rydberg_minus_qaoa_ci95": ci,
                        "direct_rydberg_win_rate": float(
                            np.mean(np.asarray(differences) > 1e-12)
                        ),
                    }
                )
    return summary, paired


def _find(rows: list[dict[str, Any]], **matching: Any) -> dict[str, Any]:
    return next(
        row
        for row in rows
        if all(row.get(key) == value for key, value in matching.items())
    )


def _render_report(results: dict[str, Any]) -> str:
    budget = int(results["config"]["advantage_budget"])
    lines = [
        "# 浅层 QAOA 与直接里德堡映射的 Wi-Fi/MWIS 对比",
        "",
        "## 比较口径",
        "",
        "两种方法均为理想态矢量经典模拟，不是量子硬件实测。直接里德堡映射复用"
        "现有 QuTiP 中性原子哈密顿量结果；QAOA 使用惩罚 QUBO 成本哈密顿量、"
        "横向 X mixer 和全局冻结参数。所有原始测量均经过同一 MWIS 修复层。",
        "",
        f"QAOA 参数只在 {results['training']['instances']} 个中心瓶颈训练实例上选择，"
        f"随后在 {results['config']['test_records']} 个留出实例上固定评估。主表使用 K={budget}。",
        "",
        "## 主要结果",
        "",
        "| 测试场景 | 直接里德堡映射 | QAOA p=1 | QAOA p=2 | 里德堡−p1 | 里德堡−p2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for family in FAMILIES:
        direct = _find(
            results["summary"],
            family=family,
            method="direct_rydberg",
            candidates_k=budget,
        )
        p1 = _find(
            results["summary"], family=family, method="qaoa_p1", candidates_k=budget
        )
        p2 = _find(
            results["summary"], family=family, method="qaoa_p2", candidates_k=budget
        )
        d1 = _find(
            results["paired_evidence"],
            family=family,
            qaoa_method="qaoa_p1",
            candidates_k=budget,
        )
        d2 = _find(
            results["paired_evidence"],
            family=family,
            qaoa_method="qaoa_p2",
            candidates_k=budget,
        )

        def pm(row: dict[str, Any], mean: str, ci: str) -> str:
            return f"{row[mean]:.4f} ± {row[ci]:.4f}"

        lines.append(
            f"| {FAMILY_LABELS[family]} | "
            f"{pm(direct, 'expected_best_ratio_mean', 'expected_best_ratio_ci95')} | "
            f"{pm(p1, 'expected_best_ratio_mean', 'expected_best_ratio_ci95')} | "
            f"{pm(p2, 'expected_best_ratio_mean', 'expected_best_ratio_ci95')} | "
            f"{d1['direct_rydberg_minus_qaoa_mean']:+.4f} ± {d1['direct_rydberg_minus_qaoa_ci95']:.4f} | "
            f"{d2['direct_rydberg_minus_qaoa_mean']:+.4f} ± {d2['direct_rydberg_minus_qaoa_ci95']:.4f} |"
        )
    lines.extend(
        [
            "",
            "![浅层 QAOA 与直接里德堡映射对比](../figures/wifi_mis/03_qaoa_vs_direct_rydberg.png)",
            "",
            "## 参数与限制",
            "",
            f"- QAOA p=1 参数：`{results['training']['qaoa_p1']['chosen']}`；",
            f"- QAOA p=2 参数：`{results['training']['qaoa_p2']['chosen']}`；",
            "- QAOA 的 QUBO 分数按实例归一化，冲突惩罚为最大节点权重的 1.25 倍；",
            "- QAOA 参数搜索预算很小，不能代表经过大规模变分优化后的最佳 QAOA；",
            "- QuTiP 与 NumPy 运行时间都是经典模拟器耗时，不能用于推断真实硬件延迟；",
            "- 正的‘里德堡−QAOA’只表示在当前实例、参数训练协议和相同 K 下，"
            "直接里德堡映射的候选质量更高。",
            "",
        ]
    )
    return "\n".join(lines)


def _plot(results: dict[str, Any], output_dir: Path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "savefig.dpi": 300,
        }
    )
    budget = int(results["config"]["advantage_budget"])
    methods = METHODS
    labels = ("直接里德堡映射", "浅层 QAOA p=1", "浅层 QAOA p=2")
    colors = ("#C23B3B", "#3677A8", "#6A8E3A")
    x = np.arange(len(FAMILIES), dtype=float)
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)

    for index, (method, label, color) in enumerate(zip(methods, labels, colors)):
        rows = [
            _find(
                results["summary"],
                family=family,
                method=method,
                candidates_k=budget,
            )
            for family in FAMILIES
        ]
        offset = (index - 1) * width
        axes[0].bar(
            x + offset,
            [row["expected_best_ratio_mean"] for row in rows],
            yerr=[row["expected_best_ratio_ci95"] for row in rows],
            width=width,
            color=color,
            capsize=2.5,
            label=label,
        )
        axes[1].bar(
            x + offset,
            [row["raw_feasible_mean"] for row in rows],
            yerr=[row["raw_feasible_ci95"] for row in rows],
            width=width,
            color=color,
            capsize=2.5,
            label=label,
        )

    family_ticks = [FAMILY_LABELS[family].replace("干扰", "干扰\n") for family in FAMILIES]
    for axis in axes:
        axis.set_xticks(x, family_ticks)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.22, linewidth=0.7)
        axis.set_axisbelow(True)
    axes[0].set_ylim(0.60, 1.01)
    axes[0].set_ylabel("平均调度质量（期望近似比）")
    axes[0].set_title(f"A  候选质量（K={budget}）", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_ylim(0.0, 1.04)
    axes[1].set_ylabel("修复前原始可行率")
    axes[1].set_title("B  原始测量可行性", loc="left", fontweight="bold")
    fig.suptitle(
        "浅层 QAOA 与直接里德堡映射的 Wi-Fi/MWIS 对比（理想模拟）",
        fontsize=13,
        fontweight="bold",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(
            output_dir / f"03_qaoa_vs_direct_rydberg.{suffix}",
            facecolor="white",
        )
    plt.close(fig)


def run_qaoa_vs_rydberg(
    baseline_json: Path,
    output_json: Path,
    output_report: Path,
    figure_dir: Path,
) -> dict[str, Any]:
    """Train shallow QAOA on the frozen split and compare on held-out frames."""

    baseline = json.loads(baseline_json.read_text(encoding="utf-8"))
    config = baseline["config"]
    budgets = tuple(int(value) for value in config["candidate_budgets"])
    advantage_budget = int(config["advantage_budget"])
    seed = int(config["seed"])

    training_instances = [
        make_wifi_instance("bottleneck", seed + index, int(config["nodes"]))
        for index in range(int(config["pulse_training_seeds"]))
    ]
    training_problems = [_prepare_problem(instance) for instance in training_instances]
    qaoa_p1, qaoa_p2, search_rows = _select_parameters(
        training_problems, advantage_budget, seed
    )

    records: list[dict[str, Any]] = []
    for baseline_record in baseline["records"]:
        instance = _instance_from_record(baseline_record)
        problem = _prepare_problem(instance)
        methods: dict[str, Any] = {
            "direct_rydberg": baseline_record["methods"]["ideal_rydberg"]
            | {"evidence_type": "ideal_direct_rydberg_statevector"}
        }
        for method, parameters in (("qaoa_p1", qaoa_p1), ("qaoa_p2", qaoa_p2)):
            start = perf_counter()
            probabilities = qaoa_probabilities(problem, parameters)
            elapsed_ms = (perf_counter() - start) * 1_000.0
            methods[method] = qaoa_metrics(
                problem,
                probabilities,
                budgets,
                float(config["near_optimal_epsilon"]),
            ) | {
                "classical_emulator_ms": float(elapsed_ms),
                "evidence_type": "ideal_gate_statevector",
                "parameters": asdict(parameters),
            }
        records.append(
            {
                "family": baseline_record["family"],
                "seed": baseline_record["seed"],
                "nodes": baseline_record["nodes"],
                "edges": baseline_record["edges"],
                "edge_density": baseline_record["edge_density"],
                "methods": methods,
            }
        )

    summary, paired = _summaries(records, budgets)
    results: dict[str, Any] = {
        "study": "wifi_mis_shallow_qaoa_vs_direct_rydberg",
        "claim_boundary": (
            "Ideal statevector comparison only. No physical QPU, wall-clock, energy, "
            "or end-to-end hardware advantage is claimed."
        ),
        "config": {
            "nodes": int(config["nodes"]),
            "training_instances": len(training_instances),
            "test_records": len(records),
            "candidate_budgets": list(budgets),
            "advantage_budget": advantage_budget,
            "near_optimal_epsilon": float(config["near_optimal_epsilon"]),
            "qaoa_penalty_multiplier": 1.25,
            "qaoa_cost_normalization": "per-instance max absolute QUBO score",
            "seed": seed,
        },
        "training": {
            "split": "same bottleneck training seeds used for Rydberg pulse selection",
            "instances": len(training_instances),
            "selection_metric": f"mean expected best-of-{advantage_budget} approximation ratio",
            "qaoa_p1": search_rows[0],
            "qaoa_p2": search_rows[1],
        },
        "records": records,
        "summary": summary,
        "paired_evidence": paired,
        "physical_qpu_evidence": False,
    }
    ResultWriter().artifacts(
        json_path=output_json,
        report_path=output_report,
        payload=results,
        render_report=_render_report,
    )
    _plot(results, figure_dir)
    return results


__all__ = [
    "METHODS",
    "PreparedQAOAProblem",
    "QAOAParameters",
    "pennylane_qaoa_qnode",
    "prepare_qaoa_problem",
    "qaoa_metrics",
    "qaoa_probabilities",
    "run_qaoa_vs_rydberg",
]
