"""Chinese academic report for the public Wi-Fi MWIS study."""

from __future__ import annotations

from typing import Any


METHOD_LABELS = {
    "ideal_rydberg": "理想 Rydberg sampler",
    "randomized_greedy": "随机加权贪心",
    "one_swap_local_search": "单交换局部搜索",
    "simulated_annealing": "模拟退火",
    "priority_greedy": "确定性优先级贪心",
    "beam_width_16": "Beam search (width=16)",
    "exact_enumeration": "精确枚举",
}

FAMILY_LABELS = {
    "bottleneck": "中心干扰瓶颈",
    "random": "随机热点",
    "crowded": "高密拥挤热点",
    "corridor": "走廊链式干扰",
}

FAMILY_ORDER = ("bottleneck", "random", "crowded", "corridor")


def _find(rows: list[dict[str, Any]], **matching: Any) -> dict[str, Any]:
    return next(
        row
        for row in rows
        if all(row.get(key) == value for key, value in matching.items())
    )


def _pm(mean: float, ci: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f} ± {ci:.{digits}f}"


def _percentage(value: float, digits: int = 1) -> str:
    return f"{100.0 * value:.{digits}f}%"


def render_wifi_mis_report(results: dict[str, Any]) -> str:
    """Render a source-grounded, claim-bounded academic report in Chinese."""

    config = results["config"]
    gates = results["gates"]
    budget = int(config["advantage_budget"])
    epsilon = float(config["near_optimal_epsilon"])
    status = (
        "通过（仅限理想 sampler 的条件优势）"
        if gates["limited_ideal_sampler_advantage_pass"]
        else "未通过"
    )
    summary = results["summary"]
    paired = results["paired_evidence"]
    chosen = results["pulse_selection"]

    quantum = _find(
        summary,
        family="bottleneck",
        method="ideal_rydberg",
        candidates_k=budget,
    )
    greedy = _find(
        summary,
        family="bottleneck",
        method="randomized_greedy",
        candidates_k=budget,
    )
    local = _find(
        summary,
        family="bottleneck",
        method="one_swap_local_search",
        candidates_k=budget,
    )
    annealing = _find(
        summary,
        family="bottleneck",
        method="simulated_annealing",
        candidates_k=budget,
    )
    vs_greedy = gates["quantum_minus_randomized_greedy"]
    vs_local = gates["quantum_minus_one_swap_local"]
    vs_annealing = gates["quantum_minus_simulated_annealing"]
    beam = _find(
        results["deterministic_summary"],
        family="bottleneck",
        method="beam_width_16",
    )

    lines = [
        "# 面向公共 Wi‑Fi 时隙调度的中性原子 MWIS 候选采样",
        "",
        "## 摘要",
        "",
        (
            "本文将原有的约束动作 sampler 具体化为公众可感知的公共 Wi‑Fi "
            "时隙调度：每个待发送设备对应一个顶点，不能同槽发送的设备之间连边，"
            "调度动作是一个加权独立集。我们在二维 unit-disk 干扰图上比较理想 "
            "Rydberg 演化、随机加权贪心、单交换局部搜索、模拟退火、beam search "
            "与精确解。脉冲只在独立训练 seed 上选择一次，随后冻结并在四类未见"
            f"热点、每类 {config['test_seeds']} 个实例上测试。"
        ),
        "",
        (
            f"预先定义的条件优势门在 K={budget} 时的状态为 **{status}**。"
            f"在中心干扰瓶颈实例上，理想 Rydberg best-of-{budget} 近似比为 "
            f"{_pm(quantum['expected_best_ratio_mean'], quantum['expected_best_ratio_ci95'])}，"
            f"随机贪心为 {_pm(greedy['expected_best_ratio_mean'], greedy['expected_best_ratio_ci95'])}；"
            f"配对差值为 {_pm(vs_greedy['quantum_minus_classical_mean'], vs_greedy['quantum_minus_classical_ci95'])}。"
        ),
        "",
        (
            "这是一项**理想量子分布的候选质量优势**，不是硬件时间优势、能耗优势或"
            "渐近复杂度优势。QuTiP 在经典计算机上模拟演化；beam 与精确解是强经典"
            "控制，物理 QPU 与端到端网络延迟均未测量。"
        ),
        "",
        "![A：公共 Wi-Fi 干扰图](../figures/wifi_mis/01A_wifi_interference_graph.png)",
        "",
        "![B：中性原子里德堡编码](../figures/wifi_mis/01B_rydberg_encoding.png)",
        "",
        "## 1. 这个问题为什么适合 quantum",
        "",
        "### 1.1 从用户体验到 MWIS",
        "",
        (
            "在咖啡馆、机场、校园或家庭 Wi‑Fi 中，同一信道上的相邻设备若同时发包会"
            "互相干扰。令二元变量 $x_i=1$ 表示本时隙调度设备 $i$，边 $(i,j)$ 表示"
            "二者不能并发；队列长度、时延紧迫度和业务等级合成为权重 $w_i$，则每槽"
            "决策为"
        ),
        "",
        "$$\\max_{x\\in\\{0,1\\}^n} \\sum_i w_i x_i, \\qquad x_i+x_j\\le 1,\\;\\forall(i,j)\\in E.$$",
        "",
        (
            "无线调度以干扰图上的 MWIS 表述并非人为包装；已有工作直接以队列 backlog "
            "作为权重，并指出逐时隙调度需要求解 MWIS [1,2]。本研究使用 unit-disk "
            "干扰模型，它对应距离阈值近似；实际多径、异构功率和隐藏终端会偏离该模型，"
            "因此必须由网络侧安全检查保底。"
        ),
        "",
        "### 1.2 中性原子的原生约束",
        "",
        (
            "在 Rydberg 阵列中，一个设备决策映射为一个原子：$x_i=1$ 对应原子处于 "
            "$|r\\rangle$。距离小于 blockade 半径的两个原子不能同时激发，恰好实现 "
            "$x_i+x_j\\le1$。实验上已用最多 289 个原子研究 unit-disk MIS，并以测量"
            "后经典修复处理少量 blockade 违例 [3]；2024 年公开数据集还给出了最高 "
            "141 原子的实验测量 [4]。因此，本问题相较一般 QUBO 少了一层惩罚项编译。"
        ),
        "",
        (
            "量子 sampler 的潜在价值不必是“一次给出最优解”，而是用一次全局相干演化"
            "生成偏向高质量、彼此多样的可行候选，再由经典安全层按已知效用做 best-of-K。"
            "中心设备会阻塞多个外围设备的场景对局部贪心尤其不利：量子演化可以在多个"
            "兼容组合之间分配振幅，而非承诺普遍超越成熟经典求解器。"
        ),
        "",
        "## 2. 我们 model 的描述",
        "",
        "### 2.1 应用状态与效用",
        "",
        (
            "每个合成热点帧包含 12 个待传输设备及二维位置。顶点权重由三项归一化"
            "信号组成：队列长度 45%、剩余时延紧迫度 35%、业务优先级 20%，再缩放到"
            "正值区间。四个测试族分别是：随机热点、高密拥挤热点、走廊两侧设备形成的"
            "链式局部干扰，以及两个“中心设备阻塞五个互相兼容外围设备”的干扰瓶颈。"
            "后者是明确的 opportunity regime，不是从测试结果中事后筛选。"
        ),
        "",
        "### 2.2 混合量子—经典路径",
        "",
        "模型使用的理想化哈密顿量为",
        "",
        "$$H(t)=\\frac{\\Omega(t)}{2}\\sum_i X_i-\\Delta(t)\\sum_i w_i n_i+U\\sum_{(i,j)\\in E}n_i n_j,$$",
        "",
        (
            "其中 $n_i=(I-Z_i)/2$。负到正 detuning 扫描把初始真空态推向高权重独立集；"
            "有限 $U$ 的残余冲突在测量后按效用确定性删除较弱端点。完整决策链为："
            "热点状态 → 权重编码 → Rydberg/经典候选 → 权威干扰图修复 → "
            "已知效用 best-of-K。"
        ),
        "",
        (
            f"候选脉冲在 {config['pulse_training_seeds']} 个训练瓶颈实例上，从 short、"
            f"balanced、adiabatic 三个预定义 regime 中按 K={budget} 的平均 best-of-K "
            f"近似比选择；在距训练最优 0.002 内以更快的 emulator 作为固定 tie-break。"
            f"最终冻结为 '{chosen['chosen_label']}'，测试 seed 与脉冲选择"
            " seed 完全分离。"
        ),
        "",
        "### 2.3 评价协议",
        "",
        (
            f"主指标是 best-of-K 期望近似比 $E[W(\\hat S_K)/W(S^*)]$，辅指标为"
            f"命中 $\\ge {1.0 - epsilon:.0%}$-optimal 解的概率、配对实例差值、原始可行率与候选"
            "多样性。理想量子分布由 QuTiP 全分布精确汇总；经典随机方法各用 "
            f"{config['classical_probability_samples']} 个离线样本估计单候选分布，再解析"
            "计算 best-of-K。reranker 只计算候选的已知线性效用，不知道全局最优解；"
            "精确枚举只提供 12 节点归一化分母，不进入 sampler 或候选选择。"
        ),
        "",
        "## 3. Classical vs quantum 对比",
        "",
        "![Classical 与 ideal Rydberg 的 held-out 对比](../figures/wifi_mis/02_classical_quantum_comparison.png)",
        "",
        f"### 3.1 主结果：中心干扰瓶颈，K={budget}",
        "",
        "| 方法 | best-of-K 近似比 | 近优解命中率 |",
        "|---|---:|---:|",
    ]
    for method, row in (
        ("ideal_rydberg", quantum),
        ("randomized_greedy", greedy),
        ("one_swap_local_search", local),
        ("simulated_annealing", annealing),
    ):
        lines.append(
            f"| {METHOD_LABELS[method]} | "
            f"{_pm(row['expected_best_ratio_mean'], row['expected_best_ratio_ci95'])} | "
            f"{_percentage(row['near_optimal_hit_mean'])} ± "
            f"{_percentage(row['near_optimal_hit_ci95'])} |"
        )
    lines.extend(
        [
            "",
            "配对优势的定义是同一测试实例上的 ideal Rydberg − classical：",
            "",
            "| 对照 | 平均差值 ± 95% CI | Rydberg 配对胜率 |",
            "|---|---:|---:|",
        ]
    )
    for label, row in (
        ("随机加权贪心", vs_greedy),
        ("单交换局部搜索", vs_local),
        ("模拟退火", vs_annealing),
    ):
        lines.append(
            f"| {label} | "
            f"{_pm(row['quantum_minus_classical_mean'], row['quantum_minus_classical_ci95'])} | "
            f"{_percentage(row['paired_win_rate'])} |"
        )

    lines.extend(
        [
            "",
            (
                "对模拟退火，Rydberg 的期望近似比差值虽然很小且 95% CI 为正，"
                f"但其近优命中率为 {_percentage(quantum['near_optimal_hit_mean'])}，"
                f"低于模拟退火的 {_percentage(annealing['near_optimal_hit_mean'])}。"
                "因此本文不把这一单指标差异解释为对强经典方法的整体优势。"
            ),
            "",
            (
                f"原始量子测量的平均可行率为 {_pm(gates['raw_feasible_mean'], gates['raw_feasible_ci95'])}；"
                "执行前修复后的可行率按构造为 1。这里的局部优势来自避免“高权重中心"
                "设备”这一局部诱饵；单交换搜索也难以一次删除中心并加入多个外围设备。"
            ),
            "",
            "### 3.2 跨拓扑外推边界",
            "",
            f"下表固定 K={budget}，避免只报告有利实例族：",
            "",
            "| 测试族 | 里德堡采样 | 随机贪心 | 差值（里德堡采样−贪心） |",
            "|---|---:|---:|---:|",
        ]
    )
    for family in FAMILY_ORDER:
        q = _find(
            summary,
            family=family,
            method="ideal_rydberg",
            candidates_k=budget,
        )
        g = _find(
            summary,
            family=family,
            method="randomized_greedy",
            candidates_k=budget,
        )
        d = _find(
            paired,
            family=family,
            comparator="randomized_greedy",
            candidates_k=budget,
        )
        lines.append(
            f"| {FAMILY_LABELS[family]} | "
            f"{_pm(q['expected_best_ratio_mean'], q['expected_best_ratio_ci95'])} | "
            f"{_pm(g['expected_best_ratio_mean'], g['expected_best_ratio_ci95'])} | "
            f"{_pm(d['quantum_minus_classical_mean'], d['quantum_minus_classical_ci95'])} |"
        )

    lines.extend(
        [
            "",
            "**图 2C**",
            "",
            "*候选预算与测试场景的配对性能差值矩阵*",
            "",
            "![不同测试族的配对性能差值热图](../figures/wifi_mis/02C_paired_performance_heatmap.png)",
            "",
            "*注.* 热图以候选预算 K 为行、测试场景为列。颜色编码配对性能差值的均值；"
            "单元格给出均值 ± 95% 置信区间。正值（红色）表示里德堡采样占优，"
            "负值（蓝色）表示随机加权贪心占优。",
            "",
            "### 3.3 为什么不称为全面 quantum advantage",
            "",
            (
                f"中心瓶颈上的 beam-width-16 平均近似比为 {_pm(beam['ratio_mean'], beam['ratio_ci95'])}，"
                "精确枚举为 1。经典 solver 对许多 unit-disk MIS 很强；已有系统研究指出，"
                "此前某些数百到数千节点的准平面实例可被定制或通用经典 solver 很快求解，"
                "放宽模拟退火限制后也可与早期量子结果竞争 [5]。因此本文只声称：在"
                "预先定义的空间复用瓶颈、相同小 K、理想量子演化下，候选质量优于两个"
                "低计算量局部经典 sampler。"
            ),
            "",
            "限制还包括：",
            "",
            "- 测试是结构化合成热点，不是路由器真实 packet trace；",
            "- QuTiP 是无噪声经典模拟，不含装载、排队、脉冲、重复 shots、读出和网络往返；",
            "- 12 节点不提供渐近 scaling 证据；",
            "- unit-disk 近似忽略多径衰落、异构发射功率、隐藏终端与多信道分配；",
            "- 通过经典修复保证安全可能改变物理分布，实际硬件必须重新标定；",
            "- 脉冲选择只针对 bottleneck 家族，跨家族结果应视作外推检验而非再优化机会。",
            "",
            "## 4. 未来展望与更多实用场景",
            "",
            "### 4.1 从理想分布到可部署证据",
            "",
            "下一阶段应按以下门逐级推进：",
            "",
            "1. 用真实家庭/校园/公共热点 trace 替代合成队列，并由测得的 SINR 构建权威冲突图；",
            "2. 在中性原子 QPU 上复现实例分布，报告原始/修复可行率、TV 距离和读出误差；",
            "3. 固定总 wall-clock 或能耗预算，把装载、队列、shots、读出和回传全部计入；",
            "4. 与调优后的 SA、局部搜索、branch-and-bound、GNN scheduler 及硬件友好 beam 比较；",
            "5. 将 sampler 接回动态队列，验证平均时延、p95 时延、丢包率、公平性和吞吐量，而不只看单槽 MWIS；",
            "6. 在数十到数百设备上做预注册 scaling，只有量子曲线在公平资源模型下保持优势才升级主张。",
            "",
            (
                "任意连接无线图也可通过 Rydberg wire/gadget 映射到 unit-disk MWIS，但已知方案"
                "最多会产生二次 qubit 开销 [6]；因此工程上应优先选择几何结构原生、嵌入"
                "开销低的热点或局部子图。"
            ),
            "",
            "### 4.2 同类公众场景",
            "",
            "同一模型可扩展到下列日常情境，但都需重新定义权威约束：",
            "",
            "- 家庭路由器/公共热点：同信道设备或链路的低时延并发调度；",
            "- 蓝牙耳机、可穿戴设备与 IoT：共存干扰下的发送窗口选择；",
            "- 共享充电站：把相互重叠且争用同一端口/馈线的预约作为冲突边；",
            "- 外卖或即时配送 batching：把时间窗或路线不兼容的订单组合表示为冲突图；",
            "- 电影院/餐厅预约：把座位、桌位或时间资源冲突建成加权独立集；",
            "- 城市路口信号相位：把不能同时放行的车流/人流相位作为冲突边。",
            "",
            "## 结论",
            "",
            (
                "当前 sampler 最自然的公众解释是无线干扰图上的 MWIS 候选器。结果支持"
                "一个窄而清楚的结论：理想中性原子演化在预定义的中心干扰瓶颈中，"
                f"以 K={budget} 的小候选预算对随机贪心和单交换局部搜索产生统计上"
                "可检验的候选质量优势；它没有击败所有强经典方法，也尚未转移到物理"
                " QPU。这个边界使结果既能展示量子方法的实际切入点，又不会把模拟中的"
                "局部优越性夸大为全面 quantum advantage。"
            ),
            "",
            "## 参考文献",
            "",
            (
                "[1] J. Choi, S. Oh, and J. Kim, “Quantum Approximation for "
                "Wireless Scheduling,” arXiv:2004.11229 (2020). "
                "https://arxiv.org/abs/2004.11229"
            ),
            "",
            (
                "[2] Z. Zhao, G. Verma, A. Swami, and S. Segarra, “Delay-Oriented "
                "Distributed Scheduling Using Graph Neural Networks,” "
                "arXiv:2111.07017 (2021). https://arxiv.org/abs/2111.07017"
            ),
            "",
            (
                "[3] S. Ebadi et al., “Quantum optimization of maximum independent "
                "set using Rydberg atom arrays,” Science 376, 1209–1215 (2022). "
                "https://doi.org/10.1126/science.abo6587"
            ),
            "",
            (
                "[4] K. Kim et al., “Quantum computing dataset of maximum independent "
                "set problem on king lattice of over hundred Rydberg atoms,” "
                "Scientific Data 11, 111 (2024). "
                "https://doi.org/10.1038/s41597-024-02926-9"
            ),
            "",
            (
                "[5] R. S. Andrist et al., “Hardness of the Maximum Independent Set "
                "Problem on Unit-Disk Graphs and Prospects for Quantum Speedups,” "
                "arXiv:2307.09442 (2023). https://arxiv.org/abs/2307.09442"
            ),
            "",
            (
                "[6] M.-T. Nguyen et al., “Quantum Optimization with Arbitrary "
                "Connectivity Using Rydberg Atom Arrays,” PRX Quantum 4, 010316 "
                "(2023). https://doi.org/10.1103/PRXQuantum.4.010316"
            ),
            "",
            "## 可复现性说明",
            "",
            (
                f"所有实例 seed、几何、权重、精确最优值、完整 sampler 指标与 95% CI "
                "均保存在配套 JSON。主实验包含 "
                f"{len(results['records'])} 个 held-out 实例；报告与 figures 由同一 JSON "
                "确定性生成。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


__all__ = ["render_wifi_mis_report"]
