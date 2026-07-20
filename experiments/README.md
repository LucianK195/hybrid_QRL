# Experiments

Experiment scripts contain reproducible benchmark orchestration rather than
reusable library implementation.

- `cartpole_benchmark.py`: offline linear-policy CartPole integration test.
- `cartpole_budget_sweep.py`: comparison across candidate budgets `K`.
- `cartpole_multiseed_study.py`: backend-by-budget matrix with paired,
  multi-seed aggregation and JSON/Markdown reporting.
- `dispatch_scaling_benchmark.py`: reward-trained dynamic resource dispatch at
  20–100 binary decisions with serious optimization/search baselines, equal-K
  and equal-latency protocols, dynamic rollouts, and robustness sweeps.

Run scripts from the repository root after installing `hybrid_qrl` in editable
mode. Their default datasets and reports are written to the repository-level
`results/` directory.

Run the complete multi-seed CartPole study with:

```powershell
.\.venv\Scripts\python.exe `
  .\hybrid_qrl\experiments\cartpole_multiseed_study.py `
  --backends dense qutip manual `
  --budgets 1 2 4 8 16 `
  --seeds 17 29 43 71
```

This produces `results/cartpole_multiseed_results.json` with compact per-trial
records and `results/cartpole_multiseed_report.md` with aggregate tables,
paired confidence intervals, and limitations.

The classical controls include direct argmax, epsilon-greedy, single-sample
Boltzmann/softmax, uniform random shooting best-of-K, softmax best-of-K, and
randomized weighted greedy best-of-K. These share the learned linear utility
model; they compare action-selection mechanisms rather than different training
algorithms such as PPO or DQN.

Run the dispatch benchmark with its preregistered minimum of 20 held-out seeds:

```powershell
.\.venv\Scripts\python.exe `
  .\hybrid_qrl\experiments\dispatch_scaling_benchmark.py `
  --seeds 20 --train-episodes 320 --k 16 --latency-ms 20
```

The output files are `results/dispatch_benchmark_results.json` and
`results/dispatch_benchmark_report.md`. The JSON retains per-instance solver
status, MIP gap, candidate counts, raw feasibility, post-repair candidates,
critic score, realized reward, reference regret, latency, and latency-budget
compliance. The Rydberg path in this experiment is explicitly a scalable
classical surrogate; it is not a neutral-atom hardware timing result.
