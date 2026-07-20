# Experiments

Experiment scripts contain reproducible benchmark orchestration rather than
reusable library implementation.

- `cartpole_benchmark.py`: offline linear-policy CartPole integration test.
- `cartpole_budget_sweep.py`: comparison across candidate budgets `K`.
- `cartpole_multiseed_study.py`: backend-by-budget matrix with paired,
  multi-seed aggregation and JSON/Markdown reporting.

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
