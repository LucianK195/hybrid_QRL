# Experiments

Experiment scripts contain reproducible benchmark orchestration rather than
reusable library implementation.

- `cartpole_benchmark.py`: offline linear-policy CartPole integration test.
- `cartpole_budget_sweep.py`: comparison across candidate budgets `K`.

Run scripts from the repository root after installing `hybrid_qrl` in editable
mode. Their default datasets and reports are written to the repository-level
`results/` directory.

