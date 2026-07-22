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
- `conditional_advantage_study.py`: sampler-in-loop reward training,
  epsilon-optimal/K95 metrics, dense/QuTiP/manual calibration at 8–12 qubits,
  an eight-qubit dynamic pipeline proof, and a 20+20-seed physical phase map.

Run scripts from the repository root after installing `hybrid_qrl` in editable
mode. Their default datasets and reports are written to the repository-level
`results/` directory.

`stable_backlog_scaling.py` is the fixed-K size-aware detuning confirmation and
delayed stable-backlog study. It includes an identically delayed beam control.

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

Run the conditional study with:

```powershell
.\.venv\Scripts\python.exe -B `
  .\hybrid_qrl\experiments\conditional_advantage_study.py `
  --seeds 20 --training-iterations 140 `
  --calibration-seeds 20 --phase-seeds 40 --k 16
```

This produces `results/conditional_advantage_results.json` and
`results/conditional_advantage_report.md`. The report uses a claim ladder:
safe pipeline proof, surrogate opportunity, distribution transfer, and manual
geometry-backend quality. Only passing every gate would justify a conditional
quantum-assisted advantage claim.

## Held-out dispatch graph dataset

`export_dispatch_test_dataset.py` freezes the paired held-out scaling states as
a JSON Lines graph dataset with geometry, authoritative edges, node features,
linear reward terms, reference reward, and a SHA-256 manifest.

```powershell
$env:PYTHONPATH = ".\hybrid_qrl\src"
.\.venv\Scripts\python.exe `
  .\hybrid_qrl\experiments\export_dispatch_test_dataset.py
```

The default dataset contains 80 test-only instances: 20 seeds at each of 20,
40, 60, and 100 binary decisions. `datasets/dispatch_test_v1.jsonl` stores one
graph per line and `datasets/dispatch_test_v1_manifest.json` records provenance,
counts, reward semantics, and content hashes.

After exporting the dataset, create the five editable SVG figures with:

```powershell
$env:PYTHONPATH = ".\hybrid_qrl\src"
.\.venv\Scripts\python.exe `
  .\hybrid_qrl\experiments\plot_dispatch_benchmark.py
```

## Latency-aware dispatch

`latency_aware_dispatch.py` assigns physical duration to each synthetic
environment step, replays a timestamped latency distribution, tracks persistent
job identities, repairs stale actions, and compares blocking quantum execution
with asynchronous beam/greedy fallback. It retains all conditional-advantage
gates and adds measured-QPU, deadline, safety, return, and utilization gates.

```powershell
$env:PYTHONPATH = ".\hybrid_qrl\src"
.\.venv\Scripts\python.exe -B `
  .\hybrid_qrl\experiments\latency_aware_dispatch.py
```

The default deterministic stress trace validates the architecture but is not
hardware evidence. Supply `--latency-trace` with timestamped observations whose
source kind is `measured_qpu` for the physical-latency gate.

## Scale-aware stable backlog

Run the scaling and future-batch study with:

```powershell
$env:PYTHONPATH = ".\hybrid_qrl\src"
.\.venv\Scripts\python.exe -B `
  .\hybrid_qrl\experiments\stable_backlog_scaling.py
```

The default protocol holds `K=16`, uses separate selection and confirmation
seeds, and compares the frozen legacy sampler, multi-size reward training,
scale-aware detuning, and beam search. The future-batch section reserves only
the highest-priority 25% of stable jobs, capped at 20, while beam handles the
unreserved immediate lane. Results are written to
`results/stable_backlog_scaling_results.json` and
`results/stable_backlog_scaling_report.md`.
