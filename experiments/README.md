# Experiments

The directory has exactly one executable Python entry point per dataset:

| Dataset | Entry point | Stages |
|---|---|---|
| CartPole trajectories | `cartpole.py` | `benchmark`, `budget-sweep`, `multiseed` |
| Synthetic dispatch graphs | `dispatch.py` | `scaling`, `conditional`, `latency`, `backlog`, `generalization`, `export`, `plot` |
| Azure Packing trace | `azure_packing.py` | `benchmark` |
| Azure bundle-conflict graphs | `azure_bundle.py` | `benchmark` |
| Public Wi-Fi interference graphs | `wifi_mis.py` | `benchmark` |

The entry points contain no reusable implementation. OOP command objects live
in `src/hybrid_qrl/applications`, domain logic lives in `src/hybrid_qrl`, and
shared result/plot helpers live in `src/hybrid_qrl/utilities`, with all
dataset-specific Markdown renderers isolated under
`src/hybrid_qrl/utilities/reports`.

Run commands from the workspace root after installing `hybrid_qrl` in editable
mode, or set `PYTHONPATH=.\hybrid_qrl\src`. The default stage remains compatible
with the former primary script, so both of these forms run the same CartPole
benchmark:

```powershell
.\.venv\Scripts\python.exe .\hybrid_qrl\experiments\cartpole.py
.\.venv\Scripts\python.exe .\hybrid_qrl\experiments\cartpole.py benchmark
```

List all stages and inspect stage-specific arguments with:

```powershell
.\.venv\Scripts\python.exe .\hybrid_qrl\experiments\dispatch.py --help
.\.venv\Scripts\python.exe .\hybrid_qrl\experiments\dispatch.py backlog --help
```

Former script names map directly to stages:

| Former entry point | New command |
|---|---|
| `cartpole_benchmark.py` | `cartpole.py benchmark` |
| `cartpole_budget_sweep.py` | `cartpole.py budget-sweep` |
| `cartpole_multiseed_study.py` | `cartpole.py multiseed` |
| `dispatch_scaling_benchmark.py` | `dispatch.py scaling` |
| `conditional_advantage_study.py` | `dispatch.py conditional` |
| `latency_aware_dispatch.py` | `dispatch.py latency` |
| `stable_backlog_scaling.py` | `dispatch.py backlog` |
| `dispatch_generalization_stress.py` | `dispatch.py generalization` |
| `export_dispatch_test_dataset.py` | `dispatch.py export` |
| `plot_dispatch_benchmark.py` | `dispatch.py plot` |
| `azure_packing_benchmark.py` | `azure_packing.py benchmark` |
| `azure_bundle_benchmark.py` | `azure_bundle.py benchmark` |

## CartPole

```powershell
.\.venv\Scripts\python.exe .\hybrid_qrl\experiments\cartpole.py multiseed `
  --backends dense qutip manual `
  --budgets 1 2 4 8 16 `
  --seeds 17 29 43 71
```

The established JSON schemas and default filenames are retained:
`cartpole_hybrid_comparison.json`, `cartpole_candidate_budget_sweep.json`,
`cartpole_multiseed_results.json`, and `cartpole_multiseed_report.md`.

## Synthetic dispatch graphs

The stages form one dataset lifecycle:

```text
scaling -> conditional -> latency/backlog -> generalization
    |
    +-> export -> plot
```

Examples:

```powershell
.\.venv\Scripts\python.exe .\hybrid_qrl\experiments\dispatch.py scaling `
  --seeds 20 --train-episodes 320 --k 16 --latency-ms 20

.\.venv\Scripts\python.exe .\hybrid_qrl\experiments\dispatch.py conditional `
  --seeds 20 --training-iterations 140 `
  --calibration-seeds 20 --phase-seeds 40 --k 16

.\.venv\Scripts\python.exe .\hybrid_qrl\experiments\dispatch.py latency
.\.venv\Scripts\python.exe .\hybrid_qrl\experiments\dispatch.py backlog
.\.venv\Scripts\python.exe .\hybrid_qrl\experiments\dispatch.py generalization
.\.venv\Scripts\python.exe .\hybrid_qrl\experiments\dispatch.py export
.\.venv\Scripts\python.exe .\hybrid_qrl\experiments\dispatch.py plot
```

Each stage keeps its previous default input dependencies and output paths under
the workspace-level `results`, `datasets`, and `figures` directories.

## Azure trace formulations

```powershell
.\.venv\Scripts\python.exe .\hybrid_qrl\experiments\azure_packing.py
.\.venv\Scripts\python.exe .\hybrid_qrl\experiments\azure_bundle.py
```

Both entry points reuse the same official Azure Packing trace path and selected
stable-backlog model by default. Their distinct result schemas, filenames, gate
logic, and direct/bundle oracle summaries are unchanged.

## Public Wi-Fi MWIS

```powershell
$env:PYTHONPATH = ".\hybrid_qrl\src"
.\.venv\Scripts\python.exe -B .\hybrid_qrl\experiments\wifi_mis.py
```

The experiment maps one busy-hotspot airtime frame to a maximum-weight
independent set on a unit-disk interference graph. It selects one ideal
Rydberg pulse on disjoint bottleneck training seeds, freezes it, and compares
held-out QuTiP distributions with randomized greedy, one-swap local search,
simulated annealing, beam search, and an exact 12-node oracle. The generated
Chinese academic report and publication-ready figures explicitly distinguish
ideal sampler quality from physical-QPU advantage.
