# Hybrid QRL

`hybrid_qrl` is a research-oriented Python package for constrained
reinforcement-learning action selection:

```text
observation
  -> classical encoder and utility head
  -> classical or quantum candidate sampler
  -> authoritative safety filter
  -> classical critic best-of-K reranking
  -> environment action
```

The package separates reusable model components from tutorials, reproducible
experiments, tests, and longer-form documentation.

## Repository layout

```text
hybrid_qrl/
├── src/hybrid_qrl/       # Installable library source
├── tests/                # Contract and regression tests
├── examples/             # Small tutorial programs
├── experiments/          # Reproducible research entry points
├── docs/                 # Architecture and backend documentation
├── pyproject.toml        # Packaging, dependencies, and tool configuration
├── CONTRIBUTING.md       # Collaboration conventions
├── CITATION.cff          # Academic software citation metadata
└── README.md
```

Only reusable implementation belongs under `src/hybrid_qrl`. Examples should
demonstrate one concept with small inputs. Experiments may generate datasets
and reports but must write them outside the package source tree.

## Installation

From the repository root, install the library in editable mode:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .\hybrid_qrl
```

The core package requires NumPy and SciPy. Optional quantum backends are grouped
as extras for a clean standalone installation:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".\hybrid_qrl[quantum]"
```

The manually downloaded neutral-atom platform remains an external local
backend. Its adapter discovers `QML-Platform-for-Neutral-Atom` at the repository
root or accepts an explicit `backend_source`.

## Quick start

Run the six-node tutorial:

```powershell
.\.venv\Scripts\python.exe .\hybrid_qrl\examples\toy_constrained_action.py `
  --sampler both --candidates 8 --seed 7
```

Run the tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover `
  -s .\hybrid_qrl\tests -v
```

Run the CartPole experiment:

```powershell
.\.venv\Scripts\python.exe .\hybrid_qrl\experiments\cartpole_benchmark.py `
  --quantum-backend dense --candidates 8
```

Run the 20-seed dynamic dispatch benchmark:

```powershell
.\.venv\Scripts\python.exe `
  .\hybrid_qrl\experiments\dispatch_scaling_benchmark.py `
  --seeds 20 --train-episodes 320 --k 16 --latency-ms 20
```

This benchmark trains a reward-only actor-critic and compares time-limited
MILP, simulated annealing, MCMC, local search, beam search, randomized greedy,
learned autoregressive proposals, and a classical Rydberg-blockade surrogate.
It covers 20, 40, 60, and 100 binary decisions under equal-K and equal-latency
protocols. The generated raw records and report are written to `results/`.

Run the conditional-advantage calibration study:

```powershell
.\.venv\Scripts\python.exe -B `
  .\hybrid_qrl\experiments\conditional_advantage_study.py `
  --seeds 20 --training-iterations 140 `
  --calibration-seeds 20 --phase-seeds 40 --k 16
```

This study trains through the sampler using reward-only SPSA, calculates
epsilon-optimal probability and K95, validates the eight-qubit end-to-end path,
calibrates dense/QuTiP/manual distributions at 8–12 qubits, and searches a
16-decision physical phase map. Its separate safety, surrogate-opportunity, and
calibration-transfer gates prevent surrogate results from being mislabeled as
quantum-backend evidence.

Run the latency-aware delayed/asynchronous extension:

```powershell
$env:PYTHONPATH = ".\hybrid_qrl\src"
.\.venv\Scripts\python.exe -B `
  .\hybrid_qrl\experiments\latency_aware_dispatch.py
```

The default run assigns 1,000 ms to each environment step, applies a 3,000 ms
quantum-result deadline, repairs stale job-identity actions, and compares a
blocking quantum policy with asynchronous beam and greedy fallbacks. The
bundled stress trace is explicitly non-hardware evidence; provide measured QPU
timestamps with `--latency-trace` to evaluate the physical-latency gate.

Run the fixed-K scale-aware and stable-backlog extension:

```powershell
$env:PYTHONPATH = ".\hybrid_qrl\src"
.\.venv\Scripts\python.exe -B `
  .\hybrid_qrl\experiments\stable_backlog_scaling.py
```

This experiment retrains the actor on 20--100 decision environments, selects a
standardized utility-to-detuning map on disjoint seeds, and confirms it at
`K=16`. It then sends only a bounded, long-lived target block to the delayed
sampler while beam search handles the immediate lane. An identically delayed
beam planner controls for the benefit of backlog partitioning.

## Documentation

- [Architecture and extension guide](docs/architecture.md)
- [Source API reference](docs/api_reference.md)
- [Manual neutral-atom backend integration](docs/manual_backend_integration.md)
- [Experiment and reporting protocol](docs/research_protocol.md)
- [Dynamic dispatch benchmark](docs/dispatch_benchmark.md)
- [Conditional quantum-assisted advantage study](docs/conditional_advantage_study.md)
- [Latency-aware dynamic dispatch extension](docs/latency_aware_dispatch.md)
- [Scale-aware stable-backlog study](docs/stable_backlog_scaling.md)
- [Migration from the former template layout](docs/migration.md)

The editable diagrams remain in the repository-level `figures/` directory, and
generated reports remain in the repository-level `results/` directory.
