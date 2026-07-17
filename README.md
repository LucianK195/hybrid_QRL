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

## Documentation

- [Architecture and extension guide](docs/architecture.md)
- [Source API reference](docs/api_reference.md)
- [Manual neutral-atom backend integration](docs/manual_backend_integration.md)
- [Experiment and reporting protocol](docs/research_protocol.md)
- [Migration from the former template layout](docs/migration.md)

The editable diagrams remain in the repository-level `figures/` directory, and
generated reports remain in the repository-level `results/` directory.
