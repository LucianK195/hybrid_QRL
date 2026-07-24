# Hybrid classical-quantum action-head architecture

This package provides a reusable implementation of the architecture:

```text
observation
  -> classical encoder
  -> state-dependent action utilities
  -> quantum constrained-action sampler
  -> classical safety filter
  -> critic reranking
  -> environment action
```

It implements the action-selection boundary only. The surrounding PPO, SAC,
CQL, DQN, or other RL training loop remains classical and can update the
encoder, utility head, and critic.

## Project map

| Path | Responsibility | Replace in a real project? |
|---|---|---|
| `src/hybrid_qrl/core.py` | Shared types and component protocols | Usually no |
| `src/hybrid_qrl/classical.py` | Identity encoder, fixed utilities, greedy baseline | Yes |
| `src/hybrid_qrl/quantum.py` | QPU integration skeleton and Rydberg adapters | Yes |
| `src/hybrid_qrl/pipeline.py` | Safety filtering, fallback, and critic reranking | Extend |
| `src/hybrid_qrl/applications/` | OOP orchestration for the four dataset entry points | Extend |
| `src/hybrid_qrl/utilities/` | Shared result serialization, metrics, paths, and plots | Usually no |
| `src/hybrid_qrl/utilities/reports/` | Dataset-specific Markdown renderers | Usually no |
| `experiments/` | Four thin dataset entry points; no reusable implementation | Usually no |
| `examples/toy_constrained_action.py` | Runnable six-decision tutorial | Use as wiring reference |
| `tests/test_pipeline.py` | Contract and safety tests | Extend |

The editable architecture diagram is in
[`../../figures/hybrid_quantum_rl_architecture.drawio`](../../figures/hybrid_quantum_rl_architecture.drawio).

The downloaded `neutral_atom.simulator` integration is documented separately in
[`manual_backend_integration.md`](manual_backend_integration.md).

## Run the example

From the repository root:

```powershell
.\.venv\Scripts\python.exe .\hybrid_qrl\examples\toy_constrained_action.py `
  --sampler both `
  --candidates 8 `
  --seed 7
```

The example uses the same six-node conflict graph as the small experiment. It
runs the classical randomized-greedy sampler and the idealized Qiskit Rydberg
emulator through the same safety and critic stages. Exact enumeration appears
only as a small-system evaluation reference.

Run the contract tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover `
  -s .\hybrid_qrl\tests `
  -v
```

## Manual integration

### 1. Define the hard constraints

Pairwise exclusions are represented by `ConflictGraph`:

```python
from hybrid_qrl import ConflictGraph

graph = ConflictGraph(
    nodes=6,
    edges=((0, 1), (1, 2), (2, 3)),
)
```

For capacity, equality, routing, or power-flow constraints, extend
`SafetyFilter.apply`. The safety filter is authoritative even when the quantum
program is intended to encode the same constraints.

### 2. Replace the encoder

Implement one method:

```python
class MyEncoder:
    def encode(self, observation):
        return learned_state_vector
```

The template's `IdentityEncoder` is only suitable when the observation is
already a one-dimensional numeric vector.

### 3. Replace the utility head

The utility head creates the local detunings or action weights for the current
state:

```python
class MyUtilityHead:
    def utilities(self, encoded_state, graph):
        # Must return shape (graph.nodes,)
        return state_dependent_node_utilities
```

Pairwise learned utilities can be added by extending the sampler request type.
The minimal template deliberately exposes only node weights and graph edges.

### 4. Connect a QPU

Subclass `QuantumSamplerTemplate` and implement three backend-specific hooks:

```python
from hybrid_qrl import QuantumSamplerTemplate

class MyQuantumSampler(QuantumSamplerTemplate):
    name = "my_quantum_backend"

    def build_program(self, utilities, graph):
        # Map utilities to detunings and graph edges to blockade interactions.
        return program

    def execute(self, program, shots, seed):
        return backend_measurements

    def decode(self, measurement, nodes):
        return tuple_of_zero_one_bits
```

The inherited `sample` method handles the pipeline contract. Do not remove the
downstream safety check: real measurements can violate intended constraints due
to noise, embedding error, or incomplete penalty strength.

`RydbergEmulatorSampler` is a working reference adapter, but it constructs dense
operators and emulates them on a CPU. Its runtime is not representative of
neutral-atom hardware.

### QuTiP sampler

`QuTiPRydbergSampler` implements the same interface with QuTiP's continuous
time solver API. It builds the Hamiltonian from tensor-product number and Pauli
operators, wraps the time-dependent pulse in `QobjEvo`, evolves `|00...0>` with
`sesolve`, and samples the final state probabilities.

```python
from hybrid_qrl import QuTiPRydbergSampler

sampler = QuTiPRydbergSampler(cache_decimals=3)
```

Run it on the six-node example:

```powershell
.\.venv\Scripts\python.exe .\hybrid_qrl\examples\toy_constrained_action.py `
  --sampler qutip `
  --candidates 8 `
  --seed 7
```

Or select it in the CartPole integration test:

```powershell
.\.venv\Scripts\python.exe .\hybrid_qrl\experiments\cartpole.py benchmark `
  --quantum-backend qutip `
  --candidates 8
```

QuTiP is installed through `requirements.txt`. On Windows, the environment-local
`msvc-runtime` dependency supplies the runtime DLLs required by QuTiP's compiled
extensions; the adapter explicitly registers the virtual-environment DLL search
directories before importing QuTiP.

### Downloaded neutral-atom simulator

`ManualNeutralAtomBackendSampler` calls the downloaded multi-layer simulator
through its public factory and adiabatic API:

```python
sampler = ManualNeutralAtomBackendSampler(
    positions=positions,
    C6=10.0,
    cache_decimals=2,
)
```

Select it in the provided examples with `--sampler manual` or
`--quantum-backend manual`. The external package stays unmodified; the adapter
automatically loads its local `src` directory.

### 5. Replace the critic

The provided `UtilityCritic` only computes the immediate weighted score. A real
RL critic should estimate long-horizon value:

```python
class MyCritic:
    def value(self, encoded_state, action, utilities):
        return learned_q_value
```

The critic receives only feasible candidates and selects the best candidate in
the batch.

### 6. Assemble the action head

```python
from hybrid_qrl import HybridActionHead

action_head = HybridActionHead(
    encoder=MyEncoder(),
    utility_head=MyUtilityHead(),
    sampler=MyQuantumSampler(),
    critic=MyCritic(),
    candidates=8,
)

decision = action_head.select(observation, graph, seed=episode_seed)
environment.step(decision.action)
```

If every quantum candidate is invalid, the template automatically invokes the
classical randomized-greedy fallback. This provides a safe action path but
should also be logged as a QPU failure or distribution-shift event.

## Recommended experiment contract

For every problem size and state distribution, compare samplers under both an
equal candidate budget and an equal wall-clock budget. Record:

- feasible-sample rate before the safety filter;
- best-of-K critic value and candidate regret;
- epsilon-optimal coverage when an oracle is available;
- unique feasible candidates and structural diversity;
- QPU calls, end-to-end decision latency, and fallback rate;
- return under validation states and distribution shift.

The unrestricted action count `2^n` is motivation, not proof of advantage.
Favorable scaling requires the probability of useful candidates to remain high
enough that K and total latency do not grow exponentially.

## CartPole integration test

`hybrid_qrl.cartpole.benchmark` contains a dependency-free implementation of the
standard CartPole-v1 dynamics. It generates an offline trajectory dataset,
fits a basic linear policy, and evaluates common action selectors on identical
environment seeds:

- uniformly random control;
- direct classical linear-policy argmax;
- epsilon-greedy exploration;
- Boltzmann/softmax action sampling;
- uniform random shooting with best-of-K utility reranking;
- softmax proposals with best-of-K utility reranking;
- classical randomized-greedy candidates followed by critic reranking;
- two-qubit Rydberg-emulated candidates followed by the same safety filter and
  critic.

The best-of-K classical policies use the same candidate budget as the quantum
sampler. This isolates candidate-distribution quality from the learned utility
model and avoids treating weak random control as the primary baseline.

CartPole uses a one-hot two-bit action, so its graph contains one conflict edge
and an exactly-one cardinality constraint. This tests the integration contract,
not the large-action-space hypothesis.

Run the primary K=8 comparison:

```powershell
.\.venv\Scripts\python.exe .\hybrid_qrl\experiments\cartpole.py benchmark `
  --training-episodes 64 `
  --evaluation-episodes 30 `
  --candidates 8 `
  --seed 17
```

Run the candidate-budget sweep:

```powershell
.\.venv\Scripts\python.exe .\hybrid_qrl\experiments\cartpole.py budget-sweep `
  --budgets 1 2 4 8 `
  --training-episodes 64 `
  --evaluation-episodes 30 `
  --seed 17
```

The Rydberg benchmark uses a cached, ideal dense statevector calculation. It
rounds utility vectors to two decimals for cache reuse and contains no hardware
noise or QPU latency. See
[`../../results/cartpole_hybrid_analysis.md`](../../results/cartpole_hybrid_analysis.md)
for the reproduced results and limitations.
