# Integrating the downloaded neutral-atom simulator

The project now includes `ManualNeutralAtomBackendSampler`, an adapter between
the hybrid RL action-head contract and the downloaded simulator at:

```text
QML-Platform-for-Neutral-Atom/
└── src/neutral_atom/simulator/
```

The execution path is:

```text
HybridActionHead
  -> state-dependent utilities
  -> ManualNeutralAtomBackendSampler
       -> create_simulator("qutip", positions=..., C6=...)
       -> simulator.reset()
       -> simulator.evolve_adiabatic(protocol, detuning_weights)
       -> final-state probabilities / simulator.sample()
  -> SafetyFilter
  -> classical critic best-of-K reranking
  -> environment action
```

The PennyLane device and HAL are not required for this path. They remain
available for circuit-level workflows, while the RL project uses the lower-level
Simulator API directly.

## Minimal use

```python
import numpy as np

from hybrid_qrl import (
    ConflictGraph,
    HybridActionHead,
    IdentityEncoder,
    ManualNeutralAtomBackendSampler,
    StaticUtilityHead,
    UtilityCritic,
)

graph = ConflictGraph(
    nodes=2,
    edges=((0, 1),),
    min_selected=1,
    max_selected=1,
)

sampler = ManualNeutralAtomBackendSampler(
    positions=np.array([[0.0, 0.0], [1.0, 0.0]]),
    C6=10.0,
    cache_decimals=2,
)

head = HybridActionHead(
    encoder=IdentityEncoder(),
    utility_head=StaticUtilityHead((1.2, 0.8)),
    sampler=sampler,
    critic=UtilityCritic(),
    candidates=16,
)

decision = head.select(np.zeros(2), graph, seed=7)
print(decision)
```

When the downloaded directory is at the repository root, the adapter finds its
`src` directory automatically. If it is elsewhere:

```python
sampler = ManualNeutralAtomBackendSampler(
    backend_source=r"D:\external\QML-Platform-for-Neutral-Atom",
    positions=positions,
)
```

This avoids installing PennyLane and InfluxDB when only the simulator API is
needed.

## Custom adiabatic protocol

Use the protocol class supplied by the downloaded backend:

```python
import sys
from pathlib import Path

backend_root = Path("QML-Platform-for-Neutral-Atom").resolve()
sys.path.insert(0, str(backend_root / "src"))

from neutral_atom.simulator import AdiabaticProtocol

protocol = AdiabaticProtocol(
    total_time=4.0,
    n_steps=80,
    omega_max=1.5,
    delta_g_initial=-3.0,
    delta_l_max=3.0,
)

sampler = ManualNeutralAtomBackendSampler(
    backend_source=backend_root,
    positions=positions,
    C6=10.0,
    protocol=protocol,
)
```

The utility vector becomes `detuning_weights`:

```text
utilities[i] * utility_scale -> detuning_weights[i]
```

The backend Hamiltonian therefore uses the learned action preference as the
local detuning weight.

## Geometry is part of the model

The downloaded backend does not accept an arbitrary conflict-edge list. It
computes every pair interaction:

```text
V_ij = C6 / distance(i, j)^6
```

Consequently:

- intended conflict pairs should be close;
- non-conflicting pairs should be farther apart;
- some residual non-edge interaction normally remains;
- not every abstract graph has an exact two-dimensional Rydberg embedding.

Inspect an embedding before experiments:

```python
print(sampler.geometry_report(graph))
```

The most useful field is `edge_to_nonedge_separation`. Larger values mean that
the weakest intended edge interaction is still stronger than the strongest
unintended non-edge interaction.

This geometric Hamiltonian is only a proposal mechanism. `SafetyFilter`
continues to check the original `ConflictGraph`, so an imperfect embedding
cannot directly execute an invalid action.

## Noise configuration

Any additional factory arguments can be passed through `simulator_kwargs`:

```python
from neutral_atom.simulator import NoiseConfig

sampler = ManualNeutralAtomBackendSampler(
    positions=positions,
    C6=10.0,
    simulator_kwargs={
        "noise": NoiseConfig(
            blockade_enabled=True,
            C6=10.0,
            gamma_r=0.001,
            gamma_d=0.001,
        )
    },
)
```

Non-zero decay or dephasing causes the downloaded backend to use QuTiP
`mesolve`; the noiseless path uses `sesolve`.

## Run the included integrations

Six-decision constrained-action example:

```powershell
.\.venv\Scripts\python.exe .\hybrid_qrl\examples\toy_constrained_action.py `
  --sampler manual `
  --candidates 32 `
  --seed 7
```

CartPole integration:

```powershell
.\.venv\Scripts\python.exe .\hybrid_qrl\experiments\cartpole_benchmark.py `
  --quantum-backend manual `
  --candidates 16 `
  --seed 17 `
  --output results\cartpole_manual_backend_k16.json
```

In the reproduced 30-episode CartPole run, K=16 matched the classical linear
controller exactly: mean return 437.90 and solved rate 83.3%. K=8 reached mean
return 426.97 and solved rate 80.0%, showing that this backend/protocol requires
a slightly larger candidate budget than the project's earlier ideal emulator.

## Caching and training

Every distinct utility vector requires a new quantum evolution. For repeated
RL states, the adapter caches final probability distributions after rounding
utilities to `cache_decimals`.

- Use `cache_decimals=2` for fast integration tests.
- Use `cache_decimals=3` or `None` for higher-fidelity studies.
- Disable or carefully audit caching during gradient-based training, because
  rounding makes the sampler piecewise constant.

The cache accelerates classical simulation only. It does not model QPU runtime.

## Production boundary

Keep these responsibilities separate:

| Component | Responsibility |
|---|---|
| Classical encoder/utility head | Learn state-dependent action preferences |
| Downloaded simulator | Geometry, blockade, adiabatic evolution, sampling |
| Adapter | Translate utilities and K into simulator calls |
| Safety filter | Enforce the real operational constraints |
| Classical critic | Select the best long-horizon candidate |
| RL algorithm | Update encoder, utilities, and critic |

This boundary also allows a future hardware backend to replace `qutip` through
the downloaded simulator's `register_backend()` mechanism without rewriting the
RL pipeline.
