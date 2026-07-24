# Azure bundle-conflict benchmark

## Purpose

The raw Azure Packing benchmark maps one VM request to one binary decision.
That representation leaves CPU, memory, SSD, and NIC as cumulative constraints
that cannot be expressed exactly by pairwise Rydberg blockade edges.

This extension uses a configuration formulation:

```text
Azure request window
  -> reward-trained request values
  -> classical capacity-feasible bundle generation
  -> (machine slot, bundle) conflict graph
  -> Rydberg or classical proposal
  -> exact conflict repair
  -> learned additive reranking
  -> safe multi-machine allocation
```

It is a trace-driven algorithmic benchmark. It is not Azure's production
allocator, a physical neutral-atom experiment, or a hardware-latency result.

## Exact pairwise formulation

A node `b` is a complete configuration for one machine slot:

```text
b = (machine, member VM requests, resource usage, utility)
```

The bundle generator verifies all four constraints before exposing the node:

```text
sum(i in b) demand[i, resource] <= machine_capacity[resource].
```

Two bundle nodes have an edge when either:

- they are alternative complete configurations for the same machine slot; or
- their member sets contain at least one common VM request.

Therefore an independent set selects at most one configuration per machine,
never allocates one request twice, and is capacity-safe without a cumulative
post-sampling repair.

## Frozen protocol

- Official Azure Packing 2020 SQLite trace, hardware generation `machineId=16`.
- Training: 30 chronological windows from days 0.25--9.75.
- Test: 20 held-out windows from days 10.00--13.75.
- Raw requests per window: 200.
- Machine slots: 2.
- Bundle decision nodes: 20, 40, 60, 80, and 100.
- Per-machine capacity scales: 0.75 and 1.00.
- Candidate budgets: 4, 16, and 64.
- Epsilon-optimal threshold: 5%.

Each stochastic layout first partitions requests between machine slots and
then greedily packs each partition. The configurations from one layout are
disjoint, so every library contains complete feasible multi-machine choices.
Larger node-count settings are nested prefixes of the same generated library.

## Two optimization references

The restricted bundle MILP chooses a maximum-weight independent set from the
generated node library. It measures sampler quality conditional on the
classical preprocessing.

The direct assignment MILP uses one variable for every request-machine pair,
the original four capacity constraints for every machine, and a one-machine
limit for every request. It measures the best value before restriction to the
finite bundle library. The solver receives a five-second time limit. Exact
objectives are used when available; otherwise ratios use the reported
maximization upper bound, making the result conservative rather than dividing
by a potentially suboptimal incumbent.

The reported ratios are:

```text
best/bundle = proposal reward / restricted bundle MILP
library coverage = restricted bundle MILP / direct assignment MILP
best/direct = proposal reward / direct assignment MILP
```

This decomposition prevents poor bundle generation from being attributed to
the sampler and prevents a strong sampler result inside a weak library from
being described as strong end-to-end packing.

## Methods

- `rydberg_geometry`: stochastic blockade proposal on a fitted two-dimensional
  unit-disk graph, followed by repair against the exact bundle conflicts.
- `blockade_exact_graph`: the same stochastic schedule on the authoritative
  graph. This is a nonphysical diagnostic upper-bound.
- `repair_only`: select every bundle and apply the same exact graph repair.
- `beam_search`: capacity-safe beam search on the exact bundle graph.
- `randomized_greedy`: Gumbel-perturbed greedy independent sets.

All methods use the same reward-trained linear request-value model. No MILP
labels are used for proposal scoring or reranking.

## Preregistered gates

The trace-to-bundle pipeline passes only if:

- every direct-assignment MILP has a finite incumbent/upper bound with at most
  a 1% reported gap, and all bundle MILPs finish with zero gap;
- all executed actions are authoritatively feasible;
- 100-node library coverage has a lower 95% confidence bound of at least 0.90;
- geometry best-of-16 end-to-end reward has a lower bound of at least 0.90.

The geometry sampler contribution passes separately only if:

- raw feasibility is at least 20%;
- exact repair removes no more than 10% of selected bundle nodes;
- the paired lower confidence bound over repair-only is positive;
- at least 80% of exact compatible bundle pairs remain non-edges after the
  two-dimensional physical embedding.

The last metric is stricter than edge Jaccard for dense graphs. A fitted
geometry can reproduce nearly all conflict edges yet add blockade edges over
the few non-edges that encode useful multi-machine bundle combinations.

These gates establish neither hardware transfer nor quantum advantage.

## Reproduction

```powershell
$env:PYTHONPATH = ".\hybrid_qrl\src"
.\.venv\Scripts\python.exe -B `
  .\hybrid_qrl\experiments\azure_bundle.py
```

Outputs are written to:

```text
results/azure_bundle_benchmark.json
results/azure_bundle_benchmark.md
```
