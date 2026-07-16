# Experiment and reporting protocol

## Goal

The research question is whether a constrained quantum sampler can maintain a
useful probability of high-value actions as the binary candidate space grows,
while a classical safety layer and critic preserve operational correctness.

## Required comparisons

For every problem size and state distribution, compare samplers under:

1. an equal candidate budget `K`; and
2. an equal end-to-end decision-time budget where measurable.

At minimum include direct classical argmax, randomized weighted greedy, and the
quantum or quantum-inspired sampler. For small instances, include an exact
solver only as an evaluation oracle.

## Required metrics

- raw feasible-sample probability;
- safety-filter acceptance and fallback rate;
- number and structural diversity of unique feasible candidates;
- best-of-`K` critic value and regret to the oracle;
- optimum or epsilon-optimum coverage;
- environment return and task-specific service metrics;
- sampling, filtering, critic, and total decision latency.

## Scaling claims

The unrestricted action count `2^n` motivates candidate sampling but is not
evidence of quantum advantage. A favorable empirical trend requires useful
candidate probability, required `K`, and end-to-end latency to scale better
than relevant classical baselines.

Dense statevector and matrix-exponential timings are emulator costs. They must
not be presented as neutral-atom hardware runtimes.

## Dataset experiments

For trace-driven scheduling, document the conversion from a trace snapshot to:

- the encoded state;
- candidate binary decisions;
- pairwise conflict edges;
- cumulative constraints checked by the safety filter;
- reward and episode termination;
- train, validation, and held-out time ranges.

This is especially important for Alibaba Cluster Trace experiments because
CPU, memory, and GPU capacities are cumulative constraints and generally cannot
be represented exactly by a pairwise conflict graph.

