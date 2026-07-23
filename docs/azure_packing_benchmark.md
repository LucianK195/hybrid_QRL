# Azure Packing 2020 trace benchmark

## Scope

This experiment evaluates the hybrid candidate-generation pipeline on the
official Microsoft Azure Packing 2020 trace. The trace contains VM requests,
priority, lifetime, and normalized CPU, memory, SSD, and network allocations
for compatible hardware generations.

The experiment is an offline, trace-driven packing benchmark. It does not
reconstruct Azure's production allocator and does not treat local surrogate
runtime as neutral-atom hardware latency.

## Frozen protocol

- Hardware generation: `machineId=16`, selected because it has the largest VM
  type coverage in the released `vmType` table.
- Training split: request anchors from trace days 0.25 through 9.75.
- Test split: request anchors from trace days 10.00 through 13.75.
- Training windows: 30; test windows: 20.
- Maximum requests per window: 100.
- Tested decision sizes: 20, 40, 60, 80, and 100.
- Capacity scales: 0.50, 0.75, and 1.00 of one normalized machine.
- Candidate budgets: 4, 16, and 64.
- Near-optimal threshold: epsilon = 5%.
- Reference: zero-gap MILP over the same requests and four cumulative resource
  constraints.

The node utility and action critic are fitted on the training split using only
trace-derived rewards from sampled feasible actions. MILP actions and
objectives are not training labels.

## Decision model

For request `i`, the binary decision is

```text
x_i = 1  if the VM request is admitted to the capacity pool
x_i = 0  otherwise.
```

The offline utility proxy gives high-priority VMs four times the base weight
of low-priority VMs and uses log lifetime as a bounded tie breaker:

```text
u_i = priority_weight_i * (0.75 + 0.25 * normalized_log_lifetime_i)
priority_weight = 4 for high priority, 1 for low priority.
```

Right-censored lifetimes are assigned the documented 90-day anonymization cap.
This makes the experiment explicitly offline; it must not be interpreted as an
online policy with perfect lifetime knowledge.

The authoritative constraints are

```text
sum_i cpu_i * x_i <= capacity
sum_i memory_i * x_i <= capacity
sum_i ssd_i * x_i <= capacity
sum_i nic_i * x_i <= capacity.
```

Pairwise edges are added only when two requests alone exceed a capacity.
Several individually compatible requests can still violate a cumulative
constraint. Therefore every candidate passes through a mandatory
capacity-repair layer before critic reranking or execution.

## Methods

- `rydberg_surrogate`: the previously selected standardized, extended-pulse
  classical surrogate, followed by authoritative capacity repair.
- `deterministic_repair`: one all-selected proposal followed by exactly the
  same authoritative capacity repair. This control separates sampler quality
  from repair quality.
- `beam_search`: capacity-aware beam search using the same learned proposal
  score and requested `K`.
- `randomized_greedy`: Gumbel-perturbed greedy packing.
- `random_shooting`: random-order feasible packing.
- `milp`: evaluation reference only.

## Metrics and gates

The report includes best-of-`K` and critic-selected reward/MILP ratios,
epsilon-5% coverage, accepted VM count, mean and peak resource utilization,
raw capacity feasibility, post-repair feasibility, repair removal rate,
candidate diversity, unique candidate count, local proposal latency, MILP
status, and MIP gap.

The pipeline gate requires:

- every MILP reference to complete with zero gap;
- post-repair feasibility equal to 1;
- the lower 95% confidence bound of best-of-16 at 100 requests and full
  capacity to be at least 0.90; and
- the corresponding critic-selected lower bound to be at least 0.90.

Passing these gates would establish a trace-driven algorithmic pipeline result,
not a hardware or quantum-advantage claim.

Sampler contribution is reported separately. It requires a positive paired
lower confidence bound over deterministic repair and at least 10% raw capacity
feasibility at the 100-request, full-capacity, `K=16` target. This prevents a
strong repair layer from being misattributed to the proposal sampler.

## Reproduction

Download and extract the official trace to:

```text
datasets/azure_packing/raw/packing_trace_zone_a_v1.sqlite
```

Then run:

```powershell
$env:PYTHONPATH = ".\hybrid_qrl\src"
.\.venv\Scripts\python.exe -B `
  .\hybrid_qrl\experiments\azure_packing_benchmark.py
```

The raw records and report are written to
`results/azure_packing_benchmark.json` and
`results/azure_packing_benchmark.md`.
