# Dynamic dispatch benchmark

## Research question

The benchmark tests whether a blockade-constrained candidate sampler retains
useful reward, feasible-sample probability, and candidate diversity as the
binary action dimension grows from 20 to 100. It compares that trend against
strong classical proposal mechanisms under both equal candidate count and
equal end-to-end decision-time budgets.

An exponentially large unrestricted action space is only motivation. It is
not evidence of quantum advantage. A favorable result would require a better
quality-versus-latency scaling trend than the relevant classical baselines.

## Scheduling process

Each node represents one pending job and one binary accept/defer decision.
Selecting a set of pairwise non-conflicting jobs completes them. Deferred jobs
age, become more urgent, and eventually incur a missed-deadline penalty.
Completed and expired jobs are replaced, so current actions affect future
states and episode return.

The six node features are:

1. job value;
2. normalized age;
3. inverse remaining time (urgency);
4. value multiplied by urgency;
5. normalized conflict degree; and
6. a bias feature.

Random unit-disk and jittered-grid positions generate pairwise conflicts using
a blockade radius. These are hardware-compatible graph families, but the
application graph remains the authoritative constraint. Geometry error creates
a separate physical graph; every measured proposal is repaired and validated
against the true application graph before execution.

Small-backend calibration additionally normalizes each register to unit minimum
atom separation. This prevents unphysical near-coincident random coordinates
from producing numerically extreme `C6/r^6` interactions.

## Learned model

`LinearAutoregressiveActor` shares one feature-weight vector across all nodes,
so a model trained at 20–60 jobs can be evaluated at 100 jobs. It visits nodes
in randomized order, samples Bernoulli decisions, and masks neighbors after a
selection. This is a learned feasible autoregressive policy.

Training uses Monte Carlo actor-critic updates from dispatch reward only:

- a state-value critic supplies the policy-gradient baseline;
- an action-value critic learns discounted returns for critic best-of-K
  reranking; and
- neither critic nor actor receives MILP, greedy, or teacher action labels.

The implementation is intentionally NumPy-only for auditability. It establishes
that the policy is genuinely reward-trained, but it is not meant to claim that
a linear actor is competitive with a tuned PPO/GNN production scheduler.

## Candidate methods

All methods receive priorities derived from the frozen learned actor. Feasible
candidates are deduplicated and reranked by the frozen learned Q critic.

| Method | Implementation |
|---|---|
| MILP | SciPy/HiGHS binary optimization; repeated no-good cuts request distinct candidates and a wall-clock limit is enforced. |
| Simulated annealing | Temperature-decayed add/remove/swap moves on feasible independent sets. |
| MCMC | Blockade-aware Gibbs sweeps with burn-in and retained samples. |
| Local search | Randomized greedy starts followed by improving add/single-swap moves. |
| Beam search | Include/exclude expansion in priority order with bounded beam width. |
| Greedy | Deterministic first pass plus Gumbel-randomized weighted restarts. |
| Autoregressive | Direct samples from the learned graph-masked actor. |
| Rydberg surrogate | Classical annealed blockade dynamics using the perturbed physical graph, pulse schedule, readout flips, and rounded cached weights. |

The Rydberg surrogate does not simulate a quantum state and is not a hardware
runtime. Its role is to test algorithmic sensitivity at sizes where dense
statevector and QuTiP evolution are intractable. Small-instance calibration to
a real backend is a separate experiment.

## Protocol

The default run uses 20 independently generated held-out seeds for every cell.
The primary scale study evaluates `n = 20, 40, 60, 100` at target edge density
0.12 in two modes:

- equal K: at most 16 raw proposals per method; and
- equal local latency: 20 ms software-pipeline target, with generation capped
  at 10 ms and 96 retained raw proposals so repair and Q reranking remain
  inside the target.

This target covers local proposal generation, repair, deduplication, learned-Q
reranking, and Python overhead. It is not a neutral-atom QPU deadline and does
not include cloud submission, queueing, atom preparation, physical shots,
measurement, or result retrieval.

The dynamic evaluation runs 12 dispatch steps for every method and seed. A
one-factor-at-a-time robustness study varies:

- graph density: 0.05, 0.12, 0.25;
- relative position error: 0, 0.03, 0.08 blockade radii;
- readout bit-flip probability: 0, 0.01, 0.05;
- uniform, lognormal, and bimodal job utility;
- short, balanced, and adiabatic surrogate pulse schedules;
- no rounding and 0, 1, 2, or 4 decimal cache precision; and
- grid, dense-grid, and combined geometry/utility distribution shifts.

OFAT isolates sensitivity but does not estimate high-order interactions. A
follow-up full or fractional factorial design is needed if interactions are a
primary research question.

## Metrics and reference

The per-state MILP reference directly maximizes realized immediate dispatch
reward, including avoided deadline penalties. The JSON stores solver status,
MIP gap, and latency. A reference is described as exact only when HiGHS reports
successful completion with zero gap.

Reported metrics include reward/reference ratio, normalized regret, exact
coverage, raw feasible rate, number and Hamming diversity of unique repaired
candidates, fallback, Q value, and proposal/end-to-end latency. A latency trial
is compliant only if measured end-to-end time is at most 105% of the target.

## Reproduction

From the repository root:

```powershell
$env:PYTHONPATH = ".\hybrid_qrl\src"
.\.venv\Scripts\python.exe `
  .\hybrid_qrl\experiments\dispatch.py scaling `
  --seeds 20 --train-episodes 320 --k 16 --latency-ms 20
```

Outputs:

- `results/dispatch_benchmark_results.json`: configuration, learned weights,
  training curves, and all per-trial records;
- `results/dispatch_benchmark_report.md`: aggregate means and paired-seed 95%
  confidence intervals.

## Current limitations

- The workload is synthetic rather than a parsed Alibaba Cluster Trace split.
- Unit-disk edges encode pairwise exclusion only. CPU, memory, and accelerator
  capacities require cumulative constraints in an extended safety layer.
- Python CPU wall time is suitable for within-machine comparisons, not neutral
  atom hardware latency claims.
- Environment steps have no assigned physical duration, and recorded actions
  are executed immediately. Hardware-latency claims require a configured
  decision interval plus delayed-action/stale-state evaluation.
- The action critic is linear and may mis-rank candidates under distribution
  shift; this is part of the measured end-to-end method, not an oracle critic.
- Hardware validation still requires mapping pulse schedules and geometry to
  the downloaded neutral-atom backend on small calibratable instances.

The follow-up implementation in `docs/latency_aware_dispatch.md` assigns a
physical duration to each step, consumes timestamped task-latency traces, and
evaluates delayed execution plus asynchronous classical fallback without
changing the original benchmark records.
