# Dispatch generalization and candidate-budget stress test

## Purpose

This test extends the fixed-`K` dispatch study without changing the trained
model or selecting new sampler hyperparameters. It asks whether the improved
Rydberg surrogate continues to place high-reward actions in its candidate
batch when the action dimension, candidate budget, constraint pressure, and
held-out data distribution change.

The study is an algorithmic surrogate evaluation. It does not replace the
dense, QuTiP, manual-backend, measured-latency, or hardware calibration gates.

## Frozen inputs

- Actor and critics: `trained_model` from
  `results/stable_backlog_scaling_results.json`.
- State warm-up actor: frozen baseline model from
  `results/dispatch_benchmark_results.json`, matching the previous
  confirmation protocol.
- Sampler regime: the previously selected
  `standardized-050-extended` encoding and pulse schedule.
- Comparator: beam search with the same requested candidate count.
- Reward reference: a paired per-state MILP solution. Exactness is claimed
  only when HiGHS completes with zero MIP gap.
- Test seeds: 20 held-out paired seeds per experimental cell.
- Near-optimal threshold: epsilon = 5%.

No model training, sampler selection, or threshold tuning is permitted on the
new test records.

## Test matrix

### A. Job count and best-of-K scaling

- Jobs: 20, 40, 60, 80, and 100.
- Candidate budgets: 1, 4, 8, 16, 32, and 64.
- Environment: unit-disk graph, density 0.12, uniform independent utilities,
  and default deadlines 3--12 steps.
- Methods: scale-aware Rydberg surrogate and beam search.

This matrix measures whether increasing `K` raises best-of-`K` quality and
epsilon-optimal coverage, and how many candidates are required as the binary
action dimension grows.

### B. Constraint-pressure scaling

- Jobs: 40 and 100.
- Conflict densities: 0.05, 0.12, 0.25, and 0.40.
- Deadline profiles: default 3--12 steps and tight 2--5 steps.
- Candidate budget: 16.
- Environment: unit-disk graph with uniform independent utilities.
- Methods: scale-aware Rydberg surrogate and beam search.

Higher edge density enlarges the pairwise-conflict constraint set. The tight
deadline profile separately increases urgency and the cost of omitting jobs.

### C. Held-out dataset shifts

All settings use 60 jobs and `K=16`:

| Setting | Graph | Density | Utility distribution | Utility correlation |
|---|---:|---:|---:|---:|
| in_distribution | unit_disk | 0.12 | uniform | none |
| grid | grid | 0.12 | uniform | none |
| clustered | clustered | 0.12 | uniform | none |
| lognormal | unit_disk | 0.12 | lognormal | none |
| bimodal | unit_disk | 0.12 | bimodal | none |
| spatial_correlation | unit_disk | 0.12 | uniform | spatial |
| degree_correlation | unit_disk | 0.12 | uniform | degree |
| combined_shift | clustered | 0.25 | bimodal | degree |

## Metrics

For every paired state and method, record:

- best-of-`K` reward divided by the MILP reward;
- critic-selected and utility-head-selected reward ratios;
- probability and batch coverage of an epsilon-5% action;
- raw feasibility and post-repair feasibility;
- unique feasible candidates and pairwise Hamming diversity;
- proposal latency and selected action size; and
- MILP status, MIP gap, exactness, and solve latency.

Means are reported with two-sided normal-approximation 95% confidence
intervals across held-out seeds.

## Preregistered interpretation

- The best-of-`K` curve is a candidate-generation diagnostic, not deployed
  policy performance. The critic-selected curve is the deployable one-step
  result.
- A nominal threshold is met when the mean ratio is at least 0.90. It is
  statistically passed only when the lower 95% confidence bound is at least
  0.90.
- Post-repair feasibility must equal 1.0.
- Any cell with a non-exact MILP reference is identified and excluded from an
  exact-optimum claim, while its incumbent-relative ratio remains auditable.
- Larger `K` is useful only if quality or coverage improves enough to justify
  the additional samples and latency.

### Reporting amendment discovered during execution

The first full execution showed that dense graphs combined with tight
deadlines can make even the optimal immediate MILP reward non-positive because
some miss penalties are unavoidable. A multiplicative `reward / MILP` ratio
is undefined in those states. Those stress cells are retained, not removed:
the multiplicative ratio is reported as `n/a`, and an additional opportunity
score is reported as

```text
(candidate reward - empty-action reward)
-------------------------------------------------
(MILP reward - empty-action reward)
```

This maps the empty action to zero and the MILP action to one. The amendment
changes only invalid-ratio handling; it does not alter the model, sampler,
test cells, seeds, or pass thresholds.

## Reproduction

From the repository root:

```powershell
$env:PYTHONPATH = ".\hybrid_qrl\src"
python .\hybrid_qrl\experiments\dispatch.py generalization
```

The command writes raw JSON records and a Markdown result report under
`results/`.
