# Scale-aware stable-backlog study

## Purpose

This study addresses the two bottlenecks exposed by the earlier dispatch work:

1. fixed-K Rydberg-surrogate proposal quality declined as the number of jobs
   increased; and
2. current-state quantum candidates arrived too late to influence dynamic
   dispatch.

It tests a size-stable utility encoding at the unchanged candidate budget
`K=16`, then moves delayed proposals onto a reserved, long-lived backlog. The
study is implemented in
`src/hybrid_qrl/dispatch/backlog_benchmark.py` and executed by
`experiments/stable_backlog_scaling.py`.

## Claim boundary

The 20--100 decision sampler is a classical stochastic blockade surrogate. The
new result is evidence about an algorithm and control architecture, not about
neutral-atom hardware speed or fidelity. Physical advancement still requires:

- distribution transfer to dense and QuTiP evolution;
- acceptable quality from the geometry-driven manual backend;
- timestamped latency from a real QPU service; and
- a repeated held-out run using those physical inputs.

## Multi-size reward training

The utility actor is retrained for 800 episodes on sizes 20, 40, 60, 80, and
100, densities
0.08, 0.12, and 0.18, and three geometric graph families. Training uses only
environment trajectories and rewards. MILP, beam search, and teacher actions
are not used as policy targets.

This separates actor generalization from the sampler encoding. Confirmation
therefore reports:

- `legacy_frozen`: the original model and original detuning map;
- `multi_size_legacy`: the multi-size actor with the original detuning map;
- `scale_aware`: the multi-size actor with the selected detuning map; and
- `beam_search`: a strong classical comparator at the same `K`.

## Standardized local detuning

The original scalable surrogate used mean normalization:

\[
\tilde u_i = \frac{u_i}{\bar u}.
\]

As size changes, this can leave too little contrast between local detunings.
The new candidate map is

\[
z_i=\frac{u_i-\bar u}{\sigma_u},\qquad
\tilde u_i=
\frac{\exp(gz_i)}{\operatorname{mean}_j\exp(gz_j)}.
\]

The gain `g` and pulse schedule are selected using a disjoint selection split.
The default grid compares the original encoding with standardized gains 0.25,
0.50, 0.75, and 1.00. The selected extended schedule uses 16 surrogate sweeps,
maximum inverse temperature 10, and a detuning sweep from 1.30 to 0.60.

These values are not silently treated as physical pulse parameters. They form
a hypothesis that must be translated and recalibrated on the small-system
backends.

## Stable-backlog policy

One environment step is assigned 1,000 ms. A future result has a 6,000 ms
deadline. With a one-step guard, a job is eligible only when

\[
r_i \ge
\left\lceil\frac{6000}{1000}\right\rceil + 1 + 1 = 8
\]

steps remain.

Reserving every eligible job caused avoidable return loss in the development
run. The final policy reserves the highest-utility 25% of all current jobs,
capped at 20 jobs and subject to a minimum of eight. This is a bounded quantum
target block inside the larger dispatch state.

While the request is pending:

1. beam search immediately serves all unreserved jobs;
2. persistent IDs protect the target block from slot replacement;
3. the candidate batch is evaluated for the predicted arrival state;
4. late results are dropped at the preregistered deadline; and
5. on-time results are remapped, merged with the fallback action, repaired
   against the full application graph, and reranked by the learned critic.

`future_beam` receives the same reservation, delay, `K`, and arrival handling.
It distinguishes a benefit of backlog partitioning from a benefit of the
Rydberg proposal distribution.

## Metrics and gates

The confirmation report includes:

- best-of-16 reward ratio, a diagnostic proposal upper bound;
- learned-critic and utility-head selected ratios;
- epsilon-5% coverage and per-candidate probability;
- raw and post-repair feasibility;
- identity survival and result utilization;
- reward/reference ratio, missed value, shots, and fallback latency; and
- deadline compliance from the complete queue/execution/retrieval trace.

The strict algorithmic gate requires the lower 95% confidence bounds of both
best-of-16 and learned-critic ratios to reach 0.90 at every tested size. The
asynchronous gate applies the same lower-bound rule to 0.90 return, 0.80
identity survival, and 0.10 utilization, plus perfect post-repair feasibility.
The physical gate additionally requires 95% deadline compliance, a measured-
QPU trace, and the retained calibration-transfer gates.

## Reproduction

From the repository root:

```powershell
python experiments/stable_backlog_scaling.py
```

To use timestamped hardware observations:

```powershell
python experiments/stable_backlog_scaling.py `
  --latency-trace path\to\measured_qpu_latency.json
```

Selection and confirmation seeds are disjoint, and increasing `K` is not part
of this experiment.

## Current result

The selected `standardized-050-extended` regime increased the held-out
best-of-16 ratio at 100 jobs from 0.683 to 0.902 without increasing `K`. The
nominal mean threshold passed, but the 95% interval was 0.902 +/- 0.015, so its
lower bound did not establish the strict 0.90 gate. The learned critic reached
only 0.771 at 100 jobs, and the combined algorithmic gate remains on hold.

The bounded stable backlog passed its software asynchronous gates. At 40 and
100 jobs, scale-aware return ratios were 0.932 and 0.962, identity survival was
1.0, result utilization was 0.958 and 0.882, and every returned action was safe
after repair. The synthetic trace met the six-second deadline 98.0% of the
time, but it cannot satisfy the measured-QPU gate.

The next algorithmic priority is therefore no longer the best-of-16 candidate
pool alone. It is a scale-aware critic that can recover the high-value sample
already present in that pool, followed by small-system physical calibration of
the standardized detuning map.
