# Latency-aware dynamic dispatch extension

## Purpose and claim boundary

This experiment adds physical time, delayed execution, stale-state repair, and
asynchronous fallback to the synthetic dispatch benchmark. It does not treat
emulator runtime as neutral-atom hardware latency.

The default run uses a deterministic preregistered stress trace to exercise the
control architecture. Because that trace has `source_kind: synthetic_stress`,
the measured-QPU gate must fail. A physical latency claim requires timestamped
observations from an actual QPU service and `source_kind: measured_qpu`.

## Preregistered protocol

The default held-out study fixes these settings before evaluation:

| Setting | Default |
|---|---:|
| physical duration per environment step | 1,000 ms |
| quantum result deadline | 3,000 ms |
| pending decisions | 40 |
| horizon | 18 steps |
| candidate budget | 16 |
| held-out seeds | 20 |
| required deadline compliance | 95% |
| required reward/reference ratio | 0.90 |
| minimum quantum-result utilization | 5% |

The physical duration is an explicit scenario assumption for the synthetic
environment. It is not inferred from Python runtime.

## Timestamped latency trace

A measured JSON trace has this structure:

```json
{
  "schema_version": 1,
  "source_kind": "measured_qpu",
  "source_name": "experiment-name",
  "device": "provider/device identifier",
  "observations": [
    {
      "request_id": "task-0001",
      "submitted_at_ms": 0.0,
      "started_at_ms": 1250.0,
      "completed_at_ms": 1810.0,
      "retrieved_at_ms": 1925.0,
      "shots": 100
    }
  ]
}
```

Timestamps must be monotonic. The benchmark derives:

- queue latency: `started - submitted`;
- provider execution: `completed - started`;
- retrieval latency: `retrieved - completed`; and
- end-to-end latency: `retrieved - submitted`.

It reports the mean, p50, p95, and p99 where applicable, rather than replacing
the distribution with emulator wall time.

## Persistent identities and stale actions

Each dynamic job now receives a monotonically increasing `job_id`. A quantum
request stores selected job identities rather than slot numbers. At arrival:

1. identities that have completed or expired are dropped;
2. surviving identities are mapped to their current slots;
3. application-graph repair runs unconditionally;
4. feasibility is checked again; and
5. the critic reranks the repaired candidates against the current state.

This prevents a stale bit from accidentally selecting an unrelated replacement
job that later occupies the same array index.

## Policies

The study evaluates five paired policies:

| Policy | Behaviour |
|---|---|
| `beam_immediate` | immediate beam search only |
| `greedy_immediate` | immediate randomized greedy only |
| `quantum_delayed` | no-op while waiting, then execute a repaired result |
| `async_beam_quantum` | beam fallback now; use a better quantum result later |
| `async_greedy_quantum` | greedy fallback now; use a better quantum result later |

Only one quantum request is in flight per policy. Requests that cannot resolve
before the finite evaluation horizon are not issued, preventing horizon
censoring from being counted as deadline failure.

## Metrics

The raw JSON retains per-seed values for:

- episode return and reward/reference ratio;
- missed value and epsilon-5% selected-action hit rate;
- per-candidate epsilon probability, batch coverage, and K95;
- requests, deadline misses, arrivals, and results actually used;
- raw generation feasibility, stale raw feasibility, post-repair feasibility,
  and repair change rate;
- mean stale steps, fallback latency, p95/p99 observed task latency, and p95
  queue latency; and
- total shots and shots per request.

`quantum_result_utilization` is the fraction of issued requests whose arrived
candidate is ultimately executed. It is stricter than arrival rate: an arrived
candidate that loses critic reranking to the fallback is not counted as used.

## Gate composition

The physical gate cannot bypass the conditional-advantage study. It retains:

1. acceptable reward ratio;
2. competitive epsilon-optimal coverage;
3. surrogate-to-dense/QuTiP calibration transfer; and
4. manual geometry-backend quality.

It additionally requires measured-QPU timestamps, at least 95% deadline
compliance, latency-aware return of at least 0.90, post-repair safety of 1.0,
and at least 5% quantum-result utilization. All gates must pass simultaneously.

## Reproduction

Run the preregistered stress experiment from the workspace root:

```powershell
$env:PYTHONPATH = ".\hybrid_qrl\src"
.\.venv\Scripts\python.exe -B `
  .\hybrid_qrl\experiments\dispatch.py latency
```

Run the unchanged protocol with measured QPU timestamps:

```powershell
$env:PYTHONPATH = ".\hybrid_qrl\src"
.\.venv\Scripts\python.exe -B `
  .\hybrid_qrl\experiments\dispatch.py latency `
  --latency-trace .\path\to\measured_qpu_latency.json
```

Default outputs are:

- `results/dispatch_latency_stress_trace.json`;
- `results/latency_aware_dispatch_results.json`; and
- `results/latency_aware_dispatch_report.md`.
