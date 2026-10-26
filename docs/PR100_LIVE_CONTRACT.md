# PR100 live path contract (worker <-> controller)

Owner of this file: the worker/gpu/selection agent. The controller reads it.
Scope: `nanobody_opt/pr100_live_worker.py`, `pr100_live_gpu.py`,
`pr100_live_selection.py`. Validator-matched semantics (developability
thresholds and order, A3D caps and provenance strings, NanoBodyBuilder2
`--n_threads 1`) are unchanged by anything described here.

## 1. job.json keys

Common:

| key | type | default | meaning |
| --- | --- | --- | --- |
| `phase` | `"sequence" \| "native" \| "score"` | required | which phase this worker runs |
| `challenge` | object | required | re-derived and compared on entry |
| `cpu_slots` | int | required | affinity budget for this worker process |
| `archive_sha256`, `archive_fetched_at` | str, int | required | archive snapshot binding |
| `references`, `native_tools` | path | per phase | bank / pinned tool paths |

Sequence phase:

| key | type | default | meaning |
| --- | --- | --- | --- |
| `proposals` | int | `600` | fresh proposals to generate (was hard-coded 64) |
| `reference_slots` | int | `16` | existing native references pulled from the bank |

Native phase:

| key | type | default | meaning |
| --- | --- | --- | --- |
| `workers` | int | `48` | concurrent native profiles, 1 thread each |
| `max_workers` | int | `64` | hard cap; `workers` is clamped to it |
| `candidate_budget_seconds` | number | `300` | per-candidate native budget, measured **from slot acquisition**, never from enqueue |
| `repeats` | int | `3` | `k`: total measurements for a first-pass survivor |
| `margin` | object | see §2 | repeat-and-margin parameters |
| `allow_marginal` | bool | `true` | publish tier B when tier A is empty (see §4) |

`workers` default is derived from available cores/memory
(~1.4 GB RSS per worker, master has 224 CPU slots and ~160 GiB free);
contention was observed at 32 and is unmeasured above it, so 48 is a
measured-conservative default, not a tuned optimum.

Score phase:

| key | type | default | meaning |
| --- | --- | --- | --- |
| `gpus` | list[int] | required | margin-safe candidates are sharded across these |

The 8-candidate handoff cap is removed; the controller no longer truncates.

## 2. Margin parameters

Tier A ("margin-safe") uses the audit's k-scaled rule
(`nova-reports/pr100-audit-20261026/7_developability_algorithm.md` §2.1),
not a flat threshold:

```
threshold(k) = min(cap - z * sigma / sqrt(k), hard_margin(k))
```

with configurable constants under `margin`:

| key | default | note |
| --- | --- | --- |
| `cap` | `89` | validator `psh_strict_rejection_threshold` |
| `z` | `1.3` | audit confidence multiplier |
| `sigma` | `4.4` | per-measurement PSH noise SD (est. from 24 repeat triples) |
| `min_repeats` | `2` | `k >= 2` required; a single measurement is never tier A |
| `hard_margin` | `{"2": 80, "3": 82}` | hard floor; `k > 3` reuses the k=3 value |

This yields ~80 at k=2 and ~82 at k=3. A flat margin of 85 is **not** the
default: the audit's own probability table puts a 3-repeat mean of 85 near a
coin flip, so 85 would publish candidates the evidence does not support.

Honest limit: `sigma` is estimated from 24 sequences. Treat these as
configurable estimates, not constants of nature.

## 3. Selection semantics

- First native measurement is the cheap filter. A candidate that fails it is
  rejected on that measurement's reason and never repeated.
- A first-pass survivor gets `k - 1` further **independent** measurements.
- Tier A requires: `k >= min_repeats`, **every** non-PSH gate passing on
  **every** repeat per existing `rules.developability_reason` semantics, and
  `mean(PSH) <= threshold(k)`.
- Tier B requires: every gate passing on every repeat, but the mean misses
  the margin.
- The cache is **compute-saving only**. A cached measurement can supply a
  measurement value but can never by itself make a candidate tier A; tier
  assignment always comes from the recorded repeat set.
- A candidate that never acquired a cpu slot within the phase budget is
  `not_run`, not `failed`. 17 of 63 earlier "failures" were queue waits.

## 4. `allow_marginal` default: validator penalty check

Checked in the official validator checkout at
`/tmp/sn68-pr100-official-validator`:

- `neurons/validator/nanobody_validity.py` rejects a submission with a bare
  `continue` at every tier (lines 50, 59-60, 65, 70, 75, 80, 85, 103, 142,
  161, 167, 171, 190, 197, 200, 208, 249, 256, 267). Developability rejection
  (red flags/error) is line 248-249, PSH strict rejection line 257, Aggrescan
  rejection line 266-267. A rejected uid is simply absent from
  `valid_nanobodies_by_uid` (returned line 270+); no score is written.
- `neurons/validator/ranking.py` builds `eligible` only from uids that have a
  non-None, finite `final_nanobody_score` (lines 181-189) and ranks those
  (`ordered`, `winner = ordered[0]`, lines 216-220). A uid with no score is
  not ranked at all.
- Local mirror: `nanobody_opt/rules.py:172` `developability_reason` returns a
  reason string; rejection is exclusion, and the reason vocabulary at
  `rules.py:257-258` carries no penalty semantics.

Finding: **an ineligible or developability-failing entry is unscored, with no
negative weight or penalty.** Since the validator rewards rank 1, an idle slot
forfeits a ticket for nothing. Therefore `allow_marginal` defaults to `true`.

## 5. result.json schema

```jsonc
{
  "challenge": {...}, "target": "...", "epoch": 25957,
  "archive_sha256": "...", "archive_checked_at_unix": 0,
  "status": "running" | "complete",
  "input_count": 0, "completed": 0,
  "workers": 48, "candidate_budget_seconds": 300,
  "repeats": 3, "margin": { "cap": 89, "z": 1.3, "sigma": 4.4,
                            "min_repeats": 2, "hard_margin": {"2": 80, "3": 82},
                            "threshold_at_k": 82.0 },
  "allow_marginal": true,
  "rejection_counts": { "passed": 0, "psh_threshold": 0, "not_run": 0 },
  "not_run": 0, "cache_hits": 0,
  "candidates": [
    {
      "sequence": "...", "sequence_hash": "...",
      "source": "fresh_baseline_proposal", "pr100_annotation": {...},
      "novelty": {...},
      "layer2_passed": true,
      "tier": "A" | "B",
      "measurements": [ {...developability...} ],
      "psh_values": [78.1, 80.4, 84.2],
      "psh_mean": 80.9, "psh_sd": 3.07,
      "repeats": 3, "margin_safe": true,
      "margin_threshold": 82.0,
      "pr100_evidence": { "developability": {...} }
    }
  ]
}
```

Ordering guarantee the controller may rely on: `candidates` is **tier A first
ordered by `psh_mean` ascending, then tier B ordered by `psh_mean`
ascending**. The controller can therefore publish a monotonic prefix and, when
`allow_marginal` is false, stop at the first non-`A` entry.

`pr100_evidence.developability` is the **first** passing measurement, so
existing `rules.validate_evidence` semantics are untouched; the repeat set
lives beside it in `measurements`.

## 6. Funnel order

`annotation / basic rules` -> `native developability (repeat-and-margin)` ->
`archive novelty check`. Novelty runs **only on native passers**, so the
expensive archive search is not spent on candidates that fail developability.
