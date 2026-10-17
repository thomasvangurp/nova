# nanobody_opt_v2 — rewritten screen + replication orchestration

A thin rewrite of the SN68 nanobody producer's screen and replication path.
Four modules are rewritten in place; every other live module (`db`, `scorer_daemon`,
`mutation_network`, `rank_exact`, `publication_gates`, the score adapters, and the
on-chain commit path) is imported **by reference** from `nanobody_opt` (v1) and is
untouched. This branch is the *orchestration* change; it is not a full rewrite of
the scorer.

## What changed (vs v1)

| module | change |
|---|---|
| `run_live_producer.py` | new flag defaults, `minimum_keep` plumbing, `--dry-run` output isolation |
| `sequential_exact_race.py` | `minimum_keep` floor in `_run_replication_rounds`, dry-run submit stub |
| `compute_budget.py` | staged-tournament pricing for the replication reserve |
| `select_elite_parents.py` | flag defaults |

Removed from the live loop (dead per import tracing): `elite_parents_from_db.py`,
`children_submit.py`, `children_lane.py`, and all `.bak.*` / `.pre-*` /
`audit_*` / `diag_*` / `backtest_*` files.

## New argparse defaults

| flag | default | note |
|---|---|---|
| `--candidate-limit` | **4800** | screen pool |
| `--finalists` | **150** | first-cut width; a ceiling the planner can lower |
| `--replication-rounds` | **8** | 8× = ranking convergence (0.988) |
| `--replication-rounds-per-stage` | **1** | progressive, not flat |
| `--replication-keep-fraction` | **0.35** | 150→52→floor(20) prune chain |
| `--replication-minimum-keep` | **20** | **new** — never prune below 20 survivors |
| `--parent-limit` / `--neighbor-parent-limit` / `--basin-parent-limit` / `--proposal-limit` | 64 / 32 / 32 / 384 | unchanged |
| `--reserve-blocks` | **36** | Pi commits at ~325 of 361; 22 if pushed to 339 |
| `--dry-run` | off | **new** — isolates output under `/root/dryrun/`, never commits |

`gpu_seconds_per_candidate` in the timing model is **4.8** (= ~0.60 s/obs × 8 GPUs),
replacing the stale 7.5.

## Why these numbers (empirical, 2026-10-16)

- **Single-obs ranking is heavy-tailed, but the screen winner is protected.** On the
  broad screen pool the true #1's single-draw p95 rank is top ~0.7% (~rank 33 of 4800),
  so a first cut at ~150 retains the winner with >95% confidence. Measuring on the
  narrow *finalist* basin (all-good sequences) overstated displacement by ~2 orders of
  magnitude — that was the trap the first cut-to-400 estimate fell into.
- **Progressive deepening, not flat rounds.** The depth curve knees at ~3 obs and
  converges at ~8; the tournament concentrates the budget on the top-150 instead of
  flat 20-round scoring. `minimum_keep=20` guarantees a family slate is always held.
- **Commit timing is Pi's decision, not a latency.** Pi (the combined submit host)
  commits at ~325 blocks; `commit_reserve = 16 blackout + 5/UID`, so it scales to 13
  UIDs (81 blocks) — the true bounding constraint on the screen window.

## Tests

```
PYTHONPATH=/root:/root/bgen_shim:/root/fastscore/release/boltzgen_src \
  /root/validator_probe/parity25747/venv/bin/python -m unittest discover -s tests
```
19 tests, mocked (no GPU/DB): prune floor, staged schedule, budget window, flag round-trips.

## Dry run

```
bash dry_run.sh --once            # one generate-and-screen iteration, no commit
```

Box4 (dev) required, in addition to the parity venv: a synthetic `chain_state.json`,
a `pixi` shim to the parity venv, the `metanano` package + `/root/utils`, IgBLAST
(binary + `internal_data` + `database` + `ncbi-igblast` shared libs), and HMMER
(`hmmscan` + MPI libs). With those in place the producer runs the full preparation
flow (archive → `build_top50_index` → `select_elite_parents` → `competitor_lane` →
IgBLAST layer-2, 62/64 pass). The final TNP developability gate (`theraprofnano`) was
not installed on the dev box, so the live screen/replication pass was not exercised
end-to-end there — it is covered by the unit tests.

## Not yet done

- Deploy to the master (production launch config still has the old 2800/300/20/0.65 values).
- Validate the TNP developability gate + real screen/replication on a box with `theraprofnano`.
- Confirm the `--reserve-blocks` floor against the real 13-UID commit window.
