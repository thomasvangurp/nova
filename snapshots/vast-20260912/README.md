# vast.ai instance 50420868 — final state, 2026-09-12

The 10×4090 box that ran the live nanobody miner for 21 days, captured minutes
before it was destroyed. Everything here was outside version control at the time.

Restore is `gunzip` / `tar xzf`; nothing needs a running instance.

| archive | expands to | what it is |
|---|---|---|
| `nanobody_mutations.sqlite.gz` | 147,714,048 B | the accumulated mutation search: 19,266 nodes, 29,931 observations, 134,669 edges, 7 tables (`nodes`, `observations`, `edges`, `mutation_effects`, `proposal_scores`, `lineage_edges`, `metadata`). `integrity_check ok`; sha256 of the expanded file is `54bbed5281e65b66282786c39f111128b9c9c5f5e6c6b5f7330e20822ea2c59c`. This is search memory accumulated across ~30 epochs and is the one artifact here that cannot be regenerated at any price. |
| `critical-state.tgz` | 13 KB | the publication ledger (36 sequences — required for the novelty gate, so losing it means republishing things we have already submitted), publisher state, re-derived strict gates, the fast-scorer pin, and the per-epoch history directory. |
| `rankings.tgz` | 159 KB | 38 per-epoch `*_rankings.json` — what the race actually chose, epoch by epoch, and the only surviving record of the ranked fields behind those choices. |
| `fastscore-pinned.tgz` | 30 MB | the complete pinned scorer package, including the ~27 MB of validation output structures (`pp24_s`, `fin_s`, `cap_structs`) deliberately excluded from `external_tools/fast_nanobody_scorer/`. Keep this if you ever need to re-derive the equivalence numbers rather than cite them. |
| `fastscore-src.tgz` | 415 MB tree, source only | the whole scorer working directory: benchmark harnesses, sweep analyses and comparison scripts that were never promoted into the repository. Useful as a record of what was tried and rejected — batching at 1.00×, multi-worker GPU packing at 1.21× — not as something to run. |

The scorer itself is not only here: it is committed at
`external_tools/fast_nanobody_scorer/`, byte-identical at sha256
`530f5dcbc54d…`. See its `PROVENANCE.md`.

## Why a snapshot is in git at all

Version control is for source, and binary snapshots in a repository are usually
a mistake. The exception earned its place: three separate pieces of load-bearing
state lived only on rented hardware — the live publisher's gate constants, the
fast scorer, and this database — and two of them were nearly lost, one on the
day this was written. 28 MB against that is cheap.

What is deliberately absent is the 11 GB of `/root/live` epoch structures. Those
are intermediate folding output: large, regenerable, and evidence of runs rather
than inputs to them.
