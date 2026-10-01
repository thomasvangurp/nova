# Claude handoff — molecule strategy, September 18, 2026 (evening)

Supersedes the morning handoff for strategy conclusions; that file remains correct
for operational/credential rules, the NIM recipe and the failed rank-parity receipt.
Everything below is offline. No live submission, no on-chain publication.

## Instance state — STOPPED, disk retained, EVIDENCE NOT COLLECTED

Vast instance **51335750** (`nova-nim19-epoch-offline-8gpu-20260917`) was stopped by
the lease watchdog at **2026-09-18 14:58:36 UTC**, `cur_state=stopped`. Disk retained
(~$0.0277/hr). Do not destroy it.

**Trial 2's receipts and archive were never collected locally.** They remain at
`/root/nim19-test/rowcolumn-trial-20260918/` on the retained disk: all per-round JSON,
`round-N-lane.json`, `final.json`, `final_records`, native validation and
`candidates.sqlite` (~1,570 self-labels). The user chose to leave them there rather
than restart to collect. Only the headline numbers below were captured locally, via a
poll of `progress.json` before the stop.

Lease `lease_safety_seconds` was raised 71457 -> 73857 (cutoff 14:18:36 -> 14:58:36)
with explicit user authorization, recorded in `data/nim19_epoch/20260917/instance.json`.
This extension was **necessary**: trial 2 ran 3144s from 13:38:45 and finished at
14:31:09, past the original cutoff. Do not revert it.

## Results

Epoch 25176, reaction 3, P40261. All reference-rescored on the pinned Pixi/Boltz worker.

| Run | Scores | Elapsed | Ours | Matched leader | Gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| old 6 x 384 (baseline) | 2304 | 2826.8s | 1.4364645260 | 1.9266314308 | 0.4901669 |
| trial 2 (coverage+sweep) | ~1570 | 3144.4s | **1.4641881708** | 1.9169221535 | **0.4527340** |

Public leader quote 1.8894035980. Trial 2: **+0.0277 over baseline on ~68% of the
exact scores**, closing 7.6% of the gap. Six rounds, within the 3600s budget.

**Trial 1** (adaptive proxy + contextual rxn3, isolated, fresh archive) was stopped by
user request at r5 before its final rescore, so it has **no canonical score**. Its
search-only NIM trajectory was ahead of baseline throughout (r3 +0.0144, r4 +0.0298,
r5 +0.0102) on ~1528 scores vs 1920. Collected to
`data/nim19_epoch/20260917/strategy-trial-20260918/` (16MB, includes archive).

Per-round NIM sum20 (search-only, NOT comparable to canonical):

| r | old 6x384 | trial 1 | trial 2 |
| --- | ---: | ---: | ---: |
| 1 | 1.233775 (384) | 1.084198 (288) | 0.943149 (288) |
| 2 | 1.295990 (768) | 1.288937 (584) | 1.163455 (544) |
| 3 | 1.350741 (1152) | 1.365145 (872) | 1.369133 (792) |
| 4 | 1.372449 (1536) | 1.402216 (1184) | 1.413859 (1048) |

## The diagnosis that actually matters

### 1. The gap decomposes into three comparable factors

From the old run's reference scores vs the matched leader portfolio:

```
ours    sum20 1.436465  mean prob 0.3759  mean aff -2.6063  mean heavy 41.50
leader  sum20 1.926631  mean prob 0.8012  mean aff -2.8507  mean heavy 37.95

counterfactual, swapping one factor into our portfolio:
  + leader heavy atoms   1.5716  (+0.1344)  27% of gap
  + leader binding prob  1.6422  (+0.2050)  42% of gap
  + leader affinity      1.5550  (+0.1178)  24% of gap
  all three              1.9246  ~= actual leader
```

### 2. Reachability was the blocker (now fixed)

The winner put role-A 65401 in 16 of 20 molecules and concentrated role C on two
reagents (12 of 16): fix an (A,C) block, vary role B. Our portfolio used 17 distinct
role-A reagents, max reuse two.

Their reagents are effectively absent from the frozen public history:
65401 x1, 74710 x2, and **1174 / 6430 / 32427 / 88530 / 100584 zero occurrences**.
Because the baseline proposal policy draws components from history-score-weighted
frequencies, we proposed them **0-1 times in 24,768 constructed candidates** and
exactly scored them **zero times**. Role C has 18,330 valid reagents; history covers
2,125 (12%). Role space is 817 x 2,731 x 18,330 ~= 4.1e10.

Our 2304 exact scores spread over **1,755 distinct (A,C) blocks — 1.3 cells per block**,
only 31 blocks with >=4 cells. No block-level statistic existed to rank on.

We *did* sample 65401 (67 proposals, 7 exact scores) and it looked bad: mean 0.03934,
max 0.05124, against our top role-A mean of 0.05651. The leader's 16 molecules on that
same anchor scored **mean 0.09678, min 0.08656** (reference). Selecting anchors by
marginal reagent statistics — mean OR max — provably fails: 65401 ranks 81/377 by mean
and 139/377 by max in our own scored pool. **The signal is in the combination, not the
reagent.**

### 3. Ranking splits by regime — this is the key result

A self-labelled chemistry proxy (ExtraTrees on `molecule_features`: 1024-bit Morgan +
11 descriptors, no component target-encodings) trained on our own NIM labels:

```
broad-range pool (trial 2, spans random to good):  0.5527 Spearman, novel 0.4848
                                                   top-10% precision 0.3733 vs random 0.0995
narrow pre-selected pool (trial 1, adaptive-filtered): -0.2387, novel -0.3172
grouped CV by role-A within one run:                0.5040, novel 0.3313
frozen historical proxy, same data:                +0.1085 overall, ~0 within-stratum
```

The negative number is **range restriction**, not a generalization failure: trial 1's
molecules were already filtered into a narrow high-quality band. Both test sets are
~85% novel-reagent, so novelty does not explain the difference.

Operationally:
* **Filtering 250k proposals down to ~300 to score: works** (0.55, 3.75x random).
* **Choosing the final 20 from already-elite candidates: fails** (~0 to negative).

The frozen historical proxy is mostly detecting "contains known-good reagents", not
chemistry quality — its within-stratum Spearman is ~0 or negative in every novelty band.

**More labels measurably help:** 2,304 -> 3,815 labels moved cross-run novel ranking
0.4179 -> 0.4848, learning curve still climbing.

### 4. In-epoch Boltz screening cannot scale

```
full quality (canary): 24 requests, 3 waves of 8, 24.355s -> 1.01 s/molecule, GPU 98-99%
reduced recipe (audit): waves of 8 at 5.372s / 4.885s    -> 0.64 s/molecule
speedup: 1.58x   (NOT the 5-10x the settings imply)
```

Fixed per-request cost (structure prediction, MSA, `max_parallel_samples=1`) dominates,
not sampling steps. A 72-minute epoch on 8 GPUs caps at ~2,800 full / ~4,300 reduced
scores. Note the existing `reduced-nim-audit.json` set `policy_accepted: false` on an
**n=8** audit with full-vs-reduced Spearman swinging 0.4286 / 0.7857 between repeats —
that is not sound evidence either way, but 1.58x makes it moot.

**Implication:** the leader is most plausibly exploiting a densely-labelled *private*
block built across epochs, not generalizing. 24h x 8 GPUs ~= 85,000 labels/day
(~$244/day at $10.16/hr). That, not in-epoch search cleverness, is where the evidence
points.

## Code delivered (local, uncommitted)

* `compound_miner/rowcolumn_strategy.py` — coverage lane (uniform draws over the full
  role space) + (A,C)-block sweep enumerating all 2,731 role-B reagents, shared
  `SweepState` so cursors persist across rounds, `quota_records` / `merge_with_quota`
  spending a scoring quota **inside** the batch. Deployed sha256 `c378b2ca1231...`.
* `tools/nim19_rowcolumn_runner.py`, `tools/nim19_rowcolumn_supervisor.py` — trial 2.
* `tools/nim19_strategy_runner.py`, `tools/nim19_strategy_supervisor.py` — trial 1;
  warmup switched to **pre-epoch history**, never leaderboard molecules.
* `tools/test_rowcolumn_lane.py` — 14 dependency-free tests (rdkit absent locally), all
  passing. The 43-test local regression suite also passes.
* Remote `oss-repo/compound_miner/experimental_search.py` was **rebuilt** from the
  pinned deployed fleet `search.py` (`57c14965...` -> `3cbedd3f...`, 4 changed lines,
  factory hook only, GPU-first novelty preserved), replacing the wrong main-derived
  CPU-novelty copy `3c26844a...`. Baseline `search.py` digest unchanged.

## Known defects in the delivered lane — fix before reusing

1. **One shared quota, ordered by `proxy_mean`.** Sweep rows outrank coverage rows, so
   coverage was crowded out completely: batch share went 222 -> 3 -> 0 -> 9 -> 8.
   **Fix: separate quotas, coverage ordered by `proxy_std`.** As shipped, trial 2 tested
   coverage only in r1 and sweep exploitation in r2-r6 — not the combined lane designed.
2. **Block lock-in.** `blocks` is recomputed each round from the best molecules, but the
   sweep concentrates scoring into those blocks, so they stop changing and the grid
   (block_count x |role_B|) saturates. `sweep_prop` froze at 38,234 from r4 to r5.
   **Fix: force block turnover, or many more blocks.**
3. **Lane receipts store counts and `blocks`, not the tagged component sets**, so
   lane-vs-lane realised-score comparison is only approximate post-hoc.
4. Lane fractions control *proposals*, not batch share. 35% coverage proposals became
   77% of r1's batch, because uniform-random molecules clear the 0.5 novelty gate easily
   while parent-mutations land near historical chemistry and get rejected. Only **1,484
   of 250,000** constructed candidates survived (0.6%). **The novelty gate structurally
   favours random chemistry and fights the exploitation lane.**

## Recommended next steps, in evidence order

1. **Build an offline label bank.** Score novel chemistry continuously between epochs
   and train the chemistry proxy on it. This is the only measured path to the broad-filter
   regime that works (0.55 / 3.75x random). Size the spend against subnet payout first.
2. **Size discipline.** Ours span 30-48 heavy atoms, leader 36-42 (mean 41.50 vs 37.95).
   Worth 27% of the gap, deterministic, needs no model.
3. Fix the two lane defects above and re-run the genuine combined lane.
4. Repeat the range-restriction analysis on a run that is not coverage-heavy before
   trusting the 0.55 figure operationally.

## Process notes

* Stopping a remote trial by scanning `/proc` cmdline **matched the scanner itself** —
  our own `python3 -c` cmdline contained the literals being searched for. The guard
  caught it. Exclude `os.getpid()`/`os.getppid()` when matching.
* The trial-1 watchdog from the previous session was **not** running when this session
  started; a Codex-supervised watcher does not survive its client.
* `watch()` reads its deadline once, so editing the manifest never extends a running
  watcher — a new watcher must be started.
