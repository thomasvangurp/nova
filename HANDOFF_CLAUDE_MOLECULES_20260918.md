# Claude handoff — molecule strategy, September 18, 2026

Written around **2026-09-18 12:38 UTC**. Everything below is offline. No live
submission or production deployment was made. This handoff and the experimental
code are **local, uncommitted, and not pushed**.

## User's current request — NOT six epochs

The user corrected the request to **one historical epoch**, focusing on algorithm
work. Demonstrate a competitive, validator-valid molecule portfolio **within
72 minutes**, rather than merely proving throughput. Use the existing eight-GPU
NIM instance and the validated Pixi/reference workflow. The user also asked
whether a row/column-like approach was included.

**The fresh adaptive six-round competitive trial has NOT started.** Only replay,
unit checks, a revival canary, and a deliberately warm-started integration smoke
test completed. Do not mistake the old `epoch/final.json` for a new result.

I said I would add a bounded explicit row/column scoring lane inside the existing
budget. **That lane is NOT implemented.** The user requested this handoff before
I made further algorithm changes. Existing contextual mutations are only a
partial row/column analogue, not systematic sweeps or matrix completion.

## Immediate operational state / money

- Only authorized test host: **Vast instance 51335750**, label
  `nova-nim19-epoch-offline-8gpu-20260917`, **8 × RTX PRO 6000 WS Blackwell,
  96 GB each**. Provider and direct-SSH check confirmed running/running and NIM
  readiness HTTP 200 around 12:37 UTC today. It is presently idle.
- Approximate provider prices last checked: GPU $10.133333/hour, total with disk
  **$10.161111/hour**. Stopped disk: **$0.0277778/hour**, about $0.67/day or
  $20/30 days, provisioned 300 GB, reported usage 82 GB. Stopping retains disk;
  destroying would delete the remaining remote evidence, so do not destroy it.
- **Hard lease cutoff: 2026-09-18 14:18:36.945604 UTC.** Local manifest:
  `data/nim19_epoch/20260917/instance.json`. Deadline is
  `created_unix + lease_safety_seconds`; the latter is currently 71457.
- Local watchdog `python3 tools/nim19_epoch_control.py watch` was still running
  when polled, Codex exec session **44065**. Do not assume a Codex session remains
  supervised after this client exits; Claude should ensure equivalent bounded
  supervision. The watch function reads its deadline **once**, so editing the
  manifest does not extend an already running watcher. No extension is needed
  for one trial if started promptly. Never silently extend the rental.
- Stop the test host after collecting terminal evidence, or promptly if no test
  will run. `nim19_epoch_control.stop()` fresh-checks the exact ID/label/GPU count
  and issues a provider stop; it does not destroy disk.
- **Do not touch production 51125470 (`nova-nanobody-pixi`, 4 × 4090).** Do not
  revive older NIM instances 51321895 or 51332541, rent additional hosts, or
  publish on-chain.

Credential isolation is mandatory. Only opaque locators: Keychain services
`vast.ai` and `NGC_API_KEY`; SSH identity path
`/Users/tvgdeena/.ssh/id_ed25519`. Retrieve inside the process, never print values,
raw account responses, headers, arbitrary subprocess output, environment,
credential files or traceback details. Capture and allowlist both output streams.
Do not read `nova.env`, `.env`, wallets, key contents, registry config, or browser
state. Existing provider helper `nim19_seed_control.api()` retrieves Keychain
credentials in-process; callers must print only explicitly safe fields.

Use **direct SSH**, not Vast gateways:
`nim19_epoch_control.instance(fresh=True)` obtains the current guarded public
route and `ssh(host)` uses `-F /dev/null`, `ProxyCommand=none`, `ProxyJump=none`,
and the mapped `22/tcp` port. Do not trust old cached route data after a restart.

## Repositories and source isolation

Main: `/Users/tvgdeena/nova-nanobody`, branch `nanobody-opt`, HEAD
`f79c4032e8e9dcc103c964fc27e75974e6a3e947` (Cut strict pose ranking overhead).
Tracked main change from this work: `compound_miner/search.py`; the new strategy,
NIM tools, data, playbook, and handoff are mostly untracked. Preserve all changes.

Fleet: `/Users/tvgdeena/nova-compound-fleet`, branch `feature/compound-fleet-10`.
This tree is **very dirty with pre-existing user work**. Our only tracked edit
there was the same four-line optional policy-factory hook in
`compound_miner/search.py`. Do not reset, checkout, copy the whole tree, or deploy
it to production. No commits were made by this turn.

The deployed remote compound modules came from the **fleet** repository;
`score_boltz_batch.py`, target sequence, and vendored Boltz came from **main**.
Main's old validity/selection modules are not the correct modern test gates:
use deployed fleet/native gates (historical similarity 0.5, AtomPair entropy
0.25), not main's older defaults.

Remote base: `/root/nim19-test`, deployed repo `oss-repo/`, old client `client/`.
Reference worker entry point **`/root/search_python.sh`**, using the unchanged
locked Pixi environment (Torch 2.7.1, CUDA 12.8). Do not run heavy scientific
workers under base Python, install packages, redownload weights, or move NIM
into Pixi. NIM retains its own extracted, unmodified image/runtime.

Original baseline remote search/runner files were not overwritten. Source pins
are in `/root/nim19-test/schedule-comparison/frozen-inputs.json`; verify them
before/after the new run. Do **not** call broad `deploy`, old `start`, or old
`monitor`: they target old trials or overwrite pinned files. Transfer only the
needed explicitly named public experimental files.

## What already proved the old strategy is not competitive

All totals below are 20-molecule portfolios for historical **epoch 25176**,
P40261, reaction 3. NIM acquisition scores are NOT validator/reference scores.

| Run | Full NIM scores | Active elapsed | Ours, reference rescore | Same-pass leader |
| --- | ---: | ---: | ---: | ---: |
| Old 6 × 384 | 2304 | 47m07s | 1.4364645260 | 1.9266314308 |
| Old 12 × 192 | 2304 | 56m03s | 1.3206321684 | See shared audit below |
| Old 12 × 384 continuation | 4608 | 95m17s | 1.5294801399 | 1.8966195141 |

The public leader quote for this epoch was **1.8894035980**. A jointly rescored
fixed-final audit produced 6 × 384 **1.3106605700**, 12 × 192 **1.2478935872**,
leader **1.8815977836**. Larger batches won this one-epoch comparison, but the
old strategy lost badly. Primary versus repeat scores show material stochastic
variation; small improvements need repeated matched evaluation.

Diagnosis from actual saved data:

- Frozen proxy had weak ranking signal (overall saved-pool Spearman ~0.108).
  Its top 192 captured none of the saved field's actual best 20; top 384 captured
  five. This is retrospective pool analysis, not unbiased generalization.
- Old acquisition learned new NIM scores for parent selection, **not the proxy
  or replacement priors**. Reaction 3 also lacked the contextual policy that
  reaction 5 already had. Good discoveries were not fully fed back into search.
- Entropy selection was not the observed bottleneck: unconstrained best 20
  equaled the selected NIM portfolio in the baseline pool.
- Ours had larger mean heavy-atom count (41.5 vs leader 37.95), much lower mean
  binding probability (~0.376 vs ~0.801), and more distinct role-A reagents
  (17 vs 3). Deeper family exploitation is a plausible hypothesis, not proof
  that we know the leader's algorithm.
- The 0.5 novelty gate can reject >99% of the high-proxy shortlist. Never relax
  it or force similar molecules into the scoring lane to manufacture gains.

## Implemented algorithm work — opt-in, default off

`compound_miner/adaptive_strategy.py`:

1. `OnlineResidualProxy`: separate 64-tree ExtraTrees residual head on the
   frozen historical proxy, fit only from previous current-challenge NIM scores.
   Historical labels/encoder stay frozen. Mixed labels, identity mismatches,
   wrong reaction and inconsistent/nonfinite objectives fail closed. Repeats
   average per chemical identity; tree disagreement is heuristic uncertainty.
2. `FeedbackProposalPolicy`: reaction-3 contextual substitutions with separately
   estimated historical and current NIM contexts. Preserves old parent/move
   mixture; experimental replacement sources are 65% current-feedback,
   20% historical-context, 15% unrestricted valid-role draw. These fractions
   are NOT proven optimal. Contextual moves hold the other roles fixed, but
   **there is no explicit grid/sweep, guaranteed scoring quota, or low-rank model**.
3. `refresh_unscored_archive`: refreshes acquisition proxy signals after updates,
   without changing already scored rows. Otherwise INSERT OR IGNORE preserves
   stale proxy rankings in SQLite.

Optional `SearchEngine(..., proposal_policy_factory=None)` hook is present in
both local repos; default policy remains unchanged.

`tools/nim19_epoch_runner.py` now accepts challenge booleans `adaptive_proxy`
and `contextual_rxn3`, both default false. Feedback updates and archive refresh
are charged to the 72-minute budget. Adaptive resume fails closed. A further
`experimental_search_module: true` imports
`compound_miner.experimental_search.SearchEngine`, requiring an adaptive flag.
That last isolated-module wiring is local and has **not** been used for a fresh
six-round trial yet.

Important: remote `experimental_search.py` currently exists from the integration
smoke and was copied from **main's CPU-novelty path**. Before a full fair trial,
derive the isolated experimental module from the **pinned deployed fleet search**
and apply only the factory hook, preserving its GPU-first novelty path. Do not
overwrite baseline `search.py`. Record both base and modified digests.

## Evidence for the new code — promising, not a win

Prediction-before-update replay (`tools/analyze_compound_adaptation.py`), twelve
saved 384-molecule batches, no leader labels:

- First cold batch matched frozen proxy exactly.
- Subsequent eleven batches: mean within-batch Spearman
  **0.0118257 frozen → 0.4917846 adaptive**.
- Mean best-20 capture in predicted top 192: **9.8182 → 16.5455**.
- Best-20 NIM subtotal improved in all eleven saved batches.
- Feedback update 0.51–1.47 seconds, whole replay 23.53 seconds.

This is conditional on old acquired pools, not a fresh trajectory, canonical
win, portfolio validation, or epoch-fit proof.

Completed real integration smoke (`tools/smoke_compound_adaptation.py`):
**explicitly warm-started with 4608 saved NIM labels**, proposed 5000 tuples,
obtained 959 novel constructed candidates, scored **32 fresh molecules in
29.4986 seconds**. Total 161.3550 seconds; selected portfolio passed native
validation, **20 valid**, AtomPair entropy **0.4017555**, NIM total **1.5450472**.
It was NOT reference-rescored and is NOT a fresh or competitive epoch result.

Revival: real all-eight-GPU CUDA probe passed, NIM ready after 62.07 seconds;
eight full-quality real-task predictions took **8.3165 seconds**. No new packages
or downloads were needed.

Testing: **12 functional strategy tests passed under locked remote Pixi**;
**43 dependency-free harness/wiring tests passed locally**, repeated at handoff.
Eight changed/new Python files parse successfully. The latest added smoke
validation assertion was not rerun, but the saved native proof above was checked.

Local regression command (no credentials):

```sh
cd /Users/tvgdeena/nova-nanobody
python3 -m unittest tools/test_compound_adaptive_wiring.py tools/test_nim19_schedule_audit_repair.py tools/test_nim19_leader_extension.py tools/test_nim19_schedule_comparison.py tools/test_nim19_offline_playbook.py tools/test_nim19_native_runtime.py tools/test_nim19_benchmark.py
```

## Next action — one isolated, timed strategy trial

1. Ensure enough lease time remains and NIM is ready. Use the same prepared
   **25176** frozen challenge as the quickest clean anchor. It is a development
   epoch used to diagnose this policy, **not a held-out generalization test**.
2. Resolve the row/column request explicitly. A sensible experimental lane is
   coordinate sweeps/recombination around promising pre-epoch or current-round
   parents, preserving other roles, with a bounded slice of the existing exact
   scoring quota and some broad exploration. A proposal-only lane can disappear
   in global proxy filtering; if guaranteeing a scoring quota, record actual
   gate-passing coverage and spend it **within**, not above, the total budget.
   No such implementation, quota, or measured benefit exists yet.
3. Create a NEW directory/archive and separately named runner/client. Reuse
   frozen public baseline training/proxy/novelty, Pixi gate receipts and pinned
   native validator; do not copy prior NIM score rows or warm-start labels.
   Configure six sequential rounds, up to 384 full scores each, strict 4320-second
   budget including search, feedback, warmup and final reference evaluation.
   The existing planner reserves 600 seconds for final work and may reduce
   batches; report actual scored counts and never call partial work nominal 2304.
4. The existing runner's warmup selects eight future leaderboard molecules.
   Those scores are discarded and not fit, but for clean deployable behavior
   switch warmup to **pre-epoch-known molecules**, rather than using the leader
   during acquisition. Leader molecules belong only in the final matched audit.
5. NIM scores remain search-only. Native gates check reaction/product identity,
   uniqueness, heavy atoms/rotatable bonds, **historical Morgan similarity ≤0.5**,
   **20 molecules**, and **AtomPair entropy ≥0.25**. Native validity is not a
   second Boltz inference pass. Finalists and same-epoch leader are independently
   rescored using the pinned reference Boltz/Pixi worker and compared separately
   against the public quote and same-pass leader.
6. Use a bounded private process group (existing `run_bounded`) and an independent
   provider stop watchdog. Never send broad kill signals to NIM, other trials,
   or production. Existing supervise scripts target OLD paths: create/adapt an
   isolated bounded supervisor, test its contracts, do not blindly relaunch them.
7. Collect small final/progress/round receipts and SQLite **before** large CSV/
   pickle transfers, then stop the host. Report score, matched leader gap,
   elapsed including final evaluation, actual counts, validation and repeat
   uncertainty. A win is not established unless score and time both pass.

The short-lived attempt to prepare six historical epochs failed with a sanitized
`KeyError`; no study launched and local `strategy-study/` has no scientific
outputs. The user canceled that scope: do not spend time fixing six-epoch setup.
If testing another epoch later, pin history available by actual chain start;
API epoch-end publication can lag chain boundary ~50 minutes. Do not infer
availability simply from epoch number or train on the tested epoch's outcome.

## Full-quality NIM recipe / correction — preserve it

NIM 1.9 stock inference: recycling 3, sampling 100, diffusion samples 1,
affinity sampling 100, affinity diffusion samples 3, stock affinity refinement,
step scale 1.638, `max_parallel_samples=1`; eight concurrent independent HTTP
requests. This is not cross-molecule tensor batching. No reduced pass.

NIM internal `affinity_mw_correction=false`; apply exactly once externally:

```text
corrected_affinity = 1.03525938 * raw_affinity - 0.59992683 * MW**0.3 + 2.83288489
score = (affinity_probability_binary - corrected_affinity) / heavy_atoms
label = nim19-mw-corrected-search-only
```

Strict NIM/reference 0.99 rank-parity acceptance **failed** (Spearman ~0.98496).
The run is eligible only for **offline acquisition + matched reference rescore**;
retain the failed-parity receipt and never promote the label to canonical or
claim validator equivalence. Reference and NIM have different runtime profiles.

## Evidence locations

Local root: `data/nim19_epoch/20260917/`:

- `results/`: old six-round terminal results, canonical receipts and baseline
  SQLite. Complete baseline CSV/proxy collection is **not certified locally**.
- `leader-12x384/`: all 53 recovered extended-run JSON receipts + SQLite.
- `schedule-comparison/`: all 11 shared-comparison reports; `12x192/` contains
  48 JSON receipts + SQLite. The repaired matched audit really completed.
- `strategy-work/adaptation-replay.json`, `unit-results.json`.
- `strategy-smoke/smoke-final.json`, `round-1-nim.json`, progress and SQLite.
- `revival-canary/round-0-nim.json`, progress.
- `instance.json`: public instance manifest/deadline, no secret values.

Remote old evidence: `/root/nim19-test/epoch/`, `leader-12x384/`,
`schedule-comparison/`, and `strategy-smoke-20260918/`. Remote frozen baseline
`training.csv` and `proxy.joblib` remain usable on the retained disk.

Further workflow: `docs/NIM19_OFFLINE_PLAYBOOK.md`,
`deploy/nim19_offline/compute.yml`, and `tools/nim19_epoch_runner.py`.
Read these and preserve source pins, current safety guards, and credential rules.
No one-epoch adaptive win or live-ready molecule strategy has been demonstrated.
