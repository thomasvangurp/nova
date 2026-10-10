# Nanobody mining redesign proposal (2026-10-09)

This proposal is based on the evidence in
[ARCHITECTURE_AUDIT.md](ARCHITECTURE_AUDIT.md), and section references like
"audit §0.1" point there. Nothing here has been deployed. It builds on the
monitors that already exist and does not replace them:

- `check_deploy_drift`
- `watch_critical_switches`
- `check_mining_liveness`
- `rank_watch`
- `validate_scoring_change`
- `sync_multipass_rivals`

---

## Priority order

The ranking is by the cost each problem actually caused or is causing, against
the effort to fix it.

| # | Change | Why (evidence) | Effort | Risk to production | Kind |
|---|---|---|---|---|---|
| P0 | **Stop the running 20x re-fold on master** | ~80 GPU-h of duplicate work that **no production code reads**. It contends with the producer's first good race in 19 h, and it overwrites the old algo2 rows in place (audit §0.1) | minutes | none: nothing consumes its output | operator action |
| P0b | **Make DB failures loud**: `MUTATION_DB_UPDATE_FAILED` into the producer log and `failed.json`; `rank_exact.py:448` logs loudly and records `history_read_failed` in the artifact instead of a silent `return {}` (do not raise: that would block the provisional publish); `multipass_all_rivals` exit non-zero on `DB_WRITE_FAILED` | the producer wrote nothing for ~27 epochs and the publisher ran on a frozen DB, and the only trace was unmonitored `failed.json` (audit §0.4) | ~1 h | low | incremental |
| P1 | **Producer-identity check on the Pi + a switch rule in `watch_critical_switches`** (R4) | 5 epochs × 11–13 uids committed from a frozen bank, undetected by the Pi (audit §0.2) | ~1 h | low: an alert-only first step | incremental |
| P2 | **Store raw per-pass metrics; compute formulas at read time** (R2) | the only reason a label rename costs a full re-fold; it also makes the scale mixing repairable | ~½ day | low: additive column/rows | incremental |
| P3 | **One rival scorer** (R1) | 94–99% of master's 20x set was already 10x-scored by the dashboard with the identical scorer | ~½ day | low: master's refresh is off the commit path | scheduled, between epochs |
| P4 | **Label registry + write-time enforcement** (R3) | 4+ scales under `local_validator_replay_rank_sum`; the live producer's `top50_index` and the hf "our top-50" pool them (audit §3) | ~1 day | medium: it touches the writers | incremental, warn → enforce |
| P5 | **Put the producer and publisher argv in the repo** (R5) | the ~70-flag tuned producer config and the publisher switch exist only as box files (audit §6) | ~2 h | low | incremental |
| P6 | **Quarantine dead code** (R6) | ~156 unreachable files; 15 in-place patchers; the children lane is one argv edit from live (audit §5) | ~2 h | none for the moves; the children-lane removal needs P1 first | incremental |
| P7 | **Supervise the orphans and fix boot** (R7) | the hf daemon, archive fetcher and MySQL tunnel on master, and everything on the dashboard box, are ppid 1 with no restart and no boot start | ~2 h | low | scheduled |
| P8 | **One host inventory file** (R8) | box addresses hard-coded in 5 tools in 3 spellings; the Pi has a retired port | ~1 h | none | incremental |

**Not recommended now:** merging the producer and the publisher, or moving the
Pi's commit logic. Both are on the money path, and neither caused today's
incidents.

### Status (updated 2026-10-09 ~20:00 EDT)

Each item was re-checked against the live boxes before it was changed. "Live"
means deployed with a `*.pre-audit-*-20261009` (boxes) or
`*.bak-<ts>-deploy` (Pi) backup beside the file. Commits `b6022ae`,
`d53598f`, `516c91f`.

| # | Status | What was done / what is left |
|---|---|---|
| P0 | **Done** (operator) | The 20x re-fold was killed by hand. Re-checked: no `hf_field_refresh` process on master. The scorer daemon it used is still resident with no client (8 shard workers, ~0.5 GB per GPU, idle); stopping it is optional. |
| P0b | **Done, live** | `nanobody_opt/failure_log.py` (`record_failure`: traceback to stderr + a line in `/root/artifacts/pipeline_failures.jsonl`, never raises). `rank_exact` history read records `EXACT_OBSERVATION_HISTORY_READ_FAILED` and writes `observation_history.status=read_failed` into the rankings payload, still without raising. `sequential_exact_race` records `MUTATION_DB_UPDATE_FAILED`. `multipass_all_rivals` exits 4 on `DB_WRITE_FAILED` (JSON still written). The dashboard scorer script is now in the repo (`deploy/dashboard/`) and logs `MULTIPASS_FAILED` for any rc other than 0/1/3. `check_mining_liveness` gains `pipeline-failures-master` / `pipeline-failures-dashboard`. **Live evidence:** at 23:58Z and 23:59Z master recorded two real `MUTATION_DB_UPDATE_FAILED` (1213 deadlock) on `live_exact_screen` rankings, which before this change were stdout-only. They coincide with the first P2 back-fill run writing to the same DB (see P2), so they were most likely caused by it. **Not done:** routing the race's stdout marker into `failed.json` needs `run_live_producer.py`, which another agent owns; the failure log covers the same event. |
| P1 | **Phase A live; Phase B built, OFF** | Pi `pi_epoch_loop.py` checks every final pull (`artifact_kind`, `publication_stage`/`status` in {provisional, qualified}, per-submission stages, `producer.name` once present). A mismatch logs `UNEXPECTED_PRODUCER` to `loop.log` and `producer_alerts.log`; the commit is unchanged. Phase B: `touch /opt/nova-submit/REFUSE_UNEXPECTED_PRODUCER` (read every epoch, no restart) makes the loop commit the last expected artifact instead, through the existing `--allow-stale-nanobody` path, or no nanobody if none exists. Tested against the real 25613 (children) and 25619 (provisional) artifacts. **Live evidence:** the epoch 25620 commit logged `producer_identity_ok stage=provisional pool=16` and committed 13/13 (rc 0), and `nanobody_result.last_expected.json` is now kept. Liveness `submitter-identity` check: committed stage from the Pi archive, master artifact stage, and `publisher_argv.bin` must launch `publish-current-nanobody`. Run on today's history, it flags 25616/25617 as children commits. **Blocked on other agents:** layer 1 (`producer` field in the artifact) needs `publish_current_nanobody.py`, and the `session_watchdog.sh` relaunch refusal touches the publisher's launch path. Both are owned by the publisher agent. |
| P2 | **Writer done, live; back-fill run** | `multipass_all_rivals` stores the raw per-pass metrics in `metrics_json`, plus provenance: `formula=sum10`, `field_rows`, `field_sha256`, `pass_index`/`passes`, `writer`, `cohort` (ours/rival). `tools/backfill_multipass_metrics.py` re-imports the passes still on the dashboard disk (no GPU) under the same observation ids, keeping each row's original score/kind in provenance.

**Back-fill progress.** It is **partial and paused**:

- Done: epochs 25533, 25534 and 25593–25602, plus 318 of 480 rows of 25603.
- Left: 25603–25615 and 25618.
- Why it is paused: its first run (one large transaction per epoch) deadlocked against the live writers. It most likely caused the two producer `MUTATION_DB_UPDATE_FAILED` above and the dashboard's `DB_WRITE_FAILED` for the 25618 rescore (old code, exit 0). The rows for 25618 are recoverable from disk by this tool.
- The batched retry run was stopped by hand.
- Resume between producer iterations, or with the producer agent's agreement: `/root/bgen_python.sh -u /root/scratch/20261009-audit/backfill_multipass_metrics.py --epochs 25603-25615,25618`. It is idempotent: rows that already have metrics are skipped.

New multipass runs (25619 onward) write metrics themselves. **Left:** `validator_formulas.py` (one shared formula module) and read-time derivation. |
| P3 | **Decided: stay off** | Re-checked: nothing in the repo, master `/root`, dashboard `/root`, Pi `/opt` or Mac LaunchAgents reads `hf_field_20x_rank_sum` / `hf_field_sum10_20x_rank_sum`. The only readers are the refresh itself and a one-off `check_multipass_data.py`. `hf_field_refresh.py` now refuses to run without `--allow-unconsumed` or `HF_FIELD_REFRESH_ENABLED=1`. It is deployed to master, including over the stale `nanobody_opt/` decoy. See the recommendation below. |
| P3/R1 | **Bounded HF/rival top-up live; 20x stays off** | The dashboard pass scorer is now the one rival scorer, and the live parent pool reads it. `nanobody_opt/hf_topup.py` (new, shared) holds the HF-archive rank_binders top-200, which `hf_field_refresh` and `build_top50_index` now import (one copy). **Writer:** `multipass_all_rivals.py` adds HF top-200 parent candidates (>= 6 edits from the validator's archive top-50) to the same 10 scorer passes it already runs. Each candidate gets only the passes it is *missing* (stored `multipass_sum10_pass` rows with metrics count), up to 10, within 500 sequence-passes per run. The rows use the same label, source, writer and formula, with `cohort=hf_top200`. They are kept out of the dashboard numbers, and a failure never fails the leaderboard run. Switches: `MULTIPASS_TOPUP=0`, `MULTIPASS_TOPUP_DEPTH`, `MULTIPASS_TOPUP_BUDGET`. **Reader:** `build_top50_index` (called by `run_live_producer` → `competitor_lane --priority-parents`) ranks HF candidates and the last 16 epochs' rival rows. The ranking key is the mean sum10 re-derived at read time from `metrics_json`, against this epoch's archive. It merges that list into `live_sequences` **by rank position only**: `merge_by_rank` takes no scores and puts top-up #k at position 4k, for up to 10 slots. The producer-scale top-50 keeps its order and its fail-closed `TOP50_EMPTY`. The top-up degrades to empty. The dedup reference `sequences` is unchanged. `touch /root/artifacts/DISABLE_TOP50_TOPUP` turns it off for the next build. Tests: `tests/test_hf_topup.py`. Commit `e38602c`. **Live evidence:** epoch 25622 iteration_01 `top50_index.json` (02:16Z) has 60 parents. The 50 producer-scale parents are in their original order, 49 of them shared with 25621. Positions 3, 7, … 39 hold 10 rival sequences that were never in the pool before (10 passes each, derived sum10 ~1.06e5–1.14e5); there was no `failed.json`. The epoch's `competitor_parents.json` has `priority_pool_size=60`, and 404 of its 2000 single edits come from top-up parents #1 and #2 (positions 3 and 7: 266 + 138 edits). The dashboard's 25620 run (01:42–02:32Z, 50 min against ~30 before, still under one 72-min epoch) was the first top-up. It logged `TOPUP_PLAN candidates=117 this_run=50 sequence_passes=500` and `TOPUP_DB_WRITE observations=500`, and the strict label guard accepted the rows; its dashboard JSON still holds only the 53 leaderboard sequences. After that run the reader ranks 168 eligible candidates, 33 of them HF top-200. The best HF candidate ranks 118th, so today the reserved slots go to current rivals. That is expected: the full-history top-200 is far from the live basin. ~67 HF candidates remain to top up, about 2 more epochs at 500 sequence-passes each. **Note:** the competitor lane cuts at `--limit 2000`, about 266 single edits per parent, with no ranker model on master. Only the first ~8 priority parents ever get edits, so the stride, not the slot count, decides the top-up's real share. |
| P4 | **Reader fix live; registry live (multipass strict, rest warn)** | `build_top50_index._live_top50_from_db` reads only `REPLAY_SCALE_SOURCES` (the ~20–370 producer-scale writers) and fails closed on an empty result. **Live evidence:** the producer's epoch 25620 iteration_03 `top50_index.json` (built 00:18Z with the fix) is identical to the offline prediction: 48/50 shared with the pre-fix iteration_01, and no `failed.json`. Measured on the live DB: 887 of our sequences had off-scale rows mixed in, and 2 of the 50 slots change (old ranks ~44,230 / ~44,400). New multipass rows are written as `multipass_sum10_pass`, not `local_validator_replay_rank_sum`. **Blocked on other agents:** see "new finding 1". **Left:** the registry module plus warn→enforce in `update_observation_store` (in `mutation_network.py`, owned now), and the other pooled readers (`elite_parents_from_db`, `children_submit`, `hf_field_refresh.OUR_TOP50_SQL`, all off the live path). **Registry (2026-10-09 ~21:10 EDT):** `nanobody_opt/observation_labels.py` is the registry; [`docs/OBSERVATION_LABELS.md`](OBSERVATION_LABELS.md) is generated from it, and a test keeps the two in sync. `update_observation_store` checks every batch before writing. Strict labels (`multipass_sum10_pass` and the closed legacy labels) refuse a wrong writer, formula or source (`OBSERVATION_LABEL_REFUSED`). Every other label only warns on stderr, and refuses only with `NOVA_LABEL_ENFORCE=1`. `multipass_all_rivals` now runs `assert_formula` before any GPU work: its `pass_formula` must reproduce the pinned golden outputs for `PASS_FORMULA_ID`, and the registry must agree. A formula change therefore cannot land under the old label. Shown live: a `formula=algo2` batch under `multipass_sum10_pass` was refused before the DB connect, and `algo2_score` fails the sum10 golden check. **Regimes, verified by recomputing all three formulas from raw metrics (rho=1.000):** algo2 = 25559–25610 (written before the 14:01:40Z category3 box deploy). category3 = 25536–25538 and 25611–25612 (written 14:01:40Z–17:19:41Z). sum10 = 25613–25618 (written after 17:25:15Z). These are the CORRECT, current scale. The boundaries follow write time, not epoch; the `.bak.*` timestamps on the box give the deploy times. **Relabelled** with `tools/relabel_multipass_legacy.py`: 24,398 rows moved, scores untouched, provenance added. The legacy rows went to `legacy_multipass_algo2_pass` (19,110) and `legacy_multipass_category3_pass` (2,490). The 25613–25618 rows joined `multipass_sum10_pass`, which stays the current label. Two 25586 rows whose formula cannot be pinned were refused by the fingerprint guard and stay as documented off-scale. `tools/check_observation_labels.py` is the read-only trip-wire, and it is CLEAN after the relabel. **Dashboard:** the panels read the box JSONs, not the DB, and those are all on the sum10 scale. But ≤25612 holds sum10 of the validator's published single-pass metrics (API back-fills; rho=1.000 vs API), while ≥25613 holds 10-pass local fast-scorer means (rho 0.2–0.5 vs API). The three multipass panels now break lines and trailing means at that boundary and label both regimes. New JSONs carry a per-epoch `method` block. Live on Vercel since 21:04 EDT. **Write-path evidence:** `backfill_multipass_metrics --epochs 25613` (run with the guard) wrote 510 rows through the registry check into `multipass_sum10_pass` with metrics, with no refusal and no failure-log entry. It took ~25 min for 510 rows, so the rest of the P2 back-fill (25603–25612, 25614–25618) is left to its owner. Deployed with `.pre-audit-p4-20261009` backups: master `nanobody_opt/{mutation_network,observation_labels}.py`; dashboard the same plus `/root/{multipass_all_rivals,relabel_multipass_legacy,check_observation_labels}.py`. |

**P3 recommendation: keep the 20x pipeline off.** Do not restart it on any
terms until a consumer exists. Its own premise ("parent pool") was never
wired into the producer, the publisher or `build_top50_index`, and P2 means
the dashboard's 10 passes per sequence now keep their raw metrics. If a 20x
pool is wanted later:

1. Show the consumer first, with `validate_scoring_change.py`.
2. Build it as R1 describes: an aggregator over stored passes, plus at most a
   bounded top-up of the *missing* passes for the ~50 pool sequences per
   epoch on the dashboard's GPUs.
3. Never run a second full GPU scorer on master's GPUs.

**New findings not covered by P0b–P4:**

1. **The producer's mutation network pools every scale too.**
   `MutationNetwork.__init__` (`mutation_network.py`) builds each node's
   robust score from every observation of the sequence, of any `score_kind`.
   Only `validator_final_rank_sum` is preferred when present. So multipass
   and hf rows (~1e5) enter the live producer's node scores, the same bug as
   P4 in a second, live reader.
   - Fix: filter to the producer-scale sources, the same whitelist as
     `REPLAY_SCALE_SOURCES`, in `_load_network_observations_uncached` or in
     `__init__`.
   - **Blocked:** `mutation_network.py` was redeployed at 17:59 EDT by the
     agent fixing the MySQL timeout.
2. **A real silent DB failure was found.** The dashboard multipass for epoch
   25602 (04:09Z) logged
   `DB_WRITE_FAILED (1205, 'Lock wait timeout exceeded')` and exited 0. Its
   rows were missing until the P2 back-fill wrote them from disk.
3. **There is a duplicate orphan scoring loop on the dashboard.** Besides the
   tmux `dashboard-refresh-loop`, pid 6103 (ppid 1, started ~21 h earlier)
   runs `while true; do bash score_latest_rival_epoch.sh; sleep 1800; done`.
   The shared flock keeps the two from overlapping, but it belongs under P7.
4. **Grouped reads of `observations` can hit MySQL 2013.** A GROUP BY over
   (sequence, source) for P20809 hit the 60 s read timeout. The live
   top-50's GROUP BY (sequence) currently finishes, measured end to end at
   61 s including pixi start-up. It is the same failure class the producer
   just hit, so watch for `TOP50` failures in `failed.json`.

---

## R1. One rival scorer: "multipass" owns raw passes, everyone else reads them

**Current state.**

- The dashboard's `multipass_all_rivals.py` (10 passes per epoch) and master's
  `hf_field_refresh.py` (20 passes over the dashboard's set) both run the same
  scorer on the same sequences.
- Neither reads the other's passes.
- Each one collapses its passes to a single field-relative rank-sum.

**First question: is a 20x pool needed at all?** The audit found no
production reader of either `hf_field_*` label. Either wire it into a real
consumer, through a `validate_scoring_change.py` run that shows it improves
parent or top-50 choice, or delete `hf_field_refresh*` and
`import_hf_field_20x.py` outright. Everything below assumes the pool earns a
consumer.

**Proposal: one owner, one table of passes.**

1. **The owner** is `multipass_all_rivals.py`, renamed in spirit to "pass
   scorer". It runs where idle GPUs exist; today that is the dashboard box.
2. **What it writes.** It writes raw per-pass metrics (R2), keyed
   `(sequence, scorer_version, pass_seed)`, for:
   - every leaderboard sequence;
   - our committed sequences, which the Pi already pushes;
   - the HF top-200.
3. **What it never writes.** No rank-sum is stored as a primary record.
4. **What master's 20x becomes.** "20x" is a *depth target*, not a separate
   pipeline.
   - The pass scorer tops up any sequence in the parent pool to 20 passes.
   - It only ever scores the *missing* passes. A sequence with 10 dashboard
     passes needs 10 more, not 20.
5. **`hf_field_refresh.py` becomes a read-only aggregator.** It takes the
   pool, reads the passes, computes sum-of-10 against the *current* archive
   field, and writes one derived row per sequence (R3 `derived` class).
   - It never invokes a GPU, so it never contends with the producer.
6. **Expected saving.**
   - Today: master's per-epoch increments of 8–150 sequences × 20 passes.
   - Proposed: 10 extra passes on the ~50 pool sequences per epoch.
   - It also removes the entire ~80 GPU-h class of "label changed, re-fold
     everything" runs.

**Migration** (safe incrementally, because master's refresh is not on the
commit path):

1. Land R2's metrics column first.
2. Make `multipass_all_rivals.py` include `metrics` in each observation. It
   already has them in `rows[seq]` and throws them away.
3. Add `--top-up-to N` to it.
4. Switch `hf_field_refresh.py` to aggregate-only.
5. Stop `hf_field_refresh_daemon.sh` on master. Do this between epochs; it is
   not a cutover of anything the Pi reads.

**Status (2026-10-10):** steps 1–3 are done. Step 3 is an env-bounded top-up
(`MULTIPASS_TOPUP_DEPTH`/`_BUDGET`), not a flag. Step 5 was done by P0.
Step 4 was replaced: the read-time aggregation lives in
`hf_topup.ranked_topup`, which `build_top50_index` calls every producer epoch,
so `hf_field_refresh.py` has no remaining job. It stays disabled and is an R6
deletion candidate together with `import_hf_field_20x.py`. Details and live
evidence are in the P3/R1 status row above.

---

## R2. Store raw metrics; never store only a formula output

**Why.** Every scoring-formula change so far (algo2 → category3 → sum-of-10)
has done one of two things:

- silently mixed scales under the old label; or
- forced a full re-fold under a new label (audit §0.1, §3).

The BoltzGen per-pass metrics are formula-independent.

**Proposal.**

- Writers fill `metrics_json` with the 10 validator metrics on every pass
  observation. `update_observation_store` already supports this.
- Rank-sums become *derived*: computed at read time against an explicit,
  recorded field. Each derived row records:
  - the formula id (`sum10`, `algo2`, `category3`);
  - the field snapshot (HF archive row count + sha256);
  - the number of passes.
- One shared module, `nanobody_opt/validator_formulas.py`, holds `sum10`,
  `algo2` and `category3`. It is imported by `hf_field_refresh`,
  `multipass_all_rivals`, `local_neighborhood`, `rank_exact` and
  `validate_scoring_change`, replacing the 4+ copies in audit §4.2.
  - `validate_scoring_change.py`'s `builtin:` sources point at the same
    functions, so the validation gate and production cannot diverge.

**Back-fill.**

- Rows with `metrics_json='{}'` can't be repaired. They keep their old label
  and are marked `legacy` (R3).
- `scoreN/exact_all.json` files still on disk (dashboard 25594–25618, master
  `hf_field_refresh/*`) do contain the metrics. They can be re-imported
  offline, with no GPU.

---

## R3. An enforced `observations` label convention

**Rule: a `score_kind` names one scale.** A value written under a label must
be comparable, by `<`, with every other value under that label. `source` only
says who wrote the row; it never changes the meaning of `score`.

**Registry.**

- `nanobody_opt/observation_labels.py` is a Python dict, so it is importable
  by both writers and readers.
- `docs/OBSERVATION_LABELS.md` is generated from it.
- One entry per `score_kind`:

  ```python
  "hf_field_sum10_20x_rank_sum": Label(
      klass="derived",                 # raw_pass | derived | legacy
      scale="sum of 10 dense ranks vs full HF archive field; ~2e4-1.2e5; lower=better",
      formula="sum10", field="hf_archive_full", passes=20,
      writers={"hf_field_refresh"}, owner="hf_field_refresh.py",
      introduced="2026-10-09",
  ),
  ```

**Enforcement, in two steps.**

1. **Warn.** `update_observation_store` looks up `score_kind` and
   `provenance.writer` in the registry. It logs and counts unknown labels, and
   writers not listed for a label.
2. **Enforce, one week later.** The same check raises. A new label then needs
   a registry entry stating scale, formula, field and passes, which is exactly
   the information whose absence caused the sum-of-10/algo2 mix.

**Repair the existing mix** without deleting anything; this is the audit trail:

- Re-label historical multipass rows by epoch range to explicit legacy kinds.
  The boundaries come from the measured per-epoch scale shifts in the audit:
  - `legacy_multipass_algo2_<range>` for the ~37 k and ~53 k ranges;
  - `multipass_sum10_pass` for 25613+.
- Re-label the producer's ~200-scale rows only if they turn out to be mixed
  too. The audit DB table shows whether they are.
- Fix the readers that pool across scales. The first one is on the live path
  today:
  - `build_top50_index._live_top50_from_db` (live: the producer's
    `top50_index.json`);
  - `hf_field_refresh.OUR_TOP50_SQL`;
  - `children_submit`;
  - `elite_parents_from_db`.

  Each one names exactly one `score_kind` **and** an explicit `source`
  whitelist. They then fail closed on an empty result instead of falling back
  to a pooled read.

**Split `source` into its two meanings.** Keep `source` as "writer" and add
`cohort` (`ours`, `rival`, `hf_top200`) in `provenance_json`. The cohort comes
from the leaderboard coldkey at write time, so "rivals" stops including us.

**Stop `hf_field_refresh` reading its own output as input.** Its rival set
should come from the leaderboard, or from `cohort='rival'`, not from
`source='public_multipass_rivals'`.

**A weekly read-only check** joins with `tools/validate_scoring_change.py`'s
family: per `score_kind`, per epoch, it computes min, p50 and max and alerts
when the p50 moves more than 3x between adjacent epochs. That one query would
have caught the 37 k → 120 k jump the day it happened.

---

## R4. One monitored "real submitter" invariant

**Current state.**

- The identity of the submitter is whatever last wrote
  `/root/artifacts/nanobody_result.json`. Two scripts can write it.
- The switch between them is `publisher_argv.bin`.
- The Pi checks only epoch freshness and an observation floor that the
  frozen bank met exactly.

**Proposal: three layers, cheapest first.**

1. **The artifact says who made it.**
   - `publish_current_nanobody.py` adds `producer: {"name":
     "publish_current_nanobody", "commit": <git sha>, "argv_sha256": ...}` to
     the artifact. This is a schema 4 → 5 additive field.
   - Any other writer must use a different `name`.
2. **The Pi checks it.**
   - `pi_epoch_loop.py` keeps an allowlist (`publish_current_nanobody`) and an
     expected `publication_stage ∈ {provisional, final}`.
   - **Phase A:** on mismatch it logs `UNEXPECTED_PRODUCER` and raises an
     alert, but still commits. Committing is still better than missing an
     epoch.
   - **Phase B, after a week without false alarms:** it refuses to commit a
     non-allowlisted artifact unless a deliberate override file exists on the
     Pi.
   - This is the only check positioned *at* the money boundary.
3. **The switch file is a watched invariant, not just a watched file.**
   - Extend `watch_critical_switches.py`, which already records
     `publisher_argv.bin` content changes, with a *rule* on top of the
     change log: the decoded argv must contain `publish-current-nanobody`.
   - If not, raise `ALERT SUBMITTER_SWAPPED` immediately, not merely as
     "file changed".
   - Add the same rule to `session_watchdog.sh`'s safety check, next to the
     existing `--commit-command` refusal. Then a swapped argv is not
     relaunched automatically after a crash.
4. **Liveness confirms from the outcome side.** `check_mining_liveness.py`'s
   publisher check adds the `producer.name` field and the
   `sequence_pool_size` to its evidence. `rank_watch.py` already covers the
   rank consequence.

**Retire the alternative.** After layer 2 Phase A is live, remove or move to
quarantine `children_lane_loop.sh`, `children_submit.py`,
`publisher_argv.pre-rollback-20261009.bin` and `children_lane.py` (R6). If a
children lane is ever wanted again, it should be a *mode of the publisher* (a
flag whose value lands in `producer`), not a second writer of the same file.

**Cutover.** Layers 1, 3 and 4 are safe incrementally. Layer 2 Phase B changes
the Pi's commit behaviour and should be scheduled. Deploy it at the start of
an epoch window, with the override file pre-staged, using pi-submit's own
deploy path (`pi_drift.py`).

---

## R5. The configuration lives in the repo, the box gets a copy

- Add `deploy/producer/producer_argv.txt` and
  `deploy/producer/publisher_argv.txt` (NUL-free, one argument per line) as
  the source of truth.
- `producer_relaunch.sh` and the watchdog compare the box file against the
  deployed copy. `check_deploy_drift.py` gains two entries; it already tracks
  `session_watchdog.sh`.
- A hand edit on the box then shows up as `LOCALMOD` within one drift run, and
  a tuning change becomes a commit with a message.
- Record provenance correctly: the deploy step writes
  `artifacts/nanobody_deployed_commit.txt` every time. Today it is stuck at a
  10-08 commit.
- Move production off the `wip/worktree-snapshot-20261001` branch, whose
  history says "do not merge", onto `nanobody-opt`, and update
  `OPERATIONS.md` §3, which already claims that branch. This is a git-only
  operation with no box impact, and it should be done by whoever owns the
  in-flight fixes.

---

## R6. Dead code: delete or quarantine, never leave in place

Moves happen in the repo only. Box copies are removed only after the drift
checker confirms that nothing executes them.

| Action | Files |
|---|---|
| Delete (git keeps history) | `nanobody_opt/patch_*.py`, `tools/apply_*_live.py` (in-place string patchers: re-running one re-applies an old edit); `select_elite_parents._database_payloads` (no caller since `8ddc474`); repo-root `0`, `root@hetzner...`; master decoys `/root/nanobody_opt/hf_field_refresh{.py,_loop.sh}`; `*.bak*`/`*.pre-*` on boxes once the critical-switch blob store covers them |
| Move to `attic/` with a one-line README stating why and when | `tools/nim19_*` (30), `tools/*_box_control.py` (10), `scoring_backfill/` (111), `nanobody_opt/diag_*`/`audit_*`/`backtest_*`, `frontier_*`, `children_*`, repo-root `patch_rank_exact_residue.py`, `simulate_residue_flip.py`, `dryrun_order.py`, the old `HANDOFF*.md` |
| Box one-offs → `analysis/oneoffs/<date>/` in the repo, then delete from the box | master `check_*.py`, `rank_*_vs_hf.py`, `corr_analysis.py`, `disambiguate.py`, `compare_multipass.py`, `rescore_master.sh`, `*rosterresync.sh`; dashboard `api_backfill.py`, `backfill_multipass.py`, `sum10_backfill.py`, `rank_epoch25612.py`, `rescore_backfill.sh`, `backfill_norivals_*.sh` |
| Keep, but bring into the repo | dashboard `dashboard_refresh_loop.sh` and its `score_latest_rival_epoch.sh` variant; master `fetch_hf_archive_daemon.sh`, `mysql_tunnel_watchdog.sh`, `onstart.sh`; Pi `push_intended_watcher.sh`; the five untracked Mac plists |

**Guard against recurrence.** Add a CI-style test (`tests/test_live_manifest.py`)
over a checked-in `deploy/LIVE_MANIFEST.txt`, which lists every file each box
executes:

- It fails if a manifest file is missing.
- `check_deploy_drift.py` reads its `ENTRYPOINTS` from the same manifest
  instead of its own hard-coded list, so the two stay one list.

**Box hygiene rule.** A one-off belongs under `/root/scratch/<date>/`, never in
`/root` or `/root/nanobody_opt`. The drift checker flags new top-level
`*.py`/`*.sh` files that are not in the manifest.

---

## R7. Every long-running process has a supervisor and a boot path

**Master.** Extend `session_watchdog.sh`, which is the established convention,
to own these as tmux sessions:

- `hf_field_refresh_daemon` (until R1 retires it);
- `fetch_hf_archive_daemon`;
- `mysql_tunnel_watchdog`.

Also have the scorer-daemon supervisor actually start the daemon. Today it
waits on a hand-started pid.

**Dashboard box.**

- Its `onstart.sh` starts the same watchdog pattern for
  `dashboard-refresh-loop` and the tunnel.
- `multipass_all_rivals.py` exits non-zero on `DB_WRITE_FAILED`, so the
  liveness checker's dashboard-multipass check can see it.

**Pi.** Add `/opt/nova-nanobody`, which holds the TAO-spending `slot_manager`
and `infra_sweep`, to `pi_drift.py`'s coverage.

**Boot.** All of this can be done incrementally. The first test is a
deliberate container restart of the *dashboard* box, which is off the money
path, before relying on it on master.

---

## R8. One host inventory

- `config/hosts.toml` is the single place that maps a role to a host, port and
  vast label: master, dashboard, Pi.
- The five Mac monitors and `sync_multipass_rivals.py` read it.
- The Pi's `push_intended_watcher.sh` resolves the dashboard box by vast label,
  the way `pi_submit.py` already resolves master. Without that, a port
  rotation silently stops the "ours" supplement.
- Remove `pi_epoch_loop.py`'s retired `ssh2:15470` `PROD` constant.

---

## What stays as it is

These are already right:

- The miner's hotkey resolution (coldkey + live metagraph + sticky file).
- `roster_watch`'s coldkey-derived uid set.
- The Pi's single-writer role.
- `check_deploy_drift` + `watch_critical_switches` + `check_mining_liveness`
  + `rank_watch`.

The gaps are the ones above: what those monitors cannot see, namely
*semantic* invariants (who produced the artifact, what scale a label holds),
plus the processes no supervisor owns.
