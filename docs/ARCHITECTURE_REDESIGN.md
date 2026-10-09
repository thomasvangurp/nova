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
