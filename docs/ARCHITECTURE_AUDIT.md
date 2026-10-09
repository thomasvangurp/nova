# Nanobody mining architecture audit (2026-10-09)

Read-only audit of what actually runs, checked against the three live hosts on
2026-10-09 between about 18:50 and 19:40 EDT. Nothing on any box was changed.
The companion proposal is [ARCHITECTURE_REDESIGN.md](ARCHITECTURE_REDESIGN.md).

All times are EDT unless marked `Z`. "Box" means the vast.ai instance, and
"repo" means the `wip/worktree-snapshot-20261001` HEAD at `d97a6a4`.

---

## 0. Live issues (fix these before anything structural)

These were still happening while the audit ran.

### 0.1 Master: a ~10 h, ~80 GPU-hour 20x re-fold whose output nothing reads is competing with the producer

- **What is running.** `hf_field_refresh.py` started at 17:14 and is scoring
  2,908 "rival" sequences at 20 passes. It runs on all 8 GPUs through the warm
  scorer daemon.
- **Why it started.** The sum-of-10 deploy renamed the label to
  `hf_field_sum10_20x_rank_sum`, so `already_scored()` returned 0 rows
  (`refresh.log`: `pool already scored=0; new: ours=50 rivals=2908 hf=200`).
- **Measured cost.** Pass 1 of the rival set ran 17:38–18:09 and pass 2 ran
  18:09–18:39, about 30 min per pass. Twenty passes is about 10 h, followed by
  another 200-sequence HF set.
- **How much is redundant.**
  - 2,870 of the 2,908 sequences (99%) already have dashboard multipass rows;
    2,735 of them have at least 10 passes.
  - 2,853 (98%) already had a 20x score under the old algo2 label.
  - The raw BoltzGen metrics do not depend on the formula. The re-fold happens
    only because neither pipeline stores those metrics (§4.1).
- **Nothing consumes the result.** No production code reads
  `hf_field_20x_rank_sum` or `hf_field_sum10_20x_rank_sum`. I checked the repo
  and master's `/root/nanobody_opt`, `/root/tools` and `/root/neurons`. The
  only readers are the refresh's own `already_scored()` and docstrings. The
  refresh's "parent pool" is not wired into the producer, the publisher or
  `build_top50_index`.
- **The write overwrites history.** The new rows reuse
  `observation_id=hf20x:{seq}`, the PRIMARY KEY of the old algo2 rows.
  `ON DUPLICATE KEY UPDATE` therefore overwrites and relabels them in place,
  despite the code comment "the stale ones are untouched". 48 of the 50
  `our_top50_20x` rows have already been relabelled. The 2,853 rival rows will
  be overwritten when this run finishes.
- **Why it matters now.**
  - The live producer just completed preparation for the first time in about
    19 h (every iteration from epoch 25605 to 25618 ended in `failed.json`;
    see §0.4).
  - It is now in its `sequential_exact_race` screen, with its own 8 scorer
    workers on the same 8 GPUs. `nvidia-smi` shows both process sets, about
    4 GB each per GPU, at 98–100% utilisation.
  - The refresh is running at about 0.6 s per sequence-pass. Its own
    uncontended rate earlier today was 0.2 s per sequence-pass (200 sequences
    in 40 s), so both jobs are slowed by roughly 3x.
- **Why the busy gate did not stop it.** The loop's GPU-busy gate checks once,
  at start. At 17:14 the producer was still crash-looping and the GPUs were
  idle. Nothing re-checks the gate once a refresh is running.
- **Operator decision needed (not taken here).**
  - Stop the refresh; it has no consumer, and it is destroying the only copy
    of the old algo2 values.
  - Or at least cut it to idle GPUs until the producer is healthy.

### 0.2 The swap switch that caused today's outage is still armed

- **What changed.** For epochs **25613–25617** the Pi committed the
  children-lane artifact (`publication_stage=children`, `sequence_pool_size=300`,
  exactly 8 observations on every uid) on all 11–13 uids. The Pi logged
  `committed rc=0` every epoch.
- **Why nobody noticed.** The Pi's only quality gate is
  `--confirmation-min-observations 8` (`tools/pi_epoch_loop.py:246`), which
  the children artifact meets exactly. Neither `pi_submit.py` nor
  `pi_epoch_loop.py` reads `publication_stage`, `artifact_kind` or a producer
  identity.
- **The switch is still armed.**
  - `/root/artifacts/publisher_argv.bin` still decides what the watchdog
    relaunches.
  - `nanobody_opt/children_lane_loop.sh` is still on the box and in the repo.
  - The watchdog's only safety check is "argv starts with `tmux` and has no
    `--commit-command`" (`session_watchdog.sh`).
  - The children argv is still saved beside it as
    `publisher_argv.pre-rollback-20261009.bin`.

### 0.3 Dashboard "multipass vs rival" panel mixes measurement methods by epoch range

- **The JSONs.** `multipass_{E}.json` for 25533–25612 were rewritten today
  (17:26–17:30Z and 18:13Z) by API single-pass sum-of-10 backfills. Their
  values are exact integers, which a 10-pass mean cannot produce.
  - Only 25613–25618 hold real 10-pass means.
  - For 25594–25612, real 10-pass outputs exist on disk but were overwritten.
  - For 25537–25593, the raw passes are gone.
- **The DB.** The rows under the same label were never corrected, and they sit
  on three scales (§3).

### 0.4 The producer wrote no new observations for about 27 epochs, silently

- **Last new data.** The newest `live_exact_screen` row is epoch 25592. The
  newest `exact_runs` ingest is 2026-10-08 22:27 (DB timestamp, likely `Z`).
- **Every iteration failed.** Every producer iteration from 25599 to 25618
  ended in `failed.json`. 144 of them failed in `local_neighborhood`, which
  hit MySQL error 2013 in `load_observations`. `a8ea00b` is the fix, and
  epoch 25619's race is the first to run.
- **The publisher kept going on stale evidence.** It kept writing artifacts
  every epoch from a DB frozen at 25592.
- **Why it was silent:**
  - `sequential_exact_race.py:1342` catches any DB write failure and prints
    `MUTATION_DB_UPDATE_FAILED` only to subprocess stdout, which
    `run_live_producer.py:346` does not capture. The marker has 0 hits on the
    box.
  - `check_mining_liveness.py` does flag the failing iterations, but it was
    added only today.
- **Two more silent-failure paths of the same kind:**
  - `rank_exact.py:448` uses `except Exception: return {}` around the history
    read.
  - `multipass_all_rivals.py` prints `DB_WRITE_FAILED` and exits 0.

### 0.5 Not nanobody, but found running and failing

- **Compound track.** Every Pi epoch since about 25610 logs
  `no_molecules why=epoch have=25609`. The compound release fleet has been
  stuck at epoch 25609, and no molecules have been committed since.
- **Blueprint.** The Mac LaunchAgent `ai.metanova.sn68.autosubmit` exits with
  status 2 every 30 min. Its log says
  `BLOCKED can_submit=false reason=insufficient_balance` and lists 30+ epochs
  never secured.

---

## 1. What actually runs

### 1.1 Master (`ssh9:10616`, 8x4090, vast label `nova-nanobody-pixi`)

**Sessions (tmux), each started by `session_watchdog.sh` from a relaunch script**

| tmux session | Start command | Live process | Reads | Writes |
|---|---|---|---|---|
| `nova-session-watchdog` | `/root/session_watchdog.sh`, started by `onstart.sh` | bash loop, 60 s | tmux sessions, `chain_state.json` mtime and roster age, `publisher_argv.bin` | (re)creates the 5 sessions below; `artifacts/session_watchdog.log` |
| `nova-nanobody-miner` | `bash /root/miner_relaunch.sh` | `neurons/miner/miner.py --chain-state-only --wallet.hotkey nova68-slot-337` | chain (finney, netuid 68); hotkey chosen by `deploy/pick_registered_hotkey.py` from the coldkey's live registrations (sticky file `artifacts/miner_signer_hotkey`) | `artifacts/chain_state.json`, the **only** writer |
| `nova-nanobody-producer` | `bash /root/producer_relaunch.sh`, which replays `artifacts/producer_argv.txt` | `pixi run -e bgen live-producer` → `nanobody_opt.run_live_producer` → `local_neighborhood`, `sequential_exact_race`, `score_boltzgen_waves`, `score_boltzgen_fast_adapter` → `fastscore/pinned-530f5dcbc54d/fast_score_batch.py` (its own 8 GPU workers, **not** the daemon) | MySQL `observations` (through `db.connect`; the `--mutation-database ...sqlite` path is ignored), HF archive, chain state | `/root/live/epoch<E>/producer/**`, including `exact_all.json`; observations through the importer |
| `nova-nanobody-publisher` | `pixi run -e chain310 publish-current-nanobody ...`; argv in `artifacts/publisher_argv.bin` | `tools/publish_current_nanobody.py` → `rank_exact.py`, `check_validator_novelty` (metanano), Layer-2 | `chain_state.json`, `live/epoch<E>/**/exact_all.json`, MySQL history | `artifacts/nanobody_result.json` (what the Pi commits), `nanobody_published_sequences.txt`, `nanobody_publisher_state.json`, `nanobody_history/` |
| `nova-nanobody-scorerd` | `bash /root/scorer_daemon_supervisor.sh` | supervisor only, in "adopt" mode: it is waiting on daemon pid 2248677, which was **started by hand 23 h ago (ppid 1)** | `artifacts/nanobody_fast_scorer.json` | `artifacts/scorer_daemon_supervisor.log` |
| `nova-rosterwatch` | `bash /root/roster_watch.sh` | bash loop, 60 s | `chain_state.json` | `artifacts/roster_watch.log`; restarted twice at 17:24–17:25 |

**Daemons outside tmux and the watchdog (ppid 1, `nohup`)**

None of these is restarted by anything, and none comes back after a container
restart: `onstart.sh` starts only the watchdog.

| Process | Script (box-only unless noted) | Cadence | Reads | Writes |
|---|---|---|---|---|
| `hf_field_refresh_daemon.sh` → `hf_field_refresh_loop.sh` → `/root/hf_field_refresh.py` (= repo HEAD) + `/root/import_hf_field_20x.py` | repo `nanobody_opt/hf_field_refresh*` | every 30 min, gated by GPU util > 40% plus a 3 h starvation override | leaderboard API, MySQL (`public_multipass_rivals` rows, replay rows), `/root/P20809_nanobodies.csv` | `/root/hf_field_refresh/<source>/score*/`, MySQL `observations` (`score_kind=hf_field_sum10_20x_rank_sum`) |
| `scorer_daemon.py` + 8 shard workers | repo `nanobody_opt/scorer_daemon.py` | resident | socket `artifacts/nanobody_scorer.sock` | GPU. **Its only client is the hf_field refresh.** The supervisor comment says "for the live producer", but `producer_relaunch.sh` deliberately does not use it |
| `fetch_hf_archive_daemon.sh` | box-only | hourly | HF dataset | `/root/P20809_nanobodies.csv` |
| `mysql_tunnel_watchdog.sh` + `ssh -L 3306` | box-only | 30 s | — | tunnel used by `hf_field_refresh_loop.sh` (`SN68_MYSQL_HOST=127.0.0.1`). Everything else uses `db.py`'s default host or the creds file. That gives **two DB routes with different failure modes** |

There is no cron on master (no `crontab` binary). `sync_loop.sh` /
`sync_structures.sh` (`.cif` rsync to Hetzner) exist but are **not running**,
and `/tmp/sync_loop.log` does not exist.

### 1.2 Dashboard box (`ssh2:12752` = `95.3.33.46:13920`, 2x4090)

- **`dashboard-refresh-loop` (tmux).** Runs `/root/dashboard_refresh_loop.sh`,
  which is box-only. Every 300 s it:
  - runs `fetch_hf_archive`, writing `/root/hf_archive.csv`;
  - runs `score_latest_rival_epoch.sh`, a **local variant**. It reads the
    current epoch over ssh from master's `chain_state.json`, has no GPU gate,
    and looks back 9 epochs;
  - for each of those epochs, runs `multipass_all_rivals.py E 10` (= HEAD).
- **`multipass_all_rivals.py`** scores 10 passes of every P20809 sequence on the
  leaderboard, plus our intended sequences (pushed by the Pi into
  `/root/our_intended_sequences/`). It writes:
  - `multipass_{E}.json`;
  - MySQL rows `source=public_multipass_rivals`,
    `score_kind=local_validator_replay_rank_sum`, one row per pass, with
    `metrics_json={}`.
  - It never reads the DB.
- **A one-off backfill is running outside tmux:**
  `backfill_norivals_20261009.sh 25617 25616` (ppid 1). It holds
  `/tmp/score_rival_epoch.lock`, so the loop's scorer silently exits 0 until
  the backfill finishes.
- **Nothing restarts anything here.**
  - There is no cron, no systemd, and `onstart.sh` is empty.
  - The MySQL tunnel runs unsupervised; `mysql_tunnel_watchdog.sh` exists but
    is not running.
  - On a DB failure, `multipass_all_rivals.py` prints `DB_WRITE_FAILED`, still
    writes the JSON, and exits 0.

### 1.3 Pi (`100.64.243.65`, on-chain writer)

| Unit | Runs | Role |
|---|---|---|
| `nova-submit-loop.service` | `/opt/nova-submit/tools/pi_epoch_loop.py --on-submit-host --reactions 1,2,3,4,5` → `pi_submit.py` | **Sole on-chain writer**, nanobody and compound. Pulls master's `/root/artifacts/nanobody_result.json`, resolving master by vast label. Commits every uid. Archives to `archive/epoch<N>/` |
| `nova-nanobody-producer-guard.timer` (1 min) | `/opt/nova-nanobody/deploy/slot_host/ensure_vast_running.py` | keeps master running |
| `nova-hotkey-sync.timer` (~10 min) | `sync_hotkeys.sh` | rsyncs `mining_ops/nova68-slot*` hotkeys to master |
| `nova-slot-manager.timer` | `nanobody_opt.slot_manager` (spends TAO) | registration |
| `nova-infra-sweep.timer` | `nanobody_opt.infra_sweep` (moves TAO) | funding sweep |
| `nova-bank-pull.timer`, `nova-producer-watchdog.timer` | compound | compound only |
| tmux `push-intended-watcher` | `/opt/nova-submit/push_intended_watcher.sh`, **not in any repo** | every 120 s, scps our committed uid→sequence pairs to the dashboard box (hardcoded `ssh2:12752`) |

Code ownership on the Pi:

- `/opt/nova-submit` is mirrored by `~/nova-sandbox-repro/pi-submit`
  (`pi_drift.py`, hourly). It reports "no drift".
- `/opt/nova-nanobody`, which holds the TAO-spending `slot_manager` and
  `infra_sweep`, is **not covered by any drift check**. Its `DEPLOYED.json`
  best-matches `5f7a70d` (2026-09-16, not exact).
- `deploy/submit_host/nova-submit-loop.service` in the repo still says
  `--reactions 3`; the live unit runs `1,2,3,4,5`.

### 1.4 Mac (controller)

The Mac runs these launchd jobs:

- `critical-switch-watch` (5 min)
- `deploy-drift`
- `liveness-watch`
- `rank-watch` (15 min)
- `multipass-sync` (600 s; the docstring says 60 s)
- `dashboard` and `dashboard-server`
- `backup`
- `pi-code-drift`

Five of the plists are **not tracked in the repo**: `dashboard`,
`multipass-sync`, `rank-watch`, `pi-code-drift`, and the Blueprint
`autosubmit`.

---

## 2. Deployed copy vs repo

### Master

I hashed every `nanobody_opt/*`, `tools/*`, `neurons/miner/*`, `deploy/**` and
root `*.py`/`*.sh` file (268 files) against HEAD:

- Every file that is executed matches HEAD, including the root-level
  `/root/hf_field_refresh.py`. The latest drift-checker run (18:42) agrees.
- **STALE but not executed:**
  - `/root/nanobody_opt/hf_field_refresh.py` and `hf_field_refresh_loop.sh`:
    the decoy copies that hid the original bug, still present.
  - `/root/multipass_all_rivals.py` (one-off).
  - `/root/tools/sync_multipass_rivals.py` (an rsync leftover).
- **Box-only, no repo counterpart:**
  - Ad-hoc: `ceiling_test.py`, `check_exact_coverage.py`,
    `check_exact_recency.py`, `check_ipae_fast.py`, `check_multipass_data.py`,
    `check_obscount{,2}.py`, `check_sources{,2}.py`, `compare_multipass.py`,
    `corr_analysis.py`, `disambiguate.py`, `rank_children_vs_hf.py`,
    `rank_prev_vs_hf.py`, `rescore_master.sh`.
  - Operational one-shots: `deploy_rosterresync.sh`,
    `restart_rosterresync.sh`, `watch_rosterresync.sh`, `probe_miner_rs.sh`.
  - Live infrastructure: `fetch_hf_archive_daemon.sh`,
    `mysql_tunnel_watchdog.sh`, `sync_loop.sh`, `sync_structures.sh`,
    `onstart.sh`.
- **The box keeps backups beside live files:** `*.bak.<ts>`, `*.pre-*`,
  `children_lane.py.pre-netobs-stream-20261009`.

### Dashboard

- `multipass_all_rivals.py` is current.
- `score_latest_rival_epoch.sh` is a local variant, pinned in
  `ACCEPTED_VARIANTS`.
- These `nanobody_opt` modules are **stale**: `db.py` (no `stream()`),
  `mutation_network.py` (about 103 lines behind), `local_neighborhood.py`,
  `run_live_producer.py`, `children_*`, `elite_parents_from_db.py`. Only `db.py`
  and `mutation_network.py` are actually imported by the live multipass job.
- `dashboard_refresh_loop.sh` and all of its one-offs are box-only:
  `api_backfill.py`, `backfill_multipass.py`, `compare_multipass.py`,
  `rank_epoch25612.py`, `sum10_backfill.py`, `rescore_backfill.sh`,
  `backfill_norivals_20261009.sh`, and five `multipass_all_rivals.py.bak*`
  files.

### Pi

`pi_submit.py`, `pi_epoch_loop.py`, `rival_margin.py` and `chain_money.py` match
the repo. `make_fleet.py` and `bank_release.py` are intentional local variants.

### Repo itself

- **Production deploys from a branch named "WIP SNAPSHOT ... do not merge".**
  `ff0c93c` is described as "unreviewed mixed work from many sessions".
  `docs/OPERATIONS.md` §3 still says "one branch (`nanobody-opt`)", which is
  not what runs.
- The master's recorded `artifacts/nanobody_deployed_commit.txt` (`4763179`,
  10-08) and `nanobody_publisher_running_commit.txt` (`83f0e49`) are
  **never updated by today's hand deploys**. They are wrong as provenance.

---

## 3. The MySQL `observations` table in practice

### Shared mechanics

- **There is one write path:**
  `nanobody_opt.mutation_network.update_observation_store`. It is an
  `INSERT ... ON DUPLICATE KEY UPDATE` keyed on `observation_id`. On a
  collision it **overwrites** `source`, `score_kind`, `score` and `epoch`.
- **The sqlite path is vestigial.**
  - Every caller passes a sqlite path. `db.connect()` ignores it and always
    connects to MySQL.
  - Master's `/root/artifacts/nanobody_mutations.sqlite` has been **0 bytes**
    since 16:54 today.
  - The path is still load-bearing in two ways:
    - the 418 MB `...sqlite.netobs_cache.P20809.127.pkl` cache is keyed off
      it;
    - the dead `_database_payloads` gate tested `.is_file()` on it, so
      deleting that empty file would have silently disabled DB reads.
- **`source` is overloaded.** It means "which population the sequence belongs
  to" for some writers (`public_multipass_rivals`, `our_top50_20x`,
  `hf_archive_top200`) and "which pipeline wrote it" for others
  (`live_exact_screen`, `children_*x`).
  - `hf_field_refresh.py` writes `source='public_multipass_rivals'`, the same
    label the dashboard multipass writes, but with a different `score_kind`,
    pass count and id scheme.
  - `hf_field_refresh.fetch_db_rivals()` then reads that same source back as
    its definition of "rivals". The refresh therefore feeds on its own output.
- **"Rivals" includes us.** `multipass_all_rivals.py` writes our own intended
  sequences under `source='public_multipass_rivals'`.
- **No raw metrics.** Both multipass writers store only the field-relative
  rank-sum, with `metrics_json='{}'`. Changing a formula, or changing the
  archive field, therefore cannot be replayed from the DB; it forces a
  re-fold. That is the direct cause of §0.1.
- **`epoch` is NULL** on every `hf_field_refresh` row (it never sets `epoch`).

### Scale mixing under one label (DB query)

`score_kind='local_validator_replay_rank_sum'` carries at least four scales
under one name.

**Producer and replay rows, about 200 scale:**

- `live_exact_screen` (3.1 M rows)
- `live_exact_repeat`
- `children_*x`
- `backfill_8x`
- `exhaustive_8x`

**`hf_archive_top200`:** about 59 k.

**`public_multipass_rivals`, 28,632 rows over 2,935 nodes, by epoch range:**

| Epochs | Mean score |
|---|---|
| 25533–25538 | ~53 k |
| 25559–25610 | ~37 k |
| 25611–25612 | ~52 k |
| 25613–25618 | ~120 k |

The multipass ranges differ because the formula (algo2 vs sum-of-10) and the
archive field size both changed over time, and the rows were never re-labelled.

**Readers that pool across the label (no `source` filter):**

- `build_top50_index._live_top50_from_db` (`:79-91`) is **live**: the producer
  calls it at `run_live_producer.py:523` to build every iteration's
  `top50_index.json` (`source: our_top50`).
- `hf_field_refresh.OUR_TOP50_SQL` (`:84-95`).
- `elite_parents_from_db.load_replay_means` (`:89-101`).
- `select_valid_deep`, `stack_neutral`, `diag_*`.

**Measured effect.**

- The multipass and HF rows (≥2e4) never reach the top of an `ORDER BY
  AVG(score) ASC`. Their effect is to *exclude* the 874 of our own sequences
  that have rows on both scales.
- A single 1e5 row sends each of those to about rank 44,000 of 46,852. On
  their small-scale rows alone, 2 would be in the top 50 (ranks 26 and 27).
- The comment at `multipass_all_rivals.py:298` says the multipass rows exist
  "so `_live_top50_from_db` sees public sequences ... (high-fidelity
  parents)". That intent is **structurally impossible** on these scales, and
  the feature has been silently inert since it was added.

**Comment and behaviour disagree.** The producer comment at
`run_live_producer.py:510-514` says the dedup top-50 "is rank_binders over the
FULL HF archive". `build_top50_index.py:147-154` actually uses our own pooled
replay top-50 whenever a DB path is passed, which the producer always does.

### 3a. Every `(source, score_kind)` in use (DB query, 2026-10-09)

`observations` has 2.84 M rows (11.6 GB). `observation_id` is the PRIMARY KEY.
The indexes are `node_id`, `(node_id,epoch)`, `(node_id,observation_count)`
and `(source,score_kind)`. There is no `created_at` and no epoch index, so
`hf_field_refresh.py:293` `SELECT MAX(epoch)` scans the whole table on every
run.

| source / score_kind | rows | epochs | score min/med/max | writer (live?) | readers |
|---|---|---|---|---|---|
| live_exact_screen / local_validator_replay_rank_sum | 3.11 M | 25041–**25592** | 20–367 | `sequential_exact_race.py:1848` → `mutation_network.py:1430` (live, but failing 25599–25618) | `_live_top50_from_db`, `OUR_TOP50_SQL`, `select_valid_deep`, `stack_neutral` |
| local_exact_raw / local_exact_validator_replay_rank_sum | 705 k | 25033–25592 | 20 / 217 / 322 | `import_exact_screens.py:240` (live) | `rank_exact.py:448` (exceptions swallowed), `_live_top50`, `OUR_TOP50` |
| live_exact_repeat / local_validator_replay_rank_sum | 11.2 k | 25085–25587 | 37 / 195 / 334 | `sequential_exact_race.py:1999,2198` (live) | generic replay readers |
| public_validator_api / validator_final_rank_sum | 39.9 k | 24074–25617 | 13 / 156 / 337 | `seed_mutation_database.py:41`, `select_elite_parents.py:716` (live) | `elite_parents_from_db`, `terminal_policy`, `mutation_network.py:148` |
| children_8x / local_validator_replay_rank_sum | 31.4 k | **25423 only** | 74 / 196 / 312 | **no writer at HEAD or on any box** (WIP commits only) | `children_submit.py` via `children_lane_loop.sh`: this is the "frozen bank" |
| backfill_8x, exhaustive_8x, children_20x, children_40x / local_validator_replay_rank_sum | 12.3 k / 4.7 k / 1 k / 200 | ≤25435 | ~100–316 | no writer at HEAD | generic |
| **public_multipass_rivals / local_validator_replay_rank_sum** | 28.6 k (per pass), 2,935 sequences | 25533–25618 | **2.1e4 / 4e4 / 1.8e5, three regimes** | `multipass_all_rivals.py:310` (live, dashboard) | pooled into every generic replay reader; `hf_field_refresh.py:168` uses it as its rival set |
| hf_archive_top200 / local_validator_replay_rank_sum | 800 | NULL | ~5.8e4 | **no writer in repo** | generic (scale mismatch) |
| public_multipass_rivals, hf_archive_top200, motif_variants_20x, our_top50_20x / hf_field_20x_rank_sum (algo2) | 2,853 / 275 / 642 / 2 | NULL | 2.6e4–6.0e4 | old `hf_field_refresh` + `import_hf_field_20x.py:45`; `motif_variants_20x` has no writer in repo | **none** |
| our_top50_20x / hf_field_sum10_20x_rank_sum | 50 (growing) | NULL | 1.06e5–1.31e5 | `hf_field_refresh.py:43` (live) | **none** except its own `already_scored` |
| local_exact_raw / unknown | 1 | NULL | 42 | a test row leaked into production | – |

**Every multipass, hf and 20x row has empty `metrics_json` and
`provenance_json`.** Nothing in such a row records its formula, its field or
its pass count.

Other facts from the same query:

- `hf_field_refresh.py:168` also reads a phantom
  `slave_public_multipass_rivals` source, which has 0 rows.
- `children_lane.py:34-44` would fail under the server's
  `ONLY_FULL_GROUP_BY`. Its dead status is real.
- Raw `sqlite3.connect` on the 0-byte sqlite path survives in non-live
  readers: `select_valid_deep`, `stack_neutral`, four `diag_*`, and
  `external_evidence.py:201`. Each would silently read nothing if revived.


---

## 4. Redundancy, with evidence

### 4.1 Two 10x/20x rival scorers that never share work

| | Dashboard `multipass_all_rivals.py` | Master `hf_field_refresh.py` |
|---|---|---|
| Sequence set | leaderboard rivals + ours, per epoch | latest rivals ∪ **every DB row with source `public_multipass_rivals`** (i.e. the dashboard's set) ∪ our replay top-50 ∪ HF top-200 |
| Passes | 10 | 20 |
| Scorer | same `score_boltzgen_fast_adapter` + `fast_score_batch` | same |
| Formula | sum-of-10 vs full HF archive | sum-of-10 vs full HF archive (algo2 before 10-09 17:14) |
| Stored | per-pass rank-sum, no metrics | 20-pass mean rank-sum, no metrics |
| Reads the other's output? | no | only to decide *what* to re-fold |
| Who consumes the result | dashboard panel (JSON); DB rows are pooled into the replay readers, where they are inert or harmful (§3) | **nobody** (§0.1) |
| Cost | ~25–27 min / epoch on 2 GPUs (~50 GPU-min) | per-epoch increments of 8–150 sequences, all just scored by the dashboard; now ~80 GPU-h for the label reset |

Master's 20x of a rival sequence repeats the 10 passes the dashboard already
ran with the identical scorer. The extra 10 passes are the only new
information, and the 10 duplicate passes are thrown away.

### 4.2 Four implementations of the two scoring formulas

| Formula | Live definitions | Notes |
|---|---|---|
| algo2 | `utils/minmax_weighted_rank.rank_binders`, `diversity.minmax_weighted_rank`, `hf_field_refresh.algo2_score`, `multipass_all_rivals.algo2_score` | algo2 is still the *correct* formula for the validator's top-50 **novelty reference** (`local_neighborhood.py:1475`, `build_top50_index`) |
| sum-of-10 | `hf_field_refresh.sum_of_ten`, `multipass_all_rivals.sum_of_ten`, and the `SUM_OF_TEN_DIRS` copies in `local_neighborhood`, `instrument_check`, `pipeline`, `recombine`, `simulate_round`, `validate_scoring_change`, `audit_commit_ordering` | |
| category3 / "confidence" | `rank_exact._compiled_category_rank_sums`, `CONFIDENCE_OBJECTIVE` | |

So two formulas are both legitimately live, for different purposes (novelty
reference vs score). The bug class is not "two formulas". It is that a stored
number does not say which formula and which field produced it.

### 4.3 Other overlaps (cheap, but each is a place to go stale)

- **Three HF-archive fetchers:**
  - master daemon, hourly, to `/root/P20809_nanobodies.csv`;
  - dashboard loop, every 5 min, to `/root/hf_archive.csv`;
  - `children_lane_loop.sh` per epoch.
  - Master also holds a stale `/root/hf_archive.csv` from 11:44, used by
    `rescore_master.sh`.
- **Two submitters that write the same artifact:** `publish_current_nanobody.py`
  and `children_lane_loop.sh`, both writing `artifacts/nanobody_result.json`
  (§0.2).
- **Box locators are hard-coded in five Mac tools, in three spellings:**
  - `check_deploy_drift`, `check_mining_liveness`, `watch_critical_switches`
    and its plist use `ssh9:10616` / `ssh2:12752`;
  - `sync_multipass_rivals` uses `95.3.33.46:13920`;
  - `pi_epoch_loop`/`pi_cutover` use the retired `ssh2:15470`;
  - plus the Pi's `push_intended_watcher.sh`.
  - Only the Pi resolves master by vast label.

---

## 5. Dead vs reachable

I ran static reachability from the live entrypoints: the pixi tasks, the
producer, the publisher, the scorer daemon, the hf refresh, multipass, the
miner, `pi_submit` and `pi_epoch_loop`. Module and `-m` references were
followed transitively. Of 231 `nanobody_opt/` + `tools/` files, about **75 are
reachable** and **about 156 are not**.

The Pi additionally runs `slot_manager` and `infra_sweep`. The Mac runs the
five monitors, `sync_multipass_rivals` and `rank_watch`.

| Category | Count | Verdict |
|---|---|---|
| `tools/nim19_*` (NIM19 experiment harness) | 30 | dead for nanobody production |
| `tools/*_box_control.py` (one per past experiment box) | 10 | dead |
| `nanobody_opt/patch_*.py`, `tools/apply_*_live.py` (string-replace patchers that edit `/root/...` in place) | 15 | **dangerous dead code**: re-running one re-applies an old edit over current code |
| `nanobody_opt/diag_*`, `audit_*`, `backtest_*` | ~20 | analysis one-shots |
| `frontier_*`, `children_lane*`, `children_submit`, `longtail_children`, `frontier_loop.sh` | 8 | **looks dead, is one argv edit away from live** (§0.2) |
| `scoring_backfill/` (111 files, hard-coded retired vast hosts) | 111 | dead |
| repo root: `patch_rank_exact_residue.py`, `simulate_residue_flip.py`, `dryrun_order.py`, `smoke_*`, stray files `0` and `root@hetzner...` | — | one-shots and typos |

**`_database_payloads()`**
(`select_elite_parents.py:575`) has **no caller at HEAD**:

- `8ddc474` removed the call.
- `patch_skip_db.py` and `patch_frontier_clean.py` are the box-side patchers
  that originally did the same thing by string replacement. They now print
  `MISS` if run.
- The function body is still the most prominent DB-read logic in the file, and
  it still reads `validator_final_rank_sum`.

**`nanobody_opt/hf_field_refresh_daemon.sh`** is unreachable as a module, but
it is the repo counterpart of a live daemon. Reachability by import is not
reachability by deployment.

---

## 6. Hard-coded identity and config

| Value | Where | Risk |
|---|---|---|
| Miner signing hotkey | **fixed**: `miner_relaunch.sh` → `pick_registered_hotkey.py` resolves from coldkey + live metagraph, sticky file. Live argv shows `nova68-slot-337`, resolved, not configured | ok |
| Our coldkey SS58 | literal in 9 runtime files (`hf_field_refresh`, `multipass_all_rivals`, `children_lane_loop.sh`, `miner_relaunch`, `roster_watch`, `pick_registered_hotkey`, `nanobody_dash`, `rank_watch`, `validate_scoring_change`) and the publisher argv | low (it never changes), but there is no single source |
| `chain_state.json` `miner_uid` | resolved once at miner start; `roster_watch.sh` documents that it goes stale | ok (watch derives from coldkey) |
| Box addresses | §4.3; `pi_epoch_loop.py:33-35` `PROD=ssh2:15470` (retired) | **medium**: latent on the Pi's non-default path |
| MySQL host | `db.py` default `46.225.101.210` vs `SN68_MYSQL_HOST=127.0.0.1` tunnel in `hf_field_refresh_loop.sh` | two routes, two failure modes |
| GPU list | `MULTIPASS_GPUS` default `0,1`; `rescore_master.sh` `1,2,3,4,5,6`; producer argv in `producer_argv.txt`; hf loop auto-detects | ok |
| Producer argv (~70 flags) | `/root/artifacts/producer_argv.txt` only, captured from a running process, not in the repo | **high**: the tuned configuration of the main pipeline exists only as a box file |
| Publisher argv | `/root/artifacts/publisher_argv.bin` only | **high**: this is the swap switch (§0.2) |
| Pi unit args (`--reactions`, score floors) | live unit ≠ `deploy/submit_host/nova-submit-loop.service` | medium |

