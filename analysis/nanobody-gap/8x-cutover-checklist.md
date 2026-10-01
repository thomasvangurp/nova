# SN68 nanobody: 4x -> 8x RTX 4090 cutover checklist

Prepared 2026-09-28. Nothing in this file has been executed against production.

| | old (production) | new (standby) |
|---|---|---|
| instance | 51125470 | 53241806 |
| label now | `nova-nanobody-pixi` | `nova-nanobody-pixi-8x` |
| GPUs | 4x RTX 4090 | 8x RTX 4090 (GPU 3 reports 23028 MiB, others 24564) |
| ssh (proxy) | `ssh -p 15470 root@ssh2.vast.ai` | `ssh -p 11806 root@ssh3.vast.ai` |
| ssh (direct) | - | `ssh -p 46916 root@8.17.147.152` (static IP) |
| host | DC, US | Florida, US, machine 148978, 256 cores, 503 GB RAM, 300 GB disk |
| price | $1.567/h | $4.271/h |

Shorthand used below (run from the Mac):

```bash
OLD="ssh -o ServerAliveInterval=30 -p 15470 root@ssh2.vast.ai"
NEW="ssh -o ServerAliveInterval=30 -p 46916 root@8.17.147.152"
PI="ssh thomasg@192.168.1.68"
VAST=/opt/nova-nanobody/deploy/slot_host/.pixi/envs/default/bin/vastai
# box-to-box push from OLD using its existing key; the new box authorizes it
PUSH_E='ssh -i /root/.ssh/id_foldcheck -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 46916'
```

## How the new box was built

1. `ansible/nanobody.yml` + `ansible/nanobody_services.yml` from a clean detached
   worktree at **fbc4dd82** (`/Users/tvgdeena/nova-nanobody-8x-deploy`), the last commit
   the playbooks recorded as running on production (`nanobody_producer_running_commit.txt`
   and `nanobody_publisher_running_commit.txt`). Inventory `ansible/inventory/vast_nanobody.py`
   with `NOVA_VAST_LABEL=nova-nanobody-pixi-8x`, profile `vars/nanobody-high-throughput.yml`,
   and `nanobody_producer_activate=false nanobody_publisher_activate=false
   nanobody_chain_writer_activate=false nanobody_hotkey_sources=[]` (plus the public coldkey
   ss58). No tmux session was started, and none exists on the new box.
2. Then the off-ansible reconciliation listed in the next section, copied from production.

## Verification on the new box (2026-09-28 ~19:55Z, no live session started)

- `nanobody.yml` @ fbc4dd82: ok=78 failed=0 (second run; the first died on a transient Zenodo
  0-byte `nanobody_model_4`, which the rerun fetched). Results: `BOLTZGEN_OK 2.7.1+cu126 8`,
  Layer-2 canary 4/4 passed, all four checkpoints sha-verified, fast scorer pin 530f5dcbc54d,
  `autonomous producer active=False`.
- `nanobody_services.yml` @ fbc4dd82: ok=44 failed=0. chain310 installed, the focused publisher
  tests pass on the deployed tree, and publisher and chain writer were NOT started.
- Drift check (checksum dry-run from production after the overlay): **0 differing files** in
  nanobody_opt, tools, neurons, utils, compound_miner, tests, config, fastscore,
  external_tools/boltzgen, NOVA-nanobody-filter, deploy, bgen_shim, metanano_shim, pixi.toml/lock,
  wrappers, supervisor scripts, nanobody.env, and the wallet tree.
- Fast scorer smoke test, 8 GPUs: `score_boltzgen_fast_adapter.py --gpus 0,1,2,3,4,5,6,7`,
  16 sequences, 8 shards (one per GPU, 2 sequences each, about 29 s/seq cold), 69.7 s wall,
  rc 0, protocol `validator-dual-sample-v1`, 16 scored, no empty or NaN values. Every GPU peaked
  at 95-100 % utilization. Artifacts are in `/root/smoke_8gpu/`.
- py_compile: all 79 `nanobody_opt/*.py` (bgen), plus publisher, miner, chain_state,
  publication_gates and pick_registered_hotkey (chain310): OK.
- `pixi run ... live-producer -- --help` and `publish-current-nanobody -- --help`: OK.
- argparse dry parse with the production parser: `producer_argv.txt` and `producer_argv.8gpu.txt`
  both parse (gpus=0..7 reserve=120 cts=0.36 rounds=20 per_stage=5 keep=0.65).
  `publisher_argv.8gpu.bin`: all 22 flags are known to the parser, and `--expected-gpus 8`.
- SQLite: `quick_check ok`, 1,996,754 observations (snapshot 19:11Z).

## Deviations from ansible (everything on the new box that did not come from the playbooks)

Copied from production because production runs them and git (at fbc4dd82) does not have them:

- **Hand-deployed source** (production content differs from fbc4dd82):
  - `nanobody_opt/`: audit_validator_score.py, chain_state.py, compute_budget.py, fleet.py,
    import_exact_screens.py, mutation_network.py, rank_exact.py, run_live_producer.py,
    score_boltzgen_fast_adapter.py, score_boltzgen_waves.py, sequential_exact_race.py,
    win_probability.py (modified); chunk_reuse.py, pool_policy.py, resident_scorer.py,
    round_parallelism.py, scorer_daemon.py, shrinkage.py (only on production).
  - `tools/publish_current_nanobody.py`, `neurons/miner/miner.py` (modified).
  - `tests/`: test_compute_budget.py, test_nanobody_rank_exact.py (modified);
    test_chunk_reuse.py, test_miner_chain_state_resync.py, test_scorer_daemon.py,
    test_shrinkage_and_pool.py (only on production).
  - The editor backup files (`*~`, `*.bak-*`) were copied too, so the trees are identical.
  - 4 audit scripts and 4 tests exist in fbc4dd82 but not on production. They were left
    in place on the new box; nothing imports them.
  - utils/, config/ (except the two downloaded validator yamls), external_tools/boltzgen,
    ansible/files/root wrappers, pixi.toml and pixi.lock are **identical** to fbc4dd82.
- **Fast scorer model package (repo defect).** `.gitignore:11` (`models/`) excludes
  `external_tools/fast_nanobody_scorer/boltzgen_src/boltzgen/model/models/` from git, at fbc4dd82
  and at HEAD. The playbook therefore deploys a pinned scorer without `boltz.py` and
  `__init__.py` there. Both files were copied from production's
  `/root/fastscore/pinned-530f5dcbc54d/`. A box built by ansible alone cannot fast-score.
  Fix in the repo with `git add -f` (or a `!` rule) before the next rebuild.
- **Supervisor scripts that are not in git** (`/root/*.sh`): producer_relaunch.sh,
  session_watchdog.sh, miner_relaunch.sh, scorer_daemon_supervisor.sh, roster_watch.sh,
  publisher_artifact_only.sh, publisher_watchdog.sh, plus `/root/deploy/pick_registered_hotkey.py`.
- **argv files:** `/root/artifacts/producer_argv.txt` and `publisher_argv.bin` are production's,
  unchanged. The 8-GPU versions are **pre-built and dry-parsed** alongside them as
  `producer_argv.8gpu.txt` and `publisher_argv.8gpu.bin`. Step 5 moves them into place.
  - producer diff vs production: `--gpus 0,1,2,3,4,5,6,7`, `--confirmation-time-share 0.36`
    (was 0.45), `--replication-rounds 20` (was 12), `--replication-rounds-per-stage 5` (was 4),
    `--replication-keep-fraction 0.65` (was 0.6), and `--reserve-blocks 120` appended
    (production passes none, so it uses the default 55). Everything else is byte-identical,
    including the empty `--strict-gates ""`.
  - publisher diff vs production: `--expected-gpus 8` (was 4), and a single
    `tmux new-session` prefix (production's file has it twice; see latent bug 1).
- **Secrets, copied box-to-box and never printed:** `/root/nanobody.env` (production's copy
  with GITHUB_TOKEN; it overwrote the playbook-rendered one, which has no token) and
  `/root/.bittensor/wallets/mining_ops/hotkeys/*` (hotkeys only; there is no coldkey).
  After cutover the Pi's `nova-hotkey-sync` keeps the hotkeys current by label.
- **State:** `/root/artifacts` (everything except the SQLite, the socket and the two
  `*_deployed_commit.txt` markers, which the playbooks write) and `/root/layer2_cache`, as of
  about 19:40Z. `nanobody_mutations.sqlite` was restored from the integrity-checked Mac
  backup `~/nova-backups/nanobody_mutations.20260928T191125Z.sqlite.gz`
  (10,623,946,752 bytes, `quick_check ok`, 1,996,754 observations). It is delta-synced at cutover.
  **`/root/live` is deliberately empty.** History lives in the SQLite, the producer creates the
  current epoch dir itself and prunes old ones, and step 3 copies only the newest two epoch dirs.
- **HF cache / checkpoints:** `/root/.boltzgen` comes from the playbook's checksum-verified fetch.
  `/root/.cache/huggingface` was downloaded straight from Hugging Face at production's pinned
  revisions (boltzgen-1 @ c1be29e1, inference-data @ c3d36fd2). Every blob is sha256-verified.
  A truncated `boltzgen1_diverse.ckpt` blob left by an aborted rsync was detected and replaced.
- **Publication gate floor** (applied at cutover, step 5): `--minimum-observations` is not a
  CLI flag. It is the module constant `EVIDENCE_GATES["--minimum-observations"] = "1"` in
  `/root/nanobody_opt/publication_gates.py`. The publisher reads it directly via `gate_defaults()`,
  and the race reads it via `gate_arguments()`. A `--strict-gates` JSON reaches only the race,
  so setting 20 consistently means editing that one line on the box. That is one more
  hand-deployed deviation.
- **ssh:** `/root/.ssh/authorized_keys` on the new box also holds production's
  `id_foldcheck.pub` (`root@4b489d5c6ae1`), so production can push state to the new box.
  Remove that line after the cutover is stable (step 9).

## Latent bugs found on production (not fixed there)

1. `publisher_argv.bin` holds `tmux new-session -d -s nova-nanobody-publisher` **twice**.
   Replaying it (which session_watchdog.sh and publisher_watchdog.sh both do) returns rc 0
   but creates no session and no process. This was tested on the new box. A publisher death on
   production is therefore NOT recovered, while the watchdog log says it was. The new box
   gets a single-prefix copy (step 5).
2. The publisher fails closed when `nvidia-smi` counts a GPU number different from
   `--expected-gpus`, which is 4 in production's argv. It must be 8 on the new box or the
   artifact is never written.

## Cutover

Do it early in an epoch: after the Pi has committed epoch N, and with at least 200 blocks
before the 55-block commitment reserve. Epoch length is 361 blocks. Check the position:

```bash
$OLD 'python3 -c "import json;d=json.load(open(\"/root/artifacts/chain_state.json\"));print(\"block_in_epoch\",d[\"block\"]%d[\"epoch_length\"])"'
```

Start when `block_in_epoch` is roughly 5-60. The whole procedure takes about 15 minutes.

### 0. Pre-flight (new box, nothing started)

```bash
$NEW 'tmux ls 2>&1; nvidia-smi -L | wc -l; df -h / | tail -1; ls -la /root/nanobody.env; ls /root/.bittensor/wallets/mining_ops/hotkeys | wc -l'
# expect: "no server running", 8, >150G free, -rw------- env, >=400 files
$PI "sudo systemctl is-active nova-submit-loop nova-nanobody-producer-guard.timer"
```

### 1. Pause the Pi's producer guard for the relabel window

The guard fails closed on 0 or 2 matching labels. It does nothing destructive, but it
should not try to "recover" either box halfway through.

```bash
$PI 'mkdir -p ~/.nova && touch ~/.nova/producer-autostart.disabled'
```

### 2. Stop the old box, in this order (watchdog first, so it restarts nothing)

```bash
$OLD 'tmux kill-session -t nova-session-watchdog'
$OLD 'tmux kill-session -t nova-nanobody-producer; tmux kill-session -t nova-nanobody-scorerd'
# the publisher only reads the DB (mode=ro); stop it and the chain-state miner too
$OLD 'tmux kill-session -t nova-nanobody-publisher; tmux kill-session -t nova-nanobody-miner; tmux kill-session -t nova-rosterwatch'
$OLD 'tmux ls 2>&1; pgrep -fa "[r]un_live_producer|[p]ublish_current_nanobody|[s]corer_daemon|[n]eurons/miner/miner.py" || echo ALL_STOPPED'
```

If any process survives, `pkill -f '[r]un_live_producer'` (and so on). Always use the bracket
form. Scorer workers are children of the daemon, so check `nvidia-smi` shows no process.

### 3. Final state sync, old to new

```bash
# 3a. fold the WAL into the main file now that no writer is left
$OLD 'python3 -c "import sqlite3;c=sqlite3.connect(\"/root/artifacts/nanobody_mutations.sqlite\");print(c.execute(\"pragma wal_checkpoint(TRUNCATE)\").fetchone());c.close()"'
# expect (0, 0, 0)

# 3b. delta-sync the DB against the pre-staged copy, then all other state
$OLD "E='$PUSH_E'; rsync -a --inplace --no-whole-file -e \"\$E\" /root/artifacts/nanobody_mutations.sqlite root@8.17.147.152:/root/artifacts/nanobody_mutations.sqlite"
$OLD "E='$PUSH_E'; for d in artifacts layer2_cache; do rsync -a -e \"\$E\" --exclude='nanobody_mutations.sqlite*' --exclude=nanobody_scorer.sock --exclude=nanobody_deployed_commit.txt --exclude=nanobody_chain_deployed_commit.txt --exclude='producer_argv*' --exclude='publisher_argv*' /root/\$d/ root@8.17.147.152:/root/\$d/ || echo FAIL \$d; done"
# only the newest two epoch dirs of /root/live (the rest is history the SQLite already holds)
$OLD "E='$PUSH_E'; cd /root/live && for d in \$(ls -d epoch* | sort -V | tail -2); do rsync -a -e \"\$E\" /root/live/\$d/ root@8.17.147.152:/root/live/\$d/ && echo synced \$d; done"
# (the argv files are excluded so the pre-built 8-GPU versions survive)

# 3c. verify the DB on the new box
$NEW 'rm -f /root/artifacts/nanobody_mutations.sqlite-wal /root/artifacts/nanobody_mutations.sqlite-shm; python3 -c "import sqlite3;c=sqlite3.connect(\"file:/root/artifacts/nanobody_mutations.sqlite?mode=ro\",uri=True);print(c.execute(\"pragma quick_check\").fetchone())"'
$OLD 'sha256sum /root/artifacts/nanobody_mutations.sqlite /root/artifacts/nanobody_published_sequences.txt /root/artifacts/nanobody_publisher_state.json'
$NEW 'sha256sum /root/artifacts/nanobody_mutations.sqlite /root/artifacts/nanobody_published_sequences.txt /root/artifacts/nanobody_publisher_state.json'
# all three digests must match; quick_check must print ('ok',)
```

The ledger (`nanobody_published_sequences.txt`) and the publisher state are what stop the
fleet re-publishing a sequence, so a mismatch there is a stop condition.

### 4. Relabel: old off the production label first, then new onto it

```bash
$PI "$VAST label instance 51125470 nova-nanobody-pixi-4x-standby"
$PI "$VAST label instance 53241806 nova-nanobody-pixi"
$PI "$VAST show instances --raw" | python3 -c "import json,sys;[print(i['id'],i['label'],i['actual_status']) for i in json.load(sys.stdin) if 'nanobody-pixi' in str(i.get('label'))]"
# expect exactly one nova-nanobody-pixi: 53241806 running
```

### 5. Point the argv files at 8 GPUs (new box)

```bash
# 5a. keep production's files as *.4gpu, move the pre-built, dry-parsed 8-GPU files into place
$NEW 'cd /root/artifacts && cp -p producer_argv.txt producer_argv.4gpu.txt && cp -p publisher_argv.bin publisher_argv.4gpu.bin \
  && cp -p producer_argv.8gpu.txt producer_argv.txt && cp -p publisher_argv.8gpu.bin publisher_argv.bin \
  && diff producer_argv.4gpu.txt producer_argv.txt; tr "\0" " " < publisher_argv.bin | grep -o -- "--expected-gpus [0-9]*"'
# expect the six producer changes listed under "Deviations", and --expected-gpus 8

# 5b. publication gate floor 1 -> 20 (module constant; read by both the race and the publisher)
$NEW 'cd /root/nanobody_opt && cp -p publication_gates.py publication_gates.py.pre-minobs20 \
  && sed -i "s/^    \"--minimum-observations\": \"1\",\$/    \"--minimum-observations\": \"20\",/" publication_gates.py \
  && grep -n "\"--minimum-observations\":" publication_gates.py \
  && /root/.pixi/envs/chain310/bin/python -c "import sys;sys.path.insert(0,\"/root\");from nanobody_opt import publication_gates as g;assert g.gate_defaults()[\"--minimum-observations\"]==\"20\";print(\"GATE_OK\")"'
```

Things to decide before 5b. These are flagged, not changed:

- With `--replication-rounds 20`, a finalist that survives every stage has 1 screen + 20 repeats
  = 21 observations. Any candidate pruned at a stage boundary (keep 0.65 every 5 rounds) ends
  with 6, 11 or 16, so it can never clear a floor of 20. Only the last-stage survivors are
  publishable. The publisher's provisional pass applies the same floor (it exists "so that an
  epoch always submits something"), so an epoch whose replication does not finish all 20 rounds
  publishes **nothing**.
- `--minimum-publishable-observations 8` and `--replication-reserve-blocks 152` are unchanged.
  The producer's help says to keep the first in step with `--replication-rounds`, because it
  sizes the shortlist for (value - 1) rounds. The reserve of 152 blocks was sized for 12 rounds
  on 4 GPUs. 20 rounds on 8 GPUs is roughly 10/12 of that GPU time per candidate, so it
  probably fits, but that is an estimate, not a measurement.
- Nothing else in the argv depends on GPU count. The scorer daemon shards each job over every
  GPU in `--gpus`, and adaptive budgeting derives from measured per-GPU seconds.

### 6. Start the new box, in dependency order

```bash
# 6a. chain-state miner first: it is the only writer of chain_state.json, which the publisher needs.
#     (the tmux commands below are the ones session_watchdog.sh uses)
$NEW 'cd /root && tmux new-session -d -s nova-nanobody-miner "bash /root/miner_relaunch.sh"'
# wait until chain_state.json is fresh (<150 s old) and has neurons
$NEW 'for i in $(seq 1 40); do python3 -c "import json,os,time,sys;p=\"/root/artifacts/chain_state.json\";d=json.load(open(p));a=time.time()-os.stat(p).st_mtime;print(int(a),d[\"block\"],len(d[\"neurons\"]));sys.exit(0 if a<150 else 1)" && break; sleep 15; done; tail -3 /root/artifacts/miner_supervisor.log'
# 6b. warm scorer daemon, then producer
$NEW 'cd /root && tmux new-session -d -s nova-nanobody-scorerd "bash /root/scorer_daemon_supervisor.sh"'
$NEW 'cd /root && tmux new-session -d -s nova-nanobody-producer "bash /root/producer_relaunch.sh"'
# 6c. publisher (artifact producer only; it has no --commit-command, the Pi signs)
$NEW 'cd /root && python3 -c "
import subprocess
a=[x for x in open(\"/root/artifacts/publisher_argv.bin\",\"rb\").read().decode().split(\"\0\") if x]
assert a[0]==\"tmux\" and \"--commit-command\" not in a
print(subprocess.run(a).returncode)"'
# 6d. roster watcher, then the watchdog last
$NEW 'cd /root && tmux new-session -d -s nova-rosterwatch "bash /root/roster_watch.sh"'
$NEW 'cd /root && tmux new-session -d -s nova-session-watchdog /root/session_watchdog.sh'
$NEW 'tmux ls'   # expect all six sessions
```

### 7. Verify

```bash
# producer is on 8 GPUs and screening
$NEW 'tr "\0" " " < /proc/$(pgrep -f "[r]un_live_producer" | head -1)/cmdline | grep -o -- "--gpus [0-9,]*"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader'
$NEW 'tmux capture-pane -pt nova-nanobody-producer | tail -15; tmux capture-pane -pt nova-nanobody-publisher | tail -15'
# publisher did not fail closed on the GPU count, and the artifact is fresh
$NEW 'tmux capture-pane -pt nova-nanobody-publisher -S -200 | grep -i "expected .* GPUs" || echo no_gpu_gate_failure; ls -la --time-style=+%T /root/artifacts/nanobody_result.json'
# Pi resolves the new box and pulls from it: look for event=pulled with port 46916 (direct) or 11806 (proxy)
$PI 'journalctl -u nova-submit-loop --since "-20 min" --no-pager | grep -E "\"event\": \"pulled\"|expected exactly one|pull from" | tail -5'
# push hotkeys by label now rather than waiting for the timer (idempotent)
$PI 'sudo systemctl start nova-hotkey-sync.service; systemctl status nova-hotkey-sync.service --no-pager | tail -3'
```

After the first pull, watch one full epoch: the Pi commits the nanobody half for epoch N+1,
and `roster_watch.log` on the new box shows `SYNC-OK` at the boundary.

### 8. Re-enable the Pi's producer guard (it now guards 53241806)

```bash
$PI 'rm -f ~/.nova/producer-autostart.disabled'
$PI 'journalctl -u nova-nanobody-producer-guard --since "-5 min" --no-pager | tail -3'
```

### 9. After 2-3 good epochs

- Remove production's push key from the new box:
  `$NEW 'sed -i "/root@4b489d5c6ae1\$/d" /root/.ssh/authorized_keys'`
- Only then destroy 51125470. Do not leave it labelled anything the Pi resolves. A queued
  `vastai start` can bring a stopped box back.
- Fix the repo: commit the fast-scorer `model/models/` files, the supervisor scripts and the
  single-prefix publisher argv. This file lists everything that is still hand-deployed.

## Rollback (any failure in steps 3-7)

```bash
$PI 'mkdir -p ~/.nova && touch ~/.nova/producer-autostart.disabled'
# 1. stop the new box (watchdog first)
$NEW 'tmux kill-session -t nova-session-watchdog; for s in nova-nanobody-producer nova-nanobody-scorerd nova-nanobody-publisher nova-nanobody-miner nova-rosterwatch; do tmux kill-session -t $s; done; tmux ls 2>&1'
# 2. swap the labels back
$PI "$VAST label instance 53241806 nova-nanobody-pixi-8x"
$PI "$VAST label instance 51125470 nova-nanobody-pixi"
# 3. if the new box published anything, carry its ledger/state/DB back so nothing is re-published
#    (the new box has no key into the old one: stream through the Mac)
$NEW 'tar -C /root/artifacts -cf - nanobody_published_sequences.txt nanobody_publisher_state.json nanobody_result.json nanobody_history' | $OLD 'tar -C /root/artifacts -xf -'
#    (the SQLite: only if the new producer ran long enough to matter; checkpoint it on NEW, then stream it back the same way)
# 4. restart the old box. Its session watchdog restarts miner, producer, scorerd and rosterwatch
#    from the unchanged 4-GPU argv. Its publisher restart path is broken (see above), so start
#    the publisher explicitly with the single-prefix command:
$OLD 'cd /root && tmux new-session -d -s nova-session-watchdog /root/session_watchdog.sh; sleep 90; tmux kill-session -t nova-nanobody-publisher 2>/dev/null; python3 -c "
import subprocess
a=[x for x in open(\"/root/artifacts/publisher_argv.bin\",\"rb\").read().decode().split(\"\0\") if x]
pre=[\"tmux\",\"new-session\",\"-d\",\"-s\",\"nova-nanobody-publisher\"]
while a[:5]==pre and a[5:10]==pre: a=a[5:]
print(subprocess.run(a).returncode)"; sleep 5; tmux ls'
$PI 'rm -f ~/.nova/producer-autostart.disabled'
```

Old-box state is untouched by steps 2-3 apart from the WAL checkpoint, so rollback needs no
data restore unless the new box published.
