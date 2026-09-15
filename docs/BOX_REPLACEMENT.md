# Replacing the GPU box

Written after a replacement took roughly four hours and cost about eight epochs.
Almost none of that was the work; it was discovering the order in which things
have to happen. Followed straight through this is about 40 minutes, most of it
waiting for a checkpoint download.

## The short version

```bash
# 1. rent (check credit first -- "Failed with error 400: Your account lacks credit")
vastai search offers 'num_gpus=2 gpu_name=RTX_4090 reliability>0.97 inet_down>300 disk_space>250 rentable=true' -o dph+
vastai create instance <id> --image nvidia/cuda:12.6.0-devel-ubuntu22.04 \
  --disk 220 --label nova-nanobody-pixi --ssh --direct --cancel-unavail
#    NEVER pass --raw to create: it prints an instance API key.
#    Check cur_state, not actual_status.

# 2. retire the old box's label BEFORE deploying, or the inventory resolves both
#    and the playbook runs against the dead one
vastai label instance <old-id> nova-nanobody-retired

# 3. push the hotkeys -- a fresh box has none, and the deploy stops without them
ssh thomasg@raspberrypi sudo systemctl start nova-hotkey-sync.service

# 4. pick a chain writer that is REGISTERED RIGHT NOW (see below), then:
cd ~/nova-nanobody
NOVA_CHAIN_WRITER_HOTKEY=<name> bash deploy/nanobody_deploy.sh nanobody.yml
NOVA_CHAIN_WRITER_HOTKEY=<name> bash deploy/nanobody_deploy.sh nanobody_services.yml

# 5. verify all THREE sessions exist and chain state is being written
ansible vast<id> -i inventory/vast_nanobody.py -m shell -a "tmux ls; ls -l /root/artifacts/chain_state.json"
```

## The five things that actually cost the time

### 1. Always use deploy/nanobody_deploy.sh, never ansible-playbook directly

The script sets `nanobody_producer_activate`, `nanobody_chain_writer_activate`
and the coldkey. Calling the playbook directly produces a **green deploy with
nothing running** -- the run reports `failed=0` and no producer or miner exists.
This happened twice in one evening.

`nanobody.yml` starts the miner and producer. `nanobody_services.yml` starts the
publisher and the chain writer. Both are needed; neither is sufficient.

### 2. The chain writer must be a hotkey registered AT THIS MOMENT

`resolve_chain_writer.py` reads `/root/artifacts/chain_state.json`, which the
miner writes -- so on a fresh box it can never resolve and you must name one.
But name a stale one and the miner dies instantly with

    ValueError: '5Eh7...' is not in list

from `metagraph.hotkeys.index(...)`. The fleet churns constantly: a hotkey that
was uid 9 an hour ago may be gone. Get a live one:

```bash
ansible vast<id> -i inventory/vast_nanobody.py -m shell -a "/root/.pixi/envs/chain310/bin/python -c \"
import bittensor as bt, json, glob, os
mg=bt.Subtensor('finney').metagraph(68)
CK='5EkCwcNTfxeNbbhNSEJMSaGbW2XMVuhzouZkEan5ozvSsJdQ'
onbox={json.load(open(f))['ss58Address']: os.path.basename(f)
       for f in glob.glob('/root/.bittensor/wallets/*/hotkeys/*') if not f.endswith('pub.txt')}
for i in range(len(mg.hotkeys)):
    if mg.coldkeys[i]==CK: print(i, onbox.get(mg.hotkeys[i], 'NOT ON BOX'))
\""
```

### 3. ImmuneBuilder weights are zero-byte placeholders

Now gated by "Ensure the NanoBodyBuilder2 weights are present and complete" in
nanobody.yml, so this should not recur. If it does: the four files are NOT the
same size (model_1 ~61 MB, models 2-4 ~214 MB), so verify against
`Content-Length` and never a size floor. Symptom is the Layer-2 canary failing
with `TNP: 0/4 pass` and `PytorchStreamReader failed reading zip archive`.

### 4. The deploy refuses a dirty tree, including untracked files

`?? deploy/slot_host/backup_artifacts.sh` was enough to fail the run. Commit
everything first; `git status --porcelain --untracked-files=all` must be empty.

### 5. Rescue the old box before it goes

If the old instance is still listed, even `exited`, try starting it. Ours came
back and still held `/root/live/epoch*` and a **200 MB mutation database with
54,778 observations** -- the private screening history, far better than anything
rebuildable from the API. It has no `sqlite3` CLI, so snapshot with Python:

```bash
ansible vast<old> -i inventory/vast_nanobody.py -m shell -a "python3 -c \"
import sqlite3
s=sqlite3.connect('/root/artifacts/nanobody_mutations.sqlite')
d=sqlite3.connect('/root/rescue.sqlite'); s.backup(d); d.close()\""
ansible vast<old> -i inventory/vast_nanobody.py -m fetch -a "src=/root/rescue.sqlite dest=./rescued.sqlite flat=yes"
```

Use `.backup`, not `cp`: the store is WAL and a copy taken mid-write restores
torn. `deploy/slot_host/backup_artifacts.sh` does this on a schedule so the next
replacement does not need a rescue at all.

## What is safe and what is not

| | survives a box death |
|---|---|
| code | yes, git |
| hotkeys | yes, on the Pi, pushed by label |
| target archive | yes, re-downloads |
| public mutation observations | yes, `pixi run seed-mutation-database` from the leaderboard API |
| **our own exact screens** | **only if backed up** |

## Cost traps

- A queued `vastai start` on a dead instance can fire hours later and bring the
  box back, billing and carrying the `nova-nanobody-pixi` label. Retire the
  label when you stop using a box.
- `pkill -f <pattern>` from an Ansible ad-hoc shell matches Ansible's own
  command line and kills the session with `rc=-15`. Use `pkill -f '[p]attern'`.
