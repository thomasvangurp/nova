# Deploying the SN68 nanobody miner

One branch, one inventory, one command. Three services:

| service | where | tmux session | playbook |
| --- | --- | --- | --- |
| design/scoring producer | GPU box | `nova-nanobody-producer` | `nanobody.yml` |
| epoch publisher | GPU box | `nova-nanobody-publisher` | `nanobody_services.yml` |
| on-chain commitment writer | GPU box | `nova-nanobody-miner` | `nanobody_services.yml` |
| slot purchase agent | **operator's machine** | `nova-nanobody-slots` | `slots.yml` |

```bash
# once per machine
ansible-galaxy collection install -r ansible/requirements.yml

# everything, in dependency order
ANSIBLE_CONFIG=ansible/ansible.cfg ansible-playbook \
  -i ansible/inventory/vast_nanobody.py ansible/site.yml \
  -e @ansible/vars/nanobody-high-throughput.yml \
  -e @ansible/vars/nanobody-identity.yml
```

`ANSIBLE_CONFIG` is not optional: `ansible/ansible.cfg` turns on SSH pipelining,
connection multiplexing and `transfer_method = piped`. On this link (~0.8 s round
trip through the vast.ai proxy) an sftp put of a 3-byte file measured 10 s versus
5 s piped over the existing session.

## Why it is shaped this way

Three failures paid for this directory's structure, and each one is a rule here.

### One runtime root, because two cost epochs

Until 2026-09-12 the producer ran from `/root`, deployed from `nanobody-opt`,
while the chain publisher ran from a **second full copy** at
`/root/nova-standalone`, deployed from `deploy/nanobody-standalone` and carrying
its own Pixi workspace. Publication gates were relaxed in `/root/nanobody_opt`
and nothing changed for hours, because the program that publishes was reading the
other copy.

So: `ansible/tasks/sync_repo.yml` is the single definition of what lands on the
box and where, `/root` is the only runtime root, `/root/nova-standalone` is
**deleted** by both GPU playbooks, and `chain310` is an environment in this
repository's own `pixi.toml` rather than a workspace somewhere else.

`config/config_loader.py` loads `config/config.yaml` **relative to the process
working directory**, which is why the cwd of every service is `/root` and why the
second root could win silently. The sync excludes `config/config.yaml` and
`config/boltzgen_config.yaml` so the committed copies can never overwrite the
live validator documents Ansible downloads.

### The endpoint is discovered, never written down

vast.ai rotates the SSH endpoint on every instance restart. On 2026-09-12 the
port moved from 21300 to 20868, every command aimed at the old port timed out,
and the conclusion drawn was "every box is down". A recorded port is not a
convenience, it is a latent false outage.

`ansible/inventory/vast_nanobody.py` reads `vastai show instances --raw` and
selects by **label**, because the label is the one identifier an operator chooses
and a rebuild preserves:

```bash
vastai create instance <offer> --label nova-nanobody-pixi ...
vastai label instance <id> nova-nanobody-pixi     # on an existing box
```

It produces `nanobody_producer`, `nanobody_publisher`, and a `slot_manager` group
containing one local host, `operator`. Every GPU playbook imports
`require_producer.yml` first, which fails loudly and prints the endpoint it found
— an empty group otherwise prints "skipping: no hosts matched" and exits 0, and a
green deploy that touched nothing is the worst possible report.

Only an allowlist of instance fields becomes host variables. The vast.ai instance
document also carries `jupyter_token` and ssh key material, and host variables are
printed by `ansible-inventory` and echoed under `-vvv`.

Environment overrides: `NOVA_VAST_LABEL`, `VASTAI_BIN`,
`NOVA_VAST_INSTANCES_FILE` (a saved document, which is how
`tests/test_vast_inventory.py` exercises port rotation with no account), and
`NOVA_VAST_ANY_STATUS=1` to include non-running instances.

`gpu_count` comes from the rental's `num_gpus`, with no fallback anywhere: the
publisher treats the GPU count as a hard gate, and a default would substitute a
guess for the one fact the rental knows.

### N hotkeys, and not one of them is configured

We hold many uids. A configured hotkey or uid list goes stale exactly the way a
remembered uid does — the miner published into "our uid 8" for twenty-one days
after that slot became somebody else's.

So the only identity in configuration is `nanobody_coldkey_ss58`, a public
address. The publisher discovers every uid that coldkey holds from the chain's own
neuron rows each epoch. `nanobody_hotkey_sources` is a **list** of controller-side
keyfile paths; the deployment installs however many there are and asserts a count,
never a set of expected addresses.

## The safety gate

`ansible/tasks/verify_entrypoint.yml`, and every service goes through it.

It runs `--help` on the entry point and then asserts that **every flag the
deployment passes appears in the help text** — before anything running is
stopped. Both halves matter:

- `--help` exercises every import on the entry point's path. An adapter refactor
  once left `score_boltzgen_fast_adapter` raising `ModuleNotFoundError`; because
  the probe runs before `tmux kill-session`, the deploy failed with the live
  producer still mining instead of taking the miner down.
- "the module imports" is not "the arguments exist". A flag renamed upstream
  otherwise starts the service, fails argparse, and the session exits seconds
  after the deploy reported success.

The argument lists are declared as data (`set_fact`) so the same list is both
what the gate verifies and what tmux starts. Typing them twice is how a verified
flag and a passed flag drift apart. Currently gated: the exact scorer's contract
flags, 42 producer flags, 20 publisher flags, 10 slot-agent flags.

A consequence worth knowing: because each gate is `--help`-based, every entry
point must be `--help`-able with no chain connection, no wallet and no Keychain.

Three gates are not `--help`-shaped, and each exists because a specific thing
crossed programs and could only fail at runtime:

- **the chain-state seam.** The publisher reads block, epoch and the neuron rows
  it discovers the fleet from out of `chain_state.json`, which only the connected
  miner writes, and it has no fallback to a remembered uid. So
  `tests/test_chain_state.py` and `tests/test_publish_current_nanobody.py` run on
  the box against the deployed tree, and a source check confirms the deployed
  miner still *calls* `nanobody_opt.chain_state` — which a passing test of that
  module cannot prove.
- **the miner's imports.** `py_compile` proves the miner parses, not that the
  chain environment can satisfy its imports, so the deployed miner module is
  imported in `chain310`. It is import-safe by construction: module level is
  only `sys.path` setup and `getattr` aliases, everything else is behind
  `__main__`.
- **dead configuration.** `/root/nanobody.env`'s keys are cross-checked against
  the deployed source, and any key nothing reads fails the deploy. The first
  draft of that file carried nine keys inherited from the other branch's miner
  (`NOVA_NANOBODY_ONLY`, `NOVA_NANOBODY_RESULT`, `NOVA_LIVE_CONFIG`,
  `NOVA_WALLET_*`, `NOVA_COLDKEY_SS58`, `NOVA_GIT_PUSH`, `NOVA_DEPLOY_KEY`,
  `NOVA_GIT_WORKDIR`, `AUTO_UPDATE`) that this branch's miner reads none of.
  Configuration nothing reads is worse than absent: it is a written record of
  behaviour that is not happening.

## Credentials

- **No coldkey ever reaches the box.** Both GPU playbooks `find` for coldkey
  material under the wallet path and fail if any exists, in every mode. A coldkey
  on a rented box is an unrecoverable loss, and "it is only the scoring box" is
  exactly the assumption under which one gets copied there.
- The slot purchase agent therefore runs on the **operator's own machine**. Its
  coldkey password comes from the macOS Keychain, and the only
  credential-adjacent string anywhere in this directory is
  `NOVA_COLDKEY_KEYCHAIN_SERVICE`: the *name* of an environment variable that in
  turn *names* a Keychain service. There is no `security find-generic-password`
  call in any playbook or vars file. Export the variable in the shell you run
  `slots.yml` from — the tmux session inherits it, and `slots.yml` asserts its
  presence without reading its value.
- **The deployed slot agent is a dry run.** `--execute-registrations` is the flag
  that spends tau and it is not passed, under any variable. `slots.yml` asserts
  that it is absent from the argument list. Turning on live purchasing is a
  deliberate manual act in the tmux session; a variable that can flip it is a
  variable that will be flipped by a copied command line.
- Hotkey keyfiles and the GitHub submission token are passed as controller-side
  **paths**. Ansible reads them locally, the tasks that write them are `no_log`,
  and no key material passes through a command line, a variable or task output.
  It is a token rather than an SSH deploy key because `utils/files.py` — the
  uploader the deployed miner calls — PUTs to `api.github.com/.../contents` with a
  bearer `GITHUB_TOKEN`. The deploy-key-and-git-push path belonged to the other
  branch's `files.py`.

## The vendored fast scorer

`external_tools/fast_nanobody_scorer/` is the ~28x exact scorer. It was built on
a rented box, lived outside version control for four days, and survived that
instance being destroyed by minutes. Read its `PROVENANCE.md`.

`nanobody.yml` deploys it to
`/root/fastscore/pinned-<first 12 of the sha256>/` — named after the digest so an
unpinned or differently pinned build cannot quietly occupy the same path — then:

1. re-hashes `fast_score_batch.py` on the box and asserts it equals
   `530f5dcbc54d…`, because `nanobody_opt/score_boltzgen_fast_adapter.py` refuses
   a scorer whose digest differs and would otherwise discover that mid-epoch;
2. asserts the `--synth-reference` design is present, since without it the
   adapter silently falls back to the slow-but-correct diffusion design stage —
   which is not the 28x path;
3. renders `/root/artifacts/nanobody_fast_scorer.json` with the pin, the launcher
   and the reference, so those three cannot drift apart;
4. links `/root/fastscore/release` at the deployed digest. The committed
   `release_python.sh` hardcodes `PYTHONPATH=/root/fastscore/release/boltzgen_src`;
   that path was a working directory on the destroyed box, and without the link
   the launcher would import the *live vendored* BoltzGen instead of its own
   pinned copy — silently, and only under load.

Per-host tuning goes in `nanobody_fast_scorer_tuning` (merged into the rendered
settings), empty by default so the scorer runs as validated.

## The trap that locks you out of the box

`ansible.posix.synchronize` is `rsync -a`, and `-a` preserves **owner, group and
mode from the controller** — including for the destination directory when the
source path ends in `/`. Syncing a laptop checkout (uid 502) into `/root/`
chowns `/root` to `502:staff`, and sshd's `StrictModes` then refuses **every new
public-key auth**.

It does not look like that. Existing multiplexed SSH sessions keep working, so
`ssh root@host` from the same shell still succeeds while every `rsync` dies with
`Permission denied (publickey)`. It reads like a flaky proxy or a bad key. By the
time the last mux socket expires you are locked out of a box you cannot log into
to fix.

Every rsync here carries `owner: false`, `group: false`, `perms: false`, and
`nanobody.yml` repairs `/root` ownership on every run. If you write another one,
carry those three flags.

## Idempotence

A second run against an untouched box should be `changed=0`. What makes it so:

- `checksum: true` on every rsync
- `creates:`/`--locked` guards on the Pixi installs, which report
  `Nothing to do`
- `modification_time: preserve` on the `combinatorial_db/__init__.py` touch, so
  it does not report changed forever
- `changed_when` on the checkpoint reaper keyed to the fetcher's `DOWNLOADED=0`

Anything reporting `changed` on a second run is a bug here, not a quirk: a deploy
that always says "changed" cannot be used to tell real drift from a no-op.

## The other playbooks

| playbook | what it does |
| --- | --- |
| `site.yml` | all three services, in order |
| `nanobody.yml` | provision the GPU box; producer; fast scorer; canaries |
| `nanobody-fast.yml` | validate a prebuilt OCI runtime and Vast cache volume, then import `nanobody.yml` |
| `nanobody_services.yml` | chain publisher and commitment writer |
| `slots.yml` | slot purchase agent, on the operator's machine |
| `publish_nanobody_epoch.yml` | publish one named epoch by hand |
| `validator-parity.yml` | install the validator's documented scoring environment without touching the live producer |
| `lock.yml` | resolve `pixi.lock` on x86 Linux and fetch it back |

`lock.yml` is not optional after a `pixi.toml` change: the chain environment's
`timelock` wheels are cp310-only and the git-backed TNP/ImmuneBuilder builds
cannot be cross-resolved on an ARM macOS controller. `nanobody_services.yml`
checks that `pixi.lock` actually describes `chain310` and says so, because a
`--locked` install of an environment the lock does not describe fails with a
resolver message that reads like a network problem.

## What is deliberately not here

The Compound (small-molecule) suite — `provision.yml`, `cluster.yml`, `live.yml`,
`replay.yml`, `diagnose-score.yml` and `files/root/*` — stayed on
`deploy/nanobody-standalone`. It depends on ~20 compound-only modules that are
not on this branch (`tools/score_boltz_distributed.py`, `compound_miner/proxy.py`,
`bootstrap_bz2.sh`, `preflight.sh`, `tools/gate_audit.py`,
`compound_miner/gpu_novelty.py`, the `bz2` Pixi environment, the 6.2 GB Boltz
cache), so copying it here would add 1,900 lines of playbook that cannot run.
Its transferable lessons — assert on artifacts rather than exit codes, the
`/root` ownership trap, the idempotence rules above — are folded into this file
and into the playbooks.

## Asserting on artifacts, not exit codes

`ssh` returns 0 for a remote command that died, `curl` returns 0 for a file it
truncated, and `tar` returns 0 for an archive it only half-extracted. Every one
of those has shipped a broken box that looked green. So the checks here are byte
counts, sha256 digests, file counts, help text, and an interpreter that actually
imports `torch` and sees the GPUs.

`--check` is not supported: in check mode the async checkpoint download never
starts, and the assertions that follow it then fail on a cold box for a reason
that has nothing to do with the box. Re-running the real playbook is the dry run.
