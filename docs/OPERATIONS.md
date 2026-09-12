# Nanobody miner operations handbook

Subnet 68 (Nova), nanobody track, target P20809. This is the working reference
for running the miner: what the reward actually rewards, which numbers are
measured rather than assumed, how the pieces fit, and the failures that have
already cost us epochs.

---

## 1. The strategic situation, in one page

**Reward is proportional to tickets, not to molecule quality.** Measured over
epochs 25030–25062: 1,379 submissions, 16 coldkeys holding 70 hotkeys, and 68 of
those 70 belong to a coldkey running more than one. Wins per coldkey against
share of submissions gives **χ² 9.9 on 15 df, p ≈ 0.82** — proportional to a
degree that is almost too clean. One operator submitted 190 times and won once;
another submitted 104 times and won five.

**A single score is mostly noise.** Two independent observations of the same
molecule correlate **0.164** on the validator's own sample-selection objective.
Averaged over the field, skill explains roughly **2% of the variance** in where a
miner places. Epoch 25062 was won by a molecule that drew 85 rank points better
than its own three-observation average and ranked 13th of 46 on averaged quality.

**Winning nothing in 16 appearances was the expected outcome.** Modelling each
submission's chance of being the field maximum from its realised percentile gives
0.15 expected wins across those epochs — P(≥1 win) was 14%. Nothing was broken.

**Consistency and tickets are worth the same.** Our win probability came almost
entirely from two epochs (rank 3 of 48 → 8.1%; rank 4 → 2.9%). Eleven of sixteen
were worth 0.00%, because being the maximum is convex — rank 13 of 50 is worth as
little as rank 45. Sixteen epochs at our own demonstrated best would yield 1.29
expected wins; nine tickets yields 1.30.

**More compute did not help.** Our two best epochs predate the high-throughput
profile. Mean normalised rank went 0.347 → 0.640 across it (t +1.83, n=4 after —
direction only, not proof). The lean profile costs a sixth as much to test.

### What follows from this
- Hold as many hotkeys as the slot mechanics allow (**9–13**; see §2).
- Submit a **distinct** molecule per hotkey — the validator dedupes by sequence,
  so the same molecule on nine hotkeys buys nothing.
- Prefer consistency over peak quality; a mediocre epoch is worth zero.
- Do not buy GPUs expecting rank improvements the data does not support.

---

## 2. Measured constants

Everything below was measured on-chain or from the leaderboard API, not assumed.
Re-measure before trusting any of it after a subnet upgrade.

| quantity | value |
|---|---|
| epoch | `block // 361`; 361 blocks × 12 s = 72.2 min, 19.94 epochs/day |
| commitment reserve | 55 blocks (660 s) at epoch end |
| uid cap | `MaxAllowedUids` 64, currently full; 6 validator permits → ~58 miner slots |
| immunity | 5000 blocks = 16.7 h = 13.85 epochs |
| incentive | winner-take-all: one uid carries all miner emission |
| winner's emission | 0.431 α/block = 155.6 α/epoch = τ3.35, or **τ1.34** at the 40% nanobody share |
| registration (recycle) | ~τ0.10–0.14, moving |
| **recycle dynamics** | **decays −0.19%/block, halving every 360 blocks**; jumps on registration (+33% observed from three competitors' buys) |
| registration throttle | `max_regs_per_block` 1, `target_regs_per_interval` 1 per 360 blocks |
| fast scorer | 6.23 GPU-s/sequence (vs 176 for the upstream path) |

`LastAdjustmentBlock` reads ~1M blocks stale, so the textbook per-interval burn
adjustment is **not** what drives `Burn` here. Trust the measurement.

### GPU sizing
At 6.23 GPU-s/sequence and 3,672 usable seconds per epoch:

| profile | sequences | GPUs | $/day | break-even TAO |
|---|---|---|---|---|
| live (2800 screen, 300×12 replicate) | 5,152 | 8.7 | $65 | $13.53 |
| mid (1400 screen, 120×6) | 1,976 | 3.4 | $25 | $5.19 |
| **lean 9-ticket (700 screen, 60×3)** | **880** | **1.5** | **$11** | **$2.31** |

Nine tickets need nine distinct qualifying sequences per epoch, which a
700-candidate screen produces comfortably. **2–4 GPUs is the right rig.**

---

## 3. Architecture

Three roles, three hosts, one branch (`nanobody-opt`), one runtime root.

**GPU rental — producer, ranker, publisher.** Rented from vast.ai, labelled
`nova-nanobody-pixi`. Everything deploys to `/root`; there is no second root.
Environments: `bgen` (scorer), `metanano` (Layer 2), `chain310` (publisher).
Holds **hotkeys only — never the coldkey**; both GPU playbooks assert this.

**Purchase host — an always-on arm64 node.** Holds the coldkey and buys
registrations. It is not the GPU box (a coldkey on rented hardware is an
unrecoverable loss) and not a laptop (the recycle price halves every 72 minutes,
so buying cheaply means being awake for the quiet intervals). Its environment is
`deploy/slot_host/pixi.toml`, separate from the workspace root so adding
linux-aarch64 cannot disturb the GPU lock.

**Controller — the operator's Mac.** Runs Ansible. Holds no secrets; every
credential is passed as a *locator* (a path, or the name of an environment
variable), never a value.

### Identity is resolved from the chain, never from configuration
The publisher discovers every uid our coldkey holds from the metagraph each
epoch, and maps each on-chain hotkey to a local keyfile. There is deliberately
**no configured hotkey or uid list**, because uids are recycled on
deregistration and a stale list fails silently — see §5.

The miner exports its chain view to `chain_state.json` (schema 2, with
`neurons`) so the publisher never opens a second WebSocket and identity and
block always come from one snapshot.

---

## 4. Runbook

### Deploy from scratch
```bash
export ANSIBLE_CONFIG=ansible/ansible.cfg
INV=ansible/inventory/vast_nanobody.py     # discovers the box from vast.ai by label

ansible-playbook -i $INV ansible/site.yml \
  -e @ansible/vars/nanobody-high-throughput.yml \
  -e @ansible/vars/nanobody-identity.yml
```
The inventory reads `vastai show instances --raw` and selects on the label, so a
rotated SSH port is never an outage. Every playbook refuses to run against an
uncommitted tree.

### Rent a box
```bash
vastai search offers 'num_gpus=4 gpu_name=RTX_4090 cpu_arch=amd64 disk_space>=200 reliability>0.98' -o dph
vastai create instance <id> --image nvidia/cuda:12.6.0-devel-ubuntu22.04 \
  --disk 300 --label nova-nanobody-pixi --ssh --direct
```
Do **not** pass `--raw` to `create instance`: it prints an instance API key.

### Re-lock after changing pixi.toml
`pixi.lock` cannot be cross-resolved on ARM macOS. Rent an x86 box, then
`ansible-playbook -i $INV ansible/lock.yml` (≈90 s). Lock the purchase host's
manifest *on the Pi* instead, natively.

### Purchase host
Environment variables select it: `NOVA_SLOT_HOST`, `NOVA_SLOT_ADDRESS`,
`NOVA_SLOT_USER`. Setting `NOVA_SLOT_HOST=operator` moves the purchaser back to
the Mac — correct only while the always-on node is down, since buying locally
beats not buying.

The coldkey password is a **systemd encrypted credential**, created on that host
because the encryption is host-bound:
```bash
systemd-ask-password -n | sudo systemd-creds encrypt \
  --name=nova-coldkey-password - /etc/nova/coldkey-password.cred
sudo chmod 0600 /etc/nova/coldkey-password.cred
```
`LoadCredentialEncrypted` rather than `Environment=` is the point: the value
never appears in the unit, in `systemctl show`, in `/proc/<pid>/environ`, or in a
child's core dump.

### Enabling live purchasing
The deployed unit is a **dry run**; the playbook asserts `--execute-registrations`
is absent. Turning on spending is a deliberate manual act. Rails that stay on:
`--max-recycle-tao`, `--max-spend-tao-per-day`, one registration per adjustment
interval, and a stagger of `immunity ÷ target` between purchases.

### Buying strategy
Ramp at roughly one registration per quiet interval until nine are held, then
throttle to replacement only (~1 per 8 h at a ~58-epoch expected lifetime).
Buying nine at once is the worst case: each jump compounds with no decay between
them, and nine bought together all leave immunity at the same block, producing a
self-inflicted price spike every 16.7 h forever.

---

## 5. Failures that have already cost us

**Mining into a slot we no longer held — 21 days.** Both hotkeys were
deregistered and `uid 8` belonged to another operator, while every diagnostic
keyed on "our uid" followed the new owner. *Check registration before diagnosing
anything else.* Identity comes from the coldkey or from matching our own
published-sequence ledger against leaderboard rows.

**Four copies of the publication gates.** Two in a second, unversioned tree, plus
a third and fourth in `publish_checkpoint` and `nanobody_provider` that were
*stricter* than the live values. Relaxing the gates changed nothing because the
program that publishes read a different copy, and even then the writer and
consumer would have refused. All four now read `nanobody_opt/publication_gates.py`.

**A relative config path.** `config_loader.load_config` defaults to
`config/config.yaml`, resolved against the process cwd — which is *why* a second
runtime root could silently win. Every service now runs `chdir=/root`.

**A gate that could not fail.** The deploy checked the miner for the string
`neurons`, which matches `metagraph.neurons` in any miner. It passed by accident.
A gate must be verified to fail against the broken state.

**A canary that failed on a healthy box.** Layer-2's first run errored under
contention with the async checkpoint download, and the failure looked like a
molecule verdict (`{"passed": false, "reason": "Error: "}`). It is not: an empty
`reason` is TNP erroring, a threshold failure names the threshold. Re-run on a
quiet box before believing it.

**A vendored launcher hardcoding a dead path.** `release_python.sh` points
`PYTHONPATH` at a directory that existed only on a destroyed box; without a
symlink at the pinned digest it silently imports the *live* BoltzGen instead of
the fast one.

**Losing the box's only copy.** The fast scorer, the publisher and the mutation
database all lived only on rented hardware. Snapshots now live in
`snapshots/vast-20260912/`; the scorer is vendored at
`external_tools/fast_nanobody_scorer/` and pinned by sha256.

---

## 6. Open items

- **We hold no registered hotkeys.** Everything else is blocked behind this.
- The purchase host needs the coldkey, its encrypted credential, and an explicit
  decision to enable spending.
- The post-25054 quality regression rests on four epochs. The clean test is to
  run the lean profile for two days and compare normalised rank.
- Registration is priced 4–15× below a ticket's EV. That gap is what makes
  ticket-stacking pay, and it is the sort of thing a market closes.
