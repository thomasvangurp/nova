# Compound (SN68) strategy — state at 2026-09-20 00:45 UTC

## What the epoch data says (this is new, and it changed the plan)

Public, unauthenticated leaderboard API — no credentials needed:

    https://compound-api.metanova-labs.ai/api/competitions/molecules?limit=10000
    https://compound-api.metanova-labs.ai/api/competitions/leaderboard/{epoch}/molecules?limit=10000&molecule_detail=full

200 P40261 epochs (25021–25220) saved under the session scratchpad. `final_score`
per UID is the SUM of its 20 molecule scores, directly comparable to our `best20`.

**A regime break at epoch 25176.** Median scored UIDs per epoch fell 25 → 15
(rxn 3 down to 10), and winner scores fell with it:

    rxn 3 winners before 25176   1.71 – 1.80
    rxn 3 winners since  25176   1.49 – 1.67   (one outlier, 25176 itself, 1.889)

**Reactions have different score SCALES, not different difficulty.** Winner
medians since 25176: rxn 1 1.626, rxn 2 1.951, rxn 3 1.576, rxn 4 1.803,
rxn 5 1.895. Never compare a score across reactions. Normalised by field median,
rxn 3 has the TIGHTEST winner-to-field gap (5.7% vs 9.9% for rxn 5) — we are
banking the most contested reaction.

## Target: 60,000, not 215,000

    bank      best20   realised   beats recent rxn3 winners   extra $
    19,799    1.6437   1.5877            6/12                     0
    40,000    1.7012   1.6452           10/12                   $99
    60,000    1.7344   1.6784           11/12                  $197
    215,000   1.8388   1.7828           11/12                  $958

`realised = best20 - 0.056`, the measured winner's-curse correction (naive best20
is a max statistic; the validator rescores fresh). Above 60,000 buys NOTHING
against the current field — the 215,000 target was $761 and 5.4 days for zero
extra wins. The only unbeaten winner is 25176's 1.889, needing ~792,000.

RISK: this is calibrated to the post-25176 weak field. If participation returns
to ~25 UIDs, winner median goes back to ~1.75 and we would need ~137,000. Banks
are cumulative, so extending costs nothing extra — watch winner scores per epoch
and extend if they rise. Decision (user, 2026-09-19): one reaction first, extend
later.

## Live infrastructure

    box1  51598530  4x L40S  $2.835/hr  92.180.21.67:40627
    box2  51620065  4x L40S  $3.204/hr  83.32.208.134:40054
    both: bank_builder.py --target-total 60000 --batch 400 --anchor-pool 20000
          box2 additionally --seed 1969
    NIM box 51335750 stopped, zero cost. Production 51125470 — DO NOT TOUCH.

    Mac keeper: tools/bank_keeper.sh, detached, PID in state/bank_keeper.pid
      backup every 30 min, sync every 40 min, log logs/bank_keeper.log
      MUST run bank_backup.py / bank_sync.py under /usr/bin/python3 — they are
      pure stdlib, and `pixi run` resolves the ROOT workspace which declares only
      linux-64, refuses on macOS, AND EXITS 0. Two "successful" runs copied
      nothing before this was caught.

    throughput 20.5 molecules/min combined, $6.04/hr

## Builder: what was wrong and what fixed it

1. SWEEP COLLAPSE. `--anchor-pool 4000` drew anchors only from the top 4,000 by
   score — the most-mined pairs. As a neighbourhood is exhausted the analogues
   that REMAIN are the ones that fail the gate, so survival fell to 2.2% and the
   sweep produced 10 molecules/round. `--anchor-pool 20000` restored it to 516.

2. COVERAGE SURVIVAL. The builder's comment claimed uniform draws clear
   validity+novelty ~82% of the time. Measured: 5.8% in a starved configuration,
   26% in a healthy one. Now tracked as an EWMA (`coverage_survival`), never
   assumed.

3. OVER-GENERATION WAS INERT. Coverage was sized against draws, not survivors, so
   every survivor was needed to fill the batch and the proxy had nothing to
   select between. Now sized against the slots the sweep leaves unfilled.
   Measured payoff: proxy ranks held-out coverage draws at +0.4228 Spearman —
   BETTER than it ranks sweep molecules (+0.2805), because coverage spans a wide
   quality range while analogues of a good anchor sit in a narrow band. Keeping
   the best tenth lifts coverage mean 0.01724 -> 0.03063.

4. SEARCH-PHASE GPU IDLE. Round 1 spent 287.7s single-threaded with all 4 GPUs at
   0%. Adaptive sizing cut round 2 to 55.7s (11% -> 2.3% of wall clock).
   compound_miner/search.py has a ProcessPoolExecutor construction path the
   builder still does not use — remaining ~2% is recoverable there.

Healthy round now: batch 400, sweep_kept ~614 (supplying 328), cover_kept ~564
from ~1,792 draws (supplying its best 72), search ~56s.

## Multi-reaction readiness (built, tested, NOT started)

`bank_builder.py --reaction {1..5}` and `--base <dir>` now work.

- rxn 1, 2, 4 have TWO roles; rxn 3 and 5 have three. The sweep now fixes
  `components[:-1]` and substitutes the last role, so both shapes work.
- `tanimoto_reagent_ranker` defaults to `reaction=3, role_index=2` and
  fingerprints only that role. Called unchanged it returns [] for every other
  reaction — SILENTLY DISABLING THE SWEEP rather than raising. Now parameterised.
  Verified: swept analogues 46/31/24/35/43 for rxn 1–5 (was 0/0/24/0/0).
- The seed-bank copy now filters by reaction; byte-copying rxn 3 labels into a
  rxn 1 bank would poison every anchor and the whole scaling series.
- Gate survival per reaction (1,500 uniform draws): rxn 1 29.6%, rxn 2 84.8%,
  rxn 3 30.7%, rxn 4 66.7%, rxn 5 72.3%. rxn 2 is ~3x cheaper per usable
  molecule than the one we are building.

Space per reaction: rxn 1 9.2e8, rxn 2 2.0e9, rxn 3 4.1e10, rxn 4 3.4e8,
rxn 5 1.7e10. Novelty archive has burned at most 5.3e-5 of any of them.

## Standing hazards

- `ast.parse` does NOT catch `name used prior to global declaration`. Use
  `compile(src, name, 'exec')` as the syntax check.
- pgrep matching a remote process by a string that appears in your own ssh
  command matches itself. Use PID files.
- macOS rsync 2.6.9 has no `--ignore-missing-args`.
- Repo config lags the live validator: yaml says entropy 0.1 / similarity 0.7,
  live is 0.25 / 0.6.
- Target P40261 is FIXED in config and passed through unchanged; only the
  reaction is drawn per epoch (`seed % 5`, utils/challenge.py).

## Not started

- Compound publisher / on-chain submission. NOTE: at 19,799 we already beat 6 of
  the last 12 rxn-3 winners, so every rxn-3 epoch skipped forfeits a real chance.
- `tools/compound_portfolio.py` should read the `rescored` table, not raw scores.

## Going live — corrected after finding the prior art

**The combined master/slave ALREADY EXISTS.** `git branch -a` / `git worktree
list` first; see memory `nova-compound-branch-topology`.

    deploy/compound-nanobody-live   (repo root)  compound MASTER, nanobody SLAVE
      neurons/miner/miner.py: _compose_commitment(mols, nanos) -> "mols|nanos"
      molecules  <- compound_miner.provider.from_challenge -> artifacts/result_rxn{N}.json
      nanobodies <- compound_miner.nanobody_provider.load_qualified_nanobody
      both degrade to "~" independently
    feature/compound-fleet-10       the above + publisher machinery + slot_manager
    nanobody-opt (this worktree)    OPPOSITE: miner barred from committing,
                                    publisher writes, molecule section = "~"

Authoritative facts recovered from those branches:
- entropy is **AtomPair 2048-bit**, gate 0.25 (`provider._entropy`). The
  `compute_maccs_entropy` in utils/molecules.py here is MACCS and is STALE.
- `provider._still_novel(..., gate=0.25)` repairs a stale submission from spares
  carried in the payload — so the result file must contain more than 20.
- `release.SELECTION_SHRINKAGE = 0.9234` (floor 0.8965) from 6 folds.
  Independently confirmed tonight at 0.9191 from a 3-pass rescore.

### What was built here and VERIFIED

`tools/bank_release.py` — bank -> `artifacts/result_rxn{N}.json` for one epoch.
Takes ONLY rows from the `rescored` table (re-measured with the exact scorer, so
honest for both NIM- and OSS-originated molecules, and where the quality is:
+0.0817 over single-pass selection). Carries 20 + 40 spares, stamps the epoch,
checks AtomPair entropy, writes atomically.

Verified end to end against the LIVE branch's own `validate_release_payload`:
accepted, and correctly rejected on wrong epoch, wrong reaction, tampered
selection_score, unqualified status, and a proxy scorer label. Test run on the
current bank: 20 + 40 spares, sum 1.5975, entropy 0.3677.

### Built but NOT the right topology — decide before deploying

`nanobody_opt/commit_submission.py` gained molecules support and
`tools/publish_current_nanobody.py` gained `--molecules-file` + `{molecules}`.
That is the nanobody-MASTER direction and duplicates the compound branch.
It is inert (defaults reproduce `~|<seq>` bit-for-bit; no existing commit
template contains `{molecules}`) and 144 tests pass, but two competing commit
paths is the hazard this repo already recorded. REVERT unless wanted as a
fallback.

### Still open for go-live

1. A dedicated box running `deploy/compound-nanobody-live` (or
   `feature/compound-fleet-10`), with the bank synced to it and wallets placed
   by the user — I do not handle wallet material.
2. Per-epoch loop on that box: detect a rxn-3 epoch, ensure the shortlist is
   rescored, run `bank_release.py --epoch N`, let the miner commit.
3. **All uids need DISTINCT portfolios or they are worth one ticket** — the
   validator dedupes by molecule. `provider.result_path` keys only on reaction,
   so each slot needs its own `NOVA_CANDIDATE_RESULT`. Varying the tail is worth
   +0.0071 single-scored, +0.0034 after three passes.
4. Commit window: reserve >=26 blocks (config `no_submission_blocks: 10`).

## Going live: the release now binds to the existing fleet allocator (2026-09-19)

**Do not build a per-uid splitter.** `compound_miner/fleet.py::allocate_compound_-
fleet_from_release` on `feature/compound-fleet-10` already does it: shared core
of the best 12, then a distinct tail of 8 + 4 spares per uid. The top uid still
receives the release's own top-20.

`fleet_contract.required_exact_pool_size(k) = 12 + k*(8+4)` = **132** exact rows
for ten uids. `tools/bank_release.py --fleet N` sizes the pool to that. One file
either way; `NOVA_CANDIDATE_RESULT` stays per-reaction.

Two facts that cost a rebuild:

- The validator Boltz-scores each unique SMILES **once** and fans the value to
  every uid that submitted it (`boltz_wrapper._preprocess_data_for_boltz`:
  `{smiles: [(uid, mol_id), ...]}`, sharded by molecule not by uid). Identical
  portfolios therefore get identical scores. Extra uids buy redundancy and
  gate-diversification, **not** a higher ceiling.
- A release row needs `reaction`, `components` and both affinity terms.
  `validate_release_payload` does not look at them; `scored_candidates_from_-
  payload` rebuilds a Candidate from every row and cross-checks them against the
  identifier. Verified by running the publisher's own allocator, not a mirror.

Measured on the merged bank (22,168 distinct rxn-3, 200 rescored), epoch 25300:

| | |
|---|---|
| release | 20 selected of a 132-row pool, sum **1.5975**, entropy 0.3677 |
| allocation | 10 uids, `validate_compound_fleet_artifact` accepted |
| per-uid sums | 1.5975, 1.5864, 1.5772, 1.5712, 1.5642, 1.5567, 1.5508, 1.5454, 1.5407, 1.5367 |
| per-uid entropy | 0.3677 - 0.4143, all clear of the 0.25 gate |

Post-regime-break rxn-3 winners run 1.49-1.67, so the top uid wins some epochs
and loses others; the fleet's floor is still inside the winning range.

**Rescore coverage is the near-term lever.** Only 109 of the current top-132 have
ever been rescored (`--shortlist 200`), so the release reaches down to about
rank 200 of the single-pass ordering to fill the pool. Run
`bank_rescore.py --reaction 3 --shortlist 300` immediately before each
submission; coverage decays as the bank grows.

**Left in place, flagged:** the `--molecules-file` seam in
`tools/publish_current_nanobody.py` and the molecule section in
`nanobody_opt/commit_submission.py` on `nanobody-opt`. They are inert and
unsynced, and the dedicated box will run `feature/compound-fleet-10`, whose
publisher already has `--compound-result` / `--molecule-portfolios`. Revert them
the next time that branch is touched for another reason; do not deploy a second
commit path.

## The dedicated chain-writing box (2026-09-19)

`tools/submit_box_control.py` — `survey | create | status | stop | destroy |
extend --hours N | watch`. Rented:

| | |
|---|---|
| instance | **51676703**, label `nova-compound-submit-20260919` |
| hardware | 4x RTX 3090, Finland, reliability 0.9991, 32 effective cores, 250 GB |
| price | **$1.0696/hr** ($25.7/day) |
| lease guard | stops **2026-09-21T02:22:05Z** unless explicitly extended |

Offer ids on this provider rotate every few minutes -- a pinned id refused with
`offer_unavailable` ninety seconds after a survey found it -- so `create()` pins
the *policy* (4 GPUs, Ampere or newer, reliability >= 0.995, >= 8 cores/GPU, not
Taiwan, <= $1.30/hr) and records which offer it took. Turing and Volta four-card
offers are cheaper and excluded: Boltz's fused kernels and bf16 do not exist
below Ampere.

`watch` runs detached (`state/submit_box_watch.pid`) and stops the box at the
manifest deadline. It re-reads the manifest each tick, so `extend --hours N`
moves the deadline without a restart. It never extends on its own.

### We hold 12 uids, not 10

Queried from the chain against the `mining_ops` coldkeypub:
uids **2, 8, 17, 18, 41, 47, 48, 50, 53, 57, 61, 63** of 64 neurons.

`bank_release.py --fleet 12` wants `12 + 12*(8+4)` = **156** rows; the bank has
200 rescored, so it fits. All twelve allocate and pass
`validate_compound_fleet_artifact` against the real bank at epoch 25300:

```
uid  2 1.5975   uid 18 1.5712   uid 48 1.5508   uid 57 1.5367
uid  8 1.5864   uid 41 1.5642   uid 50 1.5454   uid 61 1.5310
uid 17 1.5772   uid 47 1.5567   uid 53 1.5407   uid 63 1.5273
```
Entropy 0.3677-0.4143 throughout, clear of the 0.25 gate.

### Two blockers before it can commit

**1. No keyfile for any of the twelve.** `~/.nova-hotkeys` holds
`nova68-slot-1/2/3`, and all three are **deregistered** -- they match no
current neuron. Mapping every hotkeypub under `~/.bittensor/wallets` against the
twelve registered addresses finds **zero** locally. The keys are on whatever
machine last mined these uids. `nanobody_services.yml` globs
`~/.nova-hotkeys` and asserts `set(staged) == set(owned)`, so a deploy today
fails correctly rather than half-committing. **The operator has to place them;
this is wallet material and is not handled here.**

**2. The compound publisher flags are only wired in `live.yml`, which is the
search topology.** `--compound-result` / `--molecule-portfolios` appear in
`live.yml`'s `compound_publisher_argv`, and that playbook asserts
`groups['compound_workers'] >= 1` and a cluster sum of `>= 16` GPUs -- it is
built for the distributed in-epoch search. `nanobody_services.yml` runs the
publisher on a single box but passes nanobody flags only. A bank-fed single box
needs the second playbook's provisioning with the first playbook's publisher
arguments. Decide before deploying:
- override `minimum_cluster_gpus=4` and list the box as its own worker (fast,
  but the assert exists because a half-cluster cost epochs), or
- a focused playbook for the bank-fed topology (publisher with compound flags +
  `--chain-state-only` miner + the rescore/release loop), which is the honest
  shape but is new code on the signing path.

### Unrelated, noticed while inventorying

Instance **51321895** (`nova-boltz2-nim19-test-20260917`, 1x RTX 6000Ada,
$0.72/hr) is **running**. If that test is finished it is about $17/day. Not
stopped without instruction.

## Corrected topology, and the writer move to 51676703 (2026-09-19)

The Pi is **not** the writer. It is the slot purchaser and hotkey custodian:
`nova-slot-manager` buys, `nova-hotkey-sync` pushes 158 keyfiles every 10
minutes to the box it resolves from `NOVA_VAST_LABEL=nova-nanobody-pixi`.

**`nova-nanobody-pixi` resolves to instance 51125470 — production.** It did not
appear in `instances/?owner=me` because the vastai CLI and the Keychain `vast.ai`
key are **two different accounts**; the bank boxes and the new box are in one,
production is in the other. That is also why the new box's label is invisible to
the Pi's CLI.

Production, inspected 2026-09-19: 4x RTX 4090, 48 cores, 77 GB free, rdkit
2024.09.6 in chain310, `/root` rsynced (not a git repo), `compound_miner/`
present but **no `fleet.py` / `fleet_contract.py`**, publisher argv has **0**
compound flags, and `chain_state.json` carries only
`block, epoch, epoch_length, miner_uid, nanobody_target, netuid, neurons,
schema_version` -- no `small_molecule_target`, no `allowed_reaction`, which
`_refresh_molecule_portfolios` requires.

Decision: **51676703 becomes the writer**, and also bank box 3.

### Hotkeys reached it without any key material passing through here

`deploy/slot_host/sync_hotkeys.sh` gained `NOVA_SYNC_EXTRA_TARGETS`
("host:port host:port"), wired through
`ansible/templates/nova-hotkey-sync.service.j2` and `ansible/slots.yml`. It is
**additional** to the labelled box, never a replacement: during a cutover the
box being handed the fleet cannot sign for a uid whose key has not arrived, and
the box still signing must not lose keys. Unlike the label path -- which exits 0
on "no box rented" because that is its normal state -- an unreachable *extra*
target is an error, because an endpoint written down on purpose going silent is
exactly the drift this script exists to prevent.

The Pi's ed25519 public key was added to the new box's `authorized_keys`
(a public key, not credential material). One run synced 158 files to both
boxes, exit 0. All twelve registered uids now have their keyfile on 51676703:

```
uid  2 nova68-slot-76   uid 18 nova68-slot-74   uid 50 nova68-slot-70   uid 61 nova68-slot-77
uid  8 nova68-slot-69   uid 41 nova68-slot-68   uid 53 nova68-slot-72   uid 63 nova68-slot-79
uid 17 nova68-slot-78   uid 47 nova68-slot-73   uid 57 nova68-slot-71
uid 48 nova68-slot-75
```

### Inventory across two accounts

`submit_box_control.py inventory` writes a minimal instances document for
`NOVA_VAST_INSTANCES_FILE`, rebuilt from an **allowlist** -- the provider's raw
instance document carries `jupyter_token` and ssh key material, and host
variables are printed by `ansible-inventory` and echoed in `-vvv`. With it, the
fleet branch's own dynamic inventory resolves `vast51676703` into
`nanobody_producer` / `nanobody_publisher` with `gpu_count=4`.

### Cutover order (two writers must never run at once)

1. provision 51676703 (`provision.yml -e node_mode=miner`) -- running
2. bank sync + bank-3 builder
3. publisher + `--chain-state-only` miner with the compound arguments
4. rescore -> `bank_release.py --fleet 12 --epoch N` locally on the box
5. **last:** stop production's publisher process. Nothing else on 51125470 is
   modified; its `--chain-state-only` miner signs nothing and can stay.

### Provisioning 51676703: three source fixes, not workarounds

1. **`provision_hotkey_source=external`.** `-e node_mode=miner` requires
   `controller_hotkey_paths` and then runs *"Remove stale rental copies not
   present in the controller list"*. On a custodian-fed box that reconcile
   deletes every key the Pi delivered; it failed at the assert, before the
   delete, only by ordering. `external` skips the staging and the reconcile and
   verifies the box instead -- hotkeys present, all 0600, no coldkey-shaped file
   anywhere under the wallet. Stronger on the property that matters.
2. **`preflight.sh` section 9 asked for a file literally named `nova68`.** The
   fleet is `nova68-slot-N`, so a box holding twelve registered slots reported
   "hotkey missing". It now counts hotkeys and fails only on zero;
   `NOVA_HOTKEY_PATH` still pins one when a deployment really has one. Which
   uids we sign for comes from the metagraph, never from a name written down --
   the same staleness that had us publishing into another operator's uid for
   twenty-one days.
3. **Deploy key.** `git ls-remote` against
   `thomasvangurp/nova-sn68-submissions` failed because `/root/.ssh/nova_deploy`
   was absent. Generated **on the box** (`ssh-keygen -C nova-submit-51676703`),
   so the private half never moves; only the public half was read and registered
   with `gh repo deploy-key add --allow-write`, title `nova-submit-51676703`.
   Verified from the box: `DEPLOY_KEY_OK`.

`ansible-playbook` returns **2** on failure while a shell wrapper ending in
`tail` returns 0. Read `PROVISION_RC` and the PLAY RECAP, never the task
notification's exit code.

Preflight state before the fixes: 31 passed / 2 failed / 1 warning. Passing
already: 4 GPUs visible to the shim, PYTHONPATH isolation, boltz+rdkit+torch
imports, 21 MSA files, molecules.sqlite checksum, all three boltz checkpoints
byte-exact, CCD complete (45,227 entries), 221G free, fused boltz kernels
import, CUDA novelty kernel compiles, no coldkey on the box. The warning
("own-score store not built yet") is expected on a fresh box.

### Bank 3 plumbing

`bank_sync.box()` read `'bank_box_control' if name == 'box1' else
'bank_box2_control'`: box3 would have been pulled from box2's manifest and
merged into box2's bank, reporting success throughout. Replaced with an
explicit `BOXES` map where an unknown name raises. Writing the test for it
caught a second gap -- `bank_box_control.FORBIDDEN` named neither box2 nor
box3, so a swapped or restored manifest could aim a `stop`/`destroy` at a
sibling's disk. All three modules now guard each other, held by
`tests/test_bank_sync_boxes.py`.

## The scaling law has stopped holding (2026-09-20)

`bank_track.py` was written to detect exactly this: "running BELOW the fit would
mean the bank is saturating and the spend should stop early."

| milestone | box1 total | box1 best20 | box2 total | box2 best20 | error vs law |
|---|---|---|---|---|---|
| 12,500 | 12,557 | 1.631003 | 14,310 | 1.635247 | +0.025 / +0.018 |
| 17,500 | 17,840 | **1.661820** | 17,683 | 1.658551 | +0.027 / +0.024 |
| 20,000 | 20,199 | **1.661820** | 20,199 | 1.658743 | +0.017 / +0.013 |
| 25,000 | 25,674 | **1.661820** | 25,295 | 1.658743 | -0.003 / -0.005 |
| 30,000 | 30,229 | **1.661820** | 30,618 | 1.659228 | -0.017 / -0.020 |
| 35,000 | 35,242 | **1.661820** | 35,242 | 1.659228 | -0.029 / -0.032 |
| 40,000 | -- | -- | 40,269 | 1.663617 | -0.038 |

box1's best-20 is **bit-identical across 17,400 molecules and fourteen hours**.
box2 gained **+0.0049 for 20,000 molecules**. The law predicts +0.057 per
doubling, so 20k->40k realised about **9% of the predicted gain**, and the error
term has swung monotonically from +0.027 to -0.038.

Why the fit does not extrapolate: it was built by **subsampling** 7,827 scored
molecules, which measures how best-20 grows when you draw more from a FIXED
distribution. The builder does not draw that way -- it sweeps neighbourhoods of
the current best and retires pairs into `exhausted_pairs`, so the marginal
molecule comes from an increasingly picked-over region and is not an i.i.d.
draw. Depth on one reaction is now close to free-fall.

**This makes breadth the clear buy** and supersedes the 60,000 target: a bank at
zero climbs the steep part of the same curve, and each reaction covered turns
20% epoch coverage into 40%, 60%, ...

## Bugs fixed 2026-09-19/20

- `bank_builder.py --help` crashed: bare `%` in a help string (argparse does
  `help % params`). Hidden because the check was piped, so `$?` was `head`'s.
  `tests/test_tool_help_parses.py` now runs --help for every operator tool and
  rejects an unescaped `%` at source.
- **Cold start**: `floor = fresh[-1][1]` on an empty bank. The sweep proposes
  analogues of what already scored, so a new reaction has nothing to anchor on.
  Round one now runs coverage-only.
- **`reaction=3` hardcoded** in the records handed to `score_boltz_batch`, which
  filters its input by reaction: a reaction-2 batch matched nothing, the scorer
  exited "archive shortlist is empty", and the run died having scored nothing.
  Third instance of this literal (two were in `bank_rescore.py`); a test now
  rejects any literal reaction keyword in the generalised tools.
- **The failure tail was unreadable**: `builder.json` holds one record, so the
  final `failed` write overwrote every `batch_skipped` tail that led to it. Now
  also printed to the append-only log -- which is how the line above was found.
- **`_assert_non_increasing_win_probability` took the fleet down on a valid
  order.** `win_probability_order_key` sorts on `(chance, tier)`, so chance is
  the OUTER key and a clean tier-0 sequence with no chance legitimately follows
  a tier-2 one that has one. The invariant compared tiers across that boundary
  and raised "tier 0 follows tier 2". Fixed in **both trees**; the ladder now
  restarts at the chance boundary exactly as it restarts between tiers. Covered
  by four negative tests plus a 300-case randomised property check that the real
  order key always satisfies the invariant.

## Why the sweep collapsed, and what the binding constraint really is

Weighting anchors by P(beat the cut) moved the budget from 1.4% to 24% into the
fifteen prefixes holding the top 20 -- and **collapsed the sweep in one round**:
`sweep_kept` 516-649 -> **64**, `batch_cover` ~72 -> **336 of 400**, so 84% of
GPU went to coverage draws that cannot reach the cut. This is the failure the
wide weighting exists to prevent, recorded in the code at round 58.

The model assumed a fresh draw was *available*. The sweep proposes
`ranker(swept, role_ids, minimum_similarity)[:depth]` -- the most SIMILAR
analogues only, capped by `--depth 16`. In a well-mined neighbourhood the top
16 are all already `seen`.

Measured offline (no GPU -- the ranker is Tanimoto over reagents), unseen
analogues available per anchor:

| depth | linear anchors | expected-gain anchors |
|---|---|---|
| **16 (default)** | 8.12 | **2.08** |
| 32 | 13.14 | 8.41 |
| 64 | 17.91 | 17.62 |
| 128 | 20.52 | **22.53** |
| 256 | 20.90 | **25.25** |
| neighbourhood shrunk mean | 0.03062 | **0.05047** |

**Depth 16 was the binding constraint, not the weighting.** At 64 the
concentrated anchors draw level; at 128+ they exceed the diffuse ones while
sitting in neighbourhoods 65% richer. Median band size is 11 for linear anchors
and 24.5 for concentrated ones -- the good neighbourhoods have MORE reachable
analogues, and depth 16 was throwing them away.

Raising `--minimum-similarity` to 0.6 is a dead end in the other direction:
bands collapse to 4-5 candidates (5.08 and 4.01 unseen per anchor at any
depth). 0.5 stays.

**Live A/B now running:** box1 `--anchor-weighting expected-gain --depth 128
--hot-depth 256`; box2 unchanged on `linear` at depth 16 as control. Judge on
`sweep_kept`, `batch_cover`, and whether kept molecules land in higher-mean
prefixes. One round shows the collapse or its absence.
