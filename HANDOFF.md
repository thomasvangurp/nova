# Handoff — nanobody miner, subnet 68

Written 2026-09-15 ~23:10 UTC. Branch `nanobody-opt`, HEAD `66700be`, pushed to
`gitlab`. `origin` (metanova-labs/nova) is upstream and not writable from here.

## State right now

Mining is live and healthy. Epoch 25141 published **all 10 registered uids**
(7, 9, 19, 24, 38, 44, 46, 54, 55, 62) at `stage=provisional`.

Three tmux sessions on the one box, all up:

| session | what it is |
|---|---|
| `nova-nanobody-producer` | generates + screens candidates (the GPUs) |
| `nova-nanobody-publisher` | ranks, selects the fleet, commits to chain |
| `nova-nanobody-miner` | upstream reference miner, run ONLY to write `chain_state.json` |

Box: `ssh -p 15470 root@ssh2.vast.ai` (vast instance 51125470, 4 GPU).
Deploy: `./deploy/nanobody_deploy.sh nanobody_services.yml` from the repo root.
It resolves the chain-writer hotkey from chain, refuses an empty inventory, and
runs the publisher test suite ON the box against the deployed tree before
starting anything. A red deploy leaves the running services untouched.

## What changed this session (newest first)

- `66700be` — strict-qualification reserve raised 150 → **220 blocks**. Measured
  live: the strict rank runs **27+ minutes**, not the 20 the old comment
  claimed. The epoch check runs *after* the rank, so an overrun delays the NEXT
  epoch's provisional commit — the original reason the stage was switched off.
- `049a605` — tests that read `ansible/nanobody_services.yml` now skip when the
  file is absent. The deploy copies the tree to the box **without `ansible/`**
  and runs the suite there; reading it unguarded failed a deploy outright.
- `81b3ff3` — **re-enabled the qualified stage.** `--strict-qualification` was
  off, so 13+ consecutive epochs published at `provisional` (evidence floor
  only, no quality gate). Now on, gated on `--strict-qualification-reserve-blocks`.
  The chain read that measures the window is wrapped: the fleet is already
  committed by that point, so an unreadable chain view skips the upgrade rather
  than raising.
- `befea47` — **fixed a live publication outage I had caused.** Making the
  settled-deviation tier the primary sort key left `fleet.py`'s invariant
  describing the old single-column order, so every cycle died with
  `SelectionOrderError: ranking is not ordered best-first by win probability
  (0.0148 follows 0.0)` and nothing committed. Also: the ranker now orders
  chance-first then tier, so a **zero**-win-probability molecule never displaces
  one with a chance. The sort key moved to module level as
  `rank_exact.win_probability_order_key()` — it used to be a lambda inside
  `main()` that the tests re-implemented on their own fixtures, which is why
  they all passed while the publisher refused the output.
- `c414c5c` — the deploy now restarts the chain writer when its **whole command
  line** drifts, not just its hotkey. The launch argv lives in one var,
  `chain_writer_miner_argv`.
- `817b2ea` — `--chain-state-only`. `neurons/miner/miner.py` is upstream's
  example miner; its candidates are a random 1–30 mutation of a hardcoded base
  ("therefore not competitive" per its own docstring) and it must hold a
  REGISTERED hotkey, so it was committing noise over our own **uid 19**.

## Open items, in priority order

### 1. Confirm the qualified stage actually completes (highest value)

It has never run to completion in live operation. It is now enabled but had not
yet had an eligible window when this was written (epoch 25141 was past the
220-block mark). First eligible early in an epoch.

Check:
```
ssh -p 15470 root@ssh2.vast.ai \
  'tmux capture-pane -p -t nova-nanobody-publisher -S -4000 | grep -E "stage=qualified|strict upgrade|no robust repeat"'
```
Three possible outcomes, all informative:
- `NANOBODY_CHECKPOINT_OK stage=qualified` — it works, done.
- `no robust repeat-observation upgrade is available yet` — the strict gates
  reject everything. Next step is finding WHICH gate; the likely one is
  `--minimum-structure-observations 1` in `nanobody_opt/publication_gates.py`.
- `too little of the epoch is left...` every epoch — 220 blocks is unreachable
  in practice because the provisional path itself eats the early epoch. Then
  the fix is to make the strict rank cheaper, not to shrink the reserve.

### 2. An unfinished measurement is running on the box

A background probe runs the publisher's own strict rank offline to a scratch
file — it answers item 1 without waiting for epochs:
```
ssh -p 15470 root@ssh2.vast.ai 'cat /tmp/strict_probe.json | head -40'
```
It had run 27+ minutes without finishing. If `/tmp/strict_probe.json` exists,
its `candidates` / `sequences` length is the answer: non-empty means the
qualified stage will produce a real upgrade. Script is at `/tmp/strict_probe.py`
on the box. **Nothing depends on it; it writes only to /tmp.**

### 3. ~15% of lost tickets are still unaccounted for

~31% of tickets score nothing. TNP compute failures explain ~16% (validator-side
GPU fault, transient, never pre-filter on it). The rest is unexplained. W&B and
the leaderboard API are the sources — see the memory note on observability
endpoints.

### 4. Local test suite does not run on this Mac

`pytest tests/` fails collection on 30 files with
`ValueError: numpy.dtype size changed` (homebrew pytest vs the repo's numpy),
and `tests/test_publish_current_nanobody.py` needs ≥3.10 for
`dataclass(slots=True)`. Everything was verified on the box during deploy
instead (80 passed, 1 skipped). Fixing the local env would speed up every
future change. Use `PYTHONPATH=.` — there is no installed package.

## Things that have already been tried and were wrong

Do not re-derive these.

- **A crowd/top-50 similarity pre-filter.** Built and deployed twice, reverted
  twice (`a4af106`). An agent later proved 0 of 83 tickets would have been
  flagged. The repo's own memory recorded the mistake before it was repeated.
- **Raising the observation floor above 1.** Depth == number of epochs a
  sequence has appeared in, exactly. A floor above one is a survivorship filter
  that excludes 100% of fresh candidates. The long comment in
  `publication_gates.py` has the measurements.
- **Confidence-only selection.** Replay showed it LOST 1.5–2.8 rank points at
  every k.
- **Pre-filtering on TNP failures.** Validator-side transient fault, not a
  sequence property.

## Conventions that bit me

- `hmmscan` is only on PATH via `pixi run` / `metanano_python.sh`. CDR
  extraction returning `None` is almost always this.
- The validator's source is PUBLIC and vendored in this repo. Read and run it
  rather than inferring behaviour from outputs.
- Never use `tee` for logs; redirect with `>` to a fixed filename.
- Do not pass `--raw` to `vastai create instance` (prints an instance API key).
