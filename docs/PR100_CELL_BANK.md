# PR100 challenge-indexed candidate bank

Owner: `nanobody_opt/pr100_cell_bank.py`, `scripts/pr100_cell_bank_build.py` and the
bank hook in `nanobody_opt/pr100_live_controller.py` (config flag, default **off**).
Validator-matched semantics are untouched: every verdict still comes from
`nanobody_opt/rules.py` and `nanobody_opt/pr100_live_selection.py`.

## 1. Why

Exactly two scalars are revealed at epoch start: `allowed_v` and
`allowed_cdr3_length` (`rules.challenge`). Everything expensive depends only on the
sequence, not on those scalars:

| work | measured cost | needs the epoch hash? |
| --- | --- | --- |
| conditional generation + annotation | 64 proposals / 4 s | no (bucket-conditioned) |
| native developability | 55.2 CPU-s per profile, ~1.8% pass -> **0.86 core-h per accepted candidate** | no |
| exact archive novelty | 2.96 s/candidate (234 s for 79) | only the delta since the watermark |
| fresh target scoring | 16.9 GPU-s/record, target pinned to `Q9NZQ7` | no |
| top-50 `is_duplicate` gate | unmeasured at `num_top_sequences: null` | **yes**, the index moves |
| publish + commit | 78-98 s | yes |

So the irreducible post-T0 path is a bucket lookup, a novelty delta, the live
duplicate gate, fleet construction and the commit. The 171 buckets are
9 germlines x 19 CDR3 lengths; the three singleton V categories are 57 buckets
carrying **75% of epochs**. Current inventory covers 8 buckets (~6% of epochs).

## 2. Schema (version 1)

A bank is a directory: `bank.json` (manifest with `schema_version`) plus
`records/<target>/<GENE>__cdr3<L>.jsonl`. Appends are a single atomic
`O_APPEND` write of one line followed by `fsync`, so an interrupted write can only
lose the tail of the last line (skipped on read), never corrupt an earlier record.
A later line for the same `sequence_hash` supersedes the earlier one; the file
remains the audit trail.

Per record: `target`, `bucket` (`v_gene`, `cdr3_length`) and `bucket_id`,
`sequence`, `sequence_hash`, `annotation` (the full annotation evidence, so
`rules.annotation_reason` re-validates offline), `measurements` (**every** repeat,
raw, so `rules.developability_reason` re-checks the numbers rather than a `passed`
flag), `measurement_runtime` (`filter_sha`, `annotation_identity`,
`native_runtime`, `host`, `measured_at_unix`), `tier`, `psh_values`, `psh_mean`,
`psh_sd`, `repeats`, `margin` + `margin_threshold`, `novelty` (the search evidence
plus the watermark: `archive_sha256`, `archive_rows`, `archive_prefix_sha256`,
`checked_at_unix`), `scores` (a list of observations, each with `metrics`,
`scorer_sha256`, `settings`, `settings_file_sha256`, `observed_at_unix`) and
`provenance` (`source_sha`, builder, synthetic challenge hash, host).

**Target scoping.** Annotations and native profiles survive a target rotation;
score observations do not. They are a separate list carrying the target, so a
rotation invalidates the scores only -- which is the cheap part (16 GPU-h for a
complete 171-bucket bank versus ~2930 core-h of native work).

## 3. Use time: nothing cached is a certificate

`CellBank.select_for_epoch(challenge, archive, ...)` returns a dict shaped exactly
like a native-phase result, so the publication path is unchanged. It re-checks:

1. **Rules** -- `rules.annotation_reason` against the *live* challenge, so a record
   whose germline or CDR3 length does not match this epoch is dropped.
2. **Developability** -- `rules.developability_reason` on every stored repeat, then
   the tier is recomputed by `pr100_live_selection`. A record stored as tier A is
   used as tier A **only** with at least `min_repeats` (2) measurements from the
   pinned runtime identity that are not older than the freshness window
   (default 14 days); otherwise it is **downgraded to tier B** and published as
   such. This is the audit's margin rule
   (`min(89 - 1.3*4.4/sqrt(k), hard margin 80 at k=2 / 82 at k=3)`) and its
   corollary: the cache is a compute saving, never a pass certificate.
3. **Novelty, incrementally** -- the watermark's prefix digest is recomputed over
   the first `archive_rows` sequences of the live archive. Matching means the
   archive is append-only up to the watermark and only the appended rows are
   searched; not matching (rewritten or shrunk archive) yields
   `novelty_recheck_required` rather than a cached pass.
4. **The live duplicate gate** -- the two-tier `is_duplicate` index moves with the
   archive and is never cached. `select_for_epoch` takes a `duplicate_reason`
   callable and applies it when supplied.
5. **Evidence** -- `pr100_evidence` is rebuilt around the live challenge and run
   through `rules.validate_evidence`; a failure drops the record.

Candidates are returned tier A first by ascending mean PSH, then tier B, truncated
to `limit`, with a `rejected` list of (sequence hash, reason) for the receipt.

## 4. Building

`scripts/pr100_cell_bank_build.py` drives the **existing** worker phases through
the same `job.json` contract (docs/PR100_LIVE_CONTRACT.md), one bucket at a time:
sequence -> native (with repeats) -> novelty (inside the native phase) -> score.

No epoch hash is required. `pr100_cell_bank.synthetic_challenge` searches block
hashes until `rules.challenge` draws the requested bucket (expected 76 tries for a
V1-V3 bucket, 456 for a V4 one) and the result is a genuine challenge object that
the worker phases verify on entry exactly as they do live. It is deterministic per
(bucket, epoch), so a resumed run reuses the same job identity.

**What still needs the real epoch-start hash, at use time only:** the live
challenge inside `pr100_evidence` (which is why the bank stores the annotation and
raw measurements and rebuilds the evidence at selection), `rules.verified_fleet`,
`pr100_live_trial_export.build`, the publication cutoff arithmetic and the Pi
commit. Nothing built with a synthetic hash is ever published.

```
# plan only: ordering, coverage and the cost model, no measurement, no record
scripts/pr100_cell_bank_build.py --bank /bank --run-root /run --dry-run --max-buckets 10

# real build: CPU-only (no scoring), 57 high-mass buckets to 20 each, bounded
scripts/pr100_cell_bank_build.py --bank /bank --run-root /run \
  --source-sha <sha> --native-tools /tools --references /bank.json --archive /archive.csv \
  --per-bucket-target 20 --workers 48 --repeats 3 --core-hour-budget 200

# add scoring on explicitly idle GPUs
... --gpus 4 5 6 7 --scorer-settings /settings.json --gpu-hour-budget 2
```

Resumability: `--run-root/progress.json` records every finished bucket and
`--run-root/<bucket>/<phase>/result.json` is reused when its status is
`complete`/`partial`, so a killed run never re-measures a finished phase. Budgets
(`--core-hour-budget`, `--gpu-hour-budget`, `--max-buckets`) truncate the *plan*
before a bucket starts, and `--wall-seconds` stops between buckets, so a budget
stop never discards a measured candidate. GPU throttling is explicit: `--gpus`
lists the GPUs the operator has already established are idle (a GPU with a
resident daemon refuses its lease), and an empty list means a CPU-only build.

Priority order is `build_priority`: probability mass descending, so all 57
V1-V3 buckets precede any V4 bucket; ties break on shortfall then bucket id.

## 5. Budget model (audit `8_epoch_algorithm.md` 3.1)

`budget_estimate` uses the measured census numbers (55.2 CPU-s/profile, 1.79%
first-pass rate, 16.9 GPU-s/record) plus `repeats - 1` further profiles for
survivors:

| bank target | native | GPU |
| --- | --- | --- |
| 20/bucket, 57 high-mass buckets (75% of epochs warm) | **~977 core-h** | **~5.4 GPU-h** |
| 20/bucket, all 171 buckets | ~2930 core-h | ~16.1 GPU-h |

Steady state, with 13 consumed per epoch and each stocked bucket drawn about once
every 76 epochs (V4: every 456), is ~11 core-h and ~0.06 GPU-h per epoch -- far
inside the idle capacity between epochs. The build is the one-off cost; the native
gate, not the GPU, is the bottleneck by about two orders of magnitude.

## 6. Controller integration

`config.bank` (absent or `{"enabled": false}` by default):

```jsonc
"bank": {"enabled": true, "root": "/abs/bank", "limit": 13,
         "runtime_identity": {"native_runtime": "<pinned id>"},
         "max_measurement_age_seconds": 1209600}
```

With the flag on, `run_epoch` resolves the challenge, fetches the archive and then
calls `bank_publication` **before** generation. That path only *supplies*
candidates: they still go through `publish()`, which means the archive re-fetch
guard, `rules.verified_fleet`, the publication cutoff and the monotone-quality
rule, unchanged. The generation run then upgrades the fleet through the same
`publish()`, and a refused upgrade leaves the early fleet in place. Any bank
failure is recorded in `receipt['bank']` and never aborts the epoch. With the flag
off the bank is not even opened.

## 7. What must be confirmed on real hardware before relying on it

Everything here is local code with unit tests; nothing was run on a box, a GPU or
the chain.

1. **Builder end-to-end on the master** -- one bucket, small `--proposals`, with
   the real `--native-tools`, `--references` and archive, to confirm the job
   contract, the resume path and the ingest.
2. **Native throughput and yield per bucket.** The 55.2 CPU-s/profile and 1.79%
   pass rate come from a census dominated by two IGHV3-23 cells; per-bucket yield
   in the other 169 buckets is unmeasured, and the cost model scales linearly in
   `1/yield`. Concurrency above 32 is also unmeasured (contention was observed at
   32).
3. **The `is_duplicate` / `index_top_sequences` gate wall time** at the current
   archive size under `num_top_sequences: null`. `select_for_epoch` accepts the
   callable but the T0 critical path may not assume it is cheap until it is timed.
4. **The incremental-novelty assumption** -- that the HF archive is append-only.
   The prefix digest makes a violation detectable (`novelty_recheck_required`), but
   how often it fires in production is unknown.
5. **PSH noise sigma (4.4) and the freshness window (14 days).** Sigma is
   estimated from 24 repeat triples; the window is a judgement call, not a
   measurement. Both are config, not constants.
6. **Scorer identity stability.** `bank_scores` publishes only candidates sharing
   one `scorer_sha256`/settings file; a scorer bump silently shrinks an early fleet
   until the bucket is re-scored.
7. **Whether an early bank fleet is competitive.** No PR100-era validator score
   exists yet for any entry, so "best 13 by our score" remains a prior.
