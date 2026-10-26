# PR100 qualification workflow — review candidate

This is a proposal, not a production approval. Both reviewers must independently
approve the workflow; production remains blocked until the required measured
evidence passes. A missed epoch is preferable to an unqualified activation.

## Oracle and scope

Pin the upstream PR head `ce19ea8cbe656671c3440ba58b4093ff80798d4f`,
filter submodule `fe2d5868126406a0de8953d26ecd8bdccbdbd029`, target structure,
configuration, model checkpoints, Python dependencies, native binaries, scorer,
wrapper and launcher. Recheck the upstream merge before activation at 25955;
a changed pin invalidates qualification. No legacy target scores qualify the
new target. New-target rank sums cannot be compared numerically with old ones.

Execute the actual pinned `validate_nanobodies` implementation as the oracle.
Control-flow tests may inject recorded dependencies, explicitly labelled as
such. Release qualification must additionally execute its actual annotation,
novelty, developability and scoring dependencies on real inputs. A reimplemented
predicate or fabricated passing profile cannot substitute for this run.

## Test populations

1. Preserve a deterministic sequence-only census by every challenge bucket;
   distinguish distinct sequences from tied-gene bucket membership. Report
   probability-weighted coverage, eligible fleet capacity and fresh-work
   shortfall. Empty buckets and exhausted banks are normal test cases.
2. Include real measured passers and rejecters, all observed sequence lengths,
   tied V genes, threshold boundary cases, exact archive duplicates and close
   neighbours. Freeze the archive snapshot and its provenance. Use all 171
   possible challenges, plus at least 10,000 block-hash parity checks.
3. Boundary fixtures probe inclusive thresholds, wrong/missing/NaN fields,
   incomplete batches and provenance, ambiguous annotations, target changes,
   wrong epoch hash, duplicate UIDs/sequences and missing structure files.
   Artificial fixtures test boundaries only, never capacity or score quality.
4. Use recorded incident populations and new-target measurements for realistic
   integration trials. Keep held-out observations unavailable to selection.
   Do not manufacture future-target history or 4,800 eligible candidates to
   satisfy the legacy promotion schema. Missing evidence is a blocker.

## Runtime and incident trials

Run isolated, bounded trials on spare slave capacity, initiated by the Pi.
Keep production daemons, sockets, databases, submission files and wallets out
of trial writes. Capture backend logs privately in memory and emit allowlisted
diagnostics only. A timeout must kill every descendant of that trial.

Test both fresh non-replicate and independent-replicate paths over multiple
chunks, every candidate length, cold/warm startup and missing target-specific
synthesis references. Confirm full cardinality, finite ten-metric results,
correct target, independent replicate IDs and retained structure provenance.
Measure design, fold, filter and handoff times; a small GPU probe cannot prove
4,800-per-epoch throughput or biological eligibility.

Exercise interrupted/resumed waves, partial results, unavailable workers,
gate attrition, stale caches, disk shortage, late confirmation, publisher/Pi
outage, stale chain state, changing challenge and epoch rollover. No old work
may consume the next epoch's lease or publish under its challenge. Durable raw
observations and NEW-budget reservations survive restarts and rollback.

The NEW budget counts at most 4,800 previously unmeasured target-specific
candidates, independently of up to 8,000 proposals and cached-candidate
comparison. Report reserved, dispatched, completed and failed counts separately.
Cached candidates still receive the current challenge/novelty/developability
precheck when there are zero NEW candidates. Do not relabel repeated work NEW.

## Release decisions

Run shipping CI against committed bytes and qualify the complete deployed
dependency closure. Test the actual Pi consumer with measured evidence and
current fleet allocations, without committing on chain. Respect the real
submission validity window; never make a stale artifact fresh by copying it.

Application promotion must satisfy `RELEASE_PROMOTION.md`. A new model/scorer
dependency closure requires its separately qualified rollout; source-only
deployment is insufficient. Controls-only CI cannot approve application code.
Rehearse rollback without reverting raw observations or budget reservations.

The review ledger records each objection, evidence, revision and independent
verdict. A workflow GO means the test procedure is accepted. A production GO
requires its successful execution; neither agent may convert missing evidence
into a pass. Stop for unresolved blockers rather than forcing agreement.

## Current measured evidence and blockers

- The original candidate passed 812 shipping tests in both package profiles.
- The full census annotated 389,509 sequences; 4,080 passed sequence eligibility
  in eight of 171 buckets. Weighted nonempty coverage was 6.14%; full-fleet
  coverage was 3.95%, before developability, novelty or score selection.
- Fourteen stratified real profiles ran without backend errors; none passed
  the strict PSH limit. This is not an estimate of the whole bank's pass rate.
- The registered fleet scorer differs from the repository. Its non-reference,
  non-replicate path omits input regeneration after the first chunk. The staged
  correction retains replicate support and restores per-chunk input creation.
- Candidate-bank exhaustion incorrectly aborted before cached comparison.
  The staged correction emits an explicitly empty fresh pool instead.
- A small new-target GPU probe is running; no throughput conclusion yet.
- Broad candidate coverage, real passing fixtures, sustained replenishment,
  complete native runtime installation and measured full handoff remain open.
