# PR100 pipeline redesign — hypothesis to benchmark

The compatibility branch is not an optimal new pipeline. Its purpose is to
avoid invalid submissions during a rule change. The census (eight nonempty
challenge buckets out of 171) and fourteen failed developability profiles show
that the old population cannot support reliable operation under PR100.

## Alternatives

| Approach | Benefit | Main risk | Evidence required |
| --- | --- | --- | --- |
| Keep old generation and add new gates | Small operational change | Mostly empty or rejected challenge pools | Actual coverage, accepted throughput and deadlines |
| Generate entirely after the challenge arrives | Focuses all generation on the actual challenge | Generation and filtering compete with scoring and handoff | Cold-start end-to-end timing and full fleet completion |
| Maintain a challenge-indexed eligible bank, replenish asynchronously | Moves target-independent qualification out of the deadline window | Bank diversity, depletion and stale proofs need explicit management | Every bucket's measured eligible capacity, replenishment cost and expiry behavior |

No architecture is preferred before measuring conditional generation yield,
filtering cost and available inventory. A bank may move work outside the
window, but does not create eligible supply. Benchmark all three against the
same pinned validator, challenge schedule, compute allocation and held-out
observations before selecting a production policy.

## Separation of responsibilities

1. Maintain a target-independent, challenge-indexed sequence inventory. Verify
   annotation and developability using the exact validator dependencies. Cache
   measurements by the complete dependency/configuration identity, not by a
   bare pass flag. Keep sequence-only, fully prequalified and target-scored
   inventory counts distinct. Admission is never inferred from parent quality.
2. Prepare every challenge bucket before its block hash exists. Allocate extra
   replenishment capacity by actual challenge probability and observed depletion,
   while retaining a minimum viable fleet plus measured attrition allowance for
   rare buckets. Do not pretend a few common buckets provide network-wide
   coverage. The required surplus is measured, not an invented fixed count.
3. Bind an epoch job to its actual block hash, rule fingerprint, target and
   challenge. Recheck archive novelty and current eligibility, then admit
   genuinely unmeasured target-specific candidates to the NEW budget. Previously
   measured eligible candidates compete through the separate cached path.
4. Score within the remaining submission window. Use new-target uncertainty
   measurements to allocate repeated passes; independent held-out observations
   evaluate the policy. No fixed 20-pass requirement may consume the handoff
   window, and a single noisy pass must not silently become confident merit.
5. Keep publication and Pi submission independent of generation, replay and
   replenishment. Each job has an epoch lease, bounded subprocess ownership and
   cancellation; only current, fully verified evidence can be handed off.

## Parent reuse is not submission reuse

An archived sequence may be ineligible to submit again, while still being
informative as a generation reference. Conversely, an excellent old-target
parent may have no useful merit for the new target or the required challenge.
Evaluate generation sources by the fraction of novel, validator-accepted
children and their held-out new-target performance per unit compute. Maintain
multiple independent families and compare against parent-free generation.
Do not assume lineage alone predicts success, or transfer old-target scores.

Stratify the comparison into same V/same CDR3 length, same V/different length,
and different V references. Keep a parent-free arm in each challenge cell.
Report conditional yields separately; a pooled advantage can merely reflect
easier challenge cells. The old census occupies eight cells, with most of its
members in two; it does not occupy only two. Changing length or V does not by
itself prove that all useful information from a reference has been lost.

## Benchmark and decision

Freeze the real input population, archive, pins, challenge schedule and compute
budget before evaluating policies. Cover all 171 challenge combinations, their
actual probabilities, empty/depleted buckets and consecutive epoch transitions.
Do not reuse selection observations as held-out evaluation evidence.

Report validator acceptance, actual eligible fleet capacity, new candidates
completed, uncertainty calibration, held-out score, end-to-end deadline success,
cost per accepted candidate and replenishment sustainability. Average weighted
coverage alone is insufficient: report minimum bucket capacity and failure
rates for rare challenges. Use a common frozen new-target comparison field
within each paired experiment; do not compare raw old-target rank sums.

First qualify the existing bridge's correctness and failure behavior. Then
run the alternative policies as isolated experiments. Select a redesign only
when its measured benefit and operational reliability justify the migration.
If no policy supplies enough verified candidates, report that capacity gap
instead of lowering the validator gates or claiming the pipeline is ready.

## Preregistered comparison, version 1

This version is fixed before strategy outcome trials. The native census and
technical probes are feasibility work and must not be represented as a
randomized strategy comparison. First run the four-cell yield pilot described
in PR100_QUALIFICATION_WORKFLOW.md, with 256 proposals per source stratum and
cell, a fixed six-hour wall-clock cap per arm, and identical compute caps.
Stop an arm at its cap or after 256 proposals, whichever comes first; count
timeouts and rejected proposals in its denominator. Retain zero-yield arms.

The primary endpoint is sustained, distinct, actual-validator-accepted entries
delivered before cutoff per epoch over K=20 consecutive epochs, with archive
advancement. Compare policies on ten matched, probability-weighted 20-epoch
schedules (seeds 1000 through 1009, sampler and seed algorithm pinned before
execution). Analyse the ten paired schedule means, not individual correlated
observations. Predeclare a meaningful effect of at least one additional
eligible entry per epoch. Report the paired effect and a 98.333% paired bootstrap
interval for each of the three policy pairs, resampling complete schedules
(10,000 resamples, seed 271828; Bonferroni family-wise error target 5%).
Require the interval's lower bound to exceed one entry for a superiority
claim; otherwise report inconclusive or no improvement. Ten schedules do not
imply adequate statistical power: report precision and do not claim an
optimal strategy from an inconclusive result.

Secondary endpoints are worst-bucket residual capacity, deadline success,
compute cost per accepted entry, actual NEW completions, and held-out
new-target score. No secondary endpoint may replace a failed primary endpoint
after looking at results. A policy that fails the feasibility contract stops
before full-volume trials; do not extrapolate a small pilot into 4,800 NEW
evaluations or invent a nonzero yield from an empty arm. Subsequent larger
experiments require a new dated registration, retaining this result.

"Near optimal" is limited to the tested architectures and stated compute
budget. It requires feasibility, no known unresolved correctness failure,
and paired uncertainty bounds excluding an advantage of one accepted entry
per epoch for each tested alternative. Score quality and cost must also be
reported; equal fleet size alone does not imply equal quality. This experiment
cannot establish a global optimum over untested models or generation methods.
