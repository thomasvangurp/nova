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

## C2 strategy comparison, version 2 — registered 26 October 2026

This registration supersedes version 1 before strategy outcomes were measured.
Version 1 is retained in Git commit 8a4d2a0 and the unedited round-3 review. Its
600 full-volume epoch trials were not budgeted and are withdrawn. C2 is a
research contract. It cannot promote code, waive C1 production qualification,
or establish superiority at a scale it did not test.

### Resource budget and availability

Use the four already rented slave GPUs for at most six additional hours: 24
allocated GPU-hours, including idle time, startup and unsuccessful trials. No
hardware purchase is implied. Reserve 8 GPU-hours for the conditional generation
pilot, 4 for technical/existence checks and 12 for the paired comparison and
its scale check. Unused pilot budget may not silently enlarge the comparison.
Stop at the cap and report incomplete evidence. CPU preprocessing has a separate
256 core-hour cap; record actual work and reserved capacity separately.

The pilot has four cells and four reference strata (same V/same length, same
V/different length, different V, no historical reference). Publish reference
counts for all sixteen arms before starting. An unavailable reference stratum
is UNRUNNABLE, not a measured zero yield. Available arms receive at most 256
proposals and 30 GPU-minutes each, with at most four arms running concurrently.
Count timeouts and rejected proposals in their actual denominators. Failed
arms remain in the result. The algorithms and source pins for every arm must
be fixed in the execution manifest before generating candidates; an undefined
or unavailable generator cannot be treated as an implemented alternative.

### Feasibility and existence gates

Measure per-cell yield, novelty/clonotype attrition and warm compute costs
before running a policy comparison. For G GPUs and usable window W seconds,
a necessary compute condition is

    N_new * t_new + 13 * (10 + 10) * t_rep + T_other <= G * W.

Times on the left are GPU-seconds, not mixed wall-clock and device units.
`t_new` includes measured scoring work per NEW candidate; `t_rep` is per repeat.
`T_other` includes generation and any remaining GPU work, counted exactly once.
CPU filtering, serial stages, archive refresh and handoff must also pass a
measured critical-path deadline check; satisfying this inequality alone does
not prove schedulability. W is the real submission window after the existing
handoff reserve, not an invented 72-minute allocation. Repeats have a separate
ledger and never consume or inflate the 4,800 NEW-candidate count.

Using measured per-cell yield and novelty/diversity attrition, bound the count
of distinct accepted entries producible within this work and critical path.
Require at least one arm to supply 13 in the favourable cell and at least one
actual passer in IGHV3-23:5 before advancing. A measured zero in the rare cell
means this tested arm has no demonstrated rare-cell supply; it does not prove
that all possible generators have mathematically zero yield. If the bounds
cannot reach these gates, stop and report the capacity gap.

Next require one real 13-entry fleet, with at least ten selection and ten
reserved held-out observations per sequence, correct novelty/developability,
and the actual Pi consumer's dry-run acceptance within its window. Before
activation, a historical block-hash replay under future rules is explicitly
counterfactual and cannot satisfy the live-challenge gate. After activation,
use the actual epoch-start hash and authoritative epoch dimensions. Test the
favourable cell when it actually occurs; do not relabel a chosen synthetic
challenge as the live one. A failed favourable-cell existence trial stops the
full-volume programme. No on-chain write is needed for this qualification.

### Bounded paired comparison

After those gates, run ten independent matched challenge cases per policy,
using seeds 1000 through 1009 and the pinned challenge sampler. These are ten
one-epoch trials, not ten 20-epoch GPU programmes. The K=20 depletion sweep is
a separate offline replay of retained observations and measured timing traces;
its reused observations are labelled REPLAY, never NEW or newly executed GPU
work. It cannot substitute for the three full-volume C1 production trials.

For comparison volume N_cmp and a scale check at 2*N_cmp on one shared case per
policy, the conservative work bound is

    36 * N_cmp * t_new + R_cmp * t_rep + T_cmp_other <= 12 * 3600.

Thirty ordinary policy/case runs plus three double-volume scale runs give the
coefficient 36. R_cmp is the preregistered repeat count, including selection
and held-out draws; sharing an already retained observation must be explicit.
Also require the per-case critical path to fit W and 13 eligible entries to be
possible at N_cmp. Choose the largest integer N_cmp satisfying both bounds,
using measured times, before viewing strategy outcomes. Publish every input
and the derived value in an immutable execution manifest. If no positive,
feasible value exists, do not start the comparison. Changing the case count,
budget or volume after seeing outcomes requires a new registration.

Measure at both scales and report whether conclusions change. Results remain
scale-local until C1 passes; even an unchanged small scale check is not proof
of 4,800-candidate performance.

### Decision endpoints and uncertainty

Capacity is an absolute gate: each compared policy must deliver 13 distinct,
actual-validator-accepted entries per case inside the window. A policy below
that floor is infeasible at that scale, never equivalent to another empty arm.
Report every failure; do not drop inconvenient cases or replace them post hoc.

For feasible policies, held-out new-target score is the primary quality
endpoint. Freeze one genuinely measured new-target reference field per matched
case for all arms, and record its provenance. Old-target rank sums are excluded.
Compare paired fleet mean rank-sums, lower being better. Predeclare a 5% quality
margin relative to the incumbent feasible policy's mean, plus a 5% cost margin;
report absolute rank-sums alongside ratios. If no incumbent is feasible, use
the fully specified online-generation arm as the fixed comparator, and if it
also fails, report feasibility results without a relative superiority claim.

Use paired schedule-level bootstrap intervals across the ten matched cases,
10,000 resamples, seed 271828. For the three quality contrasts use 98.333%
intervals (Bonferroni family-wise target 5%). A quality win requires the upper
bound of candidate/comparator score ratio below 0.95, no capacity failure, and
no measured cost increase beyond 5%. A cost improvement may justify migration
only when the quality-ratio upper bound is below 1.05 and capacity passes.
Otherwise report inconclusive or a documented tradeoff, not a winner.

Reserve held-out observation IDs and ordinals before seeing any metric. Use
selection ordinals only for adaptive allocation, and test refusal of overlap,
late held-out assignment and duplicate raw observation IDs. The allocation
rule and minimum/maximum selection passes must be frozen before execution.
Held-out data stays inaccessible to selection. Different seeds or different
vectors alone do not prove independence; document RNG width, exact same-input
seed collisions, retry determinism and empirical dispersion checks separately.

"Near optimal" is restricted to feasible tested architectures at the measured
scale and budget. It requires all relevant quality-ratio interval upper bounds
below 1.05, interval half-widths below the 0.05 margin, acceptable cost, and no
open correctness failure. Zero-capacity or degenerate zero-variance comparisons
cannot support equivalence or near-optimality. Insufficient precision means
insufficient evidence. This does not establish a global optimum over untested
models, generators or allocation policies.
