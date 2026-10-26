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

The third approach is the preferred hypothesis, not a proven optimum. Online
generation is its bounded replenishment path. Benchmark all three against the
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
