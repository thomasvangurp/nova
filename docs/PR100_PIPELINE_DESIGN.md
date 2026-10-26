# PR100 pipeline redesign — hypothesis to benchmark

The compatibility branch is not an optimal new pipeline. Its purpose is to
avoid invalid submissions during a rule change. The annotated census occupied
eight challenge buckets out of 171. The completed native-proxy census found
73 passers among 4,080 profiles, in two buckets. This used the recorded older
annotation closure and does not establish full pinned-validator acceptance.
Fourteen initial profiles all rejected, but that pilot was not a population
conclusion. Existing inventory has not demonstrated reliable all-cell supply.

## Alternatives

| Approach | Benefit | Main risk | Evidence required |
| --- | --- | --- | --- |
| Frozen inventory bridge with current gates | Small operational change | Mostly empty or rejected challenge pools | Actual coverage, accepted throughput and deadlines |
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
The estimand for a fixed ordered pair (A, B) is exp(mean over the ten matched
cases of log(S_A_case / S_B_case)), where S is the fleet mean of the thirteen
held-out per-sequence rank sums. Lower is better. Report the absolute fleet
scores, the per-case ratios and the geometric mean ratio. The ratio of pooled
means is not this estimand. The fixed family contains all three unordered
policy pairs, declared before capacity results, with both directional margins
obtained from each pair's same interval. No comparator is selected by score.
The compatibility policy is the reporting reference when feasible; otherwise
the online policy is the reporting reference when feasible. This changes only
table presentation, not the family of tested pairs.

There are zero permitted exclusions from a quality comparison. A policy that
fails capacity in any case remains in the capacity report and is ineligible
for a ten-case quality or near-optimality claim. Do not substitute another case
or silently compute a ratio over surviving cases. If fewer than two policies
clear every case, report insufficient feasible comparators and no relative
quality verdict. Keep the 5% quality and cost margins.

C2 may use an internal non-competitive reference field of at least 49 distinct
new-target sequences with all ten finite metrics, measured and frozen before
policy outcomes. The same field may serve every case if explicitly fixed in
the execution manifest. Its digest and row count accompany every case. It is
not a completed-epoch competitive field and cannot satisfy the C1/publication
reference gate. Its measurement cost is charged once to technical work; an
already retained technical observation is labelled reused, never NEW again.

Use paired case-level bootstrap intervals of the declared log-ratio statistic
across the ten matched cases, back-transformed to ratios,
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

## Pilot precision and scheduling addendum — 26 October 2026

Before the conditional pilot, use measured stage costs to derive the actual
per-arm number n of independent proposal attempts that fit both the 256-proposal
and 1,800 allocated GPU-second caps. CPU-only rejections still count in the
attempt denominator and have their CPU cost recorded; do not charge every cheap
reject the cost of a full GPU evaluation. Correlated siblings are reported by
reference family and do not justify an independent-binomial precision claim.
For independent Bernoulli trials and miss probability beta=0.05, zero successes
bounds the success probability above by 1-beta**(1/n); detecting a chosen
minimum yield y requires n >= ceil(log(beta)/log(1-y)). The often quoted
log(1/beta)/y is an approximation, not the exact finite-sample formula.

Predeclare the yield required by the measured 13-entry work/window inequality.
If the achieved n cannot resolve that yield, a zero is INSUFFICIENT_PRECISION,
not evidence that the generator is incapable. The resource ceiling can still
stop further spending, but report RESOURCE_LIMITED with the unresolved interval.
An actual rare-cell passer remains a positive existence requirement; a failed
bounded attempt does not establish impossibility for all generators.

The measured cold scenarios are 490.208/5=98.0416 seconds per fresh record and
672.25/10=67.225 seconds per repeat on one GPU, including repeated process/model
startup and CPU waits. At that repeat rate, 260 repeats alone cost 17,478.5
allocated GPU-seconds (4.855 GPU-hours), exceeding the four-hour technical and
existence allocation. Thus the cold configuration is not an approved execution
plan. A cold sample is neither a statistical upper bound nor a lower bound on
warm cost; exceeding it cannot prove that a warm implementation is infeasible.
A separately bounded technical throughput measurement is allowed to determine
whether warming changes the decision. No conditional-generation pilot or paired
comparison starts until the measured cost inequality fits its stated allocation.

Before an existence trial, reserve its full measured repeat, screening and
handoff cost in the technical allocation, including prior technical spend.
If it does not fit, report RESOURCE_LIMITED and do not begin a partial trial
that will be called complete. A budget revision would be a new registration,
with the old registration and results retained.

Do not hold paid GPUs idle waiting for a particular future challenge. The live
favourable-cell gate has a seven-calendar-day deadline from epoch 25955; the
chosen IGHV3-23:16 cell has probability 1/76 per epoch. Offline historical-hash
or chosen-cell dry runs may execute earlier and remain counterfactual. At the
calendar deadline, report LIVE_CHALLENGE_NOT_OBSERVED if appropriate; no automatic
conversion of the dry run into a live qualification occurs. Waiting does not
extend the six-hour GPU allocation, which counts idle time while allocated.

## Executable conditional-proposal baseline — version 1

`nanobody_opt/pr100_proposal_pilot.py` is a CPU-only experimental baseline,
not a qualified production design model. Its source digest, exact annotation
runtime identity and human ANARCI panel digest belong in the immutable pilot
manifest. The following choices are fixed before measuring proposal outcomes:

- Human V reference: lexicographically first allele of the requested gene in
  the pinned panel; J reference: first IGHJ4 allele. Assemble V states through
  104, the requested-length CDR3 and J states from 118. Preserve the J reference
  tail at states 115–117, canonical cysteines and the required H49/H50 residues.
  Actual annotation must confirm numbering and the requested cell.
- Select references by sequence hash, at most 32 in each of the three declared
  strata. No old-target score participates. The fourth stratum uses no historical
  sequence reference. Report both total available and selected reference counts.
- Referenced proposals resize the CDR3 prefix before the fixed J tail by seeded
  uniform deletion/insertion, then substitute ceil(prefix_length/4) positions,
  at least one. The no-parent arm draws that prefix uniformly from canonical
  amino acids excluding cysteine. Six eligible CDR1/2 positions are substituted
  using the same alphabet to permit diversity even at CDR3 length five.
- Seeds are SHA-256-derived from protocol, requested cell, seed 1000 and ordinal.
  This is deterministic proposal construction, not evidence of independent
  scoring draws. References cycle by ordinal. Duplicate sequences within an
  arm are not emitted twice. Stop at 256 distinct proposals or 2,560 attempts.
- Dry runs mask CUDA and allow at most 120 CPU wall seconds for emission, before
  annotation. Emit all sixteen cell/stratum records, including UNRUNNABLE arms
  with no reference. Nominal cell labels are never accepted as annotation proof.
  Annotate every distinct emitted sequence with the pinned runtime and report
  cell-match and rejection counts. This uses no GPU, native developability or
  target scoring; no dry-run result may claim a validator acceptance.

This baseline is intentionally simple enough to falsify cheaply. Altering its
mutation count, alphabet, reference selection or allele after observing its
results creates a separately versioned arm, with the first result retained.
It is not evidence that this proposal family is near optimal.

The three policy definitions for the eventual C2 comparison are:

1. Inventory bridge: the committed PR100 `prepare_pr100_bank.select` and
   `pr100_epoch.prepare` path; hash-ordered current-cell candidates from the
   frozen inventory, current gates, target-specific NEW exclusion, and the
   separate cached comparison lane. This is the actual compatibility baseline;
   it has no implicit unimplemented conditional generator.
2. Online conditional policy: begin the above generator only after the actual
   challenge is known. Cycle runnable reference strata in the stated order,
   with equal proposal allotments. Use current annotation/native/novelty gates,
   then score genuinely NEW target-specific candidates up to the budget/window.
3. Bank policy: use exactly the same generator and gates before the challenge,
   in rounds over the 171 cells, filling thirteen accepted distinct candidates
   per cell before any surplus. Replenish consumed entries in the same rounds.
   All preparation cost and unselected cells are charged. A four-cell pilot is
   not a prebuilt 171-cell bank. If measured preparation cannot fit the declared
   allocation, this policy is RESOURCE_LIMITED and is not compared as if its
   preparation were free. A live scheduler implementation is still required.

Reference and scheduler policies are fixed here; their existence does not
substitute for the CPU dry runs, runtime identities, generator-yield pilot or
three real policy executions. The bank/online schedulers are implementation
obligations before a paired comparison can run. The pilot can decisively show
that this proposal family needs revision without consuming the comparison
budget. Such a result is not a working-strategy or near-optimality verdict.

## Measured capacity sufficiency — 26 October, before further GPU comparison

The 128-record, four-GPU measurement does not establish C1 capacity. With the
current 361-block epoch and the shared 13-slot timing plan, scoring must end
76 blocks before the boundary: W = (361 - 76) * 12 = 3,420 seconds. That reserve
includes publication, transfer and commitment. It cannot be borrowed for work.

| Reading of the retained measurement | GPU seconds / record | Implied GPU equivalents for 4,800 + 260 | NEW at four GPUs | NEW at nominal twelve |
| --- | ---: | ---: | ---: | ---: |
| First chunk, initialization excluded, slowest shard | 11.7013 | 17.31 (18 rounded up) | 909 | 3,247 |
| Warm eight-record partial chunk, slowest shard | 15.6081 | 23.09 (24) | 616 | 2,369 |
| Whole job including startup, four times 542.352 / 128 | 16.9485 | 25.08 (26) | 547 | 2,161 |

This is sensitivity arithmetic, not a qualified throughput forecast. The first
reading subtracts the elapsed design-load event and the folding load duration
from the first 24-record completion, then divides by 24; it excludes startup
and final collection and is optimistic. The second uses actual partial chunks,
not full-size steady-state work. The last charges all four allocated GPUs.
All rows additionally assume t_rep = t_new, which is unmeasured, and T_other = 0,
which omits generation and other GPU work. The master has eight GPUs in addition
to the four-GPU test slave; the nominal twelve column assumes equal speed and
full availability across unequal hosts, neither of which is qualified.

Decision: C1 is unsupported by the observed configuration. Do not start the
large GPU strategy comparison or buy hardware from this projection. Measure a
qualified warm configuration and its actual repeat rate, then recompute using
the live window and measured critical path. A separately reviewed smaller
production contract remains possible; C1's 4,800 completion bar is not silently
waived. The user's 4,800 NEW cap remains distinct from repeats and cached reads.
The arithmetic and source digest are retained in capacity-sufficiency-20261026.json.

The v1 reference-stratum contrast is explicitly CDR3-conditional on shared
framework draws at the same cell/ordinal. Its four arms are paired, not
independent binomial samples. Changing seeds now would create v2; retain v1.
Native pass rates alone do not demonstrate thirteen distinct accepted entries:
report duplicate tiers, feasible pairwise-novel subset counts and, where an
exact optimum is computed, its solver certificate. Do not label a greedy
independent set a proven maximum. Real IMGT hallmark and length distributions
must come from the retained actual annotations, not synthetic string indices.
