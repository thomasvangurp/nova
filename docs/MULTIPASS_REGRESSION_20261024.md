# Multipass regression and selection repair — 2026-10-24

Lower scores are better. The Pi comparison uses each epoch's preceding ten
validator fields and independent stored replay folds; numbers across epochs
have different fields. Compare our fleet and the rival on the same epoch.

| Epoch | Our replay mean | 5Gh replay mean | Our selected local mean |
| --- | ---: | ---: | ---: |
| 25901 | 1096.4 | 1035.2 | 1116.9 |
| 25902 | 1311.2 | 1086.4 | 1322.5 |
| 25903 | 1892.4 | 1121.9 | 1843.6 |
| 25904 | 2340.0 | 1178.1 | 2304.9 |
| 25905 | 1095.5 | 1054.1 | 1100.1 |
| 25906 | 1764.3 | 1106.1 | 1718.3 |
| 25907 | 1590.6 | 1210.7 | 1540.7 |
| 25908 | 1803.3 | 1201.0 | 1757.7 |

The local columns come from the Pi's archived submission artifacts. These can
be an intermediate fleet revision: epochs 25901–25903 have a one-entry cohort
difference from the final replay. This cannot explain the large regression.
Epoch 25904 has the same thirteen entries and 20–283 local observations each.
The field had 433 rows in 25901 and 437 in 25904. We selected a weak pool, rather
than a strong pool unexpectedly collapsing on replay.

## Confirmed failure paths

1. Previously scored candidates were excluded from the new-evaluation budget,
   correctly, but the confirmation ranker required a current-epoch raw file.
   The publisher required twenty observations. Candidates with 1–19 stored
   folds could therefore enter neither path. In 25901, 3,856 current-novelty
   passing proposals were in this gap, including 1,305 descendants of selected
   5Gh parents. These counts do not imply they all pass full Layer 2.
2. Already scored single-edit proposals consumed generation limits before
   filtering. In 25901 only 104 new measured-parent proposals remained; 4,696
   of the 4,800 admitted entries were historical fallback proposals. Parents
   were present (54 of 155 selected parents belonged to 5Gh); downstream
   generation and selection discarded their usefulness.
3. Candidate assembly stamped its reusable marker before child generation and
   merges completed. Cancellation could make an incomplete singles-only CSV
   reusable. The code establishes this failure mode; the retained logs do not
   establish which interruption first produced the singles-only artifact.
4. For unregistered lengths, the pinned fallback design worker deleted each
   chunk's inputs and wrote replacement YAML only while building its initial
   configuration. Later chunks raised `No designs found` in
   `FromGeneratedDataModule`. Epochs 25903 and 25904 exposed zero completed
   tranches despite reserving 2,746 and 2,525 entries respectively. Epoch 25908
   exposed zero although four interrupted tranches contained 1,106 completed
   measurements. The all-or-nothing aggregate hid those successful waves.

## Release invariants

- The 4,800 cap charges NEW discovery dispatches. Cached comparison and
  confirmation have separate accounting. Failed dispatch reservations remain
  charged so a restart cannot silently exceed the cap.
- Precheck the stored own-fold bank on current novelty and IgBLAST, with
  epoch/target/archive/config/gate hashes. Rank eligible cached and fresh folds
  on the same current field. Candidates below twenty can earn confirmation;
  candidates already at twenty compete through the publisher without another
  tournament. Full Layer 2 and publication depth remain mandatory.
- Cached prechecks run alongside fresh scoring with the owning epoch's live
  deadline. New-epoch work never joins an old precheck thread.
- Exclude evaluated proposals before the generation limit. Mark candidate
  assembly reusable only after every merge, binding the marker to its hash.
- Every ordinary-design chunk writes its own inputs. Preserve the registered
  worker argument schema and diffusion-sample contract when repinning.
- Each successful wave creates an immutable partial aggregate. Choose only the
  latest checkpoint, or the full aggregate when present. Both retain the same
  observation ID so ingestion cannot manufacture extra independent passes.
- Run the committed shipping suite, including the actual pinned worker's
  repeated-chunk input test and interrupted-wave ingestion test, before deploy.
  Record installed hashes and live verification separately from a CI pass.

The target of 900 is an objective, not a measured guarantee or a license to
weaken eligibility. A weak confirmed pool must be reported as such.

## Deployment and recovered evidence

Runtime revision `730f4100356cfdb7fb563beac2ca97f467c5c1f1` passed 538 local
shipping checks (zero failures/errors/skips) and GitHub CI:
https://github.com/thomasvangurp/nova/actions/runs/40100041637 . Installed
runtime hashes and the scorer pin were checked against the tested artifact.
Master receipt: `/root/artifacts/selection_repair_release_20261024.json`.

The ordinary-design canary used two already measured length-123 candidates,
one candidate per chunk. It completed both in 114.6 seconds with
`validator-dual-sample-v1`, zero daemon fallbacks and zero empty-design errors.
The first attempt timed out because restarting only the daemon child retained
the supervisor's old scorer pin. The supervisor reads its settings before its
loop: repinning requires restarting the supervisor as well as its workers.
Do not infer loaded-code identity from the settings file alone.

Validated retained wave hashes and protocol before recovering 4,488 raw
observations from fourteen interrupted tranches: 1,318 from 25903; 1,355 from
25904; 1,106 from 25908; 709 from 25909. All fourteen imports sealed, with zero
incomplete observations or refused files. Recovery reused the original run
identities and structures and performed no GPU evaluations. Receipt:
`/root/artifacts/partial_wave_recovery_20261024.json`.

Activated during the idle close of 25910. Its completed confirmation remains
untouched. The unstarted 25911 proposal manifest was preserved under a backup
name and rebuilt; its discovery ledger had no reservations. Epoch 25911 is the
first epoch scheduled to use the corrected generation and comparison paths.
Its resulting fleet multipass score must be measured after submission.

## Fixed-field audit and broader confirmation

The follow-up audit ranked all retained replay folds against the same 470-row
reference field (epochs 25898–25907). Paired fleet means were 1262 vs 1259 for
us and 5Gh in 25873–25882, 1190 vs 1074 in 25883–25894, and 1631 vs 1148 in
25900–25908. The early gap was slower improvement; the recent decline remains
after controlling for the changing reference field. Even our recent best two
were 14% worse than 5DZX's smaller fleet. Do not explain this as fleet size alone.

One-pass rankings are uncertain: among 1,525 replayed public candidates,
retaining the first-pass best 20% captured 40% of the held-out best decile;
five-pass screening captured 56%. These selected submissions cannot estimate
the production-wide false-negative rate among rejected proposals.

`broad-confirmation-v1` reallocates the existing repeat budget. The actual
shipping control is 64/41/26/20 candidates, five repeat rounds at each width
(65% survival), costing 755 repeats. The new 8-GPU schedule is 147 candidates
for two repeats, 40 for five, and 20 for thirteen: 3, 8, and 21 total passes
for candidates starting with one. It costs 754 repeats and no more modeled
wall time. The 4,800 NEW-only discovery ledger remains independent. Other
widths/GPU counts are priced against their own control schedule; narrowing
preserves depth and refuses a window that cannot finish even one candidate.

One sixteenth of a broadened shortlist gets second-chance allocation, split
between measured-parent merit and an epoch-deterministic random audit. All
entries must already belong to the eligible comparison pool and pass full
Layer 2 before repeats. A parent score grants only a repeat opportunity.
The one-pass metric-outlier heuristic does not cancel those opportunities;
strict publication eligibility remains in force. The gated schedule and
input hashes are frozen before GPU work so resume cannot widen its budget.

The committed numerical fixtures contain no sequences or candidate IDs.
On 1,000 separate-fold replay splits, the new policy's held-out top-13 mean
was 1027.40 vs 1027.60 on 346 public candidates with 30 folds, and 1033.89 vs
1036.03 on 367 candidates with 40 own/public folds. These gains are small;
this is a conditional rank-only regression check, not evidence of recovering
the full competitive gap or achieving 900. Eligibility and family selection
are tested separately. Parent metadata is absent from the numerical benchmark,
so all second-chance slots there use the random audit fallback.

CI now requires this replay check, verifies the benchmark's control against
the actual deployment argv, and exercises stage depth, budgets, full gates,
restart identity, and incomplete qualification. Run
`python -m tools.benchmark_confirmation_policy` to reproduce the numerical
check. A round-complete marker is insufficient: `confirmation_health.json`
reports qualified and newly confirmed counts, and propagates degradation to
the epoch's coverage and confirmation receipts. No fresh confirmed candidates
or a pool below the configured minimum is explicitly degraded.

Activation: runtime commit `93e2f7b2dae0cfc674af55b50df85fae4dc8b6e4`
passed 580 local shipping checks with no skips, failures, or errors, and
GitHub CI https://github.com/thomasvangurp/nova/actions/runs/40109977179 .
The full 635-file package was verified on the master; both replay cohorts
also passed under its runtime Python. Ten installed files matched the tested
hashes, and the restarted producer carried the explicit broad-policy flag.
The producer was restarted only after 25912 completed; 25913 is the first
epoch using the policy. Receipt:
`/root/artifacts/confirmation_policy_release_20261024.json`.

Epoch 25912 finished 4,800 discovery evaluations and twenty confirmation
rounds. The sealed DB contains 4,807 distinct candidates for that epoch,
of which exactly 4,800 have their first own raw evidence in 25912 (the other
seven are cached confirmations). Its best locally confirmed score was 882.33.
That is one candidate's local estimate, not an independent fleet replay mean.

## Follow-up: eligibility attrition defeated the broader race

The subsequent screenshot is the validator's per-epoch score distribution,
not the multipass chart. Both show continuing weakness. Independent replay
means from the Pi were 1835.67 vs 1084.65 for us/5Gh in 25912, then 1077.03 vs
1045.93 in 25913. Our 25914 mean was 1825.28 at ten stored passes; the rival's
25914 replay was not complete at the time of this audit. Do not compare it
against a missing rival mean or call one good epoch a recovery.

The prior repair was incomplete. It budgeted 147 **proposals** before full
Layer 2, then accepted the survivors without refilling. In epochs 25913–25915,
only 21, 24, and 26 of 147 passed. Deadline fitting narrowed 25913 further to
17 at the first stage. The resulting confirmation pools contained only 16 and
17 qualified entries in 25913 and 25914. Meanwhile the comparison rankings
contained 908, 781, and 658 candidates with current-version cached passing TNP
profiles. A profile alone does not establish current novelty or all Layer-2
eligibility, but these candidates warranted consideration through the gate.

Of the rejected inputs, 118/126, 113/123, and 109/121 had **no measured TNP
profile**, rather than measured red flags. Sampling twelve such inputs from
25914 found seven failing the structure builder's existing numbered-domain
requirement and five passing that prerequisite. An isolated uncached retry
of one of the latter completed in 23.87 seconds with all six flags green.
Thus unavailable results include both deterministic input failures and
recoverable execution failures. The evidence does not establish the exact
fraction attributable to each or prove an OOM for every failed job.

The producer requested 200 concurrent TNP processes; its GPU shim distributed
them without limiting occupancy. With approximately 3.8–6.3 GB free per GPU
alongside resident scorers, this was an unsafe scheduling assumption. The
repair separates TNP concurrency (16) from IgBLAST concurrency (200), and
uses host-wide OS locks to allow at most two TNP processes on each GPU. The
same upstream executable, arguments, model and eligibility thresholds remain.

Confirmation now targets **passing** seats. It skips measured cached failures,
checks the reserved second-chance sample early, and refills rejected places in
rank order. Uncached work gets bounded batches and a separate time allowance;
before that expires, measured cached passes can refill the pool through the
full current gate. The complete repeat schedule and finish guard are reserved
first. A late start narrows the schedule before gating rather than returning
an empty race because the original widest schedule no longer fits. The 4,800
NEW discovery cap and 755-repeat control cap are unchanged.

Each gate runs in an owned process group with a timeout. Timeout kills its
descendants, and a stale output file cannot become a successful result. Per-
batch receipts distinguish `developability_unavailable` from measured
`developability` rejection. Unmeasured failures never pass and are not cached
as biological decisions. Final plans remain frozen across restart.

Shipping CI now requires the previously omitted Layer-2 cache suite, plus
eligibility attrition/refill, deadline/descendant cleanup and GPU-slot exec/
release tests. The rank-only replay benchmark alone could not detect this
runtime attrition bug. A passing CI run validates these failure paths; the
first live gated width, completed depth and subsequent independent replay
must still be checked before claiming competitive recovery.

### Why 25870–25890 behaved better

The dated cutover record in `PARENT_SELECTION_LESSONS_20261022.md` places
`unseen-candidates-v2` at 25892. Before that, 25889 and 25890 evaluated only
799 and 623 genuinely new candidates while reporting 4,785 and 4,788 covered.
Those coverage counts do not show the source of the submitted winners. The
submission-level audit below finds that 96 of 101 submissions in 25883–25890
were first locally evaluated in that same epoch, correcting the earlier
inference that the better fleet mainly relied on older candidates.
The user's NEW-only budget requirement was correct; the implementation
incorrectly coupled discovery exclusions to comparison, and left candidates
with 1–19 stored folds outside both confirmation and publication. After the
new exclusion was enabled, filtering evaluated proposals after generation caps
also exposed the exhaustion/fallback problem documented above.

One-pass selection optimism was already measurable in the better window:
retained 25889/25890 finalists scored approximately 39%/24% worse on subsequent
draws than on their selected initial draws. Likewise the 200-process TNP
setting predates the regression. Neither is, alone, a demonstrated explanation
for its onset. Distinguish the original discovery/comparison and generation
failures from the later gate-attrition defect that defeated broader confirmation.

### Eligibility repair release

Runtime `e649cebe1bead936d228042649fba614dd96c794` passed 592 mandatory local
shipping checks with zero failures/errors/skips, and GitHub CI:
https://github.com/thomasvangurp/nova/actions/runs/40127567333 . All 639 packaged
files were verified on the master; thirteen installed files matched that
artifact, including the live TNP shim. A real full Layer-2 canary accepted 147
of 147 cached profiles in 2.09 seconds under the current configuration. This
does not itself assert novelty, fleet diversity or competitive replay quality.

The release was installed during 25916 discovery, whose active race has zero
repeat rounds. Its closing successor will import the new confirmation module.
The in-flight discovery process was preserved. A master-side tmux job
`nova-eligibility-producer-refresh` waits for completed confirmation and no
active race before reloading the long-lived producer; the next launch argv
explicitly sets TNP concurrency to sixteen. Until then the shared GPU-slot
wrapper already bounds newly launched TNP jobs. No Mac process owns this
handoff. Receipt: `/root/artifacts/eligibility_refill_release_20261024.json`.
Live gated width, completed depth and independent replay remain pending.

## Evaluation of reverting the earlier configuration

The user requested evaluation, explicitly not a live reversion. No runtime
configuration or release was changed during this evaluation. The reference
is the verified `5c470d8` incident release and recorded launch contract used
from 25872, compared with runtime `e649ceb`. Historical releases installed
scoped files; this is not a claim to possess a complete 25890 machine image.

### What actually changed

The versioned producer launch configuration differs by only two added options:
`--confirmation-policy broad` and `--tnp-concurrency 16`. AST comparison of the
producer, race, publisher and ranker's argument defaults found no other changed
defaults beyond the new discovery-budget argument. Parent limits, parent-merit
selection, the core local-neighborhood generator, basin settings, score formula,
twenty-observation publication floor and the 65% control pruning fraction did
not receive a parameter retune. Candidate admission and evidence flow changed
substantially inside the implementation, despite largely unchanged CLI settings.

| Change | Verified boundary | Reversion assessment |
| --- | --- | --- |
| NEW-only discovery, separate durable reservations | 25892, `d4f7974`; the interim `2309bbe` receipt also names 25892 as its first fresh epoch | Preserve honest accounting and the requested NEW-only cap. Reverting the whole implementation restores cached coverage being treated as completed discovery and the old mixed screening/deepening path. |
| Stored-bank comparison and prechecks | 25894 `6c23044`, 25899 `9b49663`, sparse-bank confirmation restored in 25911 | Preserve eligibility-before-cutoff and access to stored evidence. The first partial repairs had real omissions, not a reason to remove comparison. |
| Generation exclusions before caps, complete assembly markers, repeat-chunk inputs and partial-wave recovery | 25911, `730f410` | Keep the corrections. A whole-tree rollback restores reproducible work-loss and fallback-generation failure paths. |
| Broader confirmation, deterministic second-chance sample | 25913, `93e2f7b` | A legitimate policy to challenge. Evaluate the old schedule within the corrected gating/deadline implementation; merely switching to the legacy branch reintroduces unfilled eligibility seats. |
| Passing-seat refill, bounded gate processes and host-wide TNP slots | 25916, `e649ceb` | Keep resource limits and complete eligibility. These do not establish that the broader allocation improves quality. |

### Fresh finalists disappeared while confirmation became relatively scarce

The audit matched actual validator-scored US submissions against retained,
independent `local_exact_raw` observations up to each submission epoch. All
submitted candidates in these periods had retained own raw evidence. “Same
epoch” means first **locally** evaluated then, not a claim about every possible
public measurement or every proposed candidate.

| Period | Submitted candidates | First locally evaluated that epoch | Had earlier-epoch local evidence |
| --- | ---: | ---: | ---: |
| 25873–25882 | 123 | 103 (83.7%) | 20 |
| 25883–25890 | 101 | 96 (95.0%) | 5 |
| 25891–25894 | 39 | 8 (20.5%) | 31 |
| 25900–25908 | 115 | 17 (14.8%) | 98 |

In particular, every scored US submission in 25889 and 25890 was first locally
evaluated in its submission epoch. The earlier implication that those better
fleets predominantly came from reused inventory was unsupported. The confirmed
sparse-bank omission remains a defect, but this audit does not establish its
individual contribution to the fleet decline. The strongest observed change
is the failure to deliver competitive newly evaluated finalists.

The raw work ledger exposes a separate allocation issue:

| Epoch | Screen observations | Candidates in main confirmation | Main confirmation repeat observations |
| --- | ---: | ---: | ---: |
| 25889 | 799 | 51 | 625 |
| 25890 | 623 | 49 | 600 |
| 25901 | 4,523 | 40 | 530 |
| 25914 | 4,800 | 24 | 428 |
| 25915 | 4,800 | 26 | 442 |

These columns classify raw paths as screening or race rounds. Other recorded
raw work exists and is retained separately in the numeric audit; do not call
this table the total GPU bill. The old 64-proposal confirmation capacity was
approximately 8–10% of 623–799 new evaluations. Even a fully filled new 147-seat
pool is at most 3.1% of 4,800, and cached candidates can also occupy those seats.
Thus increasing discovery without similarly increasing confirmation changes
the selection problem, independently of the implementation defects. This is
an allocation trade-off to test, not proof that 4,800 NEW evaluations are bad.

### Controlled allocation replay

The offline comparison used the same candidates, first observations, disjoint
held-out folds and near-identical repeat budgets for both schedules. It ran
3,000 splits per cohort using three new random seeds, without GPU scoring.
Parent metadata, original proposal pools, full live gates and family selection
are not available in these fixtures, so this compares allocation only.

| Measured cohort | Old 64→41→26→20 | Current 147→40→20 | Current minus old |
| --- | ---: | ---: | ---: |
| 346 candidates, 30 folds | 1027.27 | 1027.29 | +0.02 |
| 367 candidates, 40 folds | 1036.90 | 1033.53 | −3.37 |

Lower is better. The old schedule costs 755 repeats, the new one 754. Training
uses 21 folds and evaluation uses the remaining 9 or 19 within each split.
Conditional Monte Carlo 95% intervals for the differences are −1.30 to +1.34
and −4.51 to −2.22. These intervals quantify random split variation for these
selected cohorts; they do not establish generalization to the live 4,800 pool.
Removing the random second-chance sample gives deltas −0.66 and −3.31, also
too small here to explain the several-hundred-point production regression.

The newer allocation has not demonstrated a material production advantage.
Its small original rank-only benchmark gain and passing implementation tests
were insufficient to establish that. Restoring the old confirmation schedule
inside the corrected orchestration is a reasonable controlled comparison;
there is no evidence here to justify a whole-repository rollback or to promise
that old settings will reproduce old scores against today's eligible pool.

For the next policy decision, compare old and current allocation on the same
current eligible candidate pool, preserve independent evaluation passes, hold
repeat/time budgets fixed, and record the number of NEW candidates reaching
confirmation and publication. Separately evaluate discovery/confirmation balance
without silently changing the user's 4,800-NEW requirement. Keep deadline,
eligibility, evidence-preservation and budget-accounting fixes in both variants.

Numeric evidence: `docs/evidence/config_revert_evaluation_20261024.json`.
Reproduce the allocation comparison with:
`python tools/evaluate_confirmation_rollback.py --output /tmp/rollback-replay.json`.
