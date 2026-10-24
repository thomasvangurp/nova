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

## Epoch 25914: good confirmation results missed the Pi's submission cutoff

The direct 25913/25914 comparison identifies an additional delivery defect.
This audit made no live configuration or runtime changes. Epoch 25913's
independent replay mean is 1077.03 for twelve entries, all with thirty passes.
The current 25914 snapshot is 1815.86 for thirteen entries with ten to thirty
passes. Rival 25914 replay remains incomplete. The changing replay depth means
the 25914 figure supersedes, rather than contradicts, the earlier snapshot.

The weak tail was already visible before submission. The Pi's archived 25914
input contains five entries locally scoring 2533.02–2545.40 with twenty to
fifty-four observations; their later replay scores are 2467.20–2611.20. Across
the twelve archived entries matched to replay, local/replay means are
1707.83/1754.14. For 25913 the corresponding means are 1071.97/1077.03. The Pi
rebound the sixteen-entry sequence pool to thirteen crew members in 25914;
the additional entry replayed at 2556.50. Do not conflate the twelve-entry
archived assignments with the thirteen-entry actual fleet. First-pass noise
does not explain this particular weak tail.

Seventeen stronger 25914 candidates eventually completed twenty-one local
observations, with scores 944.67–1451.95. All seventeen appear in the final
publisher ranking, pass comparison precheck, and pass the final novelty and
Layer-2 checks. Their existence corrects any inference that this epoch simply
failed to produce enough qualified alternatives. They arrived too late for
the Pi's frozen submission input:

| Event on 2026-10-24 | 25913, UTC | 25914, UTC |
| --- | --- | --- |
| Checkpoints in the artifact eventually pulled | 17:10:59–17:11:36 | 18:27:25–18:28:00 |
| Pi's final pull | 17:20:55 | 18:28:51 |
| Commit process completed, zero failures | 17:22:34 | 18:30:23 |
| Strong final confirmation file written | Earlier file no longer retained | 18:31:06.949 |
| Final publisher ranking written | Earlier file no longer retained | 18:35:04.613 |
| Final publisher Layer-2 output written | Earlier file no longer retained | 18:35:12.845 |

The 25914 confirmation file therefore arrived 136 seconds after the final
pull and 44 seconds after the commit process completed. Final publisher
checks followed approximately five minutes after commitment. These local
scores are not independent replay or a guaranteed counterfactual fleet mean,
but the later passing candidates could not improve an already committed fleet.
The actual Pi log establishes the pull and commit times. Archive `copy2`
preserves source modification times, so an archive's mtime cannot establish
when commitment happened. The master's per-entry planned commit blocks are
also not the Pi's actual schedule: the submitter loads the pulled input once
and commits using its own orchestration.

### The producer and the Pi budget against different deadlines

The master planned a 36-block commitment reserve and zero publication
preparation blocks. Its adaptive handoff reserve is the sum of those two
values. The Pi independently reserves `16 + 5 * crew_size` blocks, then ends
its late-commit wait another 120 seconds before that deadline. For thirteen
crew members, this is nominally 81 + 10 = 91 blocks before epoch end at the
assumed twelve seconds per block. The producer can therefore budget work
55 blocks, approximately eleven minutes, beyond the Pi's intended final-pull
cutoff, even before allowing for publication processing. Polling and transfer
latency affect the exact pull time; the observed logs demonstrate the miss.

Both epochs used the broad policy, but 25913's good submitted estimates were
ready before its final pull, while 25914's stronger final results were not.
This explains the observed difference without attributing it solely to random
ranking noise or claiming a configuration rollback would fix delivery. The
eligibility refill release `e649ceb` does not align these deadlines, so that
repair alone cannot establish that the timing defect is resolved.

### Required correction and regression coverage

Use one explicit, epoch-scoped handoff contract based on the Pi's actual
final-pull cutoff and current fleet size. Budget confirmation backwards from
that cutoff, reserving measured publisher preparation and transfer time.
Retain the conservative commit safety window until both parallel operation
and serial fallback have been evaluated; do not simply shorten the Pi's
reserve to conceal the mismatch. Preserve the 4,800-NEW accounting contract;
if measured capacity cannot fit discovery, confirmation and handoff, report
the capacity shortfall instead of counting late work as useful delivery.

CI must exercise the producer, publisher and Pi timing contract together:
thirteen crew members, the late-pull margin, nonzero publication latency,
eligibility refill, a final ranked pool reaching the Pi before its snapshot,
and late old-epoch work remaining isolated from the next epoch. A completed
confirmation marker or a healthy publisher process does not prove delivery.
Record the committed artifact's identity and quality separately from later
master-side results, and alert when good finalists miss the final pull.

Before any implementation, reconcile the live Pi loop against the versioned
file: the relevant reserve logic matches, but the whole-file hashes differ.
Do not overwrite unrelated live changes. Run the required shipping CI before
deployment, then verify the actual Pi snapshot and commit receipt in the
first affected epoch. No timing correction was deployed during this audit.

Numeric evidence: `docs/evidence/epoch_25914_handoff_audit_20261024.json`.

### Timing correction implementation

The shipping producer and publisher now opt into `pi-handoff-v2`, defined in
`nanobody_opt/submission_timing.py`. The Pi imports the same commitment reserve
and late-pull margin. With thirteen crew members the final-pull reserve is
41 blocks. Publication must finish at end-46, leaving sixty seconds for
transfer. Scoring must finish at end-76, leaving six minutes for publisher
polling, ranking and gates. That allowance covers the observed approximately
246-second final-confirmation-to-gate delay plus the 45-second polling interval
with margin. It is an operational allowance, not proof every workload fits.

The publisher records the current fleet capacity for the producer; both retain
a thirteen-slot floor even if registrations have not propagated. Larger
observed fleets increase the reserve. Existing larger explicit reserves remain
effective. The 4,800 NEW-only discovery cap and confirmation evidence thresholds
are unchanged. CPU preparation for the following epoch can use the idle
post-handoff window, but remains bound to the real current epoch's cancellation
clock so it cannot occupy the next epoch.

Each newly built artifact records its timing policy and the digest of the
completed confirmation receipt observed **before** publication starts. A
confirmation finishing during publication cannot falsely label an earlier
snapshot as final. A retained previous artifact retains its original receipt.
The Pi logs the digest of the exact fetched artifact and its confirmation
receipt; a policy-enabled fallback without completed confirmation produces an
alert while retaining the existing validated fallback submission behavior.

`tests/test_submission_handoff.py` is mandatory in shipping CI. It tests the
actual launch profiles, adaptive budgeting, fleet growth, confirm/publish/pull
timing, the publisher's live deadline guard, stale receipts, and preparation
across rollover. The old 36-block reserve fails the same handoff invariant.
Activation and live delivery verification are recorded separately below.


The user explicitly requested a less conservative Pi reserve before deployment.
Thirty recent log records (25888–25917) show 12–13-entry commit processes taking
78–98 seconds, with no recorded repairs, failures or blackout drops. The
parallel reserve is now sixteen blackout blocks plus twenty commit/repair
blocks (four minutes) for up to twenty UIDs. Larger fleets receive additional
batch allowance. The final-pull margin is sixty seconds, instead of 120. For
thirteen UIDs this moves the intended final pull from end-91 to end-41, about
ten minutes later, while leaving five minutes from pull to blackout. A full
serial fleet repair is no longer reserved on every ordinary parallel epoch;
existing best-first repairs can use the remaining bounded window.

CPU preparation still opens at end-136. Its elapsed time is now charged to an
epoch-end clock anchored before preparation, rather than accidentally extending
the chain deadline. The Pi's subprocess stream uses a timed queue and an owned
process group: a silent RPC cannot block its stdout iterator indefinitely.
The submission process is stopped before the blackout (with one extra block
for cleanup), and cannot occupy the next epoch. CI exercises this with a real
silent process and descendant, and verifies the next invocation can run.
Numeric timing evidence: `docs/evidence/pi_commit_timing_20261024.json`.

The valid submission window remains the hard constraint. Verified against
the published upstream `neurons/validator/commitments.py` and `config/config.yaml`
on 2026-10-24: `epoch_start < inclusion_block < epoch_end - 10`. The last
accepted block is therefore end-11. The sixteen-block value in our scheduler
is an additional operational guard, not the current upstream exclusion value.
The Pi's existing signer checks the live remaining blocks before its batch and
repairs; the outer process budget now ends at end-17, leaving cleanup time and
additional margin before the validator boundary. Shorter expected runtime does
not extend that valid window. The shipping suite executes the validator's
strict predicate on both boundaries and checks the actual Pi commit timeout.

### Deployment verification

Runtime release `f6184322fc288380a9fea80f90d7e187d3b88af2` passed all 608
mandatory shipping checks and [GitHub CI](https://github.com/thomasvangurp/nova/actions/runs/40141311873).
All 642 staged package files were verified on the master. The Pi's existing
SANQI and compound-window extensions were preserved using a targeted overlay;
the resulting Pi variant passed the sixteen handoff tests separately. The
previous hosted run exposed a test fixture startup race on Linux; the fixture
now waits for child readiness before testing cancellation. Production runtime
bytes did not change for that test correction.

The Pi activated the new timing in epoch 25918. The master's one-time activation
job runs on the master itself, so closing the Mac cannot prevent activation.
It waits until current race work and the protected publication window have
finished, verifies the installed runtime hashes, preserves the live launch
arguments, and checks that both replacement processes loaded `--pi-handoff`.
The 4,800 NEW-only cap remains in the live producer arguments. Both running
master processes were verified with the new flag in epoch 25918;
the first full affected epoch is 25919. The Pi reported thirteen committed,
zero failed and zero dropped to blackout for its first changed epoch, 25918.
That epoch still used the previous master scheduling policy. The final
activation receipt and verification details are recorded in
`docs/evidence/handoff_release_20261024.json`; the exact activation code is
retained beside it as `handoff_activation_20261024.py`.

Deployment verification does not establish a new multipass result. Quality
must be checked against the actual committed fleet and subsequent independent
replay, rather than a later master artifact or a successful process exit.

### Why 25916 was good and 25917 was poor

Read-only audit at 2026-10-24 23:33 UTC; no runtime settings changed during this
comparison. The archived Pi inputs and independent replay agree about the
quality difference:

| Epoch | New evaluations completed | Confirmation rounds | Submitted local mean | Replay mean | Replay passes per entry |
| --- | ---: | ---: | ---: | ---: | ---: |
| 25916 | 4,800 | 20 | 1,035.26 | 1,036.63 | 30 |
| 25917 | 4,646 | 0 | 1,822.28 | 1,876.27 | 10 |

Both fleets have thirteen entries. In 25917, seven entries had a local mean
of 1,185.19 and replayed at 1,282.27. The other six were already locally
2,563.91–2,566.94 after 41–98 observations and replayed at a mean of 2,569.27.
Most deterioration was therefore visible before submission. This is not
evidence that those six unexpectedly collapsed on independent replay.

The 25917 stored-bank precheck first completed with 81,462 eligible candidates.
The reference archive then changed from 74,574 to 74,622 rows. Eligibility is
bound to the archive hash, so the previous proof could no longer be accepted.
All three closing races logged `ready=False cached=0`; their comparison pools
contained only the 4,646 fresh candidates. A full recheck of 219,172 stored
candidates subsequently produced partial eligibility too late for those races
and stopped at the deadline with 40,342 pending. The safe invalidation was
correct; the failure was allowing revalidation to withhold the usable bank
throughout confirmation.

The repeat planner compounded that failure. It fits repeat breadth after
reserving only thirty seconds for the eligibility gate. The uncached gate
requires sixty seconds plus a thirty-second fallback reserve before starting.
With no usable cached passes, the fitted plans could therefore allocate zero
gate calls. The recorded first two closing attempts targeted eighty and
sixteen seats, attempted zero candidates, and completed zero repeat rounds;
the third could no longer fit a schedule at all. All reported
`degraded_no_new_confirmed_candidates`.

An offline reproduction using the recorded timing model and those fitted
widths produces gate allowances of 33.42 and 37.60 seconds and zero gate calls,
even when the synthetic gate would accept every input. These are synthetic
remaining-block positions matching the observed widths, not recovered live
chain timestamps. The reproduction uses no GPU work or biological inputs.

The shared producer/publisher/Pi deadline fix does not itself repair this
interaction. The next correction must revalidate changed-archive candidates
in bounded batches, prioritize useful measured candidates without trusting
stale eligibility, and reserve actual uncached-gate time before choosing
repeat breadth. CI must combine archive updates, unavailable cached scope,
late confirmation and the final submitted fleet. Standalone refill tests with
an already available cache did not cover this path. The 4,800 NEW cap and the
submission validity boundary must remain unchanged.

At this snapshot the rival fleets for 25917 are incomplete, so no rival mean
is used to explain the difference. Evidence and the numerical reproduction:
`docs/evidence/epoch_25916_25917_confirmation_audit_20261024.json`.
