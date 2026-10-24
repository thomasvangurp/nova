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
