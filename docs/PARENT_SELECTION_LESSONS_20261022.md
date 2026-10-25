# Parent merit, epoch deadlines, and release evidence — 2026-10-22

Release-control follow-up (2026-10-25): [RELEASE_PROMOTION.md](RELEASE_PROMOTION.md)
defines enforced staging versus activation, paired production-scale evidence,
Pi consumer verification, and rollback that preserves durable work. Passing CI
alone is not production qualification. Synthetic regression cases must never
be reported as real replay improvement; the new qualification path remains
closed until its operational evidence and dedicated runner are available.

Operational correction (2026-10-25): deployment-control updates now have a
strict install allowlist, real recorded handoff fixtures, a runnable replay
driver on GitHub-hosted infrastructure, and a separate Pi qualification timer.
They preserve and attest the observed application instead of copying a generic
repository tree over it. The previous all-host inventory incorrectly treated
Mac/Pi tooling and tests as required GPU-host files. The application snapshot
and the tested control release are labelled separately; the former is never
silently presented as having passed the latter's CI. Scoring/selection changes
still require their own performance evidence.

Parents are ranked by the mean of independent pass ranks against the same current ten-epoch field, with at least ten raw folds. Historical scores, owner quotas, family reservations, and rank-of-mean-metrics must not replace this ordering. Repair-only parents keep their validity routing and capacity cap. Child proposals visit the measured parents in merit order before spending another edit on the same parent.

The full live audit for epoch 25872 compared 1,456 measured, gated parents and selected 155. The cutoff was 934.25, best 591.35, and top-twenty mean 706.2390625, against epochs 25860–25869 (371 rows). Five of competitor 5GhREcsK's eight epoch-25868 parents qualified at positions 21, 28, 34, 54, and 122. The other three scored 963.55, 1025.35, and 1099.55 on that same field and fell below the cutoff. This supersedes the earlier limited-bank comparison. Parent quality is evidence for generation, not a guarantee of child quality.

From epoch 25872, generate at most 8,000 cheap proposals, pass fresh eligibility checks, and admit at most 4,800 to expensive screening. Confirm at most 64 finalists with at least twenty independent observations before publication. Coverage counts successful raw measurements, not attempts or aggregate rows. Keep slot purchases value-priced with a floor of five and a ceiling of thirteen.

## Epoch 25871 incident

All 10,980 eligible candidates were screened. Twenty final candidates completed twenty repeat rounds plus their initial fold. The saved tournament's top-eleven measured mean was 743.3983; top-eight mean was 721.2679. These were not submitted results. The publisher remained occupied by epoch 25870's strict rank for over 100 minutes; the submitter committed eleven stale epoch-25870 nanobodies during epoch 25871. Never label an archive directory's contents as a new result without checking its embedded epoch.

The strict-pool fallback expanded a thin shortlist back to the full 11,012-candidate scope. Structure metadata was loaded before applying the observation floor, and child commands had no live epoch deadline. A pre-launch time estimate and a post-completion epoch check do not prevent old work from blocking the next epoch.

The correction bounds the strict pool even when thin, filters evidence depth before structure-history loading, and keeps the twenty-fold floor hard. Producer and publisher commands run in private process groups and check the live chain clock every second. Epoch rollover, stale clock, or the handoff reserve terminates the command tree. Publisher commands also have a five-minute wall-clock cap. Optional strict publisher upgrades remain disabled in the deployed launch contract; normal biological and novelty gates remain required.

The slave worker must see producer confirmation rankings without waiting for the publisher. New rival/replay batches yield during the critical submission window until confirmation and current publication exist. Already completed measurements remain reusable through the observation store.

Recovery found twenty completed epoch-25871 repeat files missing their sealed import records. The normal importer recovered 720 independent observation records and sealed all twenty runs. Recovery must retain at most 64 confirmed, policy-current finalists from the immediately preceding epoch, import their completed repeat evidence, and rank them against the current field. Publication still runs the current novelty, Layer-2 and independent-observation gates. Do not rerun historical GPU work or scan old screening directories on the publisher path.

## Release contract

Run `scripts/shipping_checks.py` from a clean committed tree. It stages committed source, compiles it, executes the explicit shipping regression suites, rejects missing/zero/skipped/failed tests, and packages those exact bytes with commit, lockfile hash, file hashes, and passing test evidence. CI runs this same command. Deployment depends on that CI job and downloads its package; it verifies staged and installed hashes before activation. Both producer trees must not be confused: production invokes `nanobody_opt_v2.run_live_producer` and ships `nanobody_opt_v2`.

A local test pass is not a GitHub CI pass. Normally deploy after GitHub CI succeeds for the exact revision. For this incident the user explicitly authorized the verified local shipping run; preserve that distinction and the passing receipt, and verify each installed file against it. Do not copy an unrelated build stamp or infer production health from a running tmux session. Verify the artifact's epoch and age, the confirmation depth, and the submitter's actual commit results.

Commit `f29c07b772cd3c980c55ccaf97662afb14ef7dd7` passed all 371 shipping tests locally and in [GitHub CI](https://github.com/thomasvangurp/nova/actions/runs/39804000282). Eleven incident runtime files were deployed from that verified artifact and checked by hash; producer, publisher and slave top-up restarted successfully. The receipt is `/root/artifacts/incident_release_20261022.json`. This was a scoped incident deployment, not a claim that every runtime file matches a full release.

The HTTPS credential could not push workflow changes. Authenticated SSH successfully pushed the same commit, including workflows; the local `payout-pr` remote now uses `git@github.com:thomasvangurp/nova.git` as its persistent push URL. This fixes workflow pushes without copying credentials or granting a broader token scope.

The slave top-up was also blocked by a stale reviewed daemon hash. Review of both live sources confirmed the master's added short-shard retry sends the same scorer arguments and merges each shard once; the slave's replicate-only plumbing remains disabled for these unique-sequence rounds. Update only the exact reviewed hash pair, retain rejection of any further drift, and verify live preflight after testing. Do not overwrite the master's pinned scoring implementation to make hashes match.

## Verified recovery

The final code revision is `5c470d838076b3b28e9f4dc23e53874f64efc039`, on branch `fix/epoch-deadlines-parent-merit-20261022`. Its clean local shipping run and [GitHub CI](https://github.com/thomasvangurp/nova/actions/runs/39806353073) passed 377 tests with zero failures, errors or skips. All eleven deployed incident files match this package by SHA-256; `/root/artifacts/incident_release_current_20261022.json` records the exact scope and receipt. The master scorer implementation was preserved.

Epoch 25872's fresh published fleet contains eleven candidates, each with 21 observations, averaging 785.207792 against its current ten-epoch field. The Pi's normal `pi_submit.py --pull-only` path successfully fetched this artifact, and its embedded epoch, count, mean and observation depths were verified. This is a local multipass prediction and verified delivery, not a final validator score or a claim that the epoch's on-chain commit has already occurred.

At epoch offset 193.1 the slave passed preflight and began a submission top-up pass: fifty candidates, twelve planned rounds and 600 requested observations. These are planned measurements, not yet a claimed successful import. At offset 196 the master had screened 2,486 of the 4,800 admitted candidates. Keep these progress snapshots separate from completed coverage and final confirmation.

Follow-up verification of epoch 25871 found eleven successful chain commitments but only one validator-scored entry (UID 49). The other ten candidates were already present in epoch 25870's scored field. A successful commit must never be reported as proof of validator acceptance; verify the actual leaderboard separately.

Epoch 25872 subsequently had all eleven fleet entries scored. The dashboard snapshot initially ended at 25871. Refreshing it also exposed a cohort bug: one of our sequences matched a rival's same-epoch sequence, and global sequence deduplication incorrectly removed our ownership. Deduplicate membership by `(coldkey, sequence)` while reusing the same raw measurements across owners. Revision `39940c1b340fb8c8a4ef36611cccd001428688fb` passed 381 local shipping tests and [GitHub CI](https://github.com/thomasvangurp/nova/actions/runs/39822514038), including shared-owner membership and incomplete-coverage regressions, before the final verification. The deployed file hash is recorded in `/root/artifacts/incident_dashboard_20261022.json`.

The rebuilt and published dashboard shows epoch 25872 at **849.130303**, all eleven candidates covered, 10–30 stored replay passes per candidate, against 388 rows from the preceding ten epochs. This retrospective dashboard calculation uses its stored replay evidence; the earlier **785.207792** was the live prepublication estimate with 21 observations each and its then-available field. Keep these labelled rather than treating them as the same snapshot. The rendered Multipass trace and successful Vercel publication were verified.

Regression coverage includes current-field parent merit, proposal/screen budgets, deferred TNP, value pricing, commit side effects, validator round trips, actual v2 imports, release tampering, and a real blocked-process rollover test that proves child termination and immediate next-epoch progress.


## Pi dashboard ownership (2026-10-23)

The Mac refresh and multipass LaunchAgents were still the data producers even
though Vercel deployment ran on the Pi. Closing the Mac stopped both upstream
steps. The master multipass exporter also stopped, leaving epoch 25872 as its
last published file. Recent Mac logs contained DNS failures, transfer failures
and timeouts. A successful deploy attempt was not proof of a successful deploy.

The tested dashboard release is `d50957bb38fadc6fb45b251ad5ff6648970a7b36`.
Both local shipping CI (438 tests, zero skips) and GitHub run
https://github.com/thomasvangurp/nova/actions/runs/39911431328 passed. One initial
local run failed the existing four-second epoch cancellation timing assertion;
the focused rerun and complete shipping rerun passed without relaxing the test.

Production dashboard jobs use `/opt/nova-dashboard` on the Pi. The template
`deploy/dashboard/nova-dashboard@.service` runs each job with its own lock and
health file. Refresh and publish timers run after 60 seconds idle; multipass
after 300 seconds; compound after 600 seconds; money and proxy after 3600 seconds.
A slow collector cannot hold the render/publish lock. Health preserves the last
success separately from the latest attempt; CLI output and exception messages
are never copied to logs.

The Pi resolves the current master through its existing authenticated Vast CLI,
requests a bounded recent-six-epoch export, and pulls the resulting comparison.
The master exporter writes newest epochs first and saves after each epoch while
retaining historical rows. A failed older epoch cannot erase newer progress.
Exports are file-locked; remote and local timeouts bound work. Invalid or older
comparison downloads cannot replace the last good file. An exported epoch and
a completely measured fleet are reported separately on the dashboard.

Dashboard source and these regression suites are now included in the release
manifest and blocking shipping checks. Deploy only files whose hashes match a
passing build stamp. Pi Python dependencies are pinned in
`requirements/dashboard.txt`; Vercel CLI is pinned to 65.1.0 and uses the Pi's
existing authentication. No credentials are copied from the Mac. Bootstrap
history with a consistent SQLite backup; redeployment must preserve live data.

Operational checks: `systemctl list-timers 'nova-dashboard-*'`,
`systemctl status nova-dashboard@refresh nova-dashboard@multipass nova-dashboard@publish`,
and the non-secret `data/dashboard_*_health.json` receipts. Verify the live
`/version.json` marker and rendered multipass panel before retiring Mac writers.
The Mac dashboard and multipass LaunchAgents must remain disabled after cutover.
AWS/Taostats histories retain their last known values if Pi credentials are
unavailable; chain balances and Vast collection use existing Pi access.


Cutover verified on 2026-10-23: both Mac LaunchAgents were disabled and unloaded;
all six Pi timers were enabled and active. After cutover, an automatic Pi publish
advanced the public build from `20261023T063203Z` to `20261023T063427Z`. The public
HTML's actual US box-plot trace contains epoch 25883 with 12 candidates and mean
1127.0666666666666. Epochs 25873–25878 were backfilled on a bounded Pi-triggered
job; the public distribution has no missing epochs from 25873 through 25883.
The cutover receipt is `/opt/nova-dashboard/data/dashboard_cutover_20261023.json`.


## Ohio destruction and Pi billing access (2026-10-23)

The user explicitly requested destruction of Ohio, including clarification
"destroy, not just shut down". In account `668100850015`, region `us-east-2`,
instance `i-0c26ae9fcae0edfd3` was terminated, its 2048 GiB root volume was deleted,
AMI `ami-019b75f505b72f62e` deregistered, and snapshot `snap-0ccf9c833173ee3fe`
(1600 GiB source volume) deleted. Both Elastic IP allocations were released.
The three subnets, two custom security groups, internet gateway, regional SSH
key-pair registration, and the now-empty VPC were deleted. Verification found
zero nonterminated instances, volumes, snapshots, AMIs, Elastic IPs, VPCs,
subnets, interfaces, security groups, gateways, key pairs or launch templates.
Inventories also found no Auto Scaling groups, spot fleets/requests, EventBridge
rules/schedules, Lambda functions, CloudFormation stacks, Ohio S3 buckets,
Lightsail resources/snapshots, RDS databases/snapshots, ECS/EKS clusters, load
balancers, EFS/FSx filesystems, OpenSearch domains, DynamoDB tables, ECR
repositories, CloudWatch logs/alarms, MemoryDB/ElastiCache/Redshift clusters or
backup recovery points. Cost Explorer's prior three-day Ohio charges involved
only EC2 compute, EC2 Other (storage), and VPC (IPv4). Historical charges remain
in billing records after resource destruction.

The earlier request to give the Pi AWS access uses a dedicated IAM user,
`nova-dashboard-pi`, in that same account. Its sole inline policy is the
versioned `deploy/dashboard/aws-cost-reader-policy.json`: only
`ce:GetCostAndUsage` and `ce:GetCostForecast`. Simulation confirmed both reads
allowed and EC2 launch/termination, IAM user creation and S3 object access
implicitly denied. No managed policies or group memberships were added.

The new credential was transferred in memory through SSH stdin directly into
the Pi's established systemd encrypted store. No root-profile credentials were
copied. `/etc/nova/dashboard-aws.cred` is host-encrypted and mode 0600 inside the
existing mode-0700 credential directory. Only the money service receives it,
via the versioned `nova-dashboard-money-aws.conf` instance drop-in. The AWS CLI
reads a service-lifetime credential file through the normal named-profile
provider; systemd removes plaintext service credentials on exit. Neither a key
nor its value appears in a unit, source file, command argument or audit log.

Release `36b9e8b0a536f899e8da697ddfbf66c90e1cdd91` passed all 442 shipping tests
with zero skips, plus GitHub CI:
https://github.com/thomasvangurp/nova/actions/runs/39913578935
before the tested files were deployed. AWS CLI is pinned to 1.44.74. An operator
can revoke this Pi identity independently through IAM without affecting the
Mac's administrative profile or any other miner.


Live verification exposed a CLI compatibility defect: the pinned AWS CLI v1
rejects v2's `--no-cli-pager` argument. Revision
`eb1794d7760170e5836e4907f80ff368eda6fd17` uses `AWS_PAGER` in the child environment
instead, with a regression assertion and another complete 442-test passing
shipping run before deployment. A Pi service using the encrypted credential
successfully identified itself as `nova-dashboard-pi` and fetched 422 live AWS
cost rows. The normal money unit's profile and service credential locator were
also verified. The account's historical accrued bill remains present after
Ohio destruction; deleting resources does not erase charges already incurred.
Verification receipts are retained on the Pi at
`data/ohio_destroyed_20261023.json` and `data/dashboard_aws_access_20261023.json`.


## Previously unevaluated candidate budget and money history (2026-10-23)

The 4,800 epoch budget means first-ever exact evaluations of previously unevaluated candidates.
Cached history remains ranking evidence and contributes zero budget credit.
Epochs 25889/25890/25891 actually screened 799/623/581 new candidates while
all-history coverage claimed 4,785/4,788/4,787. Never describe that coverage as
new work. Keep up to 8,000 proposals through cheap eligibility before admitting
4,800; early cheap rejection consumes no GPU budget.

New epoch manifests use `unseen-candidates-v2`. A locked, durable per-epoch
ledger reserves dispatches before launch; restart, cached results, and repeat
confirmation cannot reset or inflate this budget. Report successful screens,
reserved dispatches, incomplete dispatches, and budget shortfall separately.
An interrupted expensive dispatch remains charged conservatively, preventing
silent overspend. Deadline/handoff reserves still apply: an unfinished budget
must be reported, never allowed to block the next epoch. Existing immutable
old-policy epochs remain intact; archive only unused future preparation during
cutover. CI exercises historical-cache exclusion, restart/cap behavior, and
current-epoch coverage independently of ranking history.

The screenshot's Money history is our Vercel dashboard. The linked Metanova
page does not contain that chart. Reward collection on the Pi lacked Taostats
access and retained data only through October 22; absence became zero income.
The bars also stacked infrastructure beneath net after it was already deducted.
Today's Vast number is a full-day estimate, not spend accrued so far. The fix
shows reward receipts, registration and infrastructure as components, with net
as a separate line; current/incompletely covered days have net pending. Winning
an epoch is not a posted reward transfer or proof of positive net cash flow.

Reward refresh success now has a persistent coverage watermark. API errors
must fail the money job visibly rather than masquerade as an empty successful
refresh. Pi Taostats access uses an encrypted systemd credential scoped to the
money service; the HTTP client consumes it in memory, without CLI arguments or
logs. Portfolio history uses the latest observation of each position per day;
missing observations remain unknown rather than creating zero-value crashes.
CI covers stale income, missing portfolio observations, duplicate snapshots,
component accounting, and failed reward queries.


The user clarified that a fresh stochastic draw of an old candidate is NOT new
work for this budget. The initial `fresh-evaluations-v1` interpretation was
withdrawn before epoch 25892 began screening. The producer was stopped at
25891 offset 358 while its completed result and independent publisher remained
available. Previously evaluated sequences now feed the generator's exclusion
file and the fallback's seen set. Admission rechecks local exact, public
validator and rival replay history before selecting 4,800; dispatch checks
again. The history read fails closed instead of treating an outage as an empty
cache. Parent measurements remain usable for ranking. Confirmation and any
future experiment on first-pass noise are separate from this discovery budget.

Epochs 25889–25891 discarded 748/799, 574/623 and 528/581 candidates before
confirmation. Only 9, 7 and 8 of those discarded candidates have multiple raw
local rows, so existing data cannot provide an unbiased missed-winner rate.
A same-field comparison of retained finalists' initial versus subsequent
observations found roughly 39%, 24% and 40% worse repeat rank sums. This shows
selection optimism, not the fraction of winners lost; do not conflate them.


## Separate discovery from comparison (2026-10-23)

Excluding 235,814 previously evaluated candidates from new GPU work must not
exclude those candidates from comparison or publication. The publisher's
producer-derived candidate scope accidentally coupled these two decisions.
With a sealed observation database, publication now compares the stored bank
and newly confirmed candidates together against the current ten-epoch field.
It does not dispatch GPU evaluations of stored candidates. Explicit score-bank
runs retain their existing scope. The twenty-observation publication floor,
current novelty checks, Layer-2 gates and strict qualification remain required.

Publisher state schema 3 invalidates the old scope's no-op cache. The aggregate
ranking path applies the requested observation floor before loading raw fold
vectors, retaining all aggregate statistics for unchanged shrinkage. This avoids
loading shallow histories that cannot qualify. The existing command timeout,
epoch cancellation and bounded strict finalist pool remain in force.

A read-only live comparison for epoch 25894 ranked 9,542 candidates with enough
stored observations and produced a top-100 comparison without GPU work.
This is a comparison probe, not a claim that its leaders pass final novelty or
submission gates. CI regressions verify that a stored candidate absent from
current exact inputs can beat a new candidate in the same-field ranking,
without another fold; discovery still rejects previously evaluated candidates.


## Pre-check the bank before the merit cutoff (2026-10-23)

The first comparison repair ranked the full bank before novelty checking its
best 1,000 entries. In epoch 25894, all 1,011 shortlisted entries (including the
diversity reserve) failed novelty. Invalid high-ranking cached entries therefore
crowded out valid candidates. Ranking success was not publication success.

The publisher now enumerates all sealed own candidates with enough evidence,
removes exact historical/retired sequences, and applies current novelty plus
full Layer-2 checks BEFORE either provisional or strict shortlist selection.
Stored and newly confirmed candidates take the same path, reusing GPU evidence.
These checks do not consume the 4,800 never-before-evaluated discovery budget.
Incremental eligibility proof is bound to epoch, target, archive, config and gate
implementation; newly confirmed arrivals are checked without repeating the full
bank. Failed checks cannot write success, and empty results cannot reuse a stale
scope. Publisher schema 4 invalidates old no-op state. Changed archive/config
also invalidate qualification fingerprints, and an older broader artifact may
only be retained when its sequences still belong to the current checked scope.

Epoch 25893 had 12 live Pi slots but only six sequences at its commit deadline.
The Pi refreshed until 17:13 UTC and committed all six successfully; all six
were scored. The master's 11-entry artifact used gates completed at 17:18 UTC,
after the Pi committed. Master history is therefore not proof of what the Pi
actually submitted: compare the Pi archive, commit results and public scores.

Shipping CI covers identical pre-checks for new/stored candidates, current-rule
cache invalidation, incremental arrivals, interrupted/empty gates, stale aggregate
fallback, sealed-own evidence, and a 1,001-invalid-entry shortlist regression.
Release verification must include a current artifact reaching the Pi, not merely
a successful ranking command or passing unit tests.


## Spawned scorer shutdown must release analysis children (2026-10-24)

At epoch 25899 the fallback scorer's eight GPU workers were waiting on live
analysis children while all eight GPUs were idle. The persistent analysis-pool
optimization withheld the shutdown sentinel until an ordinary atexit hook.
Python's multiprocessing worker bootstrap joins non-daemon children before
ordinary atexit hooks, so successful scoring or an early error could hang the
fallback until the epoch deadline. A CPU-only real-spawn reproduction failed
for both completion and exception paths before this repair.

Register pools when constructed, and use a positive-priority multiprocessing
finalizer before child joins and queue teardown. Send the sentinel, then bound
cleanup and terminate an unresponsive analysis child. Keep reuse within a live
worker. The real-spawn regression must wait for its analysis child to start;
otherwise a semaphore startup failure can masquerade as successful cleanup.
Both cases are now part of the mandatory epoch-deadline suite.


The first live pre-check found 617 novel entries, of which 580 lacked cached
TNP measurements. Waiting for this cold backlog before publishing again
recreated a deadline bottleneck. Current-gate checks now validate available
measurements first and checkpoint cold entries in batches of 32 across cycles.
No unchecked entry enters ranking, and the batch size never truncates the bank:
pending checks remain explicit and every cycle advances them. Novelty proof
survives an interrupted Layer-2 batch; eligibility does not. Scope changes
invalidate publication no-op state even without new GPU observations.

The first artifact-retention guard incorrectly treated `_live_sequences()` as
a set; its actual return type is a list. This raised TypeError after successful
pre-checks. The corrected guard converts the result explicitly. Mandatory CI
now exercises the complete database-backed publication orchestrator, using its
real artifact, scope, retry and retention helpers with simulated external gates.
It proves that both ordinary startup and interruption after pre-checks deliver
the checked fleet before taking another cold batch. Helper-only tests did not
provide enough integration coverage for this incident.

Release `128b79a846f19f3e36ef948e16ddd64fc1724f3d` passed all 473 mandatory
shipping checks locally and in GitHub Actions before the publisher deployment.
The scorer cleanup also passed both real-spawn shutdown cases in the configured
master and slave runtimes. Epoch 25900 produced a current 12-entry artifact;
the Pi independently pulled it and planned 12 entries for 12 held slots, with
no missing UID. This verifies preparation, not on-chain acceptance.

A separate live scorer check completed 32/32 previously measured candidates
in 107 seconds through the warm daemon, retaining all 32 structures under the
pinned validator-dual-sample-v1 protocol. These diagnostic evaluations were not
ingested into rankings and are not new discovery work or part of the 4,800
budget. Verify fresh production progress separately. The original daemon worker
failure trigger remains unproven; the independently reproduced fallback-exit
deadlock is repaired. Do not equate a successful diagnostic with a completed
discovery budget or promise a particular future multipass score.

The first fresh production tranche in epoch 25900 exposed a separate contract
failure that the homogeneous diagnostic did not cover: its 700 entries included
487 of one length and 213 of another. The fast adapter explicitly refuses a
mixed-length call before GPU dispatch. Reservation therefore advanced while
completed fresh evaluations stayed at zero. Recovery must reuse those reserved
inputs rather than charge another batch or mistake reservation for completion.

The wave runner now partitions admitted inputs by length, preserving first-seen
length order, order within each length, the admission limit, and all retained
structures. Homogeneous waves retain their existing boundaries. Mandatory CI
now includes the actual wave runner and its validation suites, plus mixed-length
execution and interruption/resume regressions that invoke the adapter's real
shape check. End-to-end verification must exercise production input shapes;
a small homogeneous diagnostic is insufficient evidence of production recovery.

Splitting lengths alone was insufficient: all four installed reference designs
were length 127, while 4,313 of the 4,800 admitted entries were length 125.
Synthesis raised SystemExit for a same-length reference mismatch, killing all
24 resident workers. Memory-event counters showed no OOM or OOM kill. A real
125-residue reference was generated with the pinned standard pipeline using an
already published candidate; that diagnostic completed with the required
structure and scoring protocol. Reference selection is now explicit by target
and length. Unregistered lengths use the pinned standard design stage, with
unchanged folding settings and a visible fallback event, instead of crashing
workers. Reference registration must validate chain length and target identity.

The daemon now reports scorer SystemExit as a failed request and remains
available. Reply reads check process liveness, partial-reply deadlines and
client disconnection instead of relying on blocking readline. Each worker owns
an isolated process group; failure/cancellation closes that group so abandoned
work releases GPU locks when the epoch runner terminates its client. Real-process
CI covers a dead parent whose child holds the pipe open, a partial reply,
disconnection and SystemExit. The previously deployed three-workers-per-GPU,
parallel startup and short-shard retry changes are retained in source control.

The new length-125 reference reuses all 1,282 target-chain atom coordinates
from the existing validated length-127 reference, verified exactly by atom
identity; only the candidate-chain length changes. Keep its separate reference
library so registering a new length does not alter the old length's library.

Master and slave pinned scorers have different worker argument formats. The
slave also supports replica ordinals and watches its owning daemon. A deployment
must preserve those capabilities instead of overwriting the slave with the
master's older 16-argument worker contract. The shared daemon now selects the
registered base or replica format explicitly, preserves global replica ordinals
and the slave's existing shard count, and retains owner monitoring. Owner
monitoring checks the daemon's lifetime rather than requiring it to be the
worker's direct parent: the configured Python launcher may insert a process.
CI exercises both argument formats, replica-flag forwarding and indirect-parent
ownership using real processes.

Enable daemon use in the box's scorer settings as well as supporting the
existing launcher environment. Otherwise an incident replay launched directly
can silently take the colder subprocess route even when the daemon is healthy.
Both routes must remain correct; their throughput and verification evidence
must be distinguished.

The immutable epoch-cycle path had also deferred the next pool's CPU preparation
until confirmation was completely finished. Its preparation took much longer
than the remaining handoff window, so the next epoch started by waiting for
generation. Start that CPU-only preparation in a daemon thread when confirmation
begins instead. Keep the existing low subprocess priority and the CURRENT
epoch's cancellation deadline; refresh admission gates in the real next epoch.
Never join speculative preparation from the scoring loop or start a duplicate
preparation after confirmation. A failed preparation remains retryable and
cannot abort the current race. CI covers overlap, retry, duplicate prevention,
confirmation-only dispatch, priority and current-epoch deadline ownership.

Production recovery in epoch 25900 completed both previously reserved tranches:
700/700 mixed-length entries in 559 seconds and 700/700 length-125 entries in
488 seconds, retaining all 1,400 structures under validator-dual-sample-v1.
These were fresh discovery inputs, unlike the earlier diagnostic. The first
tranche exercised the cold subprocess route; the second used the configured
daemon. Neither reservation was charged twice. The standard coverage report
now records attempted=1,400, screened=1,400, uncompleted_dispatches=0 and
budget_shortfall=3,400. This is recovery evidence, not completion of the full
4,800 budget or evidence of a particular multipass score.

Runtime release a0b8c0fbbddd141f65562b55dbcf917d1fc9a976 passed 514 mandatory
checks locally and in GitHub Actions and was installed on both scorer hosts.
The dead-worker, disconnected-client and replica-protocol checks also passed
inside both configured production Python runtimes. Producer release
90b66ced5c5df2a7c2b8220b60d5a6c3ff610400 passed 518 mandatory checks locally
and in GitHub Actions before installation; its deployed file hash matches the
tested staged package. The Pi's independent submission service has a current
12-entry artifact for its 12 held slots, with no unassigned UID. Preparation
remains distinct from on-chain commitment and validator scoring.

After the reserved-tranche recovery, the supervised producer resumed iteration
3 of epoch 25900 in confirmation mode and emitted
`PRODUCER_NEXT_POOL_STARTED epoch=25901`; that preparation directory appeared
while current confirmation was still active. The producer, scorer daemon,
publisher and watchdog sessions were present. Pi submission stayed active;
its refresh, publication and multipass systemd jobs completed with status 0.
These checks verify the autonomous handoff has resumed, without relying on a
Mac process or waiting for confirmation before beginning next-epoch CPU work.

The follow-up check in epoch 25901 verified 4,182 completed fresh evaluations
and a current 12-entry master artifact. Pi reported 12 successful commitments
and zero failures for epoch 25900. It also exposed a reporting defect: the
top-level coverage file still showed zero because its parent process only
refreshed it before and after the entire race. Refresh that same authoritative
coverage calculation after each reservation and each completed tranche. Keep
reserved and completed counts distinct, validate epoch/target identity, and
report monitoring-write failures without aborting healthy scoring. Mandatory CI
exercises the live race through reservation and completion before it returns,
including partial completion and invalid-scope/write-failure cases.

The real `producer_relaunch.sh` must invoke the verified restart gate. Installing
a gate file alone does not enforce it. The control installer preserves the
original command body and arguments, adds one pinned gate call, and records
both launcher hashes without restarting work.
