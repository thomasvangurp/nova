# Parent merit, epoch deadlines, and release evidence — 2026-10-22

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
