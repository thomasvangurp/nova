# Release promotion contract

CI passing permits **staging**, not promotion. A change to selection policy is
an experiment; an operational repair is a separate release. This contract is
enforced by `scripts/deploy_tested_release.py`, the deployment workflow, the
remote transaction, and the normal restart gate. It does not guarantee a score
or replace independent validator results.

## What is implemented, and what remains operational evidence

The release contains the evidence checker, isolated trial runner, read-only Pi
consumer receipt tool, state snapshot/quarantine tools, guarded source promotion,
rollback, workflows, and mandatory failure tests. The tests use synthetic IDs
and injected failures. They are **not** production replay evidence.

A real qualification requires approved recorded production workloads and a
workload driver that executes each release's production pipeline. There is no
approved production driver or production fixture bundle supplied by this
change. Until those have been collected and run, promotion deliberately fails
closed. Do not substitute the test fixture, fabricate receipts, or use the
new evidence schema as a reason to claim live score improvement. The current
production release is not changed by committing these deployment controls.

The evidence export workflow requires a dedicated runner with labels
`self-hosted`, `linux`, `sn68-release-qualification`. Its setup is an operational
prerequisite, not something this code assumes is already installed. Never run
it on the live producer or the Pi, and never give it submission credentials.

## Staging and qualification

1. Commit the release. Run `scripts/shipping_checks.py --output <new-directory>`
   with the shipping test environment. CI creates the exact tested package.
2. Deploy with `activate=false`. Only `/root/releases/<sha>` is populated and
   verified. This does not provision dependencies, copy to `/root`, change
   launch arguments, or restart processes.
3. Run `scripts/shadow_release.py` with `--baseline`, `--candidate`, `--fixtures`,
   `--output`, `--driver`, and `--image`. Release paths are unpacked, tested
   packages. Fixtures are a separate credential-free recorded dataset. The
   driver is a tracked Python file in the candidate release; the same driver
   runs both releases and must import the selected `/release` code.
   The image must be locally available and pinned by `@sha256:` digest.
4. Containers run without network, capabilities, host credentials, wallets,
   live submission access, a writable root, or the Docker socket. Only their
   output scratch directory is writable. The rollback trial runs the baseline
   against the candidate's resulting durable state. Timeout removes the entire
   container. Changed inputs or missing outputs invalidate the run.
5. Transfer the separate candidate trial artifact to a scratch path on the Pi.
   Run `scripts/pi_shadow_receipt.py --artifact <trial> --candidate-sha <sha>
   --target <target> --epoch <activation-epoch> --uids <current fleet UIDs>
   --output <new-receipt>`. Use the Pi's existing authenticated transport; do
   not copy credentials. This calls its installed `sequences_by_uid` consumer
   without invoking main, pulling live input, reading wallets, or committing.
   This proves consumer compatibility and allocation, not on-chain acceptance.
6. Assemble the evidence contract below in the dedicated runner's
   `/var/lib/nova/qualification/<sha>`. Run the **Qualify nanobody release
   evidence** workflow for that same commit. It first requires CI and exports
   only the checked documents as `sn68-promotion-evidence`.
7. Activate using **Deploy tested nanobody release**, that qualification run ID,
   exact host and target. It checks the successful run's repository, workflow
   identity, and commit; both controller and host verify evidence hashes and
   the exact tested manifest. Host checks also require the unchanged live
   baseline, current target, fresh chain state, correct activation epoch, an
   idle boundary within its first 12 blocks, and no in-flight owner children.
   A missed boundary refuses promotion; it does not terminate epoch work.

Documents expire within 24 hours; the actual Pi receipt must be at most 15
minutes old at activation. Refresh expired evidence through real verification.
Workflow success, a unit-test fixture, and a manually edited timestamp are not
substitutes for that verification.

## Evidence contract

All JSON uses unique keys, finite numbers, relative safe paths, and SHA-256
references. `promotion.json` has schema `sn68-promotion-v1`, full 40-character
`candidate_sha` and `baseline_sha`, `change_kind` (`repair`, `experiment`, or
`rollback`), `host`, `target`, `activation_epoch`, `created_at`, `expires_at`,
`baseline_manifest_sha256`, and `{path, sha256}` references named `fixture`,
`baseline_report`, `candidate_report`, `shadow_receipt`, `rollback_report`,
`rollback_receipt`, `state_snapshot`, and `pi_receipt`.

The fixture schema is `sn68-production-trial-v1`, with `target` and `cases`.
Each case contains `name`, `scenario`, `epoch`, `proposal_ids`,
`previously_evaluated_ids`, `currently_eligible_ids`, `new_budget`, `fleet_size`,
`field_sha256`, `held_out`, `epoch_length`, and scoring/publication/Pi/commit
cutoff offsets. Candidate and pass IDs are opaque SHA-256 identifiers. Each
held-out entry maps an ID to independent `pass_ids` and numeric `scores` from
the fixed field. These observations must never be supplied to selection.

Required scenarios are `normal`, `cold_cache`, `mixed_lengths`,
`interrupted_resume`, `partial_wave`, `epoch_rollover`, `unused_publication`,
`late_confirmation`, `gate_attrition`, `stale_proof`, and `pi_pool_spare`.
At least three different normal epochs are required. Every case starts with
at least 8,000 proposals and the 4,800 NEW evaluation cap. Normal cases must
complete all 4,800 to prove full-budget behavior; interrupted cases may finish
fewer, but the candidate may not reduce baseline completed NEW coverage.
Previously evaluated IDs cannot consume that budget.

Each report supplies `revision`, `fixture_sha256`, `handoff_artifact_sha256`,
and identically ordered `cases`. Each result contains `name`, `epoch`, `status`,
`confirmation_health`, unique `dispatched_new_ids` and `completed_new_ids`,
`selected` entries (`id`, independent `selection_pass_ids`), `handoff`
(`artifact_sha256`, `pi_artifact_sha256`, `confirmed_offset`, `published_offset`,
`pi_pull_offset`), `old_epoch_work_running`, and `unavailable_gates_admitted`.
The driver receives `--release-root`, `--fixtures`, `--output`, `--state-output`,
and for rollback `--resume-state`. It must actually exercise the incident
conditions; merely printing a report that fits the schema is not a replay.

The checker independently computes the selected fleet mean from held-out
scores. It requires at least 13 eligible, distinct selections with at least ten
selection observations and ten disjoint held-out observations each. Per-case
quality and discovery coverage cannot regress. Repairs can recover an incident
the baseline fails; the normal baseline cases must already work. Experiments
must also improve the measured aggregate. Cutoffs cannot exceed the existing
shipping submission timing contract. No late handoff, unavailable gate, false
healthy completion, or work surviving an epoch is accepted.

`shadow_release.py` writes the isolation receipt and binds all three run reports
to their release manifests. The rollback receipt names both revisions, uses
policy `preserve-durable-invalidate-derived-v1`, has `restores_durable_state:
false`, and binds `snapshot_manifest_sha256` and `post_candidate_state_sha256`.
The latter is `shadow_release.fingerprint(candidate/state)`. The rollback
report must also contain matching nonempty `protected_state_before` and
`protected_state_after` digest inventories. The snapshot manifest is produced
by `release_state.py capture`; it is evidence, not a restore source.

Receipts trust the dedicated runner/operator's recorded observations. Hashes
detect mismatched and changed files; they do not prove an arbitrary author ran
the claimed workload. Keep driver, fixture provenance, and runner access under
review. The workflow never manufactures missing evidence.

## Promotion, deadlines, and rollback

The transaction handles source-only releases with unchanged dependencies and
vendored scorer files. Dependency/model-image changes require a separately
qualified image rollout and are refused here. Existing-host legacy Ansible
provisioning is refused before its first mutating task. Fresh-host provisioning
must be planned separately; do not use the legacy services playbook as an
upgrade path over a stamped release.

Before pausing owners, take forensic backups using SQLite's backup API so
committed WAL data is included. Reverify the baseline and boundary afterwards.
Pause only idle owners, install verified files with a bounded copy interval,
and relaunch the existing producer and publisher. Readiness waits within the
actual publication window; it requires new PIDs and a fresh same-epoch fleet
with at least ten observations per entry. It does not infer validator rewards.
The regular restart gate requires an active promotion receipt matching the
installed manifest.

On a failed activation, restore verified baseline **code and launch arguments**,
quarantine only the explicit derived-cache allowlist, and relaunch baseline
code. Never restore old submission history, the raw measurements database,
publication events, Pi outcomes, committed UIDs, or NEW budget reservations.
Measurements and reservations written by the failed candidate survive. Every
attempt has a separate audit directory; a refusal can be retried with fresh
evidence. A rollback receipt is written only after restoring and verifying the
baseline and successfully issuing its relaunch. It is not proof of later
validator performance.

Mandatory tests include missing/expired/mismatched proof, held-out leakage,
quality and coverage regression, late handoff, one-pass evidence, unsafe shadow
settings, live baseline drift, legacy bypass, Pi consumer allocation, timeout
container cleanup, SQLite WAL backup, and a failed-start rollback preserving
new measurements and reservations. Full shipping CI remains mandatory.

SSH disconnect and normal cancellation enter the rollback/cleanup path;
repeated termination signals cannot interrupt restoration. Derived caches for
both the current epoch and a prepared next epoch are quarantined, while both
epochs' budget reservations survive. Host loss or `SIGKILL` cannot be handled
by a process-local exception handler: inspect the per-attempt backup and tested
manifest before restarting after such a failure. Never restore the state
snapshot over newer durable data.
