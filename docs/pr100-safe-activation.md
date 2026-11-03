# PR100 activation helper and retained incident evidence

The reusable helper is `scripts/pr100_safe_activate.py`. It performs a Pi-owned
release handoff only at a verified clean continuation boundary. This change
retains the already-used v2 helper byte-for-byte and adds a separately reviewed
successor; it does not deploy that successor or change scientific screening.

## Retained incident

The exact used v2 helper and its original 36 tests are archived under
[`docs/evidence/pr100_activation_20261027`](evidence/pr100_activation_20261027).
The retention manifest binds their SHA256 hashes. The executable regression
suite uses a repository-relative import and runs from the exact staged release
artifact in shipping CI.

The retained activation receipt confirms release
`22407a4ae6324b8c051ef89f738eb74fddaad978` activated in epoch 25967 with both hosts,
preserving continuation state and the publication artifact. The first native
batch snapshot records disjoint inputs: 440 on master and 147 on slave, with zero
overlap. At the recorded observation, 96+32 fresh evaluations had completed,
7+2 candidates passed, and observation errors were zero. These are batch
observations, not a claim about the final published fleet or epoch score.

## Required evidence and boundary

The helper requires a full passing shipping-CI stamp, exact release-file hashes,
and matching staging receipts from Pi, master and slave. Enabling or preserving
both native hosts additionally requires the approved slave runtime layout,
scientific/runtime identity or explicitly identity-bound approval of differences,
at least 32 workers/completed profiles, exit 0, zero inconclusive profiles and
verified fixed-point cleanup with zero descendants. The alternate wrapper hash
is pinned; annotation paths remain unchanged.

Runtime qualification is not validator-runtime attestation or bit-identity.
The historical guard receipt does not contain a digest of its companion numeric
measurement file; this successor does not invent one or rewrite the historical
receipt. It validates the explicit guard completion and cleanup fields.

The continuation must be idle (`active=None`), without legacy uncertain phase
ownership, and have a current watching heartbeat. A quiet publication-cutoff gap
is a valid deployment window: the controller still enforces work admission and
submission deadlines. There is deliberately no second slow chain RPC between
the last readiness check and service stop. After stopping, the helper obtains
the controller lock and checks a genuinely fresh chain sample, epoch, config
bytes and continuation ownership before installing anything.

## Successor failure behavior

The review found a post-start rollback hazard in the archived v2 helper: if
verification failed after the new service had started, restoring the old config
and calling `systemctl start` could leave the already-running new process using
old configuration. The new controller might also already own remote scientific
work. Holding the lock through startup is not a solution because this controller
acquires that lock nonblockingly and would exit.

The successor distinguishes these cases:

| Point of failure | Action |
| --- | --- |
| Before stop, including a new attempt beginning during preparation | Wait; leave the running service and files unchanged. |
| After stop, before install, when the boundary/epoch changed | Restart the old service with its existing state and config. |
| Install fails before any new start is attempted | Restore the old unit and config while stopped, then restart the old service. |
| Restoration itself fails | Keep the service stopped and report `rollback_failed_service_stopped`; do not start an uncertain unit/config pair. |
| New start has been attempted and startup/verification fails | Preserve the newly installed unit/config and current service state; report `activation_unverified_new_release_preserved`. Do not stop newly owned work or restore old config beneath it. |

Both recovery-required statuses terminate the helper with a nonzero exit. They
are not retried into a false success. Before any manual recovery, inspect local
continuation state and the corresponding remote supervisor/cleanup ownership;
use the same idempotent recovery paths when ownership is uncertain. The helper
does not claim that a preserved service is healthy.

A repeated invocation with matching config verifies the running release and
installed unit before reporting `already_configured`; matching config alone no
longer counts as a successful deployment. Code-only rollouts preserve both hosts
and reject silently disabling the qualified slave. Release 224 is an explicitly
approved predecessor; arbitrary unknown predecessors remain refused.

## Validation and use

`tests/test_pr100_safe_activate.py` covers the original boundary cases, the short
handoff window, qualification cleanup, lock/config races, source/manifest guards,
post-start failures after native ownership begins, failed restoration and
terminal recovery reporting. The suite is mandatory in `scripts/shipping_checks.py`.
The helper is part of the release-controls inventory. Full shipping CI is
required before staging or using the successor.

The CLI runs on Pi with a tested/staged release:

```text
python3 scripts/pr100_safe_activate.py SHA preserve \
  --staging-receipt PATH --qualification-receipt PATH \
  --novelty-qualification-receipt PATH
```

The novelty rollout requires the completed qualification receipt for both hosts.
The gate verifies canonical hashes of the embedded artifacts, runtime identities,
the eight scientific source files against the staged release, complete annotation
coverage of both historical archives, all 22 replay verdicts, and zero-descendant
fixed-point cleanup. Whole application commit identity is not substituted for
scientific source identity. This rollout changes only release/start-epoch fields
and each host's qualified `novelty_runtime`; it preserves native topology, native
runtimes, worker settings, GPUs, continuation state and the publication artifact.

For this qualified novelty migration, a watching controller may also hand off a
pending attempt when its persisted native dispatch matches the current source,
challenge and hosts, and every recorded launch request has collected artifacts
and verified zero-descendant cleanup. A pre-dispatch failure can leave a
conservative `native_unresolved` flag despite no native launch intent; requiring
`active=None` in that case would prevent the new recovery code from ever running.
The helper preserves the pending state unchanged and repeats the ownership proof
under the stopped-controller lock. Missing cleanup, running work, uncollected
requests, changed bindings, or uncertain legacy ownership still block handoff.

`--interleave-confirmations` explicitly enables the independently reviewed
confirmation-first mixed native waves. The flag defaults off and otherwise
preserves the current worker setting. Enabling requires the deadline regression
suite in both the passing shipping-CI test list and the hash-verified staged
manifest. It changes only `worker.native_interleave_confirmations` and refuses
the experimental accelerated-screen combination.

A source-only generation successor may preserve pending confirmations before
the cutoff when every selected native host already uses the same qualified
resident CUDA registration. This exception requires identical configuration
except application source, start epoch and the corresponding host release
suffixes. GPU allocation, CPU slots, workers, batching, sealed runtime paths,
fleet policy, novelty runtime and all other settings remain unchanged.

The helper compares actual scientific source bytes in the predecessor and
successor staged releases, including requirements, native evidence validation,
the registered implementation and observer, annotation and novelty. It also
compares the worker's complete Python syntax tree except `sequence_phase`, so a
graft proposal change cannot silently alter native resume or acceptance code.
Both copies must match the selected registration's implementation, observer,
resident module and qualification report hashes. Each pending checkpoint must
retain its execution host, challenge, requirements, registration, runtime and
fleet-policy bindings. Missing old source files, an unregistered backend,
different registrations between hosts, or an accelerated-screen checkpoint
cannot use this exception.

Compatibility never establishes ownership. The controller must still be at a
clean between-attempt boundary, with no active owner or legacy uncertainty and
a fresh watching heartbeat. Compatibility and checkpoint bindings are checked
again before stopping and under the stopped-controller lock. Continuation and
publication files remain untouched. CPU-to-GPU, registration replacements and
configuration transitions retain the existing drain-or-cutoff behavior, even
when the application SHA is unchanged. No CLI override is provided.

Subprocess stdout/stderr is captured. Reports contain only operational status,
release identifiers, counts and error classes. No credential values or raw
authentication output are retained.

## Recovery lessons from epochs 26014–26018

The October 29 audit found independent failures: release changes invalidated
active dispatch identities; a legacy resident scorer pool left production GPUs
below their admission threshold; the unseeded generator repeated only a handful
of sequences; novelty removed complete native passers; and the slave accelerated
screen rejected its installed CUDA wheel version. Aggregation failures alone did
not explain the incident.

`SN68_FORCE_ACTIVATION` no longer bypasses ownership, epoch, configuration-race
or predecessor checks. Preserve frozen dispatch provenance. Never rewrite an old
attempt's SHA or remove its evidence to force a release change. An emergency
recovery must collect or stop its identified owners and verify cleanup first.

Reserve production GPUs explicitly in `pr100-reserved-gpus.json`, using schema
`pr100-gpu-reservation-v1`, owner `pr100-live-controller`, and the allocated
integer GPU IDs. Drain the identified old daemon before restart. The legacy
daemon and adapter fallback refuse these devices; production adapter access
requires an inherited kernel lease. A larger free-memory claim or a bare
`--no-daemon` flag is not ownership. Background replay must use its own allocation.

`--recovery-rollout-receipt PATH` applies measured per-host screen settings and
an annotated, hash-pinned parent bank inside the same atomic handoff. Screen
qualification binds the exact source, host, settings and protocol and requires
completed profiles and verified cleanup. A zero-exit process with no profiles
does not qualify. The optional parent-edit lane additionally requires measured
novelty survivors. These checks do not establish native scientific parity or
competitive superiority; those need separately measured evidence.

Shipping CI now tests the packaged Pi layout from the same release bytes and
rejects changed test suites omitted from its blocking list. After deployment,
compare actual installed hashes. Historical scientific fixtures retain their
original targets; changing a label cannot create current-target evidence.

Monitor the full funnel: unique new proposals, annotation, native admissions,
developability, novelty, GPU admission, scored candidates, publications and
validator acceptance. Fresh evaluations and repeat measurements are separate
counters. Compare candidate quality against rivals on the same target, field
and scoring protocol. Diversity and throughput benchmarks alone do not prove
competitive scoring.
