# PR100 cutover: observed failures and release requirements

Recorded 26 October 2026. This is incident evidence, not production-strategy
approval. The activation epoch was 25955. Keep deployment receipts separate
from the research branch: a committed change is not necessarily installed.

## What prevented an epoch-25955 submission

The Pi's held plan recorded zero sequences, zero molecules and zero commit
attempts. The handled-epoch ledger subsequently included 25955. A logged
no-op is not an on-chain submission or reveal.

The nanobody hold deliberately prevented the legacy pipeline from publishing
without PR100 qualification. Compound preparation failed independently: the
live bank at `/opt/nova-submit/bank.sqlite` contained zero reaction-32 candidate
rows and zero corresponding rescored rows. Reducing the observation floor
could not manufacture missing inputs. An empty post-filter selection alone
does not identify the cause; check raw rows, context selection, joins, score
availability and individual gates separately. Do not blame the newly added
conformer check: it was not installed when this failure occurred.

## Deadline correction

The live Pi loop had three preparation call sites. A timeout around only the
immediate `make_fleet` child could leave its descendants running. The targeted
patch routes all three through the existing owned-process-group runner and
clamps work to the epoch collection cutoff, retaining a cleanup reserve.
Expired preparation returns a failure and produces a structured local alert.
It does not invalidate an independently valid earlier artifact.

CI passed 1,005 tests for source `a2dc0259`. A rehearsal under the actual Pi
interpreter and application directory exercised a parent and grandchild that
ignored SIGTERM: SIGKILL removed both, no late write appeared, and the next
job completed. This establishes the tested cleanup behavior; it does not
establish full-bank throughput or guarantee that every epoch has supply.

Installation checks bind the exact tested helper and source, preserve file
ownership/mode, use atomic replacement and retain a verified backup. A fresh
epoch/process guard is required before restarting the loop. Resume evidence
must come from new bytes in `loop.log`, because this service appends there;
checking only the system journal would incorrectly report a failed restart.

## Runtime experiment failure is separate from scientific failure

The first complete native import check passed on source `65b503f`, whose CI
passed 1,041 tests. It verified registered imports and libraries and left no
children. The subsequent frozen CPU-reuse comparison still stopped after two
observations because real execution loaded additional files, including a
Torch-generated module, absent from the manifest. Its 70 missing observations
remain missing. There is no parity, speedup or acceptance-yield conclusion.

An import-only check cannot establish the entire execution-time dependency
set. Any follow-up must retain the failed run, register the precise public
file or deterministic generated-source provenance, and use a separate run
identity. Never silently ignore temporary Python modules, expand to a whole
directory, or report a successful retry as the original experiment.

Review findings also require source verification. A review incorrectly
assumed the installed OpenMM XML loader attempted two file opens. The pinned
source resolved the absolute path first and opened once. The reviewer
withdrew that finding. Do not change correct code to satisfy an unsupported
review assertion.

## CI and operational evidence must remain distinct

- Regression tests establish control behavior, including deadline cancellation,
  output rejection, context binding and process cleanup. Mocked native results
  never count as measured validator acceptance.
- A deployed runtime check establishes the behavior of the exact interpreter,
  imports and installed files used in that check. Record remaining limits.
- A production readiness decision additionally needs actual challenge-bound
  eligible supply, fresh target observations and a timely Pi handoff.
- The 4,800 NEW budget counts previously unmeasured target-specific candidates.
  Cached candidates remain competitors; replay and retries have separate cost
  accounting and do not inflate that count.
- Missing inventory is an explicit failure state. Coverage labels alone are
  insufficient; report actual per-context candidate and score counts.

Faster native execution may reduce cost. It does not turn rejected candidates
into eligible ones, fill an empty bank, or prove a near-optimal strategy.

## Live learning is a separate release decision

The user subsequently authorized a bounded live attempt rather than waiting
for research qualification. Current-rule eligibility and the submission cutoff
remain requirements; a full fleet, ten or twenty observations, demonstrated
4,800-candidate capacity and a near-optimality study are not prerequisites for
submitting a valid partial fleet.

The installed Pi was missing the PR100 rules/config dependencies. The live
consumer patch installs those dependencies and replaces the blanket hold with
independent chain-context and per-candidate evidence checks. It retains the
independent molecule path if a nanobody import, artifact or validation fails.
The duplicate unconditional ten-observation check was removed: the checkpoint
validator owns the evidence floor.

The existing fast scorer uses 35 folding steps, while upstream specifies 200.
Its measured observations must not be labelled validator-equivalent. The
explicit schema-3 `live_trial` stage retains PR100 eligibility and finite actual
measurements but disclaims exact-scoring equivalence, binds the current context
and actual settings, and permits a single measured observation. Trial ranks
among a small local pool are not competitive epoch ranks or multipass estimates.

## Claude session recovery, 2026-10-26

The interrupted Claude session had committed and deployed controller revision `6a06d48`;
its background/idle status was not evidence of continuing implementation. Independent
readback matched all 1,078 manifest files on Pi, master, and slave. Both hosted workflows
were successful. The Pi controller was active but disabled at boot; recovery enabled it
without restarting the submission loop. Experimental throughput patches were not included
in this recovery release without completed validation.

Two orchestration gaps remained despite the earlier CI pass:

- Every SSH retry and backoff now shares the epoch's absolute receipt deadline.
  The previous five independent transport timeouts could outlast the epoch reserve.
- A Linux supervisor now owns a dedicated subreaper session and reaps all descendants,
  including children that start another session. SIGTERM, SIGINT, unexpected exceptions,
  and normal leader exit all use cleanup. A process-group-only test could not detect
  escaped GPU runners. Non-finite time bounds are rejected before spawning.

Required regression coverage includes deadline exhaustion during transport retries,
service termination, failure after process launch, detached descendants on Linux, and
real exporter/verifier contract checks. Future deployment receipts must distinguish
committed source, passing CI, installed hashes, enabled services, and actual submissions.
