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
