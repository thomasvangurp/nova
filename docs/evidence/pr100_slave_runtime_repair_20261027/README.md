# PR100 slave runtime evidence for 27 October 2026

The [incident record](../../PR100_SLAVE_RUNTIME_REPAIR_20261027.md) explains
the repair, qualification and submission audit. This bundle retains public
scientific file manifests and sanitized measurements; it contains no model
weights, library binaries, input sequences, raw stdout/stderr, credentials, or
full environment snapshots.

## Evidence

- `qualification-approved.json` is the complete safe E guard receipt plus the
  root agent's runtime-difference approval. `qualification-receipt.json` retains
  the unaugmented receipt. `qualification-summary.json` is a convenience summary.
- `measurements.json` contains 32 hash-identified numeric observations and
  provenance checks. All 32 are scientifically rejected but runtime-complete.
- `manifests/` retains the exact original scientific manifest, metadata
  supplement, MKL supplement, original unpack receipt and reference allowlist.
  `manifest-copy-provenance.json` binds the exact retained bytes.
- `mkl-probe-before.json` and the two timestamped probe records show the
  allowlisted missing-library diagnoses and successful CPU-math/import check.
  Only stream lengths and categories are retained, never raw streams.
- `mkl-family-copy.json` records fifteen verified libraries and their hashes.
  The core-copy receipt retains the earlier completed repair step.
- `submission-yield-audit.json` retains per-attempt counts, gate outcomes,
  scheduling times, final pool comparisons, crew counts and parsed commit
  receipts for epochs 25965 and 25966.
- `activation-root-observation.json` records the parent's dated production
  readback. It is labelled as a root-agent report, not a directly collected
  receipt or final epoch result.
- `bundle-manifest.json` binds every other file in this directory. The offline
  `verify_evidence.py` checks those hashes, 32 numeric/provenance records,
  runtime identity, source/weight equality, approval and cleanup.

## Exact historical helpers

`helper-provenance.json` records byte equality and SHA-256 for all thirteen
retained helpers. The E worker, guard and Pi service are the exact files used
for the successful qualification. The core-copy and family-copy helpers,
runtime probe and monitor preserve the repair path. `commit-readback.py` is a
read-only Pi helper that emits allowlisted commit-event fields and publication
pool cardinality for a requested epoch.

`helpers/original-provisioning/` preserves the earlier manifest builder,
packer, unpacker and Pi/master orchestration helpers. Those historical files
precede the E thread-pinned wrapper and metadata/MKL supplements; they are not
a complete description of the accepted final runtime by themselves.

Archived helpers contain incident-specific paths and executable top-level
operations. They are evidence, not installed service entrypoints or a replay
request. Reusing them requires a fresh run directory and review of the actual
runtime/host identities; an earlier receipt must not be overwritten or counted
as a new measurement. The repair family helper also expects the separately
verified core step and retains that dependency explicitly.

The only offline verification command is:

```sh
python3 docs/evidence/pr100_slave_runtime_repair_20261027/verify_evidence.py
```

This reads this bundle only and performs no SSH, profiling, publication or
production mutation. Host addresses and SSH-key paths inside the archived
helpers are locators, not authentication material.
