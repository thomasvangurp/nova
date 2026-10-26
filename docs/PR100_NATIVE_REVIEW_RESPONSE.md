# Response to Opus native-reuse review 1

No native measurements or remote staging occurred. Original reviewed commit:
`8b3ef5c`. Follow-ups preserve the 24 identities/order and primary endpoint;
configuration revision 2 is explicit, before any measurement.

| Item | Disposition |
| --- | --- |
| B1 | Explicit completeness/parity/cost verdict; nonzero exit on NO_GO. Failed rows cannot masquerade as a successful experiment. |
| B2 | OMP/OpenBLAS/MKL/OpenMM thread settings are one, checked against manifest. Prior CLI OpenMM refinement was already explicitly one; OMP=2 did not make refinement two-threaded. |
| B3 | Factual rebuttal: corrected evidence records A3D status `ok` for all 4,080 census and 351 pilot profiles. The 6/65 cases were finite cap rejections. Complete A3D remains mandatory; no `ok_no_a3d`. |
| B4 | Status/type/numeric/flag/reason/subgate/raw and normalized structure mismatch aggregates. Exact typed numeric equality and matching gates/flags/reasons required. |
| B5 | Actual cgroup mount is read-only/nondelegated. Revised resource contract: 24 GiB sampled RSS abort threshold, explicitly not a hardcap; 96 GiB starting/64 GiB ongoing available-memory requirements. CPU/time/volume unchanged; no infrastructure install. |
| B6 | Any mode reaching 300 seconds aborts the entire trial. No reuse of timed-out warm state. Partial atomic observations and all missing cases remain visible. |
| B7 | Allowlisted stage and exception class only. Raw exception/backend text remains discarded under credential isolation. Fatal exceptions stop the worker and fail the trial. |
| B8 | Dedicated session leader/no existing descendants required. Registered direct-child statuses are reaped before adopted orphans; pre/post cleanup exitcodes recorded. |
| B9 | Bounded fixed-point stop scan with diagnostics; rehearsal adds active bounded fork loop. Atomic per-worker files replace shared queue transport so killing a writer cannot wedge a partial-frame reader. Linux rehearsal remains pending. |
| B10 | Disappearing PIDs during getsid tolerated; resolved PID count explicit. |
| B11 | Configuration fields named permissions/by-configuration. Rehearsal reports installed controller native-import guard, not measured GPU call counts. |
| B12 | Atomic observations/progress/final receipts; accumulating conditions; verification timeout separate from drift/failure; mid-run failure retains NO_GO receipt. |
| B13 | Separate canonical-JSON and actual source-file digests. Population digest identifies records, not whole-file serialization. |
| B14 | Primary initialization-inclusive aggregate remains fixed. Added mode, initialization-bearing/steady and order diagnostics; none replaces the primary endpoint. |

B3 evidence: `native-rejection-breakdown-corrected.json` in the readiness
report. New grouping evidence also revises G1: combined 78 references have seven
near-copy components but only one or two per cell. G1 is explicitly descriptive,
using joint annotation/native-proxy/novelty/distinct-child yield. It cannot
support family-held-out generalization or choose a production policy.

Loaded runtime inventory, independent review and Linux rehearsal remain
prerequisites. CI is not native execution permission. No threshold, numerical
tolerance, complete profile or downstream target-quality gate is waived.

# Second review: B15–B19 and executable N0

This follow-up is a new candidate, not a claim that f8870cd's review covers new
bytes. It adds a finite import-only N0 runner and local assembly of its manifest.

- B15: empty special mapping allowlist by default; only an exact reviewed path
  and permissions can classify non-executable data. Unknown executable/deleted
  code remains a failure. File-backed scope and anonymous/kernel exclusions are
  explicit.
- B16: controller and children launch with `-B`; controller scripts refuse before
  importing project code if bytecode writes are enabled. Existing public
  bytecode is frozen separately (5,484 files in the readonly snapshot).
- B17: N0 measures verification bytes/time and first loaded audit time. N1 checks
  its frozen 1080-second projection plus both verifications, four initial audits
  and ten seconds cleanup against 1200 seconds. Deadline-aware release hashing
  and post-verification reserve prevent treating unchecked completion as parity.
- B18: applicable cgroup path and ancestor limits are resolved, with unavailable
  bounds explicit. Host available memory alone cannot satisfy the start gate.
- B19: receipts explicitly state worker-only address-space identity. CLI children
  inherit the frozen executable/library paths; their loaded maps are not claimed
  to have been observed.

Control-flow tests do not certify native equivalence. Actual N0 and N1 remain
unexecuted in this packet. Root separately measured the f887 dedicated-session
containment rehearsal successfully; that is process containment evidence only.

# Pre-execution source findings after review 3 began

Candidate b4165af was not executed. Readonly inspection found two concrete
startup blockers; its artifact is retained as superseded preregistration.

1. `ImmuneBuilder/refine.py:26` constructs a ForceField at import. Revision 2
   correctly forbade it, so it could never complete. Revision 3 permits only the
   pinned caller/line/constructor/XML, one parameter parser construction and one
   observed XML open. The XML has no includes or scripts, and inspected parser
   methods register data. Context, System, model and physics execution remain
   forbidden. Wrong source, caller, argument, extra count or XML path fails.
2. The native installation owns a regular `scripts` package for TNP region
   definitions; repository research scripts form a namespace directory. Exact
   file-based helper loading avoids that collision while preserving native
   scientific package resolution. A fresh-interpreter regression reproduces
   the collision and verifies both paths. A separate four-file readonly
   inventory supplies the native source and existing bytecode identities.

The empty special-mapping policy remains valid if actual N0 observes no mapping
that needs exemption. Unknown file-backed origins still fail. No exemption is
invented from a hypothetical mapping, and the policy has not been broadened.
No N0/N1 profiles, native imports, model construction or remote mutations were
performed during this source audit. Tests exercise guards, not native parity.

# Review 3 N0 follow-up (protocol revision 4)

- N0-1: fixed allowlisted stages distinguish environment, ForceField registration,
  filter activation, the declared import, stage locality, CPU platform, plugin
  classification, constructor counts and loaded identity. No raw backend text.
- N0-2: unresolved origins are actionable only within explicit public runtime,
  candidate source, shim and system-library roots. Everything else stays hashed;
  credential-shaped paths remain hidden. Class counts do not grant exemptions.
- N0-3: `activate_filter()` already checked metanano's top-level staged origin;
  the review's claim that it did not was incorrect. Added recorded checks for all
  loaded metanano/nanobody_opt modules and namespace paths in N0 and N1 workers.
- N0-4: N0 is now registered for 120 seconds, still two CPU slots/4 GiB sampled
  abort/zero GPU. Locals inspection is restricted to relevant guard sites. The
  prior post-check did not fail by construction: reserving `v+5` and spending at
  most three seconds cleanup left at least `v+1` before the old deadline. The
  genuine concern was total feasibility under 60 seconds, not that arithmetic.
  Import, verification and first loaded-audit times are recorded separately.
- N0-5: N1 verifies actual receipt SHA, clean complete status, source/runtime
  binding, import evidence and exact receipt-derived timing values. Truthy
  strings or edited timing values cannot substitute for the receipt.
- N0-6: entrypoints use readonly resolved-path evidence. Prior qualification
  content is restricted to two pinned binaries in the explicit external tool
  root; it cannot supply candidate source.
- N0-7..10: guard permissions are labelled as permissions, missing child receipts
  preserve signed exit codes, memory/time abort conditions are distinct, and
  N0 asserts the same PATH/LD_LIBRARY_PATH inheritance as N1.

The resource helper's fail-closed behavior is unchanged. No narrower cgroup
mount is substituted when the broadest bound cannot be established. All old
review/CI artifacts remain evidence for their own immutable revisions.


# Actual N0 and review 4 corrections (protocol revision 5)

Actual d766 N0 failed safely before identity collection; a separately bounded
diagnostic pinpointed urllib3's local IPv6 bind. No scientific profile ran.
Pinned source proves a single `::1` ephemeral capability probe and socket close.
The narrow caller/source/address/socket/count/closure exception permits only
that operation; subprocess and external connection restrictions stay intact.

Actual Opus review 4 proposed two XML opens, but direct inspection of the pinned
`openmm/app/forcefield.py` (SHA `7c909edd1bbb8c48ac0070e5ea19b6bf0817be2aeccfbd23e7ebb1302c0da432`)
refutes that premise. Lines 249–254 resolve the file using `isfile` checks before
line 256 parses it. No relative parse/IOError retry exists in this version. The
one-absolute-open guard remains correct and unchanged. The review's proposed
two-open contract would itself cause an import failure. The source evidence and
actual failed diagnostic are retained; tests cover every loopback constraint.
The full native N0 remains required.

The N1 gate now binds the supervisor and containment receipts and validates the
exact declared imports. Source/CI evidence, actual N0 closure, and actual N1
parity/cost remain distinct. Inputs, scientific methods, budgets, performance
thresholds and all prior failed evidence are unchanged.


# Second actual import refusal (protocol revision 6)

The corrected 80b27fa N0 passed the loopback gate, then stopped safely at NumPy's
`check_support_sve` read-only `lscpu` subprocess. A bounded diagnostic preserved
the original guard and identified the exact public source site. The old runtime
inventory omitted all 14 NumPy testing source/bytecode files even though SciPy
imports them. They are captured in a new explicit readonly inventory.

Revision 6 preserves native behavior with one source/caller/argv/constructor/
executable/library-bound CPU query, requiring captured output, no shell/custom
environment/stdin, clean exit, exact counts and existing owned-tree limits. No
raw output, argv or environment enters receipts. The native OSError fallback is
not forced. No other process exception, scientific method, input or budget changes.


# Actual loaded-identity result (protocol revision 7)

bc803c0 completed all declared imports and ForceField/loopback/CPU-query checks,
then failed the loaded identity audit safely. Its 224 unresolved observations
contained 164 distinct public files and two virtual module metadata labels.
An explicit readonly inventory now pins those 164 paths; no special mappings
or unknown executable code are exempted. Both defining Torch sources and their
bytecodes were already frozen and observed correctly.

The two source-proven ModuleType subclasses are now recognized by actual class
identity, exact literal label, owner origin and pinned source hash. Separate
receipt metadata records their provenance. Ordinary relative files and all
other virtual modules still fail; tests include impostors and actual source-byte
drift. All prior source commits, failures and diagnostic receipts remain retained.


# First frozen N1 result and separate follow-up (protocol revision 8)

Actual N1 on 65b503f stopped after 39.317188 wall seconds and 115.123812 CPU seconds.
One CLI observation passed identity; one warm profile completed computation but
failed its post-profile identity audit. Seventy observations are missing, zero
comparison pairs are complete, and no parity/speedup conclusion is available.
This is an infrastructure closure failure, not evidence against CPU model reuse.
All partial measurements and clean fixed-point cleanup receipts remain retained.

The failed receipt identifies 87 exact public paths via their locator hashes,
plus one generated module. No broad scan or inferred wildcard is used. The
pinned Torch instantiator and template deterministically produce one 2355-byte
non-scriptable module. Its exact content/name/owner/source/private-directory and
loader registration requires fresh content/source hashes at every audit; the
unchanged unknown-file checks cover everything else. Tests reject changed source,
payload, module identity, loader, owner directory, path escape and symlinks.

A separate reviewed follow-up must preserve all 24 identities, mode order,
72-profile completeness, scientific methods and cost/parity endpoints. It starts
fresh rather than filling only the missing rows. The failed trial is not relabelled
or combined with the follow-up. No production/full-validator qualification follows
from either CI or the import preflight.
