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
