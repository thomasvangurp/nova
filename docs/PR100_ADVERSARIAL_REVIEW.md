# Independent review ledger

Reviewer: Claude Code, requested model `opus`, effort `max`, actual CLI result
reported `claude-opus-5-5` and `claude-opus-5`. Round 1 took 794.6 seconds.
It received an explicitly scoped source bundle with tools disabled, without
credentials or production write access. It did not run tests. Root ran the
tests and measurements separately. Preserve the full original response in the
local readiness report; this ledger records disposition, not a rewritten review.

Round 1: workflow NO_GO; production NO_GO. Root agrees that the proposed
workflow lacked explicit capacity bars and enough oracle/dependency evidence.
Neither a critique nor its rebuttal is evidence of execution.

| Finding | Disposition for round 2 |
| --- | --- |
| F01 challenge RNG caller | Exact caller uses config and block_hash only, so num_antitargets defaults to zero. New parity test executes that recorded call with the actual pinned config loader over 10,000 hashes. |
| F02 unpinned oracle fixture | All twelve upstream files are now retained byte-exact and tested against the upstream manifest. Challenge test reads that verified source. Fresh live upstream drift/merge verification remains a production gate. |
| F03 uncommitted evidence | Correction e43f9687 passed 850 shipping tests in both profiles. Later edits are explicitly unqualified until committed and re-run. Evidence table separates earlier census, native profiles and registered-scorer probe. |
| F04 activation assumption | Candidate now refuses legacy config at 25955 and PR100 config before 25955. Future preparation uses pinned_config separately; it cannot itself approve publication. Tests pass; later release qualification pending. |
| F05 self-consistent context | Internal context validation is not an external chain attestation. Producer chain-state validation and final Pi checkpoint binding are supplied for review. End-to-end stale authoritative state/rollover tests remain required. |
| F06 epoch length and diagnostics | Confirmed robustness gap: default 361 and silent per-checkpoint rejection need explicit external epoch dimensions and diagnostics before production. Current live length is 361, not evidence that other values work. |
| F07/F08 developability predicates | Pinned metanano source explicitly defines all six bounds, inclusive A3D caps and rejection of a negative maximum. Removing them would diverge from upstream. Differential measured profiles and actual filter boundary tests remain required; artificial boundary profiles cannot be represented as real structures. |
| F09 novelty absent | Existing complete-screen and cached-precheck paths already call current archive novelty. Their source is supplied. This was missing review context, not proof of absent checks. Full real-archive differential and drift trials remain required. |
| F10 cache fixture path | Corrected to place the legacy profile at the actual cache path with matching outer identity; tests exercise both cached_result and cached_develop, requiring remeasurement. |
| F11 import closure | Named-sibling test is insufficient as a general closure proof. All 130 actual BoltzGen package files matched candidate bytes in the runtime audit. General import-closure rejection and image/native closure remain gates. |
| F12 worker regeneration | Unit test reproduced the installed scorer's second-chunk failure. Candidate fixes it and retains replica support. Real corrected non-replicate multi-chunk test is still required. |
| F13 fabricated replica fixture | Ten real GPU records across three lengths are now retained as a provenance-labelled projection with original-file hashes; unpack passes against those records. They belong to the old registered scorer, not the correction. |
| F14 independence/isolation claims | Renamed hardcoded write counters to explicit scope declarations; isolation is marked uninstrumented. Different vectors alone do not prove independence. Found and corrected fixed seed reuse across jobs; new runtime proof and enforced write isolation remain required. |
| F15 probe pin/PYTHONPATH | Required expected SHA is now external to the manifest; ambient PYTHONPATH is replaced with the staged root. Host launcher and complete import closure must also match the qualification receipt. |
| F16 stale daemon | Confirmed important runtime gate. Supply complete daemon code; require loaded worker identity, not an on-disk hash alone. Stale-daemon trial and immutable rollout remain blockers. |
| F17 impossible promotion volume | Contract now expressly blocks full application promotion until real generation supplies the required volume. No cold-target waiver and no fabricated future-target history. A smaller policy requires a separate reviewed experiment/contract. |
| F18 no capacity bar | Fixed in workflow: at least 13 actual-validator-accepted entries for each of 171 buckets, plus three full-volume normal trials, before full production approval. Current evidence fails this bar. |
| F19 stale annotation identity | Confirmed gate. Bank and current annotation runtime identity must match; any semantic identity migration needs its own differential proof. Do not silently reuse an old census. |
| F20 native/submodule identity | Existing staged CI checks submodule pin, but native/runtime closure requires a separate receipt and enforcement. Recorded binary digests are supplied; installation and host verification remain open. |
| F21 NEW reservations | Existing screen_budget reserves unique IDs durably before dispatch, and dispatch calls pending after resume. Source/tests supplied. Parent-directory durability, separate status accounting, and real rollback trial remain to be qualified. |
| F22 durable artifact fsync | Confirmed durability gap for some artifacts; host-loss durability cannot be proved solely by SIGKILL. Add fsync enforcement plus crash/restart fault tests before production. |
| F23 future-prep bank path | Future preparation must be exercised through its actual consumer and typed by stage; no speculative challenge may enter scoring. Remains integration gate. |
| F24 sequence-only aliases | Final checkpoint requires full pr100_evidence; source supplied. Explicit stage typing and a sequence-only-to-publication rejection trial remain required. |
| F25 replay isolation/lock | Confirmed: technical probe avoids this production entrypoint, but replay worker still needs injectable private context and bounded locking before a realistic isolated full-path trial. |
| F26 cold-start field | Source documents a temporary internal selection reference, not rival performance. Replay waits for completed same-target epochs. Full cold-start-to-expiry integration and minimum support policy remain open. |
| F27 suite-list self-check | Existing artifact verifies declared suite completeness, not independent governance. Controller-side expected suite/version policy and omitted-suite mutation test remain required. |
| F28 release profile/live config | shipping_checks.py exposes --profile and already ran both profiles successfully. Live config is deliberately external; all production readers and activation freshness must be tested, not inferred from manifest success. |
| F29 annotation helper parity | Actual pinned helper uses the same IMGT definition and H49/H50 criteria. Differential real annotation evidence is still required; matching source intent alone is insufficient. |
| F30 SQLite lifetime | Candidate now uses contextlib.closing before annotation. Cache resume survives worker errors; a partial progress file is not a qualified bank. |

Workflow approval asks whether the amended procedure and pass/fail bars are
sufficient to execute, not whether the code has already passed them. Production
remains NO_GO while any required execution gate is open. Root will not force a
positive verdict by renaming a failed test or excluding a difficult population.

The user subsequently questioned the architecture itself. Round 2 also reviews
PR100_PIPELINE_DESIGN.md: compare a compatibility adaptation, entirely online
generation, and a prequalified challenge bank with asynchronous replenishment.
Parent reuse as a generation reference is distinct from archived sequence
resubmission; benchmark it against generation without historical parents.

## Round 2 and response for round 3

Round 2 took 606 seconds using Claude Code Opus at max effort. Its workflow,
redesign experiment and production verdicts were all NO_GO. The original
response remains unchanged in the report. Root retains production NO_GO.

- R01–R07: the workflow now starts with the full native census and warm
  throughput feasibility, then fixed conditional-yield cells. No architecture
  is preferred in advance. The design preregisters source strata, compute/time
  caps, K=20 archive-advancing epochs, matched schedules, primary endpoint,
  meaningful effect, uncertainty analysis, multiplicity and futility rules.
  Timestamp capture distinguishes individual times from batch bounds. These
  are a revised plan, not completed experiment evidence.
- R03: the current contract abstains at 25955 without a completed new-target
  field. Later epochs require actual sufficient support. No own-two-row or
  fabricated-history waiver is accepted.
- R06 correction: the measured inventory occupies eight buckets, with most
  entries in two. It does not occupy at most two. Parent usefulness across
  different lengths or genes is an empirical comparison, not logically zero.
- R08: independent Pi/RPC attestation of the epoch-start hash, including a
  self-consistent but externally wrong artifact, remains an implementation
  and integration blocker.
- R09–R10: annotation identity checking, dependency-version parity and exact
  novelty batch failure semantics remain blockers. The installed AbNumber
  0.2.7 differs from upstream requirements' 0.4.4; ANARCI pins also differ.
  The old census must keep its provenance and cannot certify that new runtime.
- R11: added actual archive-lag trials, both snapshot revisions/read times,
  late rechecks and explicit residual contention risk. An unchanged archive
  need not be strictly older at validation time; the important case is drift.
- R12: loaded worker identity, patch set and schema must be bound and a
  mismatch refused without silent fallback. Current production daemon parity
  remains unqualified; no source-only hot patch has been activated.
- R13: sixteen byte-exact upstream files now include constants, ranking,
  timing and re-exports. Oracle constants are loaded from verified source;
  ranking's path loader verifies and executes the same checked bytes. The
  complete transitive dependency closure still needs runtime qualification.
- R14: the current 32-bit contiguous-window derivation is not an adequate
  structural independence claim. Hashing the pair to 32 bits also cannot make
  namespaces structurally disjoint. A sampled no-collision test is not proof.
  The RNG redesign remains open; retries and held-out identity separation
  need explicit tests. Do not implement the reviewer's false guarantee.
- R15–R16: submodule/native enforcement, separate NEW state accounting,
  directory fsync and real crash/rollback tests remain execution gates.
  SIGKILL alone cannot prove host-power-loss durability.
- R17: subsequent actual-filter differential execution found zero differences
  on 14 recorded real profiles and 42 explicitly synthetic boundary cases,
  including PSH 73.4/89 and A3D caps. All 14 real profiles reject; this is not
  positive acceptance coverage. Retain source and native binary digests.
- R18: stricter completeness checks require population-level capacity cost
  measurement; the full census is feasibility evidence, not a synthetic pass.

The slave is reserved for qualification through a Pi-initiated controller.
All three master-side slave dispatchers are parked after the active replay
batch drained and imported. The controller holds the shared GPU dispatch lock;
the master miner and publisher remain running. The corrected five-candidate
fresh probe completed across three lengths. Replica and census results are
reported from their receipts, never assumed from launch success.

Round 3 is asked to judge whether the revised methodology is sound to execute,
separately from production approval. Open code defects remain production
blockers even if the test design receives GO. Neither reviewer may label an
untested architecture near optimal or use agreement as a substitute for data.
