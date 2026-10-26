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

## Round 3 and response for round 4

Round 3 took 586.1 seconds. Claude Code reported Opus models at max effort and
returned NO_GO for workflow, experiment, production and near optimality. Its
unedited response is retained. Disagreement is resolved against source and
measurements, not by accepting every reviewer assertion.

- R19: the isolated pinned AbNumber 0.4.4/ANARCI 2026.2.13.2/ANARCII 2.0.5
  runtime re-annotated all 4,080 eligible sequences: no eligibility or recorded
  field changes. Of 20,000 hash-ordered excluded/current inventory sequences,
  19,915 had old ineligible annotations and 85 had no old annotation; none
  became eligible. J identity changed in 2,927, V gene in five, V identity in
  three. The review's certainty that eligibility would change was unsupported.
  Both default/explicit CDR-definition checks had zero differences among 64
  rows each. This is not the full new-runtime inventory census or full validator
  execution. Commit 9955822 enforces runtime and actual panel/HMM identities
  before accepting cached annotation; 864 tests passed in both CI profiles.
- R08 correction: the source already independently fetched the epoch hash and
  tempo on Pi, and the miner writer fetched its hash from RPC too. Round 3 did
  not receive those files. The real defect was the downstream implicit length
  361. The correction threads authoritative length through all consumers,
  records RPC read blocks, and refuses a preflight crossing an epoch boundary.
  Tests include a nondefault length and internally consistent wrong evidence.
  Live integration and cold-start publication gates remain open.
- R20: Claude retracted the false 32-bit structural-disjointness claim. Its
  replacement birthday arithmetic is also corrected: one million uniform
  32-bit draws have collision probability effectively one, not 69%; about
  100,000 draws give 69%. Torch accepts at most 64-bit seeds and Lightning's
  seed_everything interface is 32-bit. A 96-bit integer cannot simply be passed
  to either. RNG-consumer-specific derivation, exact realized same-input
  collision accounting and empirical dispersion remain open. Different seeds
  do not prove independent observations.
- R21–R27: C2 version 2 explicitly withdraws the unbudgeted 600 full-volume
  trials. C1 remains unchanged. C2 has a 24 allocated GPU-hour ceiling on the
  existing four GPUs, reference availability labels, repeat work in the cost
  inequality, a 13-entry existence gate, ten matched one-epoch cases, a scale
  check, quality-first comparison after capacity, fixed margins and precision
  requirements. The 171-cell bar still requires actual 2,223 distinct accepted
  candidates plus measured attrition; a four-cell pilot cannot certify it.
  Generator algorithms and measured throughput must fix the execution manifest
  before strategy outcomes. No comparison has run or passed.
- R28: qualification abstention means no nanobody publication, not no research
  scoring. The proposed minimum new-target field is 49 distinct measured
  sequences from completed epochs; after three insufficient epochs the state
  is NO_REFERENCE, without an automatic waiver. Runtime enforcement remains
  open. Publishing unqualified entries to create our own reference is rejected.
- R15: release verification now refuses a missing/substituted filter revision
  in full and runtime-only checks, including controls packages. This does not
  close native binary/transitive-runtime qualification.

Native feasibility remains labelled separately from validator equivalence.
At 1,280 of 4,080 measured candidates, 18 passed; earlier zero-of-fourteen was
only a stratified pilot, not a population finding. The current source changes
have focused tests; every new commit still requires both shipping profiles.


## Round 4 and response for round 5

Round 4 took 492.9 seconds at Opus max. It gave a conditional GO to feasibility
measurements, while redesign, production and near optimality remained NO_GO.
All statements below distinguish measured evidence from future requirements.

- Full shipping CI subsequently passed 870 tests in both profiles at fe4e11d,
  after four incomplete transaction fixtures were corrected. The exact artifact
  was staged and verified on the slave through Pi; no application activation.
- Full pinned-runtime inventory: 391,935 distinct sequences, 4,080 eligible,
  eight nonempty cells, 728.415 seconds. A first attempt failed immediately:
  the `-c` child's current directory shadowed staged namespace-package modules
  with /root/nanobody_opt/gates.py. The restarted harness prepends the release,
  uses its working directory and asserts actual module origins. Failed attempt
  and both harness identities are retained. This is a test-runner failure;
  the full result is from the corrected run, not a partial result relabelled.
- Batch throughput: 128 distinct sequences, four GPUs, two chunks per shard
  (24 then eight), all 128 finite records and structures retained, 542.352 seconds.
  Second-chunk times were 122.135–124.865 seconds per eight records. These are
  partial warm chunks, not steady full-size production-daemon throughput. The
  idle legacy daemon retained VRAM. No equivalence or C1-volume claim follows.
- R08 source correction: pinned validator setup.py initializes epoch_length
  from SubtensorModule.Tempo + 1, the same source as Pi. Its main loop normalizes
  off-boundary reads to the next boundary before processing the closed epoch.
  Added the seventeenth exact oracle file and executable AST tests for both
  lengths 361/362 at four offsets, including the closed-epoch distinction at
  an exact boundary. Removed implicit 361 defaults. A validator's private
  already-running stale configuration cannot be independently observed by Pi;
  reading the same chain value twice would not prove that process's state.
- R29 correction: run_live_producer starts cached_confirmation whenever the
  unseen-candidates policy is active, before looking at NEW pool size. The
  alleged one-NEW-suppresses-cache defect is absent. Added tests for zero, one
  and eight NEW candidates; strengthened the workflow wording to every epoch.
  Live three-epoch lane-composition evidence is still required.
- R15 correction: shipping_checks stages the package before invoking the tests,
  so the packaged assertion already ran in shipping CI. It now explicitly
  refuses a missing stamp when the shipping artifact environment is set, and
  a named filter pin has a direct equality assertion against rules.FILTER_SHA.
  Native/transitive-runtime qualification remains open. Native executable
  digests already exist in census receipts; they are not yet enforced in the
  release/runtime cache contract. PSH is a TNP result, not produced by A3D.
- R33–R35: specified the log-ratio estimand, fixed three-pair family, zero allowed
  case exclusions, internal non-competitive C2 field, exact binomial detection
  formula and resource-limited outcome. Cold-repeat arithmetic exceeds its
  allocation, so it is not an executable plan; it is not a proof of warm
  infeasibility. A seven-day live-cell calendar cap cannot extend GPU allocation.
- R30: a concrete CPU-only conditional germline/loop baseline and fixed
  reference/operator rules now replace the unnamed generation hypothesis.
  Dry-run and annotation evidence remain required. This is not an endorsed
  molecular design strategy, nor a completed online/bank scheduler.

No design improvement or reviewer agreement qualifies unmeasured runtime
behavior. R10, R12, R16, R20, R31 and publication cold-start enforcement remain
open implementation/evidence gates; none was silently waived.


### Additional execution evidence for round 5

The fixed conditional-proposal v1 CPU pilot completed in 13.585 seconds:
16 arm records, 13 runnable and three with unavailable reference strata; 3,328
emissions, 2,764 globally distinct sequences, zero GPU calls and zero annotation
errors. Each runnable arm emitted 256; 251–255 passed the requested sequence
cell. Cross-arm repeats are explicit and must not become repeated NEW charges.
No native developability, novelty, binding or full-validator claim follows.

A live Pi check found consumer ea9677c4 and signer 4065f061 still installed;
neither is the tested PR100 application. The consumer has no cutover hold.
An initial hold-only local test caught that its old plan suppresses molecule-only
submissions and its signer rejects a null sequence half. Nothing was installed.
The proposed bounded hold therefore includes both pure-path repairs: allow the
validator's molecule-only wire format, retain all existing strict molecule
identifier/reaction checks, and return no nanobody allocations from 25955 until
an explicitly qualified release replaces the hold. Pure functions from the
actual installed files are retained as regression fixtures. No wallets, signing,
chain calls or authentication implementation are included in those fixtures.
This operational hold is separate from application promotion and must receive
local CI and independent review before installation. Install signer support
first, then the consumer hold, with source-hash preconditions and exact backups.
The proposed change makes no assertion of a qualified PR100 mining strategy.

The follow-up I/O test found a further dependency before allocation: main read
the nanobody JSON, and optionally fetched it, before the hold could run. The
bounded patch now skips only that nanobody fetch/read from 25955; molecule
loading and validation remain intact. Tests cover both missing and malformed
nanobody JSON, pull-only behavior, and unchanged pre-cutover behavior. This is
an independent-path repair, not a bypass of molecule validation or deadlines.
The first round-5 CLI attempt failed before returning a review. It is recorded
as a failed attempt, not agreement; the same Opus max review was retried.
