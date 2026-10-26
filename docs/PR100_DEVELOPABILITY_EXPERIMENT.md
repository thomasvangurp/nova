# Developability feasibility and native model reuse

Status: proposed CPU experiment, not permission to execute or deploy. The
frozen population is `experiments/pr100-native-reuse-v1.population.json`.
Runtime closure review and Linux process-containment rehearsal remain required.

## Evidence and verdict

The 4,080-record native census completed at concurrency **32**, batch 64, in
7,040.49 seconds: 73 passers (1.79%). Its passers occupy two challenge cells.
The conditional pilot completed 351 distinct profiles at concurrency 16,
batch 32, in 1,298.824 seconds: five passers (1.42%). These are different
populations and cannot establish that one generator or concurrency is better.
Approximate Wilson 95% intervals are 1.43–2.24% and 0.61–3.29%, respectively;
related sequences mean these intervals do not describe independent family yield.
These are measured native-proxy outcomes under the recorded older annotation
closure, **not fully qualified validator acceptance**.

All recorded Aggrescan statuses were `ok`. The pilot's 65 Aggrescan first-stage
failures were finite threshold rejections. Its other first failures were 228
PSH and 53 red flags. Faster execution cannot make those structures eligible.
Obtaining 4,800 eligible NEW evaluations from 8,000 proposals requires at least
60% admission before additional novelty/other attrition. Neither tested
population demonstrates supply near that requirement.

The baseline retains a reference's CDR3 but discards its measured framework and
CDR1/2 context; reference selection is hash-ordered, not based on native
feasibility. Six random CDR1/2 substitutions add diversity but have not
demonstrated adequate native yield. This is a plausible design limitation,
not a measured causal attribution. Sequence-only diversity supports 13-member
witnesses in the tested arms; native acceptance remains limiting.

The pinned service already parallelizes native profiling. Direct source
inspection confirms TNP launches `NanoBodyBuilder2 --n_threads 1`; refinement
uses OpenMM's CPU platform with one thread. GPU refinement or changing that
thread count changes the numerical path and is outside this experiment.
The repository already contains a persistent-model profiler. Its historical
3.3-second load, 0.34-second forward and 27.7-second refinement figures are
historical comments, not current PR100 measurements. It lacks complete PR100
Aggrescan evidence and is not qualified. It must not simply be enabled.

## N1: preserve computation, reuse resident models

Compare CLI_A, a fresh CLI_B repeat, and the existing warm profiler augmented
with Aggrescan on the same refined heavy-atom model before temporary cleanup.
Both paths use the pinned filter's `analyze_profile` and PR100's complete
developability predicate. Preserve four checkpoints, CPU/thread-one refinement,
sidechain checks, pH 7.4, hydrophobicity scale and exact Aggrescan caps.
No cache reads, structure substitution, fast refinement, threshold relaxation,
GPU use, production daemon, database, wallet or submission is permitted.

Population digest:
`1fe6a4bab9b41a055268036e89f088c998f4fe1046d56e68a2b75c20e3dd638d`.
The 24 real identities cover all four observed outcomes, lengths 5/16/18, five
gene/length cells, and profiles nearest each PSH bound and Aggrescan cap.
This is an enriched numerical stress set, not an acceptance-yield estimator.
IDs, order, historical evidence and source-receipt digests are frozen in JSON.
Latin-rotated per-sequence ordering balances which mode runs first. Four private
workers receive every fourth identity; warm models persist within a worker.

Every mode must compute a fresh profile. Report all six TNP metrics, Aggrescan
mean/maximum, provenance, structure digests, decisions, failures and missing
observations. CLI_A/CLI_B variability is separate evidence and never excuses a
warm-path gate flip. Native output text is discarded before it can reach the
report. Injected model/metric tests are control-flow tests, not native proof.

Resources: 72 profiles, four workers, eight affinity-constrained CPU slots,
24 GiB aggregate RSS ceiling, 300 seconds per profile, 1,200-second total
run wall ceiling, zero GPUs. This is at most 2.667 allocated CPU-slot-hours,
not measured CPU-core time. Record actual self plus reaped-child CPU seconds
separately. Pin verification/setup before and after and bounded cleanup count
inside the same wall budget; ten seconds are reserved for cleanup. Prior slot
times suggest approximately 16–18 minutes of profiling at
four workers, with uncertain transfer. Stop at the cap and retain every case.

The Linux controller uses a subreaper to retain descendants that start new
sessions, stops/kills its private tree on a cap, and checks zero descendants.
Before native work, a bounded Linux rehearsal must demonstrate cleanup of a
deliberately detached sleeping grandchild. macOS unit tests cannot prove this.
All remote writes/launches go through Pi into a new private directory. Do not
stop processes outside controller ancestry or install production packages.

Before measurement, freeze and independently review the runtime manifest:
Python and system shared libraries; loaded package code/data; TNP,
ImmuneBuilder, theraprofnano; all four weights; HMM/reference databases; HMMER;
filter/config/adapter; FreeSASA/Aggrescan; logging shim; versions, hardware,
thread/device settings and executable resolution. Record dynamic library
closure, not just package versions. Verify digests before and after execution.
The runner refuses an unreviewed inventory. This technical comparison under
an older closure cannot qualify pinned-validator caches or production.

Preregistered continuation rules:

- Incomplete measurements, any gate flip in any comparison, runtime drift,
  containment failure or nonfinite evidence mean NO_GO.
- Require exact numeric equality of recorded TNP/Aggrescan fields and
  normalized structure digests before calling this probe equivalent. Report
  raw structure digests too. Do not invent a post-hoc tolerance.
- Even complete equality establishes only this 24-input stress probe.
- Before a larger follow-up, require at least 10% lower paired mean wall time
  for warm versus the per-input average of both CLIs, and no more than 5%
  higher CPU time. Include initialization in the primary comparison; report a
  separately labelled amortization sensitivity. These are continuation rules,
  not a statistical superiority or production approval claim.

## G1: separate reference merit from retained context

Propose a 2×2 factorial: hash-selected versus genuinely native-passing
references, crossed with germline/CDR3-only baseline versus retained full
reference context. Match mutation counts and ordinal seeds, preserving every
annotation, novelty and native failure. A parent's eligibility never certifies
its child; old-target binding ranks cannot choose new-target parents.

Use three separate IGHV3-23 cells: lengths 16, 18 and 5. The first two use
same-V/same-length references; length 5 is explicitly same-V/different-length.
Never pool these strata. A full native-passing reference for IGHV3-7 is
currently UNRUNNABLE, not a measured zero yield or a disguised different-V arm.
The parent-free/different-V baseline remains relevant for empty cells.

Bounded next proposal: 24 emissions per factorial/cell, 288 emissions total,
deduplicated across arms while retaining every membership; at most 288 fresh
native profiles, eight workers, 16 CPU-affinity slots, 40 minutes, 24 GiB RSS,
zero GPUs. This caps timed allocation at 10.667 CPU-slot-hours, with actual CPU
and setup separately measured. Prior slot times suggest 33–36 minutes; a cap
hit remains incomplete. G1 needs a separate reviewed registration before
generation; N1 approval does not authorize it.

Freeze reference families using lineage and the pinned family predicate before
selection. Keep all children of one family in one split; report effective
family counts. Reserve whole families and leave-cell-out evaluation where
supply permits. If the 73 passers collapse into too few families, report that
limitation instead of randomly splitting near-copies. Report native passers
per emission and distinct candidate, cell, archive/pairwise attrition, wall/CPU
cost per accepted child and changing failure categories.

Do not choose a production generator on filter yield alone. After adequate
supply, price the unchanged C2 new-target quality gate: a real 13-entry fleet,
ten selection and ten disjoint held-out observations per sequence, common
frozen field, family-separated comparisons and actual archive novelty. Measure
repeat cost before committing remaining GPU budget. G1 does not reduce those
pass counts or promise near-optimality.

Cheap annotation/sequence rejection remains first. A TNP-before-Aggrescan
short circuit could save only measured Aggrescan time for already-rejected
profiles, not model/refinement work. N1 records Aggrescan time to price that
upper bound before changing anything. Eligible entries need full evidence.
