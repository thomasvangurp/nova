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

Configuration revision 2, before any measurement: OMP, OpenBLAS, MKL and
OpenMM CPU thread settings are one in every mode. This does not mean the prior
CLI refinement used two threads: its explicit OpenMM argument was already one.

Resources: 72 profiles, four workers, eight affinity-constrained CPU slots,
24 GiB sampled RSS abort threshold, 300 seconds per profile, 1,200-second total
run wall ceiling, zero GPUs. This is at most 2.667 allocated CPU-slot-hours,
not measured CPU-core time. Record actual self plus reaped-child CPU seconds
separately. Pin verification/setup before and after and bounded cleanup count
inside the same wall budget; ten seconds are reserved for cleanup. Prior slot
times suggest approximately 16–18 minutes of profiling at
four workers, with uncertain transfer. Stop at the cap and retain every case.
The cgroup v2 mount is read-only, not delegated. The RSS threshold is not a
kernel-enforced aggregate ceiling; it overcounts shared pages and can miss
spikes. Require at least 96 GiB host/cgroup available headroom to start and
abort below 64 GiB. This does not guarantee against OOM or authorize new infrastructure.

The dedicated Linux session leader must have no existing descendants. It uses
a subreaper and a bounded fixed-point SIGSTOP scan (two consecutive unchanged
sets) before killing/reaping its private tree. Rehearsal covers a detached
sleeper and a bounded active fork loop; macOS unit tests cannot prove this.
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

- Incomplete measurements, any gate/flag/reason/type mismatch, runtime drift,
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

Any mode reaching 300 seconds aborts the whole trial; a timed-out warm process
is never reused. Atomic per-observation files preserve partial evidence. Earlier
conditions survive later verification timeouts. Error output is limited to stage
and exception class; permissions are labelled as configuration, not measured
call counts. Report initialization-bearing/steady and execution-order diagnostics
without changing the primary initialization-inclusive aggregate. Actual loaded
source/bytecode/library origins are checked against the frozen manifest before
work and after profiles; identity-audit time is separate from profile timing and
charged to the total trial budget.

## G1 revision 2: descriptive prospective joint-eligibility probe

The 73 census native passers are exactly two near-copy cliques under the pinned
predicate: 68 length-18 and five length-16 references. Combining the five pilot
passers gives 78 references in seven components: [68, 5, 1, 1, 1, 1, 1]. Per cell,
IGHV3-23:16 has six references/two groups; :18 has 68/one; :5 has two/two;
IGHV3-7:16 has two/two. These are near-copy groups, not established biological
lineages. See `native-pass-reference-groups-v2.json` in the readiness report.

This supersedes the original G1 proposal before any generation or measurement.
All four cells now have native-proxy-passing full contexts, including IGHV3-7.
There are only one or two groups per cell: family-held-out comparative merit
or generalization claims are unsupported. G1 is a descriptive feasibility
probe capable of finding a very large effect, not an arm-ranking experiment.
At 1.4–1.8% baseline yield, 16 emissions expect only 0.22–0.29 passers; zero does
not exclude a modest useful improvement. No production generator is selected.

Retain the 2×2 design to separate two mechanisms: hash-ordered references from
the same frozen annotated population versus measured native-proxy passers;
CDR3-only/germline construction versus retention of the full reference context.
Freeze the combined measured reference population, exact annotations and all
memberships first. Use same-V/same-length references in each of the four cells,
including the newly available length-5 and IGHV3-7 contexts; this changed stratum
is explicit. Match mutation counts and ordinal seeds, never inherit parent
eligibility, and never use old-target binding ranks as new-target merit.

Proposed cap: 16 emissions per factorial/cell, 256 total, retaining cross-arm
memberships while profiling each distinct sequence once. At most 256 fresh
native profiles, eight workers, 16 CPU-affinity slots, 40 minutes, zero GPUs.
Use the same explicit sampled-memory/headroom policy as N1. This is at most
10.667 allocated CPU-slot-hours, with actual CPU measured separately. Prior
slot costs suggest 29–32 minutes of profiling; uncertainty and cap exhaustion
remain explicit. G1 requires its own reviewed input/source registration before
any generation; N1 approval does not authorize G1.

The primary endpoint is the joint rate of distinct children passing exact
annotation, native-proxy thresholds AND the pinned novelty predicate against
the frozen current archive. Report target-observation-ledger NEW eligibility
separately; unknown history is not automatically NEW. Report marginal native
and novelty yields, per-reference/per-cell outcomes, all failing stages and
wall/CPU cost per jointly admitted distinct child. Neither the older native
closure nor a small selected sample certifies complete validator acceptance.

Keep all descendants of a reference group together. Report effective reference
and child-group counts and the absence of independent holdout support. The
additional pairwise within-fleet diversity policy is ours, not an upstream
same-epoch requirement; the small group count alone does not imply the validator
forbids 13 submissions. Actual archive novelty/depletion still matters.
Retaining an unchanged CDR3 may be informative now, but does not sustain later
novelty once that group is archived. Future discovery needs genuinely new CDR3
contexts and joint novelty/developability/target-quality evidence under depletion.

If context retention gives promising joint supply, price the unchanged C2
quality gate under the actual pinned runtime: a real 13-entry fleet, ten
selection and ten disjoint held-out new-target observations per sequence,
common frozen field and actual novelty. New independent contexts must first
be acquired to support broader family/cell generalization. A separately frozen
parent-free/different-V discovery probe is an option, not an implied extra
budget or permission to repeat generations until a favourable result appears.

Cheap annotation/sequence rejection remains first. An optional TNP-before-A3D
short circuit could save only measured A3D time, not model/refinement work.
Changing order changes first-failure counts; retain all failing gates to compare
such a follow-up. Eligible entries always require full native measurements.
