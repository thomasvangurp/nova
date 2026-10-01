# Offline eight-GPU NIM deployment

The requested test uses exactly one eight-GPU host, instance 51335750. The
production nanobody host 51125470 is outside this playbook's scope.

## Runtime boundary

NIM 1.9.0 keeps its digest-pinned packaged Python/PyTorch/CUDA runtime. It is
not pip-installed into Pixi and no NIM source or compiled engine is patched.
The compute client talks to its loopback HTTP API.

Search and independent reference rescoring use the isolated
[Blackwell Pixi manifest](../deploy/nim19_offline/pixi.toml). The existing fleet
`bz2` lock uses PyTorch 2.6/CUDA 12.6 and is not changed. The new profile uses
PyTorch 2.7.1/cu128 and a matching CUDA 12.8 compiler for the novelty extension;
NIM continues using its own CUDA 13.2 runtime and the host driver.

## Playbook review and execution

[The scoped compute playbook](../deploy/nim19_offline/compute.yml) derives its
gates from `nova-compound-fleet/ansible/provision.yml` and `preflight.sh`:

- Direct SSH from current Vast port metadata, not the Vast gateway.
- Pixi 0.73.0 archive SHA verification, visible initial lock generation, then
  `pixi install --locked`; no post-install pip overrides.
- Public source/data only. No `nova.env`, wallet, coldkey, chain environment,
  publisher, external submission, production service restart or deploy key.
- Checksum transfer without owner/group/perms propagation; no laptop UID can
  change `/root` ownership and break SSH authentication.
- All 21 MSAs, checkpoint/CCD hashes, complete CCD store, database hash and
  read-only database.
- Hostile `PYTHONPATH` isolation; vendored Boltz source provenance.
- Eight real CUDA matrix operations; novelty extension compilation **and an
  executed result check**; a real fused-kernel forward pass; original validator
  gate audit.
- The original compute preflight must print a summary with zero failures.
  The wrapper rejects a host containing `nova.env` before that legacy helper
  could source it, and rejects any `.bittensor` directory.

Runtime installation can use the `runtime` tag while public scientific assets
are transferred independently. A file lock serializes Pixi writers; gates
are run only after both paths finish. Package and authentication error text is
captured locally; only static task names, exit codes and allowlisted status
fields are returned.

The first complete compute preflight on this host passed 17 checks with zero
failures and one expected warning (no accumulated own-score store yet). Verified
checkpoint hardlinks are used because the legacy byte-size check measures a
symlink's size rather than its target. No checkpoint data is duplicated or
re-downloaded. The actual CUDA novelty and fused Boltz forward tests passed.
The complete 18-task scoped Ansible playbook then finished successfully in
97.75 seconds on a warm installation. The generated lock SHA-256 is
`72a047f40161a9858c621b4a0cb35739cfd3374a3748f6756ec482ce9e42be23`;
its manifest SHA-256 is
`f32ae95f46dabc7c80f5af1c881dd90c396d899d677bd46940d7f449942745b2`.

Three real NIM waves of eight molecules measured 8.016, 8.430 and 7.907 seconds;
all eight GPUs reached 98–99% utilization. This extrapolates to about 78 minutes
for six 768-molecule full-scoring passes alone in the superseded funnel, before
search and reduced screens; it never demonstrated a 72.2-minute epoch fit.

## Active strategy: direct full-quality NIM

The user requested removing the reduced funnel after reviewing the measured
costs. A proposed 1,408-per-round run was superseded by the user's explicit
72-minute total limit. The active strategy scores **up to 384 fresh molecules per sequential round**
directly at full NIM quality (`r3/s100/d1/as100/ad3`, stock affinity refinement),
with no reduced pass and no 768-candidate cut. Chemistry, historical similarity 0.5, submission
size 20, AtomPair entropy 0.25, six sequential feedback rounds, and the matched
final canonical rescore against the public leader remain unchanged.

The eight-molecule reduced NIM pilot measured 5.372 and 4.885 seconds; full NIM
waves measured 7.907–8.430 seconds. The reduced pilot overlapped the main search,
so these are preliminary scheduling projections, not an isolated speedup claim.
The old 1,024-to-768 funnel only saves scoring work if reduced cost per molecule
is below 25% of full cost. The observed ratio is roughly 60%, so direct scoring
all 1,024 would cost less. The larger 1,408 proposal would have used approximately
the old funnel's scoring budget, but could not fit six rounds into one epoch.
The pilot's reduced/full rank correlation was 0.429 and 0.786 on eight public
leader molecules; this is not sufficient to validate a reduced screening policy.

At measured warm throughput, 384 full-quality predictions project to
6.3–6.7 minutes per round (38–41 minutes across six), plus acquisition and
the independent canonical comparison. Approximately 55–65 minutes is a working
total estimate, to be replaced with actual complete-round measurements. This
is an epoch-fit trial, not an optimality or demonstrated fit claim. Startup and
warmup are included in the 4,320-second limit. The planner reserves 600 seconds
for the final canonical comparison and reduces later batch sizes if measured
throughput/search costs require it. Each round has at most three acquisition
passes; an explicitly recorded partial pool of at least 64 is permitted rather
than wasting the epoch chasing a fixed count. Both the runner and supervisor
enforce the deadline, with private process-group cleanup. Six completed rounds
must be recorded; a timeout cannot be reported as six successful rounds.
All prior partial attempts are
preserved separately; no mixed-policy scores are reused as direct-run evidence.

NVIDIA's 1.9 optimization documentation distinguishes serial stages inside
one inference request from concurrent independent requests across GPUs. The
client keeps eight requests in flight, rather than waiting serially for every
molecule. `max_parallel_samples` controls repeated diffusion samples of one
complex, not independent molecules; our structure sample count is one. The
observed roughly 4.9–5.0 GB peak VRAM per 96 GB GPU is not proof that filling
memory would increase throughput, and 98–99% peak utilization is not an SM
occupancy measurement. Optimal queue depth/per-GPU batching remains unproven.
The 1.9 backend uses Torch/CUDA graphs, not the separate TensorRT backend in
the older published benchmark tables. Sources:
https://docs.nvidia.com/nim/bionemo/boltz2/latest/optimization.html and
https://docs.nvidia.com/nim/bionemo/boltz2/latest/performance.html.

## Scientific acceptance

The preliminary venv check is diagnostic, not an accepted deployment. It
preserved the original winner and top five but measured Spearman 0.98496,
below its strict 0.99 parity criterion. That failure remains recorded.

A new calibration must run under the locked Pixi profile. Strict parity and
eligibility for an **offline matched-runtime comparison** are separate fields.
Passing the latter cannot certify private-validator runtime equivalence.
Both our final portfolio and the public leader's portfolio are independently
rescored in the same Pixi runtime. NIM acquisition scores retain a noncanonical
label and receive the explicit molecular-weight adapter exactly once.

The locked calibration completed 20 molecules in 298.81 seconds and executed
11,520 fused triangular calls. Its strict 0.99 criterion remains **failed**
(Spearman 0.98496); winner and top five were preserved, enabling only the
explicit offline matched-rescore comparison. An older baseline artifact lacked
`model_source_sha256`; that parser error was preserved separately, and the
completed predictions were reassessed against the pinned original source
digest without repeating GPU inference. A real 20-molecule NIM canary portfolio
also passed the unchanged native validator functions (entropy 0.250291).

Repeated provider metadata requests hit HTTP 429 during setup. Public SSH
transport/state fields are now cached for five minutes and HTTP 429 receives
bounded backoff; credentials and raw account responses are never cached.
Fresh provider verification is still required for shutdown. The six-round job
started after reassessment on 2026-09-17 at approximately 19:05 UTC.

The first attempt stopped before GPU screening: one 250,000-candidate search
pass yielded only 441 historically novel archive entries, but the harness
incorrectly assumed it must yield exactly 1,024. The repaired acquisition loop
performs at most eight explicit refill passes until 1,024 fresh valid molecules
are available. It retains the 0.5 novelty ceiling, excludes previously screened
InChIKeys, preserves the 1,024-to-768 funnel, and records every extra proposal
and second in the round's acquisition metrics. If it still cannot fill, it fails
with the specific `acquisition_refill_exhausted` reason. This is additional
acquisition work, not silently counted as additional scored iterations.

`tools/nim19_reduced_screen_audit.py` is a separate eight-molecule stock-NIM
pilot with two reduced repetitions (`r1/s20/d1/as10/ad1`) and a matched OSS
reduced run. It reuses full-NIM canary predictions only for a preliminary rank
comparison. It never adds the public leader's molecules/scores to the search
archive or promotes the new pre-screen policy. NIM's separate affinity recycle
and skip-affinity-structure settings are not publicly exposed, so this recipe
is not labelled equivalent. The main search may overlap this small audit;
wall times are explicitly non-isolated and cannot establish a speedup. A larger
representative acquisition-pool retention test is required before replacing
the existing 1,024-molecule pre-screen.

Epoch 25176 is frozen with a settled leaderboard and pre-challenge history;
the 361-block nominal budget is 4,332 seconds. Six rounds remain sequential
with the direct-full strategy above. An overrun is reported, never converted
into an epoch-fit claim. No live submission is authorized. The isolated host
has a bounded lease watchdog and should be stopped after scientific artifacts
are collected; its disk is retained. The host lease safety deadline is separate
from the scientific limit and allows collection/shutdown after the 72-minute
run. Only the same eight-GPU host is rented; production is untouched.

## Queued schedule comparison: 6 x 384 versus 12 x 192

On 2026-09-17 the user approved comparing the two schedules. The existing
six-round process is not patched or restarted. `tools/nim19_schedule_comparison.py`
queues the twelve-round arm after the baseline finishes, on the same eight-GPU
host, with no concurrent scientific runs. The comparison runner is transferred
under a separate filename so the current baseline cannot import the new version.

The twelve-round arm uses a new SQLite archive, seed 68, identical pinned search
and scoring source, frozen training/novelty/leaderboard data, MSA, Pixi lock and
full NIM recipe. SHA checks run before and after the second arm. Its twelve
batches are fixed at 192; it cannot quietly shrink batches to claim 12 x 192.
The existing baseline has an adaptive cap of 384. Actual counts are reported;
if that baseline shrinks, the result cannot be labelled an equal-2,304-score
comparison even if both arms finish.

Each arm separately has a 4,320-second limit including its initialization,
warmup, search, NIM scoring and independent canonical validation. The primary
comparison is the validated final 20-molecule portfolio available by that equal
deadline. A missing, partial, invalid or late arm makes the head-to-head score
comparison inconclusive. Running two trials sequentially necessarily takes
longer than one epoch of wall-clock rental time.

A subsequent matched audit rescans only the precommitted final portfolios and
public leader together, deduplicated, in the same canonical runtime with seed
68. Shared molecules receive exactly the same audit prediction. This audit has
a separate 1,200-second cap, cannot select an improved portfolio, and cannot
retroactively turn a late or incomplete trial into an epoch-ready entry.
One audit does not eliminate stochastic uncertainty or prove general optimality.

Per-round fixed portfolios are also independently evaluated using each arm's
final canonical predictions. Curves at equal 15/30/45/60/72-minute search
checkpoints are explicitly retrospective: independent validation had not
necessarily completed at those checkpoints, so they are not submission-ready
claims. Raw NIM totals are never substituted for these canonical scores.

The scientific lease watchdog has been rearmed for this approved pair, with
automatic collection and stop on comparison completion/failure. Its current
hard deadline is approximately 22:38 UTC; each individual trial still has its
unchanged 72-minute limit. Only instance 51335750 may be stopped, and production
instance 51125470 remains excluded. Direct SSH explicitly disables user
ProxyCommand/ProxyJump configuration and uses the current public Vast port.

## Leader-competitiveness continuation: 12 x 384

The user subsequently requested twelve rounds at 384 and clarified that beating
the leader is the priority. This is a longer **offline diagnostic**, not a
72-minute epoch-fit claim. At approximately seven minutes per round, twelve
rounds already require roughly 84 minutes before final validation. The
six-round trial finished in 2,826.78 seconds (47m07s), including independent
validation: our canonical score was 1.4364645 versus 1.9266314 for the public
leader rescored in that same runtime (public quoted score 1.8894036).

`tools/nim19_leader_extension.py` waits for the already-running 12 x 192
comparison to finish, then resumes the baseline through rounds 7–12 at exactly
384 each. The baseline's SQLite database is cloned with the read-only SQLite
backup API; all eighteen original search/NIM/selection receipts are preserved.
The first 2,304 scores are not recomputed and no score from the smaller-batch
arm is imported. The preserved original baseline is not overwritten.

Because the original run did not persist proposal RNG state, the continuation
replays only the pinned ProposalPolicy draws against the saved per-round
parents. It verifies tuple counts and NIM score receipts, restores the proposal
state, and does not repeat chemistry construction or GPU inference. Future
rounds persist RNG checkpoints. This reconstruction and resume warmup are
counted as diagnostic overhead.

The diagnostic has a maximum 7,200 seconds of active scientific time, including
the original baseline's full 2,826.78 seconds and the resumed computation and
validation. Waiting between independent trials is recorded separately by the
workflow rather than presented as an uninterrupted epoch. The final output
explicitly states whether it fits 72 minutes; a twelve-round completion cannot
be relabelled as an epoch-ready result. Both the final molecules and the public
leader are independently rescored with unchanged native validator gates.

The same eight-GPU host is used; no additional instance is provisioned. The
hard rental deadline for this approved extension is now approximately 22:58
UTC, superseding the earlier pair-only 22:38 deadline. Collection and shutdown
happen sooner on completion. The lease watcher and diagnostic monitor now
retry transient provider/network failures instead of exiting and silently
losing the safety guard. Production instance 51125470 remains excluded.

The two fixed-count arms subsequently completed: 6 x 384 scored 1.4364645 in
47m07s; 12 x 192 scored 1.3206322 in 56m03s. Both scored exactly 2,304 and
completed independent native validation before their 72-minute deadlines.
This is one epoch/trajectory, not a general proof that fewer rounds are optimal.

The extra shared-batch audit and retrospective curves initially hit
`ModuleNotFoundError`: their controller used the base Python instead of Pixi.
The primary independently validated scores are unaffected. The bounded repair
in `tools/nim19_schedule_audit_repair.py` waits for the active 12 x 384
diagnostic, then launches the heavy audit and curves through the locked Pixi
shim. It preserves the original report, cannot change the primary arm timings
or rerank their fixed final sets, and gates collection/shutdown until it is
terminal (or the existing hard rental deadline arrives).

## Completed diagnostic and shutdown handoff

The monitor reported the 12 x 384 continuation complete: 4,608 full-quality
NIM scores across twelve sequential rounds, with a final independently
rescored portfolio total of **1.5294801398686049** versus
**1.8966195140757447** for the leader rescored in that run. Active scientific
time, including the reused baseline and final validation, was
**5,716.838360439986 seconds (95m17s)**. This is approximately 19.4% behind
the matched leader and **does not fit a 72-minute epoch**. The final raw NIM
portfolio total was 1.5395060972977488; it is not the canonical result.

The monitor subsequently reported the locked-Pixi shared-audit repair complete.
Its numerical audit results have not been recovered locally and must not be
inferred from that completion flag. The final collection raised `TimeoutExpired`
and the monitor's shutdown cleanup stopped the host. On the user's subsequent
shutdown request, a fresh guarded provider lookup confirmed instance 51335750
is stopped/exited. No restart, new rental, or production-host change was made.

**Evidence preservation is incomplete:** local baseline artifacts include
`data/nim19_epoch/20260917/results/final.json`,
`final-canonical-scores.json`, `candidates.sqlite`, the frozen challenge,
leaderboard, novelty snapshot, and calibration/preflight records. The extended
run and schedule-comparison directories have not been collected. The numbers
above come from the structured monitor events, not locally recovered final
JSON files. Keep the stopped instance's disk until those scientific artifacts
are recovered; destroying the rental now could permanently lose their exact
scores and receipts. A stopped disk may still incur storage charges. Recovery
requiring a restart needs explicit direction; do not resume it automatically.

### Strategy evidence to work on next

Read-only analysis of the baseline's 2,304 scored candidates measured proxy
versus NIM score Spearman correlation **0.10846212323755529** within that
selected pool. The highest-proxy 192 captured **0 of the actual best 20**;
the highest-proxy 384 captured 5 and the highest-proxy 768 captured 10.
These are conditional, retrospective diagnostics, not an unbiased holdout
accuracy estimate. The scored archive held 24,768 novelty-eligible candidates
in total, of which 2,304 received scores.

The baseline's unconstrained top-20 NIM total was 1.4557381380626613, matching
its round-six selected total within floating-point precision. Entropy selection
therefore did not materially reduce that portfolio's NIM score. Its final
canonical entropy was 0.436045151042464 against the unchanged 0.25 floor.

On the baseline's matched canonical comparison, our selected molecules had
mean heavy-atom count 41.5 and mean binding probability 0.37585539147257807;
the leader had 37.95 and 0.8012141227722168. Our role-A building blocks numbered
17 distinct IDs versus the leader's 3. These are baseline-versus-leader
observations, not measurements of the unrecovered twelve-round portfolio.
They suggest investigating deeper exploitation of productive chemical families,
but do not establish the leader's actual search algorithm.

Code review found that `tools/nim19_epoch_runner.py` loads its proxy once and
adds new scores only to the archive: parent choice adapts, but the proxy and
historical component priors are not retrained between batches. The reviewed
ProposalPolicy enables context-conditioned replacements only for reaction 5;
this challenge uses reaction 3. Its replacement priors therefore ignore the
retained building-block context. The acquisition client also retains a fixed
50% proxy-mean / 25% tree-disagreement / 25% broad-rank allocation. Tree
disagreement must not be assumed to be calibrated uncertainty.

Prioritize matched-budget experiments on online proxy updates, context-aware
reaction-3 substitutions, and novelty-aware acquisition that targets improvement
of the valid best-20 portfolio. Audit target-encoding cross-fitting and label
reliability before trusting surrogate improvements. Keep NIM labels distinct
from canonical labels, preserve the molecular-weight correction exactly once,
and independently rescore final portfolios under unchanged validator gates.
Use pre-challenge-frozen inputs across several held-out epochs; the current
leader is diagnostic evidence, not training data for a fair replay. No strategy
change, further GPU rental, or live submission was authorized by this handoff.

## September 18: recovered evidence and opt-in strategy work

The user authorized reviving the same instance on September 18. A fresh provider
lookup confirmed 51335750 running; production 51125470 was untouched. The
existing extracted NIM runtime passed a real CUDA probe on all eight GPUs and
became ready after 62.07 seconds. Eight full-quality predictions on public,
previously scored search candidates completed in **8.316523443092592 seconds**.
No runtime patch, package installation, new rental or chain submission occurred.
The local lease watchdog was rearmed to stop this same instance by
**2026-09-18 14:18:36 UTC**, two hours after the resume request. Do not reuse old
monitor/supervisor start operations: the previous trials are already complete.

Prioritize small terminal artifacts before large baseline CSV/model transfers.
Compressed direct-SSH rsync recovered all 53 extended-run JSON files and its
SQLite archive into `data/nim19_epoch/20260917/leader-12x384/`, the 11 shared
comparison JSON reports into `schedule-comparison/`, and all 48 twelve-by-192
JSON files plus that arm's SQLite archive into `schedule-comparison/12x192/`.
This supersedes the incomplete-collection warning above for those datasets;
the baseline proxy/history transfer has not been certified complete locally.

The repaired shared audit really completed under Pixi. Its fixed, jointly
rescored portfolios totaled **1.3106605699598122** for 6 x 384,
**1.2478935872383559** for 12 x 192, and **1.8815977836073265** for the public
leader. Thus the larger-batch arm also won the shared audit, by 0.0627670.
These are additional stochastic rescores, not replacements for the primary
trial totals. The six-round portfolio's 1.4364645 primary versus 1.3106606 audit
demonstrates material rescore variation: repeat finalist evaluations are needed
before interpreting small gains as strategy improvements. It does not by itself
isolate NIM disagreement, inference noise, batch-order effects or bias.

### Implemented experimental controls (default off)

`compound_miner/adaptive_strategy.py` provides a separate 64-tree online residual
head on the frozen historical proxy, trained only from prior current-challenge
NIM batches. It rejects wrong reaction/identity, mixed scorer labels, nonfinite
values and inconsistent score formulas. Repeated observations are averaged per
chemical identity. Historical validator labels are not relabelled or refit;
uncertainty remains a tree-disagreement heuristic, not calibrated posterior risk.

Reaction-3 FeedbackProposalPolicy keeps the original parent/move mixture but
offers contextual replacement sources: an experimental 65% current-feedback,
20% historical-context and 15% unrestricted valid-role draw. Historical and NIM
context estimates are separate. The unrestricted branch keeps previously
unseen building blocks reachable. Those fractions are test settings, not a
validated optimal policy, and full product novelty is still a hard gate.

The archive refresh updates only unscored proxy means/disagreement values;
existing NIM or canonical score fields remain untouched. This matters because
INSERT OR IGNORE otherwise leaves old acquisition rankings in the archive.
SearchEngine's optional proposal-policy factory was added to both local
repositories without changing its default policy. The offline runner exposes
independent `adaptive_proxy` and `contextual_rxn3` boolean challenge flags,
both false by default. Adaptive resume is rejected until feedback state can
also be restored; a future fair comparison must start with a fresh archive.
Updates and reranking time are counted in the unchanged 72-minute budget.
The compute transfer includes the new helper for future deployments; the
running instance's original baseline search/runner files were not overwritten.

### Prediction-before-update diagnostic (not a competitive search trial)

`tools/analyze_compound_adaptation.py` replayed the twelve saved 384-molecule
NIM batches. Both proxies predicted each batch before its labels were added;
no public leader label was used. The first cold batch matched exactly. Across
the eleven subsequent batches, mean within-batch Spearman was **0.0118257
frozen versus 0.4917846 adaptive**. A highest-predicted-192 shortlist captured
an average **9.8182 versus 16.5455** of that batch's true best 20, and its best-20
NIM total improved in all eleven batches (mean 1.2172970 versus 1.2775745).
Observed updates took 0.51–1.47 seconds; the entire replay took 23.53 seconds.

This is conditional on the old policy's acquired pools, not an unbiased holdout
or a new trajectory, canonical validation, portfolio entropy proof, epoch-fit
result or evidence that the leader was beaten. The diagnostic JSON and unit
results are retained in `data/nim19_epoch/20260917/strategy-work/`.

Validation so far: 12 functional strategy unit tests passed under the unchanged
locked Pixi profile; 43 dependency-free harness/wiring regression tests passed
locally. `tools/smoke_compound_adaptation.py` is a bounded integration check on
real public chemistry with **explicitly warm-started 4,608 prior NIM labels**.
It must not be reported as a fresh six-round experiment or a competitive entry.
It uses a separately named experimental search module, the strict 0.5 novelty
gate, real new NIM predictions and native portfolio validity functions. The
saved completion was verified on September 18: 959 novel constructed candidates,
32 fresh full-quality NIM scores in 29.4986 seconds, 161.3550 seconds total,
20 native-valid selected molecules, AtomPair entropy 0.4017555, and NIM total
1.5450472. There was no canonical rescore. These warm-started results do not
establish fresh search competitiveness or epoch fit.

Next scientific step: a fresh 6 x 384 matched-budget trial, using pre-challenge
frozen data and no leader labels, with opt-in controls and independent canonical
validation. Isolate online learning and contextual search in subsequent arms;
do not replace the baseline or announce a live-ready strategy from this replay.

The user narrowed the next test to **one epoch**, not six epochs, and requested
a Claude handoff before the fresh trial started. Current code has contextual
mutations but no explicit row/column sweep or guaranteed scoring lane. See
`HANDOFF_CLAUDE_MOLECULES_20260918.md` at the repository root for live host,
deadline, source isolation, evidence, and the remaining implementation work.
