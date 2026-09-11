# Spending the fast scorer: breadth, replication, and the pipeline

Date: 2026-09-11

## Question

The replacement exact scorer is validated at 12.2x (176 -> 14.5 GPU-seconds per
sequence).  The committed dormant profile proposed spending it on breadth: 2,400
one-shot observations and eight finalists.  Is that the best use of it?

It is not.  Breadth is the one place the added throughput is close to worthless.

## Evidence

Sources: 2,508 live exact observations of 931 sequences from epochs 25036-25055
(2,200 one-shot screens, 308 replicated finalist observations), and the public
competitor fields of epochs 25035-25054.  Reproduced with

```bash
python -m nanobody_opt.backtest_budget_allocation \
  --live-root /root/live --field-dir <cached leaderboard payloads> \
  --output nanobody_opt/rd_results/budget_allocation_2026-09-11.json
```

The validator's published score is reproduced exactly from the metric vectors
(maximum absolute error 0.0 against the API's own `final_score`), so every
number below is on the leaderboard's own scale: a dense rank sum over ten
metrics against that epoch's field, lower being better.

### The whole contest is inside one observation's noise

Our mean deficit to the epoch winner is 75 rank points, of which 61 is the
confidence category and 15 physical interaction.  Developability is already
0.9 points *better* than the winners'; there is no headroom there.

That deficit is microscopic in metric space.  Against the winners we average
0.012 lower `design_iiptm` and 0.21 higher `interaction_pae`.  The field is
dense enough that those deltas cost 12.3 and 19.8 rank points.

One BoltzGen observation of a *fixed* sequence lands 33.8 rank points (median
per-sequence standard deviation, 42 sequences with seven observations each)
from its own mean.  Re-observing one sequence seven times and scoring each
observation into that epoch's real field spans a median of 95 rank points.  The
lottery inside one observation is therefore wider than our entire gap to the
winner.

### The validator pays the mean, not our best draw

For the nine epochs where we hold eight observations of the sequence we
actually published, the validator's published score sits on our repeat mean
(bias -5.7 points, standard deviation 33.2, consistent with a single draw) and
48.6 points *worse* than our best internal draw.  Selecting or believing a best
draw buys about 49 points of illusion that the validator does not honour.

The same effect appears a second, independent way.  Comparing screen
observations against repeat observations of the same sequence looks like a
systematic shift (+0.0091 `design_iiptm`, p = 0.001) when the screen
observation is the draw that *selected* the sequence for replication; the shift
disappears entirely (-0.0027, p = 0.20) when the screen observation comes from a
different epoch and so played no part in the selection.  The apparent shift is
winner's curse, not a scorer artefact.

That second pairing also answers a live fidelity question: our screens run two
candidates per BoltzGen predict call while our repeats run one, and the
unconfounded comparison shows no measurable difference between those groupings
on any metric.  Predict-batch composition is still load-bearing in principle,
so it is now pinned rather than derived (below), but at the current shape it is
not costing us anything.

### Selection accuracy, not candidate count, is the binding constraint

Decomposing the one-shot spread of a screened pool (47.3 points observed)
against observation noise (33.8) leaves 33.1 points of true candidate quality.
A single observation therefore correlates 0.70 with the truth across the whole
pool, and considerably less among the near-identical siblings a shortlist is
actually chosen from.

A held-out test on real data confirms the consequence.  Choosing between just
three replicated finalists, selection on one observation recovers 31% of the
available discrimination; five observations recover 56-70%.

### The pipeline, not the scorer, is what limits an epoch today

Now measured directly by the instrumentation this change adds, on the live box:
330 seconds of scoring per 20-candidate tranche against 1.3 seconds of ranking,
basin shrinkage and mutation-database work.  Roughly 140 of those 330 seconds is
scorer startup and model load before the first fold; only about 190 seconds is
folding.

An earlier estimate in this document attributed 84 seconds to inter-tranche
work, read from artifact timestamps.  That gap was the next scorer process
starting up before it first touched its work directory -- it is inside the
scoring step, not beside it.  The correction matters: overlapping the
bookkeeping is worth about a second per tranche today, not 84, and the entire
fixed cost lives inside the scorer invocation where only a larger tranche can
amortise it.  The overlap still earns its place at the target scale, where
ranking grows with the accumulated observation count (1.0s at 80 screened,
1.3s at 140).

The consequence for the 12.2x scorer is sharp.  At the current 20-candidate
tranche it folds 20 candidates in about 29 seconds but still pays the model
load, so the loop returns roughly 2x end to end.  At a 200-candidate tranche the
same load is amortised over ten times as many candidates and the realised gain
is about 8x, or close to 10x if the replacement scorer's lighter 33-second
startup holds.

The live producer is configured for 320 candidates per epoch and actually
completes 122, with 2.4 replicated finalists rather than five.  It has never
reached its own configured budget.

Dropping a 12.2x scorer into that loop without changing anything else gives
about 2x end to end, because the fixed costs do not shrink.  The change that removes
them is tranches large enough to amortise the model load: the scorer needs at
least 15 sequences per shard, so at least 150 candidates per invocation on ten
GPUs.  Overlapping the CPU bookkeeping is a second-order saving that keeps
ranking off the critical path as it grows.

## Decision

Spend the throughput on independent observations of a large shortlist, not on
more one-shot candidates.  The operating point that maximises expected published
score subject to the real epoch clock (3,672 seconds after the commitment
reserve), the scorer's amortisation floor, and the measured noise is:

| | breadth | tranche | shortlist | observations each | total observations |
| --- | ---: | ---: | ---: | ---: | ---: |
| today, measured | 122 | 20 | 2.4 | 7 | 137 |
| **recommended** | **1,600** | **200** | **60** | **10** | **2,140** |

Expected published score improves from 117 to 80, which moves the median
leaderboard rank from 15 to 5 and raises the probability of a top-three finish
from 9% to 39%.  The model reproduces our own recent history to within 1.5
places (predicted median 15, actual 13.5 over sixteen published epochs), which
is the only honest claim available for its absolute level; its comparative
conclusions are far more robust than its calibration.

Two negative results matter as much as the positive one:

- **Pure breadth is worth nothing.** 3,200 one-shot observations score slightly
  *worse* than today's 137-observation two-stage policy, because selecting the
  best of many single draws mostly selects noise.  The previously committed
  2,400-candidate / 8-finalist profile is close to that failure mode.
- **A shortlist of five is the expensive part.** At the same budget, moving the
  shortlist from 5 to 24-60 is worth several rank points on its own, and is free.

Beyond roughly 2,000 observations per epoch the two-stage policy already
captures about 97% of what perfect selection over the same candidate pool could
achieve.  Further throughput should go into a better pool -- new basins, better
cores -- not into more selection.  Allocation gets us reliably into the top five;
winning also needs the pool's mean to improve.

### Not adopted: the 100x5 folding sampler

The 1.59x sampler setting carries a small genuine bias (+0.0053 `design_iiptm`,
-0.22 `min_design_to_target_pae`) with excellent rank preservation (rho 0.989
across twenty diverse designs).  Bias is harmless where only ranking matters and
fatal where absolute gates and the validator's own scale matter, so its only
admissible use is the screening pass, never the confirmation observations that
qualification and publication read.

Modelled in exactly that role it is worth +3.3 rank points if it adds no ranking
noise, +1.4 if it adds 20%, and -0.8 if it adds 50%.  The relevant quantity is
its rank correlation across a *narrow* sibling spread, which is not what rho
0.989 measured.  Not worth the protocol risk yet.  Before adopting it, measure
rho on about 40 shortlist-like candidates and make observations carry a scale
tag that strict ranking refuses to pool, exactly as the dual-sample protocol tag
prevents mixing selection rules.

## Implemented

- `nanobody_opt/backtest_budget_allocation.py`: measures observation noise, pool
  quality spread, and the wall-clock-feasible allocation; writes a hash-addressed
  result to `rd_results/`.
- `sequential_exact_race.py`: the per-tranche ranking, basin shrinkage and
  mutation-database update now overlap the next tranche's GPU work
  (`--pipeline-depth`, default 1).  Tranche composition depends only on the
  pending queue and per-basin screened counts, never on the ranking produced by
  the overlapped work, so a prefetched tranche selects exactly the candidates a
  serial run would have.  Replication drains the in-flight tranche first, so GPU
  work never contends with GPU work.
- `sequential_exact_race.py`: `--screen-predict-batch-size` pins how many
  candidates share one predict call.  Growing the tranche from 20 to 200 would
  otherwise have taken the derived batch from 2 to 20 and silently changed every
  score.
- `sequential_exact_race.py`: per-phase timings are appended to
  `timings.jsonl` and printed as `EXACT_TRANCHE_TIMING`, so the next budget comes
  from measurement rather than from a nominal speed multiplier.
- `score_boltzgen_waves.py` / `diffusion_sample_selection.py`: a wave is refused
  unless every retained structure reports the upstream diffusion-sample count
  (`--expected-diffusion-samples`, default 5).  The confidence metrics are a
  maximum over samples, so a scorer that quietly draws fewer is biased, not
  merely noisier, and would otherwise pass every existing check.
- `sequential_exact_race.py`: `--replication-rounds` re-observes the best
  `--finalists` screened candidates as whole rounds after screening, one scorer
  invocation per round.  The existing loop admits at most one finalist per
  tranche, so a shortlist of 60 was previously unreachable in an eight-tranche
  epoch regardless of configuration.  Default 0 keeps the incumbent/challenger
  behaviour.
- `ansible/vars/nanobody-high-throughput.yml`: rewritten to the operating point
  above.  Still dormant.

Note that the dual-pose defect -- the writer selecting `refold_cif` by
`0.8*iptm + 0.2*ptm` while analysis selects the reported metrics by
`0.8*design_to_target_iptm + 0.2*design_ptm` -- is already closed here by
`validator-dual-sample-v1` (commits 78a1906, f502454, 46aafd6).  Both poses and
both indices are retained per observation and `pose_stability` reads the
confidence pose by default.  The first wave under the deployed build recorded
complete dual-sample records for 20/20 structures with the two rules disagreeing
on 1 (5%; small-sample, consistent with a 15% rate).

## Activation gates

1. Full local suite and Ruff on touched files; `ansible-playbook --syntax-check`.
2. On the box, score one fixed cohort with the current and replacement scorers
   from the same Pixi environment and compare all ten metrics, both selected
   sample indices, and both retained structure hashes.
3. Run one epoch at `--pipeline-depth 1` with the current scorer and confirm from
   `timings.jsonl` that tranche composition and per-tranche candidate lists are
   unchanged, and that bookkeeping no longer sits on the critical path.
4. Confirm the predict-batch pin: a 200-candidate tranche must still pass
   `--batch-size 2` (or whichever grouping the replacement scorer's equivalence
   evidence used -- the two scorers may not mean the same thing by that flag, and
   this must be checked before a large tranche goes live).
5. Watch resident-worker RSS across the first long tranche; resident workers have
   never been exercised across hundreds of sequences.
6. Only then raise `nanobody_candidate_limit`, `nanobody_screen_tranche_size`,
   `nanobody_finalists` and `nanobody_qualification_observations` together.
   Raising breadth alone reproduces the failure this document measures.
