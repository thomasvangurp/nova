# Mutation-ranker retrospective — 2026-09-10

## Decision

Do **not** use the v1 ExtraTrees model to allocate live exact-scoring slots.
The experiment is a useful negative control, but its training objective and
proposal-domain calibration are inadequate.  The existing conservative
elite-parent/local-mutation heuristic remains live and unchanged.

The next learned selector must predict parent-relative changes from measured
parent/child pairs drawn from the same proposal distribution.  Until it beats
the heuristic on parent-grouped chronological validation, it may reject only a
clearly bad tail and may not control the top-160 acquisition set.

## Reproducible protocol

- Code: commit `ba8477f` plus the Pixi task correction in `7831e4b`.
- Runtime: locked `metanano` Pixi environment; scikit-learn 1.9.0.
- Public API fields: 240 completed epochs, 24074–25034.
- Warm-up: 20 epochs used only to make preceding parent banks.
- Training/model-selection period: 200 epochs, 8,344 length-127 examples.
- Sealed holdout: the final 20 epochs, 25015–25034.
- Candidate budget: 160 from a mean organic pool of 28,053 sequences.
- Allocation: up to 32 parent-floor slots, 120 global model slots, and 8
  deterministic exploration slots.
- Every holdout proposal used parents and donors from strictly earlier epochs.
- Held-out field sequences were appended only to a labelled oracle retrieval
  pool after organic proposal generation.  They were never training data or
  counted as organic discoveries.

Complete immutable outputs are mirrored at
`artifacts/rd/mutation_ranker_v1_20260910/` (git-ignored because the fitted
model is 127 MB).  `manifest.json` contains per-file SHA-256 digests; all 12
entries were verified after transfer.  The directory includes the exact API
field snapshot, split and configuration, model-selection results, fitted
model, every held-out field prediction, per-epoch placements, all selected
top-160 rows, CDR1 context predictions, summary, and manifest.

## Result

ExtraTrees beat RandomForest on the training-period internal validation, but
neither showed useful validation signal.  ExtraTrees' internal median field
Spearman was -0.017 and its winner top-five rate was 20%.

On the sealed 20-epoch holdout, ExtraTrees achieved:

- median within-field Spearman: 0.172;
- median winner rank among the real field: 16.5;
- winner top-one/top-five/top-ten rates: 5% / 20% / 40%;
- winner support within the six-edit donor alphabet: 18/20;
- exact organic winner generation: 4/20;
- organic winner selected inside 160: 0/20;
- oracle-retrieval winner selected inside 160: 0/20;
- median oracle winner rank among ~28k proposals: 14,126.5.

The oracle result separates ranking from discovery.  Even when each unseen
winner was added to the already-frozen pool solely for evaluation, the model
ranked none inside the exact budget.  This is an abject ranker failure, not a
minor top-k calibration miss.

## CDR1 context diagnostic

The public API supplied eight exact CDR1 single-substitution cases in seven
holdout epochs.  Repeated labels were:

- `M29L`: API scores 81, 53, and 54.  Frozen-model ordering Spearman 0.5.  It
  predicted epoch 25022 best but predicted the winning epoch-25026 context
  worst of the three.
- `Q31R`: API scores 151, 100, and 94.  Frozen-model ordering Spearman -1.0;
  the model exactly reversed their quality ordering.

The feature representation can distinguish parent contexts, but the learned
function did not learn them reliably.

## Why v1 failed

The v1 target was validator-aligned but not decision-aligned.  It predicted ten
absolute within-epoch metric percentiles from a sequence and a nearest earlier
elite parent.  That is not the same as predicting whether a proposed mutation
improves its actual parent:

1. Public submissions do not declare ancestry.  The nearest elite parent is an
   inferred attribution and may be false.
2. Each public label is one stochastic BoltzGen observation, so sequence-level
   supervision contains large structural-sampling noise.
3. The broad recombinant pool is outside the public-field training
   distribution.  Trees interpolate bounded partitions and assigned thousands
   of artificial mosaics implausibly better scores than real held-out winners.
4. Multi-output absolute prediction diluted the acquisition objective: top-k
   parent-relative improvement.

## Next experiment

1. Reconstruct the proposal-to-parent mapping for our locally exact-scored
   neighborhood batches; unlike public data, these have known ancestry.
2. Exact-score the elite parents under the same locked pipeline and repeat
   parent and child observations so the target is a distribution of metric and
   category deltas, not a difference between unrelated one-shot fields.
3. Train a group-held-out parent-relative model to estimate improvement
   probability and uncertainty for each mutation/haplotype.
4. Evaluate against the existing heuristic on chronological parent-family
   folds.  Record top-tail recall and regret at 160, plus calibration under
   generated-distribution shift.
5. Require training-period acceptance gates before any holdout or live use:
   median field Spearman >= 0.15, winner top-five rate >= 35%, and median winner
   percentile <= 25%.  A failed gate leaves the heuristic in control.

