# Preserve completed negative confirmations

Epoch 25969 exposed a bookkeeping error: one candidate's third CPU observation
failed the numerical aggregation cap, but its earlier single-pass tier B row
remained in the final native rows and prepared publication pool. The backend
labels both unavailable aggregation and a completed cap failure
`aggrescan_unavailable_or_failed`; the continuation rejection allowlist correctly
refused to treat that ambiguous label as a scientific failure, but the worker
had not resolved the ambiguity before writing its terminal outcome.

The worker now uses the existing finite-metric and provenance checks on the
original, completed authoritative CPU profile to identify a measured
`aggrescan_threshold` rejection. It persists that reason both immediately in the
confirmation checkpoint and in the final terminal outcome. Continuation then
removes the earlier row and replaces or withdraws the publication as appropriate.
Unavailable tools, timeouts, incomplete metrics, mismatched provenance and
non-authoritative backend results cannot trigger this invalidation. Audit journal
rows are never loaded as accepted measurements. The profiling adapter, scientific
code, thresholds, draw count and native checkpoint runtime identity are unchanged.

Regression coverage exercises the real gate and continuation/export boundary,
including a phase interruption after the completion callback, checkpoint-only
recovery, ordinary and interleaved scheduling, and retention of previous evidence
after inconclusive observations. The shipping suite requires these tests.

This error does not explain epoch 25970's lower yield: its five repeat evaluations
all passed. Its first-draw yield was 3/1,754, compared with 29/1,475 in epoch 25969.
The target and requirements digest were unchanged, while the epoch constraints
changed from IGHV3-23 / CDR3 length 11 to IGHV3-48 / length 17. The configured
reference bank contained 78 rows across only four other gene/length cells, so both
epochs used entirely unguided baseline proposals. These observations do not
separate the effects of gene, length and seed or establish validator-runtime parity.

Historical evidence is retained under
`nova-reports/pr100-69-70-funnel-audit-20261027/`, including first-draw counts,
runtime comparisons, bank coverage and the retained failed-confirmation hash.
The repaired code does not retroactively reclassify old audit journals or alter
past on-chain submissions.
