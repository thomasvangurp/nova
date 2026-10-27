# Missing-cell donor experiment

`worker.missing_cell_donor_fallback` is a boolean, default **false**. This patch
does not activate it or claim a yield improvement. It is based on the archive
transport release `98ef1cb` plus terminal invalidation repair
`d9b13d773651066b396fcd52835b9f48456b6408`.

The read-only bank inventory found 78 rows at
`/root/pr100-qualification-20261026/native-pass-reference-groups-v2/input.json`,
SHA256 `58525a3a3d21b25ccc446efdf5ce437929895d53286e851c5dfc3c34f4ea4f36`:

| Cell | Rows |
|---|---:|
| IGHV3-23:16 | 6 |
| IGHV3-23:18 | 68 |
| IGHV3-23:5 | 2 |
| IGHV3-7:16 | 2 |

The bank has no exact reference for the observed cells 25969 (IGHV3-23:11),
25970 (IGHV3-48:17), or 25971 (IGHV3-43:21). Its existing provenance labels
are `full_native_census` (73) and `conditional_native_pilot` (5). These are
historical donor inputs, not current-cell acceptance evidence.

With the opt-in enabled and no exact cell, donor choice first ranks same-V
references by length distance, then other-V references by length distance.
Epoch/sequence-hash order breaks ties. At most `reference_slots` donors are
selected (existing default 16). No binding score enters donor selection.
Exact-cell reference selection, direct reference candidates and seeded
proposal order retain their previous behavior.

The existing first `floor(proposals / 2)` proposal positions receive donor
CDR3 inputs; the remaining positions stay unseeded controls. The existing
generator constructs the current requested germline and loop length. Donor
rows are not emitted as missing-cell candidates, and a generated identity
equal to any bank parent is dropped without retrying or spending extra
ordinals. Annotation, three-draw native confirmation, archive novelty, target
scoring and publication eligibility continue unchanged. All generated
identities still pass the existing exclusions and shared fresh-admission cap.

The sequence artifact records bank hash/cell coverage, selected donor hashes,
selection strata, length distances, generated source counts and eligible
source counts. Each fresh candidate retains `reference_hash` and
`reference_stratum`; the allowlisted native observation journal preserves
both for subsequent per-source outcome comparisons. An unseeded proposal
has a null reference hash and `unseeded_control` stratum.

A controlled live test can enable this one worker boolean while retaining
the existing proposals, fresh limit, repeat count, resources and exclusions.
Its receipt must report generated, annotation-eligible, native-admitted,
three-draw-passed, novel and scored counts separately for donor and control
strata. Read those counts as measured outcomes; no live experiment was run
for this patch. Disabling the boolean restores the prior missing-cell path.

Tests use synthetic identities with the measured bank's cell counts. They
cover the three missing cells, score-independent deterministic choice,
exact-cell preservation, 50% controls, uniqueness, parent-reuse rejection,
unchanged annotation gates, resume exclusions and the shared fresh cap.
They execute no native profiling, model inference or GPU work.
