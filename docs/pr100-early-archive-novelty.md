The opt-in worker setting `sequence_full_archive_novelty: true` runs the existing
qualified full-archive novelty child after cheap annotation, before native
admission. The controller supplies the sequence host's existing qualified
`novelty_runtime`. Its absence, a changed archive/runtime identity, or an
execution error fails the phase. The default remains off.

The scientific search implementation and runtime identity are unchanged:
`pr100_novelty_runtime.check` binds the original rows, archive bytes and actual
qualified runtime. It uses the full archive, its original IDs/order/batch
semantics, the original coarse retrieval defaults and the original duplicate
rules. Neither a parent-only projection nor a global CDR shortcut substitutes
for that search. The existing final novelty check remains mandatory. Early
results keep `novelty_checked: false` and claim no native qualification.

`early_archive_novelty` records the cheap-eligible input count/hash, kept/removed
counts, removed sequence hashes, archive/runtime identities and elapsed time.
Generation history, generated hashes and annotation reasons remain intact;
weighted native ordering applies to the surviving rows. These are sequence
rejections before any native reservation, not refunded scientific attempts.

The recorded epoch26038 attempt1 fixture contains 59 actual recombinations
independently reconstructed as native passers (11 tierA,48 tierB). All59 were
absent after the production final archive check. Local tests exercise this
recorded cohort through the bound child protocol with inert replies; they do
not claim a fresh scientific archive replay. The actual archive SHA is
`bf79571868a24046497e4b2a86c06ed19365efacb2fbd47708475d1aa6a61e54`.

The CPU-only replay command, to be scheduled by the coordinator under its
ordinary bounded CPU supervisor, is:

```text
python -B scripts/pr100_early_novelty_replay.py --cohort 26038-input.json --archive archive.csv --expected-rejected 26038-native-passers.json --output replay.json
```

Both cohort files and the unchanged archive are saved privately under the
report's `early-novelty-20261031/` directory. The59-row file can also be used as
`--cohort` for the first smaller replay. The harness invokes only the existing
qualified child, captures both streams, emits allowlisted counts/timing and
writes bound scientific results. It does not start a model, GPU task or native
profile. Run it from the exact reviewed source with the qualified runtime
available at its configured paths. Source file hashes are recorded in output.

Before enabling, measure the full2356-row check against the sequence phase's
existing wall allocation (25% of the remaining epoch window). Archive index
construction is repeated for the final check in this minimal implementation.
An early check against one snapshot never exempts publication from the final
archive-freshness requirements. No replay has been launched by this patch.
