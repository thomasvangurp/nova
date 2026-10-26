# Provenance

This package was built and validated on a vast.ai instance and lived only there,
outside version control, for four days. The instance was destroyed on
2026-09-12; the scorer survived only because it was pulled off the box minutes
beforehand. That is the reason it is now committed here.

`fast_score_batch.py` is byte-identical to the snapshot the live miner was
pinned to:

    sha256  530f5dcbc54d4a012571f4f9a7ddb6c016552291606652766d02c91643a7d531

`nanobody_opt/score_boltzgen_fast_adapter.py` refuses to run a scorer whose
digest differs from the one recorded in the deployment's
`nanobody_fast_scorer.json`, so this value is load-bearing -- changing the file
without updating the pin is a deliberate act, not an accident.

## What is here, and what was left behind

Committed: the scorer, its launcher, the vendored `boltzgen_src` it imports, the
reference designs used for `--synth-reference`, and the two documents recording
what was measured (`README.md`) and how equivalence was established
(`VALIDATION.md`).

Left behind: roughly 27 MB of validation output structures (`pp24_s`, `fin_s`,
`cap_structs`) and per-run logs. They are evidence of the runs, not inputs to
them, and the numbers derived from them are in `VALIDATION.md`. A local archive
of the full 415 MB working tree is at
`~/nova-nanobody-box-backup-20260912/fastscore-src.tgz` alongside the complete
pinned package; neither belongs in git.

## PR100 qualification candidate, 2026-10-26

The historical snapshot above is no longer the current source. An audit of
the slave registration found scorer digest
`0fccc5d630ef2ea98aad456e76ee65170383a6302f7eb92ac636565d2bfd98ef`,
with independent-replica support absent from the repository copy. Its
`resident.py` digest was
`5d722e836519326969e733b0a130f0fd0a0e4432ed2a3d060141c6468e1d95e0`
and wrapper digest was
`dd5f6225562191d8dccff5672887f7300b5cc320da94eb3ebd9d4c229d30f518`.
Those reviewed replica changes are retained here. The other 129 tracked
BoltzGen package files matched the registered runtime byte for byte.

The candidate also restores input regeneration on every ordinary-design chunk
and namespaces replica seeds by observation ID. A later job must not repeat
the same fixed `42 + ordinal` draws merely because its input order matches.
The candidate scorer pin is maintained in `ansible/nanobody.yml` and checked
against these bytes by CI. The complete package is included in release source
evidence; it requires separate runtime qualification and installation.

Passing unit tests or the old registered scorer's small new-target GPU probe
does not establish candidate throughput, validator equivalence on the new
target, or production approval. See the PR100 qualification workflow and the
release promotion contract. No active runtime pin is changed by this commit.
