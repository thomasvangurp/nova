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
