# Parent merit, epoch deadlines, and release evidence — 2026-10-22

Parents are ranked by the mean of independent pass ranks against the same current ten-epoch field, with at least ten raw folds. Historical scores, owner quotas, family reservations, and rank-of-mean-metrics must not replace this ordering. Repair-only parents keep their validity routing and capacity cap. Child proposals visit the measured parents in merit order before spending another edit on the same parent.

The full live audit for epoch 25872 compared 1,456 measured, gated parents and selected 155. The cutoff was 934.25, best 591.35, and top-twenty mean 706.2390625, against epochs 25860–25869 (371 rows). Five of competitor 5GhREcsK's eight epoch-25868 parents qualified at positions 21, 28, 34, 54, and 122. The other three scored 963.55, 1025.35, and 1099.55 on that same field and fell below the cutoff. This supersedes the earlier limited-bank comparison. Parent quality is evidence for generation, not a guarantee of child quality.

From epoch 25872, generate at most 8,000 cheap proposals, pass fresh eligibility checks, and admit at most 4,800 to expensive screening. Confirm at most 64 finalists with at least twenty independent observations before publication. Coverage counts successful raw measurements, not attempts or aggregate rows. Keep slot purchases value-priced with a floor of five and a ceiling of thirteen.

## Epoch 25871 incident

All 10,980 eligible candidates were screened. Twenty final candidates completed twenty repeat rounds plus their initial fold. The saved tournament's top-eleven measured mean was 743.3983; top-eight mean was 721.2679. These were not submitted results. The publisher remained occupied by epoch 25870's strict rank for over 100 minutes; the submitter committed eleven stale epoch-25870 nanobodies during epoch 25871. Never label an archive directory's contents as a new result without checking its embedded epoch.

The strict-pool fallback expanded a thin shortlist back to the full 11,012-candidate scope. Structure metadata was loaded before applying the observation floor, and child commands had no live epoch deadline. A pre-launch time estimate and a post-completion epoch check do not prevent old work from blocking the next epoch.

The correction bounds the strict pool even when thin, filters evidence depth before structure-history loading, and keeps the twenty-fold floor hard. Producer and publisher commands run in private process groups and check the live chain clock every second. Epoch rollover, stale clock, or the handoff reserve terminates the command tree. Publisher commands also have a five-minute wall-clock cap. Optional strict publisher upgrades remain disabled in the deployed launch contract; normal biological and novelty gates remain required.

The slave worker must see producer confirmation rankings without waiting for the publisher. New rival/replay batches yield during the critical submission window until confirmation and current publication exist. Already completed measurements remain reusable through the observation store.

Recovery found twenty completed epoch-25871 repeat files missing their sealed import records. The normal importer recovered 720 independent observation records and sealed all twenty runs. Recovery must retain at most 64 confirmed, policy-current finalists from the immediately preceding epoch, import their completed repeat evidence, and rank them against the current field. Publication still runs the current novelty, Layer-2 and independent-observation gates. Do not rerun historical GPU work or scan old screening directories on the publisher path.

## Release contract

Run `scripts/shipping_checks.py` from a clean committed tree. It stages committed source, compiles it, executes the explicit shipping regression suites, rejects missing/zero/skipped/failed tests, and packages those exact bytes with commit, lockfile hash, file hashes, and passing test evidence. CI runs this same command. Deployment depends on that CI job and downloads its package; it verifies staged and installed hashes before activation. Both producer trees must not be confused: production invokes `nanobody_opt_v2.run_live_producer` and ships `nanobody_opt_v2`.

A local test pass is not a GitHub CI pass. Normally deploy after GitHub CI succeeds for the exact revision. For this incident the user explicitly authorized the verified local shipping run; preserve that distinction and the passing receipt, and verify each installed file against it. Do not copy an unrelated build stamp or infer production health from a running tmux session. Verify the artifact's epoch and age, the confirmation depth, and the submitter's actual commit results.

Commit `f29c07b772cd3c980c55ccaf97662afb14ef7dd7` passed all 371 shipping tests locally and in [GitHub CI](https://github.com/thomasvangurp/nova/actions/runs/39804000282). Eleven incident runtime files were deployed from that verified artifact and checked by hash; producer, publisher and slave top-up restarted successfully. The receipt is `/root/artifacts/incident_release_20261022.json`. This was a scoped incident deployment, not a claim that every runtime file matches a full release.

The HTTPS credential could not push workflow changes. Authenticated SSH successfully pushed the same commit, including workflows; the local `payout-pr` remote now uses `git@github.com:thomasvangurp/nova.git` as its persistent push URL. This fixes workflow pushes without copying credentials or granting a broader token scope.

The slave top-up was also blocked by a stale reviewed daemon hash. Review of both live sources confirmed the master's added short-shard retry sends the same scorer arguments and merges each shard once; the slave's replicate-only plumbing remains disabled for these unique-sequence rounds. Update only the exact reviewed hash pair, retain rejection of any further drift, and verify live preflight after testing. Do not overwrite the master's pinned scoring implementation to make hashes match.

Regression coverage includes current-field parent merit, proposal/screen budgets, deferred TNP, value pricing, commit side effects, validator round trips, actual v2 imports, release tampering, and a real blocked-process rollover test that proves child termination and immediate next-epoch progress.
