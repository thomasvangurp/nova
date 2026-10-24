# Multipass regression and selection repair — 2026-10-24

Lower scores are better. The Pi comparison uses each epoch's preceding ten
validator fields and independent stored replay folds; numbers across epochs
have different fields. Compare our fleet and the rival on the same epoch.

| Epoch | Our replay mean | 5Gh replay mean | Our selected local mean |
| --- | ---: | ---: | ---: |
| 25901 | 1096.4 | 1035.2 | 1116.9 |
| 25902 | 1311.2 | 1086.4 | 1322.5 |
| 25903 | 1892.4 | 1121.9 | 1843.6 |
| 25904 | 2340.0 | 1178.1 | 2304.9 |
| 25905 | 1095.5 | 1054.1 | 1100.1 |
| 25906 | 1764.3 | 1106.1 | 1718.3 |
| 25907 | 1590.6 | 1210.7 | 1540.7 |
| 25908 | 1803.3 | 1201.0 | 1757.7 |

The local columns come from the Pi's archived submission artifacts. These can
be an intermediate fleet revision: epochs 25901–25903 have a one-entry cohort
difference from the final replay. This cannot explain the large regression.
Epoch 25904 has the same thirteen entries and 20–283 local observations each.
The field had 433 rows in 25901 and 437 in 25904. We selected a weak pool, rather
than a strong pool unexpectedly collapsing on replay.

## Confirmed failure paths

1. Previously scored candidates were excluded from the new-evaluation budget,
   correctly, but the confirmation ranker required a current-epoch raw file.
   The publisher required twenty observations. Candidates with 1–19 stored
   folds could therefore enter neither path. In 25901, 3,856 current-novelty
   passing proposals were in this gap, including 1,305 descendants of selected
   5Gh parents. These counts do not imply they all pass full Layer 2.
2. Already scored single-edit proposals consumed generation limits before
   filtering. In 25901 only 104 new measured-parent proposals remained; 4,696
   of the 4,800 admitted entries were historical fallback proposals. Parents
   were present (54 of 155 selected parents belonged to 5Gh); downstream
   generation and selection discarded their usefulness.
3. Candidate assembly stamped its reusable marker before child generation and
   merges completed. Cancellation could make an incomplete singles-only CSV
   reusable. The code establishes this failure mode; the retained logs do not
   establish which interruption first produced the singles-only artifact.
4. For unregistered lengths, the pinned fallback design worker deleted each
   chunk's inputs and wrote replacement YAML only while building its initial
   configuration. Later chunks raised `No designs found` in
   `FromGeneratedDataModule`. Epochs 25903 and 25904 exposed zero completed
   tranches despite reserving 2,746 and 2,525 entries respectively. Epoch 25908
   exposed zero although four interrupted tranches contained 1,106 completed
   measurements. The all-or-nothing aggregate hid those successful waves.

## Release invariants

- The 4,800 cap charges NEW discovery dispatches. Cached comparison and
  confirmation have separate accounting. Failed dispatch reservations remain
  charged so a restart cannot silently exceed the cap.
- Precheck the stored own-fold bank on current novelty and IgBLAST, with
  epoch/target/archive/config/gate hashes. Rank eligible cached and fresh folds
  on the same current field. Candidates below twenty can earn confirmation;
  candidates already at twenty compete through the publisher without another
  tournament. Full Layer 2 and publication depth remain mandatory.
- Cached prechecks run alongside fresh scoring with the owning epoch's live
  deadline. New-epoch work never joins an old precheck thread.
- Exclude evaluated proposals before the generation limit. Mark candidate
  assembly reusable only after every merge, binding the marker to its hash.
- Every ordinary-design chunk writes its own inputs. Preserve the registered
  worker argument schema and diffusion-sample contract when repinning.
- Each successful wave creates an immutable partial aggregate. Choose only the
  latest checkpoint, or the full aggregate when present. Both retain the same
  observation ID so ingestion cannot manufacture extra independent passes.
- Run the committed shipping suite, including the actual pinned worker's
  repeated-chunk input test and interrupted-wave ingestion test, before deploy.
  Record installed hashes and live verification separately from a CI pass.

The target of 900 is an objective, not a measured guarantee or a license to
weaken eligibility. A weak confirmed pool must be reported as such.
