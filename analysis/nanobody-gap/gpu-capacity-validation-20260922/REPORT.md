# GPU capacity and proposal-quality validation

## Verdict

Extra capacity produced no challenger that cleared the frozen 5/8 gate. The matched rival-basin gate failed at 4/12 basin wins. The literal one-step stop is 1 draw(s), but it is invalidated by the later curve: 8 draws has the lowest observed regret and 12 worsens again. No stable repeat optimum is established.

## Frozen experiment

- Equal discovery budget: 288 candidates per arm; four complete ten-metric draws.
- The confirmation shortlist contains 24 novel proposals per arm plus 24 controls.
- Selection uses only confirmation passes 1–12; every performance number below uses untouched passes 13–20.
- Fleet selection uses 22 intact public fields (25231–25252); fleet evaluation uses eight untouched later fields (25253–25260). Metrics are never resampled independently.

## Repeat-depth curve

| selection draws | mean selection regret | current fleet P(any win) | ours-H2 fleet P(any win) | rival-H2 fleet P(any win) | current champion wins | ours champion wins | rival champion wins |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 53.5 | 18.8% | 26.6% | 9.4% | 3/8 | 0/8 | 5/8 |
| 2 | 66.7 | 20.3% | 18.8% | 25.0% | 2/8 | 1/8 | 5/8 |
| 4 | 42.1 | 32.8% | 37.5% | 31.2% | 1/8 | 3/8 | 5/8 |
| 8 | 31.0 | 46.9% | 15.6% | 31.2% | 3/8 | 1/8 | 5/8 |
| 12 | 50.1 | 45.3% | 12.5% | 32.8% | 6/8 | 3/8 | 2/8 |

## Scored-proposal width (secondary 2-draw/2-draw split)

| arm | 24-proposal fleet P(any win) | widest fleet P(any win) | fleet change | champion rank-sum change |
|---|---:|---:|---:|---:|
| current_production | 50.0% | 0.0% (288 proposals) | -50.0% | -147.5 |
| ours_hamming2 | 12.5% | 12.5% (282 proposals) | +0.0% | -659.0 |
| rival_hamming2 | 0.0% | 0.0% (288 proposals) | +0.0% | -702.0 |

The preregistered one-step stopping rule literally chooses **1 draw(s)**, but a later depth beats that stop: **8 draws** has the lowest observed held-out regret. Because the curve is non-monotone and worsens again at 12, this experiment does not establish a stable optimum.

## Final 12-draw selection on eight untouched draws

- ours_hamming2 champion versus current-production champion: 2 wins, 1 ties, 5 losses; 5/8 gate failed. Mean design_iiptm delta +0.0106, favorable on 5/8 draws.
- rival_hamming2 champion versus current-production champion: 0 wins, 1 ties, 7 losses; 5/8 gate failed. Mean design_iiptm delta +0.0217, favorable on 6/8 draws.

## Matched basin result

Rival closure won/tied/lost 4/1/7 of 12 matched basins after selection on 12 draws and evaluation on eight untouched draws. The preregistered 8/12 basin gate failed.

## Artifacts

- `analysis.json` contains every selected sequence, complete paired metric delta, basin comparison, capacity curve, and exact scenario count.
- `shortlist_manifest.json` records the discovery-only shortlist decision.
- `manifest.json` freezes arm membership, basin provenance, public-novelty status, and input hashes.
