# Full-cohort held-out shortlist recall

## Verdict

The added Hamming-2 proposal capacity is useful: closure candidates supply 9/10 held-out consensus top-10 sequences, versus 1/10 from the current arm. The proposed 96/2800 retention fraction is not validated: its K=30 analogue recalls 5/10, versus 8/10 for current-fraction K=95. Although K=30 has 12/32 fleet wins versus 10/32, that reversal is confined to scorer-draw variation (effective n=4) and is not stable evidence for pruning. Keep the wider retention fraction and spend extra GPU on repeated complete-metric confirmation: K=95 top-10 recall rises from 5/10 at one draw to 8/10 at four.

## Experiment

- Four existing screening draws and four newly generated untouched draws, all over the same 888-sequence cohort.
- Selection excludes 30 known-public controls, leaving 858 eligible novel candidates; controls were still scored in every draw.
- Every comparison uses the complete ten-metric dense-rank sum. Fleet yield preserves intact sequence/pass/public-field scenarios.
- Fresh-draw candidate-rank correlations span 0.442–0.508; pairwise top-10 overlap spans 0–1/10.
- Absolute-K comparisons use 300 and 96 retained. Fraction-normalized comparisons use 95/888 for current 300/2800 (10.71%) and 30/888 for proposed 96/2800 (3.43%).

## Recall and yield on untouched draws 5–8

| screen draws | kept | top-1 consensus | top-10 consensus | draw-wise top-1 recall | draw-wise top-10 recall | coverage regret | champion regret | fleet wins | individual wins/entry | fleet top-10/entry |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 24 | no | 2/10 | 0/4 | 5.0% | 38.0 | 1019.0 | 8/32 | 8/320 | 80/320 |
| 1 | 30 | no | 2/10 | 0/4 | 7.5% | 38.0 | 1019.0 | 8/32 | 8/320 | 80/320 |
| 1 | 48 | no | 3/10 | 0/4 | 12.5% | 38.0 | 1019.0 | 8/32 | 8/320 | 80/320 |
| 1 | 95 | no | 5/10 | 1/4 | 30.0% | 28.0 | 1019.0 | 8/32 | 8/320 | 80/320 |
| 1 | 96 | no | 5/10 | 1/4 | 30.0% | 28.0 | 1019.0 | 8/32 | 8/320 | 80/320 |
| 1 | 192 | no | 7/10 | 2/4 | 45.0% | 28.0 | 1019.0 | 8/32 | 8/320 | 80/320 |
| 1 | 300 | yes | 9/10 | 2/4 | 60.0% | 0.0 | 1019.0 | 8/32 | 8/320 | 80/320 |
| 2 | 24 | no | 2/10 | 0/4 | 10.0% | 38.0 | 779.5 | 7/32 | 8/320 | 89/320 |
| 2 | 30 | no | 2/10 | 0/4 | 10.0% | 38.0 | 779.5 | 6/32 | 7/320 | 89/320 |
| 2 | 48 | no | 4/10 | 0/4 | 12.5% | 38.0 | 779.5 | 6/32 | 7/320 | 75/320 |
| 2 | 95 | no | 6/10 | 1/4 | 25.0% | 28.0 | 779.5 | 7/32 | 7/320 | 77/320 |
| 2 | 96 | no | 6/10 | 1/4 | 25.0% | 28.0 | 779.5 | 7/32 | 7/320 | 77/320 |
| 2 | 192 | yes | 9/10 | 3/4 | 50.0% | 0.0 | 779.5 | 7/32 | 7/320 | 77/320 |
| 2 | 300 | yes | 9/10 | 3/4 | 70.0% | 0.0 | 779.5 | 7/32 | 7/320 | 77/320 |
| 4 | 24 | no | 4/10 | 1/4 | 17.5% | 28.0 | 309.2 | 9/32 | 9/320 | 81/320 |
| 4 | 30 | yes | 5/10 | 1/4 | 20.0% | 0.0 | 309.2 | 12/32 | 19/320 | 84/320 |
| 4 | 48 | yes | 5/10 | 1/4 | 27.5% | 0.0 | 309.2 | 10/32 | 17/320 | 75/320 |
| 4 | 95 | yes | 8/10 | 2/4 | 32.5% | 0.0 | 309.2 | 10/32 | 17/320 | 69/320 |
| 4 | 96 | yes | 8/10 | 2/4 | 32.5% | 0.0 | 309.2 | 10/32 | 17/320 | 69/320 |
| 4 | 192 | yes | 9/10 | 2/4 | 55.0% | 0.0 | 309.2 | 11/32 | 17/320 | 67/320 |
| 4 | 300 | yes | 9/10 | 2/4 | 65.0% | 0.0 | 309.2 | 11/32 | 17/320 | 63/320 |

## Production decision at four screening draws

- Literal 300 retained: top-1 recalled, top-10 9/10, coverage regret 0.0, fleet yield 11/32 (per-draw range 0%–100%).
- Literal 96 retained: top-1 recalled, top-10 8/10, coverage regret 0.0, fleet yield 10/32 (per-draw range 0%–100%).
- Fraction-normalized current (95/888, matching 300/2800): top-10 8/10, coverage regret 0.0, fleet yield 10/32.
- Fraction-normalized proposed (30/888, matching 96/2800): top-10 5/10, coverage regret 0.0, fleet yield 12/32.
- Paired fleet scenarios, absolute K: current-only wins 1, proposed-only wins 0, both 10, neither 21.
- Variation is grouped by the four scorer draws (effective n=4); the 32 draw×field scenarios are repeated measures, not 32 independent observations.
- Literal 300 per-draw fleet win rates: 100.0%, 25.0%, 12.5%, 0.0%; individual win/entry: 17.5%, 2.5%, 1.2%, 0.0%.
- Literal 96 per-draw fleet win rates: 100.0%, 12.5%, 12.5%, 0.0%; individual win/entry: 17.5%, 2.5%, 1.2%, 0.0%.

### Effective-n=4 scorer-draw variation at four screening draws

| kept | fleet win rate by scorer draw 5/6/7/8 | individual win/entry by scorer draw 5/6/7/8 |
|---:|---|---|
| 30 | 100.0% / 12.5% / 37.5% / 0.0% | 17.5% / 2.5% / 3.8% / 0.0% |
| 95 | 100.0% / 12.5% / 12.5% / 0.0% | 17.5% / 2.5% / 1.2% / 0.0% |
| 96 | 100.0% / 12.5% / 12.5% / 0.0% | 17.5% / 2.5% / 1.2% / 0.0% |
| 192 | 100.0% / 25.0% / 12.5% / 0.0% | 17.5% / 2.5% / 1.2% / 0.0% |
| 300 | 100.0% / 25.0% / 12.5% / 0.0% | 17.5% / 2.5% / 1.2% / 0.0% |

## Arm outcomes on untouched draws

| arm | eligible candidates | best mean rank sum | consensus top-10 members |
|---|---:|---:|---:|
| current_production | 288 | 970.2 | 1 |
| ours_hamming2 | 282 | 908.8 | 6 |
| rival_hamming2 | 288 | 946.8 | 3 |

## Basin outcomes on untouched draws

| basin | eligible candidates | best mean rank sum | consensus top-10 members |
|---|---:|---:|---:|
| epoch-25257-basin | 62 | 908.8 | 1 |
| epoch-25254-basin | 35 | 936.8 | 3 |
| epoch-25252-basin | 57 | 946.8 | 3 |
| epoch-25248-basin | 55 | 1092.2 | 1 |
| epoch-25258-basin | 73 | 1151.5 | 1 |
| epoch-25256-basin | 41 | 1227.2 | 0 |
| epoch-25249-basin | 43 | 1250.5 | 0 |
| epoch-25253-basin | 55 | 1382.2 | 0 |
| epoch-25259-basin | 36 | 1382.5 | 0 |
| epoch-25250-basin | 41 | 1450.5 | 0 |
| epoch-25255-basin | 33 | 1556.0 | 0 |
| epoch-25260-basin | 39 | 1560.8 | 0 |

## Artifacts

- `analysis.json`: every depth/size result, retained order, selected fleets, exact scenario counts, and oracle identities.
- `run_metadata.json`: scorer/input hashes and exact timing metadata.
- `heldout_05`–`heldout_08`: four complete fresh full-cohort observations.
