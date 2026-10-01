# Historical calibration and literal ten-ticket A/B

Overall fail-closed decision: **NO-DEPLOY**.

## Test 1 — historical winner recovery and calibration

- Decision: **FAIL**
- Actual winner recovered first: 10/27 groups
- IPW predicted win rate: 2.1769%; full-history target: 2.2074%
- Predicted-rate 95% interval: 1.3340% to 3.1570%
- Winner-minus-control prediction: 5.3255%; 95% interval 1.6641% to 9.2278%
- Weighted Brier: 0.021501; weighted AUC: 0.6959
- Gates: `{"target_rate_inside_bootstrap_interval": true, "winner_control_separation_lower_bound_positive": true, "winner_recovery_20_of_27": false}`

## Test 2 — literal ten-ticket A/B

- Decision: **NO-DEPLOY**
- A wins/entry: 37/2560 = 1.4453%; B: 38/2560 = 1.4844%
- A top10/entry: 414/2560 = 16.1719%; B: 415/2560 = 16.2109%
- Favorable draw blocks: wins 6/16; top10 9/16
- Panel deltas: `{"panel_25261_25268": {"top10_delta": -2, "win_delta": -1}, "panel_25269_25276": {"top10_delta": 3, "win_delta": 2}}`
- Gates: `{"all_interface_liability_metrics_noninferior": false, "both_panels_top10_positive": false, "both_panels_win_positive": false, "every_lodo_top10_aggregate_favorable": false, "every_lodo_win_aggregate_favorable": false, "runtime": false, "top10_favorable_in_at_least_12_of_16_draws": false, "top10_not_two_draw_dominated": true, "wins_favorable_in_at_least_12_of_16_draws": false, "wins_not_two_draw_dominated": false}`

## Runtime and provenance

- Slowest 519-sequence draw: 441.217s
- Scoring sum: 6143.102s; estimated scoring cost: $7.066
- Preregistration SHA-256: `832882b5be45d7ae0823be1aa278ef9baeb5c2ec03ddf61bfc745d9f8c8bc0b8`
- All field comparisons use intact ten-metric vectors and whole-fleet joint dense ranks.
