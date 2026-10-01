# Prospective multi-cohort selector/allocation proof

Decision: **NO_DEPLOY**

Production was not changed.

| arm | wins / entry | top-10 / entry | fleet-field win rate |
|---|---:|---:|---:|
| A_live_selector_live_allocation | 1.673% | 14.342% | 15.039% |
| B_field_selector_live_allocation | 1.650% | 14.066% | 14.616% |
| C_field_selector_delayed_6_4 | 1.748% | 13.890% | 15.495% |
| D_field_selector_uniform_8 | 1.842% | 13.760% | 15.853% |

## Primary C versus A

- wins/entry delta: +0.075%; 95% two-way cluster bootstrap -0.788% to +0.980%
- top-10/entry delta: -0.452%; 95% two-way cluster bootstrap -3.773% to +3.132%
- favorable cohorts: wins 4/6; top-10 3/6

## Gates

- PASS — win_delta_positive
- FAIL — top10_delta_positive
- FAIL — win_bootstrap_lower_positive
- FAIL — top10_bootstrap_lower_positive
- FAIL — win_favorable_at_least_5_of_6_cohorts
- FAIL — top10_favorable_at_least_5_of_6_cohorts
- FAIL — all_leave_one_cohort_out_positive
- FAIL — all_leave_one_draw_out_positive
- PASS — win_not_concentrated_by_draw
- FAIL — win_not_concentrated_by_cohort
- FAIL — all_protected_metrics_nonregressing
- PASS — control_top10_calibrated_to_live
- PASS — control_win_rate_plausible

The 30,720 nominal entry/field observations are correlated. The inferential unit is the 6-by-32 cohort/draw grid, and the bootstrap resamples both axes.
