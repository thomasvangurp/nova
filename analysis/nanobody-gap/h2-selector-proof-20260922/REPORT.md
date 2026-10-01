# Offline H2 + repaired-selector proof

## Decision

**NO-DEPLOY**

The eight scorer draws are the independent blocks. The 64 draw×field cells per arm are repeated measurements, not 64 independent observations.

## Arm outcomes

| arm | individual wins/entries | top-10/entries | whole-fleet wins/fields |
|---|---:|---:|---:|
| A | 0/512 | 43/512 | 0/64 |
| B | 13/512 | 80/512 | 13/64 |
| C | 3/512 | 38/512 | 2/64 |

## Preregistered combined-treatment gates

- individual_wins_per_entry: FAIL
- top10_per_entry: FAIL
- leave_one_draw_out: FAIL
- whole_fleet_probability: PASS
- hard_metrics: FAIL
- runtime: PASS

## Effective-n=8 block results

| draw | A individual | C individual | A top-10 | C top-10 | A fleet | C fleet |
|---:|---:|---:|---:|---:|---:|---:|
| 9 | 0.0% | 0.0% | 3.1% | 10.9% | 0.0% | 0.0% |
| 10 | 0.0% | 3.1% | 18.8% | 14.1% | 0.0% | 12.5% |
| 11 | 0.0% | 0.0% | 9.4% | 0.0% | 0.0% | 0.0% |
| 12 | 0.0% | 0.0% | 6.2% | 10.9% | 0.0% | 0.0% |
| 13 | 0.0% | 0.0% | 18.8% | 1.6% | 0.0% | 0.0% |
| 14 | 0.0% | 0.0% | 0.0% | 3.1% | 0.0% | 0.0% |
| 15 | 0.0% | 0.0% | 3.1% | 1.6% | 0.0% | 0.0% |
| 16 | 0.0% | 1.6% | 7.8% | 17.2% | 0.0% | 12.5% |

## Runtime

Four-GPU guarded full-treatment projection: 3268.8s; current configured tournament: 3824.1s; deadline: 3660.0s.

The broad 2,800-candidate one-shot screen is unchanged and cancels in the A-versus-C runtime comparison.
