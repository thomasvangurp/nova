# Prospective multi-cohort selector/allocation proof

This protocol is frozen before any new development or fresh scorer draw.
Production remains unchanged.

## Inputs

- Target: P20809.
- Six independently generated production proposal cohorts: epochs 25247–25252.
- Each cohort contains the best 300 distinct candidates by its original live
  one-pass screen score. There are at least 900 unique sequences across cohorts.
- Selection fields: the sixteen intact public fields from epochs 25231–25246.
- Evaluation fields: the sixteen later intact public fields from epochs
  25253–25268.
- Scorer: the existing validator-dual-sample-v1 fast BoltzGen path on the
  preserved eight-RTX-4090 research instance.

## Prospective observations

Every unique candidate receives twelve new development observations. Four
frozen policies select ten-ticket fleets independently within each cohort:

- A: current internal-tournament selector and current 4/4/3 allocation.
- B: external-field per-entry selector and current 4/4/3 allocation.
- C: external-field selector and delayed 6/4 allocation, 300 -> 111.
- D: external-field selector after eight uniform repeats (diagnostic).

The fleet files and their SHA-256 hashes are written before fresh scoring. Only
the union of frozen fleets then receives 32 untouched observations. Development
observations can never be read by the fresh analyzer as outcomes.

## Outcomes and independence

Primary comparison: C versus A. Primary outcomes are wins per entry and top-10
per entry against the intact evaluation fields. Fleet-field win rate and all ten
raw validator metrics are secondary.

Nominal entry/field evaluations are correlated and are not treated as the sample
size. The evidence grid has six proposal cohorts by 32 independent scorer draws.
A deterministic two-way cluster bootstrap resamples both axes.

## Deployment gate

C is eligible only if every frozen gate passes:

1. aggregate wins/entry and top-10/entry deltas are positive;
2. both 95% two-way cluster-bootstrap lower bounds are positive;
3. both outcomes improve in at least five of six cohorts;
4. every leave-one-cohort-out and leave-one-draw-out comparison remains positive;
5. no draw supplies more than 25% and no cohort more than 35% of positive win gain;
6. the median fleet value does not regress on any protected confidence,
   interaction, or developability metric;
7. the A control is externally calibrated: top-10 within three percentage
   points of the current live 14.1%, and wins between 1.2% and 4.0% around the
   current live 2.38%.

Failure of any gate is NO-DEPLOY. Passing makes the change eligible only for a
shadow deployment; it does not itself modify production.
