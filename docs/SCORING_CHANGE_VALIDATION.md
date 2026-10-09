# Rule: validate scoring changes against real outcomes before deploying

**Any change to a scoring, ranking or selection formula that affects which
candidates we submit live must be run through `tools/validate_scoring_change.py`
(or an equivalent empirical check) against real historical validator outcomes
*before* it is deployed to a live box.** "It looks right on inspection" is not
evidence. Only exit code `0` (PROPOSED IS BETTER) supports deploying.

This covers, at least: the score used to pick children/submissions
(`children_submit`), parent/seed rankers (`select_elite_parents`,
`elite_parents_from_db`, `local_neighborhood` ordering), the multipass / HF-field
aggregation (`multipass_all_rivals`, `hf_field_refresh`), and any proxy that
replaces one of them. A tie-break-only change is still a ranking change.

## Why: two cases from 2026-10-09, one for each side

**What must never happen again.** A reverse-engineered "corrected" validator
formula (rank each of the three metric categories, then sum the category ranks)
was wired straight into live candidate selection because it looked plausible.
Nobody checked it against our published results first. It was a regression: over
our last 10 epochs (126 submissions, 25604-25613) the existing sum-of-ten-ranks
score predicts the published `final_score` at rho = 1.000. The "corrected"
formula reached only rho = 0.918. It cost submission quality (epochs 25613/25614)
before it was caught and reverted (`945e70f`). The tool flags it in seconds:

```
python tools/validate_scoring_change.py --case formula-20261009
  incumbent builtin:sum10 1.000   proposed builtin:category3 0.918
  diff 95% CI (-0.110, -0.058)  ->  VERDICT: PROPOSED IS WORSE (exit 2)
```

**What should always happen.** Ranking parents by the 20x multipass score instead
of the single-pass published score looked like an obvious improvement, since
averaging cuts noise. It was tested first. The outcome was how the validator
scored each parent's published relatives. The test covered 1-3 edit radii, all
relatives versus future-only, and three score bands. All correlations fell between
-0.02 and 0.15. Every confidence interval on the difference contained zero, and
the sign flipped between slices. The proposal was not deployed. It was not
"neutral, so harmless"; there was simply no evidence it helped.

```
python tools/validate_scoring_change.py --case parent20x-20261009
  18 slices, every diff CI contains zero, sign flips  ->  NO EVIDENCE OF IMPROVEMENT (exit 1)
```

Both cases are built into the tool and print `SELF-CHECK PASSED` when the tool
reproduces them. Run one first if you have changed the tool or suspect the API.

## How to run it

```
# Does my new formula predict our published scores better than sum-of-ten?
python tools/validate_scoring_change.py --last 10 \
    --incumbent builtin:sum10 --proposed py:my_module:my_formula

# Does a precomputed per-sequence score rank parents better?
python tools/validate_scoring_change.py --outcome neighbors --max-edits 1,2,3 \
    --relatives both --epochs 25000-25615 \
    --incumbent builtin:published --proposed csv:scores.csv:my_score \
    --slice 'score100plus=incumbent>=100' --json-out verdict.json
```

- Score sources are `builtin:{sum10,category3,algo2,published,published_pct}`,
  `csv:PATH[:COLUMN]` (per-sequence), or `py:MODULE:FUNCTION` (called per epoch
  on that epoch's published field). Lower is better. Append `@higher` if not.
- Ground truth is the public leaderboard API. Each epoch is saved to
  `~/.cache/nova-nanobody/leaderboard_nanobodies/` before parsing.
- `--outcome own` scores against each submission's own published score. Use it
  for "does formula X match the validator". `--outcome neighbors` scores against
  how the validator rated a candidate's 1..k-edit relatives. Use it for parent
  and seed selection.
- The statistic is Spearman rho, because only the order a formula induces matters
  and formulas live on unrelated scales. A paired bootstrap of
  rho(proposed) - rho(incumbent) gives the confidence interval.

## Reading the verdict

| exit | verdict | what to do |
|---:|---|---|
| 0 | PROPOSED IS BETTER | May deploy. Go to one box first and watch the next epochs. |
| 1 | NO EVIDENCE OF IMPROVEMENT | **Do not deploy.** This is not "safe" or "equivalent". It means the data cannot tell the two apart. |
| 2 | PROPOSED IS WORSE | **Do not deploy.** If it is already live, revert it. |
| 3 | INSUFFICIENT DATA | **Do not deploy.** Widen the window or wait for more epochs. |

"0.21 vs 0.19" is not a result. Read the CI of the difference, not the two
point estimates. If the sign flips between slices, the difference is noise.
`--outcome neighbors` CIs are optimistic, because relatives are shared between
candidates, so treat a CI that barely clears zero as not significant.

If the tool cannot express your change, an equivalent check must still compare
both formulas against real published outcomes, on the same candidates, with
uncertainty. Record the result in the commit message that deploys the change.
