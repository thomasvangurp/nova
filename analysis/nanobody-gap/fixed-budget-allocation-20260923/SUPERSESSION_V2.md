# V2 same-cohort budget controls

Added before the V2 run after inspecting only the V1 aggregate output.

The literal live 4/4/3 arm spends 2,244 repeats at the configured 300-candidate
width, but only 2,107 repeats on this frozen 282-candidate cohort. The V1 delayed
arms spent the full production budget, so their allocation and observation
count differed from the replayed baseline. V2 retains those production-budget
arms and adds delayed-prune controls capped at 2,107 observations. These are the
valid isolation of allocation shape on this cohort. No thresholds or outcomes
were changed.
