# Fixed-GPU allocation verdict

## Verdict

Do not add GPU capacity and do not uniformly deepen all candidates. Keep the
2,800-candidate one-pass screen and the 2,244-repeat budget. After correcting
the selector to rank the event that actually earns per entry (beating the
external public field), reallocate the same repeats from the current
`4 @ 300 -> 4 @ 180 -> 3 @ 108` tournament to:

```text
6 repeats @ 300 candidates
prune once to 111 candidates
4 repeats @ 111 candidates
```

That is exactly 2,244 repeat observations, but only ten model launches instead
of eleven. Including the initial screen, every survivor has eleven independent
current-epoch observations. The configured timing model estimates 2,160 seconds
versus 2,182.5 seconds for the current tournament; it fits the existing GPU and
epoch envelope.

## Held-out replay

The replay used the frozen 282-sequence current-production pool and all sixteen
complete validator-like observations already generated. Four draw blocks were
held out in turn; selection used the other twelve observations and 22
development public fields, while outcomes used eight separately reserved public
fields. The cohort-scaled budget is 2,107 observations.

With the correct per-entry field-win selector:

| allocation | repeats | survivor depth incl. screen | wins / 1,280 entries | top-10 / 1,280 | fleet-field wins / 128 |
|---|---:|---:|---:|---:|---:|
| current 4/4/3 | 2,107 | 12 | 16 (1.25%) | 183 (14.30%) | 16 (12.50%) |
| delayed 6/4 | 2,104 | 11 | 27 (2.11%) | 233 (18.20%) | 24 (18.75%) |
| delayed 6/3 | 2,106 | 10 | 26 (2.03%) | 214 (16.72%) | 24 (18.75%) |
| uniform 7 | 1,974 | 8 | 24 (1.88%) | 222 (17.34%) | 22 (17.19%) |

These are paired **replay-control** rates, not the miner's historical live
rates. At the time of this report the dashboard's clean live aggregate is
2.38% wins per entry and 14.1% top-10 per entry. The replay control therefore
calibrates closely for top-10 (14.30% versus 14.1%) but materially understates
live wins (1.25% versus 2.38%). The replay supports a relative allocation
comparison; it does not establish that the proposed policy's absolute live win
rate would be 2.11%, nor that it would exceed the current 2.38%.

The 6/4 allocation improved wins in four of sixteen draw blocks, tied in twelve,
and regressed in none. Top-10 count improved in eleven blocks, tied in three,
and regressed in two. The win improvement is nevertheless concentrated: six of
the eleven additional wins came from draw 15. Treat this as strong regression
evidence, not a fresh prospective proof.

## Why “raise everybody to a high count” is not optimal

Uniformly giving every candidate eight repeats used essentially the full budget
but produced 18 wins and 209 top-10 entries. Broad evidence is valuable through
roughly six repeats; after that, spending the remaining budget on the leading
~37% is better than either early pruning or uniform saturation.

The historical median depth near 40 does not change this conclusion. Historical
depth is uneven and many novel proposals have no historical folds. Within-epoch
allocation must therefore stand on its own; history should be pooled when it is
available, not treated as a reason to deny new candidates comparable evidence.

## Critical interaction with the selector

Allocation alone is not the fix. Under the current internal-tournament selector,
the scaled 6/4 policy produced 14 wins and 161 top-10 entries versus 12 and 166
for current 4/4/3. The allocation advantage appears only when candidates are
ranked by their own probability of beating the external field. This confirms
the previously identified objective bug: probabilities that sum to one across
our own cohort optimize which of our entries beats our other entries, not how
many of our tickets beat competitors.

Therefore the safe order is:

1. implement and test external-field per-entry win probability in the ranker;
2. change the fixed-budget tournament to 6/4 at 300 -> 111;
3. run one shadow epoch, recording both old and new fleets from the same folds;
4. activate only if the new artifact is complete, has eleven observations per
   survivor, and meets the existing integrity/pose gates.

No new GPU box or higher recurring GPU spend is needed.
