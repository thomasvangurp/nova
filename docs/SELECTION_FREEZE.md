# Selection freeze — from 2026-09-14

Selection is frozen at the epoch-25096 lead-ticket configuration. Do not change
anything in the first table without several epochs of a fixed configuration
behind the decision.

## Why

Results across the run:

| epoch | our best rank | our score | scored / committed |
|------:|--------------:|----------:|-------------------:|
| 25093 |         20/40 |       128 |                2/6 |
| 25094 |         13/44 |       124 |                3/7 |
| 25095 |          5/37 |        83 |                3/7 |
| 25096 |          2/42 |    **43** |                4/7 |
| 25097 |          3/40 |        71 |                2/8 |
| 25098 |         19/42 |       138 |                3/8 |
| 25099 |          5/43 |        91 |                5/7 |
| 25100 |         17/43 |       134 |                2/7 |
| 25101 |         23/42 |       139 |                2/7 |

Ranks improved to 2, then fell back. The turn coincides with selection being
changed almost every epoch. None of it is attributable: each epoch is a single
draw from a wide distribution, and reading single draws as signal is the same
error the win-probability estimator was making before it was shrunk.

## Frozen — the lead ticket

| setting | value | note |
|---|---|---|
| `fleet.DEFAULT_VALUE_SHARE` | `0.0` | pure win-probability order, as at 25096 |
| `fleet.ENFORCE_ONE_UID_PER_CLONOTYPE` | `False` | CDR3 does not predict drops; enforcing it cut 25102 to one ticket |
| crowd avoidance | not passed | 349 of 350 candidates are contested; the one clear candidate was the worst in the pool |
| `win_probability` shrinkage | on | live at 25096 |

## Kept — slots two onward only

These cannot change the first pick, so they add tickets without risking the
lead: `SHORTLIST_LIMIT` 300, separation-first assignment, the replication
family slate, round-robin parent seeding, donor grafts, public parents.

## Kept — plumbing, verified by mechanism not by score

Payload format `~|<sequence>`; commits inside the end-of-epoch reserve; one
commitment per uid per epoch; chain-derived chain-writer hotkey; the novelty
gate actually comparing; post-commit ownership verification.

## What would justify unfreezing

Five or six epochs at this configuration, then compare best-score-per-epoch and
scored-per-committed against the table above. Change one thing at a time.
