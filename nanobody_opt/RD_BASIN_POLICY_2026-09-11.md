# Leakage-free basin-policy replay

Date: 2026-09-11

## Question

Can the miner retain recent, individually weak public parents when they sit in
a strong P20809 VHH neighborhood, enumerate only family-supported terminal
residues, and iteratively reduce the resulting search to the 320-candidate exact
budget?

The replay used only public observations strictly before each held-out epoch.
The final six available fields, epochs 25046-25051, were opened only to measure
the eventual winner's parent, proposal, and masked-core rank.

## Observed winner lineage

Every winner was Hamming distance one from a sequence submitted by a different
hotkey in an earlier public field:

| Winner epoch | Prior epoch | Prior score | New substitution |
| ---: | ---: | ---: | --- |
| 25046 | 25040 | 95 | T124P |
| 25047 | 25042 | 139 | A126G |
| 25048 | 25046 | 93 | A121Q |
| 25049 | 25044 | 128 | Q3A |
| 25050 | 25047 | 101 | M3Q |
| 25051 | 25046 | 93 | P127E |

The weak individual score was not the useful signal. Those parents had 23-41
prior score-at-most-70 neighbors within Hamming distance six and 11-17 prior
score-at-most-55 neighbors in the same radius.

## Parent and proposal results

Within only the preceding eight epochs, a close-neighbor-count basin ordering
ranked the six exact parents 2, 161, 151, 17, 102, and 27. All fit in an
approximately 320-parent scan, but only three fit in the current 64-parent bank.

Restricting substitutions to residues previously present in a score-at-most-70
family member generated five of six winners. A121Q required an exploration
allowance because Q at position 121 had not yet appeared in a competitive
member of that family. Allowing any family-observed terminal residue generated
all six winners, but expanded the unique proposal sets to roughly 11k-13k per
epoch.

Neither available ordering reduced those proposals to the live budget:

| Epoch | Frozen pairwise-model winner rank |
| ---: | ---: |
| 25046 | 828 |
| 25047 | 9,127 |
| 25048 | 3,884 |
| 25049 | 9,692 |
| 25050 | 6,036 |
| 25051 | 8,883 |

The saved 320-candidate exact screen also rejects hard sibling elimination.
Same-parent total-score Spearman was 0.065. Distance-one physical-interaction
Spearman was 0.020, although distance-one developability was much smoother at
0.749. A poor first exact probe is therefore useful for developability but not
a safe reason to discard a binding family.

## Masked-core result

Masking the first 10 and last 17 residues collapses terminal uniqueness variants
to their shared binding core. Ranking these cores by prior recent score-at-most-70
count, then all-time score-at-most-55 and score-at-most-70 support, placed the
six winning cores at ranks 1, 5, 5, 1, 1, and 3.

This is the successful part of the policy: allocate by strong structural core,
not by the latest score of an individual terminal variant. It is not evidence
that the exact terminal child can be predicted. In epochs 25046, 25048, and
25050, UID 8 submitted the same masked core as the winner yet scored 159 versus
44, 103 versus 40, and 116 versus 61. Within those same-core cohorts, UID 8
ranked 6/10, 4/11, and 6/10.

## Decision

Do not deploy hard iterative parent elimination or the frozen pairwise model.
Reserve proposal and exact-screen capacity by masked core, retain an exploration
floor for unsupported terminal alleles, and treat one-shot terminal variants as
noisy samples from a core distribution. Any adaptive expansion must require
multiple family observations or a new proxy demonstrated to predict confidence
and physical-interaction outcomes; one sibling result is insufficient.

