# Masked-core basin allocator

## Verdict

Masked structural cores are useful allocation groups, but terminal haplotypes
within a core are not interchangeable molecules.  The production policy now
allocates most proposal and exact-screen capacity across the five strongest
historical cores, then samples multiple distinct terminal variants inside each
core.  It retains a bounded CDR/framework/exploration lane.

The candidate budget is:

- 384 proposals before Layer 2;
- 269 basin-terminal proposals (70%);
- 115 CDR, framework, and lineage/exploration proposals (30%);
- at most 320 Layer-2 survivors exact-scored in tranches of 20;
- five repeat finalists, allowing evidence from each leading basin.

In both the six-epoch leakage-free replay and a current 34,024-sequence archive
canary, the basin lane contained exactly five cores with 53 or 54 proposals per
core.  Therefore the policy fits the existing exact budget without increasing
the 320-candidate ceiling.

## Evidence and limitation

For held-out epochs 25046 through 25051, the eventual winner cores ranked
1, 5, 5, 1, 1, and 3 using only prior fields.  Core coverage was therefore
6/6.  The complete production generator included the exact winner sequence in
only one of six epochs.  This is not presented as exact-winner recall: large
cores contain thousands of possible terminal recombinations, and exhaustive
enumeration does not fit the budget.

The earlier result that our submission shared the winning core in epochs
25046, 25048, and 25050 also does not mean that we submitted the winner.  Our
terminal haplotypes placed 6/10, 4/11, and 6/10 within those cores and scored
159 versus 44, 103 versus 40, and 116 versus 61.  The core selects a basin;
exact scoring must still select the variant.

## Implemented policy

1. Rank structural cores by recent competitive density, all-time elite and
   competitive density, best score, recency, and median score.
2. Reserve 32 Layer-2 parent slots across the top five cores, round-robin by
   core rather than by individual parent.
3. Generate supported one-step terminal shells from every observed historical
   haplotype in each selected core, then bound this combinatorial pool before
   expensive graph evaluation.
4. Allocate 269 proposals evenly across the five cores and retain 115 proposals
   for CDR/framework/lineage exploration.
5. Exact-screen in 20-candidate tranches balanced by core.
6. Preserve raw validator-shaped scores, but shrink a one-shot terminal score
   toward the screened core median once at least three variants from that core
   exist.  Repeated exact evidence still controls strict qualification.

This policy improves the demonstrated failure mode—choosing only one poor
terminal sample from a good basin—without asserting that terminal changes are
biologically irrelevant or allowing one lucky local score to dominate.
