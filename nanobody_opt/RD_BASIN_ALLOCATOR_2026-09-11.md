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

## Terminal-policy follow-up

The original six-epoch replay did not load the older mutation database.  A fair
comparison was therefore rerun with the same 10,277 public validator
observations from 245 completed epochs (24074-25039) on both sides.

The old allocator retained the epoch-25047 exact winner at screen rank 256.  A
pure winner-position allocator instead retained the epoch-25050 exact winner
and displaced 25047, so replacing the old policy was rejected.  The production
policy now preserves the old diverse-parent order and adds a bounded
winner-position/residue order only in basin rank one.  Basin rank one receives
two proposal turns per turn allocated to each other basin.  At the unchanged
320 ceiling this retained 25047 at rank 308 and added 25050 at rank 156,
improving exact-winner inclusion from 1/6 to 2/6 while keeping 96 non-basin
proposals in the first 320.

The learned evidence is intentionally hierarchical: winning mutation position,
target-residue frequency among prior public winners and competitive sequences,
then parent recency.  The five best prior residue classes at each terminal
position form the exploration fallback, which makes previously unseen-in-core
routes such as A121Q generatable without enumerating every amino acid.

This is retrospective development evidence, not a sealed estimate: the policy
was designed after inspecting these six outcomes, and candidate Layer-2 was not
retroactively rerun.  The compact, hash-addressed result is preserved in
`rd_results/terminal_policy_replay_2026-09-11.json`.
