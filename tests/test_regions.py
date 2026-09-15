from nanobody_opt.regions import find_regions


CANONICAL = (
    "KWQLVESGGGLVKPGGSLRLSCAASGFTMSQGWYGMSWFRQAPGKEREWVSLISSSGSTWMDY"
    "PDSVKGRFTSSRDNAKNSLYGQMNSLRAEDTAVYYCAVSLGAHPCPMEGEMESFGAGTTVTVAP"
)
ELITE_MUTATED_FR4 = (
    "DWTLVESGGGLVKPGGSLRLSCAASGFTMSQGWYGMSWFRQAPGKEREWVSLISSSGSTWMDY"
    "PDSVKGRFTSSRDNAKNSLYGQMNSLRAEDTAVYYCAVSLGAHPCPMEGEMESFAAGTAVPVAP"
)


def test_mutated_fr4_uses_same_validated_cdr_boundaries():
    canonical = find_regions(CANONICAL)
    mutated = find_regions(ELITE_MUTATED_FR4)

    assert canonical == mutated
    assert mutated.cdr3 == (99, 116)


# A [WF]GxG motif inside FR3, ahead of the YYC anchor.  The FR4 search must
# begin at the cysteine anchor, not at the start of the sequence: a compiled
# pattern searched with a start offset has to behave exactly like searching the
# suffix, or CDR3 would be truncated to nothing.
EARLY_FR4_DECOY = CANONICAL[:70] + "WGAG" + CANONICAL[74:]


def test_an_fr4_motif_before_the_cysteine_anchor_is_not_taken_for_fr4():
    decoy = find_regions(EARLY_FR4_DECOY)

    assert decoy == find_regions(CANONICAL)
    assert decoy.cdr3 == (99, 116)
    assert EARLY_FR4_DECOY[70:74] == "WGAG"
