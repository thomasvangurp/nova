from nanobody_opt.check_validator_novelty import duplicate_reason


def test_validator_duplicate_tiers_match_live_rules():
    assert duplicate_reason(0.95, {"CDR3": 0.0}) == "near-identical sequence"
    assert "same clonotype" in duplicate_reason(0.70, {"CDR3": 0.85})
    assert "conserved" in duplicate_reason(
        0.70, {"CDR1": 0.90, "CDR2": 0.91, "CDR3": 0.80}
    )
    assert "no CDR" in duplicate_reason(0.90, None)


def test_validator_duplicate_rules_allow_distant_sequences():
    assert duplicate_reason(
        0.89, {"CDR1": 1.0, "CDR2": 1.0, "CDR3": 0.79}
    ) is None
