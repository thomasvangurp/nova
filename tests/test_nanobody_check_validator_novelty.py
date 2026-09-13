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


def test_the_replay_disables_the_coarse_prefilter_like_the_validator_does():
    """The engine's default prefilter made this replay useless.

    neurons/validator/nanobody_validity.py calls search() with
    coarse_min_shared=None and coarse_jaccard=None. Leaving the defaults on
    discarded every top sequence before alignment: the replay reported
    coarse=0, max_identity=0.000 and "30/30 novel" for a candidate pool that
    was 0.95-0.98 identical to the live rank-1 sequence.
    """
    import inspect

    from nanobody_opt import check_validator_novelty

    source = inspect.getsource(check_validator_novelty.main)
    call = source[source.index("engine.search("):]
    call = call[: call.index(")")]
    assert "coarse_min_shared=0" in call
    assert "coarse_jaccard=0.0" in call
    assert "coarse_min_shared=None" not in call, (
        "None selects the configured coarse filter rather than skipping it"
    )
