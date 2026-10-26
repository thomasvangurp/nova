"""Algorithm invariants only; actual annotation is a separate host experiment."""
from nanobody_opt.pr100_proposal_pilot import propose, dry_run, CELLS, GENES


def panel():
    v = {p: 'A' for p in range(1, 129)}
    j = {p: ('A' if p >= 115 else '-') for p in range(1, 129)}
    return {'V': {g+'*01': dict(v) for g in {g for g, _ in CELLS}}, 'J': {'IGHJ4*01': j}}


def test_proposals_are_reproducible_unqualified_and_length_conditioned():
    for gene, length in CELLS:
        a = propose(panel(), gene, length, seed=1000, ordinal=0, reference_cdr3='A'*16)
        assert a == propose(panel(), gene, length, seed=1000, ordinal=0, reference_cdr3='A'*16)
        assert a != propose(panel(), gene, length, seed=1000, ordinal=1, reference_cdr3='A'*16)
        assert len(a['sequence']) == 104+length+11
        assert not a['acceptance_qualified'] and not a['target_scored']
        assert a['sequence'][22] == a['sequence'][103] == 'C'
        assert a['sequence'][48:50] == 'ER'


def test_unavailable_reference_arms_are_not_reported_as_zero_yield():
    arms = dry_run([], panel(), extract_cdr3=lambda _:None, per_arm=8)
    assert len(arms) == 16 and all(a['gpu_calls'] == 0 for a in arms)
    assert sum(a['status'] == 'UNRUNNABLE' for a in arms) == 12
    for a in arms:
        if a['status'] == 'RUNNABLE':
            assert len(a['proposals']) == len({r['sequence'] for r in a['proposals']}) == 8


def test_generator_gene_scope_matches_pinned_challenge():
    from nanobody_opt import rules
    cfg=rules.pinned_config(rules.ACTIVATION_EPOCH)
    assert set(GENES)=={g for genes in cfg['germline_filtering']['v_neighborhood_categories'].values() for g in genes}
