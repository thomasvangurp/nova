"""Algorithm invariants only; actual annotation is a separate host experiment."""
import random

from nanobody_opt.pr100_proposal_pilot import (
    CDR2_MOTIF, CDR3_BODY, CDR3_BODY_FIRST, CDR3_TAIL, CELLS, GENES, propose,
    dry_run, _apply_cdr2_motif, _build_cdr3_loop,
)


def panel():
    v = {p: 'A' for p in range(1, 129)}
    j = {p: ('A' if p >= 115 else '-') for p in range(1, 129)}
    return {'V': {g+'*01': dict(v) for g in {g for g, _ in CELLS}},
            'J': {'IGHJ4*01': dict(j), 'IGHJ6*01': dict(j)}}


def test_proposals_are_reproducible_unqualified_and_length_conditioned():
    for gene, length in CELLS:
        a = propose(panel(), gene, length, seed=1000, ordinal=0, reference_cdr3='A'*16)
        assert a == propose(panel(), gene, length, seed=1000, ordinal=0, reference_cdr3='A'*16)
        assert a != propose(panel(), gene, length, seed=1000, ordinal=1, reference_cdr3='A'*16)
        assert len(a['sequence']) == 104+length+11
        assert not a['acceptance_qualified'] and not a['target_scored']
        assert a['sequence'][22] == a['sequence'][103] == 'C'
        assert a['sequence'][48:50] == 'ER'
        assert a['j_reference_allele'] == 'IGHJ6*01'


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


def test_cdr3_loop_matches_5ehs_positional_recipe():
    """The CDR3 is anchor(2)+head+body+tail(3), never a flat residue table.

    Every length in the declared 5..23 scope must decompose as the positional
    recipe: anchor ``A[RKQ]``/``S[RKQ]``, head ``YPHRF`` (P@2, R@4, F@5 with
    H/Y alternating at 1 and 3), a G/S/Q/K/P/R/D/E body, and the fixed ``FDY``
    tail. The one length-independent invariant is ZERO L/I/V/M/W in the CDR3.
    """
    rng = random.Random(20261028)
    for length in range(5, 24):
        for _ in range(50):
            cdr3 = _build_cdr3_loop(rng, length - 3) + CDR3_TAIL
            assert len(cdr3) == length
            assert cdr3.endswith(CDR3_TAIL)
            assert not set('LIVMW') & set(cdr3)
            anchor, headbody = cdr3[:2], cdr3[2:-3]
            assert anchor[0] in 'AS' and anchor[1] in 'RKQ'
            head_len = min(5, max(0, length - 5))
            head, body = headbody[:head_len], headbody[head_len:]
            assert all(aa in CDR3_BODY for aa in body)
            assert body == '' or body[0] in CDR3_BODY_FIRST
            if head_len >= 2:
                assert head[1] == 'P'
            if head_len >= 4:
                assert head[3] == 'R'
            if head_len >= 5:
                assert head[4] == 'F'
                assert set(head[0] + head[2]) == {'H', 'Y'}
            # CDR3 hydrophobic fraction must sit under the 0.44 PSH screen.
            hydro = sum(1 for aa in cdr3 if aa in 'FLIVMWYA') / length
            assert hydro <= 0.44 + 1e-9


def test_cdr2_motif_lands_in_the_17residue_window():
    base = 'G'*22 + 'C' + 'GG' + 'WVRQ' + 'G'*10 + 'AISGSGGSTYYADSVKG' + 'G'*20
    out = _apply_cdr2_motif(base)
    w = base.find('WVRQ')
    window = out[w+14:w+31]
    assert window == CDR2_MOTIF + 'YADSVKG'
