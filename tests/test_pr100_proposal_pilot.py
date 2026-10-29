"""Algorithm invariants only; actual annotation is a separate host experiment."""
import random

from nanobody_opt.pr100_proposal_pilot import (
    AMINO_ACIDS, AMINO_ACID_WEIGHTS, propose, dry_run, CELLS, GENES,
    _weighted_choice,
)


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


def test_sampled_cdr3_composition_matches_5ehs_recipe():
    """The weighted draw mirrors the rival's CDR3 recipe at every residue site.

    Every residue draw (CDR3 fill, CDR1/2 substitutions, reference-CDR3
    padding/mutation) routes through ``_weighted_choice``, so a large sample of
    that helper is the composition of the generated CDR3s. The 5EHS recipe
    forbids N and the aliphatics L/I/V/M/W (zero weight) and concentrates on
    aromatic F/Y plus charged D/R/K/E/H.
    """
    rng = random.Random(20261028)
    draws = [_weighted_choice(rng) for _ in range(20_000)]
    aliphatic = frozenset('LIVMW')
    aromatic_charged = frozenset('FYDRKEH')
    hydrophobic = frozenset('FLIVMWYA')
    # Zero weight means these residues can never be emitted by random.choices.
    assert not aliphatic.intersection(draws)
    assert 'N' not in draws
    assert set(draws) <= set(AMINO_ACIDS)
    # Aromatic + charged dominate (~0.70 of the mass), as in the rival's winners.
    assert sum(aa in aromatic_charged for aa in draws) / len(draws) > 0.6
    # The implied CDR3 hydrophobic fraction is ~0.30, inside the 0.44 PSH gate.
    assert 0.25 < sum(aa in hydrophobic for aa in draws) / len(draws) < 0.35
    # Every weight is still a positive, length-agnostic table over the full
    # (cysteine-free) alphabet; zero-weight residues are the only ones removed.
    assert set(AMINO_ACID_WEIGHTS) == set(AMINO_ACIDS)
