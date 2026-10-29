"""Algorithm invariants only; actual annotation is a separate host experiment."""
import random

from nanobody_opt.pr100_proposal_pilot import (
    AMINO_ACID_WEIGHTS, AMINO_ACIDS, CELLS, GENES, CDR3_BODY_PATTERN,
    CDR3_TAIL, j_identity, propose, dry_run, v_identity, _build_cdr3_loop,
    _weighted_choice,
)
from nanobody_opt.regions import find_regions
from pr100_germline import real_panel


def test_proposals_are_reproducible_unqualified_and_length_conditioned():
    panel = real_panel()
    for gene, length in CELLS:
        a = propose(panel, gene, length, seed=1000, ordinal=0, reference_cdr3='A'*16)
        assert a == propose(panel, gene, length, seed=1000, ordinal=0, reference_cdr3='A'*16)
        assert a != propose(panel, gene, length, seed=1000, ordinal=1, reference_cdr3='A'*16)
        assert not a['acceptance_qualified'] and not a['target_scored']
        # The hybrid J region (IGHJ4 FDY tail + IGHJ6 FR4) matches IGHJ4 best.
        assert a['j_reference_allele'] == 'IGHJ4*01'


def test_unavailable_reference_arms_are_not_reported_as_zero_yield():
    arms = dry_run([], real_panel(), extract_cdr3=lambda _: None, per_arm=8)
    assert len(arms) == 16 and all(a['gpu_calls'] == 0 for a in arms)
    assert sum(a['status'] == 'UNRUNNABLE' for a in arms) == 12
    for a in arms:
        if a['status'] == 'RUNNABLE':
            assert len(a['proposals']) == len({r['sequence'] for r in a['proposals']}) == 8


def test_generator_gene_scope_matches_pinned_challenge():
    from nanobody_opt import rules
    cfg = rules.pinned_config(rules.ACTIVATION_EPOCH)
    assert set(GENES) == {g for genes in cfg['germline_filtering']['v_neighborhood_categories'].values() for g in genes}


def test_flat_weight_draw_keeps_5ehs_composition():
    """The flat draw (used only for short loops) keeps the rival's composition."""
    rng = random.Random(20261028)
    draws = [_weighted_choice(rng) for _ in range(20_000)]
    aliphatic = frozenset('LIVMW')
    assert not aliphatic.intersection(draws)
    assert 'N' not in draws
    assert set(draws) <= set(AMINO_ACIDS)
    assert set(AMINO_ACID_WEIGHTS) == set(AMINO_ACIDS)


def test_short_loops_use_flat_weights_with_diversity():
    """L < 12 falls back to flat weights, restoring diversity (no 3-unique collapse)."""
    panel = real_panel()
    for length in (5, 8, 9, 11):
        seen = set()
        for ordinal in range(300):
            seen.add(propose(panel, 'IGHV3-23', length, seed=9, ordinal=ordinal)['sequence'])
        assert len(seen) > 100, (length, len(seen))


def test_long_loops_use_positional_template():
    """L >= 12 emits the submitted beater head + charged body, zero L/I/V/M/W."""
    rng = random.Random(1)
    for length in (12, 16, 18, 23):
        for _ in range(20):
            cdr3 = _build_cdr3_loop(rng, length - 3) + CDR3_TAIL
            assert len(cdr3) == length
            assert cdr3.endswith(CDR3_TAIL)
            assert not set('LIVMW') & set(cdr3)
            assert cdr3[0] in 'SA' and cdr3[1] in 'KR'
            head, body = cdr3[2:7], cdr3[7:-3]
            # Y-P-Y-[R/H]-F: three aromatics, R or H at position 4.
            assert head[0] == 'Y' and head[1] == 'P' and head[2] == 'Y' and head[4] == 'F'
            assert head[3] in 'RH'
            # Body is the DKTRDSE pattern cycled to the requested length.
            assert body == CDR3_BODY_PATTERN * (len(body) // len(CDR3_BODY_PATTERN)) + \
                CDR3_BODY_PATTERN[:len(body) % len(CDR3_BODY_PATTERN)]


def test_short_loop_hydro_rejection_rate_in_expected_band():
    """The cheap hydro screen trims ~35-60% of flat-weight short loops (not bypassed)."""
    from nanobody_opt.hydrophobicity import check
    panel = real_panel()
    for length in (5, 8, 9, 11):
        rejected = 0
        total = 400
        for ordinal in range(total):
            seq = propose(panel, 'IGHV3-23', length, seed=9, ordinal=ordinal)['sequence']
            if not check(seq)[1]:
                rejected += 1
        rate = rejected / total
        assert 0.35 <= rate <= 0.60, (length, rate)


def test_template_emissions_stay_under_hydro_ceiling():
    """L >= 12 template emissions all pass the 0.44 CDR3 hydro ceiling."""
    from nanobody_opt.hydrophobicity import score
    panel = real_panel()
    for gene in GENES:
        for length in range(12, 24):
            for ordinal in range(30):
                seq = propose(panel, gene, length, seed=8, ordinal=ordinal)['sequence']
                h = score(seq)
                assert h['cdr3'] <= 0.44 + 1e-9, (gene, length, h['cdr3'])
                assert h['cdr2'] <= 0.24 + 1e-9, (gene, length, h['cdr2'])


def test_v_and_j_identity_meet_gate_margin():
    """Every emitted (gene, length) cell stays >= 84% V and >= 86% J identity."""
    panel = real_panel()
    for gene in GENES:
        for length in (5, 11, 16, 18, 23):
            seq = propose(panel, gene, length, seed=11, ordinal=0)['sequence']
            non_gap = [p for p in sorted(panel['V'][gene+'*01'])
                       if p <= 104 and panel['V'][gene+'*01'][p] != '-']
            base_len = len(non_gap)
            base = seq[:base_len]
            loop = list(seq[base_len:base_len + length - 3])
            tail = seq[base_len + length - 3:base_len + length]
            suffix = seq[base_len + length:]
            prefix = {p: base[i] for i, p in enumerate(non_gap)}
            _, _, vf = v_identity(panel, gene, prefix, loop)
            jf = j_identity(panel, tail, suffix)
            assert vf >= 0.84, (gene, length, vf)
            assert jf >= 0.86, (gene, length, jf)


def test_reference_cdr3_is_grafted_not_erased():
    """The reference path grafts the donor CDR3 (trim/pad/mutate), never just seeds."""
    panel = real_panel()
    donor = 'QNLTAGVFDY'
    seq = propose(panel, 'IGHV3-23', 16, seed=1, ordinal=0,
                  reference_cdr3=donor)['sequence']
    reg = find_regions(seq)
    cdr3 = seq[reg.cdr3[0]:reg.cdr3[1]]
    # 7 donor middle residues, at most 4 mutated -> at least 3 survive.
    shared = sum(1 for aa in donor[:-3] if aa in cdr3)
    assert shared >= 3, (donor, cdr3)
