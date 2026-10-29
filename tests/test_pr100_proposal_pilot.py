"""Algorithm invariants only; actual annotation is a separate host experiment."""
import random

from nanobody_opt.pr100_proposal_pilot import (
    AMINO_ACID_WEIGHTS, AMINO_ACIDS, CELLS, GENES, CDR2_MOTIF, CDR3_BODY_PATTERN,
    CDR3_TAIL, GERMLINE_CDR2, RIVAL_5EHS_SEQUENCES,
    j_identity, propose, dry_run, v_identity, _build_cdr3_loop,
    _build_short_cdr3_loop, _weighted_choice,
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
            assert len(a['proposals']) == len({r['sequence'] for r in a['proposals']})
            # Long-regime cells emit 8 unique CDR3s; the short cell (IGHV3-23, 5)
            # is a small deterministic RAFDY-family pool of two (RAFDY / RHFDY).
            expected = 2 if tuple(a['cell']) == ('IGHV3-23', 5) else 8
            assert len(a['proposals']) == expected


def test_generator_gene_scope_matches_pinned_challenge():
    from nanobody_opt import rules
    cfg = rules.pinned_config(rules.ACTIVATION_EPOCH)
    assert set(GENES) == {g for genes in cfg['germline_filtering']['v_neighborhood_categories'].values() for g in genes}


def test_flat_weight_draw_keeps_5ehs_composition():
    """The flat draw (now the reference-graft path only) keeps the rival's composition."""
    rng = random.Random(20261028)
    draws = [_weighted_choice(rng) for _ in range(20_000)]
    aliphatic = frozenset('LIVMW')
    assert not aliphatic.intersection(draws)
    assert 'N' not in draws
    assert set(draws) <= set(AMINO_ACIDS)
    assert set(AMINO_ACID_WEIGHTS) == set(AMINO_ACIDS)


def test_short_loops_use_rafdy_family_template():
    """L < 12 uses the RAFDY-family head (R[A/H] + fixed FDY), a small pool."""
    panel = real_panel()
    for length in (5, 6, 7, 11):
        seen = set()
        for ordinal in range(300):
            seq = propose(panel, 'IGHV3-23', length, seed=9, ordinal=ordinal)['sequence']
            seen.add(seq)
            reg = find_regions(seq)
            cdr3 = seq[reg.cdr3[0]:reg.cdr3[1]]
            assert cdr3.endswith(CDR3_TAIL), (length, cdr3)
            assert cdr3.startswith('R'), (length, cdr3)
            assert not set('LIVMWN') & set(cdr3), (length, cdr3)
        # A small deterministic pool, not the open flat draw.
        assert 2 <= len(seen) <= 300, (length, len(seen))


def test_short_cdr3_head_lengths_match_rafdy_family():
    """L=5 -> RA/RH, L=6 -> R[A/H]X, never a byte-identical rival CDR3."""
    panel = real_panel()
    for length in (5, 6):
        heads = set()
        for ordinal in range(200):
            seq = propose(panel, 'IGHV3-23', length, seed=9, ordinal=ordinal)['sequence']
            reg = find_regions(seq)
            cdr3 = seq[reg.cdr3[0]:reg.cdr3[1]]
            assert cdr3.endswith(CDR3_TAIL)
            heads.add(cdr3[:-len(CDR3_TAIL)])
        if length == 5:
            assert heads <= {'RA', 'RH'}, heads
        else:
            assert all(len(h) == 3 and h[0] == 'R' and h[1] in 'AH'
                       and h[2] in 'TSHG' for h in heads), heads
        assert 'TSH' not in heads  # never byte-identical to the rival length-6 head


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


def test_short_cdr3_hydro_fraction_stays_in_rafdy_band():
    """The RAFDY-family head keeps the aromatic FDY tail but never adds aliphatics.

    Worst case is RA+FDY (A, F, Y = 3/5 = 0.6); the polar extension dilutes the
    longer loops. This documents the cheap-screen shape of the short template
    (the real PSH gate is exercised separately in
    test_short_emissions_pass_real_develop_gate).
    """
    from nanobody_opt.hydrophobicity import score
    panel = real_panel()
    for gene in GENES:
        for length in (5, 6, 7, 8, 11):
            for ordinal in range(40):
                seq = propose(panel, gene, length, seed=9, ordinal=ordinal)['sequence']
                h = score(seq)
                assert h['cdr3'] <= 0.6 + 1e-9, (gene, length, h['cdr3'])


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


def _framework_prefix(panel, gene, seq):
    """Decompose a scaffold emission back into the IMGT 1-104 prefix dict."""
    non_gap = [p for p in sorted(panel['V'][gene + '*01'])
               if p <= 104 and panel['V'][gene + '*01'][p] != '-']
    base = seq[:len(non_gap)]
    return {p: base[i] for i, p in enumerate(non_gap)}


def test_short_regime_restores_full_5ehs_framework():
    """L < 12 restores the winners' V5/G9/V13/K20 + germline K48; L >= 12 does not."""
    panel = real_panel()
    for gene in GENES:
        short = propose(panel, gene, 5, seed=9, ordinal=0)['sequence']
        long = propose(panel, gene, 16, seed=9, ordinal=0)['sequence']
        s_prefix = _framework_prefix(panel, gene, short)
        l_prefix = _framework_prefix(panel, gene, long)
        # Winners' FR1 V5/G9/V13/K20 only in the short regime; K48 is germline
        # in both regimes (the ``GKERE`` FR2 motif).
        assert (s_prefix[5], s_prefix[9], s_prefix[13], s_prefix[20], s_prefix[48]) == \
            ('V', 'G', 'V', 'K', 'K'), gene
        assert l_prefix[48] == 'K', gene  # germline K (pruned framework)
        assert (l_prefix[5], l_prefix[9], l_prefix[13], l_prefix[20]) != \
            ('V', 'G', 'V', 'K'), gene
        # The shared FR1 E->D/V->Q + FR2 G->E/L->R hallmarks are in both regimes.
        for prefix in (s_prefix, l_prefix):
            assert (prefix[1], prefix[2]) == ('D', 'Q'), gene
            assert (prefix[49], prefix[50]) == ('E', 'R'), gene


def test_short_regime_full_framework_v_identity_meets_gate():
    """The restored short-regime framework keeps V identity >= 82% per gene."""
    panel = real_panel()
    worst = {}
    for gene in GENES:
        for length in (5, 6, 7, 11):
            for ordinal in range(120):
                seq = propose(panel, gene, length, seed=11, ordinal=ordinal)['sequence']
                non_gap = [p for p in sorted(panel['V'][gene + '*01'])
                           if p <= 104 and panel['V'][gene + '*01'][p] != '-']
                base = seq[:len(non_gap)]
                loop = list(seq[len(non_gap):len(non_gap) + length - 3])
                prefix = {p: base[i] for i, p in enumerate(non_gap)}
                _, _, vf = v_identity(panel, gene, prefix, loop)
                worst[gene] = min(worst.get(gene, 1.0), vf)
                assert vf >= 0.82, (gene, length, ordinal, vf)
    # Measured 2026-10-29 with the V5/G9/V13/K20 + K48 framework: the weakest
    # gene (IGHV3-53) sits at 91.75%, comfortably above the >= 82% gate, so no
    # per-gene relaxation is needed.
    assert min(worst.values()) >= 0.82


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


def test_short_cdr2_keeps_germline_per_gene():
    """L < 12 keeps each gene's germline CDR2; the charged motif is long-only."""
    panel = real_panel()
    for gene in GENES:
        short = propose(panel, gene, 5, seed=9, ordinal=0)['sequence']
        long = propose(panel, gene, 16, seed=9, ordinal=0)['sequence']
        assert GERMLINE_CDR2[gene] in short, (gene, GERMLINE_CDR2[gene], short)
        assert CDR2_MOTIF not in short, gene
        assert CDR2_MOTIF in long, gene
        assert GERMLINE_CDR2[gene] not in long, gene


def test_long_path_byte_for_byte_unchanged():
    """L >= 12 charged CDR2 + positional template is byte-for-byte stable."""
    panel = real_panel()
    seq = propose(panel, 'IGHV3-23', 16, seed=11, ordinal=0)['sequence']
    assert seq == ('DQQLLESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKEREWVS'
                   'DISSQGDQTDYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCA'
                   'RYPYHFDKTRESFDYWGQGTTVTVSS')


def test_short_emissions_pass_real_develop_gate():
    """Short-CDR3 emissions satisfy the exact rules.developability_reason gate.

    This is the same decision function the live pipeline and
    ``pr100_filter.develop``'s caller use. The 3D fold is not run in this unit
    test, so PSH is supplied from a measured-profile stand-in (in-band <= 89);
    V%/J% are computed here against the real germline panel.
    """
    from nanobody_opt import rules
    from test_pr100_native_screen_integration import profile
    requirements = rules.pinned_config(rules.ACTIVATION_EPOCH)['nanobody_requirements']
    panel = real_panel()
    for gene in GENES:
        for length in (5, 6):
            seq = propose(panel, gene, length, seed=9, ordinal=0)['sequence']
            non_gap = [p for p in sorted(panel['V'][gene+'*01'])
                       if p <= 104 and panel['V'][gene+'*01'][p] != '-']
            base = seq[:len(non_gap)]
            loop = list(seq[len(non_gap):len(non_gap) + length - 3])
            tail = seq[len(non_gap) + length - 3:len(non_gap) + length]
            suffix = seq[len(non_gap) + length:]
            prefix = {p: base[i] for i, p in enumerate(non_gap)}
            _, _, vf = v_identity(panel, gene, prefix, loop)
            jf = j_identity(panel, tail, suffix)
            assert vf >= 0.84 and jf >= 0.86, (gene, length, vf, jf)
            for psh in (80., 89.):
                value = profile(seq, psh, cdr3_length=length, total_cdr_length=25)
                assert rules.developability_reason(value, requirements, sequence=seq) == ''
            over = profile(seq, 90., cdr3_length=length, total_cdr_length=25)
            assert rules.developability_reason(over, requirements, sequence=seq) == 'psh_threshold'


def test_short_emissions_never_byte_identical_to_rival():
    """No emitted short-regime sequence is a byte-identical 5EHS winner."""
    panel = real_panel()
    for gene in GENES:
        for length in (5, 6, 7, 11):
            for ordinal in range(120):
                seq = propose(panel, gene, length, seed=9, ordinal=ordinal)['sequence']
                assert seq not in RIVAL_5EHS_SEQUENCES


def test_short_loop_builder_handles_length_four_head():
    """L=4 (head length 1) yields 'R' before the fixed FDY tail."""
    rng = random.Random(1)
    for _ in range(20):
        assert _build_short_cdr3_loop(rng, 1) == ['R']
    for n in (2, 3, 8):
        loop = _build_short_cdr3_loop(rng, n)
        assert loop[0] == 'R' and len(loop) == n
        assert not set('LIVMWN') & set(loop)
