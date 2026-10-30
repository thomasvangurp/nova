"""Edit-based generation lane (v2 port #4): pure + worker wiring.

The module under test is a dependency-free port of ``local_neighborhood``'s
``supported_framework_candidates`` / ``supported_lineage_candidates``, so the
module-level tests run without abnumber/anarci/MySQLdb. The worker-wiring test
reuses the same monkeypatch pattern as ``test_pr100_reference_donors``.
"""
import json
import sys
import types

import pytest

from nanobody_opt import gates, hydrophobicity, rules
from nanobody_opt import pr100_edit_generation as edit
from nanobody_opt import pr100_live_worker as worker
from nanobody_opt.pr100_proposal_pilot import GENES, propose, v_identity
from nanobody_opt.regions import find_regions
from pr100_germline import real_panel


def _wiring_budget(parent):
    """Only a wiring stub; real HMM accounting is exercised by recorded tests below."""
    return types.SimpleNamespace(parent=parent, protected_j=frozenset(),
        delta=lambda position, aa: 0, restoration=lambda position: None,
        accepts=lambda child: True)


def _safe_framework_position(parent):
    """A non-anchor FR3 position whose residue is neither S nor T, so both
    substitutions introduce fresh residues for the per-position allowed table."""
    regions = find_regions(parent)
    for pos in range(regions.cdr2[1], regions.cdr3[0]):
        if parent[pos] in ('S', 'T'):
            continue
        candidate = parent[:pos] + 'S' + parent[pos + 1:]
        try:
            if find_regions(candidate).cdr3 == regions.cdr3:
                return pos
        except ValueError:
            continue
    raise AssertionError('no safe framework substitution found')


def _variant(parent, position, residue):
    return parent[:position] + residue + parent[position + 1:]


def _v_identity_of(panel, gene, seq, length):
    """Numeric V identity of a scaffold-layout sequence (see proposal tests)."""
    non_gap = [p for p in sorted(panel['V'][gene + '*01'])
               if p <= 104 and panel['V'][gene + '*01'][p] != '-']
    base = seq[:len(non_gap)]
    loop = list(seq[len(non_gap):len(non_gap) + length - 3])
    prefix = {p: base[i] for i, p in enumerate(non_gap)}
    _, _, vf = v_identity(panel, gene, prefix, loop)
    return vf


def test_generate_emits_gate_passing_framework_preserving_children():
    """Edit children of short-CDR3 parents keep V identity >= 82% per gene."""
    panel = real_panel()
    requirements = rules.pinned_config(rules.ACTIVATION_EPOCH)['nanobody_requirements']
    for gene in GENES:
        parent1 = propose(panel, gene, 5, seed=9, ordinal=0)['sequence']
        pos = _safe_framework_position(parent1)
        parent2 = _variant(parent1, pos, 'S')
        # A same-length public background with a third residue at the same
        # position supplies the non-parent substitution (mirrors the archive).
        background = _variant(parent1, pos, 'T')
        children = edit.generate([parent1, parent2],
                                 background_sequences=[parent1, parent2, background])
        assert children, gene
        assert len({c['sequence'] for c in children}) == len(children)  # deduped
        for child in children:
            seq = child['sequence']
            assert seq != parent1 and seq != parent2
            assert child['source'] == 'edit_generation'
            assert child['proposal_lane'] in {
                'framework_single_substitution', 'n_terminal_single_substitution',
                'c_terminal_single_substitution', 'framework_haplotype',
                'n_terminal_haplotype', 'c_terminal_haplotype',
                'lineage_cdr_single_substitution', 'lineage_terminal_single_substitution',
                'lineage_haplotype'}
            # Same cheap gates the worker's sequence_phase applies to every lane.
            assert gates.check_sequence(seq, requirements) == []
            assert hydrophobicity.check(seq)[1] is True
            # The framework lane preserves the parent's real framework; the
            # lineage lane only touches CDR/terminal positions, so every child
            # stays above the >= 82% V-identity gate (no per-gene relaxation).
            vf = _v_identity_of(panel, gene, seq, 5)
            assert vf >= 0.82, (gene, child['proposal_lane'], vf)


def test_generate_never_returns_a_parent_and_dedups():
    panel = real_panel()
    parent1 = propose(panel, 'IGHV3-23', 5, seed=9, ordinal=0)['sequence']
    pos = _safe_framework_position(parent1)
    parent2 = _variant(parent1, pos, 'S')
    background = _variant(parent1, pos, 'T')
    children = edit.generate([parent1, parent2, parent1],
                             background_sequences=[parent1, parent2, background])
    parents = {parent1, parent2}
    emitted = {c['sequence'] for c in children}
    assert emitted.isdisjoint(parents)
    assert len(emitted) == len(children)


def test_generate_empty_parents_is_empty():
    assert edit.generate([]) == []
    assert edit.generate(['']) == []


def test_edit_generation_settings_defaults_off_and_validates():
    assert worker.DEFAULT_EDIT_LIMIT == 300
    assert worker.edit_generation_settings({}) == {'enabled': False, 'limit': 300}
    assert worker.edit_generation_settings({'edit_generation': {'enabled': True}}) == \
        {'enabled': True, 'limit': 300}
    assert worker.edit_generation_settings({'edit_generation': {'enabled': True,
                                                                 'limit': 40}})['limit'] == 40
    with pytest.raises(ValueError, match='enabled'):
        worker.edit_generation_settings({'edit_generation': {'enabled': 'yes'}})
    with pytest.raises(ValueError, match='limit'):
        worker.edit_generation_settings({'edit_generation': {'enabled': True,
                                                              'limit': -1}})
    # limit=0 used to mean "unlimited"; an enabled lane must now be bounded.
    with pytest.raises(ValueError, match='limit'):
        worker.edit_generation_settings({'edit_generation': {'enabled': True,
                                                              'limit': 0}})
    with pytest.raises(ValueError, match='limit'):
        worker.edit_generation_settings({'edit_generation': {'enabled': True,
                                                              'limit': True}})
    # A disabled lane never consumes the limit, so a legacy 0 stays accepted.
    assert worker.edit_generation_settings(
        {'edit_generation': {'enabled': False, 'limit': 0}})['enabled'] is False


@pytest.mark.parametrize('policy', [None, '', 'chemistry', [], True])
def test_edit_policy_rejects_unknown_values_even_when_disabled(policy):
    with pytest.raises(ValueError, match='policy'):
        worker.edit_generation_settings({'edit_generation': {'policy': policy}})
    with pytest.raises(ValueError, match='policy'):
        edit.generate([], policy=policy)


def test_chemistry_requires_bounded_lane_and_annotated_parent_context():
    with pytest.raises(ValueError, match='limit'):
        worker.edit_generation_settings({'edit_generation': {
            'enabled': True, 'policy': edit.CHEMISTRY_POLICY, 'limit': 0}})
    with pytest.raises(ValueError, match='qualified'):
        edit.generate([], policy=edit.CHEMISTRY_POLICY)
    settings = {'enabled': True, 'limit': 60, 'policy': edit.CHEMISTRY_POLICY}
    assert worker.edit_generation_settings({'edit_generation': settings}) == settings


@pytest.mark.parametrize('policy,expected', [
    # Ordered sequence digests measured from the immutable pre-change sources:
    # d95cd301 baseline and 0aa2adb chemistry prototype. All 60 chemistry rows
    # are the exact inputs to the real 60-annotation/37-novel/8-native experiment.
    (edit.BALANCED_POLICY, '628534da4a128d733c72b3d51057eaf38f87686ed34868c14cddd748e1c2bcce'),
    (edit.CHEMISTRY_POLICY, '4917b2840a51de813daac2924dadbb007166db40c8e02caefa5094b0894cd7e2'),
])
def test_real_parent_policy_sequences_reproduce_frozen_experiments(policy, expected):
    import hashlib
    fixture, _, contexts = _recorded_hmm_budgets()
    cdrs = {row['sequence']: row['cdrs'] for row in fixture['parents']}
    options = dict(limit=60, novelty_aware=True, parent_cdrs=cdrs,
                   parent_identity_budgets=contexts)
    rows = edit.generate(list(contexts), **options, policy=policy)
    assert len(rows) == 60
    assert hashlib.sha256('\n'.join(row['sequence'] for row in rows).encode()).hexdigest() == expected
    assert rows == edit.generate(list(contexts), **options, policy=policy)
    assert all(row['edit_policy'] == policy for row in rows)
    assert [row['parent_sequence'] for row in rows[:3]] == list(contexts)
    if policy == edit.BALANCED_POLICY:
        assert rows == edit.generate(list(contexts), **options)
    else:
        assert all(row['proposal_lane'].endswith(edit.CHEMISTRY_POLICY) for row in rows)
        for row in rows:
            parent, child = row['parent_sequence'], row['sequence']
            assert contexts[parent].accepts(child)
            assert all(a == b or b in edit.conservative_substitutions(a)
                       for a, b in zip(parent, child))
    excluded = {rules.sequence_hash(row['sequence']) for row in rows}
    continued = edit.generate(list(contexts), **options, policy=policy, excluded_hashes=excluded)
    assert len(continued) == 60
    assert all(rules.sequence_hash(row['sequence']) not in excluded for row in continued)
    assert [row['parent_sequence'] for row in continued[:3]] == list(contexts)


def test_haplotypes_skip_donors_with_non_standard_residues():
    """A haplotype copies donor residues verbatim; one bad residue drops it all."""
    parent = 'ACDEFGHIKLMNPQRSTVWY' * 2
    allowed = [[residue] for residue in parent]
    bad = list(parent)
    bad[3], bad[8] = 'X', 'A'      # one non-standard + one standard change
    good = list(parent)
    good[3], good[8] = 'A', 'A'    # two standard changes
    bad, good = ''.join(bad), ''.join(good)
    positions = set(range(len(parent)))

    framework = edit.supported_framework_candidates(
        parent, allowed=allowed, background_sequences=[bad, good],
        cdr_positions=set(), protected=set(), terminal_width=len(parent))
    lineage = edit.supported_lineage_candidates(
        parent, allowed=allowed, background_sequences=[bad, good],
        cdr_positions=positions, protected=set(), terminal_width=len(parent))
    for name, candidates in (('framework', framework), ('lineage', lineage)):
        emitted = {sequence for sequence, _ in candidates}
        assert good in emitted, name
        assert bad not in emitted, name
        assert all(set(sequence) <= gates.ALLOWED_AAS for sequence in emitted), name


def test_generate_skips_and_counts_malformed_parents():
    panel = real_panel()
    parent = propose(panel, 'IGHV3-23', 5, seed=9, ordinal=0)['sequence']
    pos = _safe_framework_position(parent)
    sibling = _variant(parent, pos, 'S')
    malformed = 'G' * len(parent)       # no framework cysteine -> find_regions raises
    with pytest.raises(ValueError):
        find_regions(malformed)
    stats = {}
    children = edit.generate([malformed, parent], background_sequences=[sibling],
                             stats=stats)
    assert stats == {'malformed_parents': 1}
    assert children and all(c['parent_sequence'] == parent for c in children)
    clean = {}
    edit.generate([parent], background_sequences=[sibling], stats=clean)
    assert clean == {'malformed_parents': 0}


def test_generate_malformed_parent_does_not_seed_allowed_table():
    panel = real_panel()
    parent = propose(panel, 'IGHV3-23', 5, seed=9, ordinal=0)['sequence']
    pos = _safe_framework_position(parent)
    sibling = _variant(parent, pos, 'S')
    malformed = 'G' * len(parent)
    children = edit.generate([malformed, parent], background_sequences=[sibling])
    # Only the sibling's residue is supported by a located parent/background;
    # the malformed parent's G must not widen the per-position allowed table.
    assert children
    for child in children:
        diff = [(a, s) for a, p, s in zip(child['sequence'], parent, sibling) if a != p]
        assert all(a == s for a, s in diff), child
    assert ({c['sequence'] for c in children}
            == {c['sequence'] for c in edit.generate([parent], background_sequences=[sibling])})


def _lane_rich_pool(parent):
    """Background with one distinct residue per FR3 and CDR1 position, so the
    framework and lineage lanes each have several single-substitution children."""
    regions = find_regions(parent)
    positions = (list(range(regions.cdr2[1] + 2, regions.cdr3[0] - 2, 3))[:6]
                 + list(range(regions.cdr1[0], regions.cdr1[1]))[:6])
    return [_variant(parent, p, 'T' if parent[p] != 'T' else 'S') for p in positions]


def test_generate_limit_alternates_framework_and_lineage_lanes():
    panel = real_panel()
    parent = propose(panel, 'IGHV3-23', 5, seed=9, ordinal=0)['sequence']
    background = _lane_rich_pool(parent)
    everything = edit.generate([parent], background_sequences=background)
    lanes = {c['proposal_lane'].startswith('lineage_') for c in everything}
    assert lanes == {True, False}  # both lanes produce children for this pool
    limited = edit.generate([parent], background_sequences=background, limit=4)
    assert [c['sequence'] for c in limited] == [c['sequence'] for c in everything[:4]]
    kinds = [c['proposal_lane'].startswith('lineage_') for c in limited]
    assert kinds == [False, True, False, True]  # framework no longer fills the cap alone


def test_generate_limit_counts_only_kept_candidates():
    panel = real_panel()
    parent = propose(panel, 'IGHV3-23', 5, seed=9, ordinal=0)['sequence']
    background = _lane_rich_pool(parent)
    everything = edit.generate([parent], background_sequences=background)
    assert len(everything) >= 12
    excluded = {rules.sequence_hash(c['sequence']) for c in everything[:4]}
    kept = edit.generate([parent], background_sequences=background, limit=4,
                         excluded_hashes=excluded)
    # Excluded hashes do not consume the cap: the next four fresh candidates come back.
    assert [c['sequence'] for c in kept] == [c['sequence'] for c in everything[4:8]]
    # A deterministic continuation that excludes every prior hash walks the whole
    # pool in fixed-size batches instead of exhausting after one attempt.
    seen, order = set(), []
    for _ in range(len(everything)):
        batch = edit.generate([parent], background_sequences=background, limit=4,
                              excluded_hashes=seen)
        if not batch:
            break
        order.extend(c['sequence'] for c in batch)
        seen |= {rules.sequence_hash(c['sequence']) for c in batch}
    assert order == [c['sequence'] for c in everything]
    # Excluding everything leaves nothing to emit, whatever the limit.
    assert edit.generate([parent], background_sequences=background, limit=4,
                         excluded_hashes=seen) == []


def _edit_runner(monkeypatch, tmp_path, bank_rows, archive_sequences=('AAA',)):
    panel = real_panel()
    monkeypatch.setitem(sys.modules, 'abnumber', types.SimpleNamespace(
        Chain=lambda sequence, **kw: types.SimpleNamespace(**{
            key + '_seq': sequence[slice(*getattr(find_regions(sequence), key))]
            for key in ('cdr1', 'cdr2', 'cdr3')})))
    monkeypatch.setattr(worker.rules, 'pinned_config', lambda epoch:
                        {'nanobody_requirements': {'min_human_v_identity': 82}})
    from nanobody_opt import pr100_v_identity_budget
    monkeypatch.setattr(pr100_v_identity_budget, 'build', lambda parent, **kw: _wiring_budget(parent))
    monkeypatch.setattr(worker.pr100_filter, 'validate_annotation_runtime', lambda: None)
    monkeypatch.setattr(worker.pr100_filter, 'helper', lambda name:
                        types.SimpleNamespace(_anarci_panel=lambda species: (panel, None)))
    monkeypatch.setattr(worker.pr100_filter, '_annotate_uncached',
                        lambda sequence: {'fixture_annotation': True})
    monkeypatch.setattr(worker.pr100_filter, 'annotation_identity', lambda: 'a' * 64)
    monkeypatch.setattr(worker.rules, 'annotation_reason', lambda *a, **kw: '')
    monkeypatch.setattr(gates, 'check_sequence', lambda *a: False)

    def forbidden(*a, **kw):
        raise AssertionError('no scientific profiling or novelty in this lane test')
    monkeypatch.setattr(worker.pr100_filter, 'develop', forbidden)
    monkeypatch.setattr(worker, 'check_novelty', forbidden)

    reference_path = tmp_path / 'bank.json'
    reference_path.write_text(json.dumps({'rows': bank_rows}))
    runs = []

    def run(challenge, **overrides):
        directory = tmp_path / str(len(runs))
        directory.mkdir()
        rows = '\n'.join(archive_sequences)
        (directory / 'archive.csv').write_text(f'sequence\n{rows}\n')
        job = {'challenge': challenge, 'references': str(reference_path),
               'archive_sha256': worker.file_digest(directory / 'archive.csv'),
               'archive_fetched_at': 1, 'proposals': 4, **overrides}
        runs.append(job)
        return worker.sequence_phase(job, directory)

    return run


def test_worker_edit_lane_off_by_default_and_on_emits_gate_passing_children(
        monkeypatch, tmp_path):
    panel = real_panel()
    parent1 = propose(panel, 'IGHV3-23', 5, seed=9, ordinal=0)['sequence']
    pos = _safe_framework_position(parent1)
    parent2 = _variant(parent1, pos, 'S')
    background = _variant(_variant(parent1, pos, 'T'), 0, 'E')
    background2 = _variant(_variant(parent1, pos, 'N'), 0, 'E')
    bank_rows = [
        {'sequence': parent1, 'sequence_hash': rules.sequence_hash(parent1),
         'cells': ['IGHV3-23:5'], 'source': 'rival_bank_test'},
        {'sequence': parent2, 'sequence_hash': rules.sequence_hash(parent2),
         'cells': ['IGHV3-23:5'], 'source': 'rival_bank_test'},
    ]
    run = _edit_runner(monkeypatch, tmp_path, bank_rows,
                       archive_sequences=(background, background2))
    challenge = {'epoch': 25969, 'target': rules.TARGET,
                 'allowed_v': 'IGHV3-23', 'allowed_cdr3_length': 5}

    off = run(challenge)
    assert off['edit_generation_enabled'] is False
    assert off['edit_generation_candidates'] == 0
    assert off['edit_generation_source_counts'] == {}
    assert all(r.get('source') != 'edit_generation' for r in off['candidates'])

    on = run(challenge, edit_generation={'enabled': True})
    assert on['edit_generation_enabled'] is True
    assert on['edit_generation_candidates'] >= 2
    assert set(on['edit_generation_source_counts']) <= {
        'novelty_aware_multi_edit',
        'framework_single_substitution', 'n_terminal_single_substitution',
        'c_terminal_single_substitution', 'framework_haplotype',
        'n_terminal_haplotype', 'c_terminal_haplotype',
        'lineage_cdr_single_substitution', 'lineage_terminal_single_substitution',
        'lineage_haplotype'}
    edit_candidates = [r for r in on['candidates'] if r.get('source') == 'edit_generation']
    assert edit_candidates
    assert all(r['requested_v'] == 'IGHV3-23' and r['requested_cdr3_length'] == 5
               for r in edit_candidates)
    # Every emitted edit child preserves the parent framework (V identity >= 82%).
    for r in edit_candidates:
        vf = _v_identity_of(panel, 'IGHV3-23', r['sequence'], 5)
        assert vf >= 0.82, (r['proposal_lane'], vf)


def test_worker_edit_lane_limit_counts_only_non_excluded_candidates(monkeypatch, tmp_path):
    panel = real_panel()
    parent = propose(panel, 'IGHV3-23', 5, seed=9, ordinal=0)['sequence']
    archive = [_variant(s, 0, 'E') for s in _lane_rich_pool(parent)]
    bank_rows = [{'sequence': parent, 'sequence_hash': rules.sequence_hash(parent),
                  'cells': ['IGHV3-23:5'], 'source': 'rival_bank_test'}]
    run = _edit_runner(monkeypatch, tmp_path, bank_rows, archive_sequences=tuple(archive))
    challenge = {'epoch': 25969, 'target': rules.TARGET,
                 'allowed_v': 'IGHV3-23', 'allowed_cdr3_length': 5}
    settings = {'enabled': True, 'limit': 3}
    first = run(challenge, edit_generation=settings)
    assert first['edit_generation_candidates'] == 3
    assert first['excluded_generated'] == 0
    # A continuation excluding every prior edit child must still receive a full
    # ``limit`` of fresh ones; the cap is applied after exclusion, not before.
    prior = [r for r in first['candidates'] if r.get('source') == 'edit_generation']
    assert prior
    second = run(challenge, edit_generation=settings,
                 excluded_sequence_hashes=[rules.sequence_hash(c['sequence']) for c in prior])
    assert second['edit_generation_candidates'] == 3
    assert second['excluded_generated'] == 0
    assert not set(second['generated_sequence_hashes']) & {
        rules.sequence_hash(c['sequence']) for c in prior}


def test_worker_rejects_non_standard_residues_before_annotation(monkeypatch, tmp_path):
    panel = real_panel()
    parent1 = propose(panel, 'IGHV3-23', 5, seed=9, ordinal=0)['sequence']
    pos = _safe_framework_position(parent1)
    parent2 = _variant(parent1, 0, 'X')    # inherited by every child of parent2
    bank_rows = [
        {'sequence': sequence, 'sequence_hash': rules.sequence_hash(sequence),
         'cells': ['IGHV3-23:5'], 'source': 'rival_bank_test'}
        for sequence in (parent1, parent2)]
    run = _edit_runner(monkeypatch, tmp_path, bank_rows,
                       archive_sequences=(_variant(parent1, pos, 'T'),
                                          _variant(parent1, pos, 'N')))
    annotated = []

    def annotate(sequence):
        # The real annotator crashes on a residue outside the alphabet.
        assert set(sequence) <= gates.ALLOWED_AAS, sequence
        annotated.append(sequence)
        return {'fixture_annotation': True}

    monkeypatch.setattr(worker.pr100_filter, '_annotate_uncached', annotate)
    monkeypatch.setattr(gates, 'check_sequence', lambda seq, cfg:
                        ['invalid_aas'] if set(seq) - gates.ALLOWED_AAS else [])
    challenge = {'epoch': 25969, 'target': rules.TARGET,
                 'allowed_v': 'IGHV3-23', 'allowed_cdr3_length': 5}

    result = run(challenge, edit_generation={'enabled': True, 'limit': 50})
    assert result['edit_generation_stats'].get('parent_annotation_failed', 0) == 1
    assert annotated and all(set(s) <= gates.ALLOWED_AAS for s in annotated)
    assert all(set(r['sequence']) <= gates.ALLOWED_AAS for r in result['candidates'])
    assert result['edit_generation_stats'] == {'malformed_parents': 1, 'parent_annotation_failed': 1}
    assert result['edit_generation_candidates'] <= 50


def test_mixed_length_parents_each_receive_round_robin_slot():
    panel = real_panel()
    parents = [propose(panel, 'IGHV3-23', length, seed=9, ordinal=0)['sequence']
               for length in (5, 16)]
    background = []
    for parent in parents:
        pos = _safe_framework_position(parent)
        background.extend([_variant(parent, pos, 'S'), _variant(parent, pos, 'T')])
    children = edit.generate(parents, background_sequences=background, limit=2)
    assert [c['parent_sequence'] for c in children] == parents
    assert {len(c['sequence']) for c in children} == {len(p) for p in parents}


def test_zero_edit_limit_emits_no_work():
    parent = propose(real_panel(), 'IGHV3-23', 5, seed=9, ordinal=0)['sequence']
    assert edit.generate([parent], background_sequences=_lane_rich_pool(parent), limit=0) == []


def test_worker_never_sends_exact_archive_repeats_to_annotation(monkeypatch, tmp_path):
    parent = propose(real_panel(), 'IGHV3-23', 5, seed=9, ordinal=0)['sequence']
    row = {'sequence': parent, 'sequence_hash': rules.sequence_hash(parent),
           'cells': ['IGHV3-23:5'], 'source': 'test'}
    run = _edit_runner(monkeypatch, tmp_path, [row], archive_sequences=(parent,))
    calls = []
    monkeypatch.setattr(worker.pr100_filter, '_annotate_uncached',
        lambda sequence: calls.append(sequence) or {'fixture_annotation': True})
    c = {'epoch': 25969, 'target': rules.TARGET,
         'allowed_v': 'IGHV3-23', 'allowed_cdr3_length': 5}
    result = run(c)
    assert parent not in calls
    assert parent not in {r['sequence'] for r in result['candidates']}
    assert result['historical_exact_reuse_blocked'] == 1


def test_multi_edit_arm_strict_parent_identity_and_immutable_framework():
    panel = real_panel()
    for length in (5, 6, 12, 16, 23):
        parent = propose(panel, 'IGHV3-23', length, seed=9, ordinal=0)['sequence']
        regions = find_regions(parent)
        cdrs = {key: parent[slice(*getattr(regions, key))] for key in ('cdr1', 'cdr2', 'cdr3')}
        options = {'novelty_aware': True, 'parent_cdrs': {parent: cdrs},
                   'parent_identity_budgets': {parent: _wiring_budget(parent)}}
        rows = edit.generate([parent], limit=32, **options)
        assert len(rows) == 32
        mutable = set(regions.cdr_positions())
        c3 = list(range(*regions.cdr3))
        for row in rows:
            child = row['sequence']
            assert row['proposal_lane'] == 'novelty_aware_multi_edit'
            assert sum(a == b for a, b in zip(parent, child)) / len(parent) < 0.95
            assert sum(parent[i] == child[i] for i in c3) / len(c3) < 0.85
            assert all(child[i] == parent[i] for i in range(len(parent))
                       if i not in mutable or parent[i] == 'C' or i in c3[-3:])
            assert child.count('C') == parent.count('C')
        assert rows == edit.generate([parent], limit=32, **options)


def test_multi_edit_requires_annotation_and_retains_mixed_length_parents():
    parents = [propose(real_panel(), 'IGHV3-23', length, seed=9, ordinal=0)['sequence']
               for length in (5, 16)]
    cdrs = {p: {key: p[slice(*getattr(find_regions(p), key))]
                for key in ('cdr1', 'cdr2', 'cdr3')} for p in parents}
    assert edit.generate(parents, limit=2, novelty_aware=True) == []
    assert edit.generate(parents, limit=2, novelty_aware=True, parent_cdrs=cdrs) == []
    rows = edit.generate(parents, limit=2, novelty_aware=True, parent_cdrs=cdrs,
                        parent_identity_budgets={p: _wiring_budget(p) for p in parents})
    assert [r['parent_sequence'] for r in rows] == parents


def _recorded_hmm_budgets():
    from pathlib import Path
    from nanobody_opt import pr100_v_identity_budget as budget
    fixture = json.loads((Path(__file__).parent / 'fixtures/pr100/parent_hmm_identity_26019.json').read_text())
    panels = {region: {name: dict(enumerate(reference, 1)) for name, reference in references.items()}
              for region, references in fixture['reference_alignments'].items()}
    contexts = {row['sequence']: budget.from_alignment(row['sequence'], allowed_gene='IGHV3-23',
        minimum_identity_pct=82, state_vectors=row['state_vectors'], details=row['details'], panels=panels)
        for row in fixture['parents']}
    return fixture, panels, contexts


def test_recorded_parent_mapping_matches_exact_pinned_validator_identity():
    import hashlib
    from pathlib import Path
    from nanobody_opt import pr100_filter
    fixture, panels, contexts = _recorded_hmm_budgets()
    assert fixture['runtime'] == pr100_filter.ANNOTATION_RUNTIME
    assert fixture['helper_sha256'] == hashlib.sha256(
        (Path(__file__).parents[1] / 'utils/germline_identity.py').read_bytes()).hexdigest()
    assert hashlib.sha256(json.dumps(fixture['reference_alignments'], sort_keys=True,
        separators=(',', ':')).encode()).hexdigest() == fixture['panel_sha256']
    assert len(panels['V']) == 249 and len(panels['J']) == 14
    for row in fixture['parents']:
        context = contexts[row['sequence']]
        assert context.accepts(row['sequence'])
        best = max(score for _, score in context.scores(row['sequence']))
        assert str(best) == row['annotation']['germline']['V']['identity_fraction'] == '41/49'
        assert float(100 * best) == row['annotation']['germline']['V']['identity_pct']
        # 82 matches out of 98; the >=82% gate needs 81, leaving one match.
        assert len(context.anchor) == 98
        assert sum(row['sequence'][i] == aa for i, aa in context.anchor) == 82


def test_recorded_parent_budget_edits_preserve_matches_j_framework_and_cysteines():
    from fractions import Fraction
    fixture, _, contexts = _recorded_hmm_budgets()
    cdrs = {row['sequence']: row['cdrs'] for row in fixture['parents']}
    children = edit.generate(list(contexts), limit=60, novelty_aware=True,
        parent_cdrs=cdrs, parent_identity_budgets=contexts)
    assert len(children) == 60
    assert [r['parent_sequence'] for r in children[:3]] == list(contexts)
    for row in children:
        parent, child = row['parent_sequence'], row['sequence']
        context = contexts[parent]
        spans = edit._annotated_spans(parent, cdrs[parent])
        mutable = set().union(*spans.values()) - context.protected_j
        changes = [i for i, (a, b) in enumerate(zip(parent, child)) if a != b]
        assert len(changes) == 7
        assert sum(i in spans['cdr3'] for i in changes) == 3
        assert set(changes) <= mutable
        assert all(parent[i] != 'C' and child[i] != 'C' for i in changes)
        assert all(context.delta(i, child[i]) >= 0 for i in changes)
        assert context.accepts(child)
        assert context.scores(child) == [(name, Fraction(sum(i is not None and child[i] == aa
            for i, aa in positions), len(positions))) for name, positions in context.alleles]
        assert sum(parent[i] == child[i] for i in spans['cdr3']) / 19 < .85
        assert sum(a == b for a, b in zip(parent, child)) / 126 < .95
    excluded = {rules.sequence_hash(row['sequence']) for row in children}
    next_children = edit.generate(list(contexts), limit=60, novelty_aware=True,
        parent_cdrs=cdrs, parent_identity_budgets=contexts, excluded_hashes=excluded)
    assert len(next_children) == 60
    assert all(rules.sequence_hash(row['sequence']) not in excluded for row in next_children)


def test_hmm_insertion_and_germline_mismatch_do_not_spend_identity():
    from nanobody_opt.pr100_v_identity_budget import from_alignment
    # Sequence indices deliberately differ from match-state and IMGT numbers.
    # Insertion index1 contributes to neither V denominator nor match count.
    panels = {'V': {'IGHV3-23*01': {1: 'A', 2: 'S', 3: '-'}},
              'J': {'IGHJ4*01': {3: 'Y'}}}
    model = from_alignment('ATGY', allowed_gene='IGHV3-23', minimum_identity_pct=50,
        state_vectors=[[((1, 'm'), 0), ((1, 'i'), 1), ((2, 'm'), 2), ((3, 'm'), 3)]],
        details=[{'chain_type': 'H'}], panels=panels)
    assert model.delta(0, 'G') == -1
    assert model.delta(1, 'G') == 0
    assert model.delta(2, 'S') == 1
    assert model.delta(2, 'N') == 0
    assert model.restoration(2) == 'S'
    assert model.protected_j == {3}
    assert model.accepts('AGNY')
    assert not model.accepts('AGNN')  # J cannot be changed to fill a novelty quota.


def test_budget_checks_top_gene_ties_and_missing_reference_positions():
    from nanobody_opt.pr100_v_identity_budget import from_alignment
    panels = {'V': {'IGHV3-23*01': {1: 'A', 2: 'S'},
                    'IGHV3-7*01': {1: 'A', 2: 'T'}}, 'J': {'IGHJ4*01': {3: 'Y'}}}
    options = {'allowed_gene': 'IGHV3-23', 'minimum_identity_pct': 50,
               'state_vectors': [[((1, 'm'), 0), ((2, 'm'), 1), ((3, 'm'), 2)]],
               'details': [{'chain_type': 'H'}], 'panels': panels}
    model = from_alignment('AGY', **options)
    assert model.accepts('AGY')  # tied allowed genes remain valid.
    assert not model.accepts('ATY')  # V% alone does not preserve the epoch gene.
    options['state_vectors'] = [[((1, 'm'), 0), ((3, 'm'), 2)]]
    missing = from_alignment('AGY', **options)
    assert max(score for _, score in missing.scores('AGY')) == .5
    with pytest.raises(ValueError, match='current V identity challenge'):
        from_alignment('AGY', **{**options, 'minimum_identity_pct': 82})
    with pytest.raises(ValueError, match='current V identity challenge'):
        from_alignment('AGY', **{**options, 'allowed_gene': 'IGHV3-66'})


def test_budget_runtime_uses_exact_hmmer_and_j_pass(monkeypatch):
    from nanobody_opt import pr100_filter, pr100_v_identity_budget as budget
    fixture, panels, contexts = _recorded_hmm_budgets()
    row = fixture['parents'][0]
    calls = []
    alignments = [(None, row['state_vectors'], row['details'])]
    def run_hmmer(records, **options):
        calls.append(('hmmer', records, options))
        return alignments
    def check_for_j(records, observed, scheme):
        calls.append(('j', records, observed is alignments, scheme))
    monkeypatch.setitem(sys.modules, 'anarci', types.ModuleType('anarci'))
    monkeypatch.setitem(sys.modules, 'anarci.anarci', types.SimpleNamespace(
        run_hmmer=run_hmmer, check_for_j=check_for_j))
    monkeypatch.setattr(pr100_filter, 'validate_annotation_runtime', lambda: calls.append('runtime'))
    monkeypatch.setattr(pr100_filter, 'helper', lambda name:
        types.SimpleNamespace(_anarci_panel=lambda species: (panels, fixture['panel_sha256'])))
    assert budget.build(row['sequence'], allowed_gene='IGHV3-23', minimum_identity_pct=82) == contexts[row['sequence']]
    assert calls[0] == 'runtime'
    assert calls[1] == ('hmmer', [('query', row['sequence'])], {'ncpu': 1, 'hmmer_species': None})
    assert calls[2] == ('j', [('query', row['sequence'])], True, 'imgt')


def test_worker_builds_budgets_for_exact_current_challenge(monkeypatch, tmp_path):
    from nanobody_opt import pr100_v_identity_budget
    parent = propose(real_panel(), 'IGHV3-23', 5, seed=9, ordinal=0)['sequence']
    bank = [{'sequence': parent, 'sequence_hash': rules.sequence_hash(parent),
             'cells': ['IGHV3-23:5'], 'source': 'wiring_fixture'}]
    run = _edit_runner(monkeypatch, tmp_path, bank)
    calls = []
    def build(sequence, **options):
        calls.append((sequence, options))
        return _wiring_budget(sequence)
    monkeypatch.setattr(pr100_v_identity_budget, 'build', build)
    challenge = {'epoch': 25969, 'target': rules.TARGET,
                 'allowed_v': 'IGHV3-23', 'allowed_cdr3_length': 5}
    result = run(challenge, edit_generation={'enabled': True, 'limit': 4})
    assert calls == [(parent, {'allowed_gene': 'IGHV3-23', 'minimum_identity_pct': 82})]
    assert result['challenge'] == challenge
    assert result['edit_generation_candidates'] == 4


def test_worker_propagates_bounded_chemistry_policy_and_lineage(monkeypatch, tmp_path):
    fixture, _, contexts = _recorded_hmm_budgets()
    bank = [{'sequence': row['sequence'], 'sequence_hash': row['sequence_hash'],
             'cells': ['IGHV3-23:19'], 'source': 'recorded_parent_wiring_fixture'}
            for row in fixture['parents']]
    run = _edit_runner(monkeypatch, tmp_path, bank)
    from nanobody_opt import pr100_v_identity_budget
    cdrs = {row['sequence']: row['cdrs'] for row in fixture['parents']}
    monkeypatch.setattr(pr100_v_identity_budget, 'build', lambda sequence, **kw: contexts[sequence])
    monkeypatch.setitem(sys.modules, 'abnumber', types.SimpleNamespace(Chain=lambda sequence, **kw:
        types.SimpleNamespace(**{key + '_seq': value for key, value in cdrs[sequence].items()})))
    result = run({'epoch': 26019, 'target': rules.TARGET,
                  'allowed_v': 'IGHV3-23', 'allowed_cdr3_length': 19},
                 edit_generation={'enabled': True, 'limit': 6, 'policy': edit.CHEMISTRY_POLICY})
    assert result['edit_generation_policy'] == edit.CHEMISTRY_POLICY
    assert result['edit_generation_candidates'] == 6
    rows = [row for row in result['candidates'] if row.get('source') == 'edit_generation']
    assert len(rows) == 6
    assert all(row['edit_policy'] == edit.CHEMISTRY_POLICY for row in rows)
    assert all(row['proposal_lane'] == 'novelty_aware_multi_edit_chemistry_conservative_v3' for row in rows)
    # Bank merit sorting decides parent order; each receives its bounded share.
    assert {parent: sum(row['parent_sequence'] == parent for row in rows)
            for parent in contexts} == {parent: 2 for parent in contexts}


def test_recorded_local_alignment_rejects_cdr3_only_hamming_novel_child():
    import hashlib
    from pathlib import Path
    from nanobody_opt.check_validator_novelty import duplicate_reason
    fixture = json.loads((Path(__file__).parent /
        'fixtures/pr100/parent_edit_local_alignment_counterexample.json').read_text())
    assert hashlib.sha256((Path(__file__).parents[1] /
        'NOVA-nanobody-filter/metanano/utils/alignment.py').read_bytes()).hexdigest() == fixture['alignment_source_sha256']
    query, parent = fixture['query'], fixture['parent']
    assert len(query) == len(parent)
    assert sum(a == b for a, b in zip(query, parent)) / len(parent) < .95
    alignment = list(zip(fixture['aligned_query'], fixture['aligned_target'], strict=True))
    identity = sum(a == b and a != '-' for a, b in alignment) / len(alignment)
    assert identity == fixture['identity'] >= .95
    assert duplicate_reason(identity, {'CDR3': .5}) == 'near-identical sequence'
