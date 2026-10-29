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
    assert worker.edit_generation_settings({}) == {'enabled': False, 'limit': 0}
    assert worker.edit_generation_settings({'edit_generation': {'enabled': True}}) == \
        {'enabled': True, 'limit': 0}
    assert worker.edit_generation_settings({'edit_generation': {'enabled': True,
                                                                 'limit': 40}})['limit'] == 40
    with pytest.raises(ValueError, match='enabled'):
        worker.edit_generation_settings({'edit_generation': {'enabled': 'yes'}})
    with pytest.raises(ValueError, match='limit'):
        worker.edit_generation_settings({'edit_generation': {'enabled': True,
                                                              'limit': -1}})


def _edit_runner(monkeypatch, tmp_path, bank_rows, archive_sequences=('AAA',)):
    panel = real_panel()
    monkeypatch.setitem(sys.modules, 'abnumber', types.SimpleNamespace(
        Chain=lambda sequence, **kw: types.SimpleNamespace(cdr3_seq=sequence[96:-11])))
    monkeypatch.setattr(worker.rules, 'pinned_config', lambda epoch: {'nanobody_requirements': {}})
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
    background = _variant(parent1, pos, 'T')
    background2 = _variant(parent1, pos, 'N')
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
