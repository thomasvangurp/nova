"""Leaderboard graft lane: provenance, substitutions, divergence and de-novo recipe."""
import json
import sys
import types
from pathlib import Path

import pytest

from nanobody_opt import pr100_graft_generation as graft
from nanobody_opt import pr100_segment_generation as segments
from nanobody_opt import rules

CHALLENGE = {'epoch': 26030, 'target': 'Q9NZQ7', 'allowed_v': 'IGHV3-74',
             'allowed_cdr3_length': 16}


def cdr3_positions(length):
    base = list(range(105, 118))  # 105..117
    if length <= 13:
        return [(n, '') for n in base[:length]]
    positions = [(n, '') for n in base]
    extra = length - 13
    letters = [chr(ord('A') + i) for i in range(extra)]
    result = []
    for n, letter in positions:
        result.append((n, letter))
        if n == 111:
            for x in letters:
                result.append((111, x))
    return result


def build_donor(*, cdr3, cdr1='R' * 12, cdr2='S' * 10, fr1=None, fr2=None, fr3=None,
                fr4=None, v='IGHV3-74', cell_length=16, evidence=()):
    fr1 = fr1 if fr1 is not None else ('F' * 22 + 'C' + 'F' * 3)
    fr2 = fr2 if fr2 is not None else 'G' * 17
    fr3 = fr3 if fr3 is not None else ('H' * 38 + 'C')
    fr4 = fr4 if fr4 is not None else 'W' * 11
    regions = {
        'fr1': (fr1, [(n, '') for n in range(1, 27)]),
        'cdr1': (cdr1, [(n, '') for n in range(27, 39)]),
        'fr2': (fr2, [(n, '') for n in range(39, 56)]),
        'cdr2': (cdr2, [(n, '') for n in range(56, 66)]),
        'fr3': (fr3, [(n, '') for n in range(66, 105)]),
        'cdr3': (cdr3, cdr3_positions(len(cdr3))),
        'fr4': (fr4, [(n, '') for n in range(118, 129)]),
    }
    order = ('fr1', 'cdr1', 'fr2', 'cdr2', 'fr3', 'cdr3', 'fr4')
    seq = ''.join(regions[name][0] for name in order)
    numbered = []
    for name in order:
        sequence, positions = regions[name]
        numbered.extend({'number': n, 'letter': letter, 'aa': aa}
                        for (n, letter), aa in zip(positions, sequence))
    return {'sequence': seq, 'sequence_hash': rules.sequence_hash(seq),
            'cells': [f'{v}:{cell_length}'], 'quality_evidence': list(evidence),
            'numbered': numbered}


def prepared(donor):
    return segments.prepare(donor, donor['numbered'])


@pytest.fixture
def abnumber(monkeypatch):
    """A deterministic fake abnumber.Chain backed by the synthetic donor rows."""
    lookup = {}

    class Chain:
        def __init__(self, seq, **kwargs):
            self.row = lookup[seq]
            self.cdr3_seq = ''.join(p['aa'] for p in self.row['numbered']
                                    if 105 <= p['number'] <= 117)

        def __iter__(self):
            return iter((types.SimpleNamespace(number=p['number'], letter=p['letter']), p['aa'])
                        for p in self.row['numbered'])

    monkeypatch.setitem(sys.modules, 'abnumber', types.SimpleNamespace(Chain=Chain))
    return lookup


def test_settings_default_off_and_strict_bounds():
    assert graft.settings({}) == {'enabled': False, 'limit': 300, 'cdr3_donors': 8,
                                  'framework_donors': 8, 'donor_slots': 64}
    assert not graft.settings({})['enabled']
    for value in [None, True, {'enabled': 1}, {'limit': 0}, {'limit': 4801},
                  {'cdr3_donors': 0}, {'cdr3_donors': 65}, {'framework_donors': 65},
                  {'donor_slots': 1}, {'donor_slots': 129}, {'unknown': True},
                  {'limit': True}]:
        with pytest.raises(ValueError):
            graft.settings({'graft_generation': value})
    assert graft.settings({'graft_generation': {'enabled': True}})['enabled'] is True
    assert graft.settings({'graft_generation': {'enabled': True, 'limit': 16}})['limit'] == 16


def test_verbatim_graft_provenance_reconstructs(abnumber):
    cd = build_donor(cdr3='R' * 16)  # no neutral-swappable residues
    fd = build_donor(cdr3='Y' * 16)
    abnumber[cd['sequence']] = cd
    abnumber[fd['sequence']] = fd
    rows = graft.generate([cd], [prepared(fd)], CHALLENGE, limit=10, modes=('verbatim',))
    assert len(rows) == 1
    row = rows[0]
    assert row['graft_mode'] == 'verbatim'
    assert len(row['segment_provenance']) == 7
    assert len({p['source_hash'] for p in row['segment_provenance']}) == 2
    assert row['graft_neutral_subs'] == 0
    refs = {cd['sequence_hash']: cd['sequence'], fd['sequence_hash']: fd['sequence']}
    assert graft.reconstruct_with_edits(row, refs) == row['sequence']


def test_three_substitutions_identity_offsets_and_residues():
    cdr3 = 'SAG' + 'R' * 13
    edited, edits = graft.neutral_substitutions(cdr3)
    assert edited == 'TSA' + 'R' * 13
    identity = sum(a == b for a, b in zip(edited, cdr3)) / len(cdr3)
    assert identity == 13 / 16 == pytest.approx(0.8125)
    assert 0.80 <= identity < 0.85
    assert len(edits) == 3
    for edit in edits:
        assert edit['kind'] == 'neutral_cdr3_substitution'
        assert edit['from'] in graft.NEUTRAL_SWAPS
        assert edit['to'] == graft.NEUTRAL_SWAPS[edit['from']]
        assert 0 <= edit['child_offset'] < len(cdr3) - 3


def test_divergence_gate_drops_divergent_and_keeps_conserved(abnumber):
    divergent_cd = build_donor(cdr3='R' * 16, cdr1='R' * 12)
    divergent_fd = build_donor(cdr3='Y' * 16, cdr1='K' * 12)
    abnumber[divergent_cd['sequence']] = divergent_cd
    abnumber[divergent_fd['sequence']] = divergent_fd
    stats = {}
    assert graft.generate([divergent_cd], [prepared(divergent_fd)], CHALLENGE,
                          modes=('verbatim',), stats=stats) == []
    assert stats['divergence_drops'] == 1

    conserved_cd = build_donor(cdr3='R' * 16)
    conserved_fd = build_donor(cdr3='Y' * 16)
    abnumber[conserved_cd['sequence']] = conserved_cd
    abnumber[conserved_fd['sequence']] = conserved_fd
    rows = graft.generate([conserved_cd], [prepared(conserved_fd)], CHALLENGE,
                          modes=('verbatim',))
    assert len(rows) == 1


def test_exact_length_enforcement_skips_wrong_length_cdr3(abnumber):
    cd_bad = build_donor(cdr3='R' * 13, cell_length=13)
    cd_good = build_donor(cdr3='SAG' + 'R' * 13)
    fd = build_donor(cdr3='Y' * 16)
    for donor in (cd_bad, cd_good, fd):
        abnumber[donor['sequence']] = donor
    stats = {}
    rows = graft.generate([cd_bad, cd_good], [prepared(fd)], CHALLENGE,
                          modes=('verbatim',), stats=stats)
    assert stats['cdr3_length_mismatch'] == 1
    assert len(rows) == 1
    for row in rows:
        cdr3_part = next(p for p in row['segment_provenance'] if p.get('region') == 'cdr3')
        assert len(cdr3_part['sequence']) == CHALLENGE['allowed_cdr3_length']
        assert row['graft_cdr3_source_hash'] == cd_good['sequence_hash']


def test_generate_is_deterministic_and_order_invariant(abnumber):
    cd = build_donor(cdr3='SAG' + 'R' * 13)
    cd2 = build_donor(cdr3='T' * 13 + 'AAA')
    fd = build_donor(cdr3='Y' * 16)
    fd2 = build_donor(cdr3='N' * 16)
    for donor in (cd, cd2, fd, fd2):
        abnumber[donor['sequence']] = donor
    first = graft.generate([cd, cd2], [prepared(fd), prepared(fd2)], CHALLENGE,
                           limit=20, modes=('verbatim', 'de_novo'))
    second = graft.generate([cd, cd2], [prepared(fd), prepared(fd2)], CHALLENGE,
                            limit=20, modes=('verbatim', 'de_novo'))
    assert first == second
    reversed_order = graft.generate([cd2, cd], [prepared(fd2), prepared(fd)], CHALLENGE,
                                    limit=20, modes=('verbatim', 'de_novo'))
    assert first == reversed_order


def test_insufficient_exact_cell_donors_is_explicit(abnumber):
    cd = build_donor(cdr3='R' * 16)
    abnumber[cd['sequence']] = cd
    stats = {}
    assert graft.generate([cd], [prepared(cd)], CHALLENGE, stats=stats) == []
    assert stats['stop_reason'] == 'insufficient_exact_cell_donors'


def test_de_novo_cdr3_recipe_length_charge_and_determinism():
    for length in (5, 7, 10, 16, 23):
        value = graft.de_novo_cdr3(length, seed=42)
        assert len(value) == length
        assert value[:2] in graft._ANCHORS
        assert value[-3:] in graft._TAILS
        pos = sum(1 for c in value if c in 'KR')
        neg = sum(1 for c in value if c in 'DE')
        assert abs(pos - neg) <= 1
        assert graft.de_novo_cdr3(length, seed=42) == value
    assert graft.de_novo_cdr3(16, 42) != graft.de_novo_cdr3(16, 43)


def test_de_novo_cdr3_is_length_conditioned():
    # Short (5-10): no aromatic/His core or linker.
    for length in (5, 6, 8, 10):
        value = graft.de_novo_cdr3(length, seed=7)
        assert len(value) == length
        assert not any(token in value for token in ('FQP', 'PYR', 'YHF', 'QGS'))
    # Medium (11-14): a PYR/PYHF core.
    for length in (11, 12, 14):
        value = graft.de_novo_cdr3(length, seed=7)
        assert len(value) == length
        assert ('PYHF' in value) or ('PYR' in value)
        assert 'QGS' not in value
    # Long (15-18): an FQPR/PYR core plus charge network.
    for length in (15, 16, 18):
        value = graft.de_novo_cdr3(length, seed=7)
        assert len(value) == length
        assert ('FQP' in value) or ('PYR' in value)
        assert any(c in 'KR' for c in value) and any(c in 'DE' for c in value)
        assert 'QGS' not in value
    # Xlong (19-23): core plus QGSDSG/QGSGD linker plus charge network.
    for length in (19, 20, 23):
        value = graft.de_novo_cdr3(length, seed=7)
        assert len(value) == length
        assert 'QGS' in value
        assert any(c in 'KR' for c in value) and any(c in 'DE' for c in value)


def test_de_novo_graft_mode_length_charge_and_provenance(abnumber):
    cd = build_donor(cdr3='R' * 16)
    fd = build_donor(cdr3='Y' * 16)
    abnumber[cd['sequence']] = cd
    abnumber[fd['sequence']] = fd
    rows = graft.generate([cd], [prepared(fd)], CHALLENGE, limit=4, modes=('de_novo',))
    assert len(rows) == 1  # one framework/CDR3 pair -> one de-novo child
    row = rows[0]
    assert row['graft_mode'] == 'de_novo'
    cdr3_part = next(p for p in row['segment_provenance'] if p.get('region') == 'cdr3')
    assert cdr3_part['source'] == graft.DE_NOVO_PART_SOURCE
    assert 'source_hash' not in cdr3_part
    generated = cdr3_part['sequence']
    assert len(generated) == CHALLENGE['allowed_cdr3_length']
    pos = sum(1 for c in generated if c in 'KR')
    neg = sum(1 for c in generated if c in 'DE')
    assert abs(pos - neg) <= 1
    refs = {fd['sequence_hash']: fd['sequence']}
    assert graft.reconstruct_with_edits(row, refs) == row['sequence']
    # Deterministic and seed-distinct across pairs.
    assert graft.generate([cd], [prepared(fd)], CHALLENGE, modes=('de_novo',)) == rows
    cd2 = build_donor(cdr3='SAG' + 'R' * 13)
    abnumber[cd2['sequence']] = cd2
    many = graft.generate([cd, cd2], [prepared(fd)], CHALLENGE, limit=8, modes=('de_novo',))
    assert len(many) == 2
    assert len({r['sequence'] for r in many}) == 2


@pytest.fixture
def run_worker(abnumber, monkeypatch, tmp_path):
    from nanobody_opt import gates
    from nanobody_opt import pr100_live_worker as worker
    from pr100_germline import real_panel
    cd = build_donor(cdr3='SAG' + 'R' * 13)
    fd = build_donor(cdr3='AAA' + 'R' * 13)
    bank = [cd, fd]
    for row in bank:
        abnumber[row['sequence']] = row
    monkeypatch.setattr(worker.pr100_filter, 'validate_annotation_runtime', lambda: None)
    monkeypatch.setattr(worker.pr100_filter, 'helper', lambda name:
                        types.SimpleNamespace(_anarci_panel=lambda species: (real_panel(), None)))
    monkeypatch.setattr(worker.pr100_filter, '_annotate_uncached', lambda seq: {})
    monkeypatch.setattr(worker.pr100_filter, 'annotation_identity', lambda: 'a' * 64)
    monkeypatch.setattr(worker.rules, 'annotation_reason', lambda *a, **kw: '')
    monkeypatch.setattr(gates, 'check_sequence', lambda *a: None)
    from nanobody_opt import hydrophobicity
    monkeypatch.setattr(hydrophobicity, 'check', lambda sequence: (None, True))
    path = tmp_path / 'bank.json'
    path.write_text(json.dumps({'rows': bank}))
    (tmp_path / 'archive.csv').write_text('sequence\nAAA\n')

    def run(**overrides):
        return worker.sequence_phase({'challenge': CHALLENGE, 'references': str(path),
            'archive_sha256': worker.file_digest(tmp_path / 'archive.csv'),
            'archive_fetched_at': 1, 'proposals': 12, **overrides}, tmp_path)
    return run


def test_live_worker_graft_lane_and_per_lane_reporting(run_worker):
    off = run_worker()
    on = run_worker(graft_generation={'enabled': True, 'limit': 8})
    assert off['graft_generation_enabled'] is False
    assert on['graft_generation_enabled'] is True
    # Two exact-cell donors: two valid (cdr3, framework) pairs -> two verbatim
    # and two de-novo children.
    assert on['graft_generation_stats']['emitted'] == 4
    assert on['eligible_graft_generation'] == 4
    assert graft.SOURCE in on['per_lane_eligible']
    assert on['per_lane_eligible'][graft.SOURCE] == 4
    assert on['per_lane_published'][graft.SOURCE] == 4
    graft_rows = [r for r in on['candidates'] if r['source'] == graft.SOURCE]
    assert {r['graft_mode'] for r in graft_rows} == {'verbatim', 'de_novo'}


def test_history_and_lineage_resolve_graft_donors(abnumber, tmp_path):
    from nanobody_opt import pr100_history as history
    from nanobody_opt import pr100_lineage_outcomes as lineage
    cd = build_donor(cdr3='SAG' + 'R' * 13)
    fd = build_donor(cdr3='Y' * 16)
    abnumber[cd['sequence']] = cd
    abnumber[fd['sequence']] = fd
    rows = graft.generate([cd], [prepared(fd)], CHALLENGE, modes=('verbatim', 'de_novo'))
    assert len(rows) == 2
    refs = {cd['sequence_hash']: cd['sequence'], fd['sequence_hash']: fd['sequence']}
    for row in rows:
        assert graft.reconstruct_with_edits(row, refs) == row['sequence']
        origin = lineage.parent_origin(row, lineage.bank_index([cd, fd]))
        assert origin['resolved'] and origin['graft_mode'] == row['graft_mode']
    journal = history.generation_records(rows, [cd, fd])
    assert {r['sequence_hash'] for r in journal['lineage_references']} == {cd['sequence_hash'], fd['sequence_hash']}
    store = history.open_store(tmp_path / 'q9.sqlite')
    raw = history.canonical({'challenge': CHALLENGE, **journal}).encode()
    report = history.import_generation(store, raw, locator='recorded/sequence/result.json')
    assert report['unresolved_references'] == 0
    edges = store.execute('SELECT proposal_type,mutations_json,metadata_json FROM lineage_edges').fetchall()
    # Two verbatim donor edges (cdr3 donor + framework donor) plus one de-novo
    # framework-donor edge: the de-novo CDR3 has no source_hash donor.
    assert len(edges) == 3
    verbatim_edits = [json.loads(m)['graft_edits'] for _, _, m in edges
                      if json.loads(m)['graft_mode'] == 'verbatim']
    assert any(len(edits) == 3 for edits in verbatim_edits)
    store.close()
