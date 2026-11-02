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


SWAPPABLE = 'RRSAG' + 'R' * 11  # 3 swappable residues at offsets 2..4, length 16


def pair(abnumber, cd_cdr3=SWAPPABLE, fd_cdr3='Y' * 16, cd_cdr1='R' * 12, fd_cdr1='K' * 12):
    """A CDR3 donor and a framework donor whose CDR1 diverges (clears conserved rule)."""
    cd = build_donor(cdr3=cd_cdr3, cdr1=cd_cdr1)
    fd = build_donor(cdr3=fd_cdr3, cdr1=fd_cdr1)
    abnumber[cd['sequence']] = cd
    abnumber[fd['sequence']] = fd
    return cd, fd


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
    cd, fd = pair(abnumber)
    rows = graft.generate([cd], [prepared(fd)], CHALLENGE, limit=10, modes=('verbatim',))
    assert len(rows) == 1
    row = rows[0]
    assert row['graft_mode'] == 'verbatim'
    assert len(row['segment_provenance']) == 7
    assert len({p['source_hash'] for p in row['segment_provenance']}) == 2
    assert row['graft_neutral_subs'] == 3
    assert row['graft_neutral_subs_shortfall'] == 0
    refs = {cd['sequence_hash']: cd['sequence'], fd['sequence_hash']: fd['sequence']}
    assert graft.reconstruct_with_edits(row, refs) == row['sequence']


def test_three_substitutions_identity_offsets_and_residues():
    cdr3 = SWAPPABLE
    edited, edits = graft.neutral_substitutions(cdr3)
    assert edited == 'RRTSA' + 'R' * 11
    identity = sum(a == b for a, b in zip(edited, cdr3)) / len(cdr3)
    assert identity == 13 / 16 == pytest.approx(0.8125)
    assert 0.80 <= identity < 0.85
    assert len(edits) == 3
    for edit in edits:
        assert edit['kind'] == 'neutral_cdr3_substitution'
        assert edit['from'] in graft.NEUTRAL_SWAPS
        assert edit['to'] == graft.NEUTRAL_SWAPS[edit['from']]
        assert 2 <= edit['child_offset'] < len(cdr3) - 3


def test_required_substitutions_clears_clonotype_threshold_for_all_lengths():
    assert [graft.required_substitutions(n) for n in (5, 16, 19, 20, 23, 24, 27, 40)] \
        == [3, 3, 3, 4, 4, 4, 5, 7]
    for length in range(5, 41):
        k = graft.required_substitutions(length)
        assert (length - k) / length < 0.85


def test_neutral_substitutions_protect_anchor_and_skip_liability_motifs():
    # Index 0/1 are swappable but never touched.
    edited, edits = graft.neutral_substitutions('SAGSAGSAG' + 'R' * 7)
    assert edited[:2] == 'SA'
    assert min(e['child_offset'] for e in edits) == 2
    # Q@2 -> N would create the deamidation motif 'NG' with the G after it.
    cdr3 = 'RRQG' + 'R' * 12
    assert graft.NEUTRAL_SWAPS['Q'] == 'N'
    edited, edits = graft.neutral_substitutions(cdr3, 1)
    assert all(e['child_offset'] != 2 for e in edits)
    assert graft.liability_count(edited) == 0
    assert edits and edits[0]['child_offset'] == 3


def test_divergence_gate_drops_conserved_and_keeps_divergent(abnumber):
    conserved_cd, conserved_fd = pair(abnumber, fd_cdr1='R' * 12)
    stats = {}
    assert graft.generate([conserved_cd], [prepared(conserved_fd)], CHALLENGE,
                          modes=('verbatim',), stats=stats) == []
    assert stats['divergence_drops'] == 1
    divergent_cd, divergent_fd = pair(abnumber, cd_cdr3=SWAPPABLE[:-1] + 'K')
    rows = graft.generate([divergent_cd], [prepared(divergent_fd)], CHALLENGE,
                          modes=('verbatim',))
    assert len(rows) == 1


def test_gate_is_child_vs_both_donors_not_donor_vs_donor(abnumber):
    # CDR1 differs in 2/12 positions (similarity 0.833 < 0.90) so the donors
    # themselves look divergent, but the CHILD is 5 residues from the CDR3
    # donor (2 CDR1 + 3 substitutions): whole identity 126/131 >= 0.95.
    cd, fd = pair(abnumber, fd_cdr1='RRRRRRRRRRKK')
    cd_prepared = prepared(cd)
    assert graft.whole_identity(fd['sequence'], cd['sequence']) < 0.95
    stats = {}
    assert graft.generate([cd], [prepared(fd)], CHALLENGE, modes=('verbatim',),
                          stats=stats) == []
    assert stats['divergence_drops'] == 1
    # Child clears both donors on the accepted path.
    cd2, fd2 = pair(abnumber)
    row = graft.generate([cd2], [prepared(fd2)], CHALLENGE, modes=('verbatim',))[0]
    assert graft.whole_identity(row['sequence'], cd2['sequence']) < 0.95
    assert graft.whole_identity(row['sequence'], fd2['sequence']) < 0.95
    assert cd_prepared['segments']['cdr3']['sequence'] == SWAPPABLE


def test_clonotype_threshold_for_long_cdr3_uses_more_substitutions(abnumber):
    cdr3 = 'RR' + 'SAGSAG' + 'R' * 12  # length 20 -> k = 4
    cd, fd = pair(abnumber, cd_cdr3=cdr3, fd_cdr3='Y' * 20)
    challenge = {**CHALLENGE, 'allowed_cdr3_length': 20}
    rows = graft.generate([cd], [prepared(fd)], challenge, modes=('verbatim',))
    assert len(rows) == 1
    assert rows[0]['graft_neutral_subs'] == 4
    child_cdr3 = next(p for p in rows[0]['segment_provenance'] if p['region'] == 'cdr3')
    edited = list(cdr3)
    for edit in child_cdr3['edits']:
        edited[edit['child_offset']] = edit['to']
    assert segments.cdr_similarity(''.join(edited), cdr3) < 0.85


def test_donor_with_too_few_swappable_residues_is_skipped(abnumber):
    for cdr3 in ('R' * 16, 'RRSA' + 'R' * 12):
        cd, fd = pair(abnumber, cd_cdr3=cdr3)
        stats = {}
        assert graft.generate([cd], [prepared(fd)], CHALLENGE, modes=('verbatim',),
                              stats=stats) == []
        assert stats['cdr3_insufficient_swappable_residues'] == 1


def test_exact_length_enforcement_skips_wrong_length_cdr3(abnumber):
    cd_bad = build_donor(cdr3='R' * 13, cell_length=13)
    cd_good, fd = pair(abnumber)
    abnumber[cd_bad['sequence']] = cd_bad
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
    cd, fd = pair(abnumber)
    cd2 = build_donor(cdr3='T' * 13 + 'AAA')
    fd2 = build_donor(cdr3='W' * 16, cdr1='D' * 12)
    for donor in (cd2, fd2):
        abnumber[donor['sequence']] = donor
    first = graft.generate([cd, cd2], [prepared(fd), prepared(fd2)], CHALLENGE,
                           limit=20, modes=('verbatim', 'de_novo'))
    second = graft.generate([cd, cd2], [prepared(fd), prepared(fd2)], CHALLENGE,
                            limit=20, modes=('verbatim', 'de_novo'))
    assert first == second
    assert {r['graft_mode'] for r in first} == {'verbatim', 'de_novo'}
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
    cd, fd = pair(abnumber)
    rows = graft.generate([cd], [prepared(fd)], CHALLENGE, limit=4, modes=('de_novo',))
    assert len(rows) == 4  # independent of the CDR3 donor pair space
    assert len({r['sequence'] for r in rows}) == 4
    row = rows[0]
    assert row['graft_mode'] == 'de_novo'
    refs = {fd['sequence_hash']: fd['sequence']}
    cdr3s = []
    for row in rows:
        cdr3_part = next(p for p in row['segment_provenance'] if p.get('region') == 'cdr3')
        assert cdr3_part['source'] == graft.DE_NOVO_PART_SOURCE
        assert 'source_hash' not in cdr3_part
        generated = cdr3_part['sequence']
        cdr3s.append(generated)
        assert len(generated) == CHALLENGE['allowed_cdr3_length']
        assert abs(sum(c in 'KR' for c in generated) - sum(c in 'DE' for c in generated)) <= 1
        assert graft.liability_count(generated) == 0
        assert graft.whole_identity(row['sequence'], fd['sequence']) < 0.95
        assert segments.cdr_similarity(generated, 'Y' * 16) < 0.85  # framework donor CDR3
        assert graft.reconstruct_with_edits(row, refs) == row['sequence']
    for i, left in enumerate(cdr3s):
        for right in cdr3s[i + 1:]:
            assert segments.cdr_similarity(left, right) < 0.85
    # Deterministic; a different proposal_offset yields fresh candidates.
    assert graft.generate([cd], [prepared(fd)], CHALLENGE, limit=4, modes=('de_novo',)) == rows
    later = graft.generate([cd], [prepared(fd)], CHALLENGE, limit=4, modes=('de_novo',),
                           proposal_offset=4)
    assert len(later) == 4
    assert not {r['sequence'] for r in later} & {r['sequence'] for r in rows}


def test_de_novo_limit_beyond_donor_pair_space_is_reachable(abnumber):
    cd, fd = pair(abnumber)
    rows = graft.generate([cd], [prepared(fd)], CHALLENGE, limit=120, modes=('de_novo',))
    assert len(rows) == 120


def test_de_novo_skips_small_lengths_without_raising(abnumber):
    # Length 4 is below the recipe minimum: skip, do not abort.
    cd, fd = pair(abnumber, cd_cdr3='RSAG', fd_cdr3='YYYY')
    challenge = {**CHALLENGE, 'allowed_cdr3_length': 4}
    stats = {}
    assert graft.generate([cd], [prepared(fd)], challenge, modes=('de_novo',),
                          stats=stats) == []
    assert stats['de_novo_length_unsupported'] == 1
    # Length 5: the child is >= 0.952 identical to the framework donor -> gated out.
    cd, fd = pair(abnumber, cd_cdr3='RSAGR', fd_cdr3='YYYYY')
    challenge = {**CHALLENGE, 'allowed_cdr3_length': 5}
    stats = {}
    assert graft.generate([cd], [prepared(fd)], challenge, modes=('de_novo',),
                          stats=stats) == []
    assert stats['divergence_drops'] > 0


def test_liability_motifs_detected():
    assert graft.liability_count('RRNASRR') == 1  # N-x-S
    assert graft.liability_count('RRNGRR') == 1
    assert graft.liability_count('RRDGRR') == 1
    assert graft.liability_count('RRDPRR') == 1
    assert graft.liability_count('RRNAARR') == 0


def test_fill_middle_caps_runs_for_every_residue_class():
    import random
    for seed in range(300):
        for prefix in ('', 'AA', 'FQPR'):
            middle = graft._fill_middle(prefix, 30, random.Random(seed))
            assert len(middle) == 30
            assert 'AAA' not in middle and not any(c * 3 in middle for c in set(middle))
        short = graft.de_novo_cdr3(10, seed)
        assert not any(c * 3 in short for c in set(short))


def test_reconstruct_with_edits_rejects_illegitimate_provenance(abnumber):
    import copy
    cd, fd = pair(abnumber)
    refs = {cd['sequence_hash']: cd['sequence'], fd['sequence_hash']: fd['sequence']}
    good = graft.generate([cd], [prepared(fd)], CHALLENGE, limit=2,
                          modes=('verbatim', 'de_novo'))
    verbatim = next(r for r in good if r['graft_mode'] == 'verbatim')
    de_novo = next(r for r in good if r['graft_mode'] == 'de_novo')
    assert graft.reconstruct_with_edits(verbatim, refs) == verbatim['sequence']
    assert graft.reconstruct_with_edits(de_novo, refs) == de_novo['sequence']

    def mutated(row, fn):
        clone = copy.deepcopy(row)
        fn(clone)
        return clone

    def cdr3(row):
        return row['segment_provenance'][5]

    def swap_regions(row):
        parts = row['segment_provenance']
        parts[0], parts[2] = parts[2], parts[0]

    def de_novo_framework(row):
        row['segment_provenance'][0]['source'] = graft.DE_NOVO_PART_SOURCE

    def edits_on_framework(row):
        row['segment_provenance'][0]['edits'] = [dict(cdr3(row)['edits'][0])]

    def duplicate_offsets(row):
        cdr3(row)['edits'] = [cdr3(row)['edits'][0]] * 2

    def too_many(row):
        base = cdr3(row)['edits']
        cdr3(row)['edits'] = base + [{'child_offset': 10, 'from': 'S', 'to': 'T',
                                      'kind': 'neutral_cdr3_substitution'}]

    cases = [
        (verbatim, swap_regions), (verbatim, de_novo_framework),
        (verbatim, edits_on_framework), (verbatim, duplicate_offsets),
        (verbatim, too_many),
        (de_novo, lambda r: cdr3(r).__setitem__('edits', [
            {'child_offset': 0, 'from': 'A', 'to': 'S', 'kind': 'neutral_cdr3_substitution'}])),
        (verbatim, lambda r: r['segment_provenance'].pop()),
        (verbatim, lambda r: r['segment_provenance'].__setitem__(0, None)),
        (verbatim, lambda r: r['segment_provenance'].__setitem__(0, 'x')),
        (verbatim, lambda r: r['segment_provenance'][0].pop('source_hash')),
        (verbatim, lambda r: r['segment_provenance'][0].__setitem__('source_hash', ['x'])),
        (verbatim, lambda r: r['segment_provenance'][0].pop('region')),
        (verbatim, lambda r: cdr3(r).__setitem__('edits', 'bad')),
        (verbatim, lambda r: cdr3(r).__setitem__('edits', [None])),
        (verbatim, lambda r: cdr3(r).__setitem__('edits', [{'child_offset': 'x'}])),
        (verbatim, lambda r: cdr3(r)['edits'][0].pop('from')),
        (verbatim, lambda r: r.__setitem__('sequence', None)),
        (de_novo, lambda r: cdr3(r).__setitem__('sequence', None)),
    ]
    for row, fn in cases:
        with pytest.raises(ValueError):
            graft.reconstruct_with_edits(mutated(row, fn), refs)


@pytest.fixture
def run_worker(abnumber, monkeypatch, tmp_path):
    from nanobody_opt import gates
    from nanobody_opt import pr100_live_worker as worker
    from pr100_germline import real_panel
    cd = build_donor(cdr3=SWAPPABLE)
    fd = build_donor(cdr3='YY' + 'SAG' + 'Y' * 11, cdr1='K' * 12)
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
    on = run_worker(graft_generation={'enabled': True, 'limit': 4})
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
    cd, fd = pair(abnumber)
    rows = graft.generate([cd], [prepared(fd)], CHALLENGE, limit=2,
                          modes=('verbatim', 'de_novo'))
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
