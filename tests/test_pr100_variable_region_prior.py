"""variable-region prior: region extraction, co-occurrence, recombination, fallback.

Synthetic 128-residue sequences span IMGT numbers 1..128 with canonical
cysteines at 23 and 104, so ``prepare`` accepts them without ``abnumber`` (the
numbering is mocked/precomputed). Region boundaries (0-indexed): fr1 0-25,
cdr1 26-37, fr2 38-54, cdr2 55-64, fr3 65-103, cdr3 104-116, fr4 117-127.
"""
import json
import sys
import types

import pytest

from nanobody_opt import pr100_segment_generation as segments
from nanobody_opt import pr100_variable_region_prior as vrp
from nanobody_opt import rules

CHALLENGE = {'epoch': 26030, 'target': 'Q9NZQ7', 'allowed_v': 'IGHV3-74',
             'allowed_cdr3_length': 13}
_CDR3_A = 'SRFQPRYGFDYWG'   # 13 residues, medium bucket
_CDR3_B = 'AKTGGGGGYFDYW'
_REGION_LENS = {'fr1': 26, 'cdr1': 12, 'fr2': 17, 'cdr2': 10,
                'fr3': 39, 'cdr3': 13, 'fr4': 11}


def synthetic(first='E', cdr3=_CDR3_A):
    assert len(cdr3) == _REGION_LENS['cdr3']
    residues = ['A'] * 128
    residues[0] = first
    residues[22] = 'C'      # IMGT 23
    residues[103] = 'C'     # IMGT 104
    residues[104:117] = list(cdr3)  # IMGT 105..117
    sequence = ''.join(residues)
    numbered = [{'number': i + 1, 'letter': '', 'aa': aa}
                for i, aa in enumerate(sequence)]
    return sequence, numbered


@pytest.fixture
def pair():
    seq_a, num_a = synthetic('E', _CDR3_A)
    seq_b, num_b = synthetic('Q', _CDR3_B)
    return (seq_a, num_a), (seq_b, num_b)


def derived(pair):
    (seq_a, num_a), (seq_b, num_b) = pair
    return vrp.derive_variable_region_prior(
        [seq_a, seq_b], [10.0, 9.0], numbered=[num_a, num_b],
        allowed_v='IGHV3-74', derived_at_unix=1)


def group(prior):
    return prior['by_group']['IGHV3-74:medium']


def test_derive_extracts_regions_via_prepare(pair):
    (seq_a, num_a), (seq_b, num_b) = pair
    substrings = vrp.region_substrings(seq_a, num_a)
    assert set(substrings) == set(vrp.REGIONS)
    assert {name: len(part) for name, part in substrings.items()} == _REGION_LENS
    assert ''.join(substrings[name] for name in vrp.REGIONS) == seq_a
    assert substrings['cdr3'] == _CDR3_A and substrings['fr1'][0] == 'E'
    prior = derived(pair)
    assert prior['schema'] == 'variable-region-prior-v1'
    assert prior['count'] == 2 and prior['derived_at_unix'] == 1
    g = group(prior)
    assert g['germline'] == 'IGHV3-74' and g['bucket'] == 'medium'
    assert {f['sequence'] for f in g['regions']['cdr3']} == {_CDR3_A, _CDR3_B}
    assert {f['sequence'][0] for f in g['regions']['fr1']} == {'E', 'Q'}
    assert all(len(f['sequence']) == _REGION_LENS[n]
               for n in vrp.REGIONS for f in g['regions'][n])
    for h in g['haplotypes']:
        assert ''.join(h['regions'][n] for n in vrp.REGIONS) == h['sequence']


def test_haplotype_weight_support_and_deterministic_ordering(pair):
    (seq_a, _), (seq_b, _) = pair
    g = group(derived(pair))
    assert [h['sequence'] for h in g['haplotypes']] == [seq_a, seq_b]
    assert g['haplotypes'][0]['weight'] == 10.0 and g['haplotypes'][1]['weight'] == 9.0
    assert [f['sequence'] for f in g['regions']['cdr3']] == [_CDR3_A, _CDR3_B]
    assert g['regions']['cdr3'][0]['support'] == 1
    assert g['regions']['cdr3'][0]['source_hash'] == rules.sequence_hash(seq_a)


def test_derive_is_deterministic_and_order_independent(pair):
    (seq_a, num_a), (seq_b, num_b) = pair
    a = vrp.derive_variable_region_prior([seq_a, seq_b], [10.0, 9.0],
                                         numbered=[num_a, num_b], allowed_v='IGHV3-74',
                                         derived_at_unix=1)
    b = vrp.derive_variable_region_prior([seq_b, seq_a], [9.0, 10.0],
                                         numbered=[num_b, num_a], allowed_v='IGHV3-74',
                                         derived_at_unix=1)
    assert a == b


def test_negative_scores_weigh_zero_and_are_excluded(pair):
    (seq_a, num_a), (seq_b, num_b) = pair
    prior = vrp.derive_variable_region_prior([seq_a, seq_b], [10.0, -1.0],
                                             numbered=[num_a, num_b],
                                             allowed_v='IGHV3-74')
    g = group(prior)
    assert [f['sequence'] for f in g['regions']['cdr3']] == [_CDR3_A]
    assert [h['sequence'] for h in g['haplotypes']] == [seq_a]


def test_derive_skips_bad_input_and_mismatched_lengths(pair):
    (seq_a, num_a), (seq_b, num_b) = pair
    bad = synthetic('E', _CDR3_A)[0].replace('A', 'X', 1)
    prior = vrp.derive_variable_region_prior(
        [seq_a, bad, seq_b], [10.0, 10.0, float('nan')],
        numbered=[num_a, num_a, num_b], allowed_v='IGHV3-74')
    assert prior['count'] == 1 and prior['skipped'] == 2
    with pytest.raises(ValueError):
        vrp.derive_variable_region_prior([seq_a], [])
    with pytest.raises(ValueError):
        vrp.derive_variable_region_prior([seq_a, seq_b], [1.0],
                                         numbered=[num_a, num_b])


def test_derive_accepts_bank_rows_and_germline_from_cells(pair):
    (seq_a, num_a), _ = pair
    rows = [{'sequence': seq_a, 'numbered': num_a,
             'cells': ['IGHV3-74:13'], 'score': 10.0}]
    prior = vrp.derive_variable_region_prior(rows, None, derived_at_unix=1)
    assert set(prior['by_group']) == {'IGHV3-74:medium'}
    assert group(prior)['haplotypes'][0]['sequence'] == seq_a


def test_project_numbering_uses_abnumber_chain(pair, monkeypatch):
    (seq_a, num_a), _ = pair
    class Chain:
        def __init__(self, sequence, **kwargs):
            self.sequence = sequence

        def __iter__(self):
            return iter((types.SimpleNamespace(number=d['number'], letter=d['letter']), d['aa'])
                        for d in num_a)
    monkeypatch.setitem(sys.modules, 'abnumber', types.SimpleNamespace(Chain=Chain))
    assert vrp.project_numbering(seq_a) == num_a


def test_whole_haplotype_reuse_is_single_source(pair):
    (seq_a, _), (seq_b, _) = pair
    prior = derived(pair)
    rows = vrp.recombine_variable_region(prior, CHALLENGE, seed=0, limit=8,
                                         whole_haplotype_probability=1.0)
    assert len(rows) == len({r['sequence_hash'] for r in rows}) == 2
    assert all(r['variable_region_assembly'] == 'whole_haplotype' for r in rows)
    assert {r['sequence'] for r in rows} == {seq_a, seq_b}
    assert all(len(r['variable_region_sources']) == 1 for r in rows)
    assert all(len(r['variable_region_provenance']) == 7 for r in rows)


def test_independent_draws_trace_to_top_scored_sources_and_mix(pair):
    (seq_a, _), (seq_b, _) = pair
    prior = derived(pair)
    rows = vrp.recombine_variable_region(prior, CHALLENGE, seed=0, limit=64,
                                         whole_haplotype_probability=0.0)
    assert rows and all(r['variable_region_assembly'] == 'independent_draw' for r in rows)
    pools = {name: {f['sequence'] for f in group(prior)['regions'][name]}
             for name in vrp.REGIONS}
    for row in rows:
        for part in row['variable_region_provenance']:
            assert part['sequence'] in pools[part['region']]
        cdr3 = next(p['sequence'] for p in row['variable_region_provenance']
                    if p['region'] == 'cdr3')
        assert len(cdr3) == CHALLENGE['allowed_cdr3_length']
    # Independent draws must produce at least one child that is neither source.
    assert any(r['sequence'] not in (seq_a, seq_b) for r in rows)


def test_recombination_is_seeded_and_deterministic(pair):
    prior = derived(pair)
    first = vrp.recombine_variable_region(prior, CHALLENGE, seed=0, limit=16)
    assert first == vrp.recombine_variable_region(prior, CHALLENGE, seed=0, limit=16)
    outputs = {rules.digest(vrp.recombine_variable_region(prior, CHALLENGE, seed=s,
                                                          limit=16))
               for s in range(8)}
    assert len(outputs) > 1


def test_empty_prior_falls_back_to_segment_lane():
    stats = {}
    rows = vrp.recombine_variable_region(None, CHALLENGE, limit=4, donors=[],
                                         stats=stats)
    assert rows == segments.generate([], CHALLENGE, limit=4) == []
    assert stats.get('variable_region_fallback') is True


def test_unusable_prior_falls_back(pair):
    prior = derived(pair)
    wrong_v = {**CHALLENGE, 'allowed_v': 'IGHV3-66'}
    stats = {}
    assert vrp.recombine_variable_region(prior, wrong_v, donors=[], stats=stats) == []
    assert stats.get('variable_region_fallback') is True
    wrong_length = {**CHALLENGE, 'allowed_cdr3_length': 14}  # same bucket, no exact fragment
    stats = {}
    assert vrp.recombine_variable_region(prior, wrong_length, donors=[], stats=stats) == []
    assert stats.get('variable_region_fallback') is True


def test_recombination_rejects_invalid_bounds(pair):
    prior = derived(pair)
    for kwargs in ({'limit': 0}, {'limit': 4801}, {'seed': -1},
                   {'whole_haplotype_probability': 1.5},
                   {'whole_haplotype_probability': True}):
        with pytest.raises(ValueError):
            vrp.recombine_variable_region(prior, CHALLENGE, **kwargs)


def test_prior_write_load_roundtrip_and_invalid(pair, tmp_path):
    prior = derived(pair)
    path = tmp_path / 'variable_region_prior.json'
    vrp.write_variable_region_prior(prior, path)
    assert vrp.load_variable_region_prior(path) == json.loads(path.read_text())
    assert vrp.load_variable_region_prior(tmp_path / 'missing.json') is None
    bad = {'garbage': '{not json',
           'schema': json.dumps({'schema': 'v0', 'by_group': {}}),
           'bad_region': json.dumps({'schema': vrp.SCHEMA, 'by_group': {
               'IGHV3-74:medium': {'germline': 'IGHV3-74', 'bucket': 'medium',
                                   'regions': {n: [] for n in list(vrp.REGIONS)[:-1]},
                                   'haplotypes': []}}}),
           'bad_fragment': json.dumps({'schema': vrp.SCHEMA, 'by_group': {
               'IGHV3-74:medium': {'germline': 'IGHV3-74', 'bucket': 'medium',
                                   'regions': {n: ([{'sequence': 'X', 'weight': 1,
                                                     'support': 1,
                                                     'source_hash': 'a' * 64}]
                                                    if n == 'cdr3' else [])
                                               for n in vrp.REGIONS},
                                   'haplotypes': []}}}),
           'empty': json.dumps({'schema': vrp.SCHEMA, 'by_group': {
               'IGHV3-74:medium': {'germline': 'IGHV3-74', 'bucket': 'medium',
                                   'regions': {n: [] for n in vrp.REGIONS},
                                   'haplotypes': []}}})}
    for name, text in bad.items():
        path.write_text(text)
        assert vrp.load_variable_region_prior(path) is None, name
