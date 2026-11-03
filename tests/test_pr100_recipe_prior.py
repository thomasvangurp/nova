"""recipe_prior: motif derivation, loader validation, and generator wiring."""
import json

import pytest

from nanobody_opt import pr100_graft_generation as graft
from nanobody_opt import pr100_recipe_prior as rp

# 14-residue CDR3s (medium bucket): AR/AK + 9-residue middle + tail.
MEDIUM = [('ARWQPRYTSGGFDY', 10.0), ('ARWQPRYAGSGFDY', 9.0),
          ('AKWQPRYSSTGFDY', 8.0), ('SKGGGGGGGGGMDV', 0.5)]
LONG_SET = [('AKHYGKFRQSTTAFDV', 5.0), ('AKHYGKFRNGSAAFDV', 4.0)]  # 16 residues


def derived():
    rows = MEDIUM + LONG_SET
    return rp.derive_recipe_prior([c for c, _ in rows], [s for _, s in rows],
                                  derived_at_unix=1793664000)


def test_derive_extracts_weighted_anchors_tails_cores():
    prior = derived()
    assert prior['schema'] == 'recipe-prior-v1' and prior['count'] == 6
    assert prior['derived_at_unix'] == 1793664000
    assert set(prior['by_length']) == {'short', 'medium', 'long', 'xlong'}
    medium = prior['by_length']['medium']
    assert medium['anchors'] == ['AR', 'AK', 'SK']          # 19 > 8 > 0.5
    assert medium['tails'] == ['FDY', 'MDV']
    assert medium['cores'][0] == 'WQPRY'                    # recurrent, weight 27
    assert all(len(c) >= 3 for c in medium['cores'])
    assert 'WQPR' not in medium['cores']                    # implied by WQPRY, same support
    long_ = prior['by_length']['long']
    assert long_['anchors'] == ['AK'] and long_['tails'] == ['FDV']
    assert long_['cores'][0] == 'HYGKFR'
    assert prior['by_length']['short'] == {'anchors': [], 'cores': [], 'tails': []}


def test_derive_is_deterministic_and_order_independent():
    rows = MEDIUM + LONG_SET
    a = rp.derive_recipe_prior([c for c, _ in rows], [s for _, s in rows],
                               derived_at_unix=1)
    b = rp.derive_recipe_prior([c for c, _ in reversed(rows)],
                               [s for _, s in reversed(rows)], derived_at_unix=1)
    assert a == b


def test_derive_skips_bad_input_and_liability_tails():
    prior = rp.derive_recipe_prior(
        ['ARWQPRYTSGGFDP', 'ARWQPRYTSGGFDP', 'ARXQPRYTSGGFDY', 'AR', 'ARWQPRYTSGGFDY'],
        [9, 9, 9, 9, float('nan')], derived_at_unix=1)
    assert prior['count'] == 2
    assert prior['by_length']['medium']['tails'] == []      # FDP is a liability
    with pytest.raises(ValueError):
        rp.derive_recipe_prior(['A'], [])


def test_load_roundtrip_and_invalid_inputs(tmp_path):
    path = tmp_path / 'recipe_prior.json'
    rp.write_recipe_prior(derived(), path)
    assert rp.load_recipe_prior(path) == json.loads(path.read_text())
    assert rp.load_recipe_prior(tmp_path / 'missing.json') is None
    bad = {'garbage': '{not json', 'schema': json.dumps({'schema': 'v0', 'by_length': {}}),
           'bucket': json.dumps({'schema': rp.SCHEMA, 'by_length': {'huge': {'anchors': ['AR']}}}),
           'anchor_len': json.dumps({'schema': rp.SCHEMA,
                                     'by_length': {'short': {'anchors': ['ARR']}}}),
           'alphabet': json.dumps({'schema': rp.SCHEMA,
                                   'by_length': {'short': {'tails': ['FDB']}}}),
           'not_list': json.dumps({'schema': rp.SCHEMA,
                                   'by_length': {'short': {'cores': 'FQP'}}}),
           'empty': json.dumps({'schema': rp.SCHEMA, 'by_length': {
               b: {'anchors': [], 'cores': [], 'tails': []} for b in rp.BUCKETS}})}
    for name, text in bad.items():
        path.write_text(text)
        assert rp.load_recipe_prior(path) is None, name


PRIOR = {'schema': rp.SCHEMA, 'by_length': {
    'short': {'anchors': ['GG'], 'cores': ['WWW'], 'tails': ['YYY']},
    'medium': {'anchors': ['CC'], 'cores': ['WQPRY', 'HYGKFR'], 'tails': ['WWV']},
    'long': {'anchors': ['CC'], 'cores': ['HYGKFR'], 'tails': ['WWV']},
    'xlong': {'anchors': ['CC'], 'cores': ['HYGKFR'], 'tails': ['WWV']}}}


@pytest.mark.parametrize('length', [7, 12, 16, 21])
def test_de_novo_uses_prior_and_falls_back(length):
    for seed in range(40):
        value = graft.de_novo_cdr3(length, seed, PRIOR)
        assert len(value) == length
        entry = PRIOR['by_length'][graft.length_bucket(length)]
        assert value[:2] in entry['anchors'] and value[-3:] in entry['tails']
        assert value[:2] not in graft._ANCHORS and value[-3:] not in graft._TAILS
        assert abs(sum(c in 'KR' for c in value) - sum(c in 'DE' for c in value)) <= 3
    static = graft.de_novo_cdr3(length, 5)
    assert graft.de_novo_cdr3(length, 5, None) == static == graft.de_novo_cdr3(length, 5, {})
    assert static[:2] in graft._ANCHORS and static[-3:] in graft._TAILS


def test_prior_cores_appear_in_middle():
    seen = {graft.de_novo_cdr3(12, seed, PRIOR)[2:-3] for seed in range(60)}
    assert any(m.startswith(('WQP', 'HYG')) for m in seen)
    long_seen = {graft.de_novo_cdr3(16, seed, PRIOR)[2:-3] for seed in range(60)}
    assert any(m.startswith('HYG') for m in long_seen)


def test_partial_prior_falls_back_per_kind():
    partial = {'by_length': {'medium': {'anchors': ['CC'], 'cores': [], 'tails': []}}}
    for seed in range(30):
        value = graft.de_novo_cdr3(12, seed, partial)
        assert value[:2] == 'CC' and value[-3:] in graft._TAILS
    # An unrelated bucket is untouched: identical to the static recipe.
    assert graft.de_novo_cdr3(16, 9, partial) == graft.de_novo_cdr3(16, 9)


def test_determinism_with_prior():
    for length in (10, 12, 16, 21):
        first = [graft.de_novo_cdr3(length, s, PRIOR) for s in range(25)]
        assert first == [graft.de_novo_cdr3(length, s, PRIOR) for s in range(25)]
        assert len(set(first)) > 1


def test_malformed_prior_entries_are_ignored():
    junk = {'by_length': {'medium': {'anchors': ['ARR', 5, 'zz'], 'tails': ['FD'],
                                     'cores': ['', None, 'x' * 30]}}}
    assert graft.de_novo_cdr3(12, 3, junk) == graft.de_novo_cdr3(12, 3)


def test_cli_derive(tmp_path):
    src, out = tmp_path / 'scored.jsonl', tmp_path / 'recipe_prior.json'
    rows = MEDIUM + LONG_SET
    src.write_text('\n'.join(json.dumps({'cdr3': c, 'score': s}) for c, s in rows) + '\n')
    assert rp.main(['derive', '--input', str(src), '--output', str(out)]) == 0
    loaded = rp.load_recipe_prior(out)
    assert loaded['by_length']['medium']['cores'][0] == 'WQPRY'
    src.write_text('')
    out2 = tmp_path / 'none.json'
    assert rp.main(['derive', '--input', str(src), '--output', str(out2)]) == 1
    assert not out2.exists()
