"""Plan path: derive each epoch's challenge from validator-accepted submissions."""
import json
import sqlite3

from scripts import plan_from_accepted as p


def _conn(rows):
    con = sqlite3.connect(':memory:')
    con.execute('CREATE TABLE nodes (id INTEGER PRIMARY KEY, target TEXT, sequence TEXT)')
    con.execute('CREATE TABLE observations (node_id INTEGER, source TEXT, score_kind TEXT,'
                ' epoch INTEGER, metrics_json TEXT, provenance_json TEXT)')
    for node_id, sequence, epoch, cells in rows:
        con.execute('INSERT INTO nodes VALUES (?, ?, ?)', (node_id, 'Q9NZQ7', sequence))
        con.execute('INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?)',
                    (node_id, 'public_validator_api', 'validator_final_rank_sum', epoch,
                     None, json.dumps({'cells': cells}) if cells else None))
    return con


def test_cells_to_annotation():
    assert p.cells_to_annotation(['IGHV3-23:18']) == (frozenset({'IGHV3-23'}), 18)
    assert p.cells_to_annotation(['IGHV3-23:18', 'IGHV3-66:18']) == (frozenset({'IGHV3-23', 'IGHV3-66'}), 18)
    assert p.cells_to_annotation([]) is None
    assert p.cells_to_annotation(['bad']) is None


def test_resolve_category_key():
    cats = {'V1': ['IGHV3-23'], 'V2': ['IGHV3-66'], 'V3': ['IGHV3-74'],
            'V4': ['IGHV3-7', 'IGHV3-48']}
    assert p.resolve_category_key(cats, 'IGHV3-23') == 'V1'
    assert p.resolve_category_key(cats, 'IGHV3-48') == 'V4'


def test_derive_majority(tmp_path):
    con = _conn([(1, 'A' * 120, 26000, ['IGHV3-23:18']),
                 (2, 'B' * 120, 26000, ['IGHV3-23:18']),
                 (3, 'C' * 120, 26000, ['IGHV3-23:18'])])
    cats = {'V1': ['IGHV3-23'], 'V2': ['IGHV3-66']}
    result = p.derive_from_accepted(con, 26000, target='Q9NZQ7', categories=cats,
                                    annotate=lambda s: None)
    assert result['status'] == 'ok'
    assert result['allowed_v'] == 'IGHV3-23' and result['allowed_cdr3_length'] == 18
    assert result['category_key'] == 'V1'


def test_derive_split_on_tie(tmp_path):
    con = _conn([(1, 'A' * 120, 26000, ['IGHV3-23:18']),
                 (2, 'B' * 120, 26000, ['IGHV3-23:18']),
                 (3, 'C' * 120, 26000, ['IGHV3-66:18']),
                 (4, 'D' * 120, 26000, ['IGHV3-66:18'])])
    cats = {'V1': ['IGHV3-23'], 'V2': ['IGHV3-66']}
    result = p.derive_from_accepted(con, 26000, target='Q9NZQ7', categories=cats,
                                    annotate=lambda s: None)
    assert result['status'] == 'skip' and 'split' in result['reason']
