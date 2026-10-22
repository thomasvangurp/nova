"""Bug 5: serialize before asserting; use shipping v2 producer/race paths."""
import ast
import csv
import io
import json
from pathlib import Path

import pytest
from nanobody_opt_v2 import run_live_producer as producer
from nanobody_opt.filter_layer2 import _load_candidates
from sn68_source import import_shipping

FIELDS = ['sequence', 'source', 'parent_sequence', 'screen_priority', 'parent_source',
          'proposal_type', 'parent_basin_core_id', 'parent_basin_rank', 'basin_core_id', 'proposal_lane']


@pytest.fixture
def race(monkeypatch):
    return import_shipping('nanobody_opt_v2.sequential_exact_race', monkeypatch)


@pytest.fixture
def candidates():
    return [dict(zip(FIELDS, values)) for values in (
        ('AAAA', 'children', 'AAAC', '20', 'recent_archive', 'framework_single_substitution',
         'core-a', '2', 'core-a', 'primary'),
        ('AAAC', 'children', 'AACC', '1.0', 'exact_shortlist', 'exhaustive_single_substitution',
         'core-b', '1', 'core-b', 'winner'),
    )]


def producer_csv_schema():
    tree = ast.parse(Path(producer.__file__).read_text())
    schemas = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'DictWriter':
            for keyword in node.keywords:
                if keyword.arg == 'fieldnames' and isinstance(keyword.value, ast.List):
                    fields = ast.literal_eval(keyword.value)
                    if fields[:3] == ['sequence', 'source', 'parent_sequence']:
                        schemas.append(fields)
    assert len(schemas) == 2, 'candidate writer schema moved; review shipping code'
    assert schemas[0] == schemas[1] == FIELDS
    return schemas[0], tree


def serialize_csv(rows, fields):
    handle = io.StringIO(newline='')
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def test_bug5_actual_producer_csv_writer_preserves_every_column(candidates):
    """Bug 5: real shipping projection, exact ten-column schema, byte equality."""
    fields, tree = producer_csv_schema()
    expected = serialize_csv(candidates, fields)
    blocks = [n for n in ast.walk(tree) if isinstance(n, ast.With)
              and any(isinstance(b, ast.For) and isinstance(b.iter, ast.Name)
                      and b.iter.id == 'child_rows' for b in n.body)]
    assert len(blocks) == 1
    with io.StringIO(newline='') as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()
        namespace = dict(csv=csv, handle=handle, child_rows=candidates,
                         children_novel={r['sequence'] for r in candidates})
        exec(compile(ast.Module(body=blocks[0].body, type_ignores=[]), producer.__file__, 'exec'), namespace)
        handle.seek(0)
        reader = csv.DictReader(handle)
        assert reader.fieldnames == fields
        actual = list(reader)
    for before, after in zip(candidates, actual, strict=True):
        assert {k: v.encode() for k, v in before.items()} == {k: v.encode() for k, v in after.items()}
    assert serialize_csv(actual, fields).encode() == expected.encode()


@pytest.mark.parametrize('priority', ['1', '1.0'])
def test_bug5_csv_priority_numbers_equivalent_and_missing_nan_last(priority, race, tmp_path):
    """Bug 5: serialized priorities survive and beat earlier blank/NaN rows."""
    fields, _ = producer_csv_schema()
    rows = [dict.fromkeys(fields, '') for _ in range(4)]
    for row, seq, rank in zip(rows, ['AAAA', 'CCCC', 'DDDD', 'EEEE'], ['', 'NaN', '2', priority]):
        row.update(sequence=seq, screen_priority=rank, parent_source='exact_shortlist')
    csv_path = tmp_path / 'candidates.csv'
    csv_path.write_text(serialize_csv(rows, fields))
    with csv_path.open(newline='') as handle:
        restored = list(csv.DictReader(handle))
    assert restored == rows
    json_path = tmp_path / 'layer2.json'
    json_path.write_text(json.dumps({'candidates': restored}))
    result = race._candidate_records(json_path)
    assert [r['sequence'] for r in result] == ['EEEE', 'DDDD', 'AAAA', 'CCCC']
    assert {r['sequence']: r for r in result} == {r['sequence']: r for r in restored}


@pytest.mark.parametrize('format', ['csv', 'json'])
def test_bug5_full_candidate_metadata_round_trip(candidates, tmp_path, format, race):
    """Bug 5: real legacy Layer-2 reader and v2 race retain all measured fields."""
    records = [{**row, 'identity': .5, 'cdr3_similarity': .5, 'multipass': 8,
                'layer2_passed': True, 'cdr_similarity': {'CDR1': .4, 'CDR2': .5, 'CDR3': .5}}
               for row in candidates]
    path = tmp_path / f'qualified.{format}'
    if format == 'json':
        path.write_text(json.dumps({'candidates': records}))
        expected = records
    else:
        expected = [{k: json.dumps(v) if isinstance(v, dict) else str(v) for k, v in r.items()} for r in records]
        path.write_text(serialize_csv(expected, list(expected[0])))
    restored = _load_candidates(path)
    assert restored == expected
    race_input = tmp_path / 'layer2.json'
    race_input.write_text(json.dumps({'candidates': restored}))
    ordered = race._candidate_records(race_input)
    assert [r['sequence'] for r in ordered] == ['AAAC', 'AAAA']
    assert {r['sequence']: r for r in ordered} == {r['sequence']: r for r in restored}
