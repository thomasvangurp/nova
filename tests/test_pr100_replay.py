"""New-target replay must not reuse old-target observations or plot their mean."""
import hashlib
import io
import json
import sqlite3
from types import SimpleNamespace

import pandas as pd
import pytest

from nanobody_opt import rules
from tools import collect_slave_multipass_seqs as collector
from tools import multipass_all_rivals as multipass
from tools import rescale_multipass_history as rescale
from tools import replay_queue
from tools.slave_runtime import slave_multipass_writeback as worker


@pytest.mark.parametrize('module,function', [
    (collector, 'leaderboard_sequences'), (multipass, 'fetch_leaderboard'),
    (rescale, 'fetch_leaderboard')])
def test_leaderboard_target_changes_on_boundary(monkeypatch, module, function):
    rows = [{'coldkey': 'OWNER', 'nanobodies': [
        {'protein_name': 'P20809', 'sequence': 'OLD'},
        {'protein_name': rules.TARGET, 'sequence': 'NEW'}]}]
    monkeypatch.setattr(module.urllib.request, 'urlopen', lambda *a, **k: io.BytesIO(json.dumps({'data': rows}).encode()))
    old, new = getattr(module, function)(25954), getattr(module, function)(25955)
    assert old == (['OLD'] if module is collector else [('OWNER', 'OLD')])
    assert new == (['NEW'] if module is collector else [('OWNER', 'NEW')])


def test_same_sequence_old_passes_do_not_cover_new_target(tmp_path, monkeypatch):
    path = tmp_path/'observations.sqlite'
    with sqlite3.connect(path) as conn:
        conn.executescript('CREATE TABLE nodes (id INTEGER, target TEXT, sequence TEXT);'
                          'CREATE TABLE observations (node_id INTEGER, observation_id TEXT, source TEXT, '
                          'score_kind TEXT, metrics_json TEXT, epoch INTEGER);')
        for nid, target, count, epoch in [(1, 'P20809', 20, 25954), (2, rules.TARGET, 2, 25955)]:
            conn.execute('INSERT INTO nodes VALUES (?,?,?)', (nid, target, 'SHARED'))
            conn.executemany('INSERT INTO observations VALUES (?,?,?,?,?,?)', [
                (nid, f'{nid}:{n}', 'public_multipass_rivals', 'multipass_sum10_pass', '{"design_ptm":1}', epoch)
                for n in range(count)])
    monkeypatch.setattr(collector, 'DB', str(path))
    monkeypatch.setattr(rescale, 'LOCAL_DB', path)
    assert collector.pass_counts(['SHARED'], rules.TARGET) == {'SHARED': 2}
    assert len(rescale.multipass_sum10_metrics(25955, {'SHARED'})) == 2
    assert len(rescale.multipass_sum10_metrics(25954, {'SHARED'})) == 20
    with sqlite3.connect(path) as conn:
        assert replay_queue.canonical_counts(conn, ['SHARED'], rules.TARGET) == {'SHARED': 2}


def test_replay_empty_reference_and_wrong_target_are_rejected():
    payload = {'target': rules.TARGET, 'per_nanobody_components': {'0': {}}}
    with pytest.raises(ValueError, match='measured field'):
        multipass.rank_pass_payload(payload, pd.DataFrame(), rules.TARGET)
    with pytest.raises(ValueError, match='another target'):
        multipass.rank_pass_payload(payload, pd.DataFrame([{'metric': 1}]), 'P20809')


def actor(value):
    return {'mean': value, 'stdev': 0., 'fleet_size': 1, 'scores': [value],
            'coverage_complete': True, 'passes_min': 10, 'passes_max': 10}


def test_dashboard_rolls_over_separate_targets_and_rejects_unidentified_new_data(tmp_path):
    from nanobody_dash import progress
    payload = {}
    for epoch, score, target in [(25953, 100., 'P20809'), (25954, 100., 'P20809'),
                                 (25955, 900., rules.TARGET), (25956, 500., 'P20809')]:
        payload[str(epoch)] = {'ours': actor(score), 'rivals': {}, 'method': {
            'formula': 'sum10', 'measurement': 'local_fast_scorer_multipass_mean',
            'field': 'recent_validator_epochs', 'target': target}}
    path = tmp_path/'multipass.json'
    path.write_text(json.dumps(payload))
    figure = progress.fig_multipass_strongest(path)
    values = dict(zip(figure.data[0].x, figure.data[0].y))
    assert values[25955] == 900.
    assert values[25954.5] is None
    assert values[25956] is None
    assert any(rules.TARGET in a.text for a in figure.layout.annotations)


def test_incremental_builder_removes_wrong_target_post_cutover_cache(tmp_path, monkeypatch):
    from tools import build_multipass_comparison as builder
    path = tmp_path/'multipass.json'
    path.write_text(json.dumps({'25954': {'method': {}}, '25955': {'method': {
        'coverage_policy': 'complete_fleet_minimum_10_passes', 'target': 'P20809'}}}))
    monkeypatch.setattr(builder, 'rescale_epoch', lambda epoch: None)
    assert set(builder.build(path, lo=25955, hi=25955)) == {'25954'}
    assert set(json.loads(path.read_text())) == {'25954'}


def test_slave_replicates_keep_target_pass_count_and_all_lengths(tmp_path, monkeypatch):
    sequences = ['A'*120, 'C'*127, 'D'*130]
    commands = []
    def score(command, **kwargs):
        commands.append(command)
        from pathlib import Path
        inp = Path(command[command.index('--sequences')+1])
        out = Path(command[command.index('--out')+1])
        items = json.loads(inp.read_text())['sequences']
        assert len({len(i['sequence']) for i in items}) == 1
        assert command[command.index('--target')+1] == rules.TARGET
        assert '--allow-replicates' in command
        wave = {hashlib.sha256(i['sequence'].encode()).hexdigest()+str(n): {
                    rules.TARGET: {m: float(n+1) for m in worker.SUM_OF_TEN_DIRS}}
                for n, i in enumerate(items)}
        out.write_text(json.dumps({'target': rules.TARGET, 'per_nanobody_components': {'0': wave}}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(worker.subprocess, 'run', score)
    rows = worker.score_group(sequences, 3, tmp_path/'work', '0', tmp_path/'config.yaml', rules.TARGET)
    assert len(commands) == 3 and len(rows) == 9
    assert {s for s, _, _ in rows} == set(sequences)
    assert all({r for s, r, _ in rows if s == seq} == {1, 2, 3} for seq in sequences)


def test_slave_context_requires_matching_target_and_finite_complete_reference():
    config = rules.pinned_config(25955)
    field = {'target': rules.TARGET, 'epochs': [25955],
             'rows': [{m: 1. for m in worker.SUM_OF_TEN_DIRS}]}
    assert worker.validate_context(25956, config, field) == rules.TARGET
    field['target'] = 'P20809'
    with pytest.raises(ValueError, match='target'):
        worker.validate_context(25956, config, field)
    field['target'] = rules.TARGET
    field['rows'][0]['design_ptm'] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        worker.validate_context(25956, config, field)


def test_master_rejects_old_slave_target_after_boundary():
    from tools.ingest_slave_multipass import validate
    row = {'observation_id': 'fixture', 'sequence': 'FAKE', 'target': 'P20809',
           'epoch': 25955, 'source': 'public_multipass_rivals',
           'score_kind': 'multipass_sum10_pass', 'writer': 'slave_multipass_writeback'}
    assert any('target' in s for s in validate([row]))
    row['target'] = rules.TARGET
    assert validate([row]) == []
def test_unpack_actual_registered_scorer_gpu_output():
    """Recorded real folds, not a payload manufactured from unpack's key rule."""
    import json
    from pathlib import Path
    from tools.slave_runtime.slave_multipass_writeback import unpack
    fixture=json.loads((Path(__file__).parent/'fixtures/pr100/recorded_gpu_metrics.json').read_text())
    assert fixture['kind']=='recorded_gpu_metrics_projection'
    total=0
    for group in fixture['groups']:
        payload=group['payload']
        records=unpack(payload['per_nanobody_components']['0'],group['sequences'],
                       fixture['passes'],payload['target'])
        assert len(records)==len(group['sequences'])*fixture['passes']
        for sequence in group['sequences']:
            rows=[(replica,metrics) for s,replica,metrics in records if s==sequence]
            assert {r for r,_ in rows}=={1,2}
            assert len({json.dumps(m,sort_keys=True) for _,m in rows})==2
        total+=len(records)
    assert total==10
