"""Public parent merit preserves its target, board and single-pass provenance."""
import io
import json
import urllib.request

from nanobody_opt import competitor_lane as lane


def test_public_fetch_normalizes_each_field_and_excludes_previous_target(monkeypatch):
    monkeypatch.setattr(lane, 'target_for_epoch', lambda epoch: 'new' if epoch >= 100 else 'old')
    calls = []
    boards = {
        101: [('rival', 'AAAA', 200), ('other', 'CCCC', 250), ('other', 'DDDD', 300)],
        100: [('rival', 'EEEE', 10), ('other', 'FFFF', 1), ('other', 'GGGG', 5)],
    }
    def response(request, timeout):
        epoch = int(request.full_url.split('/leaderboard/')[1].split('/')[0])
        calls.append(epoch)
        return io.StringIO(json.dumps({'data': [
            {'coldkey': coldkey, 'nanobodies': [{'sequence': sequence,
             'protein_name': 'new', 'final_score': score}]}
            for coldkey, sequence, score in boards[epoch]]}))
    monkeypatch.setattr(urllib.request, 'urlopen', response)
    records = lane.fetch_rival_records(102, 'rival', n_epochs=3)
    assert calls == [101, 100]
    assert [row['sequence'] for row in records] == ['AAAA', 'EEEE']
    assert records[0]['quality_evidence'] == [{
        'kind': 'public_validator_percentile', 'target': 'new', 'epoch': 101,
        'value': 0.0, 'field_size': 3, 'raw_score': 200.0, 'passes': 1}]
    assert records[1]['quality_evidence'][0]['value'] == 1.0


def test_repeated_parent_retains_each_board_measurement(monkeypatch):
    monkeypatch.setattr(lane, 'target_for_epoch', lambda epoch: 'current')
    def response(request, timeout):
        return io.StringIO(json.dumps({'data': [
            {'coldkey': 'rival', 'nanobodies': [{'sequence': 'AAAA',
             'protein_name': 'current', 'final_score': 1}]},
            {'coldkey': 'other', 'nanobodies': [{'sequence': 'CCCC',
             'protein_name': 'current', 'final_score': 2}]}]}))
    monkeypatch.setattr(urllib.request, 'urlopen', response)
    records = lane.fetch_rival_records(102, 'rival', n_epochs=2)
    assert len(records) == 1
    assert [v['epoch'] for v in records[0]['quality_evidence']] == [101, 100]
    assert lane.fetch_rival_sequences(102, 'rival', n_epochs=2) == ['AAAA']
