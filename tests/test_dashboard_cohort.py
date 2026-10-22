"""Keep real scored ownership and complete repeat evidence in dashboard means."""
import io
import json

import pytest

from tools import rescale_multipass_history as replay


def board(monkeypatch, rows):
    payload = json.dumps({'data': rows}).encode()
    monkeypatch.setattr(replay.urllib.request, 'urlopen',
                        lambda *args, **kwargs: io.BytesIO(payload))


def row(owner, sequence, target=None):
    return {'coldkey': owner, 'nanobodies': [
        {'sequence': sequence, 'protein_name': target or replay.TARGET}]}


def test_shared_sequence_retains_both_scored_owners(monkeypatch):
    board(monkeypatch, [row('rival', 'SHARED'), row(replay.OUR_COLDKEY, 'SHARED'),
                        row(replay.OUR_COLDKEY, 'OWN')])
    assert replay.fetch_leaderboard(25872) == [
        ('rival', 'SHARED'), (replay.OUR_COLDKEY, 'SHARED'), (replay.OUR_COLDKEY, 'OWN')]


def test_duplicate_api_rows_and_other_targets_do_not_inflate_membership(monkeypatch):
    board(monkeypatch, [row('ours', 'SHARED'), row('ours', 'shared'),
                        row('ours', 'UNRELATED', 'OTHER')])
    assert replay.fetch_leaderboard(25872) == [('ours', 'SHARED')]


def prepare_replay(monkeypatch, depths):
    from nanobody_opt import rank_exact
    board(monkeypatch, [row('rival', 'SHARED'), row(replay.OUR_COLDKEY, 'SHARED'),
                        row(replay.OUR_COLDKEY, 'OWN')])
    metrics = {f'{seq}::pass-{n}': {'rank': 700 if seq == 'SHARED' else 900}
               for seq, depth in depths.items() for n in range(depth)}
    monkeypatch.setattr(replay, 'multipass_sum10_metrics', lambda *args: metrics)
    monkeypatch.setattr(replay, 'our_race_metrics', lambda *args: {})
    monkeypatch.setattr(rank_exact, 'fetch_recent_fields_from_db', lambda *a, **k: [object()])
    monkeypatch.setattr(rank_exact, 'pool_recent_fields', lambda *a: [object()] * 388)
    monkeypatch.setattr(rank_exact, 'rank_metrics_against_field',
                        lambda values, field: {k: v['rank'] for k, v in values.items()})


def test_shared_candidate_is_included_in_full_fleet_mean(monkeypatch):
    prepare_replay(monkeypatch, {'SHARED': 20, 'OWN': 10})
    result = replay.rescale_epoch(25872)
    assert result['ours']['fleet_size'] == 2
    assert result['ours']['coverage_complete'] is True
    assert result['ours']['mean'] == pytest.approx(800)
    assert result['ours']['passes_min'] == 10
    assert result['rivals']['rival']['mean'] == pytest.approx(700)


def test_missing_repeat_evidence_never_becomes_a_partial_fleet_average(monkeypatch):
    prepare_replay(monkeypatch, {'SHARED': 20, 'OWN': 9})
    result = replay.rescale_epoch(25872)
    assert result['ours']['fleet_size'] == 2
    assert result['ours']['scored_sequences'] == 1
    assert result['ours']['missing_sequences'] == 1
    assert result['ours']['coverage_complete'] is False
    assert result['ours']['mean'] is None
    assert result['ours']['scores'] == []
