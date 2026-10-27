"""Independent deadline-completion regression for interleaved native work."""
import copy
import json

from nanobody_opt import pr100_live_worker as worker, pr100_filter, rules
from tests.test_pr100_native_throughput import checkpoint_input, profile, CHALLENGE


def run_backlog(directory, monkeypatch, enabled):
    directory.mkdir()
    original = checkpoint_input(directory, monkeypatch)
    monkeypatch.setattr(worker, 'worker_count', lambda _: 3)
    monkeypatch.setattr(rules, 'validate_evidence', lambda *_a: None)
    rows = []
    for name in 'ABCDEFGHI':
        sequence = name * 4; row = copy.deepcopy(original)
        row['sequence'] = sequence
        row['native_resume']['sequence_hash'] = rules.sequence_hash(sequence)
        row['native_resume']['measurements'] = [profile(sequence)]
        rows.append(row)
    incoming = json.loads((directory / 'input.json').read_text()); incoming['candidates'] = rows
    pr100_filter.atomic(directory / 'input.json', incoming)
    now = [0.]; calls = []
    def measure(sequences, *_a, **_kw):
        calls.append(list(sequences)); now[0] += 100.
        return [profile(sequence) for sequence in sequences], 0., 100.
    result = worker.native_phase({'challenge': CHALLENGE, 'native_tools': '/tools',
        'workers': 3, 'repeats': 3, 'native_host': 'slave',
        'phase_budget_seconds': 430, 'candidate_budget_seconds': 300,
        'native_minimum_call_window_seconds': 130, 'native_cleanup_reserve_seconds': 10,
        'native_interleave_confirmations': enabled}, directory, measure=measure, clock=lambda: now[0])
    return result, calls


def test_interleaving_does_not_leave_all_carried_confirmations_unfinished(tmp_path, monkeypatch):
    legacy, _ = run_backlog(tmp_path / 'legacy', monkeypatch, False)
    interleaved, _ = run_backlog(tmp_path / 'interleaved', monkeypatch, True)
    assert len(legacy['candidates']) == 3
    assert interleaved['fresh_evaluations'] == legacy['fresh_evaluations'] == 0
    assert interleaved['repeat_evaluations'] == legacy['repeat_evaluations'] == 9
    assert len(interleaved['candidates']) >= len(legacy['candidates'])
