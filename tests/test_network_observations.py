"""load_network_observations: the narrow, streamed observation read that
local_neighborhood builds its MutationNetwork from.

The full ``load_observations`` pull (every column, server-side ORDER BY) made
MySQL filesort ~3.9M rows / ~3.6 GB before the first byte -- 182 s, past the
60 s read timeout -- so every live producer iteration died with error 2013.
These tests pin the replacement's contract without a database: what it asks
the server for, and that what it returns builds the same network.
"""
from __future__ import annotations

import json

import nanobody_opt.mutation_network as mn
from nanobody_opt.mutation_network import MutationNetwork
from nanobody_opt.terminal_policy import learn_terminal_priors

SEQ_A = "A" * 30
SEQ_C = "C" + "A" * 29
POSE = json.dumps({"consensus_nanobody_contact_positions": [2, 3]})

NODES = [(1, SEQ_A), (2, SEQ_C)]
# node_id, score, epoch, source, score_kind, observation_count, pose_json
ROWS = [
    (2, 60.0, 11, "live_exact_screen", "local_validator_replay_rank_sum", 1, POSE),
    (1, 80.0, 10, "public_validator_api", "validator_final_rank_sum", 1, None),
    (1, 70.0, None, "live_exact_screen", "local_validator_replay_rank_sum", 2, POSE),
    (2, 50.0, 10, "public_validator_api", "validator_final_rank_sum", 1, None),
    (9, 10.0, 10, "orphan", "local_validator_replay_rank_sum", 1, None),
]


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def fetchone(self):
        return self._rows[0]

    def fetchall(self):
        return list(self._rows)


class _FakeConnection:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.log.append(("execute", " ".join(sql.split()), params))
        if "COUNT(*)" in sql:
            return _Result([(len(ROWS), "zzz")])
        if "FROM nodes" in sql:
            return _Result(NODES)
        raise AssertionError(f"unexpected query {sql}")

    def stream(self, sql, params=None):
        self.log.append(("stream", " ".join(sql.split()), params))
        yield from ROWS


def _patch(monkeypatch):
    log = []
    monkeypatch.setattr(mn.db, "connect", lambda *a, **k: _FakeConnection(log))
    return log


def test_streams_only_the_columns_the_network_reads_without_server_sort(tmp_path, monkeypatch):
    log = _patch(monkeypatch)
    mn.load_network_observations(tmp_path / "obs.sqlite", target="P20809", length=30)
    streamed = [entry for entry in log if entry[0] == "stream"]
    assert len(streamed) == 1, "the observation pull must use the unbuffered stream"
    _, sql, params = streamed[0]
    assert params == ("P20809", 30)
    assert "ORDER BY" not in sql.upper(), "a server-side sort blocks the first row"
    for heavy in ("provenance_json", "metrics_json"):
        assert heavy not in sql, f"{heavy} is never read by the network"
    assert "n.length=%s" in sql


def test_records_build_the_same_network_as_full_rows(tmp_path, monkeypatch):
    _patch(monkeypatch)
    records = mn.load_network_observations(
        tmp_path / "obs.sqlite", target="P20809", length=30
    )
    # orphan node 9 (no node row) is dropped; NULL epoch first, then by epoch
    assert [r["epoch"] for r in records] == [None, 10, 10, 11]
    sequences = dict(NODES)
    full = [
        {
            "sequence": sequences[node], "score": score, "epoch": epoch,
            "source": source, "score_kind": kind, "observation_count": count,
            "metrics": {"x": 1}, "provenance": {"big": "x" * 100},
            "pose_stability": json.loads(pose) if pose else None,
        }
        for node, score, epoch, source, kind, count, pose in ROWS
        if node in sequences
    ]
    narrow_net, full_net = MutationNetwork(records), MutationNetwork(full)
    assert narrow_net.nodes == full_net.nodes
    assert narrow_net._contacts == full_net._contacts
    assert learn_terminal_priors(records, length=30) == learn_terminal_priors(full, length=30)


def test_second_read_is_served_from_the_cache(tmp_path, monkeypatch):
    log = _patch(monkeypatch)
    path = tmp_path / "obs.sqlite"
    first = mn.load_network_observations(path, target="P20809", length=30)
    log.clear()
    again = mn.load_network_observations(path, target="P20809", length=30)
    assert again == first
    assert not [entry for entry in log if entry[0] == "stream"]
