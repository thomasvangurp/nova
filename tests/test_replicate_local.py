"""Correctness tests for nanobody_opt.replicate_local against a REAL MySQL.

They need a disposable MySQL database with the production schema
(db/schema.sql) applied, named by a creds file with all five SN68_MYSQL_*
keys::

    REPL_TEST_MYSQL_CREDS=/path/creds.env pytest tests/test_replicate_local.py

Every test TRUNCATEs the remote tables, so the fixture refuses any database
whose name does not contain "test". Without the env var the module is skipped.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest

CREDS = os.environ.get("REPL_TEST_MYSQL_CREDS")
pytestmark = pytest.mark.skipif(not CREDS, reason="REPL_TEST_MYSQL_CREDS not set")

try:  # mysqlclient on the boxes; pymysql is an API-compatible stand-in locally
    import MySQLdb  # noqa: F401
except ImportError:  # pragma: no cover
    pymysql = pytest.importorskip("pymysql")
    pymysql.install_as_MySQLdb()
    import MySQLdb  # noqa: F401,E402

from nanobody_opt import db  # noqa: E402
from nanobody_opt import replicate_local as rl  # noqa: E402

AA = "ACDEFGHIKLMNPQRSTVWY"
TABLES = ["nodes", "observations", "observation_artifacts", "lineage_edges",
          "exact_runs", "exact_run_observations"]


def _creds() -> dict[str, str]:
    values = {}
    for line in Path(CREDS).read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


def remote_factory():
    values = _creds()
    return db.Connection(MySQLdb.connect(
        host=values["SN68_MYSQL_HOST"], port=int(values["SN68_MYSQL_PORT"]),
        user=values["SN68_MYSQL_USER"], password=values["SN68_MYSQL_PASSWORD"],
        database=values["SN68_MYSQL_DB"], charset="utf8mb4", autocommit=False,
    ))


@pytest.fixture
def remote():
    conn = remote_factory()
    name = conn.execute("SELECT DATABASE()").fetchone()[0]
    assert "test" in name, f"refusing to truncate non-test database {name!r}"
    for table in TABLES:
        conn.execute(f"TRUNCATE TABLE {table}")
    conn.execute("ALTER TABLE nodes AUTO_INCREMENT=1")
    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def local(tmp_path):
    conn = rl.open_local(tmp_path / "local.sqlite")
    rl.ensure_local_schema(conn)
    yield conn
    conn.close()


def vector(sequence: str) -> bytes:
    return bytes(AA.index(residue) for residue in sequence)


def seq(index: int, length: int = 12) -> str:
    out, value = [], index
    for _ in range(length):
        out.append(AA[value % 20])
        value //= 20
    return "".join(out)


def local_node(conn, target: str, sequence: str) -> int:
    conn.execute(
        "INSERT INTO nodes(target, sequence, length, residue_vector) VALUES(?,?,?,?) "
        "ON CONFLICT(target, sequence) DO NOTHING",
        (target, sequence, len(sequence), vector(sequence)),
    )
    return conn.execute("SELECT id FROM nodes WHERE target=? AND sequence=?",
                        (target, sequence)).fetchone()[0]


def local_observation(conn, oid, target, sequence, score, *, epoch=1, count=1,
                      source="live_exact_screen", parent=None):
    node = local_node(conn, target, sequence)
    conn.execute(
        """INSERT INTO observations(observation_id, node_id, epoch, source, score_kind,
               score, observation_count, metrics_json, pose_json, provenance_json)
           VALUES(?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(observation_id) DO UPDATE SET node_id=excluded.node_id,
               epoch=excluded.epoch, source=excluded.source, score_kind=excluded.score_kind,
               score=excluded.score, observation_count=excluded.observation_count,
               metrics_json=excluded.metrics_json, pose_json=excluded.pose_json,
               provenance_json=excluded.provenance_json""",
        (oid, node, epoch, source, "screen", score, count, json.dumps({"s": score}),
         None, json.dumps({"oid": oid})),
    )
    if parent:
        parent_node = local_node(conn, target, parent)
        conn.execute(
            """INSERT INTO lineage_edges(parent_node_id, child_node_id, epoch, source,
                   proposal_type, mutations_json, metadata_json) VALUES(?,?,?,?,?,?,?)
               ON CONFLICT DO UPDATE SET proposal_type=excluded.proposal_type""",
            (parent_node, node, epoch, source, "point", "[]", "{}"),
        )


def remote_observations(remote) -> dict[str, tuple]:
    rows = remote.execute(
        """SELECT o.observation_id, n.target, n.sequence, o.epoch, o.score,
                  o.observation_count, o.metrics_json
           FROM observations o JOIN nodes n ON n.id=o.node_id"""
    ).fetchall()
    return {row[0]: tuple(row[1:]) for row in rows}


def local_observations(conn) -> dict[str, tuple]:
    rows = conn.execute(
        """SELECT o.observation_id, n.target, n.sequence, o.epoch, o.score,
                  o.observation_count, o.metrics_json
           FROM observations o JOIN nodes n ON n.id=o.node_id"""
    ).fetchall()
    return {row[0]: tuple(row[1:]) for row in rows}


def lineage_by_identity(conn, placeholder="?") -> set[tuple]:
    return set(conn.execute(
        """SELECT p.target, p.sequence, c.sequence, e.epoch, e.source
           FROM lineage_edges e JOIN nodes p ON p.id=e.parent_node_id
           JOIN nodes c ON c.id=e.child_node_id"""
    ).fetchall())


def replicator(local, **kwargs) -> rl.Replicator:
    return rl.Replicator(local, remote_factory, log=lambda _m: None, **kwargs)


def test_colliding_surrogate_ids_are_remapped_by_identity(local, remote):
    # Another box already owns remote ids 1..40 for DIFFERENT sequences, and
    # also holds two of our sequences under ids that differ from ours.
    other = [("P20809", seq(10_000 + i)) for i in range(40)]
    remote.executemany(
        "INSERT INTO nodes(target, sequence, length, residue_vector) VALUES(%s,%s,%s,%s)",
        [(t, s, len(s), vector(s)) for t, s in other],
    )
    remote.executemany(
        "INSERT INTO nodes(target, sequence, length, residue_vector) VALUES(%s,%s,%s,%s)",
        [("P20809", seq(5), 12, vector(seq(5))), ("P20809", seq(7), 12, vector(seq(7)))],
    )
    remote.commit()
    for i in range(30):
        local_observation(local, f"obs-{i}", "P20809", seq(i), float(i),
                          parent=seq(i + 1) if i % 3 == 0 else None)
    # local ids 1..N overlap remote ids 1..42 exactly -- the dangerous case.
    assert local.execute("SELECT MIN(id) FROM nodes").fetchone()[0] == 1

    replicator(local).replicate_once()

    assert remote_observations(remote) == local_observations(local)
    assert lineage_by_identity(remote) == lineage_by_identity(local)
    # Pre-existing nodes kept their ids; no duplicate identities were created.
    assert remote.execute(
        "SELECT id FROM nodes WHERE target='P20809' AND sequence=%s", (seq(5),)
    ).fetchone()[0] == 41
    local_count = local.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    assert remote.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == 40 + local_count
    mapped = dict(local.execute("SELECT local_id, remote_id FROM replication_node_map"))
    local_ids = dict(local.execute("SELECT sequence, id FROM nodes"))
    assert mapped[local_ids[seq(5)]] == 41
    assert local.execute("SELECT COUNT(*) FROM observations WHERE replicated_at IS NULL"
                         ).fetchone()[0] == 0


def test_crash_after_remote_commit_before_local_mark_resends_without_damage(local, remote):
    for i in range(2500):
        local_observation(local, f"obs-{i}", "P20809", seq(i), float(i))

    class Crash(BaseException):
        pass

    calls = {"n": 0}

    def crash_on_second(table, count):
        if table == "observations":
            calls["n"] += 1
            if calls["n"] == 2:
                raise Crash()

    with pytest.raises(Crash):
        replicator(local, after_remote_commit=crash_on_second).replicate_once()
    # Batch 2 is on the remote but not marked locally.
    assert remote.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 2000
    remote.commit()
    assert local.execute("SELECT COUNT(*) FROM observations WHERE replicated_at IS NULL"
                         ).fetchone()[0] == 1500
    # A transaction left open by the crash must not survive in the store.
    assert not local.in_transaction

    stats = replicator(local).replicate_once()
    assert stats["sent"]["observations"] == 1500
    remote.commit()
    assert remote_observations(remote) == local_observations(local)
    assert remote.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == 2500


def test_row_rewritten_while_in_flight_stays_dirty_and_is_resent(local, remote):
    for i in range(10):
        local_observation(local, f"obs-{i}", "P20809", seq(i), 1.0)

    def rewrite(table, count):
        if table == "observations":
            # The producer upserts a new version after the old one reached Hetzner.
            local_observation(local, "obs-3", "P20809", seq(3), 99.0, count=2)

    stats = replicator(local, after_remote_commit=rewrite).replicate_once()
    assert stats["superseded"]["observations"] == 1
    assert local.execute("SELECT replicated_at FROM observations WHERE observation_id='obs-3'"
                         ).fetchone()[0] is None
    replicator(local).replicate_once()
    remote.commit()
    assert remote_observations(remote)["obs-3"][3] == 99.0
    assert remote_observations(remote) == local_observations(local)


def test_identical_reupsert_is_not_resent_but_real_change_is(local, remote):
    local_observation(local, "obs-1", "P20809", seq(1), 1.0)
    replicator(local).replicate_once()
    local_observation(local, "obs-1", "P20809", seq(1), 1.0)
    assert local.execute("SELECT replicated_at FROM observations").fetchone()[0] > 0
    local_observation(local, "obs-1", "P20809", seq(1), 2.0)
    assert local.execute("SELECT replicated_at FROM observations").fetchone()[0] is None


def test_poison_row_is_quarantined_without_blocking_the_rest(local, remote):
    for i in range(50):
        local_observation(local, f"obs-{i}", "P20809", seq(i), float(i))
    # varchar(64) on the server; 100 chars is a hard STRICT-mode error.
    local.execute("UPDATE observations SET source=? WHERE observation_id='obs-17'", ("x" * 100,))
    stats = replicator(local).replicate_once()
    assert stats["sent"]["observations"] == 49
    assert stats["quarantined"]["observations"] == 1
    remote.commit()
    assert "obs-17" not in remote_observations(remote)
    assert len(remote_observations(remote)) == 49
    assert local.execute("SELECT replicated_at FROM observations WHERE observation_id='obs-17'"
                         ).fetchone()[0] == -1
    # The producer fixes the row: it becomes dirty again and goes through.
    local.execute("UPDATE observations SET source='fixed' WHERE observation_id='obs-17'")
    replicator(local).replicate_once()
    remote.commit()
    assert len(remote_observations(remote)) == 50


def test_dangling_local_node_reference_is_quarantined(local, remote):
    local_observation(local, "good", "P20809", seq(1), 1.0)
    local.execute(
        """INSERT INTO observations(observation_id, node_id, source, score_kind, score,
               metrics_json, provenance_json) VALUES('dangling', 999999, 's', 'k', 1, '{}', '{}')"""
    )
    stats = replicator(local).replicate_once()
    assert stats["quarantined"]["observations"] == 1
    remote.commit()
    assert set(remote_observations(remote)) == {"good"}


def test_case_variant_identity_is_refused_not_merged(local, remote):
    remote.execute(
        "INSERT INTO nodes(target, sequence, length, residue_vector) VALUES('P20809', %s, 12, %s)",
        (seq(1), vector(seq(1))),
    )
    remote.commit()
    local.execute(
        "INSERT INTO nodes(target, sequence, length, residue_vector) VALUES('P20809', ?, 12, ?)",
        (seq(1).lower(), vector(seq(1))),
    )
    stats = replicator(local).replicate_once()
    assert stats["quarantined"]["nodes"] == 1
    assert local.execute("SELECT COUNT(*) FROM replication_node_map").fetchone()[0] == 0


def _record_exact_run(conn, target, run_id, observation_ids, complete):
    conn.execute("BEGIN")
    conn.execute(
        """INSERT INTO exact_runs(target, run_id, epoch, source_path, file_sha256,
               scorer_protocol, expected_rows, ingested_rows, complete, ingested_at)
           VALUES(?,?,1,'/x','sha','p',?,?,?,1.0)
           ON CONFLICT(target, run_id) DO UPDATE SET ingested_rows=excluded.ingested_rows,
               complete=excluded.complete, ingested_at=excluded.ingested_at + 1""",
        (target, run_id, len(observation_ids), len(observation_ids), complete),
    )
    conn.execute("DELETE FROM exact_run_observations WHERE target=? AND run_id=?",
                 (target, run_id))
    conn.executemany("INSERT INTO exact_run_observations VALUES(?,?,?,NULL,random())",
                     [(target, run_id, oid) for oid in observation_ids])
    conn.execute("COMMIT")


def test_exact_run_waits_for_its_observations_and_mirrors_membership(local, remote):
    for i in range(5):
        local_observation(local, f"obs-{i}", "P20809", seq(i), float(i))
    _record_exact_run(local, "P20809", "run-a", [f"obs-{i}" for i in range(5)], 1)
    # Simulate the barrier race: one listed observation is not yet on Hetzner.
    local.execute("UPDATE observations SET score=42 WHERE observation_id='obs-4'")
    rep = replicator(local)
    rep.replicate_nodes()
    rep.replicate_exact_runs()
    assert rep.stats["deferred"]["exact_runs"] == 1
    assert remote.execute("SELECT COUNT(*) FROM exact_runs").fetchone()[0] == 0
    remote.commit()

    replicator(local).replicate_once()
    remote.commit()
    assert remote.execute("SELECT complete FROM exact_runs").fetchone()[0] == 1
    assert remote.execute("SELECT COUNT(*) FROM exact_run_observations").fetchone()[0] == 5
    remote.commit()

    # record_exact_run rewrites membership with DELETE+INSERT: Hetzner follows.
    _record_exact_run(local, "P20809", "run-a", ["obs-0", "obs-1"], 0)
    replicator(local).replicate_once()
    remote.commit()
    assert sorted(r[0] for r in remote.execute(
        "SELECT observation_id FROM exact_run_observations").fetchall()) == ["obs-0", "obs-1"]
    assert remote.execute("SELECT complete FROM exact_runs").fetchone()[0] == 0


def test_schema_without_token_is_refused(tmp_path):
    conn = sqlite3.connect(tmp_path / "old.sqlite")
    conn.execute("CREATE TABLE observations(observation_id TEXT PRIMARY KEY, replicated_at REAL)")
    conn.commit()
    with pytest.raises(rl.SchemaError):
        rl.ensure_local_schema(conn)
