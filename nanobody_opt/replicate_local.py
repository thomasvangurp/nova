#!/usr/bin/env python3
"""Replicate master's local SQLite store to Hetzner MySQL (ARCHITECTURE_REDESIGN
new finding 8).

Durability and cross-box visibility for the local-sqlite-as-primary design.
A box's local-only data has no copy anywhere else, so losing the box loses
the data (this is what happened to the "slave" box's 20x-refresh data, which
had to be recovered from disk by hand on 2026-10-10). This replicator keeps
Hetzner a near-real-time copy; tools/pull_local_sqlite_backup.sh keeps a
second, file-level copy on the operator's machine.

Correctness rules (each one is exercised by tests/test_replicate_local.py and
by tools/replication_soak.py against a real MySQL server):

1. **Node identity, not node id.** ``nodes.id`` is a surrogate AUTOINCREMENT
   key, assigned independently here and on Hetzner (where every other box
   also inserts nodes), so local id 17 can be a different (target, sequence)
   on Hetzner. Local ids are never sent. Each local node is resolved to
   Hetzner's id for the same (target, sequence), inserting it there if it is
   missing, and every node reference (``observations.node_id``,
   ``lineage_edges.parent_node_id/child_node_id``) is rewritten through that
   mapping before it is sent. The mapping is cached in
   ``replication_node_map`` only after the Hetzner transaction that created
   or read it has committed, so an id from a rolled-back insert is never
   reused. A remote row is accepted for a local node only if its sequence is
   byte-identical and its length matches (the server's collation is
   case-insensitive; a case-variant match is refused, not merged).

2. **Mark only what was sent.** A row is dirty while ``replicated_at IS
   NULL``. Every row carries a random ``repl_token``, redrawn on every insert
   and on every update that changes a data column (schema triggers). After
   the Hetzner commit, rows are marked ``WHERE rowid=? AND repl_token=?``
   with the token that was read. A row the producer rewrote while its old
   version was in flight stays dirty and is sent again. Nothing is marked
   before the remote commit, so a crash at any point only causes a re-send,
   and every remote write is an idempotent upsert.

3. **One bad row cannot wedge the queue.** A batch Hetzner refuses with a
   data error (too long for a column, bad value, dangling node reference) is
   bisected until the offending rows are isolated. Those rows are recorded in
   ``replication_quarantine`` and marked ``replicated_at=-1``; everything else
   is sent. A later rewrite of a quarantined row makes it dirty again.
   Transient errors (lock wait, deadlock, lost connection) abort the pass;
   the next pass retries the same rows.

4. **The exact-run barrier survives replication.** ``exact_runs.complete``
   tells publication that every observation of a run is present.
   ``record_exact_run`` rewrites a run's ``exact_run_observations`` with
   DELETE + INSERT, so a run is replicated as one unit: upsert the run row,
   delete and reinsert its observation list, in one Hetzner transaction. It
   is deferred until every observation it lists is already confirmed on
   Hetzner, so Hetzner never shows a complete run whose rows are missing.

Not replicated: ``mutation_effects``, ``metadata``, ``edges``,
``proposal_scores`` (see the comment at the end of local_schema.sql). Local
deletes are not propagated, except for exact-run observation lists (rule 4);
the write path at HEAD deletes nothing else.

Usage::

    # one pass (cron-style); exit 0 when the backlog is empty
    python -m nanobody_opt.replicate_local --local /root/artifacts/nanobody_local.sqlite
    # long-running (the supported mode on master): idle poll every 30 s
    python -m nanobody_opt.replicate_local --local ... --loop-interval 30
    # backlog report only, no Hetzner connection
    python -m nanobody_opt.replicate_local --local ... --status
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

SCHEMA_PATH = Path(__file__).with_name("local_schema.sql")
REPLICATION_BATCH = 1000
# Same bound as the live writer (mutation_network._MAX_STATEMENT_BYTES): one
# multi-row statement per 1000-row chunk, far below max_allowed_packet (64 MiB).
MAX_STATEMENT_BYTES = 8 * 1024 * 1024
NODE_LOOKUP_BATCH = 1000
QUARANTINED = -1.0
REMOTE_ENV_KEYS = (
    "SN68_MYSQL_HOST", "SN68_MYSQL_PORT", "SN68_MYSQL_USER",
    "SN68_MYSQL_PASSWORD", "SN68_MYSQL_DB",
)

# MySQL errors that mean "this row's data is unacceptable", as opposed to a
# transient server/network condition. 1048 column cannot be null, 1062 dup
# (cannot happen through ON DUPLICATE KEY UPDATE, listed defensively), 1264
# out of range, 1265 truncated, 1292 incorrect value, 1366 incorrect string
# value, 1406 data too long.
DATA_ERROR_CODES = frozenset({1048, 1062, 1264, 1265, 1292, 1366, 1406})


LOCK_CONFLICT_CODES = frozenset({1205, 1213})
LOCK_RETRIES = 6


def is_lock_conflict(exc: BaseException) -> bool:
    args = getattr(exc, "args", ())
    return (type(exc).__name__ == "OperationalError" and bool(args)
            and args[0] in LOCK_CONFLICT_CODES)


class RowDataError(Exception):
    """A row that can never be sent as it is (bisected and quarantined)."""


class SchemaError(Exception):
    pass


def is_data_error(exc: BaseException) -> bool:
    if isinstance(exc, RowDataError):
        return True
    try:
        import MySQLdb
    except ImportError:  # pragma: no cover - tests without mysqlclient
        return False
    if isinstance(exc, (MySQLdb.DataError, MySQLdb.IntegrityError)):
        return True
    if isinstance(exc, MySQLdb.MySQLError):
        args = getattr(exc, "args", ())
        return bool(args) and args[0] in DATA_ERROR_CODES
    return False


# ----------------------------------------------------------------------------
# Remote SQL. Upserts overwrite every non-key column: the local store is the
# authority for the rows this box writes, the same semantics as the live
# writer's upserts (mutation_network._*_UPSERT_SQL).
# ----------------------------------------------------------------------------
_NODE_INSERT_SQL = """
    INSERT INTO nodes(target, sequence, length, residue_vector)
    VALUES(%s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE id=id
"""

ROW_TABLES: dict[str, dict[str, Any]] = {
    "observations": {
        "columns": ["observation_id", "node_id", "epoch", "source", "score_kind",
                    "score", "observation_count", "metrics_json", "pose_json",
                    "provenance_json"],
        "node_columns": ["node_id"],
        "sql": """
            INSERT INTO observations(
                observation_id, node_id, epoch, source, score_kind, score,
                observation_count, metrics_json, pose_json, provenance_json
            ) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                node_id=VALUES(node_id), epoch=VALUES(epoch),
                source=VALUES(source), score_kind=VALUES(score_kind),
                score=VALUES(score), observation_count=VALUES(observation_count),
                metrics_json=VALUES(metrics_json), pose_json=VALUES(pose_json),
                provenance_json=VALUES(provenance_json)
        """,
    },
    "observation_artifacts": {
        "columns": ["observation_id", "confidence_path", "writer_path",
                    "structure_json", "sample_selection_json"],
        "node_columns": [],
        "sql": """
            INSERT INTO observation_artifacts(
                observation_id, confidence_path, writer_path,
                structure_json, sample_selection_json
            ) VALUES(%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                confidence_path=VALUES(confidence_path),
                writer_path=VALUES(writer_path),
                structure_json=VALUES(structure_json),
                sample_selection_json=VALUES(sample_selection_json)
        """,
    },
    "lineage_edges": {
        "columns": ["parent_node_id", "child_node_id", "epoch", "source",
                    "proposal_type", "mutations_json", "metadata_json"],
        "node_columns": ["parent_node_id", "child_node_id"],
        "sql": """
            INSERT INTO lineage_edges(
                parent_node_id, child_node_id, epoch, source,
                proposal_type, mutations_json, metadata_json
            ) VALUES(%s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                proposal_type=VALUES(proposal_type),
                mutations_json=VALUES(mutations_json),
                metadata_json=VALUES(metadata_json)
        """,
    },
}
EXACT_RUN_COLUMNS = ["target", "run_id", "epoch", "source_path", "file_sha256",
                     "scorer_protocol", "expected_rows", "ingested_rows",
                     "complete", "ingested_at"]
_EXACT_RUN_UPSERT_SQL = """
    INSERT INTO exact_runs(
        target, run_id, epoch, source_path, file_sha256, scorer_protocol,
        expected_rows, ingested_rows, complete, ingested_at
    ) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        epoch=VALUES(epoch), source_path=VALUES(source_path),
        file_sha256=VALUES(file_sha256), scorer_protocol=VALUES(scorer_protocol),
        expected_rows=VALUES(expected_rows), ingested_rows=VALUES(ingested_rows),
        complete=VALUES(complete), ingested_at=VALUES(ingested_at)
"""
_EXACT_RUN_OBS_INSERT_SQL = """
    INSERT INTO exact_run_observations(target, run_id, observation_id)
    VALUES(%s, %s, %s)
    ON DUPLICATE KEY UPDATE observation_id=VALUES(observation_id)
"""
TOKEN_TABLES = ("observations", "observation_artifacts", "lineage_edges",
                "exact_runs", "exact_run_observations")


# ----------------------------------------------------------------------------
# Connections
# ----------------------------------------------------------------------------
def open_local(path: Path) -> sqlite3.Connection:
    # isolation_level=None: autocommit, so plain SELECTs never hold a read
    # transaction open (that would pin the WAL and block checkpoints); the
    # few multi-statement units below use explicit BEGIN/COMMIT.
    conn = sqlite3.connect(str(path), timeout=60, isolation_level=None)
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


def ensure_local_schema(conn: sqlite3.Connection) -> None:
    """Apply local_schema.sql (idempotent) and refuse a store whose
    replicated tables predate the repl_token column -- without it rule 2
    cannot be enforced, so fail closed instead of replicating unsafely."""
    try:
        conn.executescript(SCHEMA_PATH.read_text())
    except sqlite3.OperationalError as error:
        raise SchemaError(f"local_schema.sql does not apply to this store: {error}") from error
    for table in TOKEN_TABLES:
        info = {row[1]: row for row in conn.execute(f"PRAGMA table_info({table})")}
        token = info.get("repl_token")
        if token is None or "random" not in str(token[4]).lower():
            raise SchemaError(
                f"{table}.repl_token missing or not DEFAULT (random()); this "
                "store was created from an older local_schema.sql"
            )
        trigger = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='trigger' AND name=?",
            (f"{table}_repl_dirty",),
        ).fetchone()
        if trigger is None:
            raise SchemaError(f"trigger {table}_repl_dirty missing")


def load_remote_env(path: Path) -> None:
    """Load an explicit SN68_MYSQL_* creds file, overriding the environment.

    All five keys are required: db.connect() falls back to the production
    host/user/db for a missing key, so a partial file could silently point a
    test at production.
    """
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    missing = [key for key in REMOTE_ENV_KEYS if key not in values]
    if missing:
        raise SystemExit(f"REPLICATE_LOCAL_BAD_CREDS missing keys {missing} in {path}")
    os.environ.update({key: values[key] for key in REMOTE_ENV_KEYS})


def connect_remote() -> Any:
    from nanobody_opt import db
    return db.connect()


# ----------------------------------------------------------------------------
# Replicator
# ----------------------------------------------------------------------------
class Replicator:
    def __init__(
        self,
        local: sqlite3.Connection,
        remote_factory: Callable[[], Any],
        *,
        batch_size: int = REPLICATION_BATCH,
        log: Callable[[str], None] = lambda message: print(message, file=sys.stderr, flush=True),
        after_remote_commit: Callable[[str, int], None] | None = None,
    ) -> None:
        self.local = local
        self.remote_factory = remote_factory
        self.remote: Any = None
        self.batch_size = batch_size
        self.log = log
        # Test hook: runs after the Hetzner commit and before the local mark,
        # i.e. inside the window a crash must not be able to corrupt.
        self.after_remote_commit = after_remote_commit
        self.stats: dict[str, Any] = {}
        # Optional: called as trace(event, table, rows) at the two crash-
        # relevant points ("remote_committed", "marked"); --trace prints them.
        self.trace: Callable[[str, str, int], None] | None = None

    def _trace(self, event: str, table: str, count: int) -> None:
        if self.trace is not None:
            self.trace(event, table, count)

    # -- plumbing -------------------------------------------------------------
    def _remote(self) -> Any:
        if self.remote is None:
            self.remote = self.remote_factory()
        return self.remote

    def close(self) -> None:
        if self.remote is not None:
            try:
                self.remote.close()
            except Exception:  # noqa: BLE001
                pass
            self.remote = None

    def _rollback_remote(self) -> None:
        if self.remote is None:
            return
        try:
            self.remote.rollback()
        except Exception:  # noqa: BLE001 - a dead connection is reopened next use
            self.close()

    def _bump(self, key: str, table: str, amount: int) -> None:
        bucket = self.stats.setdefault(key, {})
        bucket[table] = bucket.get(table, 0) + amount

    def _send_isolated(
        self,
        table: str,
        items: Sequence[Any],
        send: Callable[[Sequence[Any]], None],
        quarantine: Callable[[Any, BaseException], None],
    ) -> tuple[int, int]:
        """send(items); on a data error, bisect to isolate and quarantine the
        offending items. A lock conflict (1213 deadlock / 1205 lock wait) with
        another writer re-runs the same send after a short backoff (every send
        is re-runnable: nothing local is marked before its remote commit).
        Other errors propagate and abort the pass."""
        delay = 0.2
        failure: BaseException | None = None
        for attempt in range(1, LOCK_RETRIES + 1):
            try:
                send(items)
                return len(items), 0
            except Exception as error:  # noqa: BLE001
                if self.local.in_transaction:
                    self.local.execute("ROLLBACK")
                self._rollback_remote()
                if is_lock_conflict(error) and attempt < LOCK_RETRIES:
                    self._bump("lock_retries", table, 1)
                    time.sleep(delay)
                    delay = min(delay * 2, 5.0)
                    continue
                if not is_data_error(error):
                    raise
                failure = error
                break
        assert failure is not None
        if len(items) == 1:
            quarantine(items[0], failure)
            self.log(f"REPLICATE_LOCAL_QUARANTINED table={table} "
                     f"{type(failure).__name__}: {str(failure)[:300]}")
            return 0, 1
        middle = len(items) // 2
        left = self._send_isolated(table, items[:middle], send, quarantine)
        right = self._send_isolated(table, items[middle:], send, quarantine)
        return left[0] + right[0], left[1] + right[1]

    def _mark(self, table: str, marks: Iterable[tuple[int, int]], value: float) -> int:
        """Token-guarded mark; returns how many rows matched the sent version."""
        marks = list(marks)
        if not marks:
            return 0
        before = self.local.total_changes
        self.local.executemany(
            f"UPDATE {table} SET replicated_at=? WHERE rowid=? AND repl_token=?",
            [(value, rowid, token) for rowid, token in marks],
        )
        matched = self.local.total_changes - before
        if matched < len(marks):
            self._bump("superseded", table, len(marks) - matched)
        return matched

    def _commit_remote(self, table: str, count: int) -> None:
        self._remote().commit()
        self._trace("remote_committed", table, count)
        if self.after_remote_commit is not None:
            self.after_remote_commit(table, count)

    # -- node identity --------------------------------------------------------
    def _resolve_remote_nodes(
        self, nodes: Sequence[tuple[int, str, str, int, bytes]]
    ) -> dict[int, int]:
        """{local_id: remote_id} for (local_id, target, sequence, length,
        residue_vector) rows, inserting missing nodes on Hetzner. Runs inside
        the caller's open remote transaction; the result is only valid once
        that transaction commits."""
        remote = self._remote()
        by_key = {(target, sequence): (local_id, length, vector)
                  for local_id, target, sequence, length, vector in nodes}
        if len(by_key) != len(nodes):
            raise RowDataError("duplicate (target, sequence) among local nodes")
        resolved: dict[int, int] = {}

        def lookup(keys: list[tuple[str, str]], locking: bool) -> None:
            by_target: dict[str, list[str]] = {}
            for target, sequence in keys:
                by_target.setdefault(target, []).append(sequence)
            for target, sequences in by_target.items():
                for start in range(0, len(sequences), NODE_LOOKUP_BATCH):
                    chunk = sequences[start:start + NODE_LOOKUP_BATCH]
                    wanted_exact = set(chunk)
                    wanted_folded = {sequence.upper(): sequence for sequence in chunk}
                    rows = remote.execute(
                        "SELECT id, sequence, length FROM nodes WHERE target=%s "
                        "AND sequence IN (" + ",".join(["%s"] * len(chunk)) + ")"
                        + (" FOR SHARE" if locking else ""),
                        (target, *chunk),
                    ).fetchall()
                    for remote_id, remote_sequence, remote_length in rows:
                        remote_sequence = str(remote_sequence)
                        if remote_sequence not in wanted_exact:
                            # Collation (utf8mb4_0900_ai_ci) matched a
                            # different string: never merge two identities.
                            local = wanted_folded.get(remote_sequence.upper())
                            raise RowDataError(
                                f"remote node {remote_id} {target}:{remote_sequence!r} "
                                f"collation-matches local {local!r} but is not identical"
                            )
                        local_id, length, _ = by_key[(target, remote_sequence)]
                        if int(remote_length) != int(length):
                            raise RowDataError(
                                f"remote node {remote_id} length {remote_length} != "
                                f"local node {local_id} length {length}"
                            )
                        resolved[local_id] = int(remote_id)

        keys = list(by_key)
        lookup(keys, locking=False)
        missing = [key for key in keys if by_key[key][0] not in resolved]
        if missing:
            remote.executemany(
                _NODE_INSERT_SQL,
                [(target, sequence, by_key[(target, sequence)][1],
                  bytes(by_key[(target, sequence)][2]))
                 for target, sequence in missing],
                max_stmt_length=MAX_STATEMENT_BYTES,
            )
            # Locking read: sees rows other writers committed after this
            # transaction's snapshot (the plain read above could not).
            lookup(missing, locking=True)
        unresolved = [by_key[key][0] for key in keys if by_key[key][0] not in resolved]
        if unresolved:
            raise RuntimeError(f"remote node resolution failed for local ids {unresolved[:5]}")
        return resolved

    def _remote_ids_for(self, local_ids: Iterable[int]) -> tuple[dict[int, int], dict[int, int]]:
        """(cached, pending): map local node ids through replication_node_map;
        resolve the rest on Hetzner inside the open remote transaction.
        ``pending`` must be persisted only after that transaction commits."""
        wanted = sorted(set(int(node_id) for node_id in local_ids))
        cached: dict[int, int] = {}
        for start in range(0, len(wanted), 900):
            chunk = wanted[start:start + 900]
            cached.update(self.local.execute(
                "SELECT local_id, remote_id FROM replication_node_map WHERE local_id IN ("
                + ",".join("?" * len(chunk)) + ")", chunk,
            ).fetchall())
        missing = [node_id for node_id in wanted if node_id not in cached]
        if not missing:
            return cached, {}
        nodes = []
        for start in range(0, len(missing), 900):
            chunk = missing[start:start + 900]
            nodes.extend(self.local.execute(
                "SELECT id, target, sequence, length, residue_vector FROM nodes WHERE id IN ("
                + ",".join("?" * len(chunk)) + ")", chunk,
            ).fetchall())
        found = {row[0] for row in nodes}
        dangling = [node_id for node_id in missing if node_id not in found]
        if dangling:
            raise RowDataError(f"row references local node id(s) {dangling[:5]} absent from nodes")
        resolved = self._resolve_remote_nodes(nodes)
        # Commit the node rows on their own, before the dependent upsert: it
        # releases the unique-index and FOR SHARE locks at once (holding them
        # through a multi-MB upsert deadlocked against other boxes' node
        # inserts in the soak), and makes the ids safe to cache right away. A
        # crash after this commit leaves only nodes that would be created
        # anyway; the rows that reference them are still dirty.
        self._commit_remote("nodes", len(nodes))
        self.local.execute("BEGIN IMMEDIATE")
        self._persist_map(resolved)
        self.local.execute("COMMIT")
        return {**cached, **resolved}, {}

    def _persist_map(self, pending: dict[int, int]) -> None:
        if pending:
            self.local.executemany(
                "INSERT OR IGNORE INTO replication_node_map(local_id, remote_id) VALUES(?, ?)",
                list(pending.items()),
            )

    def replicate_nodes(self) -> None:
        """Map every local node (including ones no replicated row references
        yet) to its Hetzner id, creating it there if needed."""
        def quarantine(row: tuple, error: BaseException) -> None:
            self.local.execute(
                "INSERT OR REPLACE INTO replication_quarantine VALUES('nodes', ?, NULL, ?, ?)",
                (str(row[0]), f"{type(error).__name__}: {error}"[:2000], time.time()),
            )

        def send(rows: Sequence[tuple]) -> None:
            resolved = self._resolve_remote_nodes(rows)
            self._commit_remote("nodes", len(rows))
            self.local.execute("BEGIN IMMEDIATE")
            self._persist_map(resolved)
            self.local.execute("COMMIT")
            self._trace("marked", "nodes", len(rows))

        while True:
            rows = self.local.execute(
                """SELECT n.id, n.target, n.sequence, n.length, n.residue_vector
                   FROM nodes n
                   WHERE NOT EXISTS (SELECT 1 FROM replication_node_map m WHERE m.local_id=n.id)
                     AND NOT EXISTS (SELECT 1 FROM replication_quarantine q
                                     WHERE q.table_name='nodes' AND q.row_key=CAST(n.id AS TEXT))
                   ORDER BY n.id LIMIT ?""",
                (self.batch_size,),
            ).fetchall()
            if not rows:
                return
            sent, bad = self._send_isolated("nodes", rows, send, quarantine)
            self._bump("sent", "nodes", sent)
            if bad:
                self._bump("quarantined", "nodes", bad)
            if len(rows) < self.batch_size:
                return

    # -- plain row tables -------------------------------------------------------
    def _quarantine_row(self, table: str) -> Callable[[tuple, BaseException], None]:
        def quarantine(row: tuple, error: BaseException) -> None:
            self.local.execute("BEGIN IMMEDIATE")
            self.local.execute(
                "INSERT OR REPLACE INTO replication_quarantine VALUES(?, ?, ?, ?, ?)",
                (table, str(row[0]), row[1], f"{type(error).__name__}: {error}"[:2000],
                 time.time()),
            )
            self._mark(table, [(row[0], row[1])], QUARANTINED)
            self.local.execute("COMMIT")
        return quarantine

    def replicate_rows(self, table: str) -> None:
        spec = ROW_TABLES[table]
        columns: list[str] = spec["columns"]
        node_positions = [columns.index(name) for name in spec["node_columns"]]

        def send(rows: Sequence[tuple]) -> None:
            pending: dict[int, int] = {}
            payload = [list(row[2:]) for row in rows]
            if node_positions:
                cached, pending = self._remote_ids_for(
                    row[position] for row in payload for position in node_positions
                )
                mapping = {**cached, **pending}
                for row in payload:
                    for position in node_positions:
                        row[position] = mapping[int(row[position])]
            self._remote().executemany(
                spec["sql"], [tuple(row) for row in payload],
                max_stmt_length=MAX_STATEMENT_BYTES,
            )
            self._commit_remote(table, len(rows))
            # Remote is durable from here on; a crash before COMMIT below only
            # leaves these rows dirty (re-sent, idempotent).
            self.local.execute("BEGIN IMMEDIATE")
            self._persist_map(pending)
            self._mark(table, [(row[0], row[1]) for row in rows], time.time())
            self.local.execute("COMMIT")
            self._trace("marked", table, len(rows))

        while True:
            rows = self.local.execute(
                f"SELECT rowid, repl_token, {', '.join(columns)} FROM {table} "
                "WHERE replicated_at IS NULL ORDER BY rowid LIMIT ?",
                (self.batch_size,),
            ).fetchall()
            if not rows:
                return
            sent, bad = self._send_isolated(table, rows, send, self._quarantine_row(table))
            self._bump("sent", table, sent)
            if bad:
                self._bump("quarantined", table, bad)
            if len(rows) < self.batch_size:
                return

    # -- exact runs (unit replication + completeness barrier) -------------------
    def replicate_exact_runs(self) -> None:
        keys = self.local.execute(
            """SELECT target, run_id FROM exact_runs WHERE replicated_at IS NULL
               UNION
               SELECT target, run_id FROM exact_run_observations WHERE replicated_at IS NULL"""
        ).fetchall()
        for target, run_id in keys:
            # One read transaction = one consistent snapshot of the run row,
            # its observation list, and the barrier check.
            self.local.execute("BEGIN")
            try:
                run = self.local.execute(
                    f"SELECT rowid, repl_token, {', '.join(EXACT_RUN_COLUMNS)} "
                    "FROM exact_runs WHERE target=? AND run_id=?",
                    (target, run_id),
                ).fetchone()
                members = self.local.execute(
                    "SELECT rowid, repl_token, observation_id FROM exact_run_observations "
                    "WHERE target=? AND run_id=?",
                    (target, run_id),
                ).fetchall()
                unconfirmed = self.local.execute(
                    """SELECT COUNT(*) FROM exact_run_observations e
                       LEFT JOIN observations o ON o.observation_id=e.observation_id
                       WHERE e.target=? AND e.run_id=?
                         AND (o.observation_id IS NULL OR o.replicated_at IS NULL
                              OR o.replicated_at < 0)""",
                    (target, run_id),
                ).fetchone()[0]
            finally:
                self.local.execute("COMMIT")
            if unconfirmed:
                self._bump("deferred", "exact_runs", 1)
                continue
            unit = (target, run_id, run, members)

            def send(units: Sequence[tuple]) -> None:
                (u_target, u_run_id, u_run, u_members), = units
                remote = self._remote()
                if u_run is not None:
                    remote.execute(_EXACT_RUN_UPSERT_SQL, tuple(u_run[2:]))
                remote.execute(
                    "DELETE FROM exact_run_observations WHERE target=%s AND run_id=%s",
                    (u_target, u_run_id),
                )
                if u_members:
                    remote.executemany(
                        _EXACT_RUN_OBS_INSERT_SQL,
                        [(u_target, u_run_id, member[2]) for member in u_members],
                        max_stmt_length=MAX_STATEMENT_BYTES,
                    )
                self._commit_remote("exact_runs", 1)
                self.local.execute("BEGIN IMMEDIATE")
                now = time.time()
                if u_run is not None:
                    self._mark("exact_runs", [(u_run[0], u_run[1])], now)
                self._mark("exact_run_observations",
                           [(member[0], member[1]) for member in u_members], now)
                self.local.execute("COMMIT")
                self._trace("marked", "exact_runs", 1)

            def quarantine(u: tuple, error: BaseException) -> None:
                u_target, u_run_id, u_run, u_members = u
                self.local.execute("BEGIN IMMEDIATE")
                self.local.execute(
                    "INSERT OR REPLACE INTO replication_quarantine VALUES('exact_runs', ?, ?, ?, ?)",
                    (f"{u_target}\x1f{u_run_id}", u_run[1] if u_run else None,
                     f"{type(error).__name__}: {error}"[:2000], time.time()),
                )
                if u_run is not None:
                    self._mark("exact_runs", [(u_run[0], u_run[1])], QUARANTINED)
                self._mark("exact_run_observations",
                           [(member[0], member[1]) for member in u_members], QUARANTINED)
                self.local.execute("COMMIT")

            sent, bad = self._send_isolated("exact_runs", [unit], send, quarantine)
            self._bump("sent", "exact_runs", sent)
            if bad:
                self._bump("quarantined", "exact_runs", bad)

    # -- driver -----------------------------------------------------------------
    def replicate_once(self) -> dict[str, Any]:
        self.stats = {}
        started = time.time()
        # Order matters: nodes before the rows that reference them (a miss is
        # still resolved on the fly), observations before exact runs (barrier).
        try:
            self.replicate_nodes()
            for table in ROW_TABLES:
                self.replicate_rows(table)
            self.replicate_exact_runs()
        except BaseException:
            # Never leave a local transaction open across an aborted pass (it
            # would hold the write lock against the producer); nothing in it
            # is needed -- an unmarked row is simply re-sent next pass.
            if self.local.in_transaction:
                self.local.execute("ROLLBACK")
            self._rollback_remote()
            raise
        self.stats["pass_seconds"] = round(time.time() - started, 3)
        return self.stats


def backlog(local: sqlite3.Connection) -> dict[str, Any]:
    dirty = {
        table: local.execute(
            f"SELECT COUNT(*) FROM {table} WHERE replicated_at IS NULL"
        ).fetchone()[0]
        for table in TOKEN_TABLES
    }
    dirty["nodes"] = local.execute(
        "SELECT COUNT(*) FROM nodes n WHERE NOT EXISTS "
        "(SELECT 1 FROM replication_node_map m WHERE m.local_id=n.id)"
    ).fetchone()[0]
    quarantined = {
        table: count for table, count in local.execute(
            "SELECT table_name, COUNT(*) FROM replication_quarantine GROUP BY table_name"
        )
    }
    return {"dirty": dirty, "quarantined": quarantined}


def write_status(path: Path, status: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(status, sort_keys=True, indent=1))
    os.replace(tmp, path)


def mark_all_dirty(local: sqlite3.Connection) -> None:
    """Maintenance: Hetzner lost data or was restored/renumbered. Forget every
    confirmation and node mapping; the next pass re-sends everything
    (idempotent) and re-resolves every node by identity."""
    local.execute("BEGIN IMMEDIATE")
    for table in TOKEN_TABLES:
        local.execute(f"UPDATE {table} SET replicated_at=NULL")
    local.execute("DELETE FROM replication_node_map")
    local.execute("DELETE FROM replication_quarantine")
    local.execute("COMMIT")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--local", required=True, type=Path)
    parser.add_argument("--remote-creds", type=Path,
                        help="explicit SN68_MYSQL_* file (all five keys); default: "
                             "the environment / /root/.sn68_mysql_creds as db.connect()")
    parser.add_argument("--loop-interval", type=float, default=0.0,
                        help="run forever; idle poll every N seconds (30 on master)")
    parser.add_argument("--batch-size", type=int, default=REPLICATION_BATCH)
    parser.add_argument("--status", action="store_true",
                        help="print the local backlog and exit (no Hetzner connection)")
    parser.add_argument("--trace", action="store_true",
                        help="print REPLICATE_TRACE lines at remote commit and local mark")
    parser.add_argument("--mark-all-dirty", action="store_true",
                        help="maintenance: re-send everything on the next pass")
    args = parser.parse_args(argv)

    if not args.local.is_file():
        print(f"REPLICATE_LOCAL_MISSING_DB {args.local}", file=sys.stderr)
        return 1
    local = open_local(args.local)
    try:
        ensure_local_schema(local)
    except SchemaError as error:
        print(f"REPLICATE_LOCAL_SCHEMA_REFUSED {error}", file=sys.stderr)
        return 2
    if args.status:
        print(json.dumps(backlog(local), sort_keys=True))
        return 0
    if args.mark_all_dirty:
        mark_all_dirty(local)
        print("REPLICATE_LOCAL_MARKED_ALL_DIRTY", flush=True)
        return 0

    # Single instance per store: two replicators would both be correct (every
    # step is idempotent and token-guarded) but would double the Hetzner load.
    lock_handle = open(f"{args.local}.replicate.lock", "w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"REPLICATE_LOCAL_ALREADY_RUNNING {args.local}", file=sys.stderr)
        return 75
    if args.remote_creds:
        load_remote_env(args.remote_creds)
    status_path = Path(f"{args.local}.replication_status.json")
    replicator = Replicator(local, connect_remote, batch_size=args.batch_size)
    fault = float(os.environ.get("REPLICATE_LOCAL_FAULT_KILL_AFTER_COMMIT", "0") or 0)
    if fault > 0:
        # Fault injection for tools/replication_soak.py ONLY: SIGKILL ourselves
        # right after a Hetzner commit, before the local mark -- the one window
        # where a crash leaves remote ahead of local. Never set in production.
        import random
        import signal

        def _fault(table: str, count: int) -> None:
            if random.random() < fault:
                print(f"REPLICATE_FAULT_SELF_KILL {time.time():.3f} {table} {count}", flush=True)
                os.kill(os.getpid(), signal.SIGKILL)

        replicator.after_remote_commit = _fault
    if args.trace:
        replicator.trace = lambda event, table, count: print(
            f"REPLICATE_TRACE {time.time():.3f} {event} {table} {count}", flush=True)
    failures = 0
    last_ok: float | None = None
    while True:
        try:
            stats = replicator.replicate_once()
        except Exception as error:  # noqa: BLE001
            replicator.close()
            failures += 1
            message = f"{type(error).__name__}: {str(error)[:500]}"
            print(f"REPLICATE_LOCAL_FAILED {message}", file=sys.stderr, flush=True)
            try:
                write_status(status_path, {
                    "ok": False, "error": message, "at": time.time(),
                    "consecutive_failures": failures, "last_ok_at": last_ok,
                    **backlog(local),
                })
            except Exception:  # noqa: BLE001
                pass
            if not args.loop_interval:
                return 1
            time.sleep(min(args.loop_interval * 2 ** min(failures, 4), 300.0))
            continue
        failures = 0
        last_ok = time.time()
        report = {"ok": True, "at": last_ok, "last_ok_at": last_ok, **stats, **backlog(local)}
        write_status(status_path, report)
        if stats.get("sent") or stats.get("quarantined") or stats.get("deferred"):
            print(f"REPLICATE_LOCAL_OK {json.dumps(report, sort_keys=True)}", flush=True)
        if not args.loop_interval:
            return 0
        if not stats.get("sent"):
            # Idle: drop the Hetzner connection rather than let it go stale
            # behind the tunnel; the next pass with work reconnects (~3 s).
            replicator.close()
        time.sleep(args.loop_interval)


if __name__ == "__main__":
    raise SystemExit(main())
