"""Central SN68 DB connection layer (Hetzner MySQL 8.0).

Replaces the per-machine sqlite3 in mutation_network.py / ingest_backfill.py.
Connection params come from env vars (SN68_MYSQL_HOST/PORT/USER/PASSWORD/DB),
sourced from a 0600 creds file on each host -- never hardcoded.

`connect()` returns a Connection that exposes sqlite3's `conn.execute()` /
`conn.commit()` surface (backed by pymysql), so the writers only need SQL
syntax changes (`?` -> `%s`, `ON CONFLICT DO UPDATE` -> `ON DUPLICATE KEY
UPDATE`, `INSERT OR REPLACE` -> `REPLACE INTO`) rather than cursor plumbing.
"""
from __future__ import annotations

import os
import sqlite3
import time
from functools import wraps
from pathlib import Path
from typing import Any, Iterable

import MySQLdb
import MySQLdb.cursors

DEFAULT_CREDS = "/root/.sn68_mysql_creds"

# MySQL error codes (per errmsg-utf8.txt / include/mysqld_error.h).
ER_LOCK_DEADLOCK = 1213  # "Deadlock found when trying to get lock; try restarting transaction"
ER_LOCK_WAIT_TIMEOUT = 1205  # "Lock wait timeout exceeded; try restarting transaction"

# Both are transient lock-contention errors: the blocking transaction commits
# (or is killed) and the lock is released, so a re-run of the WHOLE transaction
# succeeds. Retry both, not just 1213 -- a producer write holding a row briefly
# is the common case, and 1205 is that same contention after the wait timeout.
_RETRYABLE = (ER_LOCK_DEADLOCK, ER_LOCK_WAIT_TIMEOUT)


def is_deadlock(exc: BaseException) -> bool:
    """True when ``exc`` is a retryable MySQL lock error (1213 or 1205).

    InnoDB rolls back the deadlock victim's *entire* transaction, and a lock
    wait timeout (1205) leaves the transaction intact but the blocking lock is
    transient -- in both cases the caller must re-run its whole transaction
    block, not just the one statement that raised.
    """
    return (
        isinstance(exc, MySQLdb.OperationalError)
        and bool(getattr(exc, "args", ()))
        and exc.args[0] in _RETRYABLE
    )


def deadlock_retry(
    *, attempts: int = 5, base_delay: float = 0.1, max_delay: float = 2.0
):
    """Decorator: on a MySQL lock error (1213 deadlock or 1205 lock-wait), re-run
    the WHOLE wrapped call.

    A deadlock rolls back the victim's full transaction, so re-running only
    the failing statement would leave the transaction half-applied. The
    wrapped function (typically a ``with db.connect() ...`` write block) is
    re-executed from the top. Non-lock errors propagate unchanged. The lock
    error is re-raised only after ``attempts`` tries, with short exponential
    backoff in between.
    """

    def decorate(fn: Any) -> Any:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = base_delay
            for attempt in range(1, attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except MySQLdb.OperationalError as exc:
                    if not is_deadlock(exc):
                        raise
                    if attempt == attempts:
                        raise
                    time.sleep(delay)
                    delay = min(delay * 2, max_delay)
            raise RuntimeError("deadlock retry: unreachable")  # pragma: no cover

        return wrapper

    return decorate


def load_creds(path: str = DEFAULT_CREDS) -> None:
    """Source a KEY=value creds file into os.environ (existing values win)."""
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key, value.strip())


class _Cursor:
    """Minimal sqlite3-cursor-like facade over a pymysql cursor."""

    def __init__(self, cursor: Any) -> None:
        self._cursor = cursor

    def fetchone(self) -> Any:
        return self._cursor.fetchone()

    def fetchall(self) -> list[Any]:
        return self._cursor.fetchall()

    def __iter__(self) -> Iterable[Any]:
        return iter(self._cursor)


class Connection:
    """pymysql wrapped to expose sqlite3's conn.execute()/commit()/close()."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def execute(self, sql: str, params: Any = None) -> _Cursor:
        cur = self._conn.cursor()
        cur.execute(sql, params)
        return _Cursor(cur)

    def stream(
        self, sql: str, params: Any = None, *, batch_size: int = 20000
    ) -> Iterable[Any]:
        """Run a large read on an unbuffered (server-side) cursor.

        Rows are yielded as the server sends them, so ``read_timeout`` bounds
        each socket read rather than the time until a fully materialised
        result is ready. The caller must keep consuming promptly: the server
        aborts a stalled unbuffered result after ``net_write_timeout`` (60s on
        the Hetzner server), so do per-row heavy work after the read, not
        inside it.
        """
        cur = self._conn.cursor(MySQLdb.cursors.SSCursor)
        try:
            cur.execute(sql, params)
            while True:
                rows = cur.fetchmany(batch_size)
                if not rows:
                    return
                yield from rows
        finally:
            cur.close()

    def executemany(
        self, sql: str, seq: Any, *, max_stmt_length: int | None = None
    ) -> None:
        """``cursor.executemany``; mysqlclient folds an ``INSERT ... VALUES``
        (including ``ON DUPLICATE KEY UPDATE``) into multi-row statements of at
        most ``max_stmt_length`` bytes each (its default is 64 KiB). Raise it
        to cut round-trips for large rows; keep it below the server's
        ``max_allowed_packet``."""
        cur = self._conn.cursor()
        if max_stmt_length is not None:
            cur.max_stmt_length = max_stmt_length
        cur.executemany(sql, seq)

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Connection":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        # Match sqlite3's context manager: commit on success, rollback on error.
        if exc_type is None:
            self._conn.commit()
        else:
            self._conn.rollback()
        self.close()


LOCAL_SCHEMA_PATH = Path(__file__).resolve().parent / "local_schema.sql"


class LocalConnection:
    """Local SQLite primary store (ARCHITECTURE_REDESIGN new finding 8).

    Mirrors :class:`Connection`'s surface so mutation_network's write path can
    select either backend without changing its call sites. The two backends
    are NOT SQL-dialect-compatible (``%s`` vs ``?`` placeholders,
    ``ON DUPLICATE KEY UPDATE`` vs ``ON CONFLICT ... DO UPDATE``, no
    ``FOR SHARE`` in SQLite) -- callers that write must use the dialect-aware
    SQL constants in mutation_network.py, selected by ``isinstance(connection,
    LocalConnection)``, not pass MySQL SQL to this class.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not path.is_file() or path.stat().st_size == 0
        self._conn = sqlite3.connect(str(path), timeout=30, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=OFF")
        self._conn.execute("BEGIN")
        if is_new:
            self._conn.executescript(LOCAL_SCHEMA_PATH.read_text())

    def execute(self, sql: str, params: Any = None) -> "_LocalCursor":
        cur = self._conn.cursor()
        cur.execute(sql, params or ())
        return _LocalCursor(cur)

    def executemany(
        self, sql: str, seq: Any, *, max_stmt_length: int | None = None
    ) -> None:
        # max_stmt_length is a MySQL/mysqlclient statement-folding knob; SQLite
        # has no equivalent (executemany is already one prepared statement
        # re-bound per row, not a folded multi-row INSERT), so it's accepted
        # and ignored for call-site compatibility with Connection.executemany.
        self._conn.executemany(sql, seq)

    def commit(self) -> None:
        self._conn.commit()
        self._conn.execute("BEGIN")

    def rollback(self) -> None:
        self._conn.rollback()
        self._conn.execute("BEGIN")

    def close(self) -> None:
        try:
            self._conn.commit()
        except sqlite3.Error:
            pass
        self._conn.close()

    def __enter__(self) -> "LocalConnection":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is None:
            self._conn.commit()
        else:
            self._conn.rollback()
        self.close()


class _LocalCursor:
    def __init__(self, cursor: sqlite3.Cursor) -> None:
        self._cursor = cursor

    def fetchall(self) -> list[Any]:
        return self._cursor.fetchall()

    def fetchone(self) -> Any:
        return self._cursor.fetchone()

    def __iter__(self) -> Any:
        return iter(self._cursor)


def connect_local(path: str | Path) -> LocalConnection:
    """Open (creating/migrating if needed) master's local SQLite primary."""
    return LocalConnection(Path(path))


def connect(_path: str | None = None) -> Connection:
    """Open a connection using the SN68_MYSQL_* env vars.

    ``_path`` is accepted (and ignored) for call-site compatibility with
    ``sqlite3.connect(path)``.
    """
    load_creds()
    return Connection(
        MySQLdb.connect(
            host=os.environ.get("SN68_MYSQL_HOST", "46.225.101.210"),
            port=int(os.environ.get("SN68_MYSQL_PORT", "3306")),
            user=os.environ.get("SN68_MYSQL_USER", "sn68"),
            password=os.environ.get("SN68_MYSQL_PASSWORD", ""),
            db=os.environ.get("SN68_MYSQL_DB", "sn68"),
            charset="utf8mb4",
            autocommit=False,
            # Fail lock waits fast (10s) instead of the InnoDB default 50s so a
            # transient 1205 is retried by deadlock_retry rather than stalling
            # a writer for minutes behind a producer-held row.
            init_command="SET SESSION innodb_lock_wait_timeout = 10",
            # zlib protocol compression (CLIENT_COMPRESS): the metrics/provenance
            # JSON is text and compresses ~5-10x over the public wire.
            compress=True,
            # Socket-level timeouts so a half-dead connection (server closed but
            # the client is still waiting) fails fast instead of hanging a write
            # indefinitely; the deadlock_retry wrapper then re-runs the whole
            # transaction on a fresh connection.
            connect_timeout=10,
            read_timeout=60,
            write_timeout=60,
        )
    )
