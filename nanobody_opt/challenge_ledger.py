"""Append-only, hash-chained SQLite ledger of per-epoch challenges.

Persists the inputs needed to reproduce each challenge (the ruleset, in
derivation order) and the challenge itself. Uses only the Python standard
library (``sqlite3``, ``hashlib``, ``json``, ``random``).

Tables:
  rulesets            one row per distinct ruleset fingerprint (write-once)
  epoch_challenges    one row per epoch (write-once), linked in a sha256 chain
  challenge_conflicts audit trail of rejected conflicting writes

The ledger owns its transactions, so open it with :func:`connect`
(``isolation_level=None``, plus foreign_keys and recursive_triggers pragmas).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import random
import re
import sqlite3
from typing import Any, Mapping

# ---------------------------------------------------------------------------
# Filled from nanobody_opt/rules.py fingerprint()/challenge(). These constants
# and field sources MUST stay byte-identical to rules.fingerprint() so the
# ledger's ruleset_sha256 matches what the live writer computes; a drift fails
# loudly on the first record_challenge (it re-checks the fingerprint).
# ---------------------------------------------------------------------------
_UPSTREAM_SHA = "ce19ea8cbe656671c3440ba58b4093ff80798d4f"
_FILTER_SHA = "fe2d5868126406a0de8953d26ecd8bdccbdbd029"
_PROFILE = "nanobody-pr100-v1"
_AGGSCAN_CAPS = ["-0.68002724866200", "1.58023066762925"]

CANONICAL_ENSURE_ASCII = True  # matches rules.digest() json.dumps default


def fingerprint_payload(config: Mapping[str, Any]) -> dict:
    """Return the exact object rules.fingerprint() hashes for ``ruleset_sha256``."""
    return {
        "profile": _PROFILE,
        "upstream_sha": _UPSTREAM_SHA,
        "filter_sha": _FILTER_SHA,
        "target": config["protein_selection"]["nanobody"],
        "requirements": config["nanobody_requirements"],
        "germline_filtering": config.get("germline_filtering"),
        "aggrescan_caps": _AGGSCAN_CAPS,
    }


def load_config(config_bytes: bytes) -> dict:
    """Parse raw config bytes (JSON) keeping mapping insertion order."""
    return json.loads(config_bytes.decode("utf-8"))


def config_target(config: Mapping[str, Any]) -> Any:
    """Return the value placed in ``challenge["target"]``."""
    return config["protein_selection"]["nanobody"]["target"]

DERIVATION_VERSION = "challenge-v1/cpython-random-mt19937"
SUPPORTED_DERIVATION_VERSIONS = frozenset({DERIVATION_VERSION})
GENESIS_PREV_ROW_SHA256 = "0" * 64
SOURCES = ("live", "backfill")
CHALLENGE_KEYS = frozenset({
    "epoch", "target", "epoch_length", "epoch_start_block",
    "epoch_start_block_hash", "ruleset_sha256", "allowed_v", "allowed_cdr3_length",
})
_BLOCK_HASH_RE = re.compile(r"0x[0-9a-fA-F]+")


class LedgerError(Exception):
    """Base class for ledger errors."""


class ChallengeMismatch(LedgerError):
    """A challenge does not match its re-derivation or the stored row."""


class ChallengeConflict(LedgerError):
    """A different challenge is already recorded for this epoch (audited)."""


class RulesetConflict(LedgerError):
    """Same fingerprint but a different category order (audited).

    The fingerprint uses sort_keys=True, so it cannot see the order of the
    V-categories. The derivation, however, depends on that order.
    """


class LedgerIntegrityError(LedgerError):
    """Stored bytes, hashes or the chain do not verify."""


# ---------------------------------------------------------------------------
# Canonical JSON + sha256 (the only serializer used anywhere in this module)
# ---------------------------------------------------------------------------
def canonical_json(obj: Any) -> bytes:
    """Return canonical JSON as UTF-8 bytes.

    Uses sort_keys, compact separators and rejects NaN/Infinity.
    """
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"),
        ensure_ascii=CANONICAL_ENSURE_ASCII, allow_nan=False,
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    """Return the lowercase hex sha256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def fingerprint(config: Mapping[str, Any]) -> str:
    """Return ``ruleset_sha256``: the sha256 of the canonical fingerprint payload."""
    return sha256_hex(canonical_json(fingerprint_payload(config)))


def chain_hash(prev_row_sha256: str, seq: int, epoch: int,
               challenge_sha256: str, ruleset_sha256: str, source: str) -> str:
    """Return ``row_sha256 = sha256(prev|seq|epoch|challenge_sha256|ruleset_sha256|source)``.

    No field can contain ``|``: hashes are hex, seq and epoch are decimal, and
    source is live or backfill. The join is therefore unambiguous.
    """
    msg = f"{prev_row_sha256}|{int(seq)}|{int(epoch)}|{challenge_sha256}|{ruleset_sha256}|{source}"
    return sha256_hex(msg.encode("ascii"))


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_block_hash(block_hash: str) -> None:
    # The upstream code does int(block_hash[2:], 16), which drops two
    # characters whether or not they are "0x". int() also accepts "_" and
    # whitespace. Require a strict 0x-hex string so the seed is unambiguous.
    if not isinstance(block_hash, str) or not _BLOCK_HASH_RE.fullmatch(block_hash):
        raise ValueError(f"block hash must match 0x[0-9a-fA-F]+, got {block_hash!r}")


def _diff_keys(a: Mapping, b: Mapping) -> list[str]:
    missing = object()
    return sorted(k for k in set(a) | set(b) if a.get(k, missing) != b.get(k, missing))


# ---------------------------------------------------------------------------
# Derivation (mirrors the host's challenge() exactly)
# ---------------------------------------------------------------------------
def ordered_germline_filtering(config: Mapping[str, Any]) -> dict:
    """Return the derivation inputs with the category order made explicit.

    Shape: ``{"categories": [["V1", [...]], ...], "length_range": {...}}``.
    Store it with canonical_json: the categories list keeps its order, and
    sort_keys only reorders the object keys.
    """
    settings = config["germline_filtering"]
    categories = settings["v_neighborhood_categories"]
    return {
        "categories": [[key, list(categories[key])] for key in categories],
        "length_range": dict(settings["cdr3_length_range"]),
    }


def _draw(germline_filtering: Mapping[str, Any], block_hash: str) -> tuple[str, str, int]:
    """Make the RNG draws in the same order and count as upstream.

    The ``len(genes) > 1`` branch matters. For a single-gene category,
    upstream skips the gene draw entirely, and that shifts the length draw.
    """
    rng = random.Random(int(block_hash[2:], 16))
    categories = germline_filtering["categories"]
    category_key = rng.choice([key for key, _ in categories])
    genes = dict((key, g) for key, g in categories)[category_key]
    allowed_v = rng.choice(genes) if len(genes) > 1 else genes[0]
    lengths = germline_filtering["length_range"]
    allowed_cdr3_length = rng.choice(list(range(lengths["min"], lengths["max"] + 1)))
    return category_key, allowed_v, allowed_cdr3_length


def _build_challenge(epoch: int, target: Any, epoch_length: int, block_hash: str,
                     ruleset_sha256: str, germline_filtering: Mapping[str, Any]) -> tuple[dict, str]:
    out = {
        "epoch": epoch,
        "target": target,
        "epoch_length": epoch_length,
        "epoch_start_block": epoch * epoch_length,
        "epoch_start_block_hash": block_hash.lower(),
        "ruleset_sha256": ruleset_sha256,
    }
    category_key, allowed_v, allowed_cdr3_length = _draw(germline_filtering, block_hash)
    out["allowed_v"] = allowed_v
    out["allowed_cdr3_length"] = allowed_cdr3_length
    return out, category_key


def derive_challenge(config: Mapping[str, Any], epoch: int, block_hash: str,
                     epoch_length: int) -> tuple[dict, str]:
    """Re-derive the host's challenge. Returns ``(challenge, category_key)``.

    The returned challenge dict has exactly the host's keys. category_key is
    returned on its own because the host does not include it in the challenge.
    """
    if not (_is_int(epoch) and epoch >= 0 and _is_int(epoch_length) and epoch_length > 0):
        raise ValueError("epoch must be an int >= 0 and epoch_length an int > 0")
    _check_block_hash(block_hash)
    return _build_challenge(epoch, config_target(config), epoch_length, block_hash,
                            fingerprint(config), ordered_germline_filtering(config))


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS rulesets (
    ruleset_sha256      TEXT PRIMARY KEY
                        CHECK (length(ruleset_sha256) = 64 AND ruleset_sha256 NOT GLOB '*[^0-9a-f]*'),
    canonical_bytes     BLOB NOT NULL CHECK (typeof(canonical_bytes) = 'blob'),
    germline_filtering_json TEXT NOT NULL CHECK (json_valid(germline_filtering_json)),
    config_path         TEXT,
    config_git_commit   TEXT,
    first_seen_epoch    INTEGER NOT NULL CHECK (first_seen_epoch >= 0),
    recorded_at         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS epoch_challenges (
    seq                    INTEGER PRIMARY KEY AUTOINCREMENT,
    epoch                  INTEGER NOT NULL UNIQUE CHECK (epoch >= 0),
    epoch_length           INTEGER NOT NULL CHECK (epoch_length > 0),
    epoch_start_block      INTEGER NOT NULL,
    epoch_start_block_hash TEXT NOT NULL
                           CHECK (epoch_start_block_hash GLOB '0x?*'
                                  AND epoch_start_block_hash = lower(epoch_start_block_hash)),
    target                 TEXT NOT NULL,
    category_key           TEXT NOT NULL,
    allowed_v           TEXT NOT NULL,
    allowed_cdr3_length         INTEGER NOT NULL,
    ruleset_sha256         TEXT NOT NULL REFERENCES rulesets(ruleset_sha256),
    derivation_version     TEXT NOT NULL,
    challenge_bytes        BLOB NOT NULL CHECK (typeof(challenge_bytes) = 'blob'),
    challenge_sha256       TEXT NOT NULL CHECK (length(challenge_sha256) = 64),
    source                 TEXT NOT NULL CHECK (source IN ('live', 'backfill')),
    provenance             TEXT NOT NULL,
    recorded_at            TEXT NOT NULL,
    prev_row_sha256        TEXT NOT NULL UNIQUE CHECK (length(prev_row_sha256) = 64),  -- UNIQUE: no forks
    row_sha256             TEXT NOT NULL UNIQUE CHECK (length(row_sha256) = 64),
    CHECK (epoch_start_block = epoch * epoch_length)
);
CREATE INDEX IF NOT EXISTS epoch_challenges_by_ruleset ON epoch_challenges(ruleset_sha256);

CREATE TABLE IF NOT EXISTS challenge_conflicts (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    kind              TEXT NOT NULL CHECK (kind IN ('challenge_hash', 'ruleset_order')),
    epoch             INTEGER NOT NULL,
    ruleset_sha256    TEXT NOT NULL,
    existing_sha256   TEXT NOT NULL,   -- challenge_sha256, or sha256(germline_filtering_json)
    incoming_sha256   TEXT NOT NULL,
    incoming_bytes    BLOB NOT NULL,   -- incoming challenge_bytes, or germline_filtering_json
    source            TEXT NOT NULL,
    provenance        TEXT NOT NULL,
    recorded_at       TEXT NOT NULL
);

-- Append-only guards. INSERT OR REPLACE deletes rows internally, and SQLite
-- only fires DELETE triggers for that when PRAGMA recursive_triggers = ON
-- (set by connect()).
CREATE TRIGGER IF NOT EXISTS rulesets_no_update BEFORE UPDATE ON rulesets
BEGIN SELECT RAISE(ABORT, 'rulesets is append-only'); END;
CREATE TRIGGER IF NOT EXISTS rulesets_no_delete BEFORE DELETE ON rulesets
BEGIN SELECT RAISE(ABORT, 'rulesets is append-only'); END;
CREATE TRIGGER IF NOT EXISTS epoch_challenges_no_update BEFORE UPDATE ON epoch_challenges
BEGIN SELECT RAISE(ABORT, 'epoch_challenges is append-only'); END;
CREATE TRIGGER IF NOT EXISTS epoch_challenges_no_delete BEFORE DELETE ON epoch_challenges
BEGIN SELECT RAISE(ABORT, 'epoch_challenges is append-only'); END;
CREATE TRIGGER IF NOT EXISTS challenge_conflicts_no_update BEFORE UPDATE ON challenge_conflicts
BEGIN SELECT RAISE(ABORT, 'challenge_conflicts is append-only'); END;
CREATE TRIGGER IF NOT EXISTS challenge_conflicts_no_delete BEFORE DELETE ON challenge_conflicts
BEGIN SELECT RAISE(ABORT, 'challenge_conflicts is append-only'); END;
"""


def connect(path: str) -> sqlite3.Connection:
    """Open the ledger, apply the pragmas, create the schema and return the connection.

    ``isolation_level=None`` lets the ledger issue ``BEGIN IMMEDIATE`` itself,
    which serializes writers across processes. foreign_keys and
    recursive_triggers are per-connection settings, so every connection
    must be opened through this function.
    """
    conn = sqlite3.connect(path, isolation_level=None, timeout=30.0)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA recursive_triggers = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(SCHEMA_SQL)
    return conn


def _require_autocommit(conn: sqlite3.Connection) -> None:
    if conn.isolation_level is not None or conn.in_transaction:
        raise LedgerError("open the ledger with connect(); it must own its transactions")


def _fetch_row(conn: sqlite3.Connection, epoch: int) -> sqlite3.Row | None:
    cur = conn.cursor()
    cur.row_factory = sqlite3.Row
    return cur.execute("SELECT * FROM epoch_challenges WHERE epoch = ?", (epoch,)).fetchone()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def record_challenge(config_bytes: bytes, challenge: Mapping[str, Any], derivation_version: str,
                     source: str, provenance: str | Mapping[str, Any], *,
                     conn: sqlite3.Connection, config_path: str | None = None,
                     config_git_commit: str | None = None) -> dict:
    """Record a challenge idempotently. Returns ``{"inserted", "seq", "row_sha256", "challenge_sha256"}``.

    Steps:
      1. Re-derive the challenge from ``config_bytes`` and the challenge's own
         epoch, block hash and epoch_length. The challenge must match the
         re-derivation exactly, including ruleset_sha256, which checks the
         fingerprint TODO above.
      2. Insert the ruleset once (``ON CONFLICT DO NOTHING``). If a stored
         ruleset has the same fingerprint but a different category order, a
         ``ruleset_order`` conflict is audited and RulesetConflict is raised.
      3. Insert the epoch row with ``ON CONFLICT(epoch) DO NOTHING``.
         - Same epoch and same challenge_sha256: no-op, returns inserted=False.
         - Same epoch and different hash: writes a ``challenge_hash`` row to
           challenge_conflicts, commits it, then raises ChallengeConflict.
      4. ``row_sha256`` extends the chain from the highest seq (the genesis
         prev is 64 zeros). seq is fixed inside ``BEGIN IMMEDIATE``, so the
         hash can include it.

    Chain order is insertion order, not epoch order, so a backfill that
    arrives after live rows is appended after them. ``first_seen_epoch`` is the
    epoch of the first recorded row that used the ruleset. Use
    ``MIN(epoch)`` over epoch_challenges for the earliest epoch.
    """
    _require_autocommit(conn)
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    if derivation_version not in SUPPORTED_DERIVATION_VERSIONS:
        raise ValueError(f"unsupported derivation_version {derivation_version!r}")
    if set(challenge) != CHALLENGE_KEYS:
        raise ChallengeMismatch(f"challenge keys differ from host shape: {sorted(set(challenge) ^ CHALLENGE_KEYS)}")

    config = load_config(config_bytes)
    payload_bytes = canonical_json(fingerprint_payload(config))
    ruleset_sha = sha256_hex(payload_bytes)
    if challenge["ruleset_sha256"] != ruleset_sha:
        raise ChallengeMismatch(
            f"ruleset_sha256 {challenge['ruleset_sha256']} != local fingerprint {ruleset_sha}: "
            "FINGERPRINT_FIELDS / canonical_json differ from the host (see HOST-REPO TODO)")
    expected, category_key = derive_challenge(
        config, challenge["epoch"], challenge["epoch_start_block_hash"], challenge["epoch_length"])
    if canonical_json(expected) != canonical_json(dict(challenge)):
        raise ChallengeMismatch(f"challenge differs from re-derivation in {_diff_keys(expected, challenge)}")

    epoch = expected["epoch"]
    challenge_bytes = canonical_json(expected)
    challenge_sha = sha256_hex(challenge_bytes)
    gf_json = canonical_json(ordered_germline_filtering(config)).decode("utf-8")
    prov = provenance if isinstance(provenance, str) else canonical_json(dict(provenance)).decode("utf-8")
    target = expected["target"]
    target_col = target if isinstance(target, str) else canonical_json(target).decode("utf-8")
    now = _utc_now()

    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            "INSERT INTO rulesets (ruleset_sha256, canonical_bytes, germline_filtering_json, config_path,"
            " config_git_commit, first_seen_epoch, recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(ruleset_sha256) DO NOTHING",
            (ruleset_sha, payload_bytes, gf_json, config_path, config_git_commit, epoch, now))
        stored_bytes, stored_gf = conn.execute(
            "SELECT canonical_bytes, germline_filtering_json FROM rulesets WHERE ruleset_sha256 = ?",
            (ruleset_sha,)).fetchone()
        if bytes(stored_bytes) != payload_bytes:
            raise LedgerIntegrityError(f"rulesets[{ruleset_sha}].canonical_bytes do not match the payload")
        if stored_gf != gf_json:
            conn.execute(
                "INSERT INTO challenge_conflicts (kind, epoch, ruleset_sha256, existing_sha256,"
                " incoming_sha256, incoming_bytes, source, provenance, recorded_at)"
                " VALUES ('ruleset_order', ?, ?, ?, ?, ?, ?, ?, ?)",
                (epoch, ruleset_sha, sha256_hex(stored_gf.encode("utf-8")),
                 sha256_hex(gf_json.encode("utf-8")), gf_json.encode("utf-8"), source, prov, now))
            conn.execute("COMMIT")
            raise RulesetConflict(
                f"ruleset {ruleset_sha} already stored with a different category order; "
                "the same fingerprint would derive different challenges")

        existing = conn.execute(
            "SELECT seq, challenge_sha256, row_sha256 FROM epoch_challenges WHERE epoch = ?",
            (epoch,)).fetchone()
        if existing is not None:
            ex_seq, ex_sha, ex_row = existing
            if ex_sha == challenge_sha:
                conn.execute("COMMIT")
                return {"inserted": False, "seq": ex_seq, "row_sha256": ex_row, "challenge_sha256": ex_sha}
            conn.execute(
                "INSERT INTO challenge_conflicts (kind, epoch, ruleset_sha256, existing_sha256,"
                " incoming_sha256, incoming_bytes, source, provenance, recorded_at)"
                " VALUES ('challenge_hash', ?, ?, ?, ?, ?, ?, ?, ?)",
                (epoch, ruleset_sha, ex_sha, challenge_sha, challenge_bytes, source, prov, now))
            conn.execute("COMMIT")
            raise ChallengeConflict(
                f"epoch {epoch} already recorded as {ex_sha} (seq {ex_seq}); incoming {challenge_sha} audited")

        last = conn.execute(
            "SELECT seq, row_sha256 FROM epoch_challenges ORDER BY seq DESC LIMIT 1").fetchone()
        seq_floor = conn.execute(
            "SELECT seq FROM sqlite_sequence WHERE name = 'epoch_challenges'").fetchone()
        prev = last[1] if last else GENESIS_PREV_ROW_SHA256
        seq = max(last[0] if last else 0, seq_floor[0] if seq_floor else 0) + 1
        row_sha = chain_hash(prev, seq, epoch, challenge_sha, ruleset_sha, source)
        cur = conn.execute(
            "INSERT INTO epoch_challenges (seq, epoch, epoch_length, epoch_start_block,"
            " epoch_start_block_hash, target, category_key, allowed_v, allowed_cdr3_length,"
            " ruleset_sha256, derivation_version, challenge_bytes, challenge_sha256, source,"
            " provenance, recorded_at, prev_row_sha256, row_sha256)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(epoch) DO NOTHING",
            (seq, epoch, expected["epoch_length"], expected["epoch_start_block"],
             expected["epoch_start_block_hash"], target_col, category_key,
             expected["allowed_v"], expected["allowed_cdr3_length"], ruleset_sha,
             derivation_version, challenge_bytes, challenge_sha, source, prov, now, prev, row_sha))
        if cur.rowcount != 1:  # unreachable under BEGIN IMMEDIATE; kept as a guard
            raise LedgerIntegrityError(f"epoch {epoch} appeared mid-transaction")
        conn.execute("COMMIT")
        return {"inserted": True, "seq": seq, "row_sha256": row_sha, "challenge_sha256": challenge_sha}
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def get_challenge(epoch: int, *, conn: sqlite3.Connection, config_bytes: bytes | None = None,
                  block_hash: str | None = None, epoch_length: int | None = None) -> dict:
    """Return the challenge for ``epoch``. Reads the ledger first; never writes.

    If a row exists, its challenge_bytes are checked against challenge_sha256
    and returned with ``stored=True``. If no row exists and config_bytes,
    block_hash and epoch_length are all given, the challenge is derived in
    memory and returned with ``stored=False``. Nothing is persisted; call
    record_challenge for that. If no row exists and inputs are missing,
    LookupError is raised.

    Returns ``{"challenge", "category_key", "stored", "seq", "source", "row_sha256"}``.
    """
    row = _fetch_row(conn, epoch)
    if row is not None:
        if sha256_hex(bytes(row["challenge_bytes"])) != row["challenge_sha256"]:
            raise LedgerIntegrityError(f"epoch {epoch}: challenge_bytes do not hash to challenge_sha256")
        return {"challenge": json.loads(bytes(row["challenge_bytes"])), "category_key": row["category_key"],
                "stored": True, "seq": row["seq"], "source": row["source"], "row_sha256": row["row_sha256"]}
    if config_bytes is None or block_hash is None or epoch_length is None:
        raise LookupError(f"epoch {epoch} not recorded; pass config_bytes, block_hash, epoch_length to derive")
    challenge, category_key = derive_challenge(load_config(config_bytes), epoch, block_hash, epoch_length)
    return {"challenge": challenge, "category_key": category_key,
            "stored": False, "seq": None, "source": None, "row_sha256": None}


def verify_challenge(epoch: int, challenge: Mapping[str, Any], *, conn: sqlite3.Connection) -> dict:
    """Re-derive ``epoch`` from stored inputs only and check it against the row and ``challenge``.

    The stored inputs are the ruleset's ordered germline_filtering_json and the
    row's block hash, epoch and epoch_length. Checks:
      * ruleset canonical_bytes hash to ruleset_sha256, and their germline_filtering
        content matches germline_filtering_json (order-insensitive)
      * the re-derived category_key, allowed_v and allowed_cdr3_length match the
        row columns
      * sha256 of the re-derived challenge equals the stored challenge_sha256
        and the sha256 of the supplied ``challenge``
    The supplied challenge may also carry a ``category_key``, which is then
    compared too. Raises ChallengeMismatch, listing every failed check, on any
    difference. Returns the verified stored challenge.
    """
    row = _fetch_row(conn, epoch)
    if row is None:
        raise LookupError(f"epoch {epoch} not recorded; nothing to verify against")
    if row["derivation_version"] not in SUPPORTED_DERIVATION_VERSIONS:
        raise ChallengeMismatch(f"epoch {epoch}: unsupported derivation_version {row['derivation_version']!r}")
    rs = conn.execute(
        "SELECT canonical_bytes, germline_filtering_json FROM rulesets WHERE ruleset_sha256 = ?",
        (row["ruleset_sha256"],)).fetchone()
    if rs is None:
        raise LedgerIntegrityError(f"epoch {epoch}: ruleset {row['ruleset_sha256']} missing")
    canonical_bytes, gf_json = bytes(rs[0]), rs[1]
    gf = json.loads(gf_json)

    problems: list[str] = []
    if sha256_hex(canonical_bytes) != row["ruleset_sha256"]:
        problems.append("ruleset canonical_bytes do not hash to ruleset_sha256")
    payload_gf = json.loads(canonical_bytes).get("germline_filtering", {})
    if (payload_gf.get("v_neighborhood_categories") != dict((k, g) for k, g in gf["categories"])
            or payload_gf.get("cdr3_length_range") != gf["length_range"]):
        problems.append("germline_filtering_json content differs from fingerprinted germline_filtering")
    if sha256_hex(bytes(row["challenge_bytes"])) != row["challenge_sha256"]:
        problems.append("stored challenge_bytes do not hash to challenge_sha256")

    stored = json.loads(bytes(row["challenge_bytes"]))
    rederived, category_key = _build_challenge(
        row["epoch"], stored.get("target"), row["epoch_length"], row["epoch_start_block_hash"],
        row["ruleset_sha256"], gf)
    rederived_sha = sha256_hex(canonical_json(rederived))

    for name, want in (("category_key", category_key), ("allowed_v", rederived["allowed_v"]),
                       ("allowed_cdr3_length", rederived["allowed_cdr3_length"])):
        if row[name] != want:
            problems.append(f"stored {name}={row[name]!r} != re-derived {want!r}")
    if rederived_sha != row["challenge_sha256"]:
        problems.append(f"re-derived challenge_sha256 {rederived_sha} != stored {row['challenge_sha256']}")

    supplied = dict(challenge)
    supplied_key = supplied.pop("category_key", None)
    if supplied_key is not None and supplied_key != category_key:
        problems.append(f"supplied category_key={supplied_key!r} != re-derived {category_key!r}")
    for name in ("allowed_v", "allowed_cdr3_length"):
        if supplied.get(name) != rederived[name]:
            problems.append(f"supplied {name}={supplied.get(name)!r} != re-derived {rederived[name]!r}")
    try:
        supplied_sha = sha256_hex(canonical_json(supplied))
    except (TypeError, ValueError) as exc:
        supplied_sha = f"<unserializable: {exc}>"
    if supplied_sha != rederived_sha:
        problems.append(f"supplied challenge_sha256 {supplied_sha} != re-derived {rederived_sha}"
                        f" (differs in {_diff_keys(rederived, supplied)})")

    if problems:
        raise ChallengeMismatch(f"epoch {epoch}: " + "; ".join(problems))
    return rederived


def verify_chain(conn: sqlite3.Connection) -> str | None:
    """Walk epoch_challenges in seq order and recompute every link.

    Returns the head row_sha256, or None if the table is empty. Publish the
    head somewhere outside this file. The triggers only stop accidental
    writes; someone with write access to the file can still rewrite the
    whole chain.
    """
    prev = GENESIS_PREV_ROW_SHA256
    for seq, epoch, c_sha, rs_sha, source, c_bytes, prev_col, row_sha in conn.execute(
            "SELECT seq, epoch, challenge_sha256, ruleset_sha256, source, challenge_bytes,"
            " prev_row_sha256, row_sha256 FROM epoch_challenges ORDER BY seq"):
        if prev_col != prev:
            raise LedgerIntegrityError(f"seq {seq}: prev_row_sha256 does not link to previous row")
        if sha256_hex(bytes(c_bytes)) != c_sha:
            raise LedgerIntegrityError(f"seq {seq}: challenge_bytes do not hash to challenge_sha256")
        if chain_hash(prev, seq, epoch, c_sha, rs_sha, source) != row_sha:
            raise LedgerIntegrityError(f"seq {seq}: row_sha256 does not recompute")
        prev = row_sha
    return None if prev == GENESIS_PREV_ROW_SHA256 else prev
