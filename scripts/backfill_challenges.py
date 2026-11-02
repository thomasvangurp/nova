#!/usr/bin/env python3
"""Backfill nanobody epoch challenges into the challenge ledger, derived from validator-accepted submissions.

Modes (each one includes the mode before it):
  plan      Build the plan with plan_from_accepted.build_plan (majority vote over accepted
            submissions). Check it, then compare it with rows already in the ledger. Read-only.
  validate  plan, then replay the whole ledger: sha256(challenge_bytes), the chain_hash links,
            the ruleset canonical_bytes hash, verify_chain, and a per-row re-derivation
            canonical_json(derive_challenge(...)) == challenge_bytes. Read-only.
            Exit 1 if there are integrity errors. Calibration findings (era/annotation) are
            reported separately and do not fail validate.
  apply     validate, then write every entry with action == "record" through
            challenge_ledger.record_challenge with source="backfill". Each row is read back and
            diffed. verify_chain always runs in a finally block. Apply refuses unless
            ERAS_CONFIRMED is True, --yes is given, --report is given, and validate shows no
            blockers.

Exit codes: 0 ok, 1 integrity failure or abort, 2 refusal (gates, placeholders, usage).
"""
from __future__ import annotations

import argparse
import collections
import functools
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from nanobody_opt import challenge_ledger
from nanobody_opt import rules  # its source bytes are pinned into tool_sha (verified_by = rules.challenge)

import plan_from_accepted  # sibling script: the script dir is sys.path[0] when this runs as a script

# =============================== LIVE-STATE (filled) ===============================
# Accepted-submissions DB (read-only) and the query that lists candidate epochs. The
# epochs query takes (to_epoch, target) and returns DISTINCT integer epochs. The block hash
# comes from the Bittensor subtensor (config.subtensor.chain_endpoint), not a separate HTTP
# archive endpoint.
ACCEPTED_DB_PATH = "/root/artifacts/nanobody_mutations.sqlite"
ACCEPTED_TARGET = "Q9NZQ7"
ACCEPTED_EPOCHS_SQL = (
    "SELECT DISTINCT o.epoch FROM observations o JOIN nodes n ON n.id = o.node_id "
    "WHERE o.source = 'public_validator_api' AND o.score_kind = 'validator_final_rank_sum' "
    "AND o.epoch <= ? AND n.target = ? ORDER BY o.epoch"
)

# H4-style era gate. Set True only after a human has confirmed the config (ruleset) and
# epoch_length were the validator's rules for EVERY epoch in the backfill range.
ERAS_CONFIRMED = False
# ================================================================================

EXPECTED_DERIVATION_VERSION = "accepted-submissions-v1"
PLAN_SOURCE = "validator_accepted_submissions"   # goes into provenance.derived_from, never into the source column
LEDGER_SOURCE = "backfill"                       # the epoch_challenges.source column is always this
VERIFIED_BY = "rules.challenge(block_hash)"

EPOCH_COLS = (
    "seq", "epoch", "epoch_length", "epoch_start_block", "epoch_start_block_hash", "target",
    "category_key", "allowed_v", "allowed_cdr3_length", "ruleset_sha256", "derivation_version",
    "challenge_bytes", "challenge_sha256", "source", "provenance", "recorded_at",
    "prev_row_sha256", "row_sha256",
)
RULESET_COLS = (
    "ruleset_sha256", "canonical_bytes", "germline_filtering_json", "config_path",
    "config_git_commit", "first_seen_epoch", "recorded_at",
)
_EPOCH_SELECT = ", ".join(EPOCH_COLS)
_RULESET_SELECT = ", ".join(RULESET_COLS)
_BLOCK_HASH_RE = re.compile(r"^0x[0-9a-fA-F]{64}$")


class Refusal(Exception):
    """A gate, placeholder, or usage problem. Nothing was written. Exit 2."""


class Abort(Exception):
    """An integrity or safety violation during the run. Exit 1."""


# ------------------------------------------------------------------ small helpers
def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _canon(obj) -> bytes:
    return challenge_ledger.canonical_json(obj)


def _is_bytes(v) -> bool:
    return isinstance(v, (bytes, bytearray, memoryview))


def _as_bytes(v) -> bytes:
    if _is_bytes(v):
        return bytes(v)
    if isinstance(v, str):
        return v.encode("utf-8")
    raise TypeError(f"expected bytes/str, got {type(v).__name__}")


def _as_text(v) -> str:
    if isinstance(v, str):
        return v
    if _is_bytes(v):
        return bytes(v).decode("utf-8")
    raise TypeError(f"expected str/bytes, got {type(v).__name__}")


def _cmp_key(v) -> str:
    return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _norm_col(v):
    """Turn allowed_v / allowed_cdr3_length columns into comparable values (JSON text decoded)."""
    if _is_bytes(v):
        v = bytes(v).decode("utf-8")
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return v  # stays a raw string; the comparison below will flag any mismatch
    return v


def _short(v) -> str:
    if _is_bytes(v):
        b = bytes(v)
        return f"<{len(b)} bytes sha256={_sha256(b)[:12]}>"
    r = repr(v)
    return r if len(r) <= 120 else r[:117] + "..."


def _same(a, b) -> bool:
    if _is_bytes(a) or _is_bytes(b):
        try:
            return _as_bytes(a) == _as_bytes(b)
        except TypeError:
            return False
    return a == b


def _rows(conn, sql, params=()) -> list[dict]:
    cur = conn.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, tuple(r))) for r in cur.fetchall()]


def _row_by_epoch(conn, epoch: int) -> dict | None:
    rows = _rows(conn, f"SELECT {_EPOCH_SELECT} FROM epoch_challenges WHERE epoch = ?", (epoch,))
    return rows[0] if rows else None


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------ block hash (Bittensor subtensor)
_BLOCK_HASH_CACHE: dict[int, str] = {}


def fetch_block_hash(subtensor, block: int) -> str:
    """Return the 0x-prefixed hash of a block via subtensor.determine_block_hash.

    Errors are sanitised (type name only; never the endpoint or raw text). A
    None/malformed result means the node lacks this header (pruned / warp gap).
    """
    if isinstance(block, bool) or not isinstance(block, int) or block < 0:
        raise RuntimeError(f"invalid block number: {block!r}")
    cached = _BLOCK_HASH_CACHE.get(block)
    if cached is not None:
        return cached
    try:
        value = subtensor.determine_block_hash(block)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"block hash lookup failed for block {block} ({type(exc).__name__})") from None
    if not isinstance(value, str) or not _BLOCK_HASH_RE.match(value):
        raise RuntimeError(f"block hash unavailable for block {block}: endpoint lacks this header "
                           f"(pruned/warp-gap node); use an archive --subtensor.chain_endpoint")
    value = value.lower()
    _BLOCK_HASH_CACHE[block] = value
    return value


def _build_subtensor(args):
    """Build the Bittensor subtensor from the CLI flags (lazy import)."""
    import bittensor as bt  # lazy: only needed on the chain-connected host
    kwargs = {"network": args.subtensor_network}
    if args.subtensor_chain_endpoint:
        kwargs["chain_endpoint"] = args.subtensor_chain_endpoint
    return bt.Subtensor(**kwargs)  # capital S: the live code uses bt.Subtensor(network=...)


# ------------------------------------------------------------------ ledger access
def _open_ledger(path: Path):
    if not path.is_file():
        raise Refusal(f"ledger not found: {path} (refusing to let connect() create an empty ledger)")
    return challenge_ledger.connect(str(path))


def _check_schema(conn) -> None:
    for table, cols in (("epoch_challenges", EPOCH_COLS), ("rulesets", RULESET_COLS)):
        actual = {tuple(r)[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        missing = sorted(set(cols) - actual)
        if missing:
            raise Refusal(f"ledger table {table} lacks expected columns {missing}")


def _load_ledger(conn) -> tuple[list[dict], dict[str, dict]]:
    rows = _rows(conn, f"SELECT {_EPOCH_SELECT} FROM epoch_challenges ORDER BY seq")
    rulesets = {r["ruleset_sha256"]: r for r in _rows(conn, f"SELECT {_RULESET_SELECT} FROM rulesets")}
    return rows, rulesets


def _resolve_epoch_length(rows: list[dict], arg: int | None) -> int:
    seen = sorted({r["epoch_length"] for r in rows})
    if arg is not None:
        if arg not in seen:
            raise Refusal(f"--epoch-length {arg} is used by no ledger row (ledger has {seen})")
        return arg
    if len(seen) != 1:
        raise Refusal(f"ledger rows use several epoch_length values {seen}; pass --epoch-length explicitly")
    return seen[0]


def _column_checks(row: dict, ch: dict) -> list[str]:
    """Check that the denormalised columns agree with the challenge they were recorded from."""
    out = []
    for k in ("epoch", "epoch_start_block_hash", "ruleset_sha256"):
        if ch.get(k) != row[k]:
            out.append(f"column {k}={_short(row[k])} but challenge {k}={_short(ch.get(k))}")
    for k in ("allowed_v", "allowed_cdr3_length"):
        if k not in ch:
            out.append(f"challenge lacks {k}")
        elif _cmp_key(_norm_col(row[k])) != _cmp_key(ch[k]):
            out.append(f"column {k}={_short(row[k])} disagrees with challenge {k}")
    if "target" in ch and ch["target"] != row["target"]:
        out.append(f"column target={_short(row['target'])} but challenge target={_short(ch['target'])}")
    return out


# ------------------------------------------------------------------ validate: ledger replay
def replay_ledger(rows, rulesets, config, config_fp, chain_error) -> dict:
    integrity: list[dict] = []
    era: list[dict] = []

    def bad(row, msg):
        integrity.append({"seq": row.get("seq"), "epoch": row.get("epoch"), "error": msg})

    for rs_sha, rs in rulesets.items():
        try:
            got = _sha256(_as_bytes(rs["canonical_bytes"]))
        except TypeError as e:
            integrity.append({"ruleset_sha256": rs_sha, "error": f"canonical_bytes unreadable: {e}"})
            continue
        if got != rs_sha:
            integrity.append({"ruleset_sha256": rs_sha, "error": f"sha256(canonical_bytes)={got}"})

    # Which config to re-derive with, per ruleset. The current config file is used for its own
    # fingerprint. For any other ruleset, use the stored canonical_bytes only if they load as a config
    # whose fingerprint round-trips. Otherwise the row can't be re-derived here: that's an era finding.
    cfg_cache: dict[str, tuple] = {config_fp: (config, "config_file")}
    unrederivable: dict[str, str] = {}

    def cfg_for(rs_sha):
        if rs_sha in cfg_cache:
            return cfg_cache[rs_sha]
        if rs_sha in unrederivable:
            return None
        try:
            cand = json.loads(_as_bytes(rulesets[rs_sha]["canonical_bytes"]))
            if challenge_ledger.fingerprint(cand) == rs_sha:
                cfg_cache[rs_sha] = (cand, "ledger_ruleset")
                return cfg_cache[rs_sha]
            unrederivable[rs_sha] = "stored canonical_bytes is not a full config (fingerprint differs)"
        except Exception as e:  # noqa: BLE001 - every reason is reported
            unrederivable[rs_sha] = f"stored canonical_bytes not usable as config: {type(e).__name__}: {e}"
        return None

    prev_row_sha = None
    seen_epochs: set[int] = set()
    rederived = 0
    for i, row in enumerate(rows):
        seq, epoch = row["seq"], row["epoch"]
        if row["source"] not in ("live", "backfill"):
            bad(row, f"source={row['source']!r} not in (live, backfill)")
        if epoch in seen_epochs:
            bad(row, "duplicate epoch")
        seen_epochs.add(epoch)

        # Chain link. The genesis rule for the first row's prev_row_sha256 is left to verify_chain.
        if i > 0 and row["prev_row_sha256"] != prev_row_sha:
            bad(row, "prev_row_sha256 does not equal the preceding row's row_sha256")
        prev_row_sha = row["row_sha256"]
        exp_row = challenge_ledger.chain_hash(row["prev_row_sha256"], seq, epoch, row["challenge_sha256"],
                                              row["ruleset_sha256"], row["source"])
        if exp_row != row["row_sha256"]:
            bad(row, "row_sha256 != chain_hash(prev, seq, epoch, challenge_sha256, ruleset_sha256, source)")

        try:
            cb = _as_bytes(row["challenge_bytes"])
        except TypeError as e:
            bad(row, f"challenge_bytes unreadable: {e}")
            continue
        if _sha256(cb) != row["challenge_sha256"]:
            bad(row, "challenge_sha256 != sha256(challenge_bytes)")
        try:
            ch = json.loads(cb)
        except ValueError:
            bad(row, "challenge_bytes is not JSON")
            continue
        if not isinstance(ch, dict):
            bad(row, "challenge_bytes is not a JSON object")
            continue
        if _canon(ch) != cb:
            bad(row, "challenge_bytes is not canonical_json")
        for msg in _column_checks(row, ch):
            bad(row, msg)

        if row["ruleset_sha256"] not in rulesets:
            bad(row, "ruleset_sha256 has no rulesets row")
            continue
        picked = cfg_for(row["ruleset_sha256"])
        if picked is None:
            era.append({"kind": "not_rederivable", "seq": seq, "epoch": epoch,
                        "ruleset_sha256": row["ruleset_sha256"], "reason": unrederivable[row["ruleset_sha256"]]})
            continue
        cfg, cfg_src = picked
        try:
            rch, rck = challenge_ledger.derive_challenge(cfg, epoch, row["epoch_start_block_hash"],
                                                         row["epoch_length"])
        except Exception as e:  # noqa: BLE001
            bad(row, f"derive_challenge raised {type(e).__name__}: {e}")
            continue
        if _canon(rch) != cb:
            bad(row, f"canonical_json(derive_challenge(...)) != challenge_bytes (config from {cfg_src}; "
                     f"rederived sha256={_sha256(_canon(rch))[:12]} ledger={row['challenge_sha256'][:12]})")
        if rck != row["category_key"]:
            bad(row, f"rederived category_key={_short(rck)} != column {_short(row['category_key'])}")
        rederived += 1
        if row["ruleset_sha256"] != config_fp:
            era.append({"kind": "other_ruleset_rederived", "seq": seq, "epoch": epoch,
                        "ruleset_sha256": row["ruleset_sha256"]})

    if chain_error:
        integrity.append({"error": f"verify_chain: {chain_error}"})
    return {"rows": len(rows), "rederived": rederived, "verify_chain": chain_error or "ok",
            "integrity_errors": integrity, "era_findings": era}


# ------------------------------------------------------------------ plan
def _provenance(ent: dict, plan: dict, run_id: str, tool_sha: str) -> str:
    extra = {
        "derived_from": PLAN_SOURCE,
        "verified_by": VERIFIED_BY,
        "derivation_version": ent["derivation_version"],
        "config_file": plan["config_file"],
        "config_sha256": plan["config_sha256"],
        "ruleset_sha256": plan["ruleset_sha256"],
        "run_id": run_id,
        "tool_sha": tool_sha,
    }
    base = ent["provenance"]
    clash = sorted(k for k in extra if k in base and base[k] != extra[k])
    if clash:
        raise Abort(f"epoch {ent['epoch']}: plan provenance conflicts with backfill provenance on {clash}")
    return _canon({**base, **extra}).decode("utf-8")


def _check_record_entry(ent, plan, config, config_fp, epoch_length, run_id, tool_sha) -> list[dict]:
    e = ent["epoch"]
    out: list[dict] = []

    def err(m):
        out.append({"epoch": e, "error": m})

    if ent.get("source") != PLAN_SOURCE:
        err(f"entry source={ent.get('source')!r}, expected {PLAN_SOURCE!r}")
    if ent.get("derivation_version") != plan.get("derivation_version"):
        err(f"entry derivation_version={ent.get('derivation_version')!r} != plan's")
    if not isinstance(ent.get("provenance"), dict):
        err("entry provenance is not a dict")
    ch = ent.get("challenge")
    if not isinstance(ch, dict):
        err("entry challenge is not a dict")
        return out
    if ch.get("epoch") != e:
        err(f"challenge epoch={ch.get('epoch')!r} != entry epoch")
    if ch.get("ruleset_sha256") != config_fp:
        err("challenge ruleset_sha256 != fingerprint(config)")
    for k in ("allowed_v", "allowed_cdr3_length"):
        if k not in ch:
            err(f"challenge lacks {k}")
    bh = ch.get("epoch_start_block_hash")
    if not (isinstance(bh, str) and _BLOCK_HASH_RE.match(bh)):
        err(f"challenge epoch_start_block_hash malformed: {_short(bh)}")
        return out
    try:
        rch, rck = challenge_ledger.derive_challenge(config, e, bh, epoch_length)
    except Exception as x:  # noqa: BLE001
        err(f"derive_challenge raised {type(x).__name__}: {x}")
        return out
    if _canon(rch) != _canon(ch):
        err("plan challenge != derive_challenge(config, epoch, epoch_start_block_hash, epoch_length)")
    if rck != ent.get("category_key"):
        err(f"plan category_key={_short(ent.get('category_key'))} != rederived {_short(rck)}")
    if isinstance(ent.get("provenance"), dict):
        try:
            _provenance(ent, plan, run_id, tool_sha)
        except Abort as x:
            err(str(x))
    return out


def check_plan(plan, epochs, config, config_bytes, config_path, config_fp, epoch_length, run_id, tool_sha):
    errs: list[dict] = []
    expected_hdr = {"derivation_version": EXPECTED_DERIVATION_VERSION,
                    "config_sha256": _sha256(config_bytes),  # build_plan must have read the same bytes
                    "ruleset_sha256": config_fp,
                    "config_file": config_path.name}
    for k, v in expected_hdr.items():
        if plan.get(k) != v:
            errs.append({"error": f"plan {k}={plan.get(k)!r}, expected {v!r}"})
    raw = plan.get("entries")
    if not isinstance(raw, list):
        errs.append({"error": "plan entries is not a list"})
        return errs, []
    requested, seen, entries = set(epochs), set(), []
    for ent in raw:
        e = ent.get("epoch") if isinstance(ent, dict) else None
        if isinstance(e, bool) or not isinstance(e, int):
            errs.append({"error": f"malformed plan entry: {ent!r:.200}"})
            continue
        if e in seen:
            errs.append({"epoch": e, "error": "duplicate plan entry"})
            continue
        seen.add(e)
        if e not in requested:
            errs.append({"epoch": e, "error": "plan entry for an epoch that was not requested"})
            continue
        action, reason = ent.get("action"), ent.get("reason")
        if action not in ("record", "skip"):
            errs.append({"epoch": e, "error": f"unknown action {action!r}"})
            continue
        if action == "skip" and not (isinstance(reason, str) and reason.strip()):
            errs.append({"epoch": e, "error": "skip without a reason"})
            continue
        entries.append(ent)
        if action == "record":
            errs.extend(_check_record_entry(ent, plan, config, config_fp, epoch_length, run_id, tool_sha))
    for e in sorted(requested - seen):
        errs.append({"epoch": e, "error": "requested epoch missing from plan (silent drop)"})
    entries.sort(key=lambda x: x["epoch"])
    return errs, entries


def _era_groups(rows, config_fp) -> list[dict]:
    g: dict[tuple, dict] = {}
    for r in rows:
        k = (r["ruleset_sha256"], r["epoch_length"], r["derivation_version"], r["source"])
        x = g.setdefault(k, {"ruleset_sha256": k[0], "epoch_length": k[1], "derivation_version": k[2],
                             "source": k[3], "rows": 0, "min_epoch": r["epoch"], "max_epoch": r["epoch"],
                             "is_current_config": k[0] == config_fp})
        x["rows"] += 1
        x["min_epoch"] = min(x["min_epoch"], r["epoch"])
        x["max_epoch"] = max(x["max_epoch"], r["epoch"])
    return sorted(g.values(), key=lambda x: x["min_epoch"])


def calibrate(plan, entries, ledger_by_epoch, rows, config_fp, limit) -> dict:
    """Compare the plan with rows already recorded (live rows are ground truth)."""
    agree, different, new, skipped_recorded = [], [], [], []
    reasons: collections.Counter = collections.Counter()
    for ent in entries:
        e, row = ent["epoch"], ledger_by_epoch.get(ent["epoch"])
        if ent["action"] != "record":
            reasons[ent["reason"]] += 1
            if row is not None:
                skipped_recorded.append({"epoch": e, "reason": ent["reason"], "ledger_source": row["source"]})
            continue
        if row is None:
            new.append(e)
            continue
        want = _canon(ent["challenge"])
        try:
            same = _as_bytes(row["challenge_bytes"]) == want
        except TypeError:
            same = False
        if same:
            agree.append(e)
        else:
            different.append({"epoch": e, "ledger_source": row["source"],
                              "ledger_ruleset_sha256": row["ruleset_sha256"],
                              "ledger_challenge_sha256": row["challenge_sha256"],
                              "plan_challenge_sha256": _sha256(want),
                              "ledger_category_key": row["category_key"],
                              "plan_category_key": ent.get("category_key")})
    inside = []
    if new:
        lo, hi = new[0], new[-1]
        inside = [{"epoch": r["epoch"], "seq": r["seq"], "ruleset_sha256": r["ruleset_sha256"],
                   "source": r["source"]}
                  for r in rows if lo <= r["epoch"] <= hi and r["ruleset_sha256"] != config_fp]
    return {
        "annotation": {"overlap_agree": len(agree), "agree_epochs": agree,
                       "already_recorded_different": different,
                       "skip_reasons": dict(reasons), "skipped_but_recorded": skipped_recorded,
                       "plan_summary": plan.get("summary")},
        "era": {"groups": _era_groups(rows, config_fp),
                "other_ruleset_inside_backfill_range": inside,
                "current_config_seen_in_live_rows": any(
                    r["source"] == "live" and r["ruleset_sha256"] == config_fp for r in rows)},
        "would_record": new,
        "would_record_with_limit": new if limit is None else new[:limit],
    }


# ------------------------------------------------------------------ apply
def _readback_diffs(conn, ent, want: bytes, prov: str, plan, epoch_length, result) -> list[str]:
    row = _row_by_epoch(conn, ent["epoch"])
    if row is None:
        return ["row not found after record_challenge"]
    d = []
    expect = {"source": LEDGER_SOURCE,
              "derivation_version": ent["derivation_version"],
              "ruleset_sha256": plan["ruleset_sha256"],
              "epoch_length": epoch_length,
              "epoch_start_block_hash": ent["challenge"]["epoch_start_block_hash"],
              "category_key": ent["category_key"],
              "challenge_sha256": _sha256(want)}
    for k, v in expect.items():
        if row[k] != v:
            d.append(f"{k}: ledger={_short(row[k])} expected={_short(v)}")
    if not _same(row["challenge_bytes"], want):
        d.append("challenge_bytes != canonical_json(plan challenge)")
    try:
        if _as_text(row["provenance"]) != prov:
            d.append("provenance differs from the canonical provenance sent")
    except TypeError as e:
        d.append(f"provenance unreadable: {e}")
    d.extend(_column_checks(row, ent["challenge"]))
    prev = conn.execute("SELECT row_sha256 FROM epoch_challenges WHERE seq < ? ORDER BY seq DESC LIMIT 1",
                        (row["seq"],)).fetchone()
    if prev is not None and row["prev_row_sha256"] != tuple(prev)[0]:
        d.append("prev_row_sha256 does not link to the preceding row")
    if challenge_ledger.chain_hash(row["prev_row_sha256"], row["seq"], row["epoch"], row["challenge_sha256"],
                                   row["ruleset_sha256"], row["source"]) != row["row_sha256"]:
        d.append("row_sha256 != chain_hash(...)")
    if isinstance(result, dict):
        for k, v in result.items():
            if k in row and not _same(row[k], v):
                d.append(f"record_challenge returned {k}={_short(v)} but row has {_short(row[k])}")
    return d


def apply_plan(conn, plan, entries, *, limit, epoch_length, config_bytes, config_path_label, git_commit,
               run_id, tool_sha, log: list) -> None:
    written = 0
    for ent in entries:
        e = ent["epoch"]
        if ent["action"] != "record":
            log.append({"epoch": e, "outcome": "skip", "reason": f"plan: {ent['reason']}"})
            continue
        want = _canon(ent["challenge"])
        existing = _row_by_epoch(conn, e)  # fresh read: the live writer may have appended since validate
        if existing is not None:
            if _same(existing["challenge_bytes"], want):
                log.append({"epoch": e, "outcome": "skip",
                            "reason": f"already-recorded-identical (source={existing['source']})"})
                continue
            raise Abort(f"epoch {e}: already recorded with a DIFFERENT challenge "
                        f"(ledger sha256={existing['challenge_sha256']} source={existing['source']}, "
                        f"plan sha256={_sha256(want)})")
        if limit is not None and written >= limit:
            log.append({"epoch": e, "outcome": "skip", "reason": f"limit ({limit}) reached"})
            continue
        prov = _provenance(ent, plan, run_id, tool_sha)
        result = challenge_ledger.record_challenge(
            config_bytes, ent["challenge"], ent["derivation_version"], LEDGER_SOURCE, prov,
            conn=conn, config_path=config_path_label, config_git_commit=git_commit)
        diffs = _readback_diffs(conn, ent, want, prov, plan, epoch_length, result)
        if diffs:
            log.append({"epoch": e, "outcome": "readback-mismatch", "diffs": diffs})
            raise Abort(f"epoch {e}: read-back mismatch: {diffs}")
        row = _row_by_epoch(conn, e)
        written += 1
        log.append({"epoch": e, "outcome": "recorded", "seq": row["seq"], "row_sha256": row["row_sha256"]})


# ------------------------------------------------------------------ provenance of tool + config
def _tool_fingerprint() -> tuple[str, dict]:
    files = {"backfill_challenges": Path(__file__), "challenge_ledger": Path(challenge_ledger.__file__),
             "rules": Path(rules.__file__), "plan_from_accepted": Path(plan_from_accepted.__file__)}
    per = {k: _sha256(p.read_bytes()) for k, p in files.items()}
    return _sha256(_canon(per)), per


def _config_git(config_path: Path, config_bytes: bytes):
    """-> (commit, repo_relative_path, problem). A commit is returned only if HEAD's blob == the bytes we read."""
    cwd = str(config_path.parent)

    def git(*a, text=True):
        p = subprocess.run(["git", "-C", cwd, *a], capture_output=True, text=text, timeout=30)
        if p.returncode != 0:
            msg = p.stderr if text else p.stderr.decode("utf-8", "replace")
            raise RuntimeError(f"git {a[0]} failed: {msg.strip()[:200]}")
        return p.stdout

    try:
        head = git("rev-parse", "HEAD").strip()
        rel = git("ls-files", "--full-name", "--error-unmatch", "--", config_path.name).strip()
        committed = git("cat-file", "blob", f"HEAD:{rel}", text=False)
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as e:
        return None, None, f"config git provenance unavailable: {e}"
    if committed != config_bytes:
        return None, rel, f"{rel} differs from HEAD {head[:12]}; config_git_commit would be misleading"
    return head, rel, None


def _candidate_epochs(obs_conn, to_epoch: int) -> list[int]:
    if ACCEPTED_EPOCHS_SQL.count("?") != 2:
        raise Refusal("ACCEPTED_EPOCHS_SQL must contain exactly two '?' (to_epoch, target)")
    vals = [tuple(r)[0] for r in obs_conn.execute(ACCEPTED_EPOCHS_SQL, (to_epoch, ACCEPTED_TARGET)).fetchall()]
    if any(isinstance(v, bool) or not isinstance(v, int) for v in vals):
        raise Refusal("ACCEPTED_EPOCHS_SQL returned non-integer epochs")
    if len(set(vals)) != len(vals):
        raise Refusal("ACCEPTED_EPOCHS_SQL must return DISTINCT epochs")
    if any(v > to_epoch for v in vals):
        raise Refusal("ACCEPTED_EPOCHS_SQL returned epochs above to_epoch (bound ignored)")
    return sorted(vals)


def _open_accepted_db(db_path):
    p = Path(db_path)
    if not p.is_file():
        raise Refusal(f"accepted-submissions DB not found: {p}")
    return sqlite3.connect(f"{p.resolve().as_uri()}?mode=ro", uri=True)


def _apply_blockers(integrity, calib, git_problem) -> list[str]:
    b = []
    if integrity:
        b.append(f"{len(integrity)} integrity error(s)")
    if not ERAS_CONFIRMED:
        b.append("ERAS_CONFIRMED is False")
    ann, era = calib["annotation"], calib["era"]
    if ann["already_recorded_different"]:
        b.append(f"{len(ann['already_recorded_different'])} already-recorded epoch(s) disagree with the plan")
    if ann["overlap_agree"] == 0:
        b.append("no overlap epoch where the plan reproduces a recorded row (derivation uncalibrated)")
    if not era["current_config_seen_in_live_rows"]:
        b.append("config ruleset never recorded by a live row")
    if era["other_ruleset_inside_backfill_range"]:
        b.append(f"{len(era['other_ruleset_inside_backfill_range'])} row(s) with another ruleset inside the backfill range")
    if git_problem:
        b.append(git_problem)
    return b


# ------------------------------------------------------------------ driver
def _run(args, report: dict) -> int:
    if args.mode == "apply":
        if not ERAS_CONFIRMED:
            raise Refusal("apply refused: ERAS_CONFIRMED is False (confirm eras for the whole range first)")
        if not args.yes:
            raise Refusal("apply refused: --yes is required")
        if not args.report:
            raise Refusal("apply refused: --report is required (audit trail)")
    if args.limit is not None and args.limit < 1:
        raise Refusal("--limit must be >= 1")

    config_path = args.config.resolve()
    config_bytes = config_path.read_bytes()  # read exactly once; build_plan's config_sha256 is checked against it
    config = json.loads(config_bytes)
    config_fp = challenge_ledger.fingerprint(config)
    tool_sha, tool_files = _tool_fingerprint()
    report.update({
        "config": {"path": str(config_path), "config_sha256": _sha256(config_bytes), "ruleset_sha256": config_fp},
        "tool_sha": tool_sha, "tool_files": tool_files,
        "live_state": {"ACCEPTED_DB_PATH": ACCEPTED_DB_PATH, "ACCEPTED_TARGET": ACCEPTED_TARGET,
                       "ACCEPTED_EPOCHS_SQL": ACCEPTED_EPOCHS_SQL, "ERAS_CONFIRMED": ERAS_CONFIRMED},
    })

    conn = _open_ledger(args.ledger)
    try:
        _check_schema(conn)
        rows, rulesets = _load_ledger(conn)
        if not rows:
            raise Refusal("ledger has no rows: nothing anchors --to-epoch or calibration")
        max_epoch = max(r["epoch"] for r in rows)
        to_epoch = max_epoch if args.to_epoch is None else args.to_epoch
        if to_epoch > max_epoch:
            raise Refusal(f"--to-epoch {to_epoch} exceeds max recorded epoch {max_epoch}")
        epoch_length = _resolve_epoch_length(rows, args.epoch_length)
        ledger_by_epoch = {r["epoch"]: r for r in rows}

        subtensor = _build_subtensor(args)
        obs = _open_accepted_db(args.accepted_db)
        try:
            epochs = _candidate_epochs(obs, to_epoch)
            plan = plan_from_accepted.build_plan(
                obs, epochs, epoch_length=epoch_length,
                fetch_block_hash=functools.partial(fetch_block_hash, subtensor),
                annotate=plan_from_accepted.pinned_annotator, config_path=config_path)
        finally:
            obs.close()

        plan_errs, entries = check_plan(plan, epochs, config, config_bytes, config_path, config_fp,
                                        epoch_length, report["run_id"], tool_sha)
        calib = calibrate(plan, entries, ledger_by_epoch, rows, config_fp, args.limit)
        report.update({"to_epoch": to_epoch, "max_recorded_epoch": max_epoch, "epoch_length": epoch_length,
                       "epochs_requested": len(epochs), "plan": plan, "calibration": calib,
                       "integrity": {"plan": plan_errs}})
        if args.mode == "plan":
            report["apply_blockers"] = ["run --mode validate"]
            return 1 if plan_errs else 0

        replay = replay_ledger(rows, rulesets, config, config_fp, challenge_ledger.verify_chain(conn))
        report["integrity"]["ledger"] = replay.pop("integrity_errors")
        calib["era"]["ledger_replay"] = replay.pop("era_findings")
        report["ledger_replay"] = replay
        integrity = plan_errs + report["integrity"]["ledger"]
        git_commit, git_rel, git_problem = _config_git(config_path, config_bytes)
        report["config"].update({"git_commit": git_commit, "git_path": git_rel})
        blockers = _apply_blockers(integrity, calib, git_problem)
        report["apply_blockers"] = blockers
        if args.mode == "validate":
            return 1 if integrity else 0

        # ---- apply
        if blockers:
            raise Refusal("apply blocked: " + "; ".join(blockers))
        log: list = []
        report["apply_log"] = log
        abort = None
        try:
            apply_plan(conn, plan, entries, limit=args.limit, epoch_length=epoch_length,
                       config_bytes=config_bytes, config_path_label=git_rel, git_commit=git_commit,
                       run_id=report["run_id"], tool_sha=tool_sha, log=log)
        except Abort as e:
            abort = str(e)
        except Exception as e:  # noqa: BLE001 - record_challenge refusals etc. are aborts
            abort = f"{type(e).__name__}: {e}"
        finally:
            try:
                chain_err = challenge_ledger.verify_chain(conn)
            except Exception as e:  # noqa: BLE001
                chain_err = f"verify_chain raised {type(e).__name__}: {e}"
            report["post_apply_verify_chain"] = chain_err or "ok"
        rows2, rulesets2 = _load_ledger(conn)
        replay2 = replay_ledger(rows2, rulesets2, config, config_fp, chain_err)
        report["post_apply_replay"] = replay2
        if abort:
            raise Abort(abort)
        if chain_err or replay2["integrity_errors"]:
            raise Abort("post-apply ledger integrity failure (see post_apply_replay)")
        return 0
    finally:
        conn.close()


def _write_report(path: Path, report: dict) -> None:
    def default(o):
        if _is_bytes(o):
            return bytes(o).hex()
        raise TypeError(f"unserialisable {type(o).__name__}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True, default=default) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _print_summary(report: dict) -> None:
    out = [f"mode={report['mode']} run_id={report['run_id']} exit={report.get('exit_code')}"]
    for k in ("refused", "aborted", "crashed"):
        if k in report:
            out.append(f"{k.upper()}: {report[k]}")
    if "plan" in report:
        ents = report["plan"].get("entries") or []
        n_rec = sum(1 for e in ents if isinstance(e, dict) and e.get("action") == "record")
        out.append(f"to_epoch={report['to_epoch']} (max recorded {report['max_recorded_epoch']}) "
                   f"epochs={report['epochs_requested']} plan record={n_rec} skip={len(ents) - n_rec}")
    for scope, errs in (report.get("integrity") or {}).items():
        out.append(f"integrity[{scope}]: {len(errs)} error(s)")
        out.extend(f"  - {e}" for e in errs[:10])
    cal = report.get("calibration")
    if cal:
        ann, era = cal["annotation"], cal["era"]
        out.append(f"calibration/annotation: overlap_agree={ann['overlap_agree']} "
                   f"already_recorded_different={len(ann['already_recorded_different'])} "
                   f"skip_reasons={ann['skip_reasons']}")
        out.append(f"calibration/era: groups={len(era['groups'])} "
                   f"other_ruleset_in_range={len(era['other_ruleset_inside_backfill_range'])} "
                   f"replay_findings={len(era.get('ledger_replay', []))}")
        out.append(f"would_record={len(cal['would_record'])} (with limit: {len(cal['would_record_with_limit'])})")
    if "apply_blockers" in report:
        out.append(f"apply_blockers: {report['apply_blockers'] or 'none'}")
    if "apply_log" in report:
        c = collections.Counter(x["outcome"] for x in report["apply_log"])
        out.append(f"apply: {dict(c)} post_verify_chain={report.get('post_apply_verify_chain')}")
    print("\n".join(out))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=("plan", "validate", "apply"))
    ap.add_argument("--ledger", required=True, type=Path, help="challenge ledger SQLite path (must exist)")
    ap.add_argument("--config", required=True, type=Path, help="ruleset config, e.g. nanobody_pr100.json")
    ap.add_argument("--to-epoch", type=int, default=None, help="default and maximum: max recorded epoch")
    ap.add_argument("--limit", type=int, default=None, help="max NEW rows to record in apply")
    ap.add_argument("--epoch-length", type=int, default=None, help="required only if ledger rows disagree")
    ap.add_argument("--report", type=Path, default=None, help="JSON report path (required for apply)")
    ap.add_argument("--yes", action="store_true", help="confirm apply writes")
    ap.add_argument("--accepted-db", default=ACCEPTED_DB_PATH,
                    help="accepted-submissions SQLite path (default: master path)")
    ap.add_argument("--subtensor-network", default="finney", help="Bittensor network (default finney)")
    ap.add_argument("--subtensor-chain-endpoint", default=None,
                    help="optional wss archive endpoint if the default node is pruned")
    args = ap.parse_args(argv)

    report = {"tool": "backfill_challenges", "mode": args.mode, "run_id": uuid.uuid4().hex,
              "started_at": _utc_now(), "limit": args.limit}
    rc = 1
    try:
        rc = _run(args, report)
    except Refusal as e:
        report["refused"], rc = str(e), 2
    except Abort as e:
        report["aborted"], rc = str(e), 1
    except Exception as e:
        report["crashed"], rc = f"{type(e).__name__}: {e}", 1
        raise
    finally:
        report["exit_code"], report["finished_at"] = rc, _utc_now()
        if args.report:
            _write_report(args.report, report)
        _print_summary(report)
    return rc


if __name__ == "__main__":
    sys.exit(main())
