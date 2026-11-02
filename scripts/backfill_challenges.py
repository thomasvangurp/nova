#!/usr/bin/env python3
"""
backfill_challenges.py -- backfill missing `epoch_challenges` rows into the
hash-chained challenge ledger (challenge_ledger.py), nanobody PR100 track.

Modes
  plan      Read-only. Enumerate missing epochs FIRST_EPOCH..max(epoch), resolve
            the deployed config per epoch, fetch the epoch-start block hash from
            the archive node, re-derive with challenge_ledger.derive_challenge,
            cross-check (H8 accepted entries, H9 issued-challenge logs).
  validate  Read-only. Replay every existing row (challenge bytes / sha / columns,
            rulesets row, hash chain via verify_chain AND chain_hash) and
            re-derive each row from git config to confirm an exact byte match.
            Also measures the era table (H4), the accepted-entry rule (H8) and
            optionally the RPC (H6) against the live rows.
  apply     Calibration (= validate over existing rows, with an RPC sample)
            -> plan -> record_challenge(..., source='backfill') for every
            'ready' epoch in ascending order, with a read-back post-check per
            row and a final chain verification.

Nothing falls back silently: an epoch that cannot be resolved, derived, or
corroborated is skipped with an explicit reason and appears in the report.

Usage (run from the host repo root; challenge_ledger must be importable):
  PYTHONPATH=. python3 backfill_challenges.py --ledger PATH --mode validate
  PYTHONPATH=. python3 backfill_challenges.py --ledger PATH --mode plan --report plan.jsonl
  PYTHONPATH=. python3 backfill_challenges.py --ledger PATH --mode apply --yes --limit 3   # pilot
  PYTHONPATH=. python3 backfill_challenges.py --ledger PATH --mode apply --yes

Exit codes: 0 clean, 1 integrity/derivation error or aborted apply,
            2 calibration findings only (era table / H8 / H9 / RPC disagree).
Stdlib only + the ledger import.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import inspect
import json
import os
import re
import sqlite3
import subprocess
import sys
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from nanobody_opt import challenge_ledger as _ledger
from nanobody_opt.challenge_ledger import (  # REUSED -- do not reimplement any of these here
    canonical_json,
    chain_hash,
    connect,
    derive_challenge,
    fingerprint,
    record_challenge,
    sha256_hex,
    verify_chain,
)


# ════════════════════════════════════════════════════════════════════════════
#  LIVE-STATE PLACEHOLDERS  -- fill / confirm before `--mode apply`
# ════════════════════════════════════════════════════════════════════════════

# ---- H4: deployed era boundaries, in EPOCHS (commit time != deploy time) ----
@dataclass(frozen=True)
class Era:
    name: str
    start_epoch: Optional[int]          # inclusive; None = unbounded below
    end_epoch: Optional[int]            # inclusive; None = open-ended
    config_path: str                    # repo-relative (H1)
    git_commit: Optional[str] = None    # commit actually DEPLOYED for this era
    expected_ruleset_sha256: Optional[str] = None  # optional extra pin


ERAS: tuple[Era, ...] = (
    # Pre-activation. Below FIRST_EPOCH so never backfilled; kept so this
    # path's git history is part of the ruleset search in `validate`.
    Era("pre-pr100", None, 25954, "config/nanobody_pre_pr100.json"),
    # TODO(H4): confirm there is no later deploy boundary. If the deployed
    # config/nanobody_pr100.json ever changed after 25955, split this era at
    # the DEPLOY epoch and pin `git_commit` on every piece. An unpinned era is
    # resolvable only while the path's whole git history has ONE ruleset.
    Era("pr100", 25955, None, "config/nanobody_pr100.json", git_commit=None),
)

# Operator assertion that ERAS matches deploy history. `apply` refuses while False.
H4_ERAS_CONFIRMED = False

# ---- H6: archive-node RPC (Substrate JSON-RPC over http/https) ----
# TODO(H6): NAME of the env var holding the endpoint URL. Never put the URL or
# a key here or on the command line; it is read in-process and never printed.
ARCHIVE_RPC_URL_ENV = "SN68_ARCHIVE_RPC_URL"
RPC_TIMEOUT_S = 30.0

# ---- H8: validator-accepted entries + matching rule ----
# TODO(H8): read-only SQLite DB of accepted sequences, and a query with exactly
# one `?` (= epoch) returning rows (entry_id, v_gene, cdr3_length), where
# v_gene / cdr3_length are the ANNOTATION the validator matched against, keyed
# by the epoch whose challenge the entry was judged under (watch for an
# off-by-one vs submission epoch -- `validate` exposes it on live rows).
ACCEPTED_DB_PATH: Optional[str] = None
ACCEPTED_SQL: Optional[str] = None


def normalize_v_gene(name: Any) -> str:
    """TODO(H8): copy the validator's exact V-gene comparison (allele suffix,
    case, species prefix). Do NOT guess: identity-after-strip until then."""
    return str(name).strip()


# ---- H9: historical issued-challenge logs (W&B / validator logs) ----
# TODO(H9): JSONL extracts, one object per issued challenge:
#   {"epoch": int, "allowed_v": str, "allowed_cdr3_length": int,
#    optional "category_key", "target", "epoch_start_block_hash", "source"}
# Every record for an epoch must agree exactly with the re-derivation.
ISSUED_CHALLENGE_LOGS: tuple[str, ...] = ()


# ---- H7: per-epoch logged ruleset ----
def logged_ruleset_for_epoch(epoch: int) -> Optional[str]:
    """H7: lineage does not store ruleset_sha256 per epoch (recomputed on the
    fly), so there is no source today. If one ever exists, return the logged
    ruleset_sha256 here; it then takes precedence over the era table."""
    return None


# ════════════════════════════════════════════════════════════════════════════
#  HOST FACTS (fixed)
# ════════════════════════════════════════════════════════════════════════════
FIRST_EPOCH = 25955                          # H11 activation epoch
EPOCH_LENGTH = 361                           # H5; start block = epoch * 361
DERIVATION_VERSION_PREFIX = "challenge-v1"   # H12 -> "challenge-v1@<ledger git sha>"
GENESIS_PREV = "0" * 64
SOURCE_BACKFILL = "backfill"
SOURCE_LIVE = "live"

EXPECTED_CHALLENGE_KEYS = frozenset({
    "epoch", "target", "epoch_length", "epoch_start_block",
    "epoch_start_block_hash", "ruleset_sha256", "allowed_v", "allowed_cdr3_length",
})
ROW_COLS = (
    "seq", "epoch", "epoch_length", "epoch_start_block", "epoch_start_block_hash",
    "target", "category_key", "allowed_v", "allowed_cdr3_length", "ruleset_sha256",
    "derivation_version", "challenge_bytes", "challenge_sha256", "source",
    "provenance", "prev_row_sha256", "row_sha256",
)
RULESET_COLS = (
    "ruleset_sha256", "canonical_bytes", "germline_filtering_json",
    "config_path", "config_git_commit", "first_seen_epoch",
)
BLOCK_HASH_RE = re.compile(r"0x[0-9a-f]{64}")
OPTIONAL_RECORD_KWARGS = ("category_key", "config_path", "config_git_commit")


class Abort(SystemExit):
    def __init__(self, msg: str):
        print(f"ABORT: {msg}", file=sys.stderr)
        super().__init__(1)


# ── small helpers ───────────────────────────────────────────────────────────
def _b(x: Any) -> bytes:
    if isinstance(x, bytes):
        return x
    if isinstance(x, (bytearray, memoryview)):
        return bytes(x)
    if isinstance(x, str):
        return x.encode("utf-8")
    raise TypeError(f"expected bytes/str, got {type(x).__name__}")


def cj(obj: Any) -> bytes:
    """Ledger canonical_json, normalised to bytes."""
    return _b(canonical_json(obj))


def _col_eq(col: Any, val: Any) -> bool:
    """Column vs parsed-challenge value (non-scalar targets may be stored as JSON text)."""
    if col == val:
        return True
    if isinstance(col, (str, bytes)):
        try:
            return json.loads(col) == val
        except ValueError:
            return False
    return False


# ── git / config history (H1, H2) ───────────────────────────────────────────
class GitError(RuntimeError):
    pass


def git(repo: Path, *args: str) -> bytes:
    proc = subprocess.run(["git", "-C", str(repo), *args],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        lines = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise GitError(f"git {args[0]}: {lines[-1] if lines else f'exit {proc.returncode}'}")
    return proc.stdout


@dataclass
class ConfigVersion:
    path: str
    commit: str
    commit_time: int
    raw: bytes                       # exact file bytes -> record_challenge(config_bytes)
    config: Optional[dict]
    ruleset_sha256: Optional[str]    # ledger fingerprint(config); None if unusable
    error: Optional[str] = None


class ConfigHistory:
    """Every committed version of each config path (all local refs),
    fingerprinted with the ledger's own fingerprint(). H2: the formula never
    changed, so one fingerprint function is valid across all history."""

    def __init__(self, repo: Path):
        self.repo = repo
        self._versions: dict[str, list[ConfigVersion]] = {}
        self._at: dict[tuple[str, str], ConfigVersion] = {}

    def _load(self, path: str, commit: str, ct: int, raw: bytes) -> ConfigVersion:
        try:
            cfg = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as e:
            return ConfigVersion(path, commit, ct, raw, None, None, f"invalid JSON: {e}")
        if not isinstance(cfg, dict):
            return ConfigVersion(path, commit, ct, raw, None, None, "config is not a JSON object")
        try:
            fp = fingerprint(cfg)
        except (KeyError, TypeError, ValueError) as e:
            return ConfigVersion(path, commit, ct, raw, cfg, None,
                                 f"fingerprint() failed: {type(e).__name__}: {e}")
        return ConfigVersion(path, commit, ct, raw, cfg, fp)

    def versions(self, path: str) -> list[ConfigVersion]:
        """Oldest first. Remote branches count only if fetched."""
        if path not in self._versions:
            out = git(self.repo, "log", "--all", "--full-history", "--format=%H %ct", "--", path)
            seen: set[str] = set()
            vs: list[ConfigVersion] = []
            for line in out.decode().splitlines():
                commit, ct = line.split()
                if commit in seen:
                    continue
                seen.add(commit)
                try:
                    raw = git(self.repo, "show", f"{commit}:{path}")
                except GitError:
                    continue  # path deleted in this commit
                vs.append(self._load(path, commit, int(ct), raw))
            vs.sort(key=lambda v: (v.commit_time, v.commit))
            self._versions[path] = vs
        return self._versions[path]

    def at(self, commit: str, path: str) -> ConfigVersion:
        key = (commit, path)
        if key not in self._at:
            full = git(self.repo, "rev-parse", "--verify", f"{commit}^{{commit}}").decode().strip()
            ct = int(git(self.repo, "show", "-s", "--format=%ct", full).decode().strip())
            raw = git(self.repo, "show", f"{full}:{path}")
            self._at[key] = self._load(path, full, ct, raw)
        return self._at[key]


# ── config resolution per epoch ─────────────────────────────────────────────
@dataclass
class Resolution:
    ok: bool
    reason: str = ""
    era: Optional[Era] = None
    version: Optional[ConfigVersion] = None
    method: str = ""
    details: dict = field(default_factory=dict)


def _lo(e: Era) -> float:
    return float("-inf") if e.start_epoch is None else e.start_epoch


def _hi(e: Era) -> float:
    return float("inf") if e.end_epoch is None else e.end_epoch


def check_eras(eras: tuple[Era, ...]) -> None:
    for e in eras:
        if _lo(e) > _hi(e):
            raise Abort(f"era '{e.name}': start_epoch > end_epoch")
        if e.git_commit is not None and not re.fullmatch(r"[0-9a-f]{7,40}", e.git_commit):
            raise Abort(f"era '{e.name}': git_commit is not a hex sha")
    s = sorted(eras, key=_lo)
    for a, b in zip(s, s[1:]):
        if _hi(a) >= _lo(b):
            raise Abort(f"eras '{a.name}' and '{b.name}' overlap")


class Resolver:
    """Precedence: logged ruleset (H7, none today) > era table (H4).
    Live ledger rows are a CONSISTENCY check on the era table, never a
    resolution source: a live row inside an era with a different ruleset
    proves an undeclared deploy boundary, so the whole era is refused."""

    def __init__(self, history: ConfigHistory, eras: tuple[Era, ...], live_fp: dict[int, str]):
        check_eras(eras)
        self.history = history
        self.eras = eras
        self.live_fp = live_fp
        self._base: dict[str, Resolution] = {}

    def era_for(self, epoch: int) -> Optional[Era]:
        hits = [e for e in self.eras if _lo(e) <= epoch <= _hi(e)]
        return hits[0] if len(hits) == 1 else None

    def config_paths(self) -> list[str]:
        return list(dict.fromkeys(e.config_path for e in self.eras))

    def era_base(self, era: Era) -> Resolution:
        if era.name in self._base:
            return self._base[era.name]
        if era.git_commit:
            try:
                v = self.history.at(era.git_commit, era.config_path)
            except GitError as e:
                r = Resolution(False, f"era '{era.name}' pinned commit unreadable: {e}", era)
            else:
                r = (Resolution(True, era=era, version=v, method="era_pin") if v.ruleset_sha256
                     else Resolution(False, f"era '{era.name}' pinned config unusable: {v.error}", era))
        else:
            valid = [v for v in self.history.versions(era.config_path) if v.ruleset_sha256]
            fps = list(dict.fromkeys(v.ruleset_sha256 for v in valid))
            if not valid:
                r = Resolution(False, f"no commit of {era.config_path} yields a valid fingerprint", era)
            elif len(fps) > 1:
                r = Resolution(False, (f"ambiguous: {len(fps)} distinct rulesets in git history of "
                                       f"{era.config_path} and era '{era.name}' pins no deployed "
                                       f"commit (H4)"), era, details={"rulesets": fps})
            else:
                r = Resolution(True, era=era, version=valid[0], method="era_single_ruleset_in_history")
        if (r.ok and era.expected_ruleset_sha256
                and r.version.ruleset_sha256 != era.expected_ruleset_sha256):
            r = Resolution(False, (f"era '{era.name}' resolves {r.version.ruleset_sha256[:12]} but "
                                   f"expected_ruleset_sha256 is {era.expected_ruleset_sha256[:12]}"), era)
        self._base[era.name] = r
        return r

    def era_fp(self, epoch: int) -> tuple[Optional[str], str]:
        era = self.era_for(epoch)
        if era is None:
            return None, "no (unique) era covers epoch"
        b = self.era_base(era)
        return (b.version.ruleset_sha256, "") if b.ok else (None, b.reason)

    def resolve(self, epoch: int) -> Resolution:
        era = self.era_for(epoch)
        if era is None:
            return Resolution(False, "no (unique) era covers this epoch (H4)")

        logged = logged_ruleset_for_epoch(epoch)
        if logged is not None:
            cands = [v for v in self.history.versions(era.config_path) if v.ruleset_sha256 == logged]
            if not cands:
                return Resolution(False, f"logged ruleset {logged[:12]} not reproducible from git "
                                         f"history of {era.config_path}", era)
            chosen = cands[0]
            if era.git_commit:
                b = self.era_base(era)
                if b.ok and b.version.ruleset_sha256 != logged:
                    return Resolution(False, f"logged ruleset {logged[:12]} disagrees with era "
                                             f"'{era.name}' pin {b.version.ruleset_sha256[:12]}", era)
                if b.ok:
                    chosen = b.version
            return Resolution(True, era=era, version=chosen, method="logged_ruleset",
                              details={"logged_ruleset_sha256": logged})

        b = self.era_base(era)
        if not b.ok:
            return b
        fp = b.version.ruleset_sha256
        in_era = sorted(ep for ep in self.live_fp if _lo(era) <= ep <= _hi(era))
        contra = [ep for ep in in_era if self.live_fp[ep] != fp]
        if contra:
            ep = contra[0]
            return Resolution(False, (f"era table contradicted by {len(contra)} live row(s) in era "
                                      f"'{era.name}' (first: epoch {ep} has ruleset "
                                      f"{self.live_fp[ep][:12]}, era resolves {fp[:12]}); "
                                      f"undeclared deploy boundary (H4)"), era)
        before = [ep for ep in in_era if ep < epoch]
        after = [ep for ep in in_era if ep > epoch]
        return Resolution(True, era=era, version=b.version, method=b.method, details={
            "live_anchor_before": before[-1] if before else None,
            "live_anchor_after": after[0] if after else None,
        })


# ── archive RPC (H6) ────────────────────────────────────────────────────────
class RPCError(RuntimeError):
    pass


class ArchiveRPC:
    """Substrate JSON-RPC over http(s). The endpoint URL (which may embed a
    key) is read from the environment in-process and never printed; errors
    carry only method name, HTTP status or exception type."""

    def __init__(self, env_name: str):
        url = os.environ.get(env_name)
        if not url:
            raise Abort(f"archive RPC env var {env_name} is not set (H6)")
        if not url.startswith(("http://", "https://")):
            raise Abort(f"{env_name} must be an http(s) JSON-RPC endpoint (ws:// unsupported)")
        self._url = url
        self._finalized: Optional[int] = None

    def _call(self, method: str, params: list) -> Any:
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
        req = urllib.request.Request(self._url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=RPC_TIMEOUT_S) as resp:
                payload = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise RPCError(f"{method}: HTTP {e.code}") from None
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise RPCError(f"{method}: transport/decode error ({type(e).__name__})") from None
        if not isinstance(payload, dict):
            raise RPCError(f"{method}: malformed response")
        if payload.get("error") is not None:
            err = payload["error"]
            raise RPCError(f"{method}: rpc error code={err.get('code') if isinstance(err, dict) else '?'}")
        return payload.get("result")

    def finalized_number(self) -> int:
        if self._finalized is None:
            head = self._call("chain_getFinalizedHead", [])
            hdr = self._call("chain_getHeader", [head])
            try:
                self._finalized = int(hdr["number"], 16)
            except (TypeError, KeyError, ValueError):
                raise RPCError("chain_getHeader: unparseable block number") from None
        return self._finalized

    def block_hash(self, number: int) -> str:
        h = self._call("chain_getBlockHash", [number])
        if h is None:
            raise RPCError(f"chain_getBlockHash({number}) returned null (pruned / not an archive node?)")
        if not isinstance(h, str) or not BLOCK_HASH_RE.fullmatch(h):
            raise RPCError(f"chain_getBlockHash({number}) returned a malformed hash")
        return h


# ── corroboration sources (H8, H9, optional host oracle) ────────────────────
@dataclass(frozen=True)
class AcceptedEntry:
    entry_id: str
    v_gene: str
    cdr3_length: int


class AcceptedEntries:
    def __init__(self, db_path: Optional[str], sql: Optional[str]):
        self.configured = bool(db_path and sql)
        self._sql = sql
        self._con: Optional[sqlite3.Connection] = None
        if self.configured:
            p = Path(db_path).expanduser().resolve()
            if not p.exists():
                raise Abort("ACCEPTED_DB_PATH does not exist (H8)")
            self._con = sqlite3.connect(p.as_uri() + "?mode=ro", uri=True)

    def for_epoch(self, epoch: int) -> Optional[list[AcceptedEntry]]:
        if not self.configured:
            return None
        rows = self._con.execute(self._sql, (epoch,)).fetchall()
        return [AcceptedEntry(str(r[0]), str(r[1]), int(r[2])) for r in rows]


def load_issued_logs(paths: tuple[str, ...]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rec["_src"] = f"{Path(p).name}:{n}"
                out.setdefault(int(rec["epoch"]), []).append(rec)
    return out


class HostOracle:
    """Optional independent check against the host's rules.challenge (H3)."""

    def __init__(self, module_name: Optional[str]):
        self.fn = importlib.import_module(module_name).challenge if module_name else None

    def check(self, cfg: dict, epoch: int, block_hash: str, challenge: dict) -> Optional[str]:
        if self.fn is None:
            return None
        res = self.fn(copy.deepcopy(cfg), epoch, block_hash, EPOCH_LENGTH)
        if isinstance(res, tuple):
            res = res[0]
        if not isinstance(res, dict):
            return "host rules.challenge returned a non-dict"
        for k in ("allowed_v", "allowed_cdr3_length"):
            if k not in res:
                return f"host rules.challenge result lacks '{k}'"
        for k in ("allowed_v", "allowed_cdr3_length", "target"):
            if k in res and res[k] != challenge[k]:
                return f"host rules.challenge disagrees on {k}"
        return None


@dataclass
class Ctx:
    repo: Path
    history: ConfigHistory
    resolver: Resolver
    accepted: AcceptedEntries
    issued: dict[int, list[dict]]
    host: HostOracle
    rpc: Optional[ArchiveRPC] = None


def corroborate(ctx: Ctx, epoch: int, challenge: dict, category_key: str) -> tuple[Optional[str], dict]:
    """(failure reason or None, summary). The host oracle checks derivation,
    not config/era choice, so it never counts as corroboration."""
    summary: dict[str, Any] = {}
    corroborated = False
    observed = dict(challenge, category_key=category_key)

    recs = ctx.issued.get(epoch, [])
    for r in recs:
        for k in ("allowed_v", "allowed_cdr3_length", "category_key", "target", "epoch_start_block_hash"):
            if k in r and r[k] != observed.get(k):
                return f"issued-challenge log {r['_src']} disagrees on {k} (H9)", summary
    summary["issued_log_records"] = len(recs)
    corroborated |= bool(recs)

    entries = ctx.accepted.for_epoch(epoch)
    if entries is None:
        summary["accepted_entries"] = "not_configured"
    else:
        want_v = normalize_v_gene(challenge["allowed_v"])
        want_l = int(challenge["allowed_cdr3_length"])
        bad = [e for e in entries if normalize_v_gene(e.v_gene) != want_v or e.cdr3_length != want_l]
        if bad:
            ids = ",".join(e.entry_id for e in bad[:3])
            return (f"{len(bad)}/{len(entries)} accepted entries violate the derived challenge "
                    f"(first: {ids}) (H8)"), summary
        summary["accepted_entries"] = len(entries)
        corroborated |= bool(entries)

    summary["corroborated"] = corroborated
    return None, summary


# ── derivation (reuses ledger derive_challenge) ─────────────────────────────
class DerivationMismatch(RuntimeError):
    pass


def derive_checked(v: ConfigVersion, epoch: int, block_hash: str) -> tuple[dict, str, bytes]:
    """derive_challenge + structural sanity checks against the host facts."""
    try:
        challenge, category_key = derive_challenge(copy.deepcopy(v.config), epoch, block_hash, EPOCH_LENGTH)
        problems = []
        keys = set(challenge)
        if keys != EXPECTED_CHALLENGE_KEYS:
            problems.append(f"challenge keys differ: {sorted(keys ^ EXPECTED_CHALLENGE_KEYS)}")
        expect = {
            "epoch": epoch, "epoch_length": EPOCH_LENGTH,
            "epoch_start_block": epoch * EPOCH_LENGTH, "epoch_start_block_hash": block_hash,
            "ruleset_sha256": v.ruleset_sha256, "target": v.config["protein_selection"]["nanobody"],
        }
        problems += [f"{k} != expected" for k, val in expect.items() if challenge.get(k) != val]
        gf = v.config["germline_filtering"]
        cats = gf["v_neighborhood_categories"]
        if category_key not in cats:
            problems.append(f"category_key {category_key!r} not in v_neighborhood_categories")
        elif challenge.get("allowed_v") not in cats[category_key]:
            problems.append("allowed_v not in its category")
        lo, hi = gf["cdr3_length_range"]["min"], gf["cdr3_length_range"]["max"]
        cl = challenge.get("allowed_cdr3_length")
        if not (isinstance(cl, int) and lo <= cl <= hi):
            problems.append("allowed_cdr3_length outside cdr3_length_range")
    except (KeyError, TypeError, ValueError, IndexError) as e:
        raise DerivationMismatch(f"{type(e).__name__}: {e}") from None
    if problems:
        raise DerivationMismatch("; ".join(problems))
    return challenge, category_key, cj(challenge)


# ── ledger reads ────────────────────────────────────────────────────────────
def check_schema(conn: sqlite3.Connection) -> None:
    for table, cols in (("epoch_challenges", ROW_COLS), ("rulesets", RULESET_COLS)):
        have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        missing = [c for c in cols if c not in have]
        if missing:
            raise Abort(f"ledger table {table} lacks columns {missing} (H10 drift)")


def read_rows(conn: sqlite3.Connection) -> list[dict]:
    cur = conn.execute(f"SELECT {', '.join(ROW_COLS)} FROM epoch_challenges ORDER BY seq")
    return [dict(zip(ROW_COLS, r)) for r in cur]


def read_rulesets(conn: sqlite3.Connection) -> dict[str, dict]:
    cur = conn.execute(f"SELECT {', '.join(RULESET_COLS)} FROM rulesets")
    return {r[0]: dict(zip(RULESET_COLS, r)) for r in cur}


def fetch_row(conn: sqlite3.Connection, epoch: int) -> Optional[dict]:
    r = conn.execute(f"SELECT {', '.join(ROW_COLS)} FROM epoch_challenges WHERE epoch = ?",
                     (epoch,)).fetchone()
    return dict(zip(ROW_COLS, r)) if r else None


def predecessor_hash(conn: sqlite3.Connection, seq: int) -> str:
    r = conn.execute("SELECT row_sha256 FROM epoch_challenges WHERE seq < ? ORDER BY seq DESC LIMIT 1",
                     (seq,)).fetchone()
    return r[0] if r else GENESIS_PREV


def run_verify_chain(conn: sqlite3.Connection) -> list[str]:
    """Interpret verify_chain's result conservatively (confirm its contract)."""
    try:
        res = verify_chain(conn)
    except Exception as e:  # noqa: BLE001 -- any raise is a failed verification
        return [f"verify_chain raised {type(e).__name__}: {e}"]
    if res is None or res is True:
        return []
    if res is False:
        return ["verify_chain returned False"]
    if isinstance(res, dict) and "ok" in res:
        return [] if res["ok"] else [f"verify_chain: {res}"]
    if isinstance(res, (list, tuple, set)):
        return [f"verify_chain: {x}" for x in res]
    return [f"verify_chain returned unrecognised {type(res).__name__}; confirm its contract"]


def replay_chain(rows: list[dict]) -> list[str]:
    """Independent replay with the ledger's chain_hash, in insertion (seq) order."""
    errs, prev, last_seq = [], GENESIS_PREV, None
    for r in rows:
        s = r["seq"]
        if last_seq is not None and s <= last_seq:
            errs.append(f"seq {s}: not strictly increasing")
        if r["prev_row_sha256"] != prev:
            errs.append(f"seq {s} (epoch {r['epoch']}): prev_row_sha256 does not link to previous row")
        exp = chain_hash(prev, s, r["epoch"], r["challenge_sha256"], r["ruleset_sha256"], r["source"])
        if exp != r["row_sha256"]:
            errs.append(f"seq {s} (epoch {r['epoch']}): row_sha256 != chain_hash(...)")
        prev, last_seq = r["row_sha256"], s
    return errs


def version_for_ruleset(ctx: Ctx, fp: str, rs: Optional[dict]) -> Optional[ConfigVersion]:
    if rs and rs.get("config_git_commit") and rs.get("config_path"):
        try:
            v = ctx.history.at(rs["config_git_commit"], rs["config_path"])
            if v.ruleset_sha256 == fp:
                return v
        except GitError:
            pass
    for path in ctx.resolver.config_paths():
        for v in ctx.history.versions(path):
            if v.ruleset_sha256 == fp:
                return v
    return None


# ── validate ────────────────────────────────────────────────────────────────
@dataclass
class Findings:
    errors: list[dict] = field(default_factory=list)       # integrity / derivation
    calibration: list[dict] = field(default_factory=list)  # era table / H8 / H9 / RPC
    info: dict = field(default_factory=dict)

    def err(self, epoch, msg):
        self.errors.append({"epoch": epoch, "kind": "integrity", "msg": msg})

    def cal(self, epoch, msg):
        self.calibration.append({"epoch": epoch, "kind": "calibration", "msg": msg})


def rpc_sample(rows: list[dict], how: str, n: int) -> set[int]:
    if how == "none" or not rows:
        return set()
    if how == "all" or n >= len(rows):
        return {r["seq"] for r in rows}
    step = (len(rows) - 1) / max(n - 1, 1)
    return {rows[round(i * step)]["seq"] for i in range(n)}


def validate(ctx: Ctx, conn: sqlite3.Connection, *, rpc_check: str, rpc_samples: int) -> Findings:
    f = Findings()
    rows = read_rows(conn)
    rulesets = read_rulesets(conn)
    f.info["rows"] = len(rows)
    f.info["by_source"] = {s: sum(r["source"] == s for r in rows) for s in sorted({r["source"] for r in rows})}
    f.info["derivation_versions"] = sorted({r["derivation_version"] for r in rows})

    for m in run_verify_chain(conn):
        f.err(None, m)
    for m in replay_chain(rows):
        f.err(None, m)

    if rpc_check != "none" and ctx.rpc is None:
        raise Abort("--rpc-check needs the archive RPC (H6)")
    sample = rpc_sample(rows, rpc_check, rpc_samples)

    for r in rows:
        ep = r["epoch"]
        if ep < FIRST_EPOCH:
            f.cal(ep, f"row predates FIRST_EPOCH {FIRST_EPOCH}")
        if r["source"] not in (SOURCE_LIVE, SOURCE_BACKFILL):
            f.err(ep, f"unknown source {r['source']!r}")
        cb = _b(r["challenge_bytes"])
        if sha256_hex(cb) != r["challenge_sha256"]:
            f.err(ep, "sha256(challenge_bytes) != challenge_sha256")
        try:
            parsed = json.loads(cb)
        except ValueError:
            f.err(ep, "challenge_bytes is not JSON")
            continue
        if cj(parsed) != cb:
            f.err(ep, "challenge_bytes is not canonical_json")
        for col in ("epoch", "epoch_length", "epoch_start_block", "epoch_start_block_hash",
                    "target", "allowed_v", "allowed_cdr3_length", "ruleset_sha256"):
            if not _col_eq(r[col], parsed.get(col)):
                f.err(ep, f"column {col} disagrees with challenge_bytes")
        if r["epoch_length"] != EPOCH_LENGTH or r["epoch_start_block"] != ep * EPOCH_LENGTH:
            f.err(ep, "epoch_length/epoch_start_block disagree with H5 (361, epoch*361)")
        rs = rulesets.get(r["ruleset_sha256"])
        if rs is None:
            f.err(ep, "ruleset_sha256 has no rulesets row")
        elif sha256_hex(_b(rs["canonical_bytes"])) != r["ruleset_sha256"]:
            f.err(ep, "rulesets.canonical_bytes does not hash to ruleset_sha256")

        # Re-derive from git config; require exact bytes.
        v = version_for_ruleset(ctx, r["ruleset_sha256"], rs)
        if v is None:
            f.err(ep, f"ruleset {r['ruleset_sha256'][:12]} not reproducible from git history of "
                      f"{ctx.resolver.config_paths()}")
        else:
            try:
                ch, ck, b2 = derive_checked(v, ep, r["epoch_start_block_hash"])
            except DerivationMismatch as e:
                f.err(ep, f"re-derivation failed sanity check: {e}")
            else:
                if b2 != cb:
                    f.err(ep, "re-derived challenge bytes differ from stored challenge_bytes")
                if ck != r["category_key"]:
                    f.err(ep, f"re-derived category_key {ck!r} != stored {r['category_key']!r}")
                herr = ctx.host.check(v.config, ep, r["epoch_start_block_hash"], ch)
                if herr:
                    f.err(ep, herr)

        # H4: does the era table resolve the ruleset the row records?
        efp, why = ctx.resolver.era_fp(ep)
        if efp is None:
            f.cal(ep, f"era table cannot resolve: {why}")
        elif efp != r["ruleset_sha256"]:
            f.cal(ep, f"era table resolves {efp[:12]} but row records {r['ruleset_sha256'][:12]} (H4)")

        # H6: stored hash vs archive node.
        if r["seq"] in sample:
            try:
                if ctx.rpc.block_hash(r["epoch_start_block"]) != r["epoch_start_block_hash"]:
                    f.cal(ep, "archive RPC block hash != stored epoch_start_block_hash (H5/H6)")
            except RPCError as e:
                f.cal(ep, f"RPC: {e}")

        # H8 / H9 against this row.
        fail, _ = corroborate(ctx, ep, parsed, r["category_key"])
        if fail:
            f.cal(ep, fail)

    # Evidence for H4: contiguous live ruleset runs + the commits that produce them.
    runs: list[dict] = []
    for r in sorted((r for r in rows if r["source"] == SOURCE_LIVE), key=lambda r: r["epoch"]):
        if runs and runs[-1]["ruleset_sha256"] == r["ruleset_sha256"]:
            runs[-1]["last_epoch"] = r["epoch"]
            runs[-1]["rows"] += 1
        else:
            runs.append({"ruleset_sha256": r["ruleset_sha256"], "first_epoch": r["epoch"],
                         "last_epoch": r["epoch"], "rows": 1})
    for run in runs:
        run["commits"] = [f"{v.path}@{v.commit[:12]}" for p in ctx.resolver.config_paths()
                          for v in ctx.history.versions(p) if v.ruleset_sha256 == run["ruleset_sha256"]]
    f.info["live_ruleset_runs"] = runs
    return f


# ── plan ────────────────────────────────────────────────────────────────────
@dataclass
class EpochPlan:
    epoch: int
    status: str = "skip"   # ready | uncorroborated | skip | resolved_offline | applied
    reason: str = ""
    era: Optional[str] = None
    method: Optional[str] = None
    config_path: Optional[str] = None
    config_git_commit: Optional[str] = None
    ruleset_sha256: Optional[str] = None
    block: Optional[int] = None
    block_hash: Optional[str] = None
    category_key: Optional[str] = None
    challenge: Optional[dict] = None
    challenge_bytes: Optional[bytes] = None
    corroboration: dict = field(default_factory=dict)
    details: dict = field(default_factory=dict)
    version: Optional[ConfigVersion] = None

    def report(self) -> dict:
        return {
            "epoch": self.epoch, "status": self.status, "reason": self.reason,
            "era": self.era, "method": self.method, "config_path": self.config_path,
            "config_git_commit": self.config_git_commit, "ruleset_sha256": self.ruleset_sha256,
            "block": self.block, "block_hash": self.block_hash, "category_key": self.category_key,
            "allowed_v": (self.challenge or {}).get("allowed_v"),
            "allowed_cdr3_length": (self.challenge or {}).get("allowed_cdr3_length"),
            "challenge_sha256": sha256_hex(self.challenge_bytes) if self.challenge_bytes else None,
            "corroboration": self.corroboration, "details": self.details,
        }


def missing_epochs(conn: sqlite3.Connection, to_epoch: Optional[int]) -> tuple[list[int], int]:
    mx = conn.execute("SELECT MAX(epoch) FROM epoch_challenges").fetchone()[0]
    if mx is None:
        raise Abort("ledger has no rows: no upper anchor and no live rows to calibrate against")
    hi = mx if to_epoch is None else to_epoch
    if hi > mx:
        raise Abort(f"--to-epoch {hi} > max recorded epoch {mx}; epochs beyond max belong to the live writer")
    if hi < FIRST_EPOCH:
        return [], mx
    have = {r[0] for r in conn.execute(
        "SELECT epoch FROM epoch_challenges WHERE epoch BETWEEN ? AND ?", (FIRST_EPOCH, hi))}
    return [e for e in range(FIRST_EPOCH, hi + 1) if e not in have], mx


def build_plan(ctx: Ctx, epochs: list[int], *, offline: bool) -> list[EpochPlan]:
    plans = []
    finalized = None if offline or not epochs else ctx.rpc.finalized_number()
    for ep in epochs:
        p = EpochPlan(ep, block=ep * EPOCH_LENGTH)
        plans.append(p)
        res = ctx.resolver.resolve(ep)
        if res.era:
            p.era = res.era.name
        if not res.ok:
            p.reason = res.reason
            continue
        v = res.version
        p.method, p.details, p.version = res.method, res.details, v
        p.config_path, p.config_git_commit, p.ruleset_sha256 = v.path, v.commit, v.ruleset_sha256
        if offline:
            p.status = "resolved_offline"
            continue
        if p.block > finalized:
            p.reason = f"block {p.block} not finalized (finalized head {finalized})"
            continue
        try:
            p.block_hash = ctx.rpc.block_hash(p.block)
        except RPCError as e:
            p.reason = f"RPC: {e}"
            continue
        try:
            ch, ck, cb = derive_checked(v, ep, p.block_hash)
        except DerivationMismatch as e:
            p.reason = f"derivation sanity check failed: {e}"
            continue
        p.challenge, p.category_key, p.challenge_bytes = ch, ck, cb
        herr = ctx.host.check(v.config, ep, p.block_hash, ch)
        if herr:
            p.reason = herr
            continue
        fail, summary = corroborate(ctx, ep, ch, ck)
        p.corroboration = summary
        if fail:
            p.reason = fail
            continue
        if summary["corroborated"]:
            p.status = "ready"
        else:
            p.status, p.reason = "uncorroborated", "no H8/H9 evidence for this epoch"
    return plans


# ── apply ───────────────────────────────────────────────────────────────────
def ledger_derivation_version(require_clean: bool) -> str:
    """H12: challenge-v1@<sha of last commit touching challenge_ledger.py>,
    and only if the imported file equals its committed bytes."""
    p = Path(_ledger.__file__).resolve()
    try:
        top = Path(git(p.parent, "rev-parse", "--show-toplevel").decode().strip())
        rel = p.relative_to(top).as_posix()
        dirty = git(top, "status", "--porcelain", "--", rel).strip()
        sha = git(top, "log", "-1", "--format=%H", "--", rel).decode().strip()
    except (GitError, ValueError) as e:
        raise Abort(f"cannot determine git sha of challenge_ledger.py: {e}")
    if not sha:
        raise Abort("challenge_ledger.py is not committed")
    if dirty and require_clean:
        raise Abort("challenge_ledger.py has uncommitted changes; derivation_version would lie")
    return f"{DERIVATION_VERSION_PREFIX}@{sha}"


def record_kwargs_supported() -> set[str]:
    params = inspect.signature(record_challenge).parameters
    provided = {"conn", *OPTIONAL_RECORD_KWARGS}
    unknown_required = [n for n, pr in params.items()
                        if pr.kind is inspect.Parameter.KEYWORD_ONLY
                        and pr.default is inspect.Parameter.empty and n not in provided]
    if unknown_required:
        raise Abort(f"record_challenge requires unknown keyword(s) {unknown_required}; extend this script")
    return {n for n in OPTIONAL_RECORD_KWARGS if n in params}


def post_check(conn: sqlite3.Connection, row: Optional[dict], p: EpochPlan, dv: str) -> list[str]:
    if row is None:
        return ["row not found after record_challenge"]
    prev = predecessor_hash(conn, row["seq"])
    expect = {
        "source": SOURCE_BACKFILL, "category_key": p.category_key,
        "allowed_v": p.challenge["allowed_v"], "allowed_cdr3_length": p.challenge["allowed_cdr3_length"],
        "ruleset_sha256": p.ruleset_sha256, "challenge_sha256": sha256_hex(p.challenge_bytes),
        "derivation_version": dv, "epoch_start_block_hash": p.block_hash,
        "epoch_start_block": p.block, "epoch_length": EPOCH_LENGTH, "prev_row_sha256": prev,
    }
    probs = [f"{k}: stored {row[k]!r} != expected {v!r}" for k, v in expect.items() if row[k] != v]
    if _b(row["challenge_bytes"]) != p.challenge_bytes:
        probs.append("challenge_bytes differ from derived canonical bytes")
    if chain_hash(prev, row["seq"], row["epoch"], row["challenge_sha256"],
                  row["ruleset_sha256"], row["source"]) != row["row_sha256"]:
        probs.append("row_sha256 != chain_hash(...)")
    return probs


def apply(ctx: Ctx, conn: sqlite3.Connection, plans: list[EpochPlan], *,
          allow_uncorroborated: bool, limit: Optional[int], run_id: str, tool_sha: str) -> None:
    dv = ledger_derivation_version(require_clean=True)
    supported = record_kwargs_supported()
    eligible = {"ready"} | ({"uncorroborated"} if allow_uncorroborated else set())
    todo = [p for p in plans if p.status in eligible]
    if limit is not None:
        todo = todo[:limit]
    print(f"apply: {len(todo)} epoch(s), derivation_version={dv}, "
          f"record_challenge kwargs passed: {sorted(supported)}")
    for p in todo:
        if fetch_row(conn, p.epoch) is not None:
            p.status, p.reason = "skip", "epoch appeared in ledger after planning"
            continue
        prov = {
            "tool": "backfill_challenges.py", "tool_sha256": tool_sha, "run_id": run_id,
            "resolution": p.method, "era": p.era, "config_path": p.config_path,
            "config_git_commit": p.config_git_commit, "anchors": p.details,
            "block_hash_source": "archive_rpc:chain_getBlockHash",
            "rpc_endpoint_env": ARCHIVE_RPC_URL_ENV,  # env var NAME only
            "corroboration": p.corroboration,
            "uncorroborated_override": p.status == "uncorroborated",
            "host_oracle": ctx.host.fn is not None,
        }
        opt = {"category_key": p.category_key, "config_path": p.config_path,
               "config_git_commit": p.config_git_commit}
        kwargs = {k: v for k, v in opt.items() if k in supported}
        try:
            record_challenge(p.version.raw, p.challenge, dv, SOURCE_BACKFILL,
                             cj(prov).decode("ascii"), conn=conn, **kwargs)
        except sqlite3.IntegrityError as e:
            if conn.in_transaction:
                conn.rollback()
            if fetch_row(conn, p.epoch) is not None:
                p.status, p.reason = "skip", "epoch inserted concurrently"
                continue
            raise Abort(f"epoch {p.epoch}: record_challenge IntegrityError: {e}")
        probs = post_check(conn, fetch_row(conn, p.epoch), p, dv)
        if probs:
            if conn.in_transaction:
                conn.rollback()
                raise Abort(f"epoch {p.epoch}: post-check failed, rolled back: {probs}")
            raise Abort(f"epoch {p.epoch}: post-check FAILED ON A COMMITTED ROW (append-only; "
                        f"do not retry, record a conflict): {probs}")
        if conn.in_transaction:
            conn.commit()
        p.status, p.reason = "applied", ""
        print(f"  applied epoch {p.epoch} ({p.category_key}, {p.challenge['allowed_v']}, "
              f"cdr3={p.challenge['allowed_cdr3_length']})")


# ── output ──────────────────────────────────────────────────────────────────
def write_report(path: Optional[Path], records: list[dict]) -> None:
    if path is None:
        return
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
    print(f"report: {path} ({len(records)} records)")


def summarize_plan(plans: list[EpochPlan]) -> None:
    counts: dict[str, int] = {}
    reasons: dict[str, list[int]] = {}
    for p in plans:
        counts[p.status] = counts.get(p.status, 0) + 1
        if p.status not in ("ready", "applied") and p.reason:
            reasons.setdefault(p.reason, []).append(p.epoch)
    print("plan:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "nothing missing")
    for why, eps in sorted(reasons.items(), key=lambda kv: -len(kv[1]))[:20]:
        print(f"  [{len(eps)} epochs {eps[0]}..{eps[-1]}] {why}")


def summarize_findings(f: Findings) -> int:
    print(f"validate: rows={f.info['rows']} by_source={f.info['by_source']}")
    print(f"  derivation_versions: {f.info['derivation_versions']}")
    for run in f.info["live_ruleset_runs"]:
        print(f"  live ruleset {run['ruleset_sha256'][:12]}: epochs {run['first_epoch']}..{run['last_epoch']} "
              f"({run['rows']} rows) <- {run['commits'] or 'NO COMMIT IN HISTORY'}")
    for label, items in (("INTEGRITY", f.errors), ("CALIBRATION", f.calibration)):
        print(f"  {label}: {len(items)}")
        for it in items[:25]:
            print(f"    epoch {it['epoch']}: {it['msg']}")
        if len(items) > 25:
            print(f"    ... {len(items) - 25} more (see --report)")
    return 1 if f.errors else (2 if f.calibration else 0)


# ── main ────────────────────────────────────────────────────────────────────
def parse_args(argv: Optional[list[str]]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Backfill missing epoch_challenges rows.")
    ap.add_argument("--mode", choices=("plan", "validate", "apply"), required=True)
    ap.add_argument("--ledger", type=Path, required=True, help="ledger SQLite path (must exist)")
    ap.add_argument("--repo", type=Path, default=Path.cwd(), help="host repo root (config paths are relative)")
    ap.add_argument("--to-epoch", type=int, help="upper bound (<= max recorded epoch); default max")
    ap.add_argument("--limit", type=int, help="apply: at most N epochs (pilot)")
    ap.add_argument("--report", type=Path, help="JSONL report path")
    ap.add_argument("--offline", action="store_true", help="plan: resolve configs only, no RPC/derivation")
    ap.add_argument("--rpc-check", choices=("none", "sample", "all"), default="none",
                    help="validate: compare stored block hashes with the archive node")
    ap.add_argument("--rpc-samples", type=int, default=5)
    ap.add_argument("--allow-uncorroborated", action="store_true",
                    help="apply: also insert epochs with no H8/H9 evidence (flagged in provenance)")
    ap.add_argument("--host-rules-module", help="optional module exposing host challenge() as an oracle (H3)")
    ap.add_argument("--yes", action="store_true", help="required for --mode apply")
    a = ap.parse_args(argv)
    if a.mode == "apply" and not a.yes:
        ap.error("--mode apply writes append-only rows; pass --yes")
    if a.mode != "plan" and a.offline:
        ap.error("--offline is plan-only")
    return a


def main(argv: Optional[list[str]] = None) -> int:
    a = parse_args(argv)
    if not a.ledger.exists():
        raise Abort(f"ledger not found: {a.ledger}")  # connect() would create an empty one
    if a.mode == "apply" and not H4_ERAS_CONFIRMED:
        raise Abort("H4_ERAS_CONFIRMED is False: confirm ERAS against deploy history first")

    conn = connect(str(a.ledger))
    check_schema(conn)
    repo = a.repo.resolve()
    rows = read_rows(conn)
    live_fp = {r["epoch"]: r["ruleset_sha256"] for r in rows if r["source"] == SOURCE_LIVE}
    history = ConfigHistory(repo)
    ctx = Ctx(repo=repo, history=history, resolver=Resolver(history, ERAS, live_fp),
              accepted=AcceptedEntries(ACCEPTED_DB_PATH, ACCEPTED_SQL),
              issued=load_issued_logs(ISSUED_CHALLENGE_LOGS),
              host=HostOracle(a.host_rules_module))
    if (a.mode == "apply" or (a.mode == "plan" and not a.offline)
            or (a.mode == "validate" and a.rpc_check != "none")):
        ctx.rpc = ArchiveRPC(ARCHIVE_RPC_URL_ENV)

    if a.mode == "validate":
        f = validate(ctx, conn, rpc_check=a.rpc_check, rpc_samples=a.rpc_samples)
        write_report(a.report, f.errors + f.calibration + [{"kind": "info", **f.info}])
        return summarize_findings(f)

    if a.mode == "apply":
        print("calibration: validating existing rows (RPC sample) ...")
        f = validate(ctx, conn, rpc_check="sample", rpc_samples=a.rpc_samples)
        if summarize_findings(f) != 0:
            raise Abort("calibration failed; fix ledger / H4 / H6 / H8 / H9 before applying")

    epochs, mx = missing_epochs(conn, a.to_epoch)
    print(f"missing epochs in [{FIRST_EPOCH}, {a.to_epoch or mx}]: {len(epochs)} (max recorded {mx})")
    plans = build_plan(ctx, epochs, offline=a.offline)
    summarize_plan(plans)

    if a.mode == "apply":
        run_id = uuid.uuid4().hex
        tool_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        apply(ctx, conn, plans, allow_uncorroborated=a.allow_uncorroborated,
              limit=a.limit, run_id=run_id, tool_sha=tool_sha)
        chain_errs = run_verify_chain(conn) + replay_chain(read_rows(conn))
        write_report(a.report, [p.report() for p in plans])
        summarize_plan(plans)
        if chain_errs:
            print("POST-APPLY CHAIN ERRORS:", *chain_errs, sep="\n  ", file=sys.stderr)
            return 1
        print(f"chain verified after apply (run_id={run_id})")
        return 0

    write_report(a.report, [p.report() for p in plans])
    return 0


if __name__ == "__main__":
    sys.exit(main())