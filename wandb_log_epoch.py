#!/usr/bin/env python3
"""Standalone, READ-ONLY W&B logger for SN68 nanobody mining troubleshooting.

Logs one epoch's troubleshooting metrics to Weights & Biases (CoreWeave Forge,
project private by default). It reads three sources and writes nothing back:

  1. Leaderboard API (public, no auth):
     https://compound-api.metanova-labs.ai/api/competitions/leaderboard/{epoch}/nanobodies?limit=10000
  2. MySQL (Hetzner) via pymysql -- same connection semantics as
     nanobody_opt.db.connect(): creds loaded from /root/.sn68_mysql_creds
     (0600), overridable via SN68_MYSQL_HOST/PORT/USER/PASSWORD/DB env vars.
     (On the slave, set SN68_MYSQL_HOST=127.0.0.1 SN68_MYSQL_PORT=3306 for the
     tailscale relay; on prod the default direct host works.)
  3. Producer artifacts: /root/live/epoch{N}/producer/iteration_*/top50_index.json
     (fields: source = 'hf_archive'|'our_top50', live_sequences = parent pool).

It does NOT touch the live producer's scoring/submission code and performs only
a few aggregate SELECTs (no full-table scans).

Idempotency: one W&B run per epoch with a deterministic run id
(f"sn68-epoch-{epoch}") and resume="allow", plus a state file
/root/wandb_log_state/{epoch}.done. Re-running the same epoch resumes the same
run (never a duplicate run) or is skipped unless --force.

Config is read from the environment; the key is NEVER hardcoded or printed:
  WANDB_API_KEY   - required for real logging (from CoreWeave Forge settings)
  WANDB_BASE_URL  - OPTIONAL. For CoreWeave Forge multi-tenant, LEAVE UNSET:
                    the API is still https://api.wandb.ai. Do NOT set this to
                    https://forge.coreweave.com/wandb -- that is the web-app
                    URL; the /wandb prefix is app-only, not the SDK API base.
  WANDB_ENTITY    - OPTIONAL; entity/team name. Falls back to the key owner's
                    default entity via wandb.Api().default_entity.
  WANDB_PROJECT   - project name (default: sn68-nanobody).

Privacy: a project created here is private by default (only the entity/team
members can see it). W&B exposes no public SDK field to change visibility, so
the script best-effort upserts the project and reports its visibility if the
SDK exposes it; to restrict further (Team/Restricted) use the Forge UI.

MISSING INPUT (blocker): a CoreWeave W&B API key is NOT available in this
environment. To go live, put it on the prod box:

  # /root/.wandb_env  (chmod 600 -- do NOT commit / do NOT print)
  export WANDB_API_KEY="<key from forge.coreweave.com/settings -> API keys>"
  export WANDB_ENTITY="<your username or team name>"   # optional

The wrapper /root/wandb_log_loop.sh sources that file before running this
script, so cron never sees the key in its own environment.

Usage:
  python wandb_log_epoch.py --dry-run                 # print payload, no key needed
  python wandb_log_epoch.py --dry-run --epoch 25572   # pin an epoch
  python wandb_log_epoch.py                           # real upload (needs WANDB_API_KEY)
  python wandb_log_epoch.py --force                   # re-log an already-logged epoch
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import statistics
import sys
import urllib.error
import urllib.request

LEADERBOARD_URL = (
    "https://compound-api.metanova-labs.ai/api/competitions/leaderboard/"
    "{epoch}/nanobodies?limit=10000"
)
OUR_PREFIX = "5EkCwcNTfxeNbbhNSEJMSaGbW2XMVuhzouZkEan5ozvSsJdQ"
RIVAL_PREFIX = "5GhREcsK"
LIVE_ROOT = "/root/live"
CREDS_PATH = "/root/.sn68_mysql_creds"
STATE_DIR = "/root/wandb_log_state"
PROJECT_NAME = os.environ.get("WANDB_PROJECT", "sn68-nanobody")
USER_AGENT = "curl/8.0"


# --------------------------------------------------------------------------
# Creds loader (mirrors nanobody_opt.db.load_creds; values go into os.environ
# only and are never printed).
# --------------------------------------------------------------------------
def load_creds(path: str = CREDS_PATH) -> None:
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key, value.strip())


# --------------------------------------------------------------------------
# HTTP + leaderboard
# --------------------------------------------------------------------------
def http_get_json(url: str) -> dict:
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_leaderboard(epoch: int) -> dict:
    return http_get_json(LEADERBOARD_URL.format(epoch=epoch))


def discover_epoch() -> int:
    """Newest epoch with a published leaderboard (finalized), probing down
    from the newest /root/live/epoch* directory."""
    start = None
    try:
        nums = []
        for d in glob.glob(os.path.join(LIVE_ROOT, "epoch*")):
            b = os.path.basename(d)
            if b.startswith("epoch") and b[5:].isdigit():
                nums.append(int(b[5:]))
        if nums:
            start = max(nums)
    except Exception:
        pass
    if start is None:
        start = int(os.environ.get("SN68_EPOCH_START", "26000"))
    for e in range(start, max(start - 30, 0), -1):
        try:
            lb = get_leaderboard(e)
            if isinstance(lb, dict) and lb.get("success") and isinstance(lb.get("data"), list):
                return e
        except urllib.error.HTTPError:
            continue
        except Exception:
            continue
    return start


# --------------------------------------------------------------------------
# Metric helpers
# --------------------------------------------------------------------------
def _nanobody_scores(entries) -> list:
    out = []
    for e in entries:
        for nb in (e.get("nanobodies") or []):
            v = nb.get("final_score")
            if v is not None:
                out.append(float(v))
    return out


def _best_score(entries):
    vals = [float(e["final_score"]) for e in entries if e.get("final_score") is not None]
    return min(vals) if vals else None


def _best_rank(entries):
    vals = [int(e["nanobody_leaderboard_rank"]) for e in entries
            if e.get("nanobody_leaderboard_rank") is not None]
    return min(vals) if vals else None


def _dist(scores):
    if not scores:
        return {}
    return {
        "min": min(scores),
        "median": statistics.median(scores),
        "max": max(scores),
        "mean": statistics.mean(scores),
    }


def _add_dist(metrics, prefix, scores):
    for k, v in _dist(scores).items():
        metrics[f"{prefix}/dist/{k}"] = v


# --------------------------------------------------------------------------
# Artifact metrics (dedup reference source + parent pool)
# --------------------------------------------------------------------------
def artifact_metrics(epoch: int):
    extra = {"dedup_source": None, "parents_hash": None, "iteration": None}
    prod = os.path.join(LIVE_ROOT, f"epoch{epoch}", "producer")
    if not os.path.isdir(prod):
        return {}, extra
    iters = sorted(
        [d for d in os.listdir(prod) if d.startswith("iteration_")],
        key=lambda s: int(s.split("_", 1)[1]) if s.split("_", 1)[1].isdigit() else 0,
    )
    if not iters:
        return {}, extra
    path = os.path.join(prod, iters[-1], "top50_index.json")
    if not os.path.exists(path):
        return {}, extra
    with open(path) as fh:
        d = json.load(fh)
    source = d.get("source")
    live = d.get("live_sequences") or []
    h = hashlib.md5("\n".join(sorted(live)).encode("utf-8")).hexdigest()
    m = {
        "dedup/source_hf_archive": 1.0 if source == "hf_archive" else 0.0,
        "dedup/source_our_top50": 1.0 if source == "our_top50" else 0.0,
        "dedup/archive_count": float(d.get("archive_count") or 0),
        "dedup/index_n": float(d.get("n") or 0),
        "parents/live_sequences_count": float(len(live)),
        "parents/live_sequences_hash_fp": float(int(h[:12], 16)),
    }
    extra = {"dedup_source": source, "parents_hash": h, "iteration": iters[-1]}
    return m, extra


# --------------------------------------------------------------------------
# DB metrics (read-only aggregates)
# --------------------------------------------------------------------------
def _sanitize(s):
    return "".join(c if c.isalnum() or c == "_" else "_" for c in str(s))


def db_metrics(epoch: int):
    try:
        import pymysql  # noqa: WPS433 (lazy import so --dry-run works without it)
    except Exception as exc:  # pragma: no cover
        print(f"[db] pymysql unavailable ({type(exc).__name__}); skipping DB metrics")
        return {}
    load_creds()
    host = os.environ.get("SN68_MYSQL_HOST", "46.225.101.210")
    port = int(os.environ.get("SN68_MYSQL_PORT", "3306"))
    user = os.environ.get("SN68_MYSQL_USER", "sn68")
    password = os.environ.get("SN68_MYSQL_PASSWORD", "")
    dbname = os.environ.get("SN68_MYSQL_DB", "sn68")
    try:
        conn = pymysql.connect(
            host=host, port=port, user=user, password=password, database=dbname,
            charset="utf8mb4", autocommit=True, connect_timeout=10,
        )
    except Exception as exc:
        print(f"[db] connect failed: {type(exc).__name__}: {exc}")
        return {}
    m: dict = {}
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM observations WHERE epoch=%s", (epoch,))
            m["db/obs_total"] = float(cur.fetchone()[0])
            cur.execute(
                "SELECT COUNT(DISTINCT node_id) FROM observations WHERE epoch=%s", (epoch,)
            )
            m["db/obs_distinct_nodes"] = float(cur.fetchone()[0])
            cur.execute(
                "SELECT source, score_kind, COUNT(*), MIN(score), MAX(score) "
                "FROM observations WHERE epoch=%s GROUP BY source, score_kind",
                (epoch,),
            )
            for src, sk, cnt, mn, mx in cur.fetchall():
                base = f"db/obs/{_sanitize(src)}__{_sanitize(sk)}"
                m[f"{base}/count"] = float(cnt)
                if mn is not None:
                    m[f"{base}/min_score"] = float(mn)
                if mx is not None:
                    m[f"{base}/max_score"] = float(mx)
    finally:
        conn.close()
    return m


# --------------------------------------------------------------------------
# Gather everything
# --------------------------------------------------------------------------
def gather(epoch: int):
    lb = get_leaderboard(epoch)
    data = lb.get("data") or []
    ours = [e for e in data if (e.get("coldkey") or "").startswith(OUR_PREFIX)]
    rival = [e for e in data if (e.get("coldkey") or "").startswith(RIVAL_PREFIX)]

    our_scores = _nanobody_scores(ours)
    rival_scores = _nanobody_scores(rival)

    m = {"field/size": float(len(data))}
    m["ours/entries"] = float(len(ours))
    m["rival/entries"] = float(len(rival))
    m["ours/count"] = float(len(our_scores))
    m["rival/count"] = float(len(rival_scores))

    b = _best_score(ours)
    if b is not None:
        m["ours/best_score"] = b
    b = _best_score(rival)
    if b is not None:
        m["rival/best_score"] = b
    r = _best_rank(ours)
    if r is not None:
        m["ours/best_rank"] = float(r)
    r = _best_rank(rival)
    if r is not None:
        m["rival/best_rank"] = float(r)

    _add_dist(m, "ours", our_scores)
    _add_dist(m, "rival", rival_scores)

    art, extra = artifact_metrics(epoch)
    m.update(art)
    m.update(db_metrics(epoch))

    extra["epoch"] = epoch
    extra["field_size"] = len(data)
    extra["our_best_score"] = _best_score(ours)
    extra["our_best_rank"] = _best_rank(ours)
    extra["rival_best_score"] = _best_score(rival)
    extra["rival_best_rank"] = _best_rank(rival)
    return m, extra


# --------------------------------------------------------------------------
# Privacy (best effort)
# --------------------------------------------------------------------------
def ensure_private_project(entity):
    import wandb

    api = wandb.Api()  # reads WANDB_API_KEY / WANDB_BASE_URL / WANDB_ENTITY from env
    resolved = entity or api.default_entity
    if not resolved:
        print("[privacy] could not resolve entity; project will be created by "
              "wandb.init() on first run (private by default).")
        return resolved
    try:
        api.create_project(name=PROJECT_NAME, entity=resolved)
        print(f"[privacy] upserted project {resolved}/{PROJECT_NAME} "
              "(new projects are private by default).")
    except Exception as exc:
        print(f"[privacy] create_project failed ({type(exc).__name__}: {exc}); "
              "project will still be created private by wandb.init().")
    try:
        proj = api.project(PROJECT_NAME, entity=resolved)
        vis = getattr(proj, "visibility", None)
        print(f"[privacy] project visibility = {vis!r} "
              "(None means the SDK does not expose it; default is private).")
    except Exception as exc:
        print(f"[privacy] could not read project visibility: {type(exc).__name__}: {exc}")
    return resolved


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def state_file(epoch: int) -> str:
    return os.path.join(STATE_DIR, f"{epoch}.done")


def main() -> int:
    ap = argparse.ArgumentParser(description="Read-only W&B logger for SN68 nanobody mining")
    ap.add_argument("--epoch", type=int, default=None, help="pin an epoch (default: latest finalized)")
    ap.add_argument("--dry-run", action="store_true", help="print the payload only; no key, no upload")
    ap.add_argument("--force", action="store_true", help="re-log an epoch even if already logged")
    args = ap.parse_args()

    epoch = args.epoch or discover_epoch()
    metrics, extra = gather(epoch)

    base_url = os.environ.get("WANDB_BASE_URL")
    entity = os.environ.get("WANDB_ENTITY")
    key_set = bool(os.environ.get("WANDB_API_KEY"))

    print(f"[epoch] {epoch}")
    print(f"[project] {PROJECT_NAME}")
    print(f"[base_url] {base_url or '(unset -> SDK default https://api.wandb.ai)'}")
    print(f"[entity] {entity or '(unset -> resolved from the API key owner)'}")
    print(f"[key] WANDB_API_KEY: {'set' if key_set else 'MISSING'}")

    if args.dry_run:
        print("[mode] dry-run (no upload, no key required)")
        print("---- wandb.log payload (step=%d) ----" % epoch)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        print("---- run config / summary extra ----")
        print(json.dumps(extra, indent=2, sort_keys=True))
        return 0

    if not key_set:
        print("[error] WANDB_API_KEY is not set; refusing to run without a key.")
        print("        Export it via /root/.wandb_env (chmod 600) or run with --dry-run.")
        return 2

    sf = state_file(epoch)
    if os.path.exists(sf) and not args.force:
        print(f"[skip] epoch {epoch} already logged ({sf}); use --force to re-log")
        return 0

    import wandb

    resolved_entity = ensure_private_project(entity)
    run_kwargs = dict(
        project=PROJECT_NAME,
        name=f"epoch-{epoch}",
        id=f"sn68-epoch-{epoch}",
        resume="allow",
    )
    if resolved_entity:
        run_kwargs["entity"] = resolved_entity

    run = wandb.init(**run_kwargs)
    try:
        run.config.update(extra)
        run.summary.update(extra)
        wandb.log(metrics, step=epoch)
    finally:
        wandb.finish()

    os.makedirs(STATE_DIR, exist_ok=True)
    with open(sf, "w") as fh:
        fh.write(f"{epoch}\n")
    print(f"[done] logged epoch {epoch} -> {resolved_entity or '(default entity)'}/{PROJECT_NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
