#!/usr/bin/env python3
"""Continuous per-epoch bank-update loop (orchestrator).

Cycle:
  1. Detect new public sequences: fetch the latest leaderboard, diff against
     the DB nodes by sequence hash.
  2. 8x-score the new public sequences (source='public_8x').
  3. Promote elite parents: top-12% of public sequences by mean multi-pass
     rank-sum (local_validator_replay_rank_sum), lower is better.
  4. Generate single-substitution children of newly-promoted elite parents at
     the 7 hot framework positions x top-2 winning residues, and 8x-score them
     (source='children_8x').
  5. Update the bank: parents + children land in the DB via ingest_backfill.py.

Idempotent by sequence hash: scoring workers skip already-scored passes, and the
DB upsert is keyed on observation_id. --dry-run detects/computes/generates but
does no scoring and writes nothing.

Run automatically via cron/systemd on an always-on host after each epoch close.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
DB = "/root/artifacts/nanobody_mutations.sqlite"          # production
ARCHIVE = "/root/live/epoch25422/archive.csv"             # production (rank field)
PROD = "ssh3.vast.ai"
PROD_PORT = 11806
TARGET = "P20809"
LEADERBOARD_ROOT = "https://compound-api.metanova-labs.ai"
PASSES = 8
ELITE_FRACTION = 0.12
HOT_TOP2 = {0: "QR", 2: "TH", 113: "HF", 115: "AS", 120: "TS", 121: "ST", 123: "TP"}

SSHO = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=25"]
BOXES = [
    ("b0", "118.163.199.123", 12945), ("b1", "60.250.87.179", 58248),
    ("b2", "185.229.26.83", 41335), ("b3", "118.163.199.123", 56759),
    ("b4", "ssh4.vast.ai", 18324), ("b5", "ssh1.vast.ai", 25066),
    ("b6", "ssh3.vast.ai", 34196), ("b7", "ssh6.vast.ai", 27242),
    ("b8", "ssh3.vast.ai", 20070), ("b9", "211.72.13.201", 41764),
]


def ssh(host, port, cmd):
    return subprocess.run(["ssh", *SSHO, "-p", str(port), f"root@{host}", cmd],
                          capture_output=True, text=True)


def seq_hash(s: str) -> str:
    return hashlib.sha256(s.encode("ascii")).hexdigest()


def _remote(op, **kw):
    """Run the remote helper on production and return its JSON payload."""
    import tempfile
    subprocess.run(["scp", *SSHO, "-P", str(PROD_PORT), str(REPO / "epoch_loop_remote.py"),
                    f"{PROD}:/root/epoch_loop_remote.py"], capture_output=True)
    args = " ".join(f"{k}={v}" for k, v in kw.items())
    r = ssh(PROD, PROD_PORT, f"python3 /root/epoch_loop_remote.py {op} {args}")
    try:
        return json.loads(r.stdout)
    except Exception:
        return {}


def detect_new_public(epochs: int) -> dict[str, int]:
    payload = _remote("new_public", epochs=epochs)
    return {d["sequence"]: d["epoch"] for d in payload.get("new_public", [])}


def load_public_replay_scores() -> list[tuple[str, float]]:
    payload = _remote("replay_scores")
    return [(d["sequence"], d["mean_replay"]) for d in payload.get("replay_scores", [])]


def db_existing_sequences() -> set[str]:
    payload = _remote("existing")
    return set(payload.get("existing", []))


def promote_elite(scores: list[tuple[str, float]]) -> tuple[list[str], float]:
    """Top-12% by mean replay score (lower is better)."""
    ranked = sorted(scores, key=lambda x: (x[1], x[0]))
    k = max(1, int(len(ranked) * ELITE_FRACTION))
    elite = ranked[:k]
    threshold = elite[-1][1]
    return [s for s, _ in elite], threshold


def generate_children(parents: list[str], existing: set[str]) -> list[dict]:
    children: dict[str, dict] = {}
    for seq in parents:
        for pos, tos in HOT_TOP2.items():
            for to in tos:
                if pos >= len(seq) or seq[pos] == to:
                    continue
                child = seq[:pos] + to + seq[pos + 1:]
                if child in existing:
                    continue
                children.setdefault(child, {"sequence": child, "parent": seq,
                                            "position": pos, "to": to})
    return list(children.values())


def dispatch_and_score(items: list[dict], tag: str) -> int:
    """Shard items across boxes, deploy shards + worker, launch scoring.

    tag distinguishes the results dir (public vs children) so the running
    children bootstrap is not disturbed. Idempotent by sequence hash.
    """
    if not items:
        return 0
    buckets = [[] for _ in BOXES]
    for i, it in enumerate(items):
        buckets[i % len(BOXES)].append(it)
    work_dir = REPO / f"loop_shards_{tag}"
    work_dir.mkdir(exist_ok=True)
    for (sh, host, port), bucket in zip(BOXES, buckets):
        (work_dir / f"{sh}.json").write_text(json.dumps(bucket))
        subprocess.run(["scp", *SSHO, "-P", str(port), str(work_dir / f"{sh}.json"),
                        str(REPO / "score_worker.py"), f"root@{host}:/root/"], capture_output=True)
        ssh(host, port, f"mkdir -p /root/results_{tag} /root/bgen_shards_{tag}")
        ssh(host, port, f"nohup python3 /root/score_worker.py --shard /root/{sh}.json "
                        f"--results /root/results_{tag}/{sh}.json --work-root /root/bgen_shards_{tag} "
                        f"--batch-size 256 --gpus 0,1,2,3,4,5,6,7 "
                        f"> /root/results_{tag}/{sh}.worker.log 2>&1 < /dev/null &")
    return len(items)


def collect_and_ingest(tag: str, source: str) -> int:
    """Pull results, push to prod, run ingest with the given source."""
    results_dir = REPO / f"results_{tag}"
    results_dir.mkdir(exist_ok=True)
    for sh, host, port in BOXES:
        subprocess.run(["scp", *SSHO, "-P", str(port), f"root@{host}:/root/results_{tag}/{sh}.json",
                        str(results_dir / f"{sh}.json")], capture_output=True)
    ssh(PROD, PROD_PORT, f"mkdir -p /root/backfill_{tag}")
    for sh, _, _ in BOXES:
        p = results_dir / f"{sh}.json"
        if p.exists():
            subprocess.run(["scp", *SSHO, "-P", str(PROD_PORT), str(p),
                            f"{PROD}:/root/backfill_{tag}/"], capture_output=True)
    subprocess.run(["scp", *SSHO, "-P", str(PROD_PORT), str(REPO / "ingest_backfill.py"),
                    f"{PROD}:/root/"], capture_output=True)
    r = ssh(PROD, PROD_PORT, f"/root/bgen_python.sh /root/ingest_backfill.py "
                             f"/root/backfill_{tag} {ARCHIVE} {DB} {source}")
    print(r.stdout.strip(), r.stderr.strip())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=3, help="leaderboard epochs to scan")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--output", type=Path, default=REPO / "epoch_loop_output.json")
    args = ap.parse_args()

    new = detect_new_public(args.epochs)
    print(f"[1] new public sequences detected: {len(new)}")
    for s, ep in sorted(new.items(), key=lambda x: x[1]):
        print(f"    epoch={ep} len={len(s)} hash={seq_hash(s)[:16]}...")

    scores = load_public_replay_scores()
    elite, threshold = promote_elite(scores)
    print(f"[3] elite parents: {len(elite)} (top-12% of {len(scores)} public, "
          f"threshold={threshold:.2f})")

    existing = db_existing_sequences()
    children = generate_children(elite, existing)
    print(f"[4] children of elite parents: {len(children)}")

    payload = {
        "new_public_count": len(new),
        "new_public": [{"sequence": s, "epoch": e} for s, e in sorted(new.items(), key=lambda x: x[1])],
        "elite_parent_count": len(elite),
        "elite_threshold": threshold,
        "children_count": len(children),
    }
    args.output.write_text(json.dumps(payload, indent=2))

    if args.dry_run:
        print(f"DRY_RUN complete (no scoring, no DB writes). wrote {args.output}")
        return 0

    # --- live: 8x-score new public, then promote + children ---
    if new:
        items = [{"sequence": s, "epoch": e} for s, e in new.items()]
        dispatch_and_score(items, "public_loop")
        collect_and_ingest("public_loop", "public_8x")

    # promote elite fresh (after new public have multi-pass scores)
    scores = load_public_replay_scores()
    elite, _threshold = promote_elite(scores)
    existing = db_existing_sequences()
    children = generate_children(elite, existing)
    if children:
        items = [{"sequence": c["sequence"], "epoch": 25433} for c in children]
        dispatch_and_score(items, "children_loop")
        collect_and_ingest("children_loop", "children_8x")
    print("LIVE loop cycle complete")
    return 0


def _decode(h, m):
    return m.get(h, "")


if __name__ == "__main__":
    sys.exit(main())
