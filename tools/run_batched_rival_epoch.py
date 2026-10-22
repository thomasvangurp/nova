#!/usr/bin/env python3
"""Master orchestrator: real submissions -> slave replicate batch -> verified SQLite writeback."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import sqlite3
import subprocess
import sys

sys.path[:0] = ["/root", "/root/tools"]


def run(argv, timeout=14460):
    result = subprocess.run(argv, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{Path(argv[0]).name} failed rc={result.returncode}; output withheld")
    return result


def main():
    from live_submission_priority import critical
    if critical():
        print('BATCHED_RIVALS_DEFERRED current_submission_priority', flush=True)
        return 4
    from collect_slave_multipass_seqs import leaderboard_sequences, qualified_sequences, pass_counts
    from nanobody_opt.rank_exact import fetch_recent_fields_from_db, pool_recent_fields
    from ingest_slave_multipass import load_rows, validate
    from nanobody_opt.mutation_network import update_observation_store

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("epoch", type=int)
    parser.add_argument("--passes", type=int, default=20)
    parser.add_argument("--max-sequences", type=int)
    args = parser.parse_args()
    from recover_batched_rival_outbox import recover
    recover()
    board = leaderboard_sequences(args.epoch)
    if not board:
        print(f"RUN_SLAVE_MULTIPASS_PENDING epoch={args.epoch}", flush=True)
        return 3
    wanted = sorted(set(board) | set(qualified_sequences(args.epoch)))
    unsupported = sum(len(s) != 127 for s in wanted)
    wanted = [s for s in wanted if len(s) == 127]
    before = pass_counts(wanted)
    todo = [s for s in wanted if before.get(s, 0) < args.passes]
    if args.max_sequences:
        todo = todo[:args.max_sequences]
    if not todo:
        print(f"RUN_SLAVE_MULTIPASS_DONE epoch={args.epoch} (nothing left to score)", flush=True)
        return 0
    job_id = hashlib.sha256(json.dumps([args.passes, todo]).encode()).hexdigest()[:16]
    root = Path(f"/root/artifacts/batched_rivals/epoch{args.epoch}/{job_id}")
    root.mkdir(parents=True, exist_ok=True)
    inp, field, output = root / "sequences.json", root / "field.json", root / "observations.ndjson"
    inp.write_text(json.dumps(todo))
    db = Path("/root/artifacts/nanobody_mutations.sqlite")
    fields = fetch_recent_fields_from_db(db, target="P20809", before_epoch=args.epoch, count=10)
    if not fields:
        raise RuntimeError("no recent validator field; refusing another measurement regime")
    pooled = pool_recent_fields(fields)
    field.write_text(json.dumps({"epochs": [e for e, _ in fields], "rows": json.loads(pooled.to_json(orient="records", double_precision=15))}, allow_nan=False))
    host = "root@ssh2.vast.ai"
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-p", "28358", host]
    scp = ["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-P", "28358"]
    remote = f"/root/slave_multipass_in/epoch{args.epoch}_{job_id}"
    remote_out = f"/root/slave_multipass_out/epoch{args.epoch}_{job_id}_batched.ndjson"
    print(f"BATCHED_RIVALS_PENDING epoch={args.epoch} sequences={len(todo)} passes={args.passes} unsupported={unsupported}", flush=True)
    with open("/root/artifacts/slave_gpu.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if critical():
            print('BATCHED_RIVALS_DEFERRED current_submission_priority', flush=True)
            return 4
        # Another in-flight batch may have filled this cohort while we waited.
        # Recollect on the next scheduler pass rather than rescore identical IDs.
        current = pass_counts(todo)
        if any(current.get(s, 0) >= args.passes for s in todo):
            print(f"BATCHED_RIVALS_RECHECK epoch={args.epoch} coverage_changed_while_waiting", flush=True)
            return 0
        run(ssh + ["mkdir -p /root/slave_multipass_in /root/slave_multipass_out"], timeout=30)
        run(scp + [str(inp), f"{host}:{remote}_sequences.json"], timeout=60)
        run(scp + [str(field), f"{host}:{remote}_field.json"], timeout=60)
        command = ["/root/release_python_slave.sh", "/root/slave_multipass_writeback.py",
                   "--sequences-json", remote + "_sequences.json", "--field-json", remote + "_field.json",
                   "--epoch", str(args.epoch), "--passes", str(args.passes), "--gpus", "0,1,2,3", "--out", remote_out]
        run(ssh + [shlex.join(command)])
        run(scp + [f"{host}:{remote_out}", str(output)], timeout=90)
        rows = load_rows(output)
        if validate(rows):
            raise RuntimeError("canonical observation validation failed")
        counts = Counter(r["sequence"] for r in rows)
        if set(counts) != set(todo) or any(n != args.passes for n in counts.values()):
            raise RuntimeError("writeback lost replicate rows")
        if any(not r.get("metrics") or not r.get("allow_replicates") for r in rows):
            raise RuntimeError("writeback missing raw metrics or replicate provenance")
        written = update_observation_store(db, rows, commit_every=1000, commit_interval=5.0)
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as connection:
            connection.execute("BEGIN")
            for row in rows:
                stored = connection.execute(
                    "SELECT n.sequence,n.target,o.source,o.score_kind,o.metrics_json,"
                    "o.provenance_json,o.observation_count FROM observations o "
                    "JOIN nodes n ON n.id=o.node_id WHERE o.observation_id=?",
                    (row["observation_id"],)).fetchone()
                if (not stored or tuple(stored[:4]) != (
                        row["sequence"], row["target"], row["source"], row["score_kind"])
                        or json.loads(stored[4]) != row["metrics"]
                        or json.loads(stored[5]).get("allow_replicates") is not True
                        or stored[6] != 1):
                    raise RuntimeError("exact per-replicate database verification failed")
        after = pass_counts(todo)
        if any(after.get(s, 0) < args.passes for s in todo):
            raise RuntimeError("database replicate count verification failed")
        distinct = [len({hashlib.sha256(json.dumps(r["metrics"], sort_keys=True).encode()).hexdigest()
                         for r in rows if r["sequence"] == s}) for s in todo]
        print(f"BATCHED_RIVALS_VERIFIED epoch={args.epoch} sequences={len(todo)} rows={len(rows)} "
              f"written={written} minimum_distinct_metric_vectors={min(distinct)}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"BATCHED_RIVALS_FAILED {type(error).__name__}: {error}", flush=True)
        raise SystemExit(1)
