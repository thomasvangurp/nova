#!/usr/bin/env python3
"""Import committed slave outbox files even if their original coordinator exited."""
import hashlib
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sqlite3
import sys

sys.path[:0] = ["/root", "/root/tools"]


def _recover_unlocked():
    from ingest_slave_multipass import load_rows, validate
    from nanobody_opt.mutation_network import update_observation_store
    from nanobody_opt.simulate_round import METRICS

    root = Path("/root/artifacts/batched_rivals/recovered")
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "imported.json"
    done = json.loads(manifest.read_text()) if manifest.exists() else {}
    # This endpoint produces only a JSON inventory of committed result filenames.
    command = "python3 -c 'from pathlib import Path; import json; print(json.dumps([(p.name,p.stat().st_mtime) for p in Path(\"/root/slave_multipass_out\").glob(\"epoch*_batched.ndjson\")]))'"
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-p", "28358", "root@ssh2.vast.ai"]
    result = subprocess.run(ssh + [command], capture_output=True, timeout=30)
    if result.returncode:
        raise RuntimeError(f"outbox inventory failed rc={result.returncode}")
    inventory = sorted(json.loads(result.stdout), key=lambda item: item[1])
    files = rows_written = 0
    for name, modified_at in inventory:
        if not re.fullmatch(r"epoch[0-9]+_[a-f0-9]{16}_batched\.ndjson", name):
            continue
        if isinstance(done.get(name), dict) and done[name].get("modified_at") == modified_at:
            continue
        local = root / name
        result = subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-P", "28358",
                                 "root@ssh2.vast.ai:/root/slave_multipass_out/" + name,
                                 str(local)], capture_output=True, timeout=60)
        if result.returncode:
            raise RuntimeError(f"outbox transfer failed rc={result.returncode}")
        digest = hashlib.sha256(local.read_bytes()).hexdigest()
        rows = load_rows(local)
        if not rows or validate(rows):
            raise ValueError("outbox canonical label validation failed")
        by_sequence = {}
        for row in rows:
            metrics = row.get("metrics") or {}
            if (row.get("allow_replicates") is not True or row.get("observation_count") != 1
                    or not set(METRICS).issubset(metrics)
                    or not all(not isinstance(metrics[k], bool) and math.isfinite(float(metrics[k])) for k in METRICS)):
                raise ValueError("outbox row is not a complete individual replicate")
            by_sequence.setdefault(row["sequence"], []).append(row)
        for sequence_rows in by_sequence.values():
            passes = sequence_rows[0]["passes"]
            if passes != 20 or len(sequence_rows) != passes or {r["pass_index"] for r in sequence_rows} != set(range(1, passes + 1)):
                raise ValueError("outbox replicate cardinality is incomplete")
        update_observation_store(Path("/root/artifacts/nanobody_mutations.sqlite"), rows)
        with sqlite3.connect("file:/root/artifacts/nanobody_mutations.sqlite?mode=ro", uri=True) as connection:
            for row in rows:
                actual = connection.execute("SELECT metrics_json,provenance_json,observation_count FROM observations WHERE observation_id=?", (row["observation_id"],)).fetchone()
                if not actual or json.loads(actual[0]) != row["metrics"] or json.loads(actual[1]).get("allow_replicates") is not True or actual[2] != 1:
                    raise RuntimeError("recovered replicate database verification failed")
        done[name] = {"sha256": digest, "modified_at": modified_at}
        temporary = manifest.with_suffix(".next")
        temporary.write_text(json.dumps(done, sort_keys=True))
        os.replace(temporary, manifest)
        files += 1
        rows_written += len(rows)
    print(f"BATCHED_RIVALS_OUTBOX_RECOVERED files={files} rows={rows_written}", flush=True)
    return files, rows_written


def recover():
    root = Path("/root/artifacts/batched_rivals/recovered")
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "recovery.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _recover_unlocked()


if __name__ == "__main__":
    try:
        recover()
    except Exception as error:
        print(f"BATCHED_RIVALS_OUTBOX_FAILED {type(error).__name__}", flush=True)
        raise SystemExit(1)
