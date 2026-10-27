"""One bounded dashboard replay batch on an idle, cooperatively leased GPU."""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DATABASE = Path("/root/artifacts/nanobody_mutations.sqlite")
STATE = Path("/root/artifacts/dashboard_replay")
PASSES = 10
TTL = 900


def pending(database: Path, epoch: int, *, exclude=()) -> tuple[int | None, list[str]]:
    from nanobody_opt.rules import target_for_epoch
    from tools.collect_slave_multipass_seqs import leaderboard_sequences, pass_counts

    target = target_for_epoch(epoch)
    excluded = set(exclude)
    queue_path = STATE/'pending_cohort.json'
    try:
        queued = json.loads(queue_path.read_text())
    except (OSError, ValueError):
        queued = {}
    items = queued.get('items', []) if queued.get('target') == target else []
    if items:
        counts = pass_counts([item['sequence'] for item in items], target)
        missing = [item for item in items if counts.get(item['sequence'], 0) < PASSES]
        for item in missing:
            if item['sequence'] not in excluded:
                return int(item['epoch']), [item['sequence']]
        if missing:
            return None, []
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=30) as db:
        epochs = [int(row[0]) for row in db.execute(
            "SELECT DISTINCT o.epoch FROM observations o JOIN nodes n ON n.id=o.node_id "
            "WHERE o.source='public_validator_api' AND n.target=? AND o.epoch<? "
            "ORDER BY o.epoch DESC LIMIT 6", (target, epoch))]
    if epochs:
        items = priority_cohort(epoch, epochs[0])
        if items:
            STATE.mkdir(parents=True, exist_ok=True)
            temporary = queue_path.with_suffix('.json.next')
            temporary.write_text(json.dumps({'target': target, 'items': items}))
            os.replace(temporary, queue_path)
            counts = pass_counts([item['sequence'] for item in items], target)
            for item in items:
                if counts.get(item['sequence'], 0) < PASSES and item['sequence'] not in excluded:
                    return int(item['epoch']), [item['sequence']]
    for completed in epochs:
        sequences = sorted(set(leaderboard_sequences(completed) or []))
        counts = pass_counts(sequences, target)
        missing = [s for s in sequences if counts.get(s, 0) < PASSES and s not in excluded]
        if missing:
            return completed, missing[:1]
    return None, []


def priority_cohort(chain_epoch: int, published_epoch: int) -> list[dict]:
    """Finish a committed fleet and its strongest published rival before moving on."""
    from nanobody_dash.collect import fetch_epoch
    from nanobody_opt.rules import target_for_epoch
    from tools.rescale_multipass_history import OUR_COLDKEY
    target = target_for_epoch(chain_epoch)
    own = []
    path = DATABASE.parent/'dashboard_committed_sequences.json'
    try:
        committed = json.loads(path.read_text())
        if (committed.get('target') == target and type(committed.get('epoch')) is int
                and committed['epoch'] < chain_epoch):
            own = [{'epoch': committed['epoch'], 'sequence': row['sequence']}
                   for row in committed.get('submissions', [])
                   if isinstance(row.get('sequence'), str) and row['sequence']]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    groups, scores = {}, {}
    for row in fetch_epoch(published_epoch):
        coldkey, value = row.get('coldkey'), row.get('final_score')
        if not isinstance(coldkey, str):
            continue
        sequences = [nb['sequence'] for nb in row.get('nanobodies', [])
                     if isinstance(nb, dict) and nb.get('protein_name') == target
                     and isinstance(nb.get('sequence'), str) and nb['sequence']]
        groups.setdefault(coldkey, []).extend(sequences)
        if type(value) in (int, float) and math.isfinite(value):
            scores[coldkey] = min(scores.get(coldkey, math.inf), value)
    own.extend({'epoch': published_epoch, 'sequence': sequence}
               for sequence in groups.get(OUR_COLDKEY, []))
    rivals = [key for key in groups if key != OUR_COLDKEY and key in scores]
    best = min(rivals, key=lambda key: scores[key]) if rivals else None
    main = [{'epoch': published_epoch, 'sequence': sequence} for sequence in groups.get(best, [])]
    # Interleave a fixed cohort before filtering its existing pass counts, so
    # the strongest rival is not postponed until all thirteen owned UIDs finish.
    result, seen = [], set()
    for index in range(max(len(own), len(main))):
        for rows in (own, main):
            if index < len(rows) and rows[index]['sequence'] not in seen:
                result.append(rows[index]); seen.add(rows[index]['sequence'])
    return result


def lease_job(directory: Path, *, gpu: int, epoch: int) -> dict:
    # Indices 0-4 are reserved for the live pipeline. Background work may use
    # only 5-7, and even there must pass the existing idle/lock/contention gate.
    if isinstance(gpu, bool) or gpu not in (5, 6, 7):
        raise ValueError("dashboard_gpu_outside_background_pool")
    env = {k: os.environ[k] for k in (
        "PATH", "HOME", "USER", "LANG", "LD_LIBRARY_PATH") if k in os.environ}
    env.update(PYTHONPATH=str(ROOT) + ":/root/bgen_shim:/root/external_tools/boltzgen/src",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1")
    return {"gpu": gpu, "ttl_seconds": TTL, "minimum_free_mib": 12288,
            "scorer": str(ROOT/"nanobody_opt/score_boltzgen_fast_adapter.py"),
            "lock_directory": "/root/pr100-controller/gpu-locks",
            "receipt": str(directory/"lease.json"), "cwd": str(ROOT), "env": env,
            "command": ["taskset", "-c", "104-111", sys.executable,
                        "-m", "tools.slave_runtime.slave_multipass_writeback",
                        "--sequences-json", str(directory/"sequences.json"),
                        "--config", str(directory/"config.yaml"),
                        "--field-json", str(directory/"field.json"),
                        "--epoch", str(epoch), "--passes", str(PASSES),
                        # The pinned scorer sets CUDA_VISIBLE_DEVICES itself,
                        # so its argument must remain the physical leased GPU.
                        "--gpus", str(gpu), "--owned-daemon", "--independent-calls",
                        "--out", str(directory/"observations.ndjson"),
                        "--out-root", str(directory/"work")]}


def run(*, epoch: int, gpu: int = 5, selection=None) -> dict:
    from tools.replay_context import prepare
    from tools.ingest_slave_multipass import load_rows, validate
    from tools.collect_slave_multipass_seqs import pass_counts
    from nanobody_opt.mutation_network import update_observation_store
    from nanobody_opt.rules import target_for_epoch

    selected, sequences = selection if selection is not None else pending(DATABASE, epoch)
    if selected is None:
        return {"ok": True, "reason": "no_pending_published_cohort", "chain_epoch": epoch}
    batch = hashlib.sha256(json.dumps([selected, PASSES, sequences]).encode()).hexdigest()[:16]
    directory = STATE / f"epoch{selected}" / batch
    directory.mkdir(parents=True, exist_ok=True)
    (directory/"sequences.json").write_text(json.dumps(sequences))
    prepare(selected, directory, db=DATABASE)
    job = lease_job(directory, gpu=gpu, epoch=selected)
    job_path = directory/"job.json"
    job_path.write_text(json.dumps(job))
    result = subprocess.run([sys.executable, "-m", "nanobody_opt.pr100_live_gpu", "--job", str(job_path)],
                            cwd=ROOT, capture_output=True, timeout=TTL + 30)
    receipt = json.loads((directory/"lease.json").read_text())
    if (result.returncode or receipt.get("status") != "process_finished"
            or receipt.get("child_exit_code") != 0):
        status = receipt.get("status")
        safe = status if status in {"unavailable", "expired", "contention_abort", "failed",
                                     "containment_failure", "process_finished"} else "unknown"
        return {"ok": False, "reason": "replay_" + safe, "epoch": selected, "gpu": gpu}
    rows = load_rows(directory/"observations.ndjson")
    target = target_for_epoch(selected)
    if (validate(rows) or len(rows) != len(sequences) * PASSES
            or any(r.get("target") != target or r.get("epoch") != selected
                   or r.get("sequence") not in sequences or not r.get("metrics")
                   or r.get("allow_replicates") is not True for r in rows)):
        raise ValueError("replay_writeback_invalid")
    from collections import Counter
    if Counter(r["sequence"] for r in rows) != {s: PASSES for s in sequences}:
        raise ValueError("replay_writeback_incomplete")
    if (len({r.get("observation_id") for r in rows}) != len(rows)
            or any({r.get("pass_index") for r in rows if r["sequence"] == s}
                   != set(range(1, PASSES + 1)) for s in sequences)):
        raise ValueError("replay_writeback_duplicate_passes")
    written = update_observation_store(DATABASE, rows)
    verified = pass_counts(sequences, target)
    if any(verified.get(s, 0) < PASSES for s in sequences):
        raise ValueError("replay_writeback_verification_failed")
    return {"ok": True, "reason": "replay_verified", "epoch": selected, "gpu": gpu,
            "sequences": len(sequences), "rows": written, "passes": PASSES}


def run_parallel(epoch: int) -> dict:
    from concurrent.futures import ThreadPoolExecutor
    plans, assigned = [], set()
    for gpu in (5, 6, 7):
        selection = pending(DATABASE, epoch, exclude=assigned)
        if selection[0] is None:
            break
        plans.append((gpu, selection)); assigned.update(selection[1])
    if not plans:
        return {'ok': True, 'reason': 'no_pending_published_cohort', 'chain_epoch': epoch}
    with ThreadPoolExecutor(max_workers=len(plans)) as pool:
        futures = [pool.submit(run, epoch=epoch, gpu=gpu, selection=selection)
                   for gpu, selection in plans]
        results = []
        for future in futures:
            try:
                results.append(future.result())
            except Exception as error:
                results.append({'ok': False, 'reason': type(error).__name__})
    ok = all(row['ok'] for row in results)
    return {'ok': ok, 'reason': 'replay_verified' if ok else 'replay_pending',
            'epoch': max(row.get('epoch', 0) for row in results),
            'rows': sum(row.get('rows', 0) for row in results), 'batches': results}


def main() -> int:
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE/"refresh.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                epoch = int(json.loads(Path("/root/artifacts/chain_state.json").read_text())["epoch"])
                result = run_parallel(epoch)
        except Exception as error:
            result = {"ok": False, "reason": type(error).__name__}
        result["finished_at"] = time.time()
        pending_path = STATE/"health.json.next"
        pending_path.write_text(json.dumps(result, sort_keys=True))
        os.replace(pending_path, STATE/"health.json")
        print(json.dumps(result, sort_keys=True))
        return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
