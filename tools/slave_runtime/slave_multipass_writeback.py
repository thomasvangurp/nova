#!/usr/bin/env python3
"""Score independent replicates on slave; master alone imports the raw rows."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from nanobody_opt.rules import target_for_epoch
from nanobody_opt.simulate_round import METRICS as SUM_OF_TEN_DIRS
import yaml


def replicate_command(inp, out_dir, gpus, config, target):
    return [sys.executable, str(ROOT/'nanobody_opt/score_boltzgen_fast_adapter.py'),
            '--sequences', str(inp), '--config', str(config), '--target', target,
            '--gpus', gpus, '--shards-per-gpu', '1', '--batch-size', '2',
            '--work-root', str(out_dir/'work'), '--out', str(out_dir/'exact_all.json'),
            '--structure-dir', str(out_dir/'structures'), '--require-structures', '--allow-replicates']


def validate_context(epoch, config, field):
    target = target_for_epoch(epoch)
    if config['protein_selection']['nanobody']['target'] != target or field.get('target') != target:
        raise ValueError('replay target differs from epoch or field')
    if not field.get('rows'):
        raise ValueError('recent validator field is empty')
    epochs = field.get('epochs')
    if (not isinstance(epochs, list) or not epochs
            or any(not isinstance(e, int) or isinstance(e, bool) or e >= epoch
                   or target_for_epoch(e) != target for e in epochs)):
        raise ValueError('replay reference must use completed epochs of the same target')
    for row in field['rows']:
        for key in SUM_OF_TEN_DIRS:
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('replay reference has incomplete or nonfinite metrics')
    return target


def score_group(sequences, passes, root, gpus, config, target, *, daemon=True,
                independent_calls=False):
    # Replicate keys are indexed within a single length group. The ordinary
    # waves wrapper deduplicates inputs, so replicate batches use the adapter.
    records = []
    for length in sorted({len(s) for s in sequences}):
        group = [s for s in sequences if len(s) == length]
        if independent_calls:
            # Keep the master's registered base-v1 runtime: independent
            # ordinary calls need no replica-batching capability or new pin.
            group_root = independent_checkpoint(root/f'length{length}', group, passes, config, target)
            for replica in range(1, passes + 1):
                work = group_root / f'pass{replica}'
                work.mkdir(parents=True, exist_ok=True)
                output = work/'exact_all.json'
                completed = None
                if output.is_file():
                    try:
                        completed = ordinary_records(output, group, target, replica)
                    except (OSError, ValueError, KeyError, TypeError):
                        pass
                if completed is not None:
                    records.extend(completed)
                    continue
                inp = work/'input.json'
                inp.write_text(json.dumps({'sequences': [{'sequence': s} for s in group]}))
                command = replicate_command(inp, work, gpus, config, target)
                command.remove('--allow-replicates')
                if not daemon:
                    command.append('--no-daemon')
                env = dict(os.environ, NANOBODY_SCORER_DAEMON='1' if daemon else '0')
                started = time.monotonic()
                result = subprocess.run(command, env=env, capture_output=True, timeout=14400)
                if result.returncode:
                    raise RuntimeError(f'independent scorer failed rc={result.returncode}; output withheld')
                records.extend(ordinary_records(output, group, target, replica))
                (work/'timing.json').write_text(json.dumps({
                    'pass': replica, 'elapsed_seconds': time.monotonic() - started,
                    'model_reuse': bool(daemon)}))
                shutil.rmtree(work/'work', ignore_errors=True)
                shutil.rmtree(work/'structures', ignore_errors=True)
            continue
        work = root / f'length{length}'
        work.mkdir(parents=True, exist_ok=True)
        inp = work / 'input.json'
        inp.write_text(json.dumps({'sequences': [{'sequence': s} for s in group for _ in range(passes)]}))
        env = dict(os.environ, NANOBODY_SCORER_DAEMON='1' if daemon else '0',
                   NANOBODY_SCORER_SOCKET='/root/artifacts/nanobody_scorer.sock')
        command = replicate_command(inp, work, gpus, config, target)
        if not daemon:
            command.append('--no-daemon')
        result = subprocess.run(command,
                                env=env, capture_output=True, timeout=14400)
        if result.returncode:
            raise RuntimeError(f'replicate scorer failed rc={result.returncode}; output withheld')
        payload = json.loads((work/'exact_all.json').read_text())
        if payload.get('target', target) != target:
            raise ValueError('replicate result target mismatch')
        records.extend(unpack(payload['per_nanobody_components']['0'], group, passes, target))
        shutil.rmtree(work/'work', ignore_errors=True)
        shutil.rmtree(work/'structures', ignore_errors=True)
    return records


def independent_checkpoint(root, sequences, passes, config, target):
    """Reuse complete calls only under the same input, runtime and settings."""
    from nanobody_opt import score_boltzgen_fast_adapter as adapter
    settings = adapter.load_settings()
    runtime = [Path(__file__), Path(adapter.__file__),
               Path(settings.get('scorer', adapter.DEFAULT_SCORER))]
    if settings.get('python'):
        runtime.append(Path(settings['python']))
    identity = {'sequences': sequences, 'passes': passes, 'target': target,
                'config': hashlib.sha256(config.read_bytes()).hexdigest(),
                'settings': settings,
                'runtime': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                            if p.is_file() else None for p in runtime}}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    marker = root/'context.sha256'
    previous = marker.read_text().strip() if marker.is_file() else None
    if root.exists() and previous != signature:
        # Preserve incompatible partial work for audit; never relabel it as
        # evidence from this invocation's scientific context.
        root.rename(root.with_name(root.name + '.stale-' + str(time.time_ns())))
    root.mkdir(parents=True, exist_ok=True)
    marker.write_text(signature + '\n')
    return root


def ordinary_records(path, sequences, target, replica):
    payload = json.loads(path.read_text())
    if payload.get('target', target) != target:
        raise ValueError('independent result target mismatch')
    wave = payload['per_nanobody_components']['0']
    if set(wave) != set(sequences):
        raise ValueError('independent result cardinality mismatch')
    return [(sequence, replica, _metrics(wave[sequence].get(target, {}), replica))
            for sequence in sequences]


@contextmanager
def owned_daemon():
    """Keep model reuse inside this lease's owned process tree and socket."""
    from nanobody_opt import score_boltzgen_fast_adapter as adapter, scorer_daemon
    settings = adapter.load_settings()
    scorer = str(settings.get('scorer', adapter.DEFAULT_SCORER))
    adapter.require_pinned_scorer(Path(scorer), settings.get('scorer_sha256'))
    directory = Path(tempfile.mkdtemp(prefix='nova-dash-scorer-'))
    socket = directory/'worker.sock'
    keys = ('NANOBODY_SCORER_DAEMON', 'NANOBODY_SCORER_SOCKET', 'NANOBODY_WORKER_LOG_DIR')
    previous = {k: os.environ.get(k) for k in keys}
    process = None
    try:
        os.environ.update(NANOBODY_SCORER_DAEMON='1', NANOBODY_SCORER_SOCKET=str(socket),
                          NANOBODY_WORKER_LOG_DIR=str(directory))
        process = subprocess.Popen(
            [settings.get('python') or sys.executable, str(ROOT/'nanobody_opt/scorer_daemon.py'),
             '--scorer', scorer, '--socket', str(socket)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 120
        while not scorer_daemon.available(str(socket)):
            if process.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError('owned_daemon_unavailable')
            time.sleep(.2)
        yield
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=5)
        # The outer cooperative lease is the subreaper and confirms that all
        # owned descendants, including worker sessions, are gone before unlock.
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(directory, ignore_errors=True)


def reference_ranks(records, field):
    reference = {metric: sorted({float(row[metric]) for row in field['rows']}) for metric in SUM_OF_TEN_DIRS}
    return [(s, r, m, float(sum(1 + sum(v < m[k] if d == 'min' else v > m[k] for v in reference[k])
                               for k, d in SUM_OF_TEN_DIRS.items()))) for s, r, m in records]


def unpack(wave, sequences, passes, target):
    expected = {}
    for index, sequence in enumerate(s for s in sequences for _ in range(passes)):
        expected[hashlib.sha256(sequence.encode("ascii")).hexdigest() + str(index)] = (
            sequence, index % passes + 1)
    if set(wave) != set(expected):
        raise ValueError(f"replicate cardinality mismatch: expected={len(expected)} actual={len(wave)}")
    records = []
    for key, (sequence, replicate) in expected.items():
        records.append((sequence, replicate, _metrics(wave[key].get(target, {}), replicate)))
    return records


def _metrics(values, replicate):
    metrics = {}
    for metric in SUM_OF_TEN_DIRS:
        value = values.get(metric)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"replicate {replicate} missing finite metric {metric}")
        metrics[metric] = float(value)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences-json", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--field-json", required=True, type=Path)
    parser.add_argument("--epoch", required=True, type=int)
    parser.add_argument("--passes", default=20, type=int)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--out-root", default="/root/slave_multipass_work", type=Path)
    parser.add_argument("--direct", action="store_true",
                        help="use only the caller's leased GPU, bypassing the shared daemon")
    parser.add_argument("--independent-calls", action="store_true",
                        help="collect separate ordinary calls with the registered base-v1 worker")
    parser.add_argument("--owned-daemon", action="store_true",
                        help="reuse registered models in a private daemon owned by this lease")
    args = parser.parse_args()
    sequences = sorted(set(json.loads(args.sequences_json.read_text())))
    if not sequences or args.passes < 2:
        raise ValueError("requires supported sequences and at least two replicates")
    field_bytes = args.field_json.read_bytes()
    field = json.loads(field_bytes)
    target = validate_context(args.epoch, yaml.safe_load(args.config.read_text()), field)
    job_id = hashlib.sha256(json.dumps([args.passes, sequences]).encode()).hexdigest()[:16]
    root = args.out_root / f"epoch{args.epoch}" / f"replicates{args.passes}" / job_id
    root.mkdir(parents=True, exist_ok=True)
    if args.direct and args.owned_daemon:
        raise ValueError('direct and owned-daemon modes are mutually exclusive')
    with owned_daemon() if args.owned_daemon else nullcontext():
        records = score_group(sequences, args.passes, root, args.gpus, args.config, target,
                              daemon=not args.direct, independent_calls=args.independent_calls)
    rows = []
    for sequence, replicate, metrics, score in reference_ranks(records, field):
        rows.append(dict(observation_id=f"slave_batched:epoch{args.epoch}:{sequence}:pass{replicate}",
                         sequence=sequence, target=target, epoch=args.epoch,
                         source="public_multipass_rivals", score_kind="multipass_sum10_pass",
                         score=float(score), observation_count=1, metrics=metrics,
                         writer="slave_multipass_writeback", formula="sum10",
                         pass_index=replicate, passes=args.passes, allow_replicates=True,
                         field="recent_validator_epochs", field_epochs=field["epochs"], field_rows=len(field["rows"]),
                         field_sha256=hashlib.sha256(field_bytes).hexdigest()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".ndjson.part")
    tmp.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    os.replace(tmp, args.out)
    # Raw metrics and merged result remain durable; structures are not consumed by replay readers.
    shutil.rmtree(root / "work", ignore_errors=True)
    shutil.rmtree(root / "structures", ignore_errors=True)
    distinct = [len({json.dumps(m, sort_keys=True) for s, _, m in records if s == seq}) for seq in sequences]
    print(f"BATCHED_RIVALS_DONE rows={len(rows)} minimum_distinct_metric_vectors={min(distinct)}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        with open("/root/artifacts/rival_replicate_gpu.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            raise SystemExit(main())
    except Exception as error:
        print(f"BATCHED_RIVALS_FAILED {type(error).__name__}: {error}", flush=True)
        raise SystemExit(1)
