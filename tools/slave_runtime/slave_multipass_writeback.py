#!/usr/bin/env python3
"""Score independent replicates on slave; master alone imports the raw rows."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import shutil
import sys

sys.path.insert(0, "/root")
from nanobody_opt.rules import target_for_epoch
from nanobody_opt.simulate_round import METRICS as SUM_OF_TEN_DIRS
import yaml


def replicate_command(inp, out_dir, gpus, config, target):
    return [sys.executable, '/root/nanobody_opt/score_boltzgen_fast_adapter.py',
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


def score_group(sequences, passes, root, gpus, config, target):
    # Replicate keys are indexed within a single length group. The ordinary
    # waves wrapper deduplicates inputs, so replicate batches use the adapter.
    records = []
    for length in sorted({len(s) for s in sequences}):
        group = [s for s in sequences if len(s) == length]
        work = root / f'length{length}'
        work.mkdir(parents=True, exist_ok=True)
        inp = work / 'input.json'
        inp.write_text(json.dumps({'sequences': [{'sequence': s} for s in group for _ in range(passes)]}))
        env = dict(os.environ, NANOBODY_SCORER_DAEMON='1', NANOBODY_SCORER_SOCKET='/root/artifacts/nanobody_scorer.sock')
        result = subprocess.run(replicate_command(inp, work, gpus, config, target),
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
        values = wave[key].get(target, {})
        metrics = {}
        for metric in SUM_OF_TEN_DIRS:
            value = values.get(metric)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"replicate {replicate} missing finite metric {metric}")
            metrics[metric] = float(value)
        records.append((sequence, replicate, metrics))
    return records


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
    records = score_group(sequences, args.passes, root, args.gpus, args.config, target)
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
