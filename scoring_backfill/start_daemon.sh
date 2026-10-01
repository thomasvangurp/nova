#!/bin/bash
# Start the scorer daemon (production path) on a box. 8 GPU shard workers.
NANOBODY_SCORER_DAEMON=1
exec /usr/local/bin/pixi run --manifest-path /root/pixi.toml --environment bgen \
  env PYTHONPATH="/root/bgen_shim:/root/fastscore/release/boltzgen_src" \
  python /root/nanobody_opt/scorer_daemon.py \
  --scorer /root/fastscore/pinned-530f5dcbc54d/fast_score_batch.py \
  --socket /root/artifacts/nanobody_scorer.sock
