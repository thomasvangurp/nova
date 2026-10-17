#!/usr/bin/env bash
# Dry-run the rewritten SN68 producer loop end-to-end, without any on-chain
# submission.
#
# --dry-run isolates every producer artifact under /root/dryrun/epoch*/producer
# (the producer_subtree() helper in run_live_producer.py), so the independent
# live publisher -- which watches /root/live/epoch*/producer and is the ONLY
# thing that commits on chain -- can never see or commit the rehearsal.  The
# race prints DRY_RUN_WOULD_SUBMIT with the candidates that would have been
# handed to the publisher, then stops.
#
# Uses the real scorer daemon contract and the real sqlite mutation DB, so it
# is a genuine end-to-end validation minus the irreversible commit.
#
# Invoke on Box4 (do NOT run against a live epoch's reserve window):
#
#   bash /root/nanobody_opt_v2/dry_run.sh --once
#
# `--once` runs one generate-and-screen iteration and returns.  To also
# exercise the replication tournament in the same run, invoke the race
# directly on a synthetic/small candidate set:
#
#   PYTHONPATH=/root:/root/bgen_shim:/root/fastscore/release/boltzgen_src \
#   /root/validator_probe/parity25747/venv/bin/python -m nanobody_opt_v2.sequential_exact_race \
#     --dry-run --two-stage 1 --replication-rounds 8 \
#     --layer2 <synthetic_layer2.json> --archive <archive.csv> --config <config.yaml> \
#     --target P20809 --epoch <epoch> --output-dir /root/dryrun/race_probe \
#     --mutation-database /root/artifacts/nanobody_mutations.sqlite \
#     --chain-state /root/artifacts/chain_state.json --gpus 0,1,2,3,4,5,6,7
set -euo pipefail

export PYTHONPATH=/root:/root/bgen_shim:/root/fastscore/release/boltzgen_src
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

exec /root/validator_probe/parity25747/venv/bin/python \
  -m nanobody_opt_v2.run_live_producer --dry-run "$@"
