#!/usr/bin/env bash
# Runtime for the fast nanobody scorer.
#
# Same pixi environment as the production /root/bgen_python.sh; the only
# differences are deliberate:
#
#   PYTHONPATH  points at release/boltzgen_src instead of the live vendored
#               tree, so the live miner is unaffected by anything here and a
#               rollback is one path change.
#   thread caps production sets none, and 43 torch processes x ~71 threads
#               take a 128-core box to load 299. Each scoring worker only ever
#               needs a handful.
#
# /root/bgen_shim must stay FIRST: boltzgen_wrapper imports `bittensor` and
# `utils`, and the real ones drag in the whole validator stack.
unset PYTHONPATH
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-4}
exec /usr/local/bin/pixi run \
  --manifest-path /root/pixi.toml \
  --environment bgen \
  env PYTHONPATH="/root/bgen_shim:/root/fastscore/release/boltzgen_src" \
  python "$@"
