#!/usr/bin/env bash
# Interpreter shim for boltzgen scoring, mirroring search_python.sh on the miner.
#
# PYTHONPATH order is load-bearing: /root/bgen_shim must precede the vendored
# BoltzGen source so its minimal `utils` and `bittensor` win. boltzgen_wrapper
# imports both, and the real ones drag in the whole validator stack.
#
# Runtime variables, including CUDA compiler discovery, live in pixi.toml's
# bgen activation environment. Always enter through `pixi run` so the wrapper
# cannot drift from that declarative configuration.
unset PYTHONPATH
exec /usr/local/bin/pixi run \
  --manifest-path /root/pixi.toml \
  --environment bgen \
  python "$@"
