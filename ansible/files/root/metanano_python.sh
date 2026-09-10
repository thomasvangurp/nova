#!/usr/bin/env bash
# Compatibility entrypoint for NOVA's Layer-2 filters. Runtime discovery,
# including the GPU-spreading TNP shim, is owned by Pixi activation.
unset PYTHONPATH
exec /usr/local/bin/pixi run \
  --manifest-path /root/pixi.toml \
  --environment metanano \
  python "$@"
