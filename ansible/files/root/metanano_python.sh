#!/usr/bin/env bash
# Isolated entrypoint for NOVA's Layer-2 filters. TNP combines CPU tools with
# CUDA-backed ImmuneBuilder structure prediction.
unset PYTHONPATH
export PATH="/root/metanano_shim/bin:/root/.pixi/envs/metanano/bin:$PATH"
export IGDATA=/root/external_tools/igblast
export MPLCONFIGDIR=/root/.cache/matplotlib
export PYTHONPATH=/root/metanano_shim:/root/external_tools/NOVA-nanobody-filter
exec /root/.pixi/envs/metanano/bin/python "$@"
