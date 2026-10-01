#!/usr/bin/env bash
set -eu
unset PYTHONPATH
export CUDA_HOME=/root/.pixi/envs/bz2
export PATH="$CUDA_HOME/bin:/usr/local/bin:/usr/bin:/bin"
export CPLUS_INCLUDE_PATH="$CUDA_HOME/include"
export TORCH_CUDA_ARCH_LIST=12.0
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TRITON_CACHE_DIR=/root/.cache/triton-blackwell-reference
export CUDA_CACHE_PATH=/root/.nv/ComputeCache
export CUDA_CACHE_MAXSIZE=4294967296
export BOLTZ_CACHE=/root/nim19-test/oss-home/.boltz
export PYTHONPATH=/root/nim19-test/oss-repo/external_tools/boltz/src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export NOVA_GPU_NOVELTY=1 NOVA_BOLTZ_KERNELS=1 NOVA_MAX_SIMILARITY=0.5
exec /root/.pixi/envs/bz2/bin/python "$@"
