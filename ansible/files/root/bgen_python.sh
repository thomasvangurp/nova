#!/usr/bin/env bash
# Interpreter shim for boltzgen scoring, mirroring search_python.sh on the miner.
#
# PYTHONPATH order is load-bearing: /root/bgen_shim must precede everything so
# its minimal `utils` and `bittensor` win. boltzgen_wrapper imports both, and
# the real ones drag in the whole validator stack (metanano, datasets,
# combinatorial_db) that this scoring-only box neither has nor needs.
#
# TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD: torch 2.6+ defaults weights_only=True and
# the checkpoints carry an omegaconf DictConfig, so loading fails with
# "Unsupported global: omegaconf.dictconfig.DictConfig" without it.
export PYTHONPATH=/root/bgen_shim
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export BOLTZGEN_CACHE=/root/.boltzgen
exec /root/mamba/envs/bgen/bin/python "$@"
