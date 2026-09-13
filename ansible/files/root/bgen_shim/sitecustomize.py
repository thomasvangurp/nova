"""Opt-in deterministic seeding for cross-environment parity experiments.

Python imports ``sitecustomize`` automatically during interpreter startup when
this shim directory is on ``PYTHONPATH``.  Normal validator-shaped scoring is
unchanged unless ``BOLTZGEN_DIAGNOSTIC_SEED`` is explicitly set.  BoltzGen runs
each pipeline stage in a fresh Python subprocess, so seeding only the outer
scorer process would not control the random stream used by diffusion/refolding.
"""
from __future__ import annotations

import os
import sys


# ---------------------------------------------------------------------------
# Make this shim's `utils` win over the real package of the same name.
#
# boltzgen_wrapper does `from utils import get_sequence_from_protein_code,
# seq_hash`, and this directory supplies exactly those two symbols without
# nova's validator import graph. Putting the shim first on PYTHONPATH was enough
# while /root held no `utils` package -- consolidating the deployment onto a
# single runtime root put the real one there, and every scorer shard then died on
# `ModuleNotFoundError: No module named 'metanano'` three imports deep, having
# written nothing.
#
# Reordering sys.path here does NOT work: for `python -m` and `python -c` the
# working directory is re-inserted at sys.path[0] AFTER site initialisation, so
# anything sitecustomize does to the path is undone. Registering the module in
# sys.modules does work, because a name already present there is never searched
# for on the path again.
_shim = os.path.dirname(os.path.abspath(__file__))
_shim_utils = os.path.join(_shim, "utils", "__init__.py")
if "utils" not in sys.modules and os.path.isfile(_shim_utils):
    import importlib.util

    _spec = importlib.util.spec_from_file_location(
        "utils", _shim_utils, submodule_search_locations=[os.path.join(_shim, "utils")]
    )
    if _spec and _spec.loader:
        _module = importlib.util.module_from_spec(_spec)
        sys.modules["utils"] = _module
        try:
            _spec.loader.exec_module(_module)
        except Exception:
            # Never let the shim break an interpreter that did not need it.
            del sys.modules["utils"]

raw_seed = os.environ.get("BOLTZGEN_DIAGNOSTIC_SEED")
if raw_seed is not None:
    seed = int(raw_seed)
    if seed < 0:
        raise ValueError("BOLTZGEN_DIAGNOSTIC_SEED must be non-negative")

    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # This is diagnostic-only. It deliberately trades speed for a strict A/B
    # in which runtime packaging is the independent variable. Live scoring
    # never sets BOLTZGEN_DIAGNOSTIC_SEED and therefore retains upstream
    # behavior.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
