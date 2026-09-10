"""Opt-in deterministic seeding for cross-environment parity experiments.

Python imports ``sitecustomize`` automatically during interpreter startup when
this shim directory is on ``PYTHONPATH``.  Normal validator-shaped scoring is
unchanged unless ``BOLTZGEN_DIAGNOSTIC_SEED`` is explicitly set.  BoltzGen runs
each pipeline stage in a fresh Python subprocess, so seeding only the outer
scorer process would not control the random stream used by diffusion/refolding.
"""
from __future__ import annotations

import os


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
