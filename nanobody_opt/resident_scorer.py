"""Keep the scorer's BoltzGen stages resident ACROSS rounds, not just chunks.

The pinned scorer already ships `resident.py`, whose own notes say it best:

    of a 345 s chunk, ~176 s (51%) is process start + checkpoint load, paid
    again for every chunk -- and the live producer runs chunks of 2
    ...
    folding a single design drops from 52.9 s to 13.9 s resident
    Numerics are unchanged: same weights, same predict_args, same forward.

But `fast_score_batch._worker` holds its stages in a LOCAL dict::

    def _worker(args):
        stages: dict[str, ResidentStage] = {}

so residency amortises across chunks inside one invocation and dies when the
call returns. The producer runs one invocation per replication round, so every
round rebuilds the model.

Measured on the live producer over 168 replication rounds:

    seconds = 32.9 + 1.02 * candidates          (4 GPUs)
    9,136 s total, 3,610 s scoring, 5,526 s (60%) reloading

and a sampled 150 s of the box showed 2.5% mean GPU utilisation with two of
four GPUs at a flat zero -- the hardware waiting on import and checkpoint load.

WHAT THIS DOES. Memoises ResidentStage by (config, overrides) at module scope,
so `_worker`'s fresh local dict receives a WARM stage. The pinned scorer is
never edited and its sha256 stays valid; this wraps it, the same way
blueprint's worker_replay.py wraps Boltz2's checkpoint loader.

WHAT IT DOES NOT DO. It does not batch replicates into one call. That was
tried and measured: the scorer deduplicates by sequence at merge and returned
"MERGE LOST SEQUENCES: submitted 12, merged 2. Refusing to write a partial
result." Each round stays its own invocation with its own predict composition,
which is also what keeps the metric comparable -- the scorer's notes record
the same molecule moving by >2x depending on what shares its predict call.
"""
from __future__ import annotations

import os
import sys

#: Off unless asked for. This changes how a production scorer holds GPU state.
ENABLED = os.environ.get("NANOBODY_RESIDENT_SCORER", "0") == "1"
#: Distinct resident stages to keep. Each holds a model on its GPU, so this is
#: a VRAM bound, not a taste preference.
MAX_STAGES = int(os.environ.get("NANOBODY_RESIDENT_MAX_STAGES", "4"))

_CACHE: dict = {}
_TEMPLATE_CACHE: dict = {}
_POOL_CACHE: dict = {}
_INSTALLED = False


def stage_key(config_path, overrides) -> tuple:
    """Two stages are interchangeable only if config AND overrides match.

    Overrides carry sampler budgets (recycling/sampling steps, diffusion
    samples). Serving a stage built for one budget to a round that asked for
    another would silently change the measured metric, which is the one thing
    this must never do.
    """
    items = tuple(sorted((overrides or {}).items(), key=lambda kv: str(kv[0])))
    return (str(config_path), items)


def install(resident_module=None) -> bool:
    """Memoise ResidentStage construction. Idempotent; returns True if active."""
    global _INSTALLED
    if _INSTALLED:
        return True
    module = resident_module
    if module is None:
        try:
            import resident as module  # type: ignore
        except ImportError:
            return False

    real_init = module.ResidentStage.__init__
    base = module.ResidentStage

    class _Memoised(base):  # type: ignore[misc, valid-type]
        def __new__(cls, config_path, overrides=None, *args, **kwargs):
            key = stage_key(config_path, overrides)
            existing = _CACHE.get(key)
            if existing is not None:
                existing._resident_cache_hit = True
                return existing
            if len(_CACHE) >= MAX_STAGES:
                # Never silently exceed the VRAM budget: fall back to a fresh
                # uncached stage rather than evict a model another round may
                # still be using.
                obj = super().__new__(cls)
                obj._resident_cache_hit = False
                obj._resident_uncached = True
                return obj
            obj = super().__new__(cls)
            obj._resident_cache_hit = False
            obj._resident_uncached = False
            _CACHE[key] = obj
            return obj

        def __init__(self, config_path, overrides=None, *args, **kwargs):
            if getattr(self, "_resident_cache_hit", False):
                return          # warm: keep the model already on the GPU
            real_init(self, config_path, overrides, *args, **kwargs)

    module.ResidentStage = _Memoised
    _INSTALLED = True
    return True


def install_template_library(synth_module=None) -> bool:
    """Cache the rotamer TemplateLibrary by the .cif files it is built from.

    ``TemplateLibrary.from_cifs`` reads a fixed set of reference structures and
    holds no per-call state, so rebuilding it every round is pure repetition.
    The scorer's note that things are "rebuilt per chunk" describes the current
    code, not a constraint: the datamodule and writer carry per-chunk paths,
    this does not.
    """
    module = synth_module
    if module is None:
        try:
            import synth_fullatom as module  # type: ignore
        except ImportError:
            return False
    library = getattr(module, "TemplateLibrary", None)
    if library is None or getattr(library, "_resident_cached", False):
        return False
    real = library.from_cifs

    def cached_from_cifs(paths, *args, **kwargs):
        key = tuple(sorted(str(p) for p in paths))
        if args or kwargs:          # an unexpected signature must not be cached
            return real(paths, *args, **kwargs)
        got = _TEMPLATE_CACHE.get(key)
        if got is None:
            got = real(paths)
            _TEMPLATE_CACHE[key] = got
        return got

    library.from_cifs = staticmethod(cached_from_cifs)
    library._resident_cached = True
    return True


def install_analysis_pool(scorer_module) -> bool:
    """Reuse the spawned analysis pool instead of starting one per call.

    ``_AnalysisPool`` starts a Python process that imports torch; that is
    seconds of setup whose only per-call input is the jobs pushed through it.
    Reuse is only safe if the pool is drained between calls, so this keeps the
    SAME pool object and lets the scorer's own drain logic run against it.
    """
    pool_class = getattr(scorer_module, "_AnalysisPool", None)
    if pool_class is None or getattr(scorer_module, "_resident_pool_cached", False):
        return False
    real_init = pool_class.__init__

    class _Reused(pool_class):  # type: ignore[misc, valid-type]
        def __new__(cls, ctx, *args, **kwargs):
            got = _POOL_CACHE.get("pool")
            if got is not None and getattr(got, "proc", None) is not None \
                    and got.proc.is_alive():
                got._resident_cache_hit = True
                return got
            obj = super().__new__(cls)
            obj._resident_cache_hit = False
            _POOL_CACHE["pool"] = obj
            return obj

        def __init__(self, ctx, *args, **kwargs):
            if getattr(self, "_resident_cache_hit", False):
                return          # already spawned and alive
            real_init(self, ctx, *args, **kwargs)

    scorer_module._AnalysisPool = _Reused
    scorer_module._resident_pool_cached = True
    return True


def stats() -> dict:
    return {"installed": _INSTALLED, "stages": len(_CACHE),
            "templates": len(_TEMPLATE_CACHE), "pools": len(_POOL_CACHE),
            "max_stages": MAX_STAGES, "enabled": ENABLED}


def reset() -> None:
    """Drop every cache. For tests and for releasing VRAM between epochs."""
    _CACHE.clear()
    _TEMPLATE_CACHE.clear()
    _POOL_CACHE.clear()


def install_all(scorer_module=None) -> dict:
    """Install every safe cache. Returns what actually took effect.

    Ordered cheapest-risk first. Each is independent: a failure to install one
    never blocks the others, and every one of them can be reverted by not
    calling this.
    """
    applied = {"stages": install(), "templates": install_template_library()}
    if scorer_module is not None:
        applied["analysis_pool"] = install_analysis_pool(scorer_module)
    return applied


def bootstrap(scorer_path: str) -> bool:
    """Put the pinned scorer's directory on sys.path, then install.

    The scorer imports `resident` and `utils` as top-level modules from its own
    directory, so importing it out of tree fails with ModuleNotFoundError
    unless that directory is on the path.
    """
    directory = os.path.dirname(os.path.abspath(scorer_path))
    if directory not in sys.path:
        sys.path.insert(0, directory)
    return install()
