"""Stop the warm scorer rebuilding, per round, what only paths distinguish.

WHAT WAS ACTUALLY MEASURED. The premise handed to this module was that
``fast_score_batch.run_chunk`` rebuilds the BoltzgenWrapper, datamodule,
writer and Trainer for every chunk, and that eliminating those rebuilds is
where the remaining ~60% of replication wall clock lives. A phase profile of a
warm daemon round (4 sequences, batch-size 1, so four chunks, one GPU) says
otherwise:

    wrapper_init      0.005 s   (4 chunks)
    rb_datamodule     0.011 s   (4 chunks)
    rb_writer         0.002 s   (4 chunks)
    rb_trainer_ctor   0.022 s   (4 chunks)
    -------------------------------------
    the whole per-chunk rebuild:  0.04 s per round

Constructing those four objects is 0.2% of a round. The per-round cost is real
but it is somewhere else, and the profile names it:

    resident_stage_init  15-17 s   EVERY round
    pl setup + teardown   2.6 s    (4 chunks)
    analysis close+drain  1.7 s
    configure + yaml      1.1 s

THE MODEL IS REBUILT EVERY ROUND. ``resident_scorer`` memoises ResidentStage
on ``stage_key(config_path, overrides)`` -- the config PATH. That path is
``<shard_root>/shard<N>/chunk<M>/outputs/config/folding.yaml``, and the
producer passes a fresh ``--work-root`` per round, so the key never repeats
and the cache never hits. Worse, after four distinct keys the cache is at
``MAX_STAGES`` and every later round takes the uncached branch permanently.

Measured directly, holding everything else fixed, 4 sequences on one GPU:

    per-round shard_root (production):  round1 36.6 s  round2 38.6 s
    fixed shard_root:                   round1 21.5 s  round2 21.7 s

15 s a round, and 12.5 s of it is ``scipy.stats.truncnorm`` drawing random
initial weights in ``lecun_normal_init_`` that ``load_from_checkpoint``
immediately overwrites.

WHAT THIS MODULE DOES. Three wraps, none of which touch the pinned artifact:

1. ``install_content_keyed_stages`` keys the resident-stage cache on the
   config's CONTENT with the per-chunk paths normalised out, instead of on the
   file path. Safe because ``run_chunk`` overrides exactly those keys on every
   ``run_batch`` call, and because any OTHER difference changes the digest and
   therefore misses -- the cache cannot alias two configs that differ in
   anything the model depends on.

2. ``install_warm_lightning`` stops every ``trainer.predict()`` round-tripping
   the resident model to host memory and emptying the CUDA caching allocator.

3. ``install_persistent_analysis_pool`` keeps the spawned analysis child
   alive, which ``resident_scorer``'s pool memo intends but cannot achieve.

WHAT EACH ONE IS WORTH. The box is shared with a live producer whose scoring
waves saturate all four GPUs for minutes, so runs minutes apart are not
comparable -- the same configuration measured 21.5 s quiet and 32.3 s under a
wave. Alternating the modes round by round makes that load a shared term.
Four cycles, 4 sequences on one GPU, median wall and the paired
within-cycle saving against the same cycle's base round:

    base                     54.4 s     --
    + content-keyed stages   36.7 s     -16.9 s   1.48x
    + warm lightning         28.0 s     -22.6 s   1.94x
    + persistent pool        29.7 s     -24.7 s   1.83x

The medians are noisy at n=3 under 82-100% contention; the paired deltas are
the reliable statistic, and they are monotone.

WHAT IS LEFT, AND WHY IT STAYS. After these three, a warm round is dominated
by folding itself (~15.6 s of a ~18 s round at this shape). The largest
remaining non-inference item is ``configure_command`` plus
``_write_yaml_files``, ~1.1 s a round. It is NOT taken: the scorer's docstring
keeps that call precisely so the step configs are "the ones BoltzGen itself
emits -- not hand-written ones that could drift", and the only way to cache it
across rounds is to re-emit those YAMLs with the chunk path substituted. 1.1 s
of an 18 s round does not buy that risk against a load-bearing metric.

THE METRIC. None of these changes what is computed. The weights are the same
object; a device copy is lossless, so skipping one is bit-identical; batch
composition, shard split, predict_args and sampler budgets are untouched.
That claim is not taken on faith -- ``tests/test_chunk_reuse.py`` pins the
aliasing rules, and the live check compares scores against the current path on
identical inputs with the run-to-run floor measured alongside.

OFF BY DEFAULT. ``NANOBODY_CHUNK_REUSE=1`` turns it on. Every installer is
independent and failure-isolated: one that raises leaves the others, and the
scorer, exactly as they are today.
"""
from __future__ import annotations

import atexit
import hashlib
import os

#: Off unless asked for. This changes how a production scorer holds GPU state.
ENABLED = os.environ.get("NANOBODY_CHUNK_REUSE", "0") == "1"
#: Comma-separated installer names to leave out, so one of the three can be
#: rolled back without rolling back the other two. `warm_lightning` is the one
#: with a VRAM cost: it stops the CUDA caching allocator being emptied between
#: chunks, so a box short of device memory can drop just that.
SKIP = {name.strip() for name in
        os.environ.get("NANOBODY_CHUNK_REUSE_SKIP", "").split(",") if name.strip()}

#: The dotted keys `fast_score_batch.run_chunk` passes to `run_batch` on every
#: call, for the design stage and the folding stage respectively. They are the
#: only per-chunk state a stage config carries, so they -- and nothing else --
#: are normalised out of the content key. Normalising a key run_chunk did NOT
#: override would alias two stages that really differ, so this list must stay
#: exactly the set of overridden keys.
OVERRIDDEN_KEYS = ("output", "data.design_dir", "data.cfg.yaml_path")

_LIVE_POOLS: set = set()
_PATCHED_POOLS: set = set()
_KEYS: dict = {}


# --- 1. content-keyed resident stages ---------------------------------------

def normalised_config_text(config_path) -> str:
    """The stage config with every per-chunk path blanked out.

    Verified against the real generator rather than assumed: regenerating the
    step configs for two different chunk directories AND two different
    sequence sets leaves folding.yaml and analysis.yaml byte-identical once
    the chunk path is normalised, and design.yaml differing only in
    `data.cfg.yaml_path`, which run_chunk overrides.
    """
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(str(config_path))
    for dotted in OVERRIDDEN_KEYS:
        try:
            OmegaConf.update(cfg, dotted, None, merge=False)
        except Exception:                     # key absent in this stage
            continue
    # resolve=False keeps `${data.design_dir}` as the literal interpolation it
    # is, so blanking design_dir does not have to be repeated for the writer.
    return OmegaConf.to_yaml(cfg, resolve=False, sort_keys=True)


def content_stage_key(config_path, overrides) -> tuple:
    """Drop-in replacement for `resident_scorer.stage_key`.

    Falls back to the path-keyed identity if the config cannot be read, so an
    unreadable config degrades to today's behaviour instead of raising inside
    a round.
    """
    items = tuple(sorted((overrides or {}).items(), key=lambda kv: str(kv[0])))
    try:
        digest = hashlib.sha256(normalised_config_text(config_path).encode()).hexdigest()
    except Exception:
        return (str(config_path), items)
    _KEYS[digest] = str(config_path)
    return ("content", digest, items)


def install_content_keyed_stages(resident_scorer_module=None) -> bool:
    """Make the resident-stage cache survive a change of shard root."""
    module = resident_scorer_module
    if module is None:
        try:
            import resident_scorer as module      # type: ignore
        except ImportError:
            return False
    if getattr(module, "_chunk_reuse_content_keyed", False):
        return False
    # `_Memoised.__new__` looks `stage_key` up in the module namespace at call
    # time, so rebinding the name is enough and install order does not matter.
    module._chunk_reuse_path_stage_key = module.stage_key
    module.stage_key = content_stage_key
    module._chunk_reuse_content_keyed = True
    return True


# --- 2. keep the resident model on the GPU between predict calls ------------

def _already_on(module, device) -> bool:
    for tensor in module.parameters():
        if tensor.device != device:
            return False
    for tensor in module.buffers():
        if tensor.device != device:
            return False
    return True


def install_warm_lightning() -> bool:
    """Stop each predict() moving the resident model off the GPU and back.

    `Strategy.teardown` ends every `trainer.predict()` with
    `self.lightning_module.cpu()` and `Strategy.setup` begins the next one
    with `model_to_device()`, so the pinned scorer -- which calls predict once
    per chunk against a model it deliberately keeps resident -- walks ~50k
    tensors off the GPU and straight back on, twice per chunk.
    `CUDAAccelerator.teardown` additionally empties the CUDA caching
    allocator, so the next chunk re-allocates every activation block from the
    driver.

    Numerically inert: a device copy is lossless, the weights are the same
    tensors, and the model is on the same device when predict_step runs.
    """
    try:
        from pytorch_lightning.accelerators.cuda import CUDAAccelerator
        from pytorch_lightning.strategies.single_device import SingleDeviceStrategy
        from pytorch_lightning.strategies.strategy import Strategy
    except ImportError:
        return False
    if getattr(Strategy, "_chunk_reuse_warm", False):
        return False

    real_teardown = Strategy.teardown
    # `Strategy.model_to_device` is an empty stub; the move lives in the
    # subclass the scorer actually gets (devices=1, accelerator=gpu,
    # strategy="auto" resolves to SingleDeviceStrategy). Patching the base
    # would be a silent no-op.
    real_to_device = SingleDeviceStrategy.model_to_device
    real_accel_teardown = CUDAAccelerator.teardown

    def teardown(self) -> None:
        # The real teardown skips the .cpu() when no module is attached, and
        # still runs everything else it does (optimizers, precision plugin,
        # checkpoint io). Detaching around the call is the only way to drop
        # one statement of a method we do not own.
        saved = self._lightning_module
        self._lightning_module = None
        try:
            real_teardown(self)
        finally:
            self._lightning_module = saved

    def model_to_device(self) -> None:
        # `self.model`, not `self.lightning_module`: that is the object
        # upstream moves, and the two differ once a strategy wraps the module.
        module = self.model
        if module is not None and _already_on(module, self.root_device):
            return
        real_to_device(self)

    def accelerator_teardown(self) -> None:
        # `_clear_cuda_memory()` returns every cached block to the driver; the
        # next chunk then pays cudaMalloc for activations it had already sized.
        return None

    Strategy.teardown = teardown
    SingleDeviceStrategy.model_to_device = model_to_device
    CUDAAccelerator.teardown = accelerator_teardown
    Strategy._chunk_reuse_warm = True
    Strategy._chunk_reuse_real = (real_teardown, real_to_device, real_accel_teardown)
    return True


# --- 3. a genuinely persistent analysis child -------------------------------

def _shutdown_pools() -> None:
    """Send the sentinel the patched close() withheld.

    The analysis child is non-daemonic, so multiprocessing joins it at
    interpreter exit and would block forever on a pool nobody closed. This
    atexit hook is registered after multiprocessing's, and atexit runs LIFO,
    so it fires first.
    """
    while _LIVE_POOLS:
        pool = _LIVE_POOLS.pop()
        try:
            pool.job_q.put(None)
            pool.proc.join(timeout=30)
        except Exception:                     # nothing useful to do at exit
            pass


def install_persistent_analysis_pool(scorer_module) -> bool:
    """Keep the spawned analysis child alive across rounds.

    `resident_scorer.install_analysis_pool` memoises the pool, but the memo
    cannot hit: `_worker` ends every round with `analysis.close()`, which
    pushes the sentinel and joins the child, so the next round finds a dead
    process and spawns a fresh one. `drain()` -- which collects exactly one
    result per submitted job -- is the real barrier, so the close is pure
    teardown and withholding it loses no synchronisation.
    """
    pool_class = getattr(scorer_module, "_AnalysisPool", None)
    if pool_class is None or getattr(pool_class, "_chunk_reuse_persistent", False):
        return False

    def close(self) -> None:
        pending = getattr(self, "pending", 0)
        if pending:
            # drain() is what makes reuse safe; closing with work outstanding
            # would hand the next round a child still writing the last one's
            # results. Refuse rather than silently keep it.
            raise RuntimeError(
                "analysis pool closed with %d job(s) still pending" % pending)
        _LIVE_POOLS.add(self)

    pool_class._chunk_reuse_real_close = pool_class.close
    pool_class.close = close
    pool_class._chunk_reuse_persistent = True
    _PATCHED_POOLS.add(pool_class)
    atexit.register(_shutdown_pools)
    return True


# --- installation -----------------------------------------------------------

def install_all(scorer_module=None, force: bool = False) -> dict:
    """Install every reuse that measured its keep. Returns what took effect.

    Off unless NANOBODY_CHUNK_REUSE=1 or force=True. Each installer is
    independent and failure-isolated: one that raises is reported as its
    exception and leaves the others, and the scorer, untouched.
    """
    if not (force or ENABLED):
        return {}
    applied: dict = {}
    for name, call in (
        ("content_keyed_stages", lambda: install_content_keyed_stages()),
        ("warm_lightning", lambda: install_warm_lightning()),
        ("persistent_analysis_pool",
         lambda: (install_persistent_analysis_pool(scorer_module)
                  if scorer_module is not None else False)),
    ):
        if name in SKIP:
            applied[name] = "skipped"
            continue
        try:
            applied[name] = call()
        except Exception as error:            # noqa: BLE001 - never block a round
            applied[name] = "%s: %s" % (type(error).__name__, error)
    return applied


def stats() -> dict:
    return {"enabled": ENABLED, "skipped": sorted(SKIP),
            "live_pools": len(_LIVE_POOLS), "content_keys": len(_KEYS)}


def reset() -> None:
    """Undo every patch. For tests, and for a clean rollback in one process."""
    try:
        import resident_scorer as rs           # type: ignore
        if getattr(rs, "_chunk_reuse_content_keyed", False):
            rs.stage_key = rs._chunk_reuse_path_stage_key
            rs._chunk_reuse_content_keyed = False
    except ImportError:
        pass
    try:
        from pytorch_lightning.accelerators.cuda import CUDAAccelerator
        from pytorch_lightning.strategies.single_device import SingleDeviceStrategy
        from pytorch_lightning.strategies.strategy import Strategy
        if getattr(Strategy, "_chunk_reuse_warm", False):
            teardown, to_device, accel = Strategy._chunk_reuse_real
            Strategy.teardown = teardown
            SingleDeviceStrategy.model_to_device = to_device
            CUDAAccelerator.teardown = accel
            Strategy._chunk_reuse_warm = False
    except ImportError:
        pass
    _shutdown_pools()
    while _PATCHED_POOLS:
        pool_class = _PATCHED_POOLS.pop()
        pool_class.close = pool_class._chunk_reuse_real_close
        pool_class._chunk_reuse_persistent = False
    _KEYS.clear()
    _LIVE_POOLS.clear()
