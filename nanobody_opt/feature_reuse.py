"""Stop analysis recomputing `get_feat` for a design folding already featurised.

WHAT WAS MEASURED (prior session). `Analyze.compute_metrics` does
`feat = self.data.predict_set[idx]`, which walks `__getitem__` ->
`getitem_from_paths` -> `get_feat`, and `get_feat` re-parses the design's cif,
re-tokenizes it and re-runs the featurizer -- the SAME work the folding
stage's own DataLoader already did a few seconds earlier for the SAME file, in
the SAME process family, just discarded once `FoldingWriter.write_on_batch_end`
returns. Costed at ~0.6-0.9s/design.

THE CORRECTNESS QUESTION THIS MODULE RESOLVES FIRST. Folding's generated
`folding.yaml` sets `target_templates: true`; the generated `analysis.yaml`
sets `target_templates: false`. Those two settings do NOT produce the same
`feat` dict: `FromGeneratedDataset.get_feat` (data_from_generated.py:338-520)
branches on `self.target_templates` to either compute real
`template_from_tokens(...)` features or fill `template_*` keys with
`load_dummy_templates(...)` zeros -- read carefully, line by line:

  * The branch (data_from_generated.py:500-509) runs strictly AFTER
    `self.featurizer.process(...)` has already produced every OTHER feature
    key. `features.update(templates_features)` only ADDS/OVERWRITES the eight
    `template_*` keys (template_restype, template_frame_rot, template_frame_t,
    template_cb, template_ca, template_mask_cb, template_mask_frame,
    template_mask). No other key in the dict is touched by this branch, and
    no earlier line in `get_feat` reads `self.target_templates` -- so a
    folding-time feat (real templates) and an analysis-time feat (dummy
    templates) for the SAME design file are IDENTICAL in every key except
    those eight.
  * `boltzgen/task/analyze/analyze.py` and
    `boltzgen/task/analyze/analyze_utils.py` -- which between them define
    every metric `Analyze.compute_metrics` calls, including
    `get_fold_metrics`, `count_noncovalents` and `get_delta_sasa` -- contain
    ZERO occurrences of the substring "template" anywhere outside comments
    unrelated to this feature (verified with `grep -n template` against both
    files; the only hit in analyze.py is the word inside `compute_metrics`
    itself, not a feature key). Read in full: `get_fold_metrics` reads
    `feat["coords"]`/`res_type`/`atom_to_token`/`backbone_mask` plus the
    folded `.npz`'s OWN stored `best_sample` confidence keys (design_ptm,
    design_iptm, design_iiptm, interaction_pae, etc. -- these are read
    straight out of the npz the model wrote at fold time, never recomputed
    from `feat`). `count_noncovalents` and `get_delta_sasa` work from
    `feat["coords"]`/`atom_to_token`/structure geometry and from the written
    refolded cif file, respectively. Neither touches `template_*`.
  * Every metric in the production checkpoint set this module was built to
    protect -- design_iiptm, design_ptm, design_to_target_iptm,
    min_design_to_target_pae, interaction_pae, plip_hbonds_refolded,
    plip_saltbridge_refolded, delta_sasa_refolded, liability_score,
    liability_num_violations -- is therefore blind to which `target_templates`
    setting produced `feat`. This was additionally verified empirically (not
    just read): `tools/verify_feature_reuse.py` folds real designs once, then
    runs `Analyze.compute_metrics` twice against the IDENTICAL fold output --
    once recomputing `feat` the normal way (dummy templates, today's
    production path) and once loading the real-template `feat` this module
    persisted -- and diffs all ten fields. See that script's output for the
    per-design numbers; this module's reuse is gated OFF until that diff is
    clean.

CONCLUSION: no config normalization is needed and no field needs to be
excluded from reuse. The `target_templates` mismatch between folding and
analysis is real but provably inert for every metric this scorer reports.

WHAT THIS MODULE DOES, same two-sided pattern as chunk_reuse.py: monkey-patch
at import time, gated by an env var, without touching the pinned scorer's own
files.

1. `install_folding_side()` wraps `boltzgen.task.predict.writer.FoldingWriter
   .write_on_batch_end` (the PLAIN folding writer only, not `AffinityWriter`
   or `DesignWriter`) so that, after it does everything it already does, it
   ALSO pickles the per-design feature dict the folding DataLoader just built
   -- the exact object `batch` holds when Lightning calls the writer -- to a
   small gzip file next to `fold_out_npz/<id>.npz`. Runs in the SAME process
   that already holds those tensors; no extra GPU/CPU work recomputes
   anything, this is a serialize-to-disk of data already in memory.

2. `install_analysis_side()` wraps `FromGeneratedDataset.get_feat` so that,
   when called for a design that has such a cache file sitting next to it,
   it loads and returns the cached dict instead of re-parsing the cif,
   re-tokenizing and re-featurizing. Falls back to the real `get_feat` call
   whenever the cache is missing, unreadable, mismatched (batch size other
   than 1, `id` mismatch, or conditioning fields -- ss_type/binding_type --
   that disagree with what folding saw), so a cache miss or any surprise
   degrades to today's behaviour rather than risking a wrong metric. This
   must be installed INSIDE the spawned analysis child process
   (`fast_score_batch._analysis_child`), which is a separate process from
   folding and does its own fresh imports -- installing it in the parent has
   no effect there.

BATCH SIZE. Both the generated `folding.yaml` and `analysis.yaml` pin
`data.cfg.batch_size: 1` in production (verified against the live configure
output, not assumed). The write side only persists when the batch it was
handed truly has exactly one design in it and refuses (falls through to a
warning, no persistence) otherwise, so a future config change to batch>1
cannot silently cache the wrong thing -- it just stops caching.

OFF BY DEFAULT. `NANOBODY_FEATURE_REUSE=1` turns it on. Each installer is
independent and failure-isolated the same way chunk_reuse.py's are: an
exception from one is reported and leaves the others, and the scorer, alone.
"""
from __future__ import annotations

import gzip
import os
import pickle
import time
import traceback
from pathlib import Path

#: Off unless asked for. This changes what bytes feed the metrics a live
#: miner publishes on-chain -- keep it off until the diff in
#: verify_feature_reuse.py is clean for every field, every design.
ENABLED = os.environ.get("NANOBODY_FEATURE_REUSE", "0") == "1"

#: Companion-file naming. Lives inside the SAME directory FoldingWriter
#: already writes `<id>.npz` into (design_dir/fold_out_npz), so it travels
#: and gets cleaned up with that directory for free.
CACHE_SUFFIX = ".feat.pkl.gz"

_WARNED: set = set()
_STATS = {"persisted": 0, "reused": 0, "skipped_batch_size": 0,
          "miss": 0, "fallback_error": 0, "persist_error": 0,
          "persist_seconds": 0.0, "reuse_seconds_saved_estimate": 0.0}


def _warn_once(key: str, exc: Exception) -> None:
    if key in _WARNED:
        return
    _WARNED.add(key)
    print(f"[feature_reuse] {key}: {type(exc).__name__}: {exc}", flush=True)


def stats() -> dict:
    return dict(_STATS)


# --- 1. persist the folding batch -------------------------------------------

def _cache_path_for(outdir: Path, design_id: str) -> Path:
    return Path(outdir) / f"{design_id}{CACHE_SUFFIX}"


def _persist_batch(outdir: Path, batch: dict) -> None:
    ids = batch.get("id")
    if not isinstance(ids, list) or len(ids) != 1:
        # Only ever handle the batch_size==1 shape production actually uses.
        # Anything else (a future config change) is left uncached, not
        # miscached.
        _STATS["skipped_batch_size"] += 1
        return
    design_id = ids[0]
    try:
        # batch_size==1 means every value's leading list/stack-dim-0 entry IS
        # the single original (pre-collate) feature dict value -- true for
        # BOTH the torch.stack(dim=0) keys and the plain-list-kept keys in
        # `collate()`, since both reduce to a length-1 container here.
        #
        # By the time the writer callback runs, Lightning has already moved
        # `batch` onto the accelerator device (verified empirically: without
        # the `.cpu()` below, the persisted tensors are CUDA tensors, and
        # `get_feat` is normally called from analysis -- a CPU-only task that
        # never touches a GPU -- so a cached dict holding CUDA tensors is not
        # a drop-in replacement for what `getitem_from_paths` produces. Move
        # everything back to CPU on the way to disk, matching the real
        # dataset's own output exactly.
        import torch as _torch

        def _to_cpu(v):
            if _torch.is_tensor(v):
                return v.detach().cpu()
            return v

        feat = {k: _to_cpu(v[0]) for k, v in batch.items()}
    except Exception as exc:  # noqa: BLE001 - never block a design over this
        _warn_once("persist-unpack", exc)
        _STATS["persist_error"] += 1
        return
    dest = _cache_path_for(outdir, design_id)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    t0 = time.time()
    try:
        with gzip.open(tmp, "wb", compresslevel=1) as fh:
            pickle.dump(feat, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, dest)
        _STATS["persisted"] += 1
        _STATS["persist_seconds"] += time.time() - t0
    except Exception as exc:  # noqa: BLE001
        _warn_once("persist-write", exc)
        _STATS["persist_error"] += 1
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def install_folding_side(writer_module=None, force: bool = False) -> bool:
    """Patch FoldingWriter.write_on_batch_end to also persist feat."""
    if not (force or ENABLED):
        return False
    module = writer_module
    if module is None:
        try:
            from boltzgen.task.predict import writer as module  # type: ignore
        except ImportError:
            return False
    cls = getattr(module, "FoldingWriter", None)
    if cls is None or getattr(cls, "_feature_reuse_write_patched", False):
        return False

    real_write = cls.write_on_batch_end

    def write_on_batch_end(self, trainer=None, pl_module=None, prediction=None,
                            batch_indices=None, batch=None, batch_idx=None,
                            dataloader_idx=0, sample_id=None):
        real_write(self, trainer=trainer, pl_module=pl_module,
                    prediction=prediction, batch_indices=batch_indices,
                    batch=batch, batch_idx=batch_idx,
                    dataloader_idx=dataloader_idx, sample_id=sample_id)
        if self.designfolding:
            # Analysis only ever reads const.folding_dirname (the plain
            # fold, not fold_out_design_npz); persisting that path too would
            # cache bytes nothing reads.
            return
        try:
            _persist_batch(self.outdir, batch)
        except Exception as exc:  # noqa: BLE001 - persistence must never
            _warn_once("persist-outer", exc)   # cost a design its output

    cls.write_on_batch_end = write_on_batch_end
    cls._feature_reuse_write_patched = True
    cls._feature_reuse_real_write = real_write
    return True


# --- 2. reuse the persisted batch in analysis -------------------------------

def _load_cached_feat(path: Path, expect_id: str):
    cache_path = Path(path).parent / "fold_out_npz" / f"{expect_id}{CACHE_SUFFIX}"
    if not cache_path.is_file():
        _STATS["miss"] += 1
        return None
    try:
        with gzip.open(cache_path, "rb") as fh:
            feat = pickle.load(fh)
    except Exception as exc:  # noqa: BLE001
        _warn_once("load", exc)
        _STATS["fallback_error"] += 1
        return None
    if feat.get("id") != expect_id:
        _warn_once("load-id-mismatch", RuntimeError(f"{feat.get('id')!r} != {expect_id!r}"))
        return None
    return feat


def _consistent(feat: dict, ss_type, binding_type) -> bool:
    import numpy as np
    import torch
    if ss_type is not None:
        cached = feat.get("ss_type")
        if cached is None or not torch.equal(cached, torch.from_numpy(np.asarray(ss_type)).long()):
            return False
    if binding_type is not None:
        cached = feat.get("binding_type")
        if cached is None or not torch.equal(cached, torch.from_numpy(np.asarray(binding_type)).long()):
            return False
    return True


def install_analysis_side(data_module=None, force: bool = False) -> bool:
    """Patch FromGeneratedDataset.get_feat to reuse a persisted batch.

    Must run INSIDE the process that will actually call get_feat (the
    spawned analysis child), since that process does its own fresh imports.
    """
    if not (force or ENABLED):
        return False
    module = data_module
    if module is None:
        try:
            from boltzgen.task.predict import data_from_generated as module  # type: ignore
        except ImportError:
            return False
    cls = getattr(module, "FromGeneratedDataset", None)
    if cls is None or getattr(cls, "_feature_reuse_read_patched", False):
        return False

    real_get_feat = cls.get_feat

    def get_feat(self, path, design_mask=None, ss_type=None, binding_type=None):
        if design_mask is None:
            try:
                cached = _load_cached_feat(Path(path), Path(path).stem)
            except Exception as exc:  # noqa: BLE001
                _warn_once("lookup", exc)
                cached = None
            if cached is not None:
                try:
                    if _consistent(cached, ss_type, binding_type):
                        _STATS["reused"] += 1
                        return cached
                except Exception as exc:  # noqa: BLE001
                    _warn_once("consistency-check", exc)
        return real_get_feat(self, path, design_mask=design_mask,
                              ss_type=ss_type, binding_type=binding_type)

    cls.get_feat = get_feat
    cls._feature_reuse_read_patched = True
    cls._feature_reuse_real_get_feat = real_get_feat
    return True


def install_all(writer_module=None, data_module=None, force: bool = False) -> dict:
    applied = {}
    for name, call in (
        ("folding_side", lambda: install_folding_side(writer_module, force=force)),
        ("analysis_side", lambda: install_analysis_side(data_module, force=force)),
    ):
        try:
            applied[name] = call()
        except Exception as exc:  # noqa: BLE001 - never block a round
            applied[name] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
    return applied


def reset() -> None:
    """Undo every patch. For tests / a clean rollback in one process."""
    try:
        from boltzgen.task.predict import writer as wmod
        cls = getattr(wmod, "FoldingWriter", None)
        if cls is not None and getattr(cls, "_feature_reuse_write_patched", False):
            cls.write_on_batch_end = cls._feature_reuse_real_write
            cls._feature_reuse_write_patched = False
    except ImportError:
        pass
    try:
        from boltzgen.task.predict import data_from_generated as dmod
        cls = getattr(dmod, "FromGeneratedDataset", None)
        if cls is not None and getattr(cls, "_feature_reuse_read_patched", False):
            cls.get_feat = cls._feature_reuse_real_get_feat
            cls._feature_reuse_read_patched = False
    except ImportError:
        pass
    _WARNED.clear()
