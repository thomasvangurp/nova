"""Sharded boltzgen exact scorer -- the nanobody counterpart of the molecule
track's score_boltz_batch.py.

Ports the optimisations that are already proven on the compound side:
  * one worker process per (GPU, shard), so a 2-GPU box runs 2*shards in flight
  * longest-processing-time ordering, so the last shard does not straggle
  * results merged from per-shard outputs, order preserved

ISOLATION IS LOAD-BEARING. BoltzgenWrapper derives BOLTZGEN_TMP_FILES_DIR from
its own __file__ and writes inputs/ and outputs/ there. Two shards in one tree
would overwrite each other's YAML inputs and read each other's CSVs -- the same
class of failure as the concurrent downloads that truncated each other, and it
would surface as plausible-looking WRONG metrics rather than as an error. Each
worker therefore rebinds that module global to its own directory BEFORE
constructing the wrapper (__init__ reads the global into self.tmp_dir).
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import time
import uuid
from pathlib import Path


# Every subnet_config key BoltzgenWrapper reads, enumerated from its source:
#   grep -oE "subnet_config\['[a-z_]+'\]" boltzgen_wrapper.py | sort -u
# Enumerated rather than discovered, because each missing key surfaces as a
# KeyError inside a worker process, one per run, with the real cause buried
# under a multiprocessing traceback.
REQUIRED_SUBNET_KEYS = (
    "nanobody_target",
    "nanobody_target_clip_interval",
    "nanobody_structure",
    "nanobody_structure_chain",
    "nanobody_structure_res_index",
    "nanobody_structure_binding_site",
    "boltzgen_rank_by",
    "boltzgen_rank_mode",
)


def _flatten_config(cfg: dict, target: str) -> dict:
    """Flatten upstream config.yaml into the flat keys BoltzgenWrapper reads.

    config.yaml nests these under protein_selection.nanobody.* and
    boltzgen_config.*; the wrapper reads them flat. The validator does this
    flattening itself. Deriving them from the upstream file rather than
    hand-writing values keeps the epitope, residue range and clip interval tied
    to upstream -- hand-written values would silently score against a different
    binding site after any upstream change, and still look perfectly healthy.
    """
    nb = (cfg.get("protein_selection") or {}).get("nanobody") or {}
    flat = dict(cfg)
    flat["nanobody_target"] = [target]
    flat["nanobody_target_clip_interval"] = nb.get("clip_interval")
    for k in ("structure", "structure_chain", "structure_res_index",
              "structure_binding_site"):
        if k in nb:
            flat[f"nanobody_{k}"] = nb[k]
    for k, v in (cfg.get("boltzgen_config") or {}).items():
        flat.setdefault(k, v)

    missing = [k for k in REQUIRED_SUBNET_KEYS if flat.get(k) is None]
    if missing:
        raise SystemExit(
            "config.yaml cannot supply these keys BoltzgenWrapper requires: "
            f"{missing}. Check protein_selection.nanobody and boltzgen_config."
        )
    return flat


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_refolded_complexes(
    sequences: list[str],
    targets: list[str],
    chunk_dir: str,
    structure_dir: str | None,
    observation_id: str,
) -> dict[str, dict[str, dict[str, str]]]:
    """Copy only refolded complexes out of the disposable BoltzGen tree."""
    if structure_dir is None:
        return {}
    design_dir = Path(chunk_dir) / "outputs" / "intermediate_designs"
    metrics_path = design_dir / "aggregate_metrics_analyze.csv"
    if not metrics_path.is_file():
        return {}
    with metrics_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {str(row.get("id")): row for row in rows}
    copied: dict[str, dict[str, dict[str, str]]] = {}
    destination = Path(structure_dir) / observation_id
    destination.mkdir(parents=True, exist_ok=True)
    for sequence in sequences:
        sequence_hash = hashlib.sha256(sequence.encode("ascii")).hexdigest()
        for target in targets:
            record_id = f"{sequence_hash}_{target}_input"
            row = by_id.get(record_id)
            if row is None:
                continue
            filename = str(row.get("file_name") or f"{record_id}.cif")
            source = design_dir / "refold_cif" / filename
            if not source.is_file():
                continue
            target_path = destination / f"{sequence_hash}_{target}.cif"
            shutil.copyfile(source, target_path)
            copied.setdefault(sequence, {})[target] = {
                "path": str(target_path),
                "sha256": _sha256(target_path),
            }
    return copied


def _worker(args):
    (shard_id, gpu, seqs, subnet_config, out_path, batch_size,
     keep_work_dir, work_root, structure_dir, observation_id,
     diagnostic_seed) = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # Set this only inside the spawned shard worker.  The pipeline itself uses
    # fresh Python subprocesses, whose opt-in sitecustomize hook consumes the
    # value before importing Torch.  Normal live scoring leaves it unset.
    if diagnostic_seed is not None:
        os.environ["BOLTZGEN_DIAGNOSTIC_SEED"] = str(diagnostic_seed)
    import boltzgen.boltzgen_wrapper as bw
    shard_dir = os.path.join(work_root, f"shard{shard_id}")
    # rebind BEFORE constructing: __init__ copies the global into self.tmp_dir
    bw.BOLTZGEN_TMP_FILES_DIR = shard_dir
    os.makedirs(os.path.join(shard_dir, "inputs"), exist_ok=True)
    os.makedirs(os.path.join(shard_dir, "outputs"), exist_ok=True)
    started = time.time()
    # SMALL BATCHES ARE LOAD-BEARING. Measured on the compound side with the
    # validator's own wrapper: the SAME molecule scores 0.041 in a 24-batch and
    # 0.117 in a 128-batch -- a 2.8x swing driven purely by what else shares the
    # predict() call. Scoring a shard's whole slice at once would rank our
    # candidates by an artefact of how we happened to group them.
    components = {}
    structures = {}
    for start in range(0, len(seqs), batch_size):
        chunk = seqs[start:start + batch_size]
        # a fresh tmp dir per chunk: the wrapper reads predictions back out of a
        # fixed directory, so a reused one would pick up the previous chunk
        chunk_dir = os.path.join(shard_dir, f"chunk{start // batch_size}")
        bw.BOLTZGEN_TMP_FILES_DIR = chunk_dir
        shutil.rmtree(chunk_dir, ignore_errors=True)
        os.makedirs(os.path.join(chunk_dir, "inputs"), exist_ok=True)
        os.makedirs(os.path.join(chunk_dir, "outputs"), exist_ok=True)
        wrapper = bw.BoltzgenWrapper()
        wrapper.tmp_dir = chunk_dir
        wrapper.input_dir = os.path.join(chunk_dir, "inputs")
        wrapper.output_dir = os.path.join(chunk_dir, "outputs")
        try:
            got = wrapper.run_nanobody_inference({0: {"sequences": chunk}}, subnet_config)
        except Exception as exc:
            print(f"  shard {shard_id} chunk {start // batch_size} raised "
                  f"({type(exc).__name__}); continuing", flush=True)
            got = getattr(wrapper, "per_nanobody_components", {}) or {}
        for uid, by_seq in (got or {}).items():
            components.setdefault(uid, {}).update(by_seq)
        for sequence, by_target in _copy_refolded_complexes(
            chunk,
            list(subnet_config["nanobody_target"]),
            chunk_dir,
            structure_dir,
            observation_id,
        ).items():
            structures.setdefault(sequence, {}).update(by_target)
        if not keep_work_dir:
            shutil.rmtree(chunk_dir, ignore_errors=True)
        print(f"  shard {shard_id}: {min(start + batch_size, len(seqs))}/{len(seqs)}",
              flush=True)
    elapsed = time.time() - started
    json.dump({"shard": shard_id, "gpu": gpu, "n": len(seqs),
               "seconds": elapsed, "components": components,
               "structures": structures},
              open(out_path, "w"), default=str)
    return out_path, len(seqs), elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--shards-per-gpu", type=int, default=1)
    ap.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="sequences per predict() call; validator parity requires one",
    )
    ap.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="retain generated complexes and analysis files under "
             "--work-root (use only for small finalist/teacher runs)",
    )
    ap.add_argument(
        "--work-root",
        default="/root/bgen_shards",
        help="isolated shard/output directory; use a unique path for retained "
             "teacher complexes (default: /root/bgen_shards)",
    )
    ap.add_argument(
        "--structure-dir",
        help=(
            "copy each refolded target complex here before deleting temporary "
            "BoltzGen outputs"
        ),
    )
    ap.add_argument(
        "--require-structures",
        action="store_true",
        help="refuse the exact result unless every sequence has a retained complex",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--diagnostic-seed",
        type=int,
        help=(
            "seed every BoltzGen pipeline subprocess for controlled parity A/B "
            "tests; never set by live scoring"
        ),
    )
    a = ap.parse_args()
    if a.diagnostic_seed is not None and a.diagnostic_seed < 0:
        raise SystemExit("--diagnostic-seed must be non-negative")

    # Accept every shape we produce: a bare list, {"sequences": [...]} with
    # plain strings, or {"cases": [{"sequence": ...}]} from instrument_check.
    # Mixing them silently raised "string indices must be integers".
    raw = json.load(open(a.sequences))
    items = raw if isinstance(raw, list) else (raw.get("cases") or raw.get("sequences") or [])
    seqs = [it["sequence"] if isinstance(it, dict) else it for it in items]
    seqs = [s for s in seqs if isinstance(s, str) and s]
    if not seqs:
        raise SystemExit(f"no sequences parsed from {a.sequences}")

    import yaml
    raw_cfg = yaml.safe_load(open(a.config))
    subnet_config = _flatten_config(raw_cfg, a.target)

    gpus = [g.strip() for g in a.gpus.split(",") if g.strip()]
    n_shards = max(1, len(gpus) * a.shards_per_gpu)
    # round-robin: sequences are near-uniform cost here, and round-robin keeps
    # the assignment deterministic so a rerun reproduces the same partition
    buckets = [[] for _ in range(n_shards)]
    for i, s in enumerate(seqs):
        buckets[i % n_shards].append(s)

    work_root = os.path.abspath(a.work_root)
    os.makedirs(work_root, exist_ok=True)
    observation_id = str(uuid.uuid4())
    jobs = [(i, gpus[i % len(gpus)], b, subnet_config,
             os.path.join(work_root, f"out{i}.json"), a.batch_size,
             a.keep_work_dir, work_root, a.structure_dir, observation_id,
             a.diagnostic_seed)
            for i, b in enumerate(buckets) if b]
    print(f"{len(seqs)} sequences over {len(jobs)} shards on GPUs {gpus}", flush=True)

    started = time.time()
    ctx = mp.get_context("spawn")   # never fork a CUDA process
    with ctx.Pool(len(jobs)) as pool:
        results = pool.map(_worker, jobs)
    wall = time.time() - started

    # components is {uid: {sequence: {target: metrics}}} and EVERY shard uses
    # uid 0, so a plain dict.update() replaces the previous shard's entry
    # wholesale and silently drops its sequences -- 3 of 6 vanished before this
    # was caught. Merge at the SEQUENCE level instead.
    merged = {}
    merged_structures = {}
    for path, n, secs in results:
        d = json.load(open(path))
        for uid, by_seq in d["components"].items():
            merged.setdefault(uid, {}).update(by_seq)
        for sequence, by_target in d.get("structures", {}).items():
            merged_structures.setdefault(sequence, {}).update(by_target)
        print(f"  shard {d['shard']} gpu {d['gpu']}: {n} seqs in {secs:.1f}s "
              f"({secs/max(n,1):.1f}s/seq)", flush=True)
    n_out = sum(len(v) for v in merged.values())
    print(f"TOTAL {len(seqs)} sequences in {wall:.1f}s "
          f"({wall/max(len(seqs),1):.1f}s per sequence wall-clock)", flush=True)
    if n_out != len(seqs):
        raise SystemExit(f"MERGE LOST SEQUENCES: submitted {len(seqs)}, "
                         f"merged {n_out}. Refusing to write a partial result.")
    if a.require_structures and len(merged_structures) != len(seqs):
        raise SystemExit(
            f"STRUCTURE CAPTURE LOST COMPLEXES: submitted {len(seqs)}, retained "
            f"{len(merged_structures)}. Refusing to write an incomplete result."
        )
    json.dump({"target": a.target, "n": len(seqs), "wall_seconds": wall,
               "observation_id": observation_id,
               "diagnostic_seed": a.diagnostic_seed,
               "predict_batch_size": a.batch_size,
               "per_nanobody_components": merged,
               "structures": merged_structures}, open(a.out, "w"),
              indent=2, default=str)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
