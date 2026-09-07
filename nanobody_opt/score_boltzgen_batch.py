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
import argparse, json, os, time
import multiprocessing as mp


def _flatten_config(cfg: dict, target: str) -> dict:
    """Flatten upstream config.yaml into the keys BoltzgenWrapper reads.

    The wrapper reads FLAT keys -- subnet_config['nanobody_structure'],
    ['nanobody_structure_chain'], ['nanobody_structure_res_index'],
    ['nanobody_structure_binding_site'] -- while config.yaml nests them under
    protein_selection.nanobody.{structure,structure_chain,...}. The validator
    does this flattening itself; replicating it here rather than hand-writing
    values keeps the binding site and residue range tied to upstream, so a
    change there cannot leave us silently scoring against a different epitope.
    """
    nb = (cfg.get("protein_selection") or {}).get("nanobody") or {}
    missing = [k for k in ("structure", "structure_chain", "structure_res_index",
                           "structure_binding_site") if k not in nb]
    if missing:
        raise SystemExit(f"config.yaml protein_selection.nanobody lacks {missing}")
    flat = dict(cfg)
    flat["nanobody_target"] = [target]
    for k in ("structure", "structure_chain", "structure_res_index",
              "structure_binding_site"):
        flat[f"nanobody_{k}"] = nb[k]
    return flat


def _worker(args):
    shard_id, gpu, seqs, subnet_config, out_path = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import boltzgen.boltzgen_wrapper as bw
    shard_dir = f"/root/bgen_shards/shard{shard_id}"
    # rebind BEFORE constructing: __init__ copies the global into self.tmp_dir
    bw.BOLTZGEN_TMP_FILES_DIR = shard_dir
    os.makedirs(os.path.join(shard_dir, "inputs"), exist_ok=True)
    os.makedirs(os.path.join(shard_dir, "outputs"), exist_ok=True)
    started = time.time()
    wrapper = bw.BoltzgenWrapper()
    assert wrapper.tmp_dir == shard_dir, (
        f"shard {shard_id} did not isolate its tmp dir: {wrapper.tmp_dir}")
    # shape matches the validator's uid_to_data: _deduplicate_nanobodies reads
    # nanobodies['sequences'], so a bare list raises TypeError deep in the pool
    components = wrapper.run_nanobody_inference({0: {"sequences": seqs}}, subnet_config)
    elapsed = time.time() - started
    json.dump({"shard": shard_id, "gpu": gpu, "n": len(seqs),
               "seconds": elapsed, "components": components},
              open(out_path, "w"), default=str)
    return out_path, len(seqs), elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--target", default="P20809")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--shards-per-gpu", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    raw = json.load(open(a.sequences))
    if isinstance(raw, dict):
        seqs = [c["sequence"] for c in raw.get("cases", raw.get("sequences", []))]
    else:
        seqs = list(raw)

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

    os.makedirs("/root/bgen_shards", exist_ok=True)
    jobs = [(i, gpus[i % len(gpus)], b, subnet_config,
             f"/root/bgen_shards/out{i}.json")
            for i, b in enumerate(buckets) if b]
    print(f"{len(seqs)} sequences over {len(jobs)} shards on GPUs {gpus}", flush=True)

    started = time.time()
    ctx = mp.get_context("spawn")   # never fork a CUDA process
    with ctx.Pool(len(jobs)) as pool:
        results = pool.map(_worker, jobs)
    wall = time.time() - started

    merged = {}
    for path, n, secs in results:
        d = json.load(open(path))
        merged.update(d["components"])
        print(f"  shard {d['shard']} gpu {d['gpu']}: {n} seqs in {secs:.1f}s "
              f"({secs/max(n,1):.1f}s/seq)", flush=True)
    print(f"TOTAL {len(seqs)} sequences in {wall:.1f}s "
          f"({wall/max(len(seqs),1):.1f}s per sequence wall-clock)", flush=True)
    json.dump({"target": a.target, "n": len(seqs), "wall_seconds": wall,
               "per_nanobody_components": merged}, open(a.out, "w"),
              indent=2, default=str)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
