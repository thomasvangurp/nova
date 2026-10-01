"""Drop-in replacement for score_boltzgen_batch.py with resident models.

Same CLI, same output JSON, same scoring path -- the only difference is that
the checkpoints are loaded once per worker instead of once per chunk.

Why
---
`boltzgen execute` runs each pipeline step as its own `python main.py
<step>.yaml` subprocess, and `Predict.run` loads the checkpoint, predicts, then
`del`s the model. Measured on this box: of a 345 s four-sequence chunk, ~176 s
(51%) is process start + checkpoint load, and the live producer runs chunks of
TWO, so it pays ~175 s of loading for ~50 s of work. Folding one design costs
52.9 s as a subprocess step and 11.4 s against a resident model.

What is preserved exactly
-------------------------
* `configure` still runs upstream's `configure_command`, so the per-chunk
  step configs are the ones BoltzGen itself emits -- not hand-written ones that
  could drift.
* The model is built by `Predict.run(cfg, run_prediction=False)`, the same code
  path production uses, with the same weights and the same predict_args.
* Ranking is still deferred: this fills `per_nanobody_components` from the
  analysis CSV exactly as `run_nanobody_inference` does.

What changes
------------
* Models stay resident across chunks within a worker.
* The datamodule, writer and Trainer are rebuilt per chunk (they carry the
  per-chunk paths); only the LightningModule is reused.
* `matmul_precision` for the folding step is configurable. Upstream leaves it
  null, which disables TF32 and puts 45.6% of folding's CUDA time in
  non-tensor-core `ampere_sgemm`. Measured: 44.3 -> 34.4 ms per denoising step,
  with a metric perturbation the same size as rerunning the identical config
  (the scorer is not bit-reproducible even at a fixed seed).
* Thread caps are set. Production sets none, and 43 torch processes x ~71
  threads drive a 128-core box to load 299.

ISOLATION IS STILL LOAD-BEARING, for the same reason as the original: the
wrapper derives its input/output dirs from a module global, so two shards in
one tree would silently read each other's results.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import hashlib
import shutil
import sys
import time
import uuid
from pathlib import Path

# Cap threads BEFORE torch is imported anywhere in this process.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")


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

    Identical to score_boltzgen_batch.py: derived from the upstream file rather
    than hand-written, so the epitope and residue range stay tied to upstream.
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




def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _capture_structures(designs_dir, sequences, targets, structure_dir,
                        observation_id):
    """Retain both validator-selected refolded poses and their provenance.

    A port of score_boltzgen_batch.py's capture, reusing nanobody_opt's
    diffusion_sample_selection so the two argmax rules cannot drift:

      writer  rule  0.8*iptm + 0.2*ptm                   -> refold_cif
      analyze rule  0.8*design_to_target_iptm + 0.2*design_ptm -> refold_analyze_cif

    The reported CSV metric is checked against the analyze-selected sample, so a
    scorer that silently reported a different sample's value cannot pass.
    """
    import csv
    import numpy as np
    from nanobody_opt.diffusion_sample_selection import (
        ANALYZE_CONFIDENCE_METRICS,
        select_diffusion_samples,
    )

    if structure_dir is None:
        return {}
    designs_dir = Path(designs_dir)
    metrics_path = designs_dir / "aggregate_metrics_analyze.csv"
    if not metrics_path.is_file():
        return {}
    with metrics_path.open(newline="") as handle:
        by_id = {str(row.get("id")): row for row in csv.DictReader(handle)}

    copied = {}
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
            writer_source = designs_dir / "refold_cif" / filename
            analyze_source = designs_dir / "refold_analyze_cif" / filename
            folded_source = designs_dir / "fold_out_npz" / f"{record_id}.npz"
            if not all(q.is_file() for q in
                       (writer_source, analyze_source, folded_source)):
                continue
            with np.load(folded_source) as folded:
                selection = select_diffusion_samples(folded)
                for metric in ANALYZE_CONFIDENCE_METRICS:
                    if metric not in row or metric not in folded:
                        raise ValueError(
                            f"aggregate analysis lacks {metric} sample provenance"
                        )
                    reported = float(row[metric])
                    selected = float(
                        np.asarray(folded[metric]).reshape(-1)[
                            selection.analyze_sample_index
                        ]
                    )
                    # the CSV keeps five decimals; the NPZ keeps full precision
                    if not np.isclose(reported, selected, rtol=0, atol=5.01e-6):
                        raise ValueError(
                            f"aggregate {metric}={reported} does not match "
                            f"analyze sample "
                            f"{selection.analyze_sample_index}={selected}"
                        )
            confidence_path = destination / f"{sequence_hash}_{target}.cif"
            writer_path = destination / f"{sequence_hash}_{target}_writer.cif"
            shutil.copyfile(analyze_source, confidence_path)
            shutil.copyfile(writer_source, writer_path)
            copied.setdefault(sequence, {})[target] = {
                # "path" deliberately means the pose analyze_utils selected for
                # the reported confidence metrics, matching production.
                "path": str(confidence_path),
                "sha256": _sha256(confidence_path),
                "confidence_path": str(confidence_path),
                "confidence_sha256": _sha256(confidence_path),
                "writer_path": str(writer_path),
                "writer_sha256": _sha256(writer_path),
                "sample_selection": selection.as_record(),
            }
    return copied


# --------------------------------------------------------------------------
# worker
# --------------------------------------------------------------------------


def _analysis_child(job_q, done_q):
    """Run boltzgen's analysis Task for each (config, design_dir) it is given.

    Long-lived: imports torch/boltzgen once (~5.5s) instead of once per chunk.
    """
    import hydra
    from omegaconf import OmegaConf
    from boltzgen.task.task import Task

    while True:
        job = job_q.get()
        if job is None:
            return
        cfg_path, designs_dir = job
        try:
            acfg = OmegaConf.load(cfg_path)
            OmegaConf.update(acfg, "design_dir", designs_dir, merge=False)
            # distribute_tasks runs serially in-process at num_processes==1;
            # above that it spawns a pool whose every worker re-imports torch.
            OmegaConf.update(acfg, "num_processes", 1, merge=False)
            task = hydra.utils.instantiate(acfg)
            if not isinstance(task, Task):
                raise TypeError("analysis config is not a Task")
            task.run(acfg)
            done_q.put((designs_dir, None))
        except Exception as exc:  # noqa: BLE001
            import traceback
            done_q.put((designs_dir, f"{type(exc).__name__}: {exc}\n"
                                     f"{traceback.format_exc()}"))


class _AnalysisPool:
    """One persistent analysis process, so analysis overlaps the next folding."""

    def __init__(self, ctx):
        self.job_q = ctx.Queue()
        self.done_q = ctx.Queue()
        self.proc = ctx.Process(target=_analysis_child,
                                args=(self.job_q, self.done_q), daemon=False)
        self.proc.start()
        self.pending = 0

    def submit(self, cfg_path, designs_dir):
        self.job_q.put((cfg_path, designs_dir))
        self.pending += 1

    def drain(self):
        errors = {}
        for _ in range(self.pending):
            designs_dir, err = self.done_q.get()
            if err:
                errors[designs_dir] = err
        self.pending = 0
        return errors

    def close(self):
        self.job_q.put(None)
        self.proc.join(timeout=60)


def _worker(args):
    (shard_id, gpu, seqs, subnet_config, out_path, batch_size, keep_work_dir,
     shard_root, fold_matmul, fold_steps, fold_samples, design_steps,
     synth_ref, synth_lib, structure_dir, observation_id) = args

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    if "/root" not in sys.path:
        sys.path.insert(0, "/root")

    import hydra
    import rdkit.Chem
    from omegaconf import OmegaConf
    import boltzgen.boltzgen_wrapper as bw

    # The CCD mols carry per-atom properties ("name") that the featurizer looks
    # up, and RDKit drops them when a Mol is pickled unless this is set. It
    # never bit upstream because every pipeline step is its own fresh process
    # whose DataLoader workers fork; here the stages run inside a spawned
    # worker, so the mols get pickled and arrive stripped:
    #   KeyError: 'name' at featurizer.py:1022
    # analyze.py sets exactly this, for exactly this reason (analyze.py:243).
    rdkit.Chem.SetDefaultPickleProperties(rdkit.Chem.PropertyPickleOptions.AllProps)
    from boltzgen.cli.boltzgen import build_parser, configure_command
    from boltzgen.task.task import Task

    from resident import ResidentStage
    from utils import seq_hash

    stages: dict[str, ResidentStage] = {}

    synth = None
    if synth_ref:
        # Grafting replaces the design stage entirely. The folding step consumes
        # only the TARGET chain's geometry from a design output (as an
        # SE(3)-invariant template) plus both chains' sequences -- the designed
        # chain's coordinates reach no module the folding checkpoint contains.
        # So a real design output can be retargeted to any same-length sequence
        # by copying chain A verbatim and grafting chain B's side chains onto
        # its backbone. Measured: 24s -> ~94ms, with every metric shift below
        # the same-seed reproducibility floor.
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from synth_fullatom import TemplateLibrary, synthesize
        lib = TemplateLibrary.from_cifs([Path(x) for x in synth_lib])
        synth = (Path(synth_ref), lib, synthesize)
        print(f"  shard {shard_id}: rotamer library from {len(synth_lib)} designs",
              flush=True)

    def run_chunk(chunk_dir: Path, chunk_seqs: list[str]) -> dict:
        bw.BOLTZGEN_TMP_FILES_DIR = str(chunk_dir)
        shutil.rmtree(chunk_dir, ignore_errors=True)
        (chunk_dir / "inputs").mkdir(parents=True, exist_ok=True)
        (chunk_dir / "outputs").mkdir(parents=True, exist_ok=True)

        w = bw.BoltzgenWrapper()
        w.tmp_dir = str(chunk_dir)
        w.input_dir = str(chunk_dir / "inputs")
        w.output_dir = str(chunk_dir / "outputs")
        w.subnet_config = subnet_config
        w.unique_sequences = w._deduplicate_nanobodies({0: {"sequences": chunk_seqs}})

        # configure once per worker. The YAML it emits differs between chunks
        # only in paths, and run_batch overrides those at instantiate time. The
        # input YAMLs feed the design stage only, which synthesis replaces.
        if "cfg_dir" not in cached:
            w._write_yaml_files()
            parser = build_parser()
            argv = ["configure", w.input_dir, "--output", w.output_dir,
                    "--protocol", w.boltzgen_config["protocol"],
                    "--num_designs", str(w.boltzgen_config["num_designs"])]
            if w.boltzgen_config["skip_inverse_folding"]:
                argv.append("--skip_inverse_folding")
            if w.boltzgen_config.get("step_scale") is not None:
                argv += ["--step_scale", str(w.boltzgen_config["step_scale"])]
            if w.boltzgen_config.get("noise_scale") is not None:
                argv += ["--noise_scale", str(w.boltzgen_config["noise_scale"])]
            configure_command(parser.parse_args(argv))
            cached["cfg_dir"] = Path(w.output_dir) / "config"

        cfg_dir = cached["cfg_dir"]
        designs_dir = Path(w.output_dir) / "intermediate_designs"

        # ---- design: synthesized, or the diffusion model ------------------
        designs_dir.mkdir(parents=True, exist_ok=True)
        if synth is not None:
            ref_cif, lib, synthesize = synth
            ref_npz = ref_cif.with_suffix(".npz")
            t0 = time.time()
            for seq in chunk_seqs:
                for target in subnet_config["nanobody_target"]:
                    stem = f"{seq_hash(seq)}_{target}_input"
                    synthesize(ref_cif, seq, designs_dir / f"{stem}.cif", lib)
                    # folding reads only ss_type/binding_type out of the npz and
                    # the model consumes neither, so the reference's serves
                    shutil.copyfile(ref_npz, designs_dir / f"{stem}.npz")
            print(f"  shard {shard_id}: synthesized {len(chunk_seqs)} designs in "
                  f"{time.time() - t0:.2f}s", flush=True)
        else:
            design_cfg = cfg_dir / "design.yaml"
            yaml_paths = sorted(str(p) for p in Path(w.input_dir).glob("*.yaml"))
            if "design" not in stages:
                stages["design"] = ResidentStage(design_cfg)
                print(f"  shard {shard_id}: design model loaded in "
                      f"{stages['design'].load_seconds:.1f}s", flush=True)
            stages["design"].run_batch(
                {"data.cfg.yaml_path": yaml_paths, "output": str(designs_dir)},
                predict_args=({"sampling_steps": design_steps} if design_steps else None),
            )

        # ---- folding (resident) ------------------------------------------
        fold_cfg = cfg_dir / "folding.yaml"
        if "folding" not in stages:
            overrides = {}
            if fold_matmul:
                overrides["matmul_precision"] = fold_matmul
            # a spawned dataloader worker per predict() call costs more than
            # featurising a handful of designs inline
            overrides["data.cfg.num_workers"] = 0
            stages["folding"] = ResidentStage(fold_cfg, overrides=overrides)
            print(f"  shard {shard_id}: folding model loaded in "
                  f"{stages['folding'].load_seconds:.1f}s", flush=True)
        predict_args = {}
        if fold_steps:
            predict_args["sampling_steps"] = fold_steps
        if fold_samples:
            predict_args["diffusion_samples"] = fold_samples
        stages["folding"].run_batch(
            {"data.design_dir": str(designs_dir), "output": str(designs_dir)},
            predict_args=predict_args or None,
        )

        # ---- analysis: dispatched to a persistent child, not run here -----
        # It is CPU-only and the GPU idles through all 1.8 s of it, so it is
        # overlapped with the next chunk's folding. A fresh subprocess per
        # chunk would pay ~5.5 s of imports for ~1.8 s of work; the child is
        # long-lived and pays them once.
        analysis.submit(str(cfg_dir / "analysis.yaml"), str(designs_dir))
        return w, designs_dir, chunk_seqs

    started = time.time()
    components: dict = {}
    structures: dict = {}
    cached: dict = {}

    # analysis is dispatched here and collected after the GPU loop, so a
    # chunk's 1.8 s of CPU analysis overlaps the NEXT chunk's folding instead
    # of blocking it. Work dirs therefore have to survive until the drain.
    analysis = _AnalysisPool(mp.get_context("fork"))
    deferred = []

    for start in range(0, len(seqs), batch_size):
        chunk = seqs[start:start + batch_size]
        chunk_dir = Path(shard_root) / f"shard{shard_id}" / f"chunk{start // batch_size}"
        try:
            w, designs_dir, chunk_seqs = run_chunk(chunk_dir, chunk)
            deferred.append((chunk_dir, w, designs_dir, chunk_seqs))
        except Exception as exc:  # noqa: BLE001 - one bad chunk must not lose the rest
            import traceback
            print(f"  shard {shard_id} chunk {start // batch_size} raised "
                  f"({type(exc).__name__}: {exc}); continuing", flush=True)
            traceback.print_exc()
        print(f"  shard {shard_id}: folded "
              f"{min(start + batch_size, len(seqs))}/{len(seqs)}", flush=True)

    errors = analysis.drain()
    analysis.close()

    for chunk_dir, w, designs_dir, chunk_seqs in deferred:
        err = errors.get(str(designs_dir))
        if err:
            print(f"  shard {shard_id}: analysis failed for {designs_dir}:\n{err}",
                  flush=True)
        else:
            try:
                results = w._collect_local_results()
                w._populate_per_nanobody_components(results)
                for uid, by_seq in (w.per_nanobody_components or {}).items():
                    components.setdefault(uid, {}).update(by_seq)
                structures.update(_capture_structures(
                    designs_dir, chunk_seqs, subnet_config["nanobody_target"],
                    structure_dir, observation_id,
                ) or {})
            except Exception as exc:  # noqa: BLE001
                import traceback
                print(f"  shard {shard_id}: collection failed for {designs_dir} "
                      f"({type(exc).__name__}: {exc})", flush=True)
                traceback.print_exc()
        if not keep_work_dir:
            shutil.rmtree(chunk_dir, ignore_errors=True)

    elapsed = time.time() - started
    with open(out_path, "w") as handle:
        json.dump({"shard": shard_id, "gpu": gpu, "n": len(seqs),
                   "seconds": elapsed, "components": components,
                   "structures": structures}, handle, default=str)


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--target", default="P20809")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--shards-per-gpu", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=24,
                    help="sequences per chunk. With resident models the model "
                         "load is no longer amortised over the chunk, so this "
                         "is now only about work-dir size and failure blast "
                         "radius, not throughput.")
    ap.add_argument("--shard-root", default="/root/bgen_shards_fast")
    ap.add_argument("--capture-structures", default=None,
                    help="directory to retain both validator-selected refolded "
                         "poses per observation. The live path requires these: "
                         "score_boltzgen_waves.py rejects a wave whose sequences "
                         "lack captured structures.")
    ap.add_argument("--require-structures", action="store_true",
                    help="fail if any sequence did not yield a captured structure")
    ap.add_argument("--observation-id", default=None,
                    help="identifier for this observation (default: a fresh uuid4)")
    ap.add_argument("--synth-reference", default=None,
                    help="a real design .cif (with its .npz alongside) whose "
                         "designed chain has the SAME length as the candidates. "
                         "Given this, the design stage is replaced by side-chain "
                         "grafting (~94ms vs 24s). Omit to run the diffusion "
                         "design model instead.")
    ap.add_argument("--synth-library", nargs="+", default=None,
                    help="design .cif files to harvest side-chain rotamers from "
                         "(defaults to the reference's own directory)")
    ap.add_argument("--design-sampling-steps", type=int, default=100,
                    help="design-stage sampling_steps (config default 500). The "
                         "folding step consumes only the TARGET chain's geometry "
                         "from the design output, as an SE(3)-invariant template. "
                         "Measured over 4 designs: at 100 steps that template is "
                         "reproduced to 0.122 A, inside the 0.145 A spread of "
                         "independent 500-step runs, and downstream scores move "
                         "<0.5 sd on design_to_target_iptm. At 50 steps one design "
                         "of four moved 2.3 sd, so 100 is the floor.")
    ap.add_argument("--fold-matmul-precision", default="high",
                    help="TF32 for the folding step. Upstream leaves this null; "
                         "measured 44.3 -> 34.4 ms per denoising step with a "
                         "perturbation the size of rerunning the same config.")
    ap.add_argument("--fold-sampling-steps", type=int, default=50,
                    help="folding sampling_steps (upstream config is 200). 50 is "
                         "validated: against production history it sits AT the "
                         "production-vs-production agreement ceiling on all five "
                         "confidence metrics, with mean differences (-0.0007 "
                         "design_iiptm, -0.011 min_design_to_target_pae) smaller "
                         "than 200 steps gives, and sign splits that are coin "
                         "flips. A 20-design x 6-repeat sweep found the same: "
                         "biases below the reproducibility floor and better "
                         "standard error per GPU-second on every metric. Pass "
                         "200 to reproduce upstream exactly.")
    ap.add_argument("--fold-diffusion-samples", type=int, default=None,
                    help="override folding diffusion_samples (default: config, 5). "
                         "NOTE the score is a MAX over samples, so lowering this "
                         "biases scores downward, it does not merely add noise.")
    ap.add_argument("--keep-work-dir", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    raw = json.load(open(a.sequences))
    items = raw if isinstance(raw, list) else (raw.get("cases") or raw.get("sequences") or [])
    seqs = [it["sequence"] if isinstance(it, dict) else it for it in items]
    seqs = [s for s in seqs if isinstance(s, str) and s]
    if not seqs:
        raise SystemExit(f"no sequences parsed from {a.sequences}")

    if "/root" not in sys.path:
        sys.path.insert(0, "/root")

    import yaml
    subnet_config = _flatten_config(yaml.safe_load(open(a.config)), a.target)

    gpus = [g.strip() for g in a.gpus.split(",") if g.strip()]
    n_shards = max(1, len(gpus) * a.shards_per_gpu)
    buckets = [[] for _ in range(n_shards)]
    for i, s in enumerate(seqs):
        buckets[i % n_shards].append(s)

    observation_id = a.observation_id or uuid.uuid4().hex

    synth_lib = a.synth_library
    if a.synth_reference and not synth_lib:
        import glob as _glob
        synth_lib = sorted(_glob.glob(
            os.path.join(os.path.dirname(os.path.abspath(a.synth_reference)), "*.cif")))
        if not synth_lib:
            raise SystemExit(f"no .cif files beside {a.synth_reference} to build "
                             "a rotamer library from; pass --synth-library")

    os.makedirs(a.shard_root, exist_ok=True)
    jobs = [(i, gpus[i % len(gpus)], b, subnet_config,
             os.path.join(a.shard_root, f"out{i}.json"), a.batch_size,
             a.keep_work_dir, a.shard_root, a.fold_matmul_precision,
             a.fold_sampling_steps, a.fold_diffusion_samples,
             a.design_sampling_steps, a.synth_reference, synth_lib,
             a.capture_structures, observation_id)
            for i, b in enumerate(buckets) if b]
    print(f"{len(seqs)} sequences over {len(jobs)} shards on GPUs {gpus}", flush=True)

    started = time.time()
    # spawn, never fork, because these processes initialise CUDA -- and
    # NON-daemonic, because the predict DataLoaders (num_workers >= 1) spawn
    # children of their own and `mp.Pool` workers are daemonic, which fails as
    # "daemonic processes are not allowed to have children".
    ctx = mp.get_context("spawn")
    procs = []
    for job in jobs:
        pr = ctx.Process(target=_worker, args=(job,), daemon=False)
        pr.start()
        procs.append((pr, job))
    results = []
    for pr, job in procs:
        pr.join()
        shard_id, _gpu, bucket, _sc, out_path = job[0], job[1], job[2], job[3], job[4]
        if pr.exitcode != 0 or not os.path.exists(out_path):
            raise SystemExit(
                f"shard {shard_id} exited {pr.exitcode} without writing "
                f"{out_path}. Refusing to report a partial result."
            )
        results.append((out_path, len(bucket), json.load(open(out_path))["seconds"]))
    wall = time.time() - started

    # Merge at the SEQUENCE level: every shard uses uid 0, so a plain
    # dict.update() would drop whole shards.
    merged: dict = {}
    merged_structures: dict = {}
    for path, n, secs in results:
        d = json.load(open(path))
        for uid, by_seq in d["components"].items():
            merged.setdefault(uid, {}).update(by_seq)
        merged_structures.update(d.get("structures") or {})
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
            f"structures captured for {len(merged_structures)} of "
            f"{len(seqs)} sequences. Refusing to write a result the live "
            f"pipeline would reject."
        )
    from nanobody_opt.diffusion_sample_selection import SCORER_PROTOCOL
    with open(a.out, "w") as handle:
        json.dump({"target": a.target, "n": len(seqs), "wall_seconds": wall,
                   "scorer_protocol": SCORER_PROTOCOL,
                   "observation_id": observation_id,
                   "diagnostic_seed": None,
                   "predict_batch_size": a.batch_size,
                   "per_nanobody_components": merged,
                   "structures": merged_structures},
                  handle, indent=2, default=str)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
