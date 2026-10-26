"""Keep BoltzGen's models resident on the GPU across batches.

Why
---
`boltzgen execute` runs each pipeline step as its own `python main.py
<step>.yaml` subprocess, and `Predict.run` loads the checkpoint, builds a
Trainer, predicts, then `del`s the model. Measured on this box for the
nanobody job: of a 345 s chunk, ~176 s (51%) is process start + checkpoint
load, paid again for every chunk -- and the live producer runs chunks of 2.

`Predict.run(cfg, run_prediction=False)` already stops after loading the model
and building the Trainer. Using that entry point rather than reimplementing
checkpoint loading means the resident model is built by exactly the production
code path.

Rebuilt per batch (cheap, carries per-batch state): the datamodule, the writer,
and the Trainer (its callback list holds the writer). NOT rebuilt: the
LightningModule.

Numerics are unchanged: same weights, same predict_args, same forward.
Measured: folding a single design drops from 52.9 s (subprocess step) to 13.9 s
resident, against ~12.4 s of pure CUDA time.

predict_args override
---------------------
`Boltz.predict_step` reads recycling_steps / sampling_steps / diffusion_samples
out of `self.predict_args` at call time, so a sweep over sampler budgets needs
one model load, not one per setting.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional

import hydra
from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from boltzgen.task.predict.predict import Predict
from boltzgen.utils.pipeline_progress_bar import PipelineProgressBar


class ResidentStage:
    """One BoltzGen predict stage (design or folding) with a resident model."""

    def __init__(self, config_path: str | Path, overrides: Optional[dict] = None):
        self.config_path = Path(config_path)
        cfg = OmegaConf.load(self.config_path)
        for k, v in (overrides or {}).items():
            OmegaConf.update(cfg, k, v, merge=False)
        self.base_cfg = cfg

        t0 = time.time()
        task = hydra.utils.instantiate(cfg)
        if not isinstance(task, Predict):
            raise TypeError(f"{self.config_path} is not a Predict task")
        task.run(cfg, run_prediction=False)
        self.task = task
        self.model = task.model_module
        self.load_seconds = time.time() - t0

    def run_batch(self, overrides: dict[str, Any],
                  predict_args: Optional[dict] = None,
                  seed: Optional[int] = None) -> float:
        """Predict one batch. `overrides` are dotted config keys.

        Returns wall seconds for the predict call only (model load excluded).
        """
        # Seed before constructing data: featurization may consume RNG too.
        if seed is not None:
            seed_all(seed)
        cfg = self.base_cfg.copy()
        for k, v in overrides.items():
            OmegaConf.update(cfg, k, v, merge=False)

        if predict_args:
            # mutated in place on the resident model; predict_step reads it per call
            self.model.predict_args.update(predict_args)

        data = hydra.utils.instantiate(cfg.data)
        writer = hydra.utils.instantiate(cfg.writer)

        trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True) or {}
        callbacks = [writer]
        if os.environ.get("BOLTZGEN_PIPELINE_STEP"):
            callbacks.append(PipelineProgressBar())
        trainer = Trainer(default_root_dir=str(cfg.output), strategy="auto",
                          callbacks=callbacks, **trainer_kwargs)

        t0 = time.time()
        trainer.predict(self.model, datamodule=data, return_predictions=False)
        return time.time() - t0


def seed_all(seed: int) -> None:
    import random
    import numpy as np
    import torch

    from pytorch_lightning import seed_everything

    seed_everything(seed, workers=True, verbose=False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _main():
    import argparse, json

    ap = argparse.ArgumentParser(description="benchmark a resident stage")
    ap.add_argument("config")
    ap.add_argument("--design-dirs", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--sampling-steps", type=int, default=None)
    ap.add_argument("--diffusion-samples", type=int, default=None)
    a = ap.parse_args()

    if a.seed is not None:
        seed_all(a.seed)

    pa = {}
    if a.sampling_steps is not None:
        pa["sampling_steps"] = a.sampling_steps
    if a.diffusion_samples is not None:
        pa["diffusion_samples"] = a.diffusion_samples

    stage = ResidentStage(a.config)
    print(f"model load: {stage.load_seconds:.1f}s", flush=True)
    for i, d in enumerate(a.design_dirs):
        secs = stage.run_batch({"data.design_dir": d, "output": d}, predict_args=pa)
        print(f"batch {i} ({d}): {secs:.1f}s", flush=True)


if __name__ == "__main__":
    _main()
