#!/usr/bin/env python3
"""Load the exact local ESMFold2-Fast stack before launching long runs."""

from __future__ import annotations

import gc
import json
from pathlib import Path
import sys

import torch


STATUS = Path("/root/easynano-run/output/model-smoke.json")


def main() -> int:
    try:
        from easynano.config import NUM_RES_TYPES
        from easynano.design import init_soft_logits, load_fast_model, soft_to_hard_seq
        from easynano.transformers_compat import differentiable_trunk
        from esmscore._complex import build_complex_features

        model = load_fast_model()
        features = {
            key: value.to("cuda") if isinstance(value, torch.Tensor) else value
            for key, value in build_complex_features("ACDEFGHIK", "LMNPQRSTV").items()
            if not key.startswith("_")
        }
        features["res_type"] = torch.nn.functional.one_hot(
            features["res_type"].long(), num_classes=NUM_RES_TYPES,
        ).float().detach().requires_grad_(True)
        output = differentiable_trunk(model, features, num_loops=0)
        objective = output["distogram_logits"].float().mean()
        objective.backward()
        if features["res_type"].grad is None:
            raise RuntimeError("ESMFold2 trunk is not differentiable from res_type")
        anchored = init_soft_logits("A#A", "ACA", pin_cys_in_cdr=True)
        if soft_to_hard_seq(anchored) != "ACA":
            raise RuntimeError("native CDR cysteine was not preserved at initialization")
        checkpoint_layers = sum(
            bool(getattr(module, "gradient_checkpointing", False)) and module.training
            for module in model.modules()
        )
        if checkpoint_layers == 0:
            raise RuntimeError("no eval-safe ESMFold2 gradient checkpoint layers enabled")
        record = {
            "stage": "complete",
            "transformers_class": type(model).__name__,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "device": str(next(model.parameters()).device),
            "distogram_shape": list(output["distogram_logits"].shape),
            "res_type_gradient": True,
            "native_cdr_cysteine_preserved": True,
            "gradient_checkpoint_layers": checkpoint_layers,
        }
        del model
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as error:
        record = {"stage": "failed", "error_class": type(error).__name__}
    STATUS.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return 0 if record["stage"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
