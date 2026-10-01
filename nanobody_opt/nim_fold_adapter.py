"""Convert NIM Boltz-2 output into the fold artifacts ``Analyze`` consumes.

This is the boltzgen-side half of the backend swap.  ``nim_backend`` returns
raw CIF text plus the mapped confidence scores; this module turns those into
the three files the existing analysis step reads, so PLIP / delta-SASA /
liability / dense-rank-sum stay byte-for-byte the upstream code:

  * ``intermediate_designs/fold_out_npz/<id>.npz``
  * ``intermediate_designs/refold_cif/<id>.cif``
  * ``intermediate_designs/refold_analyze_cif/<id>.cif``

The analysis step (``boltzgen.task.analyze.analyze.Analyze.compute_metrics``)
reads, from the fold npz, only ``coords`` (per diffusion sample, atom14
layout), ``res_type`` (validated against the design-stage features) and one
scalar per ``boltzgen.data.const.eval_keys_confidence`` entry.  This module
reproduces exactly those keys and nothing else.

Runs on the fast-score host (the environment that already imports
``torch``/``hydra``/``boltzgen``), NOT on the NIM box.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np


def featurize_nim_cif(
    cif_text: str,
    data_module: Any,
    sample_id: str,
    tmp_dir: str | Path,
) -> dict:
    """Featurize one NIM CIF with the analysis step's own data module.

    ``data_module`` is the ``FromGeneratedDataModule`` the ``Analyze`` task was
    built with (``Analyze.data``).  Its ``get_feat`` parses a CIF, tokenizes it,
    forces chain B (``asym_id == 1``) to be the design chain, and returns the
    atom14 feature dict whose ``coords``/``res_type`` ordering is identical to
    the one the analysis step reads from ``fold_out_npz``.  Reusing that module
    is what keeps the atom ordering and fake-atom padding identical to the
    design stage; a hand-rolled featurization would drift here.
    """
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cif_path = tmp_dir / f"{sample_id}.cif"
    cif_path.write_text(cif_text)
    return data_module.get_feat(cif_path)


def build_fold_npz(
    coords_per_sample: Sequence[np.ndarray],
    res_type: np.ndarray,
    confidence_per_sample: Sequence[dict[str, float | None]],
) -> dict[str, np.ndarray]:
    """Assemble a fold-out npz in the shape ``FoldingWriter`` would write.

    ``coords_per_sample`` is one atom14 ``[n_atoms, 3]`` array per diffusion
    sample; ``res_type`` is the design-stage token one-hot ``[1, n_tokens, 21]``;
    ``confidence_per_sample`` maps BoltzGen confidence keys to one scalar per
    sample (PAE-derived keys are None and are omitted, matching the writer's
    ``if k in const.eval_keys`` filter).
    """
    n_samples = len(coords_per_sample)
    coords = np.stack([np.asarray(c, dtype=np.float32) for c in coords_per_sample])
    out: dict[str, np.ndarray] = {
        "coords": coords,
        "res_type": np.asarray(res_type),
    }
    for key in confidence_per_sample[0]:
        values = [c.get(key) for c in confidence_per_sample]
        if any(v is None for v in values):
            continue  # PAE-derived keys: not produced by NIM, omitted
        out[key] = np.asarray(values, dtype=np.float32)
    return out


def writer_sample_index(confidence_per_sample: Sequence[dict]) -> int:
    """Replicate the folding writer's pose rule: argmax(0.8*iptm + 0.2*ptm)."""
    best, best_i = None, 0
    for i, c in enumerate(confidence_per_sample):
        iptm, ptm = c.get("iptm"), c.get("ptm")
        if iptm is None or ptm is None:
            continue
        score = 0.8 * iptm + 0.2 * ptm
        if best is None or score > best:
            best, best_i = score, i
    return best_i


def analyze_sample_index(confidence_per_sample: Sequence[dict]) -> int:
    """Replicate the analysis step's pose rule:
    argmax(0.8*design_to_target_iptm + 0.2*design_ptm)."""
    best, best_i = None, 0
    for i, c in enumerate(confidence_per_sample):
        dtti, dptm = c.get("design_to_target_iptm"), c.get("design_ptm")
        if dtti is None or dptm is None:
            continue
        score = 0.8 * dtti + 0.2 * dptm
        if best is None or score > best:
            best, best_i = score, i
    return best_i


def write_nim_fold_artifacts(
    designs_dir: str | Path,
    record_id: str,
    nim_cif_texts: Sequence[str],
    confidence_per_sample: Sequence[dict],
    data_module: Any,
    feat_res_type: np.ndarray,
    tmp_dir: str | Path,
) -> dict[str, str]:
    """Write fold_out_npz / refold_cif / refold_analyze_cif for one record.

    ``record_id`` is ``f"{seq_hash(design_seq)}_{target}_input"`` (the id the
    analysis step addresses).  ``nim_cif_texts`` and ``confidence_per_sample``
    are the per-diffusion-sample outputs of ``nim_backend.predict_nanobody``.

    Returns the three written paths.
    """
    designs_dir = Path(designs_dir)
    fold_dir = designs_dir / "fold_out_npz"
    refold_dir = designs_dir / "refold_cif"
    analyze_dir = designs_dir / "refold_analyze_cif"
    for d in (fold_dir, refold_dir, analyze_dir):
        d.mkdir(parents=True, exist_ok=True)

    coords_per_sample = [
        featurize_nim_cif(cif, data_module, f"{record_id}_{i}", tmp_dir)["coords"][0]
        for i, cif in enumerate(nim_cif_texts)
    ]
    fold = build_fold_npz(coords_per_sample, feat_res_type, confidence_per_sample)
    npz_path = fold_dir / f"{record_id}.npz"
    np.savez_compressed(npz_path, **fold)

    w_idx = writer_sample_index(confidence_per_sample)
    a_idx = analyze_sample_index(confidence_per_sample)
    refold_cif_path = refold_dir / f"{record_id}.cif"
    refold_cif_path.write_text(nim_cif_texts[w_idx])
    analyze_cif_path = analyze_dir / f"{record_id}.cif"
    analyze_cif_path.write_text(nim_cif_texts[a_idx])

    return {
        "fold_npz": str(npz_path),
        "refold_cif": str(refold_cif_path),
        "refold_analyze_cif": str(analyze_cif_path),
    }
