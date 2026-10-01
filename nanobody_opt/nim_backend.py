"""NIM (Boltz-2) backend for the nanobody scorer's structure-prediction step.

This module replaces ONLY the structure-prediction call of the custom
``boltzgen`` scorer.  The metric-extraction and ranking code stays untouched;
this module's job is to produce the same intermediate artifacts (raw CIF
structures plus the model's per-structure confidence scores) that the
existing analysis step consumes.

Contract
--------
The scorer's structure prediction is the ``folding`` step.  In
``external_tools/fast_nanobody_scorer/fast_score_batch.py`` the resident
``folding`` ``ResidentStage`` runs the BoltzGen checkpoint and its
``FoldingWriter`` writes, per record ``<id>``:

  * ``intermediate_designs/fold_out_npz/<id>.npz``  -- per-diffusion-sample
    ``coords`` plus one scalar per ``boltzgen.data.const.eval_keys_confidence``
    (``iptm``, ``ptm``, ``design_iptm``, ``design_iiptm``,
    ``design_to_target_iptm``, ``design_ptm``, ``target_ptm``,
    ``ligand_iptm``, ``interaction_pae``, ``min_interaction_pae``,
    ``min_design_to_target_pae``, ...).
  * ``intermediate_designs/refold_cif/<id>.cif``            -- writer-selected pose.
  * ``intermediate_designs/refold_analyze_cif/<id>.cif``    -- analyze-selected pose.

Metric extraction is the ``analysis`` step (``boltzgen.task.analyze.analyze.Analyze``),
which reads the fold npz + the two refold CIFs + the design-stage features and
emits ``aggregate_metrics_analyze.csv``.  Nothing in this module reimplements
that; it only supplies the structure and confidence inputs it reads.

NIM endpoint (verified 2026-09-29)
----------------------------------
  * base URL  ``http://<host>:8000``
  * ``GET  /v1/health/ready`` -> 200
  * ``POST /biology/mit/boltz2/predict``

Request: ``{"polymers": [{"id": str, "molecule_type": "protein",
"sequence": str, "structural_templates": [{"structure": <cif>, "format":
"cif", "chain_id": str}]}, ...], "diffusion_samples": int,
"recycling_steps": int, "sampling_steps": int}``.  There is NO seed and NO
temperature field.

Response (verified) top-level keys:
  * ``structures``            : list of {format, name, source, structure=CIF text}
  * ``ptm_scores``            : list[float], one per diffusion sample (global ptm)
  * ``iptm_scores``           : list[float], one per sample (interface iptm)
  * ``protein_iptm_scores``   : list[float], one per sample (== iptm_scores)
  * ``confidence_scores``     : list[float], one per sample (0.8*iptm + 0.2*ptm)
  * ``complex_plddt_scores``  : list[float], one per sample
  * ``ligand_iptm_scores``    : list[float], one per sample (0.0 when no ligand)
  * ``chains_ptm_scores``     : list[float], one per CHAIN (self-iptm ~ chain ptm)
  * ``pair_chains_iptm_scores``: list[dict], one per sample, {chain_i: {chain_j:
    float}} pairwise chain iptm matrix (diagonal = self iptm, off-diagonal =
    interface iptm).
  * ``pae`` / ``pde``         : None.  The endpoint does NOT return the PAE
    matrix, so the two PAE-derived validator metrics (``interaction_pae``,
    ``min_design_to_target_pae``) cannot be produced by this backend.  This is
    a hard gap, not an implementation choice.

Confidence-key mapping (two-chain complex, polymer 0 = target, 1 = design)
--------------------------------------------------------------------------
BoltzGen ``eval_keys_confidence`` name  <- NIM field
  ``iptm``                 <- ``iptm_scores[sample]``
  ``protein_iptm``         <- ``protein_iptm_scores[sample]``
  ``ptm``                  <- ``ptm_scores[sample]``
  ``design_to_target_iptm``<- ``pair_chains_iptm_scores[sample]["1"]["0"]``
  ``design_iptm``          <- ``pair_chains_iptm_scores[sample]["1"]["0"]``
  ``design_iiptm``         <- ``pair_chains_iptm_scores[sample]["1"]["0"]``
                              (Boltz-2 has no separate iiptm; interface iptm is
                              the closest available quantity)
  ``target_ptm``           <- ``chains_ptm_scores[0]``
  ``design_ptm``           <- ``chains_ptm_scores[1]``
  ``ligand_iptm``          <- ``ligand_iptm_scores[sample]``
  ``interaction_pae``      <- None (not returned by NIM)
  ``min_interaction_pae``  <- None (not returned by NIM)
  ``min_design_to_target_pae`` <- None (not returned by NIM)

The module is stdlib-only so it runs on the NIM box and on any host that can
reach the NIM service; it must not import torch/hydra/biotite.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Iterable, Optional

DEFAULT_URL = "http://127.0.0.1:8000"
PREDICT_PATH = "/biology/mit/boltz2/predict"
HEALTH_PATH = "/v1/health/ready"

# Order mirrors boltzgen.data.const.eval_keys_confidence, with the three
# PAE-derived names present but always None for this backend.
BOLTZGEN_CONFIDENCE_KEYS = (
    "ligand_iptm",
    "interaction_pae",
    "min_interaction_pae",
    "min_design_to_target_pae",
    "iptm",
    "ptm",
    "protein_iptm",
    "design_iptm",
    "design_iiptm",
    "design_to_target_iptm",
    "target_ptm",
    "design_ptm",
)


class NimBoltz2Error(RuntimeError):
    """Raised for any NIM HTTP / schema failure."""


class NimBoltz2Client:
    def __init__(self, url: str = DEFAULT_URL, timeout: float = 600.0) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout

    def _request(self, path: str, payload: Optional[dict] = None) -> dict:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        req = urllib.request.Request(
            self.url + path, data=data, headers=headers,
            method="POST" if payload is not None else "GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise NimBoltz2Error(
                f"{path} -> HTTP {exc.code}: {exc.read(400)!r}"
            ) from exc
        except urllib.error.URLError as exc:
            raise NimBoltz2Error(f"{path} -> {exc.reason}") from exc

    def health(self) -> bool:
        return self._request(HEALTH_PATH).get("status") == "ready"

    def predict(self, polymers: list[dict], *, diffusion_samples: int = 2,
                recycling_steps: int = 3, sampling_steps: int = 50) -> dict:
        payload = {
            "polymers": polymers,
            "diffusion_samples": diffusion_samples,
            "recycling_steps": recycling_steps,
            "sampling_steps": sampling_steps,
        }
        return self._request(PREDICT_PATH, payload)


def build_nanobody_polymers(
    target_sequence: str,
    design_sequence: str,
    template_cif: str,
    *,
    template_chain_id: str = "A",
    target_id: str = "A",
    design_id: str = "B",
) -> list[dict]:
    """Build the ``polymers`` payload for one target + one design sequence.

    ``template_cif`` is the same reference structure the local scorer embeds as
    chain A in its input YAML (``subnet_config['nanobody_structure']``, e.g.
    the ``4O9H`` chain-A domain CIF at ``data/structures/4O9H.cif``).
    """
    return [
        {
            "id": target_id,
            "molecule_type": "protein",
            "sequence": target_sequence,
            "structural_templates": [
                {
                    "structure": template_cif,
                    "format": "cif",
                    "chain_id": template_chain_id,
                }
            ],
        },
        {
            "id": design_id,
            "molecule_type": "protein",
            "sequence": design_sequence,
        },
    ]


def response_structures(response: dict) -> list[str]:
    """Raw CIF text per diffusion sample, in response order."""
    structs = response.get("structures") or []
    return [s.get("structure", "") for s in structs]


def _scores(response: dict, key: str, index: int) -> Optional[float]:
    values = response.get(key)
    if not isinstance(values, list) or not values:
        return None
    if index >= len(values):
        return None
    value = values[index]
    return None if value is None else float(value)


def response_to_confidence(
    response: dict,
    *,
    sample: int = 0,
    target_chain: str = "0",
    design_chain: str = "1",
) -> dict[str, Optional[float]]:
    """Map one NIM diffusion sample to BoltzGen confidence-key names.

    Returns a dict keyed by ``BOLTZGEN_CONFIDENCE_KEYS``.  The three PAE-derived
    keys are always None because the endpoint returns ``pae: null``.
    """
    pair = response.get("pair_chains_iptm_scores") or []
    pair_iptm: dict[str, dict[str, float]] = (
        pair[sample] if sample < len(pair) else {}
    )

    def _pair(i: str, j: str) -> Optional[float]:
        row = pair_iptm.get(i)
        if isinstance(row, dict):
            value = row.get(j)
            return None if value is None else float(value)
        return None

    chains_ptm = response.get("chains_ptm_scores") or []
    design_to_target = _pair(design_chain, target_chain)

    return {
        "ligand_iptm": _scores(response, "ligand_iptm_scores", sample),
        "interaction_pae": None,
        "min_interaction_pae": None,
        "min_design_to_target_pae": None,
        "iptm": _scores(response, "iptm_scores", sample),
        "ptm": _scores(response, "ptm_scores", sample),
        "protein_iptm": _scores(response, "protein_iptm_scores", sample),
        "design_iptm": design_to_target,
        "design_iiptm": design_to_target,
        "design_to_target_iptm": design_to_target,
        "target_ptm": (chains_ptm[0] if chains_ptm else None),
        "design_ptm": (chains_ptm[1] if len(chains_ptm) > 1 else None),
    }


def predict_nanobody(
    client: NimBoltz2Client,
    target_sequence: str,
    design_sequence: str,
    template_cif: str,
    *,
    diffusion_samples: int = 2,
) -> dict:
    """One-shot convenience: predict and return the parsed NIM result.

    Return shape::

        {
          "elapsed_s": float,
          "confidence": [{boltzgen_key: float|None}, ...]  # one per sample
          "structures": [cif_text, ...]                   # one per sample
          "raw": {top_level_response_without_structures}
        }
    """
    t0 = time.time()
    response = client.predict(
        build_nanobody_polymers(target_sequence, design_sequence, template_cif),
        diffusion_samples=diffusion_samples,
    )
    elapsed = time.time() - t0
    n_samples = len(response.get("structures") or []) or 1
    raw = {k: v for k, v in response.items() if k != "structures"}
    return {
        "elapsed_s": round(elapsed, 3),
        "confidence": [
            response_to_confidence(response, sample=s) for s in range(n_samples)
        ],
        "structures": response_structures(response),
        "raw": raw,
    }


def predict_nanobodies(
    client: NimBoltz2Client,
    target_sequence: str,
    design_sequences: Iterable[str],
    template_cif: str,
    *,
    diffusion_samples: int = 2,
    on_error: str = "raise",
) -> dict[str, dict]:
    """Predict many design sequences against one target; keyed by sequence."""
    out: dict[str, dict] = {}
    for seq in design_sequences:
        try:
            out[seq] = predict_nanobody(
                client, target_sequence, seq, template_cif,
                diffusion_samples=diffusion_samples,
            )
        except NimBoltz2Error:
            if on_error == "skip":
                out[seq] = {"error": "nim_predict_failed"}
                continue
            raise
    return out
