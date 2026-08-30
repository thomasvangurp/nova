from __future__ import annotations

import concurrent.futures
import os
import time
from typing import Any, Iterable

import requests

from .targets import TARGETS
from .types import Candidate, ScoredCandidate


class NIMScorer:
    """Concurrent Boltz-2 NIM client using validator-matched inference settings."""

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        target: str = "P40261",
        concurrency: int = 8,
        timeout: float = 300.0,
    ) -> None:
        self.url = url or os.environ.get("NOVA_BOLTZ_NIM_URL", "https://health.api.nvidia.com/v1/biology/mit/boltz2/predict")
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        self.target = target
        self.concurrency = concurrency
        self.timeout = timeout

    def _payload(self, candidate: Candidate) -> dict[str, Any]:
        return {
            "polymers": [{"id": "A", "molecule_type": "protein", "sequence": TARGETS[self.target]}],
            "ligands": [{"id": "B", "smiles": candidate.smiles, "predict_affinity": True}],
            "recycling_steps": 3,
            "sampling_steps": 100,
            "diffusion_samples": 1,
            "sampling_steps_affinity": 100,
            "diffusion_samples_affinity": 3,
            "affinity_mw_correction": True,
            "output_format": "mmcif",
        }

    @staticmethod
    def _scalar(value: Any) -> float:
        while isinstance(value, list):
            value = value[0]
        return float(value)

    def score_one(self, candidate: Candidate) -> ScoredCandidate:
        headers = {"accept": "application/json", "content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        started = time.monotonic()
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                response = requests.post(self.url, headers=headers, json=self._payload(candidate), timeout=self.timeout)
                response.raise_for_status()
                payload = response.json()
                affinity = payload["affinities"]["B"]
                probability = self._scalar(affinity["affinity_probability_binary"])
                prediction = self._scalar(affinity["affinity_pred_value"])
                score = (probability - prediction) / candidate.heavy_atoms
                return ScoredCandidate(candidate, probability, prediction, score, f"nim:{self.url}", time.monotonic() - started)
            except (requests.RequestException, KeyError, TypeError, ValueError, IndexError) as error:
                last_error = error
                if attempt < 3:
                    time.sleep(1.0 * 2**attempt)
        raise RuntimeError(f"Boltz-2 NIM scoring failed for {candidate.name}: {last_error}")

    def score(self, candidates: Iterable[Candidate]) -> list[ScoredCandidate]:
        values = list(candidates)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = {pool.submit(self.score_one, candidate): index for index, candidate in enumerate(values)}
            indexed = [(futures[future], future.result()) for future in concurrent.futures.as_completed(futures)]
        return [value for _, value in sorted(indexed)]

