from __future__ import annotations

import concurrent.futures
import json
import time
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

API_ROOT = "https://compound-api.metanova-labs.ai/api"
NOVELTY_URL = "https://huggingface.co/datasets/Metanova/Submission-Archive/resolve/main/{target}_molecules.csv"


def _get_json(url: str, attempts: int = 4) -> dict[str, Any]:
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "nova-compound-miner/1"})
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.5 * 2**attempt)
    raise AssertionError("unreachable")


def available_epochs(target: str = "P40261") -> list[int]:
    payload = _get_json(f"{API_ROOT}/competitions/molecules?limit=10000")
    return sorted(
        int(item["epoch_number"])
        for item in payload.get("data", [])
        if target in item.get("target_proteins", [])
    )


def _epoch_rows(epoch: int) -> list[dict[str, Any]]:
    payload = _get_json(
        f"{API_ROOT}/competitions/leaderboard/{epoch}/molecules?limit=10000&molecule_detail=full"
    )
    rows: list[dict[str, Any]] = []
    for submission in payload.get("data", []):
        for molecule in submission.get("molecules", []):
            score = molecule.get("final_score")
            name = molecule.get("name")
            smiles = molecule.get("smiles")
            if score is None or not name or not smiles:
                continue
            components = name.split(":")
            if len(components) not in (4, 5) or components[0] != "rxn":
                continue
            rows.append(
                {
                    "epoch": epoch,
                    "uid": int(submission["uid"]),
                    "submission_score": float(submission.get("final_score") or 0.0),
                    "name": name,
                    "smiles": smiles,
                    "score": float(score),
                    "affinity_probability_binary": molecule.get("affinity_probability_binary"),
                    "affinity_pred_value": molecule.get("affinity_pred_value"),
                    "heavy_atom_count": molecule.get("heavy_atom_count"),
                    "reaction": int(components[1]),
                    "component_a": int(components[2]),
                    "component_b": int(components[3]),
                    "component_c": int(components[4]) if len(components) == 5 else -1,
                }
            )
    return rows


def sync_history(output: Path, target: str = "P40261", workers: int = 12) -> pd.DataFrame:
    """Download public scored submissions and atomically refresh a CSV training cache."""
    epochs = available_epochs(target)
    rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_epoch_rows, epoch): epoch for epoch in epochs}
        for future in concurrent.futures.as_completed(futures):
            rows.extend(future.result())
    frame = pd.DataFrame.from_records(rows).sort_values(["epoch", "uid", "score"], ascending=[True, True, False])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(output)
    return frame


def sync_novelty(output: Path, target: str = "P40261") -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    request = urllib.request.Request(NOVELTY_URL.format(target=target), headers={"User-Agent": "nova-compound-miner/1"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
    temporary.replace(output)
    return output


def aggregate_observations(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated stochastic scores without leaking future epochs into evaluation."""
    columns = ["name", "smiles", "reaction", "component_a", "component_b", "component_c"]
    return (
        frame.groupby(columns, as_index=False)
        .agg(score=("score", "mean"), score_std=("score", "std"), observations=("score", "size"), last_epoch=("epoch", "max"))
        .fillna({"score_std": 0.0})
    )
