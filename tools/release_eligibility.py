#!/usr/bin/env python3
"""Production-exact eligibility filtering before compound scoring.

The release gate is the objective.  Candidate generation, reduced/full Boltz
admission, adaptive confirmation, and lane accounting must not learn from a
molecule that the publisher cannot use.  This module centralises the cheap
parts of that gate:

* validator molecular validity;
* novelty against the current public Submission-Archive;
* exact and Morgan-radius-2/2048 similarity exclusion against recent public
  leaderboard rows and our not-yet-archived submissions; and
* the AtomPair entropy-constrained top-20 selector used by the publisher.

Observation count is deliberately not a pre-score filter: a new molecule must
be observed before it can reach three passes.  It remains a hard requirement
when ``production_portfolio`` is called.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator as rfg

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from compound_miner.novelty_gate import default_cache, prefilter  # noqa: E402
from compound_miner.types import Candidate  # noqa: E402
from compound_miner.validity import (  # noqa: E402
    MAX_SIMILARITY_TO_HISTORICAL,
    HistoricalIndex,
    ValidityGate,
)


MINIMUM_OBSERVATIONS = 3
MINIMUM_ENTROPY = 0.25
PORTFOLIO_COUNT = 20
ENTROPY_SEARCH_POOL = 300
RECENT_EPOCH_LOOKBACK = 32


def candidate(row: dict[str, Any]) -> Candidate:
    components = row.get("components") or ()
    if isinstance(components, str):
        components = json.loads(components)
    return Candidate(
        str(row["name"]), str(row["smiles"]), int(row["reaction"]),
        tuple(int(value) for value in components), str(row["inchikey"]),
        int(row["heavy_atoms"]), int(row["rotatable_bonds"]),
        float(row.get("proxy_mean", 0.0) or 0.0),
        float(row.get("proxy_std", 0.0) or 0.0),
    )


def _recent_public(path: Path | None, lookback: int) -> tuple[set[str], set[str], dict]:
    if path is None or not path.is_file():
        raise FileNotFoundError(f"recent public history is missing: {path}")
    frame = pd.read_csv(path, usecols=["epoch", "name", "smiles"])
    if frame.empty:
        return set(), set(), {"latest_epoch": None, "rows": 0}
    epochs = pd.to_numeric(frame["epoch"], errors="coerce")
    latest = int(epochs.max())
    recent = frame.loc[epochs >= latest - max(0, int(lookback))]
    return (
        set(recent["name"].dropna().astype(str)),
        set(recent["smiles"].dropna().astype(str)),
        {"latest_epoch": latest, "rows": len(recent)},
    )


def _submitted(path: Path | None) -> tuple[set[str], set[str], int]:
    if path is None or not path.is_dir():
        raise FileNotFoundError(f"submitted archive is missing: {path}")
    names: set[str] = set()
    smiles: set[str] = set()
    files = 0
    for item in sorted(path.glob("epoch*/compound_fleet.json")):
        try:
            payload = json.loads(item.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        files += 1
        for submission in payload.get("submissions") or []:
            for molecule in ((submission.get("molecules") or []) +
                             (submission.get("spares") or [])):
                if molecule.get("name"):
                    names.add(str(molecule["name"]))
                if molecule.get("smiles"):
                    smiles.add(str(molecule["smiles"]))
    return names, smiles, files


def _fingerprints(smiles: Iterable[str]):
    generator = rfg.GetMorganGenerator(radius=2, fpSize=2048)
    molecules = [Chem.MolFromSmiles(value) for value in smiles]
    molecules = [molecule for molecule in molecules if molecule is not None]
    return generator, list(generator.GetFingerprints(molecules, numThreads=0))


@dataclass
class ReleaseEligibility:
    historical_molecules: Path
    public_history: Path
    submitted_archive: Path
    recent_epoch_lookback: int = RECENT_EPOCH_LOOKBACK

    def __post_init__(self) -> None:
        for path, label in (
            (self.historical_molecules, "historical molecule archive"),
            (self.public_history, "public score history"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{label} is missing: {path}")
        if not self.submitted_archive.is_dir():
            raise FileNotFoundError(
                f"submitted molecule archive is missing: {self.submitted_archive}"
            )
        self.history = HistoricalIndex(self.historical_molecules)
        self.validity = ValidityGate(self.history)
        self.cache = default_cache(self.history)
        public_names, public_smiles, public_stats = _recent_public(
            self.public_history, self.recent_epoch_lookback,
        )
        own_names, own_smiles, own_files = _submitted(self.submitted_archive)
        self.recent_names = public_names | own_names
        self.recent_smiles = public_smiles | own_smiles
        self.recent_generator, self.recent_fingerprints = _fingerprints(
            self.recent_smiles,
        )
        self.source_stats = {
            "public_recent_rows": public_stats["rows"],
            "public_latest_epoch": public_stats["latest_epoch"],
            "submitted_files": own_files,
            "recent_names": len(self.recent_names),
            "recent_smiles": len(self.recent_smiles),
        }

    def filter_candidates(
        self, candidates: list[Candidate],
    ) -> tuple[list[Candidate], list[int], dict[str, Any]]:
        started = time.monotonic()
        _, public_indices, public_stats = prefilter(
            candidates, self.history, self.cache,
        )
        valid_indices = [
            index for index in public_indices
            if self.validity.accepts(candidates[index], check_similarity=False)
        ]
        invalid = len(public_indices) - len(valid_indices)
        exact_indices = [
            index for index in valid_indices
            if candidates[index].name not in self.recent_names
            and candidates[index].smiles not in self.recent_smiles
        ]
        recent_exact = len(valid_indices) - len(exact_indices)
        kept: list[int] = []
        recent_similarity = 0
        guard = MAX_SIMILARITY_TO_HISTORICAL - 1e-6
        for index in exact_indices:
            molecule = Chem.MolFromSmiles(candidates[index].smiles)
            if molecule is None:
                invalid += 1
                continue
            fingerprint = self.recent_generator.GetFingerprint(molecule)
            similarities = DataStructs.BulkTanimotoSimilarity(
                fingerprint, self.recent_fingerprints,
            ) if self.recent_fingerprints else []
            if any(value >= guard for value in similarities):
                recent_similarity += 1
            else:
                kept.append(index)
        stats = {
            "considered": len(candidates),
            "public_novelty_refused": len(candidates) - len(public_indices),
            "validator_refused": invalid,
            "recent_exact_refused": recent_exact,
            "recent_similarity_refused": recent_similarity,
            "kept": len(kept),
            "seconds": round(time.monotonic() - started, 3),
            **self.source_stats,
            "public_filter": public_stats,
        }
        return [candidates[index] for index in kept], kept, stats

    def filter_rows(
        self, rows: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        candidates = [candidate(row) for row in rows]
        _, kept, stats = self.filter_candidates(candidates)
        return [rows[index] for index in kept], stats


def _atompair_matrix(smiles_list: list[str]):
    generator = rfg.GetAtomPairGenerator(fpSize=2048)
    matrix = np.zeros((len(smiles_list), 2048), dtype=np.float64)
    parsed = np.zeros(len(smiles_list), dtype=bool)
    for index, smiles in enumerate(smiles_list):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is not None:
            matrix[index] = np.array(generator.GetFingerprint(molecule), dtype=np.float64)
            parsed[index] = True
    return matrix, parsed


def _entropy_rows(counts, size: int):
    probabilities = np.atleast_2d(counts) / size
    with np.errstate(divide="ignore", invalid="ignore"):
        bits = np.where(
            (probabilities > 0) & (probabilities < 1),
            -probabilities * np.log2(probabilities)
            - (1 - probabilities) * np.log2(1 - probabilities),
            0.0,
        )
    return bits.mean(axis=1)


def select_entropy_constrained(
    rows: list[dict[str, Any]],
    *,
    count: int = PORTFOLIO_COUNT,
    minimum_entropy: float = MINIMUM_ENTROPY,
    search_pool: int = ENTROPY_SEARCH_POOL,
) -> tuple[list[dict[str, Any]], float]:
    """Publisher-equivalent AtomPair constrained selection.

    Rows must contain ``smiles`` and ``score`` and are deduplicated before the
    release gate's top-300 search.  The repair and deterministic hill-climb are
    byte-for-byte equivalent in intent to ``bank_release.py``.
    """
    import random

    unique: list[dict[str, Any]] = []
    names: set[str] = set()
    inchikeys: set[str] = set()
    for row in sorted(rows, key=lambda value: float(value["score"]), reverse=True):
        name = str(row["name"])
        inchikey = str(row.get("inchikey") or "")
        if name in names or (inchikey and inchikey in inchikeys):
            continue
        names.add(name)
        if inchikey:
            inchikeys.add(inchikey)
        unique.append(row)
    unique = unique[:max(count, int(search_pool))]
    if len(unique) < count:
        raise ValueError(f"need {count} eligible candidates, only {len(unique)}")
    matrix, parsed = _atompair_matrix([str(row["smiles"]) for row in unique])
    scores = np.asarray([float(row["score"]) for row in unique], dtype=np.float64)
    usable = [index for index in range(len(unique)) if parsed[index]]
    if len(usable) < count:
        raise ValueError(f"need {count} parseable candidates, only {len(usable)}")
    selected = usable[:count]
    rest = usable[count:]
    counts = matrix[selected].sum(axis=0)

    def entropy(vector) -> float:
        return float(_entropy_rows(vector, count)[0])

    while entropy(counts) < minimum_entropy and rest:
        current = entropy(counts)
        rest_matrix = matrix[rest]
        rest_scores = scores[rest]
        best = None
        best_value = -1e18
        for position in range(count):
            gains = _entropy_rows(
                counts - matrix[selected[position]] + rest_matrix, count,
            ) - current
            cost = scores[selected[position]] - rest_scores
            value = np.where(
                cost > 1e-9, gains / np.where(cost > 1e-9, cost, 1.0), gains * 1e6,
            )
            value = np.where(gains > 0, value, -np.inf)
            choice = int(np.argmax(value))
            if np.isfinite(value[choice]) and value[choice] > best_value:
                best_value = float(value[choice])
                best = position, choice
        if best is None:
            break
        position, choice = best
        incoming = rest[choice]
        counts = counts - matrix[selected[position]] + matrix[incoming]
        rest.append(selected[position])
        rest.remove(incoming)
        selected[position] = incoming

    randomizer = random.Random(0)
    for _ in range(4000):
        if not rest:
            break
        position = randomizer.randrange(count)
        incoming = rest[randomizer.randrange(len(rest))]
        if scores[incoming] <= scores[selected[position]]:
            continue
        candidate_counts = counts - matrix[selected[position]] + matrix[incoming]
        if entropy(candidate_counts) < minimum_entropy:
            continue
        rest.append(selected[position])
        rest.remove(incoming)
        counts = candidate_counts
        selected[position] = incoming
    if entropy(counts) < minimum_entropy:
        raise ValueError("eligible candidates cannot satisfy AtomPair entropy floor")
    selected.sort(key=lambda index: (-scores[index], index))
    return [unique[index] for index in selected], entropy(counts)


def _select_entropy_indices(
    matrix: np.ndarray,
    parsed: np.ndarray,
    scores: np.ndarray,
    *,
    count: int = PORTFOLIO_COUNT,
    minimum_entropy: float = MINIMUM_ENTROPY,
    search_pool: int = ENTROPY_SEARCH_POOL,
    seed: int = 0,
) -> tuple[list[int], float]:
    """Matrix-cached equivalent of ``select_entropy_constrained``.

    Thompson acquisition calls the production selector hundreds of times with
    different score draws but identical chemistry.  Re-parsing SMILES and
    rebuilding AtomPair fingerprints in each draw would cost more CPU than the
    simulation itself, so this variant reuses one matrix.
    """
    import random

    ordered = sorted(
        (index for index in range(len(scores)) if parsed[index]),
        key=lambda index: (-float(scores[index]), index),
    )[:max(count, int(search_pool))]
    if len(ordered) < count:
        raise ValueError(f"need {count} parseable candidates, only {len(ordered)}")
    selected = ordered[:count]
    rest = ordered[count:]
    counts = matrix[selected].sum(axis=0)

    def entropy(vector) -> float:
        return float(_entropy_rows(vector, count)[0])

    while entropy(counts) < minimum_entropy and rest:
        current = entropy(counts)
        rest_matrix = matrix[rest]
        rest_scores = scores[rest]
        best = None
        best_value = -1e18
        for position in range(count):
            gains = _entropy_rows(
                counts - matrix[selected[position]] + rest_matrix, count,
            ) - current
            cost = scores[selected[position]] - rest_scores
            value = np.where(
                cost > 1e-9,
                gains / np.where(cost > 1e-9, cost, 1.0),
                gains * 1e6,
            )
            value = np.where(gains > 0, value, -np.inf)
            choice = int(np.argmax(value))
            if np.isfinite(value[choice]) and value[choice] > best_value:
                best_value = float(value[choice])
                best = position, choice
        if best is None:
            break
        position, choice = best
        incoming = rest[choice]
        counts = counts - matrix[selected[position]] + matrix[incoming]
        rest.append(selected[position])
        rest.remove(incoming)
        selected[position] = incoming

    randomizer = random.Random(seed)
    for _ in range(4000):
        if not rest:
            break
        position = randomizer.randrange(count)
        incoming = rest[randomizer.randrange(len(rest))]
        if scores[incoming] <= scores[selected[position]]:
            continue
        candidate_counts = counts - matrix[selected[position]] + matrix[incoming]
        if entropy(candidate_counts) < minimum_entropy:
            continue
        rest.append(selected[position])
        rest.remove(incoming)
        counts = candidate_counts
        selected[position] = incoming
    final_entropy = entropy(counts)
    if final_entropy < minimum_entropy:
        raise ValueError("eligible candidates cannot satisfy AtomPair entropy floor")
    selected.sort(key=lambda index: (-float(scores[index]), index))
    return selected, final_entropy


def portfolio_thompson_acquisition(
    incumbents: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    draws: int = 128,
    seed: int = 0,
    count: int = PORTFOLIO_COUNT,
    minimum_entropy: float = MINIMUM_ENTROPY,
    search_pool: int = ENTROPY_SEARCH_POOL,
) -> dict[str, Any]:
    """Estimate each candidate's value to the *selected portfolio*.

    Incumbent rows carry their confirmed full score. Candidate rows carry a
    calibrated ``predicted_full_score`` and ``posterior_sd``. Each Thompson
    draw samples all candidates jointly, reruns the exact AtomPair selector,
    and shares any positive portfolio improvement among the newly selected
    candidates. Joint draws capture complementary chemistry: two molecules
    can earn value together even when neither is an independent raw top-20
    replacement.
    """
    if draws < 1:
        raise ValueError("portfolio Thompson needs at least one draw")

    # The release selector only searches its top-N score rows. More incumbent
    # rows cannot affect a draw, while omitting candidate rows could hide a
    # useful replacement, so retain the top incumbent search pool plus all
    # candidates.
    incumbents = sorted(
        incumbents, key=lambda row: float(row["score"]), reverse=True,
    )[:max(count, int(search_pool))]
    combined: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    seen_inchikeys: set[str] = set()
    for row in incumbents:
        name = str(row["name"])
        inchikey = str(row.get("inchikey") or "")
        if name in seen_names or (inchikey and inchikey in seen_inchikeys):
            continue
        seen_names.add(name)
        if inchikey:
            seen_inchikeys.add(inchikey)
        combined.append(row)
    incumbent_count = len(combined)
    for row in candidates:
        name = str(row["name"])
        inchikey = str(row.get("inchikey") or "")
        if name in seen_names or (inchikey and inchikey in seen_inchikeys):
            continue
        seen_names.add(name)
        if inchikey:
            seen_inchikeys.add(inchikey)
        combined.append(row)
    if incumbent_count < count:
        raise ValueError(f"need {count} production incumbents, only {incumbent_count}")

    matrix, parsed = _atompair_matrix([str(row["smiles"]) for row in combined])
    baseline_scores = np.asarray(
        [float(row["score"]) for row in combined[:incumbent_count]],
        dtype=np.float64,
    )
    baseline_selected, baseline_entropy = _select_entropy_indices(
        matrix[:incumbent_count], parsed[:incumbent_count], baseline_scores,
        count=count, minimum_entropy=minimum_entropy, search_pool=search_pool,
        seed=seed,
    )
    baseline_score = float(baseline_scores[baseline_selected].sum())

    candidate_rows = combined[incumbent_count:]
    predicted = np.asarray(
        [float(row["predicted_full_score"]) for row in candidate_rows],
        dtype=np.float64,
    )
    posterior_sd = np.asarray(
        [max(1e-6, float(row["posterior_sd"])) for row in candidate_rows],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    selected_draws = np.zeros(len(candidate_rows), dtype=np.int64)
    positive_draws = np.zeros(len(candidate_rows), dtype=np.int64)
    gain_sum = np.zeros(len(candidate_rows), dtype=np.float64)
    portfolio_gain_sum = 0.0
    improved_draws = 0
    failed_draws = 0
    incumbent_full = np.asarray(
        [float(row["score"]) for row in combined[:incumbent_count]],
        dtype=np.float64,
    )
    for draw in range(draws):
        sampled = rng.normal(predicted, posterior_sd)
        scores = np.concatenate([incumbent_full, sampled])
        try:
            selected, _ = _select_entropy_indices(
                matrix, parsed, scores, count=count,
                minimum_entropy=minimum_entropy, search_pool=search_pool,
                seed=seed + draw,
            )
        except ValueError:
            failed_draws += 1
            continue
        new_indices = [index - incumbent_count for index in selected
                       if index >= incumbent_count]
        for index in new_indices:
            selected_draws[index] += 1
        delta = float(scores[selected].sum()) - baseline_score
        if delta <= 1e-12 or not new_indices:
            continue
        improved_draws += 1
        portfolio_gain_sum += delta
        shared = delta / len(new_indices)
        for index in new_indices:
            positive_draws[index] += 1
            gain_sum[index] += shared

    effective = max(1, draws - failed_draws)
    metrics = []
    for index, row in enumerate(candidate_rows):
        metrics.append({
            "name": str(row["name"]),
            "predicted_full_score": float(predicted[index]),
            "posterior_sd": float(posterior_sd[index]),
            "portfolio_selection_probability": float(selected_draws[index] / effective),
            "portfolio_entry_probability": float(positive_draws[index] / effective),
            "portfolio_expected_delta": float(gain_sum[index] / effective),
        })
    metrics.sort(key=lambda row: (
        row["portfolio_expected_delta"],
        row["portfolio_entry_probability"],
        row["predicted_full_score"],
    ), reverse=True)
    return {
        "baseline_score": baseline_score,
        "baseline_entropy": baseline_entropy,
        "draws": draws,
        "effective_draws": effective,
        "failed_draws": failed_draws,
        "improved_draw_probability": improved_draws / effective,
        "expected_best_portfolio_delta": portfolio_gain_sum / effective,
        "metrics": metrics,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, separators=(",", ":"), sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def filter_manifest(args: argparse.Namespace) -> int:
    payload = json.loads(args.acquisition.read_text())
    rows = payload.get("candidates") or []
    eligibility = ReleaseEligibility(
        args.historical_molecules, args.public_history, args.submitted_archive,
        args.recent_epoch_lookback,
    )
    kept, stats = eligibility.filter_rows(rows)
    payload["candidates"] = kept
    payload["expected_count"] = len(kept)
    payload["release_eligibility"] = {
        "policy": "production-exact-v1",
        "checked_at_unix": time.time(),
        **stats,
    }
    _atomic_json(args.acquisition, payload)
    print(json.dumps({"event": "release_eligibility_filter", **stats}, sort_keys=True))
    return 0 if kept else 75


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("filter-manifest")
    command.add_argument("--acquisition", type=Path, required=True)
    command.add_argument("--historical-molecules", type=Path, required=True)
    command.add_argument("--public-history", type=Path, required=True)
    command.add_argument("--submitted-archive", type=Path, required=True)
    command.add_argument("--recent-epoch-lookback", type=int, default=RECENT_EPOCH_LOOKBACK)
    args = parser.parse_args()
    return filter_manifest(args)


if __name__ == "__main__":
    raise SystemExit(main())
