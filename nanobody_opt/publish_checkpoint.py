"""Publish one fail-closed, current-challenge nanobody checkpoint atomically."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nanobody_opt import diversity, gates
from nanobody_opt.simulate_round import METRICS

SCHEMA_VERSION = 1
EXACT_SCORER = "nova-boltzgen-validator-wrapper"
MINIMUM_EXACT_OBSERVATIONS = 8
MINIMUM_COMPETITIVE_RATE = 0.5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _layer2_passes(paths: list[Path]) -> set[str]:
    passed: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_text())
        for candidate in payload.get("candidates", []):
            if candidate.get("layer2_passed"):
                passed.add(str(candidate["sequence"]).strip().upper())
    return passed


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def validate_exact_robustness(winner: dict) -> tuple[int, float]:
    """Enforce the live publisher's stochastic-repeat qualification floor."""
    observation_count = int(winner.get("observation_count", 0))
    if observation_count < MINIMUM_EXACT_OBSERVATIONS:
        raise ValueError(
            f"winner has fewer than {MINIMUM_EXACT_OBSERVATIONS} exact observations"
        )
    competitive_rate = float(winner.get("competitive_rate", float("nan")))
    if (
        not math.isfinite(competitive_rate)
        or competitive_rate < MINIMUM_COMPETITIVE_RATE
    ):
        raise ValueError(
            f"winner competitive rate {competitive_rate!r} is below "
            f"{MINIMUM_COMPETITIVE_RATE:.0%}"
        )
    return observation_count, competitive_rate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankings", required=True, type=Path)
    parser.add_argument("--layer2", action="append", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--epoch", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    nanobody = config["protein_selection"]["nanobody"]
    configured_target = str(nanobody["target"])
    if configured_target != args.target:
        raise SystemExit(
            f"live config targets {configured_target!r}, not {args.target!r}"
        )
    requirements = config["nanobody_requirements"]
    if int(requirements["num_sequences"]) != 1:
        raise SystemExit("publisher currently supports exactly one nanobody")

    ranking = json.loads(args.rankings.read_text())
    if str(ranking.get("target")) != args.target:
        raise SystemExit("ranking target does not match current challenge")
    candidates = ranking.get("candidates") or []
    if not candidates:
        raise SystemExit("ranking contains no candidate")
    winner = candidates[0]
    sequence = str(winner["sequence"]).strip().upper()
    try:
        observation_count, competitive_rate = validate_exact_robustness(winner)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if gates.check_sequence(sequence, requirements):
        raise SystemExit("winner fails current Layer-1 sequence gates")
    if sequence not in _layer2_passes(args.layer2):
        raise SystemExit("winner lacks a current Layer-2 pass")

    archive = pd.read_csv(args.archive).drop_duplicates("sequence", keep="first")
    archive_sequences = set(archive.sequence.astype(str).str.strip().str.upper())
    if sequence in archive_sequences:
        raise SystemExit("winner is already present in the current archive")
    top = diversity.minmax_weighted_rank(
        archive, k=int(requirements["num_top_sequences"])
    )
    diverse, max_jaccard = diversity.passes_diversity(
        sequence,
        [diversity.kmers(item, 5) for item in top.sequence],
        threshold=float(requirements["max_similarity_score"]),
        k=5,
    )
    if not diverse:
        raise SystemExit("winner fails the current top-sequence diversity gate")

    metrics = {name: float(winner["metrics"][name]) for name in METRICS}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise SystemExit("winner contains non-finite exact metrics")
    selection_score = float(winner["selection_score"])
    if not math.isfinite(selection_score):
        raise SystemExit("winner contains a non-finite selection score")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "qualified",
        "target": args.target,
        "epoch": args.epoch,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "sequences": [sequence],
        "validation": {
            "layer_1": {
                "passed": True,
                "historically_unique": True,
                "diversity_passed": True,
                "max_jaccard_to_top50": max_jaccard,
            },
            "layer_2": {"passed": True},
            "boltzgen": {
                "exact": True,
                "runtime_matched": True,
                "stochastic": True,
                "rank_method": "dense",
                "metrics": metrics,
            },
        },
        "exact": {
            "scorer": EXACT_SCORER,
            "observation_count": observation_count,
            "metrics": metrics,
            "observations": winner.get("observations", []),
            "per_observation_median_rank_sum": winner[
                "per_observation_median_rank_sum"
            ],
        },
        "selection": {
            "method": "competitive-rate-then-p90-observation-field-dense-rank-sum",
            "score": selection_score,
            "competitive_rate": competitive_rate,
            "competitive_score_threshold": float(
                winner["competitive_score_threshold"]
            ),
            "score_median": float(winner["score_median"]),
            "score_p90": float(winner["score_p90"]),
            "score_worst": float(winner["score_worst"]),
            "p90_category_rank_sums": winner.get("p90_category_rank_sums", {}),
            "score_semantics": (
                "validator-compatible dense-rank robustness distribution over "
                "repeated exact observations and historical live fields; not a "
                "prediction of the next stochastic validator observation"
            ),
            "reference": ranking.get("reference", {}),
            "aggregate_median_rank_sum": float(
                winner["aggregate_median_rank_sum"]
            ),
            "aggregate_p90_rank_sum": float(winner["aggregate_p90_rank_sum"]),
            "aggregate_median_category_rank_sums": winner.get(
                "aggregate_median_category_rank_sums", {}
            ),
        },
        "challenge": {
            "structure": str(nanobody["structure"]),
            "structure_chain": str(nanobody["structure_chain"]),
            "structure_res_index": str(nanobody["structure_res_index"]),
            "structure_binding_site": str(nanobody["structure_binding_site"]),
        },
        "provenance": {
            "config_sha256": _sha256(args.config),
            "archive_sha256": _sha256(args.archive),
            "rankings_sha256": _sha256(args.rankings),
            "layer2_sha256": [_sha256(path) for path in args.layer2],
        },
    }
    _atomic_json(args.output, payload)
    print(
        f"published qualified epoch {args.epoch} {args.target} checkpoint to "
        f"{args.output}: robustness_p90={selection_score:.1f} sequence={sequence}"
    )


if __name__ == "__main__":
    main()
