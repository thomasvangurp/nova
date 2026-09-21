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
from nanobody_opt import diversity, gates, publication_gates
from nanobody_opt.simulate_round import METRICS

SCHEMA_VERSION = 3
EXACT_SCORER = "nova-boltzgen-validator-wrapper"

# Every threshold below is READ from publication_gates, never restated here.
# Ranking already qualifies candidates against those gates; a second, stricter
# copy in the program that writes the checkpoint does not add safety, it makes
# publication impossible for exactly the candidates ranking just approved -- and
# a hardcoded ceiling of 90 on a drifting rank-sum scale is how re-scoring the
# sequence the validator paid 79 for came back at median 180 and blocked itself.
_PUBLICATION_GATES = publication_gates.gate_defaults()
MINIMUM_EXACT_OBSERVATIONS = int(_PUBLICATION_GATES["--minimum-observations"])
MINIMUM_COMPETITIVE_RATE = float(_PUBLICATION_GATES["--minimum-competitive-rate"])
MINIMUM_OBSERVATION_COMPETITIVE_RATE = float(
    _PUBLICATION_GATES["--minimum-observation-competitive-rate"]
)
MAXIMUM_P90_RANK_SUM = float(_PUBLICATION_GATES["--maximum-p90-rank-sum"])
MAXIMUM_P90_CATEGORY_RANK_SUMS = {
    "confidence": float(_PUBLICATION_GATES["--maximum-p90-confidence-rank-sum"]),
    "physical_interaction": float(
        _PUBLICATION_GATES["--maximum-p90-physical-interaction-rank-sum"]
    ),
    "developability": float(
        _PUBLICATION_GATES["--maximum-p90-developability-rank-sum"]
    ),
}
MINIMUM_STRUCTURE_OBSERVATIONS = int(
    _PUBLICATION_GATES["--minimum-structure-observations"]
)
MINIMUM_MEDIAN_CONTACT_JACCARD = float(
    _PUBLICATION_GATES["--minimum-median-contact-jaccard"]
)
MAXIMUM_CDR3_RMSD_ANGSTROM = float(_PUBLICATION_GATES["--maximum-cdr3-rmsd"])
# The validator-configured generation hotspot is not the contact patch used by
# the strongest public P20809 poses. Retain occupancy as a diagnostic, but do
# not gate publication on that mismatched site.
MINIMUM_EPITOPE_OCCUPANCY = float(_PUBLICATION_GATES["--minimum-epitope-occupancy"])


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


def validate_exact_robustness(winner: dict) -> tuple[int, float, float]:
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
    observation_rate = float(
        winner.get("observation_competitive_rate", float("nan"))
    )
    if (
        not math.isfinite(observation_rate)
        or observation_rate < MINIMUM_OBSERVATION_COMPETITIVE_RATE
    ):
        raise ValueError(
            f"winner independent-observation competitive rate "
            f"{observation_rate!r} is below "
            f"{MINIMUM_OBSERVATION_COMPETITIVE_RATE:.0%}"
        )
    score_p90 = float(winner.get("score_p90", float("nan")))
    if not math.isfinite(score_p90) or score_p90 > MAXIMUM_P90_RANK_SUM:
        raise ValueError(
            f"winner p90 rank sum {score_p90!r} exceeds "
            f"{MAXIMUM_P90_RANK_SUM:.0f}"
        )
    categories = winner.get("p90_category_rank_sums") or {}
    for category, maximum in MAXIMUM_P90_CATEGORY_RANK_SUMS.items():
        value = float(categories.get(category, float("nan")))
        if not math.isfinite(value) or value > maximum:
            raise ValueError(
                f"winner p90 {category} rank sum {value!r} exceeds {maximum:.0f}"
            )
    pose = winner.get("pose_stability") or {}
    structure_count = int(pose.get("structure_observation_count", 0))
    if structure_count < MINIMUM_STRUCTURE_OBSERVATIONS:
        raise ValueError(
            f"winner has fewer than {MINIMUM_STRUCTURE_OBSERVATIONS} retained "
            "complex observations"
        )
    contact_jaccard = float(pose.get("contact_jaccard_median", float("nan")))
    if (
        not math.isfinite(contact_jaccard)
        or contact_jaccard < MINIMUM_MEDIAN_CONTACT_JACCARD
    ):
        raise ValueError(
            f"winner median contact Jaccard {contact_jaccard!r} is below "
            f"{MINIMUM_MEDIAN_CONTACT_JACCARD:.2f}"
        )
    cdr3_rmsd = float(pose.get("cdr3_rmsd_max_angstrom", float("nan")))
    if not math.isfinite(cdr3_rmsd) or cdr3_rmsd > MAXIMUM_CDR3_RMSD_ANGSTROM:
        raise ValueError(
            f"winner maximum CDR3 RMSD {cdr3_rmsd!r} exceeds "
            f"{MAXIMUM_CDR3_RMSD_ANGSTROM:.1f} A"
        )
    epitope_occupancy = float(pose.get("epitope_occupancy_min", float("nan")))
    if (
        not math.isfinite(epitope_occupancy)
        or epitope_occupancy < MINIMUM_EPITOPE_OCCUPANCY
    ):
        raise ValueError(
            f"winner minimum epitope occupancy {epitope_occupancy!r} is below "
            f"{MINIMUM_EPITOPE_OCCUPANCY:.0%}"
        )
    return observation_count, competitive_rate, observation_rate


def validate_provisional_exact_evidence(winner: dict) -> tuple[int, float, float]:
    """Accept the first complete exact observation for optimistic publication."""
    observation_count = int(winner.get("observation_count", 0))
    if observation_count < 1:
        raise ValueError("winner has no exact observations")
    score_p90 = float(winner.get("score_p90", float("nan")))
    if not math.isfinite(score_p90):
        raise ValueError("winner has no finite exact p90 score")
    competitive_rate = float(winner.get("competitive_rate", 0.0))
    observation_rate = float(winner.get("observation_competitive_rate", 0.0))
    if not math.isfinite(competitive_rate) or not math.isfinite(observation_rate):
        raise ValueError("winner has non-finite competitive evidence")
    return observation_count, competitive_rate, observation_rate


def checkpoint_is_better(
    previous: object, candidate: dict, assigned_sequence: str | None = None
) -> bool:
    """Only replace a same-epoch checkpoint with better exact evidence.

    When a sequence is assigned -- one uid of the fleet carries exactly this
    molecule -- the assignment wins over the score comparison: a file left behind
    by a different sequence is stale for this uid, not better than it.
    """
    if not isinstance(previous, dict):
        return True
    if previous.get("epoch") != candidate.get("epoch"):
        return True
    if assigned_sequence:
        previous_sequences = previous.get("sequences")
        held = (
            str(previous_sequences[0]).strip().upper()
            if isinstance(previous_sequences, list) and previous_sequences
            else ""
        )
        if held != assigned_sequence.strip().upper():
            return True
    try:
        previous_score = float(previous["selection"]["score"])
        candidate_score = float(candidate["selection"]["score"])
        previous_observations = int(previous["exact"]["observation_count"])
        candidate_observations = int(candidate["exact"]["observation_count"])
        previous_sequence = str(previous["sequences"][0])
        candidate_sequence = str(candidate["sequences"][0])
    except (KeyError, IndexError, TypeError, ValueError):
        return True
    if not math.isfinite(previous_score):
        return True
    if not math.isfinite(candidate_score):
        return False
    if previous_sequence == candidate_sequence:
        return (
            candidate_observations > previous_observations
            or (
                candidate_observations == previous_observations
                and candidate_score < previous_score
            )
        )
    return (
        candidate_score < previous_score
        or (
            candidate_score == previous_score
            and candidate_observations > previous_observations
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankings", required=True, type=Path)
    parser.add_argument("--layer2", action="append", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--epoch", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--provisional",
        action="store_true",
        help="publish after one complete exact observation while repeats continue",
    )
    parser.add_argument(
        "--sequence",
        help=(
            "publish this ranked sequence instead of rank one; the fleet "
            "publisher binds one distinct sequence to each uid it holds"
        ),
    )
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
    if args.sequence:
        requested = args.sequence.strip().upper()
        matches = [
            item for item in candidates
            if str(item.get("sequence", "")).strip().upper() == requested
        ]
        if not matches:
            raise SystemExit(
                f"requested sequence is not in {args.rankings}; a uid may only "
                "publish a candidate that passed this epoch's qualification"
            )
        winner = matches[0]
    sequence = str(winner["sequence"]).strip().upper()
    try:
        validator = (
            validate_provisional_exact_evidence
            if args.provisional
            else validate_exact_robustness
        )
        observation_count, competitive_rate, observation_rate = validator(winner)
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
        "status": "provisional" if args.provisional else "qualified",
        "publication_stage": "provisional" if args.provisional else "qualified",
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
                "metric_aggregation": winner.get(
                    "metric_aggregation", "arithmetic_mean"
                ),
                "rank_method": "dense",
                "metrics": metrics,
            },
        },
        "exact": {
            "scorer": EXACT_SCORER,
            "observation_count": observation_count,
            "metrics": metrics,
            "metric_standard_errors": winner.get("metric_standard_errors", {}),
            "observations": winner.get("observations", []),
            "per_observation_median_rank_sum": winner[
                "per_observation_median_rank_sum"
            ],
            "pose_stability": winner.get("pose_stability", {}),
        },
        "selection": {
            "method": (
                "p90-observation-field-dense-rank-sum"
                if args.provisional
                else "competitive-rate-then-p90-observation-field-dense-rank-sum"
            ),
            "score": selection_score,
            "competitive_rate": competitive_rate,
            "observation_competitive_rate": observation_rate,
            "competitive_score_threshold": float(
                winner["competitive_score_threshold"]
            ),
            "score_median": float(winner["score_median"]),
            "score_p90": float(winner["score_p90"]),
            "score_worst": float(winner["score_worst"]),
            "p90_category_rank_sums": winner.get("p90_category_rank_sums", {}),
            # Carried through for the fleet artifact: the reason this sequence was
            # chosen for a uid is its chance of a winning draw, not its average.
            "win_probability": winner.get("win_probability"),
            "selection_order": ranking.get("selection_order"),
            "consensus_ranking_policy": ranking.get("consensus_ranking_policy", "tier"),
            "selection_metric_policy": ranking.get("selection_metric_policy", "plain"),
            "selection_metric_weights": ranking.get("selection_metric_weights"),
            "prediction_event": "selection_rank_sum_below_fixed_threshold",
            "score_semantics": (
                "published raw metrics are the arithmetic mean of independent "
                "local observations, matching validator score sharing; the current "
                "field is unavailable before reveal, while p90 is only a robustness "
                "qualification guard"
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
    try:
        previous = json.loads(args.output.read_text())
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError):
        previous = None
    if not checkpoint_is_better(previous, payload, args.sequence):
        print(
            f"retained better epoch {args.epoch} checkpoint at {args.output}; "
            f"candidate_score={selection_score:.1f} sequence={sequence}"
        )
        return
    _atomic_json(args.output, payload)
    stage = payload["publication_stage"]
    print(
        f"published {stage} epoch {args.epoch} {args.target} checkpoint to "
        f"{args.output}: robustness_p90={selection_score:.1f} sequence={sequence}"
    )


if __name__ == "__main__":
    main()
