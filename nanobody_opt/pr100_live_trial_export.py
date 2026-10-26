"""Build an honest, measured partial fleet for a bounded PR100 live trial."""
from __future__ import annotations

import math
import re

from nanobody_opt import rules
from nanobody_opt.simulate_round import METRICS


def build(validity, scored, settings, *, scorer_sha256, settings_file_sha256, now):
    """Join actual validity and one fast observation; never claim equivalence."""
    expected = validity["challenge"]
    epoch, target = expected["epoch"], expected["target"]
    config = rules.pinned_config(epoch)
    if epoch < rules.ACTIVATION_EPOCH or expected != rules.challenge(
            config, epoch, expected["epoch_start_block_hash"], expected["epoch_length"]):
        raise ValueError("invalid current challenge")
    if scored.get("target") != target or validity.get("target") != target:
        raise ValueError("scoring target mismatch")
    for value in (scorer_sha256, settings_file_sha256):
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("missing source or settings file identity")
    if not isinstance(settings, dict) or not settings:
        raise ValueError("actual scoring settings are required")
    if isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now):
        raise ValueError("invalid publication time")
    source = scored.get("per_nanobody_components", {}).get("0", {})
    joined = []
    seen = set()
    for row in validity["candidates"]:
        sequence = row["sequence"]
        if sequence in seen:
            raise ValueError("duplicate validity entry")
        seen.add(sequence)
        if row.get("layer2_passed") is not True or (row.get("novelty") or {}).get("validator_novel") is not True:
            raise ValueError("missing completed validity gate")
        rules.validate_evidence(row.get("pr100_evidence"), sequence, expected, config)
        metrics = (source.get(sequence) or {}).get(target)
        if metrics is None:
            continue  # A valid but not yet measured candidate is not published.
        if not isinstance(metrics, dict) or not set(METRICS) <= metrics.keys():
            raise ValueError("incomplete measured metrics")
        metrics = {key: metrics[key] for key in METRICS}
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in metrics.values()):
            raise ValueError("non-finite measured metrics")
        joined.append((sequence, metrics, row["pr100_evidence"]))
    if not joined:
        raise ValueError("no measured valid candidate")
    checkpoints = {}
    for sequence, metrics, evidence in joined:
        score = float(sum(1 + sum(value > metrics[key] if direction == "max" else value < metrics[key]
                                 for value in {other[key] for _, other, _ in joined})
                          for key, direction in METRICS.items()))
        checkpoints[sequence] = {
            "schema_version": 3, "status": "live_trial", "publication_stage": "live_trial",
            "target": target, "epoch": epoch, "sequences": [sequence], "pr100_evidence": evidence,
            "validation": {"layer_1": {"passed": True, "historically_unique": True,
                "duplicate_gate_passed": True}, "layer_2": {"passed": True},
                "boltzgen": {"exact": False, "runtime_matched": False, "validator_matched": False,
                    "scoring_equivalence_qualified": False, "mode": "live_fast_baseline", "metrics": metrics,
                    "provenance": {"scorer_sha256": scorer_sha256, "settings_sha256": rules.digest(settings),
                        "settings": settings, "settings_file_sha256": settings_file_sha256,
                        "target": target, "epoch": epoch}}},
            "exact": {"observation_count": 1, "metrics": metrics, "scoring_equivalence_qualified": False},
            "selection": {"score": score, "score_p90": score,
                "method": "one_pass_within_trial_dense_rank_sum", "reference_scope": "four_entry_live_trial" if len(joined) == 4 else "partial_live_trial",
                "reference_count": len(joined), "multipass_measured": False},
        }
    pool = sorted(checkpoints, key=lambda sequence: (checkpoints[sequence]["selection"]["score"],
                                                    rules.sequence_hash(sequence)))
    return {"schema_version": 4, "artifact_kind": "nanobody-fleet", "status": "live_trial",
            "publication_stage": "live_trial", "epoch": epoch, "target": target,
            "updated_at_unix": now, "producer": {"name": "pr100_live_collector"},
            "sequence_pool": pool, "sequence_pool_size": len(pool), "sequence_pool_checkpoints": checkpoints,
            "submissions": [], "submission_count": 0, "challenge": expected,
            "archive_sha256": validity["archive_sha256"],
            "archive_checked_at_unix": validity["archive_checked_at_unix"]}
