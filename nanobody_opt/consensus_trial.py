"""Immutable protocol, paired forecasts and attribution for a final-order trial.

No chain, wallet, authentication or third-party imports are needed here.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Mapping

from nanobody_opt import consensus_policy

ARM_POLICIES = {"A": "tier", "B": "off"}
SOURCE_FILES = (
    "nanobody_opt/consensus_policy.py", "nanobody_opt/consensus_trial.py",
    "nanobody_opt/rank_exact.py", "nanobody_opt/fleet.py",
    "nanobody_opt/publication_gates.py", "nanobody_opt/run_live_producer.py",
    "nanobody_opt/local_neighborhood.py", "nanobody_opt/sequential_exact_race.py",
    "nanobody_opt/win_probability.py", "tools/publish_current_nanobody.py",
    "nanobody_opt/publish_checkpoint.py",
    "nanobody_opt/score_boltzgen_fast_adapter.py",
    "nanobody_opt/diffusion_sample_selection.py", "nanobody_opt/terminal_policy.py",
    "nanobody_opt/score_boltzgen_batch.py", "tools/run_consensus_trial.py",
)
METRIC_NAMES = (
    "design_iiptm", "design_ptm", "design_to_target_iptm",
    "min_design_to_target_pae", "interaction_pae", "plip_hbonds_refolded",
    "plip_saltbridge_refolded", "delta_sasa_refolded", "liability_score",
    "liability_num_violations",
)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def source_hashes(root: Path) -> dict[str, str]:
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in SOURCE_FILES}


def write_immutable(path: Path, payload: dict) -> dict:
    """Atomically create, or verify the identical existing document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            existing = json.loads(path.read_text())
            if digest(existing) != digest(payload):
                raise ValueError("refusing to replace a frozen trial document")
            return existing
        return payload
    finally:
        Path(temporary).unlink(missing_ok=True)


def create_protocol(*, start_epoch: int, epoch_count: int, seed: int,
                    target: str, source_root: Path, output: Path,
                    heldout_repeats: int = 8) -> dict:
    if start_epoch < 1 or epoch_count < 2 or epoch_count % 2 or heldout_repeats < 2:
        raise ValueError("trial needs a positive start, even epoch count and >=2 heldout repeats")
    rng = random.Random(seed)
    schedule = []
    for pair in range(epoch_count // 2):
        arms = ["A", "B"]
        rng.shuffle(arms)
        for offset, arm in enumerate(arms):
            schedule.append({"epoch": start_epoch + 2 * pair + offset,
                             "pair": pair, "arm": arm,
                             "consensus_ranking_policy": ARM_POLICIES[arm]})
    payload = {
        "schema_version": 1, "target": target, "start_epoch": start_epoch,
        "epoch_count": epoch_count, "randomization_seed": seed,
        "schedule": schedule, "source_hashes": source_hashes(source_root),
        "treatment": "final_consensus_ranking_tier_only",
        "generation_policy": "unchanged_repair_only",
        "shared_shortlist": "equal_prefix_union_capped_at_existing_capacity",
        "confirmation_policy": "unchanged_current_producer",
        "heldout_repeats": heldout_repeats,
        "max_heldout_attempts_per_repeat": 3,
        "heldout_gpu_seconds_per_case": 7.5,
        "heldout_cpu_guard_seconds": 120.0,
        "heldout_cutoff_offset": 351,
        "heldout_max_seconds_per_epoch": 540.0,
        "heldout_gpu_utilization_limit": 10,
        "primary_endpoint": "validator_top10_per_scored_submission",
        "delivery_endpoint": "validator_scored_per_verified_commitment",
        "win_endpoint": "public_nanobody_rank1",
        "review_rule": "fixed_end_no_automatic_policy_promotion",
        "analysis_unit": "randomized_epoch_pair",
        "limits": ["Local heldout results are not validator ground truth.",
                   "This tests final ranking conditional on unchanged generation and confirmation.",
                   "Rank-one power must be assessed separately; 40 epochs may be inconclusive."],
    }
    payload["protocol_sha256"] = digest(payload)
    return write_immutable(output, payload)


def load_protocol(path: Path, source_root: Path | None = None) -> dict:
    payload = json.loads(path.read_text())
    expected = payload.get("protocol_sha256")
    if digest({k: v for k, v in payload.items() if k != "protocol_sha256"}) != expected:
        raise ValueError("trial protocol digest mismatch")
    if payload.get("schema_version") != 1 or payload.get("treatment") != "final_consensus_ranking_tier_only":
        raise ValueError("unsupported trial protocol")
    start, count = payload["start_epoch"], payload["epoch_count"]
    if not isinstance(start, int) or not isinstance(count, int) or count < 2 or count % 2:
        raise ValueError("invalid trial bounds")
    schedule = payload["schedule"]
    if len(schedule) != count:
        raise ValueError("invalid trial schedule length")
    for i, row in enumerate(schedule):
        if row != {"epoch": start + i, "pair": i // 2, "arm": row.get("arm"),
                   "consensus_ranking_policy": ARM_POLICIES.get(row.get("arm"))}:
            raise ValueError("invalid trial schedule entry")
    if any({schedule[i]["arm"], schedule[i + 1]["arm"]} != {"A", "B"}
           for i in range(0, count, 2)):
        raise ValueError("trial pairs must contain both policies")
    if source_root is not None and source_hashes(source_root) != payload["source_hashes"]:
        raise ValueError("trial source changed from the pinned protocol")
    return payload


def epoch_context(protocol: dict, epoch: int) -> dict | None:
    index = epoch - protocol["start_epoch"]
    if not 0 <= index < protocol["epoch_count"]:
        return None
    return {**protocol["schedule"][index], "target": protocol["target"],
            "protocol_sha256": protocol["protocol_sha256"]}


def candidate_forecast(item: Mapping[str, Any]) -> dict:
    sequence = str(item["sequence"])
    probability = float(item.get("win_probability") or 0.0)
    if not math.isfinite(probability):
        raise ValueError("invalid trial forecast probability")
    return {
        "sequence": sequence, "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
        "event": "rank_sum_below_fixed_threshold", "probability": probability,
        "threshold": item.get("competitive_score_threshold"),
        "score_median": item.get("score_median"), "score_p90": item.get("score_p90"),
        "settled_deviations": int(item.get("settled_deviations", 0)),
        "observation_count": int(item.get("observation_count", 0)),
        "per_observation_rank_sums": item.get("per_observation_median_rank_sum", []),
        "metrics": {name: item.get("metrics", {}).get(name) for name in METRIC_NAMES},
    }


def frozen_manifest(*, context: dict, ranking: dict, baskets: dict,
                    crew: list[dict], block: int, epoch_length: int,
                    stage: str, input_hashes: dict) -> dict:
    if not crew or set(baskets) != {"A", "B"} or any(not baskets[arm] for arm in ["A", "B"]):
        raise ValueError("trial requires a nonempty crew and two nonempty baskets")
    candidates = {item["sequence"]: item for item in ranking["candidates"]}
    reference = ranking.get("reference", {})
    payload = {
        "schema_version": 1, **context, "frozen_at_block": block,
        "epoch_length": epoch_length, "stage": stage, "crew": crew,
        "candidate_pool_sha256": digest([candidate_forecast(c) for c in
                                         sorted(ranking["candidates"], key=lambda c: c["sequence"])]),
        "shared_shortlist": ranking.get("consensus_shortlist"),
        "reference": {k: reference[k] for k in ["source", "epochs", "field_sizes"] if k in reference},
        "qualification": ranking.get("qualification"), "input_sha256": input_hashes,
        "arms": {arm: {"policy": ARM_POLICIES[arm],
                       "fleet": [candidate_forecast(candidates[s]) for s in seqs[:len(crew)]],
                       "spares": [candidate_forecast(candidates[s]) for s in seqs[len(crew):]]}
                 for arm, seqs in baskets.items()},
    }
    payload["manifest_sha256"] = digest(payload)
    return payload


def load_manifest(path: Path, protocol: dict) -> dict:
    manifest = json.loads(path.read_text())
    if manifest.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise ValueError("manifest belongs to another trial")
    if digest({k: v for k, v in manifest.items() if k != "manifest_sha256"}) != manifest.get("manifest_sha256"):
        raise ValueError("frozen manifest digest mismatch")
    context = epoch_context(protocol, manifest["epoch"])
    if context is None or any(manifest.get(k) != v for k, v in context.items()):
        raise ValueError("manifest disagrees with randomized assignment")
    return manifest
