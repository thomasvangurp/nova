"""Versioned nanobody rules shared by scoring and the independent Pi handoff.

No model or annotation dependencies: the Pi recomputes the challenge and checks
measured evidence without importing the GPU environment. Legacy epochs keep
their original rules; PR100 starts at the announced boundary.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

ACTIVATION_EPOCH = 25955
UPSTREAM_SHA = "ce19ea8cbe656671c3440ba58b4093ff80798d4f"
FILTER_SHA = "fe2d5868126406a0de8953d26ecd8bdccbdbd029"
PROFILE = "nanobody-pr100-v1"
TARGET = "Q9NZQ7"
MAX_AVERAGE = Decimal("-0.68002724866200")
MAX_PEAK = Decimal("1.58023066762925")
CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def target_for_epoch(epoch):
    """Target identity for historical readers as well as the live cutover."""
    return TARGET if int(epoch) >= ACTIVATION_EPOCH else "P20809"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def sequence_hash(sequence):
    return hashlib.sha256(sequence.strip().upper().encode("ascii")).hexdigest()


def pinned_config(epoch):
    name = "nanobody_pr100.json" if int(epoch) >= ACTIVATION_EPOCH else "nanobody_pre_pr100.json"
    return json.loads((CONFIG_DIR / name).read_text())


def is_pr100(requirements):
    return "min_human_v_identity" in requirements


def for_epoch(config, epoch):
    """Select announced nanobody rules only; preserve unrelated compound config."""
    result = copy.deepcopy(config)
    if int(epoch) < ACTIVATION_EPOCH and not is_pr100(config["nanobody_requirements"]):
        result.pop("nova_epoch_challenge", None)
        return result
    selected = pinned_config(epoch)
    if int(epoch) >= ACTIVATION_EPOCH:
        if is_pr100(config["nanobody_requirements"]):
            if fingerprint(config) != fingerprint(selected):
                raise ValueError("upstream nanobody rules differ from the qualified PR100 profile")
        elif int(epoch) > ACTIVATION_EPOCH:
            raise ValueError("upstream PR100 activation has not been confirmed")
    for key in ("nanobody_requirements", "germline_filtering"):
        if key in selected:
            result[key] = selected[key]
        else:
            result.pop(key, None)
    result.setdefault("protein_selection", {})["nanobody"] = selected["protein_selection"]["nanobody"]
    result.pop("nova_epoch_challenge", None)
    return result


def fingerprint(config):
    return digest({"profile": PROFILE, "upstream_sha": UPSTREAM_SHA,
                   "filter_sha": FILTER_SHA,
                   "target": config["protein_selection"]["nanobody"],
                   "requirements": config["nanobody_requirements"],
                   "germline_filtering": config.get("germline_filtering"),
                   "aggrescan_caps": [str(MAX_AVERAGE), str(MAX_PEAK)]})


def challenge(config, epoch, block_hash, epoch_length=361):
    if (not isinstance(block_hash, str)
            or not re.fullmatch(r"0x[0-9a-fA-F]{64}", block_hash)):
        raise ValueError("missing or invalid epoch-start block hash")
    epoch, epoch_length = int(epoch), int(epoch_length)
    if epoch < 0 or epoch_length < 1:
        raise ValueError("invalid epoch dimensions")
    target = str(config["protein_selection"]["nanobody"]["target"])
    req = config["nanobody_requirements"]
    if epoch >= ACTIVATION_EPOCH and (target != TARGET or not is_pr100(req)):
        raise ValueError("PR100 rules and target required after activation")
    if epoch < ACTIVATION_EPOCH and is_pr100(req):
        raise ValueError("PR100 rules are not active for this epoch")
    out = {"epoch": epoch, "target": target, "epoch_length": epoch_length,
           "epoch_start_block": epoch * epoch_length,
           "epoch_start_block_hash": block_hash.lower(),
           "ruleset_sha256": fingerprint(config)}
    if is_pr100(req):
        settings = config["germline_filtering"]
        rng = random.Random(int(block_hash[2:], 16))
        categories = settings["v_neighborhood_categories"]
        genes = categories[rng.choice(list(categories))]
        if not genes:
            raise ValueError("empty V category")
        out["allowed_v"] = rng.choice(genes) if len(genes) > 1 else genes[0]
        lengths = settings["cdr3_length_range"]
        out["allowed_cdr3_length"] = rng.choice(list(range(lengths["min"], lengths["max"] + 1)))
    return out


def bind(config, epoch, block_hash, epoch_length=361):
    out = copy.deepcopy(config)
    out["nova_epoch_challenge"] = challenge(out, epoch, block_hash, epoch_length)
    return out


def context(config):
    value = config.get("nova_epoch_challenge")
    if not isinstance(value, dict):
        raise ValueError("epoch challenge is not attached to config")
    expected = challenge(config, value["epoch"], value["epoch_start_block_hash"], value["epoch_length"])
    if value != expected:
        raise ValueError("epoch challenge does not match rules")
    return expected


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def annotation_reason(annotation, requirements, *, sequence=None, epoch_challenge=None):
    if not isinstance(annotation, dict) or annotation.get("status") != "ok":
        return "annotation_unavailable"
    if annotation.get("profile") != PROFILE:
        return "annotation_policy_mismatch"
    if sequence is not None and annotation.get("sequence_hash") != sequence_hash(sequence):
        return "annotation_sequence_mismatch"
    if annotation.get("single_heavy_domain") is not True or annotation.get("scheme") != "imgt":
        return "not_single_heavy_domain"
    if requirements.get("enforce_vhh_hallmarks", True) and annotation.get("vhh_hallmarks") is not True:
        return "not_vhh"
    if annotation.get("canonical_cysteines") is not True:
        return "canonical_cysteines"
    germline = annotation.get("germline") or {}
    for region, key in (("V", "min_human_v_identity"), ("J", "min_human_j_identity")):
        value = (germline.get(region) or {}).get("identity_pct")
        if not _finite(value) or value < requirements[key]:
            return "human_" + region.lower() + "_identity"
    for key in ("reference_sha256", "hmm_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(germline.get(key, ""))):
            return "annotation_reference_missing"
    length = annotation.get("cdr3_length")
    if not isinstance(length, int) or isinstance(length, bool):
        return "cdr3_length_unavailable"
    if epoch_challenge:
        if epoch_challenge["allowed_v"] not in (germline.get("V") or {}).get("genes", []):
            return "wrong_epoch_v_gene"
        if length != epoch_challenge["allowed_cdr3_length"]:
            return "wrong_epoch_cdr3_length"
    return ""


def developability_reason(result, requirements, *, sequence=None):
    if not isinstance(result, dict) or result.get("error"):
        return "developability_unavailable"
    if result.get("red_flags"):
        return "developability_red_flags"
    # Recheck the pinned filter's measured criteria on Pi as well. A bare
    # `passed` flag or a truncated profile is not independent evidence.
    for key, lower, upper in (("total_cdr_length", 20, 39),
                              ("cdr3_length", 5, 23),
                              ("cdr3_compactness", .56, 1.61),
                              ("surface_hydrophobic_patches", 73.4, 155.47)):
        value = result.get(key)
        if not _finite(value):
            return "developability_unavailable"
        if not lower <= value <= upper:
            return "developability_red_flags"
    for key, upper in (("positive_charge_patches", 1.18), ("negative_charge_patches", 1.88)):
        value = result.get(key)
        if not _finite(value):
            return "developability_unavailable"
        if value > upper:
            return "developability_red_flags"
    psh = result.get("surface_hydrophobic_patches")
    if not _finite(psh):
        return "psh_unavailable"
    if psh > requirements["psh_strict_rejection_threshold"]:
        return "psh_threshold"
    score = result.get("aggrescan3d")
    if not isinstance(score, dict) or score.get("status") != "ok" or score.get("passed") is not True:
        return "aggrescan_unavailable_or_failed"
    if sequence is not None and score.get("sequence_hash") != sequence_hash(sequence):
        return "aggrescan_sequence_mismatch"
    try:
        for key, expected in (("ph_scale", "default-scale"),
                              ("surface_method", "FreeSASA-2.0.1-NACCESS-LR100"),
                              ("mode", "static-no-minimization")):
            if score.get(key) != expected:
                return "aggrescan_provenance_mismatch"
        if Decimal(str(score["radius_angstrom"])) != 10:
            return "aggrescan_provenance_mismatch"
        for key in ("input_structure_sha256", "normalized_structure_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(score.get(key, ""))):
                return "aggrescan_provenance_mismatch"
        if sequence is not None and (type(score.get("residue_count")) is not int
                                    or score["residue_count"] != len(sequence)):
            return "aggrescan_sequence_mismatch"
        for key, cap in (("a3d_average", MAX_AVERAGE), ("a3d_maximum", MAX_PEAK)):
            value = Decimal(str(score[key]))
            if not value.is_finite() or value > cap:
                return "aggrescan_threshold"
        if Decimal(str(score["a3d_maximum"])) < 0:
            return "aggrescan_threshold"
    except (InvalidOperation, KeyError, ValueError, TypeError):
        return "aggrescan_unavailable_or_failed"
    if result.get("passed") is not True:
        return "developability_failed"
    return ""


def validate_evidence(evidence, sequence, expected, config):
    if not isinstance(evidence, dict) or evidence.get("challenge") != expected:
        raise ValueError("missing or mismatched current-challenge evidence")
    if evidence.get("filter_sha") != FILTER_SHA:
        raise ValueError("filter version mismatch")
    requirements = config["nanobody_requirements"]
    reason = annotation_reason(evidence.get("annotation"), requirements,
                               sequence=sequence, epoch_challenge=expected)
    reason = reason or developability_reason(evidence.get("developability"), requirements,
                                            sequence=sequence)
    if reason:
        raise ValueError(reason)


def verified_fleet(payload, epoch, block_hash):
    """Filter to independently checked current-challenge checkpointed entries."""
    from compound_miner.nanobody_provider import validate_nanobody_checkpoint
    config = pinned_config(epoch)
    expected = challenge(config, epoch, block_hash)
    if (not isinstance(payload, dict) or payload.get("epoch") != epoch
            or payload.get("target") != expected["target"]):
        raise ValueError("nanobody fleet target or epoch is stale")
    if (not isinstance(payload.get("submissions", []), list)
            or any(not isinstance(e, dict) for e in payload.get("submissions", []))
            or not isinstance(payload.get("sequence_pool", []), list)
            or not isinstance(payload.get("sequence_pool_checkpoints", {}), dict)):
        raise ValueError("invalid fleet structure")
    checkpoints = dict(payload.get("sequence_pool_checkpoints") or {})
    for entry in payload.get("submissions") or []:
        checkpoints[entry.get("sequence")] = entry.get("checkpoint")
    valid = {}
    for sequence, checkpoint in checkpoints.items():
        try:
            found = validate_nanobody_checkpoint(
                checkpoint, expected_target=expected["target"], expected_epoch=epoch,
                min_sequence_length=config["nanobody_requirements"]["min_sequence_length"],
                max_sequence_length=config["nanobody_requirements"]["max_sequence_length"],
                epoch_start_block_hash=block_hash)
            if found != sequence or int(checkpoint["exact"]["observation_count"]) < 10:
                continue
        except (ValueError, TypeError, KeyError, AttributeError):
            continue
        valid[sequence] = checkpoint
    pool = [s for s in payload.get("sequence_pool", []) if isinstance(s, str) and s in valid]
    entries = [e for e in payload.get("submissions", []) if e.get("sequence") in valid]
    for entry in entries:
        if entry["sequence"] not in pool:
            pool.append(entry["sequence"])
    return {**payload, "sequence_pool": pool, "sequence_pool_size": len(pool),
            "submissions": entries, "submission_count": len(entries),
            "sequence_pool_checkpoints": valid}
