"""Measured PR100 annotation/developability, cached independently of epochs."""
from __future__ import annotations

import asyncio
import ast
import hashlib
import importlib.util
import importlib.metadata
import inspect
import json
import os
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

from nanobody_opt import rules

ROOT = Path(__file__).resolve().parents[1]

# Measured in the isolated runtime built from PR100's main requirements.
# Consumers can verify this identity without installing annotation libraries.
ANNOTATION_RUNTIME = {
    "schema": "pr100-annotation-v2",
    "packages": {"abnumber": "0.4.4", "anarci": "2026.2.13.2",
                 "anarcii": "2.0.5", "biopython": "1.86"},
    "panel_sha256": "91813d37e96b23606652694c22f93051cd3e396169d23a7678fb9b96ad1146ef",
    "hmm_sha256": "60051f5a838704eb97fe39488f23894aa335ddadd24ad72b241f6e1cea6829ee",
    "hmmscan_sha256": "f225ab9d9c12bc2bb90b7b1994fd3de49180278df1bbe2a15002096e91cce480",
    "scheme": "imgt", "cdr_definition": "imgt", "use_anarcii": False,
}


def activate_filter():
    """Select the packaged PR100 backend without changing the legacy runtime."""
    import sys
    import importlib
    package = ROOT / "NOVA-nanobody-filter"
    if not (package / "metanano/config.py").is_file():
        raise ValueError("pinned PR100 filter package is missing")
    sys.path.insert(0, str(package))
    module = importlib.import_module("metanano")
    if not Path(module.__file__).resolve().is_relative_to(package.resolve()):
        raise ValueError("a different metanano backend was already imported")


@lru_cache(maxsize=None)
def helper(name):
    # Import the pinned upstream helpers without utils/__init__'s RDKit stack.
    spec = importlib.util.spec_from_file_location("_pr100_" + name, ROOT / "utils" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, allow_nan=False)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@lru_cache(maxsize=1)
def annotation_identity():
    # Cache identity follows scientific extraction semantics, not unrelated
    # developability/cache edits elsewhere in this adapter.
    extraction = annotation_ast_dump(ast.parse(inspect.getsource(_annotate_uncached)))
    return rules.digest({"runtime": ANNOTATION_RUNTIME, "extraction": extraction,
        "source": hashlib.sha256((ROOT / "utils/germline_identity.py").read_bytes()).hexdigest()})


def annotation_ast_dump(tree):
    """Keep the existing Python-3.12 identity when the Pi reads it on 3.11."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if 'type_params' not in node._fields:
                node._fields = (*node._fields, 'type_params')
                node.type_params = []
    return ast.dump(tree, include_attributes=False)


@lru_cache(maxsize=1)
def validate_annotation_runtime():
    for package, expected in ANNOTATION_RUNTIME["packages"].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError("annotation dependency version mismatch: " + package)
    germline = helper("germline_identity")
    _, panel = germline._anarci_panel("human")
    binary = shutil.which("hmmscan")
    if (panel != ANNOTATION_RUNTIME["panel_sha256"] or
            germline._hmm_digest() != ANNOTATION_RUNTIME["hmm_sha256"] or
            not binary or hashlib.sha256(Path(binary).read_bytes()).hexdigest()
            != ANNOTATION_RUNTIME["hmmscan_sha256"]):
        raise ValueError("annotation reference or HMMER binary mismatch")


def _annotate_uncached(sequence):
    from abnumber import Chain
    germline = helper("germline_identity").human_germline_identity_anarci(sequence)
    chain = Chain(sequence, scheme="imgt", cdr_definition="imgt", use_anarcii=False)
    return {"profile": rules.PROFILE, "sequence_hash": rules.sequence_hash(sequence),
            "status": "ok", "germline": germline,
            "single_heavy_domain": chain.chain_type == "H", "scheme": chain.scheme,
            "vhh_hallmarks": chain["H49"] in {"E", "Q"} and chain["H50"] == "R",
            "canonical_cysteines": chain["H23"] == "C" and chain["H104"] == "C",
            "cdr3_length": len(chain.cdr3_seq)}


def annotate(sequence, cache_dir):
    validate_annotation_runtime()
    identity = annotation_identity()
    key = rules.sequence_hash(sequence)
    path = Path(cache_dir) / "pr100" / "annotation" / identity / (key + ".json")
    try:
        result = json.loads(path.read_text())
        if (result.get("sequence_hash") == key and result.get("annotation_identity") == identity
                and result.get("status") == "ok"):
            return result
    except (OSError, ValueError, TypeError):
        pass
    try:
        result = {**_annotate_uncached(sequence), "annotation_identity": identity}
    except Exception as error:
        # Execution failures are not durable biological rejections.
        return {"profile": rules.PROFILE, "sequence_hash": key, "status": "error",
                "error_type": type(error).__name__}
    atomic(path, result)
    return result


async def develop(sequences, concurrency):
    activate_filter()
    from metanano.services.async_manager import AsyncServiceManager
    from metanano.services.developability_service import DevelopabilityService
    config = helper("aggrescan3d").developability_config()
    config.async_config.max_concurrent_tnp = concurrency
    config.async_config.batch_size = max(1, len(sequences))
    manager = AsyncServiceManager(config.async_config)
    try:
        result = await DevelopabilityService(config.developability, manager).analyze_batch_async(sequences)
        if not isinstance(result, list) or len(result) != len(sequences):
            raise ValueError("incomplete developability result batch")
        return result
    finally:
        await manager.shutdown()


def cached_result(sequence, cache_dir):
    key = rules.sequence_hash(sequence)
    path = Path(cache_dir) / "pr100" / "developability" / rules.FILTER_SHA / (key + ".json")
    try:
        item = json.loads(path.read_text())
        value = item["developability"]
        if (item.get("sequence_hash") != key or item.get("filter_sha") != rules.FILTER_SHA
                or not isinstance(value, dict) or value.get("error")
                or (value.get("aggrescan3d") or {}).get("status") != "ok"):
            return None
        reason = rules.developability_reason(value, rules.pinned_config(rules.ACTIVATION_EPOCH)["nanobody_requirements"],
                                            sequence=sequence)
        return {**value, "passed": not reason}
    except (OSError, ValueError, TypeError, KeyError):
        return None


def cached_develop(rows, cache_dir, concurrency):
    # Legacy TNP passes never satisfy the new aggregation checks.
    cache = Path(cache_dir) / "pr100" / "developability" / rules.FILTER_SHA
    found = {}; missing = []
    for index, row in enumerate(rows):
        seq = row["sequence"]; key = rules.sequence_hash(seq)
        try:
            item = json.loads((cache / (key + ".json")).read_text())
            if item.get("sequence_hash") == key and item.get("filter_sha") == rules.FILTER_SHA:
                value = item["developability"]
                if (isinstance(value, dict) and not value.get("error")
                        and (value.get("aggrescan3d") or {}).get("status") == "ok"):
                    found[index] = value
                    continue
        except (OSError, ValueError, TypeError, KeyError):
            pass
        missing.append(index)
    if missing:
        values = asyncio.run(develop([rows[i]["sequence"] for i in missing], concurrency))
        for index, result in zip(missing, values, strict=True):
            found[index] = result
            score = result.get("aggrescan3d") if isinstance(result, dict) else None
            if isinstance(score, dict) and score.get("status") == "ok" and not result.get("error"):
                key = rules.sequence_hash(rows[index]["sequence"])
                atomic(cache / (key + ".json"), {"sequence_hash": key,
                       "filter_sha": rules.FILTER_SHA, "developability": result})
    return [found[i] for i in range(len(rows))]


def run(rows, config, args):
    from nanobody_opt.gates import check_sequence
    requirements = config["nanobody_requirements"]
    offline = bool(getattr(args, "all_challenges", False))
    if offline and not args.igblast_only:
        raise ValueError("all-challenges preparation cannot certify publication")
    expected = None if offline else rules.context(config)
    survivors = []; rejected = []; counts = {}
    for row in rows:
        sequence = row["sequence"]
        annotation = annotate(sequence, args.cache_dir)
        reason = ("basic_sequence" if check_sequence(sequence, requirements) else
                  rules.annotation_reason(annotation, requirements, sequence=sequence,
                                          epoch_challenge=expected))
        record = {**row, "pr100_annotation": annotation, "sequence_gate_method": rules.PROFILE}
        if reason:
            record.update(layer2_passed=False, layer2_rejection=reason)
            rejected.append(record); counts[reason] = counts.get(reason, 0) + 1
        else:
            # Compatibility alias used by the existing cheap-precheck caller.
            record.update(igblast_passed=True, sequence_gate_passed=True)
            survivors.append(record)
    if not args.igblast_only and survivors:
        results = cached_develop(survivors, args.cache_dir, max(1, min(200, args.concurrency)))
        passing = []
        for row, result in zip(survivors, results, strict=True):
            reason = rules.developability_reason(result, requirements, sequence=row["sequence"])
            row.update(developability=result, layer2_passed=not reason,
                       pr100_evidence={"challenge": expected, "filter_sha": rules.FILTER_SHA,
                                       "annotation": row["pr100_annotation"], "developability": result})
            if reason:
                row["layer2_rejection"] = reason; rejected.append(row)
                counts[reason] = counts.get(reason, 0) + 1
            else:
                passing.append(row)
        survivors = passing
    output = {"target": config["protein_selection"]["nanobody"]["target"],
              "challenge": expected, "ruleset_sha256": rules.fingerprint(config),
              "all_challenges_preparation": offline, "igblast_only": args.igblast_only,
              "input_count": len(rows), "pass_count": len(survivors),
              "thresholds": requirements, "rejection_counts": counts,
              "sequences": [row["sequence"] for row in survivors],
              "candidates": survivors, "rejected": rejected}
    atomic(args.output, output)
    print(json.dumps({"stage": rules.PROFILE, "input": len(rows), "passed": len(survivors),
                      "rejected": counts, "all_challenges_preparation": offline}))
