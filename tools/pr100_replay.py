"""Protocol-scoped PR100 replay records, separate from legacy fast multipass.

The reference is one frozen, same-epoch rival field. Its scores are NOT the
historical ten-epoch multipass scale and do not estimate joint win probability.
This module uses only the standard library so Pi owns preparation/ingestion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import statistics
import tempfile
import os

SCHEMA = "pr100-frozen-rival-replay-v1"
MEASUREMENT = "full_protocol_frozen_same_epoch_rivals"
METRICS = {
    "design_iiptm": "max", "design_ptm": "max", "design_to_target_iptm": "max",
    "min_design_to_target_pae": "min", "interaction_pae": "min",
    "plip_hbonds_refolded": "max", "plip_saltbridge_refolded": "max",
    "delta_sasa_refolded": "max", "liability_score": "min",
    "liability_num_violations": "min",
}
SETTINGS = {"protocol": "nanobody-anything", "design_sampling_steps": 500,
            "fold_sampling_steps": 200, "fold_diffusion_samples": 5,
            "step_scale": 2.0, "noise_scale": 0.88, "num_designs": 1,
            "skip_inverse_folding": True, "synthetic_design": False}
TARGET_CONTEXT = {"nanobody_target": ["Q9NZQ7"], "nanobody_target_clip_interval": [[17, 132]],
                  "nanobody_structure": "8AOK", "nanobody_structure_chain": "A",
                  "nanobody_structure_res_index": "1..115",
                  "nanobody_structure_binding_site": "37,39,41..44,46,49,51,56,59,94..96,98,100,104..108",
                  "boltzgen_rank_by": "rank_sum", "boltzgen_rank_mode": "min"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, sort_keys=True, allow_nan=False)
            f.flush(); os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def metrics(value):
    if not isinstance(value, dict) or any(
        isinstance(value.get(k), bool) or not isinstance(value.get(k), (int, float))
        or not math.isfinite(value[k]) for k in METRICS
    ):
        raise ValueError("incomplete_or_nonfinite_metrics")
    return {k: float(value[k]) for k in METRICS}


def prepare(cohort, commitments, board, epoch, protocol):
    """Require external submission proof; never infer the cohort from a board."""
    if epoch < 25955 or cohort.get("epoch") != epoch or cohort.get("target") != "Q9NZQ7":
        raise ValueError("wrong_epoch_or_target")
    pool = cohort.get("sequence_pool", [])
    by_hash = {hashlib.sha256(s.encode("ascii")).hexdigest(): s for s in pool}
    if len(pool) != len(by_hash) or not pool:
        raise ValueError("duplicate_or_empty_cohort")
    if (commitments.get("epoch") != epoch or not commitments.get("source")
            or not commitments.get("verified") is True):
        raise ValueError("verified_submission_proof_required")
    submitted = commitments["submitted"]
    if (len({r["uid"] for r in submitted}) != len(submitted)
            or {r["sequence_sha256"] for r in submitted} != set(by_hash)):
        raise ValueError("submission_cohort_mismatch")
    ours = {r["uid"] for r in submitted}
    accepted = set()
    rivals = []
    for row in board:
        for item in row.get("nanobodies", []):
            if item.get("protein_name") != "Q9NZQ7":
                continue
            if row["uid"] in ours:
                actual = hashlib.sha256(item["sequence"].encode("ascii")).hexdigest()
                expected = next(r["sequence_sha256"] for r in submitted if r["uid"] == row["uid"])
                if actual != expected:
                    raise ValueError("accepted_sequence_mismatch")
                accepted.add(row["uid"])
            else:
                rivals.append({"uid": row["uid"], "metrics": metrics(item)})
    if not rivals or len({r["uid"] for r in rivals}) != len(rivals):
        raise ValueError("invalid_frozen_rival_field")
    candidates = [{**r, "sequence": by_hash[r["sequence_sha256"]],
                   "accepted_on_board": r["uid"] in accepted} for r in submitted]
    result = {"schema": SCHEMA, "epoch": epoch, "target": "Q9NZQ7",
              "passes_required": 10, "measurement": MEASUREMENT,
              "cohort": sorted(candidates, key=lambda r: r["uid"]),
              "submission_proof": commitments,
              "field": {"kind": "frozen_same_epoch_rivals", "epoch": epoch,
                        "target": "Q9NZQ7", "rows": sorted(rivals, key=lambda r: r["uid"])},
              "protocol": protocol,
              "interpretation": f"Conditional on {len(rivals)} frozen rival observations for epoch{epoch}; "
                 "not historical ten-epoch multipass, rival variability, or joint win probability."}
    result["field_sha256"] = digest(result["field"])
    result["cohort_sha256"] = digest(result["cohort"])
    result["protocol_sha256"] = digest(protocol)
    result["job_id"] = digest({k: result[k] for k in
                             ("epoch", "target", "field_sha256", "cohort_sha256", "protocol_sha256")})
    validate_manifest(result)
    return result


def validate_manifest(m):
    if (m.get("schema") != SCHEMA or m.get("measurement") != MEASUREMENT
            or m.get("target") != "Q9NZQ7" or m.get("epoch", 0) < 25955
            or m.get("passes_required") != 10):
        raise ValueError("invalid_replay_manifest")
    for name in ("field", "cohort", "protocol"):
        if m.get(name + "_sha256") != digest(m[name]):
            raise ValueError("manifest_identity_mismatch")
    expected = digest({k: m[k] for k in
                      ("epoch", "target", "field_sha256", "cohort_sha256", "protocol_sha256")})
    if m.get("job_id") != expected:
        raise ValueError("job_identity_mismatch")
    p = m["protocol"]
    if p.get("settings") != SETTINGS or p.get("seed_policy") != "independent_process_stage_seed_v1":
        raise ValueError("full_protocol_settings_required")
    if not p.get("runtime_identity") or not p.get("source_identity") or not p.get("model_identity"):
        raise ValueError("runtime_source_model_identity_required")
    if p.get("subnet_config") != TARGET_CONTEXT:
        raise ValueError("pinned_target_context_required")
    f = m["field"]
    if f.get("kind") != "frozen_same_epoch_rivals" or f.get("epoch") != m["epoch"] or f.get("target") != m["target"]:
        raise ValueError("frozen_field_context_mismatch")
    for row in f["rows"]:
        metrics(row["metrics"])
    if (not m["cohort"] or len({c["uid"] for c in m["cohort"]}) != len(m["cohort"])
            or not f["rows"] or len({r["uid"] for r in f["rows"]}) != len(f["rows"])
            or {c["uid"] for c in m["cohort"]} & {r["uid"] for r in f["rows"]}):
        raise ValueError("invalid_cohort_or_rival_membership")
    for candidate in m["cohort"]:
        if hashlib.sha256(candidate["sequence"].encode("ascii")).hexdigest() != candidate["sequence_sha256"]:
            raise ValueError("sequence_identity_mismatch")


def pass_seed(m, index):
    if isinstance(index, bool) or not isinstance(index, int) or not 1 <= index <= m["passes_required"]:
        raise ValueError("invalid_pass_index")
    return int(digest([m["job_id"], "independent-pass", index])[:8], 16)


def rank_sum(values, field):
    # Validator uses dense ranks. Duplicate competitor values occupy one rank.
    return float(sum(1 + sum(v < values[k] if direction == "min" else v > values[k]
                     for v in {r["metrics"][k] for r in field["rows"]})
                     for k, direction in METRICS.items()))


def observation(m, uid, index, values, *, artifact_sha256):
    validate_manifest(m)
    candidate = next(c for c in m["cohort"] if c["uid"] == uid)
    values = metrics(values)
    if not re.fullmatch(r"[0-9a-f]{64}", artifact_sha256):
        raise ValueError("artifact_identity_required")
    return {"schema": SCHEMA, "observation_id": digest([m["job_id"], uid, index]),
            "job_id": m["job_id"], "epoch": m["epoch"], "target": m["target"],
            "uid": uid, "sequence_sha256": candidate["sequence_sha256"],
            "pass_index": index, "seed": pass_seed(m, index), "observation_count": 1,
            "protocol_sha256": m["protocol_sha256"], "field_sha256": m["field_sha256"],
            "cohort_sha256": m["cohort_sha256"], "measurement": MEASUREMENT,
            "metrics": values, "score": rank_sum(values, m["field"]),
            "artifact_sha256": artifact_sha256}


def ingest(db, manifest, rows):
    """Strict idempotent ingestion; conflicting retry output cannot overwrite evidence.

    Dedicated table intentionally stays invisible to legacy query paths whose
    score_kind filter does not enforce protocol/runtime/field compatibility.
    """
    validate_manifest(manifest)
    validated = []
    for row in rows:
        expected = observation(manifest, row["uid"], row["pass_index"], row["metrics"],
                               artifact_sha256=row["artifact_sha256"])
        if row != expected:
            raise ValueError("incompatible_or_modified_observation")
        validated.append(row)
    with sqlite3.connect(db, timeout=10) as c:
        c.execute("CREATE TABLE IF NOT EXISTS pr100_replay_observations "
                  "(observation_id TEXT PRIMARY KEY, job_id TEXT NOT NULL, uid INTEGER NOT NULL, "
                  "pass_index INTEGER NOT NULL, payload TEXT NOT NULL, "
                  "UNIQUE(job_id,uid,pass_index))")
        for row in validated:
            encoded = json.dumps(row, sort_keys=True, allow_nan=False)
            prior = c.execute("SELECT payload FROM pr100_replay_observations WHERE observation_id=?",
                              (row["observation_id"],)).fetchone()
            if prior and prior[0] != encoded:
                raise ValueError("conflicting_pass_evidence")
            c.execute("INSERT OR IGNORE INTO pr100_replay_observations VALUES (?,?,?,?,?)",
                      (row["observation_id"], row["job_id"], row["uid"], row["pass_index"], encoded))
    return len(validated)


def summary(db, manifest):
    validate_manifest(manifest)
    with sqlite3.connect(db) as c:
        rows = [json.loads(r[0]) for r in c.execute(
            "SELECT payload FROM pr100_replay_observations WHERE job_id=?", (manifest["job_id"],))]
    by_uid = {}
    for candidate in manifest["cohort"]:
        selected = [r for r in rows if r["uid"] == candidate["uid"]]
        scores = [r["score"] for r in selected]
        by_uid[str(candidate["uid"])] = {
            "sequence_sha256": candidate["sequence_sha256"],
            "accepted_on_board": candidate["accepted_on_board"], "passes": len(selected),
            "complete": len(selected) == manifest["passes_required"],
            "mean": statistics.mean(scores) if scores else None,
            "stdev": statistics.stdev(scores) if len(scores) > 1 else None}
    complete = all(v["complete"] for v in by_uid.values())
    return {"schema": SCHEMA, "job_id": manifest["job_id"], "epoch": manifest["epoch"],
            "target": manifest["target"], "measurement": MEASUREMENT,
            "field_sha256": manifest["field_sha256"], "protocol_sha256": manifest["protocol_sha256"],
            "field_rows": len(manifest["field"]["rows"]), "cohort_size": len(by_uid),
            "coverage_complete": complete, "observations": len(rows), "candidates": by_uid,
            "fleet_mean": statistics.mean(v["mean"] for v in by_uid.values()) if complete else None,
            "native_eligibility_replayed": False,
            "includes_rejected_submissions": any(not c["accepted_on_board"] for c in manifest["cohort"]),
            "validator_worker_runtime_parity": "unattested",
            "historical_ten_epoch_comparable": False,
            "interpretation": manifest["interpretation"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--ingest", type=Path, nargs="*", default=[])
    ap.add_argument("--summary", type=Path, required=True)
    a = ap.parse_args()
    m = json.loads(a.manifest.read_text())
    for path in a.ingest:
        ingest(a.db, m, json.loads(path.read_text())["observations"])
    ingest(a.db, m, [])
    result = summary(a.db, m); atomic(a.summary, result)
    print(json.dumps({k: result[k] for k in ("epoch", "observations", "coverage_complete", "fleet_mean")}))


if __name__ == "__main__":
    main()
