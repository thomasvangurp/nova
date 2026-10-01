#!/usr/bin/env python3
"""Freeze six production cohorts and temporally separated public fields."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DB = ROOT / "analysis/nanobody-gap/mutations.sqlite"
OLD_PROOF = ROOT / "analysis/nanobody-gap/h2-selector-proof-20260922"
TARGET = "P20809"
COHORT_EPOCHS = tuple(range(25247, 25253))
DEVELOPMENT_FIELD_EPOCHS = tuple(range(25231, 25247))
EVALUATION_FIELD_EPOCHS = tuple(range(25253, 25269))


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    connection = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    cohorts: dict[str, list[str]] = {}
    scores: dict[str, dict[str, float]] = {}
    for epoch in COHORT_EPOCHS:
        rows = connection.execute(
            "SELECT n.sequence, MIN(o.score) AS best "
            "FROM observations o JOIN nodes n ON n.id=o.node_id "
            "WHERE o.epoch=? AND o.source='live_exact_screen' "
            "AND n.target=? AND o.metrics_json IS NOT NULL "
            "GROUP BY n.sequence ORDER BY best ASC,n.sequence ASC LIMIT 300",
            (epoch, TARGET),
        ).fetchall()
        if len(rows) != 300:
            raise RuntimeError(f"epoch {epoch} has only {len(rows)} cohort candidates")
        cohorts[str(epoch)] = [str(sequence) for sequence, _ in rows]
        scores[str(epoch)] = {
            str(sequence): float(score) for sequence, score in rows
        }

    development_all = json.loads((OLD_PROOF / "development_fields.json").read_text())
    development = [
        field for field in development_all
        if int(field["epoch"]) in DEVELOPMENT_FIELD_EPOCHS
    ]
    if [int(field["epoch"]) for field in development] != list(DEVELOPMENT_FIELD_EPOCHS):
        raise RuntimeError("development field epoch mismatch")

    evaluation: list[dict] = []
    for epoch in range(25253, 25261):
        records = []
        for blob, in connection.execute(
            "SELECT o.metrics_json FROM observations o "
            "JOIN nodes n ON n.id=o.node_id "
            "WHERE o.epoch=? AND o.source='public_validator_api' "
            "AND n.target=? AND o.metrics_json IS NOT NULL",
            (epoch, TARGET),
        ):
            record = json.loads(blob)
            if isinstance(record, dict):
                records.append(record)
        if len(records) < 40:
            raise RuntimeError(f"evaluation epoch {epoch} has only {len(records)} entries")
        evaluation.append({"epoch": epoch, "target": TARGET, "records": records})
    for epoch in range(25261, 25269):
        evaluation.append(
            json.loads(
                (OLD_PROOF / "reserved_fields" / f"epoch_{epoch}.json").read_text()
            )
        )
    if [int(field["epoch"]) for field in evaluation] != list(EVALUATION_FIELD_EPOCHS):
        raise RuntimeError("evaluation field epoch mismatch")

    background = [
        str(sequence)
        for sequence, in connection.execute(
            "SELECT DISTINCT n.sequence FROM observations o "
            "JOIN nodes n ON n.id=o.node_id "
            "WHERE o.epoch BETWEEN ? AND ? "
            "AND o.source='public_validator_api' AND n.target=?",
            (DEVELOPMENT_FIELD_EPOCHS[0], DEVELOPMENT_FIELD_EPOCHS[-1], TARGET),
        )
    ]
    if len(background) < 500:
        raise RuntimeError("insufficient development sequences for consensus")

    union = sorted({sequence for cohort in cohorts.values() for sequence in cohort})
    if len(union) < 900:
        raise RuntimeError("cohorts do not provide enough independent sequence coverage")
    lengths = {len(sequence) for sequence in union}
    if len(lengths) != 1:
        raise RuntimeError(f"mixed sequence lengths: {sorted(lengths)}")

    dump(HERE / "cohorts.json", {
        "target": TARGET,
        "cohort_epochs": list(COHORT_EPOCHS),
        "selection": "best 300 distinct live_exact_screen sequences by original score",
        "cohorts": cohorts,
        "original_screen_scores": scores,
    })
    dump(HERE / "development_sequences.json", {"target": TARGET, "sequences": union})
    dump(HERE / "development_fields.json", development)
    dump(HERE / "evaluation_fields.json", evaluation)
    length = len(union[0])
    usable = [sequence for sequence in background if len(sequence) == length]
    consensus = {}
    for position in range(length):
        counts: dict[str, int] = {}
        for sequence in usable:
            counts[sequence[position]] = counts.get(sequence[position], 0) + 1
        residue, hits = max(counts.items(), key=lambda item: (item[1], item[0]))
        consensus[str(position)] = [residue, hits / len(usable)]
    dump(HERE / "development_consensus.json", {
        "sequence_count": len(usable), "threshold": 0.8, "positions": consensus,
    })
    overlap = {}
    for left_index, left in enumerate(COHORT_EPOCHS):
        for right in COHORT_EPOCHS[left_index + 1:]:
            a, b = set(cohorts[str(left)]), set(cohorts[str(right)])
            overlap[f"{left}:{right}"] = {
                "intersection": len(a & b),
                "jaccard": len(a & b) / len(a | b),
            }
    manifest = {
        "experiment": "prospective-multicohort-selector-allocation-proof-v1",
        "target": TARGET,
        "cohort_count": len(cohorts),
        "candidate_count_per_cohort": 300,
        "unique_development_sequences": len(union),
        "development_draws": 12,
        "fresh_draws": 32,
        "development_field_epochs": list(DEVELOPMENT_FIELD_EPOCHS),
        "evaluation_field_epochs": list(EVALUATION_FIELD_EPOCHS),
        "cohort_overlap": overlap,
    }
    dump(HERE / "experiment.json", manifest)
    frozen = [
        "cohorts.json", "development_sequences.json", "development_fields.json",
        "evaluation_fields.json", "development_consensus.json", "experiment.json", "select_fleets.py",
        "analyze_fresh.py", "run_remote.sh", "PREREGISTRATION.md",
    ]
    (HERE / "FREEZE_MANIFEST.sha256").write_text(
        "".join(f"{sha256(HERE / name)}  {name}\n" for name in frozen)
    )
    print(
        f"FROZEN cohorts={len(cohorts)} unique={len(union)} "
        f"development_fields={len(development)} evaluation_fields={len(evaluation)}"
    )


if __name__ == "__main__":
    main()
