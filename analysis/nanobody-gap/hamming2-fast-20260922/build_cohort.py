#!/usr/bin/env python3
"""Build the deterministic empirical Hamming-2 closure used by this replay."""

from __future__ import annotations

import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
GAP = HERE.parent
REPO = GAP.parents[1]
PRIOR = GAP / "monte-carlo-20260921" / "manifest.json"
EDITS = GAP / "results" / "nearest_missing_edits.csv"
EASYNANO = REPO / "analysis" / "easynano-p20809" / "experiment.json"
EASYNANO_SUMMARY = (
    REPO
    / "analysis"
    / "easynano-p20809"
    / "results"
    / "artifacts"
    / "monte_carlo_summary.json"
)


def mutate(sequence: str, edits: list[dict]) -> str:
    chars = list(sequence)
    for edit in edits:
        pos = int(edit["position_zero"])
        if chars[pos] != edit["from"]:
            raise ValueError(f"edit mismatch at {pos}: {chars[pos]} != {edit['from']}")
        chars[pos] = edit["to"]
    return "".join(chars)


def main() -> None:
    prior = json.loads(PRIOR.read_text())
    anchors = []
    seen_anchor = set()
    for case in prior["cases"]:
        if case["edit_count"] != 0 or case["sequence"] in seen_anchor:
            continue
        seen_anchor.add(case["sequence"])
        anchors.append(case)
    if len(anchors) != 24:
        raise RuntimeError(f"expected 24 distinct basin anchors, found {len(anchors)}")

    with EDITS.open(newline="") as handle:
        empirical = [row for row in csv.DictReader(handle) if row["subset"] == "wins"][:20]
    for row in empirical:
        row["position_zero"] = int(row["position_zero"])
        row["position_one"] = int(row["position_one"])
        row["count"] = int(row["count"])
        row["entry_rate"] = float(row["entry_rate"])

    provenance: dict[str, list[dict]] = defaultdict(list)
    for anchor in anchors:
        sequence = anchor["sequence"]
        applicable = [e for e in empirical if sequence[e["position_zero"]] == e["from"]]
        provenance[sequence].append({
            "group": anchor["group"], "epoch": anchor["epoch"],
            "anchor_side": anchor["anchor_side"], "edit_count": 0, "edits": [],
        })
        for edit in applicable:
            candidate = mutate(sequence, [edit])
            provenance[candidate].append({
                "group": anchor["group"], "epoch": anchor["epoch"],
                "anchor_side": anchor["anchor_side"], "edit_count": 1,
                "edits": [edit],
            })
        for first, second in itertools.combinations(applicable, 2):
            if first["position_zero"] == second["position_zero"]:
                continue
            candidate = mutate(sequence, [first, second])
            provenance[candidate].append({
                "group": anchor["group"], "epoch": anchor["epoch"],
                "anchor_side": anchor["anchor_side"], "edit_count": 2,
                "edits": [first, second],
            })

    closure_sequences = list(provenance)
    if len(closure_sequences) != 687:
        raise RuntimeError(f"expected 687 unique closure sequences, found {len(closure_sequences)}")

    experiment = json.loads(EASYNANO.read_text())
    summary = json.loads(EASYNANO_SUMMARY.read_text())
    ours = next(x for x in experiment["anchors"] if x["cohort"] == "ours")
    rival = next(x for x in experiment["anchors"] if x["cohort"] == "competitor")
    proposal = next(
        x for x in summary["ranking"]
        if x["anchor_id"] == ours["id"] and x["hamming_from_anchor"] == 1
    )
    easynano = [
        {"id": "ours_anchor", "sequence": ours["sequence"], "kind": "anchor"},
        {"id": "easynano_h1", "sequence": proposal["sequence"], "kind": "proposal"},
        {"id": "rival_anchor", "sequence": rival["sequence"], "kind": "anchor"},
    ]

    combined = list(closure_sequences)
    for case in easynano:
        if case["sequence"] not in provenance:
            combined.append(case["sequence"])
        provenance[case["sequence"]].append({
            "group": "easynano-three-way", "anchor_side": case["id"],
            "edit_count": 0 if case["kind"] == "anchor" else 1,
            "edits": [],
        })

    cases = [
        {"sequence": sequence, "provenance": provenance[sequence]}
        for sequence in combined
    ]
    manifest = {
        "target": "P20809",
        "definition": (
            "all compatible Hamming-0/1/2 combinations of the top 20 "
            "public-win-associated single substitutions around 24 recent "
            "matched ours/rival basin anchors"
        ),
        "anchor_count": len(anchors),
        "empirical_substitution_count": len(empirical),
        "closure_unique_sequences": len(closure_sequences),
        "combined_unique_sequences": len(combined),
        "passes": 12,
        "easynano_extra_passes": 8,
        "empirical_substitutions": empirical,
        "easynano_cases": easynano,
        "cases": cases,
    }
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (HERE / "sequences.json").write_text(json.dumps({
        "target": "P20809", "sequences": combined,
    }, indent=2) + "\n")
    (HERE / "easynano_sequences.json").write_text(json.dumps({
        "target": "P20809", "sequences": [x["sequence"] for x in easynano],
    }, indent=2) + "\n")
    print(json.dumps({
        "anchors": len(anchors), "closure": len(closure_sequences),
        "combined": len(combined), "easynano": len(easynano),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
