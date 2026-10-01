#!/usr/bin/env python3
"""Build a balanced, previously unscored selector-calibration cohort.

Each group starts with the best top-miner and NOVA entries from one recent
epoch.  Three graph-unseen variants are generated around each anchor using
winner-enriched edits from ``nearest_missing_edits.csv``.  The resulting
groups test selection inside the local basins where the public contest is won,
without using a past score to choose the newly generated molecules.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sqlite3
from collections import defaultdict
from pathlib import Path


def load_known(database: Path, target: str) -> set[str]:
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        return {
            str(row[0])
            for row in connection.execute(
                "SELECT sequence FROM nodes WHERE target = ?", (target,)
            )
        }
    finally:
        connection.close()


def load_edits(path: Path) -> list[dict]:
    edits = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("subset") != "top10":
                continue
            edits.append(
                {
                    "position": int(row["position_zero"]),
                    "from": row["from"],
                    "to": row["to"],
                    "region": row["region"],
                    "count": int(row["count"]),
                }
            )
    edits.sort(key=lambda item: (-item["count"], item["position"], item["to"]))
    return edits


def mutate(sequence: str, edits: tuple[dict, ...]) -> str:
    residues = list(sequence)
    for edit in edits:
        residues[edit["position"]] = edit["to"]
    return "".join(residues)


def unseen_variants(
    sequence: str,
    edits: list[dict],
    known: set[str],
    occupied: set[str],
    count: int,
) -> list[tuple[str, tuple[dict, ...]]]:
    applicable = [
        edit
        for edit in edits
        if edit["position"] < len(sequence)
        and sequence[edit["position"]] == edit["from"]
        and edit["to"] != edit["from"]
    ]
    candidates: list[tuple[str, tuple[dict, ...]]] = []
    for depth in (1, 2, 3):
        for combination in itertools.combinations(applicable, depth):
            if len({edit["position"] for edit in combination}) != depth:
                continue
            variant = mutate(sequence, combination)
            if variant in known or variant in occupied:
                continue
            candidates.append((variant, combination))
            occupied.add(variant)
            if len(candidates) == count:
                return candidates
    raise ValueError(
        f"only generated {len(candidates)} unseen variants for {sequence[:12]}..."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--edits", required=True, type=Path)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target", default="P20809")
    args = parser.parse_args()

    replay = json.loads(args.replay.read_text())
    by_epoch: dict[int, list[dict]] = defaultdict(list)
    for item in replay:
        by_epoch[int(item["epoch"])].append(item)
    known = load_known(args.database, args.target)
    edits = load_edits(args.edits)
    anchors = {str(item["sequence"]) for item in replay}
    occupied = set(anchors)
    cases = []

    for epoch, items in sorted(by_epoch.items()):
        by_cohort = {str(item["cohort"]): item for item in items}
        if set(by_cohort) != {"TOP_MINERS", "US"}:
            raise ValueError(f"epoch {epoch} does not contain one rival/ours pair")
        group = f"epoch-{epoch}-basin"
        for cohort, side in (("TOP_MINERS", "rival"), ("US", "ours")):
            anchor = str(by_cohort[cohort]["sequence"])
            cases.append(
                {
                    "group": group,
                    "epoch": epoch,
                    "sequence": anchor,
                    "role": f"{side}_anchor",
                    "anchor_side": side,
                    "edit_count": 0,
                    "edits": [],
                    "seen_in_graph": anchor in known,
                }
            )
            for index, (variant, applied) in enumerate(
                unseen_variants(anchor, edits, known, occupied, 3), start=1
            ):
                cases.append(
                    {
                        "group": group,
                        "epoch": epoch,
                        "sequence": variant,
                        "role": f"{side}_unseen_{index}",
                        "anchor_side": side,
                        "edit_count": len(applied),
                        "edits": [
                            {
                                "position_zero": edit["position"],
                                "from": edit["from"],
                                "to": edit["to"],
                                "region": edit["region"],
                                "winner_count": edit["count"],
                            }
                            for edit in applied
                        ],
                        "seen_in_graph": False,
                    }
                )

    sequences = [item["sequence"] for item in cases]
    if len(sequences) != len(set(sequences)):
        raise ValueError("cohort unexpectedly contains duplicate sequences")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "target": args.target,
        "cohort": "recent-rival-ours-basins-with-unseen-haplotypes",
        "groups": len(by_epoch),
        "candidate_count": len(cases),
        "unseen_candidate_count": sum(not item["seen_in_graph"] for item in cases),
        "cases": cases,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (args.output_dir / "sequences.json").write_text(
        json.dumps({"target": args.target, "sequences": sequences}, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "groups": manifest["groups"],
                "candidates": manifest["candidate_count"],
                "unseen": manifest["unseen_candidate_count"],
                "lengths": sorted({len(sequence) for sequence in sequences}),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
