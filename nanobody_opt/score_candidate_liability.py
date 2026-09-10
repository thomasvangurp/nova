"""Attach the exact live BoltzGen peptide-liability score to candidate JSON."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any


def attach_liability(
    cases: list[dict[str, Any]], scores: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Copy exact scores onto cases and return a stable best-first ordering."""
    measured = []
    for case in cases:
        sequence = str(case["sequence"])
        result = scores.get(sequence)
        if not isinstance(result, dict) or "score" not in result:
            raise ValueError(f"missing liability result for {sequence}")
        violations = result.get("violations")
        if not isinstance(violations, list):
            raise ValueError(f"invalid liability violations for {sequence}")
        measured.append({
            **case,
            "liability_score": float(result["score"]),
            "liability_num_violations": len(violations),
        })
    return sorted(measured, key=lambda item: (
        item["liability_score"],
        item["liability_num_violations"],
        item.get("donor_rank_sum", float("inf")),
        item["sequence"],
    ))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text())
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise SystemExit("input has no candidate cases")
    sequences = [str(case["sequence"]) for case in cases]

    from boltzgen.task.analyze.analyze_utils import compute_liability_scores

    scores = compute_liability_scores(
        sequences, modality="peptide", peptide_type="linear"
    )
    measured = attach_liability(cases, scores)
    if args.limit:
        measured = measured[: args.limit]
    output = {
        **payload,
        "purpose": f"{payload.get('purpose', 'candidate')} + exact liability",
        "liability_modality": "peptide",
        "liability_peptide_type": "linear",
        "input_case_count": len(cases),
        "cases": measured,
    }
    _atomic_json(args.output, output)
    best = measured[0]
    print(
        f"CANDIDATE_LIABILITY_READY input={len(cases)} output={len(measured)} "
        f"best={best['liability_score']:.0f}/"
        f"{best['liability_num_violations']} path={args.output}"
    )


if __name__ == "__main__":
    main()
