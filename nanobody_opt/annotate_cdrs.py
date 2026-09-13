"""Annotate ranked candidates with their CDRs, for fleet separation.

Whole-sequence identity is not the rule the validator dedupes on. Epoch 25100
committed seven tickets that were all under 0.95 to one another and still lost
five of them: their CDR3s were AVFLGAHPCPLEGEMDA against AVSLGAHPCPLEGEMDA and
GVFLGAHPCPLEGEMDA -- one or two residues apart in seventeen, so every pair sat
above the 0.85 clonotype threshold. Two tickets of one clonotype are one
ticket.

Runs in the metanano environment because that is where the annotator lives.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    from metanano.utils.cdr_utils import extract_cdrs

    payload = json.loads(args.input.read_text())
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise SystemExit(f"{args.input} holds no candidate list")

    annotated: dict[str, dict[str, str]] = {}
    unannotated = 0
    for candidate in candidates:
        sequence = str(candidate.get("sequence") or "").strip().upper()
        if not sequence or sequence in annotated:
            continue
        cdrs = extract_cdrs(sequence)
        if not cdrs:
            # Not fatal: a candidate we cannot annotate is compared on whole
            # sequence identity alone, which is the behaviour we had before.
            unannotated += 1
            continue
        annotated[sequence] = {
            key: str(cdrs.get(key) or "")
            for key in ("cdr1", "cdr2", "cdr3")
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"cdrs": annotated}, indent=2) + "\n")
    print(
        f"cdr annotation: {len(annotated)} of {len(candidates)} candidate(s) "
        f"annotated, {unannotated} could not be",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
