"""Pre-GPU near-duplicate gate: drop candidates that are near-dups of the fresh
top-50 *before* the race spends GPU on them.

Replicates the validator's two-tier is_duplicate against the top-50 historical:
tier 1 whole-sequence identity >= 0.95 (vectorized Hamming), tier 2 CDR3
identity >= 0.85. Runs under the metanano env (needs extract_cdrs for tier 2).
The top-50 index is build_top50_index.py's output ({"sequences", "cdr3s"}).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

WHOLE_SEQ_LIMIT = 0.95
CDR3_LIMIT = 0.85


def _near_duplicate(sequence: str, published: np.ndarray, limit: float = WHOLE_SEQ_LIMIT) -> bool:
    if published.size == 0 or not sequence:
        return False
    seq = np.frombuffer(sequence.encode(), dtype=np.uint8)
    if seq.shape[0] != published.shape[1]:
        return False
    max_mismatch = int((1.0 - limit) * seq.shape[0])
    return bool((published != seq).sum(axis=1).min() <= max_mismatch)


def _cdr3_identity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if len(left) != len(right):
        return 0.0
    return sum(a == b for a, b in zip(left, right)) / len(left)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top50", required=True, type=Path,
                        help="build_top50_index.py output ({\"sequences\", \"cdr3s\"})")
    parser.add_argument("--input", required=True, type=Path, help="layer2.json")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    top50 = json.loads(args.top50.read_text())
    top50_seqs = [str(s).strip().upper() for s in top50.get("sequences", [])]
    top50_cdr3s = [str(c).strip().upper() for c in top50.get("cdr3s", [])]
    published = (
        np.array([[ord(ch) for ch in s] for s in top50_seqs], dtype=np.uint8)
        if top50_seqs else np.array([], dtype=np.uint8).reshape(0, 0)
    )

    blob = json.loads(args.input.read_text())
    rows = blob.get("candidates") or blob.get("sequences") or []
    if isinstance(rows, dict):
        rows = [{"sequence": seq, **(meta if isinstance(meta, dict) else {})}
                for seq, meta in rows.items()]

    from metanano.utils.cdr_utils import extract_cdrs

    survivors: list[dict] = []
    dropped_whole = 0
    dropped_cdr3 = 0
    for row in rows:
        seq = str(row.get("sequence", "")).strip().upper()
        if not seq:
            continue
        if _near_duplicate(seq, published):
            dropped_whole += 1
            continue
        if top50_cdr3s:
            try:
                cdrs = extract_cdrs(seq)
            except Exception:
                cdrs = None
            cdr3 = str((cdrs or {}).get("cdr3") or "").strip().upper() if cdrs else ""
            if cdr3 and any(_cdr3_identity(cdr3, ref) >= CDR3_LIMIT for ref in top50_cdr3s):
                dropped_cdr3 += 1
                continue
        survivors.append(row)

    out = {**blob, "candidates": survivors,
           "sequences": [r.get("sequence") for r in survivors]}
    out.setdefault("dedup", {})["dropped_whole_seq"] = dropped_whole
    out.setdefault("dedup", {})["dropped_cdr3"] = dropped_cdr3
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"PRE_GPU_DEDUP in={len(rows)} out={len(survivors)} "
          f"dropped_whole={dropped_whole} dropped_cdr3={dropped_cdr3}")


if __name__ == "__main__":
    main()
