#!/usr/bin/env python3
"""Generate a DIVERSE children bank from the public leaderboard.

The current children_20x bank is one tight lineage: it was built from the
top-12% elite parents, which -- because the field is one dominant clonotype --
is itself one family. Once that family entered the top-50 (our 25453 win), every
single-edit child of it became a >=0.95 whole-sequence duplicate and was
rejected wholesale (25454, 25455). The public pool actually holds ~34 distinct
families; this module mines the OTHER ~33.

Parent selection is diversity-first: a parent must be >= MARGIN edits from every
top-50 sequence AND from every already-picked parent (greedy, best score first,
bottom quartile excluded). A single-edit child then sits >= MARGIN-1 edits from
the top-50 -- safely outside the validator's 0.95 (<=6-edit) dedup line -- and
different families stay independent even after one of them wins.

Outputs two files:
  parents.json   [{sequence, family_id, final_score, epoch, dist_top50}]
  children.json  [{sequence, epoch, parent, family_id, position, to}]

Invocation (metanano env, numpy available):
  python nanobody_opt/diverse_children.py <out_dir> [epochs=14] [margin=8]
"""
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

API = "https://compound-api.metanova-labs.ai"
TARGET = "P20809"
LEN = 127

# Framework single-substitution positions x best residues, MEASURED by the
# exhaustive sweep (scoring_backfill/exhaustive_ranking.json). Pos 114 omitted
# (its best residue scores worse than most positions' worst).
HOT = {
    0: "ADQGN", 2: "AMVEK", 72: "ATDVQ", 110: "EDQKA", 113: "HMFVY",
    115: "ASTEH", 120: "ATSVP", 121: "APSVG", 122: "VATIL", 123: "ASPGV",
    125: "GASPT", 126: "APGES",
}


def _hamming(mat: np.ndarray, seq: str) -> np.ndarray:
    a = np.frombuffer(seq.encode(), dtype=np.uint8)
    return (mat != a).sum(axis=1)


def fetch_public(epochs: int) -> dict[str, dict]:
    pub: dict[str, dict] = {}
    for epoch in range(_current_epoch(), _current_epoch() - epochs, -1):
        try:
            req = urllib.request.Request(
                f"{API}/api/competitions/leaderboard/{epoch}/nanobodies",
                headers={"User-Agent": "nova-diverse-children/1"})
            payload = json.load(urllib.request.urlopen(req, timeout=30))
        except Exception:
            continue
        for sub in payload.get("data") or []:
            for nb in sub.get("nanobodies") or []:
                if nb.get("protein_name") != TARGET:
                    continue
                s = str(nb.get("sequence") or "").strip().upper()
                if len(s) != LEN or s in pub:
                    continue
                try:
                    score = float(nb["final_score"])
                except (KeyError, TypeError, ValueError):
                    score = None
                pub[s] = {"sequence": s, "final_score": score, "epoch": epoch}
    return pub


def _current_epoch() -> int:
    # chain state on the production box, else a recent default
    try:
        return int(json.load(open("/root/artifacts/chain_state.json"))["epoch"])
    except Exception:
        return 25456


def load_top50(path: str) -> np.ndarray:
    try:
        payload = json.load(open(path))
    except (OSError, json.JSONDecodeError):
        return np.zeros((0, LEN), dtype=np.uint8)
    seqs = payload.get("sequences") or []
    rows = [[ord(c) for c in s] for s in seqs if len(s) == LEN]
    return np.array(rows, dtype=np.uint8) if rows else np.zeros((0, LEN), dtype=np.uint8)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    out_dir = Path(sys.argv[1])
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    margin = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    out_dir.mkdir(parents=True, exist_ok=True)

    pub = fetch_public(epochs)
    print(f"public={len(pub)}", flush=True)
    top50 = load_top50("/root/artifacts/top50_cdr3_index.json")
    print(f"top50={top50.shape[0]}", flush=True)

    # candidates: scored, >= margin edits from the top-50, above the bottom quartile
    scored = [(v["final_score"], s) for s, v in pub.items() if v["final_score"] is not None]
    scored.sort()
    if not scored:
        return 1
    cutoff = scored[len(scored) // 4][0]  # drop bottom quartile (garbage)
    cands = []
    for score, s in scored:
        if score > cutoff:
            continue
        if top50.size and _hamming(top50, s).min() < margin:
            continue
        cands.append((score, s))
    print(f"diverse_candidates={len(cands)} (>= {margin} edits from top-50, above Q1)", flush=True)

    # greedy: best score first, exclude everything within (margin-1) edits of a pick
    all_mat = np.array([[ord(c) for c in s] for _, s in cands], dtype=np.uint8)
    parents: list[dict] = []
    taken_idx: set[int] = set()
    for idx, (score, s) in enumerate(cands):
        if idx in taken_idx:
            continue
        d = _hamming(all_mat, s)
        parents.append({"sequence": s, "family_id": len(parents),
                        "final_score": score, "epoch": pub[s]["epoch"],
                        "dist_top50": int(_hamming(top50, s).min()) if top50.size else None})
        # mark this parent and every candidate within (margin-1) edits as used
        for j, dj in enumerate(d):
            if dj < margin - 1:
                taken_idx.add(j)
    print(f"diverse_parents={len(parents)}", flush=True)
    json.dump(parents, open(out_dir / "parents.json", "w"))

    # generate framework single-edits for each parent
    parent_set = {p["sequence"] for p in parents}
    children: list[dict] = []
    seen: set[str] = set(parent_set)
    for p in parents:
        seq = p["sequence"]
        for pos, tos in HOT.items():
            if pos >= len(seq):
                continue
            for to in tos:
                if seq[pos] == to:
                    continue
                child = seq[:pos] + to + seq[pos + 1:]
                if child in seen:
                    continue
                seen.add(child)
                children.append({"sequence": child, "epoch": p["epoch"],
                                 "parent": seq, "family_id": p["family_id"],
                                 "position": pos, "to": to})
    print(f"diverse_children={len(children)}", flush=True)
    json.dump(children, open(out_dir / "children.json", "w"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
