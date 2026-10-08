#!/usr/bin/env python3
"""Shortlist children PER FAMILY (not globally), preserving bank diversity.

The global top-N shortlist re-collapses to the single best family, which is
exactly the narrowness that exhausted the old bank. Here each parent family
contributes its own best-k children by shallow mean score, so the deep-scored
pool spans every family.

Inputs:
  shallow.json    {hash: {sequence, epoch, p1..pN}}   (score_worker output)
  children.json   [{sequence, epoch, parent, family_id, position, to}]
Outputs:
  deep_shard.json [{sequence, epoch}]   -- union of per-family top-k
"""
import json
import statistics as st
import sys


def main() -> int:
    if len(sys.argv) < 5:
        print(__doc__, file=sys.stderr)
        return 2
    shallow_path, children_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    k = int(sys.argv[4])

    shallow = json.load(open(shallow_path))
    children = json.load(open(children_path))

    # sequence -> family_id
    family_of = {}
    for c in children:
        family_of[c["sequence"]] = c["family_id"]

    # per child: shallow mean (mean of whatever passes are present)
    per_family: dict[int, list[tuple[float, str, int]]] = {}
    for h, e in shallow.items():
        seq = e.get("sequence")
        if not seq or seq not in family_of:
            continue
        vals = [e[f"p{i}"] for i in range(1, 33) if f"p{i}" in e]
        if not vals:
            continue
        mean = st.mean(vals)
        epoch = e.get("epoch", 25456)
        per_family.setdefault(family_of[seq], []).append((mean, seq, epoch))

    families = sorted(per_family)
    picks: list[dict] = []
    for fid in families:
        rows = sorted(per_family[fid], key=lambda r: r[0])[:k]
        for mean, seq, epoch in rows:
            picks.append({"sequence": seq, "epoch": epoch})
    json.dump(picks, open(out_path, "w"))
    print(f"SHORTLIST_PER_FAMILY families={len(families)} picks={len(picks)} k={k}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
