#!/usr/bin/env python3
"""Select the best ~k VALID children for deep-scoring.

A child is valid only if it clears BOTH validator dedup gates:
  1. whole-sequence >= MIN_EDIT edits from every top-50 sequence (tier 1, 0.95),
  2. CDR3 positional identity < CDR3_SIM to every top-50 CDR3 (tier 2, 0.85).
Framework-only edits clear (1) but keep the parent's CDR3 and so fail (2); they
must be excluded.  This emits the top-k valid children by their existing
multi-pass mean score as a deep-score shard.

Invocation (metanano env):
  python nanobody_opt/select_valid_deep.py <top50_idx> <out_shard.json> [k=500] [shortlist=4000]
"""
import json
import sqlite3
import sys

import numpy as np

DB = "/root/artifacts/nanobody_mutations.sqlite"
MIN_EDIT = 7
CDR3_SIM = 0.85


def region_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    shared = sum(1 for a, b in zip(left, right) if a == b)
    return shared / max(len(left), len(right))


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    idx_path, out_path = sys.argv[1], sys.argv[2]
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    shortlist = int(sys.argv[4]) if len(sys.argv) > 4 else 4000

    idx = json.load(open(idx_path))
    top50_seqs = [s for s in idx.get("sequences", []) if len(s) == 127]
    top50_cdr3s = [c for c in idx.get("cdr3s", []) if c]
    T = np.array([[ord(c) for c in s] for s in top50_seqs], dtype=np.uint8)

    from metanano.utils.cdr_utils import extract_cdrs

    def dmin(seq: str) -> int:
        a = np.frombuffer(seq.encode(), dtype=np.uint8)
        return int((T != a).sum(axis=1).min())

    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = conn.execute(
        """SELECT n.sequence, AVG(o.score), MIN(o.epoch), SUM(o.observation_count)
           FROM observations o JOIN nodes n ON n.id = o.node_id
           WHERE o.score_kind = 'local_validator_replay_rank_sum' AND n.length = 127
           GROUP BY n.id
           HAVING SUM(o.observation_count) >= 8
           ORDER BY AVG(o.score) ASC""").fetchall()
    conn.close()
    print(f"multi-pass children (SUM(obs)>=8): {len(rows)}", flush=True)

    # gate 1: whole-seq
    whole_seq_valid = []
    for seq, mean, epoch, obs in rows:
        if dmin(seq) >= MIN_EDIT:
            whole_seq_valid.append((mean, seq, epoch, obs))
            if len(whole_seq_valid) >= shortlist:
                break
    print(f"whole-seq valid (>= {MIN_EDIT} edits): {len(whole_seq_valid)} (shortlisted)", flush=True)

    # gate 2: CDR3 novelty
    picks = []
    unextractable = 0
    for mean, seq, epoch, obs in whole_seq_valid:
        try:
            cdrs = extract_cdrs(seq)
        except Exception:
            cdrs = None
        c3 = (cdrs or {}).get("cdr3") if isinstance(cdrs, dict) else ""
        if not c3:
            unextractable += 1
            picks.append((mean, seq, epoch, obs))
            if len(picks) >= k:
                break
            continue
        novel = all(region_similarity(c3, other) < CDR3_SIM for other in top50_cdr3s)
        if novel:
            picks.append((mean, seq, epoch, obs))
            if len(picks) >= k:
                break
    json.dump([{"sequence": s, "epoch": e} for _, s, e, _ in picks], open(out_path, "w"))
    print(f"CDR3-novel valid: {len(picks)} (unextractable={unextractable})", flush=True)
    for mean, seq, epoch, obs in picks[:10]:
        print(f"   {mean:6.1f}  edits={dmin(seq)}  obs={obs}  {seq[:44]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
