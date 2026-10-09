#!/usr/bin/env python3
"""Re-aggregate an already-scored multi-pass run after the script crashed.

The 10 pass outputs (score{1..10}/exact_all.json) are already on disk; only the
final aggregation (rank-sum mean/stdev/fleet_size per coldkey) failed because a
rejected sequence was missing from one pass. This re-reads the pass outputs and
re-runs the aggregation with the missing-sequence fix, without re-scoring.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/root")
from multipass_all_rivals import (  # noqa: E402
    METRICS_NEEDED, OUR_COLDKEY, TARGET, algo2_score, fetch_leaderboard,
)


def main() -> int:
    epoch = int(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
        f"/root/multipass_all_rivals/{epoch}")
    archive_path = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(
        f"/root/live/epoch{epoch}/producer/archive.csv")
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 10

    by_coldkey: dict[str, list[str]] = {}
    for coldkey, seq in fetch_leaderboard(epoch):
        by_coldkey.setdefault(coldkey, []).append(seq)

    archive = pd.read_csv(archive_path).drop_duplicates("sequence", keep="last")
    archive["sequence"] = archive["sequence"].astype(str).str.upper()

    per_pass: list[dict[str, float]] = []
    for run in range(1, n + 1):
        payload = json.loads((out / f"score{run}" / "exact_all.json").read_text())
        wave = payload["per_nanobody_components"]["0"]
        rows = {}
        for seq, comp in wave.items():
            m = comp.get(TARGET, {})
            rows[seq] = {k: float(v) for k, v in m.items()
                         if k in METRICS_NEEDED and isinstance(v, (int, float))}
        rival = pd.DataFrame(rows).T  # index = sequence
        archive_s = archive.set_index("sequence")[METRICS_NEEDED]
        field = pd.concat([archive_s, rival])  # index = sequence
        scores = algo2_score(field)
        per_pass.append({seq: float(scores[seq])
                         for seq in rival.index if seq in scores.index})
        print(f"pass {run}/{n} aggregated", flush=True)

    common = sorted(set.intersection(*(set(p) for p in per_pass)))
    seq_means = {s: float(np.mean([p[s] for p in per_pass])) for s in common}

    result: dict[str, dict] = {"ours": None, "rivals": {}}
    for coldkey, seqs in sorted(by_coldkey.items(), key=lambda kv: -len(kv[1])):
        scored = [seq_means[s] for s in seqs if s in seq_means]
        entry = {
            "mean": float(np.mean(scored)) if scored else None,
            "stdev": float(np.std(scored)) if len(scored) > 1 else None,
            "fleet_size": len(seqs),
            "scores": [float(v) for v in sorted(scored)],
        }
        if coldkey == OUR_COLDKEY:
            result["ours"] = entry
        else:
            result["rivals"][coldkey] = entry

    out_json = out / f"multipass_{epoch}.json"
    out_json.write_text(json.dumps({str(epoch): result}, indent=2) + "\n")
    print(f"WROTE {out_json}", flush=True)
    print(f"  ours: {result['ours']}", flush=True)
    for ck, e in result["rivals"].items():
        print(f"  {ck[:12]}...: mean={e['mean']} stdev={e['stdev']} "
              f"fleet={e['fleet_size']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
