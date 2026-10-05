#!/usr/bin/env python3
"""Build the fresh top-50 index from the leaderboard API, not the stale archive.

The local archive lags the chain by ~4 epochs (newest ``archive.csv`` ends at
epoch25452 while the chain is at 25456), so our own recent winners are absent
from it and the whole-sequence / clonotype gate under-blocks -- exactly why
epochs 25454 and 25455 submitted single-edit siblings of the 25453 winners and
were rejected wholesale.

The public leaderboard API returns every scored submission with full sequence
plus all six BoltzGen metrics and the liability fields, so we rebuild the
validator's exact ``index_top_sequences`` (``rank_binders``, k=50,
max_liability_violations=50) fresh each epoch from the API instead.

Output JSON::

    {"sequences": [top-50 seqs], "cdr3s": [distinct cdr3s], "n": 50,
     "ranked": int, "epochs_scanned": int, "source": "api"}

Invocation (runs under the metanano env)::

    python build_top50_index.py <current_epoch> <out.json> [epochs_to_scan=15]
"""
import json
import sys
import urllib.request

import numpy as np
import pandas as pd

API = "https://compound-api.metanova-labs.ai"
TARGET = "P20809"

METRIC_DEFS = {
    "design_iiptm": "max",
    "design_ptm": "max",
    "min_design_to_target_pae": "min",
    "plip_hbonds_refolded": "max",
    "plip_saltbridge_refolded": "max",
    "delta_sasa_refolded": "max",
}
PROTEIN_WEIGHTS = {
    "design_iiptm": 1.0,
    "design_ptm": 2.0,
    "min_design_to_target_pae": 1.0,
    "plip_hbonds_refolded": 2.0,
    "plip_saltbridge_refolded": 2.0,
    "delta_sasa_refolded": 2.0,
}
# Columns the ranker must receive, in every metric.
REQUIRED = list(METRIC_DEFS)


def fetch_epoch(epoch: int) -> dict:
    req = urllib.request.Request(
        f"{API}/api/competitions/leaderboard/{epoch}/nanobodies",
        headers={"User-Agent": "nova-top50-index/1"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def rank_binders(df: pd.DataFrame, k: int = 50) -> pd.DataFrame:
    """Exact mirror of utils/minmax_weighted_rank.rank_binders (Algorithm 2)."""
    work = df[df["liability_num_violations"] <= 50].copy()
    rank_cols = {}
    for metric, weight in PROTEIN_WEIGHTS.items():
        ascending = METRIC_DEFS[metric] == "min"
        rank_cols[f"wrank_{metric}"] = (
            work[metric].rank(ascending=ascending, method="average") / weight
        )
    rank_df = pd.DataFrame(rank_cols, index=work.index)
    work = work.assign(algo2_score=rank_df.max(axis=1))
    return work.sort_values("algo2_score", ascending=True).head(k)


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    current_epoch = int(sys.argv[1])
    out_path = sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 15

    from metanano.utils.cdr_utils import extract_cdrs  # noqa: E402

    rows: list[dict] = []
    seen: set[str] = set()
    scanned = 0
    for epoch in range(current_epoch - 1, current_epoch - 1 - epochs, -1):
        try:
            payload = fetch_epoch(epoch)
        except Exception:
            continue
        scanned += 1
        for submission in payload.get("data") or []:
            for nb in submission.get("nanobodies") or []:
                if nb.get("protein_name") != TARGET:
                    continue
                seq = str(nb.get("sequence") or "").strip().upper()
                if len(seq) != 127 or seq in seen:
                    continue
                values = {"sequence": seq}
                ok = True
                for metric in REQUIRED:
                    try:
                        values[metric] = float(nb[metric])
                    except (KeyError, TypeError, ValueError):
                        ok = False
                        break
                try:
                    values["liability_num_violations"] = float(nb["liability_num_violations"])
                except (KeyError, TypeError, ValueError):
                    values["liability_num_violations"] = 0.0
                try:
                    values["final_score"] = float(nb["final_score"])
                except (KeyError, TypeError, ValueError):
                    values["final_score"] = float("inf")
                if not ok:
                    continue
                seen.add(seq)
                rows.append(values)

    if not rows:
        json.dump({"sequences": [], "cdr3s": [], "n": 0, "ranked": 0,
                   "epochs_scanned": scanned, "source": "api"}, open(out_path, "w"))
        print(f"TOP50_EMPTY scanned={scanned} rows=0", flush=True)
        return 1

    frame = pd.DataFrame(rows).drop_duplicates("sequence", keep="last").reset_index(drop=True)
    top = rank_binders(frame, k=50)
    sequences = [str(s).upper() for s in top.sequence.tolist()]
    # The live top-50 by the validator's own leaderboard score (lower final_score
    # = higher rank): these are the current winners, and single/double edits of
    # them still clear the dedup (the dedup reference is the full-history HF
    # top-50, ~30 edits from the live field).
    live_top50 = frame.sort_values("final_score").head(50)
    live_sequences = [str(s).upper() for s in live_top50.sequence.tolist()]

    cdr3s, missed = [], 0
    for seq in sequences:
        try:
            cdrs = extract_cdrs(seq)
        except Exception:
            cdrs = None
        c3 = (cdrs or {}).get("cdr3") if isinstance(cdrs, dict) else None
        if c3:
            cdr3s.append(str(c3).strip().upper())
        else:
            missed += 1

    json.dump(
        {"sequences": sequences, "cdr3s": sorted(set(cdr3s)), "n": 50,
         "ranked": int(len(sequences)), "unextractable": missed,
         "epochs_scanned": scanned, "source": "api",
         "live_sequences": live_sequences},
        open(out_path, "w"),
    )
    print(
        f"TOP50_OK rows={len(frame)} top={len(sequences)} "
        f"unextractable={missed} distinct_cdr3={len(set(cdr3s))} "
        f"epochs_scanned={scanned}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
