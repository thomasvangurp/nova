"""Does OUR boltzgen harness reproduce metrics the archive already publishes?

Run this BEFORE scoring any candidate of our own. The archive gives, for 31,027
sequences, the exact 10 metrics the validator ranks on. If our harness cannot
reproduce them for sequences whose answers are known, then any number it
produces for a NEW sequence is meaningless -- and would be indistinguishable
from a real result.

Picks sequences spanning the quality range rather than only good ones: a harness
that is right at the top and wrong in the tail would still rank our candidates
wrongly, and we select by rank.
"""
from __future__ import annotations
import argparse, json
import numpy as np, pandas as pd

METRICS = {
    "design_iiptm": "max", "design_ptm": "max", "design_to_target_iptm": "max",
    "min_design_to_target_pae": "min", "interaction_pae": "min",
    "plip_hbonds_refolded": "max", "plip_saltbridge_refolded": "max",
    "delta_sasa_refolded": "max", "liability_score": "min",
    "liability_num_violations": "min",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    df = pd.read_csv(a.archive).drop_duplicates(subset=["sequence"]).reset_index(drop=True)
    rank_sum = sum(df[m].rank(ascending=(d == "min")).values for m, d in METRICS.items())
    order = np.argsort(rank_sum)
    # span the range: best, worst, and evenly spaced quantiles between
    idx = [int(order[int(q * (len(order) - 1))]) for q in np.linspace(0, 1, a.n)]

    cases = []
    for rank_pos, i in enumerate(idx):
        row = df.iloc[i]
        cases.append({
            "sequence": row["sequence"],
            "archive_rank_sum": float(rank_sum[i]),
            "quantile": round(rank_pos / (a.n - 1), 3),
            "expected": {m: float(row[m]) for m in METRICS},
        })
    json.dump({"target": "P20809", "metrics": METRICS, "cases": cases},
              open(a.out, "w"), indent=2)
    print(f"wrote {len(cases)} instrument-check cases to {a.out}")
    print(f"{'quantile':>9} {'archive_rank_sum':>17}  design_iiptm  interaction_pae")
    for c in cases:
        print(f"{c['quantile']:>9} {c['archive_rank_sum']:>17,.0f}  "
              f"{c['expected']['design_iiptm']:>12.4f}  {c['expected']['interaction_pae']:>15.4f}")


if __name__ == "__main__":
    main()
