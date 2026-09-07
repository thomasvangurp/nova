"""Would our candidate have won the round?

The validator does NOT rank against the 31k archive -- it ranks a submission
against the ~50 OTHER submissions in that epoch, sums the 10 per-metric ranks,
and the lowest total wins. Epoch 24980's nanobody winner scored 67.0 over 50
ranked UIDs, i.e. an average rank of 6.7 per metric.

So the honest test is: drop one of our exact-scored candidates into a field of
real competitors and compute its rank_sum there. The competitor field is drawn
from RECENT archive rows -- recent because the archive spans multiple scoring
eras (oldest rows average design_iiptm 0.385 and delta_sasa 1223; recent rows
0.758 and 717), and our scorer was validated against the recent era only.

Each candidate is evaluated in its own field so the candidates do not compete
with each other, which is what actually happens: only one is submitted.
"""
from __future__ import annotations
import argparse, json
import numpy as np, pandas as pd

METRICS = {"design_iiptm": "max", "design_ptm": "max", "design_to_target_iptm": "max",
           "min_design_to_target_pae": "min", "interaction_pae": "min",
           "plip_hbonds_refolded": "max", "plip_saltbridge_refolded": "max",
           "delta_sasa_refolded": "max", "liability_score": "min",
           "liability_num_violations": "min"}


def rank_sum_in_field(mine: dict, field: pd.DataFrame) -> float:
    """rank_sum of `mine` against `field`, ties averaged, exactly as upstream."""
    total = 0.0
    for m, d in METRICS.items():
        col = np.append(field[m].values, mine[m])
        s = pd.Series(col).rank(ascending=(d == "min"), method="average")
        total += float(s.iloc[-1])
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True, help="our exact-scored candidates")
    ap.add_argument("--archive", required=True)
    ap.add_argument("--field-size", type=int, default=49,
                    help="competitors, so the field is field-size + 1 = a real epoch")
    ap.add_argument("--recent", type=int, default=300,
                    help="draw competitors from the most recent N archive rows")
    a = ap.parse_args()

    blob = json.load(open(a.scored))
    comp = blob["per_nanobody_components"]
    ours = {}
    for by_seq in comp.values():
        for seq, by_target in by_seq.items():
            ours[seq] = list(by_target.values())[0]
    print(f"our exact-scored candidates: {len(ours)}")

    df = pd.read_csv(a.archive)
    recent = df.tail(a.recent).reset_index(drop=True)
    rng = np.random.default_rng(0)
    field = recent.iloc[rng.choice(len(recent), a.field_size, replace=False)]
    print(f"competitor field: {len(field)} recent real submissions "
          f"(field of {len(field)+1}, like a real epoch)\n")

    # where the field's own members land, for calibration
    field_rs = []
    for i in range(len(field)):
        others = field.drop(field.index[i])
        field_rs.append(rank_sum_in_field(field.iloc[i], others))
    field_rs = np.array(field_rs)
    print(f"the field's own rank_sums: best {field_rs.min():.1f}, "
          f"median {np.median(field_rs):.1f}, worst {field_rs.max():.1f}")

    rows = []
    for seq, met in ours.items():
        rows.append((rank_sum_in_field(met, field), seq))
    rows.sort()
    print(f"\nour candidates in that field (lower is better, floor is 10):")
    print(f"{'rank_sum':>9}  {'beats N of 49':>13}  sequence")
    for rs, seq in rows[:10]:
        beats = int((field_rs > rs).sum())
        print(f"{rs:>9.1f}  {beats:>13}  {seq[:44]}...")

    best = rows[0][0]
    print(f"\nBEST OF OURS: rank_sum {best:.1f}")
    print(f"  would place #{int((field_rs < best).sum()) + 1} of {len(field)+1}")
    print(f"  epoch 24980's actual nanobody winner scored 67.0 over 50 UIDs")
    print(f"  -> {'WOULD HAVE WON' if best < 67.0 else 'would NOT have won'} that round")


if __name__ == "__main__":
    main()
