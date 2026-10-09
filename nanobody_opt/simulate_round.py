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
with each other, which is what actually happens: only one is submitted.  The
validator uses pandas ``method="dense"`` for every metric.  Ties therefore
consume one rank, not the average of all occupied ordinal positions.
"""
from __future__ import annotations
import argparse, json
import numpy as np, pandas as pd

METRICS = {"design_iiptm": "max", "design_ptm": "max", "design_to_target_iptm": "max",
           "min_design_to_target_pae": "min", "interaction_pae": "min",
           "plip_hbonds_refolded": "max", "plip_saltbridge_refolded": "max",
           "delta_sasa_refolded": "max", "liability_score": "min",
           "liability_num_violations": "min"}

METRIC_CATEGORIES = {
    "confidence": (
        "design_iiptm",
        "design_ptm",
        "design_to_target_iptm",
        "min_design_to_target_pae",
        "interaction_pae",
    ),
    "physical_interaction": (
        "plip_hbonds_refolded",
        "plip_saltbridge_refolded",
        "delta_sasa_refolded",
    ),
    "developability": (
        "liability_score",
        "liability_num_violations",
    ),
}


def dense_rank_sums(
    frame: pd.DataFrame,
    metrics: dict[str, str] = METRICS,
    *,
    na_option: str = "keep",
) -> np.ndarray:
    """Mirror ``BoltzgenWrapper._rank_metrics_dataframe`` exactly.

    The upstream validator calls ``Series.rank(method="dense")`` independently
    for each configured metric, then sums those dense ranks.  Keeping this in
    one helper prevents proposal generators and live selection from silently
    reverting to pandas' default average-tie behavior.
    """
    missing = set(metrics) - set(frame.columns)
    if missing:
        raise ValueError(f"field lacks validator metrics: {sorted(missing)}")
    total = np.zeros(len(frame), dtype=np.float64)
    for metric, direction in metrics.items():
        values = pd.to_numeric(frame[metric], errors="coerce")
        total += values.rank(
            method="dense",
            ascending=(direction == "min"),
            na_option=na_option,
        ).to_numpy()
    return total


def category_rank_sums(mine: dict, field: pd.DataFrame) -> dict[str, float]:
    """Return the three dashboard category sums for one candidate."""
    combined = pd.concat(
        [field[list(METRICS)], pd.DataFrame([{m: mine[m] for m in METRICS}])],
        ignore_index=True,
    )
    result = {}
    for category, names in METRIC_CATEGORIES.items():
        directions = {name: METRICS[name] for name in names}
        result[category] = float(dense_rank_sums(combined, directions)[-1])
    return result


def rank_sum_in_field(mine: dict, field: pd.DataFrame) -> float:
    """Validator ``rank_sum`` of ``mine`` against a competitor field.

    NOTE: this sums the three category rank-sums directly (equivalently, all ten
    metric dense ranks), which skips the validator's per-category rank step.  It
    is a looser proxy than :func:`validator_final_scores`, which reproduces the
    published ``final_score`` (rho ~0.92 vs the public API).
    """
    return float(sum(category_rank_sums(mine, field).values()))


def validator_final_scores(
    frame: pd.DataFrame,
    categories: dict[str, tuple[str, ...]] = METRIC_CATEGORIES,
) -> np.ndarray:
    """Exact validator ``final_score`` for every row: rank each category, sum ranks.

    The validator does NOT sum all ten metric ranks directly. It computes, per
    category, the sum of that category's dense metric ranks, then ranks that
    category-sum within the field (dense, lower = better), and sums the three
    category ranks. This is the ``final_score`` the leaderboard publishes as
    ``confidence_rank + physical_interaction_rank + developability_rank``,
    recovered by regression against the public API (rho ~0.92 vs published
    ``final_score``, versus ~0.56 for the old minmax-weighted proxy).

    Prefer this over :func:`rank_sum_in_field` for anything that must predict a
    live rank. Requires all ten metrics, including ``interaction_pae`` which the
    NIM fast adapter cannot emit; use the exact scorer's metrics.
    """
    total = np.zeros(len(frame), dtype=np.float64)
    for category, names in categories.items():
        directions = {name: METRICS[name] for name in names}
        category_sum = dense_rank_sums(frame, directions)
        total += (
            pd.Series(category_sum)
            .rank(method="dense", ascending=True, na_option="keep")
            .to_numpy()
        )
    return total


def validator_scores_from_observations(
    observations: list[dict],
    *,
    score_kind: str = "local_exact_validator_replay_rank_sum",
    by_epoch: bool = True,
) -> dict[str, float]:
    """Per-sequence validator ``final_score`` from stored exact-scorer metrics.

    ``observations`` is the list returned by
    ``mutation_network.load_observations(path, target)``; each dict carries
    ``sequence``, ``score_kind``, ``epoch`` and ``metrics`` (the ten raw BoltzGen
    metrics). Only rows whose ``score_kind`` matches are kept -- the exact
    full-BoltzGen replay, which carries ``interaction_pae`` that the fast NIM
    adapter cannot emit and that the confidence component depends on.

    The validator ranks a submission against the ~50 others in *its* epoch, so
    with ``by_epoch`` (default) the field is the observations sharing an epoch;
    otherwise one global field over all observations. Sequences scored multiple
    times are averaged. Lower is better.
    """
    records = []
    for obs in observations:
        if obs.get("score_kind") != score_kind:
            continue
        metrics = obs.get("metrics")
        if not isinstance(metrics, dict):
            continue
        if any(metrics.get(name) is None for name in METRICS):
            continue
        records.append((str(obs["sequence"]), int(obs.get("epoch") or 0), metrics))
    if not records:
        return {}

    def _score_group(group: list[tuple[str, int, dict]]) -> dict[str, float]:
        frame = pd.DataFrame([m for _, _, m in group])
        scores = validator_final_scores(frame)
        out: dict[str, list[float]] = {}
        for (seq, _epoch, _m), score in zip(group, scores):
            if not np.isnan(score):
                out.setdefault(seq, []).append(float(score))
        return {seq: float(np.mean(v)) for seq, v in out.items()}

    if by_epoch:
        by_epoch_map: dict[int, list[tuple[str, int, dict]]] = {}
        for rec in records:
            by_epoch_map.setdefault(rec[1], []).append(rec)
        merged: dict[str, list[float]] = {}
        for group in by_epoch_map.values():
            for seq, score in _score_group(group).items():
                merged.setdefault(seq, []).append(score)
        return {seq: float(np.mean(v)) for seq, v in merged.items()}

    return _score_group(records)


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

    # Boltzgen applies filter_cysteine=true for nanobody-anything and simply
    # returns NO metrics for a design it rejects ("No metrics found for
    # nanobody ..."). Those are not near-misses -- in a live epoch they score
    # nothing at all. Counting them is part of the honest result.
    required = set(METRICS)
    rows, dropped = [], []
    for seq, met in ours.items():
        if not isinstance(met, dict) or not required.issubset(met):
            dropped.append(seq)
            continue
        rows.append((rank_sum_in_field(met, field), seq))
    rows.sort()
    if dropped:
        print(f"\nWARNING: {len(dropped)}/{len(ours)} candidates produced NO metrics "
              f"(boltzgen rejected them, e.g. filter_cysteine). In a live epoch "
              f"each of these scores nothing.")
    if not rows:
        raise SystemExit("no candidate produced a complete metric set")
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
