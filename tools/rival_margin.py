#!/usr/bin/env python3
"""Rival-supply required margin, per reaction.

Replaces the flat "median winner + 0.10" floor with a forward projection of the
field: each competitor (coldkey = stable identity; hotkeys/UIDs rotate under it)
gets a per-reaction growth rate, projected from their last actual submission to
`now`, and the field's expected winner is the max projection.

required_margin(reaction, rival_rows, now) =
    predicted_winner - BIAS[reaction] + own_optimism + Z*RESIDUAL

Calibration (measured 2026-10-03, no-lookahead backtest over 185 epoch x
reaction winners, 200 epochs):
  rho(pred vs actual winner) = 0.815, bias = -0.004, RMSE = 0.120.
  CURSE = 0.0945 is the fallback estimate of how much our offline score
  overstates realised performance. RESIDUAL = 0.120 is rival-prediction
  uncertainty. Z is the aggressiveness dial: 0 -> ~50% win, 1 -> ~84% win.
  BIAS is measured as predicted rival winner minus actual rival winner, so it
  is subtracted when converting the projection to an expected realised score.
  Reactions listed in OPPORTUNITY_WIN_PROBABILITY replace the point floor with
  the requested empirical win-probability quantile, calibrated from strictly
  no-lookahead rival errors and reaction-specific realised optimism.

rival_rows is a list of {"epoch", "coldkey", "reaction", "score"} for scored
submissions. `now` is the epoch to project to (the epoch being gated).
"""
from __future__ import annotations

import math

CURSE = 0.0945
RESIDUAL = 0.120
SHRINK_K = 5          # growth is shrunk toward flat (0) by n/(n+K)
Z = 0.0               # aggressiveness: 0 = ~50% win, 1 = ~84% win
# residual bias per reaction from the backtest (pred - actual)
BIAS = {1: -0.018, 2: -0.004, 3: 0.029, 4: -0.004, 5: -0.024}

# Reactions 1 and 3 have deep enough inventory that waiting only for a
# point-estimate win leaves too many weak-field opportunities unused.  Their
# release floor is therefore the empirically calibrated raw score associated
# with a 25% win chance. Other reactions retain the point-estimate (roughly
# 50%) gate. The adjustment is learned only from no-lookahead rival forecast
# errors and our own realised raw-score optimism; it is never allowed to make
# the standard gate stricter or relax it by more than one measured RMSE.
OPPORTUNITY_WIN_PROBABILITY = {1: 0.25, 3: 0.25}
OPPORTUNITY_MIN_RESIDUALS = 12
OPPORTUNITY_MAX_RESIDUALS = 60


def epoch_maxima(hist: list[dict]) -> list[dict]:
    """One observation per miner and epoch: its best submitted portfolio."""
    best: dict[int, dict] = {}
    for row in hist:
        epoch = int(row["epoch"])
        if epoch not in best or float(row["score"]) > float(best[epoch]["score"]):
            best[epoch] = row
    return [best[epoch] for epoch in sorted(best)]


def growth(hist: list[dict], k: int = SHRINK_K) -> float:
    """Per-epoch score slope of a rival's submission history, shrunk toward 0
    so a low-n rival is pulled to flat rather than trusted on a noisy slope."""
    hist = epoch_maxima(hist)
    n = len(hist)
    if n < 2:
        return 0.0
    xs = [h["epoch"] for h in hist]
    ys = [h["score"] for h in hist]
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    g = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0
    return g * (n / (n + k))


def quantile(values: list[float], probability: float) -> float:
    """Linear empirical quantile with no numpy dependency."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("quantile requires observations")
    position = (len(ordered) - 1) * min(1.0, max(0.0, probability))
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def projected_winner(reaction: int, rival_rows: list[dict], now: int):
    """Return the current field projection and its per-rival components."""
    by_coldkey: dict[str, list[dict]] = {}
    for row in rival_rows:
        if str(row.get("reaction")) == str(reaction):
            by_coldkey.setdefault(row["coldkey"], []).append(row)

    projections = []
    for coldkey, history in by_coldkey.items():
        history = epoch_maxima(history)
        if len(history) < 2:
            continue
        rate = growth(history)
        last = history[-1]
        projection = last["score"] + rate * (now - last["epoch"])
        lower = min(item["score"] for item in history) - 0.05
        upper = max(item["score"] for item in history) + 0.05
        projection = max(min(projection, upper), lower)
        projections.append({
            "coldkey": coldkey, "last": last["score"], "proj": projection,
            "growth": rate, "n": len(history),
        })
    projections.sort(key=lambda item: -item["proj"])
    return (projections[0]["proj"] if projections else None), projections


def no_lookahead_residuals(reaction: int, rival_rows: list[dict]) -> list[float]:
    """Actual minus predicted rival winner using only earlier epochs."""
    reaction_rows = [row for row in rival_rows
                     if str(row.get("reaction")) == str(reaction)]
    epochs = sorted({int(row["epoch"]) for row in reaction_rows})
    residuals = []
    bias = BIAS.get(reaction, 0.0)
    for epoch in epochs:
        prior = [row for row in reaction_rows if int(row["epoch"]) < epoch]
        prediction, _ = projected_winner(reaction, prior, epoch)
        if prediction is None:
            continue
        actual = max(float(row["score"]) for row in reaction_rows
                     if int(row["epoch"]) == epoch)
        residuals.append(actual - (prediction - bias))
    return residuals[-OPPORTUNITY_MAX_RESIDUALS:]


def required_margin(reaction: int, rival_rows: list[dict], now: int,
                    optimism: float = CURSE,
                    optimism_samples: list[float] | None = None):
    """Return (margin, info) for `reaction` at epoch `now`.

    Projects every rival with >=2 submissions for this reaction to `now` using
    their shrunk growth, takes the max as the expected winner, then adds the
    curse, the Z-scaled residual, and the per-reaction bias.
    """
    winner, projections = projected_winner(reaction, rival_rows, now)
    info = {"reaction": reaction, "now": now, "n_rivals": len(projections)}
    if winner is None:
        return None, {**info, "source": "no_rival_history"}

    bias = BIAS.get(reaction, 0.0)
    standard_margin = winner - bias + optimism + Z * RESIDUAL
    margin = standard_margin
    opportunity = None
    target_probability = OPPORTUNITY_WIN_PROBABILITY.get(reaction)
    if target_probability is not None:
        residuals = no_lookahead_residuals(reaction, rival_rows)
        if len(residuals) >= OPPORTUNITY_MIN_RESIDUALS:
            samples = [float(value) for value in (optimism_samples or [optimism])]
            combined_errors = [
                residual + own_loss - optimism
                for residual in residuals
                for own_loss in samples
            ]
            raw_adjustment = quantile(combined_errors, target_probability)
            adjustment = max(-RESIDUAL, min(0.0, raw_adjustment))
            margin = standard_margin + adjustment
            opportunity = {
                "enabled": True,
                "target_win_probability": target_probability,
                "residual_samples": len(residuals),
                "optimism_samples": len(samples),
                "raw_adjustment": round(raw_adjustment, 4),
                "adjustment": round(adjustment, 4),
                "standard_margin": round(standard_margin, 4),
            }
        else:
            opportunity = {
                "enabled": False,
                "target_win_probability": target_probability,
                "residual_samples": len(residuals),
                "reason": "insufficient_no_lookahead_residuals",
            }
    info = {**info, "winner": round(winner, 4),
            "optimism": round(optimism, 4), "curse": round(optimism, 4),
            "z": Z, "residual": RESIDUAL,
            "bias": bias, "margin": round(margin, 4),
            "top": [{k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in t.items()} for t in projections[:5]]}
    if opportunity is not None:
        info["opportunity_gate"] = opportunity
    return round(margin, 4), info


if __name__ == "__main__":
    import json, sys
    # Accept either a directory of per-epoch leaderboard JSONs or a rival-rows
    # JSON (list of {epoch, coldkey, reaction, score}).
    if not sys.argv[1:]:
        sys.exit("usage: rival_margin.py <dir|rival_history.json> [now]")
    import glob, os
    src = sys.argv[1]
    if os.path.isdir(src):
        rows = []
        for p in glob.glob(os.path.join(src, "[0-9]*.json")):
            ep = int(os.path.basename(p).split(".")[0])
            data = json.load(open(p)).get("data") or []
            for sub in data:
                ck = sub.get("coldkey")
                sc = sub.get("final_score")
                if not ck or sc in (None, -999.99):
                    continue
                for m in (sub.get("molecules") or []):
                    n = m.get("name") or ""
                    if n.startswith("rxn:") and n.split(":")[1].isdigit():
                        rows.append({"epoch": ep, "coldkey": ck,
                                     "reaction": int(n.split(":")[1]), "score": sc})
                        break
        now = max((int(os.path.basename(p).split(".")[0])
                   for p in glob.glob(os.path.join(src, "[0-9]*.json"))), default=0) + 3
    else:
        rows = json.load(open(src))
        now = int(sys.argv[2]) if len(sys.argv) > 2 else max(r["epoch"] for r in rows) + 3
    for r in sorted({row["reaction"] for row in rows}, key=int):
        m, info = required_margin(int(r), rows, now)
        if m is None:
            print(f"rxn{r}: no history")
            continue
        t = info["top"][0]
        print(f"rxn{r}: winner={info['winner']:.4f} margin={m:.4f} "
              f"setby={t['coldkey'][:8]} {t['last']:.3f}->{t['proj']:.3f}")
