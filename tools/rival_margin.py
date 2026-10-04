#!/usr/bin/env python3
"""Rival-supply required margin, per reaction.

Replaces the flat "median winner + 0.10" floor with a forward projection of the
field: each competitor (coldkey = stable identity; hotkeys/UIDs rotate under it)
gets a per-reaction growth rate, projected from their last actual submission to
`now`, and the field's expected winner is the max projection.

required_margin(reaction, rival_rows, now) =
    predicted_winner + CURSE + Z*RESIDUAL + BIAS[reaction]

Calibration (measured 2026-10-03, no-lookahead backtest over 185 epoch x
reaction winners, 200 epochs):
  rho(pred vs actual winner) = 0.815, bias = -0.004, RMSE = 0.120.
  CURSE = 0.0945 (offline prediction overstates realised), RESIDUAL = 0.120 is
  the prediction uncertainty. Z is the aggressiveness dial: 0 -> ~50% win,
  1 -> ~84% win. BIAS is the per-reaction residual bias.

rival_rows is a list of {"epoch", "coldkey", "reaction", "score"} for scored
submissions. `now` is the epoch to project to (the epoch being gated).
"""
from __future__ import annotations

CURSE = 0.0945
RESIDUAL = 0.120
SHRINK_K = 5          # growth is shrunk toward flat (0) by n/(n+K)
Z = 0.0               # aggressiveness: 0 = ~50% win, 1 = ~84% win
# residual bias per reaction from the backtest (pred - actual)
BIAS = {1: -0.018, 2: -0.004, 3: 0.029, 4: -0.004, 5: -0.024}


def growth(hist: list[dict], k: int = SHRINK_K) -> float:
    """Per-epoch score slope of a rival's submission history, shrunk toward 0
    so a low-n rival is pulled to flat rather than trusted on a noisy slope."""
    hist = sorted(hist, key=lambda h: h["epoch"])
    n = len(hist)
    if n < 2:
        return 0.0
    xs = [h["epoch"] for h in hist]
    ys = [h["score"] for h in hist]
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    g = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0
    return g * (n / (n + k))


def required_margin(reaction: int, rival_rows: list[dict], now: int):
    """Return (margin, info) for `reaction` at epoch `now`.

    Projects every rival with >=2 submissions for this reaction to `now` using
    their shrunk growth, takes the max as the expected winner, then adds the
    curse, the Z-scaled residual, and the per-reaction bias.
    """
    by_coldkey: dict[str, list[dict]] = {}
    for r in rival_rows:
        if str(r.get("reaction")) == str(reaction):
            by_coldkey.setdefault(r["coldkey"], []).append(r)

    projections = []
    for ck, hist in by_coldkey.items():
        if len(hist) < 2:
            continue
        hist = sorted(hist, key=lambda h: h["epoch"])
        g = growth(hist)
        last = hist[-1]
        p = last["score"] + g * (now - last["epoch"])
        lo = min(h["score"] for h in hist) - 0.05
        hi = max(h["score"] for h in hist) + 0.05
        p = max(min(p, hi), lo)
        projections.append({"coldkey": ck, "last": last["score"], "proj": p,
                            "growth": g, "n": len(hist)})

    info = {"reaction": reaction, "now": now, "n_rivals": len(projections)}
    if not projections:
        return None, {**info, "source": "no_rival_history"}

    projections.sort(key=lambda p: -p["proj"])
    winner = projections[0]["proj"]
    margin = winner + CURSE + Z * RESIDUAL + BIAS.get(reaction, 0.0)
    info = {**info, "winner": round(winner, 4),
            "curse": CURSE, "z": Z, "residual": RESIDUAL,
            "bias": BIAS.get(reaction, 0.0), "margin": round(margin, 4),
            "top": [{k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in t.items()} for t in projections[:5]]}
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
