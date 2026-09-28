"""Our progress over time, and the same measures for our competitors.

Aggregate rates hide the only thing that matters: whether we are getting
better.  Everything here is a trailing window so a single lucky epoch cannot
move the line, and the same window is applied to rivals so the comparison is
like-for-like.
"""
from __future__ import annotations

import sqlite3
import statistics as st
from pathlib import Path

import plotly.graph_objects as go

from .outcomes import is_entry_win

PAPER = "#0e1117"
CARD = "#161a23"
INK = "#e6e9ef"
MUTED = "#8b93a7"
OURS = "#4fc3f7"
WIN = "#66d9a6"
WARN = "#ff9f6b"
BAD = "#ef5f6b"
NET = "#ffd166"
RIVALS = ("#b388ff", "#ff8fab", "#7ee8fa", "#ffd166")
LAYOUT = dict(paper_bgcolor=PAPER, plot_bgcolor=CARD, font=dict(color=INK, size=13),
              margin=dict(l=70, r=66, t=28, b=46), hovermode="x unified",
              legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0,
                          bgcolor="rgba(0,0,0,0)"))


def _by_coldkey(data):
    out = {}
    for e in data["entries"]:
        out.setdefault(e["coldkey"], {}).setdefault(e["epoch"], []).append(e)
    return out


def rolling_rates(per_epoch, epochs, window=20, min_entries=1):
    """Trailing field-beating-entry, top10-per-entry and fleet size.

    A window holding fewer than ``min_entries`` entries yields a gap rather
    than a rate: once a miner stops submitting, its trailing window drains to
    a handful of old entries and a single top-10 would plot as 100%.
    """
    wins, tens, fleet = [], [], []
    for i, _ in enumerate(epochs):
        chunk = epochs[max(0, i - window + 1):i + 1]
        rows = [r for e in chunk for r in per_epoch.get(e, [])]
        sizes = [len(per_epoch[e]) for e in chunk if per_epoch.get(e)]
        if len(rows) < max(1, min_entries):
            wins.append(None); tens.append(None); fleet.append(None); continue
        wins.append(100 * sum(is_entry_win(r) for r in rows) / len(rows))
        tens.append(100 * sum(1 for r in rows if r["rank"] <= 10) / len(rows))
        fleet.append(st.median(sizes) if sizes else None)
    return wins, tens, fleet


def fig_our_progress(data, database: Path | None = None, window: int = 20) -> go.Figure:
    """Our win rate, top-10 rate and fleet size, with deploys marked."""
    epochs = sorted(data["epochs"])
    mine = {e: rows for e, rows in data["by_epoch"].items()}
    wins, tens, fleet = rolling_rates(mine, epochs, window)

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=epochs, y=tens, name=f"top-10 per entry % ({window}ep)",
                                line=dict(color=OURS, width=2.4)))
    figure.add_trace(go.Scatter(
        x=epochs, y=wins, name=f"field-beating entries % ({window}ep)",
                                line=dict(color=WIN, width=2.4)))
    figure.add_trace(go.Scatter(x=epochs, y=fleet, name="fleet size (median)",
                                yaxis="y2", line=dict(color=WARN, width=1.6, dash="dot")))
    if database is not None and Path(database).is_file():
        connection = sqlite3.connect(database)
        counts: dict[int, int] = {}
        for (epoch,) in connection.execute(
                "SELECT epoch FROM deploys WHERE epoch IS NOT NULL"):
            counts[epoch] = counts.get(epoch, 0) + 1
        if counts:
            figure.add_trace(go.Bar(
                x=sorted(counts), y=[counts[e] for e in sorted(counts)],
                name="code changes", yaxis="y3", marker_color=NET, opacity=0.5,
                hovertemplate="%{y} commits<extra></extra>"))
    layout = dict(LAYOUT)
    layout["yaxis2"] = dict(title="fleet size", overlaying="y", side="right",
                            rangemode="tozero", gridcolor="rgba(0,0,0,0)")
    layout["yaxis3"] = dict(overlaying="y", side="right", range=[0, 40],
                            showticklabels=False, gridcolor="rgba(0,0,0,0)")
    figure.update_layout(**layout)
    figure.update_yaxes(title="percent of entries", gridcolor="#242a36", rangemode="tozero")
    figure.update_xaxes(title="epoch", gridcolor="#242a36")
    return figure


def strongest_rivals(data, top: int = 4, recent: int = 60,
                     min_recent_entries: int = 60) -> list[str]:
    """Rivals with the best top-10 per entry over the last ``recent`` epochs.

    Selected on the same rate the chart plots, over a recent window, among
    coldkeys that submitted at least ``min_recent_entries`` in it.  Counting
    wins instead favoured fleet size and hid 2-slot 5GhREcsK (37% top-10 per
    entry); an all-time window kept 5Fq7TBKW long after it quit at 25293.
    """
    epochs = sorted(data["epochs"])
    cutoff = epochs[-recent] if len(epochs) >= recent else epochs[0]
    tally: dict[str, list[int]] = {}
    for e in data["entries"]:
        if e["epoch"] < cutoff or e["coldkey"] == data.get("coldkey"):
            continue
        row = tally.setdefault(e["coldkey"], [0, 0])
        row[0] += 1
        row[1] += e["rank"] <= 10
    eligible = [(ck, t / n) for ck, (n, t) in tally.items() if n >= min_recent_entries]
    eligible.sort(key=lambda kv: (-kv[1], kv[0]))
    return [ck for ck, _ in eligible[:top]]


def fig_rates_vs_rivals(data, top: int = 4, window: int = 30,
                        min_entries: int = 20, recent: int = 60) -> go.Figure:
    """Our top-10 rate against the strongest rivals, same trailing window."""
    grouped = _by_coldkey(data)
    epochs = sorted(data["epochs"])
    figure = go.Figure()
    _, tens, _ = rolling_rates(data["by_epoch"], epochs, window, min_entries)
    figure.add_trace(go.Scatter(x=epochs, y=tens, name="US",
                                line=dict(color=WIN, width=3)))
    for shown, coldkey in enumerate(strongest_rivals(data, top, recent)):
        _, rival_tens, _ = rolling_rates(grouped[coldkey], epochs, window, min_entries)
        figure.add_trace(go.Scatter(
            x=epochs, y=rival_tens, name=coldkey[:8] + "...",
            line=dict(color=RIVALS[shown % len(RIVALS)], width=1.6)))
    figure.update_layout(**LAYOUT)
    figure.update_yaxes(title=f"top-10 per entry % ({window}-epoch)",
                        gridcolor="#242a36", rangemode="tozero")
    figure.update_xaxes(title="epoch", gridcolor="#242a36")
    return figure


def competitor_table(rows) -> str:
    """Entry efficiency plus the separately capped winning-epoch count."""
    rows = sorted(rows, key=lambda r: -r["win_per_entry"])
    head = ("<tr><th>coldkey</th><th>fleet</th><th>entries</th><th>epochs</th>"
            "<th>entry wins</th><th>entry win / entry</th><th>winning epochs</th>"
            "<th>top-10 / entry</th>"
            "<th>best score</th></tr>")
    body = []
    for r in rows:
        cls = ' class="us"' if r["ours"] else ""
        body.append(
            f"<tr{cls}><td>{r['label']}</td><td>{r['fleet']:.0f}</td>"
            f"<td>{r['entries']}</td><td>{r['epochs']}</td><td>{r['wins']}</td>"
            f"<td>{r['win_per_entry']:.2f}%</td>"
            f"<td>{r['winning_epochs']}</td>"
            f"<td>{r['top10_per_entry']:.1f}%</td><td>{r['best']:.0f}</td></tr>")
    return ('<table class="grid"><thead>' + head + "</thead><tbody>"
            + "".join(body) + "</tbody></table>")
