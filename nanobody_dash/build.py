"""Render the collected data into one self-contained index.html.

Everything is inlined -- plotly.js included -- so the file works from a local
``file://`` open, from ``python -m http.server``, and from a Vercel static
deploy, with no network and no server.  That is the whole reason this is not a
Dash app: a dashboard that needs a running process is a dashboard that is down
when you want it.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html as html_lib
import json
import sqlite3
import statistics as st
from pathlib import Path

import plotly.graph_objects as go

from . import BLOCK_SECONDS, DEFAULT_COLDKEY, EPOCH_LENGTH
from . import progress
from .outcomes import annotate_field_beating_entries, is_entry_win

_DATABASE = None

PAPER = "#0e1117"
CARD = "#161a23"
INK = "#e6e9ef"
MUTED = "#8b93a7"
OURS = "#4fc3f7"
FIELD = "#7e8aa2"
WIN = "#66d9a6"
WARN = "#ff9f6b"
BAD = "#ef5f6b"

#: No plotly `title` anywhere: it renders inside the plotting area and a
#: horizontal legend overlaps it at every margin setting.  Titles are HTML
#: headings above each figure instead, which also gives the tabs something
#: to label rows with.
LAYOUT = dict(paper_bgcolor=PAPER, plot_bgcolor=CARD, font=dict(color=INK, size=13),
              margin=dict(l=60, r=60, t=28, b=46), hovermode="x unified",
              legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0,
                          bgcolor="rgba(0,0,0,0)"))


def load(database: Path, coldkey: str) -> dict:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    epochs = {r["epoch"]: dict(r) for r in
              connection.execute("SELECT * FROM epochs ORDER BY epoch")}
    entries = [dict(r) for r in
               connection.execute("SELECT * FROM entries ORDER BY epoch, rank")]
    annotate_field_beating_entries(entries)
    slots = [dict(r) for r in connection.execute("SELECT * FROM slots ORDER BY time")]
    ours = [e for e in entries if e["coldkey"] == coldkey]

    first_seen: dict[str, int] = {}
    for e in ours:
        first_seen.setdefault(e["hotkey"], e["epoch"])
        first_seen[e["hotkey"]] = min(first_seen[e["hotkey"]], e["epoch"])

    by_epoch: dict[int, list] = {}
    for e in ours:
        by_epoch.setdefault(e["epoch"], []).append(e)

    # The fleet figure and headline compare validator scoring with the roster
    # that survived to the end of the epoch.  A median hides late churn: epoch
    # 25293 was at 12 for most polls, lost uid 24 before validation, and scored
    # 11; reporting the median as "12 held on chain" made a real slot loss look
    # like a commitment failure.  ``slots`` is ordered by time, so overwrite to
    # retain the final observation for each epoch.
    registered: dict[int, int] = {}
    for s in slots:
        if s["epoch"] is not None and s["registered"] is not None:
            registered[s["epoch"]] = int(s["registered"])

    return dict(epochs=epochs, entries=entries, ours=ours, slots=slots,
                by_epoch=by_epoch, first_seen=first_seen, registered=registered)


def _rolling(values, window=9):
    """Median over a trailing window, skipping gaps."""
    out = []
    for i in range(len(values)):
        chunk = [v for v in values[max(0, i - window + 1):i + 1] if v is not None]
        out.append(st.median(chunk) if chunk else None)
    return out


def _exact_epoch_axis(figure: go.Figure) -> go.Figure:
    """Never abbreviate epoch identifiers as values such as ``25.22k``.

    Plotly treats epochs as continuous numbers, which is useful for zooming but
    applies SI-prefix formatting by default.  Apply this at render time so it
    also covers figures built in ``progress`` and any future epoch figure.
    """
    for axis in figure.select_xaxes():
        if getattr(getattr(axis, "title", None), "text", None) == "epoch":
            axis.update(tickformat="d", exponentformat="none", showexponent="none")
    return figure


def fig_scores(data, recent: int = 60) -> go.Figure:
    """Raw series are unreadable over 189 epochs, so the raw lines are faint and
    the trailing median carries the signal.  The view opens on the last 60
    epochs with a range slider rather than showing everything at once."""
    order = sorted(data["epochs"])
    best, winner, median, cut = [], [], [], []
    for e in order:
        mine = data["by_epoch"].get(e) or []
        best.append(min((m["score"] for m in mine), default=None))
        winner.append(data["epochs"][e]["winner"])
        median.append(data["epochs"][e]["median"])
        cut.append(data["epochs"][e]["top10_cut"])

    figure = go.Figure()
    series = (("field median", median, FIELD, "dot"),
              ("top-10 cut", cut, WARN, None),
              ("winner", winner, WIN, None),
              ("our best", best, OURS, None))
    for name, values, colour, dash in series:
        figure.add_trace(go.Scatter(
            x=order, y=values, name=name, legendgroup=name, showlegend=False,
            mode="lines", opacity=0.22, hoverinfo="skip",
            line=dict(color=colour, width=1, dash=dash)))
    for name, values, colour, dash in series:
        figure.add_trace(go.Scatter(
            x=order, y=_rolling(values), name=name, legendgroup=name,
            mode="lines", line=dict(color=colour, width=2.4, dash=dash),
            hovertemplate=name + ": %{y:.0f}<extra></extra>"))
    layout = dict(LAYOUT)
    layout["hovermode"] = "x unified"
    figure.update_layout(**layout)
    figure.update_yaxes(title="nanobody score (9-epoch median)", gridcolor="#242a36")
    figure.update_xaxes(title="epoch", gridcolor="#242a36",
                        rangeslider=dict(visible=True, thickness=0.07),
                        range=[order[max(0, len(order) - recent)], order[-1]])
    return figure


def fig_distribution(data, window: int = 40) -> go.Figure:
    """Epochs where we scored nothing must be dropped from BOTH traces.  Drawing
    boxes for a filtered list while the winner line used the unfiltered one made
    plotly append the missing epochs as fresh categories at the right-hand end,
    which is why the line doubled back on itself."""
    order = [e for e in sorted(data["epochs"])[-window:] if data["by_epoch"].get(e)]
    labels = [str(e) for e in order]
    figure = go.Figure()
    for e in order:
        mine = data["by_epoch"][e]
        figure.add_trace(go.Box(
            y=[m["score"] for m in mine], x=[str(e)] * len(mine), name=str(e),
            marker_color=OURS, line=dict(color=OURS), fillcolor="rgba(79,195,247,.16)",
            boxpoints="all", jitter=0.5, pointpos=0, showlegend=False,
            hovertext=[f"uid {m['uid']} rank {m['rank']}" for m in mine]))
    figure.add_trace(go.Scatter(
        x=labels, y=[data["epochs"][e]["winner"] for e in order],
        name="winner", mode="lines+markers",
        line=dict(color=WIN, width=2), marker=dict(size=4)))
    figure.add_trace(go.Scatter(
        x=labels, y=[data["epochs"][e]["top10_cut"] for e in order],
        name="top-10 cut", mode="lines",
        line=dict(color=WARN, width=1, dash="dot")))
    figure.update_layout(**LAYOUT)
    figure.update_yaxes(title="nanobody score", gridcolor="#242a36")
    figure.update_xaxes(title="epoch", gridcolor="#242a36", type="category",
                        categoryorder="array", categoryarray=labels)
    return figure


def fig_fleet(data) -> go.Figure:
    order = [e for e in sorted(data["epochs"]) if e in data["registered"]]
    held = [data["registered"][e] for e in order]
    scored = [len(data["by_epoch"].get(e) or []) for e in order]
    lost = [h - s for h, s in zip(held, scored)]
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=order, y=held, name="uids held (paid for)",
                                line=dict(color=WARN, width=2)))
    figure.add_trace(go.Scatter(x=order, y=scored, name="uids that scored",
                                fill="tozeroy", line=dict(color=OURS, width=2)))
    figure.add_trace(go.Bar(x=order, y=lost, name="held but did not score",
                            marker_color=BAD, opacity=0.55))
    figure.update_layout(barmode="overlay",
                         **LAYOUT)
    figure.update_yaxes(title="uids", gridcolor="#242a36")
    figure.update_xaxes(title="epoch", gridcolor="#242a36")
    return figure


def fig_ages(data) -> go.Figure:
    scored_epochs = [e for e in sorted(data["epochs"]) if data["by_epoch"].get(e)]
    if not scored_epochs:
        figure = go.Figure()
        figure.update_layout(**LAYOUT)
        figure.add_annotation(text="No scored fleet cohort is available",
                              xref="paper", yref="paper", x=0.5, y=0.5,
                              showarrow=False, font=dict(color=MUTED))
        return figure
    # A missed public round is an operational failure, not evidence that every
    # hotkey disappeared.  Use the most recent cohort that actually scored.
    newest = scored_epochs[-1]
    live = {e["hotkey"]: e for e in data["by_epoch"].get(newest, [])}
    lifespans = []
    for hotkey, start in data["first_seen"].items():
        seen = [x["epoch"] for x in data["ours"] if x["hotkey"] == hotkey]
        if hotkey not in live and len(seen) > 2:
            lifespans.append(max(seen) - min(seen) + 1)
    typical = st.median(lifespans) if lifespans else 17
    rows = sorted(live.values(), key=lambda r: data["first_seen"][r["hotkey"]])
    ages = [newest - data["first_seen"][r["hotkey"]] + 1 for r in rows]
    labels = [f"uid {r['uid']} · {r['hotkey'][:8]}…" for r in rows]
    colour = [BAD if a >= typical else (WARN if a >= typical * 0.8 else WIN) for a in ages]
    figure = go.Figure(go.Bar(x=ages, y=labels, orientation="h", marker_color=colour,
                              text=[f"{a} ep" for a in ages], textposition="outside"))
    figure.add_vline(x=typical, line=dict(color=MUTED, dash="dash"),
                     annotation_text=f"typical lifespan {typical:.0f} epochs",
                     annotation_font_color=MUTED)
    figure.add_annotation(text=f"latest scored cohort: epoch {newest}",
                          xref="paper", yref="paper", x=1, y=1.08,
                          xanchor="right", showarrow=False,
                          font=dict(color=MUTED, size=11))
    figure.update_layout(**LAYOUT)
    figure.update_xaxes(title="epochs since first seen", gridcolor="#242a36")
    figure.update_yaxes(gridcolor="#242a36")
    return figure


def fig_wins(data) -> go.Figure:
    order = sorted(data["epochs"])
    wins, cumulative, rate = [], [], []
    total = 0
    flags = []
    for e in order:
        mine = data["by_epoch"].get(e) or []
        won = bool(mine) and min(m["rank"] for m in mine) == 1
        # Missing rounds belong in fleet reliability, not the molecule-quality
        # denominator.  Preserve the x-axis gap while excluding it from the
        # trailing scored-epoch rate.
        flags.append((1 if won else 0) if mine else None)
        total += won
        cumulative.append(total)
    for i in range(len(order)):
        lo = max(0, i - 19)
        window = [flag for flag in flags[lo:i + 1] if flag is not None]
        rate.append(100 * sum(window) / len(window) if window else None)
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=order, y=cumulative, name="cumulative winning epochs",
                                line=dict(color=WIN, width=2), yaxis="y"))
    figure.add_trace(go.Scatter(x=order, y=rate,
                                name="winning-epoch rate, 20 scored epochs rolling %",
                                line=dict(color=OURS, width=2, dash="dot"), yaxis="y2"))
    layout = dict(LAYOUT)
    layout["yaxis2"] = dict(title="winning epochs %", overlaying="y", side="right",
                            gridcolor="rgba(0,0,0,0)")
    figure.update_layout(**layout)
    figure.update_yaxes(title="cumulative winning epochs", gridcolor="#242a36")
    figure.update_xaxes(title="epoch", gridcolor="#242a36")
    return figure


def competitor_stats(data, top: int = 12):
    """Per coldkey: fleet size, field-beating entries, and rates per entry.

    ``wins`` deliberately counts every entry that beats the best rival entry,
    rather than rank-one rows.  A separate ``winning_epochs`` field preserves
    the fleet-level measure that is capped at one success per epoch.
    """
    annotate_field_beating_entries(data["entries"])
    per_epoch, stats = {}, {}
    for e in data["entries"]:
        coldkey = e["coldkey"]
        bucket = per_epoch.setdefault(coldkey, {})
        bucket[e["epoch"]] = bucket.get(e["epoch"], 0) + 1
        row = stats.setdefault(
            coldkey,
            dict(entries=0, wins=0, top10=0, scores=[], winning_epochs=set()),
        )
        row["entries"] += 1
        row["wins"] += is_entry_win(e)
        row["top10"] += e["rank"] <= 10
        if e["rank"] == 1:
            row["winning_epochs"].add(e["epoch"])
        row["scores"].append(e["score"])
    rows = []
    for coldkey, row in stats.items():
        sizes = sorted(per_epoch[coldkey].values())
        rows.append(dict(
            coldkey=coldkey, ours=(coldkey == data.get("coldkey")),
            label=("US" if coldkey == data.get("coldkey") else coldkey[:8] + "..."),
            epochs=len(per_epoch[coldkey]), entries=row["entries"], wins=row["wins"],
            fleet=st.median(sizes), top10=row["top10"],
            winning_epochs=len(row["winning_epochs"]),
            win_per_entry=100 * row["wins"] / row["entries"],
            top10_per_entry=100 * row["top10"] / row["entries"],
            best=min(row["scores"])))
    # Select established competitors by total entry wins, then let each view
    # order that stable cohort by the rate it displays.  Sorting the candidate
    # set by rate would fill the table with one-off 1/1 and 2/2 coldkeys.
    rows.sort(key=lambda r: (-r["wins"], -r["win_per_entry"]))
    keep = rows[:top]
    if not any(r["ours"] for r in keep):
        mine = next((r for r in rows if r["ours"]), None)
        if mine:
            keep = keep[:-1] + [mine]
    return keep


def fig_competitor_efficiency(data, top: int = 12) -> go.Figure:
    """Fleet size against field-beating entries; bubble area is total entries."""
    rows = competitor_stats(data, top)
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=[r["fleet"] for r in rows], y=[r["win_per_entry"] for r in rows],
        mode="markers+text", text=[r["label"] for r in rows],
        textposition="top center",
        textfont=dict(color=[WIN if r["ours"] else MUTED for r in rows], size=11),
        marker=dict(size=[max(9, min(46, (r["entries"] ** 0.5) * 2.2)) for r in rows],
                    color=[WIN if r["ours"] else OURS for r in rows],
                    opacity=[0.95 if r["ours"] else 0.45 for r in rows],
                    line=dict(width=[2 if r["ours"] else 0 for r in rows], color=INK)),
        customdata=[[r["entries"], r["wins"], r["epochs"], r["best"],
                     r["winning_epochs"]] for r in rows],
        hovertemplate=("fleet %{x:.0f} uids<br>%{y:.2f}% field-beating entries"
                       "<br>%{customdata[1]} entry wins in %{customdata[0]} entries"
                       "<br>%{customdata[4]} winning epochs"
                       "<br>active %{customdata[2]} epochs"
                       "<br>best score %{customdata[3]:.0f}<extra></extra>"),
        showlegend=False))
    layout = dict(LAYOUT)
    # Unified-x hover is actively unhelpful for a sparse bubble chart: the
    # cursor can be on a named point while Plotly searches a different x slice.
    layout["hovermode"] = "closest"
    layout["hoverdistance"] = -1
    figure.update_layout(**layout)
    figure.update_xaxes(title="median fleet size (uids submitting per epoch)",
                        gridcolor="#242a36")
    figure.update_yaxes(title="field-beating entries (%)", gridcolor="#242a36")
    return figure


def fig_competitors(data, top: int = 12) -> go.Figure:
    """The same competitors as bars: rates on the left axis, fleet on the right."""
    rows = competitor_stats(data, top)
    names = [r["label"] for r in rows]
    figure = go.Figure()
    figure.add_trace(go.Bar(
        x=names, y=[r["win_per_entry"] for r in rows],
        name="field-beating entries %",
        marker_color=[WIN if r["ours"] else OURS for r in rows],
        text=[str(r["wins"]) + "w" for r in rows], textposition="outside"))
    figure.add_trace(go.Bar(
        x=names, y=[r["top10_per_entry"] for r in rows], name="top-10 per entry %",
        marker_color=[WIN if r["ours"] else "#3b4455" for r in rows]))
    figure.add_trace(go.Scatter(
        x=names, y=[r["fleet"] for r in rows], name="median fleet size", yaxis="y2",
        mode="markers+text", marker=dict(color=WARN, size=12, symbol="diamond"),
        text=[format(r["fleet"], ".0f") for r in rows], textposition="top center",
        textfont=dict(color=WARN, size=10)))
    layout = dict(LAYOUT)
    layout["yaxis2"] = dict(title="fleet size (uids)", overlaying="y", side="right",
                            gridcolor="rgba(0,0,0,0)", rangemode="tozero")
    figure.update_layout(barmode="group", **layout)
    figure.update_yaxes(title="percent of entries", gridcolor="#242a36")
    return figure


def fig_slots(data) -> go.Figure:
    buys = [s for s in data["slots"] if s["executed"]]
    if not buys:
        return go.Figure(layout=dict(title="No slot purchases recorded", **LAYOUT))
    daily: dict[str, list] = {}
    for b in buys:
        day = dt.datetime.fromtimestamp(
            b["time"], dt.timezone.utc
        ).date().isoformat()
        price = b.get("decision_recycle_tao")
        if price is None:
            price = b.get("recycle_tao")
        if price is not None:
            daily.setdefault(day, []).append(float(price))
    days = sorted(daily)
    figure = go.Figure()
    counts = [len(daily[d]) for d in days]
    figure.add_trace(go.Bar(
        x=days, y=[sum(daily[d]) for d in days],
        name="TAO spent", marker_color=OURS,
        text=[f"{count} uid" if count == 1 else f"{count} uids" for count in counts],
        textposition="outside", customdata=counts,
        hovertemplate=("%{x}<br>spent %{y:.4f} TAO"
                       "<br>%{customdata} registrations<extra></extra>"),
    ))
    figure.add_trace(go.Scatter(x=days, y=[st.median(daily[d]) for d in days],
                                name="median price per uid", yaxis="y2",
                                mode="lines+markers",
                                line=dict(color=WARN, width=2),
                                hovertemplate=("%{x}<br>median %{y:.4f} TAO/uid"
                                               "<extra></extra>")))
    layout = dict(LAYOUT)
    layout["yaxis2"] = dict(title="TAO / uid", overlaying="y", side="right",
                            gridcolor="rgba(0,0,0,0)")
    layout["yaxis"] = dict(title="TAO spent per day", gridcolor="#242a36",
                           rangemode="tozero")
    layout["hovermode"] = "x unified"
    figure.update_layout(**layout)
    figure.update_xaxes(title="day (UTC)", type="date", gridcolor="#242a36")
    return figure


def fig_slot_history(data) -> go.Figure:
    """Every observed recycle price, purchase, and held-fleet count over time."""
    rows = [s for s in data["slots"]
            if s.get("time") is not None and s.get("recycle_tao") is not None]
    if not rows:
        figure = go.Figure()
        figure.update_layout(**LAYOUT)
        figure.add_annotation(text="No slot-price observations recorded",
                              xref="paper", yref="paper", x=0.5, y=0.5,
                              showarrow=False, font=dict(color=MUTED))
        return figure
    rows.sort(key=lambda row: row["time"])
    when = [dt.datetime.fromtimestamp(row["time"], dt.timezone.utc) for row in rows]
    custom = [[row.get("epoch"), row.get("registered"), row.get("reason") or ""]
              for row in rows]
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=when, y=[row["recycle_tao"] for row in rows], name="recycle price",
        mode="lines", line=dict(color=OURS, width=1.4), customdata=custom,
        hovertemplate=("%{x|%Y-%m-%d %H:%M UTC}<br>epoch %{customdata[0]}"
                       "<br>price %{y:.6f} TAO<br>%{customdata[2]}<extra></extra>")))
    buys = [(stamp, row) for stamp, row in zip(when, rows) if row.get("executed")]
    if buys:
        figure.add_trace(go.Scatter(
            x=[stamp for stamp, _ in buys],
            y=[row.get("decision_recycle_tao") or row["recycle_tao"]
               for _, row in buys],
            name="purchased", mode="markers",
            marker=dict(color=WIN, size=10, symbol="diamond",
                        line=dict(color=INK, width=1)),
            customdata=[[row.get("epoch"), row.get("registered")]
                        for _, row in buys],
            hovertemplate=("PURCHASE · %{x|%Y-%m-%d %H:%M UTC}"
                           "<br>epoch %{customdata[0]}<br>price %{y:.6f} TAO"
                           "<br>fleet before purchase %{customdata[1]}<extra></extra>")))
    fleet_rows = [(stamp, row) for stamp, row in zip(when, rows)
                  if row.get("registered") is not None]
    figure.add_trace(go.Scatter(
        x=[stamp for stamp, _ in fleet_rows],
        y=[row["registered"] for _, row in fleet_rows],
        name="hotkeys held", mode="lines", line_shape="hv", yaxis="y2",
        line=dict(color=WARN, width=1.5),
        hovertemplate="held %{y:.0f}<extra></extra>"))
    layout = dict(LAYOUT)
    layout["hovermode"] = "closest"
    layout["yaxis"] = dict(title="registration recycle price (TAO)",
                           gridcolor="#242a36")
    layout["yaxis2"] = dict(title="hotkeys held", overlaying="y", side="right",
                            rangemode="tozero", gridcolor="rgba(0,0,0,0)",
                            tickformat="d")
    figure.update_layout(**layout)
    figure.update_xaxes(title="time (UTC)", gridcolor="#242a36")
    return figure


def headline(data) -> list[tuple[str, str, str]]:
    order = sorted(data["epochs"])
    newest = order[-1]
    ours_epochs = [e for e in order if data["by_epoch"].get(e)]
    wins = sum(1 for e in ours_epochs if min(m["rank"] for m in data["by_epoch"][e]) == 1)
    entries = len(data["ours"])
    entry_wins = sum(is_entry_win(e) for e in data["ours"])
    top10 = sum(1 for e in data["ours"] if e["rank"] <= 10)
    latest = data["by_epoch"].get(newest) or []
    held = data["registered"].get(newest)
    current_held = data.get("current_held")
    if held is not None and current_held is not None and held != current_held:
        held_note = f"{held} at epoch close · {current_held} now"
    elif current_held is not None:
        held_note = f"{current_held} held now"
    elif held is not None:
        held_note = f"{held} held at epoch close"
    else:
        held_note = "held: unknown"
    return [
        ("latest epoch", str(newest), f"field of {data['epochs'][newest]['field_size']}"),
        ("uids scoring", str(len(latest)), held_note),
        ("best this epoch", f"{min((m['score'] for m in latest), default=0):.0f}",
         f"winner {data['epochs'][newest]['winner']:.0f}"),
        ("winning epochs", str(wins),
         f"{100*wins/max(len(ours_epochs),1):.0f}% of {len(ours_epochs)} scored epochs"),
        ("field-beating entries", f"{100*entry_wins/max(entries,1):.1f}%",
         f"{entry_wins} of {entries} entries"),
        ("top-10 per entry", f"{100*top10/max(entries,1):.0f}%", f"{top10} of {entries} entries"),
    ]


def selector_replay_table(
    result: Path = Path(
        "analysis/nanobody-gap/monte-carlo-20260921/selector_results.json"
    ),
) -> str:
    """Render the held-out arbitrary-molecule selector comparison."""
    if not result.is_file():
        return '<p class="note">Selector replay artifact is not available.</p>'
    payload = json.loads(result.read_text())
    labels = {
        "target-iptm-only": "target ipTM only (shadow)",
        "fresh-draw-win-probability": "fresh-draw probability (selected)",
        "confidence-only": "confidence only (shadow)",
        "validator-plain": "plain aggregate (old)",
        "design-iiptm-only": "design ipTM only (shadow)",
    }
    rows = []
    for item in payload.get("summary", []):
        policy = str(item["policy"])
        decisions = int(item["held_out_decisions"])
        winner_picks = round(float(item["oracle_pick_rate"]) * decisions)
        css = ' class="us"' if policy == "fresh-draw-win-probability" else ""
        rows.append(
            "<tr%s><td>%s</td><td>%d/%d</td><td>%.1f%%</td>"
            "<td>%.1f</td><td>%.1f</td></tr>"
            % (
                css,
                labels.get(policy, policy),
                winner_picks,
                decisions,
                100 * float(item["oracle_pick_rate"]),
                float(item["mean_regret"]),
                float(item["median_regret"]),
            )
        )
    return (
        '<table class="grid"><thead><tr><th>selector</th><th>winner picks</th>'
        '<th>pick rate</th><th>mean regret</th><th>median regret</th></tr></thead>'
        '<tbody>%s</tbody></table>' % "".join(rows)
    )


def easynano_status_table(
    experiment_path: Path = Path("analysis/easynano-p20809/experiment.json"),
    status_path: Path = Path("analysis/easynano-p20809/dashboard_status.json"),
    summary_path: Path = Path(
        "analysis/easynano-p20809/results/artifacts/monte_carlo_summary.json"
    ),
) -> str:
    """Render the isolated P20809 basin-search experiment without host details."""
    if not experiment_path.is_file():
        return '<p class="note">P20809 EasyNano experiment is not configured.</p>'
    experiment = json.loads(experiment_path.read_text())
    target = experiment["target"]
    config = experiment["easynano"]
    anchors = experiment["anchors"]
    status = json.loads(status_path.read_text()) if status_path.is_file() else {}

    def safe(value) -> str:
        return html_lib.escape(str(value), quote=True)

    def compact(values: list[int]) -> str:
        runs = []
        start = previous = values[0]
        for value in values[1:]:
            if value != previous + 1:
                runs.append(str(start) if start == previous else f"{start}–{previous}")
                start = value
            previous = value
        runs.append(str(start) if start == previous else f"{start}–{previous}")
        return ", ".join(runs)

    expected = len(anchors) * len(config["seeds"])
    rows = [
        ("target", f"{target['accession']} / {target['structure']} chain {target['chain']}",
         f"{target['canonical_residue_range'][0]}–{target['canonical_residue_range'][1]}"),
        ("configured epitope", compact(target["configured_epitope_residue_ids"]),
         "canonical PDB residue IDs"),
        ("EasyNano epitope", compact(target["easynano_epitope_token_indices"]),
         "zero-based chain-token indices"),
        ("basin anchors", "ours 36 vs 5EvpaL11 35",
         f"epochs {anchors[0]['epoch']} and {anchors[1]['epoch']}"),
        ("search", f"seeds {', '.join(map(str, config['seeds']))}; {config['steps']} steps",
         f"{expected} isolated design runs"),
        ("status", status.get("stage", "configured"),
         f"{status.get('completed_runs', 0)}/{expected} runs complete; production untouched"),
    ]
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        for comparison in summary.get("comparisons", []):
            rows.append((
                f"{comparison['cohort']} best",
                f"mean ipTM {comparison['anchor_mean_iptm']:.3f} → "
                f"{comparison['best_mutated_mean_iptm']:.3f}",
                f"Δ {comparison['mean_iptm_change']:+.3f}; "
                f"Hamming {comparison['best_mutated_hamming']}; shared seeds 0–2",
            ))
    body = "".join(
        "<tr><td>%s</td><td>%s</td><td>%s</td></tr>" % tuple(map(safe, row))
        for row in rows
    )
    return (
        '<table class="grid"><thead><tr><th>item</th><th>value</th><th>scope</th>'
        '</tr></thead><tbody>%s</tbody></table>' % body
    )


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<!-- The collector rebuilds this static document in the background.  An open
     tab otherwise keeps the old DOM forever even after index.html changes. -->
<meta http-equiv="refresh" content="60">
<title>Nanobody Tracker</title>
<style>
 :root {{ color-scheme: dark; }}
 body {{ margin:0; background:{paper}; color:{ink};
        font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
 header {{ padding:22px 16px 0; max-width:1180px; margin:0 auto; }}
 h1 {{ font-size:20px; margin:0 0 4px; letter-spacing:.2px; }}
 .sub {{ color:{muted}; font-size:12px; }}
 .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(158px,1fr));
           gap:10px; max-width:1180px; margin:14px auto; padding:0 16px; }}
 .card {{ background:{card}; border:1px solid #242a36; border-radius:10px; padding:12px 14px; }}
 .card .k {{ color:{muted}; font-size:11px; text-transform:uppercase; letter-spacing:.6px; }}
 .card .v {{ font-size:23px; font-weight:600; margin:4px 0 2px; }}
 .card .n {{ color:{muted}; font-size:11px; }}
 nav {{ max-width:1180px; margin:6px auto 0; padding:0 16px; display:flex; gap:4px;
        flex-wrap:wrap; border-bottom:1px solid #242a36; }}
 nav button {{ background:none; border:0; border-bottom:2px solid transparent;
        color:{muted}; font:600 13px/1 inherit; padding:11px 14px; cursor:pointer; }}
 nav button:hover {{ color:{ink}; }}
 nav button[aria-selected="true"] {{ color:{ink}; border-bottom-color:{ours}; }}
 section[role=tabpanel] {{ display:none; }}
 section[role=tabpanel].on {{ display:block; }}
 .panel {{ max-width:1180px; margin:22px auto; padding:0 16px; }}
 .panel h2 {{ font-size:15px; margin:0 0 2px; font-weight:600; }}
 .panel p.note {{ color:{muted}; font-size:12px; margin:0 0 8px; }}
 table.grid {{ width:100%; border-collapse:collapse; font-size:13px;
        background:{card}; border:1px solid #242a36; border-radius:10px; }}
 table.grid th {{ text-align:right; color:{muted}; font-weight:600; font-size:11px;
        text-transform:uppercase; letter-spacing:.5px; padding:9px 12px;
        border-bottom:1px solid #242a36; }}
 table.grid th:first-child, table.grid td:first-child {{ text-align:left; }}
 table.grid td {{ text-align:right; padding:8px 12px; border-bottom:1px solid #1d2330;
        font-variant-numeric:tabular-nums; }}
 table.grid tr.us {{ background:rgba(102,217,166,.11); }}
 table.grid tr.us td {{ color:#66d9a6; font-weight:600; }}
 footer {{ color:{muted}; font-size:11px; text-align:center; padding:26px 16px 40px; }}
 @media (max-width:640px) {{ .cards {{ grid-template-columns:repeat(2,1fr); }} }}
</style></head><body>
<header><h1>Subnet 68 &middot; nanobody tracker</h1>
<div class="sub">{subtitle}</div></header>
<div class="cards">{cards}</div>
<nav role="tablist">{tabs}</nav>
{panels}
<footer>{footer}</footer>
<script>
(function () {{
  var buttons = document.querySelectorAll('nav button');
  function show(name) {{
    buttons.forEach(function (b) {{
      b.setAttribute('aria-selected', String(b.dataset.tab === name));
    }});
    document.querySelectorAll('section[role=tabpanel]').forEach(function (s) {{
      s.classList.toggle('on', s.dataset.tab === name);
    }});
    try {{ localStorage.setItem('nanodash.tab', name); }} catch (e) {{}}
    // Plotly sizes to a hidden container as 0-wide, so every plot in the tab
    // being revealed has to be told to resize once it is actually visible.
    if (window.Plotly) {{
      // The newly revealed tab has not completed layout synchronously.  A
      // resize in this same call can still see a zero-width hidden container,
      // leaving later money plots blank.  Wait for two paint frames, then size
      // every plot against its visible panel.
      window.requestAnimationFrame(function () {{
        window.requestAnimationFrame(function () {{
          document.querySelectorAll('section.on .js-plotly-plot').forEach(function (g) {{
            window.Plotly.Plots.resize(g);
          }});
        }});
      }});
    }}
    if (location.hash.slice(1) !== name) history.replaceState(null, '', '#' + name);
  }}
  buttons.forEach(function (b) {{
    b.addEventListener('click', function () {{ show(b.dataset.tab); }});
  }});
  var start = location.hash.slice(1);
  if (!start) {{ try {{ start = localStorage.getItem('nanodash.tab'); }} catch (e) {{}} }}
  show(start && document.querySelector('[data-tab="' + start + '"]') ? start : '{first_tab}');
}})();
</script>
</body></html>"""


TABS = [
    ("progress", "Our progress", [
        ("Are we getting better?", "Trailing 20-epoch field-beating-entry rate and top-10 rate per entry, with fleet size on the right axis. An entry wins when it beats the best molecule from every rival coldkey.", "progress"),
        ("Us against the strongest rivals", "Top-10 per entry on a 30-epoch trailing window. Rivals are the four best on this rate over the last 60 epochs (at least 60 entries); a window with under 20 entries is left blank.", "vsrivals"),
    ]),
    ("performance", "Performance", [
        ("Score per epoch", "Our best entry against the winner, the top-10 cut and the field median. Lower is better.", "scores"),
        ("Our score distribution", "Every uid we scored, last 40 epochs. The green line is the winning score.", "distribution"),
        ("Winning epochs over time", "Fleet-level epoch wins, capped at one per epoch and kept separate from field-beating entries. Missed rounds remain an operational gap.", "wins"),
    ]),
    ("fleet", "Fleet", [
        (
            "Selector Monte Carlo",
            "96 molecules, including 72 graph-unseen variants; 12 independent "
            "passes and 144 leave-one-pass-out basin decisions. Fresh-draw "
            "probability keeps the validator's unweighted ten-metric objective. "
            "Single-metric rows are shadow comparisons, not deployment choices.",
            "selector_replay",
        ),
        ("Fleet purchases and recycle price", "Every price observation and purchase, with the number of registered hotkeys on the right axis.", "slot_history"),
        ("Paid for vs actually scored", "Red bars are paid-for uids that submitted nothing. This is fleet reliability only and is excluded from molecule-quality rates.", "fleet"),
        ("Hotkey age and expected expiry", "Immunity is 5000 blocks; measured median lifespan is 17 epochs.", "expiry"),
        ("Current fleet age", "Ordered by first appearance on the leaderboard.", "ages"),
    ]),
    ("competition", "Competition", [
        ("Efficiency vs fleet size", "Field-beating entries against fleet size. If one miner owns positions #1 and #2 ahead of every rival, both entries count as wins.", "efficiency"),
        ("Competitor table", "Entry wins count every molecule beating the best rival molecule; winning epochs remain a separate fleet-level column.", "table"),
        ("Rates and fleet size", "Field-beating and top-10 entries as a percentage of submissions, with median fleet size on the right axis.", "competitors"),
    ]),
    ("money", "Money", [
        ("Yield per subnet", "Daily balance-to-balance yield by subnet. Alpha purchases and sales are removed; SN68 registration burn is subtracted as a real slot cost.", "pnl_history"),
        ("Cloud cost tracking", "AWS Bills accrued cost is kept separate from the forward Cost Explorer projection. SN2 operating economics exclude explicitly identified one-time migration traffic; Vast runtime is split into SN68 production, research and unattributed spend.", "cloud_costs"),
        ("AWS charge breakdown", "Cost Explorer charges grouped into the same operational categories as AWS Bills, with billed and recurring amounts kept separate. Transfer usage is collected from the API in GB.", "aws_cost_breakdown"),
        ("Infrastructure charges per day", "Daily operational cloud charges by provider and cost category. The one-time Ohio migration is retained in the AWS bill reconciliation but excluded here and from net operating economics.", "cloud_daily"),
        ("Money history", "Daily mining revenue, registration burn, total infrastructure spend, and net result. Provider and service detail is kept in the infrastructure panels above.", "money_history"),
        ("Current yield rate", "Whole-window daily rate by subnet, retained as the current summary beside the daily history.", "pnl"),
        ("Infrastructure against revenue", "vast.ai burn per project versus what that project earns. Compound is an investment that has not paid off yet.", "infra"),
        ("Projection", "Forward cumulative TAO at the current net rate, and at a 12-entry fleet.", "projection"),
        ("Slot purchases", "Every purchase grouped by UTC day: actual spend, median unit price and registration count.", "slots"),
    ]),
]


def render(data, out: Path, extra=None) -> Path:
    extra = extra or {}
    builders = dict(
        scores=lambda: fig_scores(data), distribution=lambda: fig_distribution(data),
        wins=lambda: fig_wins(data), fleet=lambda: fig_fleet(data),
        slot_history=lambda: fig_slot_history(data),
        ages=lambda: fig_ages(data), competitors=lambda: fig_competitors(data),
        efficiency=lambda: fig_competitor_efficiency(data),
        table=lambda: progress.competitor_table(competitor_stats(data)),
        selector_replay=selector_replay_table,
        progress=lambda: progress.fig_our_progress(data, _DATABASE),
        vsrivals=lambda: progress.fig_rates_vs_rivals(data),
        slots=lambda: fig_slots(data))
    builders.update(extra)

    first = True
    panels, tabs = [], []
    for key, label, entries in TABS:
        tabs.append('<button role="tab" data-tab="%s">%s</button>' % (key, label))
        blocks = []
        for title, note, name in entries:
            builder = builders.get(name)
            if builder is None:
                continue
            made = builder()
            if isinstance(made, str):          # a table, not a figure
                html = made
            else:
                _exact_epoch_axis(made)
                html = made.to_html(
                    full_html=False,
                    include_plotlyjs=("inline" if first else False),
                    config=dict(displaylogo=False, responsive=True,
                                modeBarButtonsToRemove=["lasso2d", "select2d"]))
                first = False
            blocks.append('<div class="panel"><h2>%s</h2><p class="note">%s</p>%s</div>'
                          % (title, note, html))
        panels.append('<section role="tabpanel" data-tab="%s">%s</section>'
                      % (key, "".join(blocks)))

    cards = "".join(
        '<div class="card"><div class="k">%s</div><div class="v">%s</div>'
        '<div class="n">%s</div></div>' % (k, v, n) for k, v, n in headline(data))
    order = sorted(data["epochs"])
    built = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    page = PAGE.format(
        paper=PAPER, card=CARD, ink=INK, muted=MUTED, ours=OURS, cards=cards,
        tabs="".join(tabs), panels="".join(panels), first_tab=TABS[0][0],
        subtitle="epochs %s-%s \u00b7 built %s" % (order[0], order[-1], built),
        footer="%d leaderboard entries \u00b7 %d ours \u00b7 rebuilt by nanobody_dash"
               % (len(data["entries"]), len(data["ours"])))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("data/nanobody_dash.sqlite"))
    parser.add_argument("--out", type=Path, default=Path("public/index.html"))
    parser.add_argument("--coldkey", default=DEFAULT_COLDKEY)
    parser.add_argument("--tao-usd", type=float, default=None,
                        help="TAO price; fetched from taostats when omitted")
    parser.add_argument("--hotkeys", type=Path, default=Path("data/hotkeys_age.json"))
    parser.add_argument("--vast", type=Path, default=Path("data/vast_by_project.json"))
    arguments = parser.parse_args()

    global _DATABASE
    _DATABASE = arguments.database
    data = load(arguments.database, arguments.coldkey)
    data["coldkey"] = arguments.coldkey
    if not data["epochs"]:
        print(json.dumps(dict(error="no epochs collected; run nanobody_dash.collect")))
        return 1

    from . import economics, money, panels
    connection = sqlite3.connect(arguments.database)
    tao_usd = economics.resolve_tao_usd(connection, arguments.tao_usd)
    burn = money.registration_burn(connection, 8)
    pnl = money.subnet_pnl(connection, tao_usd, burn)
    pnl_history = money.subnet_pnl_history(connection, tao_usd)
    money_history = money.daily_money_history(connection, tao_usd)
    infrastructure_history = money.infrastructure_cost_history(connection)
    split = {}
    if arguments.vast.is_file():
        split = json.loads(arguments.vast.read_text())
    cloud_state_path = Path("data/cloud_costs.json")
    cloud_state = json.loads(cloud_state_path.read_text()) \
        if cloud_state_path.is_file() else {}
    # revenue is attributed to the project that produced it, not the subnet:
    # the nanobody box earns SN68, the compound boxes have earned nothing yet
    revenue = {"sn68-prod": next((r["net_usd_per_day"] for r in pnl
                                  if r["netuid"] == 68), 0.0),
               "research": 0.0, "unattributed": 0.0}
    net_tao = money.portfolio_tao_per_day(connection)
    vast_tao = sum(v["usd_per_day"] for v in split.values()) / tao_usd if tao_usd else 0
    entries = len(data["by_epoch"].get(max(data["epochs"]), []))

    current_held = next((slot["registered"] for slot in reversed(data["slots"])
                         if slot.get("registered") is not None), None)
    data["current_held"] = current_held
    extra = dict(
        pnl=lambda: panels.fig_subnet_pnl(pnl, tao_usd),
        pnl_history=lambda: panels.fig_subnet_pnl_history(pnl_history),
        cloud_costs=lambda: panels.fig_cloud_costs(cloud_state),
        aws_cost_breakdown=lambda: panels.fig_aws_cost_breakdown(cloud_state),
        cloud_daily=lambda: panels.fig_infrastructure_cost_history(
            infrastructure_history),
        money_history=lambda: panels.fig_money_history(money_history),
        infra=lambda: panels.fig_infra(split, revenue),
        expiry=lambda: panels.fig_hotkey_expiry(arguments.hotkeys,
                                                current_total=current_held),
        projection=lambda: panels.fig_projection(net_tao - vast_tao, tao_usd,
                                                 0.21, entries))
    out = render(data, arguments.out, extra=extra)
    print(json.dumps(dict(
        written=str(out), bytes=out.stat().st_size, epochs=len(data["epochs"]),
        our_entries=len(data["ours"]), tao_usd=round(tao_usd, 2),
        net_tao_per_day=round(net_tao, 3),
        vast_usd_per_day=round(sum(v["usd_per_day"] for v in split.values()), 2),
        net_usd_per_day=round((net_tao - vast_tao) * tao_usd, 2))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
