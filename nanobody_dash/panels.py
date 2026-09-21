"""The money and fleet panels: per-subnet yield, infra cost, hotkey expiry."""
from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go

PAPER = "#0e1117"
CARD = "#161a23"
INK = "#e6e9ef"
MUTED = "#8b93a7"
OURS = "#4fc3f7"
WIN = "#66d9a6"
WARN = "#ff9f6b"
BAD = "#ef5f6b"
#: Net has to read against BOTH a green revenue bar and a red cost bar, so
#: it cannot be either of them. Amber on a dark outline is the only value
#: here that stays legible where the bars overlap it.
NET = "#ffd166"
LAYOUT = dict(paper_bgcolor=PAPER, plot_bgcolor=CARD, font=dict(color=INK, size=13),
              margin=dict(l=70, r=60, t=28, b=46),
              legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0,
                          bgcolor="rgba(0,0,0,0)"))


def fig_subnet_pnl(rows, tao_usd: float) -> go.Figure:
    """Gross yield, cost and net, per subnet, in TAO/day with a USD axis."""
    rows = [r for r in rows if abs(r["tao_per_day"]) > 1e-6]
    names = [r["name"] for r in rows]
    figure = go.Figure()
    figure.add_trace(go.Bar(x=names, y=[r["tao_per_day"] for r in rows],
                            name="gross TAO/day", marker_color=OURS,
                            text=[f"${r['tao_per_day']*tao_usd:,.0f}/d" for r in rows],
                            textposition="outside"))
    figure.add_trace(go.Bar(x=names, y=[-r["cost_tao_per_day"] for r in rows],
                            name="registration burn", marker_color=BAD))
    figure.add_trace(go.Scatter(
        x=names, y=[r["net_tao_per_day"] for r in rows], name="net TAO/day",
        mode="markers+text",
        marker=dict(color=NET, size=17, symbol="diamond",
                    line=dict(width=2, color=PAPER)),
        text=[f"net ${r['net_usd_per_day']:,.0f}/d" for r in rows],
        textposition="top center",
        textfont=dict(color=NET, size=13, family="monospace")))
    figure.update_layout(barmode="relative", **LAYOUT)
    figure.update_yaxes(title="TAO per day", gridcolor="#242a36", zerolinecolor=MUTED)
    return figure


def fig_infra(split: dict, revenue_usd: dict) -> go.Figure:
    """Infrastructure cost per project against the revenue it produces."""
    projects = sorted(split, key=lambda p: -split[p]["usd_per_day"])
    cost = [split[p]["usd_per_day"] for p in projects]
    earn = [revenue_usd.get(p, 0.0) for p in projects]
    net = [e - c for e, c in zip(earn, cost)]
    figure = go.Figure()
    figure.add_trace(go.Bar(x=projects, y=earn, name="revenue USD/day",
                            marker_color=WIN))
    figure.add_trace(go.Bar(x=projects, y=[-c for c in cost], name="vast.ai USD/day",
                            marker_color=BAD,
                            text=[f"{split[p]['boxes']}box/{split[p]['gpus']}gpu"
                                  for p in projects], textposition="inside"))
    figure.add_trace(go.Scatter(
        x=projects, y=net, name="net USD/day", mode="markers+text",
        marker=dict(color=NET, size=17, symbol="diamond",
                    line=dict(width=2, color=PAPER)),
        text=[f"${v:,.0f}" for v in net],
        textposition=["bottom center" if v < 0 else "top center" for v in net],
        textfont=dict(color=NET, size=13, family="monospace")))
    figure.update_layout(barmode="relative", **LAYOUT)
    figure.update_yaxes(title="USD per day", gridcolor="#242a36", zerolinecolor=MUTED)
    return figure


def fig_hotkey_expiry(path: Path, life_epochs: float = 17.0) -> go.Figure:
    """Every held uid, its age, and how long before it is culled."""
    if not path.is_file():
        return go.Figure(layout=LAYOUT)
    data = json.loads(path.read_text())
    epoch_length = 361
    rows = sorted(data["hotkeys"], key=lambda r: -(r.get("age_epochs") or 0))
    labels = [f"uid {r['uid']}" for r in rows]
    age = [r.get("age_epochs") or 0 for r in rows]
    left = [max(0.0, life_epochs - a) for a in age]
    hours = [round(v * epoch_length * 12 / 3600, 1) for v in left]
    colour = [BAD if not r.get("immune") else (WARN if a > life_epochs * 0.7 else WIN)
              for r, a in zip(rows, age)]
    figure = go.Figure()
    figure.add_trace(go.Bar(y=labels, x=age, orientation="h", name="age (epochs)",
                            marker_color=colour,
                            text=[f"{a:.1f} ep" for a in age], textposition="inside"))
    figure.add_trace(go.Bar(y=labels, x=left, orientation="h",
                            name="epochs until expected cull",
                            marker_color="#2b3342",
                            text=[f"~{h}h left" for h in hours], textposition="outside"))
    figure.add_vline(x=data.get("immunity_period", 5000) / epoch_length,
                     line=dict(color=MUTED, dash="dash"),
                     annotation_text="immunity ends", annotation_font_color=MUTED)
    figure.update_layout(barmode="stack", **LAYOUT)
    figure.update_xaxes(title="epochs", gridcolor="#242a36")
    return figure


def fig_projection(net_tao_per_day: float, tao_usd: float,
                   win_rate: float, entries: int, per_entry_win: float = 0.019,
                   tao_per_win: float = 1.87, days: int = 30) -> go.Figure:
    """Where the balance goes at the current rate, and at 12 entries."""
    x = list(range(days + 1))
    base = [net_tao_per_day * d for d in x]
    extra_entries = max(0, 12 - entries)
    uplift = extra_entries * per_entry_win * 20 * tao_per_win - extra_entries * 0.34
    better = [(net_tao_per_day + uplift) * d for d in x]
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=x, y=base, name=f"current ({entries} entries)",
                                line=dict(color=OURS, width=2)))
    figure.add_trace(go.Scatter(x=x, y=better, name="at 12 entries",
                                line=dict(color=WIN, width=2, dash="dot")))
    figure.add_hline(y=0, line=dict(color=MUTED, dash="dash"))
    figure.update_layout(**LAYOUT)
    figure.update_yaxes(title="cumulative TAO", gridcolor="#242a36")
    figure.update_xaxes(title="days from now", gridcolor="#242a36")
    return figure
