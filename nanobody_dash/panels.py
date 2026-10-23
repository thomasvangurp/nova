"""The money and fleet panels: per-subnet yield, infra cost, hotkey expiry."""
from __future__ import annotations

import datetime as dt
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

SUBNET_COLOURS = {
    68: WIN,
    2: OURS,
    62: "#b392f0",
    85: WARN,
    107: "#f78fb3",
}

REACTION_COLOURS = {
    1: OURS,
    2: "#b392f0",
    3: WIN,
    4: WARN,
    5: "#f78fb3",
}

COST_NAMES = {
    "aws:sn2": "SN2 AWS production",
    "aws:sn2:compute": "SN2 compute",
    "aws:sn2:storage": "SN2 storage",
    "aws:sn2:ec2-other": "SN2 EC2 other",
    "aws:sn2:data-transfer": "SN2 data transfer",
    "aws:sn2:lightsail": "SN2 Lightsail",
    "aws:sn2:vpc": "SN2 VPC/network",
    "aws:sn2:tax": "SN2 tax",
    "aws:sn2:other": "SN2 other",
    "aws:sn2:total": "SN2 AWS production",
    "vast.ai:sn68-prod": "SN68 Vast production (nanobody)",
    "vast.ai:sn68-compound": "SN68 Vast production (compound)",
    "vast.ai:sn68-research": "SN68 Vast research",
    "vast.ai:blueprint": "SN68 Vast blueprint",
    "vast.ai:bank": "SN68 Vast molecule bank",
    "vast.ai:research": "Vast research",
    "vast.ai:unattributed": "Vast unattributed",
}

COST_COLOURS = {
    "aws:sn2": "#b392f0",
    "aws:sn2:compute": "#b392f0",
    "aws:sn2:storage": "#8c7ae6",
    "aws:sn2:ec2-other": "#9b8af5",
    "aws:sn2:data-transfer": OURS,
    "aws:sn2:lightsail": "#5f9ea0",
    "aws:sn2:vpc": "#6f7787",
    "aws:sn2:tax": "#a970ff",
    "aws:sn2:other": MUTED,
    "aws:sn2:total": "#b392f0",
    "vast.ai:sn68-prod": WARN,
    "vast.ai:sn68-compound": "#ef5f6b",
    "vast.ai:sn68-research": "#d8895b",
    "vast.ai:blueprint": "#f0a04b",
    "vast.ai:bank": "#7a86b8",
    "vast.ai:research": "#6f7787",
    "vast.ai:unattributed": BAD,
}


def _empty(message: str) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(**LAYOUT)
    figure.add_annotation(
        text=message, xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(color=MUTED),
    )
    return figure


def fig_subnet_pnl_history(rows) -> go.Figure:
    """Daily net TAO yield for every subnet, with a compact point hover."""
    if not rows:
        return _empty("No daily balance intervals recorded yet")
    figure = go.Figure()
    netuids = sorted({row["netuid"] for row in rows})
    for netuid in netuids:
        series = [row for row in rows if row["netuid"] == netuid]
        figure.add_trace(go.Scatter(
            x=[row["day"] for row in series],
            y=[row["net_tao_per_day"] for row in series],
            name=series[0]["name"], mode="lines+markers",
            line=dict(color=SUBNET_COLOURS.get(netuid), width=2.2),
            marker=dict(size=6),
            customdata=[row["registration_tao_per_day"] for row in series],
            hovertemplate=(
                "%{x}<br>%{fullData.name}: %{y:.4f} TAO/day"
                "<br>registration %{customdata:.4f} TAO/day<extra></extra>"
                if netuid == 68 else
                "%{x}<br>%{fullData.name}: %{y:.4f} TAO/day<extra></extra>"
            ),
        ))
    figure.add_hline(y=0, line=dict(color=MUTED, dash="dash"))
    layout = dict(LAYOUT)
    layout["hovermode"] = "closest"
    layout["hoverdistance"] = -1
    figure.update_layout(**layout)
    figure.update_yaxes(title="net TAO per day", gridcolor="#242a36")
    figure.update_xaxes(title="day (UTC)", gridcolor="#242a36", type="date")
    return figure


#: Marker symbol per epoch outcome, for the epoch-boundary overlay in
#: fig_gap_history -- outcome is encoded by SYMBOL, not color, so it never
#: fights the per-reaction color coding the rest of the chart uses.
_OUTCOME_SYMBOL = {
    "won": "diamond",
    "cleared": "diamond-open",
    "blocked": "x",
}


def fig_gap_history(gap_rows, recent_history) -> go.Figure:
    """Each reaction's submission-readiness in ABSOLUTE score: solid line =
    selected_score over time, dashed line (same colour) = the required margin
    (floor). Where the solid crosses above the dashed we clear the bar. Plus
    markers for real epoch draws from the Pi's own log.
    """
    if not gap_rows:
        return _empty("No gap-check history recorded yet")
    figure = go.Figure()
    for reaction in sorted({row["reaction"] for row in gap_rows}):
        series = [row for row in gap_rows if row["reaction"] == reaction]
        colour = REACTION_COLOURS.get(reaction)
        xs = [dt.datetime.fromtimestamp(row["checked_at"], dt.timezone.utc)
              for row in series]
        latest_score = float(series[-1]["selected_score"])
        previous_distinct = next(
            (float(row["selected_score"]) for row in reversed(series[:-1])
             if abs(float(row["selected_score"]) - latest_score) > 1e-9),
            None,
        )
        change = latest_score - previous_distinct if previous_distinct is not None else None
        endpoint = f"rxn {reaction}: {latest_score:.3f}"
        if change is not None and abs(change) >= 0.0005:
            endpoint += f" ({change:+.3f})"
        figure.add_trace(go.Scatter(
            x=xs, y=[row["selected_score"] for row in series],
            name=f"rxn {reaction}", mode="lines+markers",
            line=dict(color=colour, width=2.2),
            # Plotly's implicit marker styling can render pale/white centres
            # on the dark theme.  Keep this series reaction-coloured and
            # borderless instead of letting the renderer add a highlight.
            marker=dict(
                size=[4] * (len(series) - 1) + [10],
                color=colour,
                line=dict(width=0),
            ),
            legendgroup=f"rxn{reaction}",
            hovertemplate=(
                "%{x}<br>rxn " + str(reaction) + " score: %{y:.4f}<extra></extra>"
            ),
        ))
        # A multi-reaction absolute axis is useful for clearance context but
        # visually compresses small real gains (RXN4's +0.011 looked flat).
        # Keep the shared scale and make the current value/change explicit at
        # the right edge instead of relying on pixel height or hover.
        figure.add_annotation(
            x=1.005, xref="paper", y=latest_score, yref="y",
            text=endpoint, showarrow=False, xanchor="left",
            font=dict(color=colour, size=11),
            bgcolor="rgba(14,17,23,.88)", bordercolor=colour, borderwidth=1,
        )
        # required margin as a separate same-colour dashed line
        figure.add_trace(go.Scatter(
            x=xs, y=[row["floor"] for row in series],
            name=f"rxn {reaction} margin", mode="lines",
            line=dict(color=colour, width=1.4, dash="dash"),
            legendgroup=f"rxn{reaction}", showlegend=False,
            hovertemplate=(
                "%{x}<br>rxn " + str(reaction) + " margin: %{y:.4f}<extra></extra>"
            ),
        ))
        epoch_points = [
            row for row in recent_history
            if row.get("reaction") == reaction and row.get("at") is not None
               and row.get("selected_score") is not None
        ]
        if not epoch_points:
            continue
        symbols = [
            _OUTCOME_SYMBOL["won"] if row.get("real_won")
            else _OUTCOME_SYMBOL["cleared"] if row.get("outcome") == "cleared"
            else _OUTCOME_SYMBOL["blocked"]
            for row in epoch_points
        ]
        figure.add_trace(go.Scatter(
            x=[dt.datetime.fromtimestamp(row["at"], dt.timezone.utc)
               for row in epoch_points],
            y=[row["selected_score"] for row in epoch_points],
            name=f"rxn {reaction} epochs", mode="markers",
            marker=dict(color=colour, size=11, symbol=symbols,
                        line=dict(width=1.5, color=PAPER)),
            legendgroup=f"rxn{reaction}",
            customdata=[row["epoch"] for row in epoch_points],
            hovertemplate=(
                "%{x}<br>epoch %{customdata}: %{y:.4f}<extra></extra>"
            ),
        ))
    layout = dict(LAYOUT)
    layout["hovermode"] = "closest"
    layout["hoverdistance"] = -1
    layout["margin"] = dict(LAYOUT["margin"], r=180)
    figure.update_layout(**layout)
    figure.update_yaxes(title="score (solid) vs required margin (dashed)", gridcolor="#242a36")
    figure.update_xaxes(title="time (UTC)", gridcolor="#242a36", type="date")
    return figure


def fig_proxy_accuracy(rows) -> go.Figure:
    """Each reaction's proxy model accuracy (Spearman rho on a held-out
    split) across retrain cycles. Separate from fig_gap_history: that is
    submission readiness (is the selected portfolio good enough), this is
    whether the model DRIVING candidate search is actually getting more
    accurate as it retrains -- no accuracy metric existed anywhere in the
    retraining pipeline before 2026-10-02, so this starts empty and
    accumulates one point per reaction per ~4h retrain cycle going forward.
    """
    if not rows:
        return _empty("No proxy accuracy history recorded yet")
    figure = go.Figure()
    for reaction in sorted({row["reaction"] for row in rows}):
        series = [row for row in rows if row["reaction"] == reaction]
        colour = REACTION_COLOURS.get(reaction)
        figure.add_trace(go.Scatter(
            x=[dt.datetime.fromtimestamp(row["at"], dt.timezone.utc)
               for row in series],
            y=[row["rho"] for row in series],
            name=f"rxn {reaction}", mode="lines+markers",
            line=dict(color=colour, width=2.2), marker=dict(size=6),
            customdata=[[row.get("mae"), row.get("n"), row.get("train_rows")]
                        for row in series],
            hovertemplate=(
                "%{x}<br>rxn " + str(reaction) + " rho: %{y:.4f}"
                "<br>mae: %{customdata[0]:.4f} · n: %{customdata[1]} · "
                "train_rows: %{customdata[2]}<extra></extra>"
            ),
        ))
    figure.add_hline(y=0, line=dict(color=MUTED, dash="dash"))
    layout = dict(LAYOUT)
    layout["hovermode"] = "closest"
    layout["hoverdistance"] = -1
    figure.update_layout(**layout)
    figure.update_yaxes(title="held-out Spearman rho", gridcolor="#242a36")
    figure.update_xaxes(title="time (UTC)", gridcolor="#242a36", type="date")
    return figure


def fig_money_history(rows) -> go.Figure:
    """Recorded reward receipts and costs, with portfolio on a separate axis."""
    if not rows:
        return _empty("No daily money history recorded yet")
    days = [row["day"] for row in rows]
    figure = go.Figure()
    # Component bars sum to recorded net. Never stack a cost beneath a net
    # value that already deducted it. Today's income/cost periods are partial.
    figure.add_trace(go.Bar(
        x=days, y=[row["gross_usd"] for row in rows],
        name="recorded rewards (USD)", marker_color=WIN,
        hovertemplate="%{x}<br>recorded rewards $%{y:,.2f}<extra></extra>",
    ))
    figure.add_trace(go.Bar(
        x=days, y=[-row.get("registration_usd", 0) for row in rows],
        name="registration cost", marker_color="#e6a66b",
        hovertemplate="%{x}<br>registration $%{y:,.2f}<extra></extra>",
    ))
    infrastructure = [row.get("infra_usd", sum(
        float(value or 0) for value in (row.get("infrastructure") or {}).values()
    )) for row in rows]
    figure.add_trace(go.Bar(
        x=days, y=[-value for value in infrastructure],
        name="infrastructure (daily estimate)", marker_color="#b392f0",
        hovertemplate="%{x}<br>infrastructure $%{customdata:,.2f}<extra></extra>",
        customdata=infrastructure,
    ))
    figure.add_trace(go.Scatter(
        x=days, y=[row["net_usd"] for row in rows], name="recorded net (USD)",
        mode="lines+markers", connectgaps=False, line=dict(color="#ffffff", width=1.5),
        hovertemplate="%{x}<br>recorded net $%{y:,.2f}<extra></extra>",
    ))
    if rows[-1].get("income_status") == "incomplete":
        figure.add_annotation(text="Latest day incomplete — net pending; reward receipts may lag wins",
                              x=0, y=1.12, xref="paper", yref="paper", showarrow=False,
                              font=dict(color=MUTED, size=12), xanchor="left")
    figure.add_trace(go.Scatter(
        x=days, y=[row["portfolio_tao"] for row in rows],
        name="portfolio TAO", yaxis="y2", mode="lines+markers",
        line=dict(color=OURS, width=1.8, dash="dot"),
        hovertemplate="%{x}<br>portfolio %{y:,.3f} TAO<extra></extra>",
    ))
    layout = dict(LAYOUT)
    layout["barmode"] = "relative"
    layout["hovermode"] = "x unified"
    layout["yaxis"] = dict(title="USD per day", gridcolor="#242a36",
                            zerolinecolor=MUTED)
    layout["yaxis2"] = dict(title="portfolio (TAO)", overlaying="y", side="right",
                             gridcolor="rgba(0,0,0,0)")
    figure.update_layout(**layout)
    figure.update_xaxes(title="day (UTC)", gridcolor="#242a36", type="date")
    return figure


def fig_infrastructure_cost_history(rows) -> go.Figure:
    """Stacked daily operational infrastructure charges by provider/category."""
    if not rows:
        return _empty("No daily infrastructure charges recorded yet")
    days = [row["day"] for row in rows]
    item_order = {item: index for index, item in enumerate((
        "aws:sn2:compute", "aws:sn2:storage", "aws:sn2:ec2-other",
        "aws:sn2:data-transfer", "aws:sn2:lightsail", "aws:sn2:vpc",
        "aws:sn2:tax", "aws:sn2:other", "aws:sn2:total",
        "vast.ai:sn68-prod", "vast.ai:sn68-compound", "vast.ai:research",
        "vast.ai:unattributed",
    ))}
    items = sorted({
        item for row in rows for item in (row.get("infrastructure") or {})
    }, key=lambda item: (item_order.get(item, 999), item))

    daily_hover = []
    for row in rows:
        values = row.get("infrastructure") or {}
        total = sum(float(value or 0) for value in values.values())
        aws_total = sum(float(value or 0) for item, value in values.items()
                        if str(item).startswith("aws:"))
        vast_total = sum(float(value or 0) for item, value in values.items()
                         if str(item).startswith("vast.ai:"))
        lines = [f"<b>{row['day']}</b>", f"total ${total:,.2f}"]
        for prefix, label, subtotal in (
            ("aws:", "AWS", aws_total), ("vast.ai:", "Vast.ai", vast_total),
        ):
            if abs(subtotal) < .005:
                continue
            lines.append(f"<br><b>{label} ${subtotal:,.2f}</b>")
            for item in items:
                amount = float(values.get(item) or 0)
                if not str(item).startswith(prefix) or abs(amount) < .005:
                    continue
                lines.append(f"{COST_NAMES.get(item, item)} ${amount:,.2f}")
        daily_hover.append("<br>".join(lines))

    figure = go.Figure()
    for item in items:
        values = [(row.get("infrastructure") or {}).get(item, 0.0)
                  for row in rows]
        label = COST_NAMES.get(item, item)
        figure.add_trace(go.Bar(
            x=days, y=values, name=label,
            marker_color=COST_COLOURS.get(item, WARN),
            customdata=daily_hover,
            hovertemplate="%{customdata}<extra></extra>",
        ))
    figure.update_layout(**LAYOUT, barmode="stack", hovermode="closest",
                         hoverdistance=-1)
    figure.update_yaxes(title="operational USD per day", gridcolor="#242a36")
    figure.update_xaxes(title="day (UTC)", gridcolor="#242a36", type="date")
    return figure


def fig_cloud_costs(state: dict) -> go.Figure:
    """Provider/project cost status without mixing monthly and daily units."""
    aws = state.get("aws") or {}
    vast = state.get("vast") or {}
    payments = state.get("payments") or {}
    aws_live_actual = aws.get("bill_to_date_usd", aws.get("mtd_usd"))
    aws_reference = aws.get("bills_reference_usd")
    aws_actual = aws_reference if aws_reference is not None else aws_live_actual
    aws_operating = aws.get("operating_mtd_usd", aws_actual)
    aws_excluded = aws.get("one_time_excluded_usd", 0)
    aws_projection = aws.get(
        "projected_month_end_usd", aws.get("estimated_month_usd"))
    aws_run_rate = aws.get("operating_usd_per_day")
    aws_period = str(aws.get("period") or "current month")
    breakdown = aws.get("operating_breakdown_mtd_usd") or {}
    breakdown_order = (
        ("compute", "compute"), ("storage", "storage"),
        ("ec2-other", "EC2 other"),
        ("data-transfer", "transfer"), ("lightsail", "Lightsail"),
        ("vpc", "VPC"), ("tax", "tax"), ("other", "other"),
    )
    breakdown_text = " · ".join(
        f"{label} ${float(breakdown[key]):,.2f}"
        for key, label in breakdown_order if abs(float(breakdown.get(key) or 0)) >= .005
    ) or "breakdown pending"
    reference_as_of = str(aws.get("bills_reference_as_of") or "")
    vast_production = float(
        vast.get("sn68_production_usd_per_day")
        if vast.get("sn68_production_usd_per_day") is not None
        else vast.get("sn68_usd_per_day") or 0
    )
    vast_research = sum(float(vast.get(key) or 0) for key in (
        "sn68_research_usd_per_day", "research_usd_per_day",
        "unattributed_usd_per_day",
    ))
    # Compound's five dedicated reaction boxes (see infra.py's
    # _COMPOUND_LABEL) used to fall into the generic "research" bucket above,
    # silently inflating it to ~$750/day when compound's real isolated cost
    # is ~$220/day -- fixed 2026-10-02 by giving compound its own scope.
    vast_compound = float(vast.get("sn68_compound_usd_per_day") or 0)
    vast_blueprint = float(vast.get("blueprint_usd_per_day") or 0)
    vast_bank = float(vast.get("bank_usd_per_day") or 0)
    cells = [
        ["AWS Bills", "AWS operations", "Vast.ai", "Vast.ai compound", "Funding guard"],
        ["SN2 · Ohio", "SN2 · Ohio", "SN68 prod + research", "SN68 compound (5 reaction boxes)", "cloud bills"],
        [
            (f"${float(aws_actual):,.2f} accrued bill" if aws_actual is not None
             else "daily detail pending API access"),
            (f"${float(aws_operating):,.2f} operating MTD"
             if aws_operating is not None else "operating cost unavailable"),
            f"${vast_production:,.2f}/day production",
            f"${vast_compound:,.2f}/day",
            str(payments.get("cloud_funding_source") or "not configured"),
        ],
        [
            (f"reference {reference_as_of}" if reference_as_of
             else f"{aws_period} to date"),
            (f"${float(aws_projection):,.2f} projected month-end"
             if aws_projection is not None else "forecast unavailable"),
            (f"${vast_research:,.2f}/day research"
             f"<br>${vast_blueprint:,.2f}/day blueprint"
             f"<br>${vast_bank:,.2f}/day bank"),
            "still net-negative, under active optimization",
            "agent cannot charge a real card"
            if payments.get("real_card_allowed") is False
            else "payment guard not set",
        ],
        [
            (f"Cost Explorer ${float(aws_live_actual):,.2f} now"
             if aws_live_actual is not None else
             str(aws.get("status") or "not collected")),
            (f"{breakdown_text}<br>"
             + (f"${float(aws_run_rate):,.2f}/day clean run rate · "
                if aws_run_rate is not None else "")
             + f"${float(aws_excluded or 0):,.2f} one-time excluded"),
            (f"{vast.get('status') or 'not collected'} · "
             f"{int(vast.get('running_instances') or 0)} running"),
            "see Compound tab for win rate and real wins",
            "read-only collectors",
        ],
    ]
    figure = go.Figure(go.Table(
        header=dict(values=["provider", "project", "current", "period", "status"],
                    fill_color="#202633", font=dict(color=INK, size=13),
                    align="left", height=32),
        cells=dict(values=cells, fill_color=CARD, font=dict(color=INK, size=12),
                   align="left", height=31),
    ))
    figure.update_layout(**LAYOUT, height=270)
    return figure


def fig_aws_cost_breakdown(state: dict) -> go.Figure:
    """Bills-style AWS categories, recurring adjustment, and metered usage."""
    aws = state.get("aws") or {}
    billed = aws.get("breakdown_mtd_usd") or {}
    operating = aws.get("operating_breakdown_mtd_usd") or {}
    if not billed:
        return _empty("AWS cost breakdown has not been collected yet")

    usage_rows = aws.get("usage_breakdown_mtd") or []
    transfer_usage = {str(row.get("usage_type") or ""): row
                      for row in usage_rows}

    def quantity(needle: str) -> float:
        return sum(
            float(row.get("quantity") or 0)
            for key, row in transfer_usage.items() if needle in key
        )

    transfer_detail = (
        f"internet out {quantity('USE2-DataTransfer-Out-Bytes'):,.3f} GB"
        f" · regional {quantity('USE2-DataTransfer-Regional-Bytes'):,.3f} GB"
        f" · internet in {quantity('USE2-DataTransfer-In-Bytes'):,.3f} GB"
    )
    definitions = [
        ("Elastic Compute Cloud", ("compute", "storage", "ec2-other"),
         "instances + EBS storage"),
        ("Data Transfer", ("data-transfer",), transfer_detail),
        ("Lightsail", ("lightsail",), "service runtime"),
        ("Virtual Private Cloud", ("vpc",), "network/public IPv4"),
        ("Tax", ("tax",), "AWS tax line"),
        ("Other", ("other", "total"), "remaining AWS services"),
    ]
    labels, billed_values, operating_values, usage_values = [], [], [], []
    for label, keys, usage in definitions:
        billed_value = sum(float(billed.get(key) or 0) for key in keys)
        operating_value = sum(float(operating.get(key) or 0) for key in keys)
        if abs(billed_value) < .005 and abs(operating_value) < .005:
            continue
        labels.append(label)
        billed_values.append(f"${billed_value:,.2f}")
        operating_values.append(f"${operating_value:,.2f}")
        usage_values.append(usage)

    labels.append("<b>Total</b>")
    billed_values.append(f"<b>${float(aws.get('bill_to_date_usd') or 0):,.2f}</b>")
    operating_values.append(
        f"<b>${float(aws.get('operating_mtd_usd') or 0):,.2f}</b>"
    )
    usage_values.append(
        f"${float(aws.get('one_time_excluded_usd') or 0):,.2f} one-time excluded"
    )

    figure = go.Figure(go.Table(
        columnwidth=[1.1, .65, .72, 2.4],
        header=dict(
            values=["AWS charge category", "accrued bill", "operating MTD",
                    "metered usage / scope"],
            fill_color="#202633", font=dict(color=INK, size=13),
            align="left", height=32,
        ),
        cells=dict(
            values=[labels, billed_values, operating_values, usage_values],
            fill_color=CARD, font=dict(color=INK, size=12),
            align="left", height=31,
        ),
    ))
    figure.update_layout(**LAYOUT, height=max(260, 80 + len(labels) * 34))
    return figure


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


def fig_hotkey_expiry(path: Path, life_epochs: float = 17.0,
                      current_total: int | None = None) -> go.Figure:
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
    total = len(rows) if current_total is None else int(current_total)
    detail_block = data.get("block")
    detail = f" · age detail at block {detail_block}" if detail_block is not None else ""
    figure.add_annotation(text=f"{total} hotkeys currently registered{detail}",
                          xref="paper", yref="paper", x=1, y=1.08,
                          xanchor="right", showarrow=False,
                          font=dict(color=MUTED, size=11))
    figure.update_layout(barmode="stack", **LAYOUT)
    figure.update_xaxes(title="epochs", gridcolor="#242a36")
    return figure


def fig_projection(net_tao_per_day: float, tao_usd: float,
                   win_rate: float, entries: int, per_entry_win: float = 0.019,
                   tao_per_win: float = 1.87, days: int = 30) -> go.Figure:
    """Where the balance goes at the current rate, and -- only while the
    fleet is still smaller than 12 -- at a hypothetical 12-entry fleet.

    extra_entries is 0 once entries >= 12, which makes uplift exactly 0 and
    the second line IDENTICAL to the first: a real bug confirmed live
    2026-10-02 once the fleet actually reached 12 -- the chart kept drawing
    a second, perfectly-overlapping trace as if it still meant something,
    which read as a flat/broken line next to the real one. Once at or past
    12, that comparison is simply no longer meaningful; drop the trace
    instead of plotting a degenerate duplicate.
    """
    x = list(range(days + 1))
    base = [net_tao_per_day * d for d in x]
    extra_entries = max(0, 12 - entries)
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=x, y=base, name=f"current ({entries} entries)",
                                line=dict(color=OURS, width=2)))
    if extra_entries > 0:
        uplift = extra_entries * per_entry_win * 20 * tao_per_win - extra_entries * 0.34
        better = [(net_tao_per_day + uplift) * d for d in x]
        figure.add_trace(go.Scatter(x=x, y=better, name="at 12 entries",
                                    line=dict(color=WIN, width=2, dash="dot")))
    else:
        figure.add_annotation(
            text=f"fleet already at {entries} entries -- no further-growth scenario to show",
            xref="paper", yref="paper", x=0.5, y=0.95, showarrow=False,
            font=dict(color=MUTED, size=11),
        )
    figure.add_hline(y=0, line=dict(color=MUTED, dash="dash"))
    figure.update_layout(**LAYOUT)
    figure.update_yaxes(title="cumulative TAO", gridcolor="#242a36")
    figure.update_xaxes(title="days from now", gridcolor="#242a36")
    return figure
