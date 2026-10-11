"""Our progress over time, and the same measures for our competitors.

Aggregate rates hide the only thing that matters: whether we are getting
better.  Everything here is a trailing window so a single lucky epoch cannot
move the line, and the same window is applied to rivals so the comparison is
like-for-like.
"""
from __future__ import annotations

import json
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


MULTIPASS_COMPARISON = Path(__file__).resolve().parent / "multipass_comparison.json"

# How each epoch's multipass numbers were measured. New box output carries a
# per-epoch ``method`` block (tools/multipass_all_rivals.py). Epochs written
# before that are classified from the 2026-10-09 audit (docs/OBSERVATION_LABELS.md):
#
# * <= 25612: every box JSON was rewritten on 2026-10-09 (17:26-18:13Z) by
#   one-off API back-fills: sum10 of the VALIDATOR'S PUBLISHED single-pass
#   metrics vs the HF archive. Verified: rho=1.000 against sum10 of the API
#   metrics for 25533/25560/25593/25603/25610-25612. All values are integers.
# * >= 25613: 10-pass means of our local fast scorer, sum10 vs the HF archive.
#   Against the API metrics rho is only 0.2-0.5.
#
# Same formula and the same ~1.2e5 scale, but a different measurement, so
# the panels never join, smooth or compare across the boundary.
MULTIPASS_METHOD_LABELS = {
    "validator_published_single_pass": "validator-published metrics, 1 pass",
    "local_fast_scorer_multipass_mean": "local fast scorer, multi-pass mean",
}
# field (method["field"]) is a THIRD axis a formula+measurement match can
# still hide: 2026-10-11 switched the ranking field from the static,
# ever-growing HF archive (~1.2e5 scale) to a pooled real last-10-epochs
# field (~1-2e3 scale) without changing formula or measurement at all, so
# the two-part key above alone would have joined a ~100x scale break into
# one line. Folded into the regime key below instead of its own axis, since
# every panel here (_smooth, _gap_at_method_change, _mark_method_boundaries)
# already keys off one opaque method string.
MULTIPASS_FIELD_LABELS = {
    "hf_archive_full": "full archive",
    "recent_validator_epochs": "recent 10 epochs",
}
LEGACY_API_BACKFILL_LAST_EPOCH = 25612


def _multipass_method(epoch: int, row) -> str:
    """Measurement regime key for one epoch row (see MULTIPASS_METHOD_LABELS)."""
    method = row.get("method") if isinstance(row, dict) else None
    if isinstance(method, dict) and method.get("measurement"):
        key = f"{method.get('formula') or '?'}:{method['measurement']}"
        field = method.get("field")
        return f"{key}:{field}" if field else key
    if epoch <= LEGACY_API_BACKFILL_LAST_EPOCH:
        return "sum10:validator_published_single_pass"
    return "sum10:local_fast_scorer_multipass_mean"


def _method_label(method: str) -> str:
    formula, _, rest = method.partition(":")
    measurement, _, field = rest.partition(":")
    label = f"{formula} · {MULTIPASS_METHOD_LABELS.get(measurement, measurement)}"
    if field:
        label += f" ({MULTIPASS_FIELD_LABELS.get(field, field)})"
    return label


def _method_segments(epochs, methods):
    """``[(method, first_epoch, last_epoch)]`` runs of one method, in order."""
    out = []
    for epoch, method in zip(epochs, methods):
        if out and out[-1][0] == method:
            out[-1] = (method, out[-1][1], epoch)
        else:
            out.append((method, epoch, epoch))
    return out


def _gap_at_method_change(xs, ys, methods_by_x, *extra):
    """Insert a None point wherever the method changes, so no line crosses a
    regime boundary. ``extra`` lists are padded in step (with None/"")."""
    out_x, out_y = [], []
    out_extra = [[] for _ in extra]
    previous = None
    for i, (x, y) in enumerate(zip(xs, ys)):
        method = methods_by_x.get(x)
        if previous is not None and method != previous:
            out_x.append(x - 0.5); out_y.append(None)
            for lst in out_extra:
                lst.append(None)
        previous = method
        out_x.append(x); out_y.append(y)
        for lst, src in zip(out_extra, extra):
            lst.append(src[i])
    return (out_x, out_y, *out_extra)


def _mark_method_boundaries(figure, segments, *, categorical_index=None) -> None:
    """Dashed line at every boundary plus a label over each regime.

    A single segment (the common case once the whole series is on one
    consistent regime -- see rescale_multipass_history.py) has nothing to
    distinguish, so it draws nothing: the label previously shown even then
    sat at the same y as the legend (both ~y=1.0-1.02 paper coords) and
    overlapped it, worst on narrow/mobile widths where there is no spare
    vertical room between the two (user, 2026-10-11: "like crap on mobile").
    """
    if len(segments) <= 1:
        return
    for i, (method, first, last) in enumerate(segments):
        if categorical_index is not None:
            x0, x1 = categorical_index[str(first)], categorical_index[str(last)]
        else:
            x0, x1 = first, last
        if i > 0:
            figure.add_vline(x=x0 - 0.5, line=dict(color=WARN, dash="dash", width=1.5))
        figure.add_annotation(
            x=(x0 + x1) / 2, y=1.0, xref="x", yref="paper", yanchor="bottom",
            text=_method_label(method), showarrow=False,
            font=dict(color=WARN, size=11))


def _normalize_multipass_actor(value):
    """Return ``{"mean", "stdev", "fleet_size", "scores"}`` or None.

    Accepts both the new dict shape and the legacy flat float, so an old
    ``multipass_comparison.json`` keeps rendering.  A zero or null ``stdev`` is
    normalised to None so error bars are simply omitted.
    """
    if isinstance(value, dict):
        try:
            mean = float(value.get("mean"))
        except (TypeError, ValueError):
            return None
        stdev = value.get("stdev")
        try:
            stdev = float(stdev) if stdev not in (None, "") else None
        except (TypeError, ValueError):
            stdev = None
        if stdev in (None, 0):
            stdev = None
        return dict(mean=mean, stdev=stdev,
                    fleet_size=value.get("fleet_size"),
                    scores=value.get("scores"))
    try:
        return dict(mean=float(value), stdev=None, fleet_size=None, scores=None)
    except (TypeError, ValueError):
        return None


def _multipass_rivals(row, rival):
    """Yield only the target rival (5GhREcsK), ignoring the noisy rest.

    The legacy ``{"rival": <float>}`` shape becomes a single rival keyed by
    the ``rival`` parameter's coldkey prefix; the new ``rivals`` mapping is
    filtered to the coldkey that starts with ``rival`` so the panel stays a
    clean us-vs-them comparison.  Rivals without a usable ``mean`` are dropped.
    """
    rivals = row.get("rivals")
    if not isinstance(rivals, dict):
        rivals = {rival: row.get("rival")} if "rival" in row else {}
    for coldkey, value in rivals.items():
        if not str(coldkey).startswith(rival):
            continue
        actor = _normalize_multipass_actor(value)
        if actor is None:
            continue
        yield str(coldkey), actor


def _multipass_all_rivals(row):
    """Yield every rival with a usable mean, unfiltered by coldkey."""
    rivals = row.get("rivals")
    if not isinstance(rivals, dict):
        return
    for coldkey, value in rivals.items():
        actor = _normalize_multipass_actor(value)
        if actor is None:
            continue
        yield str(coldkey), actor


def _multipass_hover(actor) -> str:
    """``fleet=13 · σ=88`` for the hover text, ``fleet n/a`` when empty."""
    parts = []
    fleet = actor.get("fleet_size")
    if fleet is not None:
        parts.append(f"fleet={int(fleet)}")
    if actor.get("stdev") is not None:
        parts.append(f"σ={actor['stdev']:,.0f}")
    return " · ".join(parts) if parts else "fleet n/a"


def _trailing_mean(xs, ys, window=10):
    """Trailing-window mean of ``ys`` over the last ``window`` non-None values.

    A per-epoch multi-pass rank-sum is noisy (one epoch's sample is only ~10-20
    sequences), so the strongest-rivals chart plots this smoothed value instead
    of the raw per-epoch point.  Missing (``None``) values are kept as gaps and
    are not fed into the window.
    """
    out_x, out_y, win = [], [], []
    for x, y in zip(xs, ys):
        if y is None:
            out_x.append(x)
            out_y.append(None)
            continue
        win.append(y)
        if len(win) > window:
            win.pop(0)
        out_x.append(x)
        out_y.append(sum(win) / len(win))
    return out_x, out_y


def _empty_multipass(figure, note, ytitle) -> go.Figure:
    figure.update_layout(**LAYOUT)
    figure.add_annotation(text=note, xref="paper", yref="paper", x=0.5, y=0.5,
                          showarrow=False, font=dict(color=MUTED))
    figure.update_yaxes(title=ytitle, gridcolor="#242a36")
    figure.update_xaxes(title="epoch", gridcolor="#242a36")
    return figure


def fig_multipass_vs_rival(path: Path | None = None,
                           rival: str = "5GhREcsK") -> go.Figure:
    """US vs every rival's multi-pass mean rank-sum per epoch (lower = better).

    The values come from multi-pass replay scoring, which is too expensive for
    the collector to keep in the dashboard database, so they are read from
    ``multipass_comparison.json`` instead.  Missing epochs (or a missing file)
    are skipped rather than crashing the build.  Each rival is one line with
    its fleet stdev as error bars; a rival without a ``mean`` is skipped.
    """
    path = path or MULTIPASS_COMPARISON
    figure = go.Figure()
    if not path.is_file():
        return _empty_multipass(figure, "No multi-pass comparison data yet",
                                "multi-pass mean rank-sum (lower = better)")

    payload = json.loads(path.read_text())
    epochs = sorted(int(key) for key in payload)
    if not epochs:
        return _empty_multipass(figure, "No multi-pass comparison data yet",
                                "multi-pass mean rank-sum (lower = better)")

    ours_x, ours_y, ours_err, ours_hover = [], [], [], []
    rival_buckets: dict[str, dict] = {}
    methods: dict[int, str] = {}
    for epoch in epochs:
        row = payload.get(str(epoch))
        if not isinstance(row, dict):
            continue
        methods[epoch] = _multipass_method(epoch, row)
        tag = _method_label(methods[epoch])
        ours = _normalize_multipass_actor(row.get("ours"))
        ours_x.append(epoch)
        if ours is None:
            ours_y.append(None); ours_err.append(None); ours_hover.append("")
        else:
            ours_y.append(ours["mean"]); ours_err.append(ours["stdev"])
            ours_hover.append(_multipass_hover(ours) + "<br>" + tag)
        for coldkey, actor in _multipass_rivals(row, rival):
            bucket = rival_buckets.setdefault(
                coldkey, dict(x=[], y=[], err=[], hover=[]))
            bucket["x"].append(epoch)
            bucket["y"].append(actor["mean"])
            bucket["err"].append(actor["stdev"])
            bucket["hover"].append(_multipass_hover(actor) + "<br>" + tag)

    ours_x, ours_y, ours_err, ours_hover = _gap_at_method_change(
        ours_x, ours_y, methods, ours_err, ours_hover)
    for bucket in rival_buckets.values():
        bucket["x"], bucket["y"], bucket["err"], bucket["hover"] = _gap_at_method_change(
            bucket["x"], bucket["y"], methods, bucket["err"], bucket["hover"])

    figure.add_trace(go.Scatter(
        x=ours_x, y=ours_y, name="US", line=dict(color=OURS, width=3),
        error_y=dict(type="data", array=ours_err, color=OURS, thickness=1.4,
                     visible=any(v is not None for v in ours_err)),
        customdata=ours_hover,
        hovertemplate=("US · epoch %{x}<br>mean rank-sum %{y:,.0f}"
                       "<br>%{customdata}<extra></extra>")))
    for shown, coldkey in enumerate(rival_buckets):
        bucket = rival_buckets[coldkey]
        colour = RIVALS[shown % len(RIVALS)]
        figure.add_trace(go.Scatter(
            x=bucket["x"], y=bucket["y"], name=coldkey[:8] + "...",
            line=dict(color=colour, width=1.6),
            error_y=dict(type="data", array=bucket["err"], color=colour,
                         thickness=1.4,
                         visible=any(v is not None for v in bucket["err"])),
            customdata=bucket["hover"],
            hovertemplate=(coldkey[:8] + "... · epoch %{x}"
                           "<br>mean rank-sum %{y:,.0f}<br>%{customdata}"
                           "<extra></extra>")))
    figure.update_layout(**LAYOUT)
    _mark_method_boundaries(figure, _method_segments(
        sorted(methods), [methods[e] for e in sorted(methods)]))
    figure.update_yaxes(title="multi-pass mean rank-sum (lower = better)",
                        gridcolor="#242a36")
    figure.update_xaxes(title="epoch", gridcolor="#242a36")
    return figure


def fig_multipass_strongest(path: Path | None = None, top: int = 4) -> go.Figure:
    """US vs the strongest rivals' multi-pass mean rank-sum per epoch.

    Each line is a trailing 10-epoch mean of the per-epoch rank-sum (the raw
    point is one ~10-20 sequence sample and is noisy).  The ``top`` rivals are
    chosen by lowest all-time mean (lower is better) and plotted over the full
    replay-scored history.
    """
    path = path or MULTIPASS_COMPARISON
    figure = go.Figure()
    if not path.is_file():
        return _empty_multipass(figure, "No multi-pass comparison data yet",
                                "multi-pass mean rank-sum (lower = better)")

    payload = json.loads(path.read_text())
    epochs = sorted(int(key) for key in payload)
    if not epochs:
        return _empty_multipass(figure, "No multi-pass comparison data yet",
                                "multi-pass mean rank-sum (lower = better)")

    ours_x, ours_y, ours_hover = [], [], []
    rival_buckets: dict[str, dict] = {}
    methods: dict[int, str] = {}
    for epoch in epochs:
        row = payload.get(str(epoch))
        if not isinstance(row, dict):
            continue
        methods[epoch] = _multipass_method(epoch, row)
        ours = _normalize_multipass_actor(row.get("ours"))
        ours_x.append(epoch)
        if ours is None:
            ours_y.append(None); ours_hover.append("")
        else:
            ours_y.append(ours["mean"]); ours_hover.append(_multipass_hover(ours))
        for coldkey, actor in _multipass_all_rivals(row):
            bucket = rival_buckets.setdefault(
                coldkey, dict(x=[], y=[], hover=[]))
            bucket["x"].append(epoch)
            bucket["y"].append(actor["mean"])
            bucket["hover"].append(_multipass_hover(actor))

    def _mean(bucket: dict) -> float:
        return sum(bucket["y"]) / len(bucket["y"])

    strongest = sorted(rival_buckets, key=lambda ck: _mean(rival_buckets[ck]))[:top]

    def _smooth(xs, ys, hover):
        """Trailing mean that restarts at every method boundary, with a gap
        there: a window never averages two measurement regimes."""
        out_x, out_y, out_h = [], [], []
        start = 0
        for i in range(1, len(xs) + 1):
            if i == len(xs) or methods.get(xs[i]) != methods.get(xs[start]):
                sx, sy = _trailing_mean(xs[start:i], ys[start:i])
                if out_x:
                    out_x.append(xs[start] - 0.5); out_y.append(None); out_h.append(None)
                out_x += sx; out_y += sy; out_h += list(hover[start:i])
                start = i
        return out_x, out_y, out_h

    ours_x, ours_y, ours_hover = _smooth(ours_x, ours_y, ours_hover)
    figure.add_trace(go.Scatter(
        x=ours_x, y=ours_y, name="US", line=dict(color=OURS, width=3),
        customdata=ours_hover,
        hovertemplate=("US · epoch %{x}<br>10-epoch mean rank-sum %{y:,.0f}"
                       "<br>%{customdata}<extra></extra>")))
    for shown, coldkey in enumerate(strongest):
        bucket = rival_buckets[coldkey]
        x, y, hover = _smooth(bucket["x"], bucket["y"], bucket["hover"])
        colour = RIVALS[shown % len(RIVALS)]
        figure.add_trace(go.Scatter(
            x=x, y=y, name=coldkey[:8] + "...",
            line=dict(color=colour, width=1.6),
            customdata=hover,
            hovertemplate=(coldkey[:8] + "... · epoch %{x}"
                           "<br>10-epoch mean rank-sum %{y:,.0f}<br>%{customdata}"
                           "<extra></extra>")))
    figure.update_layout(**LAYOUT)
    _mark_method_boundaries(figure, _method_segments(
        sorted(methods), [methods[e] for e in sorted(methods)]))
    figure.update_yaxes(title="10-epoch mean rank-sum (lower = better)",
                        gridcolor="#242a36")
    figure.update_xaxes(title="epoch", gridcolor="#242a36")
    return figure


def fig_multipass_distribution(path: Path | None = None,
                               rival: str = "5GhREcsK") -> go.Figure:
    """Per-sequence multi-pass rank-sum distribution for us and every rival.

    Only actors (and epochs) that carry a non-empty ``scores`` list appear,
    as a grouped box plot -- one box per actor per epoch.  Lower is better.
    """
    path = path or MULTIPASS_COMPARISON
    figure = go.Figure()
    if not path.is_file():
        return _empty_multipass(figure, "No multi-pass distribution data yet",
                                "multi-pass rank-sum (lower = better)")

    payload = json.loads(path.read_text())
    ours_epochs: dict[str, list] = {}
    rival_epochs: dict[str, dict] = {}
    methods: dict[str, str] = {}
    for key, row in payload.items():
        if not isinstance(row, dict):
            continue
        epoch = str(key)
        methods[epoch] = _multipass_method(int(epoch), row)
        ours = _normalize_multipass_actor(row.get("ours"))
        if ours and isinstance(ours.get("scores"), list) and ours["scores"]:
            ours_epochs[epoch] = ours["scores"]
        for coldkey, actor in _multipass_rivals(row, rival):
            if isinstance(actor.get("scores"), list) and actor["scores"]:
                rival_epochs.setdefault(coldkey, {})[epoch] = actor["scores"]

    epoch_order = sorted(
        set(ours_epochs) | {e for m in rival_epochs.values() for e in m},
        key=int)
    if not epoch_order:
        return _empty_multipass(figure, "No per-sequence multi-pass scores yet",
                                "multi-pass rank-sum (lower = better)")

    def _add_boxes(label, per_epoch, colour, offset):
        xs, ys = [], []
        for epoch in epoch_order:
            scores = per_epoch.get(epoch)
            if not scores:
                continue
            xs.extend([epoch] * len(scores))
            ys.extend(float(s) for s in scores)
        if not ys:
            return
        figure.add_trace(go.Box(
            x=xs, y=ys, name=label, offsetgroup=offset, legendgroup=label,
            marker_color=colour, line=dict(color=colour),
            boxpoints="outliers", hoveron="boxes+points"))

    _add_boxes("US", ours_epochs, OURS, 0)
    for shown, coldkey in enumerate(rival_epochs, start=1):
        _add_boxes(coldkey[:8] + "...", rival_epochs[coldkey],
                   RIVALS[(shown - 1) % len(RIVALS)], shown)

    layout = dict(LAYOUT)
    layout["boxmode"] = "group"
    layout["boxgap"] = 0.25
    layout["boxgroupgap"] = 0.05
    figure.update_layout(**layout)
    figure.update_yaxes(title="multi-pass rank-sum (lower = better)",
                        gridcolor="#242a36")
    figure.update_xaxes(title="epoch", gridcolor="#242a36", type="category",
                        categoryorder="array", categoryarray=epoch_order)
    _mark_method_boundaries(
        figure, _method_segments([int(e) for e in epoch_order],
                                 [methods[e] for e in epoch_order]),
        categorical_index={e: i for i, e in enumerate(epoch_order)})
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
