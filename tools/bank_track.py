#!/usr/bin/env python3
"""Record the bank's best-20 against the scaling law, milestone by milestone.

The law was fit by subsampling 7,827 exactly-scored molecules:
    best20 = 0.0818*ln(size) + 0.8344      (+0.057 per doubling)
Its first out-of-sample test was the merge of 3,456 molecules from unrelated
processes (a uniform probe, a similarity experiment, two failed trials' round
one): predicted 1.5977, measured 1.5970, error 0.0006.

This keeps testing it as the bank grows, because the whole $183 premise rests
on the curve continuing to hold. Two outcomes are informative: the realised
curve running ABOVE the fit would mean proxy-ranked analogues beat the random
subsampling the fit was built from, and running BELOW would mean the bank is
saturating and the spend should stop early.

Append-only, on the box, so it survives any client session.
"""
import json
import math
import sqlite3
import time
from pathlib import Path

BASE = Path('/root/bank')
SERIES = BASE / 'scaling-series.json'
SLOPE, INTERCEPT = 0.0818, 0.8344
MILESTONES = [12500, 15000, 17500, 20000, 25000, 30000, 35000, 40000,
              50000, 60000, 70000, 82000, 95000, 110000, 125000, 150000,
              175000, 200000, 215000]


def predicted(size):
    return SLOPE * math.log(size) + INTERCEPT


def snapshot():
    path = BASE / 'bank.sqlite'
    if not path.is_file():
        return None
    with sqlite3.connect('file:%s?mode=ro' % path, uri=True) as connection:
        connection.execute('PRAGMA busy_timeout=60000')
        total = connection.execute(
            'SELECT COUNT(*) FROM candidates WHERE score IS NOT NULL').fetchone()[0]
        best = [r[0] for r in connection.execute(
            'SELECT score FROM candidates WHERE score IS NOT NULL '
            'ORDER BY score DESC LIMIT 20')]
        by_scorer = dict(connection.execute(
            'SELECT scorer,COUNT(*) FROM candidates WHERE score IS NOT NULL GROUP BY scorer'))
    if not best:
        return None
    return dict(total=total, best20=round(sum(best), 6), by_scorer=by_scorer)


def main():
    series = []
    if SERIES.is_file():
        try:
            series = json.loads(SERIES.read_text())['series']
        except Exception:
            series = []
    reached = {row['milestone'] for row in series if row.get('milestone')}
    while True:
        state = snapshot()
        if state:
            due = [m for m in MILESTONES if m not in reached and state['total'] >= m]
            for milestone in due:
                p = predicted(state['total'])
                series.append(dict(milestone=milestone, total=state['total'],
                                   best20=state['best20'], predicted=round(p, 6),
                                   error=round(state['best20'] - p, 6),
                                   by_scorer=state['by_scorer'],
                                   recorded_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
                reached.add(milestone)
            latest = dict(total=state['total'], best20=state['best20'],
                          predicted=round(predicted(state['total']), 6),
                          error=round(state['best20'] - predicted(state['total']), 6),
                          recorded_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
            temporary = SERIES.with_suffix('.partial')
            temporary.write_text(json.dumps(dict(series=series, latest=latest,
                                                 law='best20 = %.4f*ln(size) + %.4f' % (SLOPE, INTERCEPT))))
            temporary.replace(SERIES)
            if state['total'] >= MILESTONES[-1]:
                return 0
        time.sleep(600)


if __name__ == '__main__':
    raise SystemExit(main())
