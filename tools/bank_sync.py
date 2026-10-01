#!/usr/bin/env python3
"""Keep the parallel bank boxes from re-deriving each other's molecules.

Two boxes build the same bank with different RNG seeds. Without syncing, each
one's `seen` set only covers its own molecules, so both pay to score overlapping
chemistry and both anchor on a poorer pool than the union. This pulls each box's
bank, merges locally, and pushes the union back as a supplement that each box
merges under a busy timeout -- never overwriting a live database.

Safe to run repeatedly while both builders are working.
"""
import argparse
import json
import shlex
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
LOCAL = ROOT / 'data/bank_master.sqlite'

MERGE = r'''
import json,sqlite3,time
from pathlib import Path
B=Path('/root/bank')
src=B/'sync-supplement.sqlite'; dst=B/'bank.sqlite'
s={}
with sqlite3.connect('file:%s?mode=ro'%src,uri=True) as a:
    rows=a.execute('SELECT name,smiles,reaction,components,inchikey,heavy_atoms,'
                   'rotatable_bonds,affinity_probability_binary,affinity_pred_value,'
                   'score,scorer FROM candidates').fetchall()
c=sqlite3.connect(dst,timeout=180); c.execute('PRAGMA busy_timeout=180000')
before=c.execute('SELECT COUNT(*) FROM candidates WHERE score IS NOT NULL').fetchone()[0]
for attempt in range(6):
    try:
        c.executemany('INSERT OR IGNORE INTO candidates VALUES (?,?,?,?,?,?,?,?,?,?,?)',rows)
        c.commit(); break
    except sqlite3.OperationalError:
        time.sleep(20)
after=c.execute('SELECT COUNT(*) FROM candidates WHERE score IS NOT NULL').fetchone()[0]
best=[x[0] for x in c.execute('SELECT score FROM candidates WHERE score IS NOT NULL ORDER BY score DESC LIMIT 20')]
c.close()
print(json.dumps({'event':'source','summary':dict(before=before,after=after,
    added=after-before,best20=round(sum(best),6))}))
'''


#: Which control module drives which box. A ternary on the name silently sent
#: every box that was not box1 to box2's manifest -- adding a third box would
#: have merged it into box2's bank and reported success, which is the same
#: shape as the ranker that returned [] for four of five reactions. A name that
#: is not here is an error, not a default.
BOXES = {
    'box1': 'bank_box_control',
    'box2': 'bank_box2_control',
    'box3': 'submit_box_control',
}


def box(name):
    try:
        module = __import__(BOXES[name])
    except KeyError:
        raise SystemExit(f'unknown box {name!r}; known: {sorted(BOXES)}') from None
    host = module.instance()
    return module, host, 'root@' + host['public_ipaddr']


def pull(name, destination):
    module, host, target = box(name)
    common = ['rsync', '-t', '--no-owner', '--no-group', '--no-perms',
              '-e', shlex.join(module.ssh(host))]
    result = subprocess.run([*common, target + ':/root/bank/bank.sqlite', str(destination)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1800)
    return result.returncode == 0


def push_and_merge(name, supplement):
    module, host, target = box(name)
    common = ['rsync', '-t', '--no-owner', '--no-group', '--no-perms',
              '-e', shlex.join(module.ssh(host))]
    result = subprocess.run([*common, str(supplement), target + ':/root/bank/sync-supplement.sqlite'],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1800)
    if result.returncode:
        return dict(uploaded=False)
    result = subprocess.run([*module.ssh(host), target, 'python3 -c ' + shlex.quote(MERGE)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=900)
    for line in result.stdout.splitlines():
        try:
            event = json.loads(line)
        except Exception:
            continue
        if isinstance(event, dict) and event.get('event') == 'source':
            return event['summary']
    return dict(merge_failed=True, exit_code=result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--boxes', default='box1,box2')
    parser.add_argument('--work', default='/tmp')
    arguments = parser.parse_args()
    names = [n.strip() for n in arguments.boxes.split(',') if n.strip()]
    work = Path(arguments.work)
    merged = {}
    for name in names:
        local = work / ('pull-%s.sqlite' % name)
        if not pull(name, local):
            print(json.dumps(dict(step='pull', box=name, ok=False)))
            continue
        with sqlite3.connect('file:%s?mode=ro' % local, uri=True) as connection:
            rows = connection.execute(
                'SELECT name,smiles,reaction,components,inchikey,heavy_atoms,rotatable_bonds,'
                'affinity_probability_binary,affinity_pred_value,score,scorer '
                'FROM candidates WHERE score IS NOT NULL').fetchall()
        for row in rows:
            merged.setdefault(row[0], row)
        print(json.dumps(dict(step='pull', box=name, molecules=len(rows), union=len(merged))))
    if not merged:
        return 1
    LOCAL.parent.mkdir(parents=True, exist_ok=True)
    if LOCAL.exists():
        LOCAL.unlink()
    with sqlite3.connect(LOCAL) as connection:
        connection.execute(
            'CREATE TABLE candidates(name TEXT PRIMARY KEY,smiles TEXT,reaction INT,'
            'components TEXT,inchikey TEXT,heavy_atoms INT,rotatable_bonds INT,'
            'affinity_probability_binary REAL,affinity_pred_value REAL,score REAL,scorer TEXT)')
        connection.executemany(
            'INSERT OR IGNORE INTO candidates VALUES (?,?,?,?,?,?,?,?,?,?,?)', merged.values())
        best = [r[0] for r in connection.execute(
            'SELECT score FROM candidates ORDER BY score DESC LIMIT 20')]
    import math
    law = 0.0818 * math.log(len(merged)) + 0.8344
    print(json.dumps(dict(step='union', molecules=len(merged),
                          best20=round(sum(best), 6), predicted=round(law, 6),
                          margin=round(sum(best) - law, 6), master=str(LOCAL))))
    for name in names:
        print(json.dumps(dict(step='push', box=name, result=push_and_merge(name, LOCAL))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
