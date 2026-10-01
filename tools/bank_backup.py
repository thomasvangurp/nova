#!/usr/bin/env python3
"""Mirror everything the bank boxes hold to this machine.

The sync loop already copies the molecules themselves -- the expensive part --
into data/bank_master.sqlite every 40 minutes. It merges only the `candidates`
table, so several things existed solely on the rented boxes and would die with
them:

  * the `rescored` table: 200 molecules x 3 independent seeds, the measurement
    that showed selecting after averaging is worth +0.0817
  * scaling-series.json: the milestone record of actual-versus-predicted best-20
  * blacklist.json: molecules Boltz cannot score, learned the hard way after
    both builders died on record 844044928
  * exhausted-pairs.json: which (A,B) neighbourhoods are spent

Vast instances can be reclaimed by their host. Anything not on this machine is
one eviction away from being re-bought.
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
DEST = ROOT / 'data/bank_backup'
FILES = ['scaling-series.json', 'blacklist.json', 'exhausted-pairs.json',
         'builder.json', 'setup.json', 'rescore.log']


def backup(tag, module_name):
    module = __import__(module_name)
    host = module.instance()
    target = 'root@' + host['public_ipaddr']
    common = ['rsync', '-t', '--no-owner', '--no-group', '--no-perms',
              '-e', shlex.join(module.ssh(host))]
    folder = DEST / tag
    folder.mkdir(parents=True, exist_ok=True)
    result = dict(box=tag, files={})
    database = folder / 'bank.sqlite'
    code = subprocess.run([*common, target + ':/root/bank/bank.sqlite', str(database)],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3600)
    result['bank_pulled'] = code.returncode == 0
    if database.is_file():
        with sqlite3.connect('file:%s?mode=ro' % database, uri=True) as connection:
            tables = [r[0] for r in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")]
            result['tables'] = tables
            result['candidates'] = connection.execute(
                'SELECT COUNT(*) FROM candidates WHERE score IS NOT NULL').fetchone()[0]
            if 'rescored' in tables:
                result['rescored'] = connection.execute(
                    'SELECT COUNT(*) FROM rescored').fetchone()[0]
    for name in FILES:
        # macOS ships rsync 2.6.9, which has no --ignore-missing-args: passing it
        # failed the whole call with a usage error and silently copied nothing.
        # A missing file just fails its own transfer, which is what we want.
        subprocess.run([*common, target + ':/root/bank/' + name, str(folder / name)],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
        result['files'][name] = (folder / name).is_file()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--boxes', default='box1:bank_box_control,box2:bank_box2_control')
    arguments = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    summary = dict(taken_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), boxes=[])
    for entry in arguments.boxes.split(','):
        tag, module_name = entry.split(':')
        try:
            summary['boxes'].append(backup(tag, module_name))
        except Exception as error:
            summary['boxes'].append(dict(box=tag, error_class=type(error).__name__))
    # A merged copy of every rescored measurement, so it survives either box.
    merged = {}
    for tag in [e.split(':')[0] for e in arguments.boxes.split(',')]:
        path = DEST / tag / 'bank.sqlite'
        if not path.is_file():
            continue
        with sqlite3.connect('file:%s?mode=ro' % path, uri=True) as connection:
            tables = [r[0] for r in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")]
            if 'rescored' not in tables:
                continue
            for row in connection.execute('SELECT * FROM rescored'):
                merged[row[0]] = row
    if merged:
        out = DEST / 'rescored.sqlite'
        if out.exists():
            out.unlink()
        with sqlite3.connect(out) as connection:
            connection.execute('CREATE TABLE rescored(name TEXT PRIMARY KEY, passes INT, '
                               'mean_score REAL, sd_score REAL, seeds TEXT, single_pass_score REAL)')
            connection.executemany('INSERT OR REPLACE INTO rescored VALUES (?,?,?,?,?,?)',
                                   merged.values())
        summary['rescored_merged'] = len(merged)
    (DEST / 'backup.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
