"""Separate Pi commit receipts from later public validator results."""
from __future__ import annotations

import json
from pathlib import Path


def latest(log: Path, archives: Path) -> tuple[dict, dict]:
    from nanobody_opt.rules import target_for_epoch

    with log.open('rb') as handle:
        handle.seek(max(0, log.stat().st_size - 2_000_000))
        lines = handle.read().decode(errors='replace').splitlines()
    plan, successful, newest = None, set(), None
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict):
            continue
        if row.get('event') == 'plan':
            plan, successful = row, set()
        elif (row.get('event') == 'commit' and row.get('rc') == 0
              and row.get('sequence') is True and type(row.get('uid')) is int):
            successful.add(row['uid'])
        elif row.get('event') == 'done' and plan:
            values = [plan.get('epoch'), plan.get('committing'), plan.get('with_sequence'),
                      row.get('committed'), row.get('failed')]
            if (all(type(v) is int and v >= 0 for v in values)
                    and plan['committing'] == plan['with_sequence']
                    and row['committed'] == len(successful)
                    and row['committed'] + row['failed'] <= plan['committing']):
                if newest is None or plan['epoch'] >= newest[0]['epoch']:
                    newest = ({'epoch': plan['epoch'], 'committed': row['committed'],
                               'planned': plan['committing'], 'failed': row['failed']},
                              set(successful))
                    if type(plan.get('crew')) is int and plan['crew'] >= plan['committing']:
                        newest[0]['registered'] = plan['crew']
            plan, successful = None, set()
    if newest is None:
        return {}, {}
    status, uids = newest
    status = {**status, 'source': 'pi_commit_receipts'}
    # Successful transaction receipts are valid independently of replay input
    # reconstruction. PR100 archives contain a UID-independent candidate pool.
    try:
        artifact = json.loads((archives/f"epoch{status['epoch']}"/'nanobody_result.json').read_text())
    except (OSError, ValueError):
        return status, {}
    target = target_for_epoch(status['epoch'])
    if artifact.get('epoch') != status['epoch'] or artifact.get('target') != target:
        return status, {}
    submissions = [row for row in artifact.get('submissions', [])
                   if isinstance(row, dict) and row.get('uid') in uids
                   and isinstance(row.get('sequence'), str) and row['sequence']]
    if len({row['uid'] for row in submissions}) != status['committed']:
        return status, {}
    return (status,
            {'epoch': status['epoch'], 'target': target,
             'submissions': [{'uid': row['uid'], 'sequence': row['sequence']} for row in submissions]})
