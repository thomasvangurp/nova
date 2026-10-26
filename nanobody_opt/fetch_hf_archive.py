"""Fetch the public target archive; network errors never mean zero history."""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

BASE = 'https://huggingface.co/datasets/Metanova/Submission-Archive/resolve/main/'
INVENTORY = 'https://huggingface.co/api/datasets/Metanova/Submission-Archive/tree/main'
EMPTY = b'sequence,sequence_hash,epoch\n'


def fetch(path: Path, target: str, *, allow_cached=False) -> dict:
    from nanobody_opt import rules
    if not re.fullmatch(r'[A-Z0-9]{6,10}', target):
        raise ValueError('invalid archive target')
    filename = target + '_nanobodies.csv'
    payload = None
    absent = False
    cached = False
    for _ in range(3):
        try:
            with urllib.request.urlopen(BASE + filename, timeout=90) as response:
                payload = response.read()
            break
        except urllib.error.HTTPError as error:
            if error.code == 404 and target == rules.TARGET:
                with urllib.request.urlopen(INVENTORY, timeout=30) as response:
                    inventory = json.load(response)
                    paginated = bool(response.headers.get('Link'))
                if (not paginated and isinstance(inventory, list) and inventory
                        and all(isinstance(row, dict) and isinstance(row.get('path'), str)
                                for row in inventory)
                        and filename not in {row['path'] for row in inventory}):
                    payload, absent = EMPTY, True
                    break
                raise ValueError('archive absence could not be verified') from None
        except (OSError, TimeoutError):
            pass
    if payload is None:
        receipt_path = Path(str(path) + '.target.json')
        if (allow_cached and target != rules.TARGET and path.is_file()
                and receipt_path.is_file()
                and json.loads(receipt_path.read_text()).get('target') == target):
            payload, cached = path.read_bytes(), True
        else:
            raise ValueError('current target archive unavailable')
    reader = csv.DictReader(io.StringIO(payload.decode('utf-8')))
    if not reader.fieldnames or 'sequence' not in reader.fieldnames:
        raise ValueError('invalid target archive schema')
    count = sum(1 for _ in reader)
    if not count and target != rules.TARGET:
        raise ValueError('unexpected empty legacy archive')
    from nanobody_opt.pr100_filter import atomic
    path.parent.mkdir(parents=True, exist_ok=True)
    import os
    import tempfile
    fd, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    receipt = {'target': target, 'rows': count, 'confirmed_absent': absent, 'cached': cached}
    atomic(Path(str(path) + '.target.json'), receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--target', default='P20809')
    args = parser.parse_args()
    try:
        print(json.dumps({'status': 'archive_ready', **fetch(args.output, args.target, allow_cached=True)}))
        return 0
    except (OSError, ValueError):
        print(json.dumps({'status': 'archive_unavailable', 'target': args.target}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
