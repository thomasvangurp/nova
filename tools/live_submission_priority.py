"""Reserve slave capacity for current submissions before background replay."""
import json
from pathlib import Path
import time


def critical(root=Path('/root'), now=None):
    now = time.time() if now is None else now
    try:
        path = root/'artifacts/chain_state.json'
        state = json.loads(path.read_text())
        stamp = state.get('updated_at_unix', path.stat().st_mtime)
        if not 0 <= now - stamp <= 180:
            return True
        epoch, length, block = int(state['epoch']), int(state['epoch_length']), int(state['block'])
        if length <= 0 or block // length != epoch:
            return True
        offset = block % length
        if not 150 <= offset < length - 36:
            return False
        confirmation = root/f'live/epoch{epoch}/producer/full_scan_confirmation.json'
        if not confirmation.is_file():
            return True
        artifact = json.loads((root/'artifacts/nanobody_result.json').read_text())
        return artifact.get('epoch') != epoch or not artifact.get('submissions')
    except (OSError, ValueError, TypeError, KeyError):
        return True


if __name__ == '__main__':
    raise SystemExit(4 if critical() else 0)
