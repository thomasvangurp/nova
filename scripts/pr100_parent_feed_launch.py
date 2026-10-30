#!/usr/bin/env python3
"""Stable Pi timer launcher: resolve the currently configured certified source."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def launch(config, releases, python):
    value = json.loads(config.read_text()); source = value['source_sha']
    if not isinstance(source, str) or re.fullmatch('[0-9a-f]{40}', source) is None:
        raise ValueError('invalid_source')
    release = releases / source
    if release.is_symlink() or release.resolve().parent != releases.resolve():
        raise ValueError('invalid_release')
    stamp = json.loads((release / 'build_stamp.json').read_text())
    if stamp['git_sha'] != source or stamp['validation']['passed'] is not True:
        raise ValueError('uncertified_source')
    module = 'nanobody_opt/pr100_parent_feed.py'
    if hashlib.sha256((release / module).read_bytes()).hexdigest() != stamp['files'][module]:
        raise ValueError('parent_feed_source_changed')
    # The module validates every manifest byte before use. No environment
    # credentials are forwarded; SSH uses the user's existing key/agent setup.
    environment = {'PATH': '/usr/bin:/bin', 'HOME': str(Path.home()), 'LANG': 'C.UTF-8',
                   'PYTHONDONTWRITEBYTECODE': '1'}
    if os.environ.get('SSH_AUTH_SOCK'):
        environment['SSH_AUTH_SOCK'] = os.environ['SSH_AUTH_SOCK']
    result = subprocess.run([str(python), '-B', str(release / module), 'tick', '--config', str(config)],
                            env=environment, capture_output=True, timeout=150)
    # Child stdout is a JSON public envelope; emit only a strict tiny subset.
    try:
        public = json.loads(result.stdout)
    except (ValueError, TypeError):
        public = {}
    allowed = ('status', 'epoch', 'qualified_rows', 'epochs_fetched', 'last_success_age_seconds',
               'bank_age_epochs', 'error_type', 'owner_age_seconds', 'launch_uncertain')
    return result.returncode, {k: public[k] for k in allowed if k in public}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('/home/thomasg/pr100-controller.json'))
    parser.add_argument('--releases', type=Path, default=Path('/home/thomasg/pr100-releases'))
    parser.add_argument('--python', type=Path, default=Path('/opt/nova-submit/.pixi/envs/default/bin/python'))
    args = parser.parse_args()
    try:
        code, value = launch(args.config, args.releases, args.python)
    except Exception as error:
        code, value = 1, {'status': 'launch_failed', 'error_type': type(error).__name__}
    print(json.dumps(value)); return code


if __name__ == '__main__':
    raise SystemExit(main())
