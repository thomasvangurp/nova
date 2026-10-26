#!/usr/bin/env python3
"""Stage the PR100 controller unit and config into a directory, and print the steps.

This script WRITES FILES ONLY, into the staging directory given as an argument. It never
calls systemctl, never installs or enables anything, and never touches /etc, /opt or any
other live path. A human copies the staged files and runs the printed commands after
reviewing them.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
UNIT_SOURCE = HERE / 'nova-pr100-controller.service'
UNIT_NAME = 'nova-pr100-controller.service'
SUBSTITUTIONS = ('<RELEASE_DIR>', '<CONFIG_PATH>', '<CONTROLLER_PYTHON>')


def render(unit_text, *, release_dir, config_path, controller_python):
    values = {'<RELEASE_DIR>': release_dir, '<CONFIG_PATH>': config_path,
              '<CONTROLLER_PYTHON>': controller_python}
    for marker, value in values.items():
        if not str(value).startswith('/'):
            raise ValueError(f'{marker} must be an absolute path')
        unit_text = unit_text.replace(marker, str(value))
    left = [m for m in SUBSTITUTIONS if m in unit_text]
    if left:
        raise ValueError('unit still contains placeholders: ' + ', '.join(left))
    return unit_text


def stage(staging, *, release_dir, config_path, controller_python, config_source=None):
    staging = Path(staging)
    staging.mkdir(parents=True, exist_ok=True)
    unit = render(UNIT_SOURCE.read_text(), release_dir=release_dir, config_path=config_path,
                  controller_python=controller_python)
    written = {}
    unit_path = staging / UNIT_NAME
    unit_path.write_text(unit)
    written['unit'] = str(unit_path)
    if config_source:
        target = staging / Path(config_path).name
        target.write_text(Path(config_source).read_text())
        os.chmod(target, 0o600)
        written['config'] = str(target)
    steps = [
        f'review {written["unit"]} (and the staged config, if any) before copying anything',
        f'install -o root -g root -m 0644 {written["unit"]} /etc/systemd/system/{UNIT_NAME}',
        f'install the controller config at {config_path} owned by the service user, mode 0600',
        f'verify the release tree: {release_dir}/build_stamp.json git_sha equals the config source_sha',
        f'{controller_python} -B -m nanobody_opt.pr100_live_controller '
        f'--config {config_path} --once --dry-run   # from {release_dir}, publishes nothing',
        'systemctl daemon-reload',
        f'systemctl enable --now {UNIT_NAME}   # only after the dry run looks right',
        f'systemctl status {UNIT_NAME} and tail {release_dir}/pr100-controller.log',
    ]
    plan = {'staged_files': written, 'installed_anything': False, 'steps': steps,
            'note': 'This installer only wrote files into the staging directory.'}
    (staging / 'install-plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staging', required=True, type=Path,
                        help='directory to write the rendered unit into (nothing else is touched)')
    parser.add_argument('--release-dir', required=True, help='absolute stamped release directory')
    parser.add_argument('--config-path', required=True, help='absolute path the config will have')
    parser.add_argument('--controller-python', required=True, help='absolute interpreter path')
    parser.add_argument('--config-source', type=Path, help='optional config file to stage as well')
    args = parser.parse_args(argv)
    plan = stage(args.staging, release_dir=args.release_dir, config_path=args.config_path,
                 controller_python=args.controller_python, config_source=args.config_source)
    print(json.dumps(plan, indent=2))
    print('\nNothing was installed. Run the steps above by hand after review.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
