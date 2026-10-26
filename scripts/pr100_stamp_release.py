#!/usr/bin/env python3
"""Stamp a release directory so the controller and the remote transport will accept it.

Writes ``<release>/build_stamp.json`` with ``git_sha`` and the sha256 of every
controller-relevant file. It only reads the git revision and hashes files; it installs
nothing and contacts no host.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

# Files the live controller path needs on a host release tree.
CONTROLLER_FILES = (
    'nanobody_opt/pr100_live_controller.py',
    'nanobody_opt/pr100_live_worker.py',
    'nanobody_opt/pr100_live_gpu.py',
    'nanobody_opt/pr100_live_selection.py',
    'nanobody_opt/pr100_live_trial_export.py',
    'nanobody_opt/pr100_filter.py',
    'nanobody_opt/rules.py',
    'scripts/pr100_live_supervisor.py',
    'scripts/pr100_gpu_probe.py',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def revision(root):
    result = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    return result.stdout.decode().strip() if result.returncode == 0 else ''


def stamp(release, *, git_sha=None, names=CONTROLLER_FILES, extra=()):
    release = Path(release)
    git_sha = git_sha or revision(release)
    if not isinstance(git_sha, str) or len(git_sha) != 40:
        raise ValueError('a 40-character git revision is required')
    target = release / 'build_stamp.json'
    if target.exists():
        # A tested release already carries the full release manifest, which
        # pr100_gpu_probe verifies (pixi.lock, submodules, runtime inventory). Never
        # replace it with this minimal stamp: accept it only if it already covers us.
        existing = json.loads(target.read_text())
        if existing.get('git_sha') != git_sha:
            raise ValueError('existing release manifest is for a different revision')
        recorded = existing.get('files', {})
        for name in sorted(set(names) | set(extra)):
            if recorded.get(name) != digest(release / name):
                raise ValueError(f'existing release manifest does not cover {name}')
        return existing
    files = {}
    for name in sorted(set(names) | set(extra)):
        source = release / name
        if not source.is_file():
            raise ValueError(f'release is missing {name}')
        files[name] = digest(source)
    manifest = {'schema': 1, 'git_sha': git_sha, 'files': files,
                'runtime_files': sorted(files)}
    target = release / 'build_stamp.json'
    temporary = target.with_name('.build_stamp.json.tmp')
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, target)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', required=True, type=Path,
                        help='release directory to stamp (it is read and stamped, never installed)')
    parser.add_argument('--git-sha', help='revision to record; defaults to git HEAD of the release')
    parser.add_argument('--extra', action='append', default=[],
                        help='additional relative file to include in the stamp')
    args = parser.parse_args(argv)
    manifest = stamp(args.release, git_sha=args.git_sha, extra=tuple(args.extra))
    print(json.dumps({'git_sha': manifest['git_sha'], 'file_count': len(manifest['files']),
                      'stamp': str(args.release / 'build_stamp.json')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
