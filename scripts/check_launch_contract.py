#!/usr/bin/env python3
"""Validate the same producer argv that deployment and its watchdog execute."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(root: Path, *, installed: bool = False) -> None:
    sys.path.insert(0, str(root))
    stamp = root / 'build_stamp.json'
    if installed and stamp.is_file() and json.loads(stamp.read_text()).get('deployment_profile') == 'release-controls':
        from scripts.control_release import verify_application
        verify_application(root, json.loads((root / 'artifacts/observed_application.json').read_text()))
        return
    from nanobody_opt_v2.run_live_producer import _parser
    argv = (root / 'deploy/producer/producer_argv.txt').read_text().splitlines()
    if argv[:3] != ['/root/.pixi/envs/bgen/bin/python', '-m', 'nanobody_opt_v2.run_live_producer']:
        raise ValueError('producer launcher must execute the tested v2 module')
    flags = argv[3:]
    options = [arg for arg in flags if arg.startswith('--')]
    if len(options) != len(set(options)):
        raise ValueError('producer launch configuration repeats an option')
    args = _parser().parse_args(flags)
    expected = {'epoch_cycle_start': 25872, 'epoch_proposal_budget': 8000,
                'epoch_candidate_budget': 4800, 'epoch_confirmation_finalists': 64,
                'minimum_publishable_observations': 20, 'layer2_igblast_only': 1,
                'confirmation_policy': 'broad', 'tnp_concurrency': 16}
    for name, value in expected.items():
        if getattr(args, name) != value:
            raise ValueError(f'launch setting disagrees with the release contract: {name}')
    if args.replication_rounds < 19 or args.budget_mode != 'adaptive':
        raise ValueError('launch must reserve confirmation to reach at least twenty observations')
    if args.complete_screen_pool is not None or args.complete_screen_until_epoch is not None:
        raise ValueError('the one-time epoch 71 frozen-pool exception must not survive redeployment')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        check(args.root, installed=True)
    except (OSError, ValueError):
        print('LAUNCH CONTRACT FAILED')
        return 1
    print(json.dumps({'launch_contract': 'passed', 'proposal_budget': 8000, 'screen_budget': 4800}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
