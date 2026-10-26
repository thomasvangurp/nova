#!/usr/bin/env python3
"""One blocking regression command for CI and release packaging."""
from __future__ import annotations

import argparse
import json
import os
import py_compile
import subprocess
import sys
import tarfile
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
TEST_FILES = (
    'tests/test_pr100.py', 'tests/test_pr100_proposal_pilot.py', 'tests/test_pr100_publication_hold.py',
    'tests/test_pr100_replay.py',
    'tests/test_pr100_live_consumer.py',
    'tests/test_pr100_validator_oracle.py',
    'tests/test_pr100_probe.py',
    'tests/test_pr100_native_benchmark.py',
    'tests/test_pr100_resources.py',
    'tests/test_pr100_live_trial_export.py',
    # The whole Pi-owned live path: controller, supervisor, worker, GPU lease, selection
    # and the handoff diagnostics the publisher depends on.
    'tests/test_pr100_live_controller.py',
    'tests/test_pr100_live_supervisor.py',
    'tests/test_pr100_live_worker.py',
    'tests/test_pr100_live_gpu.py',
    'tests/test_pr100_live_selection.py',
    'tests/test_pr100_handoff_diagnostics.py',
    'tests/test_pr100_live_handoff_contract.py',
    'tests/test_molecule_conformer_gate.py',
    'tests/test_pi_preparation_deadline.py',
    'tests/test_parent_merit.py',
    'tests/test_epoch_screen_budget.py',
    'tests/test_cached_confirmation.py',
    'tests/test_confirmation_policy.py',
    'tests/test_confirmation_eligibility.py',
    'tests/test_adaptive_confirmation_depth.py',
    'tests/test_publication_ledger.py',
    'tests/test_nanobody_filter_layer2.py',
    'tests/test_nanobody_check_validator_novelty.py',
    'tests/test_tnp_slots.py',
    'tests/test_complete_screen_pool.py',
    'tests/test_deferred_tnp_pipeline.py',
    'tests/test_nanobody_waves.py',
    'tests/test_score_boltzgen_waves.py',
    'tests/test_score_boltzgen_fast_adapter.py',
    'tests/test_scorer_daemon.py',
    'tests/test_fast_scorer_vendoring.py',
    'tests/test_developability_hoist_regressions.py',
    'tests/test_competitor_position_lane.py',
    'tests/test_slot_value_pricing.py',
    'tests/test_slot_manager.py',
    'tests/test_commit_side_effects.py',
    'tests/test_round_trip.py',
    'tests/validate_against_validator.py',
    'tests/test_ci_guardrails.py',
    'tests/test_shipping_producer.py',
    'tests/test_deploy_stamp.py',
    'tests/test_release_pipeline.py',
    'tests/test_release_promotion.py',
    'tests/test_control_release.py',
    'tests/test_epoch_deadline.py',
    'tests/test_submission_handoff.py',
    'tests/test_dashboard_cohort.py',
    'tests/test_pi_dashboard.py',
    'tests/test_nanobody_dash_operational.py',
    'tests/test_compound_dashboard.py',
    'tests/test_nanobody_rank_exact.py',
    'tests/test_topup_slave.py',
    'nanobody_opt_v2/tests/test_flags.py',
    'nanobody_opt_v2/tests/test_replication.py',
    'nanobody_opt_v2/tests/test_compute_budget.py',
)


def run(root: Path, output: Path, *, profile: str = 'application') -> int:
    sys.path.insert(0, str(root))
    from scripts.release_manifest import stage, verify
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / 'staged'
    manifest = stage(root, artifact, profile=profile)
    for name in TEST_FILES:
        if not (artifact / name).is_file():
            raise ValueError(f'missing required regression suite: {name}')
    # Compile the exact staged Python files that the release will install.
    for name in manifest['files']:
        if name.endswith('.py'):
            py_compile.compile(str(artifact / name), doraise=True)
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=str(artifact), NOVA_LOCAL_SQLITE='1',
               SN68_DEPLOY_ARTIFACT_ROOT=str(artifact), SN68_TESTED_SHA=manifest['git_sha'])
    commands = (
        [sys.executable, 'scripts/ci_guardrails.py'],
        [sys.executable, '-m', 'pytest', '-q', '--tb=short', '-p', 'no:cacheprovider',
         '-o', 'junit_family=xunit1', '--junitxml=' + str(output / 'regressions.xml'), *TEST_FILES],
    )
    for command in commands:
        result = subprocess.run(command, cwd=artifact, env=env, check=False)
        if result.returncode:
            return result.returncode
    report = ElementTree.parse(output / 'regressions.xml').getroot()
    cases = list(report.iter('testcase'))
    validation = {'passed': True, 'tests': len(cases),
                  'failures': len(list(report.iter('failure'))),
                  'errors': len(list(report.iter('error'))),
                  'skipped': len(list(report.iter('skipped'))), 'test_files': list(TEST_FILES)}
    collected = {case.get('file') for case in cases}
    missing = set(TEST_FILES) - collected
    if missing:
        raise ValueError('required suites collected no tests: ' + ', '.join(sorted(missing)))
    manifest['validation'] = validation
    # A successful pytest invocation with skipped or zero tests is insufficient.
    verify(artifact, manifest, expected_sha=manifest['git_sha'])
    (artifact / 'build_stamp.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (output / 'build_stamp.json').write_text(json.dumps(manifest, indent=2) + '\n')
    with tarfile.open(output / 'sn68-release.tar.gz', 'w:gz') as archive:
        for name in [*manifest['files'], 'build_stamp.json']:
            archive.add(artifact / name, arcname=name, recursive=False)
    print(json.dumps({'release': manifest['git_sha'], **validation}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', choices=['application', 'release-controls'], default='application')
    args = parser.parse_args()
    try:
        return run(args.root.resolve(), args.output.resolve(), profile=args.profile)
    except (OSError, ValueError, py_compile.PyCompileError) as error:
        print(f'SHIPPING CHECKS FAILED: {error}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
