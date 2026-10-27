"""Run deterministic inert timing scenarios through both actual worker policies.

No native subprocess, model, accelerator or network call is executed. The
original batch path and proposed refill path receive identical synthetic call
durations and numerical observations. This is not a production speed benchmark.
"""
from pathlib import Path
import argparse
import json
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pytest import MonkeyPatch
from tests.test_pr100_native_refill import simulated_phase, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    cases = [
        ('uniform_sparse', 96, 600, 1000, lambda s, n: 80., lambda s, n: 'pass' if int(s) % 37 == 0 else 'reject'),
        ('stragglers_master', 96, 600, 1000, lambda s, n: 250. if int(s) % 96 == 0 else 60., lambda s, n: 'pass' if int(s) % 37 == 0 else 'reject'),
        ('stragglers_slave', 32, 600, 1000, lambda s, n: 250. if int(s) % 32 == 0 else 60., lambda s, n: 'pass' if int(s) % 37 == 0 else 'reject'),
        ('heterogeneous_sparse', 96, 600, 1000, lambda s, n: 30. + (int(s) * 37 + n * 17) % 220, lambda s, n: 'pass' if int(s) % 37 == 0 else 'reject'),
        ('partial_failure', 32, 150, 1000, lambda s, n: 400. if int(s) % 3 == 0 and n == 2 else 40., lambda s, n: 'pass' if n == 1 else 'reject'),
        ('late_fast_confirmation', 32, 180, 430, lambda s, n: 20. if int(s) % 37 == 0 else 100., lambda s, n: 'pass' if int(s) % 37 == 0 else 'reject'),
    ]
    output = {'schema': 'pr100-inert-refill-comparison-v1', 'native_calls': 0,
        'production_changes': False, 'simulation': True, 'cases': []}
    with tempfile.TemporaryDirectory(prefix='pr100-inert-refill-') as directory, MonkeyPatch.context() as monkeypatch:
        root = Path(directory)
        for name, width, count, deadline, durations, outcomes in cases:
            row = {'name': name, 'workers': width, 'inputs': count, 'deadline_seconds': deadline}
            for enabled, mode in ((False, 'current_interleaved'), (True, 'bounded_refill')):
                result, simulation = simulated_phase(root / name / mode, monkeypatch,
                    enabled=enabled, durations=durations, outcomes=outcomes,
                    width=width, count=count, deadline=deadline)
                row[mode] = summary(result, simulation, width)
            output['cases'].append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output))


if __name__ == '__main__':
    main()
