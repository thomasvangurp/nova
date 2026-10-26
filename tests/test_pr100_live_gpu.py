import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from nanobody_opt import pr100_live_gpu as gpu


def sample(*, free=12000, utilization=0, pid=10, start=100, ticks=5, idle=True):
    return {'gpu': 1, 'uuid': 'GPU-fixed', 'total_mib': 24000,
            'free_mib': free, 'utilization': utilization,
            'processes': [{'pid': pid, 'start_ticks': start,
                           'cpu_ticks': ticks, 'known_idle': idle}]}


class AdmissionTests(unittest.TestCase):
    def test_known_idle_same_identity_admitted(self):
        gpu.admit(sample(), sample(), 8192)

    def test_low_headroom_or_busy_gpu_refuses(self):
        for changed in [sample(free=8191), sample(utilization=1), sample(idle=False)]:
            with self.subTest(changed=changed), self.assertRaises(gpu.Unavailable):
                gpu.admit(sample(), changed, 8192)

    def test_cpu_activity_and_pid_reuse_refuse(self):
        for changed in [sample(ticks=6), sample(start=101), sample(pid=11)]:
            with self.subTest(changed=changed), self.assertRaises(gpu.Unavailable):
                gpu.admit(sample(), changed, 8192)

    def test_gpu_identity_drift_refuses(self):
        changed = sample(); changed['uuid'] = 'GPU-replaced'
        with self.assertRaises(gpu.Unavailable):
            gpu.admit(sample(), changed, 8192)

    def test_only_actual_owned_pids_may_join(self):
        changed = sample(); changed['processes'].append(sample(pid=70, idle=False)['processes'][0])
        self.assertTrue(gpu.uncontended(changed, sample(), {70}, 512))
        self.assertFalse(gpu.uncontended(changed, sample(), set(), 512))

    def test_gpu_listing_race_omits_process_already_exited(self):
        queries = [[['1', 'GPU-fixed', '24000', '12000', '0']], [['GPU-fixed', '99']]]
        missing = {'pid': 99, 'start_ticks': None, 'cpu_ticks': None, 'known_idle': False}
        with patch.object(gpu, '_query', side_effect=queries), \
             patch.object(gpu, '_resident', return_value=missing), \
             patch('pathlib.Path.exists', return_value=False):
            result = gpu.gpu_snapshot(1, '/pinned/scorer.py')
        self.assertEqual(result['processes'], [])

    def test_gpu_listing_live_unknown_process_still_blocks(self):
        queries = [[['1', 'GPU-fixed', '24000', '12000', '0']], [['GPU-fixed', '99']]]
        unknown = {'pid': 99, 'start_ticks': None, 'cpu_ticks': None, 'known_idle': False}
        with patch.object(gpu, '_query', side_effect=queries), \
             patch.object(gpu, '_resident', return_value=unknown), \
             patch('pathlib.Path.exists', return_value=True):
            result = gpu.gpu_snapshot(1, '/pinned/scorer.py')
        self.assertEqual(result['processes'], [unknown])
        with self.assertRaises(gpu.Unavailable):
            gpu.admit(result, result, 8192)

    def test_old_resident_becoming_active_is_contention(self):
        self.assertFalse(gpu.uncontended(sample(idle=False), sample(), {70}, 512))
        self.assertFalse(gpu.uncontended(sample(start=101), sample(), {70}, 512))
        self.assertFalse(gpu.uncontended(sample(free=511), sample(), {70}, 512))


class LockTests(unittest.TestCase):
    def test_busy_lease_refuses_and_exception_releases(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, 'test'):
                with gpu.lease_locks(1, d):
                    with self.assertRaises(gpu.Unavailable):
                        with gpu.lease_locks(1, d):
                            self.fail('overlapping lease accepted')
                    raise ValueError('test')
            with gpu.lease_locks(1, d):
                pass

    def test_existing_dispatcher_lock_is_actually_held(self):
        with tempfile.TemporaryDirectory() as d:
            other = Path(d) / 'replay.lock'; other.touch()
            with gpu.lease_locks(1, Path(d) / 'ours', [other]):
                with self.assertRaises(gpu.Unavailable):
                    with gpu.lease_locks(2, Path(d) / 'ours', [other]):
                        self.fail('existing dispatcher lock ignored')

    def test_missing_dispatcher_lock_does_not_invent_coordination(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(gpu.Unavailable):
            with gpu.lease_locks(1, d, [Path(d) / 'absent.lock']):
                pass


class RunTests(unittest.TestCase):
    def run_case(self, directory, snapshots, *, finishes=False):
        now = [0.0]
        job = {'gpu': 1, 'ttl_seconds': 10, 'lock_directory': str(Path(directory)/'locks'),
               'receipt': str(Path(directory)/'receipt.json'), 'scorer': '/pinned/scorer.py',
               'command': ['/bin/public-program'], 'cwd': directory, 'env': {'PATH': '/usr/bin'}}
        proc = Mock(pid=70)
        proc.poll.side_effect = [0, 0] if finishes else lambda: None
        control = Mock()
        control.descendants.return_value = {70}
        def cleanup(processes, timeout):
            self.assertEqual(processes, [proc])
            with self.assertRaises(gpu.Unavailable):
                with gpu.lease_locks(1, job['lock_directory']):
                    self.fail('lease released before owned-tree cleanup')
            return {'remaining_descendants': 0, 'fixed_point_reached': True}
        control.stop_descendants.side_effect = cleanup
        def launch(*args, **kwargs):
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '1')
            self.assertNotIn('AWS_SECRET_ACCESS_KEY', kwargs['env'])
            self.assertTrue(kwargs['start_new_session'])
            return proc
        snap = Mock(side_effect=snapshots) if isinstance(snapshots, list) else snapshots
        result = gpu.run_reserved(job, snapshot=snap, control=control, launch=launch,
                                  clock=lambda: now[0], sleep=lambda t: now.__setitem__(0, now[0]+t))
        self.assertEqual(json.loads(Path(job['receipt']).read_text()), result)
        with gpu.lease_locks(1, job['lock_directory']):
            pass
        return result, control

    def test_expiry_owns_only_its_child_and_releases(self):
        with tempfile.TemporaryDirectory() as d:
            result, control = self.run_case(d, lambda *args: sample())
        self.assertEqual(result['status'], 'expired')
        self.assertTrue(result['released'])
        self.assertFalse(result['exclusive_gpu'])
        control.stop_descendants.assert_called_once()

    def test_new_foreign_process_aborts_only_our_work(self):
        with tempfile.TemporaryDirectory() as d:
            foreign = sample(pid=999, idle=False)
            result, control = self.run_case(d, [sample(), sample(), foreign])
        self.assertEqual(result['status'], 'contention_abort')
        control.stop_descendants.assert_called_once()

    def test_completed_process_is_not_scientific_approval(self):
        with tempfile.TemporaryDirectory() as d:
            result, _ = self.run_case(d, [sample(), sample()], finishes=True)
        self.assertEqual(result['status'], 'process_finished')
        self.assertEqual(result['child_exit_code'], 0)
        self.assertNotIn('validator_matched', result)

    def test_busy_admission_never_launches(self):
        with tempfile.TemporaryDirectory() as d:
            result, control = self.run_case(d, [sample(), sample(utilization=40)])
        self.assertEqual(result['status'], 'unavailable')
        control.stop_descendants.assert_not_called()

    def test_invalid_limits_and_secret_environment_refuse_before_any_work(self):
        base = {'gpu': 1, 'ttl_seconds': 10, 'command': ['/bin/public-program']}
        for change in [{'ttl_seconds': float('inf')}, {'ttl_seconds': 1801},
                       {'gpu': True}, {'env': {'AWS_SECRET_ACCESS_KEY': 'not-a-real-secret'}}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                gpu.run_reserved({**base, **change})


if __name__ == '__main__':
    unittest.main()
