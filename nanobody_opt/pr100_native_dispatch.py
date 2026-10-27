"""Bounded native work on qualified hosts, with independent durable ownership."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import threading
from nanobody_opt import rules

COUNTERS = ('completed', 'fresh_evaluations', 'repeat_evaluations', 'native_passed',
            'total_profile_evaluations', 'carried_cpu_observations')


def host_artifacts(files, incoming, rows, terminal, owned, detail):
    """A malformed peer cannot erase another host's valid completed work."""
    assigned = {r['sequence'] for r in rows}
    hashes = {rules.sequence_hash(s) for s in assigned}
    invalid = False
    try:
        value = json.loads(files.get('result.json', b'{}'))
        if not isinstance(value, dict) or not value:
            raise ValueError('missing result')
        if any(value.get(k) != incoming.get(k) for k in ('challenge', 'epoch', 'target', 'archive_sha256')):
            raise ValueError('host result binding mismatch')
        for name in ('candidates', 'deferred_candidates', 'deferred_confirmations', 'blocked_confirmations'):
            if any(r['sequence'] not in assigned for r in value.get(name, [])):
                raise ValueError('host candidate outside assigned batch')
        if set(value.get('attempted_sequence_hashes', [])) - hashes:
            raise ValueError('host attempt outside assigned batch')
        if set(value.get('terminal_native_outcomes', {})) - hashes:
            raise ValueError('host rejection outside assigned batch')
        if any(type(value.get(k, 0)) is not int or value.get(k, 0) < 0 for k in COUNTERS):
            raise ValueError('invalid host counters')
    except (ValueError, TypeError, KeyError, AttributeError):
        invalid = True
        detail['artifact_error'] = 'unusable_native_result'
        # Missing data proves neither completed evaluations nor scientific
        # failure. Hold it unless the supervisor proves it was never launched.
        value = {'challenge': incoming['challenge'], 'status': 'failed', 'candidates': [],
                 'attempted_sequence_hashes': [],
                 'deferred_candidates': rows if terminal.get('reason') == 'not_launched' else [],
                 'blocked_confirmations': [{**r, 'continuation_blocked': 'native_result_unavailable'}
                    for r in rows] if not owned and terminal.get('reason') != 'not_launched' else []}
    try:
        progress = json.loads(files.get('native-confirmations.json', b'{}'))
        observed = progress.get('records', [])
        if observed and progress.get('challenge') != incoming['challenge']:
            raise ValueError('host checkpoint binding mismatch')
        if any((r.get('sequence') not in assigned if 'sequence' in r else
                r.get('sequence_hash') not in hashes) for r in observed):
            raise ValueError('host checkpoint outside assigned batch')
    except (ValueError, TypeError, KeyError, AttributeError):
        observed = []; detail['checkpoint_error'] = 'unusable_native_checkpoints'
    return value, [] if invalid else observed


def partition(rows, hosts, capacities):
    groups = {host: [] for host in hosts}
    blocked = []
    for row in rows:
        saved = row.get('native_resume')
        if saved:
            host = saved.get('execution_host', 'master')
            if host not in groups:
                blocked.append({**row, 'continuation_blocked': 'native_checkpoint_host_unavailable'})
                continue
        else:
            host = min(hosts, key=lambda host: len(groups[host]) / capacities[host])
        groups[host].append(row)
    return groups, blocked


class LockedChain:
    def __init__(self, chain):
        self.chain, self.lock = chain, threading.Lock()
    def get_current_block(self):
        with self.lock: return self.chain.get_current_block()
    def tempo(self, netuid):
        with self.lock: return self.chain.tempo(netuid)
    def determine_block_hash(self, block):
        with self.lock: return self.chain.determine_block_hash(block)


def run(config, challenge, payload, stop_at, chain, local, *, diagnostics=None, run_phase=None):
    from nanobody_opt import pr100_live_controller as c
    from nanobody_opt.pr100_epoch_continuation import cleanup_verified, read
    run_phase = run_phase or c.run_phase
    incoming = json.loads(payload['input.json'])
    if incoming.get('challenge') != challenge:
        raise ValueError('native dispatch challenge mismatch')
    if len({r['sequence'] for r in incoming['candidates']}) != len(incoming['candidates']):
        raise ValueError('duplicate native dispatch candidate')
    hosts = config.get('native_hosts', ['master'])
    overrides = config.get('native_worker_overrides', {})
    capacities = {host: overrides.get(host, {}).get('workers', config['worker'].get('workers', 48)) for host in hosts}
    groups, blocked = partition(incoming['candidates'], hosts, capacities)
    chain = LockedChain(chain)
    local = Path(local)
    # A retry cannot add a host or repartition an ambiguously live batch.
    plan_path = local / 'native-dispatch.json'
    plan = {'schema': 'pr100-native-dispatch-v1', 'challenge': challenge,
            'hosts': hosts, 'worker': config['worker'], 'overrides': overrides,
            'source_sha': config.get('source_sha'),
            'inputs': {host: rules.digest(rows) for host, rows in groups.items()}}
    if plan_path.exists():
        if read(plan_path, {}) != plan:
            raise ValueError('native dispatch routing or settings changed')
    else:
        c.atomic(plan_path, plan)
    records, results, pending, host_diagnostics = [], {}, [], {}

    def one(host, rows):
        child = local / 'native-hosts' / host
        child.mkdir(parents=True, exist_ok=True)
        worker = {**config['worker'], **overrides.get(host, {})}
        selected = {**config, 'worker': worker, '_native_host': host, '_bounded_phase_collection': True}
        host_payload = {**payload, 'input.json': json.dumps({**incoming, 'candidates': rows}).encode()}
        detail = {}
        try:
            files = run_phase(selected, challenge, 'native', host_payload, stop_at, chain, child, diagnostics=detail)
        except ValueError:
            # A saved-input/runtime binding refusal cannot be bypassed by
            # falling back to the cached artifact that was just rejected.
            files = {}
            detail['phase_identity_error'] = 'native_phase_binding_refused'
        except Exception:
            # Any completed child data remains durable. An unverified owner
            # prevents another native attempt, but cannot erase valid peers.
            files = {name: (child / 'native' / name).read_bytes() for name in
                     ('result.json', 'native-confirmations.json') if (child / 'native' / name).is_file()}
        try:
            terminal = read(child / 'native-supervisor.json', {})
        except (OSError, ValueError, TypeError):
            terminal = {}
        if not isinstance(terminal, dict):
            terminal = {}
        owned = not cleanup_verified(terminal)
        value, observed = host_artifacts(files, incoming, rows, terminal, owned, detail)
        return host, value, observed, owned, detail

    with ThreadPoolExecutor(max_workers=len(hosts), thread_name_prefix='pr100-native-host') as pool:
        futures = {pool.submit(one, host, rows): host for host, rows in groups.items() if rows}
        for future in as_completed(futures):
            try:
                host, value, observed, owned, detail = future.result()
            except Exception:
                host = futures[future]
                value, observed, owned = {'status': 'failed', 'candidates': []}, [], True
                detail = {'artifact_error': 'host_collection_unavailable'}
            results[host] = value; records.extend(observed); host_diagnostics[host] = detail
            if owned: pending.append(host)
    candidates, deferred, confirmations, attempted, rejected = [], [], [], set(), {}
    for host, value in results.items():
        if value.get('status') == 'complete' and value.get('novelty_checked') is True:
            candidates.extend(value.get('candidates', []))
        deferred.extend(value.get('deferred_candidates', []))
        confirmations.extend(value.get('deferred_confirmations', []))
        blocked.extend(value.get('blocked_confirmations', []))
        attempted.update(value.get('attempted_sequence_hashes', []))
        rejected.update(value.get('terminal_native_outcomes', {}))
    merged = {**incoming, 'status': 'complete', 'candidates': candidates, 'novelty_checked': True,
              'deferred_candidates': deferred, 'deferred_confirmations': confirmations,
              'blocked_confirmations': blocked, 'attempted_sequence_hashes': sorted(attempted),
              'terminal_native_outcomes': rejected, 'unresolved_native_hosts': sorted(pending),
              'native_host_status': {host: {'status': value.get('status'), 'ownership_unresolved': host in pending,
                   **{key: value.get(key, 0) for key in COUNTERS}} for host, value in results.items()},
              **{key: sum(v.get(key, 0) for v in results.values()) for key in COUNTERS}}
    merged['input_count'] = len(incoming['candidates'])
    merged['not_run'] = len(deferred) + len(blocked)
    output = {'result.json': json.dumps(merged).encode(),
              'native-confirmations.json': json.dumps({'challenge': challenge, 'records': records}).encode()}
    for name, value in output.items():
        path = local / 'native' / name
        c.atomic(path, json.loads(value))
    c.atomic(local / 'native-supervisor.json', {'reason': 'ownership_unresolved' if pending else 'completed',
        'cleanup': {'remaining_descendants': None if pending else 0, 'fixed_point_reached': not pending}})
    if diagnostics is not None:
        diagnostics['native'] = {'host_status': merged['native_host_status']}
        diagnostics['native_hosts'] = host_diagnostics
    return output
