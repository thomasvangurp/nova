#!/usr/bin/env python3
"""N0: bounded CPU imports and identity evidence, never a native profile."""
from __future__ import annotations
import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if __name__ == '__main__' and not sys.dont_write_bytecode:
    print(json.dumps({'status': 'NO_GO', 'error_class': 'BytecodeWritesEnabled'}))
    raise SystemExit(1)
sys.path.insert(0, str(ROOT))
# The native TNP runtime owns a regular `scripts` package. Do not shadow it.
_spec = importlib.util.spec_from_file_location('_pr100_native_probe', ROOT/'scripts/pr100_native_probe.py')
probe = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = probe
_spec.loader.exec_module(probe)

IMPORTS = probe.N0_IMPORTS
HIP_PLUGINS = ('libOpenMMAmoebaHIP.so', 'libOpenMMDrudeHIP.so',
               'libOpenMMHIP.so', 'libOpenMMRPMDHIP.so')
HIP_LIBRARIES = ('libhiprtc.so.6', 'libamdhip64.so.6')
NATIVE_SITE = '/root/.pixi/envs/metanano/lib/python3.12/site-packages/'
FORCEFIELD_IMPORT = {
    'caller_module': 'ImmuneBuilder.refine', 'caller_function': '<module>', 'caller_line': 26,
    'caller_path': NATIVE_SITE+'ImmuneBuilder/refine.py',
    'caller_sha256': 'e33d4bdc3e8354296ede6bde0e2fb438368e5e2fbe87ec0d7b387be0d60ebaf0',
    'constructor_path': NATIVE_SITE+'openmm/app/forcefield.py',
    'constructor_sha256': '7c909edd1bbb8c48ac0070e5ea19b6bf0817be2aeccfbd23e7ebb1302c0da432',
    'xml_argument': 'amber14/protein.ff14SB.xml',
    'xml_path': NATIVE_SITE+'openmm/app/data/amber14/protein.ff14SB.xml',
    'xml_sha256': 'd9f9779c09d67cd5f8bc657692f174ffab14c469dfd06d560ac1899fa7e976b8',
    'expected_constructor_count': 1, 'expected_xml_open_count': 1,
}
LOOPBACK_IMPORT = {
    'declared_import': 'ImmuneBuilder.NanoBodyBuilder2',
    'caller_module': 'urllib3.util.connection', 'caller_function': '_has_ipv6', 'caller_line': 127,
    'caller_path': NATIVE_SITE+'urllib3/util/connection.py',
    'caller_sha256': '2633bbdb69731e5ccb5cf4e4afd65605d86c7979cc5633126f50c92d5ad74a74',
    'parent_function': '<module>', 'parent_line': 137,
    'address': ['::1', 0], 'socket_family': 'AF_INET6', 'socket_type': 'SOCK_STREAM',
    'protocol': 0, 'expected_bind_count': 1, 'requires_closed_socket': True,
}
CPU_QUERY_IMPORT = {
    'caller_module': 'numpy.testing._private.utils', 'caller_function': 'check_support_sve',
    'caller_line': 1247, 'parent_function': '<module>', 'parent_line': 1253,
    'caller_path': NATIVE_SITE+'numpy/testing/_private/utils.py',
    'caller_sha256': 'dc5ad24cc8b439da43383043a27720883433768e4d280e9855f437b8a4524277',
    'executable': '/usr/bin/lscpu',
    'executable_sha256': '250c7ec85ceec9f50101f555704b79ed3a8a917efa14fa455d85136f27182daa',
    'command': 'lscpu', 'expected_query_count': 1,
    'libraries': {
        '/usr/lib/x86_64-linux-gnu/libsmartcols.so.1.1.0': 'cb0d1893c1cef53fda04eb64644e2dd1841c5f879efa491d984eb082d053ec5a',
        '/usr/lib/x86_64-linux-gnu/libc.so.6': '5955dead1a55f545cf9cf34a40b2eb65deb84ea503ac467a266d061073315fa7',
        '/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2': '595bcdc306999711a5a9c787e5d01618f94eb92c4c09a178e875a324f1072d89',
    },
    'library_resolution': {
        '/usr/lib/x86_64-linux-gnu/libsmartcols.so.1': '/usr/lib/x86_64-linux-gnu/libsmartcols.so.1.1.0',
        '/usr/lib/x86_64-linux-gnu/libc.so.6': '/usr/lib/x86_64-linux-gnu/libc.so.6',
        '/lib64/ld-linux-x86-64.so.2': '/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2',
    },
}
N0_WALL_SECONDS = 120


def assemble(inventories, stage_path, remote_stage, output):
    """Local assembly from explicit, already captured public runtime inventories."""
    stage = json.loads(stage_path.read_text())
    files, distributions, evidence, resolved_entrypoints = {}, {}, [], {}
    for path in inventories:
        inventory = json.loads(path.read_text())
        evidence.append({'file_sha256': probe.sha_file(path), 'schema': inventory['schema']})
        distributions.update(inventory.get('distributions', {}))
        if inventory.get('schema') == 'pr100-entrypoint-resolution-v1':
            resolved_entrypoints = inventory['entrypoints']
        for name, row in inventory['files'].items():
            sha = row['sha256'] if isinstance(row, dict) else row
            if name in files and files[name] != sha:
                raise ValueError('inventory disagreement')
            files[name] = sha
    prefix = Path('/root/.pixi/envs/metanano')
    q = Path('/root/pr100-qualification-20261026')
    entrypoints = dict(python=str(prefix/'bin/python3.12'), tnp=str(prefix/'bin/TNP'),
        nanobody_builder_cli=str(prefix/'bin/NanoBodyBuilder2'), hmmscan=str(prefix/'bin/hmmscan'),
        aggrescan=str(q/'NOVA-nanobody-filter/.tools/bin/aggrescan3d-rs'),
        freesasa=str(q/'NOVA-nanobody-filter/.tools/bin/freesasa'),
        hmm=str(prefix/'lib/python3.12/site-packages/anarci/dat/HMMs/ALL.hmm'),
        germline_reference=str(prefix/'lib/python3.12/site-packages/anarci/germlines.py'),
        logging_shim='/root/metanano_shim/bittensor.py')
    if set(resolved_entrypoints) != set(entrypoints):
        raise ValueError('resolved native entrypoint evidence required')
    for name, locator in entrypoints.items():
        record = resolved_entrypoints[name]
        if record['locator'] != locator or files.get(record['resolved_path']) != record['sha256']:
            raise ValueError('native entrypoint resolution evidence mismatch')
        entrypoints[name] = record['resolved_path']
    weights = [str(prefix/'lib/python3.12/site-packages/ImmuneBuilder/trained_model'/f'nanobody_model_{i}')
               for i in range(1, 5)]
    if any(p not in files for p in list(entrypoints.values()) + weights):
        raise ValueError('required runtime file absent from explicit inventories')
    runtime = dict(schema='pr100-native-runtime-manifest-v1', closure_reviewed=False,
        source_stage_manifest_sha256=probe.sha_file(stage_path), source_stage_git_sha=stage['git_sha'],
        source_stage_root=str(remote_stage), files=files, distributions=distributions,
        entrypoints=entrypoints, model_weights=weights, inventory_evidence=evidence,
        path=str(prefix/'bin')+':/usr/local/bin:/usr/bin:/bin', library_path='',
        cuda_visible_devices='', refinement_threads=1, cpu_thread_settings=probe.THREADS,
        n0_protocol_revision=6, n0_wall_seconds=N0_WALL_SECONDS, n0_forcefield_import=FORCEFIELD_IMPORT,
        n0_loopback_import=LOOPBACK_IMPORT, n0_cpu_query_import=CPU_QUERY_IMPORT,
        external_tool_roots=[probe.EXTERNAL_TOOL_ROOT],
        diagnostic_origin_roots=probe.diagnostic_roots(remote_stage),
        entrypoint_resolution_evidence=resolved_entrypoints,
        special_mapping_policies=[], identity_audit_scope='worker_address_space_only',
        cli_child_loaded_identity_observed=False,
        cli_closure_basis='frozen files, N0 imports, and identical PATH/LD_LIBRARY_PATH inheritance',
        limitations=['No full-validator qualification.', 'No observed CLI child address-space identity.',
                    'Delayed imports and runtime data reads remain bounded only by frozen inventory.'])
    probe.atomic_json(output, runtime)
    return {'file_count': len(files), 'runtime_sha256': probe.sha_file(output),
            'closure_reviewed': False, 'source_stage_git_sha': stage['git_sha']}


def forbidden_call(module, function, class_name=''):
    if module.startswith('torch.cuda') and function in {
            'is_available', 'device_count', 'init', '_lazy_init', 'current_device', 'get_device_properties'}:
        return 'gpu_discovery_or_context'
    if module.startswith('ImmuneBuilder') and (function in {'predict', 'save', 'refine'} or
            function == '__init__' and class_name in {'NanoBodyBuilder2', 'ABodyBuilder2', 'TCRBuilder2', 'Nanobody', 'StructureModule'}):
        return 'model_or_profile'
    if module.startswith('theraprofnano') and function in {'main', 'main_compactness', 'CreateAnnotation'}:
        return 'native_metric'
    if module.startswith('openmm') and function == '__init__' and (
            class_name in {'Context', 'System', 'Simulation', 'ForceField'} or 'Integrator' in class_name):
        return 'physics_constructor'
    if module.startswith('openmm') and function in {'createSystem', 'createContext', 'minimizeEnergy', 'step'}:
        return 'physics_execution'
    if module in {'importlib.metadata', 'importlib_metadata'} and function == 'load' and class_name == 'EntryPoint':
        return 'unregistered_entrypoint_execution'
    return None


def validate_forcefield_registration(runtime):
    import xml.etree.ElementTree as etree
    if (runtime.get('n0_protocol_revision') != 6 or runtime.get('n0_wall_seconds') != N0_WALL_SECONDS or
            runtime.get('n0_forcefield_import') != FORCEFIELD_IMPORT):
        raise ValueError('unregistered N0 ForceField exception')
    for key in ('caller', 'constructor', 'xml'):
        path, sha = FORCEFIELD_IMPORT[key+'_path'], FORCEFIELD_IMPORT[key+'_sha256']
        if runtime['files'].get(path) != sha or probe.sha_file(path) != sha:
            raise ValueError('N0 ForceField source or XML identity mismatch')
    xml = etree.parse(FORCEFIELD_IMPORT['xml_path']).getroot()
    if any(xml.findall(tag) for tag in ('Include', 'Script', 'InitializationScript')):
        raise ValueError('N0 ForceField XML includes or executable scripts forbidden')


class ForceFieldImportGuard:
    """One exact native import-time XML parser; no System or Context exception."""
    def __init__(self):
        self.constructor_count = self.xml_open_count = 0
        self.active = False

    def enter(self, caller_module, caller_function, caller_path, caller_line, constructor_path, files):
        r = FORCEFIELD_IMPORT
        if (self.constructor_count or self.active or caller_module != r['caller_module'] or
                caller_function != r['caller_function'] or caller_path != r['caller_path'] or
                caller_line != r['caller_line'] or constructor_path != r['constructor_path'] or
                files != (r['xml_argument'],)):
            raise ValueError('unexpected ForceField construction')
        self.constructor_count += 1
        self.active = True

    def xml_open(self, path, mode, flags):
        if not self.active or str(Path(path).resolve()) != FORCEFIELD_IMPORT['xml_path']:
            raise ValueError('unregistered ForceField XML read')
        if self.xml_open_count or (isinstance(mode, str) and any(c in mode for c in 'wax+')) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
            raise ValueError('unexpected ForceField XML open')
        self.xml_open_count += 1

    def finish(self):
        if self.constructor_count != 1 or self.xml_open_count != 1 or self.active:
            raise ValueError('N0 ForceField import counts incomplete')
        return {'constructor_count': self.constructor_count, 'xml_open_count': self.xml_open_count,
                'scope': 'registered_XML_parameter_parser_only'}


def validate_loopback_registration(runtime):
    r = LOOPBACK_IMPORT
    if runtime.get('n0_loopback_import') != r:
        raise ValueError('unregistered N0 loopback exception')
    if runtime['files'].get(r['caller_path']) != r['caller_sha256'] or probe.sha_file(r['caller_path']) != r['caller_sha256']:
        raise ValueError('N0 loopback source identity mismatch')


class LoopbackImportGuard:
    """One source-pinned IPv6 capability bind; no connection or external networking."""
    def __init__(self):
        self.bind_count = 0
        self.socket = None

    def bind(self, sock, address, declared_import, caller, parent):
        r = LOOPBACK_IMPORT
        if (self.bind_count or declared_import != r['declared_import'] or
                caller != (r['caller_module'], r['caller_function'], r['caller_path'], r['caller_line']) or
                parent != (r['caller_module'], r['parent_function'], r['caller_path'], r['parent_line']) or
                type(address) is not tuple or address != tuple(r['address']) or type(address[1]) is not int or
                not isinstance(sock, socket.socket) or sock.family != socket.AF_INET6 or
                sock.type != socket.SOCK_STREAM or sock.proto != 0 or sock.fileno() < 0):
            raise ValueError('unregistered N0 loopback bind')
        self.socket = sock
        self.bind_count += 1

    def finish(self):
        if self.bind_count != 1 or self.socket is None or self.socket.fileno() != -1:
            raise ValueError('N0 loopback import count or socket closure incomplete')
        return {'bind_count': self.bind_count, 'socket_closed': True,
                'scope': 'registered_IPv6_loopback_capability_probe_only',
                'external_network_operations_permitted': 0}


def validate_cpu_query_registration(runtime):
    r = CPU_QUERY_IMPORT
    if runtime.get('n0_cpu_query_import') != r or runtime.get('library_path', ''):
        raise ValueError('unregistered N0 CPU query exception')
    executable = shutil.which(r['command'], path=runtime['path'])
    if executable != r['executable'] or str(Path(executable).resolve()) != r['executable']:
        raise ValueError('N0 CPU query executable resolution mismatch')
    for locator, resolved in r['library_resolution'].items():
        if str(Path(locator).resolve()) != resolved:
            raise ValueError('N0 CPU query library resolution mismatch')
    files = dict(r['libraries'], **{r['caller_path']:r['caller_sha256'], r['executable']:r['executable_sha256']})
    for path, sha in files.items():
        if runtime['files'].get(path) != sha or probe.sha_file(path) != sha:
            raise ValueError('N0 CPU query source or executable identity mismatch')


def frame_site(frame):
    return ((frame.f_globals.get('__name__'), frame.f_code.co_name,
             frame.f_code.co_filename, frame.f_lineno) if frame is not None else None)


class CPUQueryImportGuard:
    """One native NumPy read-only CPU metadata query; no output leaves the child."""
    def __init__(self):
        self.run_count = self.popen_count = self.exec_count = self.return_count = 0
        self.active = False
        self.process = None

    def enter(self, caller, parent, popenargs, capture_output, input_value, timeout, check, kwargs):
        r = CPU_QUERY_IMPORT
        if (self.run_count or self.active or
                caller != (r['caller_module'], r['caller_function'], r['caller_path'], r['caller_line']) or
                parent != (r['caller_module'], r['parent_function'], r['caller_path'], r['parent_line']) or
                popenargs != (r['command'],) or capture_output is not True or input_value is not None or
                timeout is not None or check is not False or kwargs != {'text':True}):
            raise ValueError('unregistered N0 CPU query call')
        self.run_count += 1
        self.active = True

    def popen(self, values):
        if (not self.active or self.popen_count or values.get('args') != CPU_QUERY_IMPORT['command'] or
                values.get('shell') is not False or values.get('env') is not None or
                values.get('stdin') is not None or values.get('cwd') is not None or
                values.get('executable') is not None or values.get('stdout') != subprocess.PIPE or
                values.get('stderr') != subprocess.PIPE or values.get('text') is not True):
            raise ValueError('unregistered N0 CPU query constructor')
        self.popen_count += 1
        self.process = values['self']

    def execute(self, executable, argv, cwd, env):
        if (not self.active or self.popen_count != 1 or self.exec_count or
                executable != CPU_QUERY_IMPORT['command'] or argv != [CPU_QUERY_IMPORT['command']] or
                cwd is not None or env is not None):
            raise ValueError('unregistered N0 CPU query execution')
        self.exec_count += 1

    def returned(self, result):
        if (not self.active or self.exec_count != 1 or self.return_count or
                not isinstance(result, subprocess.CompletedProcess) or result.returncode != 0 or
                result.args != CPU_QUERY_IMPORT['command']):
            raise ValueError('N0 CPU query did not complete cleanly')
        self.return_count += 1
        self.active = False

    def finish(self):
        if (self.active or (self.run_count, self.popen_count, self.exec_count, self.return_count) != (1,1,1,1) or
                self.process is None or self.process.poll() != 0):
            raise ValueError('N0 CPU query counts or completion incomplete')
        return {'run_count':1, 'popen_count':1, 'exec_count':1, 'return_count':1, 'exit_code':0,
                'scope':'registered_readonly_CPU_capability_query_only', 'shell':False,
                'child_address_space_observed':False}


def plugin_categories(messages):
    rows = []
    for message in messages:
        plugins = [p for p in HIP_PLUGINS if p in message]
        missing = [p for p in HIP_LIBRARIES if p in message]
        if len(plugins) != 1 or not missing:
            raise ValueError('unclassified OpenMM plugin failure')
        rows.append({'plugin': plugins[0], 'missing_libraries': missing, 'scope': 'unused_CPU_only'})
    return rows


def import_child(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    private = Path(spec['private']); runtime = spec['runtime']
    startup_stage = 'environment'
    try:
        probe.assert_bytecode_disabled()
        if os.environ.get('PATH') != runtime['path'] or os.environ.get('LD_LIBRARY_PATH', '') != runtime.get('library_path', ''):
            raise ValueError('native inherited paths differ from registration')
        os.chdir(private)
        startup_stage = 'forcefield_registration'
        validate_forcefield_registration(runtime)
        startup_stage = 'loopback_registration'
        validate_loopback_registration(runtime)
        startup_stage = 'cpu_query_registration'
        validate_cpu_query_registration(runtime)
    except BaseException as error:
        probe.atomic_json(private/'child.json', {'status': 'failed', 'stage': startup_stage,
                                               'error_class': type(error).__name__})
        return 1
    forcefield = ForceFieldImportGuard()
    loopback = LoopbackImportGuard()
    cpu_query = CPUQueryImportGuard()
    def stop(category):
        probe.atomic_json(private/'child.json', {'status': 'forbidden_call', 'category': category})
        os._exit(71)
    def profile(frame, event, arg):
        if frame.f_globals.get('__name__') == 'subprocess':
            try:
                if event == 'call' and frame.f_code.co_name == 'run':
                    values = frame.f_locals
                    cpu_query.enter(frame_site(frame.f_back), frame_site(frame.f_back.f_back),
                        values.get('popenargs'), values.get('capture_output'), values.get('input'),
                        values.get('timeout'), values.get('check'), values.get('kwargs'))
                elif event == 'call' and frame.f_code.co_qualname == 'Popen.__init__':
                    cpu_query.popen(frame.f_locals)
                elif event == 'return' and frame.f_code.co_name == 'run':
                    cpu_query.returned(arg)
            except ValueError:
                stop('unregistered_cpu_query')
            return
        if event == 'call':
            module = frame.f_globals.get('__name__', '')
            function = frame.f_code.co_name
            if not module.startswith(('openmm', 'ImmuneBuilder', 'theraprofnano', 'torch.cuda', 'importlib.metadata', 'importlib_metadata')):
                return
            if module == 'openmm.app.forcefield' and frame.f_code.co_qualname == 'ForceField.loadFile':
                if not forcefield.active or frame.f_locals.get('files') != (FORCEFIELD_IMPORT['xml_argument'],):
                    stop('unexpected_forcefield_load')
            if module == 'openmm.app.forcefield' and frame.f_code.co_qualname == 'ForceField.__init__':
                caller = frame.f_back
                try:
                    forcefield.enter(caller.f_globals.get('__name__'), caller.f_code.co_name,
                        caller.f_code.co_filename, caller.f_lineno, frame.f_code.co_filename,
                        frame.f_locals.get('files'))
                except ValueError:
                    stop('unexpected_forcefield_constructor')
                return
            # Inspect locals only on the relevant constructor/entrypoint sites.
            obj = frame.f_locals.get('self') if function in {'__init__', 'load'} else None
            category = forbidden_call(module, function, type(obj).__name__ if obj is not None else '')
            if category:
                stop(category)
        elif event == 'return' and frame.f_globals.get('__name__') == 'openmm.app.forcefield' and frame.f_code.co_qualname == 'ForceField.__init__':
            forcefield.active = False
        elif event == 'c_call' and getattr(arg, '__name__', '') in {
                '_cuda_init', '_cuda_getDeviceCount', '_cuda_getDeviceProperties', '_cuda_setDevice'}:
            stop('gpu_native_call')
    def audit(event, args):
        if event == 'socket.bind':
            caller = sys._getframe(1)
            parent = caller.f_back
            try:
                loopback.bind(args[0], args[1], current_import, frame_site(caller), frame_site(parent))
            except ValueError:
                stop('unregistered_loopback_bind')
        if event == 'subprocess.Popen':
            try:
                cpu_query.execute(*args)
            except ValueError:
                stop('unregistered_cpu_query_execution')
        if event in {'os.system', 'os.posix_spawn', 'os.fork', 'socket.connect'}:
            stop('subprocess_or_network')
        if event == 'open' and forcefield.active and isinstance(args[0], (str, bytes)) and os.fsdecode(args[0]).endswith('.xml'):
            try:
                forcefield.xml_open(os.fsdecode(args[0]), args[1], args[2])
            except ValueError:
                stop('unexpected_forcefield_xml')
    sys.addaudithook(audit)
    sys.setprofile(profile)
    started, cpu = time.monotonic(), probe.cpu_seconds()
    stage, current_import = 'filter_activation', None
    try:
        from nanobody_opt.pr100_filter import activate_filter
        activate_filter()
        stage = 'declared_imports'
        for module in IMPORTS:
            current_import = module
            importlib.import_module(module)
        import_seconds = time.monotonic()-started
        stage = 'stage_locality'
        staged_origins = probe.assert_stage_origins()
        stage = 'cpu_platform'
        from openmm import Platform
        platform = Platform.getPlatformByName('CPU')
        name, properties = platform.getName(), list(platform.getPropertyNames())
        if name != 'CPU' or 'Threads' not in properties:
            raise ValueError('CPU Threads platform unavailable')
        stage = 'plugin_classification'
        failures = plugin_categories(Platform.getPluginLoadFailures())
        stage = 'forcefield_counts'
        forcefield_receipt = forcefield.finish()
        stage = 'loopback_counts'
        loopback_receipt = loopback.finish()
        stage = 'cpu_query_counts'
        cpu_query_receipt = cpu_query.finish()
        stage = 'loaded_identity'
        audit_start = time.monotonic()
        identity = probe.verify_loaded_identity(runtime, private/'loaded_identity.json', set(),
                                                deadline=spec['deadline'], n0_diagnostics=True)
        receipt = dict(status='complete', declared_imports=list(IMPORTS), cpu_platform=name,
                       cpu_threads_property=True, plugin_failure_categories=failures,
                       first_loaded_audit_seconds=time.monotonic()-audit_start,
                       loaded_identity=identity, actual_cpu_seconds=probe.cpu_seconds()-cpu,
                       forcefield_import=forcefield_receipt,
                       loopback_import=loopback_receipt,
                       cpu_query_import=cpu_query_receipt,
                       staged_module_origins=staged_origins, import_seconds=import_seconds,
                       elapsed_seconds=time.monotonic()-started,
                       model_constructions_permitted=0, native_profiles_permitted=0,
                       gpu_discovery_calls_permitted=0, registered_guard_hits_permitted=0)
    except BaseException as error:
        receipt = {'status': 'failed', 'stage': stage, 'error_class': type(error).__name__}
        if stage == 'declared_imports':
            receipt['declared_import'] = current_import
    finally:
        sys.setprofile(None)
    probe.atomic_json(private/'child.json', receipt)
    return 0 if receipt['status'] == 'complete' else 1


def check(runtime, output):
    started, cpu = time.monotonic(), probe.cpu_seconds()
    if output.exists():
        raise ValueError('private output already exists')
    output.mkdir(parents=True)
    conditions, children = [], []
    report = dict(scope='N0_CPU_import_identity_only', conditions=conditions, native_profiles_permitted=0,
                  gpu_calls_permitted=0, kernel_enforced_aggregate_memory_limit=False,
                  limits=dict(wall_seconds=N0_WALL_SECONDS, cpu_affinity_count=2, sampled_rss_abort_bytes=4*1024**3),
                  runtime_input_sha256=probe.digest(runtime), closure_reviewed=False,
                  harness_sha256=probe.sha_file(Path(__file__)))
    owned, verified, peak = False, False, 0
    old_handler = None
    try:
        probe.assert_bytecode_disabled()
        if sys.platform != 'linux':
            raise ValueError('Linux required')
        probe.dedicated_controller(); owned = True
        def interrupted(signum, frame):
            raise InterruptedError('N0 interrupted')
        old_handler = signal.signal(signal.SIGTERM, interrupted)
        cpus = sorted(os.sched_getaffinity(0))
        if len(cpus) < 2:
            raise ValueError('two CPU slots required')
        os.sched_setaffinity(0, cpus[:2])
        report['start_memory'] = probe.available_memory()
        probe.require_headroom(report['start_memory'], 96*1024**3)
        if runtime['source_stage_root'] != str(ROOT):
            raise ValueError('wrong staged research source')
        report['verification'] = probe.verify_runtime(runtime, deadline=started+60, preflight_only=True)
        verified = True
        reserve = report['verification']['seconds'] + 5
        report['post_verification_reserve_seconds'] = report['verification']['seconds']
        report['cleanup_and_receipt_reserve_seconds'] = 5
        child_deadline = started+N0_WALL_SECONDS-reserve
        if child_deadline-time.monotonic() < 5:
            raise TimeoutError('no remaining import budget')
        spec = output/'spec.json'
        probe.atomic_json(spec, {'runtime': runtime, 'private': str(output), 'deadline': child_deadline})
        env = dict(probe.THREADS, PATH=runtime['path'], LANG='C.UTF-8',
            PYTHONPATH=str(ROOT)+':'+str(Path(runtime['entrypoints']['logging_shim']).parent),
            PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='',
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TMPDIR=str(output),
            MPLCONFIGDIR=str(output/'mpl-cache'), XDG_CACHE_HOME=str(output/'cache'), TORCH_HOME=str(output/'torch-cache'))
        if runtime.get('library_path'):
            env['LD_LIBRARY_PATH'] = runtime['library_path']
        child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '--import-child', str(spec)],
                                 env=env, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        children.append(child)
        while child.poll() is None:
            rss = sum(probe.process_rss(p) for p in probe.descendants() | {os.getpid()})
            peak = max(peak, rss)
            memory = probe.available_memory()
            report['last_memory'] = memory
            if memory['cgroup_resolution'] != 'resolved_visible_ancestors':
                conditions.append('cgroup_resolution_lost')
            elif memory['available_bytes'] < 64*1024**3:
                conditions.append('available_memory_abort')
            if rss > 4*1024**3:
                conditions.append('sampled_rss_abort')
            if time.monotonic() >= child_deadline:
                conditions.append('import_deadline')
            if conditions:
                break
            time.sleep(.05)
        report['child_exitcode'] = child.returncode
        if not conditions:
            if (output/'child.json').is_file():
                report['child'] = json.loads((output/'child.json').read_text())
                if child.returncode or report['child']['status'] != 'complete':
                    conditions.append('import_or_identity_failed')
            else:
                conditions.append('child_receipt_missing')
    except BaseException as error:
        conditions.append('preflight_or_import_failed')
        report['error_class'] = type(error).__name__
    finally:
        if owned:
            report['cleanup'] = probe.stop_descendants(children, timeout=max(.1, min(3, started+N0_WALL_SECONDS-3-time.monotonic())))
            if children:
                report['child_exitcode_after_cleanup'] = children[0].poll()
            if report['cleanup']['remaining_descendants'] or not report['cleanup']['fixed_point_reached']:
                conditions.append('containment_failure')
        if verified:
            try:
                post_deadline = min(started+N0_WALL_SECONDS-1,
                                    time.monotonic()+report['verification']['seconds']+1)
                report['post_verification'] = probe.verify_runtime(runtime, deadline=post_deadline, preflight_only=True)
            except BaseException as error:
                conditions.append('post_verification_failed'); report['post_error_class'] = type(error).__name__
    report.update(elapsed_seconds=time.monotonic()-started, actual_cpu_seconds=probe.cpu_seconds()-cpu,
                  allocated_cpu_seconds=2*(time.monotonic()-started), sampled_tree_rss_sum_peak_bytes=peak)
    if report['elapsed_seconds'] > N0_WALL_SECONDS:
        conditions.append('wall_cap_failure')
    report['status'] = 'complete_for_independent_review' if not conditions else 'NO_GO'
    probe.atomic_json(output/'receipt.json', report)
    if not conditions:
        candidate = dict(runtime, n0_receipt_sha256=probe.sha_file(output/'receipt.json'),
            n0_receipt_path=str(output/'receipt.json'), n0_measurements=probe.n0_measurements_from_receipt(report))
        probe.atomic_json(output/'runtime-candidate.json', candidate)
    if old_handler is not None:
        signal.signal(signal.SIGTERM, old_handler)
    return {k: report[k] for k in ('status', 'conditions', 'elapsed_seconds', 'actual_cpu_seconds')}


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '--import-child':
        return import_child(sys.argv[2])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('assemble', 'check'))
    parser.add_argument('--inventory', action='append', type=Path)
    parser.add_argument('--stage-manifest', type=Path)
    parser.add_argument('--remote-stage', type=Path)
    parser.add_argument('--runtime', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        report = (assemble(args.inventory, args.stage_manifest, args.remote_stage, args.output)
                  if args.action == 'assemble' else check(json.loads(args.runtime.read_text()), args.output.resolve()))
    except BaseException as error:
        report = {'status': 'NO_GO', 'error_class': type(error).__name__}
    print(json.dumps(report))
    return 1 if report.get('status') == 'NO_GO' else 0


if __name__ == '__main__':
    raise SystemExit(main())
