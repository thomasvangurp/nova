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

IMPORTS = (
    'torch', 'ImmuneBuilder.NanoBodyBuilder2', 'ImmuneBuilder.refine',
    'theraprofnano.CDR_Profiler.CDR_Assigner',
    'theraprofnano.CDR_Profiler.CDR3_Conf_Assigner',
    'theraprofnano.Hydrophobicity_and_Charge_Profiler.Hydrophobicity_and_Charge_Assigner',
    'metanano.config', 'metanano.filters.developability',
    'metanano.utils.aggrescan3d_wrapper', 'metanano.utils.tnp_wrapper',
    'nanobody_opt.tnp_daemon', 'nanobody_opt.rules', 'openmm.app',
)
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


def assemble(inventories, stage_path, remote_stage, output):
    """Local assembly from explicit, already captured public runtime inventories."""
    stage = json.loads(stage_path.read_text())
    files, distributions, evidence = {}, {}, []
    for path in inventories:
        inventory = json.loads(path.read_text())
        evidence.append({'file_sha256': probe.sha_file(path), 'schema': inventory['schema']})
        distributions.update(inventory.get('distributions', {}))
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
        n0_protocol_revision=3, n0_forcefield_import=FORCEFIELD_IMPORT,
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
    if runtime.get('n0_protocol_revision') != 3 or runtime.get('n0_forcefield_import') != FORCEFIELD_IMPORT:
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
    probe.assert_bytecode_disabled()
    os.chdir(private)
    validate_forcefield_registration(runtime)
    forcefield = ForceFieldImportGuard()
    def stop(category):
        probe.atomic_json(private/'child.json', {'status': 'forbidden_call', 'category': category})
        os._exit(71)
    def profile(frame, event, arg):
        if event == 'call':
            obj = frame.f_locals.get('self')
            if frame.f_globals.get('__name__') == 'openmm.app.forcefield' and frame.f_code.co_qualname == 'ForceField.loadFile':
                if not forcefield.active or frame.f_locals.get('files') != (FORCEFIELD_IMPORT['xml_argument'],):
                    stop('unexpected_forcefield_load')
            if frame.f_globals.get('__name__') == 'openmm.app.forcefield' and frame.f_code.co_qualname == 'ForceField.__init__':
                caller = frame.f_back
                try:
                    forcefield.enter(caller.f_globals.get('__name__'), caller.f_code.co_name,
                        caller.f_code.co_filename, caller.f_lineno, frame.f_code.co_filename,
                        frame.f_locals.get('files'))
                except ValueError:
                    stop('unexpected_forcefield_constructor')
                return
            category = forbidden_call(frame.f_globals.get('__name__', ''), frame.f_code.co_name,
                                      type(obj).__name__ if obj is not None else '')
            if category:
                stop(category)
        elif event == 'return' and frame.f_globals.get('__name__') == 'openmm.app.forcefield' and frame.f_code.co_qualname == 'ForceField.__init__':
            forcefield.active = False
        elif event == 'c_call' and getattr(arg, '__name__', '') in {
                '_cuda_init', '_cuda_getDeviceCount', '_cuda_getDeviceProperties', '_cuda_setDevice'}:
            stop('gpu_native_call')
    def audit(event, args):
        if event in {'subprocess.Popen', 'os.system', 'os.posix_spawn', 'os.fork', 'socket.connect', 'socket.bind'}:
            stop('subprocess_or_network')
        if event == 'open' and forcefield.active and isinstance(args[0], (str, bytes)) and os.fsdecode(args[0]).endswith('.xml'):
            try:
                forcefield.xml_open(os.fsdecode(args[0]), args[1], args[2])
            except ValueError:
                stop('unexpected_forcefield_xml')
    sys.addaudithook(audit)
    sys.setprofile(profile)
    started, cpu = time.monotonic(), probe.cpu_seconds()
    try:
        from nanobody_opt.pr100_filter import activate_filter
        activate_filter()
        for module in IMPORTS:
            importlib.import_module(module)
        from openmm import Platform
        platform = Platform.getPlatformByName('CPU')
        name, properties = platform.getName(), list(platform.getPropertyNames())
        if name != 'CPU' or 'Threads' not in properties:
            raise ValueError('CPU Threads platform unavailable')
        failures = plugin_categories(Platform.getPluginLoadFailures())
        forcefield_receipt = forcefield.finish()
        audit_start = time.monotonic()
        identity = probe.verify_loaded_identity(runtime, private/'loaded_identity.json', set(),
                                                deadline=spec['deadline'])
        receipt = dict(status='complete', declared_imports=list(IMPORTS), cpu_platform=name,
                       cpu_threads_property=True, plugin_failure_categories=failures,
                       first_loaded_audit_seconds=time.monotonic()-audit_start,
                       loaded_identity=identity, actual_cpu_seconds=probe.cpu_seconds()-cpu,
                       forcefield_import=forcefield_receipt,
                       elapsed_seconds=time.monotonic()-started,
                       model_constructions_permitted=0, native_profiles_permitted=0,
                       gpu_discovery_calls_permitted=0, registered_guard_hits=[])
    except BaseException as error:
        receipt = {'status': 'failed', 'stage': 'guarded_import_or_identity', 'error_class': type(error).__name__}
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
                  limits=dict(wall_seconds=60, cpu_affinity_count=2, sampled_rss_abort_bytes=4*1024**3),
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
        report['verification'] = probe.verify_runtime(runtime, deadline=started+30, preflight_only=True)
        verified = True
        reserve = report['verification']['seconds'] + 5
        child_deadline = started+60-reserve
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
            probe.require_headroom(probe.available_memory(), 64*1024**3)
            if rss > 4*1024**3:
                raise MemoryError('sampled RSS abort threshold')
            if time.monotonic() >= child_deadline:
                raise TimeoutError('N0 import deadline')
            time.sleep(.05)
        report['child_exitcode'] = child.returncode
        report['child'] = json.loads((output/'child.json').read_text())
        if child.returncode or report['child']['status'] != 'complete':
            conditions.append('import_or_identity_failed')
    except BaseException as error:
        conditions.append('preflight_or_import_failed')
        report['error_class'] = type(error).__name__
    finally:
        if owned:
            report['cleanup'] = probe.stop_descendants(children, timeout=max(.1, min(3, started+57-time.monotonic())))
            if report['cleanup']['remaining_descendants'] or not report['cleanup']['fixed_point_reached']:
                conditions.append('containment_failure')
        if verified:
            try:
                report['post_verification'] = probe.verify_runtime(runtime, deadline=started+59, preflight_only=True)
            except BaseException as error:
                conditions.append('post_verification_failed'); report['post_error_class'] = type(error).__name__
    report.update(elapsed_seconds=time.monotonic()-started, actual_cpu_seconds=probe.cpu_seconds()-cpu,
                  allocated_cpu_seconds=2*(time.monotonic()-started), sampled_tree_rss_sum_peak_bytes=peak)
    if report['elapsed_seconds'] > 60:
        conditions.append('wall_cap_failure')
    report['status'] = 'complete_for_independent_review' if not conditions else 'NO_GO'
    probe.atomic_json(output/'receipt.json', report)
    if not conditions:
        candidate = dict(runtime, n0_receipt_sha256=probe.sha_file(output/'receipt.json'),
            n0_measurements={'verification': max([report['verification'], report['post_verification']], key=lambda r:r['seconds']),
                             'first_loaded_audit_seconds': report['child']['first_loaded_audit_seconds']})
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
