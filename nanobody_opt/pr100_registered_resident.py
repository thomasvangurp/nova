"""Registered opt-in resident native execution under the existing GPU lease.

Original TNP/NBB2 CLI bodies and scientific functions execute unchanged. Models
live across bounded batches; only main-refinement Context routing changes. No
scientific imports occur in the controller process. New qualified registration
is required; enabling a flag never promotes an old CPU receipt.
"""
from __future__ import annotations
import argparse,asyncio,functools,hashlib,importlib,json,locale,logging,math,os,re,selectors,subprocess,sys,threading,time,uuid
from contextlib import contextmanager,redirect_stdout,redirect_stderr
from pathlib import Path
from types import ModuleType
from concurrent.futures import ThreadPoolExecutor
from nanobody_opt import pr100_validator_reference as reference
from nanobody_opt import pr100_native_reference as native
MODULE='nanobody_opt.pr100_registered_resident'
ROOT=Path(__file__).resolve().parents[1]
_process_lock=threading.Lock()
class AdapterError(RuntimeError):
    pass

def require(value, reason):
    if not value:
        raise AdapterError(reason)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def restore_utf8_locale():
    """Re-assert a UTF-8 process locale before every resident draw.

    TNP's post-prediction stages reset the C locale to POSIX/ASCII, which
    changes the default text encoding of any file opened without an explicit
    ``encoding`` (and of the redirected stdout). Re-assert UTF-8 so a draw's
    own stdout and TNP's log files stay UTF-8 across repeated draws.
    """
    for name in ('C.UTF-8', 'C.utf8', 'en_US.UTF-8'):
        try:
            locale.setlocale(locale.LC_ALL, name)
            return
        except locale.Error:
            continue

def ensure_biopython_three_to_one():
    """Compat for the sealed TNP scripts against Biopython 1.85.

    The sealed environment's ``scripts/process_pdb.py`` still imports
    ``three_to_one`` from ``Bio.PDB.Polypeptide`` inside ``run_dssp``. That
    name no longer exists in Biopython 1.85 (the canonical mapping is
    ``protein_letters_3to1``), so ``get_modelling_details`` raises ImportError
    and never writes ``modelling_details.jsonp`` / ``sequence_liabilities.csv``.
    Expose the old name as a function before TNP executes so the sealed source
    runs unchanged.
    """
    import Bio.PDB.Polypeptide as polypeptide
    if not hasattr(polypeptide, 'three_to_one'):
        polypeptide.three_to_one = (lambda resname:
            polypeptide.protein_letters_3to1.get(resname, 'X'))

def execute_program(code):
    """Give an unchanged entrypoint its own __main__ globals and script path."""
    previous_main, previous_path = sys.modules.get('__main__'), sys.path
    module = ModuleType('__main__')
    module.__file__, module.__package__ = code.co_filename, None
    try:
        sys.modules['__main__'] = module
        script_directory = str(Path(code.co_filename).parent)
        sys.path = [script_directory, *[p for p in previous_path if p != script_directory]]
        exec(code, module.__dict__)
    finally:
        sys.path = previous_path
        if previous_main is None:
            sys.modules.pop('__main__', None)
        else:
            sys.modules['__main__'] = previous_main

class OriginalNBB2CLI:
    """Dispatch only TNP's exact registered NBB2 command to the ORIGINAL CLI.

    Only the module-global constructor is temporarily substituted; predict(),
    Nanobody.save(), model selection, refinement and argument parsing remain
    the original functions. The class object remains intact for Observer hooks.
    """
    def __init__(self, module, model, sequence, name, profiles, *, entrypoint_code):
        self.module, self.model = module, model
        self.entrypoint_code = entrypoint_code
        self.command = ['NanoBodyBuilder2', '--heavy_sequence', sequence, '--output',
                        str(Path(name) / (name + '.pdb')), '--n_threads', '1']
        self.profiles = Path(profiles).resolve()

    def cached_constructor(self, *args, **kwargs):
        require(not args and kwargs == {'numbering_scheme': 'imgt'},
                'unregistered_model_constructor')
        return self.model

    def run(self, command, **kwargs):
        require(list(command) in (self.command, self.command + ['--verbose'])
                and kwargs == {'check': True} and Path.cwd() == self.profiles,
                'unregistered_nbb_command')
        previous_argv, previous_constructor = sys.argv, self.module.NanoBodyBuilder2
        code = 0
        try:
            sys.argv = list(command)
            self.module.NanoBodyBuilder2 = self.cached_constructor
            try:
                execute_program(self.entrypoint_code)
            except SystemExit as error:
                code = error.code if type(error.code) is int else (0 if error.code is None else 1)
        finally:
            self.module.NanoBodyBuilder2 = previous_constructor
            sys.argv = previous_argv
        if code:
            raise subprocess.CalledProcessError(code, command)
        return subprocess.CompletedProcess(command, 0)

@contextmanager
def original_program_context(argv, cwd, bridge, *, preparation_threads, trace_path, path=None):
    """Restore every process-global override, including exceptional CLI exits."""
    old_cwd = Path.cwd()  # may raise; evaluate before taking the lock so it cannot leak
    require(_process_lock.acquire(blocking=False), 'resident_process_busy')
    previous = (sys.argv, old_cwd, subprocess.run, logging.root.manager.disable)
    keys = ('OPENMM_CPU_THREADS', 'NOVA_PR100_CONTEXT_TRACE', 'PATH', 'PYTHONDONTWRITEBYTECODE')
    environment = {key: os.environ.get(key) for key in keys}
    original_run = subprocess.run
    def dispatch(command, *args, **kwargs):
        if isinstance(command, (list, tuple)) and command and command[0] == 'NanoBodyBuilder2':
            require(not args, 'unregistered_nbb_command')
            return bridge.run(command, **kwargs)
        return original_run(command, *args, **kwargs)
    try:
        sys.argv = list(argv)
        os.chdir(cwd)
        os.environ.update(OPENMM_CPU_THREADS=str(preparation_threads),
                          NOVA_PR100_CONTEXT_TRACE=str(trace_path), PYTHONDONTWRITEBYTECODE='1')
        if path is not None:
            os.environ['PATH'] = path
        logging.disable(logging.CRITICAL)  # Original refine.py import effect.
        subprocess.run = dispatch
        yield
    finally:
        sys.argv, old_cwd, subprocess.run, disabled = previous
        try:
            try:
                os.chdir(old_cwd)
            finally:
                logging.disable(disabled)
                for key, value in environment.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
        finally:
            _process_lock.release()


@contextmanager
def route_contexts(observer, mm, policy, *, temporary_directory):
    """Route before Observer.install(), so its genuine hook records actual devices.

    Sidechain repair can create Contexts directly as well as through the main
    minimizer. Chirality work reuses that same Simulation.
    No force field, integrator, hydrogen selection, or minimization call changes.
    """
    if policy['final_refinement'] == {'platform':'CPU','threads':1}:
        require(policy['preparation_threads'] == 1, 'resident_cpu_requires_preparation_one')
        # Preserve the original CPU Context constructor and arguments entirely.
        # Observer still records the genuine original preparation/main devices.
        yield
        return
    require(policy['final_refinement'] == {
        'platform': 'CUDA', 'precision': 'mixed', 'deterministic_forces': True},
        'resident_requires_registered_cuda_policy')
    original = mm.Context.__init__
    cpu, cuda = (mm.Platform.getPlatformByName(name) for name in ('CPU', 'CUDA'))
    @functools.wraps(original)
    def routed(context, *args, **kwargs):
        stack = getattr(observer.local, 'stack', [])
        role = stack[-1][0] if stack else None
        if role in ('missing_atoms', 'hydrogens', 'minimize_energy', 'sidechain_repair'):
            require(not kwargs and len(args) in (2, 3, 4), 'unregistered_context_signature')
            if role in ('minimize_energy', 'sidechain_repair'):
                args = (*args[:2], cuda, {'Precision': 'mixed', 'DeterministicForces': 'true',
                    'DeviceIndex': '0', 'TempDirectory': str(temporary_directory)})
            else:
                args = (*args[:2], cpu, {'Threads': str(policy['preparation_threads'])})
        return original(context, *args, **kwargs)
    mm.Context.__init__ = routed
    try:
        yield
    finally:
        # Observer must restore its outer hook before this boundary is left.
        intact = mm.Context.__init__ is routed
        mm.Context.__init__ = original
        require(intact, 'resident_context_router_replaced')


def load_refinement_plugin(mm, settings, row):
    """CPU refinement needs no OpenMM CUDA sidecar; Torch prediction stays CUDA."""
    if row['policy']['final_refinement'] == {'platform':'CPU','threads':1}:
        require(row['policy']['preparation_threads'] == 1, 'resident_cpu_requires_preparation_one')
        return
    plugin = native.cuda_plugin_paths(settings, row)
    mm.Platform.loadPluginsFromDirectory(plugin['plugin_directory'])
    mm.Platform.getPlatformByName('CUDA')  # Never silently fall back.


def verify_cuda_loader(mm, torch, row, *, maps_path=Path('/proc/self/maps')):
    """Hash only the unique actually mapped core/NVRTC at CUDA worker startup."""
    if row['policy']['final_refinement']['platform'] != 'CUDA':
        return
    plugin=reference.validate_cuda_plugin_identity(row.get('cuda_plugin'))
    require('loader' in plugin,'cuda_loader_identity_required')
    mapped={'core':set(),'nvrtc':set()}
    for line in maps_path.read_text().splitlines():
        fields=line.split(None,5)
        if len(fields)!=6:continue
        raw=fields[5];path=Path(raw.removesuffix(' (deleted)'));name=path.name
        kind=('core' if name=='libOpenMM.so' else
              'nvrtc' if re.fullmatch(r'libnvrtc\.so(?:\.\d+)*',name) else None)
        if kind is None:continue
        require(raw==str(path),'cuda_loader_mapping_deleted')
        mapped[kind].add(path.resolve())
    require(all(len(paths)==1 for paths in mapped.values()),'cuda_loader_mapping_not_unique')
    actual={'core_sha256':native.sha(next(iter(mapped['core']))),
        'nvrtc_sha256':native.sha(next(iter(mapped['nvrtc']))),
        'torch_compiled_cuda':torch._C._cuda_getCompiledVersion(),
        'cpp_version':mm.Platform.getOpenMMVersion()}
    require(actual==plugin['loader'],'cuda_loader_identity_mismatch')
    return actual


def model_fingerprint(model):
    """Full model state at startup / batch end, never inside a draw."""
    digest = hashlib.sha256()
    digest.update(json.dumps({'device':str(model.device),'scheme':model.scheme}).encode())
    for name, member in model.models.items():
        digest.update(json.dumps([name, [(key, child.training)
            for key, child in member.named_modules()]]).encode())
        for key, tensor in member.state_dict().items():
            array = tensor.detach().cpu().contiguous().numpy()
            digest.update(json.dumps([key, str(array.dtype), list(array.shape)]).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


class Engine:
    """One registered model instance per serial scientific worker process."""
    def __init__(self, request):
        from nanobody_opt.pr100_live_gpu import inherited_lease_fd
        self.settings = request['settings']; self.gpu = request['gpu']
        self.row = native.registration(self.settings['registration_sha256'])
        require(self.settings.get('resident') is True, 'resident_execution_disabled')
        require(sys.dont_write_bytecode and inherited_lease_fd([self.gpu]) is not None,
                'resident_requires_bytecode_free_leased_process')
        require(all(os.environ.get(key) == '1' for key in
                    ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')),
                'qualified_startup_thread_environment_required')
        environment = Path(self.settings['environment'])
        require(Path(sys.prefix).resolve() == environment.resolve(), 'qualified_interpreter_required')
        require(not any(name in sys.modules for name in ('torch', 'openmm', 'ImmuneBuilder')),
                'scientific_worker_must_start_fresh')
        self.runtime = native.verify_files(self.settings, self.row)
        require(native.sha(__file__) == self.row['resident_module_sha256'], 'resident_module_changed')
        os.environ['OPENMM_CPU_THREADS'] = str(self.row['policy']['preparation_threads'])
        self.root = Path(request['directory'])
        for name in ('cuda-cache', 'openmm-cache', 'openmm-tmp', 'tmp'):
            (self.root / name).mkdir(exist_ok=True)
        os.environ.update(CUDA_CACHE_PATH=str(self.root/'cuda-cache'),
            OPENMM_CACHE_DIR=str(self.root/'openmm-cache'), TMPDIR=str(self.root/'tmp'))
        self.torch = importlib.import_module('torch'); self.torch.set_num_threads(1)
        require(self.torch.cuda.is_available(), 'registered_cuda_required')
        self.mm = importlib.import_module('openmm'); importlib.import_module('openmm.app')
        self.fixer = importlib.import_module('pdbfixer')
        load_refinement_plugin(self.mm, self.settings, self.row)
        if self.row['policy']['final_refinement']['platform'] == 'CUDA':
            verify_cuda_loader(self.mm,self.torch,self.row)
        disabled = logging.root.manager.disable
        try:
            self.builder = importlib.import_module('ImmuneBuilder.NanoBodyBuilder2')
            self.refine = importlib.import_module('ImmuneBuilder.refine')
            self.model = self.builder.NanoBodyBuilder2(numbering_scheme='imgt')
        finally:
            logging.disable(disabled)
        for module in (self.torch,self.mm,self.fixer,self.builder,self.refine):
            require(Path(module.__file__).resolve().is_relative_to(environment.resolve()),
                    'scientific_import_outside_qualified_environment')
        for name, expected in self.runtime['sources'].items():
            loaded = sys.modules.get(name)
            if loaded is not None:
                require(native.sha(loaded.__file__) == expected, 'loaded_scientific_source_changed')
        require(self.model.device == 'cuda', 'registered_prediction_device_changed')
        self.tnp_code = compile((environment/'bin/TNP').read_bytes(), str(environment/'bin/TNP'),
                                'exec', dont_inherit=True)
        self.entry_code = compile((environment/'bin/NanoBodyBuilder2').read_bytes(),
                                  str(environment/'bin/NanoBodyBuilder2'), 'exec', dont_inherit=True)
        self.refine_code = compile(Path(self.refine.__file__).read_bytes(), self.refine.__file__,
                                   'exec', dont_inherit=True)
        self.initial_state = model_fingerprint(self.model)
        self.completed = 0

    def draw(self, request, index, *, deadline):
        """Fresh original predict/save/TNP/A3D and a new genuine Observer."""
        from nanobody_opt.pr100_context_observer import Observer
        from nanobody_opt.pr100_live_gpu import inherited_lease_fd
        require(time.monotonic() < deadline and inherited_lease_fd([self.gpu]) is not None,
                'resident_deadline_or_lease_lost')
        sequence = request['sequences'][index]
        require(re.fullmatch('[ACDEFGHIKLMNPQRSTVWY]{50,300}', sequence) is not None,
                'invalid_sequence')
        # Re-assert UTF-8 before this draw's stdout/log files are opened; TNP's
        # post-prediction stages can leave the process locale at POSIX/ASCII.
        restore_utf8_locale()
        ensure_biopython_three_to_one()
        native.atomic(Path(request['directory'])/(str(index)+'.started.json'), {'started':True})
        directory = Path(request['directory']) / str(index); directory.mkdir()
        started = time.monotonic(); key = hashlib.sha256(sequence.encode()).hexdigest()
        name = 'reference_' + key[:16]; profiles = directory/'tnp'; profiles.mkdir()
        # Recreate original module globals, including ForceField, BEFORE observing.
        disabled = logging.root.manager.disable
        try:
            exec(self.refine_code, self.refine.__dict__)
            self.builder.refine = self.refine.refine
        finally:
            logging.disable(disabled)
        observer = Observer(directory/'trace.jsonl')
        bridge = OriginalNBB2CLI(self.builder, self.model, sequence, name, profiles,
                                entrypoint_code=self.entry_code)
        argv = [self.tnp_code.co_filename, '--name', name, '--output', str(profiles), '--seq', sequence]
        with route_contexts(observer, self.mm, self.row['policy'], temporary_directory=self.root/'openmm-tmp'):
            try:
                observer.install(self.mm, self.fixer, self.builder, self.refine, self.torch)
                with (directory/'private-stdout.log').open('x', encoding='utf-8') as stdout, \
                     (directory/'private-stderr.log').open('x', encoding='utf-8') as stderr, \
                     original_program_context(argv, directory, bridge,
                         preparation_threads=self.row['policy']['preparation_threads'],
                         trace_path=directory/'trace.jsonl', path=self.settings['environment']+'/bin:'+
                         self.settings['bundle']+'/tools:/usr/bin:/bin'), \
                     redirect_stdout(stdout), redirect_stderr(stderr):
                    try:
                        execute_program(self.tnp_code)
                    except SystemExit as error:
                        require(error.code in (None, 0), 'original_tnp_cli_failed')
            finally:
                try:
                    observer.finish()
                finally:
                    observer.restore()
                require(all(getattr(owner, name) is original for owner, name, original, _ in observer.hooks),
                        'resident_observer_restore_failed')
        raw_trace = (directory/'trace.jsonl').read_bytes()
        trace = [{k:v for k,v in json.loads(line).items() if k != 'atom_ids'}
                 for line in raw_trace.splitlines()]
        reference.validate_trace(trace, sequence_hash=key, expected_observer_sha256=self.row['observer_sha256'],
            expected_cuda_available=True, expected_final_refinement=self.row['policy']['final_refinement'])
        result_text = (profiles/('TNP_Results_SingleSeqEntry_'+name+'.json')).read_text()
        raw = json.loads(result_text)[name]
        metrics = {target:raw[source] for source,target in native.METRICS.items()}
        structure = profiles/'Final_Models'/(name+'_NanoBodyBuilder2_Model.pdb')
        models = [event for event in trace if event.get('event') == 'model_saved']
        geometries = [event for event in trace if event.get('event') == 'geometry_snapshot'
                      and event.get('stage') == 'final_model']
        require(len(models) == len(geometries) == 1, 'missing_final_geometry')
        geometry_name = geometries[0]['pdb_file']
        require(Path(geometry_name).name == geometry_name, 'invalid_geometry_artifact_path')
        binding = {'transformation':'published_tnp_hydrogen_removal_pdbio',
            'raw_model_sha256':models[0]['pdb_sha256'],
            'observed_geometry_pdb':native.pack_artifact((directory/'coordinates'/geometry_name).read_text()),
            'final_pdb':native.pack_artifact(structure.read_text()), 'tnp_result_json':result_text,
            'tnp_result_sha256':sha(result_text.encode())}
        native._namespace_wrappers(Path(self.settings['bundle']))
        from metanano.utils.tnp_wrapper import ProfileExecution
        from metanano.utils.aggrescan3d_wrapper import Aggrescan3DWrapper
        remaining = deadline-time.monotonic(); require(remaining > 1, 'aggregation_window_closed')
        aggregation = Aggrescan3DWrapper(self.settings['bundle']+'/tools/aggrescan3d-rs',
            self.settings['bundle']+'/tools/freesasa').profile(sequence, structure, ProfileExecution(min(60,remaining)))
        structure_sha = native.sha(structure)
        require(aggregation.get('status') == 'ok' and aggregation.get('sequence_hash') == key
            and aggregation.get('input_structure_sha256') == structure_sha, 'aggregation_failed')
        metrics.update(a3d_average=str(aggregation['a3d_average']), a3d_maximum=str(aggregation['a3d_maximum']))
        completed=time.monotonic()
        require(completed<=deadline,'resident_scientific_deadline_exceeded')
        return {'directory':str(directory), 'sequence_hash':key, 'trace':trace,
            'raw_trace_sha256':sha(raw_trace), 'structure_binding':binding,
            'structure_sha256':structure_sha, 'aggregation':aggregation, 'metrics':metrics,
            'elapsed_seconds':completed-started, 'scientific_completed_monotonic':completed,
            'scientific_deadline_monotonic':deadline}

    def finish_batch(self, request, pending, *, batch_id):
        """No authoritative measurement is written before this model-state check."""
        final_state = model_fingerprint(self.model)
        require(final_state == self.initial_state, 'resident_model_state_changed')
        require(pending and len(pending) <= 32, 'invalid_resident_batch_size')
        values = []
        for result in pending:
            key = result['sequence_hash']; binding = {
                'source_sha':request['source_sha'], 'sequence_hash':key,
                'registration_sha256':self.settings['registration_sha256'], 'policy':self.row['policy'],
                'implementation_sha256':self.row['implementation_sha256'], 'draw_id':result['directory'],
                'aggregation_loader':'exact_source_wrappers_without_application_bootstrap',
                'raw_trace_sha256':result['raw_trace_sha256'],
                'structure_binding_sha256':reference.digest(result['structure_binding']),
                'execution':'resident-v1', 'resident_module_sha256':self.row['resident_module_sha256'],
                'model_state_start_sha256':self.initial_state, 'model_state_end_sha256':final_state,
                'resident_batch_id':batch_id, 'resident_batch_draws':len(pending),
                'scientific_completed_monotonic':result['scientific_completed_monotonic'],
                'scientific_deadline_monotonic':result['scientific_deadline_monotonic']}
            metrics = result['metrics']; receipt = {'schema':reference.SCHEMA,
                'source_sha':request['source_sha'], 'sequence_hash':key, 'request':binding,
                'request_sha256':reference.digest(binding), 'reference_contract_sha256':reference.digest(reference.contract()),
                'runtime':self.runtime, 'runtime_identity_sha256':self.row['runtime_sha256'],
                'trace':result['trace'], 'trace_sha256':reference.digest(result['trace']),
                'observation':{'passed':reference.measured_gate(metrics), 'metrics':metrics},
                'actual_validator_worker_attested':False, 'structure_sha256':result['structure_sha256'],
                'structure_binding':result['structure_binding'], 'aggregation':result['aggregation']}
            value = {'schema':native.SCHEMA, 'evidence_backend':native.BACKEND, 'authoritative':True,
                'sequence_hash':key, 'passed':receipt['observation']['passed'],
                **{key:metrics[key] for key in native.METRICS.values()}, 'aggrescan3d':result['aggregation'],
                'native_reference':{'registration_sha256':self.settings['registration_sha256'],
                                    'policy':self.row['policy'], 'receipt':receipt},
                'native_started':True, 'elapsed_seconds':result['elapsed_seconds']}
            require(native.authoritative_measurement(value), 'resident_evidence_validation_failed')
            values.append(value)
        return values


def collect_worker_replies(processes, deadline, *, clock=time.monotonic):
    """Multiplex bounded replies; one hung scientific peer cannot hide the others."""
    replies, buffers = {}, {}
    with selectors.DefaultSelector() as selector:
        for process in processes:
            os.set_blocking(process.stdout.fileno(),False)
            selector.register(process.stdout,selectors.EVENT_READ,process)
            buffers[process]=bytearray()
        while selector.get_map() and clock()<deadline:
            for key,_ in selector.select(timeout=min(.1,max(0,deadline-clock()))):
                process=key.data
                try:data=os.read(key.fd,65536)
                except BlockingIOError:continue
                if not data:
                    selector.unregister(key.fileobj);continue
                buffers[process].extend(data)
                if len(buffers[process])>32*1024*1024:
                    replies[process]={'ok':False,'indices':[],'values':[],'attempted':[]}
                    selector.unregister(key.fileobj)
                elif b'\n' in buffers[process]:
                    line,tail=buffers[process].split(b'\n',1)
                    try:
                        require(not tail.strip(),'multiple_resident_replies')
                        replies[process]=json.loads(line)
                    except (ValueError,AdapterError):
                        replies[process]={'ok':False,'indices':[],'values':[],'attempted':[]}
                    selector.unregister(key.fileobj)
    return replies


def worker_main(request, channel):
    """Private protocol channel; third-party native stdout/stderr cannot enter it."""
    engine = Engine(request)
    channel.write(json.dumps({'ready':True})+'\n'); channel.flush()
    for line in sys.stdin:
        command = json.loads(line)
        if command.get('stop') is True:
            return 0
        pending, completed_indices = [], []
        error_type = None
        try:
            batch = command['request']; tasks = command['indices']
            require(1 <= len(tasks) <= 32 and len(set(tasks)) == len(tasks), 'invalid_worker_batch')
            for index in tasks:
                pending.append(engine.draw(batch, index, deadline=command['deadline']))
                completed_indices.append(index)
        except BaseException as error:
            error_type = type(error).__name__
        # Only the Engine, after lease/deadline admission, durably marks start.
        attempted = list(completed_indices)
        if isinstance(command.get('request'),dict) and 'directory' in command['request']:
            attempted = [i for i in command.get('indices',[]) if i in completed_indices or
                (Path(command['request']['directory'])/(str(i)+'.started.json')).exists()]
        try:
            values = engine.finish_batch(batch, pending, batch_id=command['batch_id']) if pending else []
            # Measurement files are committed by the coordinator only AFTER the
            # shared sealed-file postcheck has succeeded too.
            reply = {'ok':error_type is None, 'indices':completed_indices, 'values':values,
                     'attempted':attempted, 'error_type':error_type}
        except BaseException as error:
            reply = {'ok':False, 'indices':[], 'values':[], 'attempted':attempted,
                     'error_type':type(error).__name__}
        channel.write(json.dumps(reply, allow_nan=False)+'\n'); channel.flush()
        if not reply['ok']:
            return 1  # An uncertain / poisoned model never accepts another draw.
    return 0


def serve(request):
    """One normal lease owns this pool across many bounded request groups."""
    from nanobody_opt.pr100_live_gpu import inherited_lease_fd
    fd = inherited_lease_fd([request['gpu']]); require(fd is not None, 'missing_gpu_lease')
    settings = request['settings']; row = native.registration(settings['registration_sha256'])
    native.verify_files(settings, row, full=True)
    root = Path(request['directory']); workers = []; deadline = request['deadline']
    count = settings['workers_per_gpu']; require(1 <= count <= 8, 'unbounded_resident_pool')
    def stop():
        for process in workers:
            if process.poll() is None:
                try:process.stdin.write('{"stop":true}\n');process.stdin.flush()
                except (BrokenPipeError, OSError):pass
        end=time.monotonic()+2
        for process in workers:
            try:process.wait(timeout=max(.01,end-time.monotonic()))
            except subprocess.TimeoutExpired:pass
        for process in workers:
            if process.poll() is None:process.terminate()
        end=time.monotonic()+3
        for process in workers:
            try:process.wait(timeout=max(.01,end-time.monotonic()))
            except subprocess.TimeoutExpired:pass  # Existing lease owner proves final cleanup.
    try:
        for index in range(count):
            directory = root/('worker-'+str(index));directory.mkdir()
            path = directory/'worker.json'
            native.atomic(path, {**request, 'directory':str(directory)})
            workers.append(subprocess.Popen([sys.executable,'-s','-B','-m',MODULE,'--worker',str(path)],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
                text=True,bufsize=1,pass_fds=(fd,)))
        for process in workers:
            require(json.loads(process.stdout.readline()).get('ready') is True, 'resident_startup_failed')
        native.atomic(root/'READY.json', {'ready':True, 'worker_pids':[p.pid for p in workers]})
        while time.monotonic() < deadline and not (root/'STOP').exists():
            commands = sorted((root/'commands').glob('*.json'))
            pending_commands = [p for p in commands if not (root/'responses'/p.name).exists()]
            if not pending_commands:
                time.sleep(.025);continue
            path = pending_commands[0]; batch = json.loads(path.read_text())
            require(set(batch) == {'request','deadline','batch_id'}, 'invalid_resident_command')
            value = batch['request']; sequences = value['sequences']
            require(1 <= len(sequences) <= settings.get('resident_batch_draws',32) <= 32,
                    'unbounded_resident_batch')
            require(value['settings'] == settings and value['gpu'] == request['gpu']
                and value['source_sha'] == request['source_sha']
                and Path(value['directory']).resolve().is_relative_to(root.resolve()),
                'resident_batch_binding_changed')
            require(re.fullmatch('[0-9a-f]{32}', batch['batch_id']) is not None, 'invalid_batch_identity')
            allocation = [(p, list(range(i,len(sequences),count))) for i,p in enumerate(workers)]
            allocation = [(p,indices) for p,indices in allocation if indices]
            work_deadline=min(deadline,batch['deadline']-10)
            for process, indices in allocation:
                require(process.poll() is None, 'resident_worker_not_alive')
                process.stdin.write(json.dumps({**batch,'deadline':work_deadline,'indices':indices})+'\n');process.stdin.flush()
            replies = []; retire = False
            received=collect_worker_replies([p for p,_ in allocation],work_deadline)
            for process, indices in allocation:
                reply = received.get(process)
                if reply is None:
                    reply={'ok':False,'indices':[],'values':[],'attempted':[]}
                require(isinstance(reply.get('ok'),bool), 'resident_batch_failed')
                retire = retire or not reply['ok']
                replies.append(reply)
            if retire:
                native.atomic(root/'RETIRING.json',{'retiring':True})
                stop()
            # After bounded owned retirement, markers distinguish never-started
            # queued work from actual attempts. Unreaped uncertainty is held.
            attempted={i for process,indices in allocation for i in indices if
                (Path(value['directory'])/(str(i)+'.started.json')).exists() or
                (process not in received and process.poll() is None)}
            attempted.update(i for reply in replies for i in reply['attempted'])
            native.verify_files(settings, row)  # Shared disk weights/source once per batch.
            require(inherited_lease_fd([request['gpu']]) is not None,'resident_lease_lost')
            results = [{'inconclusive':True, 'native_started':index in attempted,
                        'sequence_hash':hashlib.sha256(sequence.encode()).hexdigest()}
                       for index,sequence in enumerate(sequences)]
            for reply in replies:
                if len(reply['indices'])!=len(reply['values']):
                    retire=True;continue
                for index, measured in zip(reply['indices'],reply['values']):
                    try:
                        completion=measured['native_reference']['receipt']['request']
                        completed=completion['scientific_completed_monotonic']
                        cutoff=completion['scientific_deadline_monotonic']
                        valid=(type(completed) in (int,float) and type(cutoff) in (int,float)
                            and math.isfinite(completed) and math.isfinite(cutoff)
                            and completed<=cutoff<=work_deadline
                            and measured['sequence_hash']==hashlib.sha256(sequences[index].encode()).hexdigest()
                            and native.authoritative_measurement(measured))
                    except (TypeError,KeyError,ValueError):valid=False
                    if not valid:
                        retire=True
                        continue  # A late/malformed peer cannot erase checked earlier results.
                    results[index] = measured
            for index, measured in enumerate(results):
                if not measured.get('inconclusive'):
                    native.atomic(Path(value['directory'])/str(index)/'measurement.json',measured)
            if retire:native.atomic(root/'RETIRING.json',{'retiring':True})
            native.atomic(root/'responses'/path.name, {'ok':True, 'values':results})
            if retire:return 0
        return 0
    finally:
        stop()  # The enclosing existing lease owner still verifies all descendants.


class RegisteredResidentAdapter(native.Adapter):
    """Keep model workers under the same lease across native refill groups."""
    def __init__(self, job, directory, **kwargs):
        began = time.monotonic()
        super().__init__(job,directory,**kwargs)
        require(self.settings.get('resident') is True, 'resident_execution_disabled')
        seconds = job.get('phase_budget_seconds',1800)
        require(type(seconds) in (int,float) and math.isfinite(seconds) and seconds > 0,
                'invalid_resident_phase_budget')
        self.phase_deadline = began+min(float(seconds),1800)
        self.sessions = {}

    @property
    def capacity(self):
        # Queue slots are bounded separately from actual model processes. This
        # lets a tranche amortize startup/end checks over multiple fresh draws.
        return len(self.settings['gpus'])*self.settings.get('resident_batch_draws',32)

    def _batch_limit(self,state):
        return min(self.settings.get('resident_batch_draws',32),
                   max(1,math.ceil(len(state['queue'])/max(1,state['waiting_lanes']))))

    def _clean(self,session):
        receipt = json.loads((session['root']/'lease.json').read_text())
        cleanup = receipt.get('cleanup',{})
        never_started = (receipt.get('status') == 'unavailable' and 'child_pid' not in receipt
                         and 'cleanup' not in receipt and receipt.get('released') is True)
        require(never_started or (receipt.get('released') is True
            and cleanup.get('remaining_descendants') == 0 and cleanup.get('fixed_point_reached') is True),
            'unverified_reference_cleanup')
        return never_started

    def _stop_session(self,session):
        native.atomic(session['root']/'STOP',{'stop':True})
        process=session['process']
        if process.poll() is None:
            try:process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.terminate();process.wait(timeout=12)
        self._clean(session)

    def _completed_files(self,directory,sequences):
        """Only after fixed-point owner cleanup: retain commits / prove no-start."""
        values=[]
        for index,sequence in enumerate(sequences):
            path=directory/str(index)/'measurement.json'
            if path.is_file():
                value=json.loads(path.read_text())
                require(value.get('sequence_hash')==hashlib.sha256(sequence.encode()).hexdigest()
                    and native.authoritative_measurement(value),'invalid_registered_measurement')
            else:
                value=native.failure('reference_profile_inconclusive',
                    started=(directory/(str(index)+'.started.json')).exists())
            values.append(value)
        return values

    def _start(self,gpu):
        old = self.sessions.get(gpu)
        if old is not None:
            if old['process'].poll() is None:return old
            self._clean(old)
        root = self.directory/('resident-'+uuid.uuid4().hex);root.mkdir()
        (root/'commands').mkdir();(root/'responses').mkdir()
        timeout = min(1800,self.phase_deadline-time.monotonic())
        require(timeout >= 10,'reference_admission_window_closed')
        request = {'settings':self.settings,'source_sha':self.source_sha,'gpu':gpu,
                   'directory':str(root),'deadline':time.monotonic()+timeout-5}
        native.atomic(root/'session.json',request)
        cpu_refinement = self.settings['registration']['policy']['final_refinement']['platform'] == 'CPU'
        plugin_libraries = [] if cpu_refinement else native.cuda_plugin_paths(
            self.settings,self.settings['registration'])['library_directories']
        bundle,environment = self.settings['bundle'],self.settings['environment']
        # Four measured resident workers added ~6.9 GiB. Reserve two GiB per
        # model process plus two GiB for contexts/overlap, never the cold 2 GiB.
        lease = {'gpu':gpu,'ttl_seconds':timeout,
            # CPU refinement still keeps CUDA prediction models resident. Four
            # workers reserve20GiB including conservative model/cache headroom.
            'minimum_free_mib':(4096 if cpu_refinement else 2048)*(self.settings['workers_per_gpu']+1),
            'lock_directory':self.settings['lock_directory'],'scorer':self.settings['scorer'],
            'receipt':str(root/'lease.json'),'cwd':str(root),
            # Use the same sealed interpreter directly with baseline RPATHs.
            # Only nonempty approved sidecar library directories may override
            # loader search; prefixing environment/lib can change Torch NVRTC.
            'command':[environment+'/bin/python3.12','-s','-B','-m',MODULE,'--serve',str(root/'session.json')],
            'env':{'PATH':environment+'/bin:'+bundle+'/tools:/usr/bin:/bin',
                'PYTHONPATH':str(ROOT),'PYTHONDONTWRITEBYTECODE':'1','HOME':'/root','LANG':'C.UTF-8',
                'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
                'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}}
        if plugin_libraries:
            lease['env']['LD_LIBRARY_PATH']=':'.join(plugin_libraries)
        native.atomic(root/'lease-job.json',lease)
        process = subprocess.Popen([sys.executable,'-B','-m','nanobody_opt.pr100_live_gpu',
            '--job',str(root/'lease-job.json')],cwd=ROOT,
            env={'PATH':'/usr/bin:/bin','PYTHONPATH':str(ROOT),'PYTHONDONTWRITEBYTECODE':'1'},
            stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        session = {'root':root,'process':process};self.sessions[gpu]=session;return session

    async def _launch(self,gpu,sequences,timeout):
        deadline = min(time.monotonic()+timeout,self.phase_deadline)
        prior = self.sessions.get(gpu)
        while prior is not None and (prior['root']/'RETIRING.json').exists() and prior['process'].poll() is None:
            require(time.monotonic()<deadline,'resident_retirement_window_closed')
            await asyncio.sleep(.025)
        session = self._start(gpu);root = session['root'];process = session['process']
        while not (root/'READY.json').exists():
            if process.poll() is not None:
                if self._clean(session):return [native.failure('reference_gpu_unavailable') for _ in sequences]
                return [native.failure('reference_profile_inconclusive',started=False) for _ in sequences]
            if time.monotonic()>=deadline:
                await asyncio.to_thread(self._stop_session,session)
                return [native.failure('reference_profile_inconclusive',started=False) for _ in sequences]
            await asyncio.sleep(.025)
        key = uuid.uuid4().hex;directory = root/('batch-'+key);directory.mkdir()
        request = {'settings':self.settings,'source_sha':self.source_sha,'sequences':sequences,
                   'gpu':gpu,'directory':str(directory),'timeout_seconds':max(1,deadline-time.monotonic())}
        native.atomic(root/'commands'/(key+'.json'),{'request':request,'deadline':deadline,'batch_id':key})
        response = root/'responses'/(key+'.json')
        delivery_deadline=min(self.phase_deadline,deadline+10)
        while not response.exists():
            if process.poll() is not None:
                self._clean(session)
                return self._completed_files(directory,sequences)
            # Scientific completion is independently bound before its cutoff.
            # A bounded cleanup/readback reserve may deliver already checked
            # peers afterwards; it never permits another prediction to start.
            if time.monotonic()>=delivery_deadline:
                await asyncio.to_thread(self._stop_session,session)
                return self._completed_files(directory,sequences)
            await asyncio.sleep(.025)
        value = json.loads(response.read_text());require(value.get('ok') is True,'resident_batch_failed')
        if process.poll() is not None:self._clean(session)
        values = value['values'];require(len(values)==len(sequences),'incomplete_resident_batch')
        output = []
        for sequence,measured in zip(sequences,values):
            require(measured['sequence_hash']==hashlib.sha256(sequence.encode()).hexdigest(),
                    'invalid_registered_measurement')
            if measured.get('inconclusive') is True:
                require(set(measured)=={'inconclusive','native_started','sequence_hash'}
                    and type(measured['native_started']) is bool,'invalid_resident_inconclusive')
                output.append(native.failure('reference_profile_inconclusive',started=measured['native_started']))
            else:
                require(native.authoritative_measurement(measured),'invalid_registered_measurement')
                output.append(measured)
        return output

    def shutdown(self):
        """Synchronous phase boundary, also usable after a private asyncio loop."""
        error = None
        for session in self.sessions.values():
            try:
                self._stop_session(session)
            except BaseException as failure:
                error = failure  # Still clean every other owned session.
        if error is not None:raise error

    async def close(self):
        try:await super().close()
        finally:await asyncio.to_thread(self.shutdown)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--serve',type=Path);group.add_argument('--worker',type=Path)
    args=parser.parse_args()
    # Duplicate protocol stdout before muting fd1/2 as well as Python streams.
    # Never expose third-party exceptions, logs, environment or credentials.
    channel=os.fdopen(os.dup(1),'w') if args.worker else None
    with open(os.devnull,'w') as sink:
        os.dup2(sink.fileno(),1);os.dup2(sink.fileno(),2)
        try:
            request=json.loads((args.worker or args.serve).read_text())
            return worker_main(request,channel) if args.worker else serve(request)
        except BaseException:
            if channel is not None:
                channel.write('{"ok":false,"error_type":"ResidentExecutionFailed"}\n');channel.flush()
            return 1
if __name__=='__main__':raise SystemExit(main())
