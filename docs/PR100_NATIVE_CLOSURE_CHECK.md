# N0: finite CPU import and runtime identity check

Executable protocol revision 7, not execution permission. Review before staging or running
anything remotely. No model construction, prediction, refinement, native
profile, target score, GPU context, download, production cache or wallet.
The sole constructor exception is one exact native import-time XML parameter
parser described below; System/Context/Simulation/Integrator construction and
physics execution remain forbidden.

Run one private child through Pi on master, under the inventoried native Python.
Cap the complete check at 120 seconds, two affinity CPU slots and 4 GiB sampled
RSS abort threshold (not a kernel-enforced ceiling), zero GPUs. Require the
same 96 GiB starting/64 GiB abort headroom as N1. First run the separately bounded 10-second detached-grandchild
containment rehearsal as a dedicated session using interpreter `-S` (no site
startup hooks). Its children also use `-S`. Stop on a cap or unresolved required identity; no retry
or package installation. Record actual CPU, wall time and zero descendants.

The executable is `scripts/pr100_runtime_preflight.py`. Its `assemble` action
merges only explicitly provided, previously captured public inventory files and
the exact CI `build_stamp.json` into a candidate manifest. It never marks its own
closure reviewed. The `check` action runs as a dedicated Linux session through
Pi, with interpreter `-B` and the native registered interpreter. Both controller
and import child refuse bytecode-enabled startup. The import child has guards
against model/physics construction, native metric entry points, CUDA discovery,
unregistered subprocess launch and network connection. A guard hit exits that child and makes
the receipt NO_GO; native logs and exception text are never returned.

The 120-second N0 cap is a preregistration change before any runtime measurement;
CPU slots, sampled memory threshold, scientific permissions and the N1 1200-second
cap are unchanged. N0 reserves measured pre-verification time for its post-check
plus five seconds for cleanup/receipt handling. The profile hook checks module
and function first; it inspects frame locals only at relevant guard sites.
Import, first loaded-origin audit, and both verification timings are separate.
The post-check gets at most the measured pre-check time plus one second within
the total cap. Backend load can make hashing slower and conservatively produce
NO_GO; that allowance must not be relaxed after observing a measurement.

Stage the reviewed candidate commit into a new directory through Pi and verify
the committed source manifest before and after. It must include research code,
protocols, frozen population, config and helpers. Stage pinned filter commit
`fe2d5868126406a0de8953d26ecd8bdccbdbd029` and verify its entire source manifest.
Confirm imported metanano resolves inside this stage. Current qualification
directory source cannot stand in for candidate source. No writes to the current
qualification directory or production release.

Before launch inspect the declared import modules for top-level model creation.
If present, stop. Import only candidate adapters/rules, pinned metanano
config/filter/wrappers, ImmuneBuilder's NanoBodyBuilder2 module, theraprofnano's
three metric modules, torch and OpenMM app/refinement dependencies. Import
classes without constructing models, simulations, integrators or contexts. Do
not call CUDA availability, device count, NVML or GPU discovery.

Revision 2 was stopped by source inspection before any N0 execution: native
`ImmuneBuilder/refine.py:26` constructs `ForceField("amber14/protein.ff14SB.xml")`.
Its constructor initializes Python parameter tables and calls `loadFile`; the
four relevant parsers register parameter generators. The fixed XML contains no
Include, Script or InitializationScript. Actual System construction is in the
separate `createSystem` method, which remains forbidden. No model weights are
loaded by this exception.

Revision 3 permits exactly one `ForceField.__init__`, only from module scope at
that exact caller line, using that one argument. It binds these observed files:

| File under native site-packages | SHA-256 |
| --- | --- |
| `ImmuneBuilder/refine.py` | `e33d4bdc3e8354296ede6bde0e2fb438368e5e2fbe87ec0d7b387be0d60ebaf0` |
| `openmm/app/forcefield.py` | `7c909edd1bbb8c48ac0070e5ea19b6bf0817be2aeccfbd23e7ebb1302c0da432` |
| `openmm/app/data/amber14/protein.ff14SB.xml` | `d9f9779c09d67cd5f8bc657692f174ffab14c469dfd06d560ac1899fa7e976b8` |

The import guard requires exactly one read-only open of that absolute XML during
the constructor, no subsequent loadFile call, no duplicate constructor, and no
other caller/argument/source. XML includes or scripts fail preflight. Pinned
`forcefield.py` lines 249–254 first check `isfile` and resolve the data path;
line 256 performs the only `etree.parse`. Review 4's suggested relative parse
followed by a fallback describes a different source version. Requiring a relative
open would incorrectly reject this pinned installation. A planted relative XML
would still fail the guard before it could be read. Execution
of package entrypoint callbacks is forbidden (the inspected installation has
no `openmm.forcefielddir` entrypoint group). No native source is patched. The
receipt reports observed allowed-constructor/XML counts separately from the
zero model/profile/GPU permissions. This exception is N0 only, not a change to
N1 physics or a qualification claim.

The failed d766 N0 and a separate diagnostic observed urllib3's import-time IPv6
capability check. Revision 5 allows exactly one `socket.bind` from the pinned
`urllib3.util.connection._has_ipv6` line 127, invoked at module scope line 137,
while importing the declared ImmuneBuilder module. The source SHA is
`2633bbdb69731e5ccb5cf4e4afd65605d86c7979cc5633126f50c92d5ad74a74`.
The actual socket must be AF_INET6/SOCK_STREAM/protocol 0 with address `('::1', 0)`;
retain its reference and require closed `fileno()` after imports. Count one bind.
No connection, other bind or external networking exception exists.
This permits a loopback-only capability probe, not blanket zero networking.
Both failed d766 receipts remain evidence; neither qualifies a runtime.

Revision 6 additionally permits one faithful read-only `lscpu` invocation from
NumPy's installed `check_support_sve`, reached through SciPy's normal array API
imports. The utils.py SHA is
`dc5ad24cc8b439da43383043a27720883433768e4d280e9855f437b8a4524277`;
caller line 1247 and parent module line 1253 are bound. Require the literal
command, capture_output=True/text=True, shell=False, no custom env/stdin/cwd or
executable, one Popen audit event and a successful CompletedProcess plus reaped
Popen. No output is returned in receipts. PATH must resolve `/usr/bin/lscpu` with
SHA `250c7ec85ceec9f50101f555704b79ed3a8a917efa14fa455d85136f27182daa`.
Its explicit public dependency inventory binds libsmartcols, libc and the loader,
including resolved symlinks. The normal imports and native fallback behavior are
unchanged; no OSError substitution or probe emulation is used. The process stays
inside the same owned tree, 120 seconds, two CPU slots and 4 GiB sampled abort.
Its address space is not separately observed. Every other subprocess remains
forbidden. Both prior failed N0 receipts remain preserved.

The inventory now includes the 14 required NumPy testing source/bytecode files.
Their directory name did not make them optional: SciPy imports them at runtime.
This is an explicit closure correction, not scientific execution evidence.

Revision 7 freezes the 164 public source/bytecode/library/data files actually
reported missing by the prior N0 audit (104,672,343 bytes). No wildcard trust or
special mapping exemption is added. All original runtime hashes stay fixed.

The same audit exposed two PyTorch virtual modules. Native `_Ops` and `_Classes`
are ModuleType subclasses with literal `__file__` labels `_ops.py` and `_classes.py`;
resolving those labels against the private CWD invents nonexistent paths. Register
only `torch.ops` and `torch.classes`, their exact native class object identity,
class module/name, literal label, module name and absent import spec, plus the
absolute defining module origin and source hash. Record this interpretation in
`registered_virtual_module_origins`; hash the defining source and existing
bytecode normally. Any other relative origin, impostor class, changed label,
source origin/digest or virtual module remains a failure. This changes provenance
interpretation only, not native code or scientific behavior.

The interpreter and other entrypoints come from a separate readonly resolved-path
inventory. The two Aggrescan/FreeSASA binaries may reside only under the explicitly
declared external tool root
`/root/pr100-qualification-20261026/NOVA-nanobody-filter/.tools/bin`; their contents
remain hash-pinned. That exception supplies binaries, never candidate Python
source. Imported `metanano` and `nanobody_opt` modules/namespaces must resolve to
the staged candidate; all those origins are recorded. Filter activation already
checked the top-level metanano origin before this additional recorded audit.

Native TNP requires a regular `scripts` package containing region definitions.
Research helpers are therefore loaded from exact staged file paths under unique
module names; they never replace that scientific package. The two scientific
scripts sources and their existing bytecodes are added through a separate
explicit readonly inventory, not a broad environment scan.

The child gets only registered non-sensitive runtime environment settings.
Set CUDA/HIP/ROCR visibility empty and registered CPU thread settings. Disable
bytecode writes before interpreter startup; existing bytecode can still be read.
Put Matplotlib/XDG/Torch caches under the private directory. Capture both output
streams privately; never print native logs, exception messages or environment.

Freeze Python, four checkpoint files, executable resolution, TNP/NanoBodyBuilder2
CLI, HMMER and its databases, native reference files, FreeSASA/Aggrescan and
logging shim. Record native ANARCI 1.3/AbNumber 0.2.7 separately from the pinned
main-validator ANARCI 2026.2.13.2/AbNumber 0.4.4 annotation closure.

Retain this finite evidence:

1. Both existing package/model/system and stdlib source/extension inventory
   artifact hashes; staged source manifests; exact child configuration.
2. Loaded module names and origins after the declared imports. Resolve/hash
   only registered staged source, shim, package/stdlib and system-library roots.
   Record existing bytecode digests alongside source when potentially used.
   An unregistered origin blocks the check; never inspect arbitrary application
   state to explain it or silently extend trusted roots.
3. Absolute file-backed library paths from `/proc/self/maps`; discard addresses
   and offsets. Permissions are used only to reject executable special mappings.
   Hash only registered roots, compare with the frozen
   inventory, and retain every newly discovered required dependency.
   N0 may report a literal unresolved path only within the fixed staged source,
   native library, logging shim and system-library/locale roots; credential-shaped
   names and all other locations remain hashed. Unknown files are never read to
   diagnose them. Counts classify deleted/memfd/shared-memory/private-temp/locale
   mappings without granting exemptions. A diagnostic does not imply acceptance.
   The initial `special_mapping_policies` list is empty. Only an exact reviewed
   path plus exact permissions tagged `non_executable_data` can later exempt a
   data mapping from hashing; executable mappings cannot receive an exemption.
   Unrecognized/deleted code is never permitted. Anonymous and kernel bracket
   mappings are outside this file-backed audit, explicitly not a proof of all
   machine code. A required unknown mapping stops N0 for a new frozen inventory.
4. `Platform.getPlatformByName('CPU').getName()` and property names. Require CPU
   and Threads support without constructing a Context. Pin refinement source
   with its positive-`n_threads` CPU/Threads branch and the profiler's argument
   one. This proves availability/selected source path, not executed refinement.
5. Plugin failure categories without raw text. Recognize only inventoried HIP
   plugin basenames and missing `libhiprtc.so.6`/`libamdhip64.so.6`. CPU plugin
   failure or unclassified required-library failure blocks the check. HIP
   failures may be classified unused only for this CPU-only scope, with CPU
   available and the registered refinement path preserved. They are not a
   healthy general GPU installation.
6. Post-import required-file/source hashes and resource/cleanup receipt. Drift,
   missing models/databases/binaries, unbound origin, visible GPU, prohibited
   constructor call, cap breach or remaining descendant yields NO_GO.

Record bytes and elapsed time for both complete runtime/source verifications,
and elapsed time for the first loaded-origin audit. N1 retains the prior
1080-second profiling projection and charges both verifications, four first
audits conservatively in serial, and ten seconds cleanup. N1 refuses to start
unless `1080 + 2*max(N0 verification times) + 4*N0 first audit + 10 <= 1200`.
This is a feasibility projection, not a guaranteed execution time. N1 reserves
the measured post-verification time before its profile deadline; all hash loops
and release verification receive deadlines. A cap still yields incomplete NO_GO.

N0 emits a receipt and an unreviewed `runtime-candidate.json` bound to its receipt
SHA, harness SHA, source release and complete file inventories. N1 requires a
separately recorded named closure reviewer and receipt digest. Identity audits
are labelled `worker_address_space_only`; N0 does not observe TNP or builder CLI
child maps. CLI closure evidence consists of the frozen files, this import
check, and asserted identical PATH/LD_LIBRARY_PATH inheritance. Refinement and
all scientific parameters remain unchanged.

N1 re-hashes the actual N0 receipt and checks its complete/clean status, zero
unresolved origins, zero descendants, CPU/Threads and registered constructor
evidence. Manifest timing values must equal the receipt-derived values. Removing
only the review/receipt fields must recover the exact input manifest digest in
the receipt, so an approval cannot silently alter its runtime configuration.
Only `closure_reviewed` and `closure_reviewer` may be changed to approve the
generated runtime candidate. The receipt path remains its actual private artifact.

Record delayed imports/data/libraries as an explicit completeness limitation.
If these cannot be bounded by prehashed source/data/library manifests, freeze
the missing dependency before N1; do not discover it during measurement and
retroactively declare identity complete. N0 is scoped preflight evidence, not
permission to reuse old caches or proof of full validator acceptance. Production
still needs loaded-worker evidence under its actual pinned runtime.

Revision 5 N1 verification also binds the outer supervisor and containment receipt
bytes, requires zero exit/timeout/descendants, fixed-point cleanup and the exact
declared import list. A complete child alone cannot authorize the experiment.
