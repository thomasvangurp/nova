"""Every module the vendored fast scorer imports must be vendored with it.

The scorer lived only on a rented box for four days before being committed, and
the commit that brought it under version control dropped two of its own modules:
resident.py (ResidentStage -- the resident-model loading that is most of the 28x
speedup) and synth_fullatom.py. Nothing noticed until a deploy, where each shard
died on "No module named 'resident'" after the producer had spent an epoch's
generation budget.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
import pytest

VENDOR = Path(__file__).resolve().parents[1] / "external_tools" / "fast_nanobody_scorer"


def _local_imports(path: Path) -> set[str]:
    """Top-level module names imported by one file."""
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_the_scorers_sibling_modules_are_all_present():
    """Any import that resolves to a sibling .py must actually be here."""
    present = {p.stem for p in VENDOR.glob("*.py")}
    missing: dict[str, set[str]] = {}
    for source in VENDOR.glob("*.py"):
        # A name is a sibling module if the pinned package ships it; we can only
        # check the ones we have, so assert closure over what is present.
        wanted = _local_imports(source)
        gap = {n for n in wanted if n in {"resident", "synth_fullatom"} } - present
        if gap:
            missing[source.name] = gap
    assert not missing, f"vendored scorer imports modules it does not ship: {missing}"


def test_resident_and_synth_fullatom_are_vendored():
    """Named explicitly: these are the two the first vendoring commit dropped."""
    assert (VENDOR / "resident.py").is_file()
    assert (VENDOR / "synth_fullatom.py").is_file()
    assert "ResidentStage" in (VENDOR / "resident.py").read_text()


def test_the_scorer_still_matches_its_pinned_digest():
    """The adapter refuses a scorer whose sha256 differs from the deployment's
    pin, so adding files beside it must not change the scorer itself."""
    import hashlib

    digest = hashlib.sha256((VENDOR / "fast_score_batch.py").read_bytes()).hexdigest()
    import yaml
    playbook = yaml.safe_load((VENDOR.parents[1] / 'ansible/nanobody.yml').read_text())
    pins = [p['vars']['nanobody_fast_scorer_sha256'].strip() for p in playbook
            if 'nanobody_fast_scorer_sha256' in p.get('vars', {})]
    assert pins == [digest]


@pytest.mark.parametrize('allow_replicates', [False, True])
def test_non_synthetic_chunks_rebuild_inputs_after_directory_cleanup(tmp_path, allow_replicates):
    """Execute the actual worker's chunk setup twice, without loading a GPU."""
    import shutil
    import types
    tree = ast.parse((VENDOR / "fast_score_batch.py").read_text())
    chunk = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "run_chunk")
    setup = []
    for statement in chunk.body:
        setup.append(statement)
        if isinstance(statement, ast.If) and ast.unparse(statement.test) == "'cfg_dir' not in cached":
            # Configuration/model creation is external to input regeneration.
            statement.body = ast.parse("cached['cfg_dir'] = 'configured'").body
            break
    else:
        raise AssertionError("worker setup contract changed")
    chunk.body = setup
    tree = ast.fix_missing_locations(ast.Module(body=[chunk], type_ignores=[]))
    class Wrapper:
        def _deduplicate_nanobodies(self, data, replicate_indices=None):
            return data[0]['sequences']
        def _write_yaml_files(self):
            for sequence in self.unique_sequences:
                (Path(self.input_dir) / (sequence + '.yaml')).write_text(sequence)
    namespace = dict(Path=Path, shutil=shutil, bw=types.SimpleNamespace(BoltzgenWrapper=Wrapper),
                     cached={}, synth=None, subnet_config={}, allow_replicates=allow_replicates)
    exec(compile(tree, '<chunk-setup-test>', 'exec'), namespace)
    for sequence in ['FIRST', 'SECOND', 'THIRD']:
        namespace['run_chunk'](tmp_path / 'chunk', [sequence], [0])
        assert [p.read_text() for p in (tmp_path / 'chunk/inputs').glob('*.yaml')] == [sequence]


def test_replica_ids_are_global_across_worker_shards_and_default_dedup_survives():
    import hashlib
    import types
    source = VENDOR / 'boltzgen_src/boltzgen/boltzgen_wrapper.py'
    tree = ast.parse(source.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == '_deduplicate_nanobodies')
    namespace = {'seq_hash': lambda s: hashlib.sha256(s.encode()).hexdigest(),
                 'bt': types.SimpleNamespace(logging=types.SimpleNamespace(debug=lambda *a: None))}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
    call = namespace[method.name]
    wrapper = types.SimpleNamespace(allow_replicates=True)
    first = call(wrapper, {0: {'sequences': ['SAME', 'SAME']}}, replicate_indices=[0, 2])
    second = call(wrapper, {0: {'sequences': ['SAME', 'SAME']}}, replicate_indices=[1, 3])
    ids = [i for group in (first, second) for _, i in group['SAME']]
    assert len(set(ids)) == 4
    assert set(ids) == {hashlib.sha256(b'SAME').hexdigest()+str(i) for i in range(4)}
    legacy = call(wrapper, {0: {'sequences': ['SAME', 'SAME']}}, allow_replicates=False)
    assert len(set(i for _, i in legacy['SAME'])) == 1
    with pytest.raises(ValueError, match='unique index'):
        call(wrapper, {0: {'sequences': ['SAME', 'SAME']}}, replicate_indices=[0, 0])


def test_release_contains_the_scorer_dependency_closure():
    import json
    stamp = VENDOR.parents[1] / 'build_stamp.json'
    required = {'fast_score_batch.py', 'resident.py', 'synth_fullatom.py',
                'release_python.sh', 'boltzgen_src/boltzgen/boltzgen_wrapper.py'}
    assert all((VENDOR / p).is_file() for p in required)
    if stamp.is_file():
        assert {'external_tools/fast_nanobody_scorer/' + p for p in required} <= set(
            json.loads(stamp.read_text())['files'])


def test_independent_jobs_do_not_repeat_the_same_replica_seed_namespace():
    import hashlib
    tree = ast.parse((VENDOR/'fast_score_batch.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'replicate_seed')
    namespace = {'hashlib': hashlib}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<seed-test>', 'exec'), namespace)
    seed = namespace['replicate_seed']
    first = [seed('observation-a', i) for i in range(8000)]
    second = [seed('observation-b', i) for i in range(8000)]
    assert len(set(first)) == 8000
    assert set(first).isdisjoint(second)
    assert all(0 <= s < 2**32 for s in first+second)
    assert first == [seed('observation-a', i) for i in range(8000)]
