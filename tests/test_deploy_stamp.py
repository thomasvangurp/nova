"""Bug 6: the staged artifact must be the revision/dependency lock CI tested."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

from nanobody_opt_v2 import run_live_producer as producer
from sn68_source import ROOT, import_shipping

EXPECTED_INVOKED = (
    'nanobody_opt_v2.select_elite_parents',
    'nanobody_opt_v2.sequential_exact_race',
    'nanobody_opt.local_neighborhood',
    'nanobody_opt.check_validator_novelty',
    'nanobody_opt.filter_public_neighborhood',
    'nanobody_opt.fetch_hf_archive',
)


def test_deploy_sha_stamp():
    from scripts.release_manifest import verify
    artifact_root = Path(os.environ['SN68_DEPLOY_ARTIFACT_ROOT'])
    stamp = json.loads((artifact_root / 'build_stamp.json').read_text())
    verify(artifact_root, stamp, expected_sha=os.environ['SN68_TESTED_SHA'], require_tests=False)
    assert artifact_root.resolve() == ROOT.resolve()


def test_shipping_invocations_resolve_to_tested_tree(monkeypatch):
    """Bug 6: actual producer invocations and imports must target shipping trees."""
    tree = ast.parse(Path(producer.__file__).read_text())
    invoked = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.List):
            for left, right in zip(node.elts, node.elts[1:]):
                if isinstance(left, ast.Constant) and left.value == '-m' and isinstance(right, ast.Constant):
                    invoked.add(right.value)
    assert set(EXPECTED_INVOKED) <= invoked
    assert not {'nanobody_opt.select_elite_parents', 'nanobody_opt.sequential_exact_race'} & invoked
    for name in ('nanobody_opt_v2.run_live_producer', *EXPECTED_INVOKED):
        spec = importlib.util.find_spec(name)
        assert Path(spec.origin).resolve() == ROOT / (name.replace('.', '/') + '.py')
        # Real CPU imports; never run entrypoints or import fd-redirecting
        # validator_parent_gate. No legacy producer/selector/race substituted.
        import_shipping(name, monkeypatch)
