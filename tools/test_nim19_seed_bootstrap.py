import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import nim19_pair_setup as setup
import nim19_paired_benchmark as benchmark


class SeedBootstrapTests(unittest.TestCase):
    def test_minimal_source_tree_creates_database_parent(self):
        expected='a84bd5216d38975b33670a137aaa20f2b3c2a29bc363db28d44f653d8ff2c617'
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder)
            repo=base/'oss-repo'
            with patch.object(setup,'BASE',base),patch.object(setup,'REPO',repo),patch.object(setup,'STATUS',base/'pair-setup.json'),patch.object(setup.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout=b'',stderr=b'')),patch.object(setup.urllib.request,'urlopen',return_value=io.BytesIO(b'scientific-database-fixture')),patch.object(setup.hashlib,'sha256',return_value=SimpleNamespace(hexdigest=lambda:expected)):
                self.assertEqual(setup.main(),0)
                self.assertEqual((repo/'combinatorial_db/molecules.sqlite').read_bytes(),b'scientific-database-fixture')

    def test_seed_default_preserved(self):
        with patch.object(benchmark.sys,'argv',['benchmark','reference']),patch.object(benchmark,'oss') as run:
            benchmark.main();run.assert_called_once_with('reference',0,68,False)

    def test_distinct_seed_artifact_option(self):
        with patch.object(benchmark.sys,'argv',['benchmark','reference','--seed','69','--seed-artifact']),patch.object(benchmark,'oss') as run:
            benchmark.main();run.assert_called_once_with('reference',0,69,True)


if __name__=='__main__':unittest.main()
