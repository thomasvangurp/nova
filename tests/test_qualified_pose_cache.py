import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from tools.qualified_pose_cache import PoseCache


class PoseCacheTests(unittest.TestCase):
    def test_cold_warm_and_structure_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            structure = root / "pose.cif"
            structure.write_text("first structure")
            function = Mock(return_value={"structure_observation_count": 1, "contact_jaccard_median": 1.0})
            cache = PoseCache(function, root / "cache", {"source": "v1"})
            kwargs = {"target_chain": "A", "nanobody_chain": "B", "epitope": {1, 2}}
            first = cache("AAAA", [structure], **kwargs)
            self.assertEqual(cache("AAAA", [structure], **kwargs), first)
            self.assertEqual(function.call_count, 1)
            self.assertEqual((cache.hits, cache.misses), (1, 1))
            structure.write_text("changed structure")
            cache("AAAA", [structure], **kwargs)
            self.assertEqual(function.call_count, 2)
            cache("AAAA", [structure], **{**kwargs, "epitope": {3}})
            self.assertEqual(function.call_count, 3)
            PoseCache(function, root / "cache", {"source": "v2"})("AAAA", [structure], **kwargs)
            self.assertEqual(function.call_count, 4)

    def test_corrupt_record_is_recomputed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            structure = root / "pose.cif"
            structure.write_text("structure")
            function = Mock(return_value={"structure_observation_count": 1})
            cache = PoseCache(function, root / "cache", "source-v1")
            cache("AAAA", [structure], epitope=set())
            path = next((root / "cache").glob("*.json"))
            record = json.loads(path.read_text())
            record["summary"]["structure_observation_count"] = 999
            path.write_text(json.dumps(record))
            self.assertEqual(cache("AAAA", [structure], epitope=set())["structure_observation_count"], 1)
            self.assertEqual(function.call_count, 2)


if __name__ == "__main__":
    unittest.main()
