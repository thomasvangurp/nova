"""OnlineResidualProxy must accept any base proxy, encoder or not.

Regression: the runner was wired to use BankProxy as the residual head's base.
BankProxy deliberately exposes no ComponentEncoder -- dropping the six component
target-encodings is precisely what lets it rank reagents absent from history --
but observe() and predict() both did `self.frozen.encoder` unconditionally. Two
full trials scored round one, then died with AttributeError before selection.
Every unit test passed, because none exercised the two classes together.

Requires rdkit (the analysis pixi environment).
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# test_rowcolumn_lane installs a stub for compound_miner.adaptive_strategy so it
# can run without rdkit. If that module was loaded first in the same process,
# the stub is still in sys.modules and we would silently test the stub instead
# of the real class -- which is exactly the class this file exists to check.
for _name in [m for m in list(sys.modules) if m == "compound_miner" or m.startswith("compound_miner.")]:
    del sys.modules[_name]

import numpy as np
import pandas as pd

from compound_miner.adaptive_strategy import OnlineResidualProxy
from compound_miner.bank_proxy import BankProxy
from compound_miner.proxy import TargetProxy
from compound_miner.types import Candidate, ScoredCandidate

SMILES = ["CCO", "CCN", "c1ccccc1", "CC(=O)O", "CCCC", "c1ccncc1",
          "CC(C)O", "CCOC", "C1CCCCC1", "CCCCO", "CN(C)C", "c1ccsc1"]


def _frame(n=60):
    return pd.DataFrame.from_records([
        dict(name=f"n{i}", smiles=SMILES[i % len(SMILES)], reaction=3,
             component_a=i, component_b=i, component_c=i,
             score=0.01 * (i % 7), repeats=1, source="run") for i in range(n)])


def _scored(n=12):
    out = []
    for i in range(n):
        c = Candidate(f"rxn:3:{i}:{i}:{i}", SMILES[i % len(SMILES)], 3,
                      (i, i + 1, i + 2), f"key{i}", 10, 1)
        probability, affinity = 0.3, -0.5
        out.append(ScoredCandidate(c, probability, affinity,
                                   (probability - affinity) / c.heavy_atoms, "nim", 0.1))
    return out


class ResidualBaseContract(unittest.TestCase):
    def _cycle(self, base):
        residual = OnlineResidualProxy(base, expected_scorer="nim", minimum_labels=4)
        scored = _scored()
        update = residual.observe(scored)
        predicted = residual.predict([s.candidate for s in scored])
        self.assertTrue(update["fitted"])
        self.assertEqual(len(predicted), len(scored))
        self.assertTrue(all(np.isfinite(c.proxy_mean) for c in predicted))
        self.assertTrue(all(c.proxy_std >= 0 for c in predicted))
        return predicted

    def test_encoderless_base_is_accepted(self):
        # The exact wiring that killed trials 3 and 4.
        self._cycle(BankProxy(trees=16, jobs=1).fit(_frame()))

    def test_encoder_base_still_works(self):
        self._cycle(TargetProxy(trees=16, jobs=1).fit(_frame()))

    def test_design_matrix_matches_the_base(self):
        chemistry = np.zeros((3, 1035), dtype=np.float32)
        candidates = [s.candidate for s in _scored(3)]
        bank = OnlineResidualProxy(BankProxy(trees=8, jobs=1).fit(_frame()),
                                   expected_scorer="nim", minimum_labels=4)
        frozen = OnlineResidualProxy(TargetProxy(trees=8, jobs=1).fit(_frame()),
                                     expected_scorer="nim", minimum_labels=4)
        # Chemistry-only base -> chemistry-only residual features.
        self.assertEqual(bank._design(candidates, chemistry).shape[1], 1035)
        # Encoder base -> chemistry plus reaction bits and component encodings.
        self.assertGreater(frozen._design(candidates, chemistry).shape[1], 1035)


if __name__ == "__main__":
    unittest.main()
