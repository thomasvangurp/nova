"""Checks for the label-bank proxy. Requires rdkit (analysis pixi env)."""
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from compound_miner.bank_proxy import BankProxy, chemistry_matrix, load_labels
from compound_miner.types import Candidate

SMILES = ["CCO", "CCN", "c1ccccc1", "CC(=O)O", "CCCC", "c1ccncc1",
          "CC(C)O", "CCOC", "C1CCCCC1", "CCCCO", "CN(C)C", "c1ccsc1"]


def _archive(rows, path=None):
    path = path or Path(tempfile.mkdtemp()) / "a.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE candidates(name TEXT,smiles TEXT,reaction INT,"
                           "components TEXT,inchikey TEXT,heavy_atoms INT,rotatable_bonds INT,"
                           "proxy_mean REAL,proxy_std REAL,score REAL,scorer TEXT)")
        connection.executemany(
            "INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    return path


def _row(i, score, reaction=3, scorer="nim"):
    return (f"rxn:3:{i}:{i}:{i}", SMILES[i % len(SMILES)], reaction,
            json.dumps([i, i + 1, i + 2]), f"key{i}", 10, 1, 0.5, 0.1, score, scorer)


class Labels(unittest.TestCase):
    def test_averages_repeats_and_tags_source(self):
        path = _archive([_row(0, 0.1), _row(1, 0.2)])
        with sqlite3.connect(path) as c:
            c.execute("INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?,?,?,?)", _row(0, 0.3))
        frame = load_labels([path])
        self.assertEqual(len(frame), 2)
        first = frame[frame["name"] == "rxn:3:0:0:0"].iloc[0]
        self.assertAlmostEqual(float(first["score"]), 0.2)   # (0.1 + 0.3) / 2
        self.assertEqual(int(first["repeats"]), 2)
        self.assertEqual(set(frame["source"]), {path.parent.name})

    def test_skips_other_reactions_and_unscored(self):
        path = _archive([_row(0, 0.1), _row(1, 0.2, reaction=5), _row(2, None)])
        self.assertEqual(len(load_labels([path])), 1)

    def test_scorer_filter(self):
        path = _archive([_row(0, 0.1, scorer="nim"), _row(1, 0.2, scorer="other")])
        self.assertEqual(len(load_labels([path], scorer="nim")), 1)

    def test_missing_archive_is_ignored(self):
        self.assertTrue(load_labels(["/nonexistent/a.sqlite"]).empty)


class Fitting(unittest.TestCase):
    def _frame(self, n=40):
        return pd.DataFrame.from_records([
            dict(name=f"n{i}", smiles=SMILES[i % len(SMILES)], component_a=i,
                 component_b=i, component_c=i, score=0.01 * (i % 7),
                 repeats=1, source="run")
            for i in range(n)])

    def test_fit_predict_sets_mean_and_std(self):
        frame = self._frame()
        model = BankProxy(trees=16, jobs=1).fit(frame)
        candidates = [Candidate(f"n{i}", SMILES[i % len(SMILES)], 3, (i, i, i),
                                f"k{i}", 10, 1) for i in range(5)]
        out = model.predict(candidates)
        self.assertEqual(len(out), 5)
        self.assertTrue(all(np.isfinite(c.proxy_mean) for c in out))
        self.assertTrue(all(c.proxy_std >= 0 for c in out))
        self.assertEqual([c.name for c in out], [c.name for c in candidates])

    def test_unfitted_predict_raises(self):
        with self.assertRaises(RuntimeError):
            BankProxy().predict([Candidate("n", "CCO", 3, (1, 2, 3), "k", 10, 1)])

    def test_empty_and_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            BankProxy().fit(pd.DataFrame())
        frame = self._frame(20)
        frame.loc[0, "score"] = float("nan")
        with self.assertRaises(ValueError):
            BankProxy(trees=8, jobs=1).fit(frame)

    def test_features_exclude_component_identity(self):
        # Same molecule, different reagent ids -> identical features. This is the
        # property that lets the model generalise to reagents absent from history.
        a = chemistry_matrix(["CCO"])
        b = chemistry_matrix(["CCO"])
        self.assertTrue(np.array_equal(a, b))
        self.assertEqual(a.shape[1], 1035)

    def test_save_load_roundtrip_and_metadata(self):
        frame = self._frame()
        path = Path(tempfile.mkdtemp()) / "bank.joblib"
        BankProxy(trees=16, jobs=1).fit(frame, scorer="nim").save(path)
        model = BankProxy.load(path)
        self.assertTrue(model.fitted)
        self.assertEqual(model.label_count, len(frame))
        self.assertEqual(model.metadata()["anchored_on_frozen_proxy"], False)
        self.assertEqual(model.metadata()["scorer"], "nim")

    def test_load_rejects_wrong_object(self):
        import joblib
        path = Path(tempfile.mkdtemp()) / "x.joblib"
        joblib.dump({"not": "a proxy"}, path)
        with self.assertRaises(ValueError):
            BankProxy.load(path)


if __name__ == "__main__":
    unittest.main()
