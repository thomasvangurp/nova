import unittest
from analyze_nim19_pair import agreement,ranks,mw_correct

class AnalysisTests(unittest.TestCase):
    def test_tied_ranks(self):
        self.assertEqual(ranks([2,1,2]),[2.5,1,2.5])
    def test_identical(self):
        result=agreement([1,2,3],[1,2,3])
        self.assertEqual(result['spearman'],1)
        self.assertEqual(result['max_absolute_delta'],0)
        self.assertTrue(result['winner_agrees'])
    def test_reversed(self):
        result=agreement([1,2,3],[3,2,1])
        self.assertEqual(result['spearman'],-1)
        self.assertEqual(result['pairwise_concordance'],0)
    def test_incomplete(self):
        with self.assertRaises(ValueError):agreement([1,2],[1])
    def test_pinned_mw_expression(self):
        self.assertAlmostEqual(mw_correct(1,1),3.26821744)

if __name__=='__main__':unittest.main()
