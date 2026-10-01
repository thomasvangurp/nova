import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from analyze_boltz_seed_noise import analyze, compare_samples


class SeedNoiseTests(unittest.TestCase):
    def test_identical(self):
        x=[[1,2,3],[1,2,3],[1,2,3]]
        r=compare_samples(x,x[:2])
        self.assertEqual(len(r['normal_pairs']),3)
        self.assertEqual(len(r['corrected_nim_vs_normal']),6)
        self.assertEqual(r['normal_summary']['spearman']['mean'],1)
        self.assertEqual(r['normal_summary']['winner_agrees']['mean'],1)
        self.assertEqual(r['score_noise_estimate']['estimated_persistent_squared_difference'],0)

    def test_systematic_offset(self):
        r=compare_samples([[1,2,3],[1,2,3]],[[2,3,4],[2,3,4]])
        self.assertEqual(r['nim_normal_summary']['spearman']['mean'],1)
        self.assertEqual(r['score_noise_estimate']['estimated_persistent_squared_difference'],1)

    def test_noise_is_not_clamped(self):
        r=compare_samples([[1,2,3],[1,2,3]],[[0,2,4],[2,2,2]])
        self.assertLess(r['score_noise_estimate']['estimated_persistent_squared_difference'],0)

    def test_single_pass_is_incomplete(self):
        r=compare_samples([[1,2,3]],[[1,2,3],[1,3,2]])
        self.assertEqual(r['normal_pairs'],[])
        self.assertNotIn('score_noise_estimate',r)

    def analyze_fixture(self, mismatch=False):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);paired=root/'paired';seeds=root/'seeds'
            paired.mkdir();seeds.mkdir()
            portfolio=paired/'paired-portfolio.json';portfolio.write_text('{}')
            original=dict(source_sha256='main',torch_version='torch',cuda_version='cuda',
                          records=[dict(name='a',score=1),dict(name='b',score=2)])
            (paired/'paired-reference-0.json').write_text(json.dumps(original))
            for seed in [68,69]:
                data=dict(original,seed=seed,wall_seconds=1,
                          portfolio_sha256=hashlib.sha256(portfolio.read_bytes()).hexdigest(),
                          model_source_sha256='different' if mismatch and seed==69 else 'model')
                (seeds/f'paired-reference-seed-{seed}.json').write_text(json.dumps(data))
            records=[dict(case=name,passed=True,repetition=repeat,molecular_weight=100,
                          heavy_atoms=10,affinities={'B':dict(affinity_probability_binary=1,
                                                           affinity_pred_value=value)})
                     for repeat in [0,1] for name,value in [('a',1),('b',2)]]
            (paired/'paired-nim.json').write_text(json.dumps(dict(records=records)))
            return analyze(seeds,paired)

    def test_legacy_original_without_model_digest(self):
        result=self.analyze_fixture()
        self.assertEqual(result['normal_model_source_sha256'],'model')
        self.assertEqual(result['same_seed_cross_host_calibration']['max_absolute_delta'],0)

    def test_model_digest_must_match_calibration(self):
        with self.assertRaisesRegex(ValueError,'model_provenance_mismatch'):
            self.analyze_fixture(mismatch=True)


if __name__=='__main__':unittest.main()
