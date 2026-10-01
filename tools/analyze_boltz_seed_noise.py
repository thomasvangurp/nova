#!/usr/bin/env python3
"""Compare saved normal scorer seed variability with corrected NIM rankings."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import statistics

from analyze_nim19_pair import agreement, mw_correct


def summarize(records):
    keys=['spearman','pairwise_concordance','top_k_overlap','mean_absolute_delta','winner_agrees']
    result={}
    for key in keys:
        values=[r[key] for r in records if r[key] is not None]
        result[key]=dict(defined_pairs=len(values),mean=statistics.mean(values) if values else None,
                         minimum=min(values) if values else None,maximum=max(values) if values else None)
    return result if records else {}


def compare_samples(normal, nim):
    """All vectors use the same molecule order; comparisons share observations."""
    within=[dict(first=i,second=j,**agreement(normal[i],normal[j]))
            for i,j in itertools.combinations(range(len(normal)),2)]
    between=[dict(normal=i,nim=j,**agreement(a,b))
             for i,a in enumerate(normal) for j,b in enumerate(nim)]
    output=dict(normal_pairs=within,normal_summary=summarize(within),
                corrected_nim_vs_normal=between,nim_normal_summary=summarize(between))
    if len(nim)>=2:
        output['corrected_nim_repeatability']=agreement(nim[0],nim[1])
    if len(normal)>=2 and len(nim)>=2:
        mean_o=[statistics.mean(values) for values in zip(*normal)]
        mean_n=[statistics.mean(values) for values in zip(*nim)]
        var_o=[statistics.variance(values) for values in zip(*normal)]
        var_n=[statistics.variance(values) for values in zip(*nim)]
        mse=statistics.mean((a-b)**2 for a,b in zip(mean_o,mean_n))
        expected=statistics.mean(a/len(normal)+b/len(nim) for a,b in zip(var_o,var_n))
        output['centroid_agreement']=agreement(mean_o,mean_n)
        output['score_noise_estimate']=dict(
            centroid_mean_squared_difference=mse,
            estimated_centroid_noise_mse=expected,
            estimated_persistent_squared_difference=mse-expected,
            normal_single_pass_noise_variance=statistics.mean(var_o),
            nim_single_pass_noise_variance=statistics.mean(var_n),
            limitation='Small-sample independent-draw approximation; fixed portfolio, not a percentage of rank disagreement. Negative persistent estimate is retained, not clamped. Pair comparisons are dependent.')
    return output


def analyze(directory, paired):
    original=json.loads((paired/'paired-reference-0.json').read_text())
    names=[r['name'] for r in original['records']]
    if len(set(names))!=len(names):raise ValueError('duplicate_molecule')
    original_scores=[r['score'] for r in original['records']]
    portfolio_digest=hashlib.sha256((paired/'paired-portfolio.json').read_bytes()).hexdigest()
    seeds=[];normal=[];walls=[]
    model_digest=original.get('model_source_sha256')
    calibration=directory/'paired-reference-seed-68.json'
    if model_digest is None and calibration.is_file():
        model_digest=json.loads(calibration.read_text())['model_source_sha256']
    for seed in [68,69,70,71]:
        path=directory/f'paired-reference-seed-{seed}.json'
        if not path.is_file():continue
        data=json.loads(path.read_text());by_name={r['name']:r for r in data['records']}
        if set(by_name)!=set(names) or len(data['records'])!=len(names):raise ValueError('portfolio_mismatch')
        if data['seed']!=seed or data['portfolio_sha256']!=portfolio_digest:raise ValueError('seed_or_portfolio_mismatch')
        if any(data[key]!=original[key] for key in ['source_sha256','torch_version','cuda_version']):raise ValueError('scorer_provenance_mismatch')
        if model_digest is None or data['model_source_sha256']!=model_digest:raise ValueError('model_provenance_mismatch')
        seeds.append(seed);normal.append([by_name[name]['score'] for name in names]);walls.append(data['wall_seconds'])
    data=json.loads((paired/'paired-nim.json').read_text())
    nim=[]
    for repeat in [0,1]:
        by_name={r['case']:r for r in data['records'] if r.get('passed') and r['repetition']==repeat}
        if set(by_name)!=set(names):raise ValueError('incomplete_nim')
        nim.append([(by_name[name]['affinities']['B']['affinity_probability_binary']-
                     mw_correct(by_name[name]['affinities']['B']['affinity_pred_value'],by_name[name]['molecular_weight']))/
                    by_name[name]['heavy_atoms'] for name in names])
    result=dict(count=len(names),normal_seeds=seeds,batch_wall_seconds=walls,
                portfolio_sha256=portfolio_digest,normal_model_source_sha256=model_digest,
                limitation='One target, 20 molecules, four planned normal seeds, two NIM passes. Same-source replacement GPU checked separately; not live validator or mining proof.',
                **compare_samples(normal,nim))
    if 68 in seeds:result['same_seed_cross_host_calibration']=agreement(original_scores,normal[seeds.index(68)])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--paired',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=analyze(args.directory,args.paired)
    if args.output:args.output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
