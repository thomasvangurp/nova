#!/usr/bin/env python3
"""Analyze only saved scientific scalars; no network or credential access."""
import argparse
import itertools
import json
from pathlib import Path
import statistics

def ranks(values):
    result=[0.0]*len(values)
    ordered=sorted(range(len(values)),key=lambda i:values[i])
    for value,group in itertools.groupby(ordered,key=lambda i:values[i]):
        members=list(group)
        rank=statistics.mean(ordered.index(i)+1 for i in members)
        for i in members:result[i]=rank
    return result

def pearson(a,b):
    ma,mb=statistics.mean(a),statistics.mean(b)
    numerator=sum((x-ma)*(y-mb) for x,y in zip(a,b))
    denominator=(sum((x-ma)**2 for x in a)*sum((y-mb)**2 for y in b))**0.5
    return numerator/denominator if denominator else None

def agreement(a,b):
    if len(a)!=len(b) or len(a)<2:raise ValueError('paired_sample_required')
    delta=[y-x for x,y in zip(a,b)]
    concordant=discordant=tied=0
    for i,j in itertools.combinations(range(len(a)),2):
        sign=(a[i]-a[j])*(b[i]-b[j])
        concordant+=sign>0;discordant+=sign<0;tied+=sign==0
    k=min(5,len(a))
    first=set(sorted(range(len(a)),key=lambda i:a[i],reverse=True)[:k])
    second=set(sorted(range(len(b)),key=lambda i:b[i],reverse=True)[:k])
    return dict(count=len(a),mean_signed_delta=statistics.mean(delta),
                mean_absolute_delta=statistics.mean(abs(x) for x in delta),max_absolute_delta=max(map(abs,delta)),
                spearman=pearson(ranks(a),ranks(b)),
                pairwise_concordance=concordant/(concordant+discordant) if concordant+discordant else None,
                tied_pairs=tied,top_k=k,top_k_overlap=len(first&second),winner_agrees=max(range(len(a)),key=lambda i:a[i])==max(range(len(b)),key=lambda i:b[i]))

def mw_correct(affinity,molecular_weight):
    """Pinned OSS expression, evaluated in Python float64, not bitwise parity."""
    return 1.03525938*affinity-0.59992683*molecular_weight**0.3+2.83288489

def analyze(directory):
    load=lambda name:json.loads((directory/name).read_text())
    reference=load('paired-reference-0.json')
    names=[x['name'] for x in reference['records']]
    reference_by_name={x['name']:x for x in reference['records']}
    a=[reference_by_name[name]['score'] for name in names]
    result=dict(count=len(names),reference_wall_seconds=reference['wall_seconds'],comparisons={},throughput={})
    for mode in ['reference','optimized']:
        for repeat in [0,1]:
            path=directory/f'paired-{mode}-{repeat}.json'
            if not path.exists():continue
            data=json.loads(path.read_text());by_name={x['name']:x for x in data['records']}
            if set(by_name)!=set(names):raise ValueError('portfolio_mismatch')
            label=f'{mode}_{repeat}'
            result['comparisons'][label]=agreement(a,[by_name[name]['score'] for name in names])
            result['comparisons'][label]['probability']=agreement([reference_by_name[name]['metrics']['affinity_probability_binary'] for name in names],[by_name[name]['metrics']['affinity_probability_binary'] for name in names])
            result['comparisons'][label]['affinity_pred_value']=agreement([reference_by_name[name]['metrics']['affinity_pred_value'] for name in names],[by_name[name]['metrics']['affinity_pred_value'] for name in names])
            timings=data['predict_step_timings']
            warm_sum=0
            for phase in [False,True]:
                phase_times=[x['wall_seconds'] for x in timings if x['affinity']==phase]
                if len(phase_times)!=len(names):raise ValueError('unexpected_predict_step_count')
                warm_sum+=statistics.mean(phase_times[1:])
            result['throughput'][label]=dict(batch_wall_seconds=data['wall_seconds'],amortized_wall_seconds_per_molecule=data['wall_seconds']/len(names),
                                            warm_gpu_inference_seconds_per_molecule=warm_sum,predictions_per_hour=3600*len(names)/data['wall_seconds'])
    if (directory/'paired-nim.json').exists():
        data=load('paired-nim.json');groups={}
        for record in data['records']:
            if record.get('passed'):groups.setdefault(record['repetition'],{})[record['case']]=record
        for repeat,by_name in groups.items():
            if set(by_name)!=set(names):
                result.setdefault('incomplete_nim_repeats',{})[str(repeat)]=len(by_name)
                continue
            label=f'nim_{repeat}'
            result['comparisons'][label]=agreement(a,[by_name[name]['score'] for name in names])
            result['comparisons'][label]['probability']=agreement([reference_by_name[name]['metrics']['affinity_probability_binary'] for name in names],[by_name[name]['affinities']['B']['affinity_probability_binary'] for name in names])
            result['comparisons'][label]['affinity_pred_value']=agreement([reference_by_name[name]['metrics']['affinity_pred_value'] for name in names],[by_name[name]['affinities']['B']['affinity_pred_value'] for name in names])
            if all('molecular_weight' in by_name[name] for name in names):
                corrected=[mw_correct(by_name[name]['affinities']['B']['affinity_pred_value'],by_name[name]['molecular_weight']) for name in names]
                corrected_scores=[(by_name[name]['affinities']['B']['affinity_probability_binary']-v)/by_name[name]['heavy_atoms'] for name,v in zip(names,corrected)]
                result['comparisons'][label+'_experimental_mw_corrected']=agreement(a,corrected_scores)
                result['comparisons'][label+'_experimental_mw_corrected']['affinity_pred_value']=agreement([reference_by_name[name]['metrics']['affinity_pred_value'] for name in names],corrected)
                result['comparisons'][label+'_experimental_mw_corrected']['limitation']='float64 adapter; not a fix to NIM or a bitwise parity claim'
            seconds=[by_name[name]['wall_seconds'] for name in names]
            result['throughput'][label]=dict(batch_wall_seconds=sum(seconds),amortized_wall_seconds_per_molecule=statistics.mean(seconds),
                                            excluding_first_request_mean_seconds=statistics.mean(seconds[1:]),predictions_per_hour=3600/statistics.mean(seconds))
        if len(groups)>=2 and all(set(groups[i])==set(names) for i in [0,1]):
            x,y=groups[0],groups[1]
            result['nim_repeatability']=agreement([x[name]['score'] for name in names],[y[name]['score'] for name in names])
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=analyze(args.directory)
    if args.output:args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
