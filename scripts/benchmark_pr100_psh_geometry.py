#!/usr/bin/env python3
"""Fixed-PDB component benchmark; no inference, GPU use unless explicitly opted in.

PSA fixtures must be captured from the SAME fixed PDB and native PSA executable.
Their computation time is excluded: report describes annotation after PSA, not
whole-TNP speed. The original surface parser/selection and PSH sum are retained.
"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random
import shutil
import statistics
import sys
import tempfile
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from nanobody_opt.pr100_psh_geometry import accelerated_distances, _distance_table
from nanobody_opt.pr100_psh_profile import load_pinned_modules, file_hash, quiet_native_output, METRIC_KEYS


def benchmark(fixtures, repeats=5, include_cuda=False):
    fixture_root=Path(fixtures)
    pdb_manifest=json.loads((fixture_root/'manifest.json').read_text())
    psa_manifest={row['name']:row for row in json.loads((fixture_root/'psa-manifest.json').read_text())}
    modes=['baseline','reference_cache','numpy']+(['cuda']if include_cuda else [])
    with quiet_native_output():
        surface,geometry,source_hashes,imports=load_pinned_modules()
        cases=[]
        for entry in pdb_manifest:
            pdb=fixture_root/entry['name'];psa=pdb.with_suffix('.psa');psa_info=psa_manifest[psa.name]
            if file_hash(pdb)!=entry['sha256'] or file_hash(psa)!=psa_info['sha256'] or psa_info['input_sha256']!=entry['sha256']:
                raise ValueError('fixture_identity_mismatch')
            chain=geometry.PDBchain(str(pdb),'H');residues=list(chain.residues.values())
            table=_distance_table(residues,'numpy',2048)
            mismatches=0
            for i,left in enumerate(residues):
                for j,right in enumerate(residues):
                    mismatches+=float(left.distance(left,right)).hex()!=float(table[i][j]).hex()
            original_psa=surface.runPSA
            def replay_psa(input_pdb,where_to,verbose=True):
                if file_hash(input_pdb)!=entry['sha256']:
                    raise ValueError('surface_fixture_does_not_match_structure')
                shutil.copyfile(psa,where_to)
            order=[mode for _ in range(repeats)for mode in modes];random.Random(0).shuffle(order)
            records=[]
            try:
                surface.runPSA=replay_psa
                for mode in order:
                    with tempfile.TemporaryDirectory(prefix='psh-geometry-')as directory:
                        copied=Path(directory)/'input.pdb';shutil.copyfile(pdb,copied)
                        context=nullcontext({})if mode=='baseline'else accelerated_distances(surface,geometry,backend=mode)
                        started=time.perf_counter()
                        with context as stats:
                            value=surface.CreateAnnotation(0,7.4,str(copied),'IG','imgt',verbose=False)
                        elapsed=time.perf_counter()-started
                        records.append({'mode':mode,'wall_seconds':elapsed,'metrics':{key:float(value[0][key])for key in METRIC_KEYS},'annotated_pdb_sha256':file_hash(copied.with_name('input_Annotated.pdb')),'distance_stats':stats})
            finally:
                surface.runPSA=original_psa
            reference=next(row for row in records if row['mode']=='baseline')
            medians={mode:statistics.median(row['wall_seconds']for row in records if row['mode']==mode)for mode in modes}
            cases.append({'input_pdb_sha256':entry['sha256'],'input_psa_sha256':psa_info['sha256'],'residues':len(residues),'atoms':sum(len(r.atoms)for r in residues),'numpy_distance_pairs_checked':len(residues)**2,'numpy_distance_bit_mismatches':int(mismatches),'all_metrics_identical':all(row['metrics']==reference['metrics']for row in records),'all_annotations_identical':all(row['annotated_pdb_sha256']==reference['annotated_pdb_sha256']for row in records),'median_seconds':medians,'speedup':{mode:medians['baseline']/value for mode,value in medians.items()},'records':records})
    return {'schema':'pr100-psh-geometry-experiment-v1','source_hashes':source_hashes,'repeats_per_mode_per_structure':repeats,'cuda_requested':include_cuda,'cases':cases,'all_numpy_distance_pairs_bit_identical':all(c['numpy_distance_bit_mismatches']==0 for c in cases),'all_full_annotation_outputs_identical':all(c['all_metrics_identical']and c['all_annotations_identical']for c in cases),'median_component_speedups':{mode:statistics.median(c['speedup'][mode]for c in cases)for mode in modes},'limitations':['local_cpu_component_benchmark','fixed_structure_surface_fixture_replay_excludes_actual_psa_runtime','no_model_prediction_or_refinement','no_whole_pipeline_or_production_speedup_claim','cpu_parity_does_not_qualify_cuda','six_structure_sample_is_not_exhaustive_equivalence']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures',required=True,type=Path)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--cuda',action='store_true')
    args=parser.parse_args()
    if not 1<=args.repeats<=20:parser.error('repeats must be1..20')
    try:
        report=benchmark(args.fixtures,args.repeats,args.cuda)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False))
        print(json.dumps({k:report[k]for k in ['all_numpy_distance_pairs_bit_identical','all_full_annotation_outputs_identical','median_component_speedups','cuda_requested']}))
        return 0 if report['all_numpy_distance_pairs_bit_identical'] and report['all_full_annotation_outputs_identical'] else 2
    except Exception as error:
        print(json.dumps({'status':'failed','error_type':type(error).__name__}))
        return 1

if __name__=='__main__':raise SystemExit(main())
