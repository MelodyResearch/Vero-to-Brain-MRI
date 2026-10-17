#!/usr/bin/env python3
"""Finite source-bound train/dev renders and shortcut checks; sealed test metadata only."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from PIL import Image
from mri_rl.visit_correspondence import (SOURCE, ROUTE, RENDER, build_cases, render_split,
    token, sha_file, _durable, fit_train_threshold, fixed_threshold_metrics)


def diagnostics(cases, summaries, destination):
    """Spatial vs nonspatial deterministic controls, thresholds fit on train only."""
    qc={q['source_sha256']:q for s in summaries.values() for q in s['qc_PRIVATE']}
    train_hashes={i['sha256'] for c in cases['train'] for i in (c['query'],c['candidate'])}
    all_hashes={i['sha256'] for split in ('train','internal_dev') for c in cases[split] for i in (c['query'],c['candidate'])}
    # Scaling is fitted without using held-out dev features or labels.
    train_vectors=np.array([qc[h]['nonspatial_summary'] for h in sorted(train_hashes)])
    scale=np.std(train_vectors,axis=0);scale=np.where(scale>1e-6,scale,1.)
    features={h:np.array(qc[h]['nonspatial_summary'])/scale for h in all_hashes}
    spatial={}
    for h in sorted(all_hashes):
        with Image.open(destination/'assets'/(token('volume',h)+'.png')) as image:
            a=np.asarray(image.resize((96,96),Image.Resampling.BILINEAR),dtype=float).ravel()
        a-=a.mean();norm=np.linalg.norm(a);spatial[h]=a/max(norm,1e-12)
    results={}; private={}
    for kind in ('spatial_multiview_correlation','nonspatial_intensity_foreground'):
        scores={};targets={}
        for split in ('train','internal_dev'):
            rows=cases[split];targets[split]=[r['target'] for r in rows];scores[split]=[]
            for row in rows:
                q=row['query']['sha256'];c=row['candidate']['sha256']
                value=float(spatial[q]@spatial[c]) if kind.startswith('spatial') else -float(np.linalg.norm(features[q]-features[c]))
                scores[split].append(value)
        fitted=fit_train_threshold(scores['train'],targets['train'])
        results[kind]={'fit':fitted,**{split:fixed_threshold_metrics(scores[split],targets[split],fitted['threshold']) for split in scores}}
        private[kind]={split:[{'case_id':r['case_id'],'score':s,'target':r['target'],'cluster_key':r['cluster_key']} for r,s in zip(cases[split],scores[split])] for split in scores}
        for split in scores:
            by_query={}
            for row,value in zip(cases[split],scores[split]):by_query.setdefault(row['query']['sha256'],{})[row['target']]=value
            pair_values=[float(x['same']>x['different'])+.5*float(x['same']==x['different']) for x in by_query.values()]
            results[kind][split]['paired_ranking_queries']=len(pair_values)
            results[kind][split]['paired_ranking_accuracy']=float(np.mean(pair_values))
    _durable(destination/'shortcut_scores_PRIVATE.json',(json.dumps(private,sort_keys=True)+'\n').encode())
    return {'status':'CPU_DIAGNOSTIC_ONLY_NOT_VLM_OR_RL','constant_class_accuracy':.5,
            'random_binary_expectation':.5,'independent_unit':'two-person matched cycle',
            'controls':results,'no_test_pixels_or_outcomes':True,
            'interpretation':'Spatial success is a simple image baseline, not proof of MRI understanding. Nonspatial success is an unresolved shortcut warning; small dev denominator prevents generalization claims.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--expected-plan-sha256',required=True)
    p.add_argument('--source-root',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    a=p.parse_args(); repo=Path(__file__).resolve().parents[1]
    if sha_file(a.plan)!=a.expected_plan_sha256:raise ValueError('Frozen plan hash mismatch')
    if not a.destination.resolve().is_relative_to((repo/'local_data').resolve()):raise ValueError('MRI task outputs must remain under ignored local_data')
    plan=json.loads(a.plan.read_text());cases=build_cases(plan)
    if {k:len(v) for k,v in cases.items()}!={'train':48,'internal_dev':16,'test':16}:raise ValueError('Frozen cohort task inventory differs')
    summaries={}
    for split in ('train','internal_dev'):
        summaries[split]=render_split(plan,split,a.source_root,a.destination,a.expected_plan_sha256)
        print(json.dumps({'state':'SPLIT_RENDERED','split':split,'rows':summaries[split]['rows'],'source_volumes':summaries[split]['source_volumes'],'technical_warning_volumes':summaries[split]['blank_or_low_contrast_tile_warnings']}),flush=True)
    # No test NIfTI or image is opened here. Registry uses already frozen source metadata only.
    sealed=[{'case_id':c['case_id'],'split':'test','cluster_key':c['cluster_key'],
             'query_patient_key':token('patient',c['query_subject']),
             'candidate_patient_key':token('patient',c['candidate_subject']),
             'query_source_sha256':c['query']['sha256'],'candidate_source_sha256':c['candidate']['sha256'],
             'source_equality_target':c['target'],'pixels_rendered':False} for c in cases['test']]
    _durable(a.destination/'sealed_test_registry_PRIVATE.json',(json.dumps(sealed,sort_keys=True)+'\n').encode())
    shortcut=diagnostics(cases,summaries,a.destination)
    safe={split:{k:v for k,v in summary.items() if k!='qc_PRIVATE'} for split,summary in summaries.items()}
    final={'status':'TRAIN_DEV_RENDERED_TYPED_CANDIDATE_NOT_SCIENTIFICALLY_ADMITTED',
           'evidence_tier':'OBSERVED_CPU_PREPROCESSING_AND_SHORTCUT_DIAGNOSTIC','source':SOURCE,'route':ROUTE,
           'mission_alignment':'Prepare a distinct metadata-hidden MRI visit-correspondence task for systematic Vero multi-task transfer; not a biological change verifier.',
           'source_version':plan['source'],'plan_sha256':a.expected_plan_sha256,
           'helper_source_sha256':sha_file(repo/'src/mri_rl/visit_correspondence.py'),'builder_source_sha256':sha_file(__file__),
           'split_cases':{k:len(v) for k,v in cases.items()},'split_people':{k:len(v) for k,v in plan['splits_PRIVATE'].items()},
           'split_matched_cycles':{k:len({c['cluster_key'] for c in v}) for k,v in cases.items()},
           'rendered_source_volumes':64,'rendered_pair_cases':64,'test_source_volumes_opened':0,'test_pairs_rendered':0,
           'sealed_test_registry_sha256':sha_file(a.destination/'sealed_test_registry_PRIVATE.json'),
           'summaries':safe,'shortcut_diagnostics':shortcut,
           'visual_task_admission':'NOT_VERIFIED','cross_model_review':'NOT_VERIFIED','model_or_rl_run':False,
           'limitations':['No expert visual answerability claim; technical renders and CPU controls are insufficient for admission.',
                          'Source scanner model absent; manufacturer/TR/TE/age/sex matching cannot rule out all acquisition shortcuts.',
                          'Labels are BIDS identity correspondence, not diagnosis, clinical longitudinal change or independent-patient generalization.',
                          'Negative cases connect two people; matched cycle is the conservative statistical cluster.']}
    _durable(a.destination/'task_receipt_SAFE.json',(json.dumps(final,indent=2,sort_keys=True)+'\n').encode())
    print(json.dumps({'state':'TRAIN_DEV_TASK_PACK_COMPLETE','receipt_sha256':sha_file(a.destination/'task_receipt_SAFE.json'),'rendered_source_volumes':64,'rendered_pair_cases':64,'test_pixels_opened':0}),flush=True)


if __name__=='__main__':main()
