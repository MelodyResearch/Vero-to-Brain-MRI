"""Analyze ONLY authenticated complete B28/U7 frozen10-record visual probes.

No native model imports, new scoring definitions or significance tests. Individual
image/pair keys and scores remain under local_data; public output has aggregates.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import importlib.abc
import json
from pathlib import Path
from statistics import mean
import sys

class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'mlx','mlx_vlm','torch','transformers'}:raise RuntimeError('Native import forbidden during probe analysis')
sys.meta_path.insert(0,NoNative())
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mri_rl import c40_visual_probe as probe
from mri_rl.c40_visual_controls import LABELS,build_pairs
from mri_rl.two_task_pool import canonical_sha,require

TASKS=tuple(LABELS);SCORES=('mean','raw','first_label');CHECKPOINTS=('B28','U_raw_seed7')
FROZEN_PROBE_SOURCE_SHA='e29e6c3c13af834efe6705405727ffa1c51f5599ca2dd19f6b9847ee674de94c'

def load(p):return json.loads(Path(p).read_text())
def file_sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require_hash(p,expected,what):require(Path(p).is_file() and file_sha(p)==expected,what+' digest missing or changed')
def summary(values):return {'n':len(values),'mean':mean(values),'minimum':min(values),'maximum':max(values)}
def unit_mean(values):return mean(mean(v) for v in values.values())

def margins(row):
    suffix='t1_minus_flair' if row['category']=='mri_sequence' else 'same_minus_different'
    return {'mean':row['comparisons']['mean_'+suffix],'raw':row['comparisons']['raw_'+suffix],
        'first_label':row['prefix_oracle_first_logps'][1]-row['prefix_oracle_first_logps'][0]}


def validate_records(rows,records,scope,checkpoint,tokenizer):
    require(checkpoint in CHECKPOINTS and len(rows)==len(records)==10,'Exactly10 completed records per checkpoint required')
    require(len({r['record_id'] for r in rows})==10,'Duplicate output record')
    result={}
    for row,expected in zip(rows,records,strict=True):
        require(all(k in row and row[k]==v for k,v in expected.items()),'Frozen record/order/pixels/prompt/route differs')
        require(row.get('run_identity')==scope['run_identity'] and row.get('checkpoint')==checkpoint and
            row.get('adapter_sha256')==scope['adapter']['sha256'] and row.get('weight_updates')==0,
            'Checkpoint/run identity or read-only contract differs')
        require(isinstance(row.get('response'),str) and row.get('candidate_order')==list(LABELS[row['category']]) and
            row.get('new_greedy_generation') is (expected['condition']=='no_image'),'Candidate order/greedy reuse contract differs')
        probe.validate_numeric_route(row,tokenizer)
        result[row['record_id']]=row
    for task in TASKS:
        selected=[r for r in rows if r['category']==task]
        require(len(selected)==5 and sum(r['condition']=='no_image' for r in selected)==1 and
            sum(r['condition']=='real_image' for r in selected)==4,'Both tasks require one null and four real images')
    return result


def authenticate_output(plan,scope_path,scope_sha,output,output_sha,checkpoint,tokenizer):
    require_hash(scope_path,scope_sha,checkpoint+' externally authenticated scope')
    require_hash(output,output_sha,checkpoint+' externally authenticated completed output')
    scope=load(scope_path);sources=probe.source_hashes()
    require(scope.get('kind')=='C40_visual_likelihood_v1' and scope.get('execution_tier')=='B' and scope.get('checkpoint')==checkpoint and
        scope.get('plan_sha256')==probe.PLAN_SHA and scope.get('sources')==sources and scope.get('adapter')==plan['checkpoints'][checkpoint] and
        scope.get('derived_cache')==plan['derived_cache'] and scope.get('weight_updates')==0 and scope.get('original21training_ready') is True and
        scope.get('output')==str(Path(output).resolve()),'Actual probe scope/source/weights/cache/output differs')
    identity=canonical_sha({'plan':probe.PLAN_SHA,'checkpoint':checkpoint,'adapter':scope['adapter'],'sources':sources})
    require(scope.get('run_identity')==identity and scope.get('fixed_prefix')==plan['likelihood']['prefix'] and
        scope.get('fixed_suffix')==plan['likelihood']['suffix'] and scope.get('candidate_order')=={t:list(LABELS[t]) for t in TASKS} and
        scope.get('prefix_tolerance')==.002,'Exact native candidate/scoring identity differs')
    count=scope.get('completed_records');records=plan['records_PRIVATE'];raw=Path(output).read_bytes()
    require(type(count) is int and 0<=count<=10 and scope.get('new_record_ids')==[r['record_id'] for r in records[count:]] and
        scope.get('forwards_per_record')==3 and scope.get('max_new_forwards')==3*(10-count),'Finite record scope differs')
    require(raw.endswith(b'\n') and len(raw.splitlines())==10,'Incomplete, torn or oversized probe output')
    if count:
        prefix=b''.join(raw.splitlines(keepends=True)[:count]);require(hashlib.sha256(prefix).hexdigest()==scope['existing_output_sha256'],'Resumed output prefix changed')
    else:require(scope.get('existing_output_sha256') is None,'Unexpected prior output for a fresh probe')
    rows=[json.loads(line) for line in raw.splitlines()]
    result=validate_records(rows,records,scope,checkpoint,tokenizer)
    require(file_sha(output)==output_sha and file_sha(scope_path)==scope_sha,'Evidence changed during analysis')
    return result


def private_pairs(plan):
    rows=[json.loads(x) for x in Path(plan['source_manifest']).read_text().splitlines() if x.strip()]
    chosen={r['source_record_id']:r for r in plan['records_PRIVATE'] if r['condition']=='real_image'};pairs=[]
    for task,taskpairs in build_pairs(rows).items():
        for pair in taskpairs:
            n,p=pair['negative']['id'],pair['positive']['id']
            if n in chosen or p in chosen:
                require(n in chosen and p in chosen,'Frozen subset has an unmatched visual pair')
                pairs.append({'task':task,'unit':pair['unit'],'negative_id':chosen[n]['record_id'],'positive_id':chosen[p]['record_id']})
    require(len(pairs)==4 and len({p['unit'] for p in pairs if p['task']=='mri_sequence'})==2 and
        len({p['unit'] for p in pairs if p['task']=='mri_visit_correspondence'})==1,'Frozen two-person/one-cycle pairing changed')
    return pairs


def checkpoint_summary(rows,pairs):
    private={'images_PRIVATE':{},'pairs_PRIVATE':[]};public={}
    for task in TASKS:
        null=[r for r in rows.values() if r['category']==task and r['condition']=='no_image']
        real=[r for r in rows.values() if r['category']==task and r['condition']=='real_image']
        require(len(null)==1 and len(real)==4,'Exact task/no-image coverage required')
        np=margins(null[0]);public[task]={}
        tp=[p for p in pairs if p['task']==task];units={p['unit'] for p in tp}
        for r in real:
            private['images_PRIVATE'][r['record_id']]={'task':task,'target':r['visible_target_PRIVATE'],
                'margins':margins(r),'real_minus_no_image':{k:margins(r)[k]-np[k] for k in SCORES}}
        for kind in SCORES:
            separation=defaultdict(list);effects=defaultdict(list)
            for p in tp:
                lo=private['images_PRIVATE'][p['negative_id']];hi=private['images_PRIVATE'][p['positive_id']]
                value=hi['margins'][kind]-lo['margins'][kind];separation[p['unit']].append(value)
                effects[p['unit']].extend([lo['real_minus_no_image'][kind],hi['real_minus_no_image'][kind]])
                private['pairs_PRIVATE'].append({**p,'score':kind,'separation':value})
            public[task][kind]={'no_image_margin':np[kind],'real_image_margins':summary([margins(r)[kind] for r in real]),
                'real_minus_no_image_effects':summary([margins(r)[kind]-np[kind] for r in real]),
                'by_true_label':{label:{'margins':summary([margins(r)[kind] for r in real if r['visible_target_PRIVATE']==label]),
                    'real_minus_no_image':summary([margins(r)[kind]-np[kind] for r in real if r['visible_target_PRIVATE']==label])} for label in LABELS[task]},
                'pair_count':len(tp),'independent_units':len(units),'unit':'person' if task=='mri_sequence' else 'matched_two_person_cycle',
                'unit_averaged_paired_separation':unit_mean(separation),'pair_separation_range':[min(v for vv in separation.values() for v in vv),max(v for vv in separation.values() for v in vv)],
                'unit_averaged_real_minus_no_image_effect':unit_mean(effects)}
    return public,private


def compare_checkpoints(by_checkpoint,pairs):
    require(set(by_checkpoint)==set(CHECKPOINTS),'Both exact checkpoints required')
    require(set(by_checkpoint['B28'])==set(by_checkpoint['U_raw_seed7']),'Paired checkpoint record coverage differs')
    result={'checkpoints':{},'U7_minus_B28':{}};private={}
    for checkpoint in CHECKPOINTS:result['checkpoints'][checkpoint],private[checkpoint]=checkpoint_summary(by_checkpoint[checkpoint],pairs)
    for task in TASKS:
        result['U7_minus_B28'][task]={}
        for kind in SCORES:
            a=result['checkpoints']['B28'][task][kind];b=result['checkpoints']['U_raw_seed7'][task][kind]
            result['U7_minus_B28'][task][kind]={
                'delta_no_image_margin':b['no_image_margin']-a['no_image_margin'],
                'delta_unit_averaged_paired_separation':b['unit_averaged_paired_separation']-a['unit_averaged_paired_separation'],
                'delta_unit_averaged_real_minus_no_image_effect':b['unit_averaged_real_minus_no_image_effect']-a['unit_averaged_real_minus_no_image_effect'],
                'paired_separation_null_cancellation':'A common additive answer-prior shift cancels between positive and negative images; delta_null is reported separately.'}
    return result,private


def save_exact(path,value):
    content=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode();path=Path(path)
    if path.exists():require(path.read_bytes()==content,'Existing analysis differs; preserve it and choose a new path')
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f:f.write(content)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,default=ROOT/'local_data/c_two_task_pool_v1/u7_visual_controls_v1/probe_plan_PRIVATE.json')
    for cp in ('b28','u7'):
        for name in ('output','scope'):p.add_argument('--'+cp+'-'+name,type=Path,required=True);p.add_argument('--'+cp+'-'+name+'-sha256',required=True)
    p.add_argument('--private-output',type=Path,required=True);p.add_argument('--output',type=Path,default=ROOT/'results/C40_VISUAL_PROBE_LIKELIHOOD_COMPARISON.json');a=p.parse_args()
    require(file_sha(Path(probe.__file__))==FROZEN_PROBE_SOURCE_SHA,'Frozen numerical-route helper changed')
    plan=probe.validate_plan(a.plan)
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(Path(plan['model_path'])/'tokenizer.json'))
    class RawTokenizer:
        def encode(self,text,add_special_tokens=False):return tokenizer.encode(text,add_special_tokens=add_special_tokens).ids
    inputs={};evidence={}
    for cp,arg in (('B28','b28'),('U_raw_seed7','u7')):
        output=getattr(a,arg+'_output');scope=getattr(a,arg+'_scope');oh=getattr(a,arg+'_output_sha256');sh=getattr(a,arg+'_scope_sha256')
        inputs[cp]=authenticate_output(plan,scope,sh,output,oh,cp,RawTokenizer());evidence[cp]={'output_sha256':oh,'scope_file_sha256':sh}
    result,private=compare_checkpoints(inputs,private_pairs(plan))
    require(a.private_output.resolve().is_relative_to(ROOT/'local_data') and a.output.resolve().is_relative_to(ROOT/'results'),'Private detail/public aggregate paths required')
    private.update({'input_hashes':evidence,'plan_sha256':probe.PLAN_SHA});save_exact(a.private_output,private)
    result.update({'schema':'actual-C40-frozen-visual-probe-comparison-v1','status':'ACTUAL_COMPLETE20_RECORDS_AUTHENTICATED_AND_ANALYZED',
        'evidence_tier':'REPRODUCED_NUMERICAL_ANALYSIS_OF_SAVED_MODEL_OUTPUTS','score_definitions':{
            'primary':'mean label-token log probability: candidate1 minus candidate0','raw':'sum label-token log probabilities: candidate1 minus candidate0',
            'first_label':'independent prefix-oracle first-label-token log probability: candidate1 minus candidate0'},
        'candidate_order':{t:list(LABELS[t]) for t in TASKS},'margin_direction':{'mri_sequence':'T1 minus FLAIR','mri_visit_correspondence':'same minus different'},
        'pair_direction':'positive-label image margin minus negative-label image margin, within same person/level or same query; average queries within matched cycle first',
        'input_evidence':evidence,'plan_sha256':probe.PLAN_SHA,'private_detail_sha256':file_sha(a.private_output),'analyzer_sha256':file_sha(__file__),
        'native_numeric_route_source_sha256':FROZEN_PROBE_SOURCE_SHA,'weight_updates':0,'sealed_test_used':False,'p_values_computed':False,
        'limitations':['Previously examined internal-dev subset: two sequence people at one frozen level and one complete Dallas matched cycle; descriptive only.',
            'All three score definitions retained; mean is prespecified primary. Candidate likelihood margins are not calibrated class probabilities.',
            'No-image margins have no image-grounded truth or accuracy. Null-prior and paired image-separation changes are separate estimands.',
            'Even changed paired separation does not establish brain-specific grounding, independent-patient generalization or a training causal mechanism.',
            'Whole-head/intensity/acquisition shortcuts and source/task confounding remain; panel order swapping does not change identity truth.']})
    require(not any(k.split('.')[0] in {'mlx','mlx_vlm','torch','transformers'} for k in sys.modules),'Unexpected native module import')
    save_exact(a.output,result);print(json.dumps({'status':result['status'],'records':20,'output_sha256':file_sha(a.output),'private_detail_sha256':file_sha(a.private_output),'native_models_launched':0}))

if __name__=='__main__':main()
