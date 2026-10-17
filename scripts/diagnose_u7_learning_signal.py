"""CPU-only U7 artifact diagnosis: no model, optimizer-array audit or sealed test.

Only aggregate categories/counts/hashes leave local_data. Internal-dev uncertainty
uses four sequence people or four matched Dallas cycles, not 40 independent people.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import importlib.abc
import itertools
import json
import math
from pathlib import Path
from statistics import mean
import sys

class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'mlx','mlx_vlm','torch','transformers'}:raise RuntimeError('Native import forbidden in artifact diagnosis')
sys.meta_path.insert(0,NoNative())
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from mri_rl.continuous_rl import canonical,digest,require
from mri_rl import tier_c_entry as te
from mri_rl.two_task_training import make_grader
from mri_rl.two_task_pool import canonical_sha
from mri_rl.two_task_analysis import score_predictions
from mri_rl.two_task_native_results import adapt_cli_output
from mri_rl.mlx_gspo_smoke import overlong_penalty,task_group_is_informative
from mri_rl.vero_mri_reward import _answer_and_format

TASKS=('mri_sequence','mri_visit_correspondence')
ALLOWED={'mri_sequence':{'flair','contrast-enhanced t1'},'mri_visit_correspondence':{'same','different'}}
WITHHELD='WITHHELD_NONCATEGORICAL_OR_UNPARSED'

def load(p):return json.loads(Path(p).read_text())
def jsonl(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def sample_std(v):return math.sqrt(sum((x-mean(v))**2 for x in v)/(len(v)-1))
def standardize(v,epsilon=1e-6):return [(x-mean(v))/(sample_std(v)+epsilon) for x in v]
def quartile(index,n):return min(3,4*index//n)
def prediction(text,task):
    answer,fmt=_answer_and_format(text)
    return (answer.lower() if fmt>0 and answer.lower() in ALLOWED[task] else WITHHELD),fmt

def paired_errors(a,b):
    out=Counter()
    for x,y in zip(a,b,strict=True):out['both_correct' if x and y else 'lost_correct' if x else 'gained_correct' if y else 'both_wrong']+=1
    return {k:out[k] for k in ('both_correct','lost_correct','gained_correct','both_wrong')}

def cluster_uncertainty(grouped,seed=20261017):
    values=np.array([mean(grouped[k]) for k in sorted(grouped)],dtype=float);n=len(values);observed=float(values.mean())
    null=[abs(float(np.mean(values*np.array(s)))) for s in itertools.product((-1,1),repeat=n)]
    p=sum(x>=abs(observed)-1e-12 for x in null)/len(null)
    rng=np.random.default_rng(seed);draws=values[rng.integers(0,n,size=(10000,n))].mean(axis=1)
    return {'units':n,'paired_delta':observed,'exact_sign_flip_two_sided_p':p,'sign_patterns':2**n,
        'bootstrap_95_percentile_interval':[float(x) for x in np.percentile(draws,[2.5,97.5])],
        'bootstrap_resamples':10000,'bootstrap_seed':seed,'cluster_delta_histogram':dict(Counter(str(float(x)) for x in values)),
        'interpretation':'Descriptive paired cluster uncertainty; sign-flip requires symmetric/exchangeable null signs, not a randomized-training causal test. Four clusters provide coarse uncertainty; no population coverage claim.'}

def summarize(groups):
    if not groups:return {'attempted_groups':0}
    n=len(groups);grades=[x for g in groups for x in g['grades']]
    patients=set();queries=set();images=set();sources=set();up_people=set();up_images=set();up_sources=set()
    for g in groups:
        patients.update(g['people']);queries.add(g['query']);images.add(g['image']);sources.add(g['source_id'])
        if g['updated']:up_people.update(g['people']);up_images.add(g['image']);up_sources.add(g['source_id'])
    return {'attempted_groups':n,'informative_groups':sum(g['informative'] for g in groups),
        'accepted_update_groups':sum(g['updated'] for g in groups),'accepted_update_rate':sum(g['updated'] for g in groups)/n,
        'informative_group_rate':sum(g['informative'] for g in groups)/n,
        'nonzero_total_variance_but_format_rejected_groups':sum(g['variance']>0 and any(x['format']!=1 for x in g['grades']) for g in groups),
        'nonzero_total_variance_other_rejected_groups':sum(g['variance']>0 and not g['informative'] and all(x['format']==1 for x in g['grades']) for g in groups),'informative_but_not_updated':sum(g['informative'] and not g['updated'] for g in groups),
        'zero_total_reward_variance_groups':sum(g['variance']==0 for g in groups),'zero_accuracy_variance_groups':sum(g['accuracy_variance']==0 for g in groups),
        'constant_accuracy_all_correct_groups':sum(all(x['accuracy']==1 for x in g['grades']) for g in groups),
        'constant_accuracy_all_wrong_groups':sum(all(x['accuracy']==0 for x in g['grades']) for g in groups),
        'groups_with_format_failure':sum(any(x['format']!=1 for x in g['grades']) for g in groups),
        'groups_with_any_reward_tie':sum(len(set(x['total'] for x in g['grades']))<len(g['grades']) for g in groups),
        'sample_total_reward_variance_mean':mean(g['variance'] for g in groups),
        'sample_total_reward_variance_distribution':dict(Counter(format(g['variance'],'.12g') for g in groups)),
        'rollouts':len(grades),'correct_rollouts':sum(x['accuracy']==1 for x in grades),'wrong_rollouts':sum(x['accuracy']==0 for x in grades),
        'other_accuracy_rollouts':sum(x['accuracy'] not in (0,1) for x in grades),'full_format_rollouts':sum(x['format']==1 for x in grades),
        'partial_format_rollouts':sum(0<x['format']<1 for x in grades),'zero_format_rollouts':sum(x['format']==0 for x in grades),
        'generated_categorical_answer_counts':dict(sum((Counter(g['predictions']) for g in groups),Counter())),
        'accepted_groups_correct_rollouts':sum(x['accuracy']==1 for g in groups if g['updated'] for x in g['grades']),
        'accepted_groups_wrong_rollouts':sum(x['accuracy']==0 for g in groups if g['updated'] for x in g['grades']),
        'accepted_positive_advantage_answer_counts':dict(Counter(label for g in groups if g['updated'] for label,a in zip(g['predictions'],g['advantages']) if a>0)),
        'accepted_negative_advantage_answer_counts':dict(Counter(label for g in groups if g['updated'] for label,a in zip(g['predictions'],g['advantages']) if a<0)),
        'actual_input_tokens':sum(g['input_tokens'] for g in groups),'actual_generated_tokens':sum(g['generated_tokens'] for g in groups),
        'actual_total_tokens':sum(g['tokens'] for g in groups),'tokens_in_update_producing_groups':sum(g['tokens'] for g in groups if g['updated']),
        'unique_query_people_attempted':len(queries),'unique_source_people_including_candidate_attempted':len(patients),
        'unique_image_payloads_attempted':len(images),'unique_source_rows_attempted':len(sources),
        'unique_source_people_including_candidate_contributing_updates':len(up_people),
        'unique_image_payloads_contributing_updates':len(up_images),'unique_source_rows_contributing_updates':len(up_sources)}

def training(root):
    base=root/'local_data/c_two_task_pool_v1';plan=load(base/'tier_c_v1/plan_v3_PRIVATE.json');meas=load(base/'native_tokens_v2/measurements_PRIVATE.json')
    train=base/'train.verl.jsonl';rows=jsonl(train);require(digest(train)==plan['manifest_sha256'],'Frozen train manifest changed')
    require(te.measurements_sha256(meas)==plan['measurements_sha256'],'Frozen token measurement changed')
    p=te.prepare_arm(plan,'U_raw',7,rows,meas['rows']);out=base/'tier_c_v1/U_raw_seed7_run_v1'
    initial=load(out/'boundary_0000/receipt.json');binding=initial['binding']
    require(binding['schedule_record_ids']==[r['id'] for r in p['schedule']] and binding['experiment']==p['bindings'],'Actual frozen U7 schedule/common contract differs')
    n=125;groups=[];inventory=[];prior=digest(out/'boundary_0000/receipt.json');updates=0;grader=make_grader(None)
    for i in range(n):
        d=out/f'boundary_{i+1:04d}';receipt=load(d/'receipt.json');raw=load(d/'rollout_PRIVATE.json');row=p['schedule'][i];task=row['ability']
        require(receipt['binding']==binding and receipt['parent_receipt_sha256']==prior and receipt['completed_groups']==i+1,'Rollout receipt chain/cursor differs')
        require(receipt['files_sha256']['rollout_PRIVATE.json']==digest(d/'rollout_PRIVATE.json') and receipt['record_id']==receipt['schedule_record_id']==raw['record_id']==row['id'] and
            raw['group_index']==i and raw['image_sha256']==binding['image_sha256'][row['id']] and len(raw['responses'])==4,'Rollout/schedule/image identity differs')
        grades=[];preds=[]
        for response in raw['responses']:
            g=grader(row,response['text'],{});g['overlong']=overlong_penalty(len(response['token_ids']),256,'direct_vero');g['total']=g['score']+g['overlong'];grades.append(g)
            preds.append(prediction(response['text'],task)[0])
        require(grades==receipt['grades'],'Recomputed native raw grades differ')
        informative=task_group_is_informative(grades);updated=receipt['updated'];updates+=int(updated)
        require(receipt['optimizer_updates']==updates and (not updated or informative),'Update cursor or active-group contract differs')
        rewards=[g['total'] for g in grades];adv=standardize(rewards)
        require(all(abs(a-b)<1e-12 for a,b in zip(adv,receipt['advantages'],strict=True)),'Saved group advantages differ')
        L=p['input_tokens'](row);generated=sum(len(r['token_ids']) for r in raw['responses']);e=row['extra_info'];people={row['data_source']+':'+e['patient_key']}
        if e.get('candidate_patient_key'):people.add(row['data_source']+':'+e['candidate_patient_key'])
        groups.append({'task':task,'target':row['reward_model']['ground_truth'],'updated':updated,'informative':informative,'grades':grades,'predictions':preds,
            'variance':sample_std(rewards)**2,'accuracy_variance':sample_std([g['accuracy'] for g in grades])**2,
            'input_tokens':4*L,'generated_tokens':generated,'tokens':4*L+generated,'people':people,'query':row['data_source']+':'+e['patient_key'],
            'image':raw['image_sha256'],'source_id':row['source_record_id'],'advantages':adv,'quartile':quartile(i,n)})
        prior=digest(d/'receipt.json');inventory.append([i+1,prior,digest(d/'rollout_PRIVATE.json')])
    final=load(out/'boundary_0125/receipt.json')
    require(updates==39 and sum(g['tokens'] for g in groups)==final['ledger']['used_tokens']==167071 and final['ledger']['imputed_unknown_tokens']==0,'Actual training totals differ')
    result={'overall':summarize(groups),'by_task':{},'by_task_true_label':{},'global_group_quartiles':[]}
    for task in TASKS:
        selected=[g for g in groups if g['task']==task];result['by_task'][task]=summarize(selected)
        result['by_task_true_label'][task]={t:summarize([g for g in selected if g['target']==t]) for t in sorted({g['target'] for g in selected})}
    for q in range(4):
        selected=[g for g in groups if g['quartile']==q]
        result['global_group_quartiles'].append({'quartile':q+1,'rule':'floor(4*zero_based_group_index/125); fixed global positions, not outcome-selected',
            'by_task_true_label':{task:{t:summarize([g for g in selected if g['task']==task and g['target']==t]) for t in sorted({g['target'] for g in groups if g['task']==task})} for task in TASKS}})
    result['allocation_definitions']={'attempted':'completed durable rollout group, not an optimizer update','informative':'all full-format, task reward varies, identical overlong term',
        'accepted_update':'receipt.updated=true, existing verified actual Adam/checkpoint evidence accepted without rescanning arrays',
        'tokens_in_update_producing_groups':'rollout exposure associated with accepted groups, not measured optimizer FLOPs','unique_people':'source-namespaced query/candidate keys, not demonstrated cross-source global independence',
        'trajectory':'stochastic on-policy training rollouts only; not an evaluation curve or causal mechanism'}
    evidence={'training_manifest':digest(train),'plan':digest(base/'tier_c_v1/plan_v3_PRIVATE.json'),'initial_receipt':digest(out/'boundary_0000/receipt.json'),
        'final_receipt':prior,'rollout_and_receipt_inventory':canonical_sha(inventory),'token_measurements':digest(base/'native_tokens_v2/measurements_PRIVATE.json')}
    return result,groups,evidence

def evaluation(root,expected_final,expected_base):
    base=root/'local_data/c_two_task_pool_v1';rep=root/'results/U_RAW_SEED7_FINAL_C40_PAIRED_EVALUATION.json';bp=root/'results/C40_B28_COMMON_START_BASELINE.json'
    require(digest(rep)==expected_final and digest(bp)==expected_base,'Externally authenticated C40 report digests differ')
    report=load(rep);baseline_public=load(bp);r=base/'tier_c_v1/U_raw_seed7_final_C40_release_v1';b=load(r/'analysis_bindings_PRIVATE.json')
    review=load(b['review_path']);require(digest(b['review_path'])==report['evidence_sha256']['launch_review'] and all(review[k]==v for k,v in b['scope'].items()),'Actual final inference review differs')
    normalized,proof=adapt_cli_output(source_manifest=b['source_manifest'],eval_manifest=b['eval_manifest'],projection_receipt=b['projection_receipt'],
        contract_path=b['contract_path'],output_path=b['scope']['output'],expected_config=b['expected_config'],approved_scope=b['scope'],
        expected_scope_sha256=canonical_sha(b['scope']),expected_projection_sha256=b['projection_receipt_sha256'],
        expected_output_sha256=report['evidence_sha256']['output'],expected_cache_binding=b['scope']['derived_cache'])
    private=base/'evaluation_v1/C40_B28_baseline_independent_scoring_PRIVATE.json'
    require(digest(private)==baseline_public['independent_private_receipt_sha256']==report['evidence_sha256']['baseline_private'],'Authenticated original B28 predictions changed')
    rows=jsonl(b['source_manifest']);A=score_predictions(rows,load(private)['normalized_response_components_PRIVATE'],expected_split='internal_dev')
    B=score_predictions(rows,normalized,expected_split='internal_dev');result={}
    require(A['public']==baseline_public['primary_metrics'],'Original B28 re-score differs')
    for task in TASKS:
        ids=[key for key,c in A['private']['cases'].items() if c['task']==task];a=A['private']['cases'];bb=B['private']['cases'];units=defaultdict(list)
        for key in ids:units[a[key]['unit_key']].append(bb[key]['native_task_accuracy']-a[key]['native_task_accuracy'])
        classes={}
        for label in sorted({a[k]['target'] for k in ids}):
            sub=[k for k in ids if a[k]['target']==label];by_unit=defaultdict(list)
            for k in sub:by_unit[a[k]['unit_key']].append(bb[k]['native_task_accuracy']-a[k]['native_task_accuracy'])
            classes[label]={'questions':len(sub),'B28_correct':sum(a[k]['native_task_accuracy'] for k in sub),'U7_correct':sum(bb[k]['native_task_accuracy'] for k in sub),
                'paired_errors':paired_errors([a[k]['native_task_accuracy'] for k in sub],[bb[k]['native_task_accuracy'] for k in sub]),'cluster_uncertainty':cluster_uncertainty(by_unit)}
        result[task]={'questions':len(ids),'unit':'person' if task=='mri_sequence' else 'matched_two_person_cycle','units':len(units),
            'B28_accuracy':A['public']['tasks'][task]['native_task_accuracy'],'U7_accuracy':B['public']['tasks'][task]['native_task_accuracy'],
            'paired_errors':paired_errors([a[k]['native_task_accuracy'] for k in ids],[bb[k]['native_task_accuracy'] for k in ids]),
            'cluster_uncertainty':cluster_uncertainty(units),'by_true_label':classes,
            'B28_prediction_counts':dict(Counter(prediction(a[k]['raw_response'],task)[0] for k in ids)),
            'U7_prediction_counts':dict(Counter(prediction(bb[k]['raw_response'],task)[0] for k in ids))}
    return ({'tasks':result,'B28_equal_task_macro':A['public']['equal_task_macro_native_accuracy'],'U7_equal_task_macro':B['public']['equal_task_macro_native_accuracy'],
        'macro_delta':B['public']['equal_task_macro_native_accuracy']-A['public']['equal_task_macro_native_accuracy'],
        'scope':'Previously examined internal-dev only; four sequence people and four Dallas cycles. No independence across source namespaces or prospective-test generalization established.',
        'macro_interval':'NOT_ESTIMATED: no assumption of independent cross-source units, no simultaneous macro coverage or training-seed uncertainty claim'},
        {'final_report':digest(rep),'baseline_report':digest(bp),'baseline_private':digest(private),'final_output':proof['output_sha256'],'dev_manifest':digest(b['source_manifest'])})

def calibration_invariance(groups,frozen_calibration,proof_sha256):
    eligible=[g for g in groups if g['informative'] and set(x['accuracy'] for x in g['grades'])=={0.,1.}];grid=[]
    for median in (0.,.25,.5,.75,1.):
        for iqr in (.0625,.125,.25,.5,1.):
            logistic=lambda x:1/(1+math.exp(-3*(x-median)/iqr))
            scale=logistic(1)-logistic(0);max_error=0.
            for g in eligible:
                raw=[x['total'] for x in g['grades']];trans=[.8*logistic(x['accuracy'])+.2*x['format']+x['overlong'] for x in g['grades']]
                max_error=max(max_error,max(abs(a-b) for a,b in zip(standardize(raw),standardize(trans))))
            grid.append({'hypothetical_p50':median,'hypothetical_IQR':iqr,'positive_reward_scale':scale,'max_abs_standardized_advantage_change':max_error})
    actual={}
    for task in TASKS:
        stats=frozen_calibration[task];median=stats['p50'];iqr=stats['iqr'];logistic=lambda x:1/(1+math.exp(-3*(x-median)/iqr))
        scale=logistic(1)-logistic(0);error=0.;zero_error=0.;formula_error=0.;selected=[g for g in eligible if g['task']==task]
        for g in selected:
            raw=[x['total'] for x in g['grades']];trans=[.8*logistic(x['accuracy'])+.2*x['format']+x['overlong'] for x in g['grades']]
            a=standardize(raw);b=standardize(trans);factor=(sample_std(raw)+1e-6)/(sample_std(raw)+1e-6/scale)
            error=max(error,max(abs(x-y) for x,y in zip(a,b)))
            zero_error=max(zero_error,max(abs(x-y) for x,y in zip(standardize(raw,0),standardize(trans,0))))
            formula_error=max(formula_error,max(abs(y-x*factor) for x,y in zip(a,b)))
        actual[task]={'frozen_p50':median,'frozen_IQR':iqr,'positive_reward_scale':scale,'observed_active_groups':len(selected),
            'maximum_abs_advantage_change_at_actual_epsilon':error,'maximum_abs_difference_epsilon0_float64':zero_error,
            'maximum_abs_difference_from_exact_epsilon_factor_formula':formula_error}
    return {'actual_frozen_statistics':actual,'independent_actual_calibration_proof_sha256':proof_sha256,
        'maximum_actual_active_advantage_change':max(v['maximum_abs_advantage_change_at_actual_epsilon'] for v in actual.values()),'observed_binary_informative_groups':len(eligible),'sample_standard_deviation_ddof':1,'epsilon':1e-6,
        'analytic_result':'For binary accuracy and constant format/overlong, any strictly increasing two-value calibration is r_new=c+s*r_raw, s>0. With epsilon0, per-group standardized advantages are identical; zero-variance membership is unchanged.',
        'finite_epsilon_exact_factor':'A_new/A_raw=(std_raw+epsilon)/(std_raw+epsilon/s); nonzero epsilon changes magnitude slightly, not sign or which binary groups are informative.',
        'hypothetical_grid_not_actual_calibration_statistics':grid,'maximum_grid_advantage_change':max(x['max_abs_standardized_advantage_change'] for x in grid),
        'claim_limit':'This is an analytic/effective-epsilon null-control diagnosis, not actual calibration rollouts or proof of identical finite-precision optimizer trajectories. Mixed-format/nonconstant-length groups lie outside the invariance premise and are excluded by the current active gate.'}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--expected-final-report-sha256',required=True);ap.add_argument('--expected-baseline-report-sha256',required=True)
    ap.add_argument('--expected-calibration-proof-sha256',required=True)
    ap.add_argument('--output',type=Path,default=ROOT/'results/U7_NEGATIVE_LEARNING_SIGNAL_DIAGNOSTIC.json');a=ap.parse_args()
    tr,groups,ev=training(ROOT);dev,dev_ev=evaluation(ROOT,a.expected_final_report_sha256,a.expected_baseline_report_sha256)
    proofpath=ROOT/'local_data/c_two_task_pool_v1/calibration_v1/independent_quantiles_v1/receipt_SAFE.json'
    require(digest(proofpath)==a.expected_calibration_proof_sha256,'Independently supplied calibration proof digest differs')
    proof=load(proofpath);calrun=ROOT/'local_data/c_two_task_pool_v1/calibration_v1/run';calfile=calrun/'calibration_for_arms_PRIVATE.json'
    require(proof['schema']=='common-B28-calibration-independent-quantiles-v1' and proof['verdict'] in {'PASS','PASS_WITH_LIMITATIONS'} and
        proof['ordered_raw_76_verified'] is True and proof['raw_quantiles_independently_recomputed'] is True and proof['rows']==76 and
        digest(calfile)==proof['evidence_sha256']['calibration_file'] and digest(calrun/'calibration_complete_SAFE.json')==proof['complete_receipt_sha256'],'Actual frozen calibration evidence differs')
    frozen=load(calfile);require(frozen=={t:{k:proof['statistics'][t][k] for k in ('p50','iqr')} for t in TASKS},'Actual calibrated parameters differ from independent quantiles')
    result={'schema':'U7-negative-learning-signal-artifact-diagnostic-v1','status':'ACTUAL_SAVED_ROLLOUTS_AND_PAIRED_C40_RECOMPUTED_CPU','author':'Codex /root/likelihood_finish_reviewer',
        'evidence_tier':'REPRODUCED_FROM_SAVED_ARTIFACTS','research_tags':['PRIMARY_VERO_TRANSFER','EXPLORATORY_DIAGNOSTIC'],'training':tr,'paired_C40':dev,
        'binary_reward_calibration':calibration_invariance(groups,frozen,a.expected_calibration_proof_sha256),'evidence_sha256':{**ev,**dev_ev,'diagnostic_source':digest(__file__),'independent_calibration_proof':digest(proofpath),'actual_frozen_calibration':digest(calfile)},
        'native_models_launched':0,'optimizer_arrays_reaudited':False,'sealed_test_content_used':False,'frozen_training_protocol_changed':False,'causal_reason_for_decline_identified':False,
        'five_category_evidence_suggestions':{
            'engineering_bug':{'status':'UNRESOLVED','evidence':'Raw route/targets/image-to-schedule hashes and saved reward/advantage/cursor numerics are checked here; previous native Adam/checkpoint proof accepted. Native GSPO ratio/gradient formula belongs to separate targeted numerical fixture.','action':'Fix only a concrete reproduced defect; no broad optimizer re-audit.'},
            'optimization_method':{'status':'POSSIBLE','evidence':'One stochastic seed with39 accepted updates and lower internal-dev macro does not identify GSPO ineffectiveness; binary monotone reward calibration is largely advantage-invariant.','action':'Complete frozen specialist/proportional/uniform seeds and mechanistic calibration comparison; interpret actual effective exposure.'},
            'task_reward_design':{'status':'SUPPORTED','evidence':'Compare attempted, informative and accepted groups by true class plus generated-answer skew; active filtering alters realized update balance.','action':'Preserve original18 arms; test a separate correction only if frozen controls establish its need. Do not change reward to manufacture updates.'},
            'scientific_data_validity':{'status':'POSSIBLE','evidence':'Saved categorical counts cannot distinguish relevant visual evidence from answer priors, whole-head/intensity shortcuts or source/task confounding.','action':'Separate predetermined training/internal-dev no-image, truth-updated image substitutions and fixed-prefix likelihood controls; sealed test remains closed.'},
            'statistical_uncertainty':{'status':'QUANTIFIED','evidence':'Four sequence people/four matched cycles; classwise paired errors and cluster bootstrap/sign flips provided, not40 independent patients.','action':'Finish frozen seeds and shared evaluations; avoid claiming independent-patient or clinical generalization.'}},
        'limitations':['On-policy training quartiles change both sampled examples and current policy; they are not paired evaluation curves or causal learning estimates.',
            'Source-namespaced unique people/image counts describe exposure, not verified global cross-source patient independence.',
            'One seed/internal-dev repeatedly examined; no test access, new model computation, clinical validity or Vero-transfer success claim.',
            'No raw identifiers, noncategorical answer text, per-case responses, paths to patient images or patient-level rows are exported.']}
    require(not any(k.split('.')[0] in {'mlx','mlx_vlm','torch','transformers'} for k in sys.modules),'Unexpected native runtime import')
    require(a.output.resolve().is_relative_to(ROOT/'results'),'Diagnostic aggregate must stay in results')
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'output_sha256':digest(a.output),'attempted_groups':tr['overall']['attempted_groups'],'updates':tr['overall']['accepted_update_groups'],'task_groups':{t:tr['by_task'][t]['attempted_groups'] for t in TASKS},'task_updates':{t:tr['by_task'][t]['accepted_update_groups'] for t in TASKS},'macro_delta':dev['macro_delta'],'native_imports':False}))

if __name__=='__main__':main()
