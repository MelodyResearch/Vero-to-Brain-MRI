"""Small CPU scoring core for frozen two-task outputs; no inference or model imports.

The normalized predictions argument is ID -> {response: str}; it is not a raw
model-file approval interface. A caller must verify exact file/model/prompt/image
bindings before adapting real receipts. The public return excludes all case,
patient and cycle identifiers. Only the private branch may be stored locally.
"""
from __future__ import annotations
from collections import defaultdict
from statistics import mean

from mri_rl.experiments import paired_patient_difference
from mri_rl.vero_mri_reward import _answer_and_format

TASK_SOURCES={'mri_sequence':'OpenNeuro_ds007045','mri_visit_correspondence':'OpenNeuro_ds004856'}
METRICS=('native_task_accuracy','full_format_accuracy','format','full_format')


def require(ok,message):
    if not ok:raise ValueError(message)


def _reference(rows,expected_split):
    require(expected_split in {'internal_dev','test'},'Evaluation split must be explicit internal_dev or released test')
    require(rows and len({r['id'] for r in rows})==len(rows),'Duplicate/empty frozen reference IDs')
    require({r['ability'] for r in rows}==set(TASK_SOURCES),'Both frozen tasks required')
    cycles=defaultdict(lambda:defaultdict(list));patient_cycle={}
    for row in rows:
        task=row['ability'];e=row['extra_info'];target=row['reward_model']['ground_truth']
        require(row['data_source']==TASK_SOURCES[task] and e['reward_type']==task,'Reference source/route mismatch')
        require(e['internal_split']==expected_split and e.get('patient_lineage')=='BIDS_SUBJECT_VERIFIED' and e.get('patient_key'),'Reference split/lineage mismatch')
        if task=='mri_sequence':require(target in {'FLAIR','contrast-enhanced T1'},'Reference sequence target malformed')
        else:
            q=e['patient_key'];c=e.get('candidate_patient_key');cluster=e.get('statistical_cluster_key')
            require(c and cluster and target==('same' if q==c else 'different'),'Reference equality target mismatch')
            for patient in (q,c):
                require(patient not in patient_cycle or patient_cycle[patient]==cluster,'Patient appears in mismatched cycle')
                patient_cycle[patient]=cluster
            cycles[cluster][q].append((c,target))
    for queries in cycles.values():
        require(len(queries)==2,'A Dallas cycle requires both source query people')
        for references in queries.values():
            require({p for p,_ in references}==set(queries) and {t for _,t in references}=={'same','different'},'Incomplete matched cycle reference targets')


def _units(cases):
    groups=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    for case in cases.values():
        for metric in METRICS:groups[case['task']][case['unit_key']][case['patient_key']].append((metric,case[metric]))
    result={}
    for task,units in groups.items():
        result[task]={}
        for unit,patients in units.items():
            # Within-person means first, then two equal query-person weights in
            # a Dallas cycle. Repeated cases cannot inflate independent units.
            result[task][unit]={metric:mean(mean(value for key,value in values if key==metric) for values in patients.values()) for metric in METRICS}
    return result


def score_predictions(manifest_rows,predictions_by_id,*,expected_split='internal_dev'):
    """Re-score normalized raw text; missing/invalid outputs receive zero accuracy.

    native_task_accuracy preserves the existing native parser's 0/.5/1 format
    decomposition. full_format_accuracy separately requires format exactly1.
    """
    _reference(manifest_rows,expected_split)
    ids={r['id'] for r in manifest_rows};require(not set(predictions_by_id)-ids,'Unknown prediction ID')
    cases={}
    for row in manifest_rows:
        key=row['id'];task=row['ability'];e=row['extra_info'];target=row['reward_model']['ground_truth']
        output=predictions_by_id.get(key)
        if output is not None:
            require(isinstance(output,dict),'Prediction adapter must provide a normalized object')
            for field,want in (('ground_truth',target),('data_source',row['data_source']),('ability',task),('internal_split',expected_split)):
                require(field not in output or output[field]==want,'Prediction '+field+' differs from frozen reference')
        response=output.get('response') if output else None
        answer,formatting=_answer_and_format(response) if isinstance(response,str) else ('',0.)
        allowed={'flair','contrast-enhanced t1'} if task=='mri_sequence' else {'same','different'}
        valid=bool(formatting>0 and answer.lower() in allowed)
        hit=float(valid and answer.lower()==target.lower())
        cases[key]={'task':task,'source':row['data_source'],'split':expected_split,'target':target,
            'patient_key':e['patient_key'],'candidate_patient_key':e.get('candidate_patient_key'),
            'unit_key':e['patient_key'] if task=='mri_sequence' else e['statistical_cluster_key'],
            'native_task_accuracy':hit,'format':formatting,'full_format':float(formatting==1),
            'full_format_accuracy':float(hit and formatting==1),'invalid_answer':not valid,
            'missing':output is None,'raw_response':response}
    units=_units(cases);tasks={}
    for task in TASK_SOURCES:
        values=[x for x in cases.values() if x['task']==task];u=units[task]
        tasks[task]={'questions':len(values),'patients':len({x['patient_key'] for x in values}),
            'independent_units':len(u),'unit':'patient' if task=='mri_sequence' else 'matched_two_person_cycle',
            'native_task_accuracy':mean(x['native_task_accuracy'] for x in u.values()),
            'full_format_accuracy':mean(x['full_format_accuracy'] for x in u.values()),
            'mean_format_score':mean(x['format'] for x in u.values()),
            'full_format_rate':mean(x['full_format'] for x in u.values()),
            'question_native_task_accuracy':mean(x['native_task_accuracy'] for x in values),
            'missing_predictions':sum(x['missing'] for x in values),'invalid_answers':sum(x['invalid_answer'] for x in values),
            'descriptive_only':True}
    public={'status':'SCORED_NORMALIZED_TEXT_BINDINGS_REQUIRE_CALLER_VERIFICATION','split':expected_split,'tasks':tasks,
        'equal_task_macro_native_accuracy':mean(x['native_task_accuracy'] for x in tasks.values()),
        'worst_task_native_accuracy':min(x['native_task_accuracy'] for x in tasks.values()),
        'equal_task_macro_full_format_accuracy':mean(x['full_format_accuracy'] for x in tasks.values()),
        'accuracy_contract':'Existing native tag-gated categorical accuracy; tagged unboxed answer may earn format0.5 and task accuracy1. Full-format accuracy is reported separately.',
        'missing_or_invalid_accuracy':0,'descriptive_only':True,'independent_patient_generalization_claimed':False,
        'clinical_or_method_transfer_claimed':False}
    return {'public':public,'private':{'cases':cases,'units':units}}


def compare_scores(base,candidate,*,n_bootstrap=2000,seed=0,explicit_control_targets=None):
    a=base['private']['cases'];b=candidate['private']['cases']
    require(set(a)==set(b),'Paired reference ID coverage differs')
    if explicit_control_targets is not None:require(set(explicit_control_targets)==set(a),'Explicit control target inventory differs')
    for key in a:
        for field in ('task','source','split','patient_key','candidate_patient_key','unit_key'):
            require(a[key][field]==b[key][field],'Paired source/split/cluster membership differs')
        target=a[key]['target'] if explicit_control_targets is None else explicit_control_targets[key]
        require(b[key]['target']==target,'Paired target differs; transformed controls require explicit frozen reference')
    contrasts={}
    for task in TASK_SOURCES:
        au=base['private']['units'][task];bu=candidate['private']['units'][task]
        require(set(au)==set(bu),'Paired statistical units differ')
        paired=[{'subject_id':unit,'base':au[unit]['native_task_accuracy'],'candidate':bu[unit]['native_task_accuracy']} for unit in sorted(au)]
        estimate=mean(x['candidate']-x['base'] for x in paired)
        interval=paired_patient_difference(paired,n_bootstrap=n_bootstrap,seed=seed) if len(paired)>=2 else {'subjects':len(paired),'mean_difference':estimate,'confidence_interval':None,'status':'NOT_ESTIMATED_TOO_FEW_UNITS'}
        contrasts[task]={'paired_delta_from_common_start':estimate,'independent_units':len(paired),
                         'paired_interval':interval,'descriptive_only':True}
    return {'tasks':contrasts,'equal_task_macro_delta':mean(x['paired_delta_from_common_start'] for x in contrasts.values()),
            'worst_task_delta':min(x['paired_delta_from_common_start'] for x in contrasts.values()),
            'control_targets':'explicit_frozen_transformed_reference' if explicit_control_targets is not None else 'identical_reference',
            'interval_scope':'Existing percentile cluster bootstrap per task only; no simultaneous macro/worst-task coverage or training-seed uncertainty claim.',
            'descriptive_only':True}


def build_matrix(runs,*,baseline_arm='B28',training_task_by_arm,n_bootstrap=2000):
    """Rows retain every arm and seed; specialists are evaluated on both tasks."""
    require(runs,'No completed scored runs');cells=[];summaries=[]
    for (arm,seed),scored in sorted(runs.items()):
        require((baseline_arm,seed) in runs,'Common initializer missing for a paired seed')
        require(arm in training_task_by_arm and training_task_by_arm[arm] in {None,*TASK_SOURCES},'Unknown training arm/task')
        comparison=compare_scores(runs[(baseline_arm,seed)],scored,n_bootstrap=n_bootstrap,seed=seed)
        for task in TASK_SOURCES:
            cells.append({'arm':arm,'seed':seed,'training_task':training_task_by_arm[arm],'evaluation_task':task,
                'native_task_accuracy':scored['public']['tasks'][task]['native_task_accuracy'],
                'full_format_accuracy':scored['public']['tasks'][task]['full_format_accuracy'],
                'off_diagonal':training_task_by_arm[arm]!=task if training_task_by_arm[arm] is not None else None,
                **comparison['tasks'][task]})
        summaries.append({'arm':arm,'seed':seed,'equal_task_macro_native_accuracy':scored['public']['equal_task_macro_native_accuracy'],
            'worst_task_native_accuracy':scored['public']['worst_task_native_accuracy'],
            'macro_delta_from_common_start':comparison['equal_task_macro_delta'],
            'worst_task_delta_from_common_start':comparison['worst_task_delta']})
    by_arm={}
    for arm in sorted({x['arm'] for x in summaries}):
        group=[x for x in summaries if x['arm']==arm]
        by_arm[arm]={'training_seeds':len(group),'seeds':[x['seed'] for x in group],
            'macro_native_accuracy_mean':mean(x['equal_task_macro_native_accuracy'] for x in group),
            'macro_native_accuracy_range':[min(x['equal_task_macro_native_accuracy'] for x in group),max(x['equal_task_macro_native_accuracy'] for x in group)],
            'seed_uncertainty':'Observed per-seed values/range only; seeds are not additional patients.'}
    return {'cells':cells,'arm_seed_summary':summaries,'between_seed_summary':by_arm,
        'status':'DESCRIPTIVE_MATCHED_MATRIX','descriptive_only':True,'independent_patient_generalization_claimed':False,
        'scope':'Core consumes normalized outputs only; actual file/image/prompt/model binding checks belong to the reviewed raw-output adapter.'}
